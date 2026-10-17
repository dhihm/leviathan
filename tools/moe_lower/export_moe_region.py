#!/usr/bin/env python3
"""Captures the routed MoE region with torch.export(strict=True) into region
JSON for Swift validation and Metal lowering.

Usage: export_moe_region.py <region_model.py> <out-region.json> [config.json]

Executes the user's trusted file (explicit user action, like running any
script); the module must define build(config) -> nn.Module whose forward is
the routed-expert slot region. The export runs on META tensors: static
shapes only, NO weight allocation and NO GGUF access. The exported ATen
graph is walked node-by-node with EXACT arity/kwargs/constant checks and
normalized into a canonical DAG (typed operand tree: node refs, static
ints, lists, literals) whose digest binds the region identity. Every
semantic dimension in the emitted JSON is DERIVED from the actual exported
nodes — never copied from the module's Python attributes. Any deviation
(unknown op, altered semantics, non-tanh GELU, wrong slot order, bias,
mutation, symint shape, extra nodes) fails closed naming the offending
node, with NO output files written.

The normalized graph + digest are re-validated and re-derived by the Swift
importer (fail-closed even against a recomputed digest); only the torch
version/source block is advisory.
"""

import hashlib
import importlib.util
import json
import operator
import sys

import torch
from torch.export.graph_signature import InputKind, OutputKind

GETITEM = "operator.getitem"
LINEAR = "aten.linear.default"
SELECT = "aten.select.int"
SPLIT = "aten.split.Tensor"
GELU = "aten.gelu.default"
MUL = "aten.mul.Tensor"
STACK = "aten.stack.default"

MAX_NODES = 512
MAX_DIM = 8192
MAX_TOPK = 64


class Reject(Exception):
    pass


# ---- canonical serialization (MUST match the Swift importer exactly) ----

def canon(v):
    """Canonical string for a JSON-form operand: node references (the
    {"__node__": name} marker) become @name. MUST match the Swift side."""
    if v is None:
        return "n"
    if isinstance(v, bool):
        return "b:1" if v else "b:0"
    if isinstance(v, int):
        return f"i:{v}"
    if isinstance(v, str):
        return "s:" + v
    if isinstance(v, (list, tuple)):
        return "l<" + ",".join(canon(x) for x in v) + ">"
    if isinstance(v, dict):
        if set(v) == {"__node__"} and isinstance(v["__node__"], str):
            return "@" + v["__node__"]
        return "d<" + ",".join(f"{k}:{canon(v[k])}" for k in sorted(v)) + ">"
    raise Reject(f"uncanonicalizable operand {v!r}")


class NodeRef:
    """A graph operand that is a reference to another node by name."""
    def __init__(self, name):
        self.name = name

    def canon(self):
        return "@" + self.name

    def to_json(self):
        return {"__node__": self.name}


def json_operand(v):
    if isinstance(v, NodeRef):
        return v.to_json()
    if isinstance(v, (list, tuple)):
        return [json_operand(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise Reject(f"operand {v!r} is not a canonical operand kind")


def node_meta(node):
    """Normalized metadata for one FX node.

    Scalar tensor_meta -> {"dtype","shape"}. Sequence outputs (split's
    tuple) carry tensor_meta None/sequence: item dtypes/shapes are then
    derived from node.meta["val"] (the FakeTensor list) into
    {"dtype": "seq", "shape": None, "items": [...]} — static-validated.
    Neither -> None. TensorMetadata is a NAMEDTUPLE: its shape/dtype
    attributes are checked BEFORE tuple classification so a scalar metadata
    is never misclassified as a sequence."""
    meta = node.meta.get("tensor_meta")
    if meta is not None:
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
        if shape is not None and dtype is not None \
                and isinstance(shape, tuple):
            if any(type(d) is not int for d in shape):
                raise Reject(f"node {node.name}: tensor_meta shape {shape} "
                             "has a non-int dimension (symint/bool rejected)")
            return {"dtype": str(dtype).replace("torch.", ""),
                    "shape": list(shape)}
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)) and len(val) > 0 \
            and all(hasattr(v, "dtype") and hasattr(v, "shape") for v in val):
        items = []
        for v in val:
            dims = list(v.shape)
            if any(type(d) is not int for d in dims):
                raise Reject(f"node {node.name}: item shape {dims} has a "
                             "non-int dimension (symint rejected)")
            items.append({"dtype": str(v.dtype).replace("torch.", ""),
                          "shape": dims})
        return {"dtype": "seq", "shape": None, "items": items}
    return None


def static_dims(name, meta, want_rank):
    if meta is None:
        raise Reject(f"{name}: missing static tensor_meta")
    if meta["dtype"] != "float32":
        raise Reject(f"{name}: dtype {meta['dtype']} != float32")
    dims = meta["shape"]
    if len(dims) != want_rank or any(type(d) is not int for d in dims):
        raise Reject(f"{name}: shape {dims} is not a static rank-{want_rank} "
                     "f32 tensor (symint/bool rejected)")
    return dims


def node_args(name, node, count):
    if len(node["args"]) != count or node["kwargs"]:
        raise Reject(f"{name} ({node['name']}): {node['target']} must take "
                     f"exactly {count} positional args and no kwargs; got "
                     f"args={node['args']!r} kwargs={node['kwargs']!r}")
    return list(node["args"])


def int_const(name, node, v):
    if type(v) is not int:
        raise Reject(f"{name} ({node['name']}): constant must be a static int, "
                     f"got {v!r}")
    return v


def export_and_normalize(module):
    hidden, inter, top_k = module.hidden, module.inter, module.top_k
    for attr, v in (("hidden", hidden), ("inter", inter), ("topK", top_k)):
        if type(v) is not int:
            raise Reject(f"module attribute {attr} must be a static int, got {v!r}")
    for label, v in (("hidden", hidden), ("inter", inter)):
        if not (0 < v <= MAX_DIM) or v % 32 != 0:
            raise Reject(f"{label}={v} out of bounds (positive, <=8192, 32-multiple)")
    if not (0 < top_k <= MAX_TOPK):
        raise Reject(f"topK={top_k} out of bounds (<= {MAX_TOPK})")
    x = torch.randn(1, hidden, device="meta")
    gu = torch.randn(top_k, 2 * inter, hidden, device="meta")
    dn = torch.randn(top_k, hidden, inter, device="meta")
    try:
        ep = torch.export.export(module, (x, gu, dn), strict=True)
    except Exception as e:  # noqa: BLE001 - actionable fail-closed boundary
        raise Reject(f"torch.export failed: {type(e).__name__}: {e}") from e

    specs = ep.graph_signature.input_specs
    out_specs = ep.graph_signature.output_specs
    user_inputs = [s for s in specs if s.kind == InputKind.USER_INPUT]
    if len(user_inputs) != 3 or len(specs) != 3:
        raise Reject("expected exactly 3 USER_INPUT specs and no "
                     "parameters/buffers, got "
                     f"{[str(s.kind) for s in specs]}")
    if len(out_specs) != 1 or out_specs[0].kind != OutputKind.USER_OUTPUT:
        raise Reject("expected exactly one USER_OUTPUT spec, got "
                     f"{[str(s.kind) for s in out_specs]} (mutations are "
                     "unsupported)")

    nodes = list(ep.graph_module.graph.nodes)
    if len(nodes) > MAX_NODES:
        raise Reject(f"graph has {len(nodes)} nodes (bound {MAX_NODES})")

    # ---- normalize every node now; semantic checks below operate on the
    # normalized DAG but are re-derived independently by the Swift side.
    norm = []
    meta_by_name = {}
    import re as _re
    _name_ok = _re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    seen_names = set()
    for n in nodes:
        if n.op not in ("placeholder", "call_function", "output"):
            raise Reject(f"unsupported node kind {n.op!r} ({n.name})")
        if not _name_ok.match(n.name):
            raise Reject(f"node name {n.name!r} is not a plain identifier "
                         "(canonical serialization is collision-safe)")
        # Topological order: every operand references an ALREADY-DEFINED node.
        def check_refs(value):
            if isinstance(value, torch.fx.Node):
                if value.name not in seen_names:
                    raise Reject(f"node {n.name} references {value.name} "
                                 "before its definition (not topological)")
            elif isinstance(value, (list, tuple)):
                for v in value:
                    check_refs(v)
        for a in n.args:
            check_refs(a)
        target = str(n.target) if n.op == "call_function" else (
            GETITEM if n.op == "call_function" and n.target is operator.getitem
            else ("placeholder" if n.op == "placeholder" else str(n.op)))
        if n.op == "call_function" and n.target is operator.getitem:
            target = GETITEM
        args = []
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                args.append(NodeRef(a.name))
            elif isinstance(a, (list, tuple)):
                args.append([NodeRef(x.name) if isinstance(x, torch.fx.Node)
                             else x for x in a])
            elif a is None or isinstance(a, (bool, int, str)):
                args.append(a)
            else:
                raise Reject(f"node {n.name}: operand {a!r} is not a "
                             "supported canonical operand (floats rejected)")
        kwargs = dict(n.kwargs)
        for k, v in kwargs.items():
            if not (v is None or isinstance(v, (bool, int, str))):
                raise Reject(f"node {n.name}: kwarg {k}={v!r} is not a "
                             "canonical kind (floats rejected)")
        m = node_meta(n)
        meta_by_name[n.name] = m
        seen_names.add(n.name)
        norm.append({"name": n.name, "op": n.op, "target": target,
                     "args": [json_operand(a) for a in args],
                     "kwargs": kwargs, "meta": m})

    by_name = {n["name"]: n for n in norm}

    def nmeta(nname):
        return meta_by_name[nname]

    placeholders = [n for n in norm if n["op"] == "placeholder"]
    if len(placeholders) != 3:
        raise Reject(f"expected 3 placeholders, got {len(placeholders)}")
    for i, spec in enumerate(user_inputs):
        if spec.arg.name != placeholders[i]["name"]:
            raise Reject(f"input_specs[{i}] name {spec.arg.name} != "
                         f"placeholder order {placeholders[i]['name']}")
    x_n, gu_n, dn_n = placeholders

    def ref_operand(label, v):
        if not isinstance(v, dict) or set(v) != {"__node__"}:
            raise Reject(f"{label}: expected a node reference, got {v!r}")
        if v["__node__"] not in by_name:
            raise Reject(f"{label}: reference to unknown node {v['__node__']!r}")
        return by_name[v["__node__"]]
    xd = static_dims("input x", nmeta(x_n["name"]), 2)
    gd = static_dims("input gu_weights", nmeta(gu_n["name"]), 3)
    dd = static_dims("input down_weights", nmeta(dn_n["name"]), 3)
    if xd[0] != 1:
        raise Reject(f"x batch dim must be 1, got {xd}")
    hidden_d = xd[1]
    topk_d, gu_out_d, gu_in_d = gd
    if gu_in_d != hidden_d:
        raise Reject(f"gu_weights in-dim {gu_in_d} != hidden {hidden_d}")
    dn_top_d, dn_out_d, dn_in_d = dd
    if dn_top_d != topk_d:
        raise Reject(f"down topK {dn_top_d} != gu topK {topk_d}")
    if dn_out_d != hidden_d:
        raise Reject(f"down_weights out-dim {dn_out_d} != hidden {hidden_d}")

    outputs = [n for n in norm if n["op"] == "output"]
    if len(outputs) != 1:
        raise Reject(f"expected 1 output node, got {len(outputs)}")
    if len(outputs[0]["args"]) != 1 or len(outputs[0]["args"][0]) != 1:
        raise Reject(f"output ({outputs[0]['name']}) must pack exactly one "
                     "value")
    stack = ref_operand("output value", outputs[0]["args"][0][0])
    if stack["target"] != STACK:
        raise Reject(f"output must be the {STACK} of slot outputs, got "
                     f"{stack['target']} ({stack['name']})")
    # stack(TensorList tensors, int dim=0): dim may be omitted (default 0)
    # or explicit in args/kwargs — accept exactly the dim==0 semantics.
    sargs = stack["args"]
    skwargs = stack["kwargs"]
    if len(sargs) == 1 and not skwargs:
        pass
    elif len(sargs) == 2 and not skwargs and int_const(
            "stack dim", stack, sargs[1]) == 0:
        pass
    elif len(sargs) == 1 and list(skwargs) == ["dim"] \
            and int_const("stack dim", stack, skwargs["dim"]) == 0:
        pass
    else:
        raise Reject(f"stack ({stack['name']}) must be dim=0; got "
                     f"args={sargs!r} kwargs={skwargs!r}")
    smeta = static_dims("stack output", nmeta(stack["name"]), 3)
    slot_finals = sargs[0]
    if not isinstance(slot_finals, list) or len(slot_finals) != topk_d \
            or any(not isinstance(s, dict) or "__node__" not in s
                   for s in slot_finals):
        raise Reject(f"stack inputs must be {topk_d} node references")
    if smeta != [topk_d, 1, hidden_d]:
        raise Reject(f"stack output shape {smeta} != "
                     f"[{topk_d}, 1, {hidden_d}]")

    inter_d = None
    trace = []
    consumed = {n["name"] for n in placeholders} | {stack["name"],
                                                    outputs[0]["name"]}
    for slot, sref in enumerate(slot_finals):
        final = by_name[sref["__node__"]]
        # DN linear <- mul, no bias
        if final["target"] != LINEAR:
            raise Reject(f"slot {slot}: final op must be {LINEAR}, got "
                         f"{final['target']} ({final['name']})")
        fargs = final["args"]
        if len(fargs) == 3 and fargs[2] is None:
            fargs = fargs[:2]
        if len(fargs) != 2 or final["kwargs"]:
            raise Reject(f"slot {slot}: {LINEAR} ({final['name']}) must be "
                         "(input, weight) with no bias and no kwargs")
        act_ref, dn_sel_ref = fargs
        act = ref_operand(f"slot {slot} activation", act_ref)
        dn_sel = ref_operand(f"slot {slot} down select", dn_sel_ref)
        if dn_sel["target"] != SELECT:
            raise Reject(f"slot {slot}: down weight must come from {SELECT}, "
                         f"got {dn_sel['target']} ({dn_sel['name']})")
        sa = node_args("slot %d down select" % slot, dn_sel, 3)
        if sa[0] != {"__node__": dn_n["name"]} or int_const(
                "slot %d down select dim" % slot, dn_sel, sa[1]) != 0:
            raise Reject(f"slot {slot}: down select ({dn_sel['name']}) must "
                         f"be {dn_n['name']}[0, slot]")
        dn_idx = int_const("slot %d down select idx" % slot, dn_sel, sa[2])
        yd = static_dims(f"slot {slot} down output", nmeta(final["name"]), 2)
        if yd != [1, hidden_d]:
            raise Reject(f"slot {slot}: down output {yd} != [1, {hidden_d}]")
        dsd = static_dims(f"slot {slot} down weight", nmeta(dn_sel["name"]), 2)
        if dsd != [hidden_d, dn_in_d]:
            raise Reject(f"slot {slot}: down weight slice {dsd} != "
                         f"[{hidden_d}, {dn_in_d}]")
        consumed.update([final["name"], dn_sel["name"]])
        # mul <- gelu(gate), up
        if act["target"] != MUL:
            raise Reject(f"slot {slot}: activation must be {MUL}, got "
                         f"{act['target']} ({act['name']})")
        ma = node_args("slot %d mul" % slot, act, 2)
        gelu_ref, up_ref = ma
        gelu = ref_operand(f"slot {slot} gate activation", gelu_ref)
        up = ref_operand(f"slot {slot} up", up_ref)
        if gelu["target"] != GELU:
            raise Reject(f"slot {slot}: gate activation must be {GELU}, got "
                         f"{gelu['target']} ({gelu['name']})")
        if len(gelu["args"]) != 1 or gelu["kwargs"] != {"approximate": "tanh"}:
            raise Reject(f"slot {slot}: gelu ({gelu['name']}) must take "
                         "exactly one arg with only approximate='tanh'; got "
                         f"args={gelu['args']!r} kwargs={gelu['kwargs']!r}")
        ga = gelu["args"]
        gdims = static_dims(f"slot {slot} gate", nmeta(gelu["name"]), 2)
        up = by_name[up_ref["__node__"]]
        if up["target"] != GETITEM:
            raise Reject(f"slot {slot}: up must be a split getitem, got "
                         f"{up['target']} ({up['name']})")
        gate = ref_operand(f"slot {slot} gate", ga[0])
        if gate["target"] != GETITEM:
            raise Reject(f"slot {slot}: gate must be a split getitem, got "
                         f"{gate['target']} ({gate['name']})")
        gta = node_args("slot %d gate getitem" % slot, gate, 2)
        upa = node_args("slot %d up getitem" % slot, up, 2)
        if gta[0] != upa[0]:
            raise Reject(f"slot {slot}: gate/up getitems must read the SAME "
                         "split node")
        split = by_name[gta[0]["__node__"]]
        if split["target"] != SPLIT:
            raise Reject(f"slot {slot}: gate|up must come from {SPLIT}, got "
                         f"{split['target']} ({split['name']})")
        spa = node_args("slot %d split" % slot, split, 3)
        lin_ref = spa[0]
        if not isinstance(lin_ref, dict) or "__node__" not in lin_ref:
            raise Reject(f"slot {slot}: split first operand must be a node "
                         f"reference, got {lin_ref!r}")
        split_size = int_const("slot %d split size" % slot, split, spa[1])
        if int_const("slot %d split dim" % slot, split, spa[2]) != -1:
            raise Reject(f"slot {slot}: split dim must be -1")
        if split_size <= 0:
            raise Reject(f"slot {slot}: split size must be positive")
        if inter_d is None:
            inter_d = split_size
        elif inter_d != split_size:
            raise Reject(f"slot {slot}: split size {split_size} != first "
                         f"slot's {inter_d}")
        if int_const("slot %d gate idx" % slot, gate, gta[1]) != 0 or \
                int_const("slot %d up idx" % slot, up, upa[1]) != 1:
            raise Reject(f"slot {slot}: getitem order must be (gate=0, "
                         "up=1)")
        udims = static_dims(f"slot {slot} up", nmeta(up["name"]), 2)
        if gdims != [1, inter_d] or udims != [1, inter_d]:
            raise Reject(f"slot {slot}: gate|up shapes {gdims}/{udims} != "
                         f"[1, {inter_d}]")
        spm = nmeta(split["name"])
        if spm is None or spm["dtype"] != "seq" \
                or len(spm.get("items", [])) != 2 \
                or any(it["dtype"] != "float32" or it["shape"] != [1, inter_d]
                       for it in spm["items"]):
            raise Reject(f"slot {slot}: split ({split['name']}) must carry "
                         f"exactly 2 static f32 [1, {inter_d}] item "
                         f"metas, got {spm}")
        mdims = static_dims(f"slot {slot} activation", nmeta(act["name"]), 2)
        if mdims != [1, inter_d]:
            raise Reject(f"slot {slot}: activation {mdims} != [1, {inter_d}]")
        # GU linear <- shared x, gu select
        lin = ref_operand(f"slot {slot} gate|up producer", lin_ref)
        if lin["target"] != LINEAR:
            raise Reject(f"slot {slot}: gate|up producer must be {LINEAR}, "
                         f"got {lin['target']} ({lin['name']})")
        largs = lin["args"]
        if len(largs) == 3 and largs[2] is None:
            largs = largs[:2]
        if len(largs) != 2 or lin["kwargs"]:
            raise Reject(f"slot {slot}: GU {LINEAR} ({lin['name']}) must be "
                         "(input, weight) with no bias and no kwargs")
        if largs[0] != {"__node__": x_n["name"]}:
            raise Reject(f"slot {slot}: GU linear ({lin['name']}) input must "
                         "be the shared x placeholder")
        gu_sel = ref_operand(f"slot {slot} GU weight", largs[1])
        if gu_sel["target"] != SELECT:
            raise Reject(f"slot {slot}: GU weight must come from {SELECT}, "
                         f"got {gu_sel['target']} ({gu_sel['name']})")
        gsa = node_args("slot %d gu select" % slot, gu_sel, 3)
        if gsa[0] != {"__node__": gu_n["name"]} or int_const(
                "slot %d gu select dim" % slot, gu_sel, gsa[1]) != 0:
            raise Reject(f"slot {slot}: GU select ({gu_sel['name']}) must be "
                         f"{gu_n['name']}[0, slot]")
        gu_idx = int_const("slot %d gu select idx" % slot, gu_sel, gsa[2])
        lud = static_dims(f"slot {slot} gate|up linear output",
                          nmeta(lin["name"]), 2)
        if lud != [1, gu_out_d]:
            raise Reject(f"slot {slot}: GU linear output {lud} != "
                         f"[1, {gu_out_d}]")
        gsd = static_dims(f"slot {slot} gu weight", nmeta(gu_sel["name"]), 2)
        if gsd != [gu_out_d, hidden_d]:
            raise Reject(f"slot {slot}: GU weight slice {gsd} != "
                         f"[{gu_out_d}, {hidden_d}]")
        if gu_idx != slot or dn_idx != slot:
            raise Reject(f"slot {slot}: branch selects ({gu_idx}, {dn_idx}) "
                         "are not the exact ordered slot mapping")
        consumed.update([act["name"], gelu["name"], gate["name"],
                         up["name"], split["name"], lin["name"],
                         gu_sel["name"]])
        trace.append({"slot": slot, "guLinear": lin["name"],
                      "split": split["name"], "gelu": gelu["name"],
                      "mul": act["name"], "dnLinear": final["name"]})

    # ---- every call node consumed, exact census, no strays.
    call_nodes = [n for n in norm if n["op"] == "call_function"]
    unconsumed = [n["name"] for n in call_nodes if n["name"] not in consumed]
    if unconsumed:
        raise Reject(f"unconsumed/extra graph nodes: {unconsumed}")
    counts = {}
    for n in call_nodes:
        counts[n["target"]] = counts.get(n["target"], 0) + 1
    expect = {SELECT: 2 * topk_d, LINEAR: 2 * topk_d, SPLIT: topk_d,
              GELU: topk_d, MUL: topk_d, STACK: 1, GETITEM: 2 * topk_d}
    if counts != expect:
        raise Reject(f"ATen node census {counts} != expected {expect}")

    if 2 * inter_d != gu_out_d:
        raise Reject(f"gu_weights out {gu_out_d} != 2*inter {2 * inter_d}")
    if dn_in_d != inter_d:
        raise Reject(f"down_weights in-dim {dn_in_d} != inter {inter_d}")
    for label, v in (("hidden", hidden_d), ("inter", inter_d),
                     ("topK", topk_d), ("gateUpOut", gu_out_d)):
        if not (0 < v <= MAX_DIM) or (label in ("hidden", "inter")
                                      and v % 32 != 0):
            raise Reject(f"{label}={v} out of bounds/not a 32-multiple")
    if topk_d > MAX_TOPK:
        raise Reject(f"topK {topk_d} exceeds {MAX_TOPK}")

    # ---- canonical digest over the normalized graph payload.
    lines = []
    for n in norm:
        meta = n["meta"]
        meta_c = "n" if meta is None else canon(meta)
        lines.append("|".join([n["name"], n["op"], n["target"],
                               canon(n["args"]), canon(n["kwargs"]), meta_c]))
    lines.append("inputs|" + canon([p["name"] for p in placeholders]))
    lines.append("outputs|" + canon([outputs[0]["name"]]))
    payload = "\n".join(lines)
    digest = hashlib.sha256(payload.encode()).hexdigest()

    graph = {"nodes": norm,
             "inputs": [p["name"] for p in placeholders],
             "outputs": [outputs[0]["name"]]}
    ops = sorted({n["target"] for n in call_nodes})
    return {
        "schema": 1,
        "region": "gemma4.moe.routed-region",
        "hidden": hidden_d,
        "expertIntermediate": inter_d,
        "topK": topk_d,
        "gateUpOut": gu_out_d,
        "geluApprox": "tanh",
        "stackOutputs": True,
        "slotOps": ["q4_0.expert-slice-linear-gate-up",
                    "gelu-tanh-split-mul",
                    "q4_0.expert-slice-linear-down"],
        "guTensor": "blk.0.ffn_gate_up_exps.weight",
        "dnTensor": "blk.0.ffn_down_exps.weight",
        "graph": graph,
        "graphSha256": digest,
        "torch": {
            "advisoryOnly": True,
            "strict": True,
            "version": torch.__version__,
            "atenOps": ops,
        },
    }, payload


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_path, out_path = sys.argv[1:3]
    config = None
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            config = json.load(f)
    spec = importlib.util.spec_from_file_location("user_moe_region", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    module = mod.build(config)
    for attr in ("hidden", "inter", "top_k"):
        v = getattr(module, attr, None)
        if type(v) is not int:
            print(f"REJECT: module attribute {attr} must be a static int, "
                  f"got {v!r}", file=sys.stderr)
            return 1
    if not (0 < module.hidden <= MAX_DIM) or module.hidden % 32 != 0 \
            or not (0 < module.inter <= MAX_DIM) or module.inter % 32 != 0 \
            or not (0 < module.top_k <= MAX_TOPK):
        print(f"REJECT: module geometry hidden={module.hidden} "
              f"inter={module.inter} top_k={module.top_k} out of bounds "
              "(positive, <=8192, 32-multiples, topK <=64)", file=sys.stderr)
        return 1
    try:
        region, _ = export_and_normalize(module)
    except Reject as e:
        print(f"REJECT: {e}", file=sys.stderr)
        return 1
    with open(out_path, "w") as f:
        json.dump(region, f, ensure_ascii=False, sort_keys=True, indent=1)
    print(f"captured {region['region']} hidden={region['hidden']} "
          f"inter={region['expertIntermediate']} topK={region['topK']} "
          f"gateUpOut={region['gateUpOut']} graphSha256="
          f"{region['graphSha256'][:16]}… -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
