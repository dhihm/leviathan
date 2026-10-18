"""Serializes a torch.export ExportedProgram of OrnithModel into the model IR (JSON).

    python ir_export.py <out.json> [--small]

Format "lev-model-ir-1":
  inputs   user inputs in call order: name, dtype, shape
  params   parameters by Hugging Face name: dtype, shape (values are NOT in the IR; the runtime
           binds them to the GGUF tensors, or to a weights file in tests)
  nodes    in topological order; node id = index. Each node: op, args, out (output shapes)
  outputs  list of arguments
An argument is one of {"n": node id}, {"p": param name}, {"i": input name}, {"int": v},
{"sym": "expr"} (a symbolic size such as "s77" or "32*s77"), {"f": v}, {"b": v}, {"str": s},
{"dtype": "float32"}, {"none": 1}, {"list": [arguments]}.
Shapes are lists of ints or symbolic expressions as strings; T (the token count) is symbolic.
"""
import json
import sys
from collections import Counter

import torch
from torch.export import Dim, export

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from ornith_model import OrnithModel, OrnithModelRagged

FORMAT = "lev-model-ir-1"


def real_config():
    cfg = Qwen3_5TextConfig(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 8,
        rope_parameters={"rope_type": "default", "rope_theta": 1.0e7, "partial_rotary_factor": 0.25},
    )
    cfg._attn_implementation = "eager"
    return cfg


def small_config():
    cfg = Qwen3_5TextConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    )
    cfg._attn_implementation = "eager"
    return cfg


def tiny_config():
    """Smallest model with all layer kinds, for committed test fixtures."""
    cfg = Qwen3_5TextConfig(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dim(d):
    return int(d) if isinstance(d, int) else str(d)


def _shape(v):
    if isinstance(v, torch.Tensor):
        return [_dim(d) for d in v.shape]
    if isinstance(v, (tuple, list)):
        return [_shape(x) for x in v]
    return None


def _dtype(t):
    return str(t).replace("torch.", "")


def _arg(a, names, params, inputs):
    if isinstance(a, torch.fx.Node):
        if a.name in params:
            return {"p": params[a.name]}
        if a.name in inputs:
            return {"i": inputs[a.name]}
        return {"n": names[a.name]}
    if a is None:
        return {"none": 1}
    if isinstance(a, bool):
        return {"b": a}
    if isinstance(a, int):
        return {"int": a}
    if isinstance(a, float):
        return {"f": a}
    if isinstance(a, str):
        return {"str": a}
    if isinstance(a, torch.dtype):
        return {"dtype": _dtype(a)}
    if isinstance(a, (torch.device, torch.layout, torch.memory_format)):
        return {"str": str(a)}
    if isinstance(a, (list, tuple)):
        return {"list": [_arg(x, names, params, inputs) for x in a]}
    if isinstance(a, torch.SymInt):
        return {"sym": str(a)}
    raise TypeError(f"unsupported argument {a!r} ({type(a).__name__})")


def _op_name(target):
    """aten.linear.default -> aten::linear.default, lev.rope.default -> lev::rope.default, operator.getitem -> getitem."""
    if not isinstance(target, torch._ops.OpOverload):
        return getattr(target, "__name__", str(target))
    return str(target).replace("aten.", "aten::", 1).replace("lev.", "lev::", 1)


def serialize(ep, cfg):
    sig = ep.graph_signature
    params = dict(sig.inputs_to_parameters)  # placeholder name -> parameter name
    user_inputs = list(sig.user_inputs)
    inputs = {n: n for n in user_inputs}
    ir = {"format": FORMAT, "config": {k: v for k, v in cfg.to_dict().items() if isinstance(v, (int, str, bool)) or (isinstance(v, float) and v == v and abs(v) != float("inf"))}, "inputs": [], "params": [], "nodes": [], "outputs": []}
    names = {}
    for n in ep.graph.nodes:
        if n.op == "placeholder":
            v = n.meta.get("val")
            if n.name in params:
                ir["params"].append({"name": params[n.name], "dtype": _dtype(v.dtype), "shape": _shape(v)})
            elif n.name in inputs:
                ir["inputs"].append({"name": n.name, "dtype": _dtype(v.dtype), "shape": _shape(v)})
        elif n.op == "call_function":
            names[n.name] = len(ir["nodes"])
            ir["nodes"].append(
                {
                    "op": _op_name(n.target),
                    "args": [_arg(a, names, params, inputs) for a in n.args],
                    "kwargs": {k: _arg(v, names, params, inputs) for k, v in n.kwargs.items()},
                    "out": _shape(n.meta.get("val")),
                }
            )
        elif n.op == "output":
            ir["outputs"] = [_arg(a, names, params, inputs) for a in n.args[0]]
    return ir


def build_and_export(cfg, meta=True):
    """Exports OrnithModel with a symbolic token count and returns (model, ExportedProgram)."""
    t = Dim("T", min=1, max=4096)
    s = Dim("S", min=0, max=32768)
    ctx = torch.device("meta") if meta else torch.device("cpu")
    with ctx:
        model = OrnithModel(cfg).eval()
        rd = int(cfg.head_dim * (cfg.rope_parameters.get("partial_rotary_factor", 1.0)))
        ex = [torch.zeros(3, dtype=torch.long), torch.empty(3, rd), torch.empty(3, rd)]
        dyn = {"token_ids": {0: t}, "cos": {0: t}, "sin": {0: t}}
        for i, lay in enumerate(model.state_layout()):
            if lay[0] == "linear":
                ex += [torch.empty(*lay[1]), torch.empty(*lay[2])]
                dyn[f"s{2 * i}"] = None
                dyn[f"s{2 * i + 1}"] = None
            else:
                ex += [torch.empty(2, *lay[1][1:]), torch.empty(2, *lay[1][1:])]
                dyn[f"s{2 * i}"] = {0: s}
                dyn[f"s{2 * i + 1}"] = {0: s}
    # *states are positional varargs: give them names s0.. through dynamic_shapes ordering
    ep = export(model, tuple(ex), dynamic_shapes=_ordered(model, dyn))
    return model, ep


def _ordered(model, dyn):
    """dynamic_shapes as a tuple in call order (token_ids, cos, sin, *states)."""
    head = [dyn["token_ids"], dyn["cos"], dyn["sin"]]
    states = []
    for i in range(len(model.state_layout())):
        states += [dyn[f"s{2 * i}"], dyn[f"s{2 * i + 1}"]]
    # The varargs *states count as one argument: a tuple of per-state specs.
    return tuple(head + [tuple(states)])


def build_and_export_ragged(cfg, meta=True):
    """Exports OrnithModelRagged with symbolic token count T, slot count S and slab length L."""
    t = Dim("T", min=1, max=4096)
    s = Dim("S", min=1, max=1024)
    l = Dim("L", min=1, max=262144)
    ctx = torch.device("meta") if meta else torch.device("cpu")
    with ctx:
        model = OrnithModelRagged(cfg).eval()
        rd = int(cfg.head_dim * (cfg.rope_parameters.get("partial_rotary_factor", 1.0)))
        ex = [torch.zeros(3, dtype=torch.long), torch.empty(3, rd), torch.empty(3, rd), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
        dyn = [{0: t}, {0: t}, {0: t}, {0: t}, {0: t}]
        pools = []
        for lay in model.pool_layout(2, 4):
            if lay[0] == "linear":
                ex += [torch.empty(*lay[1]), torch.empty(*lay[2])]
                pools += [{0: s}, {0: s}]
            else:
                ex += [torch.empty(*lay[1]), torch.empty(*lay[1])]
                pools += [{0: s, 1: l}, {0: s, 1: l}]
    ep = export(model, tuple(ex), dynamic_shapes=tuple(dyn + [tuple(pools)]))
    return model, ep


def main():
    out = sys.argv[1]
    small = "--small" in sys.argv
    ragged = "--ragged" in sys.argv
    cfg = small_config() if small else real_config()
    model, ep = (build_and_export_ragged if ragged else build_and_export)(cfg, meta=not small)
    ir = serialize(ep, cfg)
    ir["layout"] = "ragged-v1" if ragged else "single-v1"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ir, f)
    ops = Counter(n["op"] for n in ir["nodes"])
    print(f"{len(ir['nodes'])} nodes, {len(ir['params'])} parameters, {len(ir['inputs'])} inputs, {len(ir['outputs'])} outputs")
    for op, c in ops.most_common():
        print(f"  {c:5d}  {op}")


if __name__ == "__main__":
    main()
