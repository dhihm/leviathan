#!/usr/bin/env python3
"""torch.export(strict=True) graph frontend for LeviathanForge.

Unlike examples/export_torch.py (which walks an exact ``nn.Sequential`` of
``nn.Linear``/``nn.ReLU`` children), this exporter accepts a user-authored
``nn.Module`` with an ARBITRARY ``forward`` and captures its semantics with
``torch.export.export(..., strict=True)``. The exported ATen graph is then
validated against the supported subset and lowered to the same files the
existing imported-weight Forge path consumes:

    <prefix>.graph.json     GraphIR architecture JSON (dense/relu chain)
    <prefix>.weights.f32    per dense op: W transposed to [in, out] row-major,
                            then bias — float32 little-endian
    <prefix>.manifest.json  format "leviathan-forge-weights-1" (identical
                            validated contract to the Sequential exporter,
                            plus a clearly-labeled ADVISORY "torchExport"
                            metadata block — see below)

Supported subset (anything else FAILS CLOSED with the offending node named
— nothing is written on rejection):
- exactly one user input, rank 2, shape [1, INPUT_DIM], float32, static
  (every dimension a concrete int — dynamic/symbolic shapes are rejected);
- exactly one user-output spec (no mutations of parameters or buffers);
  the FX output node must unwrap to exactly one value;
- call nodes limited to ``aten.linear.default`` (exactly 3 positional args:
  input, weight parameter, bias parameter) and ``aten.relu.default``
  (``F.relu``/``torch.relu``/``nn.ReLU`` all export to this one op);
- a single linear chain: the first call consumes the user input, every later
  call consumes exactly the previous call's value, every intermediate has
  exactly one consumer, and the final call is the sole output — branching,
  residuals (``aten.add.Tensor``), and unused side values are rejected, and
  every call's metadata (dtype/rank/static dims) must match the expected
  chain shape;
- parameters only (exported buffers/constants/mutations rejected by name),
  float32, CPU, finite, unshared (no shared modules or parameters — checked
  without deduplication), no hooks or parametrizations.

Data-dependent control flow: ``torch.export`` refuses unbacked data-dependent
branching (a strict-export error is itself a fail-closed rejection), but if
export SPECIALIZES a branch on the example input, the bounded differential
gate below (original module vs exported graph on 8 seeded probes, exact
equality) rejects the observed divergence. Those 8 probes are BOUNDED parity
evidence, not a proof that every specialized branch is detected. The
deterministic manifest reference output is computed by the ORIGINAL PyTorch
module on the rank-2 [1, INPUT_DIM] contract input.

The "torchExport" manifest block is ADVISORY metadata only: the Swift
loader ignores it, and no identity or authentication claim is derived from
it. Artifact identity remains exactly what the validated weights-1 manifest
already carries: the SHA-256-bound weight blob, graph, and op table, which
the Swift side re-validates and which the persisted Forge plan binds.

Model file contract: defines ``INPUT_DIM`` (int in 1..4096) and
``build(seed)`` returning the ``nn.Module`` (eval mode is applied here).
Executing the model file is explicit trusted local Python — no sandboxing is
claimed.

Usage:
    python3 export_torch_graph.py model_def.py out_prefix [seed]
"""

import hashlib
import json
import struct
import sys

import torch
from torch import nn
from torch.export.graph_signature import InputKind, OutputKind

MAX_OPS = 64
MAX_DIM = 4096
MAX_WEIGHT_BYTES = 16 << 20
REF_SEED = 7
SEMANTICS_INPUTS = 8

# Exact OpOverload identifiers (str(OpOverload) is "aten.<op>.<overload>").
LINEAR = "aten.linear.default"
RELU = "aten.relu.default"


def is_real_int(value):
    return type(value) is int


def fail(message):
    raise SystemExit(f"export_torch_graph: {message}")


def op_name(node):
    """Canonical ATen overload id, e.g. 'aten.linear.default'."""
    return str(node.target)


def expect_static_tensor(node, rank, shape, dtype):
    """Every exported value must be a concrete (non-symint) float32 tensor
    of the exact expected shape."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        fail(f"node '{node.name}' has no tensor metadata (non-tensor value)")
    dims = tuple(val.shape)
    if any(type(d) is not int for d in dims):
        fail(f"non-static (dynamic/symbolic) shapes are not supported "
             f"('{node.name}' has shape {dims})")
    if len(dims) != rank or dims != shape:
        fail(f"node '{node.name}' shape {dims} != required {shape}")
    if val.dtype is not dtype:
        fail(f"node '{node.name}' dtype {val.dtype} != {dtype}")


def validate_module_side_conditions(model):
    """Fail closed on module-level conditions torch.export would otherwise
    inline silently (hooks become call nodes, shared parameters alias).
    named_modules/named_parameters DEDUPLICATE by default, so duplicates are
    requested explicitly: a shared object appears twice in the traversal."""
    for name, module in model.named_modules(remove_duplicate=False):
        for hook_dict in (module._forward_hooks, module._forward_pre_hooks,
                          module._backward_hooks, module._backward_pre_hooks):
            if hook_dict:
                fail(f"hooks are not allowed (module '{name or '(root)'}' "
                     f"of type {type(module).__name__} has {len(hook_dict)} "
                     "attached hook(s))")
        if getattr(module, "_parametrizations", None):
            fail(f"parametrized module '{name}' is not allowed")
    seen_modules = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if id(module) in seen_modules:
            fail(f"shared module instance '{name}' "
                 f"({type(module).__name__}) is not allowed")
        seen_modules.add(id(module))
    seen_params = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen_params:
            fail(f"shared parameter '{name}' is not allowed")
        seen_params.add(id(param))
        if param.dtype is not torch.float32:
            fail(f"parameter '{name}' must be float32, got {param.dtype}")
        if param.device.type != "cpu":
            fail(f"parameter '{name}' must be on CPU, got {param.device}")
        if not torch.isfinite(param).all():
            fail(f"parameter '{name}' contains non-finite values")


def validate_and_lower(ep, input_dim):
    """Validate the exported ATen graph and lower it to the dense/relu chain.

    Returns (ops, aten_ops): ops entries are ("dense", in_dim, out_dim,
    weight_tensor, bias_tensor) or ("relu",).
    """
    # Output specs: exactly one USER_OUTPUT and nothing else — any mutated
    # parameter/buffer shows up as an additional output spec and is rejected.
    out_specs = ep.graph_signature.output_specs
    if len(out_specs) != 1:
        fail(f"exactly one output is required, got {len(out_specs)} "
             f"({', '.join(str(s.kind) for s in out_specs)})")
    if out_specs[0].kind is not OutputKind.USER_OUTPUT:
        fail(f"output kind {out_specs[0].kind} is not a user output "
             "(mutations are not supported)")

    # Input specs: exactly one USER_INPUT, all others PARAMETER. Buffers,
    # constants, and token specs are rejected by name.
    param_target_by_node = {}
    user_inputs = []
    for spec in ep.graph_signature.input_specs:
        if spec.kind is InputKind.PARAMETER:
            param_target_by_node[spec.arg.name] = spec.target
        elif spec.kind is InputKind.USER_INPUT:
            user_inputs.append(spec.arg.name)
        else:
            fail(f"unsupported exported input kind {spec.kind} "
                 f"('{spec.arg.name}' -> {spec.target}): only parameters and "
                 "one user input are accepted")
    if len(user_inputs) != 1:
        fail(f"exactly one user input is required, got {len(user_inputs)}")

    state = dict(ep.state_dict)
    nodes = list(ep.graph_module.graph.nodes)

    # Placeholders, one call chain, one output node — nothing else.
    calls = [n for n in nodes if n.op == "call_function"]
    outputs = [n for n in nodes if n.op == "output"]
    if len(outputs) != 1:
        fail(f"exactly one output node is required, got {len(outputs)}")
    for n in nodes:
        if n.op not in ("placeholder", "call_function", "output"):
            fail(f"unsupported graph construct {n.op} ('{n.name}')")

    # The FX output node packs even a single tensor in a one-element tuple;
    # more than one element is a genuine multiple-output model.
    out_values = outputs[0].args[0]
    if not isinstance(out_values, (tuple, list)) or len(out_values) != 1:
        fail("multiple outputs are not supported (model must return a "
             f"single tensor, got {len(out_values) if isinstance(out_values, (tuple, list)) else 'non-sequence'} outputs")

    input_node = next((n for n in nodes
                       if n.op == "placeholder" and n.name == user_inputs[0]),
                      None)
    if input_node is None:
        fail("user input placeholder missing from the graph")

    def node_param_tensor(node):
        target = param_target_by_node.get(node.name)
        if target is None:
            fail(f"linear operand '{node.name}' is not a parameter "
                 "(constants or computed weights are not accepted)")
        if target not in state:
            fail(f"parameter '{target}' missing from exported state dict")
        return state[target]

    # Walk the chain in graph order.
    current = None            # node producing the chain value
    current_dim = input_dim
    ops = []
    aten_ops = []
    total_floats = 0
    for node in calls:
        aten_id = op_name(node)
        aten_ops.append(aten_id)
        if aten_id == LINEAR:
            if node.kwargs:
                fail(f"linear at '{node.name}' has unexpected keyword "
                     f"arguments {sorted(node.kwargs)} — silently ignoring "
                     "values is not allowed")
            args = list(node.args)
            if len(args) != 3:
                fail(f"linear at '{node.name}' must take exactly "
                     "(input, weight, bias); got {len(args)} args")
            x_node, w_node, b_node = args
            if current is None:
                if x_node is not input_node:
                    fail(f"the first op must consume the user input, "
                         f"'{node.name}' consumes '{x_node.name}'")
            elif x_node is not current:
                fail(f"unsupported graph structure at '{node.name}': input "
                     f"'{x_node.name}' is not the chain output "
                     f"'{current.name}' (branching/residual not supported)")
            for operand in (w_node, b_node):
                if operand.op != "placeholder":
                    fail(f"linear operand '{operand.name}' at '{node.name}' "
                         "must be an exported parameter placeholder")
            w = node_param_tensor(w_node)
            b = node_param_tensor(b_node)
            if w.dim() != 2 or b.dim() != 1:
                fail(f"linear at '{node.name}': weight must be rank 2 and "
                     f"bias rank 1, got {tuple(w.shape)}/{tuple(b.shape)}")
            out_dim_t, in_dim_t = w.shape
            if type(in_dim_t) is not int or type(out_dim_t) is not int:
                fail(f"linear at '{node.name}': non-static parameter shape")
            in_dim, out_dim = int(in_dim_t), int(out_dim_t)
            if in_dim != current_dim:
                fail(f"linear at '{node.name}': in_features {in_dim} != "
                     f"chain dim {current_dim}")
            for d in (in_dim, out_dim):
                if not (0 < d <= MAX_DIM):
                    fail(f"linear at '{node.name}': dimension {d} outside "
                         f"1...{MAX_DIM}")
            if tuple(b.shape) != (out_dim,):
                fail(f"linear at '{node.name}': bias shape {tuple(b.shape)} "
                     f"!= [{out_dim}]")
            for tensor, label in ((w, "weight"), (b, "bias")):
                if tensor.dtype is not torch.float32:
                    fail(f"linear {label} at '{node.name}' must be float32, "
                         f"got {tensor.dtype}")
                if not torch.isfinite(tensor).all():
                    fail(f"linear {label} at '{node.name}' is non-finite")
            expect_static_tensor(node, 2, (1, out_dim), torch.float32)
            total_floats += w.numel() + b.numel()
            ops.append(("dense", in_dim, out_dim, w, b))
            current, current_dim = node, out_dim
        elif aten_id == RELU:
            if node.kwargs:
                fail(f"relu at '{node.name}' has unexpected keyword "
                     "arguments")
            if len(node.args) != 1 or node.args[0] is not current:
                fail(f"unsupported graph structure at '{node.name}': relu "
                     "must apply to the chain output")
            expect_static_tensor(node, 2, (1, current_dim), torch.float32)
            ops.append(("relu",))
            current = node
        else:
            fail(f"unsupported ATen op '{aten_id}' at node '{node.name}': "
                 "the supported subset is aten.linear.default (with bias) "
                 "and aten.relu.default")
    if not ops:
        fail("exported graph contains no operations")
    if len(ops) > MAX_OPS:
        fail(f"op count {len(ops)} exceeds {MAX_OPS}")
    if total_floats * 4 > MAX_WEIGHT_BYTES:
        fail(f"total weights {total_floats * 4} bytes exceed {MAX_WEIGHT_BYTES}")

    expect_static_tensor(input_node, 2, (1, input_dim), torch.float32)
    # The chain's final node must be the sole output, and every intermediate
    # must have exactly one consumer (no branches, no unused side values).
    if calls and out_values[0] is not calls[-1]:
        fail("model output is not the final chain value (side outputs or "
             "multiple outputs are not supported)")
    for node in calls[:-1]:
        if len(node.users) != 1:
            fail(f"value '{node.name}' has {len(node.users)} consumers; the "
                 "supported subset is a single linear/relu chain")
    # Every parameter placeholder must be consumed exactly once by a linear.
    used_params = set()
    for node in calls:
        if op_name(node) == LINEAR:
            for operand in (node.args[1], node.args[2]):
                if operand.name in used_params:
                    fail(f"parameter operand '{operand.name}' is used by "
                         "more than one linear (shared weights not supported)")
                used_params.add(operand.name)
    return ops, aten_ops


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    model_path, prefix = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
        "leviathan_torch_export_model", model_path)
    module = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build"):
        fail(f"{model_path}: must define build(seed)")
    if not is_real_int(getattr(module, "INPUT_DIM", None)):
        fail(f"{model_path}: must define INPUT_DIM as an int in 1...{MAX_DIM}")
    input_dim = module.INPUT_DIM
    if not (0 < input_dim <= MAX_DIM):
        fail(f"INPUT_DIM {input_dim} outside 1...{MAX_DIM}")
    model = module.build(seed)
    if not isinstance(model, nn.Module):
        fail("build(seed) must return an nn.Module")
    model = model.eval()
    validate_module_side_conditions(model)

    # Export with a static rank-2 example input; strict=True forbids graph
    # breaks. Unsupported data-dependent control flow makes export itself
    # fail — that failure is a fail-closed rejection below.
    generator = torch.Generator().manual_seed(REF_SEED)
    example = torch.rand((1, input_dim), generator=generator,
                         dtype=torch.float32)
    try:
        ep = torch.export.export(model, (example,), strict=True)
    except Exception as exc:  # any export failure is a fail-closed boundary
        fail(f"torch.export failed (unsupported model): "
             f"{type(exc).__name__}: {exc}")

    ops, aten_ops = validate_and_lower(ep, input_dim)

    # Bounded differential parity gate: ep.module() preserves the ORIGINAL
    # calling convention; the exported graph must reproduce the module
    # exactly on several seeded probes. If export specialized a
    # data-dependent branch on the example input, the graphs diverge on some
    # probe and are rejected. (Bounded evidence over 8 probes — not a proof
    # that every specialized branch is caught.)
    exported = ep.module()
    for k in range(SEMANTICS_INPUTS):
        g = torch.Generator().manual_seed(REF_SEED + 100 + k)
        probe = torch.rand((1, input_dim), generator=g, dtype=torch.float32)
        with torch.no_grad():
            want = model(probe)
            got = exported(probe)
        if not (isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor)):
            fail(f"probe {k}: module/graph returned non-tensor outputs "
                 f"({type(want).__name__}/{type(got).__name__})")
        if want.shape != got.shape or want.dtype is not torch.float32 \
                or got.dtype is not torch.float32 \
                or not torch.isfinite(want).all() or not torch.isfinite(got).all():
            fail(f"probe {k}: output shape/dtype/finiteness mismatch "
                 f"({tuple(want.shape)}/{want.dtype} vs "
                 f"{tuple(got.shape)}/{got.dtype})")
        if (want - got).abs().max().item() != 0.0:
            fail(f"exported graph disagrees with the original module on "
                 f"probe input {k} (max abs diff "
                 f"{(want - got).abs().max().item():.3e}) — data-dependent "
                 "control flow or other unsupported semantics")

    # Architecture JSON in GraphIR format.
    graph_ops = []
    for op in ops:
        if op[0] == "relu":
            graph_ops.append("relu")
        else:
            graph_ops.append({"dense": {"outDim": op[2]}})
    graph = {"inputDim": input_dim, "ops": graph_ops}

    # Weight blob: per dense op, transposed W [in, out] row-major, then bias.
    blob = bytearray()
    op_table = []
    for op in ops:
        if op[0] != "dense":
            continue
        _, in_dim, out_dim, w, b = op
        wt = w.detach().to("cpu", torch.float32).t().contiguous().flatten()
        bt = b.detach().to("cpu", torch.float32).flatten()
        op_table.append({
            "inDim": in_dim, "outDim": out_dim,
            "weightFloatOffset": len(blob) // 4,
            "weightFloatCount": wt.numel(),
            "biasFloatOffset": (len(blob) + wt.numel() * 4) // 4,
        })
        blob += struct.pack(f"<{wt.numel()}f", *wt.tolist())
        blob += struct.pack(f"<{bt.numel()}f", *bt.tolist())

    # Deterministic reference computed by the ORIGINAL PyTorch module on the
    # rank-2 contract input; the manifest stores the flat 1-D vectors the
    # Swift weights-1 loader validates.
    generator = torch.Generator().manual_seed(REF_SEED)
    ref_input = torch.rand(input_dim, generator=generator,
                           dtype=torch.float32).tolist()
    output_dim = input_dim
    for op in ops:
        if op[0] == "dense":
            output_dim = op[2]
    with torch.no_grad():
        ref_out = model(torch.tensor(ref_input, dtype=torch.float32)
                        .unsqueeze(0))
    if not (isinstance(ref_out, torch.Tensor) and ref_out.dtype is torch.float32
            and tuple(ref_out.shape) == (1, output_dim)
            and torch.isfinite(ref_out).all()):
        fail("reference output is not a finite float32 [1, outDim] tensor")
    ref_output = ref_out.flatten().tolist()

    graph_json = json.dumps(graph, separators=(",", ":"))
    manifest = {
        "format": "leviathan-forge-weights-1",
        "inputDim": input_dim,
        "floatCount": len(blob) // 4,
        "weightsSha256": hashlib.sha256(blob).hexdigest(),
        "ops": op_table,
        "refInputSeed": REF_SEED,
        "refInput": ref_input,
        "refOutput": ref_output,
        "modelSeed": seed,
        # ADVISORY metadata only: the Swift loader ignores this block and no
        # identity or authentication is derived from it. Artifact identity
        # stays the validated weights-1 contract (weightsSha256 + graph +
        # op table), which Swift re-validates and the Forge plan binds.
        "torchExport": {
            "advisoryOnly": True,
            "frontend": "export_torch_graph.py",
            "strict": True,
            "torch": torch.__version__,
            "atenOps": aten_ops,
            "graphSha256": hashlib.sha256(graph_json.encode()).hexdigest(),
            "weightsSha256": hashlib.sha256(blob).hexdigest(),
            "semanticsProbes": SEMANTICS_INPUTS,
            "modelSource": model_path,
        },
    }

    # All validation done — only now is anything written.
    with open(prefix + ".graph.json", "w") as f:
        f.write(graph_json)
    with open(prefix + ".weights.f32", "wb") as f:
        f.write(blob)
    with open(prefix + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"wrote {prefix}.graph.json ({len(graph_ops)} ops from torch.export "
          f"strict graph: {' -> '.join(aten_ops)}), {prefix}.weights.f32 "
          f"({len(blob)} bytes, sha256 {manifest['weightsSha256'][:16]}…), "
          f"{prefix}.manifest.json with module-computed reference output")


if __name__ == "__main__":
    main()
