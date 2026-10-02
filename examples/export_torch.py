#!/usr/bin/env python3
"""Restricted PyTorch model exporter for LeviathanForge.

Contract (deliberately narrow and validated, not a generic compiler):
- The model file defines ``build(seed)`` returning an object whose exact
  type is ``nn.Sequential`` containing ONLY modules whose exact types are
  ``nn.Linear`` (float32, on CPU, with bias) and ``nn.ReLU``, in plain
  sequential order. Subclasses are rejected (a subclass can override
  ``forward`` with arbitrary control flow). Any forward/pre/backward
  hooks, parametrizations, shared modules, or shared parameters are
  rejected. Adjacent shapes must be consistent and the first module must
  be a Linear.
- At most 64 ops. Dimensions must be real ints (bool is explicitly
  rejected — in Python ``isinstance(True, int)`` is true). All weights
  and biases must be finite float32; total weight bytes must fit 16 MiB.
- ``nn.Linear`` stores weight as [out, in]; Leviathan's dense op expects
  row-major [in, out], so weights are transposed on export.
- Outputs (all plain files, explicitly written by the user's command):
    <prefix>.graph.json    GraphIR architecture JSON
    <prefix>.weights.f32   concatenated float32 little-endian weights:
                           per dense op, W (in*out) then bias (out)
    <prefix>.manifest.json content SHA-256 of the weights file, op table,
                           and a deterministic input + reference output
                           computed by this same PyTorch model.

Usage:
    python3 export_torch.py model_def.py out_prefix [seed]
"""

import hashlib
import json
import struct
import sys

import torch
from torch import nn

MAX_OPS = 64
MAX_DIM = 4096
MAX_WEIGHT_BYTES = 16 << 20
REF_SEED = 7


def is_real_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def reject_attached(module):
    for hook_dict in (module._forward_hooks, module._forward_pre_hooks,
                      module._backward_hooks, module._backward_pre_hooks):
        if hook_dict:
            raise SystemExit("hooks are not allowed in the export contract")
    if getattr(module, "_parametrizations", None):
        raise SystemExit("parametrized modules are not allowed")


def validate_sequential(model):
    if type(model) is not nn.Sequential:
        raise SystemExit("build(seed) must return exactly nn.Sequential")
    if not len(model):
        raise SystemExit("Sequential is empty")
    for module in model:
        reject_attached(module)
    if type(model[0]) is not nn.Linear:
        raise SystemExit("the first module must be nn.Linear (explicit input dim)")
    ops = []
    out_dim = None
    seen_module_ids = set()
    seen_param_ids = set()
    total_floats = 0
    for module in model:
        if type(module) is nn.Linear:
            for name in ("in_features", "out_features"):
                v = getattr(module, name)
                if not is_real_int(v) or not (0 < v <= MAX_DIM):
                    raise SystemExit(f"Linear.{name} must be an int in 1...{MAX_DIM}, got {v!r}")
            if out_dim is not None and module.in_features != out_dim:
                raise SystemExit(f"Linear in_features {module.in_features} != previous out {out_dim}")
            if module.bias is None:
                raise SystemExit("Linear layers must have biases (contract)")
            for tensor in (module.weight, module.bias):
                if tensor.dtype != torch.float32:
                    raise SystemExit("Linear parameters must be float32")
                if tensor.device.type != "cpu":
                    raise SystemExit("Linear parameters must be on CPU")
                if tensor.device != torch.device("cpu"):
                    raise SystemExit("Linear parameters must be plain CPU tensors")
                if not torch.isfinite(tensor).all():
                    raise SystemExit("Linear parameters must be finite")
            if module.weight.shape != (module.out_features, module.in_features) \
                    or module.bias.shape != (module.out_features,):
                raise SystemExit("Linear parameter shapes inconsistent")
            total_floats += module.weight.numel() + module.bias.numel()
            ops.append(("dense", module))
            out_dim = module.out_features
        elif type(module) is nn.ReLU:
            ops.append(("relu", None))
        else:
            raise SystemExit(f"unsupported module {type(module).__name__}: "
                             "only nn.Linear and nn.ReLU")
    if not ops or len(ops) > MAX_OPS:
        raise SystemExit(f"op count must be in 1...{MAX_OPS}")
    if ops[-1][0] != "dense":
        raise SystemExit("model must end with a Linear (dense) op")
    # Shared modules or parameters would alias weights — rejected.
    for module in model.modules():
        if id(module) in seen_module_ids:
            raise SystemExit("shared modules are not allowed")
        seen_module_ids.add(id(module))
    for param in model.parameters():
        if id(param) in seen_param_ids:
            raise SystemExit("shared parameters are not allowed")
        seen_param_ids.add(id(param))
    if total_floats * 4 > MAX_WEIGHT_BYTES:
        raise SystemExit(f"total weights exceed {MAX_WEIGHT_BYTES} bytes")
    return ops


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    model_path, prefix = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
        "leviathan_torch_model", model_path)
    module = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build"):
        raise SystemExit(f"{model_path}: must define build(seed)")
    model = module.build(seed)
    ops = validate_sequential(model)

    input_dim = model[0].in_features
    # Architecture JSON in GraphIR format.
    graph_ops = []
    for kind, module in ops:
        if kind == "relu":
            graph_ops.append("relu")
        else:
            graph_ops.append({"dense": {"outDim": module.out_features}})
    graph = {"inputDim": input_dim, "ops": graph_ops}

    # Weight blob: per dense op, transposed W [in, out] row-major, then
    # bias. detach().cpu() is explicit even though validation requires CPU.
    blob = bytearray()
    op_table = []
    for kind, module in ops:
        if kind != "dense":
            continue
        w = module.weight.detach().to("cpu", torch.float32).t().contiguous().flatten()
        b = module.bias.detach().to("cpu", torch.float32).flatten()
        op_table.append({
            "inDim": module.in_features, "outDim": module.out_features,
            "weightFloatOffset": len(blob) // 4,
            "weightFloatCount": w.numel(),
            "biasFloatOffset": (len(blob) + w.numel() * 4) // 4,
        })
        blob += struct.pack(f"<{w.numel()}f", *w.tolist())
        blob += struct.pack(f"<{b.numel()}f", *b.tolist())

    # Deterministic reference computed by the same PyTorch model.
    generator = torch.Generator().manual_seed(REF_SEED)
    ref_input = torch.rand(input_dim, generator=generator,
                           dtype=torch.float32).tolist()
    with torch.no_grad():
        ref_output = model(torch.tensor(ref_input, dtype=torch.float32)).tolist()

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
    }

    with open(prefix + ".graph.json", "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    with open(prefix + ".weights.f32", "wb") as f:
        f.write(blob)
    with open(prefix + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"wrote {prefix}.graph.json ({len(graph_ops)} ops), "
          f"{prefix}.weights.f32 ({len(blob)} bytes, sha256 {manifest['weightsSha256'][:16]}…), "
          f"{prefix}.manifest.json with reference output")


if __name__ == "__main__":
    main()
