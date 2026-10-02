#!/usr/bin/env python3
"""Model frontend for LeviathanForge (milestone 0 contract).

This is the documented, restricted Python model contract: a model file
declares an architecture as a plain Python dict with `inputDim` and an
`ops` list of {"dense": {"outDim": N}} or "relu", and this script emits
the graph JSON that `leviathan-cli forge --graph` consumes.

Honest scope (M0):
- Weights are NOT exported here. Leviathan generates deterministic
  seeded synthetic weights for dense/relu graphs, and the plan cache
  identity includes that seed. Real trained-weight export (PyTorch
  state_dict / safetensors) is planned but NOT implemented.
- Supported ops are exactly dense and relu; anything else is rejected by
  Leviathan's graph validation, not silently accepted.
- Running a model file is an explicit user action on trusted input; the
  script uses only the Python standard library.

Usage:
    python3 forge_demo.py model_definition.py out_graph.json
"""

import importlib.util
import json
import sys


def load_model(path):
    spec = importlib.util.spec_from_file_location("leviathan_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MODEL"):
        raise SystemExit(f"{path}: model file must define MODEL")
    return module.MODEL


def validate(model):
    if not isinstance(model, dict) or set(model) != {"inputDim", "ops"}:
        raise SystemExit("MODEL must be a dict with exactly inputDim and ops")
    input_dim = model["inputDim"]
    if not isinstance(input_dim, int) or not (0 < input_dim <= 4096):
        raise SystemExit("inputDim must be an int in 1...4096")
    if not isinstance(model["ops"], list) or not model["ops"]:
        raise SystemExit("ops must be a non-empty list")
    for op in model["ops"]:
        if op == "relu":
            continue
        if (isinstance(op, dict) and set(op) == {"dense"}
                and isinstance(op["dense"], dict)
                and set(op["dense"]) == {"outDim"}
                and isinstance(op["dense"]["outDim"], int)
                and 0 < op["dense"]["outDim"] <= 4096):
            continue
        raise SystemExit(f'unsupported op {op!r}: expected "relu" or '
                         '{"dense": {"outDim": N}}')


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model = load_model(sys.argv[1])
    validate(model)
    with open(sys.argv[2], "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"wrote {sys.argv[2]}: inputDim {model['inputDim']}, "
          f"{len(model['ops'])} ops")


if __name__ == "__main__":
    main()
