#!/usr/bin/env python3
"""Captures the user's Q4 op model (.py) into op JSON for Swift validation.

Usage: export_q4_op.py <op_model.py> <out-op.json>

Executes the user's trusted file (explicit user action, like running any
script); the module must define build() -> Q4MatvecOp (or a check_op-valid
dict). The emitted JSON carries no weights — representative weight BYTES are
read by the Swift side from the real GGUF tensor when a `tensor` binding is
present, and re-validated there.
"""
import importlib.util, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4_op_model import Q4MatvecOp, check_op

model_path, out_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("user_op_model", model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
built = mod.build()
obj = built.to_json() if isinstance(built, Q4MatvecOp) else built
check_op(obj)
json.dump(obj, open(out_path, "w"), ensure_ascii=False, sort_keys=True)
print(f"captured q4_0.matvec op inDim={obj['inDim']} outDim={obj['outDim']} "
      f"tensor={obj.get('tensor')} -> {out_path}")
