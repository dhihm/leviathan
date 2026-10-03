#!/usr/bin/env python3
"""Captures the user's RMS op model (.py) into op JSON for Swift validation.

Usage: export_rms_op.py <op_model.py> <out-op.json>

Executes the user's trusted file (explicit user action, like running any
script); the module must define build() -> RMSNormOp (or a check_op-valid
dict). The emitted JSON carries no weights — representative weight BYTES
are read by the Swift side from the real GGUF tensor when a `tensor`
binding is present, and re-validated there.
"""
import importlib.util, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rms_op_model import RMSNormOp, check_op

model_path, out_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("user_op_model", model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
built = mod.build()
obj = built.to_json() if isinstance(built, RMSNormOp) else built
check_op(obj)
json.dump(obj, open(out_path, "w"), ensure_ascii=False, sort_keys=True)
print(f"captured rmsnorm op n={obj['n']} scaled={obj['scaled']} "
      f"tensor={obj.get('tensor')} -> {out_path}")
