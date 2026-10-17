#!/usr/bin/env python3
"""Captures the user's restricted model.py into a graph JSON (no weights).

Usage: export_gemma_graph.py <model.py> <config.json> <out-graph.json>

Executes the user's trusted file (explicit user action, like running any
script); the emitted graph is fully re-validated in Swift against the
real GGUF before anything executes.
"""
import importlib.util, json, sys

model_path, config_path, out_path = sys.argv[1:4]
config = json.load(open(config_path))
spec = importlib.util.spec_from_file_location("user_model", model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
graph = mod.build(config)
json.dump(graph, open(out_path, "w"), ensure_ascii=False)
print(f"captured {len(graph['ops'])} ops -> {out_path}")
