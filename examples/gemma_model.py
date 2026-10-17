"""User model file for LeviathanForge Gemma capture (restricted subset).

Executed ONLY by tools/forge_gemma/export_gemma_graph.py with a config
exported from the authorized GGUF. Defines the model's compute graph in
the supported DSL; the Swift pipeline validates it against the real GGUF
tensor contract and rejects anything unsupported.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../tools/forge_gemma")
from gemma_dsl import build_gemma4


def build(config):
    return build_gemma4(config).to_json()
