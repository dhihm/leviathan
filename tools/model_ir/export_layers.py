"""torch.export of the two Ornith decoder layer types at real size, without weights.

The layers are built on the meta device (no memory for the 9B parameters) and
exported with a symbolic token count T, so the captured graph is valid for any
T. Prints the call nodes of each graph and optionally writes them as JSON.

    python export_layers.py [out.json]
"""
import json
import sys

import torch
from torch.export import Dim, export

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from ornith_layers import AttnDecoderLayer, GdnDecoderLayer


def real_config():
    cfg = Qwen3_5TextConfig(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 8,
        rope_parameters={"rope_type": "default", "rope_theta": 1.0e7, "partial_rotary_factor": 0.25},
    )
    cfg._attn_implementation = "eager"
    return cfg


def _shape(v):
    """Shape of a tensor value as strings (symbolic sizes print as their symbols)."""
    if isinstance(v, torch.Tensor):
        return [str(d) for d in v.shape]
    if isinstance(v, (tuple, list)):
        return [_shape(x) for x in v]
    return None


def describe(ep):
    return [
        {"name": n.name, "target": str(n.target), "shape": _shape(n.meta.get("val"))}
        for n in ep.graph.nodes
        if n.op == "call_function"
    ]

def main():
    cfg = real_config()
    out = {}
    t = Dim("T", min=1, max=4096)
    s = Dim("S", min=0, max=32768)
    with torch.device("meta"):
        gdn = GdnDecoderLayer(Qwen3_5DecoderLayer(cfg, 0)).eval()
        attn = AttnDecoderLayer(Qwen3_5DecoderLayer(cfg, 3)).eval()
        m = gdn.layer.linear_attn
        d, hd, hkv = cfg.hidden_size, cfg.head_dim, cfg.num_key_value_heads
        rd = int(hd * 0.25)
        ex_gdn = (
            torch.empty(5, d),
            torch.empty(m.conv_kernel_size - 1, m.conv_dim),
            torch.empty(m.num_v_heads, m.head_k_dim, m.head_v_dim),
        )
        ex_attn = (torch.empty(5, d), torch.empty(5, rd), torch.empty(5, rd), torch.empty(3, hkv, hd), torch.empty(3, hkv, hd))
    for name, mod, ex, dyn in [
        ("gated_delta_layer", gdn, ex_gdn, {"x": {0: t}, "conv_state": None, "rec_state": None}),
        ("full_attention_layer", attn, ex_attn, {"x": {0: t}, "cos": {0: t}, "sin": {0: t}, "k_cache": {0: s}, "v_cache": {0: s}}),
    ]:
        ep = export(mod, ex, dynamic_shapes=dyn)
        rows = describe(ep)
        out[name] = rows
        print(f"== {name}: {len(rows)} call nodes, {len(list(ep.state_dict))} parameters")
        for r in rows:
            print(f"  {r['target']:<48} {r['shape']}")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
