"""Restricted Gemma 4 graph DSL for LeviathanForge capture.

A user model.py builds the model compute graph by calling build_gemma4(config)
with a config exported from the actual GGUF (leviathan-cli gemma-config-export).
The DSL emits an op-list JSON (no weights) — the Swift side validates every
op and tensor binding against the real GGUF before anything executes, and
rejects anything outside this supported subset. Supported ops are exactly
the Gemma 4 26B-A4B attention/MoE op set; no other op type exists here.
"""

SUPPORTED_OPS = [
    "embed", "attn_norm", "q_proj", "q_norm", "q_rope", "k_proj",
    "v_eq_k", "v_proj", "k_norm", "k_rope", "v_norm", "cache_append",
    "attention", "o_proj", "post_attn_norm", "residual_add", "ffn_norm",
    "shared_gate", "shared_up", "shared_gelu", "shared_down", "shared_post1",
    "router_norm", "router_scale", "router_proj", "router_topk", "moe_norm2",
    "expert_loop", "moe_post2", "combine", "post_ffw_norm", "layer_scale",
    "final_norm", "lm_head", "softcap",
]




# Typed dataflow signatures: op -> (inputs, outputs) over value IDs.
# "kcache"/"vcache" are per-layer state effects (layer-suffixed at
# validation); everything else is an activation buffer.
SIGS = {
    "embed": ([], ["x"]),
    "attn_norm": (["x"], ["h"]),
    "q_proj": (["h"], ["t1"]),
    "q_norm": (["t1"], ["t1"]),
    "q_rope": (["t1"], ["t1"]),
    "k_proj": (["h"], ["t2"]),
    "v_eq_k": (["t2"], ["kv"]),
    "v_proj": (["h"], ["kv"]),
    "k_norm": (["t2"], ["t2"]),
    "k_rope": (["t2"], ["t2"]),
    "v_norm": (["kv"], ["kv"]),
    "cache_append": (["t2", "kv"], ["kcache", "vcache"]),
    "attention": (["t1", "kcache", "vcache"], ["t3"]),
    "o_proj": (["t3"], ["t2"]),
    "post_attn_norm": (["t2"], ["h"]),
    "residual_add": (["x", "h"], ["x"]),
    "ffn_norm": (["x"], ["h"]),
    "shared_gate": (["h"], ["gu"]),
    "shared_up": (["h"], ["gu"]),
    "shared_gelu": (["gu"], ["act"]),
    "shared_down": (["act"], ["shared"]),
    "shared_post1": (["shared"], ["shared"]),
    "router_norm": (["x"], ["h"]),
    "router_scale": (["h"], ["h"]),
    "router_proj": (["h"], ["router"]),
    "router_topk": (["router"], ["topk"]),
    "moe_norm2": (["x"], ["h"]),
    "expert_loop": (["h", "topk"], ["t3"]),
    "moe_post2": (["t3"], ["t3"]),
    "combine": (["shared", "t3"], ["h"]),
    "post_ffw_norm": (["h"], ["h"]),
    "layer_scale": (["x"], ["x"]),
    "final_norm": (["x"], ["h"]),
    "lm_head": (["h"], ["logits"]),
    "softcap": (["logits"], ["logits"]),
}


class GemmaGraph:
    def __init__(self, config):
        self.config = config
        self.ops = []

    def add(self, op, layer=None, tensor=None, extra=None):
        if op not in SUPPORTED_OPS:
            raise ValueError(f"unsupported op {op!r}")
        entry = {"op": op}
        if layer is not None:
            entry["layer"] = layer
        if tensor is not None:
            entry["tensor"] = tensor
        if extra:
            entry.update(extra)
        self.ops.append(entry)

    def to_json(self):
        return {"schema": 1, "model": "gemma4-26b-a4b", "config": self.config,
                "ops": self.ops}


def build_gemma4(config):
    """The real 30-layer Gemma 4 26B-A4B compute graph, from config alone."""
    g = GemmaGraph(config)
    g.add("embed", tensor="token_embd.weight")
    for layer in range(config["layerCount"]):
        is_global = layer in config["globalLayers"]
        p = f"blk.{layer}."
        g.add("attn_norm", layer=layer, tensor=p + "attn_norm.weight")
        g.add("q_proj", layer=layer, tensor=p + "attn_q.weight")
        g.add("q_norm", layer=layer, tensor=p + "attn_q_norm.weight")
        g.add("q_rope", layer=layer, extra={"global": is_global})
        g.add("k_proj", layer=layer, tensor=p + "attn_k.weight")
        if is_global:
            g.add("v_eq_k", layer=layer)
        else:
            g.add("v_proj", layer=layer, tensor=p + "attn_v.weight")
        g.add("k_norm", layer=layer, tensor=p + "attn_k_norm.weight")
        g.add("k_rope", layer=layer, extra={"global": is_global})
        g.add("v_norm", layer=layer)
        g.add("cache_append", layer=layer, extra={"global": is_global})
        g.add("attention", layer=layer, extra={"global": is_global})
        g.add("o_proj", layer=layer, tensor=p + "attn_output.weight")
        g.add("post_attn_norm", layer=layer, tensor=p + "post_attention_norm.weight")
        g.add("residual_add", layer=layer)
        g.add("ffn_norm", layer=layer, tensor=p + "ffn_norm.weight")
        g.add("shared_gate", layer=layer, tensor=p + "ffn_gate.weight")
        g.add("shared_up", layer=layer, tensor=p + "ffn_up.weight")
        g.add("shared_gelu", layer=layer)
        g.add("shared_down", layer=layer, tensor=p + "ffn_down.weight")
        g.add("shared_post1", layer=layer, tensor=p + "post_ffw_norm_1.weight")
        g.add("router_norm", layer=layer)
        g.add("router_scale", layer=layer)
        g.add("router_proj", layer=layer, tensor=p + "ffn_gate_inp.weight")
        g.add("router_topk", layer=layer, extra={"topK": config["expertUsedCount"]})
        g.add("moe_norm2", layer=layer, tensor=p + "pre_ffw_norm_2.weight")
        g.add("expert_loop", layer=layer,
              extra={"gateUp": p + "ffn_gate_up_exps.weight",
                     "down": p + "ffn_down_exps.weight"})
        g.add("moe_post2", layer=layer, tensor=p + "post_ffw_norm_2.weight")
        g.add("combine", layer=layer)
        g.add("post_ffw_norm", layer=layer, tensor=p + "post_ffw_norm.weight")
        g.add("residual_add", layer=layer)
        g.add("layer_scale", layer=layer)
    g.add("final_norm", tensor="output_norm.weight")
    g.add("lm_head", tensor="token_embd.weight")
    g.add("softcap", extra={"cap": config["finalSoftcap"]})
    return g
