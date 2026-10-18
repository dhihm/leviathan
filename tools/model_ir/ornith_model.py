"""Whole-model functional form of Ornith (Qwen3.5 text trunk): embedding, the layer stack,
final norm and the output projection, with every recurrent state and KV cache as an explicit
input and output.

Inputs:  token_ids [T], cos [T, rd], sin [T, rd], then per layer in order: conv_state and
         rec_state for linear-attention layers, k_cache and v_cache for full-attention layers.
Outputs: logits [T, V], then the updated states in the same order (for full attention layers the
         new K and V rows of the T tokens, which the runtime appends to its cache).

Parameter names are the Hugging Face names of Qwen3_5ForCausalLM (model.embed_tokens.weight,
model.layers.N..., model.norm.weight, lm_head.weight).
"""
import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from ornith_layers import AttnDecoderLayer, AttnDecoderLayerRagged, GdnDecoderLayer, GdnDecoderLayerRagged, PagedAttnDecoderLayer, rmsnorm


class _Trunk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


class OrnithModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = _Trunk(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.wrappers = [
            GdnDecoderLayer(l) if l.block_type == "linear_attention" else AttnDecoderLayer(l) for l in self.model.layers
        ]
        # Wrappers hold the layers as submodules; register them without duplicating parameters.
        object.__setattr__(self, "_wrappers_hidden", self.wrappers)

    def state_layout(self):
        """Per layer: ('linear', conv_shape, rec_shape) or ('full', kv_shape)."""
        cfg = self.cfg
        out = []
        for l in self.model.layers:
            if l.block_type == "linear_attention":
                m = l.linear_attn
                out.append(("linear", (m.conv_kernel_size - 1, m.conv_dim), (m.num_v_heads, m.head_k_dim, m.head_v_dim)))
            else:
                a = l.self_attn
                out.append(("full", (0, cfg.num_key_value_heads, a.head_dim)))
        return out

    def forward(self, token_ids, cos, sin, *states):
        x = self.model.embed_tokens(token_ids)
        outs = []
        i = 0
        for w, l in zip(self.wrappers, self.model.layers):
            if l.block_type == "linear_attention":
                x, c, r = w(x, states[i], states[i + 1])
                outs += [c, r]
            else:
                x, k, v = w(x, cos, sin, states[i], states[i + 1])
                outs += [k, v]
            i += 2
        x = rmsnorm(x, self.model.norm.weight, self.model.norm.eps)
        return (self.lm_head(x), *outs)


class OrnithModelRagged(OrnithModel):
    """The same model over a flat batch of tokens from several sequences.

    Inputs:  token_ids [T], cos [T, rd], sin [T, rd], slot [T], pos [T], then per layer in order:
             conv_pool [S, K-1, C] and rec_pool [S, H, dk, dv] for linear-attention layers,
             k_pool and v_pool [S, L, Hkv, d] for full-attention layers.
    Outputs: logits [T, V], then the updated pools in the same order.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ragged = [
            GdnDecoderLayerRagged(l) if l.block_type == "linear_attention" else AttnDecoderLayerRagged(l) for l in self.model.layers
        ]

    def pool_layout(self, slots, max_len):
        """Per layer: ('linear', conv_pool_shape, rec_pool_shape) or ('full', kv_pool_shape)."""
        out = []
        for l in self.model.layers:
            if l.block_type == "linear_attention":
                m = l.linear_attn
                out.append(("linear", (slots, m.conv_kernel_size - 1, m.conv_dim), (slots, m.num_v_heads, m.head_k_dim, m.head_v_dim)))
            else:
                a = l.self_attn
                out.append(("full", (slots, max_len, self.cfg.num_key_value_heads, a.head_dim)))
        return out

    def forward(self, token_ids, cos, sin, slot, pos, *pools):
        x = self.model.embed_tokens(token_ids)
        outs = []
        i = 0
        for w, l in zip(self.ragged, self.model.layers):
            if l.block_type == "linear_attention":
                x, c, r = w(x, pools[i], pools[i + 1], slot)
                outs += [c, r]
            else:
                x, k, v = w(x, cos, sin, pools[i], pools[i + 1], slot, pos)
                outs += [k, v]
            i += 2
        x = rmsnorm(x, self.model.norm.weight, self.model.norm.eps)
        return (self.lm_head(x), *outs)

class OrnithModelPaged(OrnithModelRagged):
    """Ragged batch with a paged KV cache.

    Inputs:  token_ids [T], cos [T, rd], sin [T, rd], slot [T], pos [T], slot_mapping [T],
             block_table [S, MB], then per layer: conv_pool and rec_pool for linear-attention layers
             (as OrnithModelRagged), k_pool and v_pool [NB, BS, Hkv, d] for full-attention layers.
    Outputs: logits [T, V], then the updated pools in the same order.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.paged = [PagedAttnDecoderLayer(l) if l.block_type == "full_attention" else None for l in self.model.layers]

    def pool_layout(self, slots, blocks, block_size=16):
        out = []
        for l in self.model.layers:
            if l.block_type == "linear_attention":
                m = l.linear_attn
                out.append(("linear", (slots, m.conv_kernel_size - 1, m.conv_dim), (slots, m.num_v_heads, m.head_k_dim, m.head_v_dim)))
            else:
                out.append(("full", (blocks, block_size, self.cfg.num_key_value_heads, l.self_attn.head_dim)))
        return out

    def forward(self, token_ids, cos, sin, slot, pos, slot_mapping, block_table, *pools):
        x = self.model.embed_tokens(token_ids)
        outs = []
        i = 0
        for w, pg, l in zip(self.ragged, self.paged, self.model.layers):
            if l.block_type == "linear_attention":
                x, c, r = w(x, pools[i], pools[i + 1], slot)
                outs += [c, r]
            else:
                x, k, v = pg(x, cos, sin, pools[i], pools[i + 1], block_table, slot, pos, slot_mapping)
                outs += [k, v]
            i += 2
        x = rmsnorm(x, self.model.norm.weight, self.model.norm.eps)
        return (self.lm_head(x), *outs)