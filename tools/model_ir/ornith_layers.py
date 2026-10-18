"""Export-friendly functional forms of the Ornith (Qwen3.5) decoder layers.

Each wrapper keeps the Hugging Face submodules (so parameters keep their
Hugging Face names and shapes) but runs the token mixer on a flat [T, D]
sequence with explicit state in and state out, calling the custom operators of
lev_ops. A wrapper must equal the Hugging Face layer (tests: test_layers.py).

Norm weights are used as Hugging Face stores them: Qwen3.5 RMSNorm scales by
(1 + weight). GGUF files store the sum, which is how the engine reads them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import lev_ops  # noqa: F401  (registers torch.ops.lev.*)


class GatedDeltaMixer(nn.Module):
    """Qwen3_5GatedDeltaNet on a flat [T, D] sequence with conv and recurrent state in/out."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, conv_state, rec_state):
        m = self.m
        t = x.shape[0]
        mixed = m.in_proj_qkv(x)  # [T, C]
        z = m.in_proj_z(x)  # [T, Hv*dv]
        b = m.in_proj_b(x)  # [T, Hv]
        a = m.in_proj_a(x)  # [T, Hv]
        mixed, conv_out = torch.ops.lev.causal_conv1d_state(mixed, conv_state, m.conv1d.weight.squeeze(1))
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(t, -1, m.head_k_dim)
        k = k.reshape(t, -1, m.head_k_dim)
        v = v.reshape(t, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        rep = m.num_v_heads // m.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        core, rec_out = torch.ops.lev.gated_delta_rule(q, k, v, g, beta, rec_state)
        core = torch.ops.lev.rmsnorm_gated(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim), m.norm.weight, m.norm.variance_epsilon)
        return m.out_proj(core.reshape(t, -1)), conv_out, rec_out


class FullAttentionMixer(nn.Module):
    """Qwen3_5Attention (gated, GQA, partial rope) with the KV cache as explicit tensors."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, cos, sin, k_cache, v_cache):
        m = self.m
        t = x.shape[0]
        hd = m.head_dim
        qg = m.q_proj(x).view(t, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)  # [T, Hq, hd] each
        gate = gate.reshape(t, -1)
        q = rmsnorm(q, m.q_norm.weight, m.q_norm.eps)
        k = rmsnorm(m.k_proj(x).view(t, -1, hd), m.k_norm.weight, m.k_norm.eps)
        v = m.v_proj(x).view(t, -1, hd)
        q = torch.ops.lev.rope(q, cos, sin)
        k = torch.ops.lev.rope(k, cos, sin)
        k_all = torch.cat([k_cache, k], dim=0)
        v_all = torch.cat([v_cache, v], dim=0)
        out = torch.ops.lev.causal_attention(q, k_all, v_all, k_cache.shape[0], m.scaling)
        out = out.reshape(t, -1) * torch.sigmoid(gate)
        return m.o_proj(out), k, v


def rmsnorm(x, weight, eps):
    """Qwen3.5 RMSNorm (zero-centered weight, as Hugging Face stores it)."""
    return torch.ops.lev.rmsnorm(x, weight, eps, True)


def mlp(l, x):
    """Qwen3_5MLP with the SiLU gate as one operator."""
    return l.mlp.down_proj(torch.ops.lev.swiglu(l.mlp.gate_proj(x), l.mlp.up_proj(x)))


class GdnDecoderLayer(nn.Module):
    """Qwen3_5DecoderLayer of type linear_attention."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.mixer = GatedDeltaMixer(layer.linear_attn)

    def forward(self, x, conv_state, rec_state):
        l = self.layer
        h, conv_out, rec_out = self.mixer(rmsnorm(x, l.input_layernorm.weight, l.input_layernorm.eps), conv_state, rec_state)
        x = x + h
        x = x + mlp(l, rmsnorm(x, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps))
        return x, conv_out, rec_out


class AttnDecoderLayer(nn.Module):
    """Qwen3_5DecoderLayer of type full_attention."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.mixer = FullAttentionMixer(layer.self_attn)

    def forward(self, x, cos, sin, k_cache, v_cache):
        l = self.layer
        h, k_new, v_new = self.mixer(rmsnorm(x, l.input_layernorm.weight, l.input_layernorm.eps), cos, sin, k_cache, v_cache)
        x = x + h
        x = x + mlp(l, rmsnorm(x, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps))
        return x, k_new, v_new


class GatedDeltaMixerRagged(nn.Module):
    """GatedDeltaMixer over a flat batch: per-token state slot, state pools in and out."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, conv_pool, rec_pool, slot):
        m = self.m
        t = x.shape[0]
        mixed = m.in_proj_qkv(x)
        z = m.in_proj_z(x)
        b = m.in_proj_b(x)
        a = m.in_proj_a(x)
        mixed, conv_out = torch.ops.lev.ssm_conv_ragged(mixed, conv_pool, slot, m.conv1d.weight.squeeze(1))
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(t, -1, m.head_k_dim)
        k = k.reshape(t, -1, m.head_k_dim)
        v = v.reshape(t, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        rep = m.num_v_heads // m.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        core, rec_out = torch.ops.lev.gated_delta_ragged(q, k, v, g, beta, rec_pool, slot)
        core = torch.ops.lev.rmsnorm_gated(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim), m.norm.weight, m.norm.variance_epsilon)
        return m.out_proj(core.reshape(t, -1)), conv_out, rec_out


class FullAttentionMixerRagged(nn.Module):
    """FullAttentionMixer over a flat batch with a per-slot KV slab."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, cos, sin, k_pool, v_pool, slot, pos):
        m = self.m
        t = x.shape[0]
        hd = m.head_dim
        qg = m.q_proj(x).view(t, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(t, -1)
        q = rmsnorm(q, m.q_norm.weight, m.q_norm.eps)
        k = rmsnorm(m.k_proj(x).view(t, -1, hd), m.k_norm.weight, m.k_norm.eps)
        v = m.v_proj(x).view(t, -1, hd)
        q = torch.ops.lev.rope(q, cos, sin)
        k = torch.ops.lev.rope(k, cos, sin)
        k_pool, v_pool = torch.ops.lev.slab_kv_write(k, v, k_pool, v_pool, slot, pos)
        out = torch.ops.lev.slab_attention(q, k_pool, v_pool, slot, pos, m.scaling)
        out = out.reshape(t, -1) * torch.sigmoid(gate)
        return m.o_proj(out), k_pool, v_pool


class GdnDecoderLayerRagged(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.mixer = GatedDeltaMixerRagged(layer.linear_attn)

    def forward(self, x, conv_pool, rec_pool, slot):
        l = self.layer
        h, conv_out, rec_out = self.mixer(rmsnorm(x, l.input_layernorm.weight, l.input_layernorm.eps), conv_pool, rec_pool, slot)
        x = x + h
        x = x + mlp(l, rmsnorm(x, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps))
        return x, conv_out, rec_out


class AttnDecoderLayerRagged(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.mixer = FullAttentionMixerRagged(layer.self_attn)

    def forward(self, x, cos, sin, k_pool, v_pool, slot, pos):
        l = self.layer
        h, k_pool, v_pool = self.mixer(rmsnorm(x, l.input_layernorm.weight, l.input_layernorm.eps), cos, sin, k_pool, v_pool, slot, pos)
        x = x + h
        x = x + mlp(l, rmsnorm(x, l.post_attention_layernorm.weight, l.post_attention_layernorm.eps))
        return x, k_pool, v_pool