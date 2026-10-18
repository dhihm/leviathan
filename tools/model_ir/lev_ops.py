"""Custom operators of the model IR, registered with torch.library.

Every operator has a CPU reference implementation (the semantics of the IR),
and a fake (meta) kernel so torch.export can trace shapes without data. The
recurrent state is functional: it goes in and a new state comes out, nothing
is mutated, so the exported graph stays pure.

Single sequence of T tokens for now; the flat multi-sequence layout (per-token
sequence ids, state slots, block tables) replaces these shapes later.

Reference semantics follow the Hugging Face Qwen3.5 implementation
(transformers.models.qwen3_5): the gated delta rule is evaluated by its
recurrent form, which equals the chunked form up to rounding.
"""
import torch
import torch.nn.functional as F

_LIB = torch.library.Library("lev", "DEF")

_LIB.define("causal_conv1d_state(Tensor x, Tensor state, Tensor w) -> (Tensor, Tensor)")
_LIB.define("gated_delta_rule(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state) -> (Tensor, Tensor)")
# past_len is a SymInt: the cache length is dynamic in the exported graph, and a plain int
# argument would specialize it to the example's value.
_LIB.define("causal_attention(Tensor q, Tensor k_all, Tensor v_all, SymInt past_len, float scale) -> Tensor")


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


@torch.library.impl(_LIB, "causal_conv1d_state", "CompositeExplicitAutograd")
def _conv_impl(x, state, w):
    """x [T, C], state [K-1, C] (the last K-1 inputs before x), w [C, K].

    y[t, c] = silu(sum_i w[c, i] * xx[t + i, c]) over xx = cat(state, x); the new state is the
    last K-1 rows of xx.
    """
    t, c = x.shape
    k = w.shape[1]
    xx = torch.cat([state, x], dim=0)  # [K-1+T, C]
    acc = torch.zeros(t, c, dtype=x.dtype)
    for i in range(k):
        acc = acc + xx[i : i + t] * w[:, i]
    return F.silu(acc), xx[t:].clone()


@torch.library.impl(_LIB, "causal_conv1d_state", "Meta")
def _conv_meta(x, state, w):
    return x.new_empty(x.shape), state.new_empty(state.shape)


@torch.library.impl(_LIB, "gated_delta_rule", "CompositeExplicitAutograd")
def _gdr_impl(q, k, v, g, beta, state):
    """q, k [T, H, dk] (already expanded to the value heads), v [T, H, dv], g and beta [T, H],
    state [H, dk, dv]. q and k are L2-normalized and q is scaled by dk^-0.5 inside. Computed in
    float32 whatever the input dtype (the decay gate is exponentiated).
    """
    dt = q.dtype
    q, k, v, g, beta, s = (a.float() for a in (q, k, v, g, beta, state))
    q = _l2norm(q) * (q.shape[-1] ** -0.5)
    k = _l2norm(k)
    outs = []
    for t in range(q.shape[0]):
        s = s * g[t].exp()[:, None, None]
        kv = (s * k[t][:, :, None]).sum(dim=1)  # [H, dv]
        delta = (v[t] - kv) * beta[t][:, None]
        s = s + k[t][:, :, None] * delta[:, None, :]
        outs.append((s * q[t][:, :, None]).sum(dim=1))
    return torch.stack(outs).to(dt), s.to(state.dtype)


@torch.library.impl(_LIB, "gated_delta_rule", "Meta")
def _gdr_meta(q, k, v, g, beta, state):
    return v.new_empty(v.shape), state.new_empty(state.shape)


@torch.library.impl(_LIB, "causal_attention", "CompositeExplicitAutograd")
def _attn_impl(q, k_all, v_all, past_len, scale):
    """q [T, Hq, d] at positions past_len..past_len+T-1; k_all, v_all [past_len+T, Hkv, d]."""
    t, hq, d = q.shape
    s, hkv, _ = k_all.shape
    rep = hq // hkv
    k_all = k_all.repeat_interleave(rep, dim=1)
    v_all = v_all.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k_all) * scale
    pos_q = torch.arange(t)[:, None] + past_len
    pos_k = torch.arange(s)[None, :]
    scores = scores.masked_fill((pos_k > pos_q)[None], float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(dim=-1), v_all)


@torch.library.impl(_LIB, "causal_attention", "Meta")
def _attn_meta(q, k_all, v_all, past_len, scale):
    return q.new_empty(q.shape)


# ---- element-wise and normalization operators (kept as IR operators, not decomposed) ----

_LIB.define("rmsnorm(Tensor x, Tensor w, float eps, bool zero_centered) -> Tensor")
_LIB.define("rmsnorm_gated(Tensor x, Tensor z, Tensor w, float eps) -> Tensor")
_LIB.define("swiglu(Tensor gate, Tensor up) -> Tensor")
_LIB.define("rope(Tensor x, Tensor cos, Tensor sin) -> Tensor")


@torch.library.impl(_LIB, "rmsnorm", "CompositeExplicitAutograd")
def _rmsnorm_impl(x, w, eps, zero_centered):
    """y = x * rsqrt(mean(x^2) + eps) * (w + 1 if zero_centered else w), float32 inside.

    Hugging Face stores Qwen3.5 norm weights zero-centered (scale 1 + w); GGUF stores 1 + w.
    """
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (w.float() + 1.0 if zero_centered else w.float())).to(x.dtype)


@torch.library.impl(_LIB, "rmsnorm", "Meta")
def _rmsnorm_meta(x, w, eps, zero_centered):
    return x.new_empty(x.shape)


@torch.library.impl(_LIB, "rmsnorm_gated", "CompositeExplicitAutograd")
def _rmsnorm_gated_impl(x, z, w, eps):
    """Norm first, then the SiLU gate (Qwen3_5RMSNormGated): w * norm(x) * silu(z)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return ((w.float() * y.to(x.dtype).float()) * F.silu(z.float())).to(x.dtype)


@torch.library.impl(_LIB, "rmsnorm_gated", "Meta")
def _rmsnorm_gated_meta(x, z, w, eps):
    return x.new_empty(x.shape)


@torch.library.impl(_LIB, "swiglu", "CompositeExplicitAutograd")
def _swiglu_impl(gate, up):
    return F.silu(gate) * up


@torch.library.impl(_LIB, "swiglu", "Meta")
def _swiglu_meta(gate, up):
    return gate.new_empty(gate.shape)


@torch.library.impl(_LIB, "rope", "CompositeExplicitAutograd")
def _rope_impl(x, cos, sin):
    """Rotate the first rd = cos.shape[-1] dims of each head (neox layout: halves), x [T, H, d],
    cos and sin [T, rd]; the remaining dims pass through."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    rot = torch.cat([-xr[..., h:], xr[..., :h]], dim=-1)
    return torch.cat([xr * cos[:, None, :] + rot * sin[:, None, :], xp], dim=-1)


@torch.library.impl(_LIB, "rope", "Meta")
def _rope_meta(x, cos, sin):
    return x.new_empty(x.shape)

# ---- ragged multi-sequence operators ----
#
# A batch is a flat list of T tokens. Token t belongs to the sequence in state slot slot[t] and sits
# at position pos[t]. The tokens of one sequence appear in increasing position order. Per-sequence
# state lives in pools indexed by slot; the operators take a pool and return the updated pool
# (functional here; the runtime updates the pool in place).

_LIB.define("ssm_conv_ragged(Tensor x, Tensor pool, Tensor slot, Tensor w) -> (Tensor, Tensor)")
_LIB.define("gated_delta_ragged(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor pool, Tensor slot) -> (Tensor, Tensor)")
_LIB.define("slab_kv_write(Tensor k, Tensor v, Tensor k_pool, Tensor v_pool, Tensor slot, Tensor pos) -> (Tensor, Tensor)")
_LIB.define("slab_attention(Tensor q, Tensor k_pool, Tensor v_pool, Tensor slot, Tensor pos, float scale) -> Tensor")


@torch.library.impl(_LIB, "ssm_conv_ragged", "CompositeExplicitAutograd")
def _conv_ragged_impl(x, pool, slot, w):
    """x [T, C], pool [S, K-1, C], slot [T], w [C, K]; token by token, per its slot's state."""
    pool = pool.clone()
    k = w.shape[1]
    ys = []
    for t in range(x.shape[0]):
        s = int(slot[t])
        xx = torch.cat([pool[s], x[t : t + 1]], dim=0)  # [K, C]
        acc = (xx * w.t()).sum(dim=0)
        ys.append(F.silu(acc))
        pool[s] = xx[1:]
    return torch.stack(ys), pool


@torch.library.impl(_LIB, "ssm_conv_ragged", "Meta")
def _conv_ragged_meta(x, pool, slot, w):
    return x.new_empty(x.shape), pool.new_empty(pool.shape)


@torch.library.impl(_LIB, "gated_delta_ragged", "CompositeExplicitAutograd")
def _gdr_ragged_impl(q, k, v, g, beta, pool, slot):
    """q, k [T, H, dk] (expanded to the value heads), v [T, H, dv], g and beta [T, H],
    pool [S, H, dk, dv]; same per-token recurrence as gated_delta_rule on the slot's state."""
    dt = q.dtype
    q, k, v, g, beta = (a.float() for a in (q, k, v, g, beta))
    pool = pool.clone().float()
    q = _l2norm(q) * (q.shape[-1] ** -0.5)
    k = _l2norm(k)
    outs = []
    for t in range(q.shape[0]):
        sl = int(slot[t])
        s = pool[sl] * g[t].exp()[:, None, None]
        kv = (s * k[t][:, :, None]).sum(dim=1)
        delta = (v[t] - kv) * beta[t][:, None]
        s = s + k[t][:, :, None] * delta[:, None, :]
        pool[sl] = s
        outs.append((s * q[t][:, :, None]).sum(dim=1))
    return torch.stack(outs).to(dt), pool


@torch.library.impl(_LIB, "gated_delta_ragged", "Meta")
def _gdr_ragged_meta(q, k, v, g, beta, pool, slot):
    return v.new_empty(v.shape), pool.new_empty(pool.shape)


@torch.library.impl(_LIB, "slab_kv_write", "CompositeExplicitAutograd")
def _kv_write_impl(k, v, k_pool, v_pool, slot, pos):
    """k, v [T, Hkv, d]; pools [S, L, Hkv, d]: pool[slot[t], pos[t]] = k[t]."""
    k_pool, v_pool = k_pool.clone(), v_pool.clone()
    for t in range(k.shape[0]):
        k_pool[int(slot[t]), int(pos[t])] = k[t]
        v_pool[int(slot[t]), int(pos[t])] = v[t]
    return k_pool, v_pool


@torch.library.impl(_LIB, "slab_kv_write", "Meta")
def _kv_write_meta(k, v, k_pool, v_pool, slot, pos):
    return k_pool.new_empty(k_pool.shape), v_pool.new_empty(v_pool.shape)


@torch.library.impl(_LIB, "slab_attention", "CompositeExplicitAutograd")
def _slab_attn_impl(q, k_pool, v_pool, slot, pos, scale):
    """q [T, Hq, d]; attends positions 0..pos[t] (inclusive) of the token's own slot."""
    t_n, hq, d = q.shape
    hkv = k_pool.shape[2]
    rep = hq // hkv
    out = torch.zeros_like(q)
    for t in range(t_n):
        n = int(pos[t]) + 1
        k = k_pool[int(slot[t]), :n].repeat_interleave(rep, dim=1)  # [n, Hq, d]
        v = v_pool[int(slot[t]), :n].repeat_interleave(rep, dim=1)
        sc = torch.einsum("hd,nhd->hn", q[t], k) * scale
        out[t] = torch.einsum("hn,nhd->hd", sc.softmax(dim=-1), v)
    return out


@torch.library.impl(_LIB, "slab_attention", "Meta")
def _slab_attn_meta(q, k_pool, v_pool, slot, pos, scale):
    return q.new_empty(q.shape)

# ---- paged KV cache ----
#
# The K and V caches of an attention layer are pools of fixed-size blocks, [blocks, block_size,
# Hkv, d]. A sequence owns a list of blocks; block_table[slot, i] is the physical block holding its
# positions i * block_size .. (i + 1) * block_size - 1. slot_mapping[t] is the flat pool position
# (block * block_size + offset) that token t writes to.

_LIB.define("paged_kv_write(Tensor k, Tensor v, Tensor k_pool, Tensor v_pool, Tensor slot_mapping) -> (Tensor, Tensor)")
_LIB.define("paged_attention(Tensor q, Tensor k_pool, Tensor v_pool, Tensor block_table, Tensor slot, Tensor pos, float scale) -> Tensor")


@torch.library.impl(_LIB, "paged_kv_write", "CompositeExplicitAutograd")
def _paged_write_impl(k, v, k_pool, v_pool, slot_mapping):
    """k, v [T, Hkv, d]; pools [NB, BS, Hkv, d]: flat position slot_mapping[t] receives token t."""
    k_pool, v_pool = k_pool.clone(), v_pool.clone()
    nb, bs, hkv, d = k_pool.shape
    kf, vf = k_pool.view(nb * bs, hkv, d), v_pool.view(nb * bs, hkv, d)
    for t in range(k.shape[0]):
        kf[int(slot_mapping[t])] = k[t]
        vf[int(slot_mapping[t])] = v[t]
    return k_pool, v_pool


@torch.library.impl(_LIB, "paged_kv_write", "Meta")
def _paged_write_meta(k, v, k_pool, v_pool, slot_mapping):
    return k_pool.new_empty(k_pool.shape), v_pool.new_empty(v_pool.shape)


@torch.library.impl(_LIB, "paged_attention", "CompositeExplicitAutograd")
def _paged_attn_impl(q, k_pool, v_pool, block_table, slot, pos, scale):
    """q [T, Hq, d]; attends positions 0..pos[t] of the token's sequence through its block table."""
    t_n, hq, d = q.shape
    bs, hkv = k_pool.shape[1], k_pool.shape[2]
    rep = hq // hkv
    out = torch.zeros_like(q)
    for t in range(t_n):
        n = int(pos[t]) + 1
        idx = torch.arange(n)
        blocks = block_table[int(slot[t]), idx // bs].long()
        k = k_pool[blocks, idx % bs].repeat_interleave(rep, dim=1)  # [n, Hq, d]
        v = v_pool[blocks, idx % bs].repeat_interleave(rep, dim=1)
        sc = torch.einsum("hd,nhd->hn", q[t], k) * scale
        out[t] = torch.einsum("hn,nhd->hd", sc.softmax(dim=-1), v)
    return out


@torch.library.impl(_LIB, "paged_attention", "Meta")
def _paged_attn_meta(q, k_pool, v_pool, block_table, slot, pos, scale):
    return q.new_empty(q.shape)