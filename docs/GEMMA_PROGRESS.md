# Gemma 4 26B-A4B Bring-Up Progress (working document)

Status date: 2026-10-02. Author: GLM-5.3 via ZCode (original implementation,
from scratch). This is a temporary working record of the first full-model
GPU milestone; scope boundaries and unverified items are stated explicitly.

## Milestone reached

The real Gemma 4 26B-A4B QAT Q4_0 checkpoint (14,439,363,584 bytes,
SHA-256 `3eca3b8f…eca51d`, Apache-2.0, license preserved beside the file)
runs a complete 30-layer forward pass on the M1 Max Metal GPU with
**quantized weights kept quantized** (zero-copy mmap'd `MTLBuffer`; no
full-checkpoint dequantized copy — individual activation vectors and the
current embedding row are F32 by design), and the observed greedy
next-token output **agrees with the same-GGUF Ollama 0.23.0 baseline on
two consecutive generated pieces** for the raw prompt "Hello"
(temperature 0, num_predict 2, baseline run 2026-10-02):

- baseline generated pieces: `" upon"` then `"-"` (Ollama response text)
- Leviathan greedy token IDs: `3324` (`▁upon`) then `236772` (`-`)
- the prompt mapping `[2, 9259]` is **inferred** (BOS=2 from GGUF
  metadata; "Hello" exact vocab lookup 9259; `prompt_eval_count=2`):
  Ollama's API exposes no token IDs, so the exact baseline tokenization
  is unverified; this is observed two-piece greedy agreement under an
  inferred prompt mapping, **not** full logit/layer numerical parity,
  and not a benchmark comparison.

## Measured evidence (valid post-fix runs)

- device Apple M1 Max; `maxBufferLength` 20,100,448,256 B;
  `recommendedMaxWorkingSetSize` 26,800,603,136 B (preflighted before
  allocation)
- weights mapped zero-copy: 14,439,366,656 B (page-rounded file)
- Metal `currentAllocatedSize` 16,286,400,512 B — this **includes** the
  mapped weights; activations + KV caches are the small remainder
- max **sampled** process footprint 1,895,033,152 B (sampled at init and
  token boundaries, not a kernel-level peak trace): only pages actually
  touched are resident — the full non-expert weights plus the 8-of-128
  selected expert slices per layer
- 1,799 kernel dispatches per token (one command buffer per kernel;
  batching is future work). Exact valid post-fix runs, totals including
  all prompt forwards (not per-token steady-state decode benchmarks):
  1 input token — GPU 585.2 ms, wall 1298.1 ms; 2 input tokens — GPU
  1088.9 ms total, wall 2117.2 ms; 3 input tokens — GPU 1358.3 ms
  total, wall 2679.4 ms. No performance claims vs other engines.
- Embedding decoder parity: GPU `g4_q6_embed_row` vs the independent
  scalar `Q6_K.decodeBlock` oracle over all 11 blocks of a full row:
  worst relative difference **1.18e-07**.

## Implementation facts (all original Swift/MSL)

- `Sources/LeviathanCore/Gemma4.swift`: config from GGUF metadata with
  cross-checks; **full 658-tensor contract validation** (exact names,
  dims, types, byte bounds) before any GPU allocation — crafted-GGUF
  rejection is test-covered; persistent device/queue/library/pipelines;
  mmap'd no-copy shared weight buffer (RAII: the buffer's deallocator
  owns the single `munmap`); all matvec/norm/rope/attention primitives
  preflight actual buffer extents.
- Exact semantics implemented from public official primary sources,
  used as reference only (no source text copied): Hugging Face
  Apache-2.0 transformers reference implementation
  [modeling_gemma4.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py)
  and
  [modeling_rope_utils.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py)
  (proportional RoPE), plus the model's own GGUF metadata and
  the model card at
  [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
  (Apache-2.0). Semantics pinned:
  embedding scaled by `sqrt(2816)`; RMSNorm `x·w` (no +1); per-head
  QK-norm; global layers (5,11,17,23,29) are `k_eq_v` (v = raw k_proj
  output, then scale-free v-norm) with head_dim 512 / 2 KV heads and
  proportional RoPE (0.25 partial, theta 1e6, via the file's
  `rope_freqs` divisor factors: first 64 entries 1.0, rest ~1e30);
  local layers head_dim 256 / 8 KV heads, sliding window 1024, theta
  1e4; attention scaling 1.0; shared dense expert (2112) + router
  (scale-free RMSNorm → ·scale·2816^-0.5 → F32 matvec → fp64 softmax →
  top-8 renormalized → ·per-expert QAT scale) + 128 routed experts
  (704); `layer_output_scale` per layer; final norm + tied Q6_K LM head
  + tanh softcap 30.
- Kernels: Q4_0/Q6_K/F32 matvecs (weights addressed base+offset,
  never expanded), Q6_K embedding-row decode, RMSNorm, fused
  gate|up GELU, RoPE, single-query attention with stable softmax, and
  elementwise helpers. Router top-k runs on CPU (exact tie-breaking).

## Bugs found and fixed during bring-up (documented, with evidence)

1. **Metal `tanh` NaN**: the driver's `tanh` returns NaN for |x| ≳ 49
   (standalone probe: 193/400 NaN), silently poisoning GELU and the
   logit softcap. Fixed with an overflow-safe `exp(-2|x|)` form used by
   the real kernels; a test compiles the runtime's own MSL helper and
   compares against scalar `tanh` over ±[0, 1e6] (no NaN, <1e-6 diff).
2. **Q6_K embedding-row decoder misindexing**: lanes ≥16 in each quarter
   read wrong ql/qh bytes (caught by the scalar-oracle parity gate;
   worst diff before fix failed the gate). Rewritten with the canonical
   quarter/lane/subgroup mapping.
3. Earlier bring-up issues fixed before any correctness claim: shared-
   MLP output buffer overwritten by the routed-expert loop; a 2048-wide
   staging buffer too small for 2816-wide projections (out-of-bounds
   GPU writes, found in review before execution); double-free risk in
   mmap cleanup (now RAII-owned by the MTLBuffer deallocator).

## Verification status

- Full suite: **181 passed / 0 failed** (default);
  **184 passed / 0 failed** with `LEVIATHAN_GEMMA_GGUF=<file>` opt-in
  (adds real-file embedding parity + the two greedy-piece baseline
  agreement checks); direct exit status 0 for both. Existing
  Forge/dense/relu tests unchanged. The crafted-GGUF adversarial tests
  exercise the parser, metadata guards, and missing-tensor-set
  rejection; the full 658-tensor geometry validation runs on the real
  file at every runner init, though each individual adversarial
  dimension mutation is not separately fixture-tested.
- Baseline: the installed Ollama 0.23.0 empirically loaded and ran the
  exact GGUF (task-owned `OLLAMA_MODELS` dir and localhost port, file
  registered from disk, no pull); its Gemma 4 support is evidenced by
  that successful run of this checkpoint. The task model was unloaded
  (`keep_alive:0`, `/api/ps` empty) and the task daemon stopped (port
  has no listener); an unrelated pre-existing user Ollama process was
  left untouched. Concurrent GPU residency of both engines fails (GPU
  OOM) — baseline and Leviathan runs must be sequential on this 32 GB
  machine.

## Explicitly NOT done / not claimed

- Tokenizer: only exact vocab piece lookup (`gemma-token-lookup`);
  no BPE/SentencePiece segmentation, no chat-template application,
  no text-prompt support. The BOS-only and "Hello" outputs are not
  presented as usable text generation.
- No logit-level or per-layer parity vs any engine; greedy agreement is
  bounded to the tested prompts.
- Forge integration for Gemma (automatic pass generation over this
  architecture), SSD expert streaming/LRU, serving, prefill batching,
  and multi-token generation loops remain future work; Forge remains
  dense/ReLU-only as documented.
- Performance is unoptimized (per-kernel command buffers); no
  comparative claims vs MLX/Ollama.
