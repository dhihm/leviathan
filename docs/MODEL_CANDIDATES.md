# Model Candidates (M1 Max, 32 GB unified memory, ~101 GiB free disk)

> **Short recommendation:** target **Gemma 4 26B-A4B (MoE), official QAT
> Q4_0 GGUF** (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) as the first real
> model. It is recent (2026), Apache-2.0, ungated, genuinely MoE (128
> experts), and its 14.44 GB Q4_0 weights are expected to fit fully
> resident in 32 GB unified memory — an **estimate until measured**. First
> milestone: full-resident GPU token-generation correctness with weights
> kept quantized on the GPU; SSD expert streaming is evaluated only later
> as a controlled-memory-budget comparison mode, never forced.
> Qwen3.8-Flash-Next is real and open but far too large as a first target.
> The prior SmolLM2-135M target is superseded.

All facts below were read from official public metadata (Google AI docs,
Hugging Face `google/` and `Qwen/` repos and API). No weights beyond the one
approved 14.44 GB GGUF download (in progress, external to git) were fetched,
and no license was accepted on any portal.

## Name verification (user-suggested names)

- **"Gemma 4"** — real. Announced by Google on 2026-04-02; family: E2B, E4B,
  12B Unified, 26B A4B (MoE), 31B (dense). Sources:
  [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4),
  [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4),
  [release notes](https://ai.google.dev/gemma/docs/releases).
- **"Qwen3.8 Flash"** — the exact open-weights repo is
  **`Qwen/Qwen3.8-Flash-Next`** (created 2026-08-24). "Qwen 3.8 Flash" also
  exists as an Alibaba API model (2026-08-26); the open-weight Flash-Next is
  the locally relevant one. Source:
  [Qwen blog](https://qwen.ai/blog?id=qwen3.8),
  [HF repo](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Candidate 1 (recommended): Gemma 4 26B-A4B — MoE

- Exact IDs: `google/gemma-4-26B-A4B` (base) → `…-it` → QAT unquantized →
  **`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`** (GGUF, Q4_0 QAT).
- Verified revision `d1c082be9cf3c8a514acf63b8761f4b41935842e` (lastModified
  2026-07-17), **`gated: false`**, license **Apache-2.0** per HF API (the
  card also links the
  [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license); no
  portal acceptance was performed).
- Architecture (official model card): 25.2B total / 3.8B active params,
  30 layers, **128 experts, 8 routed + 1 shared**, hybrid attention
  (local 1024-token sliding window interleaved with global; unified KV,
  p-RoPE), vocab 262K, 256K context, text+image input (text-only first; the
  1,194,828,160-byte `mmproj` file is intentionally not downloaded).
- Exact bytes: `gemma-4-26B_q4_0-it.gguf` = **14,439,363,584 bytes**
  (~13.4 GiB), SHA-256
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
  QAT claim: "similar quality to bfloat16" at Q4_0.
- Memory estimate: weights ≈ 13.4 GiB quantized, kept **quantized on the
  GPU** — the model is never fully expanded to F32/BF16 in memory; only
  activations are dequantized. Whether it fits resident is an estimate
  until measured. KV-cache per-layer head counts are not yet extracted
  from the GGUF headers (honest unknown); hybrid local attention bounds
  most layers' KV at the 1024-token window. A conservative first operating
  budget is **20–22 GiB of the 32 GiB** unified memory, to be replaced by
  measured peak numbers.
- Baselines: Ollama 0.23.0 is installed, but the installed version number
  does **not** by itself prove Gemma-4 support — the official model card
  lists Ollama compatibility, and local verification against the exact
  daemon is pending. **MLX/MLX-LM is not installed and no install is
  authorized yet**, and no official MLX build of this model has been
  verified — an MLX comparison would also be format-confounded (different
  quantization numerics) and is deferred.
- Download status: in progress at
  `<workspace>/models/gemma-4-26B-A4B-Q4_0/`
  (outside git; never committed; model files keep their own license — the
  repo's MIT grant does not cover them). The file must not be treated as
  complete until the SHA-256 above matches.

## Candidate 2 (rejected as first target): Qwen3.8-Flash-Next — MoE

- Exact ID: `Qwen/Qwen3.8-Flash-Next` (ungated, `qwen-community-1.0`
  license), created 2026-08-24.
- 125B-parameter LM, ~6B active, **512 experts, 10 routed + 1 shared**,
  plus a 51B n-gram embedding (20M bigram/trigram table) and 4B MTP layer;
  48 layers, hybrid Gated DeltaNet + sparse attention; 262K context
  (YaRN-extensible to 1M).
- Only official format is BF16 safetensors, ~360 GB across 131 shards —
  roughly 3.5x the free disk and far beyond 32 GB unified memory, so a
  resident run of the official release is not feasible on this machine;
  some compressed/SSD-backed approach would be required (community
  quantizations exist but are not verified here). That is exactly why it
  is wrong as a correctness-first first target.

## Candidate 3 (dense comparison only): Gemma 4 31B

- Same family/license as candidate 1; dense 31B, 256K context. A 4-bit
  weight floor would be ≈ 15.5 GB before working memory — comparable
  footprint to candidate 1 but with every parameter active per token.
  Included only as an honest dense-vs-MoE comparison; it is **not** an
  SSD-expert-streaming target (there are no experts). Exact quantized file
  sizes/IDs not yet pinned (not pursued).

## SSD expert streaming: design position (not yet implemented)

- Bounded resident set = shared components (attention, shared expert,
  embeddings) always resident; routed-expert cache with an explicit byte
  budget, LRU/heat eviction, async `read`/`prefetch` overlapping compute
  via a dedicated I/O queue; on Apple Silicon unified memory this is
  host-RAM-page/SSD interplay, **no PCIe device copy is assumed**.
- Correctness first: streaming path must produce bit-comparable logits to
  the resident path on fixed prompts before any tuning.
- **Automatic selection:** if the full model fits in the measured memory
  budget, run fully resident (SSD caching can only add latency); engage
  the SSD expert cache only under a controlled, explicitly set memory
  budget — chosen by measurement, never forced.

## Benchmark protocol (planned; no results claimed yet)

Same model, revision, exact quantization file, tokenizer, context length,
prompts, generation lengths, and sampling settings across engines.
Metrics: cold/warm TTFT, decode tok/s, prefill tok/s, peak resident/unified
memory, actual SSD read bandwidth, expert-cache hit rate, read
amplification, p50/p95 token latency. If comparing Ollama (GGUF) against a
future MLX build (MLX quant), the quantization numerics differ — that
confound must be stated with any numbers. No global advantage is claimed in
advance; performance is a measured target, not a promise, and small active
parameter counts do not by themselves guarantee speed.

## Honest unknowns

- Per-layer KV head counts / head dim for 26B-A4B (pending GGUF header
  read after checksum-verified download).
- Official MLX-format availability for Gemma 4 26B-A4B (unverified).
- All performance numbers (nothing measured yet; Ollama daemon not yet
  run; MLX not installed).
- Gemma-4 license terms text beyond the Apache-2.0 tag on the GGUF repo
  (not accepted anywhere).

## Superseded target (historical)

SmolLM2-135M was the original milestone-1 target (tiny, safe, Apache-2.0).
It is superseded by candidate 1 as the first *useful* model target; the
existing dense/relu compiler, GPU kernels, validation evidence, and all
current tests are unchanged. The historical SmolLM2 specification is
preserved in git history (see docs/MODEL_TARGET.md history).
