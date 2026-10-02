# Model Target: Gemma 4 26B-A4B (MoE), QAT Q4_0 GGUF

> Supersedes the SmolLM2-135M target (2026-10-02; see git history for the
> old spec). Selection rationale and candidate comparison live in
> [MODEL_CANDIDATES.md](MODEL_CANDIDATES.md). The existing dense/relu
> compiler, Metal kernels, and all current tests are unchanged.

## Source (official facts, pinned)

- Model: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (QAT Q4_0 GGUF of the
  instruction-tuned Gemma 4 26B-A4B).
  Official model card:
  https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf
  Official Gemma 4 model card:
  https://ai.google.dev/gemma/docs/core/model_card_4
- Pinned revision `d1c082be9cf3c8a514acf63b8761f4b41935842e`
  (lastModified 2026-07-17), `gated: false` (verified via the public HF
  API; no portal license acceptance was performed).
- License: Apache-2.0 per the HF API license tag; the card additionally
  links the Gemma 4 license
  (https://ai.google.dev/gemma/docs/gemma_4_license). The actual license
  text must be preserved alongside the local model files; model files are
  never committed to this repository and are not covered by its MIT grant.
- Architecture (official model card): 25.2B total / 3.8B active params,
  30 layers, 128 experts with 8 routed + 1 shared, hybrid attention
  (local 1024-token sliding window interleaved with global layers using
  unified KV and p-RoPE), vocab 262,144, 256K context, text+image input
  (text-only first; the `mmproj` vision file is intentionally not
  downloaded).
- Selected file, exact bytes and checksum (verified via HF API):
  `gemma-4-26B_q4_0-it.gguf` = 14,439,363,584 bytes, SHA-256
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
- Local download (approved 2026-10-02): **complete and SHA-256-verified**
  at
  `/Users/dhihm/Documents/Codex/2026-10-02/task-5/models/gemma-4-26B-A4B-Q4_0/`
  (outside git; license text preserved alongside as
  `LICENSE-APACHE-2.0.txt`; the file is 14,439,363,584 bytes and matches
  the pinned hash). The original GGUF reader has inspected the real file:
  658 tensors — 392 F32 (46,056,568 bytes), 265 Q4_0 (13,771,929,600
  bytes), 1 Q6_K (605,552,640 bytes, the tied `token_embd.weight`
  [2816, 262144] / LM head); header 15,823,936 bytes, 52 metadata
  entries, mmap read-only with no full-file copy. **Status: full-model
  GPU token generation and SSD expert streaming are NOT implemented
  yet**; the Q4_0 and Q6_K quantized matvec kernels and FP64-gated
  correctness (see [VALIDATION.md](VALIDATION.md)) are the foundation.

## Goal (first milestone)

The first real original-GPU result is **a single generated token ID from
the actual Gemma 4 graph on the Metal GPU, with numerical correctness
against the exact GGUF baseline**. A generic Llama-architecture substitute
is invalid for this milestone. Scalar-kernel parity alone does not prove
full-model correctness; reference logits/token IDs must come from the
identical GGUF file run through an existing engine (e.g. locally installed
Ollama, once its exact Gemma-4 support is verified — not assumed from the
version number; otherwise llama.cpp or another verified runner).

## Required implementation components (all original Swift/MSL)

1. **GGUF reader** — original parser for the GGUF container (header,
   metadata KV, tensor directory, alignment) validated against the
   checksum-verified file before any weight access.
2. **Q4_0 dequantizing kernels** — fused matvec (and matmul) operating
   directly on Q4_0 blocks. **Quantized weights stay quantized on the
   GPU; the model is never fully expanded to F32/BF16 in memory.**
3. **Exact Gemma 4 layer graph** — hybrid local sliding-window (1024) and
   global attention with unified KV, p-RoPE, RMSNorm, MoE routing
   (top-8 of 128 routed experts) plus the shared expert, exactly as
   specified by the GGUF metadata and official model card.
4. **Token generation** — greedy token-ID generation first; tokenizer
   wiring and text output only after token IDs match the baseline.
5. **Correctness gate** — per-kernel scalar CPU oracles plus the
   full-model reference comparison described above; tolerance stated
   before comparison.

## Memory policy (estimates until measured)

- Weights: 14,439,363,584 bytes quantized, kept quantized and fully
  resident on the GPU. "Fits in RAM" is an **estimate until measured**;
  the operating budget is confirmed only by real peak-memory numbers.
- Working set: dequantized activations only, never dequantized weights.
- KV-cache size depends on per-layer KV head counts not yet extracted
  from the GGUF headers (honest unknown); hybrid attention bounds local
  layers at the 1024-token window.
- First measured target: full-resident operation within a conservative
  budget (~20–22 GiB of the 32 GiB unified memory) on the M1 Max. The
  SSD expert-cache path is a later, separately measured comparison mode
  under an explicit controlled memory budget — never forced when the
  model fits resident.

## Fixed initial limits

- Context: 256 tokens (fixed). Prompt cap: 128 tokens. Generation: 1
  token for the first correctness milestone, then 32.

## Constraints

- The existing 73 tests must keep passing; nothing in the current
  repository may regress for this target.
- No model weights in git; license text preserved with the model files.
- No performance claims before measurement.
