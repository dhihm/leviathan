# Model Target: SmolLM2-135M (Base)

This document specifies the first real model target. **No full-model
execution is claimed yet, and no weights have been downloaded.** Download
approval is pending for the approximately 271 MB model files listed below.

## Source (official facts, pinned)

- Model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
  (base), Apache-2.0, ungated.
- Pinned revision:
  [`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)
- Configuration (from the pinned `config.json`): hidden size 576,
  intermediate size 1536, 30 layers, 9 query heads, 3 KV heads (GQA),
  head dim 64, vocab 49,152, tied embeddings, RMSNorm eps 1e-5, SiLU
  activation, RoPE theta 100000 (non-interleaved), BF16 weights.
- Language caveat: SmolLM2 is English-first; it was chosen as a small,
  permissively licensed target, not for Korean quality.
- Selected files and exact sizes:

| File | Bytes |
|---|---|
| `model.safetensors` (weights) | 269,060,552 |
| `tokenizer.json` | 2,104,556 |
| `config.json` | 704 |
| `tokenizer_config.json` | 3,658 |
| `special_tokens_map.json` | 831 |
| **Total** | **271,170,301** |

- Weights SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Local cache check performed: no compatible decoder/model files were found
  in the checked local caches, so a download is required.

## Planned implementation (all original Swift; no model weights in git)

1. **Loader** — original Swift safetensors reader for the BF16 tensor blob,
   streamed/mmap-loaded to avoid a duplicate in-memory weight copy. Weights
   are never committed to git and keep their Apache-2.0 license; the MIT
   grant of this repository does not cover model files.
2. **Validated graph** — a Llama-architecture graph for the SmolLM2-135M
   configuration, built and validated through the same IR discipline as the
   existing dense/relu pipeline.
3. **Metal kernels (original MSL)** — embedding, RMSNorm, RoPE, GQA causal
   attention, SwiGLU, and LM head, following the verified lower layer
   (runtime source compilation, explicit validation before dispatch,
   completed-command-buffer checks, CPU-oracle comparison).
4. **Generation** — greedy token-ID generation first; the exact tokenizer is
   wired in before any text output. No text-serving claims until token IDs
   are verified.
5. **Bounded measured tuning decision** — a small fixed candidate set (e.g.
   threadgroup widths 32/64/128). Each candidate must pass the correctness
   gate, then is ranked by median measured device time with warmup and a
   finite time/iteration budget. The result is best-of-tested under the
   stated budget — explicitly not a claim of global optimality.
6. **Serving** — only after real GPU generation works: localhost-only,
   bounded requests.

## Derived memory budget (conservative, before measurement)

- F32 weights in memory: 134,515,008 values x 4 B = 538,060,032 B ≈ 513 MiB.
- F32 KV cache at the initial fixed context of 256 tokens:
  2 x 30 layers x 3 KV heads x 64 head dim x 256 tokens x 4 B
  = 11,796,480 B ≈ 11.25 MiB.
- Conservative operating budget: **2 GiB of the 32 GiB unified memory**,
  leaving headroom for the runtime, command buffers, and measurement noise.
  To be replaced by measured numbers once real generation runs.

## Fixed initial limits

- Context: 256 tokens (fixed).
- Prompt cap: 128 tokens.
- Generation cap: 32 tokens.

## Approval prerequisite

Model download approval is pending; no weights have been downloaded.
Nothing in the current repository depends on the download.
