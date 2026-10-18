# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan logo" width="140">

![Whale devouring a kraken](docs/assets/leviathan-hero.png)

## Project Goal

Leviathan is a personal project building a local LLM inference engine for Mac GPUs (Metal) and Windows AMD GPUs (Vulkan). It aims to automatically profile the actual hardware and model, compile and optimize Python-described workloads without an agent involved at optimization or runtime, and efficiently serve reused, bounded execution plans. Lightweight small models are the target class — the current 26B MoE model serves as a probe — and performance competitive with MLX or Ollama is the goal, with broader Python model coverage and the performance gap still unfinished.

## Current Progress

### Windows / AMD Vulkan (Rust, `rust/`)

Ornith-1.5-9B (Q4_K_M) runs on the Radeon 890M of an ASUS ROG Ally X through Vulkan with the same Forge design: kernels emitted from an op spec, schedules derived per device and column count, correctness gates against a CPU reference, matrix-core (cooperative matrix) prefill, and MTP speculative decoding. Measured against llama.cpp b11393 (Vulkan) by alternating the engines on the same GPU under sustained load (about 35 W, 80 C), 512-token prompt, greedy:

| | Leviathan | llama.cpp | ratio |
| :--- | :--- | :--- | :--- |
| prefill (tok/s) | 292 - 294 | 312 | 0.94 |
| plain decode (tok/s) | 18.7 | 18.4 | 1.02 |
| MTP speculative decode, two chained drafts (tok/s) | 31.5 mean over five natural-language prompts (27.9 - 35.2) | not applicable | about 1.7 vs. llama.cpp plain |

Prefill and plain decode: one synthetic 512-token prompt, three rounds, one ROG Ally X. Speculative decode depends on draft acceptance and therefore on the prompt (a 242-token natural-language mix reached 37.4 tok/s in the interleaved run, a few other prompts 28 - 35). Details, caveats and reproduction commands: [Windows / Vulkan](docs/WINDOWS_VULKAN.md).

### Mac / Metal (Swift and Python)

- A Python frontend lowers static `torch.export` graphs to a validated model/compute DAG, with correctness-gated, measured Metal lowering and a persistent loaded-actual-input GPU runtime ([runtime details](docs/DAG_FORGE_RUNTIME.md)).
- Real Gemma 26B-A4B Q4_0 GGUF inference runs: tokenizer, quantized Metal GPU execution, loopback serving, and automatic bounded measured passes.
- Shape-specific SIMD kernels and a bounded hash memory fix are already published ([SIMD details](docs/Q4_SIMD_FORGE.md), [GPU KV storage](docs/GPU_KV_APPEND.md)).
- Prepared SSA execution and bounded parallel attention now compose through Forge, with unchanged full-logit gates and feature attribution switches ([batch details](docs/PREPARED_ATTENTION_BATCH.md)).
- Normal Forge and inference can now activate and reuse the verified MoE/Q4 composite: the measured batch reduced matched decode wall by 50.8%, reaching 6.47–6.50 tokens/s on the current M1 Max probe ([composite details](docs/COMPOSITE_FORGE_BATCH.md)).
- Exact-order cooperative RMS now participates in bounded Forge composition and passed three full-model fixtures; measured gains and timing instability are recorded together ([RMS batch](docs/RMS_Q6_FORGE_BATCH.md), [Ollama comparison scope](docs/OLLAMA_COMPARISON_SCOPE.md)).
- [Q6 SIMD Forge batch](docs/Q6_SIMD_FORGE.md): full production/reload and three prepared-serving gates passed; four fixed comparison pairs improved, with 10.68% median decode time reduction. Remaining performance gap and bottlenecks are documented.
- [Shared RMS statistics lowering](docs/RMS_STATISTICS_EXPERIMENT.md) is included in main by explicit user decision. The release build, 1,056 checks and three serving gates passed; four fixed pairs improved by a median 3.21%, while the earlier every-pair performance threshold remained unmet.
- [Router residency batch](docs/ROUTER_RESIDENCY_EXPERIMENT.md): automatic selection and reload, 1,533 checks, and three serving gates passed. Four fixed pairs reduced matched decode time by a median 23.65%, reaching 13.22–13.59 tokens/s; the Ollama gap remains unfinished.
- Correctness gates and test coverage throughout — currently 2,094 checks ([detailed progress](docs/FORGE_PROGRESS.md)).
- [Cached model identity](docs/CACHED_MODEL_DIGEST.md): actual model SHA-256 is computed once, reused across processes, and refreshed only explicitly; cache hits do not re-read weights.
- [Command timeline diagnosis](docs/COMMAND_TIMELINE_DIAGNOSIS.md): default-off GPU/host interval tracing passed three strict full-model fixtures; timing remains provisional under observed paging.
- Performance: closing the gap with Ollama and broader Python model coverage are in progress.

[MIT](LICENSE); model weights keep their upstream terms.
