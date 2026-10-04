# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan logo" width="140">

![Whale devouring a kraken](docs/assets/leviathan-hero.png)

## Project Goal

Leviathan is a personal project building a local LLM inference engine for Mac GPUs. It aims to automatically profile the actual hardware and model, compile and optimize Python-described workloads without an agent involved at optimization or runtime, and efficiently serve reused, bounded execution plans. Lightweight small models are the target class — the current 26B MoE model serves as a probe — and performance competitive with MLX or Ollama is the goal, with broader Python model coverage and the performance gap still unfinished.

## Current Progress

- A Python frontend lowers static `torch.export` graphs to a validated model/compute DAG, with correctness-gated, measured Metal lowering and a persistent loaded-actual-input GPU runtime ([runtime details](docs/DAG_FORGE_RUNTIME.md)).
- Real Gemma 26B-A4B Q4_0 GGUF inference runs: tokenizer, quantized Metal GPU execution, loopback serving, and automatic bounded measured passes.
- Shape-specific SIMD kernels and a bounded hash memory fix are already published ([SIMD details](docs/Q4_SIMD_FORGE.md), [GPU KV storage](docs/GPU_KV_APPEND.md)).
- Prepared SSA execution and bounded parallel attention now compose through Forge, with unchanged full-logit gates and feature attribution switches ([batch details](docs/PREPARED_ATTENTION_BATCH.md)).
- Normal Forge and inference can now activate and reuse the verified MoE/Q4 composite: the measured batch reduced matched decode wall by 50.8%, reaching 6.47–6.50 tokens/s on the current M1 Max probe ([composite details](docs/COMPOSITE_FORGE_BATCH.md)).
- Exact-order cooperative RMS now participates in bounded Forge composition and passed three full-model fixtures; measured gains and timing instability are recorded together ([RMS batch](docs/RMS_Q6_FORGE_BATCH.md), [Ollama comparison scope](docs/OLLAMA_COMPARISON_SCOPE.md)).
- [Q6 SIMD Forge batch](docs/Q6_SIMD_FORGE.md): full production/reload and three prepared-serving gates passed; four fixed comparison pairs improved, with 10.68% median decode time reduction. Remaining performance gap and bottlenecks are documented.
- [Shared RMS statistics experiment](docs/RMS_STATISTICS_EXPERIMENT.md): 1,056 experiment checks and three serving gates passed; four fixed pairs improved by a median 3.21%, but the declared every-pair performance threshold was missed. The implementation is preserved in a private experiment branch for the next optimization batch.
- Correctness gates and test coverage throughout — currently 923 checks ([detailed progress](docs/FORGE_PROGRESS.md)).
- Performance: closing the gap with Ollama and broader Python model coverage are in progress.

[MIT](LICENSE); model weights keep their upstream terms.
