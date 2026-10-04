# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan logo" width="140">

![Whale devouring a kraken](docs/assets/leviathan-hero.png)

## Project Goal

Leviathan is a personal project building a local LLM inference engine for Mac GPUs. It aims to automatically profile the actual hardware and model, compile and optimize Python-described workloads without an agent involved at optimization or runtime, and efficiently serve reused, bounded execution plans. Lightweight small models are the target class — the current 26B MoE model serves as a probe — and performance competitive with MLX or Ollama is the goal, with broader Python model coverage and the performance gap still unfinished.

## Current Progress

- A Python frontend lowers static `torch.export` graphs to a validated model/compute DAG, with correctness-gated, measured Metal lowering and a persistent loaded-actual-input GPU runtime ([runtime details](docs/DAG_FORGE_RUNTIME.md)).
- Real Gemma 26B-A4B Q4_0 GGUF inference runs: tokenizer, quantized Metal GPU execution, loopback serving, and automatic bounded measured passes.
- Shape-specific SIMD kernels and a bounded hash memory fix are already published ([SIMD details](docs/Q4_SIMD_FORGE.md)).
- Correctness gates and test coverage throughout — currently 815 checks ([detailed progress](docs/FORGE_PROGRESS.md)).
- Performance: closing the gap with Ollama and broader Python model coverage are in progress.

[MIT](LICENSE); model weights keep their upstream terms.