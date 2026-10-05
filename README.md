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
- [Shared RMS statistics lowering](docs/RMS_STATISTICS_EXPERIMENT.md) is included in main by explicit user decision. The release build, 1,056 checks and three serving gates passed; four fixed pairs improved by a median 3.21%, while the earlier every-pair performance threshold remained unmet.
- [Router residency batch](docs/ROUTER_RESIDENCY_EXPERIMENT.md): automatic selection and reload, 1,533 checks, and three serving gates passed. Four fixed pairs reduced matched decode time by a median 23.65%, reaching 13.22–13.59 tokens/s; the Ollama gap remains unfinished.
- Correctness gates and test coverage throughout: the current default regression passed 2,056 checks, with dedicated suites recorded separately ([detailed progress](docs/FORGE_PROGRESS.md)).
- [Cached model identity](docs/CACHED_MODEL_DIGEST.md): actual model SHA-256 is computed once, reused across processes, and refreshed only explicitly; cache hits do not re-read weights.
- [Command timeline diagnosis](docs/COMMAND_TIMELINE_DIAGNOSIS.md): default-off GPU/host interval tracing passed three strict full-model fixtures; timing remains provisional under observed paging.
- [Prepared GPU routing](docs/DEVICE_ROUTING_NUMERICAL_PHASE57.md): all three model fixtures and 4,170 actual routes passed; the default-off path removes CPU routing selections and reduces continuation command buffers from 37 to seven without changing output bits.
- [Fixed GPU routing comparison](docs/DEVICE_ROUTING_PERFORMANCE_PHASE58.md): all 48 runs passed correctness; observed TPOT medians were 69.60–72.19 ms versus 84.29–87.56 ms for the incumbent. Failed spread gates leave the result unqualified; defaults are unchanged and no Ollama ratio is claimed.
- [Actual kernel stage attribution](docs/KERNEL_STAGE_ATTRIBUTION_PHASE59.md): all three model fixtures were bit-identical with sampling, covering 125,457 kernel encoders. Changed encoder topology and overlapping timestamp ranges limit latency attribution.
- Isolated Q4 load experiments are closed: [shared weight staging](docs/Q4_WEIGHT_STAGING_PHASE60.md) preserved bits but was 2.48–2.86 times slower; [unrolled register decoding](docs/Q4_PACKED_WORD_PHASE61.md) and [loop-preserving register decoding](docs/Q4_REGISTER_LOOP_PHASE62.md) failed compiled-bit gates, so timing was withheld and no candidate was installed.
- [Frozen minimum-row diagnosis](docs/Q4_FIRST_DIVERGENCE_PHASE64.md) reproduced the original one-bit mismatch and equal decoded operands; trace stores changed the control output, leaving the original arithmetic cause unestablished.
- [Original-pipeline prefix probes](docs/Q4_ORIGINAL_PREFIX_PHASE65.md) reproduced the full-input mismatch and found differences at fixed mask lengths 62 and 64 only. Input perturbation does not identify original intermediate accumulators or compiled instructions.
- Performance: closing the gap with Ollama and broader Python model coverage are in progress.

[MIT](LICENSE); model weights keep their upstream terms.
