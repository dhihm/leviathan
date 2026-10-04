# GPU KV append Forge pass

The Gemma Forge candidate set adds an opt-in GPU cache append. K/V values remain on the GPU: an original uint32 copy kernel stores their words into validated cache slots before attention consumes them. The existing CPU readback and memcpy path remains the baseline. The same pass is available in the validated SSA executor and diagnostic runner path. This removes one command-buffer boundary per layer in the measured batched configuration, rather than changing attention or matrix arithmetic.

## Selection and identity

The bounded candidate set grows from 13 to 15. An unbatched copy candidate and a batched vectorized-load candidate are locally compiled, numerically gated and measured by the existing Forge. No agent or network participates in optimization or inference. The measured 32-token full-workload fallback rejected faster short-workload tiled candidates at the unchanged logit tolerance and selected the batched GPU append candidate. Serving executes the chosen validated plan; GPU append is not an unconditional default.

Candidate serialization and the engine's candidate signature include `gpuCacheAppend`. Actual shader, pipeline and executable identities change too. Pre-axis candidate records reject instead of silently interpreting the missing field; rebuilds explicitly re-forge old artifacts. The measurement artifacts bind the tested CLI executable and are not portable records for a different rebuilt executable.

## Safety and qualification

Positive width, nonnegative position, checked byte-width/offset/end arithmetic, both source/destination capacities and UInt32 word limits are validated before dispatch. Source buffers may be larger than their active values. The runner owns all scratch/cache buffers through completion, and existing ordered compute segments place append between the producers and attention consumers. Later CPU mutation/readback retains the existing completion boundary. Command failure handling and accounting remain unchanged.

The full Swift regression suite passes 837 checks. New cases cover candidate identity, old-record rejection, overflow/capacity guards, and actual Metal copies at two offsets with ragged widths, complete destination sentinels and unchanged sources. The generic loaded Python DAG runtime passes 22 acceptance cases and seven actual-worker reliability cases on freshly bound artifacts. All implementation and repository tests were externally authored through configured GLM-5.3/ZCode; exact scope-normalized patch replay was checked. Unsafe author drafts were retained for review and never applied.

The existing full-model Forge gate and a separate fresh-process artifact reload both pass raw Hello for 32 generated IDs and 31 decode steps, with finite full-vocabulary logits, normalized differences below 1e-4 and exact greedy equality. The matched CPU control also passes its own full 32-token gate. This is one fixture, not cross-language or general model qualification. No correctness tolerance was relaxed. The short Forge gate measured normalized differences of zero for the append candidates; the full gate is reported as passing its existing tolerance rather than an unrecorded bit-identity claim.

## Matched performance

Both arms use the same Apple M1 Max, existing 14,439,363,584-byte GGUF, CLI binary, validated graph and plans whose only candidate difference is GPU append. Each round has three serial AB/BA/AB process pairs. Each process generates 32 IDs three times with BOS-only reuse; repetition one is excluded as warmup, and repetitions two/three time 31 decode steps each. Startup, identity hashing and prompt prefill are excluded. No profiling, build, external author or competing owned GPU job runs during these measurements. The shared Mac is not claimed idle.

| Round | Median paired wall reduction | Per-pair reductions |
| --- | ---: | --- |
| Initial | 3.281% | 3.281%, 3.801%, 1.247% |
| Complete fresh repeat | 2.027% | 2.027%, 2.224%, 1.748% |

All six pairs improve and cross-arm generated IDs agree exactly. The combined median paired reduction is 2.126%. Two pairs are below 2%; this is a modest, variable improvement, not a universal or statistically established speedup. Candidate warm aggregate rates span 2.664–2.713 tokens/s. These synchronous CLI decode wall measurements include graph execution and output readback; Ollama server eval counters have a different scope. The separately measured Ollama reference remains substantially faster. Closing that gap is unfinished.

Command buffers fall from 127 to 97 per token while kernels rise from 1799 to 1829. Fewer host synchronization boundaries explain the intended mechanism; kernel-count or buffer-count changes alone are not throughput evidence. SIMD and this pass were measured under different configurations, so their gains must not be added.

[First round](evidence/gpu-kv-append/round-1.json), [complete repeat](evidence/gpu-kv-append/round-2.json) and [summary](evidence/gpu-kv-append/summary.json) preserve all timing outcomes, including warmup rows and below-threshold pairs. No weights, credentials or external-author conversation identifiers are included. No model or dependency was downloaded.
