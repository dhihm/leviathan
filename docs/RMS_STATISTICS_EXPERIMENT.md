# Shared RMS statistics lowering

Three RMS normalizations in each of the 30 Gemma probe layers consume the same immutable SSA input, with different scale/apply outputs. The new original pass identifies safe groups during preparation and computes the ordered inverse statistic once, leaving each apply at its original schedule position. It does not eagerly overwrite downstream consumers' physical buffers.

## Safety and execution

`RMSStatisticsPlan` derives a dense instruction-indexed action table once from the validated canonical SSA. Matching includes exact input version, F32 vector type, width, vector count, layout, attributes and epsilon bits. The analysis models physical backing aliases (including the fused gate/up buffer), closes groups at writes and opaque effects, and excludes in-place input destruction and unsupported layouts. The measured schedule has 30 groups with 60 reused reductions.

The first group member stages the input and performs the same ordered float4 statistic reduction as cooperative RMS; it stores the inverse in a small persistent GPU buffer. Later members apply that inverse using the same multiplication association. Group readiness is fresh for every token, including after reset or error. The inverse is never read by the CPU; dispatches use the existing ordered encoder. Actual pipeline thread limits, device memory limits, weight geometry/alignment, spans and slot limits gate dispatch. Unsupported planned operations fall back; an installed generated RMS kernel retains precedence. Incompatible arithmetic candidates are rejected before execution.

The prepared path does not hash, lower or analyze per token. Independent lazy pipelines do not change the shared pipeline-key identity. Kernel source/version, analysis version and the explicit required candidate flag bind artifacts to this pass; older artifacts missing the new field are rejected. Forge preserves the previous 26 candidates and adds one compatible statistics-reuse candidate, rather than silently replacing existing implementations.

## Focused development evidence

The implementation and repository tests were externally authored through the user-approved existing Z.ai GLM-5.3/ZCode route. Exact patch replay against published phase34 matched all six changed source files. Root reviewed/applied the patches and prepared independent temporary measurement drivers. [Source provenance](evidence/rms-statistics-experiment/verified-author-provenance.json) and [development attempts](evidence/rms-statistics-experiment/development-attempts.json) are retained; no source from other projects was copied.

The release build and 1,056 checks passed. Pure analysis tests cover versions, epsilon, shapes, dtype, layout, aliasing, effects and mutation boundaries; actual Metal tests compare scaled/unscaled and in-place outputs against the existing ordered implementations.

An isolated 2816-float component probe used two warmups per configuration and four alternating rounds, each with 128 three-output groups. All scaled/unscaled outputs were bit-exact. Median GPU time was 209.879 µs/group for three independent reductions versus 105.640 µs/group with shared statistics; wall time was 213.542 µs versus 109.500 µs. This is [component evidence](evidence/rms-statistics-experiment/component-results.json), not whole-model throughput.

A separate actual resident-model check used the same incumbent flags and verified all 262144 logits for four teacher-forced steps across two independent resets. Both runs were bit-exact with exact IDs, 150 computes and 300 reuses over five processed tokens, and zero fallbacks. Engine identity remained stable after lazy compilation. These [short diagnostic-install checks](evidence/rms-statistics-experiment/focused-results.json) do not approve a production artifact.

## Production and fixed comparison

Fresh production qualification passed in 875.75 seconds; persisted-artifact reload passed separately in 60.61 seconds. Forge selected statistics reuse on the same batching/vectorized-load/GPU-KV/parallel-attention/cooperative-RMS/cooperative-Q6 flags and Q4 width 64 as published phase34. The plan digest is `75ca6285b7e4076b11a3e53d750d4dc5000305bd05faa90748bfd19602648613`; the composite digest is `d63eb8f1786463a1af98f9f2801af4fd004754a4f01c763b304b8c4e51a16ad0`. All 27 candidate results, including rejected short-gate combinations, remain in [qualification.json](evidence/rms-statistics-experiment/qualification.json).

Actual prepared serving passed raw Hello, English chat and Korean chat, 32 generated tokens per fixture, all 262144 logits with normalized difference < 1e-4 and exact IDs. Statistics computed/reused totals were 1980/3960, 3120/6240 and 3240/6480, with zero fallbacks. These include prefill and candidate gate replay; they are not per-token counters.

The predeclared fixed protocol uses four serial A/B, B/A, A/B, B/A pairs, with four generations per arm and the first excluded. Each score is the median of three warm 31-step decode-wall totals, using the same original GGUF/raw Hello/BOS alignment. Startup, identity and prefill are excluded; synchronous graph execution and output readback are included. No owned author, build or profiling jobs overlapped timing. Every repetition and warmup is retained, with no adaptive stopping, reruns or exclusion.

| Pair | Order | Phase34 tokens/s | Experiment tokens/s | Decode time reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | A/B | 7.731 | 8.228 | 6.045% |
| 2 | B/A | 7.831 | 8.069 | 2.943% |
| 3 | A/B | 7.680 | 7.956 | 3.473% |
| 4 | B/A | 7.564 | 7.638 | 0.974% |

All four pairs improved. Median paired decode time reduction is **3.208%**. Equal-work pooled totals give 3.347% less decode time and 3.463% more throughput. Source, both binaries, plans and composites stayed unchanged, and every cross-arm ID matched. See [all fixed results](evidence/rms-statistics-experiment/fixed-results.json).

**The predeclared promotion criterion failed:** every pair had to improve by more than 2%; pair 4 improved by only 0.974%. The implementation passed correctness qualification. On 2026-10-04 at 11:15 UTC, the user explicitly requested its inclusion in main despite the measured variability. Main therefore includes the pass by user decision; this does not change or satisfy the earlier every-pair criterion. The original source remains preserved at private experiment commit `2562e0edceda99e64e4dd0e0635e01a853d3a6ae`. No additional timing series was substituted to obtain a pass. The final inclusion tree was rebuilt, all 1,056 checks passed again, and a separate persisted production reload/full-logit gate passed with byte-identical qualified runtime sources and executable; see [inclusion verification](evidence/rms-statistics-experiment/user-inclusion-verification.json).

The experiment's absolute 7.638–8.228 tokens/s must be compared with its contemporaneous 7.564–7.831 tokens/s incumbent, rather than the previous series' 9 tokens/s. [All 16 resource observations](evidence/rms-statistics-experiment/resource-observations.json) preserve shared-host conditions; other applications remained active, and no idle-host or causal thermal claim is made.

## Post-comparison bottlenecks

A separate warmed census measured 122.391 ms/token in normal batched execution, with 55.44% GPU intervals and 38.76% host flush work. It still used 97 command buffers and 1289 dispatches per decode. This diagnostic sample is not a replacement for the fixed comparison. Both normal and intrusive replay showed 930 statistics computes, 1860 reuses and zero fallbacks over 31 decode tokens. Intrusive replay was bit-exact against its batched reference. Its ranking was generic Q4 at 26.27%, remaining cooperative RMS 19.72%, F32 router 19.02%, statistics compute 3.51% and cooperative Q6 4.76%; intrusive rankings are not normal wall shares. See [post-budget.json](evidence/rms-statistics-experiment/post-budget.json).

Static canonical-SSA/source analysis identifies 30 CPU router-scale read/multiply/write boundaries, 30 expert-output CPU zeroing boundaries and five host copies, alongside 30 CPU top-k boundaries. The next coherent batch should move router scales, resident expert zeroing and copies into the ordered GPU path. A flush cannot simply be removed before CPU memset: GPU zeroing must preserve buffer lifetime and the original eight ordered axpy merges. Preserve the two-round float multiplication in router scaling, copy bit patterns, cache/streaming fallback lifetime and strict full-model gates. Counts are not isolated timing attribution; see [boundary evidence](evidence/rms-statistics-experiment/next-host-boundaries-analysis.json).

## Remaining scope

The throughput gap is not resolved by this pass alone. CPU routing/synchronization, many dispatches and the under-parallelized F32 router remain measured targets. No new matched Ollama/MLX benchmark is claimed. Lightweight models and general Python model coverage remain goals; the existing 14.44 GB 26B MoE is a probe. No model/dependency downloads, new secrets or purchases were needed.
