# Q6 block-internal SIMD lowering

The tied Q6_K output projection has 2816 input features and 262144 output rows. Its 605,552,640-byte tensor contains 11 quantization blocks per row. The incumbent uses one thread per row; the old tiled path assigns whole blocks to lanes and leaves many lanes without block work. The new original kernel distributes the 32 positions inside each half-block across a SIMD group, accumulates contributions per lane and uses `simd_sum` for the row result. It uses no threadgroup staging array or threadgroup barrier. Addition order changes; bit equality is not claimed.

The compiled pipeline's execution width and thread cap determine legality and one or two rows per threadgroup. Unsupported geometry records a fallback to the plain kernel. Buffer spans retain the incumbent checks. Forge retains 24 candidates and adds two cooperative-Q6 candidates on batching/vectorized-load/GPU-KV/parallel-attention, with cooperative RMS off/on. The new explicit candidate field, source SHA-256 and schedule version bind artifact identity. The independent lazy pipeline does not alter the shared pipeline dictionary. Candidate state resets on completion/error; source hashing is outside token execution. Old tiled Q6 and cooperative Q6 cannot be enabled together.

## Development evidence

The implementation and repository tests were authored through the explicitly authorized existing Z.ai GLM-5.3/ZCode route. Exact source provenance replay matched all five changed source files against the published baseline. An ordered-staging prototype passed its numerical tests but measured 409.660–816.555 ms versus a 21.763 ms incumbent in the component probe. It was rejected before production selection; its measurements remain in [probe-results.json](evidence/q6-cooperative-forge/probe-results.json).

The SIMD revision uses the same actual model tensor with a deterministic synthetic input in a standalone shared buffer. After two excluded warmups, four rounds alternate configuration order. GPU medians were 19.777854 ms for plain, 2.999000 ms for one-row SIMD groups and 3.016438 ms for two-row groups. Maximum normalized output difference was 1.728535e-6 with identical synthetic argmax. This approximately 6.6-fold component improvement is not an end-to-end claim: input, residency and command isolation differ from actual serving. All component measurements remain in [probe-simd-results.json](evidence/q6-cooperative-forge/probe-simd-results.json).

The integrated release build and 923 checks passed. New tests dispatch the actual shader against the independent FP64 oracle at widths 256/512/2816, with mixed-sign scales, saturated quant bytes, cancellation-prone inputs, one/two-row groups and row tails. The hand-computed -315 case remains exact. Four-step actual-model checks compared all 262144 logits: both new candidates retained exact IDs and normalized error 8.344650e-6, with five actual applications and zero fallbacks. This short check did not approve a serving artifact.

## Production qualification

Fresh production selection passed in 788.52 seconds, followed by a separate persisted-artifact reload/full gate in 54.07 seconds. Forge selected cooperative Q6 together with cooperative RMS, batching, vectorized loads, GPU KV and parallel attention; old Q6/F32/Q4 tiling and reassociated tiled RMS remain disabled. No candidate failed the final full gate in this selection. The plan digest is `8875c011801c8186ef3adb1fa15f9dc5eab08227b8fdbd3522da325ac75dd01c`.

The independently qualified Q4 composite selected width 64, whereas the preserved phase33 incumbent uses width 32. Its digest is `1d2accedb489cb5a1182e68243ddb06d5cddb08762a7dffd1d3462d4d81716a4`. Therefore the following comparison measures the accepted batch and its selected layout, not isolated attribution to Q6.

Actual prepared-serving invocations passed every full-logit and exact-ID comparison across raw Hello, English chat and Korean chat, 32 generated tokens each. Cooperative-Q6 applications were 66/104/108 and RMS applications 21846/34424/35748, with zero fallbacks. These totals include candidate gate replay and generation including prefill; they are not per-token counts. See [qualification.json](evidence/q6-cooperative-forge/qualification.json).

## Fixed whole-batch comparison

The protocol was declared before timing: four serial A/B, B/A, A/B, B/A process pairs; each arm gets four generations with the first excluded and the median of three warm 31-step wall totals. The same original GGUF, raw Hello, 32 generated IDs and BOS-only alignment are used. Initialization/hash/prefill are excluded; synchronous graph execution and output readback are included. No owned author/build/profiling jobs overlap timing. Every run is retained, with no adaptive stopping or outlier removal. A is preserved phase33 and B is qualified phase34, each with its own approved artifact and layout.

| Pair | Order | A tokens/s | B tokens/s | Decode time reduction | Throughput increase |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | A/B | 8.271 | 9.096 | 9.069% | 9.973% |
| 2 | B/A | 8.174 | 9.146 | 10.629% | 11.893% |
| 3 | A/B | 8.249 | 9.260 | 10.923% | 12.262% |
| 4 | B/A | 8.010 | 8.972 | 10.728% | 12.017% |

All four pairs improved by more than 2%. Median paired decode time reduction is **10.678%**. Summing the equal-work warm arm totals gives **10.341% less decode time** and **11.534% more throughput**. These metrics are distinct. Source, both binaries, plans and both composites remained unchanged; all IDs were exact. Full repetitions and warmups are preserved in [fixed-results.json](evidence/q6-cooperative-forge/fixed-results.json).

Sixteen read-only before/after snapshots reported AC power and no recorded thermal/performance warnings, with zero throttled VM pages. Actual temperature/clocks were not measured. Other applications remained active and cumulative system swap counters increased, so an idle host or causal explanation is not claimed. [Resource snapshots](evidence/q6-cooperative-forge/resource-observations.json) preserve measured conditions. This is a reproducible improvement within this fixed shared-host series, not a guarantee across load conditions. The earlier phase33 noisy series, its regressing pair and later confirmations remain preserved; the prior fixed series is also retained in [rms-q6-forge/fixed-results.json](evidence/rms-q6-forge/fixed-results.json).

No new matched Ollama or MLX baseline was measured. Historical Ollama remains substantially faster and uses a different timing boundary; see [comparison scope](OLLAMA_COMPARISON_SCOPE.md). The existing 14.44 GB Gemma 4 26B-A4B model is a probe, not completion of lightweight/general Python-model support. No model/dependency downloads, conversion, new secrets or purchases were required.

## Remaining bottlenecks and next compiler pass

A separate warmed census measured normal batched GPU intervals at 61.37% and host flush work at 32.89%, with 97 command buffers and 1289 dispatches per decode. The normal sample was 117.62 ms/token and is not a replacement for the fixed comparison. Intrusive replay was bit-exact against its batched reference. Its ranking is cooperative RMS 28.74%, generic Q4 24.36%, F32 router 17.62% and cooperative Q6 4.53%. Intrusive rankings are not normal wall or GPU shares. See [post-budget.json](evidence/q6-cooperative-forge/post-budget.json).

The selected SSA has 331 RMS operations. In every one of 30 layers, three normalizations consume the same immutable input version and attributes, with different scale/apply outputs: `ffn_norm`, unscaled routing norm and `pre_ffw_norm_2`. This exposes **60 potentially redundant statistic reductions**. A bounded common-statistics lowering should compute the ordered RMS inverse once for matching input/width/epsilon and reuse it for the three separately scaled outputs. It must validate SSA identity and physical-buffer liveness, preserve each multiplication association, keep inverse values on the GPU, bind the new pass source/schedule to artifacts and retain strict full-logit gates. This is an implementation proposal supported by [SSA evidence](evidence/q6-cooperative-forge/rms-statistics-candidates.json), not an implemented pass or promised gain.

The F32 router's 128 output rows currently launch as one 128-thread group; this is a separate parallelism target. Existing tree-reduced F32 combinations previously failed a strict full-model gate, so a new candidate must preserve that rejection and numerical threshold. The Q6 result does not license weaker router gates.

Production qualification also repeats paired Q4 selection when the whole engine identity changes. Splitting artifact dependencies by validated pass could reuse unchanged approvals, but requires a correctness-preserving identity design; no gate was bypassed in this batch.
