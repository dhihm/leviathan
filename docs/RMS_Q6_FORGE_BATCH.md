# Cooperative RMS and output/router composition batch

After the published E4/Q4 composition, a diagnostic census after a full generation of warmup found a different kernel ranking: RMS normalization accounted for 41.77% of intrusive kernel intervals, the tied Q6_K output projection 16.02%, generic Q4 matvec 15.12%, and F32 router matvec 10.41%. The normal batched sample measured 79.53% GPU intervals and 15.87% host flush work. The diagnostic replay compared all 262144 logits at each of 31 decode steps with zero normalized difference. Its per-kernel submission intervals are investigation rankings, not normal batched GPU or decode-wall shares.

This batch changes the largest measured kernel family and makes existing output/router tiling available in composed candidates. It preserves the existing 17 candidates and adds seven distinct nonempty combinations of cooperative RMS, tiled Q6, and tiled F32 on the batching/vectorized-load/GPU-KV/parallel-attention baseline, for 24 bounded candidates. It adds no Q6/F32 kernel implementation and keeps the existing strict gates and fallback.

## Kernel and runtime

The original RMS kernel assigns one GPU thread per vector, with serial input reads, serial sum and serial output writes. The new original `RMSCooperative` kernel stages float4 input cooperatively into dynamic threadgroup memory, uses lane zero to sum float4 components in the incumbent's exact order, then writes each output element in parallel with the same multiplication association. Two barriers separate staging, serial reduction and writes; in-place inputs are consumed completely before output writes. It preserves the reduction order instead of using the previously rejected reassociated tiled-RMS reduction.

Thread count derives from the actual pipeline execution width and maximum, bounded by vector width. Dynamic plus static threadgroup memory is checked against the device cap. Positive multiple-of-four widths up to 8192 and scaled-weight alignment use the production guards; ineligible geometry counts a fallback and retains the incumbent. Existing generated-RMS installation retains precedence. Cooperative and reassociated tiled RMS cannot be enabled together.

The independent lazy pipeline is outside the shared pipeline dictionary. Engine identity binds source SHA, schedule derivation version and the new explicit `rmsnormCooperative` candidate field; pre-axis artifacts are rejected. Per-call axis state resets on completion/error, and prepared execution does not hash source per token. Actual application and fallback counters are logged.

## Development checks

The implementation and repository tests were authored through the user-approved configured GLM-5.3/ZCode. Exact applied source provenance was replayed against the published baseline. Patch-format conflicts, Swift argument order and test naming/type issues were corrected before any full model qualification. Release compilation passed; 892 checks passed with zero failures, including hardware bit equality across widths 256/704/2816, weighted/unweighted, in-place/out-of-place and multiple vectors.

A separate four-step actual-model check used the existing E4/width 32 diagnostic installation and current prepared GPU-KV/parallel-attention execution. RMS-only output was bit-exact over all 262144 logits with 1655 actual applications and zero fallbacks. All seven short composition checks retained exact IDs and normalized error below 1e-4. These development checks do not approve a production artifact or demonstrate a performance gain.

## Qualification and limits

Fresh production selection and full Forge qualification passed in 863.75 seconds, followed by a separate fresh-process artifact reuse/full gate in 100.98 seconds. The Q4 composite retained width 32 and passed its existing three-fixture production gates. Its artifact digest is `e813d9f70714f1a0e0bbd4bc3c49268b4106daeccb3d18507e74d922a03e8c15`.

The initially fastest cooperative-RMS plus tiled-F32 candidate failed the full 32-token gate at step 13 with normalized difference 1.091e-4. It was rejected under the unchanged threshold. Forge automatically reselected cooperative RMS with batching, vectorized loads, GPU KV and parallel attention. Tiled Q6 and tiled F32 are both disabled in the final plan; no speedup from those paths is credited. The full-gated plan digest is `1c6ffccf39887a6f6582e88a9176ef3fcc6760f2c29aa7c84a7830e096c985ec`.

Three separate actual CLI prepared-serving invocations passed every full-logit and exact-ID comparison for 32 generated tokens: rawHello (2 prompt tokens), English chat (21), and Korean chat (23). These use the public verified composite installer and the persisted plan. Their RMS applications were 21846, 34424 and 35748 respectively, with zero fallbacks. Counts cover graph gate replay plus one generation; the serial reference replay does not use the new cooperative axis. The same-source/binary checks passed. See [qualification.json](evidence/rms-q6-forge/qualification.json).

The original matched series used three serial AB/BA/AB process pairs against the preserved published phase 32 executable and composite/plan. The GGUF, rawHello 32, BOS-only alignment, 31 decode steps and first-generation warmup exclusion were identical. Each arm used the median of two subsequent warm generation wall totals; no profiling, builds or owned author jobs ran during timing.

| Pair | Incumbent warm 31-step wall | Cooperative RMS wall | Wall reduction |
| --- | ---: | ---: | ---: |
| 1 | 4,590.30 ms | 3,574.75 ms | 22.124% |
| 2 | 4,551.35 ms | 3,549.40 ms | 22.014% |
| 3 | 5,563.45 ms | 7,362.00 ms | -32.328% |

The median paired wall reduction was 22.014%, but the third pair regressed 32.328%. It is retained in [paired-results.json](evidence/rms-q6-forge/paired-results.json), not excluded or replaced. The slow candidate warm generation took 9997.9ms with 4734.4ms reported GPU time; the next generation recovered to 4726.1ms with 2984.2ms GPU time. This shows substantial non-GPU waiting variation as well as GPU variation; the record does not establish its cause or prove that it was external interference.

Because of that observed instability, one additional reverse-order BA confirmation pair used four generations, excluding the first and taking the median of three warm wall totals. The incumbent measured 4528.7ms/6.845tokens/s and cooperative RMS 3549.2ms/8.734tokens/s, a 21.629% wall reduction. This is separately preserved in [stability-results.json](evidence/rms-q6-forge/stability-results.json). It corroborates the first two gains without erasing the original regression, and does not establish uniform improvement across load conditions.

All compared IDs were exact. Source, binaries, plans and both composite artifacts remained unchanged. No individual feature gains are summed; tiled Q6/F32 are disabled in the chosen serving plan. The shared Mac was not claimed idle, and the comparison is between Leviathan versions rather than a new matched Ollama or MLX benchmark. Full qualification/reload times include deliberate reference replays and are not request latency. No throughput is predicted from the intrusive kernel percentages. If only GPU time changed and the other measured categories remained fixed, the first warmed batched sample's hypothetical maximum speedup would be 4.89 times; this is a conditional sample bound, not an architecture ceiling or an Ollama-comparable TPS prediction.

The model remains the existing 14.44 GB Gemma 4 26B-A4B Q4_0 GGUF on M1 Max with 32 GiB. No model/dependency downloads, conversion or paid setup were performed. The model is a probe rather than completion of the lightweight-model goal. Broader Python LLM architecture support and competitive performance remain unfinished. [Ollama comparison scope](OLLAMA_COMPARISON_SCOPE.md) documents the known model, cache, precision and timing conditions without claiming a new matched Ollama benchmark.

## Residual after this batch

A separate census after full-generation warmup measured 115.19ms/token, with 76.04% GPU intervals, 18.91% host flush work, 0.84% encoding, 0.30% read/copy, 0.11% write/copy and 3.80% residual work. The intrusive replay remained full-logit bit-exact against its batched reference. Its largest intervals were cooperative RMS 24.22% (331 calls/token), generic Q4 20.48% (205), Q6 output 20.03% (one), and F32 router 14.88% (30). These are intrusive rankings, not normal GPU shares or feature attribution percentages.

The existing Q6 tiled shader distributes complete 256-element quant blocks across lanes. The actual 2816-wide output row has 11 blocks, leaving many lanes unused at the current tile width. A finer within-block Q6 schedule and generic-Q4 memory/lane layout are concrete follow-up targets, with the same numerical gates. RMS remains substantial because its exact-order reduction is still serial. Host flush work remains a separate limit requiring measurement before changing routing/readback. No additional implementation is claimed in this report. See [post-budget.json](evidence/rms-q6-forge/post-budget.json).
