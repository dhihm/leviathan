# Packed Q4 weight staging component: correct, slower, rejected

This original component cooperatively copies packed Q4_0 WEIGHT bytes into threadgroup memory. Each lane retains its scalar per-row block/quant-byte arithmetic, and input activations remain device scalar loads. It differs from the closed phase-37 INPUT-staging experiment. The two fixed variants use 32 output rows and 22 or eight blocks per tile, with 12,672 or 4,736 bytes of static threadgroup storage. Padded word pitch is a hypothesis; no bank count or occupancy was measured.

## Actual result

The native component exited zero after END, with all committed command buffers terminal and both owned locks released. The supervisor recorded 0.754133 seconds. Forty pure checks and the cached-model EOF refusal passed. The final executable SHA-256 is `26b617d63d115664f42db5f5c0df7422626e52355651119125d158e194be2f2d` and is frozen locally with its actual report.

Both variants passed all 18 cases: nine supported synthetic cases, one predeclared unsupported odd-block case, and eight bounded real layer-0/5 2-D tensors. A total of 4,894 supported candidate output rows were Float-bit identical to the production scalar control; both arms passed the existing independent FP64 `1e-4` contract and all guards. Unsupported shapes were declined before candidate dispatch. This covers bounded tensor rows, not the full model or 3-D expert tensors.

The 148 actual command buffers comprise 18 shared control correctness runs, 34 candidate correctness runs, and 96 sequential timing-arm submissions. Each timing shape had two warmup pairs and six timed pairs, alternating arm order, with two command buffers per pair. The second arm was encoded only after the first was observed terminal. Timestamp ranges were resolved twice after both completed; identity, ordering, nonzero, staleness and guard checks passed. `timingPassed` means valid samples, not an improvement.

| Variant | Shape | Candidate/control ratio of medians | Median pair ratio: control first | Median pair ratio: candidate first |
| --- | --- | ---: | ---: | ---: |
| q4ws_matvec_R32_B22_P99 | timing-2816x1024 | 2.478 | 2.473 | 2.483 |
| q4ws_matvec_R32_B22_P99 | timing-2112x1024 | 2.855 | 2.253 | 2.565 |
| q4ws_matvec_R32_B22_P99 | timing-real-blk.0.attn_output.weight | 2.624 | 2.527 | 2.638 |
| q4ws_matvec_R32_B8_P37 | timing-2816x1024 | 2.694 | 2.432 | 2.679 |
| q4ws_matvec_R32_B8_P37 | timing-2112x1024 | 2.502 | 2.672 | 2.420 |
| q4ws_matvec_R32_B8_P37 | timing-real-blk.0.attn_output.weight | 2.617 | 2.545 | 2.692 |

Every observed ratio exceeds one in both execution orders. The component is performance-rejected, with no model integration, default promotion or unchanged rerun. These unitless raw stage-tick ratios are not GPU busy time or full-model latency. They do not prove the cause, bandwidth, cache efficiency or occupancy. Cache warmth and clock/order effects remain; three timed pairs per order establish no stability claim.

## Review and protocol provenance

Before the first GPU run, independent Opus review corrected the independent-encoder comparison to two sequential completed command buffers, actual commit counting, unsupported sampling classification and empty-output validation. The fixed synthetic scales were also bounded before any GPU observation, while retaining seeds, shapes, counts, FP64 thresholds, zero/subnormal edges and explicit large-scale edges. No result was used to retune or reseed. The original compiler access error was fixed through public Codable test construction rather than exposing an unchecked initializer. The conditional compiler patches were not applied because their conditions did not occur.

The earlier phase-59 raw ranges overlap 3,221 times and cannot partition production GPU time. Fixed per-encoder overhead dominance remains unmeasured. That ranking motivated a component hypothesis; the actual slower result closes this hypothesis as implemented.

No model download, original weight upload or model-weight rehash occurred. Bounded local structural tensor reads and the cached explicit-refresh-only model identity were reused. The report carries no original weight bytes or scales. `fullModelQualified`, `gainQualified`, `installable`, `productionPerformanceClaim` and `busyTimeClaim` remain false.

A distinct next component is being authored: aligned register word decoding of two Q4 blocks at a time, without threadgroup storage or barriers, using this frozen corpus and comparison protocol. It must satisfy the same compiled-bit and FP64 gates before timing; no speedup is assumed. The overall Ollama goal remains unmet.

[Raw evidence](evidence/q4-weight-staging-phase60/) preserves the full actual cases, samples and review amendments. After refreshing the debug CLI, the final default regression passed all 2,056 checks with actual exit 0, END and released supervision lock; its complete receipt and output are preserved in the evidence directory.
