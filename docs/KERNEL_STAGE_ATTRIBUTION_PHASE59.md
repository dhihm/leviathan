# Actual kernel stage attribution, with topology limits

This opt-in diagnostic sampled the incumbent OFF path on the actual M1 Max: modern candidate, v1 E4R32, Q4 launch width 32 and unchanged CPU router arithmetic. The diagnostic keeps the kernels, order, bindings, command buffers, commits and waits, but places each deferred dispatch in its own serial encoder inside the same command buffer. When inactive, the runner hooks only check an optional slot. No production default or installed candidate changes.

## Actual verification

The reviewed final executable `e67a2683c5a2a9cefda8e737320aa465ab1a0fc00ba178b5362aff996ea61bd4` exited zero after END, restored the pristine runner and released both owned locks. Actual GO was 2026-10-05 11:43:18 UTC and END was 11:46:53 UTC. The supervisor recorded 216.049067 seconds.

All three frozen fixtures passed. Each plain and sampled run produced the 32 frozen IDs from previous-actual-argmax continuations, with finite full-vocabulary outputs and the existing fresh-reference vector/per-element `1e-4` gate. All 96 sampled/plain outputs were Float-bit identical. Each sampled continuation retained 1,349 dispatches and 37 command buffers, and the kernel-name counts matched the runner counters.

Actual sampling covered 93 continuations, 125,457 encoders and 250,914 timestamp values. The 2,698-sample buffer was created on this device. Every sample range was resolved only after completed, error-free command buffers; two resolves were byte-identical. Counter-error, zero, end-before-start, overflow, missing-dispatch and stale-sample checks passed. Seven live refusal probes preserved execution state and performed no GPU dispatch, with the rejection attributed to the intended sampling guard. Unknown-tag rejection was tested in pure validation, not injected into a live model. Device-entry rejection while sampling is an additive reviewed guard; the live mutual-scope probe exercised the opposite entry direction.

The report is 1,261,302 bytes, below the fixed 16 MiB limit, with no trimming. No model download, original weight upload or weight rehash occurred. The cached explicit-refresh-only model identity remains in use.

## Observed stage values

The following are medians of family sums per sampled token, in raw resolved timestamp ticks. They are not milliseconds, GPU busy time, bandwidth, utilization or recoverable time.

| Family | English | Korean | Raw hello |
| --- | ---: | ---: | ---: |
| q4Matvec | 19,941,716 | 20,063,792 | 20,044,553 |
| rmsnorm | 18,038,931 | 18,065,325 | 18,147,338 |
| routerF32Matvec | 9,979,584 | 10,062,959 | 10,019,707 |
| moeGateUp | 6,846,789 | 6,906,291 | 6,908,587 |
| moeDown | 3,288,666 | 3,270,585 | 3,264,082 |

Generic Q4 has 205 dispatches per continuation. RMS has 331, including 241 remaining cooperative RMS calls. The router F32 matvec has 30. These are useful investigation rankings, not percentages of normal latency.

There were 3,221 overlapping adjacent encoder timestamp ranges. Disjoint-share reporting is omitted when a token has overlap. Family medians must not be added to obtain a total or converted into baseline GPU shares. Encoder boundaries and sampling can particularly distort short kernels. The sampled-minus-plain wall-median differences were 30.829 ms, 4.127 ms and 21.997 ms for the three fixtures; these are diagnostic overhead observations from different runs, not a performance qualification.

## Memory observations and limits

Own-process run-boundary TASK_EVENTS page-in deltas were plain/sampled: English 33/23, Korean 11/16, raw hello 11/9. These are actual nonzero own-process events. They do not identify the faulting objects, prove model-weight paging, exclude GPU-driver paging or explain the earlier host-wide swapping. Own TASK_VM and Metal allocation snapshots are retained in the raw report; allocation length and virtual size are not physical residency.

## Next bounded experiment

The largest observed stage values include generic Q4, while prior exact-order scalar router geometry and Q4 input-staging candidates have negative results. A distinct original packed-weight staging component is being authored: cooperatively copy Q4 weight bytes into bounded threadgroup memory while each row retains its serial arithmetic. It must prove compiled output-bit identity and the independent FP64 component contract before any component timing. This hypothesis does not establish that coalescing is the cause. There is no model integration or default promotion, and the Ollama goal remains unmet.

The initial release build passed, dedicated pure suites passed 614 checks and the default regression passed 2,056 checks. Independent review required exact terminal-token, refusal-attribution and additive device-entry corrections; the final build and affected pure suites passed after applying them. The final default regression also passed all 2,056 checks after refreshing the CLI. EOF was refused before device creation or model mapping.

[Raw reports and receipts](evidence/kernel-stage-attribution-phase59/) preserve the actual results. `gainQualified`, `installable`, `productionPerformanceClaim` and `busyTimeClaim` remain false.
