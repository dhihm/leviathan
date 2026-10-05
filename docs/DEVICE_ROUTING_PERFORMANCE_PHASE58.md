# Prepared device routing: final executable qualification and fixed comparison

The prepared GPU routing implementation has a fixed performance comparison against the incumbent CPU routing path. The runner, v1 E4R32 expert math, Q4 launch width 32, model, fixtures and modern candidate are identical across arms. This is a new GPU selection and record-bound expert execution candidate, not a retry of the rejected phase-53 overlap candidate.

## Actual prerequisites

The final benchmark executable SHA-256 is `1fcd09971c174a2690271d28d30fe79855a280d86219f17296dfef76c82a8d7e`. Its own qualification-only invocation passed all three frozen fixtures, all 4,170 checked and exact routes, the full component replay and 16 pre-work refusal probes. All ON/OFF model outputs were bit-identical, finite and within the existing reference tolerance. The actual process exited zero after END and restored the pristine runner. The supervisor recorded 174.648008 seconds and released its lock.

The unchanged selector source is `6c34d907cefc0895b256d566bd7ad99529e61c4d5399b0ac635eb1ccf064d3fd`. The frozen selector and real expert component executable/report pairs are independently hash-bound. Earlier standalone gate reports cannot qualify this benchmark executable.

Release build passed. Related pure tests passed 62 benchmark, 42 runtime, 33 model-gate, 126 selector and 64 replay checks. Existing default regression passed 2,056 checks with model opt-ins disabled. Independent Fable review accepted the implementation with two exact corrections distinguishing an unfinished GPU window from a terminal but unrestored runner; both were applied before final qualification. Performance-only accepted the actual qualification receipt and refused EOF before creating a device or mapping the model.

The existing 14,439,363,584-byte model was reused. No model download, model-weight upload or weight rehash occurred. The explicit-refresh-only digest cache remains unchanged.

## Fixed performance protocol

The executable provides separate qualification-only and performance-only invocations. Performance-only requires the report of this exact executable, its hashed executable, the cached model digest, the same selector source and prerequisite receipt, and the live identity after GO. No rebuild follows qualification.

Each fixture has one OFF and one ON warmup, followed by seven alternating OFF/ON or ON/OFF pairs. The complete protocol has six warmups and 42 timed runs. Each run produces 32 frozen IDs with 31 previous-actual-argmax continuations and full-vocabulary finiteness checks. OFF continuation counters are pinned at 1,349 kernels and 37 command buffers; ON at 1,379 kernels and seven command buffers, with zero CPU selections, 30 GPU selections and 30 record chains per forward. Every fallback must be zero.

TPOT is the median of the 31 continuation wall times per run, then the median of seven run medians per arm. Each timed call includes encoding, GPU waits, logits readback and argmax; ON also includes the selector, record-bound expert work and post-token flag checks. Setup, compilation, scope entry/exit, counter snapshots and post-run checks are outside the declared timers. ON captures no numerical snapshots and runs no CPU route recomputation inside the timed path.

Every fixture must improve by at least 2%, and every arm must have spread `(max - min) / median` at most 5%. There are no retries, outlier removals, trimming, reseeding or count changes. External resource eligibility is reported separately by the supervisor. A relative pass alone does not install a candidate or establish an Ollama comparison.

## Measurement status

The fixed performance comparison is pending a fresh acknowledgement coordinating only known parent-owned builds and benchmarks. It does not claim global machine idleness. No timing result or performance gain is claimed yet. The incumbent main and production defaults remain unchanged.

## Timestamp instrumentation prerequisite

A separate temporary original two-encoder Metal probe actually sampled four nonzero, ordered encoder-stage timestamps on the M1 Max. Both outputs were bit-identical and all guards passed. Dispatch-boundary sampling is unsupported on this device; encoder-stage sampling is supported. Recorded intervals remain raw ticks and are not GPU busy time, utilization, bandwidth or recoverable time. This isolated test does not attribute any model kernel cost.

Raw receipts and reports are in [the evidence directory](evidence/device-routing-performance-phase58/). They contain diagnostic metadata, counts and timings, not model-weight originals or authentication information.
