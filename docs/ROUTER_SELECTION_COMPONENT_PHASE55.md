# GPU router selection component: phase 55

This isolated, default-off component derives a bounded GPU selector from a validated router IR. It is not connected to model inference and has no installed winner or performance claim. The purpose is to qualify removal of the incumbent's 30 CPU router readback boundaries without changing expert IDs or final Float32 coefficient bits.

## Actual native results

On Apple M1 Max, the release build passed. The independent Fable source review accepted the complete component without required implementation fixes. CPU tests initially reported 124 passes and two failures: an assertion incorrectly compared the sum of field bytes with their aligned span, and another counted the GPU-only offset-domain refusal among host-constructible cases. Exact externally authored corrections changed only those predicates. The observed corrected result is **126 passed, 0 failed**. All **651 predetermined ordinary CPU cases** had zero flags and exact IDs, quotient bits and final coefficient bits against an independent copy of the incumbent CPU FP64 calculation. Fifteen host-constructible refusal cases remain distinct from equivalence passes. No seed, numerical tolerance or case set was changed.

The actual synthetic GPU run exited zero in 0.487837 seconds with READY, GO and END observed, an owned terminal receipt, and released supervisor lock. All **187 ordinary cases** matched the CPU reference bit for bit with zero GPU refusal flags; **16 declared refusal cases** were refused as declared. All 203 records passed padding, magic, layer, coefficient-slot, GU/DN offset and guarded producer/consumer checks. There were zero consumer errors, unexpected flags, preflight errors and unflagged mismatches. The offset table validates 7,680 entries. The component executable SHA256 was `8acf00eafc197fe586c8ecdb5331af651fcb0308fd3283b16dba2d74f8b4cafc`; emitted and compiled source SHA256 was `bf04e4347caaad9a902fa9d77f16c8e3a7ceac4b1af025f368608a3e9d73aa3a`.

The no-GO invocation printed READY, refused EOF and exited 3 without creating a Metal device, opening a model or writing the report. The GPU command used the existing recoverable supervisor with a 60-second deadline and 15-second graceful cleanup allowance; the component timeout was 10 seconds. No unrelated processes were inspected or stopped.

The existing default regression suite also passed **2,056 tests, 0 failures**, after the debug CLI prerequisite build. Real-model opt-ins were disabled; this is a default-path regression result, not model qualification. The supervised run observed END and actual exit zero, and its lock was released.

The existing pure diagnostic, model-gate, runtime and planner suites passed 66, 177, 170 and 333 checks respectively, with zero failures after the new component was linked.

## Representation and numerical policy

The IR validates 128 experts, top 8, 128 threads per group, SIMD width 32 and bounded exponential input. The layout derives a 2,816-byte record; declared fields total 236 bytes but span 2,564 bytes because Metal coefficient bindings require 256-byte offsets. GU and DN offsets and eight individual coefficient slots are checked rather than assuming a 256-byte record.

GPU rank selection is lane-parallel. Selected exponentials and division use double-float arithmetic and an estimated rounding certificate. Conservative domain, tiny-input, near-tie, rounding-boundary, scale and offset flags refuse unsupported cases. The `2^-42` certificate budget is **not a universal FP64 equivalence theorem**. Direct CPU-versus-GPU bit comparisons determine each tested equivalence. The sampled CPU quotient relative error was approximately `1.26e-14`; this only describes the fixed sample.

A concurrent compute encoder inserts a record-buffer memory barrier between selection and its small consumer. The consumer dereferences synthetic GU/DN payload tags and coefficient slots. Canary checks and these reads qualify this component's plumbing; they do not qualify the real E4 model kernels.

## Limits and next gate

No original model weights were opened, hashed, cached, downloaded or included in the synthetic report. No inference kernels, default runtime math, existing correctness gates or artifacts changed. The GPU command-buffer interval in the report is not GPU busy time or a production TPOT measurement. `gainQualified`, `installable`, `productionPerformanceClaim`, `realE4GateComplete` and `refusalsProveCoverage` are false.

Real router activation flag rates remain unknown. In particular, checking the full 128-score exponential range may conservatively refuse a frame even if all selected experts are within range. The next prerequisite captures the unchanged incumbent's complete routing decisions locally for all three frozen fixtures and replays every frame against this selector. Original expert scales must remain local and ephemeral. Real E4 consumption, full-model numerical equivalence and a new fixed matched performance comparison remain necessary before any promotion. Phase 53's rejected overlap candidate remains rejected.

Evidence is preserved in [the phase 55 evidence directory](evidence/router-selection-component-phase55/). The initial failing test log is retained alongside the exact correction and passing native log.
