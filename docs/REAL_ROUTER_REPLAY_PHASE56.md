# Actual model routing capture and GPU replay: phase 56

All **4,170 real routing frames** from the three frozen Gemma 4 fixtures matched the GPU selector exactly. This qualifies the selector on those frames; it does not remove a CPU synchronization boundary or establish a runtime speedup.

## Actual native results

The owned supervised run completed in **177.028729 seconds**, exited zero, observed READY/GO/END, restored the pristine runner and released its supervisor lock. It reused the cached explicit-refresh model digest without rehashing or downloading weights. The executable SHA256 was `0dc1fb4f0194978b4ce8e25ff336caa1b3a808c892b65fe20a8c2ea657bb9271`.

| Fixture | Captured frames | Captured versus plain output bit mismatches | Maximum per-element normalized reference error |
| --- | ---: | ---: | ---: |
| en-chat | 1,560 | 0 | 3.740191459655762e-5 |
| kr-chat | 1,620 | 0 | 3.272294998168945e-5 |
| raw-hello | 990 | 0 | 7.529184222221375e-5 |

For each fixture the tool ran a fresh direct reference, the plain incumbent prepared SSA path and the same path with capture enabled. Each produced all 32 frozen output IDs with 31 continuation steps fed by the previous actual argmax. All 262,144 logits were finite at every output. Captured output Float bits equaled the plain output bits throughout; maximum per-element normalized error against the fresh reference stayed below the unchanged `1e-4` contract. Captured and plain kernel and command-buffer totals agreed, derived axis counters matched, and all fallback counts were zero.

The complete frame sequence is `(21+31 + 23+31 + 2+31) * 30 = 4170`. No frame was sampled, omitted, reseeded or reclassified. Five fixed chunks of `[1024, 1024, 1024, 1024, 74]` each completed two GPU passes: a selector using the real model's GU/DN offset metadata and the selector plus synthetic payload consumer. All chunk guards were intact. Every frame had zero flags and exact expert IDs, Float32 quotient bits and final coefficient bits against the **actual captured incumbent CPU decision**, with the host mirror excluded as the oracle. Real offset equality and the metadata extent proof covered 7,680 offsets. Refusal, numerical mismatch, offset mismatch, record error, consumer error, host-offset mismatch and CPU-reference mismatch counts were all zero.

The emitted selector source remained `bf04e4347caaad9a902fa9d77f16c8e3a7ceac4b1af025f368608a3e9d73aa3a`, the source qualified by the phase 55 synthetic run. This is direct evidence for these fixtures, not a universal numerical equivalence theorem.

## Scope and verification

The new capture hook is inactive by default. Active capture records the unchanged CPU routing calculation after the router readback and captures checked installed binding offsets. A deferred scope cleanup clears it on both success and failure. It refuses nesting, overlap scopes, instrumentation and probes; the executor additionally refuses unprepared and overlap-prepared execution. The tool constructs only the declared modern control preparation. The general direct `forward` API does not capture frames, and the public capture scope alone does not exclude every possible non-modern caller. Those are narrow API limitations, not broader guarantees.

Native release and debug prerequisite builds passed. The new CPU-only suite passed **64 checks**. The no-GO invocation printed READY and refused EOF with exit 1 before mapping the model or creating a Metal device. The existing default regression suite passed **2,056 checks**, and existing selector, diagnostics, model-gate, runtime and planner pure suites passed **126, 66, 177, 170 and 333 checks**, all with zero failures. Independent Fable review accepted the source, conditional on successful native build and those actual prerequisites; they were fulfilled before this model run.

Original expert scales stayed in local process/GPU memory. They were neither hashed nor serialized. Coefficient bits were compared in memory and only counted in the report; report frame records and their digest omit coefficient bits to avoid exposing the scale values alongside quotient bits. External author/review requests contained original Leviathan source and derived validation facts, never credentials or model weight bytes.

## Remaining prerequisite

The real-offset selector writes actual offsets but does not dereference those weight addresses. The consumer dereferences synthetic tagged payloads. Existing E4R32 GU/ACT/DN and merge kernels still receive the incumbent CPU bindings. Therefore `realE4GateComplete`, `gainQualified`, `installable` and `productionPerformanceClaim` remain false. Reported GPU microseconds are command-buffer intervals, not busy time or recoverable latency.

The next implementation binds GPU-written GU/DN offsets and aligned coefficients to the existing real expert and merge kernels, behind a separate prepared diagnostic scope. Small constant-address-space consumer tests and a real E4 component comparison must precede its full-model gate. Only a passing full gate can precede a new fixed matched performance comparison. The failed phase 53 scheduling experiment stays rejected; no default, artifact winner or main promotion has occurred.

Full raw evidence is preserved in [the phase 56 evidence directory](evidence/real-router-replay-phase56/).
