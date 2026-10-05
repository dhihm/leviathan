# Prepared GPU device routing: numerical qualification, phase 57

The new default-off prepared path performs router selection and passes expert addresses and coefficients entirely on the GPU. On all three frozen Gemma 4 fixtures it is bit-identical to the incumbent at every output and every routing decision. Actual per-forward command buffers decreased from **37 to 7**, with **0 CPU selections and 30 GPU selections**. This is a numerical and execution-path qualification; performance is not qualified yet.

## Implementation

A validated router IR emits one arithmetic body shared by the component and runtime kernel entries. The runtime entry consumes the existing projected scores and local model scale bytes, writes an aligned record and logs scores for optional qualification snapshots. The real E4R32 GU/DN kernels consume its absolute offset arrays through constant-buffer bindings; the eight ordered merge dispatches consume the eight aligned Float32 coefficient slots. Their mathematical MSL, parameter geometry, scratch layouts and merge order are unchanged.

A separate prepared binding admits the declared modern candidate only inside a diagnostic scope. Entry checks device, install/source identity, metadata bounds, 7,680 expert offsets, local scale equivalence and aliases against weights, K/V, activations, scratch, statistics and readback buffers. Records are retained for the whole token. After the existing final-logits drain, the runtime checks that all 30 records were written and chained with zero flags before advancing the context. Failed tokens poison the scope; there is no partial CPU fallback or retry. Scope exit drains and restores its state.

The inactive path adds optional checks and a CPU selection counter increment; no new dispatch, clock, syscall or mathematical operation is added to inference. The feature remains default-off, with no artifact winner or main promotion.

## Actual prerequisites and component results

Native release and final debug CLI builds passed. Selector, device runtime, capture/replay and model-gate pure suites passed **126, 42, 64 and 33 checks**. The final default regression suite passed **2,056 checks**. All had zero failures.

The author delivered part 1 in two explicitly connected response fragments. The first mechanical extraction incorrectly included the continuation instruction in a source file and failed compilation before any GPU run. Exact reassembly of the author's two fenced payloads restored valid JSON and original source without semantic edits; the reassembly receipt, clean prose scan and both build logs are retained.

Independent reviews accepted the component and numerical gate. The gate review identified a real boundary hole: direct `Gemma4Runner.forward` bypassed the device scope's prepared-only contract. Exact reviewer-authored patches add a before-work refusal and the sixteenth predetermined live refusal probe. The final build and tests passed with that correction; conditional compile patches were not applied because the actual builds succeeded.

The newly emitted source invalidated earlier source-bound selector evidence. Its current SHA256 is `6c34d907cefc0895b256d566bd7ad99529e61c4d5399b0ac635eb1ccf064d3fd`. Fresh GPU qualification passed all **187 ordinary cases** exactly and all **16 declared refusals** correctly. The paired entry component additionally passed **102 identical records**, **94 ordinary CPU-exact cases**, **8 declared refusals**, **816 constant-address-space consumer slots**, padding, logged scores and canaries.

On the actual local weights, layers **0, 5 and 29** passed host-binding versus GPU-record-binding comparisons: GU, activation and DN scratch and the accumulator were bit-identical. IDs, quotient bits, coefficient bits and real offsets matched the local CPU decision; every record flag was zero.

## Actual full-model gate

The owned supervised gate completed in **177.464377 seconds**, exited zero, observed READY/GO/END, restored the pristine runner and released its lock. Its executable SHA256 was `b9d64d033c688a751d34f797bd5e420edc6beceb94898ab8e7f95483fa5b7415`. Before GO, prerequisite reports were bound to their actual frozen executable hashes, the current emitted source/IR and the cached model digest. EOF was separately refused before device creation or mapping.

All **16 live refusal probes** passed without GPU work or changed counters/position, including the new direct-forward probe. Each fixture ran a fresh direct reference, the incumbent OFF arm and device-routing ON arm. Every arm produced 32 frozen output IDs with 31 continuation steps fed by the previous actual argmax; all 262,144 logits were finite at every output. Every ON logit Float bit equaled OFF.

| Fixture | Real routing frames | Output bit mismatches | Maximum per-element normalized reference error |
| --- | ---: | ---: | ---: |
| en-chat | 1,560 | 0 | 3.740191459655762e-5 |
| kr-chat | 1,620 | 0 | 3.272294998168945e-5 |
| raw-hello | 990 | 0 | 7.529184222221375e-5 |

All **4,170 frames** matched an in-process CPU FP64 recomputation on expert IDs, Float32 quotient/coefficient bits and actual expert offsets. Five fixed component-entry replay chunks covered those same frames; all completed, every runtime/component record matched and the mismatch count was zero. Nothing was omitted, reseeded or accepted as a partial pass.

Every ON continuation used exactly **1,379 kernels and 7 command buffers**, versus the incumbent's **1,349 and 37**. The extra 30 kernels are GPU selections. CPU selections, failed/flagged tokens and all fallbacks were zero; GPU selections and record chains were 30 per accepted forward, including prefill. These are observed counts, not projected performance gains.

The final status was `qualificationPassed`, and `realE4GateComplete` was recomputed true from numerical coverage and valid in-process prerequisite receipts. `gainQualified`, `installable` and `productionPerformanceClaim` remain false. The rounding-certificate budget is estimated; direct gates establish agreement only on these tested inputs, not universal FP64 equivalence.

## Profiling facts and remaining validation

A standalone own-process SDK query created no command buffer or counter sample buffer. This M1 Max reports **32,768 bytes** of threadgroup memory, stage-boundary timestamp sampling supported, dispatch/draw/blit/tile-boundary sampling unsupported, and only the timestamp counter set. No timestamp samples or GPU utilization measurements were taken. Encoder-level profiling is required for later kernel attribution; command-buffer intervals and host wait residuals are not busy time or gain bounds.

Original scale and coefficient bits remain local and are neither serialized nor uploaded. Model weights were not downloaded or rehashed; the existing explicit-refresh cached SHA record was reused. Guards for aliases and wrong Q4 width are enforced in code but were not mutated in live model probes; the overlap-entry probe cannot by itself attribute which nested guard refused. Actual GPU failure recovery remains unprovoked.

The next step is a fresh fixed seven-block AB/BA comparison of this path and the same incumbent, after the final performance executable is numerically qualified. All fixtures must achieve at least 2% gain and both arms must stay within 5% spread before a performance qualification. There is no current qualified Ollama ratio, and the previous failed overlap comparison remains rejected.

Full evidence is preserved in [the phase 57 evidence directory](evidence/device-routing-numerical-phase57/).
