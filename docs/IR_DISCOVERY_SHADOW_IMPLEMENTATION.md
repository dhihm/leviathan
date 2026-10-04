# Cache-append IR opportunity discovery: shadow implementation

`GemmaOpportunityDiscovery.discover` now derives a diagnostic verdict for the existing GPU cache-append copy from typed SSA. It proves the static binding and geometry contract and complete per-layer coverage. `shadowCandidates` re-runs analysis and compares admission with the existing manual pool; it explicitly preserves index zero and manual order.

This is a pure library analysis, not an active forge generator. The forge still uses the same manual 32 tuples. No existing runner, executor, kernel, candidate enumeration, plan schema or selection policy was changed. All existing semantic runtime source files are byte-identical to private main `23b3eed`. The new code performs no Metal calls, file access or runner mutation. It makes no performance claim.

## Design-to-code and proof boundary

| Requirement | Code and evidence |
| --- | --- |
| Live operands and valid dataflow | Existing pure `GemmaSSA.validate` is called before discovery; malformed schedules throw. |
| Exact K/V binding | `t2` then `kv`, vector/f32, zero or absent source slice offsets. |
| Layer geometry and copy word ABI | Local/global heads and dimension must match config; checked width is positive and fits UInt32; both input lengths and matching kvstate output agree. |
| Complete unique coverage | Count every append per layer; report duplicate and missing coverage explicitly. Each valid current-model layer is proven once. |
| Deterministic provenance | Ordered proofs and typed reasons; body includes analysis version, config, measured profile and SSA digest. SHA-256 covers canonical sorted-key body JSON excluding its own digest. |
| Diagnostic reports are not trusted proofs | `shadowCandidates` takes SSA/config/profile and re-analyzes them; it accepts no decoded report. |
| Baseline preserved | Index zero is pinned structurally regardless of flags. Current catalog tests also verify baseline preservation on every valid unsupported fixture; the current baseline has the flag disabled, so these tests alone do not simulate a flag-enabled baseline. |

Discovery-owned sanity checks are explicit additions: positive hidden/vocab/query/KV/expert/intermediate dimensions, valid expert-use count, nonnegative sliding window, finite positive bounded softcap, unique legal global layers and checked geometry. The unchecked `8 * hidden` in the existing validator is guarded first. Layer count is limited to **4096** solely to bound diagnostic coverage allocation; it is not a model fact or hardware limit. Profile fields must be positive, the device name nonempty and execution width no larger than max threads; there is no fixed hardware cap.

The analysis does not certify actual buffer allocation, byte spans, token positions, command queues or compiled pipeline legality. Those remain fail-closed runtime guards. Known unrelated operations are outside this axis; unknown malformed IR is rejected by the existing validator. No compilation or resource error is caught as an unsupported static opportunity.

## Concrete verification

Opus 5.5 authored source and tests with tools disabled. Fable 5.1 reviewed the full patch and the follow-ups, finally accepting with no blockers. Exact ordered replacement replay matches both final source files; supervising-agent semantic source edits are zero. Independent temporary census/measurement drivers and English documentation are separately identified in evidence.

Release builds succeeded in 111.48 and 69.65 seconds. Final existing real-model/Metal regression plus new pure gates: **1,808 passed, 0 failed**, including 139 new checks. Valid-but-unsupported fixtures first pass `GemmaSSA.validate`; a rejection there increments failure rather than silently skipping acceptance.

Tests cover canonical support, ordered whole-tuple equivalence, repeated/Codable determinism, tampered metadata/proofs, swapped inputs, duplicate/missing coverage, local geometry on a global layer, absent/zero/nonzero slices, nil layer and exact multi-reason ordering, stale references, unknown operators, config overflow and invalid profiles. Recorded canonical SSA/report digest constants pin this fixture and current codec; they do not prove cross-Foundation digest portability. Some defensive reason branches are unreachable after SSA validation, and this is not a proof for every possible arbitrary graph configuration.

An independent read-only driver validated the existing GGUF's **658 tensors**, validated its graph and exported **934 SSA instructions**. New discovery over that actual IR produced **30 proofs, zero reasons**, covering layers 0–29 exactly once, and the same ordered 32 manual tuples. Local widths: 25 at 2048 words; global widths: 5 at 1024 words. The measured hardware profile was captured in the immediately preceding validated main lifecycle on this M1 Max; this probe did not perform new hardware timing.

Actual SSA digest: `28a04586de4fe9b62d56c56723e55ba9f8668548d3cb29db1a565a42097f8583`. Actual diagnostic report digest: `d957577fe026b8d57898dd64935115482262c1be0728b333e1927a84ba546e06`. The final follow-up build produced the identical diagnostic report.

## Artifact and execution limits

Adding library source changes the executable digest even though no analysis call was added to the runtime. The new CLI build correctly rejected the prior executable-bound artifact with identity mismatch, without generation or artifact writes; both prior artifact files remain unchanged. Ordinary automatic re-forging is required for a newly built executable. No identity field was overridden or rewritten. A new full production selection/positive plan reload was not run for this pure shadow slice; the prior validated execution remains preserved as a frozen baseline.

The forge has not begun consuming these reports. Later activation needs separately reviewed identity/schema versioning, candidate-set binding and ordinary full correctness/lifecycle gates. Candidate-list equivalence is consistency with the existing policy, not an independent numerical oracle or a full replacement of the manual list.

## Next smallest discovery and performance steps

The next discovery implementation should compose two shadow verdicts: cache append and RMS statistics reuse. Call `RMSStatisticsPlan.analyze` unchanged rather than reimplementing its grouping, liveness or alias analysis. Its epsilon is an explicit input: GraphConfig does not contain it, so obtain it from validated model configuration, validate it and record its exact Float bit pattern instead of silently hardcoding it. Version the diagnostic shape and preserve deterministic per-axis proofs/reasons, baseline pinning and ordered whole-tuple equivalence. Add a valid SSA with no reusable RMS groups as the new unsupported fixture. Keep runtime generation and executor unchanged.

For performance, prepare a new **baseline A/A control**, not another rejected tuning variant. The predeclared protocol and runnable orchestration are included: seven visits of the same frozen validated executable and artifacts, four generations per visit, first excluded and median of three warm 31-step decode totals; retain all visits and exact IDs/fallback checks. Relative spread is `(max visit median - min visit median) / overall median`. A spread over 5% requires fixing baseline reproducibility before judging future sub-5% changes. Prior rejected experiments remain rejected.

This measurement has **not started**. The runner requires a fresh parent quiet-window receipt for this exact scope and no owned parallel GPU work. It does not enumerate processes or stop user applications. Frozen baseline executable/artifact hashes are checked before and after the run.

A separate segment-gap trace needs externally authored instrumentation, because current flush-minus-GPU counters do not measure CPU busy time or completion-to-next-commit gap. Use a single host monotonic clock for commit and wait-return events; label wait-return-to-next-commit as a host proxy unless GPU/host clock alignment and delay are established. Do not mix trace overhead into A/A timing. Before any new batching implementation, reject that hypothesis if the confirmed per-token commit count is at most two or the properly measured gap is below 2 ms. No RMS-width, SSA-copy or Q4-staging failure is reopened by this protocol.

The performance target remains unmet. No current matched Ollama ratio is claimed. The existing 14,439,363,584-byte model was reused, with no download or new service setup.
