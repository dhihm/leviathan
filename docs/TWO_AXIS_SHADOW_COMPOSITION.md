# Two-axis opportunity composition and baseline reproducibility

This experiment adds a pure `GemmaShadowComposition` wrapper over the existing cache-append discovery and RMS statistics analysis. It carries both complete proofs, ordered verdicts, analysis versions and the caller's exact positive finite epsilon bits in a deterministic digest. Candidate comparison always re-analyzes the inputs, pins manual index zero and preserves complete tuples and order.

The existing analyzers, runner, executor, kernels, candidate catalog and plan lifecycle are unchanged. This is passive discovery; it does not generate or activate a new optimized runtime pass. Static RMS eligibility does not certify buffer readiness, kernel legality or a runtime reuse hit. Other optimization axes remain manual.

## Verification and review status

Opus 5.5 authored the new source and tests. Exact author receipt hashes were verified. All pre-existing source files except the test file are byte-identical to main `7af75f6`; 419 test lines were inserted, with zero removals. Release build completed in 113.33 seconds. The real-model/Metal regression run completed with **1,963 passed, zero failed** (155 more than the prior main run).

All four valid-SSA support quadrants are checked: both axes supported retains 32 tuples; cache supported/RMS unsupported retains 26; cache unsupported/RMS supported retains 14; neither supported retains 14. Tests also compare complete underlying analysis reports, fixed group/reuse counts, epsilon and version fields, candidate ordering, deterministic coding, body tampering and typed failures. The whole-tuple expected-list helper repeats the filtering predicate; its comment calling it an independent oracle is imprecise. These are policy consistency tests, not an independent numerical oracle.

Fable 5.1 accepted the design. Its first final implementation-review response was fabricated tool markup rather than a review; tools were disabled and those strings were not executed. The response is preserved as invalid. Automatic approval review initially rejected the corrected read-only invocation because explicit source-payload-to-destination authorization was not established. After the user explicitly authorized the exact 71,331-byte packet and existing firstParty Fable 5.1 route, the same tools-disabled invocation completed and returned **ACCEPT, no blocking findings**. Exact model receipt and review are retained. The reviewer identified the predicate-mirror oracle wording and unexercised present-but-ineligible RMS fixtures as nonblocking limitations. The review did not inspect the synthetic-fixture helpers, forge catalog or SSA validator; their validity and pinned counts depend on the completed test run and local source-scope checks. No code or tests changed after that test run.

An independent read-only driver validated the existing GGUF tensor contract and graph, lowered 934 instructions, and composed the two reports with `Gemma4Config.rmsNormEps`. That epsilon is the validated model-family configuration constant, not metadata inferred from the graph and not a tunable default in this API. Actual result: **30 cache proofs, 30 RMS groups, 60 reuses, 934 dense actions**, both axes supported, and the same ordered 32 manual tuples. Its profile was captured in the prior validated M1 Max lifecycle; this static probe does not constitute new hardware timing.

Actual composition digest: `8d6b0e9688e38b6d300014894df88fe2421d2a7f54d56d0d9f4c930df7dfb93a`. Canonical sorted-key Foundation JSON hashes are bookkeeping, not authentication or a guarantee across Foundation versions. Decoded reports are never admitted as candidate-generation proofs.

## Seven-visit A/A control

After owned compilation and tests finished, the parent confirmed its Dokkabi developer and utility idle and no parent-owned Mac builds or benchmarks scheduled. No claim was made that user apps were idle. The frozen already validated executable `b77e692ff46dd80116500335c86000273e1744cd35ebb31a5a5aed48a1ef0f5a` and its original executable-bound plan/composite were used without rewriting identities. New shadow-library executables require ordinary re-forging before serving; this control deliberately uses the frozen artifact-compatible binary.

The predeclared protocol used seven visits, four raw-Hello/BOS-aligned generations per visit, 32 generated IDs, and a median of the last three 31-step decode totals per visit. Every visit and repetition is retained. There was no candidate variant, kernel profiler, process inspection, app termination, adaptive exclusion or rerun. Existing strict full-vocabulary three-fixture gates belong to the prior frozen baseline validation; this control checks IDs and fallback counters and does not claim to have re-run those logit gates.

| Visit | Warm median ms/token |
| --- | ---: |
| 1 | 83.9968 |
| 2 | 92.3516 |
| 3 | 82.9452 |
| 4 | 97.9194 |
| 5 | 71.4032 |
| 6 | 108.7710 |
| 7 | 92.4387 |

Overall median: **92.3516 ms/token**. Relative spread `(max-min)/median`: **40.4625%**, exceeding the fixed 5% threshold. All 28 generations matched the frozen validated baseline IDs; all reported fallback counters were zero. Executable and both artifact files remained unchanged.

Fourteen outside-decode observations recorded `pmset -g therm`, `pmset -g batt`, `vm_stat` and aggregate `uptime`. They reported no recorded thermal or performance warnings and AC power with battery at least 20%. Across the control, pageouts increased by **884** 16 KiB pages, swapins by **42,871** and swapouts by **58,296**. Paging activity and timing variability are observed together; these observations do not isolate causation, identify another process or prove GPU scheduling interference.

**Decision: skip the conditional RMS-reuse ON/OFF comparison.** Baseline reproducibility fails the predeclared gate, and pageout/swapout growth also fails its prerequisites. Do not remove visits, relax the threshold, repeat failed candidates or claim an improvement. RMS ON was already present in main, so a later successful ON/OFF attribution would still not be a new improvement over main. No current matched Ollama ratio has been established; the performance goal remains unmet.

## Activation decision and next bounded work

The whole manual pool is not ready for replacement: only two static axes are proven. Future activation needs immutable candidate-set binding, exact epsilon and analysis identities, schema/lifecycle versioning and ordinary full correctness/serving gates. Runtime buffer/pipeline checks remain mandatory.

The next measurement task is baseline reproducibility with an explicit memory-resource precondition and proper host/commit tracing in a separately instrumented run. The observed swap growth warrants resource investigation; it does not justify stopping user applications or attributing the variance without evidence. Existing flush-minus-GPU timing is not CPU busy time or an exact GPU-completion-to-next-commit gap. Any new product hook or pass must again be externally authored and reviewed, and must preserve the failed controls and existing correctness gates.

Evidence is under `docs/evidence/two-axis-shadow-phase44/`. No model downloads were performed. Main promotion contains passive analysis, tests and evidence only. No runtime optimization was activated.
