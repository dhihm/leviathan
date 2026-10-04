**Corrected decision: a diagnostics foundation plus pure-validation rejection only.** Pipeline compile, resource, geometry and function-missing failures all stay global and fail-closed. The first proposal's `prepareCandidatePipelines` isolation is withdrawn: the accessors discard the `NSError` from `makeComputePipelineState` via `try?`, so OOM, allocation and source failures are indistinguishable today, and `badTensorGeometry` and `functionMissing` are global corruption signals by requirement. Classification cannot be reviewed until provenance exists. Finding (B) remains deferred; softcap 30.0 is not an accuracy failure.

## Scope 1: typed compile provenance (global, no behavior change)

**Sources/LeviathanCore/Gemma4.swift**

1. Add `public struct Gemma4CompileDiagnostic: Equatable, Codable { stage: Stage; kernel: String; domain: String?; code: Int?; message: String }` with `enum Stage: String, Codable { case library, function, pipeline }`, and `static func make(stage:kernel:error:)` that reads `(error as NSError).domain` and `.code`.
2. Add `Gemma4Error.compileFailed(Gemma4CompileDiagnostic)`. Keep every existing case unchanged.
3. In the five lazy-axis accessors only (`rmsCooperativePipeline`, `q6CooperativePipeline`, `parallelAttentionPipeline`, `rmsStatsComputePipeline`, `rmsStatsCachedPipeline`): replace `try? makeComputePipelineState` with `do/catch` throwing `compileFailed(.make(stage: .pipeline, ...))`, and replace the library catch with `compileFailed(.make(stage: .library, ...))`. `functionMissing` is unchanged. Cached slots are written only on success, as now. Non-lazy accessors are untouched in this phase.
4. `GemmaForge` surfaces the diagnostic in the thrown error description only. No forge record, plan, digest or hash includes it.

**Classification helper.** `GemmaForge.isCandidateLocal(_:)` returns true only for `candidateRejected`. It is false for `compileFailed`, `shaderCompileFailed`, `functionMissing`, `badTensorGeometry`, `metalUnavailable`, `commandBufferFailed`, `nonFiniteActivation`, `executionFailed`, `invalidPlan`, graph and SSA errors. No string inspection.

## Scope 2: pure axis validation is the only recoverable rejection

**Sources/LeviathanCore/GemmaForge.swift**

5. `validateCandidate` throws `candidateRejected(String)` for its existing rules plus `rmsnormCooperative && rmsnormTiled` and `matvecCooperativeQ6 && matvecTiledQ6`. It is pure and runs in `executeGraph` and `executeGraphContinue` before `reset()`, before any slot mutation and before any dispatch. The runner's mid-dispatch `badTensorGeometry` checks stay as the global second line.
6. The forge loop snapshots `entryContext = runner.contextPosition()`, `entryDispatch = runner.dispatchTotal`, `entrySubmitted` before each candidate. On `candidateRejected` it asserts `openEnc == nil` and all three counters equal their entry values, not zero. Any mismatch throws `executionFailed`. It appends a result with `correct: false`, nil medians, and note `"rejected before dispatch: ..."`, then continues.
7. `executeGraphContinue` on rejection leaves `consumedPosition` and the open segment exactly as at entry; it performs no reset.

**Counted fallbacks are untouched.** Ineligible RMS, statistics or router geometry keeps recording production fallback counters and returning false. No fallback path is promoted to rejection.

## Honest limit

Every entry of `candidates()` passes `validateCandidate`, so this change does not let the forge skip any candidate that fails to compile on hardware. A lazy-axis pipeline failure still aborts the forge, now with domain, code and stage recorded. Deciding which of those are candidate-local is a future review that needs the diagnostics this task produces.

## Tests

Pure:

- `Gemma4CompileDiagnostic.make` with a synthetic `NSError(domain: "TestDomain", code: 42)` preserves domain, code, stage and kernel; round-trips through `Codable`.
- `isCandidateLocal` matrix per the list above.
- `validateCandidate` rejects both new pairs and existing illegal inputs; all `candidates()` entries pass.
- Rejected result encodes with null timings and decodes equal.

Real model, one fixture, prompt only:

- After a valid prompt, `executeGraphContinue` with `rmsnormCooperative: true, rmsnormTiled: true` throws `candidateRejected`; `contextPosition()`, `dispatchTotal`, `submittedBufferTotal` equal their pre-call values and the open segment is nil. A following valid continue reproduces the reference argmax within the existing 1e-4 gate.
- Existing Metal, regression and one-prompt checks pass unchanged. Each lazy axis still compiles on the installed device.

Existing tests asserting `invalidGraph` from `validateCandidate` or `shaderCompileFailed` from these five accessors must be updated.

## Acceptance

No full forge rerun is scheduled for this diagnostic-only change. The next ordinary CLI invocation re-forges because `executableDigest` changes. Its selected candidate is reported as the actual legal selection under the unchanged objective and gates. A different winner than the published main is logged as information, not suppressed and not forced.
