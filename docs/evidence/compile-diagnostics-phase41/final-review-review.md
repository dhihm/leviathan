ACCEPT, conditional on the full `leviathan-tests` target building and the pre-existing gates still passing. The source tree is not present in this directory, so the review below is of the diff text only.

**Evidence the patch stays inside the diagnostics-only scope**

- All five lazy accessors are touched and only those: RMS cooperative, Q6 cooperative, RMS statistics compute, RMS statistics cached-apply, and parallel attention. Each has exactly two new catch sites, one `.library` and one `.pipeline`.
- Every catch site calls `Gemma4CompileDiagnostic.make` with the correct kernel constant for its accessor. The two RMS statistics accessors correctly use `computeKernelName` and `applyKernelName` respectively.
- The `make` helper copies `domain`, `code`, and `localizedDescription` straight off the bridged NSError. Nothing parses message text.
- Missing-function paths are byte-identical: all five still throw `functionMissing(kernelName)` from the same `guard let fn` line.
- The parallel attention geometry guard and its `badTensorGeometry` throw are unchanged context. Only the comment above it was rewritten.
- Caching is unchanged in position: the `...PipelineState = pipe` assignment still follows the last throwing step in every accessor. No path assigns before success.
- No `try?`, no `nil` return, no fallback library, no candidate rejection was introduced. The four former `try?` guards were converted to `do/catch` that still throws, which strictly adds information and does not change control flow.
- The pipeline-stage errors previously discarded by `try?` are now preserved. That is the one real diagnostic gain of the patch.

**Tests**

- Synthetic coverage checks stage, kernel, domain, code, and message for all three stages plus a Codable roundtrip and `Gemma4Error` equality and inequality. Correct and self-contained.
- The real Metal gate compiles genuinely invalid MSL on the default device and compares against the captured original NSError, not literal text. It skips cleanly without a device.
- As you note, the Metal gate exercises a test-local wrapper, not the five production accessors. Production catch-site correctness therefore rests on the source review above, which I consider sufficient here because the ten sites are mechanically identical and visible in full.

**Non-blocking observations**

- `Stage.function` is public API that no production path ever produces, since missing functions stay `functionMissing`. It is dead but harmless. Consider dropping it or documenting it as reserved.
- The test increments `passed` once without a named check inside the Metal catch branch. Cosmetic.

**Limitations and the single thing to confirm before merge**

Adding `compileFailed` to the public `Gemma4Error` enum will break any exhaustive `switch` over `Gemma4Error` elsewhere in the package or in existing test gates, and any existing gate that asserted `.shaderCompileFailed` from one of these five accessors will now fail. The diff shows no such edits, so either none exist or the build and suite will tell you immediately. I could not grep for this because the source is not on disk here. A green build of both targets is the only acceptance condition I am attaching.
