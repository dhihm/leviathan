The review directory returned nothing on every listing, so this verdict rests on the pasted replacements alone, checked against the previously confirmed source and the SSA contract.

**Verdict: ACCEPT, no blockers.** All six replacements do what the prior review asked, and none touches the forge or the semantic runtime path.

**Verification of the four requested points**

- **Baseline index 0 is now explicit.** The helper keeps the element at offset zero regardless of its flags, and the doc comment says so. The test mirrors the same expression. That mirror is tautological on current main because the baseline tuple has gpuCacheAppend false, so the data never exercises the pin. Acceptable, but declare it as a structural pin rather than a data-proven one.
- **Nil-layer fixture passes the existing validator before the unsupported assertion.** The mutation removes the layer key, rewrites the output to the negative-layer kvstate name, and renames the matching value entry including its id. It is gated on the validator helper first, then asserts layerAbsent for instruction three followed by missingLayer for layer zero in ascending order, then runs the shadow check with supported false. Confirm the validator helper counts a rejection as a test failure, otherwise a silently skipped fixture would look like a pass.
- **Multi-reason fixture is validator-consistent.** Attrs two by eight give sixteen, equal to the operand length and under the hidden bound, so the validator accepts it. Against the config's local two by four geometry, discovery emits geometry, length, and output reasons on the same instruction in that order, pinning intra-instruction ordering as requested.
- **Recorded digests are scoped correctly.** The comment states they are regression constants under the fixed test hardware profile, not a portable proof, and that the separate SSA pin tells lowering drift from codec drift. That is the right claim. The two hex constants cannot be checked from pasted text and must be confirmed by one test run.

**Other confirmations**

- The softcap comment now gives both justifications: JSONEncoder rejects non-finite floats, and lowering converts the cap to an integer. Correct.
- Semantic runtime files remain byte-identical. The report is analysis only and the forge never reads it.
- Docs should declare the extra sanity checks and the existing bound of 4096 as the diagnostic resource limit. Do not claim this is the first Float-bearing digested artifact. The graph digest already hashes the final softcap, so that earlier statement was wrong and should be struck.

**Unverifiable, settle by one compile and run:** the mutation helper taking an inout dictionary, the append spec initializer labels, the body and report variable names in the digest scope, and the two constants.

**Next discovery step, bounded**

Add one axis to the shadow helper using the existing RMS statistics analysis entry point. Call it as-is, convert its result into the same report reason and proof shape, and compose the two axes into a per-axis verdict map. Do not reimplement or extend the analysis, do not activate anything at runtime, and do not touch the forge. Acceptance stays the same pattern: whole-tuple equivalence with the manual list including the pinned baseline, repeated and Codable determinism, and one valid-but-unsupported fixture for the new axis. Flag names come from the real candidate tuple definition.

**Next performance step, bounded, as two separate measurements**

The failed variants remain failed under their declared criteria. Variance does not make those outcomes uninterpretable or invalid. The purpose below is to establish a noise floor for future sub-five-percent decisions, not to reopen old ones.

1. **Baseline A/A noise control.** Seven visits of the unmodified baseline in one session, interleaved with nothing else. Keep all seven, drop no outliers. Report per-visit median ms/token, the overall median, and the relative spread defined as the maximum minus the minimum per-visit median, divided by the overall median. Require a quiet window and no GPU work of our own in parallel. Declare up front that a spread above five percent means the harness is fixed before any sub-five-percent effect is judged.
2. **Separate instrumented gap trace.** The current breakdown's flush-minus-GPU figure does not measure CPU busy time or the gap from command completion to the next commit, and there is no public segment timestamp hook to read one. Measuring that gap therefore needs a new instrumentation hook, added in its own run, never mixed into the noise-control run. Report commits per token and summed completion-to-next-commit gap. Drop the flush-batching hypothesis before trying it if commits are two or fewer per token or the gap is under two milliseconds.

Repeat none of the failed RMS width, copy rewrite, or Q4 staging variants. Make no claim about discovery improving throughput and no ratio against another runner.
