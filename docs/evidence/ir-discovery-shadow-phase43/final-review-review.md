The review directory is empty, so this review is based solely on the pasted source, tests, SSA contract and census.

**Verdict: CONFIRM with no blockers.** The implementation matches the published boundary, the fixtures are validated before assertions, and arithmetic cannot trap after sanity. Four items need declaration or follow-up, none of which blocks shadow landing.

**Correctness against the boundary**

- All four proof rules are implemented as specified: legal layer, exactly two operands named t2 then kv, vector f32 with zero or absent slice offset, config geometry per layer globalness, checked UInt32 width, matching lengths and kvstate output. The census rows (30 appends, 30 unique layers, t2/kv order, nil slices, 2048 or 1024 widths) all satisfy these rules, so a supported verdict on the real graph is consistent with independent inspection.
- Coverage counts every present append toward its layer before proof checks, so a withheld append yields its own reason rather than a spurious missing-layer reason. Duplicate and missing reasons are emitted in ascending layer order, never dictionary order.
- `supported` requires empty reasons and proofs equal to layer count. The second condition is redundant given the first but fails closed. Fine.
- Unreachable-after-validate branches exist: `operandCount`, the dtype half of `operandKind`, and the `overflow` half of the width check. They are harmless dead guards, not defects. `operandKind` can only fire alongside `operandBinding` because vector names imply vector kind in the validator.
- The decoder-synthesized init preserves a stale `reportDigest`, which is exactly what makes the tamper tests meaningful. `verifyDigest` recomputes from body alone. Correct.

**Trap and bounds**

- The only unchecked product in `GemmaSSA.validate` is `8 * c.hidden` for `maxLen`. Config sanity rejects that overflow before the validator runs. Every other product in the validator uses `checkedMul`. The lowering path has unchecked arithmetic but discovery never calls it.
- Discovery's own width uses `multipliedReportingOverflow`, and the output-length comparison is gated behind `fitsWord`. Layer count is bounded at 4096 before the coverage array is built. Per-layer loop starts from a positive layer count.
- The softcap check is not only sanity: JSONEncoder throws on NaN or infinity by default, so without it a report digest could throw after a valid verdict. Keep it, and say so in the comment.

**Fixture validity**

- Every valid-but-unsupported fixture (swapped operands, duplicate, missing, local geometry on global layer, nonzero slice) goes through `odValidates` first and short-circuits on rejection. Traced against the validator, each one passes: the validator checks attr product against operand length but never against config geometry, and never checks input slice offsets or operand order. So these are genuine boundary tests.
- Synthetic instruction indices 3 and 6 (and 9 for the duplicate case), value IDs t2@1/kv@1 and kvstate_0@0/kvstate_0@1, and the explicit zero-slice accepted case all check out by hand enumeration.
- The stale fixture is correctly a throw test, not a boundary test, and is asserted to fail the validator first.

**Items to declare or follow up (non-blocking)**

1. **Config sanity exceeds the plan's list.** The plan names layer count, KV heads and dims, global membership, and the approved 8*hidden check. The implementation also requires positive vocab, qHeads, expert counts, intermediates, nonnegative window, and a bounded softcap. The softcap check is justified above. The expert and window checks are undeclared scope widening. Either trim them or add one sentence to the receipt declaring them as discovery-owned invariants.
2. **Baseline retention is implicit.** `shadowCandidates` keeps the baseline only because the baseline tuple has `gpuCacheAppend` false. The tests catch regression on current main, but the helper should pin index zero explicitly before any later axis is added.
3. **Untested reachable reasons.** `layerAbsent` is reachable (a nil-layer append writing `kvstate_-1` passes the validator). The multi-reason case for one instruction (attr geometry whose product differs from config width, yielding geometry, length and output reasons together) is also untested. Add one fixture each so reason ordering within an instruction is pinned before activation.
4. **Digest stability across Foundation versions is unproven.** The body is the first digested artifact in this codebase that contains a Float. Adequate for diagnostic bookkeeping now, but before the later activation step binds this digest into identity, pin float encoding or test the digest against a recorded constant.

**Unverifiable from supplied material:** the canonical-setup helper names, `GemmaForgeCandidate` Equatable, `GraphConfig` and `ForgeHardwareProfile` Sendable and Equatable conformance, and the Opus provenance claim. The pasted source carries no in-text provenance marker, so the receipt in the evidence directory must record the file hash. Compile the test target once to settle the conformance questions.

**Next smallest implementation step toward replacing the manual pool**

Generalize the shadow helper to a per-axis verdict map and add one more axis rule that needs no new analysis: the RMS statistics reuse flag, decided by calling the existing pure `RMSStatisticsPlan.analyze` and converting its result into the same report shape. Same acceptance pattern: canonical whole-tuple equivalence with the manual list, repeated and Codable determinism, one valid-but-unsupported fixture. This exercises the two-axis composition path without touching the forge or inventing any new rule. Flag names must be taken from the real candidate tuple definition.

**Next concrete performance step**

The failed results share one pattern: effects under 5% judged against runs whose first sample moved by 10%. No noise floor has been established, so none of those outcomes is interpretable. The bounded step is a measurement, not a tuning:

- Run the unmodified baseline seven times, interleaved, same session, and record median and spread of ms/token.
- In the same runs, count command-buffer commits per token and sum the host wait between GPU completion and the next commit, since host flush is the largest non-GPU bucket in the current census.
- Rejection criteria, declared up front: if baseline spread exceeds 5%, no sub-5% result including the staged 1.38% Q4 component can be accepted or rejected and the harness must be fixed first. If commits per token are two or fewer, or the idle gap is under 2 ms, drop the flush-batching hypothesis without trying it.

This repeats none of the failed RMS width, copy rewrite or Q4 staging tunings, needs no new models, and makes no claim about discovery improving tokens per second or about any ratio against another runner.
