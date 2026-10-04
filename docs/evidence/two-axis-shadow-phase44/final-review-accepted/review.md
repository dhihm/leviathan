ACCEPT

**Blocking findings:** none.

I checked the composite against the two wrapped analyzers it calls and against the test file. The implementation does what it claims and nothing more.

**Correctness of the implementation**

- **Order and fail-closed behaviour are right.** The epsilon guard runs before either analysis, so an invalid epsilon never reaches the analyzer's string-message path. Cache errors and RMS errors propagate as their original typed values. Nothing is swallowed or converted to a silent unsupported verdict.
- **Verdict predicates match the source.** The cache verdict is the embedded report's own supported flag. The RMS verdict is groups positive and reuses positive. Reason is nil exactly when supported, in both branches.
- **No cross-check is needed for version or epsilon bits.** The analyzer always stamps the plan with its own constant and the caller's bit pattern. Recording them from the fresh plan is sufficient. The test asserts both against the constants, so a drift would surface there. The redundant cross-check from the design sketch is future scope, not a blocker.
- **Digest determinism holds.** The plan has no dictionaries. Groups are sorted by first member index, which is unique because an instruction joins at most one group. Actions are dense by index. The embedded cache report is already deterministic. Sorted-key JSON over the body alone matches the existing pattern byte for byte.
- **Shadow filter is correct and never consumes a report.** It re-runs discovery from raw inputs, pins index 0, and withholds only tuples whose enabled composed axes are unsupported. All other flags pass through in manual order.
- **No duplication of analysis.** The RMS analyzer is called once, unmodified. No grouping, liveness, or alias rule is re-derived.
- **Invalid epsilon typing is sound.** Carrying the bit pattern avoids the NaN inequality trap. Negative zero is correctly rejected since it fails the positive test.

**Fixture coverage of the four quadrants**

The hardcoded counts are arithmetically consistent with the pinned manual list, which is what makes them meaningful despite the predicate mirror.

| Pinned fact | Consequence |
|---|---|
| 32 tuples, 18 carry cache, 6 carry RMS, all 6 RMS tuples carry cache | cache off removes 18 in both RMS cases, RMS off alone removes 6 |

- **Canonical** gives both supported. Direct cache report equality, direct plan equality, 30 proofs, 30 groups, 60 reuses, dense action array length, and per-layer ordering are all asserted independently of the composite.
- **Swapped first cache inputs** gives cache off, RMS on. The swap only reorders two already-live inputs of an opaque op, so the analyzer is unaffected. The test confirms that with the 30 and 60 counts and the single operand-binding reason.
- **Cache-only synthetic** gives cache on, RMS off. Two proven appends, zero groups, every action nil, dense array length equal to instruction count.
- **Zero-append synthetic** gives both off with two missing-layer reasons.

All four quadrants are reached by real fixtures that pass the existing validator. The quadrants also compare the composite's evidence to the direct analyses, so the predicate-mirror helper is not the only oracle.

**Nonblocking observations**

- The test comment calling the predicate helper an independent oracle is wrong. It mirrors the filter. The independent checks are the pinned counts, the manual-list facts, and the direct analysis equalities. Reword the comment externally.
- The RMS-only withholding branch, a tuple with RMS on and cache off, is unreachable with the current catalog. The counts for the two cache-off quadrants coincide at 14. The filter is hardwired to the forge's candidate list, so that branch cannot be unit tested without a catalog change.
- The RMS-unsupported quadrant uses schedules with no rmsnorm at all. A schedule with present but ineligible norms, such as in-place or single-member groups, is not exercised here. The existing analyzer tests presumably cover that.
- The body stores the RMS analysis version twice, once as a top-level field and once inside the plan. Harmless redundancy.
- The verdict struct does not enforce the reason-nil-iff-supported invariant on decode. A tampered decoded verdict can be inconsistent, but the digest check catches it and decoded reports are diagnostic only.
- The hex formatting in the error description is unpadded. Cosmetic.

**Limitations of this review**

I did not see the test helpers for the synthetic SSA, mutation, validation, or error matching. I take the reported pass of all 1963 tests as evidence they behave as named. I also did not see the forge catalog or the SSA validator, so the pinned counts and fixture validity rest on the test run, not on my reading.

**Activation status**

This proves two axes statically and nothing else. The other manual axes have no discovery. The RMS verdict does not certify the runtime path, which still depends on the legality guard, the installed-kernel check, matching epsilon bits at execution, and slot readiness. The matched RMS diagnostic remains gated on the frozen baseline with A/A spread at or under 5 percent, correctness gates on all three fixtures outside the timer, a zero fallback count for the ON arm, and all four interleaved pairs above 2 percent in one direction. Any failing pair is inconclusive, not negative. ON is already main, so no gain claim follows from this work.
