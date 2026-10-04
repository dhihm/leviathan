**Verdict: ACCEPT** the two-axis shadow composition as designed, with the precisions below. No blocker found in the three sources. Two fixture prerequisites and one catalog observation need confirming before tests are written.

## Design brief for the new file

**One new file, two new types, one new error enum.** Suggested names `GemmaTwoAxisOpportunityDiscovery` and `GemmaTwoAxisOpportunityReport`, in a new file next to the existing discovery. Nothing in the existing discovery file, the RMS analyzer, Forge, the catalog, plan identity, or schema changes. The existing report's `init(body:)` is internal, which is fine, since the composite never constructs a cache report, it only embeds the one `discover` returns.

**Inputs.** The SSA, the graph config, the hardware profile, and an explicit `epsilon: Float`. The caller supplies the validated model configuration's RMS epsilon. The composite never reads a runner and never hardcodes a value.

**Evaluation order, fail-closed throughout.**
1. Validate epsilon finite and strictly positive. Throw the composite's own `.invalidEpsilon` before anything else, so tests get a typed error rather than the analyzer's `GemmaSSAError.badLowering` for the same condition. Subnormals pass, matching the analyzer.
2. Call the existing `discover` as-is. It owns config, hardware, and SSA validation. Wrap its error as `.cacheAxis(GemmaOpportunityDiscoveryError)`.
3. Call the existing `analyze` as-is with the SSA and epsilon. Wrap any thrown `GemmaSSAError` as `.rmsAxis(String)`. A throw here is malformed input, not an unsupported verdict.
4. Cross-check and throw `.inconsistentAnalysis` if the plan's `analysis` field differs from `RMSStatisticsPlan.analysisVersion`, if the plan's epsilon bits differ from the input's bit pattern, or if group count is positive while reuse count is zero. The last cannot happen with the current analyzer, since every group needs two members, but the check is cheap and documents the invariant.

**Composed body fields.** Overall analysis version constant 1. The full embedded `GemmaOpportunityReport` including its own digest. The full embedded `RMSStatisticsPlan`, which carries version 2, groups, and the dense action array as the proof. The epsilon bits as `UInt32`, never as a Float. An ordered array of exactly two axis verdicts, cache first then RMS, each with the axis name string, a supported flag, and a typed reason list. The RMS verdict is supported iff group count and reuse count are both positive; otherwise it carries the single typed reason `noReusableGroups`. The cache verdict mirrors the embedded report's `supported` and summarizes its reason codes.

**Digest.** SHA-256 hex over the sorted-key JSON of the body alone, computed exactly as the existing report does it, with a `verifyDigest` that recomputes. Determinism holds: the analyzer sorts groups by first member index, which is unique per group, and the action array is dense by instruction index, so dictionary iteration order never leaks into the plan.

**shadowCandidates.** Takes the same four inputs, re-runs the full composite, never accepts a report. Keeps index 0 unconditionally. For every later tuple, withholds it if it carries `gpuCacheAppend` and the cache verdict is unsupported, or carries `rmsStatisticsReuse` and the RMS verdict is unsupported. All other flags and the original order pass through untouched.

**Catalog observation your tests must state explicitly.** Every catalog tuple with `rmsStatisticsReuse` also carries `gpuCacheAppend`. The RMS verdict therefore never changes the shadow result independently of the cache verdict in the current catalog. Expected counts from the pasted catalog:

| Cache verdict | RMS verdict | Tuples kept |
|---|---|---|
| supported | supported | 32 |
| supported | unsupported | 26 |
| unsupported | supported | 14 |
| unsupported | unsupported | 14 |

The swapped case equals the no-cache case. Assert that equality rather than implying the RMS axis has independent filtering effect today.

**Fixture prerequisites, confirm before writing tests.**
- The cache-unsupported negatives need an SSA that still passes `GemmaSSA.validate` and still passes `analyze`. Reuse whatever mutation the existing discovery negative tests already use, and add an assertion that the analyzer still returns a plan on it.
- The RMS-unsupported negative needs a validate-passing SSA with zero reusable groups. The canonical lowering may not admit one. If no such fixture can be built, test the verdict mapping through an internal non-public helper that takes a plan, and document the quadrant as unreachable from canonical lowering.

**Tests to specify.** Positive canonical fixture yields 30 cache proofs, 30 groups, 60 reuses, 32 tuples, both verdicts supported, embedded epsilon bits equal to the input bit pattern, embedded nested digest verifies. The four negative quadrants above. Epsilon zero, negative, infinite, and NaN each throw `.invalidEpsilon`. Changing epsilon to a different positive value changes the recorded bits and the overall digest while leaving the cache report digest unchanged. Same inputs twice give byte-identical encoded bodies and digests. Codable round-trip preserves equality; mutating any body field after decoding makes `verifyDigest` return false. No test or API path feeds a decoded report back into candidate filtering.

**Activation readiness.** This proves two axes statically. The other manual axes stay manual: parallel attention, cooperative RMS, cooperative Q6, router residency, launch divisor, tiled kernels, and segment batching have no discovery at all. A static RMS verdict also does not certify the runtime path, which still requires the cooperative plus vectorized legality guard, no installed generated RMS kernel, matching epsilon bits at execution, and per-token slot readiness. The pool cannot be swapped wholesale on this evidence.

## Matched diagnostic assessment

**Yes, this is the smallest useful matched experiment, and discovery supports it only as existence evidence.** The 30 groups and 60 reuses establish that the opportunity is real in the IR. They say nothing about magnitude. A cached apply still dispatches one kernel per reused norm, so launch count barely moves and the saving is reduction work only. On a wall-time objective dominated by CPU launch overhead, a sub-2% effect is a plausible honest outcome, and that would be a resolution result, not a refutation of the mechanism.

**The OFF arm is legal but unlisted.** Residency true with divisor 1 and reuse false passes `validateCandidate`, since divisor 1 only requires residency and no tiled F32. It is not in the catalog, so it stays a diagnostic tuple passed to the public prepared `executeGraph`, never added to candidates and never a production selection. The Q4 launch width and the MoE install state are bound by the prepared execution's binding check, so they are held constant across arms automatically.

**Critical prerequisite before any timing.** Confirm the ON arm actually serves statistics. If the runner has a generated RMS kernel installed, every planned action falls back and ON equals OFF by construction. Read the statistics fallback counter after the correctness gate runs and require it to be zero for ON. Without this, the attribution is void regardless of the timing result.

**Correctness gates.** All three fixtures, 32 steps, exact greedy IDs and the existing 1e-4 relative logit gate, run outside the timed region before the first pair. Record the worst relative difference between arms. It should be very small, since reuse is bit-exact by construction, but do not add an exactness requirement the suite does not already assert.

**Conditions for not proceeding to the A/B.** Do not run the pairs if the A/A spread exceeds 5% of the median, if any of the 7 visits was dropped or had fewer than 3 warm runs, if the thermal state is anything but nominal, if the machine is not on AC with a healthy battery state, or if page-out or swap activity grows across visits. Also note the gate mismatch: a 5% A/A spread is looser than the 2% effect gate. Four interleaved AB and BA pairs with every pair above 2% in the same direction is strong evidence. Any pair failing is inconclusive, not a negative result, and must be reported that way.

**Framing.** ON is already main. This is attribution of an IR-proven opportunity within the current stack, not a gain claim against main and not a claim from the shadow composition. No noise-eliminated statement, no external throughput ratio.
