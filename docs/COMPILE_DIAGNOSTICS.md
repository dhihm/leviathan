# Lazy Metal compilation diagnostics

The five optional Gemma runtime pipelines previously erased compilation provenance. Four pipeline-state creation calls used `try?`, replacing every Metal error with a generic string. Library and attention pipeline errors also lost their original error domain and code. This made source compilation failures and resource failures difficult to distinguish during review.

This experiment preserves structured diagnostics while keeping every compilation failure fatal to the caller. It does not classify hardware errors as candidate-local or skip failed candidates. Missing functions and illegal pipeline geometry retain their existing errors. No runtime fallback, shader, launch geometry, candidate order, timing objective, artifact schema or correctness threshold changes.

## Design-to-code trace

| Reviewed requirement | Implementation location | Verification |
| --- | --- | --- |
| Preserve compilation stage, kernel identity, NSError domain/code and message | `Gemma4CompileDiagnostic` in `Gemma4.swift` | Synthetic NSError and Codable checks |
| Preserve errors instead of suppressing them with `try?` | RMS cooperative, Q6 cooperative, RMS statistics compute/apply, parallel attention lazy accessors | Source review and regression suite |
| Keep failure global and fail-closed | `Gemma4Error.compileFailed`; no candidate recovery handling | Existing propagation unchanged; invalid Metal source test |
| Cache only successful pipeline creation | Existing cache assignments remain after successful compilation and geometry checks | Exact patch review |
| Preserve successful execution | All shader sources and other runtime/compiler sources remain identical to validated main | File hash comparison and real-model regression |

The Fable 5.1 correction review completed on 2026-10-04 at 16:55:53 UTC. Its wider candidate-isolation proposal was deliberately narrowed to this foundation: the existing errors cannot safely establish whether an allocation failure is recoverable. Pure candidate validation changes are deferred to a separate review.

Implementation and tests are externally authored through the user-authorized Opus 5.5 route, with no tools enabled. The supervising agent reviews and applies exact replacement patches, records provenance and runs verification. Documentation is written independently in English.

## Artifact identity and limits

The executable digest remains part of artifact identity, so a newly built executable must not silently reuse artifacts bound to another executable. Existing artifact roundtrip and rejection gates apply. No production artifact is rewritten to bypass identity checks.

This diagnostics change makes no performance improvement claim. Experiment `541d0cf` and its rejected RMS launch choices remain preserved on their private branch. The diagnostics branch was based on validated main `cc1219f`; publication follows the integrated acceptance below. The existing 14,439,363,584-byte model is reused; no download or additional service setup is required.

## Recorded acceptance

- Release build completed successfully in 108.82 seconds. The initial sandboxed build could not write the existing compiler cache; the authorized build succeeded without changing the toolchain or source.
- Existing real-model and Metal regression suite: **1,669 passed, 0 failed**. The first run had one harness-layout failure because the existing CLI checks require `.build/debug/leviathan-cli`. A symlink to the identical release executable resolved that path; no test or product code changed between runs. Both logs are retained.
- Externally authored focused real-model replay, completed at **17:10:27 UTC**: all five lazy paths engaged. Per arm, RMS cooperative 723, Q6 cooperative 3, parallel attention 90, statistics compute 90 and cached apply 180; all counted fallbacks zero. Both 262,144-logit steps and IDs were bit-identical on replay, and engine identity was stable after lazy compilation. This is repeatability of the unchanged successful path, not an independent numerical oracle or a full production qualification.
- Actual CLI reload of preserved previous-executable artifacts, completed at **17:11:15 UTC**: rejected with `q4 artifact invalid: identity digest mismatch`, exit 1, no generation, and both artifact files unchanged. This is the negative identity gate; the subsequent fresh generation and normal reload are recorded below.
- Fable 5.1 accepted the exact diff conditional on build and regression success; those conditions passed. No broad error recovery or candidate skipping was introduced. Missing-function and geometry checks and all successful cache assignments remain unchanged.

The synthetic diagnostics tests cover all stage values; `function` is reserved and is not emitted by these accessors, which retain `functionMissing`. The real invalid-source test uses a test-local wrapper, not fault injection into a production accessor. Production failure wiring is verified by reviewing all ten exact catch sites. Resource exhaustion is not deliberately induced.

The evidence directory contains exact externally authored replacements, application hashes, model receipts, source-scope comparison, build/test logs, focused driver and replay results, and negative artifact-reload receipts. Model receipt cost fields are provider list-cost metadata, not a claim about account charges.

## Integrated production artifact lifecycle

The remaining lifecycle validation is complete. The unchanged diagnostics executable generated fresh optimized artifacts through the normal automatic pipeline, then reloaded those artifacts normally. Fresh selection exited 0 in 756.23 seconds; reload exited 0 in 36.64 seconds. Source and executable hashes were unchanged across both. The original incompatible artifacts remain preserved and rejected.

Actual automatic selection: batched vectorized loads, GPU cache append, parallel attention, cooperative RMS and Q6, RMS statistics reuse and router GPU residency, router divisor **1**, E4R32 and Q4 width **32**. This is the actual legal selection from this run; no winner was forced and no timing result is presented as a speed gain.

Raw Hello, English chat and Korean chat each generated 32 IDs and passed the existing untimed reference replay: all 262,144 logits at each step within the unchanged 1e-4 normalized threshold and exact greedy IDs. All nine reported fallback counters were zero in every fixture. Elapsed lifecycle checks were 39.63, 59.00 and 62.14 seconds respectively; these include preparation and reference replay and are not throughput comparisons.

Internal plan digest: `50107816f506952aa3bee1389563d59cb849841c427566ee92ae2f4b844114b8`. Internal composite digest: `a82d1319a75a2575a2b9ebfa63696df31c700a6a38afdb773c0c409d44cb4795`. These are internal digests, not artifact-file SHA-256 values; both are recorded distinctly in acceptance evidence.

The integrated tree has the same source and executable that passed 1,669 regression checks, Fable final review, actual lazy-path replay, negative identity reload and positive fresh/reload/three-input gates. Only documentation and evidence were added afterward. This diagnostic correctness improvement is eligible for private main publication without a performance claim.
