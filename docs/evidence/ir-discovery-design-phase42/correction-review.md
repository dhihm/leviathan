Corrected design follows. It is a shadow-only, pure, device-free slice; the manual list stays the runtime authority.

## Corrections to the prior plan

- **Acceptance is no longer byte-identical plans.** The artifact carries measured timings, timestamps and the actual winner, which legally varies. Acceptance becomes three independent gates: deterministic discovery reports, candidate-set equivalence against the manual list, and the existing correctness and artifact-lifecycle tests passing unchanged.
- **No generator signature change in shadow.** The prior plan made `candidates()` take SSA and hardware. Retracted. In shadow the manual list is untouched and discovery is never consulted by forge selection, plan creation or plan validation.
- **Invalid hardware throws.** It is a global input error, the same class as an invalid SSA. It never yields an empty candidate list, because every non-throwing output contains the baseline.
- **No RMS duplication.** The shipped analyzer already does rmsnorm grouping with its own version, effect model and backing map. Discovery treats rmsnorm as out of scope and does not re-derive liveness, aliasing or grouping. It reuses two of its conventions only: a static analysis version constant, and fail-closed errors for malformed input with skip-not-throw for merely unsupported instructions.
- **Coverage is per-layer uniqueness, not a count.** The axis is admissible only when the set of layers carrying one cache_append each equals the canonical layer range. A duplicate append on one layer or a missing layer is unsupported with a reason naming the layer.

## Pure discovery contract

**Entry point.** One pure function, proposed name `GemmaOpportunityDiscovery.discover`, taking the SSA, the graph config and the hardware profile. It touches no runner, Metal or GGUF.

**Global input checks, all throwing.**
- SSA fails `GemmaSSA.validate` against the config. Discovery runs validation itself so a caller cannot skip it.
- Config is invalid by the existing config validator.
- Hardware profile is invalid by the profile type's own existing validation. Discovery adds no cap of its own. The thread-geometry contract for the copy kernel is whatever `gpuCacheAppendCopy` and the profile already enforce. If that contract exposes a bound, discovery checks the kernel's unsigned thread-count parameter against that bound. It does not fix a number such as the DAG threadgroup cap. This remains an open fact until the copy kernel and profile sources are read.

**Per-instruction static proof for cache_append.** Every item must hold, mirroring the static subset of the executor guard and the operand order the executor relies on.
1. Layer non-nil and inside the config layer range.
2. Exactly two inputs. The first resolves to a live value named `t2`, the second to `kv`. Both kind vector, dtype f32, slice offset nil or zero, since the copy kernel binds whole buffers with no offset argument.
3. Attributes kvHeads and headDim present, positive, and equal to the config geometry for that layer, chosen by whether the layer is in the config's global layer set.
4. Both input lengths equal the overflow-checked product. The output is kind kvstate, named for the same layer, same length.

Note that SSA validation checks lengths and the output but not the input names or kinds. Items two and three are where discovery adds proof the validator does not.

**Axis verdict.** Supported, with one evidence entry per layer sorted by layer, or unsupported with a reason naming the instruction index, value ids and layer. One unsupported instruction withholds the whole axis because the runner flag is global. The baseline is always present.

**Report.** A Codable, Equatable value with the analysis version, the axis verdict, the evidence list ordered by instruction index, and a digest computed the way `GemmaSSA.digest` does it: sorted-keys JSON hashed with the existing SHA256. Reasons are an enum with associated values rendered to stable strings. Nothing in the report comes from time, randomness or device state.

**Shadow equivalence.** Given the manual list and the verdict, the derived list is the manual list with entries carrying the cache-append flag kept only when the verdict is supported. For the real config the derived list must equal the manual list in order and flag content. This comparison lives in tests and an optional diagnostic entry point, not in forge.

**Static versus runtime.** Position bounds, cache buffer lengths, pipeline compilation and buffer binding stay runtime `executionFailed` errors. The executor guard is unchanged. A discovered candidate can still be rejected by the unchanged numeric gates.

## Activation rule, tests, limitations, next brief

**When discovery later becomes active**, identity must change, not just the results-set check. The plan identity gains the discovery analysis version, the report digest and the candidate-set digest, and the artifact schema is bumped so old artifacts are rejected on schema and identity, independently of candidate membership. This is not part of the shadow slice.

**Pure tests, no kernels**, driven by `GemmaGraph.expectedSequence` and `GemmaSSA.lower` on the real config, plus hand-built SSA that passes validation.
- Real config: axis supported, evidence covers each canonical layer exactly once, derived list equals the manual list.
- Determinism: two runs produce equal reports and equal digests.
- Swapped inputs, `kv` then `t2`: validation passes, axis unsupported, reason names the instruction index and value ids.
- Duplicate append on one layer, and a schedule missing one layer: both unsupported, reason names the layer.
- Local geometry on a global layer with internally consistent lengths: unsupported with a geometry reason.
- Stale SSA reference throws. Invalid profile throws. Every non-throwing report contains the baseline.

**Limitations.** Only the cache-append axis is derived. The hardware check is delegated and may reduce to profile validity. No performance claim is made or needed, since the axis is bit-exact by construction. The RMS analyzer continues to treat cache_append as opaque; discovery does not alter that.

**Next implementation brief.** Add one pure type in LeviathanCore with the analysis version constant, the report type and the discover function. Add one test file covering the six cases above. Make no change to `GemmaForge.candidates`, `makePlan`, `validatePlanIdentity`, the executor or the runner. Before coding, read three sources not yet supplied: the hardware profile type and its validator, `gpuCacheAppendCopy` for the thread-count parameter and any offset argument, and the manual candidate tuple type for the equivalence comparison.
