Here is the full plan from the start.

## Current evidence

- **The manual list is the only candidate source.** `GemmaForge.candidates()` in `Sources/LeviathanCore/GemmaForge.swift` returns a hand-written array of flag tuples with prose justifications. Legality is partially enforced by the pure `GemmaForge.validateCandidate`, which covers only the statistics-reuse and router-divisor rules.
- **The set is already a plan identity input in effect.** `GemmaForge.makePlan` and `GemmaForge.validatePlanIdentity` both require the results to equal the legal set exactly. Any change to the generator therefore rejects old plans and re-forges. That is correct fail-closed behavior and it shapes the migration.
- **Typed SSA with use evidence exists.** `GemmaSSA.lower` emits versioned values with dtype, length, kind and slice offset. `GemmaSSA.validate` enforces def-before-use, live-version reads, per-op signatures and length geometry. It does not check value names per op beyond kvstate and routing prefixes.
- **An SSA-derived plan already exists.** `GemmaPreparedExecution.prepare` calls `RMSStatisticsPlan.analyze` once on the validated SSA. The executor rechecks each planned action against the live instruction and counts fallbacks. Its source was not supplied, so this plan relies only on its observed interface. It is the template, not something to rebuild.
- **The generic DAG side already derives opportunities.** `tools/forge_run/dag_frontend.py` imports linear, relu and add from torch.export with strict static checks. `DAGForge.legalCandidates` derives the fusion axis from `program.safeFusionPairs` and the device profile, and `DAGForge.kernelSequence` predicts dispatch on the host. The Gemma side should mirror that shape, not copy its ops.
- **Executor guards are already per instruction.** In `executeToken` the cache_append case checks layer bounds, kv width against kvHeads times headDim, and cache buffer bounds before either the GPU copy or the CPU memcpy path.

## Proposed boundary and first slice

**Pure boundary.** Add one device-free analysis, proposed name `GemmaOpportunityDiscovery`, taking a validated SSA, the graph config and a `ForgeHardwareProfile`. It returns one report per axis. A report is either supported with an evidence list, or unsupported with a reason that names the instruction index and value ids. It also returns the deterministic bounded candidate list and a digest of the report. It touches no runner, no Metal, no GGUF.

**One transformation.** The first slice derives only the `gpuCacheAppend` axis. It is bit-exact by construction and its preconditions are fully visible in the SSA. All other axes stay on the manual list and are passed through unchanged.

**Static preconditions for one cache_append instruction.** All must hold or the instruction is unsupported.

1. The op is cache_append with a non-nil layer inside the config layer count.
2. Exactly two inputs, both kind vector, dtype f32, names in `GemmaSSA.vectorNames`.
3. Attributes kvHeads and headDim present and positive. Both input lengths equal their checked product.
4. The output is kind kvstate, named for the same layer, same length as the inputs.
5. Both inputs are live versions at that point. This is already guaranteed by SSA validation and the discovery requires validation to have run first.

**Coverage rule.** The runner flag is global per candidate, so the axis is generated only when every cache_append in the schedule is supported and the count equals the layer count. One unsupported instruction withholds the whole axis and emits an explicit unsupported diagnostic. The baseline is always retained.

**Hardware filter.** For this axis the only hardware precondition is profile validity, using the same bounds `DAGForge.select` applies to the threadgroup cap. No occupancy or bandwidth claim is made because none is measured. The filter interface exists so later axes can use profile fields deterministically.

**Static versus runtime.** Cache buffer length bounds, pipeline compilation, position bounds and buffer binding failures stay as `GemmaForgeError.executionFailed` and remain global. Discovery never reasons about them. A candidate that passes discovery can still be rejected by the unchanged numeric gates, and the per-instruction executor guards stay exactly as they are.

**Enumeration and identity.** The generated list for slice 1 must be identical, in order and content, to the current manual list. Discovery decides only whether the entries carrying the cache-append flag are admissible. Order is the manual order, baseline first. Identity is unchanged, so no schema bump. Provenance goes to a forge-time diagnostic record, not the plan, in this slice.

**Call sites.** `candidates()` is parameterless today. The derived generator needs the SSA and hardware. All three callers already hold both: `forge` has the runner and profile, `makePlan` and `validatePlanIdentity` receive the graph and profile and can lower it. The signature change is mechanical.

## Tests, migration, acceptance, risks

**Pure tests, no kernels.** `GemmaGraph.expectedSequence` plus `GemmaSSA.lower` produce the real schedule from a config alone.

- Real config: every cache_append supported, evidence count equals the layer count, axis generated.
- Hand-built SSA that passes `GemmaSSA.validate` but names a cache_append input outside the allowed vector set: axis withheld, diagnostic names the instruction index.
- Determinism: two runs give equal reports, lists and digests.
- Hardware profile out of bounds: no candidates, explicit reason.
- Equivalence: derived list equals the manual list exactly.
- Every unsupported report has a non-empty reason, and the baseline is present in every output.

**Integration lifecycle.** Discovery runs once in `forge` before `GemmaPreparedExecution.prepare`, and again in plan validation on reuse. The existing device tests, numeric gates, warmup and timed policy and the two percent baseline threshold are unchanged.

**Migration.** Phase one is shadow mode: the manual list drives the forge, discovery runs alongside, and any discrepancy fails a test. Phase two swaps the generator and keeps the manual list as the test oracle. Phase three removes the manual list only after the real graph and the mutated schedules agree. Old plans re-forge by the existing results-set check.

**Acceptance.** The slice is done when the pure tests above pass, the forged plan on hardware is byte-identical to one forged from the manual list, and the diagnostic record lists one supported cache-append report per layer.

**Risks.**

- Naming the vector set in discovery duplicates a fact the executor also knows. Mitigate by reading `GemmaSSA.vectorNames`, not a private table.
- A future axis with per-geometry eligibility, such as parallel attention, cannot use the all-or-nothing coverage rule. That is out of scope here and must be designed with its runtime fallback counter.
- `RMSStatisticsPlan` source was not supplied, so interface alignment is inferred from call sites.

**Next task.** Implement the pure discovery type and the six tests above, in shadow mode only, with no change to the forge selection path.
