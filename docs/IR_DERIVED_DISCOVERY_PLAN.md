# Bounded IR-derived candidate discovery

The current Gemma forge uses a manually assembled list of execution flag tuples. The next step is a pure opportunity-discovery boundary that explains candidate admissibility from validated IR, beginning with the existing GPU cache-append copy transformation. It does not add another backend, support another model family, replace the entire engine, or claim a speed gain.

This is the original design. The pure shadow slice is now implemented; see [the implementation and verification record](IR_DISCOVERY_SHADOW_IMPLEMENTATION.md). Runtime candidate discovery remains inactive. Fable 5.1 reviewed the original source, corrected the initial proposal, and confirmed the hardware-profile and cache-copy source facts. The final implementation brief and model receipts are preserved in `docs/evidence/ir-discovery-design-phase42`.

## Existing evidence

| Source | Existing capability and remaining gap |
| --- | --- |
| `GemmaForge.candidates()` in `GemmaForge.swift` | Manual bounded execution tuples remain the runtime authority. There is no general opportunity report explaining why each axis applies to this SSA. |
| `GemmaSSA.lower` and `GemmaSSA.validate` | Versioned operands, kinds, f32 dtype, lengths, slice offsets and per-op signatures are already checked. Cache append validates matching lengths and state output, but does not establish the exact input binding order used by the executor. |
| `RMSStatisticsPlan.analyze` and `GemmaPreparedExecution.prepare` | Reusable RMS statistics are already derived from immutable SSA inputs, exact attributes, epsilon and conservative effects. This analysis must remain intact; it is not missing functionality. |
| `DAGGraph` and `DAGForge.legalCandidates` | The separate restricted DAG already derives safe dense-ReLU fusion from consumer evidence and filters candidate geometry. The Python frontend already imports supported torch.export linear/relu/add graphs. Neither needs rebuilding for this slice. |
| `Gemma4Runner.gpuCacheAppendCopy` | Existing uint32 bit copy binds whole buffers at offset zero and passes width and destination word offset as UInt32. Actual position, spans, buffer lengths and pipeline legality remain runtime concerns. |
| `ForgeHardwareProfile` | Plain measured Codable value; there is no existing validator to call. |

## First supported transformation and pure boundary

Proposed `GemmaOpportunityDiscovery.discover` accepts SSA, the `GraphConfig` from an already validated/lowered graph, and the measured profile. It performs no file access, runner mutation, Metal calls or model loading. It invokes the existing pure `GemmaSSA.validate` itself. It does not call the GGUF-dependent `GemmaGraph.validate`.

New discovery-owned sanity checks are explicit: positive layer count and local/global KV heads and dimensions, global layer membership inside the canonical range, nonempty device name, positive execution width/max threads/working set, and execution width no larger than max threads. These are new input invariants, not a nonexistent public validator. No fixed hardware cap or occupancy estimate is introduced.

For each `cache_append`, prove the static subset of the existing copy contract:

1. A present legal layer, with one proven append per canonical layer.
2. Exactly two live operands named **t2 then kv**, both vector/f32, each with zero or absent slice offset, matching the whole-buffer binding ABI.
3. Positive `kvHeads` and `headDim` equal to the supplied layer's local/global config geometry. Their checked product is positive and fits UInt32.
4. Both operand lengths equal that product. The output is the matching layer's kvstate with the same length.

The global axis is supported only if every append is proven and the layer set covers every canonical layer exactly once. Duplicate or missing layers and valid-but-unsupported bindings produce stable reasons with layer, instruction index and operand IDs. Malformed SSA, invalid config sanity or invalid profile sanity throws. Other known operations are outside this axis; unknown malformed IR still fails the existing validator. Unsupported static opportunity is never used to catch compilation or resource errors.

The Codable/Equatable report records analysis version, verdict, ordered evidence and stable reason codes. Canonical sorted-key JSON and the existing SHA-256 implementation produce the report digest from the body **excluding its own digest field**. Include relevant config and profile inputs in the canonical body. No timing, random state, transient runner flags or dictionary iteration order enters the result.

## Shadow migration

The first implementation has no forge call-site integration. `GemmaForge.candidates`, plan creation/validation, engine identity, executor, runner and kernels remain unchanged. Reports and whole-tuple comparisons run in tests or a standalone diagnostic only.

A shadow helper retains the full existing baseline and uses the verdict to admit or withhold only manual tuples carrying `gpuCacheAppend`; all other axes are passed through. On the current canonical graph the derived ordered list must equal the manual list exactly. This is a consistency check against the current policy, not an independent correctness oracle or a completed replacement of the manual generator.

Later activation is a separate reviewed change. It must bind discovery version, report digest and generated candidate-set digest into immutable identity and version the persisted artifact schema, reject old artifacts, and preserve baseline-first bounded enumeration and all numerical/timing policy. Removing the manual list requires independently tested rules for the remaining axes. No Cartesian expansion or automatic new operator support is authorized by this slice.

## Independently testable acceptance

The externally authored implementation task should add one pure discovery type and focused tests in the existing executable test harness. The package has no separate XCTest target; do not invent one just to match a generic review phrase.

- Canonical lowered config: supported, each layer evidenced once, ordered whole-tuple equivalence with the manual list, baseline retained.
- Repeated discovery and Codable roundtrip: equal body, reports and digest; tampered metadata or proof changes the digest.
- Swapped cache operands that still pass SSA validation: unsupported with the exact instruction/operand evidence.
- Valid synthetic schedules with duplicate coverage and missing coverage: unsupported with a layer reason. Build fresh SSA versions and consistent downstream uses; a malformed mutation that merely fails SSA validation does not test the unsupported boundary.
- Internally consistent local geometry on a global layer: unsupported for config mismatch. Include nonzero input slice offsets as another valid-but-unsupported binding case.
- Stale references and malformed config/profile: throw before report generation. Every nonthrowing derived result preserves the baseline.

Use the existing real-model/Metal suite and artifact lifecycle checks when integration changes are later proposed. Pure shadow analysis needs no full performance rerun. Measured artifacts and winners are not required to be byte-identical; deterministic discovery is distinct from variable runtime timing.

## Next external implementation brief

Delegate the pure type and tests to Opus 5.5 or verified GLM 5.3 using the current private main as base. Supply `GemmaSSA`, `GemmaGraph` config definitions, `ForgeHardwareProfile`, the exact copy ABI/geometry guard and candidate tuple definitions. Return original English source with exact patch provenance. Keep compile diagnostics global and fail-closed. Review pure proofs and negative fixtures before enabling any selection path.

The performance target remains unmet. This boundary aims to make optimization decisions reproducible and explainable; it does not promise that cache-append discovery alone will close the gap against another model runner.
