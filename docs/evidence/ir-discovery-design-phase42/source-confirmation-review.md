Implementation brief: shadow-only cache_append opportunity discovery, corrected against the sources now read.

## Confirmed source facts

- **Hardware profile.** `ForgeHardwareProfile` is a plain Codable value with four fields and no validator. Discovery must not call one.
- **Graph validation is not pure.** `GemmaGraph.validate` takes a GGUF file. Discovery never calls it and never takes a file.
- **SSA validation is pure.** `GemmaSSA.validate` takes the SSA and a `GraphConfig`. Discovery calls it itself.
- **Copy kernel contract.** `gpuCacheAppendCopy` binds all four buffers at offset zero. It passes the word count and the word offset as UInt32 values. The width is kvHeads times headDim. Position, byte spans and destination lengths are checked at runtime by the geometry predicate.
- **Candidate type.** `GemmaForgeCandidate` carries the gpuCacheAppend flag as an explicit Codable key with all other axes. Equivalence compares whole candidate values.

## Discovery contract

**Inputs.** The SSA, the `GraphConfig` taken from an already validated and lowered graph, and the hardware profile. No runner, Metal, GGUF or file access.

**Global checks, all throwing.**
- SSA fails `GemmaSSA.validate` against the supplied config.
- Config sanity, stated as a new pure invariant owned by discovery: positive layer count, positive kv heads and head dim for both local and global geometry, global layer set inside the layer range. This is not a public config validator and is not exported as one.
- Profile sanity, stated as a new explicit input contract: non-empty device name, positive execution width, positive max threads, positive working set, execution width at most max threads. No fixed cap such as 1024. Actual pipeline legality stays a runtime check in dispatch.

**Per-instruction proof for cache_append.**
1. Layer present and in range.
2. Exactly two inputs resolving to the live versions of `t2` then `kv`, in that order. Both vector kind, f32, slice offset nil or zero.
3. kvHeads and headDim attributes present, positive, and equal to the config geometry chosen by global membership.
4. Both input lengths equal the overflow-checked product. The product is positive and fits UInt32, because the kernel receives it as a UInt32 word count. Output is kvstate for the same layer with the same length.

**Axis verdict.** Supported only when the layers carrying exactly one proven append form the full canonical layer range. Any failure yields unsupported with a reason naming instruction index, value ids and layer. Baseline is always present.

**Report.** Codable, Equatable, with analysis version, verdict, evidence sorted by instruction index, and a digest computed as `GemmaSSA.digest` does it. No time, randomness or device state. Unsupported ops are skipped, malformed input throws.

## Acceptance

Six pure tests, no kernels, built from the real config through expectedSequence and lower, plus hand-built SSA that passes validation.
1. Real config: supported, each canonical layer once, derived list equals the manual list.
2. Determinism: two runs give equal reports and digests.
3. Swapped inputs: validation passes, unsupported, reason names index and ids.
4. Duplicate append on one layer, and one missing layer: unsupported, reason names the layer.
5. Local geometry on a global layer with consistent lengths: unsupported with a geometry reason.
6. Throwing paths: stale reference, failed config sanity, failed profile sanity. Every non-throwing report contains the baseline.

Integration exclusions. `GemmaForge.candidates`, makePlan, validatePlanIdentity, the executor and the runner are untouched. The manual list stays the runtime authority. Equivalence comparison lives only in tests and an optional standalone diagnostic.

Later activation rule. When discovery becomes active, plan identity must gain the analysis version, report digest and candidate-set digest, and the artifact schema must bump so old artifacts fail on schema and identity, not on candidate membership. Not part of this slice.

Non-claims. No byte-identical timings or plans, since the artifact carries measurements. No independent oracle: the manual list is derived from the same graph, so equivalence is a consistency check, not external proof. The RMS analyzer continues to treat cache_append as opaque.
