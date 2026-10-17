# Cooperative SIMD MoE component prototype

This prototype emits fused routed gate/up/activation kernels and cooperative down-projection kernels from validated MoE shape facts and measured Metal pipeline limits. It is an original correctness experiment. It does not install a production lowering or select a performance winner.

The existing v1 MoE emitter, artifact installer, selected schedule, and production dispatch remain unchanged. The new numerical class is `simd-reassociated-oracle`: lane reductions change floating-point addition order, so agreement with v1 bit patterns is not claimed. The full independent Double oracle uses the unchanged normalized error threshold of 1e-4.

## Actual validation

On Apple M1 Max, the release build passed and the Swift/Metal executable reported 2,168 passed, zero failed. The standalone real-model component screen accepted all 48 derived schedules against four inputs: seeded uniform, signed alternating, one-hot, and a captured layer-0 pre-FFW normalization result. Every expert slot and output row was checked. Repeat execution and E4/E8 batching were bit-identical within each identical kernel configuration.

Worst normalized errors across the screen were 4.0304082726816404e-7 for fused activation, 2.858389812029635e-7 for end-to-end down output, and 2.11719789811049e-7 for isolated down output. The unchanged v1 E4R32 incumbent also passed the same full oracle on all four inputs.

The model capture performs a reference forward pass through the model before releasing that runner. Component allocations then contain only bounded packed expert slices. Neither model weights nor captured activation values are exported in the report. The report uses the previously cached historical model SHA; no model rehash or download occurred.

GPU intervals in this report are diagnostic only. Earlier A/A controls failed under paging, so these times establish neither a production winner nor a gain over Ollama. A complete model gate, normal artifact provenance, prepared-graph binding, and a coordinated quiet performance window remain necessary before production activation.

## Environment recovery

The first restricted test invocation could not access Metal and terminated with `Gemma4Error.metalUnavailable`. The standalone compiler also lacked access to its module cache. Both subsequently succeeded through permission-reviewed native execution; the original failure receipts are retained. The first restricted review invocation could not access the existing CLI authentication. The authenticated Fable 5.1 source review returned ACCEPT with no blockers. It explicitly noted that this non-installable report does not validate all embedded result fields and that its historical paging note is static. Production integration must use stricter artifact validation. The reviewer lacked the runner excerpt for probe detachment; the supplied implementation reads the stored closure anew on each layer, and the real probe test passed.
