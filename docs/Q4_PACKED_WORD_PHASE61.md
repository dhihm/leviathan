# Register packed-weight decoding: phase 61

The original register-word candidate is **correctness-rejected**. It is isolated from the production runner and Forge defaults. No timing ran and no speed claim follows. The main branch and incumbent kernel remain unchanged.

Two consecutive Q4_0 blocks occupy 36 bytes. The emitter loads nine aligned UInt32 words, reconstructs both FP16 scales and all quantization bytes, then emits the original scalar terms in their source order. It uses one thread per row, width 32, scalar activation loads, no threadgroup storage and no barriers. Plans decline odd block counts, misaligned offsets, unsupported pipeline geometry and spans outside the weight payload. The candidate differs from both the rejected activation vector loads and phase-60 shared weight staging.

## Actual checks

Independent Opus review required three exact patches before the first GPU run: eliminate misleading emitted-comment matches in a pure test, and bound correctness and timing reads by the weight payload rather than the trailing guard. The actual initial and reviewed release builds passed. After refreshing the debug CLI, all 2,056 existing regression checks passed with actual exit 0, END and released supervision lock. Pure checks passed 29 for the descriptor/emitter and 40 for the reused staging helpers. With the existing cached model identity, EOF reached READY and exited 1 without device creation, GGUF mapping or a report.

The first Apple M1 Max component run completed all 35 committed command buffers. Actual exit was 3, END was observed, and both supervision and advisory GPU locks were released. All 17 supported cases contained 2,447 rows; all were finite, inside the unchanged FP64 1e-4 contract and had intact guards. However, 1,368 rows differed in Float bits from the production control. Sixteen supported cases failed; the one-row minimum case and the predeclared odd-block decline passed. Eight real layer-0/5 two-dimensional tensor prefixes contributed 256 rows each. No three-dimensional expert or full-model qualification is implied.

The frozen inputs, seeds, oracle, compiler options and gates were not adapted to this failure. Timing was withheld automatically. Fully unrolled source terms with the same apparent expression order do not establish compiled-bit equivalence; compiler contraction or reassociation is a possible explanation, not an observed instruction-level diagnosis. Load count, bandwidth, cache and occupancy were not measured.

The candidate source SHA-256 is `593f912dd20cf14c0117652484498038fcf3d5f7bbf9f82ed7dc6cd20cf0f957`; the actual executable SHA-256 is `cb13c313069774cade7e2e86c0b430de296f5f096d392b0f18800523b2fd9c6b`. The executable is frozen locally beside the terminal receipt. Full reports and independent review are in [raw evidence](evidence/q4-packed-word-phase61/).

A distinct follow-up will examine whether retaining the production block-loop and inner-loop structure can avoid this compiled arithmetic change. It requires independent review and the same frozen gates; no improvement is assumed. The overall Ollama performance goal remains unmet. No original weight upload, download or model rehash occurred. All full-model, gain, installation, production-performance and busy-time claim flags remain false.
