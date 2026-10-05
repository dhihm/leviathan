# Original-pipeline input-prefix diagnosis: phase 65

The unchanged original kernels reproduce the frozen one-row mismatch. Across 65 predeclared derived activation masks, only prefixes 62 and 64 have different output bits. **This localizes input sensitivity, not an observed original accumulator or compiled instruction cause.** No candidate, gate or production kernel was changed.

The production control and phase-62 register source were each compiled once with options nil, without trace stores. The original `min-1x64` weights, pipeline objects, buffers and launch geometry were retained. Prefix k keeps original activations for the first k terms in the production source order and fills the rest with positive zero. Indices are 0,16,1,17,...,15,31,32,48,...,47,63. Only prefix 64 is the unchanged full input; the other probes are explicit input perturbations, never replacement correctness fixtures.

## Actual result

The in-session original baseline was control `bd6b6117` versus candidate `bd6b6116`, matching the phase-64 record. Prefix 64 reproduced both baseline bits. All 65 probes were finite, guard-intact and inside the unchanged FP64 1e-4 contract. The explicit prefix oracle agreed with the masked-input oracle. No signed-zero-only difference was found; the raw and zero-pair-excluded first differing prefix are both 62.

| Prefix | Control bits | Candidate bits | Outcome |
| --- | --- | --- | --- |
| 61 | bda69d7c | bda69d7c | Last prefix before the first difference |
| 62 | bd6367fd | bd6367fc | First masked-input difference |
| 63 | bd900ac8 | bd900ac8 | Difference is not monotonic |
| 64 | bd6b6117 | bd6b6116 | Full input matches the baseline |

Including term index 61, block 1 byte 14 high nibble, first exposes the difference on this mask series. It is not a direct measurement of that term's original product or running sum. Under compiler reassociation, changing the suffix activations changes which terms combine. The result cannot establish where an unmodified full-input accumulator first differs. Four serial CPU association/FMA hypotheses matched the candidate at all 65 prefixes and the control at 63; consistency with those hypotheses is not an instruction-sequence proof.

The actual Apple M1 Max run completed 132 command buffers, exited 0, emitted END and released both locks in 0.280227 supervisor seconds. Exit 0 means diagnostic integrity checks passed, not that phase 62 became bit-correct or a performance gain was measured. Actual executable SHA-256: `19c55a49c6256b700753a8f120b57f2b294574bfd8a75633cbfd0ba4cc0d7b98`. The frozen executable is beside the local receipt. [Raw evidence](evidence/q4-original-prefix-phase65/) preserves every prefix and the full native receipt.

Independent Opus review required payload-bound preservation, no candidate-buffer reads after timeout, infrastructure-failure status handling, signed-zero reporting and restricted predecessor/one-hot wording before the first GPU run. All seven exact patches were applied. Conditional oracle-access compiler patches were unnecessary because the actual build passed. The reviewed release build passed, all 2,056 existing regression checks passed after the debug CLI refresh with actual exit 0 and END, pure checks passed 18 + 18 + 22 + 29 + 40, and EOF reached READY, exited 1 and created no Metal device or report.

The delayed Opus review of phase 64 was separately collected and requires a hardened version with host unpack expectations, unique original-kernel mirror checks and stronger integrity gates. Its original v1 evidence is preserved and is not retagged as hardened qualification. That follow-up is diagnosis only, not a repair candidate. No offline Metal tools, full Xcode or Metal toolchain are installed at the checked local paths; no compiler instruction inspection was claimed or downloaded.

No GGUF/model file, model hash, download, weight upload, timing comparison, full-model qualification or default promotion occurred. All five claim flags remain false. The Ollama performance goal remains unmet and the existing normal kernel is retained.
