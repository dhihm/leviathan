# Experimental Q4 launch qualification

This branch measures a bounded launch-geometry pass for the existing Metal
`g4_q4_matvec` kernel. It changes the threadgroup width, preserves the compiled
kernel and its serial per-row arithmetic, and selects only after correctness
gates and paired timing. It does not implement cooperative Q4 staging, change
main, or enable a serving default. Normal generation does not automatically
load a Q4 launch artifact.

The experiment used an Apple M1 Max with 32 GPU cores and 32 GiB unified memory,
macOS 26.2 and Swift 6.2.1. The existing Gemma4 26B A4B Q4_0 GGUF occupies
14,439,363,584 bytes; no weights were downloaded for this qualification. Its
SHA-256 is `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
The model is a heavy experimental target, not evidence of a lightweight engine.

## What passed

The exact seven-file source snapshot passed 778 default checks and 41 standalone
synthetic checks. The synthetic qualifier includes deliberately unwritten rows,
33/1,000-row tails, required versus shared K/V metadata, buffer bounds and
illegal geometry. Real component checks covered all output rows of 20 sampled
matrices at widths 32, 64, 128, 256 and 1,024: finite and bit-identical to the
incumbent. Independent FP64 checks on the first 64 rows per real shape had worst
relative error `9.537770927018384e-7`, below `1e-4`.

Selection used E4_R32 in both arms, the same binary and SSA settings, three
canonical fixtures (English chat, Korean chat and raw Hello), 32 generated
tokens each, and all 262,144 logits per step. Width 64 passed with relative
error zero. A separate process loaded the persisted composite artifact through
the public installer and passed the same three-fixture gate with zero error,
complete geometry counters and no fallback.

For three paired repetitions, each arm timed 93 decode forwards: the median
wall totals were 37.836656 seconds for E4 with incumbent Q4 geometry and
32.105660 seconds for E4 with width 64, a 15.15% reduction. The fixed selection
policy, which uses sums of per-fixture token medians, recorded 14.71% reduction.
These are different summaries of the same records, not interchangeable metrics.

A separate warm raw-Hello measurement excluded BOS prefill and timed 31 actual
decode forwards after the first generated token. Five paired repetitions,
with two discarded arm warmups and identical 32-token output IDs, gave median
wall totals of 10.856017 versus 8.273775 seconds: 2.856 versus 3.747 tokens/s,
a 23.79% reduction in median wall totals. All five pairs favored width 64.
This single-fixture result must not be mixed with the three-fixture aggregate.

There is no claim of Ollama superiority. Historical same-GGUF Ollama results
were around 63–65 tokens/s with first-to-last normalization; server
`eval_duration` and synchronous SSA adapter wall time have different scopes and
were collected at different times. A large performance gap remains.

## Current-binary E4 regression check

The existing `moe-lower select` command separately compared the current binary's
E4_R32 against the original serial MoE path, using the same English chat,
Korean chat and raw-Hello fixtures with 32 actual steps and full-vocabulary
finite/greedy/relative-error checks. E4_R32 passed all three fixtures with worst
relative error zero, 25,020 generated stage dispatches and no fallback. The
re-emitted E4 source SHA matches the pinned source used by Q4 qualification.
A separate process publicly reloaded the new E4_R32 artifact and repeated
the serial-incumbent comparison: all three 32-step fixtures passed with worst
relative error zero and exact stage/merge counters, no fallback, and exit zero.
This closes the common-E4 regression gap for these bounded histories; it does
not establish independent full-model semantic quality.

## Reproduce after rebuilding

Run from the repository root on an Apple Silicon Mac. Supply your separately
obtained, appropriately licensed model at the relative path below. Commands do
not download weights. Keep other GPU inference jobs idle during qualification.
Use Python with the project's existing PyTorch dependency when regenerating the
region; the supplied input JSON files preserve the exact measured graph inputs.

```sh
swift build
swift build -c release
.build/release/leviathan-tests

swiftc -O -I .build/release/Modules \
  tools/q4_qualification/main.swift \
  .build/release/LeviathanCore.build/*.swift.o \
  -o .build/release/q4-qualification
.build/release/q4-qualification

model='models/gemma-4-26B_q4_0-it.gguf'
region='docs/evidence/q4-launch-qualification/moe-region.json'
graph='docs/evidence/q4-launch-qualification/gemma-graph.json'
mkdir -p .build/qualification

.build/release/q4-qualification --component --gguf "$model" \
  --out .build/qualification/component.json

.build/release/leviathan-cli q4-lower launch-select \
  --region "$region" --graph "$graph" --gguf "$model" \
  --out .build/qualification/q4-launch-new.json

# Separate process using the same unchanged release executable.
.build/release/leviathan-cli q4-lower launch-verify-production \
  --artifact .build/qualification/q4-launch-new.json \
  --region "$region" --graph "$graph" --gguf "$model"
```

Require exit zero and `Q4LAUNCH_VERIFY_PRODUCTION_OK`. Selection completion
alone is insufficient: no-win records can be persisted but cannot install.
A new run can choose a different width. Use a new output filename for each
attempt; smoke/no-win output can overwrite an existing qualified artifact if
pointed at the same path. The standalone qualifier is not run automatically by
the default test executable. The debug build is needed by existing CLI tests.

For a current-binary serial-incumbent comparison, the existing bounded MoE
selector and separate verifier are available:

```sh
.build/release/leviathan-cli moe-lower select \
  --region "$region" --graph "$graph" --gguf "$model" \
  --out .build/qualification/moe-new.json
.build/release/leviathan-cli moe-lower verify-production \
  --artifact .build/qualification/moe-new.json \
  --graph "$graph" --gguf "$model"
```

Inspect the emitted schedule and candidate gates: component screening can
prune E4_R32. A verifier for another selected schedule is not an E4_R32 proof.
Only a policy-selected production record can pass the public verifier.

## Identity and limitations

Public Q4 installation validates the canonical artifact body digest, schema,
codegen, actual hardware profile, actual supplied graph digest, complete
embedded primitive MSL SHA, re-emitted E4_R32 source SHA, toolchain identifier,
GGUF digest and running executable bytes. It compares a fresh model binding,
compiles E4 kernels on the runner's device and installs E4 plus width coherently;
an installation error restores prior MoE/width state. Rebuilt executable bytes
invalidate old artifacts. The historical saved artifact is evidence only and
is not distributed as an installable artifact for another binary.

Width 64 changes only the generic Q4 dispatch geometry. It does not select an
old Q4 shader or import a cached external kernel. Generated/tiled Q4 paths are
outside the allowlist. The artifact body hash is integrity checking, not an
author signature.

Qualification is limited to fresh runners and the recorded short fixtures.
The library API accepts a caller-provided GGUF digest and does not independently
tie that argument to the runner's mapped file. The selector also accepts the
runner and GGUF URL separately. The measured CLI used the same verified file
for both. Other preinstalled RMS/Q4 lowerings and diagnostic axes are not bound
as a full stack, and selection does not preserve a previous MoE/width install.
Do not generalize the measured result to a reused or mixed-stack runner.
Persisted timing records lack per-arm geometry counters; correctness replays
and the separate Hello measurement provide those checks for this experiment.
No occupancy, bandwidth, long-context, multi-device or concurrent-serving claim
is made. Independent full-model semantic quality remains outside these gates.

The Python frontend strictly captures one PyTorch routed MoE region. The full
GGUF graph input is a restricted stage-list DSL. This is not an arbitrary
whole-Python-model compiler. Runtime schedule selection uses bounded local
measurements without an agent, but the engine remains experimental and far from
competitive serving performance.

## Authorship and evidence

Existing phase14/15 handoff records attribute those accumulated changes to GLM;
an exact native model/session/patch chain for that portion was not independently
recovered. Earlier phase13 integration records do not certify later authorship.
Phase16 contains 11 exact replacements and an original standalone qualifier
written by `claude-opus-5-5-high`, mechanically applied and checked by Codex.
The author response SHA-256 is
`9966242674731bbb286497e2f8f890e63a2d3e3bacdb7fe894c40df2e175926c` and patch SHA-256
is `fad0caf607bd07e3dbb7fad86da0e6ade4cb4b4e98ab6d390db12f84940a569c`.
Codex prepared measurement orchestration, documentation and evidence, with no
additional core implementation. No further optimization was authored here.

[Machine-readable results](evidence/q4-launch-qualification/results.json),
[synthetic checks](evidence/q4-launch-qualification/synthetic-checks.json) and
[default test output](evidence/q4-launch-qualification/default-tests.log) contain
no weights, binaries, private machine paths, external-author prompts,
credentials or conversation identifiers.
