# Phase 9 design — parameterized Q4_0 operator lowering (one operator family)

Written 2026-10-03 UTC (initially drafted 2026-10-03 KST / 2026-10-02 UTC
and corrected before the measured runs), before implementation. This is the
concrete plan responding to [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md)
stage 1–4 for exactly ONE operator family: the Q4_0 quantized matrix–vector
product (`y[row] = Σ_col W[row,col]·x[col]`, W in Q4_0 blocks), the dominant
Gemma 4 matvec. It is NOT a claim of a general arbitrary-Python compiler.

## 1. Operation input (Python, real dataflow)

`tools/q4_lower/q4_op_model.py` (stdlib only) defines a `Q4MatvecOp` with the
metadata the audit demands — semantics, shape, layout, quant, precision:

- `op = "q4_0.matvec"`, `semantics = "dot(W_row, x)"`,
- `inDim`, `outDim` (must satisfy `inDim % 32 == 0`, bounded by the existing
  `ModelWorkloadLimits`),
- `layout = "row-major-q4_0-blocks"` (18-byte blocks, FP16 delta + 32 nibbles,
  biased by −8),
- `quant = "q4_0"`, `accumulate = "f32"`, `inputPrecision = "f32"`,
- optional `tensor` binding (a real GGUF tensor name used for representative
  weight bytes in component validation).

A small export script executes the user's `.py` and emits op JSON — the same
trusted-file capture pattern as `tools/forge_gemma/export_gemma_graph.py`.
The Swift side re-validates every field; nothing outside the supported op is
accepted.

## 2. Operation IR (Swift)

`Q4OpIR` (new, `Sources/LeviathanCore/Q4Lowering.swift`): decodes the JSON,
validates the op name, dims, block geometry, and precision tags against the
Q4_0 constants already used by the reference kernels. Digest
(SHA-256 over canonical JSON) becomes an artifact identity input.

## 3. Schedule derivation (shape × measured hardware — not booleans)

`Q4Schedule` parameters, each with a REAL effect on emitted code:

- `threadsPerRow T` — threads cooperating on one row (1 reproduces the
  incumbent one-thread-per-row mapping and its serial summation order);
- `rowsPerGroup R` — rows per threadgroup (grid/threadgroup geometry);
- `accum a ∈ {f32, f32compensated}` — per-thread partial accumulation and
  ordered ascending-thread combine, both in float32, compiled with the
  same default Metal compiler options as the incumbent kernels. Metal on
  Apple GPUs has no usable FP64 path, so NO GPU double strategy exists
  here; the FP64 reference is a CPU-ONLY oracle (below). `f32compensated`
  carries a Neumaier compensation term through the partial accumulation —
  a documented deterministic reduction order that is NOT bit-identity
  with the incumbent, measured against the FP64 oracle (an accuracy
  IMPROVEMENT is measured per-case, not proven for all inputs);
- `xLoads ∈ {scalar, float4}` — activation load width in the emitted inner
  loop.

`Q4ScheduleDeriver.derive(op, hardware)` enumerates the legal space from the
operation SHAPE and the MEASURED device (`threadExecutionWidth w`,
`maxTotalThreadsPerThreadgroup M`): T ∈ {1} ∪ {2,4,8,16,32} with `T·R ≤ M`
and `blocksPerRow/T ≥ 4` — the block count per row (shape-derived
`inDim/32`) is a hard legality input: small geometries legally admit only
small T, and the emitted chunk loop is generated FROM that count. R ∈
{1,2,4}. Pruning keeps at most 8 candidates: the T=1 baseline (always
candidate 0 — the stable fallback), three R=1 combinations (plain f32,
compensated scalar, compensated float4) for EACH of the two
execution-width-nearest legal T values, and one R=2 cooperative row-group
entry on the width-nearest T. The measured working-set size is a profile
input only — no occupancy or utilization claim is derived from it. No giant buffers: the only new
allocation is a threadgroup partials array sized T·R floats, bounded by M.

The op's Python-declared shape is validated against the BOUND GGUF tensor's
actual dimensions when `tensor` is set (e.g. attn_q reads inDim 2816 —
hidden size — in this checkpoint); a mismatch is rejected before any
lowering.

## 4. Metal source emission

`Q4KernelEmitter.emit(schedule)` generates MSL source from the parameters —
thread mapping, block-chunk striding, reduction tree, and load width are
EMITTED text derived from (T, R, a, xLoads) and the op shape, not a lookup
table of prewritten kernels. Kernel name
`q4gen_matvec_T{T}_R{R}_{a}_{xLoads}`; compiled with
`device.makeLibrary(source:)` exactly like the incumbent path. Emission is
deterministic; the source SHA-256 is recorded in the artifact.

## 5. Independent FP64 oracle (CPU-only, component gate)

`Q4Oracle.matvecFP64` — independent Swift Double dequant/dot (FP16 scale
decoded via Float16→Double, nibble levels in Double) over REAL weight bytes
of the bound GGUF tensor (or synthetic deterministic blocks when unbound) and
a deterministic seeded input. Scope as MEASURED: the run validated and
oracle-checked only the 2-D bound tensor blk.0.attn_q.weight (256 rows of
[2816, 4096]) with the seeded input, plus synthetic unbound geometries in
the test suite; 3-D expert-slice bindings are NOT supported by this
family. Every generated candidate AND the incumbent kernel are compared
to the same CPU FP64 oracle. The full-model oracle shares the incumbent
kernel primitives — the model-level gate is a bounded 32-step,
three-fixture equivalence proxy, not an independent full-model
reference.

Component acceptance uses the EXISTING contract, unchanged — the same one
the current Q4/Q6 kernels are held to in the test suite: finite outputs and
relative-to-FP64 `|gpu−ref64| / max(1, |ref64|)) < 1e-4`. No new or looser
numerical policy is introduced; the per-candidate rel-to-FP64 values and the
incumbent's own value are recorded as diagnostics (the incumbent f32 serial
kernel is measured, not assumed exact).

## 6. Artifact identity and persisted execution

`Q4LoweringArtifact` (JSON, ≤ 1 MiB): codegen version, op digest, schedule
parameters, EMITTED SOURCE + its SHA-256, derived hardware profile, toolchain,
GGUF content digest, executable digest, gate records (component + model), and
an artifact digest over all fields. Reuse re-emits from the recorded
schedule, requires byte-identical source, and RECOMPILES the MSL on load
(the artifact is source/schedule JSON, not a compiled binary archive;
tamper/stale → reject, fail closed to the incumbent kernel). Execution
genuinely dispatches the emitted, compiled pipeline — asserted by the
dedicated generated-kernel dispatch counters (> 0 on the generated path
for the bound tensor; the incumbent kernel continues to serve every
other Q4 matvec).

## 7. Model-level gates (unchanged strictness) and matched measurement

One-process orchestration (`q4-lower select`, mirrors the
gemma-diagnose teacher-forcing pattern): the incumbent reference
(pass-7 selected batching+RMS-VL configuration) generates greedy
histories per fixture; EVERY component-accepted candidate is then
teacher-forced replayed on the SAME histories over length-32
generations on bounded representative fixtures (canonical-template
English AND Korean chat prompts plus the raw Hello continuity
fixture), gating full 262144 logits at the UNCHANGED strict 1e-4
relative formula + exact greedy token equality + finiteness at every
step. Matched performance: same binary, same GGUF, same prompt,
sequential GPU residency, THREE paired repetitions per passing
candidate with alternating incumbent-first/candidate-first order,
raw per-arm samples and actual history windows recorded in the
artifact; selection only beyond the documented 2% policy threshold.
If the generated path fails the gate or shows no win, that is
REPORTED HONESTLY; the incumbent remains the stable fallback and the
artifact records the failure. No claim of a win from microbenchmarks
alone, and the measured paired repetitions can be (and in the actual
run, were) sign-changing — a median-based selection is not a claim of
stable speedup.

## 8. Bounds

≤ 8 emitted candidates per forge; component validation reads bounded weight
slices (≤ a few MB per shape, never the whole matrix set resident twice);
model gates bounded to the existing 4096-context limit with 2...64
generation tokens accepted by the CLI (the measured run used 32
generation / 31 timed tokens); no new dependencies (Python stdlib
only; Swift stdlib + Metal + CryptoKit only).

## 9. Honest scope statement

This adds ONE real parameterized operator-level lowering family (Q4_0 matvec)
with shape/hardware-derived schedule generation, independent numerical
validation, artifact-bound persisted execution, and matched measurement. The
broader joint goal (general Python model compiler, competitive performance)
remains UNMET until demonstrated; nothing here asserts it.

## 10. Pass-11 as-built update — expert 3-D view, row-tile geometry,
codegen v2, screen-first funnel (2026-10-03 UTC)

Sections 1–9 above are the phase-9 design as written then; this section
records what phase 11 changed in the ACTUAL code (frozen executable
SHA-256
8fdb03f5ec85b919e5a5a8358531fe502f0026b428f305deb698cad153f207c6):

- Operation input / IR: `Q4MatvecOp` and `Q4OpIR` gained a `view` field;
  `view = "expert-row-slice"` binds a 3-D expert tensor family
  (`.ffn_gate_up_exps.weight` / `.ffn_down_exps.weight`) whose declared
  shape is ONE expert's [outDim, inDim] slice, validated against the
  GGUF's actual 3-D dims [inDim, outDim, experts] by
  `op.validateAgainst(file:)`. The Python models
  `tools/q4_lower/model_q4_expert_{gu,dn}.py` capture the two families
  (GU [2816→1408], DN [704→2816]). 2-D bindings validate exactly as in
  phase 9 (section 2).
- Schedule space: `rowsPerGroup` legality widened to powers of two
  ≤ 1024; the deriver emits up to 13 candidates — the T1R1 baseline,
  the execution-width-nearest cooperative T ladder × {f32 scalar,
  compensated scalar, compensated float4}, one R=2 cooperative entry,
  and the T=1 row tiles [w, 2w, 4w, 8w, min(1024, cap)] derived from
  the measured execution width w = 32 with the 1024 threadgroup cap as
  a measured control.
- Emission (codegen "q4gen-v2"): T=1 emits a direct row-tile kernel —
  `row = threadgroup_position * R + thread_index`, one thread per row,
  direct `y[row] = sum;` (or `sum + comp` compensated) store, no
  threadgroup partials or barriers — with the incumbent's per-row
  serial term order (ascending blocks, low then high nibble) unchanged.
  T>1 keeps the phase-9 partials+barrier cooperative structure.
- Component gates: T=1/f32/scalar row-tile candidates additionally
  REQUIRE measured bitPattern identity to the ACTUAL incumbent
  `g4_q4_matvec` output; every candidate keeps the section-5 FP64
  oracle 1e-4 contract. The section-5 statement that expert-slice
  bindings are unsupported is superseded by the expert view.
- Selection funnel: a component timing screen at the REAL hot-row count
  (outDim rows; bounded few-MB weight slice) runs BEFORE the model
  gates; the T1R1 baseline is excluded from finalists and at most the
  2 fastest candidates reach the full section-7 gates (3 fixtures × 32
  steps, unchanged strictness). Pruned candidates are recorded as a
  screen prune, not a gate failure. `--smoke-test` is a raw-fixture
  plumbing mode (2 gen / 1 timed) only.
- Runtime: install validates the actual GGUF rank; a pure predicate
  (`Gemma4Runner.q4GeneratedServesTensor`) serves the bound tensor AND
  every same-family, same-shape expert slice in all 30 layers; all
  other Q4 matvecs keep the incumbent kernels. The runner holds ONE Q4
  generated binding at a time — the measured GU and DN artifacts are
  SEPARATE single bindings; combined/composed execution (GU + DN + RMS
  in one process) is NOT implemented, and the measured gains are not
  additive-claimable. Measured outcomes: pass 11 of
  [FORGE_PROGRESS.md](FORGE_PROGRESS.md).
