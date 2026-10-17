# Forge architecture audit — why the current design does not generalize

Written 2026-10-03 (KST; 2026-10-02 UTC), after pass 8
([FORGE_PROGRESS.md](FORGE_PROGRESS.md),
[evidence](evidence/gemma-diagnose-hello32.log)). This is a design
audit only — no redesign implementation was made in this phase (the
phase's actual implementation work was the opt-in diagnostic
subcommand and one bounded candidate-composition entry).

## What exists today

`GemmaForge.candidates()` is a hand-enumerated global list of 13
candidate configurations over a fixed axis set (segment batching,
Q4/Q6/F32 tiled matvec, tiled RMSNorm, vectorized loads). Selection
runs one warm and three timed samples per candidate on one workload,
then a full-logit correctness gate at fullGateTokens=32 over the same
"Hello" fixture. Hardware capability records (core counts, limits)
exist but are inputs to limits only, not to lowering.

The Python frontend accepts restricted `nn.Sequential` Linear/ReLU
exports and a separate trusted Gemma stage-list DSL; it is not a
compiler for arbitrary PyTorch models.

## Why this is not general operator/hardware lowering

1. **Axes, not operators.** Candidates toggle scheduling flags on
   five hand-written kernels. The Gemma path does have a typed SSA
   schedule, but nothing below it: no parameterized operator-level
   lowering derives thread mapping, tiling shape, memory layout, or
   fusion from that IR per operator; a new operator (or a new quant
   format) means new hand-written kernels plus new hand-written
   candidate entries.
2. **No hardware-derived search.** The candidate grid is fixed and
   global, independent of the measured device. Nothing synthesizes
   per-operator parallel strategies (row vs vector vs tile
   decomposition, threadgroup geometry, occupancy-driven splits) from
   model shapes and device capabilities. The retained
   one-thread-per-row Q4 matvec and one-thread-per-vector serial-sum
   RMSNorm are exactly the kind of under-parallelized lowering such a
   search would replace — pass 8 shows the parallel tiled variants
   fail the strict 1e-4 gate on this workload, and the grid offers no
   middle designs (e.g. parallel partial staging with a preserved
   serial final sum — UNTESTED hypothesis, not a proven win).
3. **Selection is not generation.** Automatic candidate SELECTION
   exists and works (gated, measured, fail-closed). Automatic pass
   GENERATION does not: the space of legal transformations is exactly
   what a human enumerated. Adding one more combination (the pass-8
   Q6+VL entry) does not change this.
4. **Workload coupling.** Timing and the 32-token gate both use the
   same Hello fixture, and selection is the FIRST gate-passing
   candidate in the short-workload timing order — not a
   representative-workload acceptance and not a claim that the
   selected candidate is the best gate-passing configuration on any
   other workload. "GPU 84-86%" figures are elapsed GPU command
   durations, not core-occupancy counters; they cannot diagnose
   underutilization.
5. **Numerics bound the search.** The strict 1e-4 full-logit gate
   (deliberately unchanged; pass 8 attributes tiled-candidate
   failures to float32 reduction-order differences consistent with
   reassociation, single-Q4 FP64 sample, no exclusive cause proven)
   excludes reassociated kernels wholesale. A general design would
   need per-operator numerical strategies as part of lowering rather
   than a single post-hoc output gate.

**Assessment:** the joint goal — a general Python model compiler with
competitive performance — is UNMET. What exists is a validated,
gated, evidence-producing executor for one model family with
human-enumerated optimizations.

## Concrete staged redesign (recommendation, not implemented)

1. **Documented opset + frontend.** An explicit operator spec
   (semantics, shapes, precision, quant formats) that the Python
   frontend checks models against, replacing the per-model DSL.
2. **Semantic IR with parameterized lowering.** Each operator carries
   layout/precision/tiling/fusion parameters; backend rules map
   (operator, parameters, device capabilities) to kernel skeletons
   (thread geometry, reduction tree, memory access pattern).
3. **Legal/cost-guided bounded search.** Enumerate candidate
   parameterizations per operator from the rules, prune by a cost
   model seeded from measured hardware, keep the existing bounded
   timing budget.
4. **Independent numerical/quality validation.** Per-operator
   component oracles (FP64 decode references as in pass 8) plus
   representative-workload correctness and quality gates, separated
   from the performance gate, so numerically-sound reassociation is
   not conflated with semantic error.
5. **Measured hot-path acceptance.** Occupancy/utilization counters
   and representative workloads (multiple prompts/lengths) replacing
   the single-fixture decision.

Compiler strategies, rules, and templates remain authored code
throughout; the generalization is in parameterized derivation plus
bounded search, not in unreviewed synthesis.
