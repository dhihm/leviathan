# MoE Routed-Expert Batch Lowering (implemented, pass 13)

This documents the ACTUAL pass-13 implementation and root's measured
evidence. The earlier `MOE_BATCH_LOWERING_DESIGN.md` is kept as the
historical design document (it says "UNIMPLEMENTED" — that was true when
written; this file is the current record).

## What it is — and is not

A pattern-based lowering for ONE region: the existing Gemma 4 26B-A4B
routed MoE expert region (all 30 layers' experts in the approved Q4_0
GGUF). The region is captured from a user-authored PyTorch `nn.Module`
with REAL `torch.export(strict=True)` on META tensors (no weight
allocation, no full float32 shadow — metadata shapes are verified, the
semantics are the actual exported ATen DAG). The Swift importer
independently revalidates the graph, re-derives every dimension, and
treats the canonical digest as identity: frontend fields are never
trusted.

Supported ops, routed-slot region ONLY: no-bias `aten.linear.default`,
`aten.split.Tensor` + `operator.getitem`, `aten.gelu.default`
(approximate=tanh), `aten.mul.Tensor`, `aten.stack.default` (dim 0) over
the topK routed slots. Everything else fails closed with the offending
op named. The host-side router and the SAME 8 original ordered axpy
merges stay OUTSIDE the capture and run unchanged. This is NOT an
arbitrary whole-LLM model.py compiler.

Execution stays on the original Swift/Metal backend — no MLX, no MPS
runner wrapper, no new dependencies.

## Pipeline

1. **Capture (Python, `tools/moe_lower/`)** — user's local module calls
   `build(config)`; the exporter runs `torch.export.export(strict=True)`
   on META tensors, normalizes the ATen DAG, checks the exact pattern
   and census, and emits region JSON with a canonical SHA-256 digest.
   Fail-closed: no files are written on rejection. Default config is the
   real Gemma geometry `{"hidden": 2816, "inter": 704, "topK": 8}`
   (`model_moe_region.py`); the Swift side fail-closed crosschecks every
   dimension against the actual GGUF metadata.
2. **Binding (Swift, `MoEModelBinding.swift`)** — immutable binding to
   the ACTUAL GGUF: every layer's `ffn_gate_up_exps`/`ffn_down_exps`
   tensors verified Q4_0 with exact dims, checked strides/offsets
   (GU stride 2,230,272 B = 1408·88·18; DN 1,115,136 B = 2816·22·18),
   spans bounded by the real
   page-rounded resident mapping (16 KiB Apple Silicon pages), GPU
   scratch = 8×(1408+704+2816)×4 = 157,696 B ≤ 160 KiB budget.
3. **Kernel generation (`MoELowering.swift`)** — hardware-derived
   candidates E∈{1,2,4,8} (experts per chunk, runtime chunk tails) ×
   row-tile R from the measured threadgroup caps; three generated MSL
   kernels per schedule — `moegen_gu_E{E}_R{R}` and
   `moegen_dn_E{E}_R{R}` (row-tiled) and `moegen_act_E{E}` (elementwise,
   no R suffix) — whose per-row serial
   arithmetic is EXACTLY the incumbent `g4_q4_matvec`'s and whose
   activation helper is the LITERAL incumbent source text
   (`g4_tanh`/`g4_gelu_up_value`) extracted from the runner's shader —
   never retyped.
4. **Component gates** — every candidate is bit-Pattern-identical to the
   ACTUAL incumbent kernels on ALL topK (8) slots; independently, both
   sit inside a bounded FP64 oracle contract evaluated on a bounded
   prefix (first 64 GU/DN rows over the first 2 slots, and the ACT
   stage over the first 2 slots) — the oracle bound is component- and
   prefix-bounded, not a whole-model proof.
5. **Model selection (`moe-lower select`)** — EVERY arm (reference,
   gates, timing) runs the SAME compiled Forge SSA execution path:
   strict 3 fixtures × 32 teacher-forced steps, all 262,144 logits per
   step gated at absdiff/max(1,|ref|) < 1e-4, exact greedy argmax, all
   finite; 3 paired AB/BA repetitions × 31 matched decode steps; fixed
   2% policy recomputed from the recorded arms; per-stage
   GU/ACT/DN dispatch and ordered-merge and fallback DELTA proofs.
   Exact-32 references are required — early EOS fails the selection.
6. **Artifact + fresh process** — the artifact binds model, hardware,
   toolchain, executable, source, graph, schedule, and gate-policy
   identities. Integrity is local-trust (digests), not authentication:
   a stale executable or any mismatch requires fresh gates — no
   rebinding. `moe-lower verify-production` is a SEPARATELY LAUNCHED
   process that revalidates identity, generates an unoptimized incumbent
   SSA reference (same engine primitives — not an independent semantic
   proof), publicly installs the artifact, and replays the full 3×32
   strict gates with exact workload digests and dispatch counters.
   A `--smoke-test` record is diagnostic plumbing only and can NEVER
   install. Default serving configuration is NOT automatically changed;
   the eligible public install path is the saved artifact + runtime,
   demonstrated via the actual CLI.

No runtime agents, no unbounded search — the candidate set is derived
from the graph and measured device profile.

## Reproducible commands

```bash
# 0. Build (release; PyTorch is frontend-only, backend is Swift/Metal):
swift build -c release

# 1. Capture the routed region (default config = Gemma 2816/704/topK 8):
python3 tools/moe_lower/export_moe_region.py \
    tools/moe_lower/model_moe_region.py /tmp/moe-region.json
# Optional custom geometry: a THIRD positional argument (after the
# module and output paths) — a JSON file with
# {"hidden": …, "inter": …, "topK": …}.

# 2. Gemma graph (existing trusted stage-list DSL, unchanged):
.build/release/leviathan-cli gemma-config-export --file <gguf> --out /tmp/cfg.json
python3 tools/forge_gemma/export_gemma_graph.py examples/gemma_model.py /tmp/cfg.json /tmp/graph.json

# 3. Production selection: 3×32 strict full-logit/greedy/finite gates on
#    the same Forge SSA path for every arm, 3 paired AB/BA × 31 matched
#    timing, fixed 2% policy, persisted artifact + reload/install proof:
.build/release/leviathan-cli moe-lower select \
    --region /tmp/moe-region.json --gguf <gguf> \
    --graph /tmp/graph.json --out /tmp/moe-selected.json

# 4. Fresh-process verification (separately launched; full 3×32 gates,
#    workload digests, dispatch counters; public install):
.build/release/leviathan-cli moe-lower verify-production \
    --artifact /tmp/moe-selected.json --gguf <gguf> \
    --graph /tmp/graph.json

# Optional plumbing smoke (internal trial record; NEVER installs):
.build/release/leviathan-cli moe-lower select \
    --region /tmp/moe-region.json --gguf <gguf> \
    --graph /tmp/graph.json --out /tmp/moe-smoke.json --smoke-test
```

Tests: `swift run leviathan-tests` (713 passed / 0 failed, includes GPU
bit-identity of generated E×R tail kernels vs the incumbent kernels and
the pre-refactor-vs-factored GELU GPU bit gate) and
`python3 tools/moe_lower/test_moe_export.py` (41 passed / 0 failed).

## Root's measured evidence (2026-10-03, release binary)

Raw logs and the numeric summary are copied verbatim in
[docs/evidence/pass13-moe/](evidence/pass13-moe/). Identity: executable
SHA-256 `451b16b2…dd950`, artifact `931deb8e…4a610`, model
`3eca3b8f…a51d` (the approved existing 14,439,363,584 B GGUF; no
downloads). Sources unchanged across both runs.

- **Component forge**: 20/20 E∈{1,2,4,8} × R∈{32,64,128,256,1024}
  candidates bit-identical to the ACTUAL incumbent kernels; bounded
  FP64 worst relative 6.00147318248e-07.
- **Selection** (`moe-lower select`, EXIT 0, 802.543 s): both finalists
  (E4_R32, E4_R64) passed all 3 canonical prompts × 32 steps × all
  262,144 logits with worst relative 0, exact greedy, all finite;
  25,020 batched stage dispatches, 0 fallbacks each. Winner **E4_R32**:
  6 batched GU/ACT/DN math dispatches per layer/token vs 24 incumbent
  (18 fewer — a count claim; the winner is E4, not E8).
- **Performance (same SSA path, both arms)**: fixed 2% policy metric —
  21.9388542098% reduction in the median summed per-fixture wall median
  (ratio 1.281047). A DIFFERENT, correctly-labeled aggregate figure:
  93 decode steps' whole-wall median TPS 3.007 vs 2.342 incumbent —
  this is an aggregate median, not the inverse-median policy metric.
  The earlier tiny-MLP microsecond numbers are NOT LLM performance and
  are not added to these gains.
- **Same-process reload proof**: disk artifact reloaded, validated,
  publicly installed; en-chat replay worst rel 0; stage deltas
  3120/3120/3120 (expected 3120 each); ordered merges 12,480; 0 fallbacks.
- **Fresh process** (`verify-production`, separately launched, EXIT 0,
  133.650 s): all 3×32 gates worst rel 0; en-chat 3120/3120/3120,
  merges 12,480; kr-chat 3240/3240/3240, merges 12,960; raw-hello
  1980/1980/1980, merges 7,920; fallbacks 0 everywhere; recorded
  workload digests checked.
- **Resources** (observed job including selector, component gates,
  references, and compiler — NOT physical GPU occupancy): selector peak
  footprint 31,265,588,544 B / maxRSS 3,641,393,152 B; fresh-process
  peak footprint 16,631,420,736 B.

## Honest scope

The 14.4 GB Q4_0 target is NOT lightweight. The 21.9% win is incremental
vs this engine's own incumbent path — it is NOT a comparison to Ollama's
historical ~66 tok/s reference figure (kept only as history, not a fair
comparison). Whole-model quality shares the engine's own primitives;
independence is the bounded FP64 component oracle. The overall goals —
general PyTorch frontend for arbitrary models, competitive serving, and
a lightweight target — remain UNMET and active.
