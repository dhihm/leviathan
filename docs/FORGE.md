# LeviathanForge: automatic pass generation — current contract

LeviathanForge generates optimized GPU execution passes for a supplied
model **automatically**: no coding agent, no LLM, no per-model hand
tuning at runtime. The pipeline is validate → profile → generate a
bounded candidate set → gate on numerical correctness → measure real GPU
time → select the best valid candidate → cache a reusable plan keyed to
model + weights + hardware + toolchain + workload.

## What runs today (actual, measured on this M1 Max)

End-to-end user workflow with **real PyTorch parameters** (not synthetic
weights):

```bash
# 1. User writes a model file (restricted contract, examples/torch_model.py):
#    build(seed) -> nn.Sequential(nn.Linear..., nn.ReLU...)
python3 examples/export_torch.py examples/torch_model.py /tmp/model 0

# 2. Forge: profile, candidates, correctness gates, measured selection,
#    PyTorch parity check, plan cache:
.build/release/leviathan-cli forge --export-prefix /tmp/model \
    --batch 1 --plan /tmp/model.forge.json

# 3. Re-run: cache hit — the plan validates (model+weights+hardware+
#    toolchain+workload identity) and the selected candidate re-executes
#    through the same correctness and parity gates without re-benchmarking:
.build/release/leviathan-cli forge --export-prefix /tmp/model \
    --batch 1 --plan /tmp/model.forge.json
```

Measured run (Apple M1 Max, 2026-10-02, one bounded micro-graph): device
profile max 1024 threads/threadgroup, execution width 32, recommended
working set 26,800,603,136 bytes. Candidates (deduplicated to the actual
dispatch width 8 by the smallest dense output dim): unfused 187–196 µs
median GPU time, fused dense+ReLU 187.3 µs, 11 timed runs after 3
warmups. The fused pass was selected. PyTorch parity worst relative
difference 2.235e-08 (fresh) / 2.235e-08 (reused). Different model seeds
produce different parameter weights, different reference outputs, and
different plan identities (digest `93ad96b5…` vs `f0cdd8c9…`). This is
one measured micro-graph; no general performance claim is made.

`swift run leviathan-tests` — 163 checks, 0 failures, including the
exporter e2e, parity, cache-hit/stale/tamper rejection, and adversarial
input cases.

## Pipeline stages (as implemented)

1. **Model input.** Two frontends, both explicit user actions:
   - Restricted **PyTorch exporter** (`examples/export_torch.py`):
     `nn.Sequential` of exactly `nn.Linear` (float32, CPU, biased) and
     `nn.ReLU`. Subclasses, other module types, hooks, parametrizations,
     shared modules/parameters, non-finite weights, bool dims, >64 ops,
     >16 MiB weights, and inconsistent shapes are rejected. `Linear`
     weight [out,in] is transposed to the engine's row-major [in,out].
     The export contains the architecture JSON, the float32
     little-endian weight blob, its SHA-256, and a deterministic
     PyTorch-computed reference input/output. These example weights are
     randomly initialized parameters, not a trained model.
   - Plain **graph JSON** (`{"inputDim":…,"ops":[…]}`, also produced by
     `examples/forge_demo.py` from a Python dict) with seeded synthetic
     weights.
2. **Swift-side validation** (`WeightExport`): bounded reads (an
   oversized file is rejected without full reads), format tag, blob size
   and 4-byte divisibility, SHA-256 content hash, op-table agreement with
   the graph layout, manifest/graph inputDim match, and reference
   vectors finite in Float — a finite Double like 1e300 that converts to
   Float infinity is rejected.
3. **Hardware profile (measured, not a fixed list):** device name,
   pipeline `maxTotalThreadsPerThreadgroup` (min over the dispatchable
   pipelines), execution width, recommended working set — recorded in
   the plan and compared on reuse.
4. **Bounded candidate generation.** Width variants {32,64,128,256}
   clamped to the legal maximum AND the smallest dense output dim, then
   deduplicated to actually-distinct dispatches; each width with and
   without the **fused dense+ReLU pass** when (and only when) the graph
   contains a fusable dense→relu pair. No runtime search, no agents.
5. **Correctness gate before measurement.** Every candidate's output
   must be within 1e-4 of both independent CPU references (scalar and
   Accelerate/vDSP); rejected candidates are never timed.
6. **Measured selection.** One `MetalBackend.Prepared` program compiles
   shaders and uploads buffers ONCE; timed runs dispatch only. GPU time
   is the command buffer's own `gpuStartTime`/`gpuEndTime` — shader
   compilation and uploads are never part of a measurement. 3 warmups,
   then at most 11 timed runs; the per-candidate GPU-time budget is
   2,000,000 microseconds = 2 seconds (a generous cap that in practice
   never binds for these micro-graphs — actual runs complete all 14
   dispatches). Median wins. Unavailable/non-positive GPU timestamps
   reject that candidate instead of aborting the set.
7. **Plan cache and reuse.** Plans are JSON files validated on load:
   bounded size (re-checked on the actual decoded bytes), schema,
   positive bounded integers before any UInt64 conversion, workload
   (batch/seed), verbatim toolchain identity (swift version + OS +
   schema + MSL shader fingerprint), exact hardware profile, graph
   digest, **weights content digest**, selected-candidate legality
   against the regenerated candidate set, and result consistency. A
   stale, foreign, corrupt, or different-weights plan is rejected and
   re-forged. On reuse the selected candidate still passes the
   correctness gate and the PyTorch parity gate before any output is
   trusted. The digest covers model/weights/hardware/toolchain/workload/
   selection identity — it is not an integrity proof of the stored
   measurement list.
8. **Parity.** With `--export-prefix`, the selected candidate with the
   real weights on the manifest's deterministic input must match the
   PyTorch reference within 1e-4 (measured 2.2e-08). The parity and Q6
   FP64 gates use |a − ref| / max(1, |ref|) — a mixed absolute/relative
   bound: for |ref| ≤ 1 it behaves as absolute, so near-zero references
   are not blown up by a tiny denominator. The Forge candidate gates
   against the CPU references and the base `check` command use plain
   absolute |a − ref| < 1e-4. Each gate states its own bound; they are
   not all the same formula.

## Honest limitations

- Supported ops are exactly `dense` and `relu`. Anything else is
  rejected explicitly — the Swift engine consumes JSON IR only and never
  executes Python. The `export_torch.py` step DOES execute the user's
  trusted model Python file (an explicit user action, like running any
  script), but its output is fully re-validated in Swift before use.
- The width cap is the graph-global smallest dense output dim, which can
  suppress wider per-layer tilings on mixed-width graphs (bounded
  milestone choice, not a full tiling optimizer).
- The measurement budget is finite; a candidate whose budget is spent
  before a timed run is reported unmeasured, not guessed.
- The toolchain probe reads `swift --version` with a bounded read but no
  wall-clock timeout on process exit (documented, fixed executable).
- Trained weights: the pipeline fully supports real parameter values
  (demonstrated with different random-initialization seeds); a genuinely
  trained checkpoint has not been run through it yet.
- The Gemma 4 26B-A4B Q4_0 GGUF work (reader, Q4_0/Q6_K quantized Metal
  matvec kernels with FP64-gated correctness) is separate and NOT yet
  connected to Forge; full-model Gemma token generation and SSD expert
  streaming are unimplemented. No Gemma text generation, MLX, or Ollama
  comparison claims are made anywhere in this repository.
