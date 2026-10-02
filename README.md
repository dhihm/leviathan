# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan project logo: a small whale emblem marking the Leviathan inference engine project" width="140">

Leviathan is being built as a local LLM serving engine for Apple Silicon
Macs. The intended engine profiles the machine and the workload, generates a
bounded set of legal optimization candidates, checks each one for numerical
agreement with an independent reference within a tolerance, measures real
device time, selects the best valid plan, and reuses compiled artifacts
keyed by model, hardware, and toolchain — with execution on the Metal GPU.
Serving itself is not implemented yet; it is planned as localhost-only.

![Leviathan hero illustration: an original drawing of a giant whale devouring a kraken, symbolizing the Leviathan engine swallowing workloads](docs/assets/leviathan-hero.png)

**What runs today** is the verified foundation of that goal, honestly
scoped: a small inference compiler for `dense`/`relu` graphs with real
Metal GPU execution, integrity-checked compiled artifacts, and two
independent CPU reference implementations. LeviathanForge additionally
imports a restricted PyTorch export (`nn.Sequential` of `nn.Linear`/`nn.ReLU`)
with **real parameter weights** — validated, hashed, parity-checked
against PyTorch (measured 2.2e-08) — and automatically generates, gates,
and selects GPU pass candidates with a reusable measured plan cache; see
[docs/FORGE.md](docs/FORGE.md). The example weights are random
initializations, not a trained model.

**Real-model Gemma 4 26B-A4B (QAT Q4_0) now runs end-to-end through the
Forge pipeline** on an M1 Max — the original GGUF-backed tokenizer
(bounded 13-case exact-ID parity vs an independent Rust oracle),
bounded greedy generation, and the model.py capture path (a bounded,
trusted stage-list DSL — honestly NOT arbitrary Python/torch tracing):

```bash
# 1. Capture: user's trusted restricted model.py DSL -> graph JSON
.build/release/leviathan-cli gemma-config-export --file <gguf> --out /tmp/cfg.json
python3 tools/forge_gemma/export_gemma_graph.py examples/gemma_model.py /tmp/cfg.json /tmp/graph.json

# 2. Forge: validate against the GGUF -> typed SSA lowering -> full-logit
#    correctness gates (vs the independent runner forward) -> measured
#    candidates (wall-median objective, baseline fallback) -> plan artifact
#    embedding the lowered executable schedule + real digests:
.build/release/leviathan-cli gemma-forge --file <gguf> --graph /tmp/graph.json \
    --plan /tmp/plan.json --prompt-ids 2,9259 --gen-tokens 8

# 3. Generation + GPU-time profile:
.build/release/leviathan-cli gemma-generate --file <gguf> --text "Hello" --chat --max-tokens 8
.build/release/leviathan-cli gemma-profile --file <gguf>
```

Measured on this M1 Max (2026-10-02; raw logs under
[docs/evidence/](docs/evidence/)): the graph executor matches the
original hand-written runner.forward — a separate execution path that
drives the same validated kernel primitives, not a mathematically
independent full-model implementation — bit-identically on the
per-kernel path and within rel 1.6e-05 on the selected optimized pass
(all 262,144 logits, gated per decode step), with exact greedy
next-token equality over the tested 8-token workload
(" upon-the-sands-of" — also identical TEXT with matching token counts
to the same-GGUF Ollama 0.23.0 raw greedy baseline; the baseline API
exposes no internal IDs, so that agreement is text+count, and no
global no-quality-loss claim is made). Quantized kernels carry their
own independent FP64 component gates. Selected pass = cross-op
command-buffer batching (1799 → 127 submissions/token) + a tiled
cooperative Q4_0 matvec: 921,952 → 290,382 µs wall per token within one
gated run (3.17× vs the per-kernel baseline; ~3.4 decode tokens/s).
The gap to the same-GGUF Ollama baseline (server-reported ~71 eval
tokens/s, cold load 7.5 s, prefill 163 ms; the API's streamed chunks
arrive coalesced client-side, so no per-token interval is claimed)
remains large; optimization continues one measured, gated pass at a
time. Limits: the captured DSL is a bounded Gemma-specific subset;
serving, SSD expert streaming, and prefill batching are not
implemented. See [docs/GEMMA_PROGRESS.md](docs/GEMMA_PROGRESS.md) and
[docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md).

## Features

- **Metal GPU execution** — MSL compute kernels for dense and ReLU are
  compiled from source at run time (`device.makeLibrary(source:)`),
  dispatched on shared-memory buffers, and read back after a verified
  command-buffer completion. No silent CPU fallback: any Metal failure
  throws.
- **Validated graph IR** — structural and resource-limit validation of
  rank-2 float32 `[batch, features]` graphs (`dense {outDim}`, `relu`).
- **Deterministic compilation** — seeded lowering produces byte-identical
  artifacts with a SHA-256 integrity digest, full semantic re-validation on
  every load (bounds, offsets, buffer-chain topology, finite weights).
- **Independent CPU oracles** — Accelerate (vDSP) and a plain-Swift scalar
  backend; the `check` and test commands compare Metal outputs against both
  within 1e-4.
- **CLI** — `compile`, `run` (default backend `metal`; explicit
  `cpu-accelerate` / `cpu-scalar` for reference), `check` for the
  multi-batch correctness gate, `gguf-inspect` for memory-mapped GGUF
  metadata inspection, and `forge` — the LeviathanForge automatic
  pass-generation pipeline (restricted PyTorch export or graph JSON in;
  measured hardware profile, bounded legal candidates, correctness
  gates, GPU-time selection, validated plan cache reuse; see
  [docs/FORGE.md](docs/FORGE.md)).

## Installation

Requires an Apple Silicon Mac with Metal, macOS 13+, and Swift 6
(Command Line Tools suffice; no Xcode test frameworks needed).

```sh
git clone https://github.com/dhihm/leviathan.git
cd leviathan
swift build
```

The current Swift package has no third-party dependencies; building it
needs no downloads or credentials.

## Usage

```sh
# Correctness gate: Metal vs both CPU references across batches
swift run leviathan-cli check

# Compile a graph JSON into an integrity-checked artifact
swift run leviathan-cli compile --graph examples/demo-graph.json --out /tmp/demo.lvi

# Run on the GPU (default)
swift run leviathan-cli run --artifact /tmp/demo.lvi --batch 4

# Run an explicit CPU backend for reference
swift run leviathan-cli run --artifact /tmp/demo.lvi --batch 2 --backend cpu-scalar

# Test suite (exits nonzero on any failure)
swift run leviathan-tests
```

Example graph JSON (`dense` takes an output dimension; `relu` is a string):

```json
{
  "inputDim": 16,
  "ops": [
    { "dense": { "outDim": 32 } },
    "relu",
    { "dense": { "outDim": 32 } },
    "relu",
    { "dense": { "outDim": 8 } }
  ]
}
```

## Architecture

```
model import / graph JSON        [model import: planned; graph JSON: runs]
        │
        ▼
 validate ── lower (seeded, deterministic) ── LoweredProgram      [runs]
        │                                          │
        │                    compile: canonical bytes + SHA-256 ──▶ .lvi artifact
        │                                          │            [runs]
        │                    load: digest + semantic plan validation
        ▼                                          ▼
 hardware & workload profiling ──▶ bounded legal candidates ──▶ correctness gate
                          [planned]                              [runs]
                                                                       │
        reused model/hardware/toolchain-keyed artifact ◀── measured selection
                    [artifact reuse: runs for graphs; keyed reuse: planned]   [planned]
                                                                       │
                                              Metal GPU runtime ──▶ loopback serving
                                                    [runs]           [planned]
```

Today, the solid path is graph JSON → validation → lowering → artifact →
Metal dispatch → CPU-oracle comparison. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detail and
[docs/VALIDATION.md](docs/VALIDATION.md) for measured hardware evidence.

## Resource limits

Enforced in code (`Sources/LeviathanCore/Limits.swift`), and re-enforced on
the GPU path before any device memory is touched:

| Limit | Value |
|---|---|
| Tensor shape | fixed rank-2 `[batch, features]` float32 |
| Tensor elements per tensor | 2^20 (4 MiB float32) |
| Single dimension | 4096 |
| Graph ops | 64 |
| Batch size | 256 |
| Weights blob | 16 MiB |
| Whole artifact file | 17 MiB |
| Graph JSON source file | 1 MiB |

## Limitations

- Two op types only (`dense`, `relu`); single input/output, strictly
  sequential graphs. No convolutions, attention, or control flow yet.
- Weights are synthetic (seeded); no real-model weights load yet — the first
  real target is specified in [docs/MODEL_TARGET.md](docs/MODEL_TARGET.md).
- The artifact digest provides integrity, not authentication or signing.
- Float32 only; backend comparisons use a 1e-4 absolute tolerance.
- No benchmark suite; timings printed by `check` are indicative only.
- No CI, no local serving API yet (planned, localhost-only).

## Roadmap

1. **First real model** — original Swift GGUF reader and Q4_0 dequantizing
   Metal kernels for [Gemma 4 26B-A4B](docs/MODEL_TARGET.md): exact hybrid
   local/global attention with unified KV and p-RoPE, RMSNorm, MoE routing
   (top-8 of 128) plus shared expert, weights kept quantized on the GPU.
   **Status 2026-10-02: the full 30-layer forward runs on the M1 Max GPU
   from the quantized checkpoint (zero-copy mmap), and the greedy
   next-token matches the same-GGUF Ollama 0.23.0 baseline on two
   consecutive generated tokens** — bring-up evidence and exact scope in
   [docs/GEMMA_PROGRESS.md](docs/GEMMA_PROGRESS.md). Tokenizer, text
   generation, and logit-level parity remain open.
2. **One measured automatic optimization decision** — a bounded candidate
   set (e.g. threadgroup widths 32/64/128) gated by correctness, then
   selected by median measured device time under a finite tuning budget.
   Best-of-tested, not a claim of global optimality.
3. **Loopback local serving** — localhost-only, bounded requests, after
   real GPU generation works.

Model weights are never committed to git; Apache-2.0 model files remain
under their own license and are not relicensed by this repository's MIT
terms.

## Artwork

The logo and hero images in `docs/assets/` are original generated Leviathan
artwork.

## License

[MIT](LICENSE) © 2026 dhihm. Model files, if later downloaded, are governed
by their upstream licenses and are not covered by this repository's MIT
grant.
