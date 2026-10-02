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
scoped: a small inference compiler for synthetic `dense`/`relu` graphs with
real Metal GPU execution, integrity-checked compiled artifacts, and two
independent CPU reference implementations. Real LLM model import and
serving are the next milestone (see
[docs/MODEL_TARGET.md](docs/MODEL_TARGET.md)); no full-model execution is
claimed yet.

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
  `cpu-accelerate` / `cpu-scalar` for reference), and `check` for the
  multi-batch correctness gate.

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

1. **First real model** — original Swift/safetensors BF16 loader and a
   validated Llama-architecture graph for
   [SmolLM2-135M](docs/MODEL_TARGET.md): Metal embedding, RMSNorm, RoPE,
   GQA causal attention, SwiGLU, LM head; greedy token-ID generation, then
   exact tokenizer, then text. No full-model execution is claimed until it
   runs and is verified.
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
