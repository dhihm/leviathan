# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan project logo: a small whale emblem marking the Leviathan inference compiler project" width="140">

Leviathan is a personal, from-scratch experiment in building a small **CPU
inference compiler for Apple Silicon** in Swift. It is an exploration of the
generic compiler pipeline — validated graph IR, deterministic lowering,
reproducible compiled artifacts with integrity validation, and a correctness
gate comparing an optimized backend against an independent reference
implementation. It is **not** a trained model, not an LLM, and makes no claims
about production-grade inference.

![Leviathan hero illustration: an original drawing of a giant whale devouring a kraken, symbolizing the Leviathan compiler swallowing workloads](docs/assets/leviathan-hero.png)

## Status: first milestone (M1)

What actually runs today, end to end, on an Apple M1 Max (10 CPU cores,
macOS 26.2, Swift 6.2.1, arm64):

- A small **tensor graph IR** with exactly two op types, `dense` (float32
  affine transform) and `relu`, operating on row-major `[batch, features]`
  tensors, with structural and resource-limit validation.
- A **deterministic lowering** pass that assigns buffers and synthesizes
  reproducible weights from a seeded SplitMix64 stream (a clearly synthetic
  network — no training, no pretrained downloads).
- A **compiled artifact format** (`LVI1`) with canonical JSON header,
  little-endian float32 weights blob, and a SHA-256 digest that is
  recomputed and enforced on every load, plus full semantic re-validation of
  the decoded plan (bounds, offsets, buffer-chain topology, non-finite
  weight rejection). The digest is an integrity check, not a digital
  signature; there is no author authentication.
- Two execution backends: **Apple Accelerate (vDSP)** and an independent
  **scalar reference** in plain Swift, cross-checked with a 1e-4 tolerance.
- A CLI (`leviathan-cli`) with `compile`, `run`, and `check` commands.
- An executable test harness (`leviathan-tests`) with 41 checks; the
  standard XCTest / swift-testing modules are not available under the
  CommandLineTools toolchain, so this project ships its own minimal harness.

## Quick start

```sh
swift build

# Run the correctness gate (Accelerate vs scalar reference)
swift run leviathan-cli check

# Compile a graph JSON into an integrity-checked artifact
swift run leviathan-cli compile --graph examples/demo-graph.json --out /tmp/demo.lvi

# Load, validate, and run the artifact
swift run leviathan-cli run --artifact /tmp/demo.lvi --batch 4

# Run the test suite (exits nonzero on any failure)
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
graph JSON ──validate──▶ GraphIR ──lower (seeded, deterministic)──▶ LoweredProgram
                                                                        │
                                              compile: canonical bytes + SHA-256 ──▶ .lvi artifact
                                                                        │
                                              load: digest + semantic plan validation
                                                                        │
                                     ┌──────────────────────────────────┴──────────┐
                              AccelerateBackend (vDSP)                    ScalarBackend (reference)
                                     └──────────────────┬──────────────────────────┘
                                              max |diff| < 1e-4 correctness gate
```

See `docs/ARCHITECTURE.md` for details.

## Resource limits

Enforced in code (`Sources/LeviathanCore/Limits.swift`), not by convention:

| Limit | Value |
|---|---|
| Tensor shape | fixed rank-2 `[batch, features]` float32 (no general rank support) |
| Tensor elements per tensor | 2^20 (4 MiB float32) |
| Single dimension | 4096 |
| Graph ops | 64 |
| Batch size | 256 |
| Weights blob | 16 MiB |
| Whole artifact file | 17 MiB |
| Graph JSON source file | 1 MiB |

## Limitations (honest scope)

- Two op types only (`dense`, `relu`); no convolutions, attention, or
  control flow. Single input, single output, strictly sequential graphs.
- Weights are **synthetic**, generated deterministically from a seed. This
  milestone demonstrates compiler mechanics, not model inference quality.
- The artifact digest provides integrity (corruption/tamper detection), not
  authentication or signing.
- Float32 only; tolerance-based comparison between backends (1e-4 absolute).
- No benchmark suite yet; timings printed by `check` are indicative only.
- No CI; no third-party dependencies (deliberately, to keep provenance
  simple and downloads at zero).

## Roadmap

1. M1 (this): validated IR, deterministic artifacts, dual backends, CLI. ✅
2. M2: more elementwise ops, real weight import, batched buffers reuse.
3. M3: BNNS/Metal Performance Shaders backend comparison, benchmark harness.
4. M4: small real model import (requires owner approval before any model
   download), kernel-level optimization experiments.

## References

- [vDSP / Accelerate documentation (Apple)](https://developer.apple.com/documentation/accelerate/vdsp)
- [FIPS 180-4 (Secure Hash Standard)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [Swift Package Manager](https://www.swift.org/documentation/package-manager/)

## Project note: "Connect to Forge"

"Forge" in this project refers to the internal Forge-inspired compiler and
optimization architecture — the local validate → lower → artifact → execute
pipeline described above. No external Forge service is used or needed; future
work may add optimization hooks inside this pipeline.

## Artwork

The logo and hero images in `docs/assets/` are original artwork generated
specifically for the Leviathan project. They are not copied from any other
project or organization.

## License

None granted yet. All rights reserved by the owner until a license is chosen.
