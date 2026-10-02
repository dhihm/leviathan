# Leviathan Architecture (M1)

Leviathan is organized as a classic small compiler pipeline. Everything runs
locally; there are no external services and no third-party dependencies.

## Pipeline

1. **Source graph (JSON)** — `GraphIR` (`Sources/LeviathanCore/IR.swift`):
   `inputDim` plus an ordered list of ops (`dense {outDim}`, `relu`).
   `validate()` enforces structural rules and resource limits (positive and
   bounded dimensions, op count, per-tensor element budget) and returns the
   ordered per-op feature widths.
2. **Lowering** (`Lowering.swift`) — `LoweredProgram` walks the validated
   graph and assigns an execution plan: buffer indices (buffer 0 is the
   graph input; each dense op allocates the next buffer; relu is in-place)
   and weight offsets into a single weights blob. Weights are synthesized
   from a seeded SplitMix64 stream, so lowering is a pure function of
   `(graph, seed)` and byte-identical across runs and machines.
3. **Artifact** (`Artifact.swift`) — `CompiledArtifact` serializes the plan
   as a canonical JSON header (sorted keys) plus the little-endian float32
   weights blob, followed by a SHA-256 digest over all preceding bytes.
   Layout: magic `LVI1` | UInt32 header length | header JSON | weights |
   SHA-256.
4. **Load-time validation** — recomputes the digest (rejecting any
   corruption or tampering), bounds the header length and weight count
   before arithmetic that could overflow, then semantically re-validates the
   plan: dimensions, step count, buffer indices and the topological buffer
   chain, weight offsets/counts against the blob, declared output shape, and
   finiteness of every weight. Even a malformed artifact with a correctly
   recomputed digest is rejected here.
5. **Execution** (`Backends.swift`) — two independent backends:
   - `AccelerateBackend`: dense = `vDSP_mmul` + `vDSP_vadd` (tiled bias),
     relu = `vDSP_vmax` against zeros.
   - `ScalarBackend`: plain Swift loops, no Accelerate. This is the ground
     truth for the correctness gate (`maxAbsDiff` < 1e-4), which also
     rejects NaN/infinite outputs rather than comparing them.

## Determinism

- Same `(graph, seed)` ⇒ identical artifact bytes ⇒ identical SHA-256.
- The input vector in `run`/`check` is generated from a seed, so outputs are
  reproducible.
- Canonical JSON (`.sortedKeys`) removes dictionary-ordering nondeterminism.

## Testing

`leviathan-tests` (run: `swift run leviathan-tests`) is an original minimal
harness because neither XCTest nor swift-testing is available under the
active CommandLineTools toolchain. It covers: graph validation acceptance
and rejection cases, artifact determinism and byte-exact round-trip,
tamper/truncation rejection, malformed-but-correctly-checksummed headers
(wrong output dim, bad buffer count, negative/`Int.max` weight counts,
out-of-range offsets, empty step list), NaN weights with a matching digest,
SHA-256 known-answer vectors, backend agreement, a hand-computed dense+relu
case, run-time resource limits, NaN-safety of the comparison helper, and CLI
rejection of malformed flags without trapping. It exits nonzero on failure.

## Resource discipline

All caps live in `Limits.swift` and are enforced before allocation: file
sizes are bounded before reading, weight counts before multiplication,
batch sizes before buffer allocation, and per-tensor element counts before
weight synthesis.

## Forge note

"Connect to Forge" refers to the internal Forge-inspired compiler and
optimization architecture implemented locally in this repository. No external
Forge service is used; future optimization hooks will live inside this
pipeline.
