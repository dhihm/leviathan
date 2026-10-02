# Leviathan Architecture

Leviathan's product goal is a local-LLM serving engine for Apple Silicon:
model import → hardware/workload profiling → bounded legal optimization
candidate generation → correctness gate → measured selection → reusable
model/hardware/toolchain-keyed compiled artifacts → Metal GPU runtime →
loopback local serving API. Everything runs locally; there are no external
services and no third-party dependencies.

The currently implemented, verified slice of that pipeline is: source graph
JSON → validated IR → deterministic lowering → integrity-checked artifact →
real Metal GPU execution → independent CPU-oracle comparison. Real model
import (see [MODEL_TARGET.md](MODEL_TARGET.md)) builds on this foundation.

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
   chain, weight offsets/counts against the blob (overflow-safe: offsets
   are bounded by the blob size before any addition), declared output
   shape, and finiteness of every weight. Even a malformed artifact with a
   correctly recomputed digest is rejected here.
5. **Execution** — three backends:
   - `MetalBackend` (`MetalBackend.swift`, product primary target): fully
     validates the raw `LoweredProgram` through the same plan validator
     (via a temporary header) before any GPU resource is allocated, so an
     unvalidated plan can never become out-of-bounds device memory access.
     It then compiles the MSL dense/ReLU kernels from source at run time
     (`device.makeLibrary(source:)`), creates compute pipelines, uploads
     the weights blob and input to explicit `.storageModeShared` MTLBuffers
     (zero-copy readback, no managed-buffer sync), encodes one compute
     encoder per step on a single command buffer (encoders serialize in
     encode order, which is the ordering barrier between dependent steps),
     commits, waits, requires `status == .completed` and a nil error, and
     reads the output buffer back. Failures throw (`MetalError`); there is
     no silent CPU fallback. `runReporting` returns evidence (device name,
     shader compiled, steps dispatched, command status, elements read back)
     captured from the actual run.
   - `AccelerateBackend` (CPU): dense = `vDSP_mmul` + `vDSP_vadd` (tiled
     bias), relu = `vDSP_vmax` against zeros.
   - `ScalarBackend` (CPU): plain Swift loops, no Accelerate. This is the
     ground truth for the correctness gate (`maxAbsDiff` < 1e-4), which
     also rejects NaN/infinite outputs rather than comparing them.

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
tamper/truncation rejection, malformed-but-correctly-checksummed headers,
NaN weights with a matching digest, SHA-256 known-answer vectors, CPU
backend agreement and a hand-computed dense+relu case, run-time resource
limits, NaN-safety of the comparison helper, CLI rejection of malformed
flags, and — on the GPU side — hand-computed dense/relu values on Metal
(batches 1 and 2), a dense-only negative pass-through, a ReLU-only graph
with an empty weights blob and odd non-power-of-two shapes, malformed raw
plans rejected before any GPU access (bad buffer index, `Int.max` weight
offset, zero buffer count, non-finite weights), run-time limit enforcement
on the Metal path, and Metal-vs-scalar / Metal-vs-Accelerate parity at
batches 1, 4, 8, 16, and 64. It exits nonzero on failure.

## Resource discipline

All caps live in `Limits.swift` and are enforced before allocation: file
sizes are bounded before reading, weight counts before multiplication,
batch sizes before buffer allocation, and per-tensor element counts before
weight synthesis. The Metal path re-validates the complete plan before any
device memory is allocated or dispatched.
