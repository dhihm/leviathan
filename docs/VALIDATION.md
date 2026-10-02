# Validation Evidence

Hardware/software at time of validation: Apple M1 Max (32 GPU cores, 32 GiB
unified memory, Metal supported), macOS 26.2, Swift 6.2.1, Xcode Command
Line Tools (no XCTest/swift-testing). Model invoked for code generation:
GLM-5.3 (via ZCode); all authored code, docs, and commits are in English.

All results below are from the final rerun after the buffer-safety fixes
(canonical fresh output buffers, buffer-count match, overflow-safe offsets).

## Test suite — `swift run leviathan-tests` (2026-10-02 update)

```
metal device: Apple M1 Max
163 passed, 0 failed
ALL TESTS PASSED
```

The suite now covers, beyond the original 73 checks:

- **GGUF reader** (original implementation): synthetic known-field
  containers; arrays with real GGUF element layout (string/uint32, with
  trailing metadata proving cursor position); malformed-input rejection
  (bad magic, truncation, UInt64.max counts/lengths, nested/oversized
  arrays, zero/huge dims, ragged quantized blocks, unsupported GGML
  types, data beyond EOF, misaligned/overlapping tensors, duplicate
  names/keys, zero-tensor files, near-Int.max offsets through checked
  error paths).
- **Q4_0**: hand-computed block decode (nibble order low-16 then
  high-16, FP16 scales, non-finite scale rejection); fused Metal matvec
  vs a Double-accumulating scalar oracle; geometry/count/finiteness
  guards firing before dispatch.
- **Q6_K**: hand-computed known block including negative signed subgroup
  scales and chunk-1 nibble maps (GPU = scalar = hand, exactly -315);
  random 64x512 matvec gated at 1e-4 relative against an FP64 reference
  (measured worst GPU 2.4e-05, scalar 5.7e-08; the worst row's small
  result |−15.70| from 195,570 of |products| documents the cancellation,
  not a bit-mapping error).
- **LeviathanForge**: candidate generation (width dedup to actual
  dispatch, fusion only on fusable graphs), correctness-gated measured
  selection with GPU command-buffer timing, plan cache round-trip,
  stale-plan rejection (wrong seed / workload), illegal-selected-width
  rejection, zero-batch rejection, fused==unfused output equivalence.
- **Python frontends end-to-end** (explicit user-run processes):
  dict-contract frontend, and the restricted PyTorch exporter with REAL
  parameter weights — two seeds give different weight hashes and
  different references; GPU output with the real weights matches the
  PyTorch reference at 2.235e-08 worst relative difference; weight
  content hashing (plan forged for one parameter set never validates for
  another); tampered weight bytes rejected by the manifest SHA-256.

## Original 73-check run (historical scope)

```
metal device: Apple M1 Max
73 passed, 0 failed
ALL TESTS PASSED
```

The 73 checks include the original CPU-side coverage (graph validation,
artifact determinism/round-trip, tamper/truncation rejection,
malformed-but-checksummed headers, NaN weights, SHA-256 known answers,
backend agreement, hand-computed dense+relu, run-time limits, CLI rejection)
plus the GPU suite: hand-computed dense/relu values on Metal (batches 1 and
2), dense-only negative pass-through, ReLU-only graph with an empty weights
blob and odd non-power-of-two shapes, malformed raw plans rejected before
any GPU access (bad buffer index, `Int.max` weight offset, zero buffer
count, non-finite weights, dense writing to buffer 0, dense reusing an
earlier buffer, unused declared buffer), run-time limit enforcement on the
Metal path, and Metal-vs-scalar / Metal-vs-Accelerate parity at batches
1, 4, 8, 16, and 64.

## Correctness gate — `swift run leviathan-cli check` (Metal default)

```
metal device: Apple M1 Max
shader library compiled at runtime; 5 step(s) dispatched; command buffer completed; 8 value(s) read back
batch   1: max |metal - reference| = 2.384e-06
batch   4: max |metal - reference| = 2.861e-06
batch  16: max |metal - reference| = 2.861e-06
batch  64: max |metal - reference| = 5.722e-06
all batches agree within 1e-4 (worst 5.722e-06)
PASS
```

Each batch line's evidence is captured from the actual run: the MSL library
was compiled at runtime via `device.makeLibrary(source:)`, 5 compute
encoders were dispatched on one command buffer, the command buffer reported
`completed`, and the output buffer was read back from shared storage.
Explicit CPU backends also pass: `check --backend cpu-accelerate` and
`check --backend cpu-scalar` both end in `PASS`.

## Independent output comparison (CPU scalar vs Metal, same seeded input)

First output row of the demo graph (`run --artifact /tmp/demo.lvi`):

- Metal, batch 4: `0.445163 -9.033243 3.789005 -0.006480 8.112945 ...`
- CPU scalar, batch 2: `0.445164 -9.033243 3.789005 -0.006479 8.112946 ...`

Elementwise agreement to ~1e-6, well inside the 1e-4 gate.

## Artifact compile evidence

```
compiled 5 ops -> /tmp/demo.lvi
digest: b9a00b5bfe2101da610f971c49c6ba78e0219e6d1510c14b77784109e92d747f
weights: 1864 float32, artifact bytes: 8086
```

## Limitations of this evidence

- Timings printed by `check` are single-shot wall-clock debug-build numbers
  (indicative only); no benchmark suite exists yet.
- Parity is checked at a 1e-4 absolute tolerance on float32 synthetic
  graphs up to batch 64.
- No real-model (LLM) execution is claimed; see [MODEL_TARGET.md](MODEL_TARGET.md)
  for the approved-next-step prerequisite and memory budget.
