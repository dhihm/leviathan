# Leviathan Forge Python runtime (generic Linear/ReLU subset)

`leviathan_forge.py` (orchestrator) + `leviathan-forge-helper` (small Swift target) turn a **trusted** local PyTorch `nn.Module` file into a bound, reloadable Metal plan bundle, run it on a real input, and check it against the source module.

## Scope and trust
- Only FP32, static input `[1, INPUT_DIM]`, one chain of `aten.linear.default` (with bias) and `aten.relu.default`, <=64 ops, <=16 MiB weights. Not an LLM runtime, not a general compiler. Other ATen nodes fail closed with op and node named.
- The model file (`INPUT_DIM`, `build(seed)`) is **executed as ordinary local Python with your privileges**; no sandbox. Bundles, input JSON, and helper output are untrusted data (bounded, validated).
- No agent/LLM/optimizer at inference time. `run` never re-tunes; any mismatch is a rejection.
- The existing frontend `tools/torch_frontend/export_torch_graph.py` is unchanged; its validation/lowering functions are reused.

## Usage (from the repository root)
```sh
swift build -c release --product leviathan-forge-helper
mkdir -p /tmp/lev
python3.11 tools/forge_run/leviathan_forge.py sample-input --dim 128 --seed 1 --out /tmp/lev/ex.json
python3.11 tools/forge_run/leviathan_forge.py sample-input --dim 128 --seed 2 --out /tmp/lev/in.json
python3.11 tools/forge_run/leviathan_forge.py sample-input --dim 128 --seed 3 --out /tmp/lev/in2.json
python3.11 tools/forge_run/leviathan_forge.py build tools/forge_run/examples/mlp_tower.py --example-input /tmp/lev/ex.json --input /tmp/lev/in.json --out /tmp/lev/tower.bundle
python3.11 tools/forge_run/leviathan_forge.py run /tmp/lev/tower.bundle --model tools/forge_run/examples/mlp_tower.py --input /tmp/lev/in2.json
python3.11 tools/forge_run/test_leviathan_forge.py -v
```
Input JSON: `[[x0, ..., xD-1]]`, finite and representable as float32, <=1 MiB. Python API: `build_bundle(...)`, `run_bundle(...)`. Output directories and `--report` files must not exist (never overwritten).

## Selection (baseline first)
The helper generates the legal set via `Forge.candidates` (device-measured threadgroup cap, fusion only where the graph allows). The unfused baseline is verified first on 4 inputs (example + 3 seeded) against scalar, Accelerate, and (example) the source PyTorch output. Each optimization must pass the same qualification and is benchmarked with real command-buffer GPU timestamps; it is selected only if it beats the baseline median by >=2%. Otherwise the bundle records an explicit baseline fallback.

`--diagnostic-fail-optimizations` (API `diagnostic_fail_optimizations=True`) is **diagnostic only**: after each real GPU run of an optimization candidate (never the baseline) its first output is perturbed, so qualification fails and fallback is exercised. It is recorded in the bundle.

## Bundle and binding
`model.graph.json`, `model.weights.f32`, `model.manifest.json`, `model_source.py`, `bundle.json`, `bundle.sha256` (integrity check, not a signature). `bundle.json` binds: model source SHA-256, seed, ATen ops, example input digest, torch/Python versions, backend source digest (relative paths + content of Package.swift, LeviathanCore, helper, frontend, orchestrator; portable across checkouts), file digests (graph, actual weights, manifest, source copy), helper binary SHA-256 (computed by the helper at run time), Metal shader fingerprint, Swift toolchain/OS, device profile, shape, selection.

`run` rejects before GPU execution: digest/file corruption, different model file (before executing it), different weights or graph after re-capture, altered backend sources, rebuilt/altered helper, shader/toolchain change, other device, illegal saved selection, wrong input shape. Changing the helper executable content invalidates earlier bundles; re-forge explicitly.

## Timing labels
`gpuCommandBufferMicros` = GPU only (one command buffer). `helperProcessMillis` = fresh helper process wall time. `endToEndMillis` = orchestrator function wall time to GPU output (validation, re-capture, helper), excluding Python process startup/imports and the final CPU reference comparison. Full-process wall time must be measured separately. Forge benchmarking timings are separate (`forgeTiming`).

## Limitations
Parity probes and qualification inputs are bounded evidence, not proofs. File reads have small TOCTOU windows. GPU timings on tiny models are noisy; the 2% selection threshold does not establish statistical significance or a general speedup. The Python run path re-captures the trusted model to validate graph/weights and performs eager CPU parity; that validation is included in end-to-end time. Kernel-sequence/count evidence is derived from host lowering after a successful command buffer, not a GPU profiler trace.

## Input validation and failure handling

The Python API accepts CPU, dense strided FP32 tensors with shape `[1, D]`,
width 1..4096 and finite values. JSON uses the same validator after explicit
float32 parsing. Invalid dtype/device/layout/value/rank inputs fail before
model construction, capture, GPU work or artifact writes. Declared/bound
width mismatches fail before model construction or dispatch. Accepted input
tensors are detached and copied to stable contiguous snapshots.

`build` forges into its own hidden stage and executes the supplied input
with source-PyTorch parity before publishing the final directory. On macOS,
publication uses `renamex_np(..., RENAME_EXCL)` to create the final bundle
atomically without replacing any existing output, including a directory
created during validation. Failed stages keep raw files and a
`FORGE_FAILED.json` marker under a `FAILED` name; the Python `run` entry point
rejects those stages. They are diagnostic evidence, not usable bundles.

Optional build reports are prepared and linked exclusively before bundle
publication. Handled report failures leave no final bundle. If bundle
publication subsequently fails, rollback removes only its own report with
matching inode and bytes; foreign outputs remain untouched. Bundle
visibility is atomic. Bundle and report are not a two-artifact crash
transaction: a process crash between report linking and directory rename
may leave a report without a bundle. Report rollback has a small
check/unlink race window; no hostile concurrent mutation guarantee is made.

Run all GPU acceptance and focused fault tests with:
```sh
LEVIATHAN_FORGE_KEEP_TEST_OUTPUT=1 python3.11 -m unittest discover -s tools/forge_run -p 'test_*.py' -v
```

### Declared native Metal Conv2D

Build the single-operator helper with `swiftc -O`. The CLI consumes a
JSON declaration and raw little-endian FP32 input and weight files.
Tensor layouts are contiguous NCHW and OIHW. The output directory must
not exist. Its parent must exist.

```sh
swiftc -O tools/forge_run/metal_conv2d.swift -o .build/leviathan-metal-conv2d
python3 tools/forge_run/conv2d_build_declared.py \
  --helper .build/leviathan-metal-conv2d --contract contract.json \
  --input input.f32 --weights weights.f32 --out conv-result --cache conv-plans
```

For the ResNet-18 stem with a 113×151 image, set `format` to
`leviathan-declared-conv2d-1`, `inputShape` to `[1,3,113,151]`,
`weightShape` to `[64,3,7,7]`, `outputShape` to `[1,64,57,76]`,
`stride` to 2, `padding` to 3, and `groups` and `dilation` to 1.
Set `weightsSha256` to the SHA-256 of the raw weights file. Bias and
batch sizes above one are outside the version 1 contract. A successful
run writes `output.f32` and `report.json` after independent CPU parity.
The optional private cache reuses dispatch parameters after checking
the current device and pipeline. Each invocation executes the kernel
and performs the reference comparison.

Reproduce the full feature-map test with the public TorchVision weights
and Python with NumPy, PyTorch, and pytest installed.

```sh
scratch=$(mktemp -d)
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth -o "$scratch/resnet18.pth"
LEVIATHAN_RESNET18_CHECKPOINT="$scratch/resnet18.pth" \
  python3 -m pytest -q -s tools/forge_run/test_conv2d_declared.py
```

### Declared Metal ResNet basic block

Build the same Metal helper as for Conv2D. For a residual block use
`resnet_block_declared.py` with the same flags. Its versioned JSON
contract has these fields:

- `format` is `leviathan-declared-resnet-basic-block-1`.
- `inputShape` and `outputShape` are contiguous FP32 NCHW with batch 1.
- `skip` is `identity` with `stride` 1, or `projection` with `stride` 2.
- `folding` is `eval-batchnorm-fp64-fold-to-fp32-1`.
- `epsilon` lists the inference BatchNorm epsilons in layer order.
- `originalSha256` binds the bytes of the original weights file.

Pack the raw little-endian FP32 weights file for main convolution 1,
main convolution 2, then the projection if present. For each layer
write the OIHW convolution weights, BatchNorm scale, offset, running
mean, and running variance, in that order. The CLI verifies full
output parity against a separate unfused CPU reference and publishes
`output.f32` and `report.json`. The report records the folded digest
and Metal command-buffer time separately from BUILD timing.

Use the same pinned TorchVision checkpoint as above and run
`python3 -m pytest -q -s tools/forge_run/test_resnet_block_declared.py`
with `LEVIATHAN_RESNET18_CHECKPOINT` set. The test covers both skip
paths and prints cold/warm medians, ranges, and the maximum PyTorch
comparison error.

For the complete stem and layer1 at 224×224, use the same helper and CLI
with format `leviathan-declared-resnet18-stage1-1`. The exact `ops`
descriptor is `STAGE_OPS` in `resnet_block_declared.py`. Set input to
`[1,3,224,224]`, output to `[1,64,56,56]`, and provide five epsilons.
Pack original convolution and BatchNorm groups in order: stem, both
convolutions of layer1 block 0, then both of block 1. The same FP32
layout, folding, and source-digest rules apply. Successful builds also
publish `stem.f32`, `pool.f32`, and `block1.f32` after GPU completion
and boundary validation. The public checkpoint test, with
`LEVIATHAN_RESNET18_CHECKPOINT` set as above, is
`python3 -m pytest -q -s tools/forge_run/test_resnet_stage1_declared.py`.

For full ResNet-18 inference, use format `leviathan-declared-resnet18-1`
with the exact `FULL_OPS` descriptor, input `[1,3,224,224]` and
output `[1,1000]`. Pack the stem and all eight basic blocks in forward
order. For projection blocks, pack the two main convolutions followed
by the projection convolution, each with its original BatchNorm
parameters. Provide the 20 BatchNorm epsilons in that same order.
Append the classifier's FP32 `[1000,512]` weights and `[1000]` bias,
and hash the complete source file. The CLI also emits all major stage
outputs and the pooled 512-vector after validation.

With NumPy, pytest, PyTorch, TorchVision and Pillow installed, reproduce
the checkpoint regression and independent-process measurements with:

```sh
scratch=$(mktemp -d)
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth -o "$scratch/resnet18.pth"
curl -fL https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png -o "$scratch/astronaut.png"
LEVIATHAN_RESNET18_CHECKPOINT="$scratch/resnet18.pth" \
  LEVIATHAN_RESNET18_IMAGE="$scratch/astronaut.png" OPENBLAS_NUM_THREADS=1 \
  python3 -m pytest -q -s --basetemp="$scratch/pytest" tools/forge_run/test_resnet18_declared.py
```

Full ResNet-18 BUILD reports `phaseMillis` for source weights read,
original SHA-256, BatchNorm folding and packing, and temporary input
writes. `gpu.pipelineMillis` covers Metal library and pipeline
creation, and `gpu.gpuMicros` covers the command buffer. The independent
`cpuReferenceMillis` can overlap Metal preparation and execution.
Use subprocess wall time for the full cost including Python startup
and publication. Keep checkpoint conversion outside the timed CLI.

### Portable declared ResNet-18 execution

Reuse the same residual block, stage1, or full-model declaration,
original FP32 weights file, and FP32 input file from the Metal example:

```sh
python3 tools/forge_run/resnet_run_declared.py --backend cpu \
  --contract contract.json --weights weights.f32 --input input.f32 \
  --out boundaries.npz
```

This PyTorch runner executes the original Conv2D and inference BatchNorm
separately, then checks every output against the independent NumPy FP64
reference. It writes the validated tensors to a new NPZ archive and
reports device and error information on stdout. Metal's folding
provenance remains in the declaration. No format conversion is required.
On NVIDIA hardware with CUDA-enabled PyTorch, select `--backend cuda`.
That choice requires CUDA, checks the device of each output and reports
CUDA event time. An unavailable GPU causes an explicit error.

The public-checkpoint case uses the environment variables above. Set
`LEVIATHAN_RESNET18_BACKEND=cuda` on an NVIDIA runtime to test the same
files there, and omit it for CPU. Stage1 alone can be tested with:

```sh
python3 -m pytest -q -s tools/forge_run/test_resnet_portable.py -k stage1
```

On a CUDA runtime with the same public checkpoint and image setup,
run the full-model portability test once for each input:

```sh
export LEVIATHAN_RESNET18_BACKEND=cuda LEVIATHAN_RESNET18_REPEAT=6
export LEVIATHAN_RESNET18_CHECKPOINT="$scratch/resnet18.pth"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
env -u LEVIATHAN_RESNET18_IMAGE python3 -m pytest -q -s tools/forge_run/test_resnet_portable.py -k full
LEVIATHAN_RESNET18_IMAGE="$scratch/astronaut.png" \
  python3 -m pytest -q -s tools/forge_run/test_resnet_portable.py -k full
```

The repeated CLI run verifies identical boundary hashes every time.
Its CUDA report separates initialization, transfer and first-forward
timing from the later forward events, and includes an untimed
device audit of dispatcher operations. Remove the backend environment
variable to run the same check on CPU.

### Stories15M generation timing

`llama-generate --timing json` adds one versioned JSON line to stderr
and leaves the normal token and backend lines on stdout. For CUDA
timing, rebuild the pinned llama.cpp adapter with its v2 timing ABI.
Untimed generation accepts the existing v1 adapter. Use the public
Stories15M F32 GGUF and the release CLI:

```sh
python3 tools/test_llama_timing.py --cli .build/release/leviathan-cli --model stories15M.gguf
# On an NVIDIA host, add --library /path/to/libleviathan_llama_cuda.so.
```

The test pins the model SHA-256, checks exact greedy token sequences
for two fixed prompts, and reports six fresh-process samples per
prompt. It checks that enabling timing preserves stdout byte for
byte and that CUDA generation reports a completed device matrix
multiply in each decode call.

The five non-overlapping timing fields use monotonic host clocks.
On CPU, model load maps and parses GGUF, and preparation materializes
the model tensors, validates generation inputs and sets up the KV
cache. Prefill covers the prompt transformer work. First-token time
covers final projection and greedy selection. Each later decode
sample covers one new input token and the output selection.

On CUDA, model load is llama.cpp's combined load/upload API. The
preparation field includes adapter loading, backend initialization,
and context creation. Prefill is the synchronized prompt decode,
which already computes first-token logits. First-token time covers
retrieval, validation, and greedy selection of those logits. A later
decode sample includes synchronized inference and selection.
CUDA values are host-observed elapsed time, separate from device
evidence. Compare within each backend's timing boundary. The phase
fields omit process startup, reporting and teardown.

### Reuse loaded Llama weights across requests

`llama-serve` is a serial stdin/stdout JSON Lines interface. Start it
once with a GGUF path, wait for the `ready` record, then send requests
with a positive integer ID, a prompt token array and a generation count:

```sh
printf '%s\n' \
  '{"id":1,"promptIds":[1],"count":5}' \
  '{"id":2,"promptIds":[1,9038,2501,263],"count":5}' |
  .build/release/leviathan-cli llama-serve --file stories15M.gguf
```

Each successful `result` includes the ID, token IDs and timings.
Adding `"cancelAfterTokens":2` to a five-token request stops after two
internal token steps and produces a `cancelled` result without partial
tokens. This protocol cancels at a token boundary using the budget
attached to the request. It does not accept an asynchronous cancel
message. Invalid requests return `error` and the next complete request
can still run. Input is newline-terminated and bounded to 16 KiB per
request. EOF closes the session. Open a new process to use different
weights.

The ready record reports one-time weight loading and model preparation.
Each CPU request starts with fresh KV and position state. CUDA shares
the model but creates and frees a context per request. Greedy selection
has no persistent sampler state. For CUDA, use `--backend cuda` and
`--cuda-library PATH` with an adapter built from this version. CUDA
requests require completed device work, and the command does not fall
back to the scalar backend.

```sh
python3 tools/test_llama_serve.py --cli .build/release/leviathan-cli --model stories15M.gguf
# Add --library /path/to/libleviathan_llama_cuda.so for the NVIDIA check.
```

The harness alternates the two pinned prompts through one loaded model
and compares each result with a fresh timed `llama-generate` call. It
reports six pairs per prompt. Its time-to-first-token figure sums the
non-overlapping in-engine phases through first-token selection. The
JSON protocol returns the full response at once. Reply wall time
includes later decode and serialization and is reported separately.

For the larger FP16 regression, use the pinned
[SmolLM2-135M-Instruct conversion](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/blob/b57b6475ae6f5f2b1b51c43a1190a64cab325ef4/SmolLM2-135M-Instruct-f16.gguf):

```sh
curl -fL 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/b57b6475ae6f5f2b1b51c43a1190a64cab325ef4/SmolLM2-135M-Instruct-f16.gguf' -o smol.gguf
python3 tools/test_llama_serve.py --fixture smollm2-135m \
  --cli .build/release/leviathan-cli --model smol.gguf
```

The fixture checks the GGUF SHA-256, exact five-token outputs, and six
fresh/resident pairs per raw prompt. Add `--library` to check the CUDA
path with the same CPU golden tokens. It samples host RSS at ready and
after the requests and macOS physical footprint when available. It reports per-process GPU memory when the driver
exposes it. These are idle endpoint samples. They exclude transient
memory peaks during loading or decode.

### Stream token events to a pipe consumer

Pass `--stream jsonl` to `llama-serve` to opt into the
`leviathan-llama-serve-stream-1` ready format. Requests use the
same fields as above. Each generated token is written immediately to
stdout as a newline-terminated `token` event with the request `id`,
zero-based `index`, and integer `tokenId`. The terminal `result`
contains the same full token list and timings as the default mode.
The default emits no token events.

A cancelled stream can contain tokens already delivered before the
terminal `cancelled` event. A terminal error or cancellation does not
carry a completed token list. Complete frames after a recoverable
request error still run with fresh request state. Clean stdin EOF closes
the session after its complete frames. A truncated frame at EOF ends
with an error. The worker writes synchronously to the output pipe,
so a slow consumer can stall its serial request loop. Cancellation
still uses the request's `cancelAfterTokens` budget.

```sh
python3 tools/test_llama_stream.py --cli .build/release/leviathan-cli \
  --fixture stories15m --model stories15M.gguf
python3 tools/test_llama_stream.py --cli .build/release/leviathan-cli \
  --fixture smollm2-135m --model smol.gguf
```

Add `--library PATH` to the harness on NVIDIA hardware. The adapter
must provide the streaming request ABI for opt-in CUDA streaming.
The harness is a separate process from the worker and records receipt
of complete JSON lines from the stdout pipe before parsing. It
measures first token event and terminal result against the start of
the request write. It runs both warm resident modes with six requests
per prompt and verifies the pinned token lists, IDs and EOF behavior.
