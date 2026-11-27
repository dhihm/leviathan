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
