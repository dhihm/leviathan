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
