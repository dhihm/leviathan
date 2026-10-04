# Static value DAG Forge experiment

This local experiment extends the validated Linear/ReLU path with immutable values, activation fanout, and same-shape residual addition. It uses real `torch.export(strict=True)` captures and imported model parameters, then executes the resulting graph on Metal. It does not invoke an agent during capture, pass selection, or inference.

## Supported contract

A trusted local Python file defines `INPUT_DIM` and `build(seed) -> nn.Module`. The file runs with ordinary Python privileges; this is not an untrusted upload sandbox.

The supported operators are biased `aten.linear.default`, `aten.relu.default`, and `aten.add.Tensor` with two identically shaped tensor values and `alpha=1`. Input and output are static CPU FP32 tensors with shape `[1, features]`. Feature dimensions are 1–4096, graphs contain at most 64 nodes, and imported weights occupy at most 16 MiB. Broadcast, scalar addition, dynamic shapes, unsupported ATen operations, exported constants/buffers, unused parameters, and shared linear parameters are rejected.

These are small original randomly initialized modules, not trained language models. No weights or dependencies are downloaded.

## Representation and execution

`DAGGraph` assigns value 0 to the input and sequential topological IDs to operation results. Validation checks references, arity, exact dimensions, output reachability, weight limits and finite parameters. The artifact graph format is `leviathan-value-dag-1`.

Every node writes a distinct buffer. Consumer counts and last use are computed explicitly; no buffer reuse is attempted yet. Dense/ReLU fusion is legal only when the ReLU immediately follows the dense node and is that dense value's sole consumer. The fused kernel writes the ReLU result buffer. For `h + relu(h)`, the original negative values in `h` remain available to addition.

Scalar Swift and Accelerate provide independent CPU references. Metal executes dense, functional ReLU and addition kernels, using separate compute encoders for dependencies. It uploads the actual exported weights. Completion status, positive command-buffer GPU timestamps and readback are checked. Dispatch counts and kernel names are derived from host encoding, not a hardware profiler trace.

## Automatic selection

Selection profiles the actual Metal device and pipeline thread limits. It generates the bounded legal set from threadgroup widths 32/64/128/256 and safe fusion eligibility, with dimension/device clamping and duplicate removal.

An unfused baseline is qualified first against scalar and Accelerate references on four deterministic inputs: the source example and three seeded inputs. The source PyTorch example is also checked. Every candidate uses the same finite per-element criterion:

`abs(gpu - reference) <= 1e-4 + 1e-4 * abs(reference)`.

Qualified candidates receive three warmups and up to eleven measured GPU runs under the existing finite budget. A candidate wins only when its median is strictly more than 2% faster than the verified baseline. Otherwise the baseline is retained. A diagnostic switch perturbs only optimization readbacks after real GPU execution to test rejection and fallback.

This searches a small existing pass space. It does not synthesize arbitrary kernels or prove global optimality. Dense reduction remains serial within each output thread; this milestone does not solve every GPU parallelization problem.

## Artifact reuse

The new helper is `leviathan-dag-helper`; the existing helper and chain validators remain intact. DAG bundles use `leviathan-dag-forge-bundle-1` and `leviathan-dag-weights-1` manifests.

Bundles bind graph bytes and edges, actual weights and layouts, model-source copy and seed, example input, ATen sequence, Python/Torch versions, backend/frontend sources, helper binary, shaders, toolchain/OS, shape, device profile and measured selection. Reuse validates bindings and regenerates the legal candidate set. It does not retune silently. The Python entry point additionally re-captures the trusted model and compares graph/parameter bytes before invoking the helper on the actual input.

SHA-256 checks provide integrity, not authenticity. The helper hashes the source copy without executing Python. Current Python/source enforcement belongs to the Python entry point.

Builds use a hidden stage. The actual user input must run on the GPU and pass source PyTorch parity before an exclusive atomic directory rename publishes the local bundle. Failures retain a marked `.FAILED-*` sibling for evidence. Existing outputs are never overwritten. Optional report publication and bundle publication are not a joint crash transaction.

## Run locally

Use the existing Python environment containing PyTorch; on the development Mac this is `/opt/anaconda3/bin/python`.

```sh
swift build -c release --product leviathan-dag-helper

/opt/anaconda3/bin/python tools/forge_run/dag_forge.py build \
  --model tools/forge_run/examples/residual_block.py \
  --example-input /tmp/example-32.json \
  --input /tmp/input-32.json \
  --out /tmp/my-new-dag-bundle

/opt/anaconda3/bin/python tools/forge_run/dag_forge.py run \
  --bundle /tmp/my-new-dag-bundle \
  --model tools/forge_run/examples/residual_block.py \
  --input /tmp/different-input-32.json
```

Each input JSON is a finite nested array with one row and the declared number of features. The second original example, `projected_branches.py`, uses a different topology and 24 input features through the same engine.

Run the actual framework and GPU acceptance checks:

```sh
swift build -c debug
/opt/anaconda3/bin/python -m unittest tools/forge_run/test_dag_frontend.py -v
/opt/anaconda3/bin/python -m unittest tools/forge_run/test_dag_forge.py -v
```

Reports separate GPU command-buffer time, helper startup/validation/preparation, Python capture, and total API runtime. A fresh Python process adds interpreter/import startup outside the API timer. GPU microseconds are not token latency or a serving benchmark.

Source/export equality is checked on the example and eight seeded probes. This is bounded differential evidence, not a proof against every input-dependent specialization.

## Validation on the development Mac

The local M1 Max run used the existing Swift 6.2.1 and PyTorch 2.9.1 environments. Debug and release builds passed. Validation passed 33 DAG representation/CPU checks, 54 actual Metal checks, 4 framework importer test methods, 14 DAG workflow acceptance tests, all 29 existing Python acceptance tests, and all 778 existing regression checks. Optional pretrained Gemma gates were not enabled.

Both source examples were also run independently through the release helper with actual input and a different fresh-process input:

| Original module | Nodes | Weight bytes | Baseline median GPU µs | Selected median GPU µs | Fresh-input source max error | Dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual block, 32 → 96 → 32 | 4 | 25,088 | 107.708 | 103.875 | 5.96e-8 | 3 |
| Projected branches, 24 → 48 | 8 | 28,416 | 134.500 | 123.583 | 7.45e-8 | 6 |

Both selected width 32 with safe fusion in this sweep. The first graph offered one alternative to the baseline; the second offered three. These are single baseline-first sweep measurements, not an alternating repeated performance study or a comparison with MLX/PyTorch MPS. The full fresh Python process took about 3.23 s and 2.67 s respectively. It re-captures the model and starts a helper that validates and recompiles Metal pipelines. This is not a low-latency serving path.

Independent direct-helper checks rejected self-referencing graph edges, altered binary/shader identity, altered shape/device/selection, and a marked failed stage even after the relevant checksums were recomputed. A separate fault test changed a finite value in an actual completed GPU readback: source parity rejected it, no final bundle appeared, and the failed stage retained raw evidence. Forced optimization failures also retained the verified baseline.

The existing validated MVP working copy remained byte-identical across all 254 recorded files. This experiment is a local copy; no new commit, push or PR is part of this validation.

## Official API references

- [PyTorch 2.9 torch.export](https://docs.pytorch.org/docs/2.9/export.html): the framework graph-capture API used by this experiment.
- Apple Metal command-buffer [gpuStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) and [gpuEndTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime): the device-time fields used after completion.

These references describe upstream APIs; acceptance evidence above comes from the actual local implementation and tests. No upstream example implementation was copied.
