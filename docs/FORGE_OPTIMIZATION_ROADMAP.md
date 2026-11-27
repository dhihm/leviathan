# Forge optimization roadmap

Baseline: `603f4fa` on `experiment/router-overlap-phase53`. Measurements
below are scoped to the home M1 Max unless marked otherwise.

| Track | Baseline and status | Completion criterion |
|---|---|---|
| Single BUILD | First strict export measured 1,108 ms on the residual fixture. Instrumenting two model topologies is in progress. | Cold median within 5% of baseline, or document the tradeoff. |
| Separate-process warm rebuild | Repeated invocations currently initialize PyTorch and run all checks. | At least 30% lower median across five or more interleaved pairs on two models. |
| Same-session repeated BUILD | Three builds: 9.558 s independent, 4.452 s batched. Available through `dag_build_many.py`. | Preserve per-build validation and independent bundles. |
| Validation | Fresh bundle, helper identity, source/export probes, GPU and eager parity. | Keep these gates and exercise changed inputs, options, backends and corrupt bundles. |
| Serving | Existing Metal prepared worker reuses startup. Colab T4 previously executed a Forge DAG and generated five matching Stories15M tokens through the separate CUDA Llama path. | Measure startup separately from per-request latency. |

1. Measure both model topologies with fresh helpers and independent builds.
2. Try a bounded, explicitly selected reuse path. Unknown Python model
   provenance defaults to a fresh export.
3. Measure the largest remaining independently removable stage. Keep
   correctness checks, preserve old bundle identity, and record evidence.
4. Run Metal regression and validate CUDA on an available personal runtime
   when the change reaches that path.

## Stage 1 — Independent Metal BUILD experiment

An isolated packing prototype replaced per-float Python conversion with
contiguous little-endian tensor bytes. The experiment left candidate
qualification, artifact validation and eager parity active. Independent
CLI processes built fresh output directories. One first pair and five
subsequent pairs used alternating order on each topology. The second
topology is the repository's wider 128→512→256→64 MLP, with about 213K
parameters. Both models are synthetic fixtures.

| Model | First BUILD, baseline → prototype | Warm BUILD median and range, baseline → prototype | Warm lowering median |
|---|---:|---:|---:|
| Residual | 4,691 → 3,327 ms | 2,802 (2,620–3,555) → 2,942 (2,423–3,546) ms | 1.01 → 0.73 ms |
| MLP tower | 4,651 → 3,886 ms | 4,102 (3,248–6,444) → 3,653 (2,982–5,850) ms | 21.41 → 5.92 ms |

All twelve output pairs matched exactly. The first pair per topology is a
sample rather than a cold median. The warm changes were +5.0% and −10.9%;
neither satisfied the 30% gate. This experiment was discarded. The default
Metal source and bundle identity inputs still match `603f4fa` byte for byte.
Arbitrary Python models continue through strict capture. A future
cross-process reuse path needs a declarative model contract that binds
architecture, weights and all relevant dependencies before benchmarking.

## Stage 2 — CUDA preparation reuse

The CUDA DAG CLI accepts repeated `--input` arguments. It captures once and
uploads the validated weights once per invocation. It reports capture and
synchronized device preparation separately from each request. Each input
still receives new CUDA output, device timing and eager source parity.
The single-input report retains its existing fields and adds request wall
time. The prepared tensors have per-invocation lifetime.

```sh
python3 tools/forge_run/dag_cuda.py \
  --model tools/forge_run/examples/residual_block.py \
  --example-input /tmp/example-32.json \
  --input /tmp/input-32.json --input /tmp/changed-32.json
```

Validation coverage on the home Mac: existing Metal manifest corruption,
binding identity, diagnostic candidate failures and prepared-worker suites,
plus the CUDA CLI's unavailable-device branch and a prepared-tensor test
with repeated and changed inputs. Pytest reported 41 passed, 7 skipped and
11 passing subtests. The Swift CPU GGUF CLI regression also passed.

### Tesla T4 results

On personal Colab, compare three independent CUDA CLI invocations with
one invocation given the same three inputs (original, changed, original).
The first pair is separate and five subsequent pairs per model alternate
order. End-to-end times include Python startup. Capture includes model and
input validation. This CUDA CLI captures and executes a DAG without a
persistent BUILD bundle. Its timings are separate from Metal BUILD.

| Model | First group, independent → prepared | Warm group median and range, independent → prepared | Warm capture / device preparation / changed request medians |
|---|---:|---:|---:|
| Residual | 18,273 → 5,594 ms | 18,193 (17,702–18,596) → 6,049 (5,502–6,820) ms | 2,889 / 0.57 / 0.68 ms |
| MLP tower | 15,377 → 4,976 ms | 15,382 (15,244–15,466) → 5,584 (4,838–5,646) ms | 2,081 / 0.93 / 0.87 ms |

All 36 paired CUDA outputs matched exactly, and each request produced a
positive CUDA event measurement. The paired timing used `5054f8b`.
The follow-up `b40c4dc` restores the caller's matmul precision before
the eager source check and passed the three-test T4 regression.
The full Swift regression on the home Mac passed 2,063 checks.
A bundle built before the roadmap loaded with its original helper
binary and passed actual Metal/source parity.

**Next gate.** Prototype an explicit declarative model input with a closed
operator and dependency contract. Keep strict export for arbitrary Python
and compare five independent warm rebuild pairs before enabling reuse.

## Stage 3 — Opt-in declarative Metal BUILD

`tools/forge_run/dag_build_declared.py` accepts a versioned JSON declaration
and two separate files: a `leviathan-value-dag-1` graph and its packed
little-endian FP32 weights. The declaration contains exactly
`format: "leviathan-declared-dag-1"`, `graphSha256`, `weightsSha256`,
`inputShape`, and `outputShape`. Shapes are `[1, features]`. Dense
weights use row-major `[in, out]` followed by the bias, in graph order.
The supported operators are Dense, ReLU, and same-shape Add. A verified
legacy Metal bundle's graph and weight files can seed a declaration.

```sh
python3 tools/forge_run/dag_build_declared.py \
  --contract /tmp/declared.json --graph /tmp/model.graph.json \
  --weights /tmp/model.weights.f32 --example-input /tmp/example.json \
  --input /tmp/input.json --helper .build/debug/leviathan-dag-helper \
  --cache /tmp/private-forge-plans --out /tmp/declared-bundle
```

The optional cache has a caller-owned private directory. Its plan key
binds the schema, graph and weights, qualification input and reference,
helper binary, macOS version, NumPy version, CLI source and diagnostic
option. The helper checks its live device and toolchain and validates all
bundle files. A miss runs the existing candidate qualification and
benchmark. A hit uses its prior selection evidence and runs the requested
input in a fresh helper, gated against an independent NumPy CPU evaluator.
Each build computes CPU references from the declared parameters. A changed
request input reuses the tuning decision and gets fresh verification.
Changed qualification input or options select another key. A corrupted
cache record or rejected hardware binding triggers a new tune. Publication
uses exclusive rename after verification.

The strict-export Python and CUDA paths retain their existing entry points
and bundle validation. The declarative path has its own provenance contract
and emits Metal helper bundles. It does not execute Python model source.

### Independent process results

Five fresh-process triples per model alternated the execution order of
the `603f4fa` strict-export BUILD and the declarative cold and warm BUILDs.
Each cold build used an empty cache and each warm build used a primed
cache. Input, weights, helper and output checks were equivalent for the
declared graph. Strict export additionally establishes provenance for
executable Python. Times below include interpreter startup.

| Model | Strict-export baseline | Declared cold | Declared warm |
|---|---:|---:|---:|
| Residual | 2,794 (2,466–3,288) ms | 843 (544–1,032) ms | 497 (364–646) ms |
| MLP tower | 3,069 (2,858–4,267) ms | 1,059 (949–1,474) ms | 549 (464–635) ms |
| Handwritten-digit classifier | 2,877 (2,358–2,998) ms | 660 (569–853) ms | 435 (315–497) ms |

The digit model is a 64→32→10 classifier trained locally on the public
[scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
The benchmark freezes the learned weights before either path runs and
excludes training and one-time declaration export. All 15 triples had
exactly matching Metal outputs, fresh parity, a cold miss and a warm hit.
The warm path reduced the declared cold median by 41%, 48% and 34%,
respectively. It reduced the strict-export medians by 82%, 82% and 85%.
The cold medians also met the +5% bound. These results apply to the tested
Metal DAGs and do not indicate quantized LLM build performance.

## Stage 4 — Released checkpoint and CUDA portability

The [Badak MNIST MLP release](https://huggingface.co/building-an-inference-engine-from-scratch/mnist-mlp/tree/ff9e117d7f1f3902e55c68b29eda74700c9282c5)
is MIT licensed. Its complete inference graph is Dense(784→128), ReLU,
Dense(128→10). The pinned release includes normalized input rows and PyTorch
logits. The importer verifies the hashes of both safetensors files and
writes the declaration without executing publisher code.

```sh
revision=ff9e117d7f1f3902e55c68b29eda74700c9282c5
base="https://huggingface.co/building-an-inference-engine-from-scratch/mnist-mlp/resolve/$revision"
curl -fL "$base/model.safetensors" -o /tmp/mnist-model.safetensors
curl -fL "$base/reference.safetensors" -o /tmp/mnist-reference.safetensors
python3 tools/forge_run/import_public_mnist.py \
  --checkpoint /tmp/mnist-model.safetensors --reference /tmp/mnist-reference.safetensors \
  --out /tmp/mnist-declared
python3 tools/forge_run/dag_cuda.py --contract /tmp/mnist-declared/contract.json \
  --graph /tmp/mnist-declared/graph.json --weights /tmp/mnist-declared/weights.f32 \
  --input /tmp/mnist-declared/input-0.json --input /tmp/mnist-declared/input-1.json
```

Mac Metal: six alternating cold and warm pairs, separate processes,
empty plan cache for cold, same qualification row and a different
request row per pair. The 16 locally recomputed PyTorch rows matched the
published logits exactly. Every measured Metal output passed against the
corresponding PyTorch row. Cold and warm outputs were exactly equal.
Wall time includes process startup and excludes the one-time import.

| Metal BUILD | Median | Range |
|---|---:|---:|
| Empty cache | 1,038 ms | 921–1,298 ms |
| Verified cache hit | 661 ms | 419–805 ms |

CUDA shares declaration validation and the NumPy reference with Metal
and uses the existing prepared CUDA executor. Device tensors live for
one invocation. Metal tuning records are scoped to Metal and each CUDA
process prepares its own device tensors. Tests cover invalid hashes and
unreachable operations before CUDA initialization, plus the existing
Metal invalidation and corruption-recovery cases.

Colab Tesla T4, implementation `efc954c`: all 16 published normalized
rows ran through the declared CUDA CLI with positive CUDA events.
Maximum absolute error against the published PyTorch logits was
9.06e-6. A corrupted copy of the weights was rejected with no output.
The CUDA executor creates its own device weights each process;
the cold/warm table above measures Metal plan reuse.

The [dacorvo alternative](https://huggingface.co/dacorvo/mnist-mlp/blob/7ed1e96d182395fd7380ea8495b9eb2364ce9dc1/modeling_mlp.py)
returns Softmax probabilities and remains outside the current contract.
A future output-operator expansion should implement and test Softmax
end to end on both backends before accepting that model.

## Stage 5 — Final-axis Softmax

Version `leviathan-declared-dag-2` adds an explicit `outputActivation:
"softmax"`. Metal verifies the original DAG logits, then dispatches
Softmax with a separate GPU helper. For v2, `run.output` contains
probabilities and `run.logits` preserves the verified logits. CUDA includes
Softmax in its GPU event. An independent Float64 NumPy reference subtracts
each row maximum before exponentiation. The Metal helper accepts up to 64
rows; each DAG request remains a single row. Existing v1 bundles keep
their helper identity. The v2 cache key and bound source metadata include
the activation helper hash.

```sh
swiftc -O tools/forge_run/metal_softmax.swift -o .build/leviathan-metal-softmax
revision=7ed1e96d182395fd7380ea8495b9eb2364ce9dc1
curl -fL "https://huggingface.co/dacorvo/mnist-mlp/resolve/$revision/model.safetensors" \
  -o /tmp/softmax-mnist.safetensors
python3 tools/forge_run/import_public_softmax_mnist.py \
  --checkpoint /tmp/softmax-mnist.safetensors --out /tmp/softmax-declared
```

Use the earlier Metal BUILD command with `--softmax-helper
.build/leviathan-metal-softmax`, or the same declared CUDA CLI. The pinned
Apache-2.0 [dacorvo release](https://huggingface.co/dacorvo/mnist-mlp/tree/main)
uses Dense–ReLU–Dense–ReLU–Dense–Softmax. The benchmark used public
normalized MNIST rows from the Badak release as numeric inputs and
recomputed the published forward in PyTorch with the pinned dacorvo
weights. It makes no accuracy claim about preprocessing.

Six alternating independent-process Metal pairs gave cold median
1,511 ms (1,257–1,987) and verified warm median 788 ms (623–890).
Import time is excluded. All six paired probability vectors were exactly
equal and passed PyTorch comparison. Multiple rows, finite FP32 logits
through ±3e38, invalid numbers and ragged shapes were tested on Metal.
Changing either declaration version or the Softmax helper invalidates
selection reuse.

Warm BUILD costs ranked by measured medians:

1. Existing Metal DAG helper work: 369 ms (inside the process).
2. Declaration preparation: 86 ms, including validation and hashing.
3. Separate Softmax process: 78 ms including 61 ms Metal preparation.
   The GPU Softmax itself took 0.025 ms.

The implementation reuses the already verified weight digest to construct
the manifest. A 100-run hash microbenchmark on the 1.08 MB package
measured 0.90 ms for two hashes and 0.45 ms for one. This isolates hashing,
not a claimed BUILD speedup. Next, split DAG helper startup from GPU
preparation, then compare a shared prepared session against fresh-helper
validation. Keep the fresh helper on the default path until six
interleaved pairs show lower end-to-end time and matching old-bundle and
corruption gates. Kernel tuning is lower priority at the measured width.

Colab Tesla T4 at `d309c45` verified all 16 public input rows against
PyTorch CPU probabilities. Maximum absolute error was 1.19e-7. The
declared CUDA CLI also verified two extreme rows and rejected an FP32
overflow input before execution. The preserved pre-change Metal bundle
loaded with its recorded helper and passed source parity.

## Stage 6 — Independent-process Metal BUILD

Compared with `54efc02` on the home Mac, using six interleaved fresh-process
trials per configuration, model, and cache state. Cold trials had fresh
cache directories. Warm trials reused the corresponding verified cold
selection. The pinned Apache-2.0 MNIST checkpoint and a second residual DAG
with final-axis Softmax used fixed numeric inputs. Checkpoint conversion
was outside the timed command. Full process wall time includes Python
startup, bundle publication, fresh Metal execution, and independent CPU
comparison. Each paired GPU output matched within 2e-6 absolute error.

| Model and state | Baseline debug, ms | Release + one OpenBLAS thread, ms | Same + concurrent Softmax, ms | Combined change |
|---|---:|---:|---:|---:|
| Public cold | 1239 (1028–1432) | 715 (641–845) | 667 (583–828) | −46% |
| Public warm | 568 (472–591) | 404 (336–432) | 308 (289–390) | −46% |
| Residual cold | 815 (580–1018) | 803 (659–1390) | 687 (636–796) | −16% |
| Residual warm | 616 (497–640) | 468 (370–489) | 391 (306–549) | −36% |

Cells show median and min–max. The earlier 788 ms was a separate baseline
run. Both paired warm improvements exceed 20%, and both cold medians
improve. With the debug helper alone, overlapping Softmax was inconsistent
across models. With release and bounded OpenBLAS, overlapping improved
warm medians by a further 24% and 17%.

Warm component medians on the serial release arm, in milliseconds:

| Exclusive timed component | Public | Residual |
|---|---:|---:|
| DAG subprocess overhead (launch, exit, IPC) | 9.88 | 12.07 |
| DAG bundle validation and I/O, including Swift identity | 183.67 | 204.34 |
| DAG Metal preparation | 37.58 | 45.35 |
| DAG GPU execution | 1.01 | 0.11 |
| Independent CPU reference and comparisons | 0.64 | 0.68 |
| Separate Softmax subprocess envelope | 51.25 | 57.84 |

Phase medians are individually calculated. Additional time includes Python
startup, declaration preparation, report reading, dispatch/readback overhead
and publication. The isolated system Swift version subprocess took a median
129 ms. It is part of bundle validation and is not added to the table.
CPU comparisons exclude Softmax subprocess time. With concurrency the
reported activation wall time covers the remaining critical-path wait,
not the overlapped pipeline preparation.

To reproduce, build the release DAG helper with
`swift build -c release --product leviathan-dag-helper` and rebuild the
Softmax helper using the `swiftc -O` command in Stage 5. Prefix the declared
Metal BUILD command with `OPENBLAS_NUM_THREADS=1` when NumPy uses OpenBLAS,
pass `--helper .build/release/leviathan-dag-helper`, and add
`--concurrent-softmax`. The option starts pipeline preparation before the
DAG verification completes and sends only verified logits to the GPU for
Softmax. Device identity and independent reference checks remain in place.
The DAG bundle format and recorded helper matching rules are unchanged.
Use the originally recorded helper when loading an existing bundle.

The next measurement should split toolchain identity probing from bundle
reading and digest checks. Benchmark any proposed change against this
release profile and require the same identity and compatibility decisions
before changing the verification path. CUDA hardware was not rerun in
this Metal milestone.

## Stage 7 — Bundle validation audit and ResNet-18 stem

The Stage 6 release profile at `b747be4` was rerun on the home Mac.
Each cold BUILD used an empty cache and a fresh process. Its paired warm
BUILD used a fresh process and the resulting cache. All six pairs per
model passed the independent reference and matched GPU outputs within
2e-6 absolute.

| Model | Cold wall, ms | Warm wall, ms | Warm helper validation, ms |
|---|---:|---:|---:|
| Public MNIST Softmax | 825 (715–2233) | 464 (405–706) | 234 (203–264) |
| Residual DAG + Softmax | 677 (565–803) | 420 (298–488) | 201 (123–241) |

Cells show median and min–max. These are baseline repeatability runs,
without a second implementation arm. They do not demonstrate a 15%
speedup or establish a cold regression delta.

An isolated audit ran each phase 12 times on a warm bundle and the exact
release helper. The Python SHA-256 and JSON timings below are proxies
for work on the real bytes. They are separate from the production Swift
helper envelope above and do not include Swift typed decoding, semantic
validation, or process startup for the DAG helper.

| Isolated phase median, ms | Public | Residual |
|---|---:|---:|
| `/usr/bin/swift --version` subprocess | 107.14 | 121.40 |
| Read helper executable | 2.05 | 2.16 |
| SHA-256 of helper executable | 4.78 | 4.84 |
| Read six bundle files | 0.54 | 0.32 |
| SHA-256 of five bound payloads | 0.47 | 0.03 |
| Two JSON parses of each of three documents | 0.20 | 0.13 |

`loadValidatedBundle` and `loadModel` each read their bound files once
per invocation and reuse those bytes for validation. Python hashes the
helper for cache lookup and the Swift process independently hashes its
own executable to verify the recorded bundle identity. Removing one of
these checks would change the trust boundary. There is a second read
of the emitted bundle when writing a cache record on a cold miss, with
no effect on warm BUILD. The observed file-read costs do not justify
changing the validation path. No runtime speed patch was made.

### Next architecture unit

Target the FP32 NCHW stem convolution of TorchVision ResNet-18 with
IMAGENET1K_V1 weights. The published checkpoint has 11.7M parameters.
The first unit is a single image, three input channels, 64 output
channels, 7×7 kernel, stride 2, zero padding 3, groups 1, no bias.
[Official model and weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18)

The Metal declared BUILD CLI passed a first checkpoint-backed test of
the weight layout and padding. The checkpoint came from TorchVision,
with SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Load `conv1.weight` with PyTorch's weights-only loader, flatten the
three input-channel and 7×7 kernel dimensions, transpose to a
147×64 declared Dense weight matrix, and append 64 zero biases.
Use `torch.linspace(-1, 1, 3*11*13).reshape(1,3,11,13)` as input.
For output coordinates (0,0), (2,3), and (5,6), gather each patch from
the zero-padded image and present it to the CLI as one 147-element row.
All 192 Metal values matched the corresponding independent PyTorch
`conv2d(stride=2, padding=3)` outputs exactly on this input.

This exercise used patch extraction on the host and existing Dense on
Metal. Native Conv2D, whole-image execution, and CUDA remain open.
Next, add a shape-bound Conv2D declaration and compare an entire
non-square feature map, including border positions, against PyTorch.
Require exact shape, finite values, numerical parity, rejection of
unsupported geometry, and cache invalidation on kernel or stride
changes before adding batch norm and pooling. The Stage 6 validation
binding remains the baseline for that work.

## Stage 8 — Native Metal ResNet-18 stem convolution

The declared Conv2D CLI runs a single native FP32 Metal cross-correlation
over the full NCHW image. Version 1 accepts batch size 1 and contiguous
little-endian FP32 input and OIHW weights, with symmetric zero padding,
a uniform stride, groups 1 and dilation 1, and no bias. It bounds tensor
sizes and total work before dispatch. A separate NumPy float64 reference
checks every output value before publishing the result. The optional
cache records a dispatch plan bound to the declaration, weight digest,
helper, host OS, and a GPU registry ID checked by the helper. It does
not cache output or GPU execution, and no warm speedup is claimed.

The official IMAGENET1K_V1 ResNet-18 stem took input `[1,3,113,151]`,
including all padding borders, and produced `[1,64,57,76]`.
All 277,248 FP32 values matched PyTorch on the tested image. Metal
reported `conv2d_fp32` completion on Apple M1 Max. The checkpoint and
exact test are reproducible from the [Forge CLI guide](../tools/forge_run/README.md).
The existing value DAG schema and helper were unchanged.

For a full checkpoint the next slice is one complete residual block.
It needs intermediate NCHW values, 3×3 convolution chained with ReLU
and residual add, and either an inference BatchNorm op or verified
folding into convolution weights and bias. Then cover the stem 3×3
max pool with its padding semantics, stride-2 projection shortcuts
(1×1 convolution and normalization), four residual stages, global
average pooling, flatten, and the existing Dense classifier. Check
layer boundaries and final logits against PyTorch on a standard
224×224 input before making any whole-model support claim. Revisit the
bounded work limit for larger layers using measured GPU cost and add
separate CUDA parity when a GPU is available.

Existing DAG warm hits call `loadValidatedBundle` in a fresh helper.
It compares the complete recorded toolchain string against
`Forge.toolchainIdentifier`, which queries `/usr/bin/swift --version`
once per process. This probe is part of the current warm validation
contract. No replacement was validated that detects changes in toolchain
resolution and version under all of the current launch configurations.

## Stage 9 — Public ResNet-18 basic block

The declared residual CLI runs the two 3×3 convolutions, ReLU, skip add,
and final ReLU in one Metal command buffer. It keeps intermediate tensors
on the device. The versioned declaration supports an identity skip at
stride 1 or a 1×1 projection at stride 2. BatchNorm is folded from FP32
source tensors into FP32 filters and bias using FP64 arithmetic.
The declaration records the folding identifier, ordered epsilon values,
and a digest covering original weights, affine parameters, and running
statistics. The result records a separate folded digest. The CPU reference
executes direct FP64 convolution and unfused BatchNorm on every BUILD.
Cached dispatch metadata still requires current GPU and pipeline validation.

On Apple M1 Max, five fresh-process cold/warm pairs per block used the
pinned TorchVision IMAGENET1K_V1 checkpoint and deterministic FP32 input
of height 17 and width 23. Every output was compared against the original,
unfolded PyTorch block operations, including all padding edges and the
stride-2 boundary. Test fixture materialization and the separate PyTorch
comparison are outside the subprocess wall time. Within the subprocess,
the independent unfused CPU comparison runs before publication.

| Block | Output | Cold / warm wall medians, ms | Cold / warm GPU command medians, ms | Max PyTorch error |
|---|---|---:|---:|---:|
| `layer1.0`, identity | 1×64×17×23 | 310 / 327 | 1.226 / 1.230 | 6.68e-6 |
| `layer2.0`, stride-2 projection | 1×128×9×12 | 311 / 308 | 1.062 / 1.057 | 2.98e-6 |

Wall ranges in the same order were 306–1548, 318–348, 274–438, and
280–417 ms. GPU timing covers the aggregate Metal command buffer, which
includes the convolution and epilogue dispatches. These are correctness
runs at small spatial dimensions. A warm speedup was not established.
Source weight, running mean, and epsilon changes each produced cache
misses. Damaged plan records rebuilt while preserving output.

The next unit is the stem through the first residual stage at the public
224×224 input size. It needs folded stem BatchNorm, compatible padded max
pool, tensor chaining without per-block serializations, and benchmarks or
tiling to handle the existing per-block work cap. Then bind and validate
all eight blocks across the four stages, including output shapes at every
transition, global average pooling, flatten, and the Dense classifier.
Require layer-boundary parity and final logits against TorchVision before
claiming full ResNet-18 support. CUDA hardware remains unverified for these
new operators.

## Stage 10 — 224×224 ResNet-18 through layer1

The stage declaration runs folded stem convolution and ReLU, padded
max pool, then both layer1 identity blocks in one Metal command buffer.
It reuses the residual scheduler and keeps intermediates on the device
until the command completes. The CLI compares all four boundary
tensors with an independent, unfused FP64 reference before publication.
The pinned TorchVision checkpoint regression also compares those
boundaries with the unfused PyTorch sequence.

| Boundary | Shape | Maximum absolute error vs. PyTorch |
|---|---|---:|
| Stem after ReLU | 1×64×112×112 | 7.15e-7 |
| Max pool | 1×64×56×56 | 5.96e-7 |
| First identity block | 1×64×56×56 | 3.10e-6 |
| Second identity block | 1×64×56×56 | 1.01e-5 |

The fixed MaxPool2D contract uses kernel 3, stride 2, padding 1 and
dilation 1, floor output size, and values only. Out-of-bounds samples
are excluded. A direct Metal run with negative edge values verifies
the absence of implicit zero padding. Unsupported operation
descriptors are rejected before lookup. Source-statistic changes,
helper-binary changes, and damaged plan records produced misses.

On Apple M1 Max, five fresh-process cold/warm pairs measured these
medians and min–max ranges. Checkpoint extraction and the external
PyTorch comparison are outside the timed CLI. The CLI includes the
internal CPU reference, Metal execution, readback and publication.

| Phase | Cold median (range), ms | Warm median (range), ms |
|---|---:|---:|
| Full BUILD wall | 1688 (1654–1911) | 1689 (1689–1717) |
| Metal command buffer | 18.23 (9.06–18.46) | 12.38 (10.49–20.06) |
| Independent CPU reference | 1462 (1441–1467) | 1474 (1454–1518) |

CPU reference accounts for roughly 87% of wall time in this test.
The dispatch-plan cache did not establish an end-to-end warm speedup.

The next unit is layer2 through layer4, reusing the tested projection
block with channel transitions and stage-size guards. Then add
global average pooling, flatten and the Dense head. External boundary
snapshots and unfused verification must remain available for complete
224×224 logits parity. Measure an independent CPU reference that uses
optimized convolution before attempting to improve BUILD latency.
The timed stage does not establish end-to-end model support or CUDA
parity.

## Stage 11 — Full public ResNet-18 on Metal

The declared FP32 route executes the full IMAGENET1K_V1 ResNet-18 at
224×224 in one Metal command buffer. It extends the existing schedule
through layers 2–4, 7×7 global average pooling, channel-vector flatten
and the 512-to-1000 classifier. Every BUILD checks all snapshots
against the independent FP64 reference with the existing absolute
and relative tolerances of 2e-4. The checkpoint test additionally
compares every boundary and every logit with an unmodified TorchVision
model in evaluation mode. Inputs were a deterministic ramp and NASA's
public-domain Eileen Collins photograph, preprocessed with the
official TorchVision weights transform.
[Image provenance](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

| Boundary | Shape | Synthetic maximum error | Photo maximum error |
|---|---|---:|---:|
| Stem | 1×64×112×112 | 7.15e-7 | 2.88e-6 |
| Max pool | 1×64×56×56 | 5.96e-7 | 2.26e-6 |
| First residual block | 1×64×56×56 | 3.10e-6 | 3.81e-6 |
| Layer1 | 1×64×56×56 | 1.01e-5 | 9.03e-6 |
| Layer2 | 1×128×28×28 | 5.19e-6 | 6.26e-6 |
| Layer3 | 1×256×14×14 | 7.39e-6 | 5.25e-6 |
| Layer4 | 1×512×7×7 | 1.87e-5 | 2.57e-5 |
| Pooled features | 1×512 | 2.26e-6 | 4.05e-6 |
| Logits | 1×1000 | 3.70e-6 | 1.00e-5 |

Metal and PyTorch ordered top-5 class indices matched exactly.
Synthetic: 623, 111, 499, 446, 812. Photo: 758, 823, 542, 570, 560.
This validates inference parity and does not measure classification accuracy.

Five fresh-process cold/warm pairs per input on Apple M1 Max, with
`OPENBLAS_NUM_THREADS=1`, gave the following medians in milliseconds.
Cold runs used an empty plan cache. Warm runs reused that plan and
repeated all correctness checks. File extraction and the external
TorchVision oracle are outside the CLI wall time.

| Input | Full BUILD cold / warm | Independent CPU cold / warm | GPU command cold / warm |
|---|---:|---:|---:|
| Synthetic | 600 / 584 | 185 / 190 | 43.79 / 34.70 |
| Photo | 738 / 655 | 239 / 223 | 31.74 / 29.99 |

Wall ranges were 590–1701, 581–796, 597–1415, and 585–699 ms in
table order. Timing variability precludes a strong claim about the plan
cache's wall-time benefit. In an isolated synthetic-input comparison,
the original direct FP64 convolution reference took 6630 ms and the
matrix-multiplication reference took 212 ms. All nine boundary tensors
agreed at 1e-10 absolute/relative tolerance between CPU implementations.
Production validation still runs unfused convolution and BatchNorm.

The next candidate is to profile BUILD outside the CPU reference and GPU
command buffer. Separate the 45 MB source read and digest, folding and
staging I/O, helper startup and pipeline creation, and snapshot readback.
Consider optimizing repeated staging only if complete boundary checks,
original source provenance and cache invalidation remain intact.
GPU kernel tuning and CUDA parity need separate measurements.

## Stage 12 — Full ResNet-18 BUILD scheduling and packing

Full-model BUILD starts the Metal subprocess and computes the independent
unfused CPU reference concurrently. Both must complete before any boundary
is accepted or published. Folding now views the immutable original
bytes and assembles the folded filters and classifier in a single join,
avoiding an original-weight slice copy and a second full folded-weight
copy. Source SHA-256, folding provenance, output checks, and tolerances
are unchanged.

Six pairs per input alternated the serial and optimized routes on Apple
M1 Max. Both routes used the same instrumented helper and
`OPENBLAS_NUM_THREADS=1`. The comparator retained the `20204fc` ordering
and packing with phase timers added. Each arm used a separate fresh
plan cache for cold and reused it for warm. Every process matched all
nine external PyTorch boundaries and the ordered top-5. Both arms
produced identical folded and boundary digests within each pair.

| Input / state | Serial median (range), ms | Optimized median (range), ms | Median change |
|---|---:|---:|---:|
| Synthetic cold | 778 (625–974) | 536 (501–614) | −31.1% |
| Synthetic warm | 746 (692–885) | 564 (451–621) | −24.4% |
| Photo cold | 655 (567–789) | 510 (433–705) | −22.1% |
| Photo warm | 662 (547–789) | 563 (463–608) | −14.9% |

Both cold results meet the +5% maximum regression criterion. The
synthetic warm run exceeds the 20% goal and the photo warm run misses it.
Checkpoint conversion, measured separately as packing, hashing and
writing the source payload after model loading, took 76 and 146 ms in
the two final benchmark setups. In a separate six-run measurement,
checkpoint reading and deserialization had a median of 17 ms and model
construction and state loading had a median of 175 ms. These timings,
image preparation, and the external PyTorch checks are outside BUILD wall.

| Warm phase median, ms | Synthetic serial → optimized | Photo serial → optimized |
|---|---:|---:|
| Source weights read | 9.99 → 8.78 | 7.12 → 7.75 |
| Original SHA-256 | 24.98 → 23.77 | 22.31 → 22.45 |
| BatchNorm folding and packing | 95.17 → 53.90 | 64.32 → 58.01 |
| Temporary input writes | 39.07 → 33.49 | 18.58 → 23.06 |
| Metal pipeline preparation | 2.61 → 2.71 | 2.32 → 3.06 |
| Metal command buffer | 37.03 → 30.76 | 38.18 → 40.44 |
| Independent CPU reference | 221.13 → 241.66 | 199.81 → 226.45 |

The phase timers cover different work. CPU reference can overlap pipeline
preparation and GPU execution in the optimized arm, so these medians
cannot be summed to recover wall time. Warm Metal pipeline preparation
was a few milliseconds in this run. Next, isolate helper-side input I/O,
memory transfers and snapshot publication, then evaluate a staging
change that preserves the original digest and every boundary check.
CUDA performance was outside this comparison.

## Stage 13 — Folded-weight cache verdict

The production baseline is `d1873d9`. On Apple M1 Max, eight balanced,
seeded randomized pairs per input compared that exact CLI with a scratch
folded-weight cache candidate. Each arm used a separate empty directory
for cold BUILD and reused it for warm BUILD. Every BUILD ran in a new
process with one OpenBLAS thread and the same Metal helper. The synthetic
ramp and pinned NASA photograph matched the prior milestone's inputs.
All runs retained the original source digest, the independent unfused
CPU oracle, and comparison with unmodified TorchVision at all nine
boundaries and for ordered top-5. The folded and output digests matched
between arms.

| Input / state | `d1873d9` median (range), ms | Candidate median (range), ms | Median change |
|---|---:|---:|---:|
| Synthetic cold | 641 (503–1665) | 685 (524–1283) | +6.7% |
| Synthetic warm | 606 (485–748) | 622 (511–698) | +2.7% |
| Photo cold | 567 (444–630) | 587 (525–625) | +3.6% |
| Photo warm | 556 (502–748) | 562 (499–588) | +1.1% |

Each input split four faster and four slower warm pairs. The median of
the paired percent changes was −1.6% synthetic and +0.5% photo,
distinct from the percent changes between medians above. Neither
summary supports a 20% win. Synthetic cold also breached the +5% limit.

The prototype keyed the cache on the complete declaration, including
source bytes digest, epsilon, op descriptor and folding identifier,
plus the Python implementation and NumPy/machine identity. It checked
the cached byte digest and verified the FP32 result of folding each
tensor against the original source, bit for bit. The latter catches
a forged digest on a changed payload and preserves binding independently
of a single test input. Original-weight and epsilon changes caused
misses, and a deliberately modified, rehashed payload caused a rebuild
followed by a verified hit. The candidate used about 44.6 MiB per entry.

The original folding segment took 71.5 ms synthetic and 60.7 ms photo
on warm baseline runs. Cache read, hash and exact folding verification
together took 75.6 and 82.7 ms. Skipping serialized copies did not
offset those costs, and cold runs added a cache write. The largest
measured warm component on the baseline remained independent CPU
reference at 240–241 ms, compared with GPU command time of 36–40 ms
and pipeline preparation around 3 ms. Temporary write times varied
widely across pairs.

The prototype was discarded. The retained improvement is Stage 12's
same-session comparison of `d1873d9` with the serial `20204fc` route:
24.4% synthetic and 14.9% photo warm. There is no additional cumulative
gain from this milestone. The historical serial medians were measured
in a different session and are not a new comparison against today's
baseline. Keep the production path at `d1873d9` behavior and close this
folded-cache candidate. A later BUILD optimization should measure
snapshot or helper-side staging separately and preserve exact source
binding and all parity gates.

## Stage 14 — ResNet-18 hardware portability

The earlier `dag_cuda.py` loader checks a versioned flat value DAG and
its separate graph file. It only dispatches Dense, ReLU, Add, and final
Softmax. The ResNet declaration instead uses bounded NCHW shapes, OIHW
weights, inference BatchNorm statistics, strided projection, spatial
pooling, and a flatten/classifier boundary. The old loader cannot read
this contract. The portable runner uses the ResNet declaration validator
and original tensor decoder shared with Metal, while dispatching those
operations through PyTorch. The preexisting CUDA value-DAG CLI and
Metal bundle formats retain their separate entry points.

On the home Mac CPU, the stage1 and full ResNet-18 CLIs used the pinned
TorchVision checkpoint and the public astronaut input. The stage1
boundaries matched TorchVision exactly. The full run matched all nine
boundaries within tolerance, with a maximum logit error of 1.43e-6 and
the same ordered top-5. Both variants rejected an incorrect original
weights digest. An explicit CUDA request on this Mac failed without
CPU fallback. The NumPy FP64 reference runs on CPU for either backend
and is reported separately from accelerator evidence.

The existing free Colab runtime assigned a Tesla T4 (compute capability
7.5). Implementation `725bcda` ran the stage1 test using only the public
checkpoint and photograph. Its input and original weight SHA-256 values
matched the home Mac CPU stage1 run. All four CUDA device buffers
matched the original CPU TorchVision outputs at the unchanged
tolerance. The test also rejected the wrong source digest.

| Boundary | Shape | T4 max absolute error vs. TorchVision |
|---|---|---:|
| Stem after ReLU | 1×64×112×112 | 2.58e-6 |
| Max pool | 1×64×56×56 | 1.79e-6 |
| First block | 1×64×56×56 | 2.26e-6 |
| Stage1 output | 1×64×56×56 | 8.79e-6 |

The CLI recorded a positive CUDA event duration of 1148.45 ms on the
fresh process and separately reported 111.52 ms for its CPU reference.
This is execution evidence, not a steady-state benchmark. The test
passed in 18.91 s including the external oracle and rejection test.
The full-model CUDA route remains unverified on hardware. The home Mac
passed three portable checks, seven existing Metal/value-DAG checks,
and an explicit historical bundle load.

## Stage 15 — Full ResNet-18 portability closed

The existing free Colab Tesla T4 (compute capability 7.5) ran commit
`20eb817` with the same FP32 declaration, source weights, and 224×224
inputs used on the home Mac. Each input used a new CLI process with
the same resident device weights across one first and five warm
executions. Each execution verified all nine boundaries against the
independent NumPy FP64 reference. Repeats produced identical boundary
hashes, and those tensors matched the unmodified TorchVision CPU
model and exact ordered top-5. A separate untimed dispatch audit
observed 72 PyTorch operations per input with every tensor operand
and result on CUDA. The explicit CUDA CLI has no CPU fallback.
The test also rejected an incorrect source digest for each input.

| Boundary | T4 max absolute error, synthetic | T4 max absolute error, photo |
|---|---:|---:|
| Stem | 7.00e-7 | 2.58e-6 |
| Max pool | 5.66e-7 | 1.79e-6 |
| First block | 1.31e-6 | 2.26e-6 |
| Layer1 | 4.71e-6 | 8.79e-6 |
| Layer2 | 2.92e-6 | 4.44e-6 |
| Layer3 | 4.29e-6 | 3.34e-6 |
| Layer4 | 1.38e-5 | 1.40e-5 |
| Pooled features | 3.10e-6 | 5.36e-6 |
| Final 1000 logits | 3.58e-6 | 8.11e-6 |

Top-5 class indices were 623, 111, 499, 446, 812 for synthetic and
758, 823, 542, 570, 560 for the photograph.

| T4 timing, ms | Synthetic | Photo |
|---|---:|---:|
| CUDA availability and context initialization, host wall | 110.86 | 107.63 |
| Tensor materialization, transfer and synchronization, host wall | 191.53 | 195.89 |
| First forward, CUDA event | 360.59 | 288.13 |
| Warm forward, CUDA event median (five runs) | 4.04 | 4.03 |
| Warm forward, CUDA event range | 4.02–4.27 | 4.03–4.33 |
| Independent NumPy CPU reference, host wall | 218.97 | 222.10 |

The first forward includes lazy library and kernel startup effects
inside the event interval. The warm events measure the forward
stream with resident tensors and exclude CPU comparison, boundary
readback, archive publication, and process startup. Stage 12's home
Mac Metal numbers were 30.76 and 40.44 ms of command-buffer time,
and 564 and 563 ms of fresh-process warm BUILD wall time, for the
same inputs. Those were separate Metal processes and their timings
do not measure the same execution boundary as T4 warm forwards.
Stage 14 established home Mac CPU parity, without a matched CPU
forward-only benchmark. CPU reference times above are oracle costs.

Completion is scoped to batch-1 FP32 ResNet-18 with the pinned weights
and tested inputs. CPU, Metal and T4 preserve the declaration and
source digest and pass all boundary and top-5 gates. For the next Forge
LLM milestone, baseline the existing `llama-generate` CLI on the pinned
public Stories15M GGUF and its fixed prompt. Time GGUF identity and
model load, backend preparation, prefill, first token, and steady
decode separately across fresh processes and a reused model session.
Require the existing exact greedy-token sequence and CUDA evidence
for each decode before attempting any preparation or cache reuse.

## Stage 16 — Stories15M LLM generation baseline

The real `llama-generate` CLI at `82ee63f` emits a structured stderr
record only with `--timing json`. It keeps the normal output and
untimed CUDA v1 ABI. CUDA timing uses the v2 ABI. The timing contract
and reproduction command are in the [Forge guide](../tools/forge_run/README.md).

On the home Mac Apple M1 Max, the Release scalar CLI ran the pinned
public F32 Stories15M GGUF six times per prompt, each in a fresh
process. The file was not evicted from the OS cache. Medians and
min–max ranges are in milliseconds. The decode total covers four
tokens after the first greedy token.

| Phase | Prompt `1` | Prompt `1,9038,2501,263` |
|---|---:|---:|
| GGUF map and parse | 6.61 (6.37–6.87) | 6.52 (6.22–6.80) |
| Backend and KV preparation | 95.82 (85.89–124.35) | 83.56 (79.63–88.88) |
| Prompt prefill | 6.16 (5.41–7.62) | 17.51 (16.78–18.18) |
| First token selection, including CPU output projection | 6.96 (6.16–8.29) | 5.84 (5.70–7.02) |
| Later decode total | 40.27 (37.50–49.22) | 38.13 (36.66–39.39) |
| Overall process wall | 175.39 (159.42–204.71) | 164.75 (157.55–171.98) |

All timed stdout matched the untimed stdout byte for byte.
The first prompt produced `9038,2501,263,931,29892`, and the second
produced `931,29892,727,471,263`. The synthetic fixture also passed
single-token EOS, multiple-token generation, invalid token and
truncated-file rejection, and absence of timing on the default path.

Backend preparation is the largest measured CPU phase. Target at
least 30% less host preparation for both prompts, from 95.82 to at
most 67 ms and from 83.56 to at most 59 ms in a same-session paired
baseline. First isolate tensor materialization from per-request KV
setup, then measure reuse of a validated loaded model with a new
generation state per request. Require no more than 5% regression in
fresh-process wall time, exact greedy IDs and unchanged failure
behavior. Do not infer throughput for larger GGUFs from Stories15M.

The existing free Colab T4 ran the same CLI commit on Linux with
the C++ adapter linked to the official llama.cpp b12808 CUDA 12.8
shared libraries. The release tag resolves to the repository's pinned
`ed2153e708de813c35559102aed43b3085179c27` commit. The archive digests
were verified and Colab's existing NVIDIA driver path was retained.
The slower dedicated static dependency build was stopped before the
measurement. [Pinned llama.cpp CUDA release](https://github.com/ggml-org/llama.cpp/releases/tag/b12808)

| T4 phase | Prompt `1`, median (range) ms | Prompt `1,9038,2501,263`, median (range) ms |
|---|---:|---:|
| llama.cpp model load and upload | 261.48 (239.83–285.62) | 233.72 (229.25–249.01) |
| Adapter, backend and context preparation | 194.26 (173.32–257.49) | 175.38 (171.99–181.74) |
| Synchronized prompt prefill | 136.27 (124.29–175.44) | 133.15 (128.58–139.78) |
| Logit validation and first-token selection | 0.157 (0.139–0.200) | 0.153 (0.141–0.163) |
| Later decode total | 6.54 (6.20–7.56) | 6.56 (6.07–7.77) |
| Overall process wall | 853.55 (751.28–875.81) | 758.28 (739.07–771.91) |

Each prompt had six fresh-process samples after a device smoke run.
One initial probe measured 104.38 seconds inside prefill while the
source build was also running. The table does not characterize a
clean driver-cache first start. Both prompts matched the pinned CPU
greedy IDs and untimed stdout. The adapter reported a completed CUDA
matrix multiply in every decode call, and llama.cpp reported offloading
7/7 layers in the smoke run. With the device hidden, both timed and
untimed explicit CUDA requests failed without generating tokens.
Host orchestration and greedy output selection are part of the backend.

On T4 the largest measured phase was model load/upload, followed by
backend and context preparation. The next CUDA experiment should keep
a validated model resident across requests and report model reuse
separately from fresh context creation. Aim for at least 30% lower
warm request time to the first token versus newly opened fresh
processes at the same measurement boundary. Retain per-decode device
evidence and exact token parity, and verify that identity changes
reject reuse.
