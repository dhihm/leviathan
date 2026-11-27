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
