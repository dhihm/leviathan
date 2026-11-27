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
