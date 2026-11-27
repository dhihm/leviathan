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
