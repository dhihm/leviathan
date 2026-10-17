# Generic Python-to-Metal Forge runtime: implementation and validation

This increment connects a trusted, user-authored Python `nn.Module` and an example input to actual `torch.export(strict=True)`, the existing restricted ATen validator, dense/ReLU lowering, hardware-dependent candidate generation, baseline-first qualification and GPU measurement, a bound plan bundle, and execution of a real input. A fresh Python process can reload the saved selection and run a different input. No inference-time agent participates.

The implementation was authored through the configured agy Opus tool. Codex applied its complete JSON patch mechanically and independently ran builds, GPU acceptance, source-PyTorch comparisons, compatibility tests, and prepared documentation. The provider reported an output-limit error; the raw response contained a truncated first object and a complete second object. Only the complete second object was applied. A separate GLM draft failed fresh reload and is preserved outside this implementation; none of that draft is included here.

## Scope

Static FP32 `[1, INPUT_DIM]`, input width 1..4096, biased `aten.linear.default` and `aten.relu.default` in a single chain, at most 64 operations and 16 MiB of weights. These examples have random initialization. This is a generic compiler/runtime milestone for this bounded operator subset, not a pretrained language model, Transformer compiler, or performance comparison with MLX or Ollama. Gemma experiments remain separate and unchanged.

The Python file is trusted executable code, not sandboxed model data. `INPUT_DIM` and `build(seed)` remain the input contract. Unsupported ATen operations report their operation and node. Other architectures require explicit operator/representation work rather than model-specific engine edits.

## Architecture

- `tools/forge_run/leviathan_forge.py`: Python `build`/`run` commands and APIs; accepts real example/input JSON, captures with `torch.export`, reuses the existing frontend validator, serializes real weights, checks source and output parity.
- `Sources/LeviathanCore/ForgeBaseline.swift`: generates the legal set through existing `Forge.candidates`, verifies the unfused baseline first, qualifies candidates on the same four inputs, and benchmarks all configurations on the same example input. Compilation/upload are outside GPU measurements.
- `Sources/leviathan-forge-helper/main.swift`: a small Swift executable that forges and validates a persisted selection, executes the original Metal kernels, and returns output, command-buffer GPU time, and host-derived kernel sequence.

Numerical qualification uses absolute tolerance 1e-4 plus relative tolerance 1e-4 against scalar/Accelerate and the source-PyTorch example reference; it is not bit identity. Actual-input parity uses the same absolute/relative tolerances. An optimization must pass qualification and beat the verified baseline median by more than 2%. Otherwise the baseline is selected. The diagnostic failure switch perturbs candidate output after an actual GPU run to exercise rejection; it never bypasses baseline verification. Tiny-model timings are noisy: this threshold does not establish statistical significance or a general speedup.

The bundle binds graph, real weights, source model copy, example input, frontend/backend source digests, helper executable content, shader fingerprint, hardware, toolchain, shape, and selection. The SHA-256 record checks integrity; it is not an authenticity signature. Changed identities reject before model dispatch; they do not silently retune. Runtime recreates Metal pipelines from source. This is a reloadable plan/weights bundle, not a persisted `.metallib`. The Python `run` path also recaptures the trusted model to validate graph/weights and evaluates it on CPU for parity; this validation cost is included in its end-to-end time. It is not a low-latency serving API.

## Atomic publication and early input repair

Pre-publication review found that the initial build finalized its bundle
before actual-input parity, and that Tensor inputs bypassed early JSON
checks. These were reproduced as RED, then fixed through the configured
external Opus author. The first repair response was complete JSON despite
a provider output-limit status; the final refinement and isolated
negative-test correction returned SUCCESS. All source/test replacements
were applied mechanically. No Swift kernel, graph operator, numeric
tolerance or weight limit changed.

The actual-input helper execution and source parity now run from a hidden
stage before the final atomic no-replace directory rename. Handled parity,
late helper, report and publication failures leave no new final bundle.
Their raw stages are retained with an explicit failure marker. Existing
bundles and foreign concurrent outputs are preserved. CPU/FP32/dense
strided/finite/static-shape validation is shared by JSON and Tensor APIs;
accepted inputs are copied before use. A shape-specific gate validates the
trusted model's declared input dimension before `build(seed)`.

Focused RED/GREEN tests include injected finite output errors after actual
GPU execution, late helper errors, existing bundle preservation, concurrent
empty/foreign output creation, report failure/rollback, 14 invalid-input
cases across build-example/build-input/run, and successful build/report
plus different-input fresh-process reuse. The legacy source-negative test
now supplies a correct bundle input shape while changing only the model,
so it reaches the intended model-source rejection gate.

Only the bundle directory is the atomic commit point; optional report and
bundle publication are not a two-artifact crash transaction. Report rollback
checks its inode/content before removing its own file, with a documented
small check/unlink race window. Trusted Python execution remains unsandboxed.

## Actual Mac results

Apple M1 Max, 32 GPU cores, 32 GiB unified memory, macOS 26.2, Swift 6.2.1, existing Python 3.11 / PyTorch 2.9.1. No dependency or model downloads. Two original example models reached the runtime without changes to the engine:

- `mlp_tower.py`: 128 -> 512 -> 256 -> 64, two ReLUs; 855,296 weight bytes.
- `bottleneck_stack.py`: 24 -> 96 -> 48 -> 96 -> 10, three ReLUs, a ModuleList loop and final activation; 50,920 weight bytes.

Release helper and full debug builds passed. The final real-GPU acceptance and focused failure suite passed all 29 tests with no skips (17 original acceptance tests and 12 atomic-publication/input tests). The existing debug regression suite passed 778 tests with zero failures. Tests cover actual-input/source-PyTorch parity, fresh-process reuse, forced rejection fallback, illegal selection even with a recomputed digest, corrupt JSON/weights, model/source/helper/device mismatch, bounded inputs, no overwrite, and compatibility with the existing frontend.

Historical kernel-selection evidence from before the Python failure-handling repair recorded one tower round and three predefined bottleneck rounds. Kernel/Swift implementation bytes are unchanged by the repair. Each used identical example input and weights for baseline/candidate timing, three warmups and eleven timed GPU runs. These are internal selection medians, measured baseline first; they are not paired end-to-end performance trials. A separately launched Python process used a different input and an independent source-module comparison.

| Case | Baseline GPU median (us) | Selected GPU median (us) | Reduction in this round | Fresh-input max abs error | Orchestrator call to output (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tower | 706.125 | 564.750 | 20.02% | 1.043e-07 | 919.06 |
| bottleneck-1 | 193.667 | 180.583 | 6.76% | 4.470e-08 | 1071.51 |
| bottleneck-2 | 193.625 | 176.459 | 8.87% | 4.470e-08 | 945.19 |
| bottleneck-3 | 193.041 | 179.500 | 7.01% | 4.470e-08 | 907.13 |

Separate full-process timing included Python launch/imports, re-capture, helper execution, CPU reference comparison and JSON output: tower 3.506 s, bottleneck-1 3.061 s. The table above measures the orchestrator function to GPU output; it excludes Python startup/imports and the final reference comparison. These were fresh processes with existing OS caches, not cold-machine trials. This validation-heavy path does not yet provide low-latency serving.

The tower selected width 64 with Dense/ReLU fusion. Bottleneck selected width 10 with fusion; device/output limits collapsed all requested widths to 10. Successful runs returned completed command buffers and positive GPU timestamps. Kernel-sequence/count fields are derived from the selected host lowering, not a GPU profiler trace. GPU execution time and Python/process wall time are separate metrics.

Non-optimization outcomes were also verified:

- Forced qualification failure: all three tower optimization candidates were rejected; the verified width-32 unfused baseline ran correctly and reloaded. The baseline median was 705.000 us in this acceptance round. This is a diagnostic rejection, not a natural performance regression.
- Independent one-linear 8 -> 4 acceptance fixture: no alternative candidate was legal; the ordinary baseline selection ran with a 13.375 us GPU median and source-PyTorch max abs error 5.960e-8. No optimization or speedup is claimed. This test fixture is external validation evidence, not an additional engine specialization.

## Reproduction

See [the command/API documentation](../tools/forge_run/README.md). On the test Mac the existing Torch environment is `/opt/anaconda3/bin/python`.

```sh
swift build -c release --product leviathan-forge-helper
LEVIATHAN_FORGE_KEEP_TEST_OUTPUT=1 /opt/anaconda3/bin/python -m unittest discover -s tools/forge_run -p 'test_*.py' -v
swift build
.build/debug/leviathan-tests
```

Raw source-author responses, failed draft, build logs, acceptance artifacts and independent run reports are retained locally outside the repository. No commit, push, PR, or merge is part of this implementation handoff.
