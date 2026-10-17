# Loaded DAG runtime

`LoadedRuntime` reuses one validated Python model and one Metal helper process for multiple actual inputs. It loads and exports the model once, validates a bound artifact once, compiles Metal pipelines once and uploads weights once. Each request replaces the input buffer, executes the saved selected plan, reads the GPU result and checks it against the cached source PyTorch model. No agent, export capture, model construction, candidate search or helper launch occurs per request.

This is the same bounded static FP32 Linear/ReLU/same-shape Add experiment documented in [DAG Forge](DAG_FORGE_RUNTIME.md). It is not a general Transformer/LLM runtime.

## Use

Build the existing helper and explicitly forge a bundle using this source revision:

```sh
swift build -c release --product leviathan-dag-helper
/opt/anaconda3/bin/python tools/forge_run/dag_forge.py build \
  --model tools/forge_run/examples/residual_block.py \
  --example-input /tmp/example-32.json \
  --input /tmp/input-32.json \
  --out /tmp/my-new-loaded-bundle
```

Then keep the context open in a Python process:

```python
import sys
import torch
sys.path.insert(0, "tools/forge_run")
from dag_runtime import LoadedRuntime

with LoadedRuntime(
    "/tmp/my-new-loaded-bundle",
    "tools/forge_run/examples/residual_block.py",
) as runtime:
    for seed in range(10):
        x = torch.randn(1, 32, generator=torch.Generator().manual_seed(seed))
        report = runtime.run(x)
        print(report["output"], report["timing"]["requestWallMillis"])
```

The helper path must match the binary that forged the artifact. Pass `helper=...` when choosing an explicit debug/release binary. Old artifacts reject changed source or binary identity; re-forging is explicit.

## Lifetime and correctness

The constructor retains all existing artifact, model-source, backend/frontend, weight/layout, example, version, shape, helper, shader, toolchain, device and selection checks. Shared loading functions serve both the original fresh-process command and the reusable runtime. Existing atomic build publication and unsupported-operation rejection remain intact.

A session pins a verified memory snapshot. Later disk changes do not change its loaded weights, graph or cached model. A fresh session validates the files again and rejects mismatches. This avoids claiming per-request disk validation while quietly skipping it.

Every request validates a detached input snapshot as finite dense CPU FP32 `[1, D]`, validates GPU completion and evidence, and uses the original source-parity criterion `abs(diff) <= 1e-4 + 1e-4 * abs(reference)`. Consumer-safe fusion and distinct per-node buffers preserve negative values and multiple consumers. All needed node outputs are recomputed for each input.

Prepared Metal objects share an execution lock across their struct copies. Input upload, dispatch, GPU completion and readback are serialized, preventing another call from replacing buffers mid-run. The Python context also serializes requests and source parity. Concurrent callers are supported through serialization; parallel GPU request execution is not claimed.

Invalid client tensors fail before IPC and leave the worker usable. Wire errors, EOF, timeouts, malformed results or parity failure close the worker and poison the context. There is no restart, retuning or CPU fallback. Use the context manager or call `close()`.

The worker communicates only over bounded local stdio: `serve --bundle DIR`, 128 KiB request lines and 1 MiB client response lines, sequential integer request IDs, strict request fields and finite FP32 values. It opens no network port. IPC uses deadlines. Trusted model Python still executes with ordinary local privileges; an IPC timeout is not a sandbox or a limit on arbitrary trusted Python execution.

## Why fresh runs took about three seconds

Three fresh processes reused an existing artifact on the M1 Max. Median component measurements were:

| Component | Median |
| --- | ---: |
| Whole Python process, including launch/import/finalization | 2,936.75 ms |
| Module imports after entering the profiling script | 1,156.96 ms |
| Complete fresh `run_bundle` API | 1,265.06 ms |
| Source hashing, within that API | 9.93 ms |
| Model load/construction, within that API | 2.34 ms |
| Export capture/lowering/differential validation, within that API | 982.05 ms |
| Helper process, within that API | 267.03 ms |
| Helper validation, within helper time | 194.27 ms |
| Metal preparation, within helper time | 56.06 ms |
| Actual GPU command buffer, within helper time | 0.108 ms |
| Source CPU parity function, within that API | 0.36 ms |

Rows are nested measurements and must not be added together. Median component values need not sum exactly. Process launch/finalization and other uninstrumented overhead account for the remaining wall time. The GPU was not responsible for the multi-second delay.

## Reusable request measurements

Each source example was explicitly forged with the release helper. A fresh Python process then constructed one session and processed 45 distinct real inputs, keeping the same helper PID throughout. The first five were treated as warmups; the remaining 40 were measured. CPU source parity remained enabled for every request.

| Model | Complete request p50 | Complete request p95 | GPU command p50 | Source parity p50 | Maximum source error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual block | 0.994 ms | 1.728 ms | 104.354 µs | 0.121 ms | 2.38e-7 |
| Projected branches | 0.744 ms | 1.012 ms | 125.709 µs | 0.101 ms | 4.77e-7 |

Cold constructor validation/capture still took 791/689 ms respectively, followed by helper launch/ready time of 276/239 ms. Python imports were also incurred once. This improvement amortizes setup; it is not a claim that the dense kernels became thousands of times faster. The table describes small local randomly initialized modules and a single session per model, not a serving throughput or token-latency benchmark.

## Validation evidence

The prior DAG audit checked four actual fanout/duplicate-edge models over sixteen additional distinct-input runs, with zero observed source error and unchanged artifacts. NaN weights were rejected. An independent Metal test exercised eighty simultaneous calls across copied prepared objects and verified invalid inputs/widths do not overwrite the last valid input.

The runtime acceptance suite verifies one-time model/capture setup, repeated distinct inputs, exact negative alias canaries, serialized concurrent callers, immutable loaded snapshots, invalid input handling, worker cleanup and poisoning after corruption of an actual completed GPU readback. Direct stdio tests cover multiple requests in one pipe chunk, decimal-to-FP32 conversion, malformed and oversized requests and truncated EOF. Existing DAG and legacy acceptance gates are retained.

Run with the already installed Python testing environment:

```sh
swift build -c debug
/opt/anaconda3/bin/python -m pytest tools/forge_run/test_dag_runtime.py -q
```

No model or dependency download, new commit, push or PR is part of this local experiment.

Final local checks passed 22 reusable-runtime pytest cases, 18 existing DAG test methods, 29 legacy Python acceptance methods and 778 Swift regression checks. Independent boundary tests passed both signs of maximum finite FP32 input across all 4096 features and an IPC timeout against a deliberately stopped worker created by the test, followed by process cleanup with no restart.

## Exact timing boundaries and sample counts

`requestWallMillis` starts before the Python session lock is acquired. It includes queued wait, input snapshot/validation, request JSON encoding, IPC, result validation, output conversion, and CPU source parity. It ends while assembling the return mapping, so a caller timer around the complete `run()` also includes the small remaining return overhead. Constructor validation and module imports are excluded. The request table above uses that external complete-call timer, not the shorter internal timer.

`sourceParityMillis` covers eager CPU source execution, reference shape/finiteness checks, difference calculation, and the allclose gate. `ipcRoundTripMillis` starts after request JSON encoding and ends after response JSON parsing; it includes helper response encoding and transport. `helperTotalMicros` starts after reading the input line and covers request parsing, upload, encoding, GPU wait/readback, and GPU evidence checks; it excludes waiting for stdin and encoding/writing the response. `gpuCommandBufferMicros` is the completed command buffer's GPU end minus start timestamp. These nested durations must not be added together.

The published hot results use exactly 45 requests per model: five warmups and forty timed requests, with forty timing samples for each column. p50 is the median; p95 is sorted sample rank 38 of 40 (nearest-rank definition). Both models use batch one, sequential calls, and CPU parity on every request. No delayed fault-injection request contributes to those measurements. Cold constructor wall time was 1,070.451 ms for residual and 932.141 ms for projected branches; module import time was measured separately as 972.403/969.158 ms. Helper startup validation/preparation occurs once and is reported separately from zero per-request validation/preparation counters.

## Bounded reliability audit

Seven additional real-worker pytest cases passed on the M1 Max: SIGKILL/crash, orderly stdin EOF, SIGSTOP/IPC timeout, sequential input isolation with invalid inputs interleaved, cached CPU model mutation causing terminal source-parity failure, timing inclusion of a deliberate parity delay, and four concurrent callers serialized on one PID. The independent timeout audit used a 50 ms IPC deadline and observed worker cleanup in 1.057 seconds; the cleanup grace period is separate from the request deadline. EOF closed the session correctly, although its client diagnostic for an intentionally closed stdin handle was `unexpected ValueError: I/O operation on closed file`.

Independent reload checks changed an artifact's weights and recomputed all applicable file/bundle checksums. The existing session retained its verified snapshot; a new session rejected the weights because they disagreed with `build(seed)` before launching a worker. An independent model-source disk mutation likewise left the live snapshot unchanged and rejected a new session before worker launch. Checksums alone are not authentication.

The timing fault audit delayed the real eager source parity by 20 ms. It observed 26.228 ms source-parity time, 26.960 ms internal request wall time, and 26.987 ms external complete-call time. These fault measurements demonstrate timer inclusion; they are not performance benchmark samples. No production source changes were needed after this audit.

To run the additional tests, explicitly supply an existing matching helper and a bundle forged from the unmodified `tools/forge_run/examples/residual_block.py` with the current backend. The module skips when these prerequisites are absent; a skipped run is not acceptance evidence.

```sh
LEVIATHAN_RELIABILITY_HELPER=/absolute/path/to/leviathan-dag-helper \
LEVIATHAN_RELIABILITY_BUNDLE=/absolute/path/to/residual-bundle \
/opt/anaconda3/bin/python -m pytest -p no:cacheprovider \
  tools/forge_run/test_dag_runtime_reliability.py -q
```
