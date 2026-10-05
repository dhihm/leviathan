# Independent RMS dispatch diagnostics (phase 66)

This optional component compares serial and concurrent Metal compute dispatch types on the same cooperative RMS pipeline. It does not install a pass, change production defaults, modify kernel mathematics, or establish a model-level speedup. The source is an original implementation and the developer CLI reviews were static reviews; native results below are the actual verification evidence.

The three frozen groups are local Q/K/V normalization, global Q/K/V normalization, and the independent post-feedforward normalization pair. Inputs are deterministic synthetic activations; scale tensors come from the existing local model. The initial arms must be finite, Float-bit identical, inside the unchanged vector-normalized FP64 1e-4 contract, and preserve canaries. Every warmup and timed arm must match the already-gated serial output bits and preserve its canaries after actual command-buffer completion.

Two warmup pairs and six timed pairs alternate order. Each arm completes before the other is submitted. Stage timestamps are resolved twice, must agree, and must not be stale. Both order medians in all groups must be at or below the fixed 0.85 concurrent/serial ratio before integration evidence exists. Component evidence alone never qualifies or installs a model pass.

## What was verified

| Check | Actual result |
| --- | --- |
| Swift toolchain | Swift 6.2.1, native arm64 Mac |
| Framework import-only typecheck | Passed, 51.323 seconds |
| Initial isolated release component build | Passed, 161.878 seconds |
| Final CLI actor-safe incremental build | Passed, 2.836 seconds |
| Final pure-test incremental build | Passed, 3.164 seconds |
| RMS pure tests | 71 passed, 0 failed |
| Pre-GO EOF and malformed GO | Refused before setup; only an exclusively reserved empty report remains |
| Existing report path | Refused before READY |
| Existing default regression | 2,056 passed, 0 failed; actual END; supervisor lock released |
| Model-based component | Incomplete: 600-second timeout, actual SIGTERM 15, only READY observed, no END or numeric report |
| Setup-only localization | Incomplete: 120-second timeout at `gguf-map-parse`, before any later setup stage or kernel dispatch |
| Small local I/O probe | Open completed in 0.000285 seconds; offset-0 read of at most 4 KiB did not return within 20 seconds; actual owned SIGTERM |

No new model correctness or performance result follows from these incomplete attempts. The existing default suite includes small synthetic Metal and PyTorch parity checks, while its real-model opt-ins were disabled. Its historical diagnostic timings are not a new RMS or whole-model benchmark.

## Model file availability

The existing model path is recorded in [model-availability.json](evidence/phase66-rms-dispatch/model-availability.json). It reports 14,439,363,584 bytes, filesystem flags `0x40000060`, the `SF_DATALESS` bit, and zero allocated blocks. The installed SDK's `sys/stat.h` defines `SF_DATALESS` as `0x40000000` and describes it as a dataless object. An exact cached-digest blob was absent in the checked Ollama and Hugging Face cache locations. This is not an exhaustive search of the Mac.

The bounded 4 KiB read did not return, so the observed block is file I/O before a GGUF parser or Metal kernel could run in that probe. The provider and underlying materialization mechanism are not established. No explicit model download, forced hydration, model rehash, or original weight-content export was performed. A readable materialized copy of the existing model is required before model-specific numeric or performance gates can proceed.

The cached explicit-refresh model digest is reused. The CLI does not hash the model; the caller supplies the existing cache record value. Executable and small source digests identify compiled artifacts and are separate from model-weight hashing.

## Setup-only diagnosis

`rms-coschedule-component --setup-only` waits for exactly GO and runs the same loader with nine ordered CPU-side boundaries: GGUF map/parse, config, tensor contract, norm packing, Metal device/queue, weight buffer, library compilation, pipeline creation, and context readiness. It never calls the component runner or encodes/commits a command buffer. Its serialized correctness, timing, gain, installation and model qualification flags are forced false. `setupCompleted` requires the full ordered boundary list and a successfully compiled pipeline.

The existing GGUF reader virtually maps the whole file read-only and returns zero-copy tensor slices. The component copies only the small F32 normalization tensors into one Metal buffer. It creates no full-model runner, whole-weight Metal buffer, or KV cache. No resident-memory inference is made from those capacities or virtual mappings.

Source builds in the worktree twice reported an unrelated input as modified during compilation. The diagnosed content hash and Git diff were unchanged. Native validation therefore used a byte-identical source snapshot in a separate temporary package. Core compilation completed there. The new CLI's Swift 6 actor isolation error was corrected with an externally authored non-escaping closure, without disabling concurrency checks. The final CLI and tests then compiled and passed. These build diagnostics are separate from GPU correctness and performance.

All command buffers and timed-arm semantics remain unchanged; added boundary lines occur outside sampled encoders. Every actual native timeout was collected, and owned locks were released after actual child termination.

## Evidence and remaining work

[Evidence directory](evidence/phase66-rms-dispatch/) contains authored patches, independent static review, actual build and protocol receipts, terminal supervisors, the file-I/O probe and the default regression output. No raw CLI thinking/authentication events or model weights are included.

After existing model bytes become locally readable, complete setup-only diagnosis, then the original fixed component correctness/timing protocol. Any qualifying component would still require all three frozen model fixtures, actual argmax continuation, full-vocabulary 1e-4 gates, and separately qualified whole-model performance evidence before adoption. The independently reviewed workload-dependent KV capacity proposal remains unapplied and unqualified; it is not a claimed memory or performance win. No main/default/gain promotion occurred.
