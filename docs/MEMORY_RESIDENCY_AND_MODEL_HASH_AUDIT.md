# Memory residency diagnosis and model-file hash audit

The seven-process frozen-baseline A/A control failed its fixed reproducibility gate: 40.46% spread and global paging growth. This follow-up separates idle allocation, streaming file hashing and one-process inference using the same existing model, quantization, max context, prompt and generated-token count. It changes no product source and makes no optimization claim.

## Staged local diagnosis

An independent temporary driver held five phases for 15 seconds each, with aggregate `vm_stat`, thermal/power and load observations outside inference: process only, tokenizer, one bounded streaming hash, one resident runner idle, then release. This run performed no inference and installed no compiled artifacts. It used the built library whose runtime source is byte-identical to the frozen baseline. No user apps were enumerated or stopped.

The initial process-only idle interval had **82 pageouts** and zero swapouts. The hash transition had **176 pageouts**, file-backed pages increased by 223,828 (3.42 GiB), and compressor occupancy increased by 114,575 pages (1.75 GiB). These are simultaneous global changes, not isolated attribution to our process. Neither the hash transition nor its subsequent idle hold increased swapouts.

One resident runner reported **16,286,433,280 allocated Metal bytes (15.168 GiB)** before dispatch, but wired pages did not increase by that amount and swapouts remained unchanged. Metal allocation accounting alone does not establish physical resident bytes or pinning. Across the complete staged run, swapout growth was zero; pageouts also occurred before model loading and after release. Therefore the earlier control's paging cannot all be attributed to hashing or merely creating a resident mapping.

Source inspection confirms a no-copy shared file mapping for weights. Metadata loading uses mmap; the tokenizer retains its vocabulary structures, not a second whole-file GPU buffer. Streaming SHA-256 reads 8 MiB chunks inside per-chunk autorelease pools. No unbounded full-weight copy or read-buffer accumulation was identified in these paths. These source facts do not prove physical-page aliasing or an absence of OS/driver copies.

The resident runner allocates both K and V at maximum context 4096 for all 30 layers: `(25*2048 + 5*1024)*4096*4*2 = 1,845,493,760 bytes`, **1.71875 GiB**. The model itself is 14,439,363,584 bytes. The context allocation was not shortened to make measurements look better.

## Single-process reproducibility control

A second fixed protocol used the original artifact-compatible executable and artifacts, with exactly one runner and one startup identity-preparation pass. It ran 32 repetitions: discard four initial warmups, then retain seven blocks of four; discard the first repetition of each block and take the median of its other three 31-step decode totals. Both old and new controls retain 21 measured warm generations, although process lifetime and initial warmup handling intentionally differ. This is a harness residency diagnosis, not a paired optimization benchmark. No telemetry sampler ran during decode windows.

The model, Q4_0 quantization, max context, raw Hello/BOS alignment, 32 generated IDs, prepared SSA, selected flags, E4R32 and Q4 width32 were unchanged. Executable/plan/composite fingerprints remained unchanged. Every one of the 32 generations matched the prior validated baseline IDs and every reported fallback counter was zero. No additional full-vocabulary logit gate is claimed for this control.

| Block | Median warm ms/token |
| --- | ---: |
| 1 | 84.7516 |
| 2 | 97.8839 |
| 3 | 78.4323 |
| 4 | 78.6323 |
| 5 | 78.4290 |
| 6 | 81.6935 |
| 7 | 84.6484 |

Overall median: **81.6935 ms/token**. Spread: **23.8144%**, still above the predeclared 5% gate. Across the 113.59-second invocation, global pageouts increased by **2253**, swapins by **453,533** and swapouts by **841,468** 16 KiB pages. Swap counters measure cumulative page movement, not current swap footprint; approximately 12.84 GiB moved out does not mean this process owns 12.84 GiB of swap.

**Result: fail; no RMS ON/OFF timing or runtime optimization activation.** Keeping one process does not resolve the measured instability. The apparent median difference from the earlier control is not a speed gain: environmental memory state, protocol and time differ, and both resource/reproducibility gates failed. The observations do not identify another application or distinguish scheduling, host delay and residency faults causally. No isolated quiet-window claim is made for the new diagnosis.

The smallest feasible comparison protocol retains a single resident runner and preinstalled artifacts, uses interleaved identical token histories and fixed complete-output gates outside timers, and requires a separately observed resource window without new pageout/swapout activity plus a passing same-runner A/A floor before judging small changes. Current telemetry does not satisfy that prerequisite. A smaller resident footprint through the already available expert-cache backend could be investigated with the same GGUF and context, but is a separate runtime variant requiring strict three-fixture 32-ID/full-logit gates and its own resource-valid baseline; the existing resident composite cannot simply be installed on a streaming runner.

## Full-model hash audit at main 13fbc59

`GemmaForge.fileDigest` at `GemmaForge.swift:734` is full-byte SHA-256, not metadata validation. Each model pass reads **14,439,363,584 logical bytes**, using bounded chunks; this is logical read volume, not measured SSD traffic. The seven-process control read **101,075,545,088 model bytes** for identity alone. Recorded preparation times were 8.19–8.34 seconds per visit; they include graph parse, model hashing and artifact/plan preparation, so they are not isolated hash timings. Decode timers excluded this startup work, but its resource effects may persist.

| Path | Direct full-model passes |
| --- | --- |
| `gemma-generate --plan`, CLI790 | One per process before the repetition loop. Composite receives this same value. |
| Cached `gemma-forge`, CLI2983 | One even when plan/composite cache hits. No persistent content-hash cache. |
| Fresh/stale `gemma-forge` | CLI2983 plus CoreGemmaForge1381: at least two (28,878,727,168 bytes), plus composite-selection work on its cache miss. |
| `gemma-serve`, CLI3613 | One at server startup; zero additional passes per request or token in this path. |
| Q4/RMS `selectAndPersist` | One outer pass plus one inside its bound component forge: two. |
| MoE `selectAndPersist` | Component artifact persistence1956 plus final selected persistence2263: two. |
| Q4 launch selection | Selected branch persists1239 and reload-install1171: two; retained no-win branch persists once. |
| `MoEProfile` | One at report creation572 per profile invocation, not per token. |
| Explicit CLI bound-artifact verification/installation | Each audited callsite hashes once; detailed inventory is retained. |

These counts are source-level control-flow findings; no new execution was launched to prove every branch. This Mac path has no distributed-rank hash loop. Do not project this finding onto Kraken: its code was not inspected in this task.

Metadata/header validation parses tensor directories, sizes, supported types and execution geometry; it is not a full-byte cryptographic hash. Source/MSL hashes, graph/SSA hashes, executable hashes and artifact-body digests identify different objects and do not read model weights. `Gemma4.executableDigest` calls the generic file hasher on the executable, not on the GGUF. Numerical reference/accuracy gates compare output IDs and logits; a matching file hash does not prove those gates passed.

The original hash rationale was byte-identical model binding for persisted plans and generated artifacts. The final user policy is **compute once, persist and reuse, refresh only explicitly when needed**. An intermediate unverified-policy draft was tested locally but never published to main. It was superseded by the user's explicit cache instruction. No new full-model hash was launched after that instruction; the existing phase45 stream-hash receipt supplies the historical cache entry.

The cached digest is a real historical full-byte SHA-256, not a constant marker and not a claim that this run verified model bytes. Same-path model changes deliberately require `--rehash-model`. Header, tensor geometry and buffer bounds checks remain execution requirements. Numerical accuracy gates, graph/source/executable and compiled-artifact checks identify different objects and remain active. See `CACHED_MODEL_DIGEST.md` for the final policy and its validation evidence.

The performance goal remains unmet. A two-axis runtime admission gate should bind candidate-set/epsilon/analysis identities and pass normal artifact lifecycle plus all three strict 32-ID/full-logit fixtures; no performance-driven selection is admissible under the failed controls.

Raw evidence and original independent drivers are under `docs/evidence/memory-residency-phase45/`.
