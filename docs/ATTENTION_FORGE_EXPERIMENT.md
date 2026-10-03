# Bounded Attention Forge Experiment

The incumbent single-query attention kernel gives each of 16 query heads one GPU thread. Each thread recomputes each Q/K dot product three times and accumulates every output dimension serially. This diagnostic candidate uses one 64-thread group per head, computes each score once, and distributes tokens and output dimensions while preserving serial dot and chronological normalization/output accumulation order. Shared score storage is bounded to 16 KiB for at most 4096 tokens.

Both the standard runner and Forge SSA attention call sites use one common diagnostic dispatch helper. No candidate installed means the original attention dispatch and bindings are used. The candidate is installed only inside a scoped diagnostic wrapper; nested installs are refused and the original pipeline/state is restored on returns and throws. Serving defaults remain unchanged.

## Qualification and Numerical Policy

The release and debug builds passed. All 18 local/global synthetic cases passed against the actual cached incumbent and an independent FP64 stable-attention oracle: token counts 1, 2, 17, 33, 65, 257, 1024, 1025, 4096; head dimensions 256/512; 16 query heads; 8/2 KV heads; local window 1024/global window 0. A 4097-token capacity guard rejected the candidate. Outputs were finite and deterministic.

The full-model policy is unchanged from existing LoweringSelect qualification: for every logit, abs(candidate-reference)/max(1,abs(reference)) < 1e-4, finite values, exact first argmax, and identical generated IDs. Three fixtures (English chat, Korean chat, raw Hello) each covered 32 steps and all 262144 logits. Maximum full-logit absolute error was 7.867813110e-5 and maximum relative error was 7.529184222e-5. Outputs are tolerance-equivalent, not bit-identical, and the tolerance margin is limited.

Attention component acceptance also checks the independent FP64 oracle with max absolute error / max(1,max absolute baseline) < 1e-4. Unlike the separate Q4 bit-identity gate, attention component bit identity was not required from the initial experiment specification; no tolerance was increased after failure.

Attention applications were 1560, 1620, and 990 for the respective full-model fixtures, with attempts equal to applications and zero fallback. No-leak, canary reachability, and bit-identical reference restoration passed. A separate measurement binary linked to the same release Core objects checked current serial versus E4_R32 with Q4 width 64 in both arms: all three fixtures were bit-identical, with exact expected stage/merge counts and zero fallback. This is not a serial comparison inside the attention qualifier executable itself.

Existing regression suite: 778 passed, 0 failed. Five artifact negative cases passed before candidate execution: invalid digest, altered timing with recomputed digest, incorrect candidate source identity with recomputed digest, different region bytes, and refusal to overwrite an existing output. These cases are not an exhaustive validator security proof.

## Matched Decode Measurement

Both arms use the same model, pinned E4_R32 lowering, Q4 width 64, SSA path, and prompt. After an untimed warm-up of both arms, three paired AB/BA/AB raw-Hello runs measure 31 decode calls; the two-token prompt prefill and its first generated ID are excluded from timing. Each arm returns 32 generated IDs in total. Timed IDs and candidate counters are checked; full-logit finite/error checks occur in the preceding quality gate, not on every timed call.

| Round | Pair/order | Reference seconds | Candidate seconds | Paired reduction |
| --- | --- | ---: | ---: | ---: |
| Selection | 0 / AB | 7.836621208 | 5.525078125 | 29.497% |
| Selection | 1 / BA | 8.088393833 | 5.792760375 | 28.382% |
| Selection | 2 / AB | 7.998460458 | 5.801783834 | 27.464% |
| Fresh verification | 0 / AB | 7.570714125 | 5.715637416 | 24.503% |
| Fresh verification | 1 / BA | 7.701782708 | 5.349352416 | 30.544% |
| Fresh verification | 2 / AB | 7.517219583 | 5.464920875 | 27.301% |

Selection median walls: 7.998460458 -> 5.792760375 seconds (3.876 -> 5.352 tokens/s computed as 31 / median wall). Median paired reduction: 28.382%.

Fresh-verification median walls: 7.570714125 -> 5.464920875 seconds (4.095 -> 5.673 tokens/s). Median paired reduction: 27.301%.

Every pair passed candidate <= 0.98 * reference. These are separate rounds, not one aggregated gain. A median of paired reductions differs from the reduction of wall medians. Measurements are limited to this short warm workload; they are not serving throughput or comparisons with Ollama, MLX, another machine, or historical experiments. Intrusive per-kernel attribution was a hypothesis-selection aid, not end-to-end causal proof.

## Local Selection and Experiment Record

The standalone tool performs deterministic local comparison of one predefined eligible attention candidate against the incumbent, without runtime agents or network calls. It persists a record only after component, full-model, dispatch/restore, and timing gates pass. Existing files are never overwritten. Fresh verification validates the record before candidate compilation, recompiles the candidate on the runner's device, and reruns all gates.

```
swift build -c release
swiftc -O -I .build/release/Modules tools/attention_qualification/main.swift .build/release/LeviathanCore.build/*.swift.o -o .build/release/attention-forge
.build/release/attention-forge --select --gguf MODEL.gguf --region REGION.json --graph GRAPH.json --artifact NEW_RECORD.json
.build/release/attention-forge --verify-artifact --gguf MODEL.gguf --region REGION.json --graph GRAPH.json --artifact NEW_RECORD.json
```

The record binds the exact executable, streamed model SHA256/size, graph/region bytes, candidate/incumbent source, compile options, engine, OS, actual hardware profile/geometry, E4 source/schedule, and Q4 width. A rebuild invalidates it. The digest supplies integrity, not authenticity or authorization. The included record is evidence for the tested binary, not an installable plan for a user's rebuilt binary; rerun selection after rebuilding.

The model is the already available Gemma4 26B A4B Q4_0 GGUF (14,439,363,584 bytes). No model or dependency was downloaded. This is not the lightweight first target and does not implement arbitrary new Python/PyTorch model capture or model-specific pass generation. The record is metadata/evidence, not a saved Metal binary. No public serving installer or default is added. Diagnostic counters are process-wide and only sequential experiments are qualified. The standalone qualifier/validator should be refactored before broader integration.

## Evidence and Provenance

[Structured results](evidence/attention-forge/results.json), [selection log](evidence/attention-forge/select.log), [fresh verification](evidence/attention-forge/fresh-verify.log), [serial/E4 check](evidence/attention-forge/serial-e4-reference.log), and [regression log](evidence/attention-forge/regression.log) preserve the exact evidence. The graph and region fixtures are included next to those files. The generated [experiment record](evidence/attention-forge/selected-experiment-record.json) has productionQualified=false.

Implementation was externally authored through configured agy/Opus; exact response/patch/source hashes are recorded in results.json. The orchestrator applied exact replacements, wrote measurement-only regression orchestration and English documentation, and ran the gates. No Kraken/company source or other project's history was supplied. Prior rejected Q4 and attention experiments and their raw records remain preserved separately; their code is not included in this change. No main merge is part of this experiment.
