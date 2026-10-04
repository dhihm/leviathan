# Prepared execution and parallel attention batch

This batch follows an end-to-end bottleneck inventory rather than repeated full evaluations after individual edits. It combines a retained, validated SSA execution representation with the existing original bounded parallel-attention recipe, selected through the normal Gemma Forge candidate workflow. GPU KV append remains part of the composed candidate. Runtime optimization and execution do not call an agent.

## Measured priorities

The baseline is private-main commit `39be1daea5a9e2a2f842cc8e8bfd55c6e0331beb`, running the existing Gemma 4 26B-A4B Q4_0 model on Apple M1 Max with 32 GiB memory. Raw `Hello`, BOS-only prefix alignment, 32 generated IDs and 31 decode steps were used. In the second instrumented repetition:

| Decode wall category | Share |
| --- | ---: |
| GPU busy intervals | 86.2% |
| Flush host share | 8.6% |
| Encoding | 0.7% |
| Read copies | 0.4% |
| Write copies | 0.1% |
| Residual CPU work | 3.9% |

Flush wall overlaps GPU execution; host share is the nonnegative difference between flush and GPU time. This single instrumented workload does not establish stable percentages across prompts, contexts or system load. Metal allocated memory sampled 15.168 GiB. The baseline completed 97 command buffers and 1,829 dispatches per token.

CPU-only measurement of repeated SSA validation, canonical lowering, equality and operand-index construction over the actual 934-instruction graph took median 2.050 ms and p95 7.994 ms across 100 iterations after five warmups. Removing it alone cannot explain or close an approximately 400 ms/token baseline.

An existing intrusive diagnostic ranked Q4 matvec, attention and RMSNorm at approximately 32%, 27% and 22% of its GPU intervals. Per-kernel submissions inflated those intervals by about 1.45 times relative to batching. These historical percentages are investigation targets, **not current batched GPU shares or additive speedup predictions**. Previously rejected tiled candidates remain subject to the same numerical gate.

## Architecture and safeguards

`GemmaPreparedExecution.prepare` validates a graph against the mapped GGUF and actual configuration, lowers canonical SSA, validates its dataflow and retains its value index. Private construction binds the result to the exact retained runner object and immutable graph. Reuse checks object identity, configuration and structural graph equality without hashing, lowering or rebuilding the index. Another runner, a changed graph, out-of-range token or simultaneous raw schedule and prepared input is rejected before GPU execution. Existing completion checks and error cleanup remain. Prepared execution is not declared `Sendable`; a runner remains a serial execution resource.

The production attention candidate compiles the project's original, unchanged 64-lane kernel into a separate lazy pipeline on the runner's device. It computes scores once, distributes attended tokens and output dimensions across lanes, and preserves the original sequential dot and chronological reduction orders. Host geometry and actual pipeline limits guard dispatch; ineligible geometry retains the incumbent. Production application and fallback counters are separate from the original diagnostic counters and installation slot. Kernel arithmetic remains tolerance-gated rather than assumed bit-identical.

Forge retains its previous 15 candidates and adds parallel attention alone and composed parallel attention plus GPU KV append, for 17 bounded candidates. The new flag is explicit in Codable artifacts and candidate identity. Shader source identity is stable across enabled states and lazy compilation. Old artifacts missing the new axis are rejected. The `1e-4` full-logit tolerance and exact greedy requirements are unchanged; no winner is forced.

`gemma-generate` prepares a validated stored plan once and checks that its embedded SSA equals the prepared lowering. `--unprepared` retains the schedule-validation path for attribution. Selected flags and actual attention counters are logged.

## Qualification

The release build and 846 regression checks passed. Development used a four-step real-model gate and binding rejection checks before the completed batch entered full qualification.

Fresh Forge selection and a separate-process artifact reload both passed the 32-step full-model gate. A fast tiled candidate was rejected at step 25 with normalized difference `1.171e-4`; Forge selected the composed GPU KV append and parallel attention candidate through its existing fallback policy. The saved plan records the rejection and successful 32-step gate.

A separate measurement executable linked the same compiled core objects and checked the prepared serving path over all 262,144 logits at every one of 32 steps. Maximum normalized difference was `7.529184222221375e-5`, greedy IDs matched exactly, and attention applied 990 times with zero fallback. The serial prepared arm was bit-identical to the unprepared serial reference. Four rejection cases performed no GPU dispatch. This gate used a 128 MiB expert cache and makes no performance claim. It covers one raw-Hello fixture, not arbitrary inputs or models.

The generic `torch.export` frontend and persistent small-DAG runtime remain separate capabilities. This Gemma model input is still a restricted stage-list DSL rather than arbitrary PyTorch LLM capture. The original attention diagnostic artifact remains diagnostic; it is not reused as production authorization.

## Performance evaluation

Three matched pairs measured median paired decode-wall reduction **18.29%**. Every pair improved by more than 2%, and all generated IDs matched. New warm aggregate throughput was **3.18–3.31 tokens/s**, versus **2.65–2.71 tokens/s** for the preserved baseline. This is a measured improvement for this fixture and configuration; Ollama-level performance remains unmet.

| Pair | Baseline decode wall | Batch decode wall | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 11542.00 ms | 9427.50 ms | 18.32% |
| 2 | 11450.15 ms | 9355.50 ms | 18.29% |
| 3 | 11705.40 ms | 9749.00 ms | 16.71% |


The completed batch is evaluated against the preserved previous executable with its qualified plan. Both versions use the same model, graph, raw-Hello input, BOS-only prefix alignment and 32 generated IDs. Three serial AB/BA/AB process pairs each use one excluded warmup and two timed repetitions. Decode wall excludes startup, identity checks and prefill; it includes synchronous execution and required output readback. Each arm's objective is the median of its two 31-step decode totals. Generated IDs must match across all arms, and source, binary and plan identities must remain unchanged.

Feature-attribution runs use the same new executable with attention and prepared execution switches. These single sequential runs are diagnostic and cannot establish small causal speedups. The separate full serving gate verifies the serial prepared control; its diagnostic artifact honestly retains its original four-step screening record.

The single same-binary attribution runs measured:

| Feature arm | Warm median 31-step decode wall | Aggregate throughput |
| --- | ---: | ---: |
| Serial attention, prepared execution | 12,939.7 ms | 2.396 tokens/s |
| Serial attention, unprepared execution | 12,268.3 ms | 2.527 tokens/s |
| Parallel attention, unprepared execution | 10,026.7 ms | 3.092 tokens/s |

The serial prepared arm was slower in this sequential window. These runs do not establish an independent throughput gain from prepared execution, and their variability prevents a causal percentage attribution to that feature. The structural removal of repeated validation is verified; the replicated whole-batch comparison remains the performance evidence. No individual gains are added together.

Complete timing evidence and limitations are recorded alongside the qualification evidence in [docs/evidence/prepared-attention-batch](evidence/prepared-attention-batch). The Mac is shared and is not claimed idle. Ollama's existing server evaluation counters use a different timing scope. MLX/MLX-LM were absent from the inspected Python environment and have not been benchmarked under matched conditions. No dependency or model download was performed. Competitive performance remains unfinished.
