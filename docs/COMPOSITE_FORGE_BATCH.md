# Composite Forge activation batch

The published prepared-execution and parallel-attention runtime still launched 685 generic Q4 matrix-vector kernels per decode token in a diagnostic replay. The existing E4 MoE lowering and hardware-derived Q4 launch selector were available as separate experiments, but normal `gemma-forge` and `gemma-generate` did not activate their verified composite artifact. This batch connects that production selection and reuse path before the regular Forge plan is selected or validated.

## Measured scope and choice

A first generation after a two-token prefill measured 389.08 ms per decode token: 73.42% GPU intervals, 17.39% host flush work, 1.37% encoding, 2.15% output read/copy, 0.10% input write/copy, and 5.57% unassigned residual. There was no full-generation warmup. Cold page/resource effects and shared-Mac variation can affect these shares; they are not a stable warm serving budget.

A separate intrusive replay, checked against all logits with zero normalized difference, ranked generic Q4 matrix-vector work at 65.63% of its measured kernel intervals, RMS normalization at 18.74%, and the Q6 output head at 6.06%. Intrusive per-kernel intervals do not represent normal batched GPU shares. The E4 path reduces the diagnostic generic Q4 launch population from 685 to 205 per decode token by batching expert work, while preserving the original arithmetic and ordered merges.

If only GPU time changed and every measured host component remained fixed, the hypothetical maximum speedup for that first-generation sample would be 3.76 times. This is a conditional bound for that sample, not an architecture ceiling or a prediction comparable with Ollama. The 14.44 GB GGUF remains fixed; no controlled MLX or alternative-layout comparison has established backend or model-layout causality.

## Activation and reuse

Pass both `--composite-plan PATH` and `--moe-region PATH`. `gemma-forge` requires the full 32-token setting, selects a missing artifact using the existing production selector, validates it, and activates it through the public atomic installer before ordinary Forge selection. `gemma-generate` also requires `--plan`; it only reuses an existing validated artifact and never profiles or selects during inference. Streaming expert-cache mode is incompatible with this resident composite.

The selector retains the original hardware width ladder, FP64-prefix and full-output bit-identity component gates, three canonical fixtures with 32 generated steps and all 262144 logits, exact token IDs, the unchanged normalized error threshold below 1e-4, and the existing paired 31-step timing policy with a fixed 2% selection margin. A malformed or stale artifact fails. An internally consistent no-win record keeps the incumbent active and explicitly reports that nothing was installed.

Engine identity now binds the actual installed MoE artifact, region, emitted source, schedule, and Q4 width. Prepared execution rejects changed installed state before GPU work. There is no per-token source hashing. Artifact and region reads are bounded to 8 MiB. The CLI reuses one streaming GGUF hash for composite and ordinary plan validation.

## Qualification

Release compilation, 850 regression checks, six command-line preflight rejection checks, and a four-step all-logit bit-exact diagnostic composition check passed. The latter also checked identity changes/restoration, stale prepared-state rejection before dispatch, actual E4 and Q4 application counts, and zero fallbacks. It does not approve a production artifact or prove a performance gain.

Fresh production selection and full Forge qualification passed in 779.53 seconds. The composite selector chose Q4 width 32; both width finalists passed all three fixtures with zero normalized logit difference. The recorded eligible and applied populations matched at 28495, with derived MoE stage and ordered merge proofs. The public installer replayed the selected artifact after persistence.

The regular 17-candidate Forge selection retained batching, vectorized loads, GPU KV append, and parallel attention. Its initially fastest tiled-Q4 plus tiled-RMS candidate failed the full gate at step 13 with normalized difference 1.103e-4 and was rejected. No threshold was relaxed. The persisted full-gate length is 32.

A separate fresh process reused both artifacts, validated and publicly installed the composite, and passed the full Forge gate in 45.50 seconds. A separate actual prepared-serving invocation passed the 32-token full-logit gate and reproduced the exact generated IDs. Across reference replay, graph gate replay, and one generation it recorded 1980 parallel-attention applications, 5940 applications of each MoE stage, 23760 ordered merges, and 20295 Q4-width applications, with zero fallbacks. These are cumulative counts, not per-token counts. The serving reference uses the existing serial runner with the validated composite installed.

Three matched serial process pairs used AB/BA/AB order, the preserved published `4e59c10` executable and plan as incumbent, and the new executable plus publicly installed composite and full-gated plan as candidate. Both arms used the same GGUF, raw Hello, 32 generated IDs, 31 decode steps, BOS-only prefix alignment, and three repetitions. The first repetition was excluded as warmup; each arm used the median wall total of repetitions two and three. No profiling, build, or owned author job ran during this timing.

| Pair | Incumbent warm decode wall | Composite warm decode wall | Wall reduction | Composite throughput |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9,702.60 ms | 4,772.15 ms | 50.816% | 6.496 tokens/s |
| 2 | 10,042.95 ms | 4,790.30 ms | 52.302% | 6.471 tokens/s |
| 3 | 9,625.75 ms | 4,779.95 ms | 50.342% | 6.485 tokens/s |

The median paired decode-wall reduction was **50.816%**. Incumbent aggregate throughput ranged from **3.087 to 3.221 tokens/s**; the composite ranged from **6.471 to 6.496 tokens/s**. All generated IDs matched exactly across arms and repetitions. Sources, binaries, both plans, and the composite artifact stayed unchanged. This is the whole activation/composition batch's result, not a causal percentage for one individual feature. The selector's separate Q4-width timings hold E4 fixed and use its own adapter settings; they cannot be substituted for this combined serving comparison. See [paired-results.json](evidence/composite-forge/paired-results.json).

The Mac is shared and was not claimed idle. Synchronous CLI decode wall includes graph execution and output readback; Ollama server evaluation counters have a different scope, and no matched MLX comparison was run. Full selection and reload times include deliberate qualification work and are not request latency. The evidence is in [qualification.json](evidence/composite-forge/qualification.json). No model or dependency was downloaded for this batch. The runtime remains an incomplete personal Mac inference experiment; arbitrary new Python LLM architectures and the performance gap with Ollama remain unfinished.

## Remaining measured budget

One separate instrumented invocation used two full generations; its second generation measured 4927.2 ms over 31 decode steps after a complete generation warmup. GPU intervals contributed 78.4%, host flush work 17.0%, encoding 0.8%, read/copy 0.3%, write/copy 0.1%, and residual work 3.4%. Instrumented values are separate from the matched timing result and may vary on the shared Mac.

Holding every other measured component fixed, removing only GPU time would give a hypothetical maximum speedup of 4.63 times; removing only host time while retaining GPU time would give 1.28 times. Neither is an architecture ceiling or a transferable throughput prediction. GPU execution remains the largest measured category. This invocation did not collect a new per-kernel census, so the old intrusive percentages cannot identify the dominant kernel after composition. The next coherent optimization batch needs that post-composition ranking before choosing kernel changes. The unchanged GGUF/backend comparison supports an execution-composition improvement; it does not establish superiority over MLX or a different weight layout. See [post-warm-budget.json](evidence/composite-forge/post-warm-budget.json).

## Reproduction

Use the existing locally licensed GGUF as `$MODEL` and an output directory as `$OUT`. The checked-in graph and original PyTorch-exported MoE region below are identical to this batch's inputs. Selection is automatic and bounded; it invokes no agent. Artifacts are bound to the actual executable, hardware, model, graph, region, emitted source, and toolchain. Rebuilding requires new artifact paths rather than rewriting identities on an old result.

```sh
swift build -c release
.build/release/leviathan-cli gemma-forge \
  --file "$MODEL" \
  --graph docs/evidence/q4-launch-qualification/gemma-graph.json \
  --plan "$OUT/forge-plan.json" --text Hello --gen-tokens 32 \
  --composite-plan "$OUT/composite-pass.json" \
  --moe-region docs/evidence/q4-launch-qualification/moe-region.json
.build/release/leviathan-cli gemma-generate \
  --file "$MODEL" \
  --graph docs/evidence/q4-launch-qualification/gemma-graph.json \
  --plan "$OUT/forge-plan.json" --text Hello --max-tokens 32 \
  --composite-plan "$OUT/composite-pass.json" \
  --moe-region docs/evidence/q4-launch-qualification/moe-region.json
```

The Gemma graph comes from the restricted Python model DSL; the routed MoE region comes from `torch.export(strict=True)` of the original trusted Python module. This integration does not make arbitrary new Python LLM architectures supported.
