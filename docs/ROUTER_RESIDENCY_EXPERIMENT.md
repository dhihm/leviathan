# Router residency and scalar router launch lowering

This original batch addresses three related router-path costs: the CPU round-trip for the second router input multiply, CPU initialization of the resident expert output accumulator, and the single-threadgroup geometry of the original F32 router matvec. It extends the bounded automatic Forge candidate search on the user-selected RMS-inclusive main baseline `7dd0456020f7afe0a1da67b518fc9afe98a2a211`.

## Execution and ownership

The first router scale stays a separate dispatch on the original `g4_scale` pipeline. The new second-scale kernel uses the current canonical layer tensor `blk.<layer>.ffn_gate_inp.scale`, checking F32 type, exact vector geometry, byte span, alignment, overflow, buffer bounds, and UInt32 count. The current reference and weight buffer are resolved each call because cache compaction remaps offsets; no buffer or offset is cached across remapping. Unsupported inputs retain the original CPU multiply fallback.

For fully resident experts, a guarded uint32 GPU zero kernel replaces the CPU accumulator memset inside the ordered encoder. Expert-cache execution retains its flush/prefetch/memset path and lifetime semantics. Expert selection, CPU Double exponentiation and tie rules, expert kernels, and the eight ordered axpy merges are unchanged.

A nonzero router launch divisor derives threads per group from the actual original F32 pipeline's execution width and maximum threads. Only the canonical hidden-by-expert router dispatch qualifies. The kernel, bindings, per-row accumulation order, and other F32 operations stay unchanged. Divisor zero retains original geometry. Invalid or unsupported configurations fall back explicitly; nonzero divisors without residency or with tiled F32 are rejected before execution.

Five appended candidates use residency with divisors 0, 1, 2, 4, and 8, preserving the 27 existing candidates and RMS legality guards. Kernel source/version, compile options, launch derivation, and every candidate field bind persisted artifacts. Missing required fields reject stale artifacts. Private lazy pipelines do not change the shared pipeline dictionary or depend on transient armed flags in engine identity.

## Gradual underflow correction

The original direct GPU second multiply failed three strict tests on this M1 Max. After the unchanged first GPU scale, the CPU second multiply produced bits `0x00148512` while the GPU produced zero for widths 256, 704, and 2816. Precise compile options alone did not preserve gradual underflow. The failures remain recorded in [underflow-failure.json](evidence/router-residency-experiment/underflow-failure.json).

Version 3 retains hardware multiplication for ordinary normal-range products. Potential subnormal operands/results use an original integer significand-product helper and one round-to-nearest, ties-to-even operation from the exact product. It reconstructs output bits without floating arithmetic on a subnormal. Signed zero, subnormal inputs, minimum-normal rounding, and zero-underflow ties are covered. An earlier proposed double-rounding implementation was rejected during review; independent counterexamples are retained. No test tolerance or input coverage was weakened.

A separate host mathematical review compared 248,007 finite input pairs against exact binary32 operands multiplied in binary64 and rounded once to binary32, with no mismatch. That review does not substitute for Metal execution. Actual Metal boundary checks cover 401 direct second-multiply cases, including deterministic subnormal-binade sweeps. The complete release regression suite passed **1,533 checks with zero failures**.

## Actual-model correctness

All five router candidates passed a four-step teacher-forced full-vocabulary comparison against the RMS-inclusive main flags, with five processed tokens per arm. Every one of 262,144 logits and every selected token was bit-exact. Each arm recorded 150 GPU second scales, 150 resident GPU zeros, 150 RMS computes, 300 RMS reuses, and no fallback. Nonzero divisor arms recorded 150 derived router launches; divisor zero intentionally recorded none. Identity stayed stable after lazy compilation. These diagnostic E4/width64 checks are [focused evidence](evidence/router-residency-experiment/focused-results.json), not persisted production qualification or throughput approval.

The runtime and repository tests were externally authored through the user-approved existing Z.ai GLM-5.3/ZCode route. Exact original patch replay against main matched all five changed source/test files. Root reviewed and applied exact patches, mechanically normalized file paths and indentation, and authored independent temporary measurement drivers. Only authorized router code, tests, and the minimal resident Q4 reset wrapper were transmitted. No expert-cache implementation, company source, other project content, or credentials were sent.

## Production qualification and performance

Fresh production qualification passed in **719.57 seconds** and persisted reload passed separately in **41.51 seconds**. Forge selected residency with divisor 2 (16 threads per group on the actual 32-wide scalar router pipeline), along with the same batching, vectorized loads, GPU KV, parallel attention, cooperative RMS/Q6, and shared RMS statistics as main. Q4 width 64 and the existing E4/R32 MoE lowering remain selected. All 32 candidates and rejections are retained in [qualification.json](evidence/router-residency-experiment/qualification.json).

The plan digest is `73dccbebce7f11aeb2c8c31681c2ab9849a4ef75a9825b949b0a2ca664d182df`; composite digest is `1f620a9566867e690f33b119edc9151f3f137c9a5c4670dd3f0763e6ebacb64c`; qualified executable SHA-256 is `d4253487ce3762a309a1b39ffc777963044f7c5807d5c0b6efd48329b3ebfa6a`. Raw Hello, English chat, and Korean chat each passed prepared serving with 32 generated tokens, all 262,144 logits under the strict < 1e-4 gate, exact IDs, and zero router or RMS statistics fallbacks. See [production/reload](evidence/router-residency-experiment/full-run.json) and [serving results](evidence/router-residency-experiment/serving-three-results.json).

The predeclared fixed protocol uses four serial A/B, B/A, A/B, B/A pairs. Each arm runs four generations, retaining the excluded first warmup and scoring the median of the remaining three warm 31-step decode-wall totals. Original GGUF, raw Hello, BOS-only prefix reuse, and capacity are matched. Startup, hashing, and prefill are excluded; synchronous graph execution and required output readback are included. No owned author, build, or profiling jobs overlapped this series. Every run is retained, with no adaptive stopping or exclusion.

| Pair | Order | RMS main tokens/s | Router tokens/s | Decode time reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | A/B | 10.358 | 13.223 | 21.671% |
| 2 | B/A | 9.718 | 13.241 | 26.603% |
| 3 | A/B | 10.240 | 13.585 | 24.625% |
| 4 | B/A | 10.334 | 13.363 | 22.671% |

Median paired decode time reduction is **23.648%**. Equal-work pooled totals give **23.937%** less decode time and **31.471%** more throughput. All four pairs exceed the predeclared 2% threshold; every cross-arm ID matches. Sources, both executables, plans, and composites stayed unchanged. This qualifies the composed router batch for main; it does not isolate the benefit of each of the three changes. See [all fixed results](evidence/router-residency-experiment/fixed-results.json) and [all resource observations](evidence/router-residency-experiment/resource-observations.json).

## Remaining measured costs

A separate post-comparison normal census measured **74.507 ms/token**, **37 command buffers**, and **1,349 dispatches** per decode. The prior RMS implementation used 97 command buffers and 1,289 dispatches: two added GPU dispatches per layer remove 60 host boundaries. The census is separate from the fixed comparison and is not an additional paired performance result.

Normal accounting gives 81.57% GPU intervals, 12.54% host flush work, 1.25% encoding, 0.28% read copies, and 4.36% residual work. Over 31 decode tokens there were 930 second scales, 930 resident zeros, and 930 derived router launches, all with zero fallback; RMS statistics retained 930 computes, 1,860 reuses, and zero fallback. Separate intrusive replay passed all full-logit comparisons with zero normalized difference and exact IDs. Its largest kernel diagnostic shares were generic Q4 (27.06%), remaining cooperative RMS (21.27%), F32 router (14.56%), generated expert gate/up (9.91%), generated expert down (5.25%), and Q6 output (5.11%). These intrusive intervals are **not normal wall shares**; the intrusive residual counter includes uninstrumented dispatch/wait work and overlaps GPU time. See [census and limits](evidence/router-residency-experiment/post-budget.json).

CPU top-k synchronization, five host-copy boundaries, substantial GPU matrix/reduction work, and many dispatches remain. The next useful batch should profile actual batched GPU occupancy and memory access for Q4/RMS/router composition, then optimize the demonstrated costs together with unchanged full-model gates. The current result does not establish a hardware bandwidth limit, MLX causality, or competitive throughput against Ollama. No new matched Ollama or MLX benchmark was run.

This experiment uses the existing 14,439,363,584-byte Gemma probe; lightweight models and broader Python model coverage remain unfinished goals. No model/dependency download, secret setup, purchase, or new service was needed. The source and focused proof were first preserved at private experiment commit `c17153ed31ebcad6f3b6b09ce86500b5083c7796`.
