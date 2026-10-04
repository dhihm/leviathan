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

Fresh qualification, persisted reload, three serving fixtures, and the predeclared four AB/BA comparison pairs are still running. No router throughput gain, promotion decision, command-buffer reduction, or closure of the Ollama gap is claimed yet. This experiment uses the existing 14,439,363,584-byte Gemma probe; no model, dependency, or new service download is required.
