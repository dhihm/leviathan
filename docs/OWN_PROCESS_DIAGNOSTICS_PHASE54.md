# Own-process memory and submission diagnostics

The phase54 tool profiles the unchanged incumbent after the existing all-three model qualification. It adds opt-in timers and own-process snapshots; it does not implement or select a faster candidate. The completed native supervised run took 308.564368 seconds, returned exit 0, emitted the supervisor END marker, restored the runner and released its lock. All three qualification gates and all six profiled runs passed.

Every profiled run retained 32 frozen output IDs, 31 continuations using the actual prior argmax, full-vocabulary finiteness, exact OFF counters, the qualified command-buffer/kernel totals, and a fresh reference comparison with maximum per-element normalized error strictly below 0.0001. The measured runs reached 0.00003740, 0.00003272 and 0.00007529 respectively. No tolerance was relaxed.

## Actual timing attribution

All values below are medians of the 31 continuation calls in one instrumented run per input. They are diagnostic timings under normal concurrency, not production TPOT or a qualified comparison. Different fields are nested; do not add the columns together.

| Input | Encode ms | Commit ms | Wait ms | Router selection ms | Binding check ms | Read copy ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en-chat | 1.1118 | 0.2691 | 81.8416 | 0.5290 | 0.0047 | 0.1256 |
| kr-chat | 0.8229 | 0.2100 | 77.4143 | 0.4198 | 0.0050 | 0.1123 |
| raw-hello | 0.7299 | 0.2125 | 68.9148 | 0.3950 | 0.0047 | 0.1023 |

Each continuation submitted and synchronously drained 37 command buffers: one embedding site, 30 router sites, five copy sites and the final logits site. CPU router selection itself is small. Wait spans include device execution and host/driver scheduling, so they are not pure CPU overhead. Command-buffer intervals are coverage intervals, not GPU busy time. Binding checks measured only a few microseconds and do not explain the prior 37–39 ms residual.

The current diagnostic residual varied from 12.8 to 26.7 ms across input runs. It cannot be compared to the earlier uninstrumented residual as a speedup. The next change should address a measured synchronization boundary and retain the strict numerical contracts.

## Actual memory evidence

The source-derived K/V allocation is 1,845,493,760 bytes (1,760 MiB) at 4096 positions, matching actual buffer lengths. The fixture upper bound is 55 positions and 24,780,800 bytes; this is an upper bound rather than the number of positions actually consumed. Metal allocated capacity was approximately 16.29 GB. Capacity is not residency.

Own-process physical footprint was 10.86 MB before initialization, 47.60 MB immediately after loading, and about 2.02–2.06 GB around measured runs. Therefore the allocation does not immediately appear in the process footprint. These snapshots do not isolate which resource caused later growth or count all GPU residency. Do not identify a second file mapping as a second physical copy.

Own-process page-ins increased by zero in all three measured runs. Page faults increased by 2,431, 2,381 and 2,219; faults are not equivalent to page-ins. This evidence does not attribute the earlier system-wide swap activity to Leviathan and does not rule out GPU-specific residency or concurrent system interference. The API event fields are signed 32-bit counters; a decrease is reported as nonmonotonic rather than fabricated as zero.

## Validation and invocation

Native release and debug prerequisite builds passed. The new own-process pure suite passed 66 checks, and the current gate/runtime/plan suites passed 177/170/333 checks. The existing default-path suite passed 2,056 checks. Canonical Opus 5.5 authored the source and tests, and canonical Fable 5.1 independently accepted the implementation and the narrow protocol correction. Root applied the authored source mechanically.

The first supervised start emitted an incompatible READY literal. It was interrupted through the live owned supervisor before GO; the child terminal receipt records SIGTERM 15 and no model or GPU work. The corrected tool aliases the existing gate protocol tokens, and a native integration assertion checks literal BENCHMARK_READY/GO/TIMED_WINDOW_END. EOF refusal was verified before model/device creation. Evidence from both attempts is retained; the first is not model-validation evidence.

Build `owned-process-diagnostics` and run it through `tools/owned_benchmark_supervisor.py run` with a new output directory, 1200-second timeout, 15-second grace and the existing GGUF, graph, region and fixtures. The tool takes `--gguf`, `--region`, `--graph`, `--fixtures` and `--report`. Its report path must be a new JSON file in an existing directory. GO is issued by the supervisor only after exact READY. The model digest must already exist in the cache; the tool never hashes weights.

Minor reviewed limitations remain explicit: initial activation accounting precedes lazy scratch allocation; short successful VM replies gate physical footprint but not the earlier virtual/resident fields; oversized reports are rejected after END; encoder-forced false gain/installable flags are not a permission system. They did not occur as failures in this native run.

API definitions were checked against the installed macOS SDK and Apple’s public [task info header](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/task_info.h). No Apple source was copied. Existing dependency licenses are unchanged.

No new model download, automatic rehash, main merge, default activation, process enumeration, unrelated process signaling, or production gain claim occurred. The rejected phase53 candidate remains unpromoted.
