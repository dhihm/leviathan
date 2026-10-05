# Router overlap matched performance result

The bounded matched comparison completed on the M1 Max with all three correctness gates passing. The candidate is rejected for promotion: every fixture failed the fixed 5% stability limit and none met the 2% gain threshold. No default or artifact was changed.

| Fixture | Baseline TPOT (ms) | Candidate TPOT (ms) | Gain | Baseline spread | Candidate spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| en-chat | 98.985 | 100.413 | -1.44% | 15.17% | 15.91% |
| kr-chat | 98.434 | 101.348 | -2.96% | 27.46% | 17.40% |
| raw-hello | 99.070 | 98.910 | 0.16% | 5.56% | 12.20% |

All 48 complete runs retained their raw continuation timings: six excluded warmups and 42 measured runs in seven fixed AB/BA pairs per fixture. Every run passed token, finite-value, counter and fallback checks. The repeated qualification compared all 32 output steps per fixture with zero ON/OFF bit mismatches; the largest reference absolute difference was below 0.0001.

The supervisor observed END, actual exit code 3 and successful lock cleanup after 410.913562 seconds. Exit 3 records the failed performance gates, not a crash. Runtime state was restored and identity failures were empty.

The parent reserved a known-task coordination window; this does not establish that all user or system work was idle. No applications were stopped. Pre-GO and post-END thermal reports recorded no warning status, which does not certify an unthrottled device. Earlier prelaunch thermal queries were unavailable.

Aggregate host counters increased during the window: Pageins +2,246,533 pages, Pageouts +30,177 pages, Swapins +601,344 pages, Swapouts +816,524 pages, Compressions +14,132,226 pages, Decompressions +13,250,867 pages. These counters are system-wide and do not identify a process or prove the cause of variability.

The overlap keeps all 30 CPU router waits, adds 30 command buffers per continuation, and preserves the same 1,349 kernels. Overlap therefore cannot by itself remove the kernel-launch or router-readback floor. This experiment does not isolate which overhead caused the observed medians. GPU interval counters have different coverage across arms and are not GPU busy time or performance evidence.

The existing 14.44 GB model and cached digest were reused. No model download or weight rehash occurred. The unchanged candidate will not be rerun to seek a favorable sample. The next investigation must change a justified bottleneck or resource footprint rather than reinterpret these results.
