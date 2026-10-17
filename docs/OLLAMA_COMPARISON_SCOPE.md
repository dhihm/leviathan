# Ollama comparison scope

The historical Ollama result is a useful gap indicator, but it is not a new matched benchmark against the current Leviathan executable. This review reads the existing probe source, saved results, cache-manager debug log, and model tensor directory. It starts no server and measures no new Ollama result.

| Condition | Historical Ollama probe | Published Leviathan phase32 comparison |
| --- | --- | --- |
| Hardware | Apple M1 Max on this shared Mac | Same Mac, measured later |
| Model bytes | Gemma 4 26B-A4B Q4_0 GGUF; model blob SHA matches original GGUF | Same original GGUF SHA |
| Quantization | Identical mixed GGUF tensor bytes | Identical bytes; 265 Q4_0 tensors, one Q6_K tied embedding/output tensor, 392 F32 tensors |
| Model SHA | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` | Same SHA |
| Prompt | Raw Hello, no chat template; two prompt tokens | Raw Hello, IDs 2 and 9259 |
| Generation | 32 tokens, temperature0, top_k1, top_p1, repeat_penalty1, seed42 | 32 greedy IDs; exact across compared arms |
| Model residency | Warm persistent model, keep_alive -1 | New process per arm; first full generation warmup excluded, subsequent generations warm |
| Cache prefix | Distractor before each measured round; cache manager proved one BOS token reused and one prompt token freshly evaluated | Reset and explicit BOS-only untimed prefill, then freshly evaluate the non-BOS prompt token |
| Capacity | num_ctx4096, one serial slot | Bounded max context4096, serial runner |
| KV/runtime precision | Debug log compiled f16 flash-attention variants; explicit cache-type setting was empty and cache precision was not separately recorded | F32 KV buffers and original F32 arithmetic; no reduced-precision cache conversion |
| Kernel backend | Ollama0.23.0 Metal runner; log reports flash attention | Original Leviathan Metal kernels with accepted E4/width32, parallel attention, GPU KV and prepared SSA |
| Timing | Server eval_count / eval_duration, excludes prompt/load counters | 31 synchronous decode steps, includes graph execution and required logits/argmax readback, excludes initialization/hash/prefill |
| Sampling | Five historical warm rounds | Three matched old/new process pairs; each warm arm median of two31-step wall totals |
| Token receipt timing | HTTP chunks buffered/coalesced; not TPOT measurements | Synchronous CLI; no HTTP chunk TPOT claim |

Ollama's historical median `32 / eval_duration` is 58.078 tokens/s. Using 31 instead of32 for that same recorded duration gives56.263 tokens/s. The count convention changes the value by about3.23%, so it cannot explain the large observed gap by itself. A remaining difference is the timing boundary and runtime/readback behavior. The two engines were not measured in alternating matched runs in that result, and no per-token Ollama IDs or full logits were independently captured; the returned text matches the current32-token raw-Hello sequence. Kernel/layout superiority or MLX causality cannot be established from these records.

The Q6_K tensor `token_embd.weight` is605,552,640bytes and also supplies the tied output projection. Leviathan's post-composition intrusive replay measured the Q6 output dispatch at32.684ms per decode token. Dividing tensor bytes by that diagnostic interval yields about18.5GB/s of nominal weight-byte throughput. This is a diagnostic ratio, not measured DRAM bandwidth; it includes the kernel's arithmetic and cannot establish a hardware bandwidth ceiling. The scalar one-row-per-thread access and existing tiled-Q6 alternative provide a concrete layout/parallelism hypothesis to test with strict gates.

Source records: [saved Ollama evidence](evidence/q4-simd-forge/ollama-current.json), [probe implementation](../tools/baseline/ollama_fairdecode.py), and [phase32 comparison](COMPOSITE_FORGE_BATCH.md). No new baseline claim or matched MLX comparison is made here.
