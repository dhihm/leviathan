# Windows / AMD (Vulkan): Ornith-1.5-9B

The second backend target. The same Forge design as the Mac/Metal path (op spec, device-derived schedule, emitted kernel source, correctness gate, measured selection, cached artifact) drives a Rust implementation that runs Ornith-1.5-9B (`qwen35`, Q4_K_M GGUF) on a Radeon 890M iGPU through Vulkan.

Code: `rust/` (workspace: `lev-gguf`, `lev-forge`, `lev-vulkan`, `lev-model`, `lev-cli`), benchmark tooling in `tools/bench/`.

## Target

| | |
| :--- | :--- |
| Machine | ASUS ROG Ally X (handheld, AMD APU with Radeon 890M, shared LPDDR5X memory), Windows 11 |
| GPU | Radeon 890M (RDNA 3.5, UMA, 15.6 GB LPDDR5X-8000), subgroup 64 default, wave32 selectable |
| API | Vulkan 1.3 through `ash`, kernels as GLSL compiled with `glslc`, SPIR-V cached by SHA-256 of the source |
| Model | Ornith-1.5-9B Q4_K_M: 32 layers (24 gated-delta-net SSM, 8 full attention) plus an MTP block |
| Reference | llama.cpp b11393 Vulkan, same model file |

## What is implemented

- **Decode**: int8-activation Q4_K / Q6_K matrix-vector kernels (integer dot product), GPU-side embedding and argmax, fused norm+quantize and SwiGLU+quantize kernels (bit-identical to the unfused ones), schedules derived per column count and verified against a CPU reference.
- **MTP speculative decoding**: the model's own MTP head drafts one token; a 2-column batched verification pass with double-buffered recurrent state accepts or rolls back. Greedy output equals plain greedy decoding up to int8 numerical noise (see below).
- **Prefill**: cooperative-matrix (WMMA, f16 in, f32 accumulate) GEMMs for Q4_K and Q6_K over 64 or 128 token chunks, with whole-block dequantization into shared memory, a two-lane chunked gated-delta kernel and a chunked SSM convolution.
- **Measurement**: `lev gpu-profile` per-dispatch timestamps, driver statistics through `VK_KHR_pipeline_executable_properties` (ISA, VGPR, LDS, scratch), and an interleaved benchmark with AMD ADL telemetry (`tools/bench/soak.ps1`, `plot-soak.ps1`).

## Results

All numbers below were measured on the ROG Ally X above (a power-limited handheld, so the sustained state differs from a desktop GPU). They were measured by alternating the two engines on the same GPU for 80 s slices, three rounds, with GPU telemetry running, so both sides see the same power and thermal history. The sustained state on this machine is about 35 W ASIC power and about 80 C. Prompt: 512 tokens, 64 generated, greedy.

| | this engine | llama.cpp | ratio |
| :--- | :--- | :--- | :--- |
| prefill (tok/s) | 292 - 294 | 312 | 0.94 |
| plain decode (tok/s) | 18.7 | 18.4 | 1.02 |
| MTP speculative decode (tok/s) | 30.4 (acceptance 0.88) | not applicable | 1.65 vs. llama.cpp plain |

In a short cold-GPU run (boost power state, before the sustained state is reached) prefill is 305 tok/s here against 406 - 415 tok/s for llama.cpp (0.75). Under sustained load llama.cpp drops to about 312 tok/s; the cause of the larger drop was not investigated.

Caveats:

- One synthetic prompt and three rounds. Numbers describe this one ROG Ally X in the sustained 35 W state only.
- The speculative-decode rate depends on draft acceptance, which depends on the prompt (0.88 on this prompt in the latest run, 0.73 in an earlier run of the same prompt before a numerically different kernel change). Use natural-language prompts when comparing.
- Decode is memory-bandwidth bound (about 98 GB/s effective here, against a measured read ceiling of about 103 GB/s); plain decode is 2% ahead of llama.cpp, within what one prompt and one machine can show.
- Int8 activation quantization amplifies floating-point accumulation-order differences on some inputs: two correct matvec schedules produced residual streams that differ by more than 10x at layer 10 on a repetitive synthetic prompt. Spec-versus-plain divergences on such inputs come from this and are not specific to the 2-column pass (the kernels match their sequential references to 1e-7 or exactly).

## Reproduce

```powershell
cd rust
cargo build --release -p lev-cli
$lev = "<target dir>\release\lev.exe"
& $lev vk-devices                                   # Vulkan devices and extensions
& $lev gpu-check <model.gguf> <comma separated token ids>   # GPU vs CPU reference
& $lev gpu-spec  <model.gguf> <ids> 64              # prefill, plain and speculative decode
& $lev vk-gemm   <model.gguf> 12288                 # GEMM microbenchmark and correctness gate
..\tools\bench\soak.ps1 -Model <model.gguf> -Ids <ids file> -OutDir <dir> -Rounds 3 -SliceSeconds 80
```

Environment switches (debugging and ablation): `LEV_NO_FUSE`, `LEV_NO_BIG` (disable matrix-core prefill), `LEV_MAX_BIG` (64 or 128 token chunks), `LEV_NO_CHUNK`, `LEV_NO_MTP_CHUNK`, `LEV_MTP_VOCAB`, `LEV_OLD_SCHED`, `LEV_MAX_LAYERS`.

## Open items

- Overlap dequantization with matrix multiply (double buffering), larger token tiles.
- Attention and gated-delta chunk kernels (about 5% of a chunk each).
- Benchmark on natural-language prompts; investigate the layer-10 error amplification (outlier channels, group quantization).
- DRAM/SSD offload and MoE models.
