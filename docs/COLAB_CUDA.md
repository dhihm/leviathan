# Run Leviathan on a Colab NVIDIA GPU

Use a **GPU** Colab runtime for these cells. The existing
`leviathan-cli llama-generate` command runs the model through
the pinned llama.cpp CUDA library, loaded in the Swift process.
The library parses GGUF, places model layers and KV cache on CUDA,
and executes decoding. This backend uses llama.cpp kernels.
Forge's Dense/ReLU/Add value DAG and the Gemma Metal path have
separate execution contracts.

First use the [Colab CPU quickstart](COLAB_CPU.md) clone and Swift
installation cells. Then build the CLI and CUDA library. The GPU
runtime needs the CUDA toolkit (`nvcc`), a matching NVIDIA driver,
a C++ build toolchain, Git, and CMake.

```bash
%%bash
set -euo pipefail
cd /content/leviathan
export PATH="/content/leviathan-swift/usr/bin:$PATH"
swift build -c release --product leviathan-cli
nvidia-smi
nvcc --version
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git /content/llama.cpp
git -C /content/llama.cpp checkout --detach ed2153e708de813c35559102aed43b3085179c27
capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1{s/\.//;p;}')
test -n "$capability"
cmake -S tools/llama_cuda -B /content/leviathan-cuda \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/content/llama.cpp \
  -DLEVIATHAN_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$capability"
cmake --build /content/leviathan-cuda --target leviathan_llama_cuda --parallel 2
```

Run the final cell in the [CPU quickstart](COLAB_CPU.md) to download
and checksum the public Stories15M model. Use the same model and
prompt for the CUDA run:

```bash
%%bash
set -euo pipefail
cd /content/leviathan
.build/release/leviathan-cli llama-generate --file /tmp/stories15M.gguf \
  --prompt-ids 1 --gen-tokens 5 --backend cuda \
  --cuda-library /content/leviathan-cuda/libleviathan_llama_cuda.so
python3 tools/test_llama_cuda.py --cli .build/release/leviathan-cli \
  --model /tmp/stories15M.gguf \
  --library /content/leviathan-cuda/libleviathan_llama_cuda.so
```

The regression expects token IDs `9038,2501,263,931,29892` on both
backends. CUDA mode reports the CUDA device and checks for a completed
device-resident matrix multiply during prefill and every subsequent
decode call. llama.cpp can still run some operators on the host.
Missing CUDA hardware, a load failure, or a decode without GPU evidence
makes the command fail. There is no switch to the Swift CPU executor.
Pass only a trusted shared library you built, since it runs inside the
CLI process.

For a build-only check on a host without NVIDIA hardware, configure a
separate build directory with `-DLEVIATHAN_CUDA=OFF` and run
`test_llama_cuda.py --expect-unavailable` with the resulting library
(`.dylib` on macOS). This exercises real library loading and the refusal
to generate CPU tokens in explicit CUDA mode.

## Execute the Forge value DAG on CUDA

The Forge frontend can also run the existing validated
`leviathan-value-dag-1` IR directly on CUDA. This entry point uses
CUDA-enabled PyTorch and CUDA tensors. It checks the result against
the source module and reports a synchronized CUDA event and device
name. This path runs the captured Dense/ReLU/Add IR and does not
implement Metal bundle reuse or autotuning.

```bash
%%bash
set -euo pipefail
cd /content/leviathan
python3 - <<'PY'
import json
from pathlib import Path
import torch

assert torch.version.cuda and torch.cuda.is_available()
Path("/tmp/lev-example.json").write_text(json.dumps([[0.25] * 32]))
Path("/tmp/lev-input.json").write_text(json.dumps([[0.75] * 32]))
PY
python3 tools/forge_run/dag_cuda.py \
  --model tools/forge_run/examples/residual_block.py \
  --example-input /tmp/lev-example.json --input /tmp/lev-input.json
python3 -m unittest tools.forge_run.test_dag_cuda
```
