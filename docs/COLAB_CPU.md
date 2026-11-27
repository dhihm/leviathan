# Run Leviathan on a Colab CPU runtime

The Swift CLI's `llama-generate` command runs dense, bias-free Llama
GGUF models with F32/F16 weights and unscaled RoPE. It takes prompt
token IDs and prints greedy continuation IDs, including EOS if emitted.
The CPU backend is the default for this command, and
`--backend cpu-scalar` selects it explicitly. The model loader allows
up to 512 MiB of decoded weights, 512 context positions, and 128 new
tokens per call. Gemma, quantized GGUF, text tokenization, and CUDA are
outside this CPU path.

In a new Google Colab notebook, select a **CPU** runtime. Run the
following cells in order. They need only the public Leviathan branch
and a fixture generated locally in the notebook.

```bash
%%bash
set -euo pipefail
cd /content
git clone --depth 1 --branch experiment/router-overlap-phase53 \
  https://github.com/dhihm/leviathan.git
```

Use Swift 6.2.3 for Ubuntu x86_64. This cell selects the matching package
for Ubuntu 22.04 or 24.04. See the [official Swift Linux installation
page](https://www.swift.org/install/linux/) for other environments.

```bash
%%bash
set -euo pipefail
source /etc/os-release
test "$(uname -m)" = x86_64
case "$ID:$VERSION_ID" in
  ubuntu:22.04|ubuntu:24.04) ;;
  *) echo "Use the Swift toolchain for $ID $VERSION_ID"; exit 1 ;;
esac
release=swift-6.2.3-release
platform="ubuntu${VERSION_ID/./}"
curl -fL --retry 3 \
  "https://download.swift.org/$release/$platform/swift-6.2.3-RELEASE/swift-6.2.3-RELEASE-ubuntu$VERSION_ID.tar.gz" \
  -o /tmp/leviathan-swift.tar.gz
mkdir -p /content/leviathan-swift
tar -xzf /tmp/leviathan-swift.tar.gz --strip-components=1 -C /content/leviathan-swift
/content/leviathan-swift/usr/bin/swift --version
```

```bash
%%bash
set -euo pipefail
cd /content/leviathan
export PATH="/content/leviathan-swift/usr/bin:$PATH"
swift build -c release --product leviathan-cli
python3 tools/test_llama_cpu.py --write-fixture /tmp/leviathan-tiny.gguf
.build/release/leviathan-cli llama-generate --file /tmp/leviathan-tiny.gguf \
  --prompt-ids 0 --gen-tokens 4 --backend cpu-scalar
python3 tools/test_llama_cpu.py --cli .build/release/leviathan-cli
```

The generated fixture is a synthetic one-layer model. The expected
generation is `backend: cpu-scalar` followed by
`generated token IDs: 1,2`. The second token is EOS. To run locally,
use the third cell's commands from the repository root with Swift 6+
on your path.

For a trained public model, the same CLI accepts the
[Stories15M GGUF](https://huggingface.co/ggml-org/tiny-llamas/blob/main/stories15M.gguf)
(about 98 MB). From the repository root after the release build:

```bash
%%bash
set -euo pipefail
cd /content/leviathan
curl -fL --retry 3 \
  https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M.gguf \
  -o /tmp/stories15M.gguf
echo '61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a  /tmp/stories15M.gguf' \
  | sha256sum -c
.build/release/leviathan-cli llama-generate --file /tmp/stories15M.gguf \
  --prompt-ids 1 --gen-tokens 5
```
