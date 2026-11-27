"""Generate a synthetic Llama GGUF and exercise the actual Swift CLI."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_fixture(path):
    meta = [
        string("general.architecture") + struct.pack("<I", 8) + string("llama")
    ]
    for key, value in {
        "llama.embedding_length": 4,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 1,
        "llama.context_length": 16,
        "llama.block_count": 1,
        "llama.feed_forward_length": 4,
        "llama.rope.dimension_count": 2,
        "tokenizer.ggml.eos_token_id": 2,
    }.items():
        meta.append(string(key) + struct.pack("<II", 4, value))

    identity = [float(i == j) for i in range(4) for j in range(4)]
    tensors = [
        ("token_embd.weight", [4, 3], identity[:12]),
        ("output_norm.weight", [4], [1] * 4),
        # Advance 0 -> 1 -> 2 (EOS). With token 2 the best score is token 0.
        ("output.weight", [4, 3], identity[8:12] + identity[:8]),
        ("blk.0.attn_norm.weight", [4], [1] * 4),
        ("blk.0.attn_q.weight", [4, 4], identity),
        ("blk.0.attn_k.weight", [4, 2], identity[:8]),
        ("blk.0.attn_v.weight", [4, 2], identity[:8]),
        ("blk.0.attn_output.weight", [4, 4], [0.125 * x for x in identity]),
        ("blk.0.ffn_norm.weight", [4], [1] * 4),
        ("blk.0.ffn_gate.weight", [4, 4], identity),
        ("blk.0.ffn_up.weight", [4, 4], identity),
        ("blk.0.ffn_down.weight", [4, 4], [0.125 * x for x in identity]),
    ]
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(meta)))
    header.extend(b"".join(meta))
    data = bytearray()
    for index, (name, shape, values) in enumerate(tensors):
        data.extend(bytes(-len(data) % 32))
        # Mix F32 and F16 to exercise both decode paths.
        half = index % 2
        header.extend(string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", half, len(data)))
        data.extend(struct.pack("<" + ("e" if half else "f") * len(values), *values))
    header.extend(bytes(-len(header) % 32))
    path.write_bytes(header + data)


def verify(cli, model):
    def generate(prompt, *flags):
        return subprocess.run(
            [str(cli), "llama-generate", "--file", str(model),
             "--prompt-ids", prompt, "--gen-tokens", "4", *flags],
            capture_output=True, text=True, timeout=30,
        )

    for prompt, flags, expected in [
        ("0", (), "1,2"),
        ("0,1", ("--backend", "cpu-scalar"), "2"),
    ]:
        result = generate(prompt, *flags)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "backend: cpu-scalar", f"generated token IDs: {expected}",
        ], result.stdout
        assert not result.stderr
        instrumented = generate(prompt, *flags, "--timing", "json")
        assert instrumented.returncode == 0 and instrumented.stdout == result.stdout
        timing = json.loads(instrumented.stderr)
        assert timing["format"] == "leviathan-llama-timing-1" and timing["backend"] == "cpu-scalar"
        assert len(timing["decodeMillis"]) == expected.count(",")

    result = generate("0", "--timing", "invalid")
    assert result.returncode != 0 and not result.stdout and "--timing must be json" in result.stderr

    result = generate("3")
    assert result.returncode != 0 and "token IDs" in result.stderr, result
    result = generate("0", "--backend", "cuda")
    assert result.returncode != 0 and "--cuda-library" in result.stderr, result

    model.write_bytes(model.read_bytes()[:32])
    result = generate("0")
    assert result.returncode != 0 and "truncated" in result.stderr, result
    print("PASS: Swift CPU GGUF load, prompt prefill, greedy generation, EOS, and rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--write-fixture", type=Path)
    command.add_argument("--cli", type=Path)
    args = parser.parse_args()
    if args.write_fixture:
        write_fixture(args.write_fixture)
    else:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tiny.gguf"
            write_fixture(model)
            verify(args.cli.resolve(), model)
