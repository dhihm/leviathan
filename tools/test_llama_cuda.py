"""Verify the Leviathan CLI's llama.cpp CUDA backend or its fail-closed path."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="public Stories15M F32 GGUF")
    parser.add_argument("--expect-unavailable", action="store_true")
    args = parser.parse_args()
    command = [
        str(args.cli.resolve()), "llama-generate", "--file", str(args.model),
        "--prompt-ids", "1", "--gen-tokens", "5",
    ]
    cuda = subprocess.run(
        command + ["--backend", "cuda", "--cuda-library", str(args.library.resolve())],
        text=True, capture_output=True, timeout=180,
    )
    if args.expect_unavailable:
        assert cuda.returncode != 0, cuda.stdout
        assert "CUDA backend or NVIDIA device is unavailable" in cuda.stderr, cuda.stderr
        assert "generated token IDs:" not in cuda.stdout, cuda.stdout
        print("PASS: explicit CUDA rejects the non-CUDA library without producing CPU tokens")
        return

    assert cuda.returncode == 0, cuda.stderr
    cpu = subprocess.run(command + ["--backend", "cpu-scalar"],
                         text=True, capture_output=True, timeout=180)
    assert cpu.returncode == 0, cpu.stderr
    expected = "generated token IDs: 9038,2501,263,931,29892"
    assert expected in cpu.stdout.splitlines(), cpu.stdout
    assert expected in cuda.stdout.splitlines(), cuda.stdout
    assert "backend: cuda (llama.cpp)" in cuda.stdout.splitlines(), cuda.stdout
    assert any(line.startswith("device: CUDA") and
               "completed CUDA matmul in each of 5 decode calls" in line
               for line in cuda.stdout.splitlines()), cuda.stdout
    print(cuda.stdout, end="")
    print("PASS: Stories15M CPU/CUDA token parity and per-decode CUDA evaluation")


if __name__ == "__main__":
    main()
