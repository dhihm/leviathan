"""Measure opt-in Stories15M timings and verify the production CLI contract."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

MODEL_SHA256 = "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"
EXPECTED = {
    "1": "generated token IDs: 9038,2501,263,931,29892",
    "1,9038,2501,263": "generated token IDs: 931,29892,727,471,263",
}
FIELDS = ("modelLoadMillis", "backendPreparationMillis", "prefillMillis", "firstTokenMillis")


def invoke(command):
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    elapsed = (time.perf_counter() - start) * 1000
    assert result.returncode == 0, result.stderr
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == MODEL_SHA256
    backend = "cuda" if args.library else "cpu-scalar"
    options = ["--backend", backend]
    if args.library:
        options.extend(["--cuda-library", str(args.library.resolve())])

    for prompt, expected in EXPECTED.items():
        base = [str(args.cli.resolve()), "llama-generate", "--file", str(args.model.resolve()),
                "--prompt-ids", prompt, "--gen-tokens", "5"]
        reference, _ = invoke(base)
        assert reference.stderr == ""
        tokens = reference.stdout.splitlines()[-1]
        assert tokens == expected
        if args.library:
            legacy, _ = invoke(base + options)
            assert legacy.stdout.splitlines()[-1] == tokens
            assert "completed CUDA matmul in each of 5 decode calls" in legacy.stdout
            assert "leviathan-llama-timing-1" not in legacy.stderr
        else:
            legacy = reference
        rows = []
        for _ in range(6):
            result, wall = invoke(base + options + ["--timing", "json"])
            assert result.stdout == legacy.stdout
            payloads = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{"backend":')]
            assert len(payloads) == 1
            report = payloads[0]
            assert report["format"] == "leviathan-llama-timing-1" and report["backend"] == backend
            values = [report[field] for field in FIELDS]
            decode = report["decodeMillis"]
            assert len(decode) == tokens.count(",")
            assert all(math.isfinite(value) and value >= 0 for value in values + decode)
            assert sum(values + decode) <= wall + 1
            rows.append([*values, sum(decode), statistics.median(decode), wall])
        names = (*FIELDS, "decodeTotalMillis", "decodeMedianPerTokenMillis", "processWallMillis")
        print(json.dumps({"backend": backend, "prompt": prompt, "tokens": tokens.split(": ")[1],
                          "samples": len(rows), "millis": {
                              name: [round(f(column), 3) for f in (statistics.median, min, max)]
                              for name, column in zip(names, zip(*rows), strict=True)
                          }}), flush=True)


if __name__ == "__main__":
    main()
