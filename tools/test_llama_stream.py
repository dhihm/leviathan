"""Measure actual stdout receipt in a separate consumer process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import tempfile
import time

from test_llama_serve import SMOLLM2_EXPECTED, SMOLLM2_SHA256
from test_llama_timing import EXPECTED, MODEL_SHA256


class Consumer:
    def __init__(self, stdout):
        self.stdout = stdout
        self.pending = bytearray()
        self.received = 0

    def next(self):
        while b"\n" not in self.pending:
            assert select.select([self.stdout], [], [], 180)[0], "stdout read timed out"
            chunk = os.read(self.stdout.fileno(), 4096)
            self.received = time.perf_counter_ns()
            if not chunk:
                assert not self.pending, "incomplete stdout event"
                return None, self.received
            self.pending.extend(chunk)
        i = self.pending.index(b"\n")
        line = self.pending[:i]
        del self.pending[:i + 1]
        return json.loads(line), self.received


def measure(command, expected, stream, cuda):
    fmt = "leviathan-llama-serve-stream-1" if stream else "leviathan-llama-serve-1"
    argv = command + (["--stream", "jsonl"] if stream else [])
    rows = {prompt: [] for prompt in expected}
    ids = 0
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, bufsize=0) as worker:
        try:
            stdout = Consumer(worker.stdout)
            ready, _ = stdout.next()
            assert ready["kind"] == "ready" and ready["format"] == fmt
            assert ready["backend"] == ("cuda" if cuda else "cpu-scalar")

            def request(prompt, terminal="result", **options):
                nonlocal ids
                ids += 1
                payload = json.dumps(dict(id=ids, promptIds=list(map(int, prompt.split(","))),
                                          count=5, **options)).encode() + b"\n"
                started = time.perf_counter_ns()
                assert worker.stdin.write(payload) == len(payload)
                tokens = []
                first = None
                while True:
                    event, receipt = stdout.next()
                    assert event is not None and event["id"] == ids, event
                    if event["kind"] != "token":
                        break
                    assert stream and event["index"] == len(tokens), event
                    tokens.append(event["tokenId"])
                    if first is None:
                        first = receipt
                assert event["kind"] == terminal, event
                if terminal == "result":
                    want = expected[prompt]
                    assert event["tokens"] == want and tokens == (want if stream else []), event
                    if cuda:
                        assert "completed CUDA matmul in each of 5 decode calls" in event["device"]
                else:
                    assert "tokens" not in event
                    if terminal == "cancelled" and stream:
                        assert tokens == expected[prompt][:options["cancelAfterTokens"]]
                    else:
                        assert not tokens
                return ((first - started) / 1e6 if first is not None else None,
                        (receipt - started) / 1e6)

            prompts = list(expected)
            request(prompts[0])
            request("2147483647", terminal="error")
            request(prompts[1])
            request(prompts[0], terminal="cancelled", cancelAfterTokens=2)
            request(prompts[1])
            assert worker.stdin.write(b"not-json\n") == len(b"not-json\n")
            malformed, _ = stdout.next()
            assert malformed["kind"] == "error" and "id" not in malformed and "tokens" not in malformed
            for _ in range(6):
                for prompt in prompts:
                    rows[prompt].append(request(prompt))
            worker.stdin.close()
            assert stdout.next()[0] is None
            assert worker.wait(timeout=30) == 0
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
            if worker.returncode:
                errors.seek(0)
                print(errors.read().decode(errors="replace"))
    truncated = subprocess.run(argv, input=b'{"id":1', capture_output=True, timeout=180)
    assert truncated.returncode != 0 and b"unterminated request" in truncated.stderr
    assert [json.loads(line)["kind"] for line in truncated.stdout.splitlines()] == ["ready"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", choices=("stories15m", "smollm2-135m"), required=True)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    smol = args.fixture == "smollm2-135m"
    digest = hashlib.sha256()
    with args.model.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == (SMOLLM2_SHA256 if smol else MODEL_SHA256)
    expected = {prompt: list(map(int, value.split(": ")[1].split(",")))
                for prompt, value in (SMOLLM2_EXPECTED if smol else EXPECTED).items()}
    command = [str(args.cli.resolve()), "llama-serve", "--file", str(args.model.resolve())]
    if args.library:
        command.extend(["--backend", "cuda", "--cuda-library", str(args.library.resolve())])
    default = measure(command, expected, stream=False, cuda=bool(args.library))
    streamed = measure(command, expected, stream=True, cuda=bool(args.library))

    def summary(values):
        return [round(f(values), 3) for f in (statistics.median, min, max)]

    for prompt, rows in streamed.items():
        assert all(first is not None and first < completed for first, completed in rows)
        print(json.dumps(dict(fixture=args.fixture, backend="cuda" if args.library else "cpu-scalar",
                              prompt=prompt, samples=len(rows),
                              firstEventMillis=summary([first for first, _ in rows]),
                              streamDoneMillis=summary([done for _, done in rows]),
                              defaultDoneMillis=summary([done for _, done in default[prompt]]))), flush=True)


if __name__ == "__main__":
    main()
