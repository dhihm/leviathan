"""Check resident Stories15M isolation and paired first-token timings."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import statistics
import subprocess
import tempfile
import time

from test_llama_timing import EXPECTED, FIELDS, MODEL_SHA256, invoke


def read_reply(process):
    assert select.select([process.stdout], [], [], 180)[0], "worker response timed out"
    line = process.stdout.readline()
    assert line, "worker closed stdout"
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == MODEL_SHA256
    cli, model = str(args.cli.resolve()), str(args.model.resolve())
    options = ["--backend", "cuda", "--cuda-library", str(args.library.resolve())] if args.library else []
    backend = "cuda" if args.library else "cpu-scalar"

    def independent(prompt):
        command = [cli, "llama-generate", "--file", model, "--prompt-ids", prompt,
                   "--gen-tokens", "5", *options]
        result, elapsed = invoke(command + ["--timing", "json"])
        report, = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{"backend":')]
        assert result.stdout.splitlines()[-1] == EXPECTED[prompt]
        assert report["backend"] == backend
        if args.library:
            assert "completed CUDA matmul in each of 5 decode calls" in result.stdout
        return row(report, elapsed)

    def row(timing, elapsed):
        phases = [timing[field] for field in FIELDS]
        return [*phases, sum(timing["decodeMillis"]), sum(phases), elapsed]

    rows = {prompt: [[], []] for prompt in EXPECTED}
    with tempfile.TemporaryFile(mode="w+t") as errors, subprocess.Popen(
            [cli, "llama-serve", "--file", model, *options], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=errors, text=True) as worker:
        try:
            ready = read_reply(worker)
            assert ready["kind"] == "ready" and ready["backend"] == backend
            request_id = 0

            def request(prompt, **extra):
                nonlocal request_id
                request_id += 1
                payload = {"id": request_id, "promptIds": list(map(int, prompt.split(","))), "count": 5, **extra}
                start = time.perf_counter()
                worker.stdin.write(json.dumps(payload) + "\n")
                worker.stdin.flush()
                answer = read_reply(worker)
                return answer, (time.perf_counter() - start) * 1000

            def served(prompt):
                answer, elapsed = request(prompt)
                assert answer["kind"] == "result" and answer["id"] == request_id, answer
                expected = list(map(int, EXPECTED[prompt].split(": ")[1].split(",")))
                assert answer["tokens"] == expected
                timing = answer["timing"]
                assert timing["backend"] == backend and timing["modelLoadMillis"] == 0
                assert len(timing["decodeMillis"]) == len(expected) - 1
                if args.library:
                    assert "completed CUDA matmul in each of 5 decode calls" in answer["device"]
                return row(timing, elapsed)

            prompts = list(EXPECTED)
            # Exclude the first request from warm statistics.
            served(prompts[0])
            bad, _ = request("2147483647")
            assert bad["kind"] == "error" and "tokens" not in bad, bad
            served(prompts[1])
            cancelled, _ = request(prompts[0], cancelAfterTokens=2)
            assert cancelled["kind"] == "cancelled" and "tokens" not in cancelled, cancelled
            served(prompts[1])
            unexpected, _ = request(prompts[0], model="another.gguf")
            assert unexpected["kind"] == "error"
            for i in range(6):
                for prompt in prompts:
                    if i % 2:
                        hot, fresh = served(prompt), independent(prompt)
                    else:
                        fresh, hot = independent(prompt), served(prompt)
                    rows[prompt][0].append(fresh)
                    rows[prompt][1].append(hot)
            worker.stdin.close()
            assert worker.wait(timeout=30) == 0
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
            errors.seek(0)
            if worker.returncode:
                print(errors.read())

    print(json.dumps(ready, sort_keys=True))
    names = (*FIELDS, "decodeTotalMillis", "timedPathToFirstTokenMillis", "processOrReplyWallMillis")
    for prompt, groups in rows.items():
        medians = [[round(statistics.median(column), 3) for column in zip(*samples)] for samples in groups]
        print(json.dumps({"prompt": prompt, "pairs": 6,
                          **{name: [v[i] for v in medians] for i, name in enumerate(names)}}))


if __name__ == "__main__":
    main()
