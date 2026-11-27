"""Check pinned Llama model isolation and paired first-token timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import select
import statistics
import subprocess
import sys
import tempfile
import time

from test_llama_timing import EXPECTED, FIELDS, MODEL_SHA256, invoke


SMOLLM2_SHA256 = "f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
SMOLLM2_EXPECTED = {
    "6403,1980,253,655": "generated token IDs: 28,339,436,253,1805",
    "504,3575,282,4649,314": "generated token IDs: 7042,30,7042,314,260",
}


def memory(pid, cuda):
    result = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    snapshot = {"hostRSSMiB": round(int(result.strip()) / 1024, 3)}
    if sys.platform == "darwin":
        detail = subprocess.run(["vmmap", "-summary", str(pid)], text=True, capture_output=True, timeout=15)
        footprint = re.search(r"^Physical footprint:\s+(\S+)", detail.stdout, flags=re.MULTILINE)
        if footprint:
            snapshot["hostPhysicalFootprint"] = footprint.group(1)
    if cuda:
        usage = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        )
        if usage.returncode == 0:
            for line in usage.stdout.splitlines():
                row = [field.strip() for field in line.split(",")]
                if len(row) == 2 and row[0] == str(pid) and row[1].isdigit():
                    snapshot["gpuProcessMiB"] = int(row[1])
    return snapshot


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
    parser.add_argument("--fixture", choices=("stories15m", "smollm2-135m"), default="stories15m")
    args = parser.parse_args()
    smol = args.fixture == "smollm2-135m"
    digest = hashlib.sha256()
    with args.model.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == (SMOLLM2_SHA256 if smol else MODEL_SHA256)
    cli, model = str(args.cli.resolve()), str(args.model.resolve())
    options = ["--backend", "cuda", "--cuda-library", str(args.library.resolve())] if args.library else []
    backend = "cuda" if args.library else "cpu-scalar"

    def fresh_command(prompt, backend_options):
        return [cli, "llama-generate", "--file", model, "--prompt-ids", prompt,
                "--gen-tokens", "5", *backend_options]

    expected = SMOLLM2_EXPECTED if smol else EXPECTED
    if smol:
        for prompt, want in expected.items():
            reference, _ = invoke(fresh_command(prompt, []))
            assert not reference.stderr
            assert reference.stdout.splitlines() == ["backend: cpu-scalar", want], reference.stdout
            if args.library:
                untimed, _ = invoke(fresh_command(prompt, options))
                assert untimed.stdout.splitlines()[-1] == want, untimed.stdout
                assert "completed CUDA matmul in each of 5 decode calls" in untimed.stdout

    def independent(prompt):
        command = fresh_command(prompt, options)
        result, elapsed = invoke(command + ["--timing", "json"])
        report, = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{"backend":')]
        assert result.stdout.splitlines()[-1] == expected[prompt], result.stdout
        assert report["backend"] == backend
        if args.library:
            assert "completed CUDA matmul in each of 5 decode calls" in result.stdout
        return row(report, elapsed)

    def row(timing, elapsed):
        phases = [timing[field] for field in FIELDS]
        return [*phases, sum(timing["decodeMillis"]), sum(phases), elapsed]

    rows = {prompt: [[], []] for prompt in expected}
    with tempfile.TemporaryFile(mode="w+t") as errors, subprocess.Popen(
            [cli, "llama-serve", "--file", model, *options], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=errors, text=True) as worker:
        try:
            ready = read_reply(worker)
            assert ready["kind"] == "ready" and ready["backend"] == backend
            snapshots = {"ready": memory(worker.pid, bool(args.library))} if smol else {}
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
                want = list(map(int, expected[prompt].split(": ")[1].split(",")))
                assert answer["tokens"] == want, answer
                timing = answer["timing"]
                assert timing["backend"] == backend and timing["modelLoadMillis"] == 0
                assert len(timing["decodeMillis"]) == len(want) - 1
                if args.library:
                    assert "completed CUDA matmul in each of 5 decode calls" in answer["device"]
                return row(timing, elapsed)

            prompts = list(expected)
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
            if smol:
                snapshots["afterRequests"] = memory(worker.pid, bool(args.library))
            worker.stdin.close()
            assert worker.wait(timeout=30) == 0
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
            errors.seek(0)
            logs = errors.read()
            if worker.returncode:
                print(logs)

    print(json.dumps(ready, sort_keys=True))
    if smol:
        print(json.dumps({"memory": snapshots, "cudaKVAllocations": sorted({
            line.strip() for line in logs.splitlines() if re.search(r"KV buffer size", line)
        })}, sort_keys=True))
    names = (*FIELDS, "decodeTotalMillis", "timedPathToFirstTokenMillis", "processOrReplyWallMillis")
    for prompt, groups in rows.items():
        medians = [[round(statistics.median(column), 3) for column in zip(*samples)] for samples in groups]
        print(json.dumps({"prompt": prompt, "pairs": 6, "tokens": expected[prompt].split(": ")[1],
                          **{name: [v[i] for v in medians] for i, name in enumerate(names)}}))


if __name__ == "__main__":
    main()
