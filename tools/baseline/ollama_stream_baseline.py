#!/usr/bin/env python3
"""Bounded stdlib-only streaming Ollama baseline: raw "Hello", greedy
settings, num_predict 8. Saves the request, every JSON chunk with
monotonic client receipt timestamps, and the server's final durations
separately. Reports first-nonempty-chunk TTFT and first-to-last
nonempty-chunk interval. Client stream metrics are labeled separately
from the API's server metrics; no internal-ID claim is made. Fails
closed: malformed JSON, missing done:true, an error field, or counts
other than prompt_eval_count=2 / eval_count=8 exit nonzero with full
diagnostics saved. Bounds: 64 chunks, 1 MiB/line, 4 MiB total."""
import json, time, urllib.request, sys

MAX_CHUNKS, MAX_LINE, MAX_TOTAL = 64, 1 << 20, 4 << 20
PORT = 11499
req_body = {
    "model": "gemma4-task5", "prompt": "Hello", "raw": True, "stream": True,
    "options": {"temperature": 0, "top_k": 1, "top_p": 1, "repeat_penalty": 1,
                "seed": 42, "num_predict": 8, "num_ctx": 4096},
    "keep_alive": 0,
}
url = f"http://localhost:{PORT}/api/generate"
data = json.dumps(req_body).encode()
request = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
chunks = []
total_bytes = 0
t0 = time.monotonic()
fail = None
try:
    with urllib.request.urlopen(request, timeout=600) as resp:
        while True:
            line = resp.readline(MAX_LINE + 1)
            if not line:
                break
            t = time.monotonic()
            total_bytes += len(line)  # blank lines count toward totals too
            if len(line) > MAX_LINE or total_bytes > MAX_TOTAL:
                fail = f"bound exceeded: line {len(line)}B, total {total_bytes}B"
                break
            if len(chunks) >= MAX_CHUNKS:
                fail = f"chunk bound {MAX_CHUNKS} exceeded"
                break
            if not line.strip():
                chunks.append({"t_rel_s": round(t - t0, 6), "blank": True})
                continue
            if len(line) > MAX_LINE or total_bytes > MAX_TOTAL:
                fail = f"bound exceeded: line {len(line)}B, total {total_bytes}B"
                break
            if len(chunks) >= MAX_CHUNKS:
                fail = f"chunk bound {MAX_CHUNKS} exceeded"
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                fail = f"malformed JSON chunk: {e}"
                chunks.append({"t_rel_s": round(t - t0, 6),
                               "raw_prefix": line[:200].decode("utf-8", "replace")})
                break
            if not isinstance(obj, dict):
                fail = f"JSON chunk is not an object: {type(obj).__name__}"
                break
            chunks.append({"t_rel_s": round(t - t0, 6), "chunk": obj})
            if obj.get("error"):
                fail = f"server error field: {obj['error']}"
                break
except Exception as e:
    fail = f"transport: {e!r}"

out_path = sys.argv[1]
report = {"request": req_body, "chunk_count": len(chunks), "failure": fail}
def save():
    with open(out_path, "w") as f:
        json.dump({"request": req_body, "chunks": chunks, "report": report}, f, indent=1)

if fail:
    print(json.dumps(report, indent=1))
    save()
    sys.exit(1)

final = None
for c in reversed(chunks):
    if isinstance(c.get("chunk"), dict) and c["chunk"].get("done"):
        final = c["chunk"]
        break
if final is None:
    report["failure"] = "no final done:true chunk"
    print(json.dumps(report, indent=1))
    save()
    sys.exit(1)
if final.get("error"):
    fail = f"final error: {final['error']}"
if final.get("prompt_eval_count") != 2:
    fail = f"prompt_eval_count {final.get('prompt_eval_count')} != 2"
if final.get("eval_count") != 8:
    fail = f"eval_count {final.get('eval_count')} != 8"
nonempty = [c for c in chunks if isinstance(c.get("chunk"), dict) and c["chunk"].get("response")]
text = "".join(c["chunk"]["response"] for c in nonempty)
if text != " upon-the-sands-of":
    fail = f"unexpected text {text!r} (expected ' upon-the-sands-of')"

if fail:
    report["failure"] = fail
    print(json.dumps(report, indent=1))
    save()
    sys.exit(1)

report.update({
    "failure": None,
    "nonempty_chunk_count": len(nonempty),
    "text": text,
    "client_stream": {
        "ttft_first_nonempty_s": nonempty[0]["t_rel_s"] if nonempty else None,
        "first_to_last_nonempty_interval_s":
            (nonempty[-1]["t_rel_s"] - nonempty[0]["t_rel_s"])
            if len(nonempty) > 1 else None,
        "note": ("client stream metrics; NOT token-interval TPS unless "
                 "nonempty_chunk_count == eval_count"),
        "nonempty_matches_eval_count": len(nonempty) == final["eval_count"],
    },
    "api_server_metrics": {
        "load_duration_s": final.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": final.get("prompt_eval_duration", 0) / 1e9,
        "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_duration_s": final.get("eval_duration", 0) / 1e9,
        "eval_count": final.get("eval_count"),
        "total_duration_s": final.get("total_duration", 0) / 1e9,
    },
})
print(json.dumps(report, indent=1))
save()
