#!/usr/bin/env python3
"""Matched warm/cold Ollama acceptance baseline (stdlib only).

Protocol (matches Leviathan's resident gemma-generate --repeat harness):
  - the GGUF declared for the installed Ollama 0.23.0 server model
    gemma4-task5 (GGUF SHA-256 from the shared cache; blob not hashed),
  - raw prompt "Hello", greedy (temperature 0, top_k 1, top_p 1,
    repeat_penalty 1, seed 42), num_predict 8, num_ctx 4096, streaming,
  - COLD rounds (default 3): keep_alive 0 so the model unloads after each
    request; unload is VERIFIED by polling GET /api/ps before the next round.
    "Cold" means the model is not resident in the server process; the OS page
    cache is NOT purged, so storage is not guaranteed cold.
  - WARM rounds (default 5): keep_alive -1 keeps the model loaded; round 1
    still pays the load, rounds >= 2 are warm.
Every round fails closed on: transport/JSON errors, missing done:true, an
error field, prompt_eval_count != 2, eval_count != 8, or text other than
" upon-the-sands-of". Client TTFT (request start to first nonempty chunk)
is reported SEPARATELY from the API's server-side load/prompt_eval/eval
durations. Only exposed text/count agreement is claimed — hidden token IDs
are not observable through this API and are not claimed.
Ollama memory is sampled with ps (no privilege): the user's own daemon on
11434 is never touched; this harness talks ONLY to the task-owned daemon
on 11499 (verified alive by /api/version + /api/ps before any request).
Samples cover the serve processes AND the per-model runner child (found
by the GGUF blob path in its command line) while the model is resident.
ps RSS is KiB (divisor 1024); these are point samples with the runner's
shared runtime overhead included — not a residency or peak-memory claim.
Model identity policy cached-explicit-refresh-v1: the GGUF SHA-256 comes
from the shared model digest cache (hashed once on a cache miss or on an
explicit LEVIATHAN_REHASH_MODEL=1 refresh, never re-read on a hit, so it
is a historical digest rather than a fresh verification). /api/show's
parent_model blob is never opened or hashed; the digest in its file name
is declared metadata, only compared with the cached GGUF digest, never
certified as the blob's current bytes. Preflight still requires that
no leviathan-cli process is resident (sequential GPU use).
Bounds: 64 chunks, 1 MiB/line, 4 MiB total per request; 300 s unload poll.
"""
import json, os, subprocess, sys, time, urllib.request

from model_digest_cache import MalformedRecord
from model_identity import MODEL_FILE_VERIFICATION, resolve_model_identity

MAX_CHUNKS, MAX_LINE, MAX_TOTAL = 64, 1 << 20, 4 << 20
PORT = 11499
MODEL = "gemma4-task5"
EXPECTED_TEXT = " upon-the-sands-of"
COLD_ROUNDS, WARM_ROUNDS = 3, 5

def req_body(keep_alive):
    return {"model": MODEL, "prompt": "Hello", "raw": True, "stream": True,
            "options": {"temperature": 0, "top_k": 1, "top_p": 1,
                        "repeat_penalty": 1, "seed": 42, "num_predict": 8,
                        "num_ctx": 4096},
            "keep_alive": keep_alive}

def one_request(keep_alive):
    """Returns (record, failure). record has client + api server metrics."""
    body = req_body(keep_alive)
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://localhost:{PORT}/api/generate", data=data,
        headers={"Content-Type": "application/json"})
    # Initialized BEFORE the request so mid-loop updates cannot hit an
    # unbound name; chunk_count is finalized after the loop.
    rec = {"keep_alive": keep_alive, "chunk_count": 0}
    chunks, total, t0, fail = [], 0, time.monotonic(), None
    try:
        with urllib.request.urlopen(request, timeout=600) as resp:
            while True:
                line = resp.readline(MAX_LINE + 1)
                if not line:
                    break
                t = time.monotonic()
                total += len(line)
                if len(line) > MAX_LINE or total > MAX_TOTAL:
                    fail = f"bound exceeded: line {len(line)}B total {total}B"
                    break
                if len(chunks) >= MAX_CHUNKS:
                    fail = f"chunk bound {MAX_CHUNKS} exceeded"
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    fail = f"malformed JSON chunk: {e}"
                    break
                if not isinstance(obj, dict):
                    fail = f"chunk is not an object"
                    break
                chunks.append((t, obj))
                if obj.get("error"):
                    fail = f"server error field: {obj['error']}"
                    break
    except Exception as e:
        fail = f"transport: {e!r}"
    rec["chunk_count"] = len(chunks)
    if fail:
        rec["failure"] = fail
        return rec, fail
    final = next((o for _, o in reversed(chunks) if o.get("done")), None)
    if final is None:
        return rec, "no final done:true chunk"
    if final.get("error"):
        return rec, f"final error: {final['error']}"
    if final.get("prompt_eval_count") != 2:
        return rec, f"prompt_eval_count {final.get('prompt_eval_count')} != 2"
    if final.get("eval_count") != 8:
        return rec, f"eval_count {final.get('eval_count')} != 8"
    nonempty = [(t, o) for t, o in chunks if o.get("response")]
    text = "".join(o["response"] for _, o in nonempty)
    if text != EXPECTED_TEXT:
        return rec, f"unexpected text {text!r}"
    done_t = chunks[-1][0]
    rec.update({
        "failure": None, "text": text, "nonempty_chunk_count": len(nonempty),
        "client": {
            "ttft_first_nonempty_s": nonempty[0][0] - t0,
            "total_to_done_s": done_t - t0,
        },
        "api_server": {
            "load_duration_s": final.get("load_duration", 0) / 1e9,
            "prompt_eval_duration_s": final.get("prompt_eval_duration", 0) / 1e9,
            "prompt_eval_count": final.get("prompt_eval_count"),
            "eval_duration_s": final.get("eval_duration", 0) / 1e9,
            "eval_count": final.get("eval_count"),
        },
    })
    return rec, None

def model_loaded():
    """True while /api/ps lists MODEL (i.e. resident in the server)."""
    try:
        with urllib.request.urlopen(
                f"http://localhost:{PORT}/api/ps", timeout=30) as r:
            ps = json.load(r)
    except Exception as e:
        sys.exit(f"cannot read /api/ps: {e!r}")
    return any(m.get("name", "").startswith(MODEL) or
               m.get("model", "").startswith(MODEL) for m in ps.get("models", []))

def wait_unloaded(timeout_s=300.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not model_loaded():
            return time.monotonic() - t0
        time.sleep(2.0)
    sys.exit("model still loaded after unload poll timeout")

def rss_samples_mib():
    """ps RSS (MiB, KiB/1024) for serve processes and, while the model is
    resident, the per-model runner child (blob path in its command line).
    Point samples including shared runtime overhead — not peak claims."""
    out = {}
    try:
        for pattern, key in (("ollama serve", "serve"),
                             ("3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d",
                              "runner_child")):
            pids = subprocess.run(["pgrep", "-f", pattern],
                                  capture_output=True, text=True, timeout=10)
            for pid in pids.stdout.split():
                r = subprocess.run(["ps", "-o", "rss=", "-p", pid],
                                   capture_output=True, text=True, timeout=10)
                rss = r.stdout.strip()
                if rss.isdigit():
                    out[f"{key}_pid{pid}"] = round(int(rss) / 1024.0, 1)
    except Exception as e:
        out["error"] = repr(e)
    return out

def load_average():
    try:
        o = subprocess.run(["sysctl", "-n", "vm.loadavg"],
                           capture_output=True, text=True, timeout=10)
        return o.stdout.strip()
    except Exception as e:
        return repr(e)

def main(out_path):
    # Preflight, in order: task-owned daemon alive on 11499, no resident
    # leviathan-cli (sequential GPU), model exists, cached GGUF SHA-256 plus
    # declared blob metadata (the blob is never hashed).
    try:
        with urllib.request.urlopen(
                f"http://localhost:{PORT}/api/version", timeout=30) as r:
            version = json.load(r).get("version")
    except Exception as e:
        sys.exit(f"task-owned daemon on {PORT} not answering: {e!r}")
    resident = subprocess.run(["pgrep", "-fl", "leviathan-cli"],
                              capture_output=True, text=True, timeout=10)
    if resident.stdout.strip():
        sys.exit(f"leviathan-cli resident; refusing to share the GPU:\n"
                 f"{resident.stdout}")
    try:
        show = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:{PORT}/api/show",
            data=json.dumps({"model": MODEL}).encode(),
            headers={"Content-Type": "application/json"}), timeout=30))
    except Exception as e:
        sys.exit(f"model {MODEL} missing on {PORT}: {e!r}")
    blob_path = show.get("details", {}).get("parent_model")
    if not blob_path or not os.path.exists(blob_path):
        sys.exit(f"parent_model blob missing: {blob_path!r}")
    gguf = os.environ.get(
        "GGUF", "<workspace>/models/"
                "gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf")
    # GGUF SHA-256 from the shared digest cache (hashed once on a miss or on
    # an explicit LEVIATHAN_REHASH_MODEL=1 refresh; a hit reads no model
    # bytes). The Ollama blob is never opened or hashed.
    try:
        identity = resolve_model_identity(gguf, blob_path)
    except MalformedRecord as e:
        sys.exit(f"model digest cache: {e}")
    print(json.dumps({"model_file_identity": identity}, sort_keys=True))
    if model_loaded():
        sys.exit("model already loaded at start; run a keep_alive:0 request "
                 "first or wait for unload so round 1 is honestly cold")
    result = {"model": MODEL, "port": PORT, "ollama_version": version,
              "loadavg_start": load_average(),
              "warm_protocol_note": ("Ollama's server may reuse the KV "
                                     "prefix cache across identical warm "
                                     "requests (observed warm "
                                     "prompt_eval_duration far below "
                                     "cold); Leviathan's harness resets "
                                     "and re-prefills every repetition. "
                                     "TTFT/prefill numbers are therefore "
                                     "NOT strictly like-for-like; decode "
                                     "per-token medians are comparable."),
              "model_file_verification": MODEL_FILE_VERIFICATION,
              "model_file_identity": identity,
              "show_details": show.get("details"),
              "cold_note": ("model unloaded between rounds (verified via "
                            "/api/ps); OS page cache NOT purged — storage "
                            "is not guaranteed cold"),
              "rss_note": ("ps RSS point samples (KiB/1024 -> MiB) of the "
                           "task-owned serve process and the runner child "
                           "matched by blob path; include shared runtime "
                           "overhead — not residency or peak claims")}
    failures = []
    result["loadavg_end"] = None
    result["rss_before_cold_mib"] = rss_samples_mib()
    # Post-request RSS samples ONLY (never during streaming — pgrep/ps
    # would block the reader and inflate client timestamps/totals). The
    # model stays loaded for keep_alive -1 rounds, so the runner child
    # is sampled right after each warm request; cold rounds are sampled
    # before wait_unloaded(). Reported per phase is the MAX sample across
    # rounds — a sampled maximum across rounds, NOT a peak.
    sampled_max = {}
    def sample_rss(phase):
        for k, v in rss_samples_mib().items():
            if isinstance(v, float):
                cur = sampled_max.get(k)
                sampled_max[k] = v if cur is None else max(cur, v)
        result[f"rss_max_{phase}_mib"] = dict(sorted(sampled_max.items()))
    result["cold"] = []
    for i in range(COLD_ROUNDS):
        rec, fail = one_request(0)
        sample_rss("cold")
        result["cold"].append(rec)
        print(json.dumps(rec))
        if fail:
            failures.append(f"cold round {i + 1}: {fail}")
            break
        wait_unloaded()
    result["rss_after_cold_mib"] = rss_samples_mib()
    result["warm"] = []
    if not failures:
        for i in range(WARM_ROUNDS):
            rec, fail = one_request(-1)
            sample_rss("warm")
            result["warm"].append(rec)
            print(json.dumps(rec))
            if fail:
                failures.append(f"warm round {i + 1}: {fail}")
                break
    result["rss_after_warm_mib"] = rss_samples_mib()
    result["loadavg_end"] = load_average()
    # Restore the pre-run state: unload the model we asked to stay resident.
    if model_loaded():
        try:
            data = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request(
                f"http://localhost:{PORT}/api/generate", data=data,
                headers={"Content-Type": "application/json"}), timeout=120).read()
        except Exception as e:
            failures.append(f"cleanup unload failed: {e!r}")
    result["failures"] = failures
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    if failures:
        print(json.dumps(failures, indent=1))
        sys.exit(1)
    print("OLLAMA_WARMCOLD_PASS")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ollama-warmcold.json")
