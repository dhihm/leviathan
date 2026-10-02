#!/usr/bin/env python3
"""Fair decode-focused Ollama baseline with AUTHORITATIVE KV-cache policy
proof from the runner's own debug log (stdlib only).

Server: the task-owned daemon on 127.0.0.1:11498 started with
OLLAMA_DEBUG=1, OLLAMA_NUM_PARALLEL=1, OLLAMA_MAX_LOADED_MODELS=1,
OLLAMA_MODELS=/tmp/task5-ollama-models (all non-secret, set by the
operator). The debug log (path passed as argv[2]) is the evidence source;
Ollama 0.23.0's cache manager logs per request:
  source=cache.go:151 msg="loading cache slot" id=0 cache=<len> prompt=<n> used=<u> remaining=<r>
observed semantics (measured 2026-10-02 UTC, probe A/B/D/C in the same log):
`used` is the number of prompt tokens served from the retained KV prefix
and `remaining` the number freshly evaluated; a distractor whose tokens
diverge after BOS leaves used=1 (BOS only). runner.parallel=1 appears in
the same log, proving ONE slot per runner.

Protocol:
  1. CONTROL: request A loads the model (keep_alive -1) with the
     measured prompt at num_predict 1; identical request B follows. For
     this 2-token prompt the control also shows used=1 (BOS-only reuse —
     Ollama re-evaluates the final prompt token either way), so the
     control documents the reuse signature, not a longer prefix. Both
     are recorded; no performance claim is taken.
  2. MEASURED ROUNDS (default 5, warm model): each round is preceded by a
     serial distractor request (raw "Xqztv probe", num_predict 1,
     keep_alive -1). The measured request uses the canonical inputs (raw
     "Hello", temperature 0, top_k 1, top_p 1, repeat_penalty 1, seed 42,
     num_ctx 4096, num_predict 32; EOS-aware: done before 32 tokens is
     accepted with its done_reason recorded).
  3. EVICTION PROOF per measured round, fail-closed from the LOG (not
     timing): the round's "loading cache slot" line must show prompt=2
     and remaining>=1 — the non-BOS prompt token was freshly evaluated.
     The line is copied into the evidence JSON verbatim.
  4. Determinism: text, eval_count, done_reason must match round 1.
Decode metric: server-side eval_duration/eval_count (eval_count-1
first-to-last intervals, matching Leviathan's decode accounting of
generated-1 steps after the first token). Client chunk timestamps are
recorded with the caveat that HTTP chunk receipt is buffered/coalesced
and is NOT a token-receipt-interval claim. Text/count only; no hidden
token-ID claims. Same digest linkage, /api/ps, ps-RSS (KiB/1024,
post-request point samples, max across rounds — not a peak claim),
loadavg, and no-page-cache-purge rules as tools/baseline/ollama_warmcold.
The user's own 11434 daemon is never touched. Model unloaded + verified
at the end. Bounds: 128 chunks, 1 MiB/line, 8 MiB total per request.
"""
import hashlib, json, os, re, subprocess, sys, time, urllib.request

MAX_CHUNKS, MAX_LINE, MAX_TOTAL = 128, 1 << 20, 8 << 20
PORT = 11498
MODEL = "gemma4-task5"
MEASURED_ROUNDS = 5
NUM_PREDICT = 32
DISTRACTOR = "Xqztv probe"
CACHE_RE = re.compile(
    r'msg="loading cache slot" id=(\d+) cache=(\d+) prompt=(\d+) '
    r'used=(\d+) remaining=(\d+)')

def req_body(prompt, num_predict, keep_alive):
    return {"model": MODEL, "prompt": prompt, "raw": True, "stream": True,
            "options": {"temperature": 0, "top_k": 1, "top_p": 1,
                        "repeat_penalty": 1, "seed": 42,
                        "num_predict": num_predict, "num_ctx": 4096},
            "keep_alive": keep_alive}

def one_request(prompt, num_predict, keep_alive):
    """Returns (record, failure). record keeps client + api server metrics."""
    body = req_body(prompt, num_predict, keep_alive)
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://localhost:{PORT}/api/generate", data=data,
        headers={"Content-Type": "application/json"})
    rec = {"prompt": prompt, "num_predict": num_predict,
           "keep_alive": keep_alive, "chunk_count": 0}
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
                    fail = "chunk is not an object"
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
    if prompt == "Hello":
        if final.get("prompt_eval_count") != 2:
            return rec, (f"prompt_eval_count {final.get('prompt_eval_count')}"
                         " != 2")
    elif not isinstance(final.get("prompt_eval_count"), int) or \
            final.get("prompt_eval_count", 0) < 1:
        return rec, f"bad distractor prompt_eval_count {final.get('prompt_eval_count')!r}"
    eval_count = final.get("eval_count")
    if not isinstance(eval_count, int) or not (1 <= eval_count <= NUM_PREDICT):
        return rec, f"eval_count {eval_count!r} outside 1..{NUM_PREDICT}"
    nonempty = [(t, o) for t, o in chunks if o.get("response")]
    text = "".join(o["response"] for _, o in nonempty)
    rec.update({
        "failure": None, "text": text,
        "nonempty_chunk_count": len(nonempty),
        "done_reason": final.get("done_reason"),
        "client": {
            "ttft_first_nonempty_s": nonempty[0][0] - t0 if nonempty else None,
            "total_to_done_s": chunks[-1][0] - t0,
            "note": ("HTTP chunk receipt is buffered/coalesced; these are "
                     "NOT token-receipt-interval claims"),
        },
        "api_server": {
            "load_duration_s": final.get("load_duration", 0) / 1e9,
            "prompt_eval_duration_s": final.get("prompt_eval_duration", 0) / 1e9,
            "prompt_eval_count": final.get("prompt_eval_count"),
            "eval_duration_s": final.get("eval_duration", 0) / 1e9,
            "eval_count": eval_count,
            "decode_note": ("eval_duration/eval_count is the standard "
                            "server decode metric (per generated token, "
                            "N=eval_count); eval_duration/(eval_count-1) "
                            "is an alternate first-to-last normalization "
                            "reported only for comparison"),
        },
    })
    return rec, None

def model_loaded():
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

def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_average():
    try:
        o = subprocess.run(["sysctl", "-n", "vm.loadavg"],
                           capture_output=True, text=True, timeout=10)
        return o.stdout.strip()
    except Exception as e:
        return repr(e)

def main(out_path, daemon_log):
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
    blob_digest = os.path.basename(blob_path).removeprefix("sha256-")
    gguf = os.environ.get(
        "GGUF", "/Users/dhihm/Documents/Codex/2026-10-02/task-5/models/"
                "gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf")
    t0 = time.monotonic()
    gguf_digest = sha256_file(gguf)
    if blob_digest != gguf_digest:
        sys.exit(f"digest mismatch: ollama blob {blob_digest} != gguf {gguf_digest}")
    if model_loaded():
        sys.exit("model already loaded at start; run a keep_alive:0 request "
                 "first or wait for unload so the control round pays the load")

    log_lines = []
    def refresh_log():
        nonlocal log_lines
        try:
            with open(daemon_log, "r", errors="replace") as f:
                log_lines = f.read().splitlines()
        except OSError as e:
            sys.exit(f"cannot read daemon log {daemon_log}: {e!r}")

    def latest_cache_line(before_count):
        """Newest 'loading cache slot' line index >= before_count."""
        refresh_log()
        for i in range(len(log_lines) - 1, before_count - 1, -1):
            if "loading cache slot" in log_lines[i]:
                m = CACHE_RE.search(log_lines[i])
                return i, (m.groups() if m else None), log_lines[i]
        return None, None, None

    result = {"model": MODEL, "port": PORT, "ollama_version": version,
              "daemon_log": daemon_log,
              "daemon_env": {"OLLAMA_DEBUG": "1", "OLLAMA_NUM_PARALLEL": "1",
                             "OLLAMA_MAX_LOADED_MODELS": "1",
                             "OLLAMA_MODELS": "/tmp/task5-ollama-models",
                             "note": "set by the operator when starting the "
                                     "task-owned 11498 daemon; single slot "
                                     "also confirmed by runner.parallel=1 "
                                     "in the debug log"},
              "gguf": gguf, "gguf_sha256": gguf_digest,
              "ollama_blob_sha256": blob_digest,
              "digest_linkage": "ollama model blob sha256 == authorized GGUF sha256 (byte-linked)",
              "digest_computed_s": round(time.monotonic() - t0, 1),
              "loadavg_start": load_average(),
              "num_predict": NUM_PREDICT,
              "measured_rounds": MEASURED_ROUNDS,
              "distractor_prompt": DISTRACTOR,
              "cache_policy_evidence": ("per-round 'loading cache slot' lines "
                                        "from the runner's cache manager "
                                        "(cache/prompt/used/remaining); proof "
                                        "the measured prompt's non-BOS token "
                                        "was freshly evaluated after each "
                                        "distractor — log-authoritative, not "
                                        "a timing heuristic"),
              "protocol_note": ("measured rounds are WARM (model resident) "
                                "with one serial distractor request before "
                                "each; Leviathan's harness re-prefills all "
                                "prompt tokens after a full reset, so the "
                                "remaining asymmetry is at most the single "
                                "reused BOS token"),
              "rss_note": ("ps RSS point samples (KiB/1024 -> MiB); max "
                           "across rounds; not a peak claim")}
    failures = []
    sampled_max = {}
    def sample_rss(phase):
        for k, v in rss_samples_mib().items():
            if isinstance(v, float):
                cur = sampled_max.get(k)
                sampled_max[k] = v if cur is None else max(cur, v)
        result[f"rss_max_{phase}_mib"] = dict(sorted(sampled_max.items()))

    # CONTROL: A loads the model with the measured prompt; identical B
    # documents the reuse signature (BOS-only for this 2-token prompt;
    # both discarded).
    refresh_log()
    markA = len(log_lines)
    recA, failA = one_request("Hello", 1, -1)
    _, gA, lineA = latest_cache_line(markA)
    recB, failB = one_request("Hello", 1, -1)
    _, gB, lineB = latest_cache_line(markA)
    result["control_A_load"] = {**recA, "cache_slot_line": lineA,
                                "cache_slot_groups": gA}
    result["control_B_reuse"] = {**recB, "cache_slot_line": lineB,
                                 "cache_slot_groups": gB}
    if failA:
        failures.append(f"control A: {failA}")
    if failB:
        failures.append(f"control B: {failB}")
    sample_rss("control")

    result["rounds"] = []
    reference = None
    if not failures:
        for i in range(MEASURED_ROUNDS):
            refresh_log()
            mark = len(log_lines)
            recD, failD = one_request(DISTRACTOR, 1, -1)
            if failD:
                failures.append(f"round {i + 1} distractor: {failD}")
                break
            rec, fail = one_request("Hello", NUM_PREDICT, -1)
            idx, groups, line = latest_cache_line(mark)
            result["rounds"].append({"distractor": recD, "measured": rec,
                                     "measured_cache_slot_line": line,
                                     "measured_cache_slot_groups": groups})
            print(json.dumps({"round": i + 1, "measured": rec,
                              "cache_slot_line": line}))
            sample_rss("measured")
            if fail:
                failures.append(f"round {i + 1}: {fail}")
                break
            if groups is None:
                failures.append(f"round {i + 1}: no 'loading cache slot' "
                                "line found after the distractor")
                break
            slot_id, cache_len, prompt_n, used, remaining = map(int, groups)
            # Exact aligned signature: slot 0 (single parallel slot), the
            # 2-token measured prompt, exactly the BOS token reused, and
            # the non-BOS "Hello" token freshly evaluated. The Leviathan
            # harness aligns by prefilling BOS untimed and timing the
            # one-token continuation, so both engines freshly evaluate
            # exactly one prompt token per round.
            if (slot_id, prompt_n, used, remaining) != (0, 2, 1, 1):
                failures.append(
                    f"round {i + 1}: cache slot line shows slot={slot_id} "
                    f"prompt={prompt_n} used={used} remaining={remaining}; "
                    f"expected 0/2/1/1 (BOS-only reuse, non-BOS token "
                    f"fresh) — KV-prefix policy NOT aligned")
                break
            if reference is None:
                reference = (rec["text"], rec["api_server"]["eval_count"],
                             rec["done_reason"])
            elif (rec["text"], rec["api_server"]["eval_count"],
                  rec["done_reason"]) != reference:
                failures.append(
                    f"round {i + 1}: output differs from round 1 "
                    f"(text/count/reason {rec['text']!r}, "
                    f"{rec['api_server']['eval_count']}, {rec['done_reason']!r})")
                break
    result["reference_output"] = ({"text": reference[0], "eval_count": reference[1],
                                   "done_reason": reference[2]}
                                  if reference else None)
    result["loadavg_end"] = load_average()
    if model_loaded():
        try:
            data = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request(
                f"http://localhost:{PORT}/api/generate", data=data,
                headers={"Content-Type": "application/json"}), timeout=120).read()
        except Exception as e:
            failures.append(f"cleanup unload failed: {e!r}")
        wait_unloaded()
    result["failures"] = failures
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    if failures:
        print(json.dumps(failures, indent=1))
        sys.exit(1)
    print("OLLAMA_FAIRDECODE_PASS")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ollama-fairdecode.json",
         sys.argv[2] if len(sys.argv) > 2 else
         "/tmp/leviathan-forge2-tooling/ollama-11498-debug.log")
