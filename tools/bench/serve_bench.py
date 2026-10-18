"""Serving benchmark with the metrics of `vllm bench serve`, for any OpenAI-compatible server.

    python serve_bench.py --url http://127.0.0.1:8100 --num-prompts 32 --input-len 256 --output-len 128 \
        --concurrency 8 [--request-rate 4] [--model NAME]

Requests carry random token ids as `prompt` (a list of ids) when --ids is given, else a text made
of common words; `ignore_eos` fixes the output length. Streams every response and reports
time to first token (TTFT), time per output token after the first (TPOT), inter-token latency
(ITL), request and token throughput, like vllm's benchmark. (vllm itself does not install on
Windows: its build needs CUDA/Linux toolchains.)
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

WORDS = "the of and to in is that it for with as on be at by this from or an are was but not have had you all can her one our out day get use man new now way may say".split()


def make_prompt(rng, n_tokens, ids, vocab):
    if ids:
        return [rng.randrange(1000, vocab) for _ in range(n_tokens)]
    # About one token per word for common English words.
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


async def one(session, url, model, prompt, out_len, sem, start_barrier):
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "stream": True, "ignore_eos": True, "temperature": 0}
    async with sem:
        t0 = time.perf_counter()
        stamps = []
        try:
            async with session.post(url + "/v1/completions", json=body) as resp:
                if resp.status != 200:
                    return {"ok": False, "err": f"{resp.status} {(await resp.text())[:100]}"}
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[5:])
                    if ev.get("choices") and ev["choices"][0].get("text"):
                        stamps.append(time.perf_counter())
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "err": repr(e)}
    if not stamps:
        return {"ok": False, "err": "no tokens"}
    return {"ok": True, "t0": t0, "stamps": stamps, "end": time.perf_counter()}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def line(name, xs):
    xs = [x * 1000 for x in xs]
    return f"  {name:<22} mean {statistics.mean(xs):8.1f}   median {statistics.median(xs):8.1f}   p99 {pct(xs, 99):8.1f}  ms"


async def main(a):
    rng = random.Random(a.seed)
    prompts = [make_prompt(rng, a.input_len, a.ids, a.vocab) for _ in range(a.num_prompts)]
    sem = asyncio.Semaphore(a.concurrency if a.concurrency else a.num_prompts)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600)) as s:
        # Warm-up: one short request so shader/pipeline setup is not measured.
        await one(s, a.url, a.model, prompts[0], 4, asyncio.Semaphore(1), None)
        t_start = time.perf_counter()
        tasks = []
        for i, p in enumerate(prompts):
            tasks.append(asyncio.create_task(one(s, a.url, a.model, p, a.output_len, sem, None)))
            if a.request_rate:
                await asyncio.sleep(rng.expovariate(a.request_rate))
        res = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_start
    ok = [r for r in res if r["ok"]]
    failed = len(res) - len(ok)
    if not ok:
        print("all requests failed:", res[0]["err"])
        return
    ttft = [r["stamps"][0] - r["t0"] for r in ok]
    tpot = [(r["stamps"][-1] - r["stamps"][0]) / (len(r["stamps"]) - 1) for r in ok if len(r["stamps"]) > 1]
    itl = [b - a_ for r in ok for a_, b in zip(r["stamps"], r["stamps"][1:])]
    out_tokens = sum(len(r["stamps"]) for r in ok)
    print("============ Serving Benchmark Result ============")
    print(f"Successful requests:        {len(ok)}   (failed {failed})")
    print(f"Concurrency limit:          {a.concurrency or 'none'}   request rate: {a.request_rate or 'inf'}")
    print(f"Input / output tokens:      {a.input_len} / {a.output_len}")
    print(f"Benchmark duration (s):     {wall:.2f}")
    print(f"Request throughput (req/s): {len(ok) / wall:.2f}")
    print(f"Output token throughput:    {out_tokens / wall:.1f} tok/s")
    print(f"Total token throughput:     {(out_tokens + len(ok) * a.input_len) / wall:.1f} tok/s")
    print(line("TTFT", ttft))
    if tpot:
        print(line("TPOT", tpot))
    if itl:
        print(line("ITL", itl))
    for r in res:
        if not r["ok"]:
            print("  error:", r["err"])
            break
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"duration": wall, "ok": len(ok), "failed": failed, "output_tokens": out_tokens,
                       "ttft_ms_median": statistics.median(ttft) * 1000, "tpot_ms_median": statistics.median(tpot) * 1000 if tpot else None,
                       "itl_ms_median": statistics.median(itl) * 1000 if itl else None,
                       "output_tok_s": out_tokens / wall}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="m")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--request-rate", type=float, default=0.0)
    ap.add_argument("--ids", action="store_true", help="send random token ids instead of text")
    ap.add_argument("--vocab", type=int, default=240000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    asyncio.run(main(ap.parse_args()))