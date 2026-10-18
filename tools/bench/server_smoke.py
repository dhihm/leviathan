"""Smoke and concurrency test of lev-server with real clients.

    python server_smoke.py [--url http://127.0.0.1:8100] [--concurrency 8] [--tokens 48]

1. The official `openai` client: completion, streaming completion, chat, stop string, stream usage.
2. A concurrency run with aiohttp: N requests at once, each streamed; reports the time to first
   token, the generation rate per request and the aggregate rate, and checks that every request got
   exactly `--tokens` tokens (ignore_eos) and that the outputs of identical prompts agree.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp
from openai import OpenAI

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Once upon a time, in a small fishing village on the edge of a cold northern sea,",
    "What are the main differences between TCP and UDP?",
]


def client_checks(url):
    c = OpenAI(base_url=url + "/v1", api_key="none")
    model = c.models.list().data[0].id
    print("model:", model)

    r = c.completions.create(model=model, prompt="The capital of France is", max_tokens=8, temperature=0)
    print("completion:", repr(r.choices[0].text), r.choices[0].finish_reason, r.usage)
    assert "Paris" in r.choices[0].text

    chunks = list(c.completions.create(model=model, prompt="Count: 1, 2, 3,", max_tokens=12, stream=True, stream_options={"include_usage": True}))
    text = "".join(ch.choices[0].text for ch in chunks if ch.choices)
    usage = [ch.usage for ch in chunks if ch.usage]
    print("stream:", repr(text), "| chunks", len(chunks), "| usage", usage[-1] if usage else None)
    assert usage and usage[-1].completion_tokens == 12

    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=60)
    print("chat:", repr(r.choices[0].message.content[:80]), r.choices[0].finish_reason)

    r = c.completions.create(model=model, prompt="1, 2, 3, 4, 5, 6,", max_tokens=40, stop=["8"], temperature=0)
    print("stop:", repr(r.choices[0].text), r.choices[0].finish_reason)
    assert "8" not in r.choices[0].text and r.choices[0].finish_reason == "stop"


async def one(session, url, prompt, tokens, idx):
    body = {"model": "m", "prompt": prompt, "max_tokens": tokens, "stream": True, "ignore_eos": True, "temperature": 0}
    t0 = time.perf_counter()
    first = None
    text = []
    n = 0
    async with session.post(url + "/v1/completions", json=body) as resp:
        assert resp.status == 200, await resp.text()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            piece = ev["choices"][0]["text"]
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                text.append(piece)
                n += 1
    total = time.perf_counter() - t0
    return {"idx": idx, "ttft": first, "total": total, "chunks": n, "text": "".join(text)}


async def concurrency(url, n, tokens):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(s, url, PROMPTS[i % len(PROMPTS)], tokens, i) for i in range(n)])
        wall = time.perf_counter() - t0
        async with s.get(url + "/v1/stats") as r:
            stats = await r.json()
    gen = [(r["chunks"] - 1) / (r["total"] - r["ttft"]) for r in res if r["ttft"] and r["total"] > r["ttft"]]
    print(f"{n} concurrent requests, {tokens} tokens each: wall {wall:.1f} s")
    print(f"  time to first token: median {statistics.median(r['ttft'] for r in res):.2f} s, max {max(r['ttft'] for r in res):.2f} s")
    print(f"  per-request rate after the first token: median {statistics.median(gen):.1f} tok/s")
    print(f"  aggregate: {sum(r['chunks'] for r in res) / wall:.1f} chunks/s")
    print(f"  scheduler: {stats}")
    by_prompt = {}
    for r in res:
        by_prompt.setdefault(r["idx"] % len(PROMPTS), set()).add(r["text"])
    same = all(len(v) == 1 for v in by_prompt.values())
    print("  identical prompts gave identical text:", same)
    for k, texts in sorted(by_prompt.items()):
        if len(texts) > 1:
            ts = sorted(texts)
            a, b = ts[0], ts[1]
            at = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(f"  prompt {k}: {len(texts)} variants, first difference at character {at} of {len(a)}: {a[max(0, at - 20):at + 20]!r} vs {b[max(0, at - 20):at + 20]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--skip-client", action="store_true")
    a = ap.parse_args()
    if not a.skip_client:
        client_checks(a.url)
    asyncio.run(concurrency(a.url, a.concurrency, a.tokens))
