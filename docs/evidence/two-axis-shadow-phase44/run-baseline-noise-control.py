import datetime, hashlib, json, re, statistics, subprocess, time
from pathlib import Path
b = Path(__file__).parent
protocol = json.loads((b / "next-baseline-noise-protocol.json").read_text())
quiet = b / "quiet-window-ready.json"
if not quiet.exists():
    raise SystemExit("Fresh parent quiet-window coordination is required; no measurement started")
coordination = json.loads(quiet.read_text())
if coordination.get("scope") != "phase43-baseline-aa-seven-visits" or not coordination.get("ready"):
    raise SystemExit("Quiet-window receipt is not for this measurement")
lock = b.parent / "moe-root-gpu-lock.json"
if json.loads(lock.read_text()).get("gpuLock") == "running":
    raise SystemExit("An owned GPU measurement is already active")
exe = Path(protocol["baselineExecutable"])
assert hashlib.sha256(exe.read_bytes()).hexdigest() == protocol["baselineSha256"]
plan = Path(protocol["plan"])
composite = Path(protocol["composite"])
files = [exe, plan, composite]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
root = Path("<workspace>")
prior = json.loads((plan.parent / "serving-three-results.json").read_text())
expected = next(x["generatedIDs"] for x in prior["fixtures"] if x["fixture"] == "raw-hello")
out = b / "baseline-aa-measurement"
out.mkdir(exist_ok=False)
args = [str(exe), "gemma-generate", "--file", str(root / "models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf"), "--graph", str(b.parent / "gemma-graph.json"), "--plan", str(plan), "--composite-plan", str(composite), "--moe-region", str(b.parent / "moe-region-root.json"), "--text", "Hello", "--max-tokens", "32", "--repeat", "4", "--bos-align"]
def observe(label):
    raw = {}
    for name, command in [("thermal", ["pmset", "-g", "therm"]), ("power", ["pmset", "-g", "batt"]), ("memory", ["vm_stat"]), ("aggregateLoad", ["uptime"])]:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        raw[name] = {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    counters = {}
    for key in ["Pageouts", "Swapins", "Swapouts"]:
        match = re.search(r"^" + key + r":\s*(\d+)", raw["memory"]["stdout"], re.M)
        counters[key] = int(match.group(1)) if match else None
    sample = {"label": label, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "raw": raw, "memoryCounters": counters}
    receipt["observations"].append(sample)
    return sample

receipt = {"scope": "Frozen validated baseline A/A control only, no candidate variant", "protocol": protocol, "coordination": coordination, "startedUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": args, "visits": [], "observations": []}
lock.write_text(json.dumps({"gpuLock": "running", "mode": "phase43-baseline-aa-control"}))
try:
    for visit in range(7):
        observe(f"visit-{visit+1}-before")
        t = time.monotonic()
        log = out / f"visit-{visit+1}.log"
        with log.open("x") as stream:
            result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        observe(f"visit-{visit+1}-after")
        text = log.read_text()
        ids = re.findall(r"^generated ids: (.+)$", text, re.M)
        totals = [float(x) for x in re.findall(r"repetition \d+/4:.*?decode: ([0-9.]+) ms total \(31 steps\)", text)]
        fallback = [int(x) for x in re.findall(r"fallback (\d+)", text)]
        row = {"visit": visit+1, "exit": result.returncode, "elapsedSeconds": time.monotonic()-t, "decodeTotalsMs": totals, "generatedIDs": ids, "fallbackValues": fallback, "completedUtc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        receipt["visits"].append(row)
        (out / "progress.json").write_text(json.dumps(receipt, indent=2))
        assert result.returncode == 0 and "GENERATION_COMPLETED" in text and len(totals) == 4
        assert len(ids) == 4 and all(x == expected for x in ids)
        assert len(fallback) >= 9 and all(x == 0 for x in fallback)
        row["warmMedianMsPerToken"] = statistics.median(totals[1:]) / 31
        print(json.dumps(row), flush=True)
    medians = [x["warmMedianMsPerToken"] for x in receipt["visits"]]
    median = statistics.median(medians)
    receipt.update(overallMedianMsPerToken=median, relativeSpread=(max(medians)-min(medians))/median, allVisitsRetained=True, filesUnchanged=before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, completedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    firstMemory = receipt["observations"][0]["memoryCounters"]
    lastMemory = receipt["observations"][-1]["memoryCounters"]
    receipt["memoryActivityDeltas"] = {key: lastMemory[key]-firstMemory[key] if firstMemory[key] is not None and lastMemory[key] is not None else None for key in firstMemory}
    receipt["sub5PercentJudgmentPermittedByNoiseCriterion"] = receipt["relativeSpread"] <= 0.05
    assert receipt["filesUnchanged"]
    (out / "results.json").write_text(json.dumps(receipt, indent=2))
except BaseException as error:
    receipt["failure"] = repr(error)
    (out / "failed.json").write_text(json.dumps(receipt, indent=2))
    raise
finally:
    lock.write_text(json.dumps({"gpuLock": "finished", "mode": "phase43-baseline-aa-control"}))
