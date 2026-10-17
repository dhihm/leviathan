#!/bin/zsh
# Matched warm/cold resident Leviathan acceptance harness (PRIMARY path).
# Same protocol as tools/baseline/ollama_warmcold.py: raw "Hello", greedy,
# 8 new tokens, ctx 4096 (runner default), exact-output agreement asserted.
#
# The timed path is the SELECTED SSA artifact (batching + tiledQ4 +
# rmsnormTiled): step 0 (re)forges the plan with gemma-forge so its identity
# binds THIS executable SHA / GGUF digest / hardware. Step 1 is a SEPARATE
# gated verification invocation (--gate: one untimed reference replay with
# exact greedy ids, rel < 1e-4 per logit array, finiteness on both arrays).
# The measurement invocations below run WITHOUT the gate so a first cold
# repetition is not pre-warmed by any replay; they still validate the plan
# identity and time startup/identity prep separately from prefill TTFT.
#   COLD: 3 separate CLI processes, no --gate (fresh model load each; the OS
#         page cache is NOT purged, so storage is not guaranteed cold — same
#         caveat as the Ollama harness). /usr/bin/time -l captures the
#         client-side process peak RSS (includes Swift runtime; not a GPU
#         residency claim) and total client wall from process start.
#   WARM: 1 process, --repeat 5 (round 1 pays load only; rounds >= 2 are
#         warm in-process; cross-repetition determinism asserted by the CLI).
# Preflight refuses to start while ANY model is resident in Ollama (/api/ps)
# so the GPU is never shared. Machine load is recorded (sysctl vm.loadavg)
# at start/end for comparability context. PROTOCOL DISCLOSURE: every
# Leviathan repetition resets the KV cache and re-prefills the prompt;
# Ollama's server may reuse the KV prefix across identical warm requests
# (its warm prompt_eval_duration is far below cold) — TTFT/prefill are NOT
# strictly like-for-like across engines; decode per-token medians are.
# Asserts the greedy continuation matches the previously verified ids
# (3324 236772 1437 236772 236751 2964 236772 1340 -> " upon-the-sands-of").
# CLIENT-TTFT DISCLOSURE: this CLI is not a streaming server — stdout is
# emitted when generation completes, so a client sees the first token only
# at process exit. The reported prefill/TTFT and per-step decode timings
# are IN-PROCESS ContinuousClock measurements (printed with the output);
# /usr/bin/time "real" is the whole-process client wall. For streamed
# client-visible tokens see gemma-serve (bounded localhost HTTP path).
set -euo pipefail
repo=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
gguf=${GGUF:-<workspace>/models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf}
cli="$repo/.build/release/leviathan-cli"
tool=/tmp/leviathan-forge2-tooling
graph=$tool/gemma-graph.json
plan=$tool/gemma-plan-acceptance.json
ev="$repo/docs/evidence"
expect_ids="3324 236772 1437 236772 236751 2964 236772 1340"
[[ -x $cli ]] || { echo "missing $cli (swift build -c release)" >&2; exit 1; }
[[ -r $graph ]] || { echo "missing graph $graph (gemma-compile)" >&2; exit 1; }

# Preflight: no model may be resident in Ollama (sequential GPU use).
ps_json=$(curl -s --max-time 10 http://localhost:11499/api/ps)
loaded=$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)["models"]))' <<<"$ps_json")
[[ $loaded -eq 0 ]] || { echo "Ollama has $loaded model(s) resident; aborting" >&2; exit 1; }
{
  echo "start: $(date -u +%FT%TZ) loadavg: $(sysctl -n vm.loadavg)"
} > "$ev/gemma-acceptance-load.txt"
trap 'echo "end: $(date -u +%FT%TZ) loadavg: $(sysctl -n vm.loadavg)" >> "$ev/gemma-acceptance-load.txt"' EXIT

# Step 0: (re)forge the resident selected plan bound to THIS executable.
"$cli" gemma-forge --file "$gguf" --graph "$graph" --plan "$plan" \
  --text "Hello" --gen-tokens 8 > "$ev/gemma-acceptance-forge.log" 2>&1
grep -q "^FORGE_PASS\|plan saved: $plan" "$ev/gemma-acceptance-forge.log" \
  || { echo "forge step failed" >&2; tail -5 "$ev/gemma-acceptance-forge.log" >&2; exit 1; }
grep -E "^(selected|plan saved)" "$ev/gemma-acceptance-forge.log"

gen_args=(--file "$gguf" --graph "$graph" --plan "$plan" --text "Hello" --max-tokens 8)

# Step 1: SEPARATE gated verification (untimed replay inside its process).
"$cli" gemma-generate "${gen_args[@]}" --gate > "$ev/gemma-acceptance-gate.log" 2>&1
grep -q "plan gate passed" "$ev/gemma-acceptance-gate.log" \
  || { echo "gate failed" >&2; tail -5 "$ev/gemma-acceptance-gate.log" >&2; exit 1; }
echo "--- gate ---"; grep -E "^(plan identity|plan gate|generated ids)" "$ev/gemma-acceptance-gate.log"

run_one() {
  local label=$1; shift
  local log="$ev/gemma-acceptance-$label.log"
  if [[ $label == cold* ]]; then
    /usr/bin/time -l "$cli" gemma-generate "${gen_args[@]}" "$@" > "$log" 2>&1
  else
    "$cli" gemma-generate "${gen_args[@]}" "$@" > "$log" 2>&1
  fi
  grep -q "^generated ids: $expect_ids$" "$log" || { echo "ids mismatch in $log" >&2; exit 1; }
  grep -q "GENERATION_COMPLETED" "$log" || { echo "incomplete $log" >&2; exit 1; }
  grep -q "plan gate passed" "$log" && { echo "unexpected gate in measurement $log" >&2; exit 1; }
  echo "--- $label ---"; grep -E "^(load|identity validation|repetition|generated text)" "$log"
  if [[ $label == cold* ]]; then grep -E " real|maximum resident set size" "$log" | tail -2; fi
}

for i in 1 2 3; do run_one "cold$i"; done
run_one warm --repeat 5
echo "LEVIATHAN_WARMCOLD_PASS"
