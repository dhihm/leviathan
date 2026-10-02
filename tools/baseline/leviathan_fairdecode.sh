#!/bin/zsh
# Fair decode-focused Leviathan harness matching tools/baseline/
# ollama_fairdecode.py (32 tokens, KV-prefix policy ALIGNED).
#
# The Ollama side proves from the 11498 debug daemon's cache-manager log
# that each warm measured round reuses ONLY the BOS token and freshly
# evaluates the non-BOS "Hello" token (slot 0, prompt 2, used 1,
# remaining 1). This harness aligns: the timed warm repetitions use the
# CLI's --bos-align mode — untimed [BOS] prefill, then a TIMED one-token
# continuation, then 31 timed decode steps (32 generated tokens, 31
# decode intervals after the first; the Ollama eval_duration/eval_count
# convention covers all 32 generated tokens — both conventions are
# reported, neither is silently reinterpreted).
#
# Steps: (0) re-forge the selected SSA plan bound to THIS executable;
# (1) SEPARATE gated verification (--gate, untimed reference replay,
# exact greedy ids, rel < 1e-4, finiteness); (2) one cold plain-protocol
# process; (3) one warm --repeat 5 --bos-align process (in-process
# cross-repetition determinism asserted by the CLI; round 1 pays load).
# All runs assert identical generated ids and that the decoded text
# equals the Ollama fair-decode reference text. Preflight refuses to
# start while any model is resident in Ollama. Loadavg recorded at
# start/end; the OS page cache is NOT purged. Client-TTFT disclosure:
# this CLI is not streaming; the printed prefill/decode timings are
# in-process ContinuousClock measurements, /usr/bin/time "real" is
# whole-process wall.
set -euo pipefail
repo=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
gguf=${GGUF:-/Users/dhihm/Documents/Codex/2026-10-02/task-5/models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf}
cli="$repo/.build/release/leviathan-cli"
tool=/tmp/leviathan-forge2-tooling
graph=$tool/gemma-graph.json
plan=$tool/gemma-plan-fairdecode.json
ev="$repo/docs/evidence"
tokens=32
# Verified greedy ids for raw "Hello" (BOS 2 + token 9259), 32 tokens:
# reference runner path == SSA path == Ollama text, cross-checked below.
expect_ids="3324 236772 1437 236772 236751 2964 236772 1340 236772 2289 236772 1340 236772 1437 236772 236751 2964 236772 1340 236772 236751 2964 236772 1340 236772 1437 236772 236751 2964 236772 1340 236772"
[[ -x $cli ]] || { echo "missing $cli (swift build -c release)" >&2; exit 1; }
[[ -r $graph ]] || { echo "missing graph $graph (gemma-compile)" >&2; exit 1; }

# Ollama reference text for exact cross-engine agreement (32 tokens).
ollama_json=${OLLAMA_JSON:-$ev/ollama-fairdecode.json}
ollama_text=$(python3 - "$ollama_json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ref = d.get("reference_output") or {}
text = ref.get("text")
count = ref.get("eval_count")
assert text is not None and count == 32, f"bad Ollama reference: {ref!r}"
print(text)
EOF
)
[[ -n $ollama_text ]] || { echo "cannot read Ollama reference text" >&2; exit 1; }

# Preflight: no model may be resident in EITHER task daemon (sequential GPU).
for port in 11498 11499; do
  ps_json=$(curl -s --max-time 10 http://localhost:$port/api/ps) \
    || { echo "daemon on $port not answering" >&2; exit 1; }
  loaded=$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)["models"]))' <<<"$ps_json")
  [[ $loaded -eq 0 ]] || { echo "Ollama:$port has $loaded model(s) resident; aborting" >&2; exit 1; }
done
{
  echo "fairdecode start: $(date -u +%FT%TZ) loadavg: $(sysctl -n vm.loadavg)"
} > "$ev/gemma-fairdecode-load.txt"
trap 'echo "fairdecode end: $(date -u +%FT%TZ) loadavg: $(sysctl -n vm.loadavg)" >> "$ev/gemma-fairdecode-load.txt"' EXIT

# Step 0: (re)forge the plan with the FULL 32-token workload: candidate
# gating covers all 31 continuation steps at the strict rel < 1e-4 gate,
# with the legal-candidate fallback — measured tiled candidates that
# fail the long gate are rejected and the fastest passing candidate is
# selected (full-gate record persisted in the plan).
"$cli" gemma-forge --file "$gguf" --graph "$graph" --plan "$plan" \
  --text "Hello" --gen-tokens $tokens > "$ev/gemma-fairdecode-forge.log" 2>&1
grep -q "FORGE_PASS" "$ev/gemma-fairdecode-forge.log" \
  || { echo "forge step failed" >&2; tail -5 "$ev/gemma-fairdecode-forge.log" >&2; exit 1; }
grep -E "^(selected|plan saved|full-workload|verified generation)" "$ev/gemma-fairdecode-forge.log"

gen_args=(--file "$gguf" --graph "$graph" --plan "$plan" --text "Hello" --max-tokens $tokens)

# Step 1: SEPARATE gated verification at the SAME 32-token length
# (untimed reference replay inside its process; strict 1e-4 + exact ids).
"$cli" gemma-generate "${gen_args[@]}" --gate > "$ev/gemma-fairdecode-gate.log" 2>&1
grep -q "plan gate passed" "$ev/gemma-fairdecode-gate.log" \
  || { echo "gate failed" >&2; tail -5 "$ev/gemma-fairdecode-gate.log" >&2; exit 1; }
echo "--- gate ---"; grep -E "^(plan gate|generated ids|generated text)" "$ev/gemma-fairdecode-gate.log"

check_run() {
  local label=$1
  local log="$ev/gemma-fairdecode-$label.log"
  grep -q "^generated ids: $expect_ids$" "$log" \
    || { echo "ids mismatch vs verified ids in $log" >&2; exit 1; }
  grep -q "GENERATION_COMPLETED" "$log" || { echo "incomplete $log" >&2; exit 1; }
  grep -q "plan gate passed" "$log" && { echo "unexpected gate in measurement $log" >&2; exit 1; }
  local text
  text=$(grep "^generated text: " "$log" | head -1 | sed 's/^generated text: //')
  [[ $text == $ollama_text ]] \
    || { echo "text mismatch vs Ollama: '$text' != '$ollama_text'" >&2; exit 1; }
  echo "--- $label ---"; grep -E "^(load \(|identity validation|repetition|generated text)" "$log"
}

# Step 2: one cold plain-protocol process (fresh load; page cache NOT purged).
/usr/bin/time -l "$cli" gemma-generate "${gen_args[@]}" \
  > "$ev/gemma-fairdecode-cold.log" 2>&1
check_run cold
grep -E " real|maximum resident set size" "$ev/gemma-fairdecode-cold.log" | tail -2

# Step 3: warm aligned repetitions (untimed BOS prefill per repetition;
# timed one-token continuation + 31 decode steps; determinism asserted).
"$cli" gemma-generate "${gen_args[@]}" --repeat 5 --bos-align \
  > "$ev/gemma-fairdecode-warm.log" 2>&1
check_run warm
echo "LEVIATHAN_FAIRDECODE_PASS"
