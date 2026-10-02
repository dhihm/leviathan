# LeviathanForge working progress (GLM-5.3/ZCode, original implementation)

Status date: 2026-10-02. Concise evidence log for the coordinator. All
work in this repo is authored by GLM; licensed public references are
used for semantics only, with attribution, no source copying.

## Bounded tokenizer equivalence (13 diverse cases; input IDs)

Environment (installed, no downloads): Ollama 0.23.0
(/opt/homebrew/bin/ollama); Anaconda Python 3.11 with transformers
4.57.3 and Rust-backed tokenizers 0.22.1.

Baseline endpoint findings (evidence, task-owned daemon port 11499):
- Public `POST /api/tokenize` does NOT exist in Ollama 0.23.0 → HTTP
  404 (observed live; also absent from the pinned v0.23.0 routes.go,
  which registers /api/generate, /api/chat, /api/embed, /api/embeddings,
  /api/show, /api/ps, etc.).
- From the pinned official v0.23.0 `llm/server.go` (lines 1810–1833):
  `Tokenize` is an IN-PROCESS Go method (`s.llamaModel.Tokenize(...)`
  or `s.tokenizer.Encode`); the runner HTTP surface only exposes
  /load, /health, /completion, /embedding. No local tokenize endpoint
  to query. User's own Ollama process untouched; task daemon runs only
  on port 11499 with OLLAMA_MODELS=/tmp/task5-ollama-models.

GGUF tokenizer contract (exported via `leviathan-cli gguf-tokenizer-export`,
reusing the validated bounded Swift GGUFReader — no duplicate parser):
- `tokenizer.ggml.model = "gemma4"`; 262,144 tokens; 514,906 merges;
  BOS=2 (forced add_bos=true), EOS=1, add_space_prefix=false; 256
  `<0xXX>` byte tokens (ids 238–493); 17 CONTROL + 7 USER_DEFINED
  special pieces (`<|turn>`=105, `<turn|>`=106, `<|channel>`=100,
  `<channel|>`=101, `<|think|>`=98, …); full Jinja chat template
  embedded in `tokenizer.chat_template` (simple user→model path:
  `<bos><|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`).
- Semantics pinned from llama.cpp MIT `src/llama-vocab.cpp` at commit
  `a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd` (fetched 2026-10-02,
  master is mutable; reference semantics only, no source copied): SPM-style BPE — spaces escaped to U+2581 before BPE, the only
  pre-split is newline runs (`[^\n]+|[\n]+`), ranked merges with a
  lowest-(rank, left-index) priority queue, all-newline runs that
  exactly match a vocab piece are emitted directly (PR #21343), unknown
  final symbols fall back to `<0xXX>` uppercase-hex bytes, duplicate
  vocab pieces resolve LAST-WINS (3 exist: `//`, `<?`, `#`), 4 merge
  entries with no separator at byte offset ≥ 1 (` //`, ` <?`, ` #`,
  ` \ufeff`) become unmatchable empty pairs, and the U+FEFF merge chain
  produces an intermediate-only symbol outside the final vocab.

Swift implementation: `Sources/LeviathanCore/GemmaTokenizer.swift`
(original). Byte-exact ([UInt8]) vocab/merge keys — Swift String
equality is Unicode-canonicalizing and would wrongly merge NFC/NFD
pieces. Init validates: model==gemma4, array element counts vs parsed
lengths, token_type codes in range, complete 256 byte-fallback set,
bos/eos/unk in range, add_bos true. Missing byte-fallback token or
unmapped symbol is a hard error, never a silent drop; malformed decode
IDs rejected; input size bounded (1 MiB).

Independent oracle: `tokenizers` 0.22.1 Rust BPE instantiated from the
exported vocab+merges with byte_fallback, Replace(" ","▁") normalizer,
Split(Regex(r"\n+"), isolated) pre-tokenizer, specials as added tokens,
BOS appended manually. Adapter for reference-only semantics: merge
halves/outputs that are not final vocab pieces (exactly 1: the U+FEFF
chain) get virtual ids ≥ vocab size; virtual ids are expanded through
the original byte fallback after encoding and provably never reach the
model (asserted on every encode; the assertion proves no leak in observed
runs, not a general equivalence proof). Two-pass construction preserves
all merge rules and rank order.

Verification: `/tmp/leviathan-forge2-tooling/crosscheck-authored.sh`
compares full ID lists on 13 diverse cases — ASCII, Korean (NFC),
digits/punctuation, emoji+skin-tone, café/naïve, NFD é vs NFC,
newlines/CRLF/blank lines, tabs, leading/trailing spaces, U+FEFF BOM,
special-token overlap text, and the full canonical chat prompt.
**Result: 13/13 exact match, exit 0.** This is bounded adapter
equivalence against an independent Rust implementation — it is not a
proof of Ollama's in-process tokenizer, which cannot be queried.

`Hello` → [2, 9259]: the previously inferred prompt mapping is now
derived from an exact tokenizer implementation, cross-verified.

## Next (in order)

1. Matched greedy generation ≥8 tokens (or early EOS) for English and
   Korean canonical single-user chat prompts, exact matched input IDs,
   sequential Ollama-then-Leviathan (GPU residency constraint).
2. Bounded generation loop + text decoding in the runner on verified
   IDs only.
3. Tokenizer validation tests folded into `leviathan-tests`.
4. Gemma through the ACTUAL Forge path: restricted model.py subset →
   validated quantized graph capture/IR → lowering/pass candidates
   (dispatch width, batching, fused/quant matvec variants) →
   correctness gates → measured GPU selection → cached artifact keyed
   on model/weights/hardware/toolchain/workload. The hand-written
   Gemma4Runner stays a resident reference, NOT the Forge path.
5. Profiling (measured, not assumed): CPU logits readback/sort (the
   262,144-logit top-5 diagnostic) vs GPU time; 1,799 per-kernel
   command buffers per token. Optimizations one at a time through
   legal Forge candidates with before/after parity and matched decode
   metrics.
6. Early matched baseline timings (TTFT, prefill, decode tok/s, peak
   memory, cold/warm, load excluded where stated) vs same-GGUF Ollama;
   non-equivalent quantization disclosed rather than hidden.

## Generation + matched baseline (2026-10-02, later session segment)

- `gemma-generate` CLI (token IDs in/out, exact in-repo tokenizer, chat
  canonical single-user path, incremental KV continuation, separate
  load / prefill-TTFT / per-token-decode metrics, bounded context).
- EN "Hello" (chat, 8 greedy tokens) and KO "안녕하세요, 오늘 날씨가
  어떤가요?" (10 tokens) both produce coherent text. Exact transcripts:
  /tmp/leviathan-forge2-tooling/leviathan-generations-2026-10-02.txt
  (IDs, text, timings).
- Matched Ollama raw baseline (same GGUF, raw canonical template text,
  temp 0/top_k 1/top_p 1/repeat_penalty 1/seed 42, num_ctx 4096,
  EN num_predict 8 / KO 10): generated TEXT identical for all tokens on
  both prompts; prompt token COUNTS match the exact tokenizer (14, 23).
  Honest scope: text+count agreement only — Ollama exposes no generated
  IDs. Baseline timings: EN cold load 6962.8 ms, prefill 657.2 ms,
  13.5 ms/token; KO warm. Leviathan bring-up: ~19–21 s TTFT,
  ~807–842 ms/token (per-kernel command buffers, unoptimized) — NO
  performance claim. Evidence JSONs saved in the tooling dir.
- Task Ollama daemon remains on 127.0.0.1:11499; task model unloaded
  (keep_alive 0, /api/ps empty).

## Gemma→Forge path (IN PROGRESS, not complete — no completion claim)

Built so far (compiles; default suite 181/0; NOT yet wired to CLI/tests,
not yet executed on the real model):
- tools/forge_gemma/gemma_dsl.py + examples/gemma_model.py +
  export_gemma_graph.py: restricted, honest Gemma DSL (declaration DSL,
  NOT arbitrary Python/torch tracing) with a SIGS dataflow table.
- Sources/LeviathanCore/GemmaGraph.swift: captured-graph IR validated
  against the real GGUF (config cross-check, exact op-sequence + tensor
  bindings; unknown ops/mutations rejected).
- Sources/LeviathanCore/GemmaForge.swift: graph executor driving the
  runner's kernel PRIMITIVES from the IR (not runner.forward), bounded
  candidates {segment batching × top-5 diagnostic}, gates = full
  262,144-logit rel<1e-4 vs the independent runner-forward reference +
  argmax equality, measured per-token GPU selection incl. baseline.
- Gemma4Runner: internal kernel primitives + checked CPU-readback
  flush (read/write now throw), lazy cross-op auto-segment command
  buffers (kernel order unchanged; kernel count tracked separately
  from buffer count).

Known open items before ANY completion claim (review-guided): SSA
versioned values with explicit shapes/dtypes (SIGS is alias metadata,
not typed SSA); executor currently derives buffers implicitly per op
stage rather than from IR operands; plan artifact is a placeholder
(empty gguf/plan digests) — no executable cache claim; timing workload
must be real prompt-position decode (not position-0 single tokens);
top5-diagnostic candidate must be selected on wall time (CPU cost), not
GPU time; CLI/tests for capture→validate→reject-mutation→execute still
to be written; real-model run of the graph path pending.

## Gemma→Forge gates consolidated (still NOT executed on real model)

GemmaForge.forge now: bounds workload first (prompt + warmup + timed ≤
4096, decodeTokens == warmup+timed); gates the graph executor (per-kernel
reference candidate) against the ORIGINAL runner.forward over ALL
262,144 raw logits (rel < 1e-4) + argmax + finiteness/count; rebuilds an
independent KV state via a fresh runner.forward prefill, captures the
reference greedy decode sequence AND per-step full logits with the
original runner; every candidate replays the identical reference token
sequence at true decode positions and is gated per warmup and timed step
on exact next-token equality + full-logit rel < 1e-4 before its GPU
times count; measurement = median per-token GPU micros over timed steps
under a finite budget; the baseline per-kernel candidate participates in
selection (no forced "optimized" winner); top5 diagnostic removed from
the candidate set (reporting, not a compiler pass). Segment batching is
cross-op (lazy auto-segments, CPU readback flushes; kernel order and
count unchanged). Remaining before any claim: typed SSA IR with
input-driven lowering, real identity (GGUF digest, plan digest) artifact
+ cache validation, CLI wiring, mutation-rejection and tokenizer
malformed-metadata tests, and an actual real-model run with saved raw
output.

## Gemma→Forge FIRST REAL-MODEL EXECUTION (2026-10-02, measured, saved raw logs)

Command (M1 Max, 32 GB, task daemon idle, model unloaded from Ollama;
sequential GPU residency):
`.build/release/leviathan-cli gemma-forge --file <authorized GGUF>
 --graph /tmp/leviathan-forge2-tooling/gemma-graph.json
 --plan /tmp/leviathan-forge2-tooling/gemma-plan.json
 --prompt-ids 2,9259 --gen-tokens 8`
The graph is captured by `tools/forge_gemma/export_gemma_graph.py` from
`examples/gemma_model.py` (restricted stage-list DSL — an HONEST bounded
subset, not arbitrary Python/torch tracing) and validated op-for-op
against the GGUF before any GPU work. Raw logs:
`/tmp/leviathan-forge2-tooling/gemma-forge-run1.log` (fresh) and
`gemma-forge-run2-cachehit.log` (plan reuse). Both exit 0, both end
`FORGE_PASS`.

Measured results (934-op graph, prompt [2, 9259], 8 greedy tokens):

- Correctness: graph executor vs the independently validated
  runner.forward — worst relative difference over ALL 262,144 logits
  **0.0e+00** (bit-identical) at prefill and at every decode step; exact
  greedy next-token equality at every step. Greedy sequence
  `3324 236772 1437 236772 236751 2964 236772 1340` = " upon-the-sands-of"
  (first two pieces match the earlier Ollama raw-"Hello" baseline).
- Candidates (both correct, identical outputs):
  - per-kernel baseline: median wall 858,944 µs/token, GPU 524,029 µs,
    1799 kernels, **1799 command buffers**/token.
  - cross-op segment batching: median wall 434,357 µs/token, GPU
    399,532 µs, 1799 kernels, **127 command buffers**/token (~1.98× wall).
- Selection objective is per-token WALL median (GPU-only time misses CPU
  launch/driver overhead — the dominant cost of the per-kernel path);
  GPU median is reported separately. Baseline fallback with a CHOSEN 2%
  noise threshold (policy, not an empirically calibrated noise floor).
- Verified final generation through the selected candidate: 7 gated
  steps, median wall 447,349 µs/token.
- Plan cache reuse (run 2): streaming GGUF SHA-256 (one pass), plan
  validated on all identity fields + planDigest self-consistency, the
  cached candidate RE-RUNS the full-logit correctness gate before any
  output is trusted, then FORGE_PASS again. Plan digest `2cfe077e…`.

Scope honesty: ~2.3 decode tokens/s at this milestone vs the same-GGUF
Ollama's reported ~72 — a large gap remains; no performance superiority
claim. The top-5 logit diagnostic sort still exists in the reference
`forward` (not in the graph executor path); quantized matvec GPU time
(~400 ms/token) is the dominant bottleneck and the next optimization
target (SIMD/tiled quant kernels as a future legal Forge candidate with
independent numerical gates).

## Typed SSA IR + operand-driven executor + executable artifact (same date)

`Sources/LeviathanCore/GemmaSSA.swift`: deterministic lowering of the
validated stage graph into a typed SSA schedule — every write a fresh
`(name, version)` value with dtype/length/kind (vector | kvstate |
routing), explicit slice offsets recorded in the artifact (gate=0,
up=2112 in the fused gate|up buffer), per-op operand signature
validation (arity, lengths, checked products, layer bounds, tensor
arity, live-version discipline: every read references the currently-
live version of its name — the property that makes post-validation
fixed-buffer aliasing safe), and a softcap→logits terminal requirement.
The lowering derives layer globalness from the CONFIG (cross-checking
stage flags), residual axpy carries BOTH operands (x and h), expert
scratch buffers are internal effects of the compound op, and gate/up are
distinct logical values.

The GemmaForge executor dispatches runner kernel primitives FROM the SSA
operands (buffer resolved per operand kind; rmsnorm n-per-vector; matvec
physical capacity includes slice offsets; cache_append/attention/router/
expert layer geometry guarded with checked arithmetic; final logits read
from the final artifact output value). The plan artifact embeds the FULL
lowered schedule + scheduleDigest; `validatePlan` requires the embedded
schedule to equal the current build's deterministic lowering (a Swift
lowering change invalidates caches automatically), plus the running
executable's own SHA-256 in the engine identity (automatic build
identity; fails closed if the binary cannot be hashed) alongside the
embedded-MSL source hash (options=default) and candidate-set signature.
Cache-hit and final generation execute the ARTIFACT schedule operands.

Adversarial tests (default suite, 220 passed / 0 failed, direct exit 0):
crafted tokenizer metadata rejections (malformed byte syntax, missing
byte token, add_bos/add_space_prefix violations, unknown token_type,
out-of-range BOS; byte-fallback and merge-rank behavior verified), graph
mutation rejections (typed case matching: dropped/reordered ops, unknown
op, wrong layer, config mismatch, schema/model tag, tensorless file),
and SSA mutation rejections via JSON-level adversarial edits (undefined
operand, stale/non-live version read, shape mutation, absurd dimension,
bad kind, undecodable dtype, layer out of range, missing cache attrs,
gate/up slice-interference, unknown op, non-logits terminal, duplicate
value id, digest sensitivity).

## GPU-time profile + expert-scratch alias fix (same date)

`gemma-profile` (per-kernel reference forward, prompt [2, 9259], 2 tokens,
857.3 ms GPU total): matvec 451.6 ms (53%), non-matvec 405.6 ms (47%)
observed in this per-kernel submission/profile mode — which can distort
attribution; the CAUSE of the non-matvec share is not yet measured and
remains to be profiled by kernel family before any fusion pass. Matvec
breakdown by tensor family: expert gate_up 173.5 ms/480 calls (38.4% of
matvec), expert down 47.7 ms (10.6%), LM head (token_embd) 39.3 ms,
attn_output 37.3 ms, router (ffn_gate_inp, only 128 rows) 28.4 ms/60
calls. Ranking (not absolute per-call costs) guides the next pass. Raw
log: docs/evidence/gemma-profile-run2.log.

Alias-correctness fix: the expert compound op now uses DEDICATED scratch
buffers (bufExpertGU/Act/Acc), physically separate from every SSA-bound
activation buffer, so its internal effects can never clobber live SSA
values (previously it reused bufGU/bufAct/bufT2 — a hole in the
live-version safety argument). Public schedule execution additionally
requires EXACT equality with the canonical lowering of the validated
graph (tensor bindings/geometry against the mapped GGUF), not just
standalone arity validation. Re-run gate with the fix: rel 0.0 over all
logits, identical 8-token greedy sequence, FORGE_PASS exit 0
(docs/evidence/gemma-forge-ssa-run3-scratch.log, plan digest 058c260c…).
Next optimization target (measured, one at a time): tiled/SIMD
cooperative Q4_0 matvec kernel for the expert gate_up/down bottleneck,
gated against the independent FP64 oracle (mixed-sign/cancellation
fixtures, multiple row widths) and the full SSA 8-token parity gate —
SIMD reduction reassociation will not be bit-identical, so tolerances
must be justified and greedy equality enforced.

## Measured optimization pass 1: tiled cooperative Q4_0 matvec (same date)

New `g4_q4_matvec_tiled` kernel (one threadgroup per row, strided block
partials + tree reduction; width is the largest legal power of two ≤
min(128, device max, 2× execution width) — the shared PRODUCTION
validator `legalCooperativeWidth` rejects 0/33/256-style widths before
any dispatch; no silent clamping). The reduction reassociates the
addition order, so this is NOT bit-identical by construction:
independence comes from (a) synthetic FP64-oracle gates compiling the
ACTUAL runtime `Gemma4Runner.shaderSource` (mixed-sign FP16 scales,
cancellation-prone inputs, inDim 32/704/2112/2816, widths 32/64/128 —
all worst rel < 1e-5 with the mixed absolute/relative bound
|a−ref|/max(1,|ref|)) and (b) the full Forge gate: all 262,144 logits
rel 1.6e-05 (< 1e-4) vs the reference executor AND exact greedy
next-token equality at every decode step.

Measured selection, all within the SAME run (real GGUF, prompt
[2,9259], 8 tokens, wall-median objective, all four candidates gated):
- per-kernel baseline: 921,952 µs/token wall, 577,401 µs GPU
- segment batching: 440,816 µs wall, 399,539 µs GPU, 127 buffers/token
- tiled Q4 (per-kernel): 589,942 µs wall, 276,586 µs GPU
- batching + tiled Q4: **290,382 µs wall, 249,313 µs GPU** — selected:
  1.52× vs the batched baseline and 3.17× vs the per-kernel baseline
  IN THIS RUN (~3.4 decode tokens/s). Verified final generation through
  the selected candidate: identical greedy sequence
  `3324 236772 1437 236772 236751 2964 236772 1340`, FORGE_PASS exit 0
  (docs/evidence/gemma-forge-tiled-run1.log, plan digest cd2b9607…).
  Default suite: 240 passed / 0 failed, exit 0. Still far below the
  same-GGUF Ollama (~72 tok/s reported); remaining measured targets:
  Q6_K tied-embedding/LM-head tiling, measured non-matvec kernel-family
  profiling, then SSD expert streaming.

## Matched raw-"Hello"-8 Ollama streaming baseline (same date, sequential residency)

`tools/baseline/ollama_stream_baseline.py` (stdlib-only, fail-closed:
bounds 64 chunks / 1 MiB per line / 4 MiB total; requires final
done:true, prompt_eval_count 2, eval_count 8, exact expected text;
saves request + every chunk with monotonic client receipt timestamps —
raw evidence docs/evidence/ollama-rawhello8-stream.json; the saved run
predates a diagnostic-hardening patch, run content identical):

- SAME text as Leviathan's greedy sequence: " upon-the-sands-of"
  (8 eval tokens, prompt count 2 — raw mode, temp 0/top_k 1/top_p 1/
  repeat_penalty 1/seed 42/num_ctx 4096, keep_alive 0, task daemon
  :11499, model unloaded after; /api/ps empty).
- API SERVER metrics (labeled as server-reported): cold load 7.537 s,
  prompt eval 162.6 ms (2 tokens), eval 112.53 ms for 8 tokens (~71
  reported eval tokens/s), total 7.843 s.
- CLIENT STREAM metrics, honestly limited: first-nonempty-chunk TTFT
  7.921 s (includes cold load); the 8 nonempty chunks arrive within a
  3.20 ms span — BUFFERED/COALESCED receipt, NOT actual per-token
  intervals; chunk-count equality with eval_count is necessary but NOT
  sufficient, so no client-side token-interval TPS is claimed.
- Leviathan at the same checkpoint: ~3.4 decode tokens/s wall (gated
  measured run above) — the gap to the baseline remains large and is
  the active work item.

## Measured optimization pass 2 (Q6_K tiling NOT selected) + chat workloads (same date)

- `g4_q6_matvec_tiled` (tied-embedding/LM-head) was implemented with
  its own FP64 gates (runtime MSL vs `Q6_K.matvecReferenceF64` at
  inDim 256/512/2816, widths 32/64/128, mixed-sign scales, saturated
  nibbles, cancellation inputs, plus the exact hand-computed -315
  known block; per-pipeline tile width derived from the Q6 pipeline's
  OWN limits — suite 250 passed / 0 failed, exit 0).
- Full-model gated selection with the Q6 candidates (six candidates,
  real GGUF, docs/evidence/gemma-forge-q6tiled-run1.log): batching+
  tiledQ4 285,045 µs vs batching+tiledQ4+tiledQ6 293,341 µs wall — the
  Q6 tiling measured SLIGHTLY WORSE and was NOT selected (no forced
  winner; the fallback/selection discipline kept the measured best).
  Raw-Hello-8 remains the scope of the Ollama text/count proof.
- NEW bounded workloads through the SAME SSA artifact path with exact
  tokenizer chat-template prompts and multi-token prefill/KV/rope/
  router coverage (each with per-step full-logit + exact greedy gates
  vs runner.forward; separate plans keyed by the prompt workload
  digest): EN chat (14 prompt IDs, 8 tokens) → "Hello! How can I help
  you today", median wall 342,120 µs/token; KO chat (23 prompt IDs, 10
  tokens) → "안녕하세요! 저는 실시간 기상 정보 시스템에", median wall
  374,342 µs/token (tiledQ4 logits rel 2.5e-05, greedy identical).
  Both FORGE_PASS exit 0 (docs/evidence/gemma-forge-{enchat,kochat}-run1.log).
- Next active items (not done): measured non-matvec kernel-family
  profile and one legal gated fusion pass; an EXPLICIT bounded SSD
  expert cache (separate bounded expert-slice buffers — OS paging of
  the whole-GGUF mmap is NOT an SSD engine; in-flight pinning/flush
  accounting in the budget; forced eviction/miss/hit/parity gates
  through the actual SSA path vs resident; bytes/hits/evictions/
  peak-allocated reported separately; no cold-SSD speed claim without
  measurement; no sudo/purge); then a bounded localhost server driving
  the compiled artifact.

## Measured optimization pass 3: tiled cooperative RMSNorm (same date, 6.2x cumulative)

Per-kernel profile (docs/evidence/gemma-profile-run3.log; per-kernel
submission mode can distort attribution — ranking only): g4_rmsnorm
382.1 ms = 44.6% of GPU over 662 calls (~577 µs/call). Root cause: the
original kernel dispatched ONE THREAD PER VECTOR — plain norms reduced
2816 floats serially on a single thread. New `g4_rmsnorm_tiled` (one
threadgroup per vector, strided sum-of-squares partials + tree
reduction, barrier between the read phase and the in-place write phase;
own pipeline-derived tile width; the shared production width validator
rejects illegal widths before dispatch).

Independence: Double-oracle gates on the ACTUAL runtime MSL (weighted
and scale-free, vectors 1/2/8/16, n 256/512/2816, widths 32/64/128,
in-place and out-of-place; deterministic corners: all-zero (exact-zero
output), eps-dominated tiny values gated with an ABSOLUTE 1e-7
tolerance and a nonzero requirement — a /max(1,|ref|) denominator
would be vacuous there — and a 193400-magnitude finite input; all GPU
outputs and Double references have separate explicit finiteness checks
(any non-finite check fails the suite); thresholds 1e-5). The reduction
reassociates the sum order — NOT bit-identical by design.

Full-model measured selection (same run, real GGUF, seven candidates,
wall-median objective): per-kernel baseline 754,092 µs/token; batching
450,094; batching+tiledQ4 280,866; **batching+tiledQ4+rmsTiled
122,565 µs wall, 89,016 µs GPU** — selected (2.29× over the previous
winner, 6.2× vs the per-kernel baseline in the same run; logits rel
2.5e-05 < 1e-4, exact greedy equality). Verified generation median
124,893 µs/token (~8.0 decode tokens/s wall). EN chat (14 IDs, 8
tokens → "Hello! How can I help you today", median wall 172,463
µs/token) and KO chat (23 IDs, 10 tokens → "안녕하세요! 저는 실시간 기상
정보 시스템에", median 203,548 µs/token) both FORGE_PASS exit 0 with
identical greedy sequences through the same artifact path
(docs/evidence/gemma-forge-{rms-run1,enchat-run2,kochat-run2}.log).
Suite: 408 passed / 0 failed. Still below the Ollama server-reported
~71 eval tok/s; remaining measured targets: launch-count fusion
(1799 kernels/token remain), explicit bounded SSD expert cache, and
artifact-driven local serving.
