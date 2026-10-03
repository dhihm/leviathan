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

## Pass 4: explicit bounded SSD expert cache + free-slot prefetch (2026-10-02 UTC)

The prior execution path kept the whole 14.44 GB file mapped into the
GPU address space with no explicit weight-residency policy of its own;
OS paging of that mapping is not an explicit SSD engine. This pass adds
an explicit bounded alternative, selected at construction:

- **Construction** (`Gemma4Runner.init(fileAt:expertCacheBudgetBytes:)`)
  never creates the whole-file GPU mapping (note: a CPU whole-file
  GGUF mapping is still created and retained for parsing/metadata and
  the small F32 parameter reads; the byte budget covers the expert
  slot-pool LENGTHS only — it is not a process-RSS bound):
  every non-expert tensor is compacted
  straight from the GGUF into a dedicated 1,577,172,088 B GPU buffer
  (checked chunked preads), and the two expert tensor families
  (~12.8 GB) stream through two bounded LRU slot pools sized by
  `ExpertCacheGeometry` from the byte budget (checked arithmetic; the
  CLI preflights malformed budgets against the validated tensor
  directory BEFORE any runner/KV allocation). A post-hoc
  `enableExpertCache` on a resident runner also exists (transient
  overlap documented, rollback-tested); the bounded-startup
  constructor is the preferred path. The switch is fail-closed:
  nothing mutates until every step succeeded.
- **Pinning protocol**: the slice in use is ACTIVE for one expert
  iteration; with a segment open it moves to IN-FLIGHT until a segment
  flush; victims are never chosen from active/in-flight slots; when
  every slot is pinned the runner's segment is flushed first
  (flush-before-reuse, counted); a DISCARDED (never-submitted) segment
  releases pins wholesale. Pool allocation is the budget — pinned
  bytes are inside it by construction and reported separately.
- **Transactional loads**: the victim's mapping is dropped BEFORE the
  pread overwrites its bytes; the new key is installed ONLY after a
  full read (EINTR retried; short/EOF aborts) — no partial slice is
  ever addressable. Component tests drive real loads against a
  truncated synthetic file: the failed key is NOT cached and a retry
  reads exact bytes.
- **Prefetch**: SYNCHRONOUS free-slots-only initial read-ahead of each
  layer's routed experts between iterations (blocking preads — no
  async I/O or GPU overlap is claimed or implemented; it never evicts,
  so it cannot change any result). Attribution is per-family,
  per-slot, first-consumption-only. ON vs OFF total bytes read are
  IDENTICAL (6,556,999,680 B in the A/B) — prefetch reclassifies which
  loads are served from installed slots; it is not an I/O reduction or
  speed improvement. The mode is part of the engine identity
  (`expertPrefetch=free-only`) and the plan records the cache budget
  (`expertCacheBudgetBytes`).

Real-model verification (`gemma-expert-cache-verify`,
docs/evidence/gemma-expert-cache-verify2.log; a first capture's byte
counters were printed through a 32-bit `%d` format and are INVALID —
that log is retained renamed
`gemma-expert-cache-verify1-INVALID-FORMATTING.log`; the corrected run
uses 64-bit-safe interpolation plus a Codable JSON stats line):

- Phase 1 (resident snapshot, then RELEASED): failed budget-1 enable
  rejected; resident forward intact.
- Phase 2a (8 MB budget, per-kernel): exact greedy tokens, full
  262,144-logit worst rel **0.0** (same kernels, same bytes — only the
  weight binding differs), 2,880 misses / 2,875 evictions / 0 hits.
- Phase 2b (same budget, batching): identical logits, **540
  flushes-before-reuse observed** — the in-flight pinning path is
  actually exercised — peak pinned 6,690,816 B within the 7,805,952 B
  pools.
- Phase 3 (1 GB budget, 2 replays): prefetch ON 42.3% hits (598
  prefetch-served) vs OFF 31.9%, evictions identical (3,322), logits
  and tokens bit-identical both ways, total bytes read identical.
- Resource ledger: pools 999,161,856 B at a 1 GB budget, non-expert
  compact 1,577,172,088 B, peak pinned ≤ pools. device
  .currentAllocatedSize SAMPLES (4.42 GB at end of construction in the
  1 GB run) cover the driver's view of everything this process had
  allocated at that point — slot pools, the compact buffer, KV caches,
  activations, and allocator overhead — for the REQUESTED sizes; they
  are not a physical-RSS cap or a startup-peak claim. Reads
  are preads that may hit the OS page cache — no cold-SSD latency
  claim anywhere.

Cache-mode FORGE (docs/evidence/gemma-forge-cache-run1.log): plan
identity carries the budget; all seven candidates gated (logits rel ≤
2.5e-05, exact greedy " upon-the-sands-of"); selected
batching+tiledQ4+rmsTiled at 382,115 µs wall in that run. The FINAL
1 GB serving artifact (gemma-forge-serve-artifact.log, plan digest
5ae137ee…) selected the same candidate at 334,232 µs/token wall
(102,395 µs GPU); its ungated served decode median was 325,149 µs
over 7 decode intervals. The resident-mode 122,565 µs selection was
measured in a SEPARATE earlier run — cross-run observations only, no
matched-memory causal comparison.

Suite contributions this pass: 53 cache/prefetch component checks plus
18 plan-identity mutation checks (479 passed / 0 failed total; the
pre-pass suite was 408).

## Pass 5: bounded localhost serving from the persisted artifact (2026-10-02 UTC)

`gemma-serve` (docs/evidence/gemma-serve-run1.log, serve-*.txt,
serve-probes2.txt): loads the 1 GB cache-mode
plan artifact (5ae137ee…), validates its identity — graph/GGUF
content/engine/hardware/toolchain/schedule/planDigest — via
`validatePlanIdentity`, i.e. everything EXCEPT the workload binding:
the selected candidate was measured on the plan's profiling workload
(recorded and disclosed at startup; no fastest-on-all-prompts claim).
Runtime executes the STORED typed SSA schedule and selected flags
only — no model.py, no agents, no Python at runtime. One POST
/generate at a time (serial GPU) plus GET /health; request count,
prompt length, output tokens, header/body sizes, and one absolute
15 s per-request read deadline (socket timeouts shrink to the
remaining budget; {0,0} timeval rounding fixed; EAGAIN classified as
timeout) are all bounded; duplicate Content-Length, Transfer-Encoding,
malformed header lines, oversized headers, wrong media type,
fractional/boolean max_tokens, and numeric chat all fail closed.

Measured/evidence: the first request ran with --gate-first-request —
gated step-by-step against the runner forward path (a SEPARATE
execution path sharing the same validated kernel primitives, NOT a
mathematically independent implementation) — exact tokens, full-logit
rel < 1e-4, both-arrays finiteness, final exact sequence equality:
ids 3324 236772 1437 236772 236751 2964 236772 1340 (" upon-the-sands-
of"), identical to the forge run. The ungated repeat returned the
exact same 8 IDs; max_tokens=1 returned [3324]; EN chat ("I am Gemma
4, a large") and KO chat ("안녕하세요! 저는 실시간 기상 정보 시스템에")
served 200; the EN chat IDs are EXACTLY reproduced by the resident
`gemma-generate` CLI reference (a separate execution path sharing the
same validated kernel primitives) on the same input
(gemma-generate-serve-en-parity.log). Negative HTTP evidence: 20
cases total — 13 body/route probes (serve-negatives.txt) + 6 raw
header/protocol probes + 1 idle-connection read timeout (408 after
15.0 s) — all asserted, with /health verified recovered after each
full probe batch. Route note (documented as-is, not changed):
GET /generate and POST /health currently return 404 (path-first
routing), not 405. The 20-request bound ended the first server
cleanly (SERVE_DONE); only task-owned PIDs were started/stopped.
Remaining known limits: ~330 ms (325,149 µs) per served decode token
in 1 GB cache mode vs the Ollama server-reported ~71 tok/s (external
baseline, text+count agreement only — no global quality or
competitiveness claim), single-model serial GPU, greedy subset only.

## Pass 6 — matched warm/cold acceptance vs the same-GGUF Ollama, and a measured F32 router candidate (2026-10-02 UTC)

External baseline harness (tools/baseline/ollama_warmcold.py, stdlib
only): talks ONLY to the task-owned Ollama 0.23.0 daemon on 127.0.0.1:
11499 (the user's own daemon on 11434 is never touched). Preflight
fails closed unless: the daemon answers, no leviathan-cli process is
resident (sequential GPU), the model exists, the /api/show
parent_model blob sha256 EQUALS the sha256 of the authorized GGUF
computed by the harness itself (byte-linked: both are
3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d),
and no model is resident in Ollama. Protocol: raw "Hello", greedy
(temperature 0, top_k 1, top_p 1, repeat_penalty 1, seed 42),
num_predict 8, num_ctx 4096, streaming; 3 COLD rounds (keep_alive 0,
unload VERIFIED via /api/ps before the next round; "cold" = not
resident in the server — the OS page cache is NOT purged, so storage
is not guaranteed cold) + 5 WARM rounds (keep_alive -1). Client TTFT
(request start to first nonempty chunk) is reported separately from
the API's server-side load/prompt_eval/eval durations; only exposed
text/count agreement is claimed (" upon-the-sands-of",
prompt_eval_count 2 and eval_count 8 in every round) — hidden token
IDs are not claimed, and per-token latency is NOT derived from chunk
timing. Memory: post-request ps RSS point samples only (in-stream
pgrep/ps would block the reader and inflate client timestamps); the
reported value is the max across rounds (a sampled maximum, NOT a
peak): runner child 14.4 GiB, task-owned serve 124 MiB. Machine load
recorded via sysctl vm.loadavg (start ~{9.6-10.7}, end up to
{15.3 13.6 12.0} — nontrivial background load; comparability
context only).

Measured Ollama, FINAL run (docs/evidence/ollama-warmcold.json; the
initial run is preserved as ollama-warmcold-run1.json and is NOT
mixed into these statistics): COLD client TTFT 6.09-10.01 s (server
load 5.87-9.67 s), WARM round-1 TTFT 5.45 s (still pays a 5.27 s
load) then steady warm TTFT 0.23-0.33 s, steady-warm api_server
eval_duration median ~0.117 s per 8-token request → ~14.6 ms/token
(68.4 tok/s, server-reported; no per-token client claim).

Leviathan harness (tools/baseline/leviathan_warmcold.sh): PRIMARY
path is the persisted selected SSA artifact — step 0 (re)forges the
plan so its identity binds the CURRENT executable SHA + GGUF digest
+ hardware; step 1 is a SEPARATE gated verification invocation
(gemma-generate --gate: one untimed reference replay, exact greedy
ids + rel < 1e-4 per logit array + finiteness on both arrays); the
measurement invocations run WITHOUT the gate (so a first cold
repetition is not pre-warmed by any replay), still validate plan
identity, and time identity-prep separately from prefill TTFT.
gemma-generate gained --repeat N (in-process warm repetitions via KV
reset, cross-repetition determinism asserted), --plan/--graph
(stored-schedule execution), and --gate (optional verification).
COLD = 3 fresh processes with /usr/bin/time -l; WARM = 1 process x
5 repetitions. Client-token-timing disclosure: this CLI is not a
streaming server and stdout is emitted at completion, so no
client-side first-token timing is claimed for it; the reported
TTFT/decode are in-process ContinuousClock measurements. PROTOCOL
DISCLOSURE: every Leviathan repetition resets the KV cache and
re-prefills; Ollama's server may reuse the KV prefix across
identical warm requests (its warm prompt_eval 0.02 s vs cold
0.14-0.18 s) — TTFT/prefill are NOT strictly like-for-like across
engines; decode per-token medians are the comparable metric.

Measured Leviathan, FINAL run only (docs/evidence/gemma-acceptance-
cold1..3.log / -warm.log / -gate.log / -forge.log; earlier runs
preserved as -before.log and -intermediate.log and NOT mixed in):
COLD client total 29.29-31.50 s, process peak RSS 2,126,528,512-
2,322,399,232 bytes and peak memory footprint 16,474,838,464-
16,477,377,984 bytes (client-side /usr/bin/time -l, incl. Swift
runtime; not a GPU residency claim), composed of load 498.7-508.3 ms
+ identity prep (graph parse + 14.4 GB GGUF hash + plan load)
11,112.8-11,410.9 ms + first prefill 15,355.0-17,329.0 ms (cause
not causally measured — plausibly first-touch/lazy pipeline
compilation, labeled unknown) + decode. WARM steady state
(repetitions 2-5): prefill TTFT 198.0-214.2 ms, decode medians
118.70/118.96/121.33/124.59 ms/token, GPU total 764.9-769.7 ms per
9 token-forwards. The residual wall-minus-GPU difference is
reported as measured only — NOT attributed to launch overhead or
readback without a dedicated measurement. Metal device allocated
sample 15.168 GiB (point sample of currentAllocatedSize — not a
peak). Cross-run caveat: these runs and the pass-3 122,565 µs/token
are separate executions; no causal comparison across runs is
claimed. Comparison caveat: the Leviathan decode median is over 7
inter-token steps of an 8-token greedy output (the first token comes
from prefill) while Ollama reports eval_count 8 — an approximation,
not an identical token accounting. With that caveat, the Ollama
steady-warm baseline is ~8x faster per decode token (~14.6 vs
~120 ms) on the same GGUF; both engines agree exactly on the greedy
continuation text and count.

Bottleneck analysis (gemma-profile --q4-tiled --rms-tiled, exact
per-kernel attribution over 2 tokens — per-kernel mode; batched
execution timing may differ; gemma-profile-selected.log): the
selected-kernel GPU time splits as g4_q4_matvec_tiled 32.4% /
g4_q6_matvec (LM head) 23.8% / g4_f32_matvec (MoE router
ffn_gate_inp) 17.4% / g4_rmsnorm_tiled 12.3% / g4_attn 8.0%. The
router is dispatched as the scalar one-thread-per-row kernel: 31.9
ms per 2 tokens (532 us/call x 30 calls/token ≈ 16 ms/token) — one
thread per row for a 128-row x hidden-dim matvec underparallelizes
the GPU.

The ONE optimization this pass (g4_f32_matvec_tiled): a tiled
cooperative F32 matvec (one threadgroup per row, strided input,
128-slot partials tree reduction), a new legal Forge candidate
matvecTiledF32 in the grid, plumbed through dispatch, engine
identity (candidates + f32TileWidth), plan schema, and the CLI
profiler. Numerical gates: FP64-oracle test on the ACTUAL runtime
shader at router-relevant geometries (inDim 256/2112/2816/3840 x
outDim 128, cancellation-prone data, tolerance 1e-4 * sum|terms| +
1e-6 — fp32 dot-product rounding scales with the term-magnitude
sum; a flat 1e-5 rel gate is not met even by the plain kernel's own
summation order), plus the existing full Forge gates. Suite: 499
passed / 0 failed (479 + 20 new f32-tiled checks incl. illegal-width
rejection via the production validator).

Measured outcome (gemma-acceptance-forge*.log, distinct plans/logs
per diagnostic; intermediate runs preserved as -intermediate.log):
the kernel itself is 20x faster per call (532 -> 26 us; per-kernel
GPU 31.9 -> 1.6 ms per 2 tokens,
gemma-profile-selected-f32tiled.log). As a STANDALONE candidate it
is correct (full-logits rel 1.8e-05). STACKED with the selected
tiled Q4/RMS candidates it FAILS the pre-registered 1e-4 full-logit
gate at decode step 2 with worst rel 1.066e-04 (greedy token
identical; the gate now computes the true full-array worst
explicitly after length/finiteness checks — the previous
allSatisfy-based note could short-circuit and understate the worst;
on a shape/token mismatch the worst is reported as 0.0 and the
mismatch is labeled separately). A Kahan-compensated variant
measured the SAME worst value (and default Metal fast-math may
cancel compensation), so the plain reduction is kept; no claim is
made about which term dominates the stacked drift. The Forge
therefore did NOT select the stacked combination: selection remains
batching+tiledQ4+rmsTiled at 113,090-120,389 us/token across this
pass's forges (same configuration as pass 3 within run-to-run noise;
no end-to-end win is claimed). This mirrors the pass-2 Q6-tiled
precedent: measured, gated, not selected — the machinery (kernel +
candidate + identity + oracle tests) is the reusable artifact.

## Pass 7 (2026-10-02 UTC): fair KV-aligned decode baseline, batched
## wall/GPU breakdown, and the full-workload legal-candidate fallback

### Motivation
Pass 6's warm comparison had a KV-prefix asymmetry: Ollama could reuse
the identical request's prompt KV while Leviathan reset and re-prefilled
every repetition. This pass eliminates the ambiguity with AUTHORITATIVE
cache-policy evidence, measures the ACTUAL batched wall-vs-GPU split of
the selected SSA path, and delivers one gate-clean improvement chosen
from that breakdown.

### Authoritative cache policy (not a timing heuristic)
A second task-owned Ollama 0.23.0 daemon (127.0.0.1:11498, started with
OLLAMA_DEBUG=1, OLLAMA_NUM_PARALLEL=1, OLLAMA_MAX_LOADED_MODELS=1,
OLLAMA_MODELS=/tmp/task5-ollama-models; the user's own 11434 daemon and
the task's 11499 daemon were never modified) exposes the runner cache
manager's own log line per request:
  cache.go:151 "loading cache slot" id=0 cache=<len> prompt=<n> used=<u> remaining=<r>
Observed (A/B/D/C probe preserved in docs/evidence/
ollama-11498-debug-fairdecode.log): for the 2-token "Hello" prompt both
the identical-request control AND the post-distractor request show
used=1 remaining=1 — only the BOS token is retained and the non-BOS
token is freshly evaluated in both cases (the distractor itself shows
prompt=6 used=1 remaining=5), and runner.parallel=1 appears in the same
log, proving a single slot. The official docs additionally state
OLLAMA_NUM_PARALLEL defaults to 1 (docs.ollama.com/faq). The daemon was
stopped after the phase; its model was verified unloaded first, and the
11499 daemon was left running and empty.

### Fair decode baseline (32 tokens, policies aligned)
tools/baseline/ollama_fairdecode.py: 5 warm measured rounds, each
preceded by a serial distractor; per-round PROOF from the log that slot
0 shows prompt=2 used=1 remaining=1 (docs/evidence/
ollama-fairdecode.json; the earlier protocol run is kept as
ollama-fairdecode-run1.json and never mixed with final statistics).
Leviathan aligns via the new gemma-generate --bos-align mode
(tools/baseline/leviathan_fairdecode.sh): each warm repetition
prefills [BOS] untimed and times the one-token continuation + 31 decode
steps — both engines then freshly evaluate exactly one prompt token per
round. Decode metric conventions are disclosed side by side: Ollama
eval_duration/eval_count is the standard server metric (per generated
token, N=32); the first-to-last normalization is an alternate only.
Measured (loadavg 1/5/15-min: Ollama run { 5.86 12.59 11.83 } ->
{ 5.93 12.27 11.73 }; Leviathan harness { 13.61 9.52 9.74 } ->
{ 10.09 9.54 9.68 } — recorded, not controlled, no causal claim; the
final VL cold/warm runs were not individually loadavg-recorded):
  Ollama (final JSON): eval 0.472677954-0.494303125 s / 32 tokens,
            median 0.480543623 s = 15.02 ms/eval-token (66.59 tok/s);
            prompt_eval 0.0194865-0.027267 s (median 0.022870417);
            identical text all 5 rounds.
  Leviathan (selection below): warm aligned decode totals
            12236.1-13097.8 ms for 31 steps across 5 repetitions,
            median 12788.6 ms = 2.424 tok/s (pre-improvement selection:
            15103.9-15545.2 ms totals — a descriptive ~16% reduction
            across different binaries, times, and uncontrolled load; no
            controlled claim). Final cold process (final binary, final
            plan): total 41.44 s, load 257.0 ms of which runtime Metal
            library + pipeline compile 6.6 ms (printed separately;
            driver-side caching not claimed either way), identity
            validation prep 12466.9 ms (graph parse + 14.4 GB GGUF hash
            + plan load), first prefill 15976.9 ms (first-touch/
            lazy-init not causally measured), decode 11607.8 ms
            (median 374.44 ms/token), peak RSS 2,253,258,752 B, peak
            memory footprint 16,477,181,376 B, Metal device allocated
            sample 15.168 GiB (a sample, not a peak claim; distinct
            from OS RSS/high-water marks).
Both engines produce the exact same 32-token text
" upon-the-sands-of-time-of-the-sands-of-sands-of-the-sands-of-"
(token-ID agreement asserted on the Leviathan side; text/count only on
the Ollama side — no hidden-ID claim). Descriptively, the Ollama N=32
per-eval-token rate (66.59 tok/s) and the Leviathan warm per-step rate
(2.424 tok/s) differ by roughly 27x; these accountings differ (Ollama
includes the first generated token in eval_duration's N=32; Leviathan's
31 steps exclude it), machine load was uncontrolled, and no causal
attribution is made. The competitive-performance objective vs Ollama
is UNMET at this scope; the reusable assets are the aligned protocol,
the gate machinery, and the measured improvement.

### Full-workload legal-candidate fallback (strict gates unchanged)
The pass-6 8-token gate hid a real fact: at 32 tokens the accumulated
cross-path drift of the reassociating tiled kernels crosses the strict
rel < 1e-4 gate (measured worst 1.032e-04-1.171e-04 at steps 25-28;
greedy ids/text still exact). Rather than relax anything, gemma-forge
now gates candidates over the REQUESTED --gen-tokens length with a
legal fallback: candidates are tried in the SAME order the short
selection used (measured wall-median order at the decodeTokens=4
measurement workload; current selection first, then medians with a
deterministic legal-enumeration tie-break) and the FIRST candidate in
that order that passes prefill + EVERY step of the fullGateTokens=32
gate (rel < 1e-4 per full logit array + exact greedy ids) is selected.
Caveat stated plainly: the gate compares the SSA path against the
runner reference path, which shares the same kernel primitives — the
only independent numerical check is the per-kernel FP64 oracle in the
test suite. Rejections are recorded IN THE PLAN
(fullGateTokens/fullGateRejections) with pairing/bounds/digest-
coverage gates in the test suite; stale plans fail closed and re-forge.
GemmaForge.reselect rebuilds the plan (schedule/identity unchanged,
digest recomputed) and MERGES same-length re-gate history: a cached
plan re-gated at the same length keeps its prior rejection reasons (a
now-passing selection can produce an empty list this run) and appends
only new unique reasons; a different gate length resets the record.
Verified in production: the cached gemma-forge re-run preserved all 3
rejections with the parsed whole-plan JSON object identical before and
after (root-verified object equality; digest
575adf9f9f17886d29bbd6b414a41bfac8218e4b76c2c4444e9a5969e8454c93).
First-fallback artifact preserved at
docs/evidence/gemma-plan-full32-first-fallback.json.

### Batched wall-vs-GPU breakdown (--breakdown, opt-in counters)
Runner-primitive counters (monotonic DispatchTime; no timer reads when
off — one branch remains; submission/synchronization untouched): encode
(deferred-segment kernel encoding; the GPU is idle then in batched
mode, no overlap), flush wall (endEncoding+commit+waitUntilCompleted;
OVERLAPS GPU busy time — the host share is flush-gpu, clamped at 0),
readCopy, writeCopy; GPU busy from gpuMicrosTotal; everything else is
the residual (CPU router softmax/top-k, operand resolution, CLI).
Overhead control — same workload, same binary and plan, instrumentation
off (medians 482.15/491.92 ms/token) vs on (496.90/484.64), n=2 per
arm: the elapsed times look similar; no statistical claim is made from
two observations. Result (31 decode steps, batching-only selection, no
per-run loadavg recorded for this diagnostic): GPU 89.4-91.7% of wall,
flush-host 6.1-7.4%, encode 0.3-0.5%, readCopy 0.2%, writeCopy ~0%,
residual 1.8-2.4% — GPU busy time dominates, with material host-side
flush overhead. (Per-kernel attribution remains exact only in
per-kernel mode; it does not substitute for this batched breakdown.)

### The improvement: vectorized-load RMSNorm (bit-identical, selected)
Priority from the breakdown: GPU busy time dominates the wall, with
material host overhead. Plain-mode per-kernel attribution
(docs/evidence/gemma-profile-run3.log) has g4_q4_matvec 44.7% and
g4_rmsnorm 44.6% of GPU time. g4_rmsnorm_vl consumes x (and w) in
float4 chunks with an arithmetic sequence IDENTICAL to g4_rmsnorm —
asserted EXACTLY bit-equal on hardware by the new test block (widths
256/2112/2816, weighted/scale-free, in-place and out-of-place) plus the
production predicate Gemma4Runner.rmsVectorLoadGuards (positive
multiple of 4; scaled weight offset 16-byte aligned; fallback to the
original kernel otherwise) tested directly.
  - A Q4 matvec vectorized-load variant was also written and measured
    ~2.1x per call in the synthetic harness (856 -> 420 us — but that
    number includes per-call MTLBuffer allocation and a dedicated
    command-buffer submit per call, so it is NOT a pure GPU-kernel
    time; kept as a printed diagnostic), yet its COMPILED outputs
    differed from the original kernel (worst rel up to 1.0e-05; cause
    not established), so bit-equality could not be proven: the variant
    is REJECTED from legal dispatch (excluded from the production
    pipeline list) and survives only as a printed diagnostic in the
    test suite.
The new vectorizedLoads Forge candidate (RMS VL only; batched and
unbatched entries) measures with logits rel 0.0e+00 vs the reference:
  batching+plain:  439,691 us/token wall, 398,715 us/token GPU
  batching+VL:     315,820 us/token wall, 278,191 us/token GPU
                   (-28% wall, -30% GPU, zero measured drift)
and it is the first candidate in the short-selection order that the
fullGateTokens=32 gate passes (the three faster tiled candidates were
rejected and recorded first; the same-session end-to-end numbers are
above). Post-improvement breakdown: GPU 84.4-85.5%, flush-host
10.5-11.1% (the freed GPU time shifts the split toward host share).
Default suite: 527 passed / 0 failed.

## Pass 8 (measured 2026-10-03 local / KST, 2026-10-02 UTC): teacher-forced layerwise numerical diagnosis + bounded Q6 composition candidate

Goal: determine WHY the parallel tiled candidates exceed the strict
1e-4 logits gate at 32 tokens, before writing any further kernels. A
new opt-in diagnostic subcommand `gemma-diagnose` (Sources/leviathan-cli,
`--file`, `--tokens`/`--text`, `--gen-tokens 2...64`, optional
`--graph`) teacher-forces the reference-generated greedy token history
and replays each kernel/scheduling axis separately against per-layer
residual checkpoints (attn_norm_h, post_attn, post_layer). The probe is
OFF by default and adds no readbacks in production runs; branch cost at
the three checkpoint call sites is unmeasured. Fail-closed checks:
reference probe-on vs probe-off must be bit-identical, every axis's
probe-on vs probe-off logits must be bit-identical, and metric helpers
fail on shape/count/nonfinite inputs.

Workload: Hello fixture, prompt ids 2 9259, 31 reference continuation
inputs + 2 prompt tokens = 33 teacher-forced positions for 32 output
steps. Same token history for every axis, which rules out different
greedy-token feedback on this fixture; numeric KV/residual propagation
differences remain possible. Full log:
docs/evidence/gemma-diagnose-hello32.log.

Findings (full 262144-logit gate formula |a-b|/max(1,|ref|)):
  - segment_only and vl+segment are BIT-IDENTICAL to reference over
    all 32 steps (snapshot bitPattern comparison, anyBitsDiffer=false).
    CORRECTION (2026-10-03, pass 9): a reset-order bug in the
    diagnostic's teacher-forced replay (runner.reset() ->
    discardOpenSegment() clears autoSegment AFTER the axis was applied)
    means the segment_only / vl+segment axis replays actually executed
    WITHOUT deferred-segment batching — per-kernel submissions with
    the plain kernels. The bit-identity result is therefore a valid
    plain-kernel replay comparison (and segment batching is
    numerically identical in dispatch ORDER by construction), but it
    did NOT measure batched execution itself; the batched-mode
    evidence for the selected configuration remains the pass-7 FORGE
    gates (graph executor, 0.0e+00 measured drift), which are
    unaffected by this bug. The bug is fixed in pass 9 (the axis is
    re-applied after every reset with a fail-closed
    segmentBatchingActive guard).
  - q4_tiled alone reproduces the pass-7 rejection digit-for-digit:
    worst rel 1.032e-04 at output step 28; per-axis composed SSA
    re-runs of the three persisted pass-7 rejections give
    batch+q4+rms 1.892e-04 (its former first failure was at step 25),
    batch+q4 1.032e-04, batch+q4+q6 1.041e-04; selected batching+rmsVL
    is 0.0e+00. argmax mismatch 0/32 on EVERY axis including the
    tiled ones. At the q4_tiled worst step (28): reference argmax
    margin (top1-top2) 2.5625, KL 1.151e-10, relL2 8.883e-06.
  - rms_tiled worst rel 2.882e-04, f32_tiled 1.347e-04,
    q6_tiled 9.332e-06 — all with 0/32 argmax mismatches.
  - Per-layer residual error is nonzero from layer 0 and its
    magnitude oscillates, with larger errors observed at some later
    layers (not monotonic, not universal); per-step error is 2-6e-05
    with input-dependent spikes (steps 25/28), not monotonic in
    position.
  - Bounded ORIGINAL independent FP64 oracle on the actual GGUF bytes
    (layer 17 attn_q, 256 rows, input captured at position 29):
    rel-to-FP64 worst plain 1.300e-06 vs tiled 4.734e-07. This is a
    SINGLE Q4 sample; it does not bound all tiled reductions, all
    operators, or overall math quality.

Scoped reading: the observations are CONSISTENT WITH per-op float32
reassociation (reduction-order rounding) differences amplified by
later-layer sensitivity on this fixture. They do NOT prove absence of
semantic/layout/cache-routing/quant decode bugs, do not establish an
exclusive root cause, and the tiny KL / matching argmax on this one
fixture does not demonstrate broad lexical quality. The production
strict 1e-4 gate, exact-ID, and finiteness contracts are UNCHANGED; no
gate loosening and no FP64-anchored gate change is proposed. The FP64
sample is a one-operator oracle, not an independent full-model
reference.

Bounded candidate-composition experiment (the one pass this phase):
two legal Q6-tiled+RMS-VL entries (batched and unbatched) appended to
GemmaForge.candidates(). Fresh full Forge run at fullGateTokens=32:
FORGE_PASS, engine 988efa10ae6d287bf3d5dd285ea39911afedf5feaca0867cdba1c32c14fd31cf,
plan digest 57a2d5d9b942df0e51db39287be0616a59f404f40d1669ee3b6ea92711f70686
(docs/evidence/gemma-plan-q6vl.json, full log
docs/evidence/gemma-forge-q6vl-32.log). fullGateRejections = 2 in this
run's order (batch+q4+rms, batch+q4); the third pass-7 rejection
(batch+q4+q6) sorts after the first passer in this run's timing order
and was not re-gated, so the composed Q6 candidate's own 32-token gate
outcome is UNKNOWN (selection stops at the first passer). Short
3-sample timing in the same forge observed the Q6+VL batched candidate
at 326,671 us/token vs incumbent RMS-VL 317,388 us/token, so Q6+VL was
NOT SELECTED this run; occupancy/utilization counters were NOT
measured, so no causal or global "Q6 is slower" claim is made. The Q6
~4.6% headroom figure remains a HISTORICAL per-kernel plain-mode GPU
diagnostic, not a current batched-breakdown attribution.

Default suite with the real GGUF on Metal: 627 passed / 0 failed
(docs/evidence/tests-q6vl-627.log). No competitive-performance claim is
made for this phase.

## Pass 9 (2026-10-03 UTC): parameterized Q4_0 operator lowering — one operator family, emitted Metal, FP64 oracle, artifact-bound persisted execution

Design first (root-reviewed before any model run):
[docs/Q4_LOWERING_DESIGN.md](Q4_LOWERING_DESIGN.md), responding to the
pass-8 [architecture audit](ARCHITECTURE_AUDIT.md) stages 1–4 for
exactly ONE operator family: the dominant Q4_0 quantized matvec.

What was implemented (all code/tests/docs original, this phase):

- **Python op input** (`tools/q4_lower/q4_op_model.py`,
  `export_q4_op.py`, example `model_q4_attn.py`, stdlib only): ONE
  bounded operation with full semantics/shape/layout/quant/precision
  metadata — `q4_0.matvec`, inDim 2816, outDim 4096, bound to
  `blk.0.attn_q.weight` exactly as the GGUF tensor directory pins it.
  The Swift side re-validates every field and cross-checks the
  declared shape against the actual tensor (mismatch rejected).
- **Operation IR + schedule derivation** (`Q4OpIR`,
  `Q4ScheduleDeriver`): legal schedules derived from the operation
  SHAPE (blocksPerRow = inDim/32 governs the T ladder; small
  geometries legally admit only small T) and the MEASURED device
  (execution width, threadgroup maximum from the compiled incumbent
  pipeline). Parameters with real emitted-code effect: threadsPerRow,
  rowsPerGroup, accumulation (plain f32 or Neumaier-compensated f32 —
  Apple GPUs expose no usable FP64, so no GPU double strategy exists),
  and scalar/float4 activation loads. Baseline (T=1, incumbent-mapping
  serial order) always present; ≤ 8 candidates; threadgroup partials
  bounded by the device maximum.
- **Metal source EMISSION** (`Q4KernelEmitter`, codegen q4gen-v1):
  the complete MSL kernel is generated from the schedule parameters —
  contiguous block-chunk decomposition, uniform threadgroup barrier
  with an active-row guard (inactive rows never read OOB), ascending-
  thread ordered combine — not a table of prewritten kernels. Emission
  is deterministic and its SHA-256 is an artifact identity input.
- **Independent CPU FP64 oracle** (`Q4Oracle.matvecFP64`): Double
  dequant/dot over REAL bounded weight rows read from the bound GGUF
  tensor (256 rows) with a deterministic seeded activation; the
  incumbent kernel is measured against the same oracle (measured, not
  assumed exact). Component contract UNCHANGED: finite outputs and
  rel-to-FP64 < 1e-4, the same tolerance the hand-written Q4/Q6 GPU
  kernels are held to in the test suite. Measured on the real tensor:
  incumbent 7.211e-07; all 8 emitted candidates accepted — the T=1
  baseline at 7.211e-07 (same reduction order as the incumbent) and
  the seven cooperative candidates at 1.865e-07–2.546e-07 (component
  line of docs/evidence/q4-select-root-final.log; per-candidate
  records in docs/evidence/q4-artifact-root-final.json).
- **Artifact identity + persisted execution** (`Q4LoweringArtifact`):
  binds op digest, emitted source + SHA, codegen version, schedule,
  hardware, toolchain, GGUF content digest, and the running
  executable's bytes (artifact JSON bounded to 1 MiB); validation
  re-emits from the schedule and
  requires byte equality (tamper/rule drift rejected), re-validates
  the decoded op, requires the component record's oracle distance to
  be finite and within 1e-4, and — whenever a model-gate record exists
  — requires the persisted schedule to have passed EVERY fixture gate
  even when the policy selected the incumbent. The PRODUCTION install
  (`Gemma4Runner.installQ4GeneratedKernel`) additionally refuses
  component-only artifacts and checks the hardware profile against the
  actual compiled incumbent pipeline. The generated kernel serves ONLY
  its bound tensor (exact name + inDim match); every other Q4 matvec —
  other layers, other shapes, expert slices — keeps the incumbent
  kernel. Dedicated dispatch counters (correct in segment-batched
  mode) prove the emitted kernel actually executed.
- **Bounded per-candidate orchestration** (`q4-lower select`, ONE
  process so the executable digest is stable): component gates for
  every derived candidate; canonical chat fixtures (the same
  user→model template derived from the GGUF's embedded Jinja template)
  for English AND Korean plus the raw Hello continuity fixture;
  teacher-forced same-history replay per candidate gating the full
  262144 logits at the UNCHANGED strict 1e-4 relative formula + exact
  greedy argmax equality + finiteness at every step; THREE paired
  matched-timing repetitions per passing candidate with alternating
  incumbent-first/candidate-first order (raw per-arm samples and the
  actual history windows recorded in the artifact); policy selection
  only when a candidate beats ITS OWN paired incumbent beyond the
  documented 2% threshold (GemmaForge.noiseRelWall), else the
  incumbent stays the production selection; final persisted-execution
  proof reloads the artifact from disk, validates, installs through
  the production path, and gates one more fixture replay with
  dispatch-counter evidence.
- **Fail-closed everywhere**: artifact rejection at any check leaves
  the incumbent kernels active; a generated replay that never
  dispatches is an error, not a silent fallback; timing arms throw on
  any forward failure (a failed run is never timed).

Tests (root-executed on the final release binary): default suite 552
passed / 0 failed
(docs/evidence/tests-q4lower-root-final-default.log) and with the real
GGUF on Metal 653 passed / 0 failed
(docs/evidence/tests-q4lower-root-final-realgguf.log) — 627
pre-existing plus 26 new: op-IR validation (typed error cases),
deriver shape/hardware bounds, emitter determinism/distinctness/
parameter baking, hand-computed FP64 oracle block, GPU component
gates for the emitted T=1 and compensated T=8 kernels, full component
forge round-trip, artifact tamper/hardware-mismatch rejection, and a
deferred-segment batching regression (one batched forward after a
reset must submit fewer command buffers than dispatched kernels — the
reset-order bug class root measured live).

Measured selection outcome (root-executed orchestration, same release
binary, engine SHA-256 0a3aa9bdf9896dcf80d19099ee958db1f305144d0ab125bad6397d9c654e1fb4, full log docs/evidence/q4-select-root-final.log,
artifact digest recorded in docs/evidence/q4-artifact-root-final.json;
47.3 min wall (2839.69 s), peak memory footprint 31,226,840,192 B /
max RSS 3,328,737,280 B — near the 32 GB hardware RAM capacity, NOT
lightweight; the existing authorized 14,439,363,584 B GGUF was used,
no new download; swap of 52-53 GiB and system LOAD AVERAGES of 8-17
during the run are recorded in
docs/evidence/q4-resource-root-final.jsonl without a causal claim):

- Component (real tensor bytes, 256 rows): incumbent rel-to-FP64
  7.211e-07; all 8 emitted candidates accepted — the T=1 baseline at
  7.211e-07 (same reduction order as the incumbent) and the seven
  cooperative candidates at 1.865e-07–2.546e-07.
- Model gates (32 steps, EN chat + KO chat + raw Hello, teacher-forced
  same-history, full 262144 logits, unchanged strict 1e-4 + exact
  argmax + finiteness): 6/8 candidates passed 3/3 fixtures (worst
  full-logit rel 7.04e-05–8.85e-05; the emitted T=1 baseline measured
  0.000e+00 full-logit drift with exact argmax on these fixtures — a
  zero measured drift, not a bitPattern-level bit-identity proof).
  Two float4 variants (T16 compensated float4, T16 R2 compensated
  float4) FAILED the raw-Hello fixture at step 9 (worst 1.026e-04,
  outside the gate) — recorded as measured rejections, not retried or
  loosened.
- Matched timing (3 paired repetitions each, alternating
  incumbent-first/candidate-first order, 31 timed tokens per fixture
  arm, per-fixture sums over the three fixtures). The timings ARE
  whole-model decode wall times over the three fixtures; what is
  bounded is the CHANGE (one bound tensor's kernel).
  q4gen_matvec_T8_R1_f32_scalar — the policy selection — measured
  per-repetition candidate/incumbent sums of 1472045/1386046,
  1388390/1444605, and 1377647/1489661 us: paired gains of −6.20%,
  +3.89%, and +7.52% (the FIRST repetition was SLOWER than its paired
  incumbent); the median-of-repetitions comparison (1388390 vs
  1444605 us, +3.89%) crossed the 2% policy threshold and selected
  it. OBSERVED 2%-policy selection with SIGN-CHANGING paired
  repetitions: stable or competitive performance is NOT demonstrated
  — the per-repetition spread includes a slower repetition, other
  candidates showed similar or larger variance, and only one bound
  tensor's kernel changed; this is not a robust general speedup and
  not a complete whole-model compiler result.
- Persisted execution proof: the artifact was saved, RELOADED from
  disk, identity-validated, installed through the production path,
  and gated one more EN-chat replay (worst rel 4.965e-05) with 52
  generated-kernel dispatches on the bound tensor counted by the
  dedicated counters.

Fail-closed checks exercised in this run: the two gate failures were
recorded and those candidates were excluded from timing and selection,
and the production install path rejects any schedule without a fully
gate-passed model-gate record.

Standalone tooling in the same run: `q4-lower verify` re-validated the
persisted artifact against the actual GGUF
(docs/evidence/q4-verify-root-final.log); the standalone component
`q4-lower forge` artifact (model gate nil) remains
diagnostic-and-test-only by construction — production install refuses
component-only artifacts.

Honest scope: this is ONE parameterized operator-level lowering family
with shape/hardware-derived schedule generation, independent numerical
validation, artifact-bound persisted execution, and paired matched
measurement. The broader joint goal (general Python model compiler
with competitive performance) remains UNMET.

## Pass 10 (2026-10-03 UTC): serial-order-preserving cooperative RMSNorm
lowering — SELECTED at 6.31% policy margin after a clean full root run;
bounded evidence, joint goal still UNMET

Motivation: the production RMSNorm (`g4_rmsnorm_vl`, one thread per vector,
bit-identical vectorized loads) serially reduces n = 2816 floats on a
single lane; pass 3's cooperative `g4_rmsnorm_tiled` was faster but
REASSOCIATES the sum and was correctly rejected by the strict gate
(2.882e-04). Phase 10 tests the audit's untried structure: PARALLEL staging
of exact input copies into threadgroup memory, a SERIAL lane-0 sum of
squares in the ORIGINAL scalar order, then a PARALLEL strided output apply
whose per-element arithmetic is identical to the incumbent's. Design:
docs/RMS_LOWERING_DESIGN.md (root-approved with corrections folded in:
`--vectorized-loads` profile flag; n=512 global head norms documented;
per-dispatch eps/span/alignment/alias checks; T ladder derived from the
measured execution width with the proximity rule stated; bitPattern-level
equality; warmed component GPU timing screen; exact Float-bitPattern eps
install guard).

Implementation (all GLM-authored, original):
- Sources/LeviathanCore/RMSLowering.swift — RMSOpIR (Python-captured op
  JSON validated into IR; capture-time vector batch vs per-call runtime
  dispatch documented), RMSSchedule (threadsPerVector + staging width;
  legality includes the ACTUAL static threadgroup memory n*4+4 bytes —
  the staged array plus the invShared slot), hardware-derived deriver
  (powers of two ranked by log2 proximity to the measured
  threadExecutionWidth; an over-cap shape derives an EMPTY set, never an
  illegal baseline), deterministic MSL emitter (codegen "rmsgen-v1"), CPU
  FP64 oracle, component inputs (real 11 KB tensor pread or synthetic),
  RMSEmittedKernel (+ warmed GPU timing screen at the PRODUCTION vector
  count, ranking/pruning only), RMSIncumbentKernel (standalone
  g4_rmsnorm / g4_rmsnorm_vl runner defining the bit-equality reference),
  artifact with full identity binding, component forge (accept =
  bitPattern-identical to the incumbent production kernel in BOTH
  out-of-place and in-place modes AND rel-to-FP64 < 1e-4), and the
  phase-9-style selectAndPersist orchestration (screen funnel: at most the
  two fastest non-baseline candidates reach the model gates; 3 paired
  matched-timing repetitions; 2% policy; persisted-execution proof).
- Gemma4.swift — rmsnorm() routes the bound (n, scaled) family through the
  generated pipeline after per-dispatch checks (eps validated at install by
  Float bitPattern equality — a one-ULP-off eps is rejected; span bounds
  already in rmsnorm(); float4 staging passes rmsVectorLoadGuards);
  installRmsGeneratedKernel (requires the model-gate record + full
  artifact identity; internal trial installer plus a public
  diagnostic-only wrapper mirroring diagnosticMatvec), uninstall,
  dedicated rmsGeneratedDispatchCount, engineIdentity gains
  "|rmsgen=<digest|none>".
- CLI — `rms-lower forge|verify|select`, `gemma-rmsgen-gate`,
  `gemma-profile --vectorized-loads`.
- tools/rms_lower/ — rms_op_model.py / model_rms_norm.py /
  export_rms_op.py (real .py input demonstrated: the captured op JSON in
  docs/evidence/rms-op-p10.json came from the .py, not hand-written).

Measured so far (this pass; the COMPONENT forge reads real GGUF weight
bytes with bounded reads only — the opt-in real-GGUF test suite and the
select/gate runs of course instantiate the full resident engine):
- Component forge on the real n = 2816 scaled op bound to
  blk.0.attn_norm.weight: incumbent rel-to-FP64 1.057e-06; ALL 5 emitted
  candidates (T1 baseline, T16/T32 × scalar/float4 staging) ACCEPTED as
  bitPattern-identical to the incumbent production kernel in both
  out-of-place and in-place modes (docs/evidence/rms-forge-p10.log). The
  serial-order-preservation structure reproduced the incumbent's bits
  exactly on these bounded inputs — synthetic activations plus the real
  weight bytes — which is BOUNDED evidence, not a universal FMA/real-
  activation proof; the artifact binds toolchain + executable bytes so
  any rebuild re-gates.
- Test suites (historical, pre-format-fix; current counts below): default
  585/0 (was 552; includes per-schedule hardware
  bit-identity, in-place staging safety, the n*4+4 threadgroup memory cap
  deriving an EMPTY schedule set, oracle hand case, artifact
  tamper/hardware rejection); real-GGUF opt-in 691/0 (was 653; includes
  the real-op forge bit-identity, trial-install dispatch-counter proof
  via the diagnostic wrapper, and one-ULP-off eps rejection with
  behavioral no-install verification).

Root long run (run 57271, one process, 893.76 s wall, executable SHA-256
012ff4d7a99be8a98d8e8843401cebf434ae81bcff91fb6b3eefa74b0e3ca890 with
sources and binary independently frozen): `rms-lower select` completed —
- Component screen (warmed, production vector count, pruning only):
  T32_float4 65.6 µs/call, T32_scalar 72.1 µs/call, T16_float4 98.9,
  T16_scalar 134.6; the two fastest reached the model gates.
- Model gates: BOTH candidates passed all 3 fixtures × 32 steps with
  worst measured numerical drift 0 (relative), exact greedy agreement,
  and finiteness — the full unchanged strict gate, not bit identity
  (component gates are the ones that check bits).
- Matched timing (3 paired repetitions, no sampled arms): T32_float4
  fixture rep sums 1,187,450 / 1,211,788 / 1,251,922 µs vs paired
  incumbent 1,387,527 / 1,293,425 / 1,292,686 — all three positive;
  T32_scalar gains 8.295 / 4.852 / 4.644%. Policy: SELECTED
  rmsgen_rms_T32_float4 at 6.31% median-ratio margin (> 2% threshold).
  This is a bounded 3-rep measurement on a shared Mac — evidence of a
  real paired win on this machine, NOT a universal or stable-occupancy
  claim.
- Persistence: artifact
  digest 65be14e564af17b20d4d9e2cb6caf208ab1d53d2d84b7c0ac085c16712696748
  (40,603 bytes) saved, reloaded, identity-validated, publicly installed;
  EN 32-step replay worst rel 0; 10,972 actual generated dispatches
  (52 positions × 211 bound scaled n = 2816 norms each). A separate root
  verification process (fresh model/hardware/toolchain/executable check)
  re-validated the artifact: VALID.
- Honest scope and cost: maxRSS 2,888,663,040 B; peak footprint
  31,156,126,720 B (~31.156 GB decimal) — the engine is NOT lightweight.
  The failed predecessor run (SIGSEGV in the final Bool-%@ formatter,
  951.31 s, log preserved) is NOT success: the crash hit at the FINAL log
  line, after that run had already reloaded and publicly installed its own
  artifact and replayed the gate — the run is still counted FAILED, and its
  artifact was NOT reused or re-bound after the formatter fix and
  executable change (the successful fixed run produced and bound a fresh
  artifact). The `--smoke-test` EXIT 0 run proves end-to-end
  plumbing only (2 steps, raw fixture), not quality or performance.
  Current test suites: default 589/0; historical real-GGUF opt-in 691/0
  before the format-only fix (not re-run since — the current code's
  gates were exercised by the full 32-step run and the smoke run above).
  The broader joint goal (general Python model compiler with competitive
  performance) remains UNMET; next hypothesis is the MoE routed-expert
  batch lowering (docs/MOE_BATCH_LOWERING_DESIGN.md, root-corrected,
  UNIMPLEMENTED).
- Evidence (docs/evidence/): rms-select-root-p10-fixed.log (full select,
  time/maxRSS/footprint; SHA-256 3f8aa9b9…), rms-artifact-root-p10-fixed.json
  (persisted artifact; file SHA-256 5d598925…, internal artifact digest
  65be14e5…), rms-select-root-p10-fixed-source-snapshot.json,
  rms-verify-root-p10-fixed.log (separate-process VALID, b90b3527…),
  tests-rmsfix-default-2.log (589/0, 1f2eb8db…), tests-rmslower-realgguf-2.log
  (historical pre-format-fix real 691/0, 129d8e17…),
  rms-profile-root-vl-p10.log (2-token unbatched VL diagnostic: RMS VL
  kernels 19.1% / 147.6 ms / 662 calls, expert gate_up+down 49.1% within
  matvec total 74% — ranking only, not occupancy attribution),
  rms-select-root-p10-FAILED-boolformat-segv.log +
  rms-resource-root-p10-failedrun-snapshot.jsonl (the failed 951.31 s
  predecessor run, preserved and labeled as failure).
