# Leviathan

<img src="docs/assets/leviathan-logo.png" alt="Leviathan project logo: a small whale emblem marking the Leviathan inference engine project" width="140">

Leviathan is being built as a local LLM serving engine for Apple Silicon
Macs. The intended engine profiles the machine and the workload, generates a
bounded set of legal optimization candidates, checks each one for numerical
agreement with an independent reference within a tolerance, measures real
device time, selects the best valid plan, and reuses compiled artifacts
keyed by model, hardware, and toolchain — with execution on the Metal GPU.
Serving is implemented for the real-model Gemma path as a bounded
localhost-only API driven by the validated plan artifact (see
`gemma-serve` below); the dense/ReLU engine has no serving API.

![Leviathan hero illustration: an original drawing of a giant whale devouring a kraken, symbolizing the Leviathan engine swallowing workloads](docs/assets/leviathan-hero.png)

**What runs today** is the verified foundation of that goal, honestly
scoped: a small inference compiler for `dense`/`relu` graphs with real
Metal GPU execution, integrity-checked compiled artifacts, and two
independent CPU reference implementations. LeviathanForge additionally
imports a restricted PyTorch export (`nn.Sequential` of `nn.Linear`/`nn.ReLU`)
with **real parameter weights** — validated, hashed, parity-checked
against PyTorch (measured 2.2e-08) — and automatically generates, gates,
and selects GPU pass candidates with a reusable measured plan cache; see
[docs/FORGE.md](docs/FORGE.md). The example weights are random
initializations, not a trained model.

**Real-model Gemma 4 26B-A4B (QAT Q4_0) now runs end-to-end through the
Forge pipeline** on an M1 Max — the original GGUF-backed tokenizer
(bounded 13-case exact-ID parity vs an independent Rust oracle),
bounded greedy generation, and the model.py capture path (a bounded,
trusted stage-list DSL — honestly NOT arbitrary Python/torch tracing):

```bash
# 0. Build (release):
swift build -c release

# 1. Capture: user's trusted restricted model.py DSL -> graph JSON
.build/release/leviathan-cli gemma-config-export --file <gguf> --out /tmp/cfg.json
python3 tools/forge_gemma/export_gemma_graph.py examples/gemma_model.py /tmp/cfg.json /tmp/graph.json

# 2. Forge: validate against the GGUF -> typed SSA lowering -> full-logit
#    correctness gates (vs the original runner forward — a separate
#    execution path driving the same validated kernel primitives) ->
#    measured candidates (wall-median objective, baseline fallback) ->
#    plan artifact embedding the lowered executable schedule + digests:
.build/release/leviathan-cli gemma-forge --file <gguf> --graph /tmp/graph.json \
    --plan /tmp/plan.json --prompt-ids 2,9259 --gen-tokens 8

# 3. Generation + GPU-time profile. gemma-generate also accepts
#    --plan/--graph (execute the persisted SSA artifact; the PRIMARY
#    measured path), --repeat N (in-process warm repetitions with KV
#    reset and cross-repetition determinism asserted), --gate
#    (optional untimed reference verification — OFF for cold
#    measurements because the replay warms all model resources),
#    --bos-align (warm protocol matched to Ollama's proven BOS-only
#    KV-prefix reuse: untimed [BOS] prefill, timed one-token
#    continuation; requires --plan and a 2-token prompt), and
#    --breakdown (opt-in per-decode-step wall/GPU split from runner
#    primitive counters: encode / flush-wait (overlaps GPU) / copies /
#    residual). Forge with --gen-tokens 32 tries candidates in the
#    short-selection timing order and selects the FIRST that passes
#    the full requested-generation gate, recording measured
#    rejections in the plan (strict gates unchanged). See
#    tools/baseline/{ollama_fairdecode.py,leviathan_fairdecode.sh} for
#    the aligned 32-token comparison harnesses:
.build/release/leviathan-cli gemma-generate --file <gguf> --text "Hello" --chat --max-tokens 8
.build/release/leviathan-cli gemma-profile --file <gguf> --q4-tiled --rms-tiled --f32-tiled

# 4. Optional bounded SSD expert mode: non-expert weights compacted to
#    a ~1.58 GB GPU buffer, expert slices streamed through an LRU slot
#    cache within the byte budget (same gated candidates; identity and
#    plan record the budget):
.build/release/leviathan-cli gemma-forge --file <gguf> --graph /tmp/graph.json \
    --plan /tmp/cache-plan.json --text "Hello" --gen-tokens 8 \
    --expert-cache-budget 1000000000
.build/release/leviathan-cli gemma-expert-cache-verify --file <gguf> \
    --graph /tmp/graph.json   # forced eviction/miss/hit/in-flight gates

# 5. Bounded localhost serving of the validated plan artifact (no
#    runtime agents/Python; one POST /generate at a time + GET /health):
.build/release/leviathan-cli gemma-serve --file <gguf> --graph /tmp/graph.json \
    --plan /tmp/cache-plan.json --port 8377 --expert-cache-budget 1000000000 \
    --gate-first-request
curl -s http://127.0.0.1:8377/generate -H 'Content-Type: application/json' \
     -d '{"text":"Hello","max_tokens":8}'
```

Resources and architecture-diagram sections below describe the
DENSE/RELU engine (the Gemma path has its own scoped limits above).
Measured on this M1 Max (2026-10-02; raw logs under
[docs/evidence/](docs/evidence/)): the graph executor matches the
original hand-written runner.forward — a separate execution path that
drives the same validated kernel primitives, not a mathematically
independent full-model implementation — bit-identically on the
per-kernel path and within rel 2.5e-05 on the selected optimized pass
(all 262,144 logits, gated per decode step; the gate's reference is
the runner's plain forward path, which shares the same kernel
primitives — the independent numerical checks are the per-kernel FP64
oracle component gates), with exact greedy
next-token equality over the tested 8-token workload
(" upon-the-sands-of" — also identical TEXT with matching token counts
to the same-GGUF Ollama 0.23.0 raw greedy baseline; the baseline API
exposes no internal IDs, so that agreement is text+count, and no
global no-quality-loss claim is made; pass 7 extends the tested
workload to 32 tokens). Quantized kernels carry their
own independent FP64 component gates. HISTORICAL passes 2-6 selected
cross-op command-buffer batching (1799 → 127 submissions/token) + a
tiled cooperative Q4_0 matvec and tiled RMSNorm: 754,092 → 122,565 µs
wall per token within one gated 8-token run (6.2× vs the per-kernel
baseline of that run) — SUPERSEDED at 32 tokens: the pass-7
full-length gate measured-rejected the three attempted tiled
candidates (accumulated cross-path drift 1.03-1.17e-04 over the strict
1e-4 gate; the gate stops after the first passing candidate, so not
all legal candidates were tested at 32) and selected segment batching
+ the bit-equal vectorized-load RMSNorm instead (-28% wall vs plain
batching at zero measured drift).
The gap to the same-GGUF Ollama baseline remains large; optimization
continues one measured, gated pass at a time. A matched warm/cold
ACCEPTANCE pass (tools/baseline/ollama_warmcold.py +
leviathan_warmcold.sh; same raw "Hello" greedy 8-token protocol,
byte-linked GGUF verified by sha256 both sides, repeated rounds,
sequential GPU, honest cold/page-cache/KV-prefix-reuse/client-TTFT
disclosures; final-run statistics only) measured on this machine:
Ollama 0.23.0 steady-warm server-reported ~14.6 ms/eval-token
(~68.4 tok/s median), cold TTFT 6.1-10.0 s (load 5.9-9.7 s) — vs
the selected resident SSA artifact's steady-warm decode medians
118.7-124.6 ms/token, warm prefill 198-214 ms, cold-process client
total 29.3-31.5 s (load 0.5 s; identity prep incl. a 14.4 GB GGUF
hash 11.1-11.4 s; first prefill 15.4-17.3 s, cause not causally
measured), peak memory footprint 16.47 GB, process peak RSS
2.13-2.32 GB. Decode accounting caveat: the Leviathan median is
over 7 inter-token steps of the 8-token output while Ollama reports
eval_count 8; warm prefill is additionally not like-for-like (Ollama
may reuse the KV prefix across identical warm requests — its warm
prompt_eval 0.02 s vs cold 0.14-0.18 s — while every Leviathan
repetition resets and re-prefills), and the machine carried
nontrivial background load during both harnesses (loadavg ~10-15,
recorded in the evidence). Both engines agree exactly on the greedy
text and count; ~8x per-token difference, reported as-is. The same
pass profiled the selected kernels per-kernel (Q4 matvec 32.4% /
Q6 LM head 23.8% / F32 router matvec 17.4% / tiled RMSNorm 12.3% /
attention 8.0% of GPU; per-kernel-mode attribution, batched
execution timing may differ) and implemented ONE new measured
candidate:
a tiled cooperative F32 router matvec (g4_f32_matvec_tiled + the
matvecTiledF32 candidate + FP64-oracle tests) — 20x faster per
kernel call (532 → 26 µs) and correct standalone (rel 1.8e-05), but
its stacking with the selected tiled Q4/RMS candidates reaches
worst rel 1.066e-04, just over the pre-registered 1e-4 full-logit
gate, so the Forge measured and did NOT select it (selection and
wall/token unchanged within run-to-run noise). Two earlier passes
are also implemented and measured:
an EXPLICIT bounded SSD expert mode (`--expert-cache-budget`; whole
expert families no longer GPU-resident — non-expert weights compact to
a 1,577,172,088 B buffer and expert slices stream through LRU slot
pools with checked transactional preads, active/in-flight pinning with
flush-before-reuse, and synchronous free-slots-only prefetch; the
per-kernel cache path in the dedicated verify run reproduced all
262,144 logits bit-exactly (worst rel 0.0) vs the resident snapshot,
while the 1 GB-artifact FORGE run's selected tiled/batched candidate
is gated within tolerance (rel ≤ 2.5e-05, exact greedy IDs) and
measured 334,232 µs wall / 102,395 µs GPU per token — a cross-run
contrast with the earlier 122,565 µs resident selection, not a
matched-memory comparison) and BOUNDED LOCALHOST SERVING from the
persisted plan artifact (`gemma-serve`: identity validation without
the workload binding — the candidate's medians were measured on the
plan's own profiling workload, disclosed at startup — serial greedy
generation executing the stored SSA schedule, strict HTTP bounds with
a 15 s absolute read deadline; 20 asserted negative probes incl. a
408 idle timeout, a gated first request, and exact-ID parity with the
resident CLI reference on the same input; ~330 ms per served decode
token in 1 GB cache mode). Remaining limits: the captured DSL is a
bounded Gemma-specific subset; prefill batching and kernel-count
reduction (1799 kernels/token) are not implemented; cache reads may
hit the OS page cache (no cold-SSD claim). See
[docs/GEMMA_PROGRESS.md](docs/GEMMA_PROGRESS.md) and
[docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md).

## Features

- **Metal GPU execution** — MSL compute kernels for dense and ReLU are
  compiled from source at run time (`device.makeLibrary(source:)`),
  dispatched on shared-memory buffers, and read back after a verified
  command-buffer completion. No silent CPU fallback: any Metal failure
  throws.
- **Validated graph IR** — structural and resource-limit validation of
  rank-2 float32 `[batch, features]` graphs (`dense {outDim}`, `relu`).
- **Deterministic compilation** — seeded lowering produces byte-identical
  artifacts with a SHA-256 integrity digest, full semantic re-validation on
  every load (bounds, offsets, buffer-chain topology, finite weights).
- **Independent CPU oracles** — Accelerate (vDSP) and a plain-Swift scalar
  backend; the `check` and test commands compare Metal outputs against both
  within 1e-4.
- **CLI** — `compile`, `run` (default backend `metal`; explicit
  `cpu-accelerate` / `cpu-scalar` for reference), `check` for the
  multi-batch correctness gate, `gguf-inspect` for memory-mapped GGUF
  metadata inspection, and `forge` — the LeviathanForge automatic
  pass-generation pipeline (restricted PyTorch export or graph JSON in;
  measured hardware profile, bounded legal candidates, correctness
  gates, GPU-time selection, validated plan cache reuse; see
  [docs/FORGE.md](docs/FORGE.md)).

## Installation

Requires an Apple Silicon Mac with Metal, macOS 13+, and Swift 6
(Command Line Tools suffice; no Xcode test frameworks needed).

```sh
git clone https://github.com/dhihm/leviathan.git
cd leviathan
swift build
```

The current Swift package has no third-party dependencies; building it
needs no downloads or credentials.

## Usage

```sh
# Correctness gate: Metal vs both CPU references across batches
swift run leviathan-cli check

# Compile a graph JSON into an integrity-checked artifact
swift run leviathan-cli compile --graph examples/demo-graph.json --out /tmp/demo.lvi

# Run on the GPU (default)
swift run leviathan-cli run --artifact /tmp/demo.lvi --batch 4

# Run an explicit CPU backend for reference
swift run leviathan-cli run --artifact /tmp/demo.lvi --batch 2 --backend cpu-scalar

# Test suite (exits nonzero on any failure)
swift run leviathan-tests
```

Example graph JSON (`dense` takes an output dimension; `relu` is a string):

```json
{
  "inputDim": 16,
  "ops": [
    { "dense": { "outDim": 32 } },
    "relu",
    { "dense": { "outDim": 32 } },
    "relu",
    { "dense": { "outDim": 8 } }
  ]
}
```

## Architecture

Two engine paths share the validated-IR + measured-plan core:

- **Dense/ReLU engine (runs)** — restricted PyTorch capture
  (`nn.Sequential` of `nn.Linear`/`nn.ReLU`, real weights, GPU-vs-
  PyTorch parity 2.2e-08) or graph JSON → validation → deterministic
  lowering → `.lvi` artifact (canonical bytes + SHA-256, semantic
  re-validation on load) → hardware/workload profiling → bounded
  legal candidates → correctness gate → measured selection → Metal
  GPU runtime → CPU-oracle comparison.
- **Real-model Gemma 4 path (runs, scoped)** — model.py capture DSL →
  graph JSON → validation against the actual GGUF → typed SSA
  schedule → gated, measured candidate selection → plan artifact
  (digests + embedded executable schedule) → Metal GPU runtime with
  optional bounded SSD expert cache → bounded localhost serving
  (`gemma-serve`).
- **Parameterized Q4_0 operator lowering (pass 9, one operator
  family)** — a Python op input (semantics/shape/layout/quant/
  precision metadata) → operation IR validation against the bound
  GGUF tensor → shape+hardware-derived schedule enumeration (thread
  mapping, row/block decomposition, reduction strategy, load width)
  → Metal source EMISSION from the parameters → independent CPU FP64
  oracle gate under the existing 1e-4 component contract → per-
  candidate teacher-forced model gates on EN/KR/raw fixtures → 3
  paired matched-timing repetitions (alternating order) against the
  incumbent → documented policy selection → artifact whose identity
  binds emitted source/SHA, codegen version, schedule, op digest,
  hardware, toolchain, GGUF content, and executable bytes →
  persisted execution that reloads, re-validates, and installs
  through the production path (component-only artifacts are not
  installable; fail-closed incumbent fallback). Deliberately ONE
  bound tensor (`blk.0.attn_q.weight`) — every other Q4 matvec keeps
  the incumbent kernel. See
  [docs/Q4_LOWERING_DESIGN.md](docs/Q4_LOWERING_DESIGN.md) and pass 9
  of [docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detail and
[docs/VALIDATION.md](docs/VALIDATION.md) for measured hardware
evidence. A pass-8 design audit of why the Forge candidate grid does
not generalize to arbitrary operators/hardware — and a staged
redesign recommendation — is in
[docs/ARCHITECTURE_AUDIT.md](docs/ARCHITECTURE_AUDIT.md). Status
assessment: automatic candidate selection exists, but the joint goal
of a general Python model compiler with competitive performance is
UNMET (see the audit).

## Resource limits

Enforced in code (`Sources/LeviathanCore/Limits.swift`), and re-enforced on
the GPU path before any device memory is touched:

| Limit | Value |
|---|---|
| Tensor shape | fixed rank-2 `[batch, features]` float32 |
| Tensor elements per tensor | 2^20 (4 MiB float32) |
| Single dimension | 4096 |
| Graph ops | 64 |
| Batch size | 256 |
| Weights blob | 16 MiB |
| Whole artifact file | 17 MiB |
| Graph JSON source file | 1 MiB |

## Limitations

Dense/ReLU engine (this section; the Gemma path below has its own
scoped limits and evidence docs):

- Two op types only (`dense`, `relu`); single input/output, strictly
  sequential graphs. No convolutions, attention, or control flow.
- Weights in this engine are synthetic (seeded) or restricted PyTorch
  exports; real quantized model weights run through the separate
  Gemma Forge path described above.
- The artifact digest provides integrity, not authentication or signing.
- Float32 only; backend comparisons use a 1e-4 absolute tolerance.
- No benchmark suite; timings printed by `check` are indicative only.
- No CI; the dense/ReLU engine has no serving API (the Gemma path's
  bounded localhost server is implemented — `gemma-serve`).

Gemma Forge path (measured, gated, still bounded):

- The model.py capture is a bounded, trusted stage-list DSL lowered to
  a typed SSA schedule — NOT arbitrary Python/torch tracing; unknown
  ops/orders/shapes are rejected before any GPU work.
- Correctness scope: per-step full-logit gates vs the original
  hand-written runner.forward (a separate execution path driving the
  same validated kernel primitives — not a mathematically independent
  full-model implementation), quantized-kernel FP64/Double component
  gates, and exact greedy equality over the TESTED workloads (raw
  "Hello" 8 tokens; EN chat 8; KO chat 10) — no global no-quality-loss
  claim.
- KV-aligned 32-token comparison (pass 7): warm decode median
  12788.6 ms / 31 steps = 2.424 tok/s vs the same-GGUF Ollama's
  66.59 eval tok/s (N=32 accounting includes Ollama's first generated
  token; Leviathan's 31 steps exclude it — different accountings,
  uncontrolled machine load, ~27x descriptively). A large gap
  remains; optimization continues one measured gated pass at a time.
- Remaining scope limits: greedy decoding only, single model, serial
  GPU, and the strict 1e-4 full-logit gate (checked over the full
  requested generation length) currently blocks the faster tiled
  candidates — including the 20x-faster-per-kernel tiled F32 router —
  from 32-token selection; the selected vectorized-load RMSNorm
  passes it with zero measured drift (bit-identical kernel). A
  teacher-forced layerwise diagnosis
  (`leviathan-cli gemma-diagnose`, opt-in, probe off by default with
  fail-closed bit-invariance checks) attributes the tiled-candidate
  gate failures to differences consistent with float32
  reduction-order reassociation on the tested fixture (single-Q4
  FP64 sample; no exclusive cause proven, gates unchanged); see
  [docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md) pass 8 and
  [docs/evidence/gemma-diagnose-hello32.log](docs/evidence/gemma-diagnose-hello32.log).
- The pass-9 parameterized Q4 lowering is ONE operator family
  (Q4_0 matvec) for ONE bound tensor — not a general operator
  compiler and not a whole-model candidate combination. Its
  generated kernels for T>1 use an explicitly reassociated
  deterministic float32 reduction (plain or Neumaier-compensated;
  Apple GPUs expose no usable FP64), gated against an independent
  CPU FP64 oracle at the same 1e-4 contract as the hand-written
  kernels and against the unchanged full-logit/exact-greedy model
  gates; the incumbent kernel remains the stable fallback unless a
  candidate passes every gate AND beats its own paired incumbent
  beyond the documented 2% policy threshold. Measured outcome and
  calibrated limits: pass 9 of
  [docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md).

Runnable end-to-end (measured configuration; ~47 min on M1 Max 32 GB,
peak ~31.2 GB, using the existing authorized GGUF — no new download):

```sh
GGUF="../models/gemma-4-26B-A4B-Q4_0/gemma-4-26B_q4_0-it.gguf"  # authorized checkpoint
# 1. Capture the Python op model (semantics/shape/layout/quant metadata)
python3 tools/q4_lower/export_q4_op.py tools/q4_lower/model_q4_attn.py /tmp/q4-op.json
# 2. One-process orchestration: component FP64 gates for every derived
#    candidate, per-candidate teacher-forced model gates on EN/KR chat
#    + raw Hello fixtures, 3 paired matched-timing repetitions, policy
#    selection, persistence, and a persisted-execution proof.
.build/release/leviathan-cli q4-lower select --op /tmp/q4-op.json \
    --gguf "$GGUF" --out /tmp/q4-artifact.json \
    --text-en "Hello, can you help me today?" \
    --text-kr "안녕하세요, 오늘 날씨는 어떤가요?" \
    --gen-tokens 32 --timed-tokens 31
# 3. Re-validate the persisted artifact against the GGUF
.build/release/leviathan-cli q4-lower verify --artifact /tmp/q4-artifact.json --gguf "$GGUF"
```

Measured on that run: 6 of 8 emitted candidates passed every fixture
gate (two float4 variants failed raw Hello at step 9 and were
excluded); the persisted execution replay counted 52 generated-kernel
dispatches on the bound tensor; the policy selected
`q4gen_matvec_T8_R1_f32_scalar` on a +3.9% median paired wall gain —
with one of its three paired repetitions SLOWER than the incumbent, so
no stable gain is claimed. One bound tensor out of the model's many Q4
matvec sites changed; the broader general-compiler goal remains UNMET.

## Roadmap

1. **First real model** — original Swift GGUF reader and Q4_0 dequantizing
   Metal kernels for [Gemma 4 26B-A4B](docs/MODEL_TARGET.md): exact hybrid
   local/global attention with unified KV and p-RoPE, RMSNorm, MoE routing
   (top-8 of 128) plus shared expert, weights kept quantized on the GPU.
   **Status 2026-10-02: DONE and superseded — the full model now runs
   through the Forge SSA artifact pipeline with the original
   GGUF-backed tokenizer (bounded 13-case Rust-oracle parity), bounded
   text generation, and per-step full-logit + greedy gates; see
   [docs/GEMMA_PROGRESS.md](docs/GEMMA_PROGRESS.md) and
   [docs/FORGE_PROGRESS.md](docs/FORGE_PROGRESS.md).**
2. **One measured automatic optimization decision** — a bounded candidate
   set (e.g. threadgroup widths 32/64/128) gated by correctness, then
   selected by median measured device time under a finite tuning budget.
   Best-of-tested, not a claim of global optimality.
3. **Loopback local serving** — localhost-only, bounded requests, after
   real GPU generation works.

Model weights are never committed to git; Apache-2.0 model files remain
under their own license and are not relicensed by this repository's MIT
terms.

## Artwork

The logo and hero images in `docs/assets/` are original generated Leviathan
artwork.

## License

[MIT](LICENSE) © 2026 dhihm. Model files, if later downloaded, are governed
by their upstream licenses and are not covered by this repository's MIT
grant.
