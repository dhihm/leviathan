# Phase 10 design: generated serial-order-preserving cooperative RMSNorm

Status: APPROVED by root with corrections (folded in below): profiling gets a
`--vectorized-loads` flag so the incumbent VL kernel is profileable; the head
norms include n = 512 (global layers) besides n = 256; the n/scaled family
dispatch performs PER-DISPATCH eps/span/alignment/alias/bounds checks before
routing; the T ladder is derived from the measured execution width and device
caps with the rule stated, not an unexplained fixed set. Strict gates (per-step full 262144-logit worst rel < 1e-4, exact
greedy argmax equality, finiteness, 32 steps, EN/KR chat + raw Hello) are
UNCHANGED. Policy threshold stays 2%. All work is original GLM-authored code in
this repository only.

Outcome (root run 57271, 2026-10-03 UTC): selected `rmsgen_rms_T32_float4`
at a 6.31% paired matched-timing median margin (all 3 repetitions positive),
worst measured 32-step drift 0 on all 3 fixtures, artifact 65be14e5…
verified in a separate root process and publicly installed (10,972 replayed
generated dispatches). Bounded 3-rep evidence on a shared Mac; the joint
competitive goal remains UNMET (see FORGE_PROGRESS.md Pass 10).

## 1. Problem

The production configuration is segment batching + `g4_rmsnorm_vl`
(bit-identical vectorized-load RMSNorm). `g4_rmsnorm_vl` still assigns ONE
THREAD PER VECTOR and reduces the sum of squares serially: at the hidden size
n = 2816 a single lane performs 2816 sequential FMA-class operations while the
SIMD lanes of the GPU sit idle. Pass 3's `g4_rmsnorm_tiled` fixed exactly this
with a cooperative tree reduction and was measured faster, but it REASSOCIATES
the addition order, drifted to 2.882e-04 worst rel, and was correctly REJECTED
by the strict full-logit gate (pass 7). The untested hypothesis from the
architecture audit: stage the input cooperatively (parallel, coalesced), reduce
SERIALLY in the ORIGINAL scalar order on one lane, then apply the scaling in
parallel — bit-identical arithmetic with cooperative memory traffic.

## 2. Proposed kernel structure (codegen "rmsgen-v1")

One threadgroup per vector, T threads (schedule parameter):

1. **Stage (parallel):** threads copy `x[vec*n ..< vec*n+n]` into threadgroup
   memory `threadgroup float staged[n]` — element copies, NOT squared values
   (squaring on staging would change the reduction's FMA contraction
   behavior). Staging granularity is a schedule parameter: `scalar` or
   `float4` (float4 requires n % 4 == 0; every actual norm width here
   qualifies: 2816, 256).
2. `threadgroup_barrier`.
3. **Reduce (serial, lane 0 only):** `for (i = 0; i < n; i++) ms += staged[i] * staged[i];`
   — the exact incumbent expression, in the exact incumbent order, on exact
   copies of the incumbent input values. `ms = ms / n + eps; inv = 1/sqrt(ms);`
   written to threadgroup.
4. `threadgroup_barrier`.
5. **Apply (parallel, strided):** `for (i = lid; i < n; i += T) yv[i] = staged[i] * inv * (scaled ? w[i] : 1.0f);`
   — reads the staged copy (no second device round-trip). Per-element output
   arithmetic is independent across elements and identical to the incumbent,
   so parallelism here cannot change any output bit.

**Bit-identity claim and its limit.** Steps 3 and 5 reproduce the incumbent's
arithmetic one-for-one. The residual risk is compiler FMA contraction
differing between the incumbent kernel and the generated kernel (e.g. if the
incumbent's `ms += xv[i]*xv[i]` contracts to FMA from device memory but the
generated form from threadgroup memory does not). This is NOT assumed away:
a hardware component test asserts bitPattern equality of full outputs
against the production incumbent kernel (-0.0/+0.0 sign flips count as
differences), on deterministic synthetic activations for the unbound op and
on real weight bytes + synthetic activations for the bound op, in both
out-of-place and in-place modes; any candidate that differs in any bit
pattern on any element is rejected at the component stage and never reaches
the model gate. The evidence is BOUNDED to those inputs and shapes — it is
not a universal proof over all real activations or future toolchains (the
artifact binds the toolchain and executable bytes, so a rebuild re-gates).
If ALL candidates fail bit-equality, the phase reports that honestly and
falls back (the unchanged 1e-4 model gate applies to any survivor
regardless).

**In-place x==y safety.** All model call sites with x == y (q_norm, k_norm,
v_norm, moe/share in-place sites) are safe by construction: the FULL vector is
staged to threadgroup before any device write occurs, and the two barriers
order staging before apply. The apply phase reads only `staged[i]` and `w[i]`,
never `x` again, so no thread can observe another thread's output write.
Distinct vectors are disjoint. `n*4` bytes of threadgroup memory are required
(2816 → 11.3 KB; 256 → 1 KB), checked against
`MTLDevice.maxThreadgroupMemoryLength` at derivation time; oversize n is
illegal for this family and falls back.

## 3. Operation IR and Python input

`tools/rms_lower/rms_op_model.py` (mirrors `tools/q4_lower/q4_op_model.py`):
`RMSNormOp(n, vectors, scaled, eps, tensor)` → JSON schema 1 with semantics
`"y[i] = x[i] * inv_rms(x) * w[i]"`, layout `"row-major-f32-vectors"`,
quant `"f32"`, reduction `"serial-order-i0..n"`, precision `"f32"`. The Swift
IR validates: n > 0, vectors > 0, eps matches the GGUF config's rmsNormEps,
and — when scaled — the named tensor exists in the GGUF with type f32 and
length exactly n*4. A capture script exports the dominant production site
(`blk.0.attn_norm.weight`, n = 2816, scaled) as the representative op.

## 4. Schedule derivation (shape + hardware)

Parameters: threadsPerVector T (power of two) and staging ∈ {scalar, float4}.
T is DERIVED from measured hardware, not a fixed list: legal powers of two are
those with `T <= maxTotalThreadsPerThreadgroup`, `T <= n`, and
`n * 4 <= maxThreadgroupMemoryLength` (the staging array bound); they are
ranked by |log2(T) − log2(threadExecutionWidth)| — proximity to the SIMD
execution width, the occupancy-relevant measured quantity — and the TWO
nearest survive (the rule, stated: cooperative width near the execution width
keeps whole SIMD lanes busy without oversubscribing the quads). Staging
`float4` additionally requires `n % 4 == 0` (all actual widths qualify).
Baseline candidate `T = 1, scalar` reproduces the incumbent serial semantics
trivially (sanity arm, not expected to win). Pruning: at most 6 candidates.
`RMSHardwareProfile.profile()` records maxTotalThreadsPerThreadgroup,
threadExecutionWidth, maxThreadgroupMemoryLength, recommendedMaxWorkingSetSize
from the live device and its compiled incumbent pipeline.

## 5. Binding scope (honest bound)

Like phase 9 this binds ONE operation family, but dispatch is keyed on the
VALIDATED op shape rather than a single tensor: the installed generated kernel
handles rmsnorm calls with `n == bound n` and the same `scaled` flag, and ONLY
after PER-DISPATCH checks in `rmsnorm()`: the runtime eps equals the op's
validated eps (config rmsNormEps), the x/y span bounds already enforced by
`rmsnorm()` cover this call, a float4-staging schedule additionally passes
`rmsVectorLoadGuards` (n multiple of 4, weight offset 16-byte aligned when
scaled), and the in-place x == y case is legal by the staging/barrier
construction above (documented, not re-derived per dispatch). The weight
offset stays a runtime argument so every n = 2816 scaled norm site (attn/
post-attn/ffn/router/pre-ffw2/final norms) can route through it. The head
norms (n = 256 local, n = 512 GLOBAL head dim — both exist in this model) are
a SECOND op instance only if the first succeeds within budget. No claim of
general norm compilation is made.

## 6. Pipeline (reuses the phase-9 machinery pattern)

- `RMSLowering.swift`: op IR + validate/validateAgainst(GGUF), schedule
  deriver, kernel emitter (deterministic, digest-bound), CPU FP64 oracle
  (Double sum of squares in serial order — the oracle is the ground truth the
  1e-4 component contract measures against, incumbent included), component
  inputs (real GGUF bytes via FileHandle pread of the norm weight + synthetic
  activations), emitted-kernel compile, forge (component phase: FP64 oracle
  distance < 1e-4 AND hardware bit-equality vs `g4_rmsnorm_vl`), artifact
  with body digest binding emitted source SHA, codegen version, schedule, op
  digest, hardware profile, toolchain, GGUF digest, gate records.
- Selection: identical to phase 9 — per-candidate teacher-forced gates on the
  same 3 canonical chat fixtures (32 steps, exact greedy history replay,
  per-step full-logit 1e-4 + argmax + finite), then 3 paired matched-timing
  repetitions with alternating order, candidate judged on ITS OWN paired
  incumbent ratio, 2% median policy, `resetIncumbentConfig()` (reset THEN
  re-apply vectorizedLoads + segmentBatching, guarded) before every arm,
  fail-closed incumbent fallback, persisted-artifact reload/verify/install
  with dispatch-counter proof, engineIdentity gains `|rmsgen=<digest|none>`.
- Bit equality is defined STRICTLY: every output Float's `bitPattern` must
  match the incumbent production kernel's output on the same inputs, so
  -0.0/+0.0 sign flips count as differences (numerical `==` would accept
  them). Both the out-of-place and the in-place (x == y) mode are gated on
  hardware; the incumbent's own in-place stability is verified first (it
  defines the reference).
- Component-first funnel per root instruction: candidates must pass the
  component bit-equality + oracle filter BEFORE any model gate; the
  survivors then run a WARMED component GPU timing screen (median
  GPU-busy µs/call over 30 calls after 5 warmups, buffers allocated once —
  ranking/pruning only, never a win claim), and only the TWO FASTEST
  proceed to full-model gates and matched timing. No repeat of the 8-arm
  47-minute search.

## 7. Measurement plan (bounded, ordered)

1. **Profiling first (short):** `gemma-profile --vectorized-loads` (new flag,
   mirrors the other kernel flags) per-kernel run on the real GGUF to
   re-confirm `g4_rmsnorm_vl`'s GPU share under the incumbent kernels.
   Caveat to disclose: profile mode runs unbatched per-kernel dispatch for
   exact attribution, so absolute shares differ from batched production;
   ranking only, labeled as such. Elapsed GPU fraction is NOT occupancy.
2. **Component forge (minutes):** all candidates on real n = 2816 weights +
   representative activations; bit-equality + 1e-4 oracle records; a quick
   component wall-time screen (single-vector microbenchmark, used ONLY to
   prune obviously non-competitive schedules, never to claim wins).
3. **Model gates + paired timing (the only long run):** ≤ 2 candidates ×
   3 fixtures × (gate + 3 paired repeats). Budget: one process, target
   ≤ ~40 min wall, footprint comparable to pass 9 (~31 GB peak, near the
   32 GB hardware RAM capacity — disclosed, no causal claim). Resource
   numbers recorded via /usr/bin/time as before.
4. Report honestly: sign-changing repetitions disclosed; no stable-gain claim
   without medians over repetitions; if no candidate clears 2%, the incumbent
   stays and the phase reports the negative result.

## 8. Tests

Emitter determinism/distinctness; deriver legality (threadgroup memory bound,
T cap); oracle hand-computed small case; hardware bit-equality vs incumbent
for every bound shape (the load-bearing test); non-finite input rejection;
artifact tamper rejection; hardware-mismatch rejection; batching-regression
guard (submitted buffers vs dispatch counters after reset+re-apply, mirroring
the phase-9 test); generated-dispatch counter increments only on bound calls.

## 9. Not claimed

No arbitrary-kernel-invention claim; no general Python compiler claim; no
performance claim from microbenchmarks; component oracle shares the
incumbent's dequantization philosophy where applicable (here inputs are raw
f32, so the FP64 oracle is independent arithmetic on the same bytes —
disclosed); one norm-shape family bound per artifact instance.
