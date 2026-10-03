# MoE Routed-Expert Batch Lowering — Design (UNIMPLEMENTED, for root review)

Status: **DESIGN ONLY. Not implemented, not measured, no performance claim.**
Authored while the frozen engine `012ff4d7…` ran the strict 32-step RMS select
(completed: run 57271 EXIT 0, selected T32_float4 at 6.31% policy; see
FORGE_PROGRESS.md Pass 10). No source, binary, or benchmark was touched by
this document. Scope is bounded to ONE MoE region — this is not a general
Python compiler. Root-approved corrections to fold in during implementation:
21 (not 24−⌈8/E⌉·3) fewer dispatches per layer/token at E8, verified from
actual supported calls; scratch is GU 44 KiB + ACT 22 KiB + DOWN 88 KiB =
154 KiB (DOWN keeps all 8 slots; GU/ACT sized E per chunk) plus offset
arrays; FMA preservation will be checked by component bit gates (never
disabled by flag); batch size E derives from the global scratch budget and
actual device buffer limits, not threadgroup memory; E ∈ {1, 2, 4, 8} all
legal candidates from the first iteration, with at most 2 timed finalists.

## 1. Hypothesis

In `Gemma4.forwardLayer` (Sources/LeviathanCore/Gemma4.swift:2129-2153) the
routed loop executes, **serially per slot** (8 slots, top-8 of 128 experts):

1. `expertMatvec`/`matvec` — `ffn_gate_up_exps.weight` slice → `bufGU`
2. `geluUp(gu: bufGU, y: bufAct, n: e)` → `bufAct`
3. `expertMatvec`/`matvec` — `ffn_down_exps.weight` slice → `bufT2`
4. `axpy(x: bufT2, y: bufT3, c: weights[slot], n: hidden)`

The eight expert computations are **data-independent after
`pre_ffw_norm_2`** (the shared normed input `bufH` is read-only for all
slots); the only cross-expert dependency is the final weighted merge into the
accumulator `bufT3`. Today each of the 3 math ops issues 8 dispatches per
layer per token (24 total) with tiny per-call row parallelism
(gate|up output = 2·704 = 1408 rows; down output = 2816 rows).

**Hypothesis (unmeasured):** batching the 3 ops across E ∈ {2,4,8} experts
into dedicated scratch slices raises row parallelism and cuts kernel count
predictably — at E=8, 3 batched dispatches replace 24 (**21 fewer per
layer/token**, verified at implementation time from the actual supported
call sequence, not derived from a closed-form expression), while
**keeping all 8 original ordered `axpy` calls** so the
accumulation arithmetic order into `bufT3` is bit-unchanged. Fewer dispatches
is an arithmetic count, **not** a speedup claim; no occupancy or bandwidth
claim is made without hardware counters.

## 2. Numerical-equivalence contract (the hard gate)

The incumbent production Q4 matvec preserves an exact serial FP32 sequence:
low/high nibble interleaved sum, `d * (q - 8) * x` per term, variable-trip
loops (tails). A generated batched kernel MUST reproduce that **same FP32
operation sequence per output row**, not merely the same C expression —
"same source expression" is not a compiler-bit-proof. Requirements:

- Per-row accumulation order identical to the scalar kernel: same nibble
  order, same multiply/add chaining, no NEW reassociation, same tail
  handling (variable-loop semantics preserved under E-chunking, including
  partial last chunk when topK % E ≠ 0 — here topK=8 divides all candidate
  E, but the generated schedule must still be correct for dynamic chunk
  tails). The incumbent kernel's compiled FMA contraction WILL BE checked
  empirically: we will not forbid FMA with a compiler flag; we will demand
  bitPattern-level component gates on real expert weights + bounded
  activations, and only bit-equal candidates will advance.
- `geluUp` math identical: reuse the incumbent helper's exact expression
  from our own source (provenance within this repo only; no third-party
  source). Generated MSL may compile alongside our `shaderSource` so the
  common function dependency is the literal incumbent function; the
  artifact then hashes the full compiled source.
- **Merge untouched:** the 8 ordered `axpy(x: bufT2Slot, y: bufT3,
  c: weights[slot])` calls run in original slot order reading each slot's
  batched-down output via offset. No norm/merge fusion, no reordering, no
  scale folding. This avoids any merge reassociation risk entirely.

## 3. Region op-IR (Python frontend, parameterized — not a global flag)

A bounded Python op/region IR expresses the loop as dataflow:

```
Region "moe-routed-e{E}" (per layer, per token):
  in:  xN        = normed hidden input (read-only, shared)
       experts   = ordered list[8] of expert ids (runtime values)
       weights   = ordered list[8] of merge coefficients
       GU, DOWN  = 3D Q4_0 GGUF sliced weight tensors (zero-copy base)
  ops:
    for chunk c in chunks(experts, E):            # control node, static E
      GUout[c]  = batched_q4_matvec(GU, experts[c], xN)      # E·1408 rows
      ACT[c]    = gelu_up(GUout[c])                          # elementwise
      DNout[c]  = batched_q4_matvec(DOWN, experts[c], ACT[c])# E·2816 rows
    # ordered merge stays OUTSIDE the region as 8 incumbent axpy nodes
  out: DNout slots (8 × hidden)
```

Every control (E, chunk count, thread geometry) must genuinely change the
generated grid mapping or resource lifetimes; the codegen is parameterized
from region shape/control, never a single global boolean flag enumerating
hand-written variants.

## 4. Hardware-derived legality + schedule

- Batch size E = largest value in {1,2,4,8} such that threadgroup scratch
  and thread geometry are legal against **actual device caps** read at
  runtime (`maxThreadgroupMemoryLength`, threadgroup size limits, buffer
  count), with E=1 degenerating to the incumbent loop (sanity baseline).
- Thread mapping: one threadgroup per output row-tile; expert index arrives
  via a small host-initialized offset/index array bound as a buffer
  (validated: indices are distinct legal route values in [0,128), host
  top-8 sorted order, arbitrary at runtime).
- Weight access stays **zero-copy on the original resident GGUF buffer**:
  per-expert slice via `expert * guStride` / `expert * dnStride` exactly as
  the incumbent computes them; both GU and down offsets/strides are
  checked against per-tensor complete extents before dispatch.
- Bounds/alias checks: output bounds, unaligned offset rejection, and
  scratch non-aliasing with any original mutable buffer (`bufT3`, `bufX`,
  …) are validated host-side per dispatch.

## 5. Scratch budget (~160 KiB, no weight copies)

Separate batched scratch, sized at the Gemma-4 26B-A4B point:
`hidden = 2816`, `expertIntermediate e = 704`, `topK = 8`, `E = 8`:

- GU scratch:  `8 × 2 × 704 × 4 B = 45 056 B` (44 KiB, FP32; chunk-of-E rows
  per batched call, full 8 retained across chunks)
- ACT scratch: `8 × 704 × 4 B = 22 528 B` (22 KiB)
- Down scratch: `8 × 2816 × 4 B = 90 112 B` (88 KiB; ALL 8 slots retained —
  the 8 ordered axpy merges read their slot via validated byte offset)
- Expert id array `8 × UInt32` (32 B) + GU/DN offset arrays `2 × 8 × UInt64`
  (128 B), host-initialized from validated route indices

Total ≈ **154 KiB + 160 B ≤ ~160 KiB budget**, three separate allocations,
owned once per engine (not per layer/token), no weight duplication of any
kind (the later Q4 byte-plane packing idea is explicitly out of scope here;
if explored later it stays under a 64 MiB family cap — not a 14 GB copy).
The batch size E is derived from this GLOBAL scratch budget and the
device's actual `recommendedMaxWorkingSetSize`/`maxBufferLength` limits —
NOT from threadgroup memory, which these one-thread-per-row kernels do not
use.

## 6. Fallback and cache interaction

- The generated batched path is used **only** in the resident-GGUF
  (non-SSD) mode that today takes the plain `matvec(extraOffset:)` branch
  (Gemma4.swift:2140-2149).
- In `expertCache`/SSD mode the incumbent loop is retained unchanged as a
  **guarded fallback**: batching over streamed experts would risk unsafe
  unpinning / read-before-completion. No claim that the new batch supports
  streaming — that is future work.
- `prefetchExperts` and `endExpertIteration` semantics stay on the fallback
  path; the resident path has no cache interaction to preserve.

## 7. Artifact identity + install + gates (reuse existing machinery)

Persisted artifact binds: region IR hash, layout/schedule parameters,
generated MSL hash, full compiled source hash (including shared function
dependency), model identity, toolchain, hardware caps snapshot, executable
digest, and gate policy — same identity discipline as the RMS lowering
(`rmsgen-v1`-style codegen tag, e.g. `moegen-v1`). Install is public and
gated by the **full-model gate only**; all runtime passes are autogenerated
and agent-free.

Gates, in order (all unchanged from existing policy):

1. **Component bit-gates**: `bitPattern`-level equality of batched vs
   incumbent outputs on real expert weights + bounded activations, for GU,
   gelu, and down separately, across E ∈ {1,2,4,8} and chunk tails.
2. **Independent FP64 oracle per operator** from the original Q4 bytes
   (disclosed bounded equivalence, shared full-model primitives).
3. **Full strict 32-step select**: 3 fixtures, 262144 logits, worst
   rel < 1e-4, exact greedy match, finite check, per step — thresholds
   NOT loosened.
4. Component GPU timing screen (pruning only), then matched AB/BA paired
   timing with the 2% policy threshold — production protocol unchanged,
   run by root sequentially after code + tests + short smoke
   (`--smoke-test`, plumbing-only).

## 8. Integration touchpoints (when implemented)

- `Gemma4.swift:2129-2153` — routed loop gains a resident-mode branch
  emitting batched dispatches + 8 ordered axpys reading down-scratch slots.
- `RMSLowering.swift`-style new `MoELowering.swift` — forge/select/persist
  pipeline mirroring the RMS one (component forge, screens, gates, paired
  timing, artifact, install, replay proof).
- `leviathan-cli/main.swift` — `moe-lower select` with the same
  `--smoke-test` semantics.
- `leviathan-tests/main.swift` — component bit-equality + format/log
  regressions calling actual production helpers.

## 9. Measurement plan (bounded, root-executed)

1. Default test suite green, real-GGUF suite per root's discretion.
2. `--smoke-test` EXIT 0 through persisted replay proof line.
3. Component screens at E ∈ {1, 2, 4, 8} — E=1 is a sanity baseline and
   ALL of {2,4,8} are legal candidates from the first iteration (no
   artificial ≤4 blast-radius bound, no repeated low-yield runs); screen
   is bounded to at most the 2 fastest actual candidates for timing.
4. One strict 32-step 3-fixture select: full gates + paired timing vs
   incumbent, 2% policy. No sampled timed arms; no new run types.
5. If batching does not win, the next queued hypothesis is the significant
   Q4 byte-plane/row-tile packing for the dominant expert 2D slice
   (hypothesis only; resource cap 64 MiB family).

## 10. Open questions for root review

1. RESOLVED by root: E ∈ {1, 2, 4, 8} are all legal first-iteration
   candidates (E=1 sanity); no ≤4 staging. Folded into §9.
2. RESOLVED by root: persistent engine-owned scratch allocation,
   three separate buffers (GU/ACT/DOWN), one-time.
3. Confirm: profile basis is the historical scalar 2-token family profile
   (`ffn_gate_up_exps` 202 ms / 37.7%, `down` 54 ms / 10.1%) — the newer
   VL profile (Q4 62.4%, RMS 19.1%) is unbatched-GPU ranking, not
   production occupancy; no claim is drawn from either without the
   matched run.

---
*Authored by GLM (same authorized SID), English, original content, own
source only. No agents, no external services, no third-party material.
RMS 32-step result: run 57271 EXIT 0 against frozen 012ff4d7… — selected
rmsgen_rms_T32_float4 at 6.31% policy margin, artifact 65be14e5…
verified; see FORGE_PROGRESS.md Pass 10.*
