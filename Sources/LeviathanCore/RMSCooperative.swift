// Exact-order cooperative RMSNorm lowering (public enum source of
// truth): the measured per-kernel profile attributes 41.77% of GPU time
// to RMSNorm (331 calls/token on the intrusive strict-bit-exact replay),
// where the incumbent kernel runs ONE thread per vector with serial
// reads and a serial apply — poor coalescing and poor GPU lane use.
// This kernel keeps the arithmetic EXACTLY the incumbent g4_rmsnorm_vl
// sequence (no tree/reassociated reduction, identical eps/sqrt/inv and
// per-element x*inv*(scale) association) while making the memory
// movement cooperative:
//
//   phase 1 — ALL threads cooperatively stage the input vector into
//   dynamically sized threadgroup memory as float4s;
//   barrier;
//   phase 2 — lane 0 sums the staged float4 components in the EXACT
//   incumbent order (ms += x^2; ms += y^2; ms += z^2; ms += w^2),
//   computes ms/n + eps and 1/sqrt(ms), and stores inv in the extra
//   staged slot at index n;
//   barrier;
//   phase 3 — ALL threads apply per element in parallel, reading ONLY
//   the staged copy (never x), so in-place x == y is safe by
//   construction.
//
// The test suite asserts EXACT bit equality against the original kernel
// on hardware for the tested geometries; bit equality on hardware is a
// measured property of the compiled pair, not a proof for all inputs.
// The runner dispatches one threadgroup per vector; the thread count is
// HARDWARE-DERIVED from the compiled pipeline's threadExecutionWidth —
// never a GPU-name guess.

import Foundation

public enum RMSCooperative {
    /// Metal function name (also the dispatch accounting name).
    public static let kernelName = "g4_rmsnorm_coop"

    /// Version of the thread-count/threadgroup-memory DERIVATION in this
    /// enum (not the source text — that is bound separately by SHA-256).
    /// Bumped whenever the schedule derivation changes so engine
    /// identities can never interchange schedules across versions.
    public static let scheduleVersion = 1

    /// Largest supported vector width (dynamic threadgroup staging of
    /// n+4 floats; the model's widest norm is 2816).
    public static let maxN = 8192

    /// Standalone MSL (its own library — never part of the runner's
    /// shared shaderSource, never stored in the shared pipelines
    /// dictionary, so the pipeline identity key list is unchanged).
    public static let source = """
#include <metal_stdlib>
using namespace metal;

// Exact-order cooperative RMSNorm. Same binding ABI as g4_rmsnorm /
// g4_rmsnorm_vl (buffers 0..6); the dynamically sized threadgroup array
// `staged` (n + 4 floats, 16-byte aligned: n is a multiple of 4) is set
// per dispatch via setThreadgroupMemoryLength(index: 0). One threadgroup
// per vector. Phase 1 stages the input; barrier; lane 0 reduces the
// staged float4s in the EXACT incumbent component order and stores inv
// at staged[n]; barrier; phase 3 applies per element in parallel reading
// ONLY the staged copy, so in-place x == y is safe: no thread reads x
// after any write to y (all reads come from threadgroup memory).
kernel void g4_rmsnorm_coop(constant const uint2& p [[buffer(0)]],
                            constant const float& eps [[buffer(1)]],
                            constant const int& withScale [[buffer(2)]],
                            constant const ulong& wOff [[buffer(3)]],
                            device const uchar* wbase [[buffer(4)]],
                            device const float* x [[buffer(5)]],
                            device float* y [[buffer(6)]],
                            threadgroup float* staged [[threadgroup(0)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tcount [[threads_per_threadgroup]]) {
    if (tgid >= p.y) { return; }
    uint n = p.x;
    device const float* xv = x + (ulong)tgid * n;
    // Phase 1: cooperative float4 staging of the full input vector.
    for (uint i = 4u * tid; i < n; i += 4u * tcount) {
        *(threadgroup float4*)(staged + i) = *(device const float4*)(xv + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Phase 2: lane 0 serial reduction — EXACTLY the incumbent
    // g4_rmsnorm_vl order (no tree, no reassociation).
    if (tid == 0u) {
        float ms = 0.0f;
        for (uint i = 0; i < n; i += 4u) {
            float4 v = *(const threadgroup float4*)(staged + i);
            ms += v.x * v.x;
            ms += v.y * v.y;
            ms += v.z * v.z;
            ms += v.w * v.w;
        }
        ms = ms / float(n) + eps;
        staged[n] = 1.0f / sqrt(ms);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Phase 3: parallel per-element apply from the staged copy only,
    // with the identical per-element association.
    float inv = staged[n];
    device float* yv = y + (ulong)tgid * n;
    if (withScale != 0) {
        device const float* w = (device const float*)(wbase + wOff);
        for (uint i = tid; i < n; i += tcount) {
            yv[i] = staged[i] * inv * w[i];
        }
    } else {
        for (uint i = tid; i < n; i += tcount) {
            yv[i] = staged[i] * inv;
        }
    }
}
"""

    /// SHA-256 of the actual kernel source, bound into engineIdentity
    /// alongside the schedule derivation version. Computed fresh on
    /// every call — engineIdentity runs at initialization/validation
    /// time, not per token.
    public static func sourceSha256() -> String {
        SHA256.hex(SHA256.hash(Array(source.utf8)))
    }

    /// Geometry eligibility: positive width, multiple of 4 (float4
    /// staging + 16-byte alignment), bounded by maxN (dynamic
    /// threadgroup memory budget). Buffers spans/offsets and the scaled
    /// weight span are validated by the caller (rmsnorm) before the
    /// dispatch hook; scaled weight-offset alignment reuses the
    /// production rmsVectorLoadGuards predicate in the hook itself.
    public static func eligible(n: Int) -> Bool {
        n > 0 && n % 4 == 0 && n <= maxN
    }

    /// Hardware-derived thread count: 2x the ACTUAL compiled pipeline's
    /// threadExecutionWidth, bounded by the pipeline's own maximum
    /// threads per threadgroup and by n/4 (one float4 per thread covers
    /// the staging loop). HARDENED: invalid inputs (n <= 0,
    /// executionWidth <= 0, maxThreads <= 0) return 0 — the caller MUST
    /// guard threads > 0 before dispatch; the doubling is overflow-safe
    /// (a doubled width that would overflow Int or exceed maxThreads
    /// clamps straight to maxThreads, always the tighter real pipeline
    /// cap). Never a GPU-name guess; on M1 Max (width 32, maxThreads
    /// 1024) n >= 256 still derives exactly 64 threads.
    public static func threadCount(n: Int, executionWidth: Int, maxThreads: Int) -> Int {
        guard n > 0, executionWidth > 0, maxThreads > 0 else { return 0 }
        let (doubled, over) = executionWidth.multipliedReportingOverflow(by: 2)
        var threads = (over || doubled > maxThreads) ? maxThreads : doubled
        threads = min(threads, max(1, n / 4))
        return max(1, min(threads, maxThreads))
    }

    /// Dynamic threadgroup memory length in bytes: n staged floats plus
    /// one extra float slot at index n for inv (padded to a full float4
    /// row so every float4 access in the kernel is 16-byte aligned:
    /// n is a multiple of 4, so (n + 4) * 4 is a multiple of 16).
    /// HARDENED: invalid or gigantic n (non-positive, not a multiple of
    /// 4, or beyond maxN) returns 0 instead of trapping or overflowing;
    /// within maxN the arithmetic is trivially overflow-free. The caller
    /// guards tgLen > 0 and the dynamic+static budget against the real
    /// device cap before dispatch.
    public static func threadgroupMemoryLength(n: Int) -> Int {
        guard n > 0, n % 4 == 0, n <= maxN else { return 0 }
        return (n + 4) * 4
    }
}
