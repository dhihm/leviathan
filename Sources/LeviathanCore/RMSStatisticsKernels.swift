// Reusable RMS statistics kernels.
//
// g4_rmsnorm_stats is the FIRST compute of a statistics group: it
// duplicates the original cooperative RMS arithmetic EXACTLY (same
// ordered float4 square sum, eps/div/sqrt, inverse, and the identical
// x*inv*w left association) while ADDITIONALLY storing the inverse
// statistic to a new device buffer (binding 7) from lane 0. All later
// group members run g4_rmsnorm_cached: they read inverse[tgid] and
// re-apply the original per-element association in parallel — no new
// normalization, no host readback.
//
// Binding ABI is the incumbent 0..6 (p, eps, withScale, wOff, wbase,
// x, y) plus buffer 7 = device float* inverse. The cached kernel keeps
// the full ABI (eps is unused there but the slot stays reserved) so the
// runner binds identically for both. Bit-exactness versus the incumbent
// kernel is a property the test suite verifies on hardware for the
// tested geometries; it is not a proof for all inputs.

import Foundation

public enum RMSStatisticsKernels {

    public static let computeKernelName = "g4_rmsnorm_stats"
    public static let applyKernelName = "g4_rmsnorm_cached"

    /// Version of the kernel pair ABI/arithmetic contract.
    public static let version = 1

    /// Standalone MSL (its own library, never merged into the runner's
    /// shared shaderSource or shared pipeline dictionary).
    public static let source = """
#include <metal_stdlib>
using namespace metal;

// First compute of a statistics group. Identical arithmetic to the
// incumbent g4_rmsnorm_coop (staged full input copy, lane-0 serial
// reduction in the exact incumbent float4 component order, parallel
// apply reading ONLY the staged copy so in-place x == y is safe), plus
// inverse[tgid] = inv stored from lane 0. One threadgroup per vector;
// the dynamically sized threadgroup array is (n + 4) floats, 16-byte
// aligned, set per dispatch via setThreadgroupMemoryLength(index: 0).
kernel void g4_rmsnorm_stats(constant const uint2& p [[buffer(0)]],
                             constant const float& eps [[buffer(1)]],
                             constant const int& withScale [[buffer(2)]],
                             constant const ulong& wOff [[buffer(3)]],
                             device const uchar* wbase [[buffer(4)]],
                             device const float* x [[buffer(5)]],
                             device float* y [[buffer(6)]],
                             device float* inverse [[buffer(7)]],
                             threadgroup float* staged [[threadgroup(0)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tcount [[threads_per_threadgroup]]) {
    if (tgid >= p.y) { return; }
    uint n = p.x;
    device const float* xv = x + (ulong)tgid * n;
    // Phase 1: cooperative float4 staging of the full input vector.
    // This staged copy IS the full copy that makes an in-place
    // x == y dispatch safe: no thread reads x after any write to y.
    for (uint i = 4u * tid; i < n; i += 4u * tcount) {
        *(threadgroup float4*)(staged + i) = *(device const float4*)(xv + i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Phase 2: lane 0 serial reduction — EXACTLY the incumbent order
    // (no tree, no reassociation) — plus the inverse statistic store.
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
        float inv = 1.0f / sqrt(ms);
        staged[n] = inv;
        inverse[tgid] = inv;
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

// Cached apply: re-uses the statistic computed by g4_rmsnorm_stats.
// Same ABI (buffers 0..7; eps is reserved and unused). Reads
// inverse[tgid], applies the original per-element association
// x[i] * inv * w[i] (or x[i] * inv) per index in parallel — no new
// normalization, no host readback. In-place x == y is safe: each
// element is read and then written by exactly one thread, and every
// thread only touches its own indices. Bounds are guarded by the
// tgid >= p.y early-out and the i < n loop bounds; the logical grid
// (width * vectors) is overflow-checked on the host before dispatch.
kernel void g4_rmsnorm_cached(constant const uint2& p [[buffer(0)]],
                              constant const float& eps [[buffer(1)]],
                              constant const int& withScale [[buffer(2)]],
                              constant const ulong& wOff [[buffer(3)]],
                              device const uchar* wbase [[buffer(4)]],
                              device const float* x [[buffer(5)]],
                              device float* y [[buffer(6)]],
                              device const float* inverse [[buffer(7)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tcount [[threads_per_threadgroup]]) {
    (void)eps; // ABI slot reserved; the cached apply never re-normalizes.
    if (tgid >= p.y) { return; }
    uint n = p.x;
    float inv = inverse[tgid];
    device const float* xv = x + (ulong)tgid * n;
    device float* yv = y + (ulong)tgid * n;
    if (withScale != 0) {
        device const float* w = (device const float*)(wbase + wOff);
        for (uint i = tid; i < n; i += tcount) {
            yv[i] = xv[i] * inv * w[i];
        }
    } else {
        for (uint i = tid; i < n; i += tcount) {
            yv[i] = xv[i] * inv;
        }
    }
}
"""

    /// SHA-256 of the actual kernel source (local SHA API; CryptoKit is
    /// deliberately NOT imported).
    public static func sourceSha256() -> String {
        SHA256.hex(SHA256.hash(Array(source.utf8)))
    }

    // Eligibility and schedule derivation delegate to the public
    // RMSCooperative helpers so the stats kernels can never drift from
    // the incumbent geometry contract.

    public static func eligible(n: Int) -> Bool {
        RMSCooperative.eligible(n: n)
    }

    public static func threadCount(n: Int, executionWidth: Int, maxThreads: Int) -> Int {
        RMSCooperative.threadCount(n: n, executionWidth: executionWidth, maxThreads: maxThreads)
    }

    public static func threadgroupMemoryLength(n: Int) -> Int {
        RMSCooperative.threadgroupMemoryLength(n: n)
    }

    /// Weighted-offset alignment guard, same 16-byte rule as the
    /// production rmsVectorLoadGuards predicate family.
    public static func weightOffsetIsAligned(_ wOff: Int) -> Bool {
        wOff >= 0 && wOff % 16 == 0
    }

    /// Checked logical grid width * vectors; nil on non-positive inputs
    /// or integer overflow (overflow handling lives on the host).
    public static func elementCount(width: Int, vectors: Int) -> Int? {
        guard width > 0, vectors > 0 else { return nil }
        let (count, overflow) = width.multipliedReportingOverflow(by: vectors)
        return overflow ? nil : count
    }
}
