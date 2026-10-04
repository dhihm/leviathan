import Foundation
import CryptoKit

/// Standalone intra-block SIMD-reduction Q6_K matvec candidate.
/// Deliberately isolated from the production plan: this source, its
/// version and its content hash form the artifact identity until the
/// candidate is independently qualified. Each 32-lane SIMD group serves
/// one output row; lanes compute the same four dequantized products per
/// half-block as the plain kernel (for l = lane) over a serial b/c loop,
/// accumulating into a lane-local float accumulator that is finally
/// reduced with simd_sum and written by lane 0. There is no threadgroup
/// staging array, no threadgroup barrier, and no serial n-element sum.
/// The simd_sum reduction reassociates the floating-point addition order
/// relative to the plain kernel, so bit-exactness and order preservation
/// are NOT claimed; qualification is against the strict FP64 oracle with
/// a normalized error below 1e-4, finiteness, and the exact hand-computed
/// -315 known block.
public enum Q6Cooperative {
    /// Schedule version; bump on any change to `source`.
    public static let version: Int = 2

    public static let kernelName = "g4_q6_matvec_coop"
    public static let minInDim = 256
    public static let maxInDim = 2816
    public static let maxRowsPerGroup = 2
    /// Lane mapping (rowInGroup = tid >> 5, lane = tid & 31) only holds
    /// for a 32-wide SIMD group; other widths must fall back to plain.
    public static let requiredExecutionWidth = 32
    /// No threadgroup staging array is used; the reduction is intra-SIMD.
    public static let staticThreadgroupBytes = 0
    public static let maxThreadgroups = 1 << 20

    public static let source = """
#include <metal_stdlib>
using namespace metal;

// Intra-block SIMD-reduction Q6_K matvec candidate.
// ABI matches plain g4_q6_matvec: buffer0 uint2 params (inDim, outDim),
// buffer1 ulong weight offset, buffer2 weight base, buffer3 input,
// buffer4 output; buffer5 carries rowsPerGroup. Each output row is
// served by one 32-lane SIMD group (rowInGroup = tid >> 5,
// lane = tid & 31). Lanes compute the same four dequantized products
// per half-block as the plain kernel (for l = lane) while b and c run
// serially, accumulating into a lane-local float accumulator. simd_sum
// reduces the accumulator across the SIMD group and lane 0 writes the
// row. This reassociates the floating-point addition order relative to
// the plain kernel; bit-exactness is not claimed. The row-tail guard
// (validRow) is uniform within each SIMD group and involves no barrier
// and no early return divergence across a SIMD group. Host code must
// guarantee n in [256, 2816] and a multiple of 256, rowsPerGroup in
// {1, 2}, and threads = 32 * rowsPerGroup within the pipeline cap.

kernel void g4_q6_matvec_coop(constant const uint2& p [[buffer(0)]],
                              constant const ulong& wOff [[buffer(1)]],
                              device const uchar* wbase [[buffer(2)]],
                              device const float* x [[buffer(3)]],
                              device float* y [[buffer(4)]],
                              constant const uint& rowsPerGroup [[buffer(5)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint groupID [[threadgroup_position_in_grid]]) {
    uint lane = tid & 31u;
    uint rowInGroup = tid >> 5u;
    uint row = groupID * rowsPerGroup + rowInGroup;
    uint n = p.x;
    uint blocksPerRow = n / 256u;
    bool validRow = (row < p.y) && (rowInGroup < rowsPerGroup);
    float acc = 0.0f;

    for (uint b = 0; b < blocksPerRow; b++) {
        for (uint c = 0; c < 2u; c++) {
            if (validRow) {
                device const uchar* blk = wbase + wOff
                    + (ulong)row * blocksPerRow * 210u + b * 210u;
                uint ql = c * 64u;
                uint qh = 128u + c * 32u;
                uint sc = c * 8u;
                uint xbase = b * 256u + c * 128u;
                ushort dbits = ushort(blk[208]) | (ushort(blk[209]) << 8u);
                float d = float(as_type<half>(dbits));
                uint isub = lane / 16u;
                uchar h = blk[qh + lane];
                float s0 = float((char)blk[192u + sc + isub]);
                float s1 = float((char)blk[192u + sc + isub + 2u]);
                float s2 = float((char)blk[192u + sc + isub + 4u]);
                float s3 = float((char)blk[192u + sc + isub + 6u]);
                float q1 = float((int)((blk[ql + lane] & 0x0Fu) | ((h & 3u) << 4u)) - 32);
                float q2 = float((int)((blk[ql + 32u + lane] & 0x0Fu) | (((h >> 2u) & 3u) << 4u)) - 32);
                float q3 = float((int)((blk[ql + lane] >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                float q4 = float((int)((blk[ql + 32u + lane] >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                acc += d * s0 * q1 * x[xbase + lane];
                acc += d * s1 * q2 * x[xbase + 32u + lane];
                acc += d * s2 * q3 * x[xbase + 64u + lane];
                acc += d * s3 * q4 * x[xbase + 96u + lane];
            }
        }
    }
    float total = simd_sum(acc);
    if (lane == 0u && validRow) {
        y[row] = total;
    }
}
"""

    /// SHA-256 content hash of the shader source, hex-encoded. This is
    /// the artifact identity used by tests via `hash`.
    public static func sourceSha256() -> String {
        return SHA256.hex(SHA256.hash(Array(source.utf8)))
    }

    /// Content hash of `source`; kept under this name for existing tests.
    public static let hash: String = sourceSha256()

    public struct LaunchPlan: Equatable {
        public let threadsPerThreadgroup: Int
        public let threadgroups: Int
        public let rowsPerGroup: Int
    }

    /// Pure checked geometry eligibility. No Metal types; callers pass the
    /// actual pipeline executionWidth and maxTotalThreadsPerThreadgroup
    /// and the device maxThreadgroupMemoryLength. Negative capability
    /// values are rejected.
    public static func eligible(inDim: Int, outDim: Int, rowsPerGroup: Int,
                                executionWidth: Int, pipelineMaxThreads: Int,
                                maxThreadgroupMemory: Int) -> Bool {
        return plan(inDim: inDim, outDim: outDim, rowsPerGroup: rowsPerGroup,
                    executionWidth: executionWidth,
                    pipelineMaxThreads: pipelineMaxThreads,
                    maxThreadgroupMemory: maxThreadgroupMemory) != nil
    }

    /// Pure checked launch plan; nil means the caller must fall back to
    /// the plain kernel. All arithmetic is overflow-safe. The kernel uses
    /// no threadgroup memory (staticThreadgroupBytes is 0), so the memory
    /// check only rejects negative capabilities.
    public static func plan(inDim: Int, outDim: Int, rowsPerGroup: Int,
                            executionWidth: Int, pipelineMaxThreads: Int,
                            maxThreadgroupMemory: Int) -> LaunchPlan? {
        guard inDim >= minInDim, inDim <= maxInDim, inDim % 256 == 0 else {
            return nil
        }
        guard rowsPerGroup == 1 || rowsPerGroup == maxRowsPerGroup else {
            return nil
        }
        guard outDim >= 1 else { return nil }
        guard executionWidth == requiredExecutionWidth else { return nil }
        guard pipelineMaxThreads >= 0 else { return nil }
        let threads = requiredExecutionWidth * rowsPerGroup
        guard threads <= pipelineMaxThreads else { return nil }
        guard maxThreadgroupMemory >= staticThreadgroupBytes else { return nil }
        let (sum, o1) = outDim.addingReportingOverflow(rowsPerGroup - 1)
        guard !o1 else { return nil }
        let groups = sum / rowsPerGroup
        guard groups >= 1, groups <= maxThreadgroups else { return nil }
        return LaunchPlan(threadsPerThreadgroup: threads,
                          threadgroups: groups,
                          rowsPerGroup: rowsPerGroup)
    }
}
