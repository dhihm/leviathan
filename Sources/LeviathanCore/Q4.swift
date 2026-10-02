import Foundation
import Metal

/// Bounded-workload caps for a full-scale model run. These are separate
/// from the small synthetic-graph `Limits` on purpose: a 26B Q4_0 model is
/// ~14 GB of weights, which is legal here but would be nonsense for the
/// dense/relu demo graphs.
public enum ModelWorkloadLimits: Sendable {
    /// Max elements per Q4_0 matvec row (inner dimension).
    public static let maxQ4InputDim = 1 << 22
    /// Max output rows per Q4_0 matvec.
    public static let maxQ4OutputDim = 1 << 22
    /// Max total Q4_0 weight bytes (32 GiB, above the 14.44 GB Gemma file).
    public static let maxQ4WeightBytes = 32 << 30
}

public enum Q4Error: Error, Equatable, Sendable {
    case badDimensions(String)
    case wrongInputCount(Int, expected: Int)
    case nonFiniteInput(String)
    case nonFiniteScale(block: Int)
    case nonFiniteOutput(index: Int)
    case exceedsModelWorkload(String)
    case integerOverflow(String)
    case metalUnavailable
    case shaderCompileFailed(String)
    case functionMissing
    case commandBufferFailed(String)
    case exceedsMaxBufferLength(needed: Int, max: Int)
    case exceedsRecommendedWorkingSet(needed: Int, recommended: Int)
}

/// Q4_0 quantization: 32 values per 18-byte block — a 2-byte FP16 scale
/// delta followed by 16 bytes holding 32 signed 4-bit levels. Within each
/// block, the low nibble of byte `l` holds value `l` and the high nibble
/// holds value `16 + l`; levels are stored biased by +8, so a stored
/// nibble `n` decodes to `d * (n - 8)`.
public enum Q4_0 {
    public static let valuesPerBlock = 32
    public static let bytesPerBlock = 18

    /// Decode one block's FP16 scale, rejecting non-finite scales.
    static func blockScale(_ block: UnsafeBufferPointer<UInt8>, at base: Int) throws -> Float {
        let bits = UInt16(block[base]) | UInt16(block[base + 1]) << 8
        // FP16 exponent all-ones bits are Inf/NaN.
        guard bits & 0x7C00 != 0x7C00 else {
            throw Q4Error.nonFiniteScale(block: base / bytesPerBlock)
        }
        return Float(Float16(bitPattern: bits))
    }

    /// Validated per-block decode. The oracle is deliberately block-wise:
    /// the full model must never be dequantized into one giant F32 array.
    public static func decodeBlock(_ block: ArraySlice<UInt8>) throws -> [Float] {
        guard block.count == bytesPerBlock else {
            throw Q4Error.badDimensions("block slice is \(block.count) bytes, need \(bytesPerBlock)")
        }
        var out = [Float](repeating: 0, count: valuesPerBlock)
        try block.withUnsafeBytes { raw in
            let u8 = raw.bindMemory(to: UInt8.self)
            let d = try blockScale(u8, at: 0)
            for l in 0..<16 {
                let byte = u8[2 + l]
                out[l] = d * (Float(byte & 0xF) - 8)
                out[16 + l] = d * (Float(byte >> 4) - 8)
            }
        }
        return out
    }

    /// Validated block dot product `sum_j blockValues[j] * x[blockStart + j]`
    /// — the streaming CPU oracle primitive.
    public static func blockDot(_ block: ArraySlice<UInt8>, x: [Float], blockStart: Int) throws -> Float {
        let values = try decodeBlock(block)
        var sum: Float = 0
        for j in 0..<valuesPerBlock {
            let xi = x[blockStart + j]
            guard xi.isFinite else { throw Q4Error.nonFiniteInput("x[\(blockStart + j)]") }
            sum += values[j] * xi
        }
        return sum
    }

    /// Validated streaming scalar matvec oracle: `y[row] = W·x` computed
    /// one block at a time, never materializing the dequantized matrix.
    public static func matvecReference(blocks: [UInt8], outDim: Int, inDim: Int, x: [Float]) throws -> [Float] {
        guard inDim > 0, outDim > 0, inDim % valuesPerBlock == 0,
              inDim <= ModelWorkloadLimits.maxQ4InputDim,
              outDim <= ModelWorkloadLimits.maxQ4OutputDim else {
            throw Q4Error.badDimensions("inDim \(inDim), outDim \(outDim)")
        }
        let blocksPerRow = inDim / valuesPerBlock
        let (weightBytes, overflow) = outDim.multipliedReportingOverflow(
            by: blocksPerRow * bytesPerBlock)
        guard !overflow, weightBytes <= ModelWorkloadLimits.maxQ4WeightBytes,
              blocks.count == weightBytes else {
            throw Q4Error.badDimensions(
                "weights are \(blocks.count) bytes, geometry needs \(overflow ? -1 : weightBytes)")
        }
        guard x.count == inDim else { throw Q4Error.wrongInputCount(x.count, expected: inDim) }
        var y = [Float](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum: Float = 0
            for b in 0..<blocksPerRow {
                let base = (row * blocksPerRow + b) * bytesPerBlock
                sum += try blockDot(blocks[base..<(base + bytesPerBlock)], x: x,
                                    blockStart: b * valuesPerBlock)
            }
            guard sum.isFinite else { throw Q4Error.nonFiniteOutput(index: row) }
            y[row] = sum
        }
        return y
    }
}

/// Evidence from a real Q4_0 GPU matvec run.
public struct Q4MatvecEvidence: Sendable {
    public let deviceName: String
    public let deviceMaxBufferLength: Int
    public let recommendedMaxWorkingSetSize: Int
    public let outDim: Int
    public let inDim: Int
    public let commandStatus: String
}

/// GPU backend for the fused Q4_0 matvec. Weights are uploaded exactly as
/// stored — quantized — never expanded to F32; the kernel dequantizes
/// block-by-block in registers. All dimension arithmetic is checked before
/// any Metal call, and outputs must be finite.
public enum Q4MetalBackend {

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Q4_0 block: 2-byte FP16 delta + 16 bytes of nibbles (32 values).
    // y[row] = sum_col W[row, col] * x[col]; W row-major [outDim, inDim].
    // p = (inDim, outDim); gid.x = row.
    kernel void leviathan_q4_matvec(constant const uint2& p [[buffer(0)]],
                                    device const uchar* w [[buffer(1)]],
                                    device const float* x [[buffer(2)]],
                                    device float* y [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
        uint row = gid;
        if (row >= p.y) { return; }
        uint inDim = p.x;
        uint blocksPerRow = inDim / 32u;
        device const uchar* rowBytes = w + (ulong)row * blocksPerRow * 18u;
        float sum = 0.0f;
        for (uint b = 0; b < blocksPerRow; b++) {
            device const uchar* blk = rowBytes + b * 18u;
            ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
            half dh = as_type<half>(dbits);
            float d = float(dh);
            for (uint l = 0; l < 16u; l++) {
                uchar q = blk[2u + l];
                sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];
                sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];
            }
        }
        y[row] = sum;
    }
    """

    public static func matvecReporting(blocks: [UInt8], outDim: Int, inDim: Int,
                                       x: [Float]) throws
            -> (output: [Float], evidence: Q4MatvecEvidence) {
        guard inDim > 0, outDim > 0, inDim % Q4_0.valuesPerBlock == 0,
              inDim <= ModelWorkloadLimits.maxQ4InputDim,
              outDim <= ModelWorkloadLimits.maxQ4OutputDim else {
            throw Q4Error.badDimensions("inDim \(inDim), outDim \(outDim)")
        }
        guard inDim <= Int(UInt32.max), outDim <= Int(UInt32.max) else {
            throw Q4Error.integerOverflow("dims beyond uint32 for kernel parameters")
        }
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        let (weightBytes0, overflow0) = blocksPerRow.multipliedReportingOverflow(by: Q4_0.bytesPerBlock)
        let (weightBytes, overflow1) = overflow0
            ? (0, true)
            : outDim.multipliedReportingOverflow(by: weightBytes0)
        guard !overflow0, !overflow1, weightBytes <= ModelWorkloadLimits.maxQ4WeightBytes,
              blocks.count == weightBytes else {
            throw Q4Error.badDimensions(
                "weights are \(blocks.count) bytes, geometry needs \(overflow0 || overflow1 ? -1 : weightBytes)")
        }
        guard x.count == inDim else {
            throw Q4Error.wrongInputCount(x.count, expected: inDim)
        }
        guard x.allSatisfy(\.isFinite) else {
            throw Q4Error.nonFiniteInput("activation vector")
        }

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw Q4Error.metalUnavailable
        }
        // Hard device limits and the memory-pressure guard, checked before
        // any allocation.
        guard weightBytes <= device.maxBufferLength else {
            throw Q4Error.exceedsMaxBufferLength(needed: weightBytes, max: device.maxBufferLength)
        }
        // Recommended working set always fits in Int on current devices;
        // clamping keeps the report safe regardless.
        let workingSet = Int(clamping: device.recommendedMaxWorkingSetSize)
        let (totalNeeded, overflow2) = weightBytes.addingReportingOverflow(
            inDim * 4 + outDim * 4)
        guard !overflow2, totalNeeded <= workingSet else {
            throw Q4Error.exceedsRecommendedWorkingSet(
                needed: overflow2 ? -1 : totalNeeded, recommended: workingSet)
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw Q4Error.shaderCompileFailed("\(error)")
        }
        guard let fn = library.makeFunction(name: "leviathan_q4_matvec"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw Q4Error.functionMissing
        }

        // The weights array is copied into shared storage exactly once,
        // quantized; no F32 expansion of weights happens anywhere.
        guard let weightBuffer = device.makeBuffer(bytes: blocks, length: weightBytes,
                                                   options: .storageModeShared),
              let xBuffer = device.makeBuffer(bytes: x, length: x.count * 4,
                                              options: .storageModeShared),
              let yBuffer = device.makeBuffer(length: outDim * 4,
                                              options: .storageModeShared) else {
            throw Q4Error.metalUnavailable
        }

        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            throw Q4Error.metalUnavailable
        }
        var p = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(xBuffer, offset: 0, index: 2)
        encoder.setBuffer(yBuffer, offset: 0, index: 3)
        let width = pipeline.threadExecutionWidth
        let groups = (outDim + width - 1) / width
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4Error.commandBufferFailed(
                "status \(cmd.status.rawValue), error \(String(describing: cmd.error))")
        }

        let out = [Float](UnsafeBufferPointer(
            start: yBuffer.contents().assumingMemoryBound(to: Float.self),
            count: outDim))
        guard let bad = out.firstIndex(where: { !$0.isFinite }) else {
            return (out, Q4MatvecEvidence(
                deviceName: device.name,
                deviceMaxBufferLength: device.maxBufferLength,
                recommendedMaxWorkingSetSize: workingSet,
                outDim: outDim, inDim: inDim,
                commandStatus: "completed"))
        }
        throw Q4Error.nonFiniteOutput(index: bad)
    }
}
