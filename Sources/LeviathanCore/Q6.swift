import Foundation
import Metal

/// Q6_K quantization: 256 values per 210-byte block. Layout (GGML public
/// spec): a 2-byte FP16 block scale `d`; 16 signed int8 subgroup scales
/// (one per 16 values); 128 bytes `ql` holding the low 4 bits of each
/// 6-bit level; 64 bytes `qh` holding the top 2 bits of four levels per
/// byte. A stored 6-bit level `L` decodes to the signed level `L - 32`,
/// and value = `d * subgroupScale * (L - 32)`.
public enum Q6_K {
    public static let valuesPerBlock = 256
    public static let bytesPerBlock = 210
    public static let valuesPerSubgroup = 16

    /// Validated per-block decode (CPU oracle primitive). Block order
    /// within the 256 values, for each 128-value half-chunk `c` (ql offset
    /// 64c, qh offset 32c, scale offset 8c) and `l` in 0..<32, subgroup
    /// `is = l / 16`:
    ///   v[l]        = d * sc[is+0] * ((ql[l]     & 0xF) | (2bits(qh[l], 0)) - 32)
    ///   v[32+l]     = d * sc[is+2] * ((ql[32+l]  & 0xF) | (2bits(qh[l], 2)) - 32)
    ///   v[64+l]     = d * sc[is+4] * ((ql[l]     >> 4) | (2bits(qh[l], 4)) - 32)
    ///   v[96+l]     = d * sc[is+6] * ((ql[32+l]  >> 4) | (2bits(qh[l], 6)) - 32)
    public static func decodeBlock(_ block: ArraySlice<UInt8>) throws -> [Float] {
        guard block.count == bytesPerBlock else {
            throw Q4Error.badDimensions(
                "q6_k block slice is \(block.count) bytes, need \(bytesPerBlock)")
        }
        var out = [Float](repeating: 0, count: valuesPerBlock)
        try block.withUnsafeBytes { raw in
            let u8 = raw.bindMemory(to: UInt8.self)
            // Reject non-finite FP16 block scales (exponent all ones).
            let dBits = UInt16(u8[208]) | UInt16(u8[209]) << 8
            guard dBits & 0x7C00 != 0x7C00 else {
                throw Q4Error.nonFiniteScale(block: -1)
            }
            let d = Float(Float16(bitPattern: dBits))
            // scales[16] start at byte 192 (after ql 128 + qh 64).
            let scales = (0..<16).map { Int(Int8(bitPattern: u8[192 + $0])) }
            for c in 0..<2 {
                let ql = c * 64      // ql offset within the block
                let qh = 128 + c * 32
                let sc = c * 8       // subgroup-scale offset
                let base = c * 128   // output offset
                for l in 0..<32 {
                    let isub = l / 16
                    let b = u8[qh + l]
                    func level(_ low: Int, _ shift: Int) -> Int {
                        (low | (Int((b >> UInt8(shift)) & 3) << 4)) - 32
                    }
                    out[base + l] = d * Float(scales[sc + isub + 0])
                        * Float(level(Int(u8[ql + l] & 0xF), 0))
                    out[base + 32 + l] = d * Float(scales[sc + isub + 2])
                        * Float(level(Int(u8[ql + 32 + l] & 0xF), 2))
                    out[base + 64 + l] = d * Float(scales[sc + isub + 4])
                        * Float(level(Int(u8[ql + l] >> 4), 4))
                    out[base + 96 + l] = d * Float(scales[sc + isub + 6])
                        * Float(level(Int(u8[ql + 32 + l] >> 4), 6))
                }
            }
        }
        return out
    }

    /// Validated streaming scalar matvec oracle, block by block.
    public static func matvecReference(blocks: [UInt8], outDim: Int, inDim: Int,
                                       x: [Float]) throws -> [Float] {
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
        // Accumulation is in Double and converted to Float at the end:
        // Q6_K values are dyadic (fp16 d * int8 scale * int level, exact
        // in Float), but a 256-column dot product of ~1e4-magnitude terms
        // can cancel to a small result, where Float accumulation order
        // alone moves the answer by more than 1e-4 relative. Double
        // accumulation makes the CPU oracle independent of GPU kernel
        // reduction order.
        var y = [Float](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum: Double = 0
            for b in 0..<blocksPerRow {
                let base = (row * blocksPerRow + b) * bytesPerBlock
                let values = try decodeBlock(blocks[base..<(base + bytesPerBlock)])
                for j in 0..<valuesPerBlock {
                    let xi = x[b * valuesPerBlock + j]
                    guard xi.isFinite else {
                        throw Q4Error.nonFiniteInput("x[\(b * valuesPerBlock + j)]")
                    }
                    sum += Double(values[j]) * Double(xi)
                }
            }
            let f = Float(sum)
            guard f.isFinite else { throw Q4Error.nonFiniteOutput(index: row) }
            y[row] = f
        }
        return y
    }

    /// Same computation kept entirely in Double — the independent
    /// reference used by the correctness gates to bound Float rounding.
    /// Geometry and bounds are validated exactly like `matvecReference`.
    public static func matvecReferenceF64(blocks: [UInt8], outDim: Int, inDim: Int,
                                          x: [Float]) throws -> [Double] {
        guard inDim > 0, outDim > 0, inDim % valuesPerBlock == 0,
              inDim <= ModelWorkloadLimits.maxQ4InputDim,
              outDim <= ModelWorkloadLimits.maxQ4OutputDim else {
            throw Q4Error.badDimensions("inDim \(inDim), outDim \(outDim)")
        }
        let blocksPerRow = inDim / valuesPerBlock
        let (bytes0, of0) = blocksPerRow.multipliedReportingOverflow(by: bytesPerBlock)
        let (weightBytes, of1) = of0 ? (0, true) : outDim.multipliedReportingOverflow(by: bytes0)
        guard !of0, !of1, weightBytes <= ModelWorkloadLimits.maxQ4WeightBytes,
              blocks.count == weightBytes else {
            throw Q4Error.badDimensions(
                "weights are \(blocks.count) bytes, geometry needs \(of0 || of1 ? -1 : weightBytes)")
        }
        guard x.count == inDim, x.allSatisfy(\.isFinite) else {
            if x.count != inDim { throw Q4Error.wrongInputCount(x.count, expected: inDim) }
            throw Q4Error.nonFiniteInput("activation vector")
        }
        var y = [Double](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum: Double = 0
            for b in 0..<blocksPerRow {
                let base = (row * blocksPerRow + b) * bytesPerBlock
                let values = try decodeBlock(blocks[base..<(base + bytesPerBlock)])
                for j in 0..<valuesPerBlock {
                    sum += Double(values[j]) * Double(x[b * valuesPerBlock + j])
                }
            }
            guard sum.isFinite else { throw Q4Error.nonFiniteOutput(index: row) }
            y[row] = sum
        }
        return y
    }
}

/// GPU backend for the fused Q6_K matvec; weights stay quantized.
public enum Q6MetalBackend {

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Q6_K block: d (fp16), scales[16] (int8), ql[128], qh[64] = 210 bytes
    // for 256 values. y[row] = sum_col W[row, col] * x[col].
    kernel void leviathan_q6_matvec(constant const uint2& p [[buffer(0)]],
                                    device const uchar* w [[buffer(1)]],
                                    device const float* x [[buffer(2)]],
                                    device float* y [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
        uint row = gid;
        if (row >= p.y) { return; }
        uint inDim = p.x;
        uint blocksPerRow = inDim / 256u;
        device const uchar* rowBytes = w + (ulong)row * blocksPerRow * 210u;
        float sum = 0.0f;
        for (uint b = 0; b < blocksPerRow; b++) {
            device const uchar* blk = rowBytes + b * 210u;
            ushort dbits = ushort(blk[208]) | (ushort(blk[209]) << 8u);
            float d = float(as_type<half>(dbits));
            for (uint c = 0; c < 2u; c++) {
                uint ql = c * 64u;
                uint qh = 128u + c * 32u;
                uint sc = c * 8u;
                uint xbase = b * 256u + c * 128u;
                for (uint l = 0; l < 32u; l++) {
                    uint isub = l / 16u;
                    uchar h = blk[qh + l];
                    float s0 = float((char)blk[192u + sc + isub]);
                    float s1 = float((char)blk[192u + sc + isub + 2u]);
                    float s2 = float((char)blk[192u + sc + isub + 4u]);
                    float s3 = float((char)blk[192u + sc + isub + 6u]);
                    float q1 = float((int)((blk[ql + l] & 0x0Fu) | ((h & 3u) << 4u)) - 32);
                    float q2 = float((int)((blk[ql + 32u + l] & 0x0Fu) | (((h >> 2u) & 3u) << 4u)) - 32);
                    float q3 = float((int)((blk[ql + l] >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                    float q4 = float((int)((blk[ql + 32u + l] >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                    sum += d * s0 * q1 * x[xbase + l];
                    sum += d * s1 * q2 * x[xbase + 32u + l];
                    sum += d * s2 * q3 * x[xbase + 64u + l];
                    sum += d * s3 * q4 * x[xbase + 96u + l];
                }
            }
        }
        y[row] = sum;
    }
    """

    public static func matvecReporting(blocks: [UInt8], outDim: Int, inDim: Int,
                                       x: [Float]) throws
            -> (output: [Float], evidence: Q4MatvecEvidence) {
        guard inDim > 0, outDim > 0, inDim % Q6_K.valuesPerBlock == 0,
              inDim <= ModelWorkloadLimits.maxQ4InputDim,
              outDim <= ModelWorkloadLimits.maxQ4OutputDim,
              inDim <= Int(UInt32.max), outDim <= Int(UInt32.max) else {
            throw Q4Error.badDimensions("inDim \(inDim), outDim \(outDim)")
        }
        let blocksPerRow = inDim / Q6_K.valuesPerBlock
        let (bytes0, of0) = blocksPerRow.multipliedReportingOverflow(by: Q6_K.bytesPerBlock)
        let (weightBytes, of1) = of0 ? (0, true) : outDim.multipliedReportingOverflow(by: bytes0)
        guard !of0, !of1, weightBytes <= ModelWorkloadLimits.maxQ4WeightBytes,
              blocks.count == weightBytes else {
            throw Q4Error.badDimensions(
                "weights are \(blocks.count) bytes, geometry needs \(of0 || of1 ? -1 : weightBytes)")
        }
        guard x.count == inDim, x.allSatisfy(\.isFinite) else {
            if x.count != inDim {
                throw Q4Error.wrongInputCount(x.count, expected: inDim)
            }
            throw Q4Error.nonFiniteInput("activation vector")
        }

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw Q4Error.metalUnavailable
        }
        guard weightBytes <= device.maxBufferLength else {
            throw Q4Error.exceedsMaxBufferLength(needed: weightBytes, max: device.maxBufferLength)
        }
        let workingSet = Int(clamping: device.recommendedMaxWorkingSetSize)
        let (totalNeeded, of2) = weightBytes.addingReportingOverflow(inDim * 4 + outDim * 4)
        guard !of2, totalNeeded <= workingSet else {
            throw Q4Error.exceedsRecommendedWorkingSet(
                needed: of2 ? -1 : totalNeeded, recommended: workingSet)
        }

        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw Q4Error.shaderCompileFailed("\(error)") }
        guard let fn = library.makeFunction(name: "leviathan_q6_matvec"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw Q4Error.functionMissing
        }
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
            start: yBuffer.contents().assumingMemoryBound(to: Float.self), count: outDim))
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
