import Foundation
import Metal

// Phase 60: COMPONENT-ONLY candidate "cooperative packed Q4_0 WEIGHT-byte staging".
// The 32 lanes of a threadgroup own 32 consecutive output rows. For each tile of TB blocks the lanes cooperatively
// copy the packed 18-byte Q4_0 blocks of all 32 rows into threadgroup memory with lane-consecutive 4-byte word loads,
// barrier, and then each lane evaluates its OWN row with the unchanged scalar arithmetic of the production
// g4_q4_matvec: same FP16 scale decode, same nibble decode, same two sequential sums per quant byte in the same
// order, x read as device scalar loads at the same indices. No tree or SIMD reassociation, no FP64, no shared
// activation copy, no default pipeline change and no model integration. This is distinct from the rejected phase-37
// INPUT activation staging, the phase-39 launch geometry and the phase-50/52 SIMD candidates.
// Identical source text does NOT guarantee identical compiled bits (the vectorized-load variant proved that), so the
// component measures Float bit identity against the production kernel compiled with the SAME options nil, plus the
// independent FP64 contract, on fixed synthetic edge corpora and bounded real Q4 rows, before any timing. Raw stage
// ticks are reported without units; fullModelQualified, gainQualified, installable and productionPerformanceClaim
// are always false.

public struct Q4WeightStageError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

// MARK: - Schedule descriptor

/// Validated tile geometry: rowsPerTile lanes (the measured execution width), an even number of Q4_0 blocks per tile
/// step (so every row segment is whole 4-byte words) and an odd threadgroup row pitch in words (padding hypothesis:
/// an odd pitch spreads the 32 lanes' same-offset reads over distinct residues for any power-of-two bank count;
/// the bank count is unmeasured and no fixed count is claimed).
public struct Q4WeightStageSchedule: Codable, Equatable, Hashable, Sendable {
    public let rowsPerTile: Int
    public let blocksPerTile: Int
    public let segmentBytes: Int
    public let segmentWords: Int
    public let rowPitchWords: Int
    public let paddedPitch: Bool

    public static let maxThreadgroupBytes = 32 * 1024
    /// The two fixed variants of this component.
    public static let fixedBlockTiles = [22, 8]

    public var threadgroupBytes: Int { rowsPerTile * rowPitchWords * 4 }
    public var kernelName: String {
        "q4ws_matvec_R" + String(rowsPerTile) + "_B" + String(blocksPerTile) + "_P" + String(rowPitchWords)
    }

    public static func derive(blocksPerTile: Int, rowsPerTile: Int) throws -> Q4WeightStageSchedule {
        guard rowsPerTile >= 8, rowsPerTile <= 64, rowsPerTile & (rowsPerTile - 1) == 0 else {
            throw Q4WeightStageError("rowsPerTile " + String(rowsPerTile) + " must be a power of two in 8...64")
        }
        guard blocksPerTile >= 2, blocksPerTile <= 64, blocksPerTile % 2 == 0 else {
            throw Q4WeightStageError("blocksPerTile " + String(blocksPerTile)
                                     + " must be even in 2...64 so every tile segment is whole 4-byte words")
        }
        let segmentBytes = blocksPerTile * Q4_0.bytesPerBlock
        let segmentWords = segmentBytes / 4
        let padded = segmentWords % 2 == 0
        let s = Q4WeightStageSchedule(rowsPerTile: rowsPerTile, blocksPerTile: blocksPerTile,
                                      segmentBytes: segmentBytes, segmentWords: segmentWords,
                                      rowPitchWords: padded ? segmentWords + 1 : segmentWords, paddedPitch: padded)
        try s.validate()
        return s
    }

    public func validate() throws {
        guard rowsPerTile >= 8, rowsPerTile <= 64, rowsPerTile & (rowsPerTile - 1) == 0,
              blocksPerTile >= 2, blocksPerTile <= 64, blocksPerTile % 2 == 0,
              segmentBytes == blocksPerTile * Q4_0.bytesPerBlock, segmentBytes % 4 == 0,
              segmentWords == segmentBytes / 4, paddedPitch == (segmentWords % 2 == 0),
              rowPitchWords == (paddedPitch ? segmentWords + 1 : segmentWords), rowPitchWords % 2 == 1 else {
            throw Q4WeightStageError("schedule is not the canonical derivation")
        }
        guard threadgroupBytes <= Self.maxThreadgroupBytes else {
            throw Q4WeightStageError("schedule threadgroup bytes " + String(threadgroupBytes) + " exceed "
                                     + String(Self.maxThreadgroupBytes))
        }
    }
}

public struct Q4WeightStageLaunch: Equatable, Sendable {
    public let threadgroups: Int
    public let threadsPerThreadgroup: Int
}

// MARK: - Fail-closed plan and host mirror of the cooperative copy

public enum Q4WeightStagePlan {
    /// Control launch width: the Q4 diagnostic reference width every model gate used (same threadgroup count as
    /// the candidate); the scalar kernel's arithmetic does not depend on it.
    public static let controlLaunchWidth = 32

    /// nil when the dispatch is supported; otherwise the exact reason the candidate is declined BEFORE any dispatch.
    public static func declineReason(inDim: Int, outDim: Int, weightOffset: Int, weightLength: Int,
                                     schedule: Q4WeightStageSchedule, executionWidth: Int, pipelineMaxThreads: Int,
                                     staticThreadgroupBytes: Int, deviceThreadgroupCap: Int) -> String? {
        if (try? schedule.validate()) == nil { return "schedule invalid" }
        guard inDim > 0, inDim % Q4_0.valuesPerBlock == 0, inDim <= ModelWorkloadLimits.maxQ4InputDim,
              inDim <= Int(UInt32.max) else {
            return "inDim " + String(inDim) + " is not a positive bounded multiple of 32"
        }
        guard outDim > 0, outDim <= ModelWorkloadLimits.maxQ4OutputDim, outDim <= Int(UInt32.max) else {
            return "outDim " + String(outDim) + " is not positive and bounded"
        }
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        guard blocksPerRow % 2 == 0 else {
            return "odd block count " + String(blocksPerRow)
                + ": a row is not whole 4-byte words, so every tile tail would need a half-word read"
        }
        let (rowBytes, o1) = blocksPerRow.multipliedReportingOverflow(by: Q4_0.bytesPerBlock)
        guard !o1, rowBytes % 4 == 0 else { return "row byte count invalid" }
        guard weightOffset >= 0, weightOffset % 4 == 0 else {
            return "weight offset " + String(weightOffset) + " is not 4-byte aligned"
        }
        let (span, o2) = rowBytes.multipliedReportingOverflow(by: outDim)
        let (end, o3) = o2 ? (0, true) : weightOffset.addingReportingOverflow(span)
        guard !o2, !o3, end <= weightLength else { return "weight span exceeds the bound buffer" }
        guard executionWidth == schedule.rowsPerTile else {
            return "rows per tile " + String(schedule.rowsPerTile) + " != measured execution width "
                + String(executionWidth)
        }
        guard schedule.rowsPerTile <= pipelineMaxThreads else {
            return "rows per tile exceed the compiled pipeline thread maximum " + String(pipelineMaxThreads)
        }
        guard staticThreadgroupBytes <= deviceThreadgroupCap, schedule.threadgroupBytes <= deviceThreadgroupCap,
              staticThreadgroupBytes >= schedule.threadgroupBytes else {
            return "threadgroup memory " + String(staticThreadgroupBytes) + " / " + String(schedule.threadgroupBytes)
                + " outside the device cap " + String(deviceThreadgroupCap)
        }
        let groups = (outDim + schedule.rowsPerTile - 1) / schedule.rowsPerTile
        guard groups > 0, groups <= Int(UInt32.max) else { return "threadgroup count invalid" }
        return nil
    }

    public static func launch(outDim: Int, schedule: Q4WeightStageSchedule) -> Q4WeightStageLaunch {
        Q4WeightStageLaunch(threadgroups: (outDim + schedule.rowsPerTile - 1) / schedule.rowsPerTile,
                            threadsPerThreadgroup: schedule.rowsPerTile)
    }

    public static func controlLaunch(outDim: Int, pipelineMaxThreads: Int) -> Q4WeightStageLaunch {
        let width = max(1, min(controlLaunchWidth, pipelineMaxThreads))
        return Q4WeightStageLaunch(threadgroups: (outDim + width - 1) / width, threadsPerThreadgroup: width)
    }

    /// Copy index t of one tile step maps to (row within the tile, word within the segment).
    public static func tileWord(index t: Int, segmentWords: Int) -> (row: Int, word: Int) {
        let r = t / segmentWords
        return (r, t - r * segmentWords)
    }

    /// CPU mirror of one tile step of the kernel's cooperative copy for one threadgroup: the staged segment bytes
    /// per lane (nil for lanes whose row is beyond outDim), read as little-endian 4-byte words exactly as the GPU does.
    public static func stagedRows(blocks: [UInt8], inDim: Int, outDim: Int, group: Int, tileStartBlock: Int,
                                  schedule: Q4WeightStageSchedule) -> [[UInt8]?] {
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        let rowBytes = blocksPerRow * Q4_0.bytesPerBlock
        let tb = min(schedule.blocksPerTile, blocksPerRow - tileStartBlock)
        guard tb > 0, blocks.count >= outDim * rowBytes else { return [] }
        let segWords = tb * Q4_0.bytesPerBlock / 4
        var tile = [UInt32](repeating: 0, count: schedule.rowsPerTile * schedule.rowPitchWords)
        for t in 0..<(schedule.rowsPerTile * segWords) {
            let (r, w) = tileWord(index: t, segmentWords: segWords)
            let srcRow = group * schedule.rowsPerTile + r
            var v: UInt32 = 0
            if srcRow < outDim {
                let o = srcRow * rowBytes + tileStartBlock * Q4_0.bytesPerBlock + 4 * w
                v = UInt32(blocks[o]) | UInt32(blocks[o + 1]) << 8 | UInt32(blocks[o + 2]) << 16
                    | UInt32(blocks[o + 3]) << 24
            }
            tile[r * schedule.rowPitchWords + w] = v
        }
        return (0..<schedule.rowsPerTile).map { lane -> [UInt8]? in
            guard group * schedule.rowsPerTile + lane < outDim else { return nil }
            var bytes = [UInt8]()
            bytes.reserveCapacity(tb * Q4_0.bytesPerBlock)
            for w in 0..<segWords {
                let v = tile[lane * schedule.rowPitchWords + w]
                bytes.append(UInt8(v & 0xFF))
                bytes.append(UInt8((v >> 8) & 0xFF))
                bytes.append(UInt8((v >> 16) & 0xFF))
                bytes.append(UInt8(v >> 24))
            }
            return bytes
        }
    }
}

// MARK: - Emitter

public enum Q4WeightStageEmitter {
    public static let codegenVersion = "q4ws-v1"
    public static let controlFunction = "g4_q4_matvec"

    /// Deterministic MSL for one schedule. The per-row arithmetic lines are the production scalar kernel's.
    public static func emit(_ s: Q4WeightStageSchedule) throws -> String {
        try s.validate()
        let lines: [String] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "// " + s.kernelName + " - EMITTED by LeviathanForge Q4 weight staging (" + codegenVersion + ").",
            "// Cooperative copy of the packed Q4_0 WEIGHT bytes of " + String(s.rowsPerTile) + " rows x "
                + String(s.blocksPerTile) + " blocks per tile step into threadgroup memory (row pitch "
                + String(s.rowPitchWords) + " words), then the unchanged scalar per-row dot from the copy.",
            "kernel void " + s.kernelName + "(constant const uint2& p [[buffer(0)]],",
            "        constant const ulong& wOff [[buffer(1)]],",
            "        device const uchar* wbase [[buffer(2)]],",
            "        device const float* x [[buffer(3)]],",
            "        device float* y [[buffer(4)]],",
            "        uint lid [[thread_position_in_threadgroup]],",
            "        uint wid [[threadgroup_position_in_grid]]) {",
            "    constexpr uint R = " + String(s.rowsPerTile) + "u;",
            "    constexpr uint TB = " + String(s.blocksPerTile) + "u;",
            "    constexpr uint PITCH = " + String(s.rowPitchWords) + "u;",
            "    threadgroup uint tile[" + String(s.rowsPerTile * s.rowPitchWords) + "];",
            "    uint row = wid * R + lid;",
            "    bool active = row < p.y;",
            "    uint blocksPerRow = p.x / 32u;",
            "    uint rowBytes = blocksPerRow * 18u;",
            "    device const uint* words = (device const uint*)(wbase + wOff);",
            "    threadgroup const uchar* rowTile = (threadgroup const uchar*)(tile + lid * PITCH);",
            "    float sum = 0.0f;",
            "    for (uint t0 = 0u; t0 < blocksPerRow; t0 += TB) {",
            "        uint tb = min(TB, blocksPerRow - t0);",
            "        uint segWords = (tb * 18u) / 4u;",
            "        uint total = R * segWords;",
            "        for (uint t = lid; t < total; t += R) {",
            "            uint r = t / segWords;",
            "            uint w = t - r * segWords;",
            "            uint srcRow = wid * R + r;",
            "            uint v = 0u;",
            "            if (srcRow < p.y) {",
            "                ulong byteOff = (ulong)srcRow * (ulong)rowBytes + (ulong)t0 * 18u;",
            "                v = words[byteOff / 4u + (ulong)w];",
            "            }",
            "            tile[r * PITCH + w] = v;",
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "        if (active) {",
            "            for (uint b = 0u; b < tb; b++) {",
            "                threadgroup const uchar* blk = rowTile + b * 18u;",
            "                ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);",
            "                float d = float(as_type<half>(dbits));",
            "                uint xb = (t0 + b) * 32u;",
            "                for (uint l = 0; l < 16u; l++) {",
            "                    uchar q = blk[2u + l];",
            "                    sum += d * (float(q & 0x0Fu) - 8.0f) * x[xb + l];",
            "                    sum += d * (float(q >> 4) - 8.0f) * x[xb + 16u + l];",
            "                }",
            "            }",
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    }",
            "    if (active) { y[row] = sum; }",
            "}",
        ]
        return lines.joined(separator: "\n") + "\n"
    }

    public static func sourceDigest(_ source: String) -> String {
        RouterSelectionKernel.sha256Hex(Data(source.utf8))
    }
}

// MARK: - Deterministic corpus

struct Q4WeightStageRNG {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func below(_ n: UInt64) -> UInt64 { next() % n }
}

public struct Q4WeightStageCase {
    public let name: String
    public let origin: String
    public let inDim: Int
    public let outDim: Int
    public let blocks: [UInt8]
    public let x: [Float]
    /// A predeclared unsupported shape: the candidate must decline before any dispatch.
    public let expectedDecline: Bool

    public var blocksPerRow: Int { inDim / Q4_0.valuesPerBlock }
}

public enum Q4WeightStageCorpus {
    public static let seed: UInt64 = 0x5157_5354_4147_4531
    public static let timingSeed: UInt64 = 0x5157_5354_4147_4532
    /// FP16 scale encodings exercised by the edge cases: +0, -0, +subnormal, -subnormal, +255.875, -255.875.
    public static let edgeScaleBits: [UInt16] = [0x0000, 0x8000, 0x0001, 0x8001, 0x5BFF, 0xDBFF]
    public static let edgeNibbleBytes: [UInt8] = [0x00, 0xFF, 0x0F, 0xF0]
    /// (name, inDim, outDim, scaleEdges, nibbleEdges)
    public static let syntheticShapes: [(String, Int, Int, Bool, Bool)] = [
        ("min-1x64", 64, 1, false, false),
        ("row-tail-33x64", 64, 33, false, false),
        ("row-tail-31x256", 256, 31, false, false),
        ("block-tail-100x704", 704, 100, false, false),
        ("block-tail-65x2112", 2112, 65, false, false),
        ("even-64x2816", 2816, 64, false, false),
        ("edge-scales-32x2816", 2816, 32, true, false),
        ("edge-nibbles-40x704", 704, 40, false, true),
        ("edge-both-33x256", 256, 33, true, true),
    ]
    /// Predeclared unsupported shapes (odd block count): the candidate must decline before dispatch.
    public static let unsupportedShapes: [(String, Int, Int)] = [("unsupported-odd-blocks-8x96", 96, 8)]
    public static let timingShapes: [(String, Int, Int)] = [("timing-2816x1024", 2816, 1024),
                                                            ("timing-2112x1024", 2112, 1024)]
    public static let realTensors = [
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.5.attn_k.weight",
    ]
    public static let realRows = 256
    public static let realTimingTensor = "blk.0.attn_output.weight"
    public static let realTimingRows = 1024

    /// Synthetic scales are finite FP16 with |d| < 1/16 (exponent field 0...10, random sign and mantissa), the
    /// magnitude range of real Q4_0 checkpoint scales, so the existing FP64 contract on the CONTROL is not dominated
    /// by cancellation among synthetic terms of up to 5e5. Edge rows: blocks with b % 8 in 0...3 carry +0, -0,
    /// +subnormal, -subnormal; only the LAST block of each edge row carries +/-255.875, so its large terms round over
    /// 32 terms instead of the whole row. Corpus only: no kernel, oracle or contract change.
    static func blocks(rows: Int, inDim: Int, rng: inout Q4WeightStageRNG, scaleEdges: Bool,
                       nibbleEdges: Bool) -> [UInt8] {
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        var out = [UInt8]()
        out.reserveCapacity(rows * blocksPerRow * Q4_0.bytesPerBlock)
        for index in 0..<(rows * blocksPerRow) {
            let row = index / blocksPerRow
            let b = index - row * blocksPerRow
            let bits: UInt16
            if scaleEdges, b == blocksPerRow - 1 {
                bits = row % 2 == 0 ? edgeScaleBits[4] : edgeScaleBits[5]
            } else if scaleEdges, b % 8 < 4 {
                bits = edgeScaleBits[b % 8]
            } else {
                bits = UInt16(rng.below(2)) << 15 | UInt16(rng.below(11)) << 10 | UInt16(rng.below(1024))
            }
            out.append(UInt8(bits & 0xFF))
            out.append(UInt8(bits >> 8))
            if nibbleEdges, index % 5 < edgeNibbleBytes.count {
                for _ in 0..<16 { out.append(edgeNibbleBytes[index % 5]) }
            } else {
                for _ in 0..<16 { out.append(UInt8(rng.below(256))) }
            }
        }
        return out
    }

    static func activation(_ n: Int, rng: inout Q4WeightStageRNG) -> [Float] {
        (0..<n).map { _ in Float(Int64(rng.below(2049)) - 1024) / 1024 }
    }

    public static func synthetic() -> [Q4WeightStageCase] {
        var rng = Q4WeightStageRNG(state: seed)
        var out = [Q4WeightStageCase]()
        for (name, inDim, outDim, scaleEdges, nibbleEdges) in syntheticShapes {
            let b = blocks(rows: outDim, inDim: inDim, rng: &rng, scaleEdges: scaleEdges, nibbleEdges: nibbleEdges)
            out.append(Q4WeightStageCase(name: name, origin: "synthetic", inDim: inDim, outDim: outDim, blocks: b,
                                         x: activation(inDim, rng: &rng), expectedDecline: false))
        }
        for (name, inDim, outDim) in unsupportedShapes {
            let b = blocks(rows: outDim, inDim: inDim, rng: &rng, scaleEdges: false, nibbleEdges: false)
            out.append(Q4WeightStageCase(name: name, origin: "synthetic", inDim: inDim, outDim: outDim, blocks: b,
                                         x: activation(inDim, rng: &rng), expectedDecline: true))
        }
        return out
    }

    public static func syntheticTiming() -> [Q4WeightStageCase] {
        var rng = Q4WeightStageRNG(state: timingSeed)
        return timingShapes.map { name, inDim, outDim in
            let b = blocks(rows: outDim, inDim: inDim, rng: &rng, scaleEdges: false, nibbleEdges: false)
            return Q4WeightStageCase(name: name, origin: "synthetic", inDim: inDim, outDim: outDim, blocks: b,
                                     x: activation(inDim, rng: &rng), expectedDecline: false)
        }
    }

    /// Bounded real rows of one 2-D Q4_0 tensor through the existing bounded pread path; nothing is hashed.
    public static func real(file: GGUFFile, tensor: String, rows: Int, name: String) throws -> Q4WeightStageCase {
        let info = try file.tensor(tensor)
        guard info.dims.count == 2, info.ggmlType == GGMLTensorType.q4_0.rawValue else {
            throw Q4WeightStageError(tensor + " is not a 2-D Q4_0 tensor")
        }
        let inDim = info.dims[0]
        let outDim = info.dims[1]
        let bounded = min(rows, outDim)
        let op = Q4OpIR(schema: Q4OpIR.supportedSchema, op: "q4_0.matvec", semantics: "dot(W_row, x)", inDim: inDim,
                        outDim: outDim, layout: "row-major-q4_0-blocks", quant: "q4_0", accumulate: "f32",
                        inputPrecision: "f32", tensor: tensor, view: nil, representativeRows: bounded)
        try op.validate()
        let bytes = try Q4ComponentInputs.weightBytes(op: op, file: file, overrideRows: bounded)
        var rng = Q4WeightStageRNG(state: seed ^ UInt64(inDim) ^ (UInt64(bounded) << 20))
        return Q4WeightStageCase(name: name, origin: tensor, inDim: inDim, outDim: bounded, blocks: bytes,
                                 x: activation(inDim, rng: &rng), expectedDecline: false)
    }
}

// MARK: - Tick checks

public struct Q4WeightStageTickChecks: Codable, Equatable, Sendable {
    public let resolveIdentical: Bool
    public let resolvedSizeExact: Bool
    public let noErrorValues: Bool
    public let noZeroSamples: Bool
    public let pairsMonotonic: Bool
    public let encodersOrdered: Bool
    public let laterThanPrevious: Bool

    public var all: Bool {
        resolveIdentical && resolvedSizeExact && noErrorValues && noZeroSamples && pairsMonotonic
            && encodersOrdered && laterThanPrevious
    }
}

public enum Q4WeightStageTicks {
    public static let sampleCount = 4
    public static let counterErrorValue = UInt64.max

    public static func analyze(first: Data?, second: Data?, previousMax: UInt64?) -> (samples: [UInt64]?,
                                                                                       checks: Q4WeightStageTickChecks) {
        func failed(_ identical: Bool, _ size: Bool) -> (samples: [UInt64]?, checks: Q4WeightStageTickChecks) {
            (nil, Q4WeightStageTickChecks(resolveIdentical: identical, resolvedSizeExact: size, noErrorValues: false,
                                          noZeroSamples: false, pairsMonotonic: false, encodersOrdered: false,
                                          laterThanPrevious: false))
        }
        guard let a = first, let b = second else { return failed(false, false) }
        let identical = a == b
        let size = a.count == sampleCount * 8
        guard identical, size else { return failed(identical, size) }
        let v: [UInt64] = a.withUnsafeBytes { raw in
            (0..<sampleCount).map { raw.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self) }
        }
        let noError = v.allSatisfy { $0 != counterErrorValue }
        let noZero = v.allSatisfy { $0 != 0 }
        let monotonic = noError && v[1] >= v[0] && v[3] >= v[2]
        let ordered = noError && v[2] >= v[1]
        let later = noError && (previousMax.map { (v.min() ?? 0) > $0 } ?? true)
        return (v, Q4WeightStageTickChecks(resolveIdentical: true, resolvedSizeExact: true, noErrorValues: noError,
                                           noZeroSamples: noZero, pairsMonotonic: monotonic, encodersOrdered: ordered,
                                           laterThanPrevious: later))
    }
}

// MARK: - Report

public enum Q4WeightStageStatus: String, Codable, Sendable {
    case correctnessPassed
    case timingChecksFailed
    case correctnessFailed
    case unsupported
    case commandBufferError
    case notEnded
    case aborted
}

public struct Q4WeightStageHardware: Codable, Equatable, Sendable {
    public let device: String
    public let controlThreadExecutionWidth: Int
    public let controlMaxTotalThreads: Int
    public let deviceMaxThreadgroupMemoryLength: Int
    public let deviceMaxThreadsPerThreadgroupWidth: Int
    public let supportsAtStageBoundary: Bool
    public let supportsAtDispatchBoundary: Bool
    public let timestampCounterSetFound: Bool
    public let sampleBufferCreated: Bool
}

public struct Q4WeightStageCaseResult: Codable, Equatable, Sendable {
    public let name: String
    public let origin: String
    public let inDim: Int
    public let outDim: Int
    public let blocksPerRow: Int
    public let expectedDecline: Bool
    public let declined: String?
    public let controlCompletion: String
    public let candidateCompletion: String?
    public let controlFinite: Bool
    public let candidateFinite: Bool
    public let controlFP64WorstRel: Double?
    public let candidateFP64WorstRel: Double?
    public let controlFP64Pass: Bool
    public let candidateFP64Pass: Bool
    public let bitIdentical: Bool
    public let mismatchRows: Int
    public let firstMismatchRow: Int?
    public let guardsIntact: Bool
    public let passed: Bool
    public let failure: String?
}

public struct Q4WeightStageTimingPair: Codable, Equatable, Sendable {
    public let order: String
    public let completion: String
    public let samplesRaw: [UInt64]?
    public let controlTicks: UInt64?
    public let candidateTicks: UInt64?
    public let checks: Q4WeightStageTickChecks
    public let bitIdentical: Bool
    public let guardsIntact: Bool
    /// Bounded-semaphore wall per arm command buffer (commit to observed completion); diagnostic envelope only.
    public let controlWallMicros: Double?
    public let candidateWallMicros: Double?
    /// gpuEndTime - gpuStartTime per arm command buffer: interval coverage, not busy time.
    public let controlCommandBufferIntervalSeconds: Double?
    public let candidateCommandBufferIntervalSeconds: Double?
}

public struct Q4WeightStageTimingShape: Codable, Equatable, Sendable {
    public let name: String
    public let origin: String
    public let inDim: Int
    public let outDim: Int
    public let warmupPairs: Int
    public let timedPairs: Int
    public let pairs: [Q4WeightStageTimingPair]
    public let controlMedianTicks: Double?
    public let candidateMedianTicks: Double?
    /// candidate median / control median of raw stage ticks over the 6 timed pairs; unitless, component-only, not
    /// busy time or gain. The second arm of a pair reads weights the first arm just read (cache warmth), so the
    /// per-order medians below are reported next to it; a disagreement between them is an order effect.
    public let rawRatioCandidateOverControl: Double?
    /// Median per-pair raw ratio candidate/control over the 3 timed pairs in each execution order.
    public let ratioMedianControlFirst: Double?
    public let ratioMedianCandidateFirst: Double?
    public let controlMedianWallMicros: Double?
    public let passed: Bool
    public let failures: [String]
}

public struct Q4WeightStageVariantResult: Codable, Equatable, Sendable {
    public let schedule: Q4WeightStageSchedule
    public let kernelName: String
    public let sourceSha256: String?
    public let compileError: String?
    public let pipelineThreadExecutionWidth: Int?
    public let pipelineMaxTotalThreads: Int?
    public let staticThreadgroupBytes: Int?
    public let cases: [Q4WeightStageCaseResult]
    public let correctnessPassed: Bool
    public let timing: [Q4WeightStageTimingShape]
    public let timingPassed: Bool?
}

public struct Q4WeightStageReport: Codable {
    public private(set) var kind = Q4WeightStageDiagnostics.kind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var timestampUnits = Q4WeightStageDiagnostics.timestampUnits
    public private(set) var notes = Q4WeightStageDiagnostics.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var controlFunction = Q4WeightStageEmitter.controlFunction
    public var controlSourceSha256: String? = nil
    public var controlLaunchWidth = Q4WeightStagePlan.controlLaunchWidth
    public var hardware: Q4WeightStageHardware? = nil
    public var corpusCases = 0
    public var realCases = 0
    public var variants: [Q4WeightStageVariantResult] = []
    public var commandBuffersCommitted = 0
    public var terminal = true
    public var error: String? = nil
    public var status: Q4WeightStageStatus = .aborted

    public init() {}
}

extension Q4WeightStageReport {
    func withContract() -> Q4WeightStageReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

// MARK: - GPU execution

struct Q4WeightStageOutcome {
    let completion: String
    let terminal: Bool
    let error: String?
    let wallMicros: Double
    let commandBufferIntervalSeconds: Double?
}

struct Q4WeightStageBuffers {
    let w: MTLBuffer
    let x: MTLBuffer
    let weightBytes: Int
    let inDim: Int
    let outDim: Int
}

final class Q4WeightStageGPU {
    static let guardBytes = 256
    static let canaryByte: UInt8 = 0xA5
    static let timeoutSeconds = 10.0

    let device: MTLDevice
    let queue: MTLCommandQueue
    let control: MTLComputePipelineState
    var retained: [MTLBuffer] = []
    var commandBuffers: [MTLCommandBuffer] = []

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else { throw Q4WeightStageError("no command queue") }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil) } catch {
            throw Q4WeightStageError("control library compile (options nil): " + String(describing: error))
        }
        guard let fn = library.makeFunction(name: Q4WeightStageEmitter.controlFunction) else {
            throw Q4WeightStageError("control function missing")
        }
        self.device = device
        self.queue = queue
        self.control = try device.makeComputePipelineState(function: fn)
    }

    func compile(_ schedule: Q4WeightStageSchedule) throws -> (pipeline: MTLComputePipelineState, source: String) {
        let source = try Q4WeightStageEmitter.emit(schedule)
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) } catch {
            throw Q4WeightStageError("candidate library compile (options nil): " + String(describing: error))
        }
        guard let fn = library.makeFunction(name: schedule.kernelName) else {
            throw Q4WeightStageError("candidate function missing")
        }
        return (try device.makeComputePipelineState(function: fn), source)
    }

    func makeBuffers(_ c: Q4WeightStageCase) throws -> Q4WeightStageBuffers {
        let g = Self.guardBytes
        guard let w = device.makeBuffer(length: 2 * g + c.blocks.count, options: .storageModeShared),
              let x = device.makeBuffer(length: 2 * g + c.x.count * 4, options: .storageModeShared) else {
            throw Q4WeightStageError("input buffer allocation failed")
        }
        memset(w.contents(), Int32(Self.canaryByte), w.length)
        memset(x.contents(), Int32(Self.canaryByte), x.length)
        c.blocks.withUnsafeBytes { raw in _ = memcpy(w.contents().advanced(by: g), raw.baseAddress!, raw.count) }
        c.x.withUnsafeBytes { raw in _ = memcpy(x.contents().advanced(by: g), raw.baseAddress!, raw.count) }
        retained.append(w)
        retained.append(x)
        return Q4WeightStageBuffers(w: w, x: x, weightBytes: c.blocks.count, inDim: c.inDim, outDim: c.outDim)
    }

    /// Guards 0xA5 outside the payload; the payload is prefilled with 0xFF bytes (NaN) so an unwritten row fails.
    func makeOutput(outDim: Int) throws -> MTLBuffer {
        let g = Self.guardBytes
        guard let y = device.makeBuffer(length: 2 * g + outDim * 4, options: .storageModeShared) else {
            throw Q4WeightStageError("output buffer allocation failed")
        }
        resetOutput(y, outDim: outDim)
        retained.append(y)
        return y
    }

    func resetOutput(_ y: MTLBuffer, outDim: Int) {
        memset(y.contents(), Int32(Self.canaryByte), y.length)
        memset(y.contents().advanced(by: Self.guardBytes), 0xFF, outDim * 4)
    }

    func guardsIntact(_ b: MTLBuffer, payloadBytes: Int) -> Bool {
        let g = Self.guardBytes
        let all = UnsafeRawBufferPointer(start: b.contents(), count: b.length)
        for i in 0..<g where all[i] != Self.canaryByte || all[g + payloadBytes + i] != Self.canaryByte { return false }
        return true
    }

    func outputBits(_ y: MTLBuffer, outDim: Int) -> [UInt32] {
        let p = y.contents().advanced(by: Self.guardBytes).assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: p, count: outDim))
    }

    func encode(_ enc: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, launch: Q4WeightStageLaunch,
                b: Q4WeightStageBuffers, y: MTLBuffer) {
        var p = SIMD2<UInt32>(UInt32(b.inDim), UInt32(b.outDim))
        var off = UInt64(Self.guardBytes)
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(b.w, offset: 0, index: 2)
        enc.setBuffer(b.x, offset: Self.guardBytes, index: 3)
        enc.setBuffer(y, offset: Self.guardBytes, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: launch.threadgroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: launch.threadsPerThreadgroup, height: 1, depth: 1))
    }

    /// Commit and observe completion through a bounded semaphore; never waitUntilCompleted; nothing throws after commit.
    func submit(_ cb: MTLCommandBuffer) -> Q4WeightStageOutcome {
        let done = DispatchSemaphore(value: 0)
        cb.addCompletedHandler { _ in done.signal() }
        commandBuffers.append(cb)
        let t0 = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        if done.wait(timeout: .now() + Self.timeoutSeconds) == .timedOut {
            return Q4WeightStageOutcome(completion: "NOT_ENDED", terminal: false,
                                        error: "completion not observed within the bound", wallMicros: -1,
                                        commandBufferIntervalSeconds: nil)
        }
        let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000
        let interval: Double? = cb.gpuEndTime > cb.gpuStartTime ? cb.gpuEndTime - cb.gpuStartTime : nil
        guard cb.status == .completed, cb.error == nil else {
            let why = cb.error.map { String(describing: $0) } ?? ("status " + String(cb.status.rawValue))
            return Q4WeightStageOutcome(completion: "ERROR", terminal: true, error: why, wallMicros: wall,
                                        commandBufferIntervalSeconds: interval)
        }
        return Q4WeightStageOutcome(completion: "COMPLETED", terminal: true, error: nil, wallMicros: wall,
                                    commandBufferIntervalSeconds: interval)
    }

    func runSingle(pipeline: MTLComputePipelineState, launch: Q4WeightStageLaunch, b: Q4WeightStageBuffers,
                   y: MTLBuffer) -> Q4WeightStageOutcome {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return Q4WeightStageOutcome(completion: "NOT_COMMITTED", terminal: true,
                                        error: "command buffer or encoder unavailable", wallMicros: -1,
                                        commandBufferIntervalSeconds: nil)
        }
        encode(enc, pipeline: pipeline, launch: launch, b: b, y: y)
        enc.endEncoding()
        return submit(cb)
    }

    /// One timing pair: up to two arm command buffers in execution order plus the two resolves.
    struct PairOutcome {
        let arms: [Q4WeightStageOutcome]
        let firstResolve: Data?
        let secondResolve: Data?

        var terminal: Bool { arms.allSatisfy { $0.terminal } }
        var completion: String {
            if !terminal { return "NOT_ENDED" }
            if arms.contains(where: { $0.completion == "ERROR" }) { return "ERROR" }
            if arms.count == 2, arms.allSatisfy({ $0.completion == "COMPLETED" }) { return "COMPLETED" }
            return "NOT_COMMITTED"
        }
    }

    /// One timing pair: TWO command buffers, each with ONE sampled serial single-dispatch encoder. The first arm's
    /// command buffer is committed and its terminal completion observed (bounded semaphore) BEFORE the second arm is
    /// encoded and committed, so the two stage intervals cannot overlap whatever the cross-encoder scheduling
    /// (dispatchType .serial orders dispatches inside one encoder only). The first arm in execution order samples
    /// indices 0/1 and the second 2/3; both resolves run only after both command buffers completed. No extra copy,
    /// barrier or dispatch is added to either arm.
    func runPair(controlFirst: Bool, b: Q4WeightStageBuffers, yControl: MTLBuffer, yCandidate: MTLBuffer,
                 candidate: MTLComputePipelineState, candidateLaunch: Q4WeightStageLaunch,
                 controlLaunch: Q4WeightStageLaunch, sampleBuffer: MTLCounterSampleBuffer) -> PairOutcome {
        let arms: [(MTLComputePipelineState, Q4WeightStageLaunch, MTLBuffer)] = controlFirst
            ? [(control, controlLaunch, yControl), (candidate, candidateLaunch, yCandidate)]
            : [(candidate, candidateLaunch, yCandidate), (control, controlLaunch, yControl)]
        func notCommitted(_ why: String) -> Q4WeightStageOutcome {
            Q4WeightStageOutcome(completion: "NOT_COMMITTED", terminal: true, error: why, wallMicros: -1,
                                 commandBufferIntervalSeconds: nil)
        }
        var outcomes = [Q4WeightStageOutcome]()
        for (e, arm) in arms.enumerated() {
            guard let cb = queue.makeCommandBuffer() else {
                outcomes.append(notCommitted("command buffer unavailable"))
                break
            }
            let pass = MTLComputePassDescriptor()
            pass.dispatchType = .serial
            guard let attachment = pass.sampleBufferAttachments[0] else {
                outcomes.append(notCommitted("sample attachment 0 unavailable"))
                break
            }
            attachment.sampleBuffer = sampleBuffer
            attachment.startOfEncoderSampleIndex = 2 * e
            attachment.endOfEncoderSampleIndex = 2 * e + 1
            guard let enc = cb.makeComputeCommandEncoder(descriptor: pass) else {
                outcomes.append(notCommitted("sampled encoder unavailable"))
                break
            }
            encode(enc, pipeline: arm.0, launch: arm.1, b: b, y: arm.2)
            enc.endEncoding()
            let outcome = submit(cb)
            outcomes.append(outcome)
            if !(outcome.terminal && outcome.completion == "COMPLETED") { break }
        }
        let pending = PairOutcome(arms: outcomes, firstResolve: nil, secondResolve: nil)
        guard pending.completion == "COMPLETED" else { return pending }
        let first = try? sampleBuffer.resolveCounterRange(0..<Q4WeightStageTicks.sampleCount)
        let second = try? sampleBuffer.resolveCounterRange(0..<Q4WeightStageTicks.sampleCount)
        return PairOutcome(arms: outcomes, firstResolve: first ?? nil, secondResolve: second ?? nil)
    }
}

// MARK: - Driver

public enum Q4WeightStageDiagnostics {
    public static let kind = "q4-packed-weight-staging-component-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 16 << 20
    public static let correctnessContract = 1e-4
    public static let warmupPairs = 2
    public static let timedPairs = 6
    public static let timestampUnits = "raw GPU timestamp ticks as resolved at stage boundaries; the installed SDK "
        + "declarations state no unit; nothing is converted; intervals are not busy time"
    public static let usage = "usage: q4-weight-stage-component --report NEW_REPORT.json [--gguf M.gguf]"

    public static let notes: [String] = [
        "Component only: fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim "
            + "are always false; no default, kernel, guard or artifact changes.",
        "Control is the production g4_q4_matvec from Gemma4Runner.shaderSource compiled with options nil; candidates "
            + "are compiled with options nil. Identical source does not guarantee identical compiled bits; bit identity "
            + "is measured on every row of every case.",
        "Correctness gates per case and variant: finite outputs, existing 1e-4 FP64 contract for control and candidate "
            + "(a control failure is retained), Float bit identity candidate versus control, intact 256-byte guards, and "
            + "a pre-dispatch decline for the predeclared unsupported odd-block shape.",
        "Timing (only after a variant passes every case): per shape 2 warmup pairs then 6 timed pairs in alternating "
            + "execution order. Each pair runs its two arms in TWO sequential command buffers, each with one sampled "
            + "serial single-dispatch encoder; the second is encoded and committed only after the first was observed "
            + "terminal, so the stage intervals cannot overlap (dispatchType .serial orders dispatches within one "
            + "encoder, not independent encoders). Samples 0/1 then 2/3 are resolved twice after both are terminal; "
            + "raw ticks, a raw ratio and per-order ratios only. The second arm reads weights the first just read, so "
            + "an order effect is reported, not removed.",
        "Real tensor rows are bounded reads of the local GGUF file; no weight is hashed, serialized or uploaded "
            + "anywhere but the local GPU buffer of this process.",
        "Motivation is not measurement: phase 59 found 3221 of 125457 adjacent encoder ranges overlapping, so its "
            + "raw family medians cannot be summed into GPU shares, and per-encoder fixed-overhead dominance of small "
            + "families is a hypothesis. This component's raw ratio is a single-kernel stage-tick comparison and "
            + "implies no full-model gain.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public struct Options: Equatable, Sendable {
        public let report: String
        public let gguf: String?
    }

    public static func parseOptions(_ argv: [String]) throws -> Options {
        var v = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard key == "--report" || key == "--gguf", i + 1 < argv.count, v[key] == nil else {
                throw Q4WeightStageError(usage)
            }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw Q4WeightStageError(usage) }
            v[key] = value
            i += 2
        }
        guard let report = v["--report"] else { throw Q4WeightStageError(usage) }
        return Options(report: report, gguf: v["--gguf"])
    }

    public static func encodeReport(_ r: Q4WeightStageReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(r.withContract())
    }

    public static func exitCode(_ s: Q4WeightStageStatus) -> Int32 {
        switch s {
        case .correctnessPassed: return 0
        case .timingChecksFailed: return 6
        case .correctnessFailed: return 3
        case .unsupported: return 2
        case .commandBufferError: return 4
        case .notEnded: return 5
        case .aborted: return 1
        }
    }

    public static func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty, v.allSatisfy({ $0.isFinite }) else { return nil }
        let s = v.sorted()
        return s.count.isMultiple(of: 2) ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2 : s[s.count / 2]
    }

    static func worstRel(_ bits: [UInt32], _ ref: [Double]) -> (worst: Double?, finite: Bool) {
        guard !bits.isEmpty, bits.count == ref.count else { return (nil, false) }
        var worst = 0.0
        for (i, b) in bits.enumerated() {
            let y = Float(bitPattern: b)
            guard y.isFinite else { return (nil, false) }
            worst = max(worst, abs(Double(y) - ref[i]) / max(1.0, abs(ref[i])))
        }
        return (worst, true)
    }

    public static func finalStatus(_ r: Q4WeightStageReport) -> Q4WeightStageStatus {
        if !r.terminal { return .notEnded }
        if r.error != nil { return .aborted }
        guard r.hardware != nil, !r.variants.isEmpty,
              r.variants.contains(where: { $0.compileError == nil }) else { return .unsupported }
        let cbError = r.variants.contains { v in
            v.cases.contains { $0.controlCompletion == "ERROR" || $0.candidateCompletion == "ERROR" }
                || v.timing.contains { $0.pairs.contains { $0.completion == "ERROR" } }
        }
        if cbError { return .commandBufferError }
        guard r.variants.allSatisfy({ $0.correctnessPassed }) else { return .correctnessFailed }
        if r.variants.contains(where: { $0.timingPassed == false }) { return .timingChecksFailed }
        // Correctness passed but timing never ran (no stage sampling / sample buffer): not a pass of both.
        if r.variants.contains(where: { $0.timingPassed == nil }) { return .unsupported }
        return .correctnessPassed
    }

    static func failedVariant(_ schedule: Q4WeightStageSchedule, _ why: String) -> Q4WeightStageVariantResult {
        Q4WeightStageVariantResult(schedule: schedule, kernelName: schedule.kernelName, sourceSha256: nil,
                                   compileError: why, pipelineThreadExecutionWidth: nil, pipelineMaxTotalThreads: nil,
                                   staticThreadgroupBytes: nil, cases: [], correctnessPassed: false, timing: [],
                                   timingPassed: nil)
    }

    /// Never throws; every failure is recorded. Called only after GO.
    /// Holds the committed command-buffer count across the inner run's defer (a defer cannot update an already
    /// returned struct copy).
    final class CommitCounter {
        var committed = 0
    }

    public static func run(ggufURL: URL?, modelSha256: String?, executableDigest: String,
                           emit: (String) -> Void) -> Q4WeightStageReport {
        let counter = CommitCounter()
        var report = runCore(ggufURL: ggufURL, modelSha256: modelSha256, executableDigest: executableDigest,
                             counter: counter, emit: emit)
        report.commandBuffersCommitted = counter.committed
        return report
    }

    static func runCore(ggufURL: URL?, modelSha256: String?, executableDigest: String, counter: CommitCounter,
                        emit: (String) -> Void) -> Q4WeightStageReport {
        var report = Q4WeightStageReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.controlSourceSha256 = Q4WeightStageEmitter.sourceDigest(Gemma4Runner.shaderSource)
        guard let device = MTLCreateSystemDefaultDevice() else {
            report.error = "no Metal device"
            report.status = finalStatus(report)
            return report
        }
        let gpu: Q4WeightStageGPU
        do { gpu = try Q4WeightStageGPU(device: device) } catch {
            report.error = String(describing: error)
            report.status = finalStatus(report)
            return report
        }
        defer { counter.committed = gpu.commandBuffers.count }
        let atStage = device.supportsCounterSampling(.atStageBoundary)
        let atDispatch = device.supportsCounterSampling(.atDispatchBoundary)
        var timestampSet: MTLCounterSet? = nil
        for set in device.counterSets ?? [] where set.name == MTLCommonCounterSet.timestamp.rawValue {
            if set.counters.contains(where: { $0.name == MTLCommonCounter.timestamp.rawValue }) { timestampSet = set }
        }
        var sampleBuffer: MTLCounterSampleBuffer? = nil
        if atStage, let set = timestampSet {
            let d = MTLCounterSampleBufferDescriptor()
            d.counterSet = set
            d.label = "q4-weight-stage-component"
            d.storageMode = .shared
            d.sampleCount = Q4WeightStageTicks.sampleCount
            sampleBuffer = try? device.makeCounterSampleBuffer(descriptor: d)
        }
        report.hardware = Q4WeightStageHardware(
            device: device.name, controlThreadExecutionWidth: gpu.control.threadExecutionWidth,
            controlMaxTotalThreads: gpu.control.maxTotalThreadsPerThreadgroup,
            deviceMaxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            deviceMaxThreadsPerThreadgroupWidth: device.maxThreadsPerThreadgroup.width,
            supportsAtStageBoundary: atStage, supportsAtDispatchBoundary: atDispatch,
            timestampCounterSetFound: timestampSet != nil, sampleBufferCreated: sampleBuffer != nil)
        emit("device " + device.name + ": control width " + String(gpu.control.threadExecutionWidth)
             + ", max threads " + String(gpu.control.maxTotalThreadsPerThreadgroup) + ", stage sampling "
             + String(sampleBuffer != nil))

        // Corpus: fixed synthetic cases first, then bounded real rows when a GGUF is given.
        var cases = Q4WeightStageCorpus.synthetic()
        var timingCases = Q4WeightStageCorpus.syntheticTiming()
        if let url = ggufURL {
            do {
                let file = try GGUFReader.read(url)
                for (i, t) in Q4WeightStageCorpus.realTensors.enumerated() {
                    cases.append(try Q4WeightStageCorpus.real(file: file, tensor: t, rows: Q4WeightStageCorpus.realRows,
                                                              name: "real-" + String(i) + "-" + t))
                    report.realCases += 1
                }
                timingCases.append(try Q4WeightStageCorpus.real(
                    file: file, tensor: Q4WeightStageCorpus.realTimingTensor,
                    rows: Q4WeightStageCorpus.realTimingRows, name: "timing-real-" + Q4WeightStageCorpus.realTimingTensor))
            } catch {
                report.error = "real corpus: " + String(describing: error)
                report.status = finalStatus(report)
                return report
            }
        }
        report.corpusCases = cases.count

        // Control outputs and FP64 references, shared by both variants.
        struct ControlRun {
            let buffers: Q4WeightStageBuffers
            let outcome: Q4WeightStageOutcome
            let bits: [UInt32]
            let guards: Bool
            let ref: [Double]?
            let refError: String?
        }
        var controls = [ControlRun]()
        for c in cases {
            let b: Q4WeightStageBuffers
            let y: MTLBuffer
            do {
                b = try gpu.makeBuffers(c)
                y = try gpu.makeOutput(outDim: c.outDim)
            } catch {
                report.error = "allocation: " + String(describing: error)
                report.status = finalStatus(report)
                return report
            }
            let launch = Q4WeightStagePlan.controlLaunch(outDim: c.outDim,
                                                         pipelineMaxThreads: gpu.control.maxTotalThreadsPerThreadgroup)
            let outcome = gpu.runSingle(pipeline: gpu.control, launch: launch, b: b, y: y)
            if !outcome.terminal {
                report.terminal = false
                report.error = "control " + c.name + ": " + (outcome.error ?? "not ended")
                report.status = finalStatus(report)
                return report
            }
            var ref: [Double]? = nil
            var refError: String? = nil
            do { ref = try Q4Oracle.matvecFP64(blocks: c.blocks[...], outDim: c.outDim, inDim: c.inDim, x: c.x) } catch {
                refError = String(describing: error)
            }
            let bits = outcome.completion == "COMPLETED" ? gpu.outputBits(y, outDim: c.outDim) : []
            let guards = gpu.guardsIntact(b.w, payloadBytes: b.weightBytes)
                && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4) && gpu.guardsIntact(y, payloadBytes: c.outDim * 4)
            controls.append(ControlRun(buffers: b, outcome: outcome, bits: bits, guards: guards, ref: ref,
                                       refError: refError))
        }

        for blocksPerTile in Q4WeightStageSchedule.fixedBlockTiles {
            let schedule: Q4WeightStageSchedule
            do {
                schedule = try Q4WeightStageSchedule.derive(blocksPerTile: blocksPerTile,
                                                           rowsPerTile: gpu.control.threadExecutionWidth)
            } catch {
                let placeholder = Q4WeightStageSchedule(rowsPerTile: 8, blocksPerTile: blocksPerTile,
                                                        segmentBytes: blocksPerTile * 18,
                                                        segmentWords: blocksPerTile * 18 / 4, rowPitchWords: 1,
                                                        paddedPitch: false)
                report.variants.append(failedVariant(placeholder, "schedule derivation: " + String(describing: error)))
                continue
            }
            let compiled: (pipeline: MTLComputePipelineState, source: String)
            do { compiled = try gpu.compile(schedule) } catch {
                report.variants.append(failedVariant(schedule, String(describing: error)))
                continue
            }
            let pipe = compiled.pipeline
            let sourceSha = Q4WeightStageEmitter.sourceDigest(compiled.source)
            var results = [Q4WeightStageCaseResult]()
            var allPassed = true
            for (i, c) in cases.enumerated() {
                let ctl = controls[i]
                let decline = Q4WeightStagePlan.declineReason(
                    inDim: c.inDim, outDim: c.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
                    weightLength: ctl.buffers.w.length, schedule: schedule,
                    executionWidth: pipe.threadExecutionWidth, pipelineMaxThreads: pipe.maxTotalThreadsPerThreadgroup,
                    staticThreadgroupBytes: pipe.staticThreadgroupMemoryLength,
                    deviceThreadgroupCap: device.maxThreadgroupMemoryLength)
                let (ctlWorst, ctlFinite) = ctl.ref.map { worstRel(ctl.bits, $0) } ?? (nil, false)
                let ctlPass = ctlFinite && ctl.outcome.completion == "COMPLETED" && ctl.guards
                    && (ctlWorst.map { $0 < correctnessContract } ?? false)
                var f = [String]()
                if ctl.outcome.completion != "COMPLETED" { f.append("control " + ctl.outcome.completion) }
                if let e = ctl.refError { f.append("FP64 oracle: " + e) }
                if !ctlPass { f.append("control outside the FP64 contract, non-finite or guard violation") }
                if let reason = decline {
                    if !c.expectedDecline { f.append("unexpected decline: " + reason) }
                    results.append(Q4WeightStageCaseResult(
                        name: c.name, origin: c.origin, inDim: c.inDim, outDim: c.outDim, blocksPerRow: c.blocksPerRow,
                        expectedDecline: c.expectedDecline, declined: reason,
                        controlCompletion: ctl.outcome.completion, candidateCompletion: nil, controlFinite: ctlFinite,
                        candidateFinite: false, controlFP64WorstRel: ctlWorst, candidateFP64WorstRel: nil,
                        controlFP64Pass: ctlPass, candidateFP64Pass: false, bitIdentical: false, mismatchRows: 0,
                        firstMismatchRow: nil, guardsIntact: ctl.guards, passed: f.isEmpty,
                        failure: f.isEmpty ? nil : f.joined(separator: "; ")))
                    if !f.isEmpty { allPassed = false }
                    continue
                }
                if c.expectedDecline { f.append("predeclared unsupported shape was not declined") }
                let y: MTLBuffer
                do { y = try gpu.makeOutput(outDim: c.outDim) } catch {
                    report.error = "allocation: " + String(describing: error)
                    report.status = finalStatus(report)
                    return report
                }
                let launch = Q4WeightStagePlan.launch(outDim: c.outDim, schedule: schedule)
                let outcome = gpu.runSingle(pipeline: pipe, launch: launch, b: ctl.buffers, y: y)
                if !outcome.terminal {
                    report.terminal = false
                    report.error = "candidate " + schedule.kernelName + " " + c.name + ": " + (outcome.error ?? "not ended")
                    report.status = finalStatus(report)
                    return report
                }
                let bits = outcome.completion == "COMPLETED" ? gpu.outputBits(y, outDim: c.outDim) : []
                let guards = ctl.guards && gpu.guardsIntact(ctl.buffers.w, payloadBytes: ctl.buffers.weightBytes)
                    && gpu.guardsIntact(ctl.buffers.x, payloadBytes: c.inDim * 4)
                    && gpu.guardsIntact(y, payloadBytes: c.outDim * 4)
                let (candWorst, candFinite) = ctl.ref.map { worstRel(bits, $0) } ?? (nil, false)
                let candPass = candFinite && outcome.completion == "COMPLETED"
                    && (candWorst.map { $0 < correctnessContract } ?? false)
                var mismatches = 0
                var firstMismatch: Int? = nil
                if bits.count == ctl.bits.count {
                    for r in 0..<bits.count where bits[r] != ctl.bits[r] {
                        mismatches += 1
                        if firstMismatch == nil { firstMismatch = r }
                    }
                } else {
                    mismatches = max(bits.count, ctl.bits.count)
                }
                let identical = outcome.completion == "COMPLETED" && ctl.outcome.completion == "COMPLETED"
                    && bits.count == ctl.bits.count && mismatches == 0
                if outcome.completion != "COMPLETED" { f.append("candidate " + outcome.completion) }
                if !candFinite { f.append("candidate output not finite") }
                if !candPass { f.append("candidate outside the FP64 contract") }
                if !identical { f.append(String(mismatches) + " row(s) differ in Float bits from the control") }
                if !guards { f.append("guard bytes violated") }
                results.append(Q4WeightStageCaseResult(
                    name: c.name, origin: c.origin, inDim: c.inDim, outDim: c.outDim, blocksPerRow: c.blocksPerRow,
                    expectedDecline: c.expectedDecline, declined: nil, controlCompletion: ctl.outcome.completion,
                    candidateCompletion: outcome.completion, controlFinite: ctlFinite, candidateFinite: candFinite,
                    controlFP64WorstRel: ctlWorst, candidateFP64WorstRel: candWorst, controlFP64Pass: ctlPass,
                    candidateFP64Pass: candPass, bitIdentical: identical, mismatchRows: mismatches,
                    firstMismatchRow: firstMismatch, guardsIntact: guards, passed: f.isEmpty,
                    failure: f.isEmpty ? nil : f.joined(separator: "; ")))
                if !f.isEmpty { allPassed = false }
            }
            emit("variant " + schedule.kernelName + ": " + String(results.filter { $0.passed }.count) + " of "
                 + String(results.count) + " cases passed")
            var timing = [Q4WeightStageTimingShape]()
            var timingPassed: Bool? = nil
            if allPassed, let sb = sampleBuffer {
                var okAll = true
                for tc in timingCases {
                    let shape = timedShape(gpu: gpu, device: device, schedule: schedule, pipeline: pipe, sampleBuffer: sb,
                                           tc: tc)
                    guard let s = shape.shape else {
                        report.terminal = false
                        report.error = "timing " + tc.name + ": completion not observed"
                        report.status = finalStatus(report)
                        return report
                    }
                    timing.append(s)
                    if !s.passed { okAll = false }
                    emit("timing " + schedule.kernelName + " " + tc.name + ": raw ratio "
                         + (s.rawRatioCandidateOverControl.map { String($0) } ?? "n/a") + " (raw ticks, not busy time)")
                }
                timingPassed = okAll
            }
            report.variants.append(Q4WeightStageVariantResult(
                schedule: schedule, kernelName: schedule.kernelName, sourceSha256: sourceSha, compileError: nil,
                pipelineThreadExecutionWidth: pipe.threadExecutionWidth,
                pipelineMaxTotalThreads: pipe.maxTotalThreadsPerThreadgroup,
                staticThreadgroupBytes: pipe.staticThreadgroupMemoryLength, cases: results,
                correctnessPassed: allPassed, timing: timing, timingPassed: timingPassed))
        }
        report.status = finalStatus(report)
        return report
    }

    /// Fixed protocol for one shape: 2 warmup pairs then 6 timed pairs, alternating execution order (even pairs run
    /// the control first). Each pair runs its two arms in two sequential command buffers (see runPair), so the two
    /// stage intervals never overlap. Returns a nil shape when a command buffer never completed within the bound (the
    /// caller marks the run non-terminal and withholds END).
    static func timedShape(gpu: Q4WeightStageGPU, device: MTLDevice, schedule: Q4WeightStageSchedule,
                           pipeline: MTLComputePipelineState, sampleBuffer: MTLCounterSampleBuffer,
                           tc: Q4WeightStageCase) -> (shape: Q4WeightStageTimingShape?, terminal: Bool) {
        func notTimed(_ why: String) -> Q4WeightStageTimingShape {
            Q4WeightStageTimingShape(name: tc.name, origin: tc.origin, inDim: tc.inDim, outDim: tc.outDim,
                                     warmupPairs: warmupPairs, timedPairs: timedPairs, pairs: [],
                                     controlMedianTicks: nil, candidateMedianTicks: nil,
                                     rawRatioCandidateOverControl: nil, ratioMedianControlFirst: nil,
                                     ratioMedianCandidateFirst: nil, controlMedianWallMicros: nil, passed: false,
                                     failures: [why])
        }
        if let reason = Q4WeightStagePlan.declineReason(
            inDim: tc.inDim, outDim: tc.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
            weightLength: 2 * Q4WeightStageGPU.guardBytes + tc.blocks.count, schedule: schedule,
            executionWidth: pipeline.threadExecutionWidth, pipelineMaxThreads: pipeline.maxTotalThreadsPerThreadgroup,
            staticThreadgroupBytes: pipeline.staticThreadgroupMemoryLength,
            deviceThreadgroupCap: device.maxThreadgroupMemoryLength) {
            return (notTimed("declined: " + reason), true)
        }
        let b: Q4WeightStageBuffers
        let yControl: MTLBuffer
        let yCandidate: MTLBuffer
        do {
            b = try gpu.makeBuffers(tc)
            yControl = try gpu.makeOutput(outDim: tc.outDim)
            yCandidate = try gpu.makeOutput(outDim: tc.outDim)
        } catch {
            return (notTimed("allocation: " + String(describing: error)), true)
        }
        let controlLaunch = Q4WeightStagePlan.controlLaunch(outDim: tc.outDim,
                                                            pipelineMaxThreads: gpu.control.maxTotalThreadsPerThreadgroup)
        let candidateLaunch = Q4WeightStagePlan.launch(outDim: tc.outDim, schedule: schedule)
        var failures = [String]()
        var pairs = [Q4WeightStageTimingPair]()
        var previousMax: UInt64? = nil
        var controlTicks = [Double]()
        var candidateTicks = [Double]()
        var walls = [Double]()
        var ratiosControlFirst = [Double]()
        var ratiosCandidateFirst = [Double]()
        func wall(_ o: Q4WeightStageOutcome?) -> Double? {
            guard let o, o.completion == "COMPLETED" else { return nil }
            return o.wallMicros
        }
        for k in 0..<(warmupPairs + timedPairs) {
            let controlFirst = k % 2 == 0
            gpu.resetOutput(yControl, outDim: tc.outDim)
            gpu.resetOutput(yCandidate, outDim: tc.outDim)
            let pair = gpu.runPair(controlFirst: controlFirst, b: b, yControl: yControl, yCandidate: yCandidate,
                                   candidate: pipeline, candidateLaunch: candidateLaunch,
                                   controlLaunch: controlLaunch, sampleBuffer: sampleBuffer)
            if !pair.terminal { return (nil, false) }
            let completed = pair.completion == "COMPLETED"
            let (samples, checks) = Q4WeightStageTicks.analyze(first: pair.firstResolve, second: pair.secondResolve,
                                                               previousMax: previousMax)
            if let s = samples, checks.noErrorValues { previousMax = s.max() }
            let cBits = completed ? gpu.outputBits(yControl, outDim: tc.outDim) : []
            let kBits = completed ? gpu.outputBits(yCandidate, outDim: tc.outDim) : []
            let identical = !cBits.isEmpty && cBits == kBits && cBits.allSatisfy { Float(bitPattern: $0).isFinite }
            let guards = gpu.guardsIntact(b.w, payloadBytes: b.weightBytes)
                && gpu.guardsIntact(b.x, payloadBytes: tc.inDim * 4)
                && gpu.guardsIntact(yControl, payloadBytes: tc.outDim * 4)
                && gpu.guardsIntact(yCandidate, payloadBytes: tc.outDim * 4)
            var ctl: UInt64? = nil
            var cand: UInt64? = nil
            if completed, let s = samples, checks.all {
                let firstInterval = s[1] - s[0]
                let secondInterval = s[3] - s[2]
                ctl = controlFirst ? firstInterval : secondInterval
                cand = controlFirst ? secondInterval : firstInterval
            }
            let firstArm = pair.arms.first
            let secondArm: Q4WeightStageOutcome? = pair.arms.count > 1 ? pair.arms[1] : nil
            let controlArm = controlFirst ? firstArm : secondArm
            let candidateArm = controlFirst ? secondArm : firstArm
            pairs.append(Q4WeightStageTimingPair(
                order: controlFirst ? "control-then-candidate" : "candidate-then-control",
                completion: pair.completion, samplesRaw: samples, controlTicks: ctl, candidateTicks: cand,
                checks: checks, bitIdentical: identical, guardsIntact: guards,
                controlWallMicros: wall(controlArm), candidateWallMicros: wall(candidateArm),
                controlCommandBufferIntervalSeconds: controlArm?.commandBufferIntervalSeconds,
                candidateCommandBufferIntervalSeconds: candidateArm?.commandBufferIntervalSeconds))
            if !completed {
                failures.append("pair " + String(k) + " " + pair.completion + ": "
                                + (pair.arms.last?.error ?? "no arm committed"))
            }
            if !checks.all { failures.append("pair " + String(k) + " tick checks failed") }
            if !identical { failures.append("pair " + String(k) + " outputs not bit-identical") }
            if !guards { failures.append("pair " + String(k) + " guards violated") }
            if k >= warmupPairs, let c = ctl, let d = cand {
                controlTicks.append(Double(c))
                candidateTicks.append(Double(d))
                if let w = wall(controlArm) { walls.append(w) }
                if c > 0 {
                    let r = Double(d) / Double(c)
                    if controlFirst { ratiosControlFirst.append(r) } else { ratiosCandidateFirst.append(r) }
                }
            }
        }
        let cMed = controlTicks.count == timedPairs ? median(controlTicks) : nil
        let kMed = candidateTicks.count == timedPairs ? median(candidateTicks) : nil
        var ratio: Double? = nil
        if let c = cMed, let k = kMed, c > 0 { ratio = k / c }
        let perOrder = timedPairs / 2
        return (Q4WeightStageTimingShape(
            name: tc.name, origin: tc.origin, inDim: tc.inDim, outDim: tc.outDim, warmupPairs: warmupPairs,
            timedPairs: timedPairs, pairs: pairs, controlMedianTicks: cMed, candidateMedianTicks: kMed,
            rawRatioCandidateOverControl: ratio,
            ratioMedianControlFirst: ratiosControlFirst.count == perOrder ? median(ratiosControlFirst) : nil,
            ratioMedianCandidateFirst: ratiosCandidateFirst.count == perOrder ? median(ratiosCandidateFirst) : nil,
            controlMedianWallMicros: median(walls), passed: failures.isEmpty, failures: failures), true)
    }
}
