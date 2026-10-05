import Foundation
import Metal

// Phase 64: FIRST-DIVERGENCE DIAGNOSTIC for the frozen min-1x64 case. Not an optimization candidate; nothing here
// is installable or a production kernel. Phase 61 and phase 62 remain closed as correctness-rejected and are not
// rerun as candidates: the phase-62 emitter is executed here only as the uninstrumented object of the diagnosis.
//
// Steps (all on the SAME frozen Q4WeightStageCorpus min-1x64 case, no reseeding):
//  1. Reproduce: production g4_q4_matvec (Gemma4Runner.shaderSource, options nil) and the actual phase-62
//     Q4RegisterLoopEmitter source (options nil), both uninstrumented; record both Float bit patterns.
//  2. Unpack: instrumented copies of each decoder store the half scale bits, the quant bytes and the x operand bits
//     exactly as that decoder reads them; first difference per stage.
//  3. Trace: instrumented copies store the running sum after each serial add (sum-only variant, term lines kept
//     verbatim) and the sub-steps w, d*w, (d*w)*x and sum (split variant, altered arithmetic). First difference per
//     stage. A trace may be attributed to the originals ONLY if its final equals the respective uninstrumented
//     output; otherwise the report says ORIGINAL CAUSE NOT ESTABLISHED. Instrumentation may change compiled
//     contraction or scheduling; that is why the gate exists.
//  4. CPU accumulation models (strict / fma, left / right association) are evaluated in Swift Float and reported as
//     consistent-with hypotheses against the two uninstrumented finals, never as the cause.
// fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false.

public enum Q4FirstDivergenceShape: String, Codable, Sendable, CaseIterable {
    case production
    case register
}

public enum Q4FirstDivergenceVariant: String, Codable, Sendable, CaseIterable {
    case unpack
    case sumOnly
    case split
}

// MARK: - Trace layout

public enum Q4FirstDivergenceLayout {
    public static let splitStages = ["w", "dw", "p", "sum"]

    public static func unpackWords(blocksPerRow: Int, inDim: Int) -> Int { blocksPerRow * 17 + inDim }
    public static func scaleIndex(block b: Int) -> Int { b }
    public static func quantIndex(blocksPerRow: Int, block b: Int, byte l: Int) -> Int { blocksPerRow + b * 16 + l }
    public static func xIndex(blocksPerRow: Int, element i: Int) -> Int { blocksPerRow * 17 + i }
    /// Serial term order of the production kernel: block b, byte l, low nibble (h = 0) then high nibble (h = 1).
    public static func termIndex(block b: Int, byte l: Int, high h: Int) -> Int { b * 32 + 2 * l + h }

    public static func traceWords(inDim: Int, blocksPerRow: Int, variant: Q4FirstDivergenceVariant) -> Int {
        switch variant {
        case .unpack: return unpackWords(blocksPerRow: blocksPerRow, inDim: inDim)
        case .sumOnly: return inDim
        case .split: return 4 * inDim
        }
    }

    /// Stage-wise views of one trace buffer; empty when the word count is not the layout's.
    public static func stages(_ words: [UInt32], variant: Q4FirstDivergenceVariant, blocksPerRow: Int,
                              inDim: Int) -> [(name: String, values: [UInt32])] {
        switch variant {
        case .unpack:
            guard words.count == unpackWords(blocksPerRow: blocksPerRow, inDim: inDim) else { return [] }
            return [("scaleBits", Array(words[0..<blocksPerRow])),
                    ("quantBytes", Array(words[blocksPerRow..<(blocksPerRow * 17)])),
                    ("xBits", Array(words[(blocksPerRow * 17)...]))]
        case .sumOnly:
            guard words.count == inDim else { return [] }
            return [("sum", words)]
        case .split:
            guard words.count == 4 * inDim else { return [] }
            return splitStages.enumerated().map { s, name in
                (name, stride(from: s, to: words.count, by: 4).map { words[$0] })
            }
        }
    }

    public static func firstDifference(_ a: [UInt32], _ b: [UInt32]) -> Int? {
        if a.count != b.count { return min(a.count, b.count) }
        for i in 0..<a.count where a[i] != b[i] { return i }
        return nil
    }

    public static func locator(stage: String, index: Int) -> String {
        switch stage {
        case "scaleBits": return "block " + String(index)
        case "quantBytes": return "block " + String(index / 16) + " byte " + String(index % 16)
        case "xBits": return "x element " + String(index)
        default:
            let r = index % 32
            return "term " + String(index) + " = block " + String(index / 32) + " byte " + String(r / 2)
                + (r % 2 == 0 ? " low nibble" : " high nibble")
        }
    }

    public static func hex8(_ v: UInt32) -> String {
        let s = String(v, radix: 16)
        return String(repeating: "0", count: max(0, 8 - s.count)) + s
    }
}

// MARK: - CPU accumulation hypotheses (Swift Float, no implicit contraction)

public enum Q4FirstDivergenceCPU {
    public enum Model: String, Codable, Sendable, CaseIterable {
        /// p = (d * w) * x; sum = sum + p
        case strictLeftProductThenAdd
        /// sum = fma(d * w, x, sum)
        case fmaLeftProductAccumulate
        /// p = d * (w * x); sum = sum + p
        case strictRightProductThenAdd
        /// sum = fma(d, w * x, sum)
        case fmaRightProductAccumulate
    }

    /// Exact IEEE binary16 to binary32 conversion by integer manipulation (sign, subnormals, infinities; NaN payload
    /// not preserved, which the frozen case never exercises).
    public static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h >> 15) << 31
        let exp = Int((h >> 10) & 0x1F)
        let man = UInt32(h & 0x3FF)
        if exp == 0 {
            if man == 0 { return Float(bitPattern: sign) }
            let mag = Float(man) * Float(bitPattern: 0x3380_0000)
            return sign != 0 ? -mag : mag
        }
        if exp == 31 {
            if man != 0 { return Float.nan }
            return sign != 0 ? -Float.infinity : Float.infinity
        }
        return Float(bitPattern: sign | (UInt32(exp + 112) << 23) | (man << 13))
    }

    /// Running sums (bit patterns) after each of the inDim serial terms of row 0 under one model.
    public static func runningSums(model: Model, blocks: [UInt8], inDim: Int, x: [Float]) -> [UInt32]? {
        guard inDim > 0, inDim % 32 == 0 else { return nil }
        let blocksPerRow = inDim / 32
        guard blocks.count >= blocksPerRow * 18, x.count >= inDim else { return nil }
        var out = [UInt32]()
        out.reserveCapacity(inDim)
        var sum: Float = 0
        for b in 0..<blocksPerRow {
            let o = b * 18
            let dbits = UInt16(blocks[o]) | (UInt16(blocks[o + 1]) << 8)
            let d = halfToFloat(dbits)
            for l in 0..<16 {
                let q = blocks[o + 2 + l]
                for h in 0..<2 {
                    let nibble = h == 0 ? (q & 0x0F) : (q >> 4)
                    let w = Float(nibble) - 8
                    let xv = x[b * 32 + (h == 0 ? l : 16 + l)]
                    switch model {
                    case .strictLeftProductThenAdd: sum = sum + ((d * w) * xv)
                    case .fmaLeftProductAccumulate: sum = sum.addingProduct(d * w, xv)
                    case .strictRightProductThenAdd: sum = sum + (d * (w * xv))
                    case .fmaRightProductAccumulate: sum = sum.addingProduct(d, w * xv)
                    }
                    out.append(sum.bitPattern)
                }
            }
        }
        return out
    }
}

// MARK: - Instrumented kernel emitter (copies for one traced row; never production)

public enum Q4FirstDivergenceEmitter {
    public static let codegenVersion = "q4fd-v1"

    public static func kernelName(shape: Q4FirstDivergenceShape, variant: Q4FirstDivergenceVariant) -> String {
        "q4fd_" + shape.rawValue + "_" + variant.rawValue
    }

    public static func emit(shape: Q4FirstDivergenceShape, variant: Q4FirstDivergenceVariant) -> String {
        let name = kernelName(shape: shape, variant: variant)
        var lines: [String] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "// " + name + " - EMITTED by LeviathanForge Q4 first-divergence diagnostic (" + codegenVersion + ").",
            "// Instrumented copy for one traced row; never a production kernel. Trace stores may change compiled",
            "// contraction or scheduling; the driver checks the trace final against the uninstrumented output.",
            "kernel void " + name + "(constant const uint2& p [[buffer(0)]],",
            "        constant const ulong& wOff [[buffer(1)]],",
            "        device const uchar* wbase [[buffer(2)]],",
            "        device const float* x [[buffer(3)]],",
            "        device float* y [[buffer(4)]],",
            "        device uint* tr [[buffer(5)]],",
            "        uint gid [[thread_position_in_grid]]) {",
            "    uint row = gid;",
            "    if (row >= p.y) { return; }",
            "    uint blocksPerRow = p.x / 32u;",
        ]
        switch shape {
        case .production:
            lines += [
                "    device const uchar* rowBase = wbase + wOff + (ulong)row * (ulong)blocksPerRow * 18u;",
                "    float sum = 0.0f;",
                "    for (uint b = 0; b < blocksPerRow; b++) {",
                "        device const uchar* blk = rowBase + b * 18u;",
                "        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);",
                "        float d = float(as_type<half>(dbits));",
            ]
        case .register:
            lines += [
                "    uint pairsPerRow = blocksPerRow / 2u;",
                "    device const uint* words = (device const uint*)(wbase + wOff) + (ulong)row * (ulong)pairsPerRow * 9u;",
                "    float sum = 0.0f;",
                "    for (uint b = 0; b < blocksPerRow; b++) {",
                "        device const uint* pw = words + (ulong)(b >> 1u) * 9u;",
                "        uint b0; uint b1; uint b2; uint b3; uint b4;",
                "        if ((b & 1u) == 0u) {",
                "            b0 = pw[0]; b1 = pw[1]; b2 = pw[2]; b3 = pw[3]; b4 = pw[4];",
                "        } else {",
                "            uint t4 = pw[4]; uint t5 = pw[5]; uint t6 = pw[6]; uint t7 = pw[7]; uint t8 = pw[8];",
                "            b0 = (t4 >> 16u) | (t5 << 16u);",
                "            b1 = (t5 >> 16u) | (t6 << 16u);",
                "            b2 = (t6 >> 16u) | (t7 << 16u);",
                "            b3 = (t7 >> 16u) | (t8 << 16u);",
                "            b4 = t8 >> 16u;",
                "        }",
                "        ushort dbits = ushort(b0 & 0xFFFFu);",
                "        float d = float(as_type<half>(dbits));",
            ]
        }
        if variant == .unpack { lines.append("        if (row == 0u) { tr[b] = uint(dbits); }") }
        lines.append("        for (uint l = 0; l < 16u; l++) {")
        switch shape {
        case .production:
            lines.append("            uchar q = blk[2u + l];")
        case .register:
            lines += [
                "            uint bi = 2u + l;",
                "            uint wsel = bi >> 2u;",
                "            uint word = (wsel == 0u) ? b0 : (wsel == 1u) ? b1 : (wsel == 2u) ? b2 : (wsel == 3u) ? b3 : b4;",
                "            uchar q = uchar((word >> ((bi & 3u) * 8u)) & 0xFFu);",
            ]
        }
        switch variant {
        case .unpack:
            lines += [
                "            if (row == 0u) {",
                "                tr[blocksPerRow + b * 16u + l] = uint(q);",
                "                tr[blocksPerRow * 17u + b * 32u + l] = as_type<uint>(x[b * 32u + l]);",
                "                tr[blocksPerRow * 17u + b * 32u + 16u + l] = as_type<uint>(x[b * 32u + 16u + l]);",
                "            }",
            ]
        case .sumOnly:
            lines += [
                "            sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];",
                "            if (row == 0u) { tr[b * 32u + 2u * l] = as_type<uint>(sum); }",
                "            sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];",
                "            if (row == 0u) { tr[b * 32u + 2u * l + 1u] = as_type<uint>(sum); }",
            ]
        case .split:
            lines += [
                "            uint k0 = b * 32u + 2u * l;",
                "            float w0 = float(q & 0x0Fu) - 8.0f;",
                "            float dw0 = d * w0;",
                "            float p0 = dw0 * x[b * 32u + l];",
                "            sum += p0;",
                "            if (row == 0u) { tr[4u * k0] = as_type<uint>(w0); tr[4u * k0 + 1u] = as_type<uint>(dw0); tr[4u * k0 + 2u] = as_type<uint>(p0); tr[4u * k0 + 3u] = as_type<uint>(sum); }",
                "            uint k1 = k0 + 1u;",
                "            float w1 = float(q >> 4) - 8.0f;",
                "            float dw1 = d * w1;",
                "            float p1 = dw1 * x[b * 32u + 16u + l];",
                "            sum += p1;",
                "            if (row == 0u) { tr[4u * k1] = as_type<uint>(w1); tr[4u * k1 + 1u] = as_type<uint>(dw1); tr[4u * k1 + 2u] = as_type<uint>(p1); tr[4u * k1 + 3u] = as_type<uint>(sum); }",
            ]
        }
        lines += ["        }", "    }", "    y[row] = sum;", "}"]
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Report

public enum Q4FirstDivergenceStatus: String, Codable, Sendable {
    case completed
    case notReproduced
    case unsupported
    case commandBufferError
    case notEnded
    case aborted
}

public struct Q4FirstDivergenceStageDiff: Codable, Equatable, Sendable {
    public let stage: String
    public let count: Int
    public let identical: Bool
    public let firstIndex: Int?
    public let locator: String?
    public let productionHex: String?
    public let registerHex: String?

    public init(stage: String, count: Int, identical: Bool, firstIndex: Int?, locator: String?, productionHex: String?,
                registerHex: String?) {
        self.stage = stage
        self.count = count
        self.identical = identical
        self.firstIndex = firstIndex
        self.locator = locator
        self.productionHex = productionHex
        self.registerHex = registerHex
    }

    public static func make(stage: String, production: [UInt32], register: [UInt32]) -> Q4FirstDivergenceStageDiff {
        let i = Q4FirstDivergenceLayout.firstDifference(production, register)
        return Q4FirstDivergenceStageDiff(
            stage: stage, count: min(production.count, register.count), identical: i == nil, firstIndex: i,
            locator: i.map { Q4FirstDivergenceLayout.locator(stage: stage, index: $0) },
            productionHex: i.flatMap { $0 < production.count ? Q4FirstDivergenceLayout.hex8(production[$0]) : nil },
            registerHex: i.flatMap { $0 < register.count ? Q4FirstDivergenceLayout.hex8(register[$0]) : nil })
    }
}

public struct Q4FirstDivergenceVariantDiff: Codable, Equatable, Sendable {
    public let variant: String
    public let productionTraceFinalEqualsControl: Bool
    public let registerTraceFinalEqualsCandidate: Bool
    public let attributableToOriginals: Bool
    public let stages: [Q4FirstDivergenceStageDiff]

    public init(variant: String, productionTraceFinalEqualsControl: Bool, registerTraceFinalEqualsCandidate: Bool,
                stages: [Q4FirstDivergenceStageDiff]) {
        self.variant = variant
        self.productionTraceFinalEqualsControl = productionTraceFinalEqualsControl
        self.registerTraceFinalEqualsCandidate = registerTraceFinalEqualsCandidate
        self.attributableToOriginals = productionTraceFinalEqualsControl && registerTraceFinalEqualsCandidate
        self.stages = stages
    }
}

public struct Q4FirstDivergenceTraceRun: Codable, Equatable, Sendable {
    public let shape: String
    public let variant: String
    public let kernelName: String
    public let sourceSha256: String
    public let completion: String
    public let guardsIntact: Bool
    public let wordsRecorded: Int
    public let unwrittenWords: Int
    public let finalHex: String?
    public let lastTracedSumHex: String?
    public let lastTracedSumEqualsOwnFinal: Bool?
    public let finalEqualsUninstrumented: Bool
}

public struct Q4FirstDivergenceCPUModelResult: Codable, Equatable, Sendable {
    public let model: String
    public let finalHex: String?
    public let matchesControlFinal: Bool
    public let matchesCandidateFinal: Bool
    public let firstDifferenceVsProductionSumTrace: Int?
    public let firstDifferenceVsRegisterSumTrace: Int?
}

public struct Q4FirstDivergenceReport: Codable {
    public private(set) var kind = Q4FirstDivergenceDiagnostics.kind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var notes = Q4FirstDivergenceDiagnostics.notes
    public private(set) var closedRecords = Q4FirstDivergenceDiagnostics.closedRecords
    public var executableDigest: String? = nil
    public var controlFunction = Q4WeightStageEmitter.controlFunction
    public var controlSourceSha256: String? = nil
    public var registerKernelName: String? = nil
    public var registerSourceSha256: String? = nil
    public var registerCompileError: String? = nil
    public var declined: String? = nil
    public var hardware: Q4WeightStageHardware? = nil
    public var caseName = Q4FirstDivergenceDiagnostics.caseName
    public var inDim: Int? = nil
    public var outDim: Int? = nil
    public var blocksPerRow: Int? = nil
    public var controlCompletion: String? = nil
    public var candidateCompletion: String? = nil
    public var controlFinalHex: String? = nil
    public var candidateFinalHex: String? = nil
    public var controlFP64WorstRel: Double? = nil
    public var candidateFP64WorstRel: Double? = nil
    public var uninstrumentedGuardsIntact: Bool? = nil
    public var reproducedMismatch: Bool? = nil
    public var instrumentedCompileErrors: [String] = []
    public var traces: [Q4FirstDivergenceTraceRun] = []
    public var unpack: [Q4FirstDivergenceStageDiff] = []
    public var unpackIdentical: Bool? = nil
    public var variants: [Q4FirstDivergenceVariantDiff] = []
    public var cpuModels: [Q4FirstDivergenceCPUModelResult] = []
    public var originalCauseEstablished = false
    public var originalCauseStatement = "not evaluated"
    public var commandBuffersCommitted = 0
    public var terminal = true
    public var error: String? = nil
    public var status: Q4FirstDivergenceStatus = .aborted

    public init() {}
}

extension Q4FirstDivergenceReport {
    func withContract() -> Q4FirstDivergenceReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

// MARK: - GPU helpers for the extra trace binding (originals untouched)

extension Q4WeightStageGPU {
    func makeTrace(words: Int) throws -> MTLBuffer {
        let g = Self.guardBytes
        guard let t = device.makeBuffer(length: 2 * g + words * 4, options: .storageModeShared) else {
            throw Q4WeightStageError("trace buffer allocation failed")
        }
        memset(t.contents(), Int32(Self.canaryByte), t.length)
        memset(t.contents().advanced(by: g), 0xFF, words * 4)
        retained.append(t)
        return t
    }

    func traceWords(_ t: MTLBuffer, count: Int) -> [UInt32] {
        let p = t.contents().advanced(by: Self.guardBytes).assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    /// Same binding as encode plus the trace buffer at index 5; one dispatch, bounded semaphore via submit.
    func runTraced(pipeline: MTLComputePipelineState, launch: Q4WeightStageLaunch, b: Q4WeightStageBuffers,
                   y: MTLBuffer, trace: MTLBuffer) -> Q4WeightStageOutcome {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return Q4WeightStageOutcome(completion: "NOT_COMMITTED", terminal: true,
                                        error: "command buffer or encoder unavailable", wallMicros: -1,
                                        commandBufferIntervalSeconds: nil)
        }
        var p = SIMD2<UInt32>(UInt32(b.inDim), UInt32(b.outDim))
        var off = UInt64(Self.guardBytes)
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(b.w, offset: 0, index: 2)
        enc.setBuffer(b.x, offset: Self.guardBytes, index: 3)
        enc.setBuffer(y, offset: Self.guardBytes, index: 4)
        enc.setBuffer(trace, offset: Self.guardBytes, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: launch.threadgroups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: launch.threadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()
        return submit(cb)
    }
}

// MARK: - Driver

public enum Q4FirstDivergenceDiagnostics {
    public static let kind = "q4-first-divergence-diagnostic-v1"
    public static let caseName = "min-1x64"
    public static let usage = "usage: q4-first-divergence-diagnostic --report NEW_REPORT.json"
    public static let closedRecords: [String] = [
        "q4-packed-word-register-decode-component-v1 (phase 61): correctness-rejected, frozen",
        "q4-register-loop-decode-component-v1 (phase 62): correctness-rejected on 17 of 17 supported cases "
            + "(1858 of 2447 rows) including min-1x64, frozen",
    ]
    public static let notes: [String] = [
        "Diagnostic only: no optimization candidate, no production or default change; fullModelQualified, "
            + "gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false.",
        "The frozen min-1x64 case is taken from Q4WeightStageCorpus.synthetic() unchanged (same seed, same bytes).",
        "Uninstrumented reproduction: production g4_q4_matvec and the actual phase-62 emitter source, both options "
            + "nil; both final Float bit patterns are recorded and compared before any instrumented kernel runs.",
        "Instrumented kernels are copies with trace stores, never the originals. A trace variant is attributed to the "
            + "originals only if the production-shaped trace final equals the uninstrumented control bits AND the "
            + "register-shaped trace final equals the uninstrumented candidate bits; otherwise the original cause is "
            + "NOT ESTABLISHED and the trace is reported as altered arithmetic.",
        "The split variant (named w, d*w, (d*w)*x, sum) is altered arithmetic by construction; the sum-only variant "
            + "keeps the two production term lines verbatim but still adds stores.",
        "CPU accumulation models are consistent-with hypotheses on one 64-term row; a match is not a proof of the "
            + "compiled instruction sequence.",
        "No fast-math, fma or volatile flag is applied to make either original pass; options nil selects the Metal "
            + "compiler default (not the safe math mode), under which contraction and reassociation are permitted in "
            + "both originals and in every instrumented copy; no timing is measured.",
    ]

    public struct Options: Equatable, Sendable {
        public let report: String
    }

    public static func parseOptions(_ argv: [String]) throws -> Options {
        guard argv.count == 2, argv[0] == "--report", !argv[1].isEmpty, !argv[1].hasPrefix("--") else {
            throw Q4WeightStageError(usage)
        }
        return Options(report: argv[1])
    }

    public static func encodeReport(_ r: Q4FirstDivergenceReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(r.withContract())
    }

    public static func exitCode(_ s: Q4FirstDivergenceStatus) -> Int32 {
        switch s {
        case .completed: return 0
        case .notReproduced: return 7
        case .unsupported: return 2
        case .commandBufferError: return 4
        case .notEnded: return 5
        case .aborted: return 1
        }
    }

    public static func frozenCase() -> Q4WeightStageCase? {
        Q4WeightStageCorpus.synthetic().first { $0.name == caseName }
    }

    /// Status reflects run integrity only; whether the cause was established is a report field.
    public static func finalStatus(_ r: Q4FirstDivergenceReport) -> Q4FirstDivergenceStatus {
        if !r.terminal { return .notEnded }
        if r.error != nil { return .aborted }
        guard r.hardware != nil, r.registerCompileError == nil, r.declined == nil,
              r.instrumentedCompileErrors.isEmpty else { return .unsupported }
        if r.controlCompletion == "ERROR" || r.candidateCompletion == "ERROR"
            || r.traces.contains(where: { $0.completion == "ERROR" }) { return .commandBufferError }
        guard r.reproducedMismatch == true else { return .notReproduced }
        return .completed
    }

    /// Pure attribution rule. Returns established plus an explicit statement.
    public static func attribute(reproduced: Bool?, unpackIdentical: Bool?,
                                 unpack: [Q4FirstDivergenceStageDiff],
                                 variants: [Q4FirstDivergenceVariantDiff]) -> (established: Bool, statement: String) {
        guard reproduced == true else {
            return (false, "mismatch not reproduced on the frozen case; nothing to attribute")
        }
        if unpackIdentical == false, let d = unpack.first(where: { !$0.identical }) {
            return (false, "ORIGINAL CAUSE NOT ESTABLISHED: decode operands differ in the instrumented unpack kernels "
                    + "at stage " + d.stage + " index " + String(d.firstIndex ?? -1) + " (" + (d.locator ?? "")
                    + "); this points at the register decoder, but the unpack kernels have no arithmetic final to tie "
                    + "to the originals, so the rule does not attribute it")
        }
        guard unpackIdentical == true else {
            return (false, "ORIGINAL CAUSE NOT ESTABLISHED: unpack stage incomplete")
        }
        for name in ["sumOnly", "split"] {
            guard let v = variants.first(where: { $0.variant == name }), v.attributableToOriginals else { continue }
            if let d = v.stages.first(where: { !$0.identical }) {
                let indexText: String = String(d.firstIndex ?? -1)
                let locatorText: String = d.locator ?? ""
                let productionText: String = d.productionHex ?? "?"
                let registerText: String = d.registerHex ?? "?"
                var s: String = "ESTABLISHED by the " + name
                s += " trace, calibrated by its finals (both trace finals equal their uninstrumented originals and each "
                s += "last traced sum equals its own final; this is trace evidence about the traced copies, not an "
                s += "observation of the originals' compiled instruction sequence): first divergence at stage "
                s += d.stage
                s += " index " + indexText
                s += " (" + locatorText + "), production " + productionText
                s += " register " + registerText + "; decode operands identical"
                return (true, s)
            }
            return (false, "ORIGINAL CAUSE NOT ESTABLISHED: the " + name + " trace finals equal their originals yet "
                    + "every traced stage is identical, which contradicts the differing finals; treat as a trace "
                    + "coverage gap, not an explanation")
        }
        let flags = variants.map {
            $0.variant + " production=" + String($0.productionTraceFinalEqualsControl) + " register="
                + String($0.registerTraceFinalEqualsCandidate)
        }
        return (false, "ORIGINAL CAUSE NOT ESTABLISHED: no trace variant reproduced both uninstrumented finals ("
                + flags.joined(separator: "; ") + "); instrumentation altered the compiled arithmetic, so the traces "
                + "describe altered kernels, not the originals; decode operands identical")
    }

    public static func run(executableDigest: String, emit: (String) -> Void) -> Q4FirstDivergenceReport {
        let counter = Q4WeightStageDiagnostics.CommitCounter()
        var report = runCore(executableDigest: executableDigest, counter: counter, emit: emit)
        report.commandBuffersCommitted = counter.committed
        return report
    }

    static func runCore(executableDigest: String, counter: Q4WeightStageDiagnostics.CommitCounter,
                        emit: (String) -> Void) -> Q4FirstDivergenceReport {
        var report = Q4FirstDivergenceReport()
        report.executableDigest = executableDigest
        report.controlSourceSha256 = Q4WeightStageEmitter.sourceDigest(Gemma4Runner.shaderSource)
        func finish(_ r: inout Q4FirstDivergenceReport) -> Q4FirstDivergenceReport {
            r.status = finalStatus(r)
            return r
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            report.error = "no Metal device"
            return finish(&report)
        }
        let gpu: Q4WeightStageGPU
        do { gpu = try Q4WeightStageGPU(device: device) } catch {
            report.error = String(describing: error)
            return finish(&report)
        }
        defer { counter.committed = gpu.commandBuffers.count }
        report.hardware = Q4WeightStageHardware(
            device: device.name, controlThreadExecutionWidth: gpu.control.threadExecutionWidth,
            controlMaxTotalThreads: gpu.control.maxTotalThreadsPerThreadgroup,
            deviceMaxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            deviceMaxThreadsPerThreadgroupWidth: device.maxThreadsPerThreadgroup.width,
            supportsAtStageBoundary: device.supportsCounterSampling(.atStageBoundary),
            supportsAtDispatchBoundary: device.supportsCounterSampling(.atDispatchBoundary),
            timestampCounterSetFound: false, sampleBufferCreated: false)
        emit("device " + device.name + ": control width " + String(gpu.control.threadExecutionWidth) + ", no sampling")

        guard let c = frozenCase(), c.outDim == 1 else {
            report.error = "frozen case " + caseName + " not found or not single-row"
            return finish(&report)
        }
        report.inDim = c.inDim
        report.outDim = c.outDim
        report.blocksPerRow = c.blocksPerRow

        let descriptor: Q4RegisterLoopDescriptor
        let registerPipe: MTLComputePipelineState
        do {
            descriptor = try Q4RegisterLoopDescriptor.derive(executionWidth: gpu.control.threadExecutionWidth)
            let source = try Q4RegisterLoopEmitter.emit(descriptor)
            report.registerKernelName = descriptor.kernelName
            report.registerSourceSha256 = Q4WeightStageEmitter.sourceDigest(source)
            let library = try device.makeLibrary(source: source, options: nil)
            guard let fn = library.makeFunction(name: descriptor.kernelName) else {
                throw Q4WeightStageError("register function missing")
            }
            registerPipe = try device.makeComputePipelineState(function: fn)
        } catch {
            report.registerCompileError = "phase-62 register source (options nil): " + String(describing: error)
            return finish(&report)
        }
        let g = Q4WeightStageGPU.guardBytes
        if let reason = Q4RegisterLoopPlan.declineReason(
            inDim: c.inDim, outDim: c.outDim, weightOffset: g, weightLength: 2 * g + c.blocks.count,
            descriptor: descriptor, executionWidth: registerPipe.threadExecutionWidth,
            pipelineMaxThreads: registerPipe.maxTotalThreadsPerThreadgroup,
            staticThreadgroupBytes: registerPipe.staticThreadgroupMemoryLength) {
            report.declined = reason
            return finish(&report)
        }

        let b: Q4WeightStageBuffers
        let yControl: MTLBuffer
        let yCandidate: MTLBuffer
        do {
            b = try gpu.makeBuffers(c)
            yControl = try gpu.makeOutput(outDim: c.outDim)
            yCandidate = try gpu.makeOutput(outDim: c.outDim)
        } catch {
            report.error = "allocation: " + String(describing: error)
            return finish(&report)
        }
        let controlLaunch = Q4WeightStagePlan.controlLaunch(outDim: c.outDim,
                                                            pipelineMaxThreads: gpu.control.maxTotalThreadsPerThreadgroup)
        let registerLaunch = Q4RegisterLoopPlan.launch(outDim: c.outDim, descriptor: descriptor)

        // 1. Uninstrumented reproduction.
        let ctlOutcome = gpu.runSingle(pipeline: gpu.control, launch: controlLaunch, b: b, y: yControl)
        if !ctlOutcome.terminal {
            report.terminal = false
            report.error = "control: " + (ctlOutcome.error ?? "not ended")
            return finish(&report)
        }
        report.controlCompletion = ctlOutcome.completion
        let candOutcome = gpu.runSingle(pipeline: registerPipe, launch: registerLaunch, b: b, y: yCandidate)
        if !candOutcome.terminal {
            report.terminal = false
            report.error = "candidate: " + (candOutcome.error ?? "not ended")
            return finish(&report)
        }
        report.candidateCompletion = candOutcome.completion
        let ctlBits = ctlOutcome.completion == "COMPLETED" ? gpu.outputBits(yControl, outDim: 1) : []
        let candBits = candOutcome.completion == "COMPLETED" ? gpu.outputBits(yCandidate, outDim: 1) : []
        report.controlFinalHex = ctlBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
        report.candidateFinalHex = candBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
        report.uninstrumentedGuardsIntact = gpu.guardsIntact(b.w, payloadBytes: b.weightBytes)
            && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4) && gpu.guardsIntact(yControl, payloadBytes: 4)
            && gpu.guardsIntact(yCandidate, payloadBytes: 4)
        if let ref = try? Q4Oracle.matvecFP64(blocks: c.blocks[...], outDim: c.outDim, inDim: c.inDim, x: c.x) {
            report.controlFP64WorstRel = Q4WeightStageDiagnostics.worstRel(ctlBits, ref).worst
            report.candidateFP64WorstRel = Q4WeightStageDiagnostics.worstRel(candBits, ref).worst
        }
        if ctlBits.count == 1, candBits.count == 1 {
            let finite = Float(bitPattern: ctlBits[0]).isFinite && Float(bitPattern: candBits[0]).isFinite
            if !finite || report.uninstrumentedGuardsIntact != true {
                report.error = "uninstrumented reproduction not trustworthy: finite=" + String(finite) + " guards="
                    + String(report.uninstrumentedGuardsIntact ?? false)
                return finish(&report)
            }
            report.reproducedMismatch = ctlBits[0] != candBits[0]
        }
        emit("uninstrumented: control " + (report.controlFinalHex ?? "n/a") + " candidate "
             + (report.candidateFinalHex ?? "n/a") + " reproduced="
             + (report.reproducedMismatch.map { String($0) } ?? "n/a"))

        // 2 + 3. Instrumented copies.
        var words = [String: [UInt32]]()
        for variant in Q4FirstDivergenceVariant.allCases {
            for shape in Q4FirstDivergenceShape.allCases {
                let name = Q4FirstDivergenceEmitter.kernelName(shape: shape, variant: variant)
                let source = Q4FirstDivergenceEmitter.emit(shape: shape, variant: variant)
                let sha = Q4WeightStageEmitter.sourceDigest(source)
                let pipe: MTLComputePipelineState
                do {
                    let library = try device.makeLibrary(source: source, options: nil)
                    guard let fn = library.makeFunction(name: name) else { throw Q4WeightStageError("function missing") }
                    pipe = try device.makeComputePipelineState(function: fn)
                } catch {
                    report.instrumentedCompileErrors.append(name + ": " + String(describing: error))
                    continue
                }
                let count = Q4FirstDivergenceLayout.traceWords(inDim: c.inDim, blocksPerRow: c.blocksPerRow,
                                                                variant: variant)
                let trace: MTLBuffer
                let y: MTLBuffer
                do {
                    trace = try gpu.makeTrace(words: count)
                    y = try gpu.makeOutput(outDim: 1)
                } catch {
                    report.error = "allocation: " + String(describing: error)
                    return finish(&report)
                }
                let outcome = gpu.runTraced(pipeline: pipe, launch: shape == .production ? controlLaunch : registerLaunch,
                                            b: b, y: y, trace: trace)
                if !outcome.terminal {
                    report.terminal = false
                    report.error = name + ": " + (outcome.error ?? "not ended")
                    return finish(&report)
                }
                let completed = outcome.completion == "COMPLETED"
                let w = completed ? gpu.traceWords(trace, count: count) : []
                let yBits = completed ? gpu.outputBits(y, outDim: 1) : []
                let guards = gpu.guardsIntact(trace, payloadBytes: count * 4) && gpu.guardsIntact(y, payloadBytes: 4)
                    && gpu.guardsIntact(b.w, payloadBytes: b.weightBytes) && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4)
                var lastSum: UInt32? = nil
                if completed, variant != .unpack, let last = w.last { lastSum = last }
                let uninstr = shape == .production ? ctlBits.first : candBits.first
                let finalEquals = completed && yBits.count == 1 && uninstr != nil && yBits[0] == uninstr!
                words[name] = w
                report.traces.append(Q4FirstDivergenceTraceRun(
                    shape: shape.rawValue, variant: variant.rawValue, kernelName: name, sourceSha256: sha,
                    completion: outcome.completion, guardsIntact: guards, wordsRecorded: w.count,
                    unwrittenWords: w.filter { $0 == 0xFFFF_FFFF }.count,
                    finalHex: yBits.first.map { Q4FirstDivergenceLayout.hex8($0) },
                    lastTracedSumHex: lastSum.map { Q4FirstDivergenceLayout.hex8($0) },
                    lastTracedSumEqualsOwnFinal: lastSum.map { s in yBits.count == 1 && yBits[0] == s },
                    finalEqualsUninstrumented: variant == .unpack ? false : finalEquals))
            }
        }

        func traced(_ shape: Q4FirstDivergenceShape, _ variant: Q4FirstDivergenceVariant) -> [UInt32]? {
            words[Q4FirstDivergenceEmitter.kernelName(shape: shape, variant: variant)]
        }
        func stageDiffs(_ variant: Q4FirstDivergenceVariant) -> [Q4FirstDivergenceStageDiff]? {
            guard let p = traced(.production, variant), let r = traced(.register, variant) else { return nil }
            let ps = Q4FirstDivergenceLayout.stages(p, variant: variant, blocksPerRow: c.blocksPerRow, inDim: c.inDim)
            let rs = Q4FirstDivergenceLayout.stages(r, variant: variant, blocksPerRow: c.blocksPerRow, inDim: c.inDim)
            guard ps.count == rs.count, !ps.isEmpty else { return nil }
            return zip(ps, rs).map { Q4FirstDivergenceStageDiff.make(stage: $0.name, production: $0.values,
                                                                       register: $1.values) }
        }
        if let u = stageDiffs(.unpack) {
            report.unpack = u
            report.unpackIdentical = u.allSatisfy { $0.identical }
        }
        /// A trace run is calibrated to its original only if it completed, its final equals the uninstrumented bits,
        /// its guards are intact, every trace word was written and its last traced sum equals its own final.
        func calibrated(_ run: Q4FirstDivergenceTraceRun?) -> Bool {
            guard let run = run else { return false }
            return run.completion == "COMPLETED" && run.finalEqualsUninstrumented && run.guardsIntact
                && run.unwrittenWords == 0 && run.lastTracedSumEqualsOwnFinal == true
        }
        for variant in [Q4FirstDivergenceVariant.sumOnly, .split] {
            guard let s = stageDiffs(variant) else { continue }
            let pRun = report.traces.first { $0.shape == "production" && $0.variant == variant.rawValue }
            let rRun = report.traces.first { $0.shape == "register" && $0.variant == variant.rawValue }
            report.variants.append(Q4FirstDivergenceVariantDiff(
                variant: variant.rawValue,
                productionTraceFinalEqualsControl: calibrated(pRun),
                registerTraceFinalEqualsCandidate: calibrated(rRun), stages: s))
        }

        // 4. CPU hypotheses.
        let pSum = traced(.production, .sumOnly)
        let rSum = traced(.register, .sumOnly)
        for model in Q4FirstDivergenceCPU.Model.allCases {
            let sums = Q4FirstDivergenceCPU.runningSums(model: model, blocks: c.blocks, inDim: c.inDim, x: c.x)
            let final = sums?.last
            report.cpuModels.append(Q4FirstDivergenceCPUModelResult(
                model: model.rawValue, finalHex: final.map { Q4FirstDivergenceLayout.hex8($0) },
                matchesControlFinal: final != nil && ctlBits.first == final,
                matchesCandidateFinal: final != nil && candBits.first == final,
                firstDifferenceVsProductionSumTrace: sums.flatMap { s in pSum.map { Q4FirstDivergenceLayout.firstDifference(s, $0) } } ?? nil,
                firstDifferenceVsRegisterSumTrace: sums.flatMap { s in rSum.map { Q4FirstDivergenceLayout.firstDifference(s, $0) } } ?? nil))
        }

        let verdict = attribute(reproduced: report.reproducedMismatch, unpackIdentical: report.unpackIdentical,
                                unpack: report.unpack, variants: report.variants)
        report.originalCauseEstablished = verdict.established
        report.originalCauseStatement = verdict.statement
        emit(verdict.statement)
        return finish(&report)
    }
}
