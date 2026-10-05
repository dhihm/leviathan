import CryptoKit
import Foundation
import Metal

// Phase 55 stage 1: GPU-resident router selection COMPONENT prototype (expertCount 128, topK 8).
// Default OFF and unwired: Gemma4.swift, GemmaForge.swift, existing gates, kernels and defaults never reference it.
// Unsupported domains are REFUSED (flag bits, ids 0, zero coefficients); they are never approximated or counted as passes.

public struct RouterSelectionError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public struct RouterSelectionFlags: OptionSet, Codable, Hashable, Sendable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let nonFiniteInput = RouterSelectionFlags(rawValue: 1)
    public static let tinyInput = RouterSelectionFlags(rawValue: 2)
    public static let rangeUnsupported = RouterSelectionFlags(rawValue: 4)
    public static let nearTieCollapse = RouterSelectionFlags(rawValue: 8)
    public static let roundBoundary = RouterSelectionFlags(rawValue: 16)
    public static let scaleDomain = RouterSelectionFlags(rawValue: 32)
    public static let offsetDomain = RouterSelectionFlags(rawValue: 64)
    public static let named: [(RouterSelectionFlags, String)] = [
        (.nonFiniteInput, "NONFINITE_INPUT"), (.tinyInput, "TINY_OR_SUBNORMAL_INPUT"),
        (.rangeUnsupported, "D_OUTSIDE_SUPPORTED_RANGE"), (.nearTieCollapse, "NEAR_TIE_COLLAPSE"),
        (.roundBoundary, "ROUND_BOUNDARY_AMBIGUOUS"), (.scaleDomain, "SCALE_DOMAIN"),
        (.offsetDomain, "OFFSET_DOMAIN"),
    ]
    public var labels: [String] { Self.named.filter { contains($0.0) }.map { $0.1 } }
}

/// Byte layout of one selection record: derived, then validated (alignment, no overlap, inside the stride).
/// GU/DN offset arrays and every coefficient slot start on bindingAlignment so a later consumer can bind them directly
/// as setBuffer offsets (256 is the conservative macOS constant-buffer offset alignment; 8 ids + 2x8 offsets + 8 slots > 256 B).
public struct RouterSelectionLayout: Codable, Equatable, Sendable {
    public typealias Field = (name: String, offset: Int, length: Int, alignment: Int)
    public let topK: Int
    public let bindingAlignment: Int
    public let idsOffset: Int
    public let flagsOffset: Int
    public let magicOffset: Int
    public let layerOffset: Int
    public let rawCoefficientOffset: Int
    public let guOffsetsOffset: Int
    public let dnOffsetsOffset: Int
    public let coefficientOffset: Int
    public let coefficientStride: Int
    public let recordStride: Int

    public static func derive(topK: Int, bindingAlignment: Int) -> RouterSelectionLayout {
        func up(_ v: Int, _ a: Int) -> Int { (v + a - 1) / a * a }
        let flags = 4 * topK
        let raw = up(flags + 12, 16)
        let gu = up(raw + 4 * topK, bindingAlignment)
        let dn = up(gu + 8 * topK, bindingAlignment)
        let coefficient = up(dn + 8 * topK, bindingAlignment)
        let slot = up(4, bindingAlignment)
        return RouterSelectionLayout(
            topK: topK, bindingAlignment: bindingAlignment, idsOffset: 0, flagsOffset: flags,
            magicOffset: flags + 4, layerOffset: flags + 8, rawCoefficientOffset: raw,
            guOffsetsOffset: gu, dnOffsetsOffset: dn, coefficientOffset: coefficient,
            coefficientStride: slot, recordStride: up(coefficient + slot * topK, bindingAlignment))
    }

    public var fieldRanges: [Field] {
        var fields: [Field] = [
            ("ids", idsOffset, 4 * topK, 4), ("flags", flagsOffset, 4, 4), ("magic", magicOffset, 4, 4),
            ("layer", layerOffset, 4, 4), ("rawCoefficients", rawCoefficientOffset, 4 * topK, 4),
            ("guOffsets", guOffsetsOffset, 8 * topK, bindingAlignment),
            ("dnOffsets", dnOffsetsOffset, 8 * topK, bindingAlignment),
        ]
        for slot in 0..<max(topK, 0) {
            fields.append(("coefficient" + String(slot), coefficientOffset + slot * coefficientStride, 4, bindingAlignment))
        }
        return fields
    }

    public var recordWords: Int { recordStride / 4 }

    public func validate() throws {
        guard topK > 0, topK <= 32 else { throw RouterSelectionError("layout topK out of range") }
        guard bindingAlignment >= 16, bindingAlignment & (bindingAlignment - 1) == 0 else {
            throw RouterSelectionError("layout bindingAlignment must be a power of two >= 16")
        }
        guard coefficientStride >= 4, coefficientStride % bindingAlignment == 0, recordStride % bindingAlignment == 0 else {
            throw RouterSelectionError("layout strides must be binding aligned")
        }
        var end = 0
        for field in fieldRanges.sorted(by: { $0.offset < $1.offset }) {
            guard field.offset >= end, field.length > 0, field.offset % field.alignment == 0 else {
                throw RouterSelectionError("layout field overlaps or is misaligned: " + field.name)
            }
            end = field.offset + field.length
        }
        guard end <= recordStride else { throw RouterSelectionError("layout fields exceed recordStride") }
        guard self == Self.derive(topK: topK, bindingAlignment: bindingAlignment) else {
            throw RouterSelectionError("layout is not the canonical derivation")
        }
    }
}

/// Typed IR: every numeric policy the emitted MSL and the host mirror depend on. Validated before any emission.
public struct RouterSelectionIR: Codable, Equatable, Sendable {
    public let schema: String
    public let expertCount: Int
    public let topK: Int
    public let threadsPerGroup: Int
    public let simdWidth: Int
    public let rangeLow: Float
    public let tinyInputLog2: Int
    public let nearTieGapLog2: Int
    public let boundaryBudgetLog2: Int
    public let taylorDegree: Int
    public let layout: RouterSelectionLayout

    public init(
        schema: String, expertCount: Int, topK: Int, threadsPerGroup: Int, simdWidth: Int, rangeLow: Float,
        tinyInputLog2: Int, nearTieGapLog2: Int, boundaryBudgetLog2: Int, taylorDegree: Int, layout: RouterSelectionLayout
    ) {
        self.schema = schema
        self.expertCount = expertCount
        self.topK = topK
        self.threadsPerGroup = threadsPerGroup
        self.simdWidth = simdWidth
        self.rangeLow = rangeLow
        self.tinyInputLog2 = tinyInputLog2
        self.nearTieGapLog2 = nearTieGapLog2
        self.boundaryBudgetLog2 = boundaryBudgetLog2
        self.taylorDegree = taylorDegree
        self.layout = layout
    }

    public static let gemma4 = RouterSelectionIR(
        schema: RouterSelectionKernel.schema, expertCount: 128, topK: 8, threadsPerGroup: 128, simdWidth: 32,
        rangeLow: -80, tinyInputLog2: -100, nearTieGapLog2: -40, boundaryBudgetLog2: -42, taylorDegree: 13,
        layout: .derive(topK: 8, bindingAlignment: 256))

    public func validate() throws {
        func require(_ ok: Bool, _ why: String) throws {
            if !ok { throw RouterSelectionError("invalid RouterSelectionIR: " + why) }
        }
        try require(schema == RouterSelectionKernel.schema, "schema")
        try require(expertCount == 128 && topK == 8, "first scope is expertCount 128 and topK 8")
        try require(threadsPerGroup == expertCount && simdWidth == 32 && threadsPerGroup % simdWidth == 0, "one 128-lane group of 32-wide SIMD groups")
        try require(rangeLow >= -80 && rangeLow < 0 && rangeLow == rangeLow.rounded(), "rangeLow must be an integer in [-80, 0)")
        try require(tinyInputLog2 >= -100 && tinyInputLog2 <= -60, "tinyInputLog2 in [-100, -60]")
        try require(nearTieGapLog2 >= -44 && nearTieGapLog2 <= -20, "nearTieGapLog2 in [-44, -20]")
        try require(boundaryBudgetLog2 >= -42 && boundaryBudgetLog2 <= -24, "boundaryBudgetLog2 in [-42, -24]")
        try require(taylorDegree >= 13 && taylorDegree <= 17, "taylorDegree in [13, 17]")
        try require(layout.topK == topK, "layout topK")
        try layout.validate()
    }
}

/// Constants shared by the host mirror and the emitted MSL (emitted as exact shortest-decimal Float literals).
public struct RouterSelectionConstants: Equatable, Sendable {
    public let invLn2: Float
    public let ln2: [Float]
    public let taylorHi: [Float]
    public let taylorLo: [Float]
    public let gapThreshold: Float
    public let budget: Float
    public let halfUlp: Float
    public let quarterUlp: Float
    public let rangeLow: Float
    public let tinyLimitBits: UInt32

    public init(_ ir: RouterSelectionIR) {
        // ln2 as double-double (hi, lo); split into a 16-bit head (k * head exact for |k| <= 128) and two tails.
        let ln2Hi = 0.6931471805599453
        let ln2Lo = 2.319046813846299558e-17
        let c0 = Float(bitPattern: Float(ln2Hi).bitPattern & 0xFFFF_FF00)
        let rest = ln2Hi - Double(c0)
        let c1 = Float(rest + ln2Lo)
        let c2 = Float((rest - Double(c1)) + ln2Lo)
        ln2 = [c0, c1, c2]
        invLn2 = Float(1.4426950408889634)
        var hi: [Float] = []
        var lo: [Float] = []
        var factorial = 1.0
        for n in 0...ir.taylorDegree {
            if n > 0 { factorial *= Double(n) }
            let v = 1.0 / factorial
            let h = Float(v)
            hi.append(h)
            lo.append(Float(v - Double(h)))
        }
        taylorHi = hi
        taylorLo = lo
        gapThreshold = Float(sign: .plus, exponent: ir.nearTieGapLog2, significand: 1)
        budget = Float(sign: .plus, exponent: ir.boundaryBudgetLog2, significand: 1)
        halfUlp = Float(sign: .plus, exponent: -24, significand: 1)
        quarterUlp = Float(sign: .plus, exponent: -25, significand: 1)
        rangeLow = ir.rangeLow
        tinyLimitBits = UInt32(127 + ir.tinyInputLog2) << 23
    }
}

/// Unevaluated double-float (hi + lo). Operation order matches the emitted MSL helpers one for one.
struct RouterSelectionDF: Equatable, Sendable {
    var hi: Float
    var lo: Float
    init(_ hi: Float, _ lo: Float) {
        self.hi = hi
        self.lo = lo
    }
}

enum RouterSelectionDFMath {
    static func twoSum(_ a: Float, _ b: Float) -> RouterSelectionDF {
        let s = a + b
        let bb = s - a
        return RouterSelectionDF(s, (a - (s - bb)) + (b - bb))
    }
    static func quickTwoSum(_ a: Float, _ b: Float) -> RouterSelectionDF {
        let s = a + b
        return RouterSelectionDF(s, b - (s - a))
    }
    static func twoProd(_ a: Float, _ b: Float) -> RouterSelectionDF {
        let p = a * b
        return RouterSelectionDF(p, (-p).addingProduct(a, b))
    }
    static func neg(_ a: RouterSelectionDF) -> RouterSelectionDF { RouterSelectionDF(-a.hi, -a.lo) }
    static func add(_ a: RouterSelectionDF, _ b: RouterSelectionDF) -> RouterSelectionDF {
        var s = twoSum(a.hi, b.hi)
        let t = twoSum(a.lo, b.lo)
        s = quickTwoSum(s.hi, s.lo + t.hi)
        return quickTwoSum(s.hi, s.lo + t.lo)
    }
    static func mul(_ a: RouterSelectionDF, _ b: RouterSelectionDF) -> RouterSelectionDF {
        let p = twoProd(a.hi, b.hi)
        let e = p.lo.addingProduct(a.hi, b.lo).addingProduct(a.lo, b.hi)
        return quickTwoSum(p.hi, e)
    }
    static func pow2(_ e: Int) -> Float { Float(bitPattern: UInt32(e + 127) << 23) }
    /// exp(d) = m * 2^k with m in about [0.707, 1.415]; only called for supported d in [rangeLow, 0].
    static func exp(_ d: Float, _ c: RouterSelectionConstants) -> (RouterSelectionDF, Int) {
        let kf = (d * c.invLn2).rounded(.toNearestOrEven)
        var r = twoSum(d, -(kf * c.ln2[0]))
        r = add(r, neg(twoProd(kf, c.ln2[1])))
        r = add(r, RouterSelectionDF(-(kf * c.ln2[2]), 0))
        let top = c.taylorHi.count - 1
        var p = RouterSelectionDF(c.taylorHi[top], c.taylorLo[top])
        for i in stride(from: top - 1, through: 0, by: -1) {
            p = add(mul(p, r), RouterSelectionDF(c.taylorHi[i], c.taylorLo[i]))
        }
        return (p, Int(kf))
    }
}

/// Host software mirror of the emitted MSL arithmetic. ratio holds Double diagnostics of m_i 2^k_i / T for error sampling.
public struct RouterSelectionMirrorResult: Equatable, Sendable {
    public let ids: [Int]
    public let rawBits: [UInt32]
    public let weightBits: [UInt32]
    public let flags: RouterSelectionFlags
    public let ratio: [Double]

    static func refused(_ flags: RouterSelectionFlags, topK: Int) -> RouterSelectionMirrorResult {
        let zeros = [UInt32](repeating: 0, count: topK)
        return RouterSelectionMirrorResult(ids: [Int](repeating: 0, count: topK), rawBits: zeros, weightBits: zeros, flags: flags, ratio: [])
    }
}

/// Exact unchanged CPU router math (GemmaForge router_topk): FP64 softmax, lower-index tie-break, Float rounding, Float scale.
public struct RouterSelectionCPUResult: Equatable, Sendable {
    public let ids: [Int]
    public let weights: [Float]
    public let quotients: [Double]
    public var weightBits: [UInt32] { weights.map { $0.bitPattern } }
    public var rawBits: [UInt32] { quotients.map { Float($0).bitPattern } }
}

public struct RouterOffsetFamily: Codable, Equatable, Sendable {
    public let base: UInt64
    public let stride: UInt64
    public let expertBytes: UInt64
    public init(base: UInt64, stride: UInt64, expertBytes: UInt64) {
        self.base = base
        self.stride = stride
        self.expertBytes = expertBytes
    }
}

/// Host-validated GU (packed gate+up) and DN offset metadata: one family pair per layer; there is no separate UP family.
public struct RouterOffsetTable: Codable, Equatable, Sendable {
    public let expertCount: Int
    public let alignment: UInt64
    public let gu: [RouterOffsetFamily]
    public let dn: [RouterOffsetFamily]
    public let guBufferLength: UInt64
    public let dnBufferLength: UInt64

    public init(expertCount: Int, alignment: UInt64, gu: [RouterOffsetFamily], dn: [RouterOffsetFamily], guBufferLength: UInt64, dnBufferLength: UInt64) {
        self.expertCount = expertCount
        self.alignment = alignment
        self.gu = gu
        self.dn = dn
        self.guBufferLength = guBufferLength
        self.dnBufferLength = dnBufferLength
    }

    /// Overflow-checked proof that base + id * stride + expertBytes <= bufferLength for every id < expertCount.
    /// Offsets increase with id, so the last expert bounds all of them. Returns the number of offsets proven.
    public func validate() throws -> Int {
        guard expertCount > 0, !gu.isEmpty, gu.count == dn.count, alignment > 0, alignment & (alignment - 1) == 0 else {
            throw RouterSelectionError("offset table shape invalid")
        }
        var proven = 0
        for (families, length) in [(gu, guBufferLength), (dn, dnBufferLength)] {
            for family in families {
                guard family.expertBytes > 0, family.stride >= family.expertBytes,
                      family.base % alignment == 0, family.stride % alignment == 0 else {
                    throw RouterSelectionError("offset family misaligned or overlapping")
                }
                let (span, o1) = family.stride.multipliedReportingOverflow(by: UInt64(expertCount - 1))
                let (last, o2) = family.base.addingReportingOverflow(span)
                let (end, o3) = last.addingReportingOverflow(family.expertBytes)
                guard !o1, !o2, !o3, end <= length else {
                    throw RouterSelectionError("offset range overflows its buffer")
                }
                proven += expertCount
            }
        }
        return proven
    }

    public func offset(layer: Int, dn isDN: Bool, expert: Int) -> UInt64 {
        let family = isDN ? dn[layer] : gu[layer]
        return family.base + UInt64(expert) * family.stride
    }

    /// Synthetic table only (no model): contiguous per-layer families of expertBytes-sized experts.
    public static func synthetic(layers: Int, expertCount: Int, expertBytes: UInt64) -> RouterOffsetTable {
        let perLayer = UInt64(expertCount) * expertBytes
        let families = (0..<layers).map { RouterOffsetFamily(base: UInt64($0) * perLayer, stride: expertBytes, expertBytes: expertBytes) }
        return RouterOffsetTable(
            expertCount: expertCount, alignment: 16, gu: families, dn: families,
            guBufferLength: UInt64(layers) * perLayer, dnBufferLength: UInt64(layers) * perLayer)
    }

    public static func syntheticTag(dn isDN: Bool, layer: Int, expert: Int) -> UInt32 {
        (UInt32(isDN ? 2 : 1) << 24) | (UInt32(layer) << 8) | UInt32(expert)
    }

    public func syntheticPayload(dn isDN: Bool) -> [UInt32] {
        var words = [UInt32](repeating: 0, count: Int((isDN ? dnBufferLength : guBufferLength) / 4))
        for layer in 0..<gu.count {
            for expert in 0..<expertCount {
                words[Int(offset(layer: layer, dn: isDN, expert: expert) / 4)] = Self.syntheticTag(dn: isDN, layer: layer, expert: expert)
            }
        }
        return words
    }
}

public struct RouterSelectionGPURecord: Codable, Equatable, Sendable {
    public let ids: [UInt32]
    public let flags: UInt32
    public let magic: UInt32
    public let layer: UInt32
    public let rawBits: [UInt32]
    public let weightBits: [UInt32]
    public let guOffsets: [UInt64]
    public let dnOffsets: [UInt64]
    public let paddingClean: Bool
    public var refusal: RouterSelectionFlags { RouterSelectionFlags(rawValue: flags) }
}

public enum RouterSelectionKernel {
    public static let schema = "leviathan.router-selection.v1"
    public static let recordMagic: UInt32 = 0x5253_4C31
    public static let guardBytes = 256
    public static let canaryByte: UInt8 = 0xA5
    public static let selectFunction = "leviathan_router_select"
    public static let consumeFunction = "leviathan_router_consume"
    public static let maxCases = 4096

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(Array(data)).map { String(format: "%02x", $0) }.joined()
    }

    public static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    public static func irDigest(_ ir: RouterSelectionIR) throws -> String { sha256Hex(try canonicalJSON(ir)) }

    public static func sourceDigest(_ source: String) -> String { sha256Hex(Data(source.utf8)) }

    /// Exact copy of the existing CPU router math. Returns nil for non-finite scores, mirroring the existing guard that throws.
    public static func cpuReference(scores: [Float], scales: [Float], topK: Int) -> RouterSelectionCPUResult? {
        guard !scores.isEmpty, scores.count == scales.count, topK > 0, topK <= scores.count,
              scores.allSatisfy({ $0.isFinite }) else { return nil }
        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp(Double($0 - maxScore)) }
        let sumExp = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExp }
        var idx = Array(0..<scores.count)
        idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
        let topIdx = Array(idx.prefix(topK))
        let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
        let quotients = topIdx.map { probs[$0] / probSum }
        let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }
        return RouterSelectionCPUResult(ids: topIdx, weights: weights, quotients: quotients)
    }

    /// Host mirror of the emitted selector (same flag order, same double-float operation order).
    public static func mirror(_ ir: RouterSelectionIR, scores: [Float], scales: [Float]) throws -> RouterSelectionMirrorResult {
        try ir.validate()
        let n = ir.expertCount
        let k = ir.topK
        guard scores.count == n, scales.count == n else {
            throw RouterSelectionError("mirror input must have expertCount scores and scales")
        }
        let c = RouterSelectionConstants(ir)
        typealias M = RouterSelectionDFMath
        var flags = RouterSelectionFlags()
        for s in scores {
            let mag = s.bitPattern & 0x7FFF_FFFF
            if mag >= 0x7F80_0000 {
                flags.insert(.nonFiniteInput)
            } else if mag != 0 && mag < c.tinyLimitBits {
                flags.insert(.tinyInput)
            }
        }
        var maxScore = scores[0]
        for s in scores.dropFirst() where maxScore < s { maxScore = s }
        let d = scores.map { $0 - maxScore }
        if d.contains(where: { $0 < c.rangeLow }) { flags.insert(.rangeUnsupported) }
        if flags.contains(.nonFiniteInput) { return .refused(flags, topK: k) }
        let order = (0..<n).sorted { d[$0] > d[$1] || (d[$0] == d[$1] && $0 < $1) }
        for r in 0..<k {
            let a = d[order[r]]
            let b = d[order[r + 1]]
            if a != b && a - b <= c.gapThreshold { flags.insert(.nearTieCollapse) }
        }
        guard flags.isEmpty else { return .refused(flags, topK: k) }
        var m: [RouterSelectionDF] = []
        var e: [Int] = []
        for r in 0..<k {
            let x = M.exp(d[order[r]], c)
            m.append(x.0)
            e.append(x.1)
        }
        let p0 = M.pow2(e[0])
        var t = RouterSelectionDF(m[0].hi * p0, m[0].lo * p0)
        for r in 1..<k {
            let p2 = M.pow2(e[r])
            t = M.add(t, RouterSelectionDF(m[r].hi * p2, m[r].lo * p2))
        }
        var ids: [Int] = []
        var raw: [UInt32] = []
        var weights: [UInt32] = []
        var ratio: [Double] = []
        for r in 0..<k {
            let mr = m[r]
            let q1 = mr.hi / t.hi
            let rr = M.add(mr, M.neg(M.mul(t, RouterSelectionDF(q1, 0))))
            let fp = q1 + rr.hi / t.hi
            let fb = fp.bitPattern
            let p2e = Float(bitPattern: fb & 0x7F80_0000)
            let hUp = p2e * c.halfUlp
            let hDn = (fb & 0x007F_FFFF) == 0 ? p2e * c.quarterUlp : hUp
            let rl = M.add(mr, M.neg(M.mul(RouterSelectionDF(fp, -hDn), t)))
            let rh = M.add(mr, M.neg(M.mul(RouterSelectionDF(fp, hUp), t)))
            let thr = c.budget * mr.hi
            if !(fp > 0 && fp < 2 && rl.hi > thr && rh.hi < -thr) { flags.insert(.roundBoundary) }
            let fq = fp * M.pow2(e[r])
            let id = order[r]
            let smag = scales[id].bitPattern & 0x7FFF_FFFF
            if smag >= 0x7F80_0000 || (smag != 0 && smag < 0x0080_0000) { flags.insert(.scaleDomain) }
            let w = fq * scales[id]
            if smag != 0 && (w.bitPattern & 0x7FFF_FFFF) < 0x0080_0000 { flags.insert(.scaleDomain) }
            ids.append(id)
            raw.append(fq.bitPattern)
            weights.append(w.bitPattern)
            ratio.append((Double(mr.hi) + Double(mr.lo)) * Double(M.pow2(e[r])) / (Double(t.hi) + Double(t.lo)))
        }
        guard flags.isEmpty else { return .refused(flags, topK: k) }
        return RouterSelectionMirrorResult(ids: ids, rawBits: raw, weightBits: weights, flags: flags, ratio: ratio)
    }

    /// Deterministic MSL emitted from the validated IR (no timestamps, no host-dependent text).
    public static func emitSource(_ ir: RouterSelectionIR) throws -> String {
        try ir.validate()
        let c = RouterSelectionConstants(ir)
        let l = ir.layout
        let n = String(ir.expertCount)
        let k = String(ir.topK)
        func fl(_ v: Float) -> String { String(describing: v) + "f /* 0x" + String(v.bitPattern, radix: 16) + " */" }
        func uw(_ name: String, _ v: Int) -> String { "constant uint " + name + " = " + String(v) + "u;" }
        func cf(_ name: String, _ v: Float) -> String { "constant float " + name + " = " + fl(v) + ";" }
        var s: [String] = []
        s.append("// Generated by RouterSelectionKernel.emitSource from " + ir.schema + "; deterministic, do not edit.")
        s.append("#include <metal_stdlib>")
        s.append("using namespace metal;")
        s.append(uw("RS_EXPERTS", ir.expertCount))
        s.append(uw("RS_TOPK", ir.topK))
        s.append(uw("RS_RECORD_WORDS", l.recordWords))
        s.append(uw("RS_W_IDS", l.idsOffset / 4))
        s.append(uw("RS_W_FLAGS", l.flagsOffset / 4))
        s.append(uw("RS_W_MAGIC", l.magicOffset / 4))
        s.append(uw("RS_W_LAYER", l.layerOffset / 4))
        s.append(uw("RS_W_RAW", l.rawCoefficientOffset / 4))
        s.append(uw("RS_W_GU", l.guOffsetsOffset / 4))
        s.append(uw("RS_W_DN", l.dnOffsetsOffset / 4))
        s.append(uw("RS_W_COEF", l.coefficientOffset / 4))
        s.append(uw("RS_COEF_STRIDE_WORDS", l.coefficientStride / 4))
        s.append(uw("RS_MAGIC", Int(recordMagic)))
        s.append(uw("RS_TINY_BITS", Int(c.tinyLimitBits)))
        s.append(uw("RS_F_NONFINITE", Int(RouterSelectionFlags.nonFiniteInput.rawValue)))
        s.append(uw("RS_F_TINY", Int(RouterSelectionFlags.tinyInput.rawValue)))
        s.append(uw("RS_F_RANGE", Int(RouterSelectionFlags.rangeUnsupported.rawValue)))
        s.append(uw("RS_F_TIE", Int(RouterSelectionFlags.nearTieCollapse.rawValue)))
        s.append(uw("RS_F_ROUND", Int(RouterSelectionFlags.roundBoundary.rawValue)))
        s.append(uw("RS_F_SCALE", Int(RouterSelectionFlags.scaleDomain.rawValue)))
        s.append(uw("RS_F_OFFSET", Int(RouterSelectionFlags.offsetDomain.rawValue)))
        s.append(cf("RS_RANGE_LOW", c.rangeLow))
        s.append(cf("RS_GAP", c.gapThreshold))
        s.append(cf("RS_BUDGET", c.budget))
        s.append(cf("RS_HALF_ULP", c.halfUlp))
        s.append(cf("RS_QUARTER_ULP", c.quarterUlp))
        s.append(cf("RS_INV_LN2", c.invLn2))
        s.append(cf("RS_LN2_0", c.ln2[0]))
        s.append(cf("RS_LN2_1", c.ln2[1]))
        s.append(cf("RS_LN2_2", c.ln2[2]))
        s.append("constant int RS_DEG = " + String(ir.taylorDegree) + ";")
        s.append("constant float RS_TAYLOR_HI[" + String(c.taylorHi.count) + "] = { " + c.taylorHi.map(fl).joined(separator: ", ") + " };")
        s.append("constant float RS_TAYLOR_LO[" + String(c.taylorLo.count) + "] = { " + c.taylorLo.map(fl).joined(separator: ", ") + " };")
        s += [
            "struct rs_df { float hi; float lo; };",
            "static inline rs_df rs_make(float h, float l) { rs_df r; r.hi = h; r.lo = l; return r; }",
            "static inline rs_df rs_two_sum(float a, float b) { float s = a + b; float bb = s - a; return rs_make(s, (a - (s - bb)) + (b - bb)); }",
            "static inline rs_df rs_quick(float a, float b) { float s = a + b; return rs_make(s, b - (s - a)); }",
            "static inline rs_df rs_two_prod(float a, float b) { float p = a * b; return rs_make(p, fma(a, b, -p)); }",
            "static inline rs_df rs_neg(rs_df a) { return rs_make(-a.hi, -a.lo); }",
            "static inline rs_df rs_add(rs_df a, rs_df b) { rs_df s = rs_two_sum(a.hi, b.hi); rs_df t = rs_two_sum(a.lo, b.lo); s = rs_quick(s.hi, s.lo + t.hi); return rs_quick(s.hi, s.lo + t.lo); }",
            "static inline rs_df rs_mul(rs_df a, rs_df b) { rs_df p = rs_two_prod(a.hi, b.hi); float e = fma(a.lo, b.hi, fma(a.hi, b.lo, p.lo)); return rs_quick(p.hi, e); }",
            "static inline float rs_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }",
            "kernel void leviathan_router_select(",
            "    device const uint* scoreBits [[buffer(0)]],",
            "    device const uint* scaleBits [[buffer(1)]],",
            "    device const uint* caseLayer [[buffer(2)]],",
            "    device const ulong* offsetMeta [[buffer(3)]],",
            "    device uint* records [[buffer(4)]],",
            "    constant uint& caseCount [[buffer(5)]],",
            "    constant uint& layerCount [[buffer(6)]],",
            "    uint lid [[thread_position_in_threadgroup]],",
            "    uint gid [[threadgroup_position_in_grid]])",
            "{",
        ]
        s.append("    threadgroup float tgD[" + n + "];")
        s.append("    threadgroup uint tgFlag[" + n + "];")
        s.append("    threadgroup uint tgSlot[" + String(ir.topK + 1) + "];")
        for decl in ["float tgMHi", "float tgMLo", "int tgK", "uint tgSlotFlag", "uint tgIds", "uint tgRaw", "uint tgWeight", "ulong tgGu", "ulong tgDn"] {
            s.append("    threadgroup " + decl + "[" + k + "];")
        }
        s += [
            "    threadgroup float tgMax[1];",
            "    threadgroup uint tgFinal[1];",
            "    if (gid >= caseCount) { return; }",
            "    uint layer = caseLayer[gid];",
            "    uint layerFlag = 0u;",
            "    if (layer >= layerCount) { layerFlag = RS_F_OFFSET; layer = 0u; }",
            "    uint bits = scoreBits[gid * RS_EXPERTS + lid];",
            "    uint mag = bits & 0x7fffffffu;",
            "    uint inFlag = 0u;",
            "    if (mag >= 0x7f800000u) { inFlag = RS_F_NONFINITE; } else if (mag != 0u && mag < RS_TINY_BITS) { inFlag = RS_F_TINY; }",
            "    float s = as_type<float>(bits);",
            "    tgD[lid] = s;",
            "    if (lid <= RS_TOPK) { tgSlot[lid] = 0u; }",
            "    if (lid < RS_TOPK) { tgSlotFlag[lid] = 0u; tgRaw[lid] = 0u; tgWeight[lid] = 0u; }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (lid == 0u) {",
            "        float m = tgD[0];",
            "        for (uint j = 1u; j < RS_EXPERTS; ++j) { float v = tgD[j]; if (m < v) { m = v; } }",
            "        tgMax[0] = m;",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    float d = s - tgMax[0];",
            "    if (d < RS_RANGE_LOW) { inFlag |= RS_F_RANGE; }",
            "    tgD[lid] = d;",
            "    tgFlag[lid] = inFlag;",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    uint rank = 0u;",
            "    for (uint j = 0u; j < RS_EXPERTS; ++j) { float o = tgD[j]; if (o > d || (o == d && j < lid)) { rank += 1u; } }",
            "    if (rank <= RS_TOPK) { tgSlot[rank] = lid; }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (lid == 0u) {",
            "        uint agg = layerFlag;",
            "        for (uint j = 0u; j < RS_EXPERTS; ++j) { agg |= tgFlag[j]; }",
            "        if ((agg & RS_F_NONFINITE) == 0u) {",
            "            for (uint r = 0u; r < RS_TOPK; ++r) {",
            "                float a = tgD[tgSlot[r]];",
            "                float b = tgD[tgSlot[r + 1u]];",
            "                if (a != b && (a - b) <= RS_GAP) { agg |= RS_F_TIE; }",
            "            }",
            "        }",
            "        tgFinal[0] = agg;",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    uint early = tgFinal[0];",
            "    if (early == 0u && lid < RS_TOPK) {",
            "        float dv = tgD[tgSlot[lid]];",
            "        float kf = rint(dv * RS_INV_LN2);",
            "        rs_df r = rs_two_sum(dv, -(kf * RS_LN2_0));",
            "        r = rs_add(r, rs_neg(rs_two_prod(kf, RS_LN2_1)));",
            "        r = rs_add(r, rs_make(-(kf * RS_LN2_2), 0.0f));",
            "        rs_df p = rs_make(RS_TAYLOR_HI[RS_DEG], RS_TAYLOR_LO[RS_DEG]);",
            "        for (int i = RS_DEG - 1; i >= 0; --i) { p = rs_add(rs_mul(p, r), rs_make(RS_TAYLOR_HI[i], RS_TAYLOR_LO[i])); }",
            "        tgMHi[lid] = p.hi;",
            "        tgMLo[lid] = p.lo;",
            "        tgK[lid] = int(kf);",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (early == 0u && lid < RS_TOPK) {",
            "        float p0 = rs_pow2(tgK[0]);",
            "        rs_df t = rs_make(tgMHi[0] * p0, tgMLo[0] * p0);",
            "        for (uint r = 1u; r < RS_TOPK; ++r) { float p2 = rs_pow2(tgK[r]); t = rs_add(t, rs_make(tgMHi[r] * p2, tgMLo[r] * p2)); }",
            "        rs_df m = rs_make(tgMHi[lid], tgMLo[lid]);",
            "        float q1 = m.hi / t.hi;",
            "        rs_df rr = rs_add(m, rs_neg(rs_mul(t, rs_make(q1, 0.0f))));",
            "        float fp = q1 + rr.hi / t.hi;",
            "        uint fb = as_type<uint>(fp);",
            "        float p2e = as_type<float>(fb & 0x7f800000u);",
            "        float hUp = p2e * RS_HALF_ULP;",
            "        float hDn = ((fb & 0x007fffffu) == 0u) ? p2e * RS_QUARTER_ULP : hUp;",
            "        rs_df rl = rs_add(m, rs_neg(rs_mul(rs_make(fp, -hDn), t)));",
            "        rs_df rh = rs_add(m, rs_neg(rs_mul(rs_make(fp, hUp), t)));",
            "        float thr = RS_BUDGET * m.hi;",
            "        uint sf = 0u;",
            "        if (!(fp > 0.0f && fp < 2.0f && rl.hi > thr && rh.hi < -thr)) { sf |= RS_F_ROUND; }",
            "        float fq = fp * rs_pow2(tgK[lid]);",
            "        uint eid = tgSlot[lid];",
            "        uint sb = scaleBits[layer * RS_EXPERTS + eid];",
            "        uint smag = sb & 0x7fffffffu;",
            "        if (smag >= 0x7f800000u || (smag != 0u && smag < 0x00800000u)) { sf |= RS_F_SCALE; }",
            "        float w = fq * as_type<float>(sb);",
            "        if (smag != 0u && (as_type<uint>(w) & 0x7fffffffu) < 0x00800000u) { sf |= RS_F_SCALE; }",
            "        tgSlotFlag[lid] = sf;",
            "        tgRaw[lid] = as_type<uint>(fq);",
            "        tgWeight[lid] = as_type<uint>(w);",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    if (lid == 0u) {",
            "        uint agg = tgFinal[0];",
            "        for (uint r = 0u; r < RS_TOPK; ++r) { agg |= tgSlotFlag[r]; if (tgSlot[r] >= RS_EXPERTS) { agg |= RS_F_OFFSET; } }",
            "        ulong guB = offsetMeta[layer * 4u];",
            "        ulong guS = offsetMeta[layer * 4u + 1u];",
            "        ulong dnB = offsetMeta[layer * 4u + 2u];",
            "        ulong dnS = offsetMeta[layer * 4u + 3u];",
            "        for (uint r = 0u; r < RS_TOPK; ++r) {",
            "            uint eid = (agg == 0u) ? tgSlot[r] : 0u;",
            "            tgIds[r] = eid;",
            "            tgGu[r] = guB + ulong(eid) * guS;",
            "            tgDn[r] = dnB + ulong(eid) * dnS;",
            "            if (agg != 0u) { tgRaw[r] = 0u; tgWeight[r] = 0u; }",
            "        }",
            "        tgFinal[0] = agg;",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    uint base = gid * RS_RECORD_WORDS;",
            "    uint fin = tgFinal[0];",
            "    for (uint w = lid; w < RS_RECORD_WORDS; w += RS_EXPERTS) {",
            "        uint v = 0u;",
            "        if (w >= RS_W_IDS && w < RS_W_IDS + RS_TOPK) { v = tgIds[w - RS_W_IDS]; }",
            "        else if (w == RS_W_FLAGS) { v = fin; }",
            "        else if (w == RS_W_MAGIC) { v = RS_MAGIC; }",
            "        else if (w == RS_W_LAYER) { v = layer; }",
            "        else if (w >= RS_W_RAW && w < RS_W_RAW + RS_TOPK) { v = tgRaw[w - RS_W_RAW]; }",
            "        else if (w >= RS_W_GU && w < RS_W_GU + 2u * RS_TOPK) { ulong o = tgGu[(w - RS_W_GU) / 2u]; v = (((w - RS_W_GU) & 1u) == 0u) ? uint(o) : uint(o >> 32); }",
            "        else if (w >= RS_W_DN && w < RS_W_DN + 2u * RS_TOPK) { ulong o = tgDn[(w - RS_W_DN) / 2u]; v = (((w - RS_W_DN) & 1u) == 0u) ? uint(o) : uint(o >> 32); }",
            "        else if (w >= RS_W_COEF && (w - RS_W_COEF) % RS_COEF_STRIDE_WORDS == 0u && (w - RS_W_COEF) / RS_COEF_STRIDE_WORDS < RS_TOPK) { v = tgWeight[(w - RS_W_COEF) / RS_COEF_STRIDE_WORDS]; }",
            "        records[base + w] = v;",
            "    }",
            "}",
            "kernel void leviathan_router_consume(",
            "    device const uint* records [[buffer(0)]],",
            "    device const uint* guPayload [[buffer(1)]],",
            "    device const uint* dnPayload [[buffer(2)]],",
            "    device uint* consumed [[buffer(3)]],",
            "    constant uint& caseCount [[buffer(4)]],",
            "    constant ulong& guLength [[buffer(5)]],",
            "    constant ulong& dnLength [[buffer(6)]],",
            "    uint tid [[thread_position_in_grid]])",
            "{",
            "    if (tid >= caseCount * RS_TOPK) { return; }",
            "    uint c = tid / RS_TOPK;",
            "    uint slot = tid % RS_TOPK;",
            "    uint base = c * RS_RECORD_WORDS;",
            "    uint status = 0u;",
            "    if (records[base + RS_W_MAGIC] != RS_MAGIC) { status |= 1u; }",
            "    ulong gu = ulong(records[base + RS_W_GU + 2u * slot]) | (ulong(records[base + RS_W_GU + 2u * slot + 1u]) << 32);",
            "    ulong dn = ulong(records[base + RS_W_DN + 2u * slot]) | (ulong(records[base + RS_W_DN + 2u * slot + 1u]) << 32);",
            "    uint guTag = 0u;",
            "    uint dnTag = 0u;",
            "    if ((gu & ulong(3)) == ulong(0) && gu < guLength && guLength - gu >= ulong(4)) { guTag = guPayload[gu >> 2]; } else { status |= 2u; }",
            "    if ((dn & ulong(3)) == ulong(0) && dn < dnLength && dnLength - dn >= ulong(4)) { dnTag = dnPayload[dn >> 2]; } else { status |= 4u; }",
            "    uint o4 = tid * 4u;",
            "    consumed[o4] = guTag;",
            "    consumed[o4 + 1u] = dnTag;",
            "    consumed[o4 + 2u] = records[base + RS_W_COEF + slot * RS_COEF_STRIDE_WORDS];",
            "    consumed[o4 + 3u] = status;",
            "}",
        ]
        let newline = String(Character(Unicode.Scalar(UInt8(10))))
        return s.joined(separator: newline) + newline
    }

    /// Decodes one record and checks that every byte outside the declared fields is zero.
    public static func decodeRecord(_ raw: UnsafeRawBufferPointer, layout: RouterSelectionLayout) -> RouterSelectionGPURecord {
        func u32(_ o: Int) -> UInt32 { raw.load(fromByteOffset: o, as: UInt32.self) }
        func u64(_ o: Int) -> UInt64 { UInt64(u32(o)) | (UInt64(u32(o + 4)) << 32) }
        let k = layout.topK
        var used = [Bool](repeating: false, count: layout.recordStride)
        for field in layout.fieldRanges {
            for b in field.offset..<(field.offset + field.length) { used[b] = true }
        }
        var clean = true
        for b in 0..<layout.recordStride where !used[b] && raw[b] != 0 { clean = false }
        return RouterSelectionGPURecord(
            ids: (0..<k).map { u32(layout.idsOffset + 4 * $0) },
            flags: u32(layout.flagsOffset), magic: u32(layout.magicOffset), layer: u32(layout.layerOffset),
            rawBits: (0..<k).map { u32(layout.rawCoefficientOffset + 4 * $0) },
            weightBits: (0..<k).map { u32(layout.coefficientOffset + layout.coefficientStride * $0) },
            guOffsets: (0..<k).map { u64(layout.guOffsetsOffset + 8 * $0) },
            dnOffsets: (0..<k).map { u64(layout.dnOffsetsOffset + 8 * $0) },
            paddingClean: clean)
    }
}

public struct RouterSelectionHardware: Codable, Equatable, Sendable {
    public let deviceName: String
    public let selectThreadExecutionWidth: Int
    public let selectMaxTotalThreads: Int
    public let consumeThreadExecutionWidth: Int
    public let consumeMaxTotalThreads: Int
    public let deviceMaxThreadsPerThreadgroup: Int
    public let threadsPerGroup: Int
    public let simdGroupsPerGroup: Int
    public let mathMode: String
    public let policy: String
}

public struct RouterSelectionGPUOutcome: Codable, Sendable {
    public let completion: String
    public let terminal: Bool
    public let errorDescription: String?
    public let records: [RouterSelectionGPURecord]
    public let consumed: [[UInt32]]
    public let guardsIntact: Bool
    public let gpuMicros: Double?
}

/// Standalone GPU selector + tiny consumer. Hardware is queried only here (construct after GO). Single use per instance.
public final class RouterSelectionGPUComponent {
    public let ir: RouterSelectionIR
    public let source: String
    public let sourceSHA256: String
    public let hardware: RouterSelectionHardware
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let selectPipeline: any MTLComputePipelineState
    private let consumePipeline: any MTLComputePipelineState
    private var used = false

    public init(ir: RouterSelectionIR) throws {
        try ir.validate()
        guard RouterSelectionKernel.guardBytes % ir.layout.bindingAlignment == 0 else {
            throw RouterSelectionError("guard bytes are not binding aligned")
        }
        let source = try RouterSelectionKernel.emitSource(ir)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw RouterSelectionError("no Metal device or command queue")
        }
        let options = MTLCompileOptions()
        var mode = "fastMathEnabled=false"
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            mode = "mathMode=safe,mathFloatingPointFunctions=precise"
        } else {
            options.fastMathEnabled = false
        }
        let library = try device.makeLibrary(source: source, options: options)
        guard let sf = library.makeFunction(name: RouterSelectionKernel.selectFunction),
              let cf = library.makeFunction(name: RouterSelectionKernel.consumeFunction) else {
            throw RouterSelectionError("emitted library is missing a kernel")
        }
        let sp = try device.makeComputePipelineState(function: sf)
        let cp = try device.makeComputePipelineState(function: cf)
        let deviceMax = device.maxThreadsPerThreadgroup.width
        guard sp.threadExecutionWidth == ir.simdWidth, sp.maxTotalThreadsPerThreadgroup >= ir.threadsPerGroup,
              deviceMax >= ir.threadsPerGroup, cp.maxTotalThreadsPerThreadgroup >= 64 else {
            throw RouterSelectionError("hardware policy infeasible: width " + String(sp.threadExecutionWidth) + " maxThreads " + String(sp.maxTotalThreadsPerThreadgroup))
        }
        self.ir = ir
        self.source = source
        self.sourceSHA256 = RouterSelectionKernel.sourceDigest(source)
        self.device = device
        self.queue = queue
        self.selectPipeline = sp
        self.consumePipeline = cp
        self.hardware = RouterSelectionHardware(
            deviceName: device.name, selectThreadExecutionWidth: sp.threadExecutionWidth,
            selectMaxTotalThreads: sp.maxTotalThreadsPerThreadgroup, consumeThreadExecutionWidth: cp.threadExecutionWidth,
            consumeMaxTotalThreads: cp.maxTotalThreadsPerThreadgroup, deviceMaxThreadsPerThreadgroup: deviceMax,
            threadsPerGroup: ir.threadsPerGroup, simdGroupsPerGroup: ir.threadsPerGroup / ir.simdWidth, mathMode: mode,
            policy: "one threadgroup of " + String(ir.threadsPerGroup) + " lanes per case; lane-parallel rank select by (d desc, index asc); lanes below topK run double-float exp and the rounding certificate; concurrent compute encoder with memoryBarrier(resources:) on the record buffer between selector and consumer")
    }

    /// Runs the selector and the consumer over a small batch. Never calls waitUntilCompleted; completion is observed
    /// through a completed handler bounded by timeoutSeconds, otherwise the outcome is NOT_ENDED (terminal false).
    public func run(
        scores: [[Float]], caseLayers: [UInt32], scales: [[Float]], offsets: RouterOffsetTable,
        guPayload: [UInt32], dnPayload: [UInt32], timeoutSeconds: Double
    ) throws -> RouterSelectionGPUOutcome {
        guard !used else { throw RouterSelectionError("component instance is single-use") }
        used = true
        let n = ir.expertCount
        let k = ir.topK
        let layout = ir.layout
        let cases = scores.count
        guard cases > 0, cases <= RouterSelectionKernel.maxCases, caseLayers.count == cases, scores.allSatisfy({ $0.count == n }) else {
            throw RouterSelectionError("invalid case batch")
        }
        guard offsets.expertCount == n, scales.count == offsets.gu.count, scales.allSatisfy({ $0.count == n }) else {
            throw RouterSelectionError("scales or offset table do not match the IR")
        }
        _ = try offsets.validate()
        guard UInt64(guPayload.count) * 4 == offsets.guBufferLength, UInt64(dnPayload.count) * 4 == offsets.dnBufferLength else {
            throw RouterSelectionError("payload lengths do not match the offset table")
        }
        guard timeoutSeconds > 0, timeoutSeconds <= 60 else { throw RouterSelectionError("timeout out of range") }
        var meta: [UInt64] = []
        for l in 0..<offsets.gu.count {
            meta += [offsets.gu[l].base, offsets.gu[l].stride, offsets.dn[l].base, offsets.dn[l].stride]
        }
        var stagedInputs: [(any MTLBuffer, any MTLBuffer)] = []
        func stage<T>(_ values: [T]) throws -> any MTLBuffer {
            let shared = try values.withUnsafeBytes { raw -> any MTLBuffer in
                guard let base = raw.baseAddress, let b = self.device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared) else {
                    throw RouterSelectionError("staging allocation failed")
                }
                return b
            }
            guard let priv = self.device.makeBuffer(length: shared.length, options: .storageModePrivate) else {
                throw RouterSelectionError("private allocation failed")
            }
            stagedInputs.append((shared, priv))
            return priv
        }
        let scoreBuf = try stage(scores.flatMap { $0.map { $0.bitPattern } })
        let scaleBuf = try stage(scales.flatMap { $0.map { $0.bitPattern } })
        let layerBuf = try stage(caseLayers)
        let metaBuf = try stage(meta)
        let guBuf = try stage(guPayload)
        let dnBuf = try stage(dnPayload)
        let g = RouterSelectionKernel.guardBytes
        let recordBytes = cases * layout.recordStride
        let consumedBytes = cases * k * 16
        guard let recPriv = device.makeBuffer(length: 2 * g + recordBytes, options: .storageModePrivate),
              let conPriv = device.makeBuffer(length: 2 * g + consumedBytes, options: .storageModePrivate),
              let recRead = device.makeBuffer(length: 2 * g + recordBytes, options: .storageModeShared),
              let conRead = device.makeBuffer(length: 2 * g + consumedBytes, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let blitIn = cb.makeBlitCommandEncoder() else {
            throw RouterSelectionError("output allocation or command buffer failed")
        }
        for (shared, priv) in stagedInputs {
            blitIn.copy(from: shared, sourceOffset: 0, to: priv, destinationOffset: 0, size: shared.length)
        }
        blitIn.fill(buffer: recPriv, range: 0..<recPriv.length, value: RouterSelectionKernel.canaryByte)
        blitIn.fill(buffer: conPriv, range: 0..<conPriv.length, value: RouterSelectionKernel.canaryByte)
        blitIn.endEncoding()
        guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            throw RouterSelectionError("compute encoder failed")
        }
        var caseCount = UInt32(cases)
        var layerCount = UInt32(offsets.gu.count)
        var guLength = offsets.guBufferLength
        var dnLength = offsets.dnBufferLength
        enc.setComputePipelineState(selectPipeline)
        enc.setBuffer(scoreBuf, offset: 0, index: 0)
        enc.setBuffer(scaleBuf, offset: 0, index: 1)
        enc.setBuffer(layerBuf, offset: 0, index: 2)
        enc.setBuffer(metaBuf, offset: 0, index: 3)
        enc.setBuffer(recPriv, offset: g, index: 4)
        enc.setBytes(&caseCount, length: 4, index: 5)
        enc.setBytes(&layerCount, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: cases, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: ir.threadsPerGroup, height: 1, depth: 1))
        enc.memoryBarrier(resources: [recPriv])
        enc.setComputePipelineState(consumePipeline)
        enc.setBuffer(recPriv, offset: g, index: 0)
        enc.setBuffer(guBuf, offset: 0, index: 1)
        enc.setBuffer(dnBuf, offset: 0, index: 2)
        enc.setBuffer(conPriv, offset: g, index: 3)
        enc.setBytes(&caseCount, length: 4, index: 4)
        enc.setBytes(&guLength, length: 8, index: 5)
        enc.setBytes(&dnLength, length: 8, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (cases * k + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        guard let blitOut = cb.makeBlitCommandEncoder() else { throw RouterSelectionError("readback encoder failed") }
        blitOut.copy(from: recPriv, sourceOffset: 0, to: recRead, destinationOffset: 0, size: recPriv.length)
        blitOut.copy(from: conPriv, sourceOffset: 0, to: conRead, destinationOffset: 0, size: conPriv.length)
        blitOut.endEncoding()
        let done = DispatchSemaphore(value: 0)
        cb.addCompletedHandler { _ in done.signal() }
        cb.commit()
        if done.wait(timeout: .now() + timeoutSeconds) == .timedOut {
            return RouterSelectionGPUOutcome(completion: "NOT_ENDED", terminal: false, errorDescription: "completion not observed within bound", records: [], consumed: [], guardsIntact: false, gpuMicros: nil)
        }
        guard cb.status == .completed, cb.error == nil else {
            let why = cb.error.map { String(describing: $0) } ?? ("status " + String(cb.status.rawValue))
            return RouterSelectionGPUOutcome(completion: "ERROR", terminal: true, errorDescription: why, records: [], consumed: [], guardsIntact: false, gpuMicros: nil)
        }
        let recAll = UnsafeRawBufferPointer(start: recRead.contents(), count: recRead.length)
        let conAll = UnsafeRawBufferPointer(start: conRead.contents(), count: conRead.length)
        func guardsOK(_ all: UnsafeRawBufferPointer, _ body: Int) -> Bool {
            for i in 0..<g where all[i] != RouterSelectionKernel.canaryByte || all[g + body + i] != RouterSelectionKernel.canaryByte { return false }
            return true
        }
        let intact = guardsOK(recAll, recordBytes) && guardsOK(conAll, consumedBytes)
        var records: [RouterSelectionGPURecord] = []
        var consumed: [[UInt32]] = []
        for c in 0..<cases {
            let start = g + c * layout.recordStride
            records.append(RouterSelectionKernel.decodeRecord(UnsafeRawBufferPointer(rebasing: recAll[start..<(start + layout.recordStride)]), layout: layout))
            let cs = g + c * k * 16
            consumed.append((0..<(4 * k)).map { conAll.load(fromByteOffset: cs + 4 * $0, as: UInt32.self) })
        }
        let micros: Double? = cb.gpuEndTime > cb.gpuStartTime ? (cb.gpuEndTime - cb.gpuStartTime) * 1e6 : nil
        return RouterSelectionGPUOutcome(completion: "COMPLETED", terminal: true, errorDescription: nil, records: records, consumed: consumed, guardsIntact: intact, gpuMicros: micros)
    }
}
