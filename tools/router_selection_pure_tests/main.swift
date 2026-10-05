import Foundation
import LeviathanCore

// Phase 55 stage 1 pure tests for RouterSelectionKernel. No Metal device, no GGUF, no model weights.
// Fixed seeds and fixed case counts with no adaptive exclusions. Refused cases are never counted as equivalence passes.

struct Harness {
    var passed = 0
    var failed = 0

    mutating func check(_ ok: Bool, _ name: String) {
        if ok {
            passed += 1
        } else {
            failed += 1
            print("FAIL " + name)
        }
    }
}

enum Fixed {
    static let seed: UInt64 = 0x5253_4C31_0055_0001
    static let refusalSeed: UInt64 = 0x5253_4C31_0055_00F1
    static let roundBoundarySearchLimit = 1 << 25
}

struct SplitMix64 {
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

struct RSCase {
    let family: String
    let scores: [Float]
    let scales: [Float]
    let expected: RouterSelectionFlags
}

struct ExistingMath {
    let ids: [Int]
    let weights: [Float]
    let quotients: [Double]
}

/// Typed sampled-error result over a fixed-seed sample; it never proves a universal bound.
struct ErrorSample {
    let coefficients: Int
    let maxRelative: Double
    let budget: Double
    let provesUniversalBound = false
}

/// Independent verbatim copy of the existing GemmaForge router_topk math (Float subtract, FP64 softmax).
func existingRouterTopK(scores: [Float], scales: [Float], topK: Int) -> ExistingMath {
    let maxScore = scores.max() ?? 0
    let exps = scores.map { exp(Double($0 - maxScore)) }
    let sumExp = exps.reduce(0, +)
    let probs = exps.map { $0 / sumExp }
    var idx = Array(0..<scores.count)
    idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
    let topIdx = Array(idx.prefix(topK))
    let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
    let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }
    return ExistingMath(ids: topIdx, weights: weights, quotients: topIdx.map { probs[$0] / probSum })
}

func pow2f(_ e: Int) -> Float { Float(sign: .plus, exponent: e, significand: 1) }

func randomScales(_ rng: inout SplitMix64) -> [Float] {
    var s: [Float] = []
    for _ in 0..<128 {
        let m = Float(Int64(rng.below(3840)) + 64) / 256
        s.append(rng.below(2) == 0 ? m : -m)
    }
    return s
}

func gridScores(_ rng: inout SplitMix64, span: UInt64, offset: Int64, denominator: Float) -> [Float] {
    var s: [Float] = []
    for _ in 0..<128 {
        s.append(Float(Int64(rng.below(span)) + offset) / denominator)
    }
    return s
}

func nearMidpoint(_ q: Double, toleranceLog2: Int) -> Bool {
    guard q.isNormal, q > 0 else { return false }
    let t = Double(sign: .plus, exponent: 23 - q.exponent, significand: q)
    let frac = t - t.rounded(.down)
    return abs(frac - 0.5) < t * Double(sign: .plus, exponent: -toleranceLog2, significand: 1)
}

/// Bounded deterministic construction of an adversarial Float round-boundary case (scores 0, x seven times, -8 elsewhere).
/// Exact top-8 quotients are 1/(1+7e^x) and e^x/(1+7e^x). x is the first Float below -0.25 that puts one of them
/// within a relative 2^-46 of a Float midpoint.
func findRoundBoundaryScore(limit: Int) -> Float? {
    var x: Float = -0.25
    for _ in 0..<limit {
        x = x.nextDown
        let e = exp(Double(x))
        let t = 1.0 + 7.0 * e
        if nearMidpoint(1.0 / t, toleranceLog2: 46) || nearMidpoint(e / t, toleranceLog2: 46) { return x }
    }
    return nil
}

func roundBoundaryScores(_ x: Float) -> [Float] {
    var s = [Float](repeating: -8, count: 128)
    s[0] = 0
    for j in 1...7 { s[j] = x }
    return s
}

/// Predetermined ordinary families: every case must be unflagged and bit exact.
func ordinaryCases(uniform: Int, ties: Int, wide: Int, dominant: Int, sweep: Int, seed: UInt64) -> [RSCase] {
    var rng = SplitMix64(state: seed)
    var out: [RSCase] = []
    for v in [Float(0), 1.5, -3.25, 7, -8] {
        out.append(RSCase(family: "allEqual", scores: [Float](repeating: v, count: 128), scales: randomScales(&rng), expected: []))
    }
    for _ in 0..<uniform {
        let s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
        out.append(RSCase(family: "uniform", scores: s, scales: randomScales(&rng), expected: []))
    }
    for _ in 0..<ties {
        let s = gridScores(&rng, span: 33, offset: -16, denominator: 4)
        out.append(RSCase(family: "ties", scores: s, scales: randomScales(&rng), expected: []))
    }
    for _ in 0..<wide {
        let s = gridScores(&rng, span: 286_720, offset: -245_760, denominator: 4096)
        out.append(RSCase(family: "wideRange", scores: s, scales: randomScales(&rng), expected: []))
    }
    for _ in 0..<dominant {
        var s = gridScores(&rng, span: 12544, offset: -20224, denominator: 256)
        s[Int(rng.below(128))] = 0
        out.append(RSCase(family: "dominant", scores: s, scales: randomScales(&rng), expected: []))
    }
    for c in 0..<sweep {
        var s = [Float](repeating: -80, count: 128)
        s[0] = 0
        for j in 1...7 { s[j] = Float(-80.0 * Double(c * 7 + j) / Double(sweep * 7 + 1)) }
        out.append(RSCase(family: "rangeSweep", scores: s, scales: randomScales(&rng), expected: []))
    }
    var e1 = [Float](repeating: -80, count: 128)
    for i in 0..<7 { e1[i] = 0 }
    out.append(RSCase(family: "rangeEdge", scores: e1, scales: randomScales(&rng), expected: []))
    var e2 = [Float](repeating: -76.5, count: 128)
    e2[64] = 3.5
    out.append(RSCase(family: "rangeEdge", scores: e2, scales: randomScales(&rng), expected: []))
    for v in 0..<4 {
        var s = gridScores(&rng, span: 2048, offset: -2048, denominator: 256)
        for j in 0..<8 { s[16 + j] = 1 + Float(j + v) / 16 }
        var sc = randomScales(&rng)
        sc[0] = .nan
        sc[1] = .infinity
        sc[2] = .leastNonzeroMagnitude
        sc[3] = pow2f(-100)
        sc[16] = 0
        sc[17] = -0.0
        out.append(RSCase(family: "unselectedBadScale", scores: s, scales: sc, expected: []))
    }
    return out
}

/// Explicit adversarial classes that must be REFUSED (flag bits, ids 0, zero coefficients).
func refusalCases(roundBoundaryX: Float, seed: UInt64) -> [RSCase] {
    var rng = SplitMix64(state: seed)
    let ones = [Float](repeating: 1, count: 128)
    var out: [RSCase] = []
    func add(_ family: String, _ scores: [Float], _ scales: [Float], _ flag: RouterSelectionFlags) {
        out.append(RSCase(family: family, scores: scores, scales: scales, expected: flag))
    }
    var s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[3] = .nan
    add("nonFiniteNaN", s, ones, .nonFiniteInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[50] = .infinity
    add("nonFinitePosInf", s, ones, .nonFiniteInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[127] = -.infinity
    add("nonFiniteNegInf", s, ones, .nonFiniteInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[9] = pow2f(-110)
    add("tinyNormal", s, ones, .tinyInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[5] = .leastNonzeroMagnitude
    add("subnormalPositive", s, ones, .tinyInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[6] = -Float.leastNormalMagnitude / 2
    add("subnormalNegative", s, ones, .tinyInput)
    s = gridScores(&rng, span: 32768, offset: -16384, denominator: 2048)
    s[0] = 9
    s[100] = -81
    add("rangeUnselected", s, ones, .rangeUnsupported)
    s = [Float](repeating: -80, count: 128)
    for i in 0..<8 { s[i] = 0 }
    s[127] = Float(-80).nextDown
    add("rangeJustBelow", s, ones, .rangeUnsupported)
    s = [Float](repeating: -100, count: 128)
    s[0] = 0
    add("underflowSubnormalCoefficient", s, ones, .rangeUnsupported)
    s = [Float](repeating: -1, count: 128)
    s[1] = pow2f(-30)
    s[0] = pow2f(-30).nextDown
    add("nearTieCollapse", s, ones, .nearTieCollapse)
    add("roundBoundary", roundBoundaryScores(roundBoundaryX), ones, .roundBoundary)
    let badScales: [Float] = [.nan, .infinity, .leastNonzeroMagnitude]
    for i in 0..<3 {
        s = gridScores(&rng, span: 2048, offset: -2048, denominator: 256)
        s[i] = 2
        var sc = ones
        sc[i] = badScales[i]
        add("scaleDomain" + String(i), s, sc, .scaleDomain)
    }
    s = [Float](repeating: -40, count: 128)
    for i in 10..<17 { s[i] = 0 }
    s[3] = -30
    var sc = ones
    sc[3] = pow2f(-100)
    add("scaleProductUnderflow", s, sc, .scaleDomain)
    return out
}

func casesDigest(_ cases: [RSCase]) -> String {
    var words: [UInt32] = []
    for c in cases {
        words += c.scores.map { $0.bitPattern }
        words += c.scales.map { $0.bitPattern }
        words.append(c.expected.rawValue)
    }
    return words.withUnsafeBytes { RouterSelectionKernel.sha256Hex(Data($0)) }
}

func makeIR(
    expertCount: Int = 128, topK: Int = 8, threads: Int = 128, simd: Int = 32, rangeLow: Float = -80,
    tiny: Int = -100, gap: Int = -40, budget: Int = -42, degree: Int = 13,
    layout: RouterSelectionLayout = .derive(topK: 8, bindingAlignment: 256),
    schema: String = RouterSelectionKernel.schema
) -> RouterSelectionIR {
    RouterSelectionIR(
        schema: schema, expertCount: expertCount, topK: topK, threadsPerGroup: threads, simdWidth: simd,
        rangeLow: rangeLow, tinyInputLog2: tiny, nearTieGapLog2: gap, boundaryBudgetLog2: budget,
        taylorDegree: degree, layout: layout)
}

func valid(_ ir: RouterSelectionIR) -> Bool { (try? ir.validate()) != nil }

func mutatedLayout(_ key: String, _ value: Int) -> RouterSelectionLayout? {
    guard let data = try? JSONEncoder().encode(RouterSelectionIR.gemma4.layout),
          var dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
    dict[key] = value
    guard let back = try? JSONSerialization.data(withJSONObject: dict) else { return nil }
    return try? JSONDecoder().decode(RouterSelectionLayout.self, from: back)
}

func checkIR(_ h: inout Harness) {
    h.check(makeIR() == RouterSelectionIR.gemma4 && valid(.gemma4), "gemma4 IR is the canonical 128x8 IR and validates")
    let invalid: [(String, RouterSelectionIR)] = [
        ("expertCount 64", makeIR(expertCount: 64)),
        ("topK 4", makeIR(topK: 4, layout: .derive(topK: 4, bindingAlignment: 256))),
        ("layout topK mismatch", makeIR(layout: .derive(topK: 4, bindingAlignment: 256))),
        ("threads 64", makeIR(threads: 64)),
        ("simd 16", makeIR(simd: 16)),
        ("rangeLow -81", makeIR(rangeLow: -81)),
        ("rangeLow 0", makeIR(rangeLow: 0)),
        ("rangeLow -79.5", makeIR(rangeLow: -79.5)),
        ("rangeLow nan", makeIR(rangeLow: .nan)),
        ("tiny -59", makeIR(tiny: -59)),
        ("tiny -101", makeIR(tiny: -101)),
        ("gap -45", makeIR(gap: -45)),
        ("gap -19", makeIR(gap: -19)),
        ("budget -43", makeIR(budget: -43)),
        ("budget -23", makeIR(budget: -23)),
        ("degree 12", makeIR(degree: 12)),
        ("degree 18", makeIR(degree: 18)),
        ("schema", makeIR(schema: "leviathan.router-selection.v0")),
    ]
    for (name, ir) in invalid { h.check(!valid(ir), "IR rejects " + name) }
    let alternatives = [makeIR(degree: 17), makeIR(budget: -30), makeIR(rangeLow: -60), makeIR(layout: .derive(topK: 8, bindingAlignment: 512))]
    h.check(alternatives.allSatisfy { valid($0) }, "IR accepts bounded alternatives")
    if let data = try? RouterSelectionKernel.canonicalJSON(RouterSelectionIR.gemma4),
       let back = try? JSONDecoder().decode(RouterSelectionIR.self, from: data) {
        h.check(back == .gemma4, "IR canonical JSON round trip")
    } else {
        h.check(false, "IR canonical JSON round trip")
    }
    let bits = RouterSelectionFlags.named.map { $0.0.rawValue }
    h.check(bits.count == 7 && Set(bits).count == 7 && bits.allSatisfy { $0 != 0 && $0 & ($0 - 1) == 0 }, "refusal flags are distinct single bits")
}

func checkLayout(_ h: inout Harness) {
    let l = RouterSelectionIR.gemma4.layout
    h.check((try? l.validate()) != nil, "gemma4 layout validates")
    h.check(l.recordStride == 2816 && l.recordStride > 256 && l.recordWords * 4 == l.recordStride, "record stride is the derived 2816 B ABI, not an assumed 256 B")
    let declared = l.fieldRanges.reduce(0) { $0 + $1.length }
    let declaredSpan = l.fieldRanges.map { $0.offset + $0.length }.max() ?? 0
    h.check(declared == 236 && declaredSpan == 2564 && declaredSpan > 256 && declaredSpan <= l.recordStride,
            "declared fields total 236 B, span 2564 B exceeds 256 B and fits the stride")
    let binding = l.fieldRanges.filter { $0.alignment == l.bindingAlignment }
    h.check(binding.count == 2 + l.topK && binding.allSatisfy { $0.offset % 256 == 0 }, "GU, DN and every coefficient slot are 256-byte binding aligned")
    h.check(l.fieldRanges.count == 7 + l.topK && l.coefficientStride % l.bindingAlignment == 0, "one binding-aligned coefficient slot per selected expert")
    var allOK = true
    for topK in 1...32 {
        for a in [16, 32, 64, 128, 256, 512, 1024] {
            let d = RouterSelectionLayout.derive(topK: topK, bindingAlignment: a)
            if (try? d.validate()) == nil || d.fieldRanges.count != 7 + topK { allOK = false }
        }
    }
    h.check(allOK, "derived layouts validate for topK 1...32 and alignments 16...1024")
    h.check((try? RouterSelectionLayout.derive(topK: 0, bindingAlignment: 256).validate()) == nil, "layout rejects topK 0")
    h.check((try? RouterSelectionLayout.derive(topK: 33, bindingAlignment: 256).validate()) == nil, "layout rejects topK 33")
    h.check((try? RouterSelectionLayout.derive(topK: 8, bindingAlignment: 24).validate()) == nil, "layout rejects alignment 24")
    h.check((try? RouterSelectionLayout.derive(topK: 8, bindingAlignment: 8).validate()) == nil, "layout rejects alignment 8")
    let mutations: [(String, Int, String)] = [
        ("guOffsetsOffset", 264, "misaligned GU offsets"),
        ("dnOffsetsOffset", 256, "DN overlapping GU"),
        ("recordStride", 2560, "short record stride"),
        ("coefficientStride", 128, "unaligned coefficient stride"),
        ("rawCoefficientOffset", 40, "raw coefficients overlapping layer"),
    ]
    for (key, value, name) in mutations {
        if let m = mutatedLayout(key, value) {
            h.check((try? m.validate()) == nil, "layout rejects " + name)
        } else {
            h.check(false, "layout mutation decodes: " + name)
        }
    }
    var words = [UInt32](repeating: 0, count: l.recordWords)
    func put32(_ v: UInt32, _ o: Int) { words[o / 4] = v }
    func put64(_ v: UInt64, _ o: Int) {
        put32(UInt32(truncatingIfNeeded: v), o)
        put32(UInt32(truncatingIfNeeded: v >> 32), o + 4)
    }
    for s in 0..<l.topK {
        put32(UInt32(s + 1), l.idsOffset + 4 * s)
        put32(0x3E00_0000 + UInt32(s), l.rawCoefficientOffset + 4 * s)
        put64(0x1_0000_0000 + UInt64(s) * 0x1_0000_0010, l.guOffsetsOffset + 8 * s)
        put64(0x2_0000_0000 + UInt64(s), l.dnOffsetsOffset + 8 * s)
        put32(0x3F00_0000 + UInt32(s), l.coefficientOffset + l.coefficientStride * s)
    }
    put32(RouterSelectionKernel.recordMagic, l.magicOffset)
    put32(5, l.layerOffset)
    put32(RouterSelectionFlags.roundBoundary.rawValue, l.flagsOffset)
    let decoded = words.withUnsafeBytes { RouterSelectionKernel.decodeRecord($0, layout: l) }
    h.check(decoded.ids == (1...8).map { UInt32($0) } && decoded.paddingClean, "record decode ids, clean padding")
    h.check(decoded.magic == RouterSelectionKernel.recordMagic && decoded.layer == 5 && decoded.refusal == .roundBoundary, "record decode magic, layer, flags")
    h.check(decoded.guOffsets[3] == 0x1_0000_0000 + 3 * 0x1_0000_0010 && decoded.dnOffsets[7] == 0x2_0000_0007, "record decode 64-bit GU/DN offsets (lo word first)")
    h.check(decoded.rawBits[2] == 0x3E00_0002 && decoded.weightBits[7] == 0x3F00_0007, "record decode raw and strided coefficient slots")
    var dirty = words
    dirty[11] = 1
    h.check(!dirty.withUnsafeBytes { RouterSelectionKernel.decodeRecord($0, layout: l) }.paddingClean, "nonzero padding between layer and raw is detected")
    dirty = words
    dirty[100] = 1
    h.check(!dirty.withUnsafeBytes { RouterSelectionKernel.decodeRecord($0, layout: l) }.paddingClean, "nonzero padding between GU and DN is detected")
}

func checkConstantsAndSource(_ h: inout Harness) {
    let ir = RouterSelectionIR.gemma4
    let c = RouterSelectionConstants(ir)
    let literals = c.taylorHi + c.taylorLo + c.ln2 + [c.invLn2, c.gapThreshold, c.budget, c.halfUlp, c.quarterUlp, c.rangeLow]
    h.check(literals.allSatisfy { Float(String(describing: $0))?.bitPattern == $0.bitPattern }, "every emitted Float literal round-trips bit exactly")
    h.check(c.ln2[0].bitPattern & 0xFF == 0, "ln2 head has 8 trailing zero significand bits")
    var exactHead = true
    for kk in -128...128 where Double(Float(kk) * c.ln2[0]) != Double(kk) * Double(c.ln2[0]) { exactHead = false }
    h.check(exactHead, "k * ln2 head is exact for |k| <= 128")
    let ln2Sum = Double(c.ln2[0]) + Double(c.ln2[1]) + Double(c.ln2[2])
    h.check(abs(ln2Sum - 0.6931471805599453) <= 0x1p-52, "three-part ln2 reconstructs ln2")
    var taylorOK = c.taylorHi.count == ir.taylorDegree + 1 && c.taylorLo.count == ir.taylorDegree + 1
    if taylorOK {
        var factorial = 1.0
        for n in 0...ir.taylorDegree {
            if n > 0 { factorial *= Double(n) }
            let exact = 1.0 / factorial
            if abs(Double(c.taylorHi[n]) + Double(c.taylorLo[n]) - exact) > exact * 0x1p-46 { taylorOK = false }
        }
    }
    h.check(taylorOK, "double-float Taylor coefficients represent 1/n! to 2^-46 relative")
    h.check(c.budget == pow2f(ir.boundaryBudgetLog2) && c.gapThreshold == pow2f(ir.nearTieGapLog2) && c.rangeLow == -80, "budget, gap and range follow the IR")
    h.check(Float(bitPattern: c.tinyLimitBits) == pow2f(ir.tinyInputLog2) && c.halfUlp == pow2f(-24) && c.quarterUlp == pow2f(-25), "tiny limit and ulp fractions follow the IR")
    guard let s1 = try? RouterSelectionKernel.emitSource(ir), let s2 = try? RouterSelectionKernel.emitSource(ir) else {
        h.check(false, "emitSource succeeds")
        return
    }
    h.check(s1 == s2, "MSL emission is deterministic")
    let digest = RouterSelectionKernel.sourceDigest(s1)
    h.check(digest.count == 64 && digest == RouterSelectionKernel.sha256Hex(Data(s1.utf8)), "source digest is SHA-256 of the emitted UTF-8")
    h.check(s1.unicodeScalars.allSatisfy { $0.isASCII }, "emitted MSL is ASCII")
    h.check(s1.contains("kernel void " + RouterSelectionKernel.selectFunction + "(") && s1.contains("kernel void " + RouterSelectionKernel.consumeFunction + "("), "emitted MSL defines the selector and consumer entry points")
    h.check(s1.contains("constant uint RS_RECORD_WORDS = " + String(ir.layout.recordWords) + "u;"), "emitted record ABI is the validated layout")
    h.check(s1.contains(String(describing: c.budget) + "f") && s1.contains(String(describing: c.rangeLow) + "f"), "budget and range literals come from the IR constants")
    let alt = makeIR(degree: 14)
    let altSource = try? RouterSelectionKernel.emitSource(alt)
    h.check(altSource != nil && altSource != s1, "a different IR emits different MSL")
    let d1 = try? RouterSelectionKernel.irDigest(ir)
    let d2 = try? RouterSelectionKernel.irDigest(ir)
    let d3 = try? RouterSelectionKernel.irDigest(alt)
    h.check(d1 != nil && d1 == d2 && d1 != d3, "IR digest is stable and IR-sensitive")
    h.check((try? RouterSelectionKernel.emitSource(makeIR(expertCount: 64))) == nil, "emission refuses an invalid IR")
}

func checkOffsets(_ h: inout Harness) {
    let table = RouterOffsetTable.synthetic(layers: 30, expertCount: 128, expertBytes: 16)
    h.check((try? table.validate()) == 7680, "offset proof covers 30 layers x 128 experts x 2 families = 7680")
    let gu = table.syntheticPayload(dn: false)
    let dn = table.syntheticPayload(dn: true)
    var tagsOK = true
    var tags = Set<UInt32>()
    for layer in 0..<30 {
        for e in 0..<128 {
            let g = gu[Int(table.offset(layer: layer, dn: false, expert: e) / 4)]
            let d = dn[Int(table.offset(layer: layer, dn: true, expert: e) / 4)]
            if g != RouterOffsetTable.syntheticTag(dn: false, layer: layer, expert: e) { tagsOK = false }
            if d != RouterOffsetTable.syntheticTag(dn: true, layer: layer, expert: e) { tagsOK = false }
            tags.insert(g)
            tags.insert(d)
        }
    }
    h.check(tagsOK && tags.count == 7680, "synthetic payload tags are distinct and addressed by the proven offsets")
    func single(base: UInt64, stride: UInt64, bytes: UInt64, length: UInt64, alignment: UInt64 = 16) -> RouterOffsetTable {
        let f = RouterOffsetFamily(base: base, stride: stride, expertBytes: bytes)
        return RouterOffsetTable(expertCount: 128, alignment: alignment, gu: [f], dn: [f], guBufferLength: length, dnBufferLength: length)
    }
    func proven(_ t: RouterOffsetTable) -> Int? { try? t.validate() }
    h.check(proven(single(base: 32, stride: 64, bytes: 48, length: 32 + 127 * 64 + 48)) == 256, "exact-fit family is proven")
    h.check(proven(single(base: 32, stride: 64, bytes: 48, length: 32 + 127 * 64 + 47)) == nil, "one byte short is rejected")
    h.check(proven(single(base: UInt64.max - 15, stride: 16, bytes: 16, length: .max)) == nil, "base + span overflow is rejected")
    h.check(proven(single(base: 0, stride: (UInt64.max / 64) & ~UInt64(15), bytes: 16, length: .max)) == nil, "stride * 127 overflow is rejected")
    h.check(proven(single(base: UInt64.max - 127 * 16 - 15, stride: 16, bytes: 32, length: .max)) == nil, "end overflow is rejected")
    h.check(proven(single(base: 8, stride: 16, bytes: 16, length: 1 << 20)) == nil, "misaligned base is rejected")
    h.check(proven(single(base: 0, stride: 16, bytes: 32, length: 1 << 20)) == nil, "overlapping experts are rejected")
    h.check(proven(single(base: 0, stride: 16, bytes: 0, length: 1 << 20)) == nil, "empty experts are rejected")
    h.check(proven(single(base: 0, stride: 24, bytes: 16, length: 1 << 20, alignment: 24)) == nil, "non power-of-two alignment is rejected")
    let f = RouterOffsetFamily(base: 0, stride: 16, expertBytes: 16)
    let mismatch = RouterOffsetTable(expertCount: 128, alignment: 16, gu: [f, f], dn: [f], guBufferLength: 1 << 20, dnBufferLength: 1 << 20)
    h.check(proven(mismatch) == nil, "GU/DN family count mismatch is rejected")
    let empty = RouterOffsetTable(expertCount: 128, alignment: 16, gu: [], dn: [], guBufferLength: 1 << 20, dnBufferLength: 1 << 20)
    h.check(proven(empty) == nil, "empty table is rejected")
    h.check(table.offset(layer: 29, dn: true, expert: 127) == 29 * 128 * 16 + 127 * 16 && table.offset(layer: 0, dn: false, expert: 0) == 0, "offset formula base + id * stride")
}

func checkSelection(_ h: inout Harness) {
    let ir = RouterSelectionIR.gemma4
    let k = ir.topK
    let budget = Double(RouterSelectionConstants(ir).budget)
    let ordinary = ordinaryCases(uniform: 256, ties: 96, wide: 96, dominant: 64, sweep: 128, seed: Fixed.seed)
    let again = ordinaryCases(uniform: 256, ties: 96, wide: 96, dominant: 64, sweep: 128, seed: Fixed.seed)
    h.check(ordinary.count == 5 + 256 + 96 + 96 + 64 + 128 + 2 + 4 && casesDigest(ordinary) == casesDigest(again), "ordinary families are a fixed, reproducible set")
    var totals: [String: Int] = [:]
    var flagged: [String: Int] = [:]
    var mismatched: [String: Int] = [:]
    var cpuIdentity = true
    var maxRel = 0.0
    var coefficients = 0
    for tc in ordinary {
        totals[tc.family, default: 0] += 1
        let ex = existingRouterTopK(scores: tc.scores, scales: tc.scales, topK: k)
        guard let cpu = RouterSelectionKernel.cpuReference(scores: tc.scores, scales: tc.scales, topK: k),
              let mir = try? RouterSelectionKernel.mirror(ir, scores: tc.scores, scales: tc.scales) else {
            mismatched[tc.family, default: 0] += 1
            continue
        }
        let exBits = ex.weights.map { $0.bitPattern }
        if cpu.ids != ex.ids || cpu.weightBits != exBits || cpu.quotients != ex.quotients { cpuIdentity = false }
        if !mir.flags.isEmpty {
            flagged[tc.family, default: 0] += 1
            continue
        }
        let exRaw = ex.quotients.map { Float($0).bitPattern }
        if mir.ids != ex.ids || mir.weightBits != exBits || mir.rawBits != exRaw || mir.ratio.count != k {
            mismatched[tc.family, default: 0] += 1
            continue
        }
        for r in 0..<k {
            maxRel = max(maxRel, abs(mir.ratio[r] - ex.quotients[r]) / ex.quotients[r])
            coefficients += 1
        }
    }
    h.check(cpuIdentity, "public cpuReference is bit-identical to the existing router_topk math on every ordinary case")
    for family in totals.keys.sorted() {
        let total = totals[family] ?? 0
        let f = flagged[family] ?? 0
        let m = mismatched[family] ?? 0
        print("family " + family + " cases " + String(total) + " flagged " + String(f) + " mismatches " + String(m))
        h.check(total > 0 && f == 0, "ordinary family " + family + " is never refused")
        h.check(m == 0, "ordinary family " + family + " matches CPU ids, quotient bits and final coefficient bits exactly")
    }
    let sample = ErrorSample(coefficients: coefficients, maxRelative: maxRel, budget: budget)
    print("error sample: coefficients " + String(sample.coefficients) + " max relative " + String(sample.maxRelative) + " log2 " + String(log2(sample.maxRelative)) + " budget/4 " + String(sample.budget / 4) + " provesUniversalBound " + String(sample.provesUniversalBound))
    h.check(sample.coefficients == k * ordinary.count && sample.maxRelative <= sample.budget / 4, "sampled mirror-vs-CPU FP64 quotient error is within budget/4 (fixed sample, not a proof)")

    guard let x = findRoundBoundaryScore(limit: Fixed.roundBoundarySearchLimit) else {
        h.check(false, "a round-boundary score exists within the fixed search bound")
        return
    }
    print("round-boundary score bits 0x" + String(x.bitPattern, radix: 16))
    let refusals = refusalCases(roundBoundaryX: x, seed: Fixed.refusalSeed)
    h.check(refusals.count == 15, "fixed refusal set (15 host-constructible classes; the GPU-only offsetDomain layer refusal is the 16th in the component tool)")
    for tc in refusals {
        var refused = false
        if let mir = try? RouterSelectionKernel.mirror(ir, scores: tc.scores, scales: tc.scales) {
            refused = mir.flags.isSuperset(of: tc.expected) && mir.ids == [Int](repeating: 0, count: k)
                && mir.rawBits == [UInt32](repeating: 0, count: k) && mir.weightBits == [UInt32](repeating: 0, count: k) && mir.ratio.isEmpty
        }
        h.check(refused, "REFUSED " + tc.family + " with " + tc.expected.labels.joined(separator: ","))
        let cpu = RouterSelectionKernel.cpuReference(scores: tc.scores, scales: tc.scales, topK: k)
        if tc.scores.allSatisfy({ $0.isFinite }) {
            let ex = existingRouterTopK(scores: tc.scores, scales: tc.scales, topK: k)
            var same = false
            if let cpu { same = cpu.ids == ex.ids && cpu.weightBits == ex.weights.map({ $0.bitPattern }) }
            h.check(same, "CPU reference handles refused " + tc.family + " unchanged")
        } else {
            h.check(cpu == nil, "CPU reference rejects non-finite " + tc.family + " like the existing guard")
        }
    }
    if let under = refusals.first(where: { $0.family == "underflowSubnormalCoefficient" }),
       let cpu = RouterSelectionKernel.cpuReference(scores: under.scores, scales: under.scales, topK: k) {
        h.check(cpu.weights[1].isSubnormal, "extreme underflow: the exact CPU coefficient is a nonzero subnormal, so flushing below -80 would be wrong")
    } else {
        h.check(false, "extreme underflow case evaluates on the CPU")
    }
    if let tie = refusals.first(where: { $0.family == "nearTieCollapse" }),
       let cpu = RouterSelectionKernel.cpuReference(scores: tie.scores, scales: tie.scales, topK: k) {
        print("near-tie: CPU FP64 order starts with expert " + String(cpu.ids[0]) + ", Float d order starts with expert 1" + (cpu.ids[0] == 0 ? " (exp collapse observed)" : ""))
    }
    let short = [Float](repeating: 0, count: 127)
    h.check((try? RouterSelectionKernel.mirror(ir, scores: short, scales: short)) == nil, "mirror rejects a wrong-length vector")
}

func runAll() -> Int32 {
    var h = Harness()
    h.check(RouterOverlapModelGate.readyToken == "BENCHMARK_READY" && RouterOverlapModelGate.goToken == "GO" && RouterOverlapModelGate.endToken == "TIMED_WINDOW_END", "gate protocol tokens used by the component")
    checkIR(&h)
    checkLayout(&h)
    checkConstantsAndSource(&h)
    checkOffsets(&h)
    checkSelection(&h)
    print("router-selection-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
