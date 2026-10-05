import Foundation
import LeviathanCore

// Phase 55 stage 1 standalone GPU router-selection component.
// Only synthetic scores, scales, offset tables and payload tags are used.
// Before GO: arguments, the exclusive report path, CPU-only cases, CPU references and the host mirror; no MTLDevice.
// After an exact GO: one selector + barrier + consumer command buffer via RouterSelectionGPUComponent, bounded completion.
// This is not the E4 real-weights model gate. It never sets gainQualified or installable, and makes no performance claim.

enum ExitCode {
    static let pass: Int32 = 0
    static let fail: Int32 = 1
    static let usage: Int32 = 2
    static let protocolRefused: Int32 = 3
    static let gpuError: Int32 = 4
    static let notEnded: Int32 = 5
    static let reportError: Int32 = 6
}

enum Synthetic {
    static let layers = 30
    static let badScaleLayer = 28
    static let sweepLayer = 29
    static let expertBytes: UInt64 = 16
    static let seed: UInt64 = 0x5253_4C31_0055_0002
    static let scaleSeed: UInt64 = 0x5253_4C31_0055_0003
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

func stderrLine(_ s: String) {
    FileHandle.standardError.write(Data(s.utf8))
    FileHandle.standardError.write(Data([10]))
}

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

/// Bounded deterministic search for a score x where 1/(1+7e^x) or e^x/(1+7e^x) is within a relative 2^-46 of a Float midpoint.
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

struct GPUCase {
    let name: String
    let family: String
    let layer: UInt32
    let scores: [Float]
    let declared: RouterSelectionFlags
}

/// Synthetic per-layer scales. Layer 28 carries NaN, +inf, subnormal, 2^-100, +0 and -0 at experts 0...5.
func layerScales() -> [[Float]] {
    var rng = SplitMix64(state: Synthetic.scaleSeed)
    var all: [[Float]] = []
    for layer in 0..<Synthetic.layers {
        var s = randomScales(&rng)
        if layer == Synthetic.badScaleLayer {
            s[0] = .nan
            s[1] = .infinity
            s[2] = .leastNonzeroMagnitude
            s[3] = Float(sign: .plus, exponent: -100, significand: 1)
            s[4] = 0
            s[5] = -0.0
        }
        all.append(s)
    }
    return all
}

/// Fixed case list: 187 ordinary cases (must be unflagged and bit exact) and 16 declared refusals.
func makeCases(roundBoundaryX x: Float) -> [GPUCase] {
    var rng = SplitMix64(state: Synthetic.seed)
    var out: [GPUCase] = []
    func add(_ family: String, _ layer: Int, _ scores: [Float], _ declared: RouterSelectionFlags) {
        out.append(GPUCase(name: family + "-" + String(out.count), family: family, layer: UInt32(layer), scores: scores, declared: declared))
    }
    func uniform() -> [Float] { gridScores(&rng, span: 32768, offset: -16384, denominator: 2048) }
    for v in [Float(0), 1.5, -3.25, 7, -8] { add("allEqual", out.count % 28, [Float](repeating: v, count: 128), []) }
    for _ in 0..<64 { add("uniform", out.count % 28, uniform(), []) }
    for _ in 0..<32 { add("ties", out.count % 28, gridScores(&rng, span: 33, offset: -16, denominator: 4), []) }
    for _ in 0..<32 { add("wideRange", out.count % 28, gridScores(&rng, span: 286_720, offset: -245_760, denominator: 4096), []) }
    for _ in 0..<16 {
        var s = gridScores(&rng, span: 12544, offset: -20224, denominator: 256)
        s[Int(rng.below(128))] = 0
        add("dominant", out.count % 28, s, [])
    }
    for c in 0..<32 {
        var s = [Float](repeating: -80, count: 128)
        s[0] = 0
        for j in 1...7 { s[j] = Float(-80.0 * Double(c * 7 + j) / Double(32 * 7 + 1)) }
        add("rangeSweep", Synthetic.sweepLayer, s, [])
    }
    var e1 = [Float](repeating: -80, count: 128)
    for i in 0..<7 { e1[i] = 0 }
    add("rangeEdge", 0, e1, [])
    var e2 = [Float](repeating: -76.5, count: 128)
    e2[64] = 3.5
    add("rangeEdge", 1, e2, [])
    for v in 0..<4 {
        var s = gridScores(&rng, span: 2048, offset: -2048, denominator: 256)
        for j in 0..<8 { s[4 + j] = 1 + Float(j + v) / 16 }
        add("unselectedBadScale", Synthetic.badScaleLayer, s, [])
    }
    var s = uniform()
    s[3] = .nan
    add("nonFiniteNaN", 0, s, .nonFiniteInput)
    s = uniform()
    s[50] = .infinity
    add("nonFinitePosInf", 0, s, .nonFiniteInput)
    s = uniform()
    s[127] = -.infinity
    add("nonFiniteNegInf", 0, s, .nonFiniteInput)
    s = uniform()
    s[9] = Float(sign: .plus, exponent: -110, significand: 1)
    add("tinyNormal", 0, s, .tinyInput)
    s = uniform()
    s[5] = .leastNonzeroMagnitude
    add("subnormalPositive", 0, s, .tinyInput)
    s = uniform()
    s[6] = -Float.leastNormalMagnitude / 2
    add("subnormalNegative", 0, s, .tinyInput)
    s = uniform()
    s[0] = 9
    s[100] = -81
    add("rangeUnselected", 0, s, .rangeUnsupported)
    s = [Float](repeating: -80, count: 128)
    for i in 0..<8 { s[i] = 0 }
    s[127] = Float(-80).nextDown
    add("rangeJustBelow", 0, s, .rangeUnsupported)
    s = [Float](repeating: -100, count: 128)
    s[0] = 0
    add("underflowSubnormalCoefficient", 0, s, .rangeUnsupported)
    s = [Float](repeating: -1, count: 128)
    s[1] = Float(sign: .plus, exponent: -30, significand: 1)
    s[0] = s[1].nextDown
    add("nearTieCollapse", 0, s, .nearTieCollapse)
    s = [Float](repeating: -8, count: 128)
    s[0] = 0
    for j in 1...7 { s[j] = x }
    add("roundBoundary", 0, s, .roundBoundary)
    for i in 0..<3 {
        s = gridScores(&rng, span: 2048, offset: -2048, denominator: 256)
        s[i] = 2
        add("scaleDomain" + String(i), Synthetic.badScaleLayer, s, .scaleDomain)
    }
    s = [Float](repeating: -40, count: 128)
    for i in 10..<17 { s[i] = 0 }
    s[3] = -30
    add("scaleProductUnderflow", Synthetic.badScaleLayer, s, .scaleDomain)
    add("offsetDomainLayer", Synthetic.layers, uniform(), .offsetDomain)
    return out
}

struct Prepared {
    let cases: [GPUCase]
    let scales: [[Float]]
    let table: RouterOffsetTable
    let offsetsProven: Int
    let guPayload: [UInt32]
    let dnPayload: [UInt32]
    let cpu: [RouterSelectionCPUResult?]
    let mirror: [RouterSelectionMirrorResult]
    let sourceSHA256: String
    let sourceBytes: Int
    let irSHA256: String
    let inputSHA256: String
    let executableSHA256: String
    let roundBoundaryBits: UInt32
}

func effectiveLayer(_ layer: UInt32) -> Int { Int(layer) < Synthetic.layers ? Int(layer) : 0 }

func inputDigest(cases: [GPUCase], scales: [[Float]]) -> String {
    var d = Data()
    func put(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    for c in cases {
        d.append(contentsOf: Array(c.name.utf8))
        d.append(0)
        put(c.layer)
        put(c.declared.rawValue)
        for s in c.scores { put(s.bitPattern) }
    }
    for layer in scales {
        for s in layer { put(s.bitPattern) }
    }
    return RouterSelectionKernel.sha256Hex(d)
}

func executableDigest() -> String {
    guard let url = Bundle.main.executableURL, let data = try? Data(contentsOf: url) else { return "unavailable" }
    return RouterSelectionKernel.sha256Hex(data)
}

/// CPU-only preparation before GO: no Metal device is created here.
func prepare(_ ir: RouterSelectionIR) throws -> Prepared {
    try ir.validate()
    guard let x = findRoundBoundaryScore(limit: Synthetic.roundBoundarySearchLimit) else {
        throw RouterSelectionError("no round-boundary score within the fixed search bound")
    }
    let cases = makeCases(roundBoundaryX: x)
    guard cases.count <= RouterSelectionKernel.maxCases else { throw RouterSelectionError("too many cases") }
    let scales = layerScales()
    let table = RouterOffsetTable.synthetic(layers: Synthetic.layers, expertCount: ir.expertCount, expertBytes: Synthetic.expertBytes)
    let proven = try table.validate()
    var cpu: [RouterSelectionCPUResult?] = []
    var mir: [RouterSelectionMirrorResult] = []
    for c in cases {
        let layerScale = scales[effectiveLayer(c.layer)]
        cpu.append(RouterSelectionKernel.cpuReference(scores: c.scores, scales: layerScale, topK: ir.topK))
        mir.append(try RouterSelectionKernel.mirror(ir, scores: c.scores, scales: layerScale))
    }
    let source = try RouterSelectionKernel.emitSource(ir)
    return Prepared(
        cases: cases, scales: scales, table: table, offsetsProven: proven,
        guPayload: table.syntheticPayload(dn: false), dnPayload: table.syntheticPayload(dn: true),
        cpu: cpu, mirror: mir, sourceSHA256: RouterSelectionKernel.sourceDigest(source), sourceBytes: source.utf8.count,
        irSHA256: try RouterSelectionKernel.irDigest(ir), inputSHA256: inputDigest(cases: cases, scales: scales),
        executableSHA256: executableDigest(), roundBoundaryBits: x.bitPattern)
}

struct CaseReport: Codable {
    let name: String
    let family: String
    let layer: UInt32
    let declared: [String]
    let gpuFlags: [String]
    let ids: [UInt32]
    let coefficientBits: [UInt32]
    let cpuIds: [Int]?
    let cpuCoefficientBits: [UInt32]?
    let verdict: String
    let problems: [String]
}

struct ComponentReport: Codable {
    let schema: String
    let status: String
    let completion: String
    let terminal: Bool
    let errorDescription: String?
    let protocolTokens: [String]
    let ir: RouterSelectionIR
    let irSHA256: String
    let sourceSHA256: String
    let componentSourceSHA256: String?
    let sourceDigestMatches: Bool?
    let sourceBytes: Int
    let executableSHA256: String
    let casesInputSHA256: String
    let hardware: RouterSelectionHardware?
    let offsetsProven: Int
    let roundBoundaryScoreBits: UInt32
    let caseCount: Int
    let ordinaryCount: Int
    let declaredRefusalCount: Int
    let exactMatches: Int
    let refusedAsDeclared: Int
    let unflaggedMismatches: Int
    let unexpectedFlags: Int
    let missingDeclaredFlags: Int
    let refusedShapeErrors: Int
    let consumerErrors: Int
    let recordErrors: Int
    let preflightErrors: Int
    let guardsIntact: Bool
    let gpuMicros: Double?
    let syntheticInputsOnly: Bool
    let originalModelAccessed: Bool
    let modelDownloadedOrCached: Bool
    let gainQualified: Bool
    let installable: Bool
    let productionPerformanceClaim: Bool
    let realE4GateComplete: Bool
    let refusalsProveCoverage: Bool
    let limitations: [String]
    let cases: [CaseReport]
}

struct Evaluation {
    var cases: [CaseReport] = []
    var exactMatches = 0
    var refusedAsDeclared = 0
    var categories: [String: Int] = [:]

    func count(_ category: String) -> Int { categories[category] ?? 0 }
}

/// Strict per-case check. Ordinary cases: zero flags plus ids, quotient bits and coefficient bits equal to the exact CPU
/// reference. Declared refusals: the declared flag is present, no unexplained flag, ids 0 and zero coefficients.
/// In both cases the GU/DN offsets and the consumer's reads through the record must match.
func evaluateCase(
    _ tc: GPUCase, cpu: RouterSelectionCPUResult?, mirror mir: RouterSelectionMirrorResult,
    record rec: RouterSelectionGPURecord, consumed: [UInt32], table: RouterOffsetTable, topK k: Int
) -> [String] {
    var p: [String] = []
    let eff = effectiveLayer(tc.layer)
    let zeros = [UInt32](repeating: 0, count: k)
    if rec.magic != RouterSelectionKernel.recordMagic { p.append("RECORD:magic") }
    if rec.layer != UInt32(eff) { p.append("RECORD:layer") }
    if !rec.paddingClean { p.append("RECORD:padding") }
    guard rec.ids.count == k, rec.guOffsets.count == k, rec.dnOffsets.count == k, consumed.count == 4 * k else {
        return p + ["RECORD:shape"]
    }
    var expectIds = zeros
    var expectWeights = zeros
    if tc.declared.isEmpty {
        if let cpu {
            if !mir.flags.isEmpty || mir.ids != cpu.ids || mir.weightBits != cpu.weightBits || mir.rawBits != cpu.rawBits {
                p.append("PREFLIGHT:host mirror differs from the CPU reference")
            }
            if rec.flags != 0 {
                p.append("UNEXPECTED_FLAGS:" + rec.refusal.labels.joined(separator: ","))
            } else {
                if rec.ids.map({ Int($0) }) != cpu.ids { p.append("MISMATCH:ids") }
                if rec.rawBits != cpu.rawBits { p.append("MISMATCH:quotientBits") }
                if rec.weightBits != cpu.weightBits { p.append("MISMATCH:coefficientBits") }
                if !rec.weightBits.allSatisfy({ Float(bitPattern: $0).isFinite }) { p.append("MISMATCH:nonFiniteCoefficient") }
                expectIds = cpu.ids.map { UInt32($0) }
                expectWeights = cpu.weightBits
            }
        } else {
            p.append("PREFLIGHT:no CPU reference for an ordinary case")
        }
    } else {
        if !rec.refusal.isSuperset(of: tc.declared) { p.append("MISSING_DECLARED_FLAG") }
        if !rec.refusal.isSubset(of: tc.declared.union(mir.flags)) {
            p.append("UNEXPECTED_FLAGS:" + rec.refusal.labels.joined(separator: ","))
        }
        let hostDeclared = tc.declared.subtracting(.offsetDomain)
        if !hostDeclared.isEmpty && !mir.flags.isSuperset(of: hostDeclared) { p.append("PREFLIGHT:host mirror did not refuse") }
        if rec.ids != zeros || rec.rawBits != zeros || rec.weightBits != zeros { p.append("REFUSED_SHAPE:nonzero ids or coefficients") }
    }
    for s in 0..<k {
        let id = Int(expectIds[s])
        let gu = table.offset(layer: eff, dn: false, expert: id)
        let dn = table.offset(layer: eff, dn: true, expert: id)
        if rec.guOffsets[s] != gu || rec.dnOffsets[s] != dn { p.append("RECORD:offsets slot " + String(s)) }
        let got = Array(consumed[(4 * s)..<(4 * s + 4)])
        let want: [UInt32] = [
            RouterOffsetTable.syntheticTag(dn: false, layer: eff, expert: id),
            RouterOffsetTable.syntheticTag(dn: true, layer: eff, expert: id),
            expectWeights[s], 0,
        ]
        if got != want { p.append("CONSUMER:slot " + String(s)) }
    }
    return p
}

func evaluateAll(_ p: Prepared, outcome: RouterSelectionGPUOutcome, topK k: Int) -> Evaluation {
    var ev = Evaluation()
    for (i, tc) in p.cases.enumerated() {
        let rec = outcome.records[i]
        let problems = evaluateCase(tc, cpu: p.cpu[i], mirror: p.mirror[i], record: rec, consumed: outcome.consumed[i], table: p.table, topK: k)
        for category in Set(problems.map { String($0.split(separator: ":").first ?? "") }) {
            ev.categories[category, default: 0] += 1
        }
        let verdict: String
        if !problems.isEmpty {
            verdict = "FAIL"
        } else if tc.declared.isEmpty {
            verdict = "EXACT_MATCH"
            ev.exactMatches += 1
        } else {
            verdict = "REFUSED_AS_DECLARED"
            ev.refusedAsDeclared += 1
        }
        ev.cases.append(CaseReport(
            name: tc.name, family: tc.family, layer: tc.layer, declared: tc.declared.labels, gpuFlags: rec.refusal.labels,
            ids: rec.ids, coefficientBits: rec.weightBits, cpuIds: p.cpu[i]?.ids, cpuCoefficientBits: p.cpu[i]?.weightBits,
            verdict: verdict, problems: problems))
    }
    return ev
}

func buildReport(
    _ p: Prepared, ir: RouterSelectionIR, status: String, completion: String, terminal: Bool, error: String?,
    hardware: RouterSelectionHardware?, componentSourceSHA256: String?, guardsIntact: Bool, gpuMicros: Double?,
    evaluation ev: Evaluation?
) -> ComponentReport {
    let ordinary = p.cases.filter { $0.declared.isEmpty }.count
    let limitations = [
        "Synthetic score vectors, scales, offset tables and payload tags only; this tool opened, hashed, cached and downloaded no model file.",
        "Declared refusals are counted separately and are never equivalence passes; refusals do not prove that every input is preserved.",
        "The 2^-42 rounding-certificate budget is an estimate from published double-word bounds plus fixed-seed samples, not a proven theorem.",
        "Stage 1 component with a tiny offset consumer; it is not the E4 real-weights model gate and claims no runtime gain.",
    ]
    return ComponentReport(
        schema: "leviathan.router-selection-component.v1", status: status, completion: completion, terminal: terminal,
        errorDescription: error,
        protocolTokens: [RouterOverlapModelGate.readyToken, RouterOverlapModelGate.goToken, RouterOverlapModelGate.endToken],
        ir: ir, irSHA256: p.irSHA256, sourceSHA256: p.sourceSHA256, componentSourceSHA256: componentSourceSHA256,
        sourceDigestMatches: componentSourceSHA256.map { $0 == p.sourceSHA256 }, sourceBytes: p.sourceBytes,
        executableSHA256: p.executableSHA256, casesInputSHA256: p.inputSHA256, hardware: hardware,
        offsetsProven: p.offsetsProven, roundBoundaryScoreBits: p.roundBoundaryBits,
        caseCount: p.cases.count, ordinaryCount: ordinary, declaredRefusalCount: p.cases.count - ordinary,
        exactMatches: ev?.exactMatches ?? 0, refusedAsDeclared: ev?.refusedAsDeclared ?? 0,
        unflaggedMismatches: ev?.count("MISMATCH") ?? 0, unexpectedFlags: ev?.count("UNEXPECTED_FLAGS") ?? 0,
        missingDeclaredFlags: ev?.count("MISSING_DECLARED_FLAG") ?? 0, refusedShapeErrors: ev?.count("REFUSED_SHAPE") ?? 0,
        consumerErrors: ev?.count("CONSUMER") ?? 0, recordErrors: ev?.count("RECORD") ?? 0,
        preflightErrors: ev?.count("PREFLIGHT") ?? 0, guardsIntact: guardsIntact, gpuMicros: gpuMicros,
        syntheticInputsOnly: true, originalModelAccessed: false, modelDownloadedOrCached: false, gainQualified: false,
        installable: false, productionPerformanceClaim: false, realE4GateComplete: false, refusalsProveCoverage: false,
        limitations: limitations, cases: ev?.cases ?? [])
}

/// Exclusive, bounded write: never overwrites, never trims. Over 1 MiB is an error.
func writeReport(_ report: ComponentReport, to path: String) -> Bool {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(report) else {
        stderrLine("report encoding failed")
        return false
    }
    guard data.count <= 1 << 20 else {
        stderrLine("report exceeds 1 MiB; refusing to trim")
        return false
    }
    do {
        try data.write(to: URL(fileURLWithPath: path), options: [.withoutOverwriting])
        return true
    } catch {
        stderrLine("report write failed: " + String(describing: error))
        return false
    }
}

struct Options {
    let reportPath: String
    let timeoutSeconds: Double
}

func parseOptions(_ args: [String]) -> Options? {
    var report: String?
    var timeout = 10.0
    var i = 0
    while i < args.count {
        guard i + 1 < args.count else { return nil }
        let value = args[i + 1]
        if args[i] == "--report" {
            report = value
        } else if args[i] == "--timeout-seconds", let t = Double(value), t > 0, t <= 30 {
            timeout = t
        } else {
            return nil
        }
        i += 2
    }
    guard let r = report, !r.isEmpty else { return nil }
    return Options(reportPath: r, timeoutSeconds: timeout)
}

func reportPathUsable(_ path: String) -> Bool {
    let url = URL(fileURLWithPath: path)
    var isDir: ObjCBool = false
    let parent = url.deletingLastPathComponent().path
    return !FileManager.default.fileExists(atPath: url.path)
        && FileManager.default.fileExists(atPath: parent, isDirectory: &isDir) && isDir.boolValue
}

func runMain() -> Int32 {
    guard let options = parseOptions(Array(CommandLine.arguments.dropFirst())) else {
        stderrLine("usage: router-selection-component --report <new-json-path> [--timeout-seconds <(0,30]>]")
        return ExitCode.usage
    }
    guard reportPathUsable(options.reportPath) else {
        stderrLine("report path already exists or its parent directory is missing; nothing written")
        return ExitCode.usage
    }
    let ir = RouterSelectionIR.gemma4
    let p: Prepared
    do {
        p = try prepare(ir)
    } catch {
        stderrLine("CPU preparation failed before GO; no GPU work: " + String(describing: error))
        return ExitCode.fail
    }
    print(RouterOverlapModelGate.readyToken)
    fflush(nil)
    guard let line = readLine(strippingNewline: true), line == RouterOverlapModelGate.goToken else {
        stderrLine("REFUSED: exact " + RouterOverlapModelGate.goToken + " not received; no Metal device was created and no GPU work was dispatched")
        return ExitCode.protocolRefused
    }
    let component: RouterSelectionGPUComponent
    let outcome: RouterSelectionGPUOutcome
    do {
        component = try RouterSelectionGPUComponent(ir: ir)
        outcome = try component.run(
            scores: p.cases.map { $0.scores }, caseLayers: p.cases.map { $0.layer }, scales: p.scales, offsets: p.table,
            guPayload: p.guPayload, dnPayload: p.dnPayload, timeoutSeconds: options.timeoutSeconds)
    } catch {
        // Every throw in the component precedes commit, so nothing is in flight and the process is quiescent.
        let report = buildReport(
            p, ir: ir, status: "ERROR", completion: "NOT_COMMITTED", terminal: true, error: String(describing: error),
            hardware: nil, componentSourceSHA256: nil, guardsIntact: false, gpuMicros: nil, evaluation: nil)
        let written = writeReport(report, to: options.reportPath)
        stderrLine("component error before commit: " + String(describing: error))
        print(RouterOverlapModelGate.endToken)
        fflush(nil)
        return written ? ExitCode.gpuError : ExitCode.reportError
    }
    if !outcome.terminal {
        let report = buildReport(
            p, ir: ir, status: "NOT_ENDED", completion: outcome.completion, terminal: false, error: outcome.errorDescription,
            hardware: component.hardware, componentSourceSHA256: component.sourceSHA256, guardsIntact: false, gpuMicros: nil,
            evaluation: nil)
        _ = writeReport(report, to: options.reportPath)
        stderrLine("GPU completion not observed within " + String(options.timeoutSeconds) + " s; quiescence unknown")
        print("NOT_ENDED")
        fflush(nil)
        return ExitCode.notEnded
    }
    guard outcome.completion == "COMPLETED", outcome.records.count == p.cases.count, outcome.consumed.count == p.cases.count else {
        let report = buildReport(
            p, ir: ir, status: "ERROR", completion: outcome.completion, terminal: true, error: outcome.errorDescription,
            hardware: component.hardware, componentSourceSHA256: component.sourceSHA256, guardsIntact: outcome.guardsIntact,
            gpuMicros: outcome.gpuMicros, evaluation: nil)
        let written = writeReport(report, to: options.reportPath)
        stderrLine("GPU command buffer ended with " + outcome.completion + ": " + (outcome.errorDescription ?? "record count mismatch"))
        print(RouterOverlapModelGate.endToken)
        fflush(nil)
        return written ? ExitCode.gpuError : ExitCode.reportError
    }
    let ev = evaluateAll(p, outcome: outcome, topK: ir.topK)
    let ordinary = p.cases.filter { $0.declared.isEmpty }.count
    let refusals = p.cases.count - ordinary
    let sourceMatches = component.sourceSHA256 == p.sourceSHA256
    let pass = ev.categories.isEmpty && outcome.guardsIntact && sourceMatches
        && ev.exactMatches == ordinary && ev.refusedAsDeclared == refusals
    let report = buildReport(
        p, ir: ir, status: pass ? "PASS" : "FAIL", completion: outcome.completion, terminal: true, error: nil,
        hardware: component.hardware, componentSourceSHA256: component.sourceSHA256, guardsIntact: outcome.guardsIntact,
        gpuMicros: outcome.gpuMicros, evaluation: ev)
    let written = writeReport(report, to: options.reportPath)
    let head = "router-selection-component " + (pass ? "PASS" : "FAIL")
    let exact = " exact " + String(ev.exactMatches) + "/" + String(ordinary)
    let refused = " refusedAsDeclared " + String(ev.refusedAsDeclared) + "/" + String(refusals)
    let guards = " guards " + (outcome.guardsIntact ? "intact" : "VIOLATED") + " source " + (sourceMatches ? "matches" : "DRIFTED")
    stderrLine(head + exact + refused + guards)
    for (category, n) in ev.categories.sorted(by: { $0.key < $1.key }) {
        stderrLine("  " + category + " cases " + String(n))
    }
    print(RouterOverlapModelGate.endToken)
    fflush(nil)
    if !written { return ExitCode.reportError }
    return pass ? ExitCode.pass : ExitCode.fail
}

exit(runMain())
