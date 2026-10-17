// tools/q4_qualification/main.swift
//
// Phase-15 Q4 launch-geometry QUALIFIER. Standalone, original source,
// compiled locally against LeviathanCore through its PUBLIC API only.
//
// Default (synthetic) mode needs no model file and checks:
//   1. the v_eq_k hot-shape metadata resolver with synthetic lookups:
//      global layers omit attn_v; a missing LOCAL attn_v, a present
//      global attn_v, missing/malformed required tensors and lookup
//      errors all fail closed;
//   2. runMatvec pipeline/buffer bounds, rejected before any encode;
//   3. NaN-sentinel poisoning: a deliberately partial writer cannot
//      reuse a previous run's rows and is rejected by the finite +
//      bit-identity acceptance rule;
//   4. tail coverage at rows 33 and 1000 across the legal width ladder:
//      full-row finite, bit-identical to the default width, and within
//      1e-4 of an independent FP64 oracle written here from scratch.
//
// --component --gguf PATH [--out FILE] builds the ACTUAL runner on an
// existing approved GGUF and runs the full-row component evaluation on
// the runner's cached pipeline, printing JSON. Screen timings are
// per-dispatch GPU medians for RANKING ONLY: they are not end-to-end
// performance and carry no occupancy claim.
//
// Exit status: 0 when every check passes, 1 on any failed check or
// error, 2 on usage errors.

import Foundation
import Metal
import LeviathanCore

struct QualError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

final class Checks {
    struct Entry: Encodable {
        let name: String
        let pass: Bool
        let detail: String
    }
    private(set) var entries: [Entry] = []

    func record(_ name: String, _ pass: Bool, _ detail: String = "") {
        entries.append(Entry(name: name, pass: pass, detail: detail))
    }

    func expectThrows(_ name: String, _ body: () throws -> Void) {
        do {
            try body()
            record(name, false, "expected rejection, but the call succeeded")
        } catch {
            record(name, true, "rejected: \(error)")
        }
    }

    var failed: Int { entries.filter { !$0.pass }.count }
}

func encodeJSON<T: Encodable>(_ value: T) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.nonConformingFloatEncodingStrategy = .convertToString(
        positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    guard let data = try? encoder.encode(value),
          let text = String(data: data, encoding: .utf8) else {
        return "{\"error\": \"report encoding failed\"}"
    }
    return text
}

// MARK: - Deterministic synthetic data and independent FP64 oracle

struct QualRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Double { Double(next() >> 11) / 9_007_199_254_740_992.0 }
}

/// Synthetic Q4_0 matrix (rows x inDim) in the GGUF block layout: per
/// block a little-endian FP16 scale followed by 16 nibble bytes (low
/// nibble = value l, high nibble = value 16 + l of the block).
struct SyntheticQ4 {
    let inDim: Int
    let rows: Int
    let bytes: [UInt8]
    let x: [Float]

    init(inDim: Int, rows: Int, seed: UInt64) {
        var rng = QualRNG(state: seed)
        var b = [UInt8]()
        b.reserveCapacity(rows * (inDim / 32) * 18)
        for _ in 0..<(rows * (inDim / 32)) {
            let magnitude = 0.005 + 0.045 * rng.unit()
            let scale = Float16(Float(rng.unit() < 0.5 ? -magnitude : magnitude))
            let bits = scale.bitPattern
            b.append(UInt8(bits & 0xFF))
            b.append(UInt8(bits >> 8))
            for _ in 0..<16 { b.append(UInt8(truncatingIfNeeded: rng.next())) }
        }
        var v = [Float]()
        v.reserveCapacity(inDim)
        for _ in 0..<inDim { v.append(Float(rng.unit() * 2 - 1)) }
        self.inDim = inDim
        self.rows = rows
        self.bytes = b
        self.x = v
    }

    /// Independent FP64 reference over EVERY row.
    func oracle() -> [Double] {
        let blocksPerRow = inDim / 32
        var out = [Double](repeating: 0, count: rows)
        for r in 0..<rows {
            var sum = 0.0
            for blk in 0..<blocksPerRow {
                let base = (r * blocksPerRow + blk) * 18
                let dBits = UInt16(bytes[base]) | (UInt16(bytes[base + 1]) << 8)
                let d = Double(Float(Float16(bitPattern: dBits)))
                for l in 0..<16 {
                    let q = bytes[base + 2 + l]
                    sum += d * (Double(q & 0x0F) - 8) * Double(x[blk * 32 + l])
                    sum += d * (Double(q >> 4) - 8) * Double(x[blk * 32 + 16 + l])
                }
            }
            out[r] = sum
        }
        return out
    }
}

func worstRel(_ out: [Float], _ oracle: [Double]) -> Double {
    guard out.count == oracle.count else { return .infinity }
    var worst = 0.0
    for i in 0..<out.count {
        let rel = abs(Double(out[i]) - oracle[i]) / max(1.0, abs(oracle[i]))
        worst = rel.isNaN ? .infinity : max(worst, rel)
    }
    return worst
}

func bitIdentical(_ a: [Float], _ b: [Float]) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
}

/// The component acceptance rule restated: every row finite AND
/// bit-identical to the incumbent output.
func wouldAccept(_ out: [Float], _ incumbent: [Float]) -> Bool {
    out.allSatisfy { $0.isFinite } && bitIdentical(out, incumbent)
}

// MARK: - 1. v_eq_k metadata resolver (pure, synthetic lookups)

typealias DirEntry = (dims: [Int], ggmlType: UInt32)

func syntheticDirectory(layerCount: Int, isGlobal: (Int) -> Bool) -> [String: DirEntry] {
    let shapes: [String: [Int]] = [
        "attn_q": [2816, 4096], "attn_k": [2816, 1024], "attn_v": [2816, 1024],
        "attn_output": [4096, 2816], "ffn_gate": [2816, 2112],
        "ffn_up": [2816, 2112], "ffn_down": [2112, 2816],
    ]
    var directory = [String: DirEntry]()
    for layer in 0..<layerCount {
        for (kind, dims) in shapes {
            if kind == "attn_v" && isGlobal(layer) { continue }
            directory["blk.\(layer).\(kind).weight"] = (dims: dims, ggmlType: 2)
        }
    }
    return directory
}

func resolveSynthetic(_ directory: [String: DirEntry], layerCount: Int = 30,
                      isGlobal: (Int) -> Bool = { $0 % 6 == 5 }) throws -> [Q4LaunchHotShape] {
    try Q4LaunchHotShapes.resolve(layerCount: layerCount, isGlobalLayer: isGlobal,
                                  lookup: { directory[$0] })
}

func resolverChecks(_ checks: Checks) {
    let layerCount = 30
    let isGlobal: (Int) -> Bool = { $0 % 6 == 5 }
    let sampled = Q4LaunchHotShapes.sampledLayers.filter { $0 >= 0 && $0 < layerCount }
    guard let local = sampled.first(where: { !isGlobal($0) }),
          let global = sampled.first(where: isGlobal) else {
        checks.record("resolver: synthetic architecture exercises a local AND a global sampled layer",
                      false, "sampled layers \(sampled)")
        return
    }
    let base = syntheticDirectory(layerCount: layerCount, isGlobal: isGlobal)
    let headline = "resolver: v_eq_k global layer omits attn_v, every other required tensor resolves"
    do {
        let shapes = try resolveSynthetic(base)
        let names = Set(shapes.map(\.tensor))
        let expected = sampled.reduce(0) { $0 + (isGlobal($1) ? 6 : 7) }
        let ok = shapes.count == expected
            && !names.contains("blk.\(global).attn_v.weight")
            && names.contains("blk.\(global).attn_k.weight")
            && names.contains("blk.\(local).attn_v.weight")
            && shapes.allSatisfy {
                $0.inDim % 32 == 0
                    && $0.oracleRows == min($0.outDim, Q4LaunchHotShapes.maxOracleRows) }
        checks.record(headline, ok,
                      "\(shapes.count) shapes (expected \(expected)), sampled \(sampled)")
    } catch {
        checks.record(headline, false, "threw: \(error)")
    }
    var noLocalV = base
    noLocalV["blk.\(local).attn_v.weight"] = nil
    checks.expectThrows("resolver: missing LOCAL attn_v fails closed (never treated as shared)") {
        _ = try resolveSynthetic(noLocalV)
    }
    var globalV = base
    globalV["blk.\(global).attn_v.weight"] = (dims: [2816, 1024], ggmlType: 2)
    checks.expectThrows("resolver: attn_v present on a GLOBAL (v_eq_k) layer fails closed") {
        _ = try resolveSynthetic(globalV)
    }
    var noGlobalK = base
    noGlobalK["blk.\(global).attn_k.weight"] = nil
    checks.expectThrows("resolver: missing GLOBAL attn_k fails closed") {
        _ = try resolveSynthetic(noGlobalK)
    }
    var f16 = base
    f16["blk.\(local).attn_q.weight"] = (dims: [2816, 4096], ggmlType: 1)
    checks.expectThrows("resolver: non-Q4_0 required tensor fails closed") {
        _ = try resolveSynthetic(f16)
    }
    var rank3 = base
    rank3["blk.\(local).ffn_up.weight"] = (dims: [2816, 2112, 2], ggmlType: 2)
    checks.expectThrows("resolver: 3-D required tensor fails closed") {
        _ = try resolveSynthetic(rank3)
    }
    var ragged = base
    ragged["blk.\(local).ffn_down.weight"] = (dims: [2100, 2816], ggmlType: 2)
    checks.expectThrows("resolver: inDim not a 32-multiple fails closed") {
        _ = try resolveSynthetic(ragged)
    }
    checks.expectThrows("resolver: lookup errors propagate (never mapped to absent)") {
        _ = try Q4LaunchHotShapes.resolve(layerCount: layerCount, isGlobalLayer: isGlobal,
                                          lookup: { name in
            if name.hasSuffix(".attn_k.weight") { throw QualError("synthetic directory failure") }
            return base[name]
        })
    }
    checks.expectThrows("resolver: layerCount 0 fails closed") {
        _ = try resolveSynthetic(base, layerCount: 0)
    }
    let allLocal = syntheticDirectory(layerCount: layerCount, isGlobal: { _ in false })
    let allLocalHeadline = "resolver: all-local architecture resolves attn_v on every sampled layer"
    do {
        let shapes = try resolveSynthetic(allLocal, isGlobal: { _ in false })
        checks.record(allLocalHeadline, shapes.count == sampled.count * 7,
                      "\(shapes.count) shapes")
    } catch {
        checks.record(allLocalHeadline, false, "threw: \(error)")
    }
    var allLocalNoV = allLocal
    allLocalNoV["blk.\(global).attn_v.weight"] = nil
    checks.expectThrows("resolver: all-local architecture with a missing attn_v fails closed") {
        _ = try resolveSynthetic(allLocalNoV, isGlobal: { _ in false })
    }
}

// MARK: - GPU harness (synthetic weights only; no model file)

enum Sources {
    /// Deliberately DEFECTIVE writer with the g4_q4_matvec binding
    /// layout: it writes only even rows (value 0) and skips every odd
    /// row. Used only to prove that sentinel poisoning exposes rows a
    /// launch failed to write.
    static let partialWriter = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void q4qual_partial_writer(constant const uint2& p [[buffer(0)]],
                                      constant const ulong& wOff [[buffer(1)]],
                                      device const uchar* wbase [[buffer(2)]],
                                      device const float* x [[buffer(3)]],
                                      device float* y [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {
        if (gid >= p.y || (gid & 1u) == 1u) { return; }
        y[gid] = 0.0f;
    }
    """
}

struct GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let q4: MTLComputePipelineState
    let partial: MTLComputePipelineState
}

func makeGPU() throws -> GPU {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw QualError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    guard let fn = library.makeFunction(name: Gemma4Runner.q4LaunchKernel) else {
        throw QualError("g4_q4_matvec missing from the runner shader source")
    }
    let q4 = try device.makeComputePipelineState(function: fn)
    let partialLibrary = try device.makeLibrary(source: Sources.partialWriter, options: nil)
    guard let pfn = partialLibrary.makeFunction(name: "q4qual_partial_writer") else {
        throw QualError("partial writer kernel missing")
    }
    let partial = try device.makeComputePipelineState(function: pfn)
    return GPU(device: device, queue: queue, q4: q4, partial: partial)
}

func sharedBuffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
    let made = values.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress, raw.count > 0 else { return nil }
        return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
    }
    guard let buffer = made else { throw QualError("buffer allocation failed") }
    return buffer
}

func emptyShared(_ device: MTLDevice, _ length: Int) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw QualError("buffer allocation failed")
    }
    return buffer
}

/// Legal width ladder mirroring the deriver: executionWidth x {1,2,4,8}
/// (powers of two within the pipeline maximum) plus the control, the
/// pipeline maximum itself (which may be a non-power-of-two).
func widthLadder(_ p: MTLComputePipelineState) -> [Int] {
    let tew = p.threadExecutionWidth
    let maxThreads = p.maxTotalThreadsPerThreadgroup
    var widths = Set<Int>()
    for m in [1, 2, 4, 8] {
        let (v, overflow) = tew.multipliedReportingOverflow(by: m)
        if !overflow, v > 0, v & (v - 1) == 0, v <= maxThreads { widths.insert(v) }
    }
    widths.insert(maxThreads)
    return widths.sorted()
}

// MARK: - 2. runMatvec bounds (rejected before any encode)

func boundsChecks(_ g: GPU, _ checks: Checks) throws {
    let rows = 33, inDim = 256
    let s = SyntheticQ4(inDim: inDim, rows: rows, seed: 0x424F_554E_4453)
    let wb = try sharedBuffer(g.device, s.bytes)
    let xb = try sharedBuffer(g.device, s.x)
    let yb = try emptyShared(g.device, rows * 4)
    let maxThreads = g.q4.maxTotalThreadsPerThreadgroup
    let defaultWidth = max(1, min(rows, maxThreads))
    func run(w: MTLBuffer? = nil, x: MTLBuffer? = nil, y: MTLBuffer? = nil,
             inDim d: Int = 256, rows r: Int = 33, width: Int, offset: Int = 0,
             pow2: Bool = true) throws -> [Float] {
        try Q4LaunchForge.runMatvec(
            pipe: g.q4, queue: g.queue, wb: w ?? wb, xb: x ?? xb, yb: y ?? yb,
            inDim: d, rows: r, width: width, weightOffset: offset,
            requirePowerOfTwo: pow2).out
    }
    let reference = try run(width: defaultWidth, pow2: false)
    checks.expectThrows("bounds: width above the pipeline maximum") {
        _ = try run(width: maxThreads + 1, pow2: false)
    }
    checks.expectThrows("bounds: zero width") { _ = try run(width: 0, pow2: false) }
    let oddWidth = 48
    if oddWidth <= maxThreads {
        checks.expectThrows("bounds: non-power-of-two CANDIDATE width \(oddWidth) rejected") {
            _ = try run(width: oddWidth)
        }
        let out = try run(width: oddWidth, pow2: false)
        checks.record("bounds: non-power-of-two width \(oddWidth) only via the explicit control path, bit-identical",
                      wouldAccept(out, reference))
    }
    checks.expectThrows("bounds: zero rows") { _ = try run(rows: 0, width: 1) }
    checks.expectThrows("bounds: inDim not a 32-multiple") { _ = try run(inDim: 48, width: 1) }
    let shortY = try emptyShared(g.device, rows * 4 - 4)
    checks.expectThrows("bounds: undersized output buffer") { _ = try run(y: shortY, width: 1) }
    let shortX = try emptyShared(g.device, inDim * 4 - 4)
    checks.expectThrows("bounds: undersized input buffer") { _ = try run(x: shortX, width: 1) }
    let shortW = try sharedBuffer(g.device, Array(s.bytes.dropLast()))
    checks.expectThrows("bounds: undersized weight buffer") { _ = try run(w: shortW, width: 1) }
    checks.expectThrows("bounds: weight offset past the extent") { _ = try run(width: 1, offset: 18) }
    checks.expectThrows("bounds: negative weight offset") { _ = try run(width: 1, offset: -1) }
    checks.expectThrows("bounds: input aliases output") { _ = try run(y: xb, width: 1) }
    checks.expectThrows("bounds: weights alias output") { _ = try run(y: wb, width: 1) }
    if let privateY = g.device.makeBuffer(length: rows * 4, options: .storageModePrivate) {
        checks.expectThrows("bounds: private (non-CPU-visible) output storage rejected") {
            _ = try run(y: privateY, width: 1)
        }
    } else {
        checks.record("bounds: private output buffer allocation", false, "allocation failed")
    }
    let groupsOK = Q4LaunchForge.groups(grid: 33, width: 32) == 2
        && Q4LaunchForge.groups(grid: 1000, width: 256) == 4
        && Q4LaunchForge.groups(grid: 1, width: 1024) == 1
        && Q4LaunchForge.groups(grid: 0, width: 32) == nil
        && Q4LaunchForge.groups(grid: 32, width: 0) == nil
    checks.record("bounds: overflow-safe ceil group formula", groupsOK)
}

// MARK: - 3. Sentinel poisoning versus a partial writer

func partialWriterCheck(_ g: GPU, _ checks: Checks) throws {
    let rows = 33, inDim = 256
    let s = SyntheticQ4(inDim: inDim, rows: rows, seed: 0x5041_5254)
    let wb = try sharedBuffer(g.device, s.bytes)
    let xb = try sharedBuffer(g.device, s.x)
    let yb = try emptyShared(g.device, rows * 4)
    let defaultWidth = max(1, min(rows, g.q4.maxTotalThreadsPerThreadgroup))
    // The incumbent writes EVERY row first: without poisoning, the
    // partial writer's skipped rows would silently keep these values.
    let incumbent = try Q4LaunchForge.runMatvec(
        pipe: g.q4, queue: g.queue, wb: wb, xb: xb, yb: yb,
        inDim: inDim, rows: rows, width: defaultWidth, requirePowerOfTwo: false).out
    var width = 1
    while width * 2 <= min(32, g.partial.maxTotalThreadsPerThreadgroup) { width *= 2 }
    let out = try Q4LaunchForge.runMatvec(
        pipe: g.partial, queue: g.queue, wb: wb, xb: xb, yb: yb,
        inDim: inDim, rows: rows, width: width).out
    let skipped = Array(stride(from: 1, to: rows, by: 2))
    let written = Array(stride(from: 0, to: rows, by: 2))
    let sentinelOnSkipped = out.count == rows && skipped.allSatisfy {
        out[$0].bitPattern == Q4LaunchForge.outputSentinelBits }
    let staleReuse = out.count == rows && skipped.contains {
        out[$0].bitPattern == incumbent[$0].bitPattern }
    let writtenRows = out.count == rows && written.allSatisfy { out[$0] == 0 }
    checks.record("sentinel: rows skipped by a partial writer carry the exact NaN sentinel (no stale incumbent reuse)",
                  sentinelOnSkipped && !staleReuse && writtenRows,
                  "\(skipped.count) skipped rows, \(written.count) written rows, width \(width)")
    checks.record("sentinel: finite + bit-identity acceptance rejects the partial writer",
                  incumbent.allSatisfy { $0.isFinite } && !wouldAccept(out, incumbent))
}

// MARK: - 4. Tail coverage across the legal ladder

func tailCoverage(_ g: GPU, rows: Int, _ checks: Checks) throws {
    let inDim = 256
    let s = SyntheticQ4(inDim: inDim, rows: rows, seed: 0x5134_5155_414C &+ UInt64(rows))
    let wb = try sharedBuffer(g.device, s.bytes)
    let xb = try sharedBuffer(g.device, s.x)
    let yb = try emptyShared(g.device, rows * 4)
    let oracle = s.oracle()
    let maxThreads = g.q4.maxTotalThreadsPerThreadgroup
    let defaultWidth = max(1, min(rows, maxThreads))
    let incumbent = try Q4LaunchForge.runMatvec(
        pipe: g.q4, queue: g.queue, wb: wb, xb: xb, yb: yb,
        inDim: inDim, rows: rows, width: defaultWidth, requirePowerOfTwo: false).out
    let incRel = worstRel(incumbent, oracle)
    checks.record("tail rows=\(rows): incumbent default width \(defaultWidth) full-row finite, FP64 < 1e-4",
                  incumbent.count == rows && incumbent.allSatisfy { $0.isFinite } && incRel < 1e-4,
                  "worst rel " + String(format: "%.3e", incRel))
    for width in widthLadder(g.q4) {
        let isControl = width == maxThreads
        let tag = isControl ? " (control)" : ""
        let out = try Q4LaunchForge.runMatvec(
            pipe: g.q4, queue: g.queue, wb: wb, xb: xb, yb: yb,
            inDim: inDim, rows: rows, width: width, requirePowerOfTwo: !isControl).out
        let rel = worstRel(out, oracle)
        let groups = Q4LaunchForge.groups(grid: rows, width: width) ?? -1
        checks.record("tail rows=\(rows) width=\(width)\(tag): full-row finite, bit-identical, FP64 < 1e-4",
                      out.count == rows && wouldAccept(out, incumbent) && rel < 1e-4,
                      "groups \(groups), worst rel " + String(format: "%.3e", rel))
    }
}

// MARK: - Modes

func runSynthetic() -> Int32 {
    let checks = Checks()
    resolverChecks(checks)
    do {
        let g = try makeGPU()
        checks.record("gpu: pipeline geometry recorded", true,
                      "maxTotalThreadsPerThreadgroup \(g.q4.maxTotalThreadsPerThreadgroup), "
                      + "threadExecutionWidth \(g.q4.threadExecutionWidth), ladder \(widthLadder(g.q4))")
        try boundsChecks(g, checks)
        try partialWriterCheck(g, checks)
        try tailCoverage(g, rows: 33, checks)
        try tailCoverage(g, rows: 1000, checks)
    } catch {
        checks.record("gpu: synthetic GPU checks completed", false, "error: \(error)")
    }
    struct Report: Encodable {
        let mode: String
        let passed: Int
        let failed: Int
        let checks: [Checks.Entry]
    }
    let report = Report(mode: "synthetic", passed: checks.entries.count - checks.failed,
                        failed: checks.failed, checks: checks.entries)
    print(encodeJSON(report))
    return checks.failed == 0 ? 0 : 1
}

func runComponent(ggufPath: String, outPath: String?) -> Int32 {
    struct Report: Encodable {
        let mode: String
        let ggufPath: String
        let hardware: ForgeHardwareProfile
        let pipelineMaxThreads: Int
        let controlWidth: Int
        let candidateWidths: [Int]
        let hotShapes: [Q4LaunchHotShape]
        let incumbentWorstRelToOracle: Double
        let results: [Q4LaunchComponentResult]
        let finalistWidths: [Int]
        let allAccepted: Bool
        let allBitIdentical: Bool
        let controlAccepted: Bool
        let note: String
    }
    do {
        let url = URL(fileURLWithPath: ggufPath)
        let runner = try Gemma4Runner(fileAt: url)
        let file = try GGUFReader.read(url)
        let pipeMax = try runner.q4MatvecPipelineMaxThreads()
        let hardware = runner.hardwareProfile()
        let candidates = try Q4LaunchDeriver.derive(hardware: hardware,
                                                    pipelineMaxThreads: pipeMax)
        let control = Q4LaunchDeriver.control(pipelineMaxThreads: pipeMax)
        var all = candidates
        if !all.contains(control) { all.append(control) }
        let shapes = try Q4LaunchHotShapes.derive(file: file,
                                                  layerCount: runner.config.layerCount)
        let (results, incumbentRel) = try Q4LaunchForge.component(
            runner: runner, file: file, shapes: shapes, candidates: all)
        let finalists = Q4LaunchForge.finalists(results, controlWidth: control.width).finalists
        let allAccepted = results.count == all.count && results.allSatisfy { $0.accepted }
        let allBits = results.allSatisfy { $0.bitIdenticalToIncumbent == true }
        let controlAccepted = results.contains {
            $0.schedule == control && $0.accepted && $0.bitIdenticalToIncumbent == true }
        let report = Report(
            mode: "component", ggufPath: ggufPath, hardware: hardware,
            pipelineMaxThreads: pipeMax, controlWidth: control.width,
            candidateWidths: all.map(\.width), hotShapes: shapes,
            incumbentWorstRelToOracle: incumbentRel, results: results,
            finalistWidths: finalists.map(\.width), allAccepted: allAccepted,
            allBitIdentical: allBits, controlAccepted: controlAccepted,
            note: "Full-row bit identity on the runner's cached pipeline; FP64 oracle bounded "
                + "to the first min(outDim, 64) rows. screenMicrosPerShape are per-dispatch GPU "
                + "medians for RANKING ONLY: not end-to-end performance, no occupancy claim.")
        let text = encodeJSON(report)
        print(text)
        if let outPath {
            try text.write(toFile: outPath, atomically: true, encoding: .utf8)
        }
        return allAccepted && allBits && controlAccepted ? 0 : 1
    } catch {
        print(encodeJSON(["mode": "component", "error": "\(error)"]))
        return 1
    }
}

func usage() -> Never {
    FileHandle.standardError.write(Data(
        "usage: q4_qualification [--component --gguf PATH [--out FILE]]\n".utf8))
    exit(2)
}

let arguments = CommandLine.arguments
if arguments.contains("--component") {
    guard let i = arguments.firstIndex(of: "--gguf"), i + 1 < arguments.count else { usage() }
    var outPath: String? = nil
    if let j = arguments.firstIndex(of: "--out") {
        guard j + 1 < arguments.count else { usage() }
        outPath = arguments[j + 1]
    }
    exit(runComponent(ggufPath: arguments[i + 1], outPath: outPath))
} else {
    guard arguments.count == 1 else { usage() }
    exit(runSynthetic())
}
