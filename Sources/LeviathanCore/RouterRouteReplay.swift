import Foundation
import Metal

// Phase 56 stage 2: default-off, scoped, ephemeral capture of the EXISTING incumbent CPU router_topk
// decision in the GemmaForge SSA executor (control prepared execution only), plus a real-fixture GPU
// replay of every captured frame through the stage-1 RouterSelectionKernel component.
// Diagnostic only: no artifact, installation, promotion or default change; gainQualified, installable,
// productionPerformanceClaim and realE4GateComplete are always false.
// Expert scales stay in this process: the replay reads them from the loaded runner and hands them to the
// GPU in memory. They are never serialized, hashed or reported, and per-frame coefficient bits (which with
// the quotient bits would expose scale values) are never encoded.

public struct RouterRouteCaptureError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

// MARK: - Captured frame

/// One incumbent router_topk decision, captured after the unchanged Float subtract, FP64 softmax, top-K,
/// Float quotient and Float scale product. Not Codable: frames live in process memory only.
public struct RouterRouteFrame: Equatable, Sendable {
    public let layer: Int
    public let position: Int
    public let scoreBits: [UInt32]
    public let ids: [Int]
    public let quotientBits: [UInt32]
    /// In-memory only. Never encoded, hashed or reported: with quotientBits it would expose expert scales.
    public let coefficientBits: [UInt32]
    public let guOffsets: [UInt64]
    public let dnOffsets: [UInt64]

    public init(layer: Int, position: Int, scoreBits: [UInt32], ids: [Int], quotientBits: [UInt32],
                coefficientBits: [UInt32], guOffsets: [UInt64], dnOffsets: [UInt64]) {
        self.layer = layer
        self.position = position
        self.scoreBits = scoreBits
        self.ids = ids
        self.quotientBits = quotientBits
        self.coefficientBits = coefficientBits
        self.guOffsets = guOffsets
        self.dnOffsets = dnOffsets
    }

    public var scores: [Float] { scoreBits.map { Float(bitPattern: $0) } }
}

// MARK: - Capture buffer

/// Bounded in-memory frame buffer. Malformed frames are rejected (counted, never stored) and throw.
public final class RouterRouteCapture {
    public static let maxFramesLimit = 1 << 16
    public let expertCount: Int
    public let topK: Int
    public let layerCount: Int
    public let maxFrames: Int
    public private(set) var frames: [RouterRouteFrame] = []
    public private(set) var rejected = 0

    public init(expertCount: Int, topK: Int, layerCount: Int, maxFrames: Int) throws {
        guard expertCount > 0, topK > 0, topK <= expertCount, layerCount > 0,
              maxFrames > 0, maxFrames <= Self.maxFramesLimit else {
            throw RouterRouteCaptureError("route capture shape invalid")
        }
        self.expertCount = expertCount
        self.topK = topK
        self.layerCount = layerCount
        self.maxFrames = maxFrames
        frames.reserveCapacity(maxFrames)
    }

    public static func frameFailures(_ f: RouterRouteFrame, expertCount: Int, topK: Int,
                                     layerCount: Int) -> [String] {
        var r = [String]()
        if f.layer < 0 || f.layer >= layerCount { r.append("layer out of range") }
        if f.position < 0 { r.append("negative position") }
        if f.scoreBits.count != expertCount {
            r.append("score count " + String(f.scoreBits.count))
        } else if !f.scoreBits.allSatisfy({ Float(bitPattern: $0).isFinite }) {
            r.append("non-finite score")
        }
        if f.ids.count != topK {
            r.append("id count " + String(f.ids.count))
        } else {
            if !f.ids.allSatisfy({ $0 >= 0 && $0 < expertCount }) { r.append("id out of range") }
            if Set(f.ids).count != f.ids.count { r.append("duplicate id") }
        }
        let quotientsOK = f.quotientBits.count == topK && f.quotientBits.allSatisfy { q in
            Float(bitPattern: q).isFinite && q & 0x8000_0000 == 0
        }
        if !quotientsOK { r.append("quotient bits malformed") }
        if f.coefficientBits.count != topK || !f.coefficientBits.allSatisfy({ Float(bitPattern: $0).isFinite }) {
            r.append("coefficient bits malformed")
        }
        if f.guOffsets.count != topK || f.dnOffsets.count != topK { r.append("GU/DN offset count") }
        return r
    }

    public func append(_ f: RouterRouteFrame) throws {
        let failures = Self.frameFailures(f, expertCount: expertCount, topK: topK, layerCount: layerCount)
        guard failures.isEmpty else {
            rejected += 1
            throw RouterRouteCaptureError("malformed route frame: " + failures.joined(separator: ", "))
        }
        guard frames.count < maxFrames else {
            rejected += 1
            throw RouterRouteCaptureError("route capture capacity " + String(maxFrames) + " exceeded")
        }
        frames.append(f)
    }

    public func reset() {
        frames.removeAll(keepingCapacity: true)
        rejected = 0
    }

    /// Forwards of one greedy run: every prompt token plus outputs - 1 continuations.
    public static func expectedForwards(promptCount: Int, outputs: Int) -> Int {
        promptCount + outputs - 1
    }

    /// Complete, ordered sequence: frame i is layer i % layers at position (forward) i / layers.
    public static func sequenceFailures(_ frames: [RouterRouteFrame], forwards: Int, layerCount: Int) -> [String] {
        guard layerCount > 0, forwards >= 0 else { return ["layer count and forwards must be positive"] }
        var f = [String]()
        let expected = forwards * layerCount
        if frames.count != expected {
            f.append("captured " + String(frames.count) + " route frames, expected " + String(expected))
        }
        for (i, frame) in frames.enumerated() {
            let wantLayer = i % layerCount
            let wantPosition = i / layerCount
            if frame.layer != wantLayer || frame.position != wantPosition {
                f.append("frame " + String(i) + " is layer " + String(frame.layer) + " position "
                         + String(frame.position) + ", expected layer " + String(wantLayer) + " position "
                         + String(wantPosition))
                break
            }
        }
        return f
    }
}

// MARK: - Scope slot

/// Default-inactive slot held by every runner. begin/end bracket one scope; nesting is refused.
public final class RouterRouteCaptureSlot {
    public private(set) var active: RouterRouteCapture? = nil

    public init() {}

    public var isActive: Bool { active != nil }

    public func begin(_ capture: RouterRouteCapture) throws {
        guard active == nil else { throw RouterRouteCaptureError("route capture scope refused (already active)") }
        active = capture
    }

    public func end() { active = nil }

    /// Runs body with the capture active and clears the slot on every return and throw.
    public func scoped<T>(_ capture: RouterRouteCapture, _ body: () throws -> T) throws -> T {
        try begin(capture)
        defer { end() }
        return try body()
    }
}

extension Gemma4Runner {
    public var routeCaptureActive: Bool { routeCaptureSlot.isActive }

    /// DIAGNOSTIC ONLY. Captures every SSA router_topk decision of control prepared executions run inside
    /// body. Refuses an active router overlap scope, breakdown/timeline instrumentation and probes.
    public func withRouteCapture<T>(_ capture: RouterRouteCapture, _ body: () throws -> T) throws -> T {
        guard capture.expertCount == config.expertCount, capture.topK == config.topKExperts,
              capture.layerCount == config.layerCount else {
            throw RouterRouteCaptureError("route capture shape differs from the runner config")
        }
        guard !routerOverlapScopeActive, !routerDeviceRoutingActive, !breakdownInstrumentation,
              !commandTimelineEnabled, layerProbe == nil, moeInputProbe == nil else {
            throw RouterRouteCaptureError("route capture refuses the overlap scope, instrumentation and probes")
        }
        return try routeCaptureSlot.scoped(capture, body)
    }

    /// Called only from the SSA router_topk case while a capture is active. Quotient bits use the
    /// incumbent expression; offsets come from the installed binding's checked offsets call.
    func routeCaptureRecord(_ capture: RouterRouteCapture, layer: Int, position: Int, scores: [Float],
                            probs: [Double], probSum: Double, topIdx: [Int], weights: [Float]) throws {
        var gu = [UInt64]()
        var dn = [UInt64]()
        if let installed = moeGenerated, expertCache == nil {
            let offs = try installed.binding.offsets(forLayer: layer, expertIds: topIdx)
            gu = offs.gu.map { UInt64($0) }
            dn = offs.dn.map { UInt64($0) }
        }
        try capture.append(RouterRouteFrame(
            layer: layer, position: position, scoreBits: scores.map { $0.bitPattern }, ids: topIdx,
            quotientBits: topIdx.map { Float(probs[$0] / probSum).bitPattern },
            coefficientBits: weights.map { $0.bitPattern }, guOffsets: gu, dnOffsets: dn))
    }
}

// MARK: - Selector-only GPU replay

extension RouterSelectionGPUComponent {
    /// Selector-only run (no payload consumer) of the same emitted MSL and pipeline as run(), against a
    /// validated offset table whose lengths may be the real resident mapping. Offsets are written, never
    /// dereferenced. Bounded completion handler, no waitUntilCompleted. Single use per instance.
    public func runSelector(scores: [[Float]], caseLayers: [UInt32], scales: [[Float]],
                            offsets: RouterOffsetTable, timeoutSeconds: Double) throws -> RouterSelectionGPUOutcome {
        guard !used else { throw RouterSelectionError("component instance is single-use") }
        used = true
        let n = ir.expertCount
        let layout = ir.layout
        let cases = scores.count
        guard cases > 0, cases <= RouterSelectionKernel.maxCases, caseLayers.count == cases,
              scores.allSatisfy({ $0.count == n }) else {
            throw RouterSelectionError("invalid case batch")
        }
        guard offsets.expertCount == n, scales.count == offsets.gu.count, scales.allSatisfy({ $0.count == n }) else {
            throw RouterSelectionError("scales or offset table do not match the IR")
        }
        _ = try offsets.validate()
        guard timeoutSeconds > 0, timeoutSeconds <= 60 else { throw RouterSelectionError("timeout out of range") }
        var meta: [UInt64] = []
        for l in 0..<offsets.gu.count {
            meta += [offsets.gu[l].base, offsets.gu[l].stride, offsets.dn[l].base, offsets.dn[l].stride]
        }
        var staged: [(any MTLBuffer, any MTLBuffer)] = []
        func stage<T>(_ values: [T]) throws -> any MTLBuffer {
            let shared = try values.withUnsafeBytes { raw -> any MTLBuffer in
                guard let base = raw.baseAddress,
                      let b = self.device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared) else {
                    throw RouterSelectionError("staging allocation failed")
                }
                return b
            }
            guard let priv = self.device.makeBuffer(length: shared.length, options: .storageModePrivate) else {
                throw RouterSelectionError("private allocation failed")
            }
            staged.append((shared, priv))
            return priv
        }
        let scoreBuf = try stage(scores.flatMap { $0.map { $0.bitPattern } })
        let scaleBuf = try stage(scales.flatMap { $0.map { $0.bitPattern } })
        let layerBuf = try stage(caseLayers)
        let metaBuf = try stage(meta)
        let g = RouterSelectionKernel.guardBytes
        let recordBytes = cases * layout.recordStride
        guard let recPriv = device.makeBuffer(length: 2 * g + recordBytes, options: .storageModePrivate),
              let recRead = device.makeBuffer(length: 2 * g + recordBytes, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let blitIn = cb.makeBlitCommandEncoder() else {
            throw RouterSelectionError("output allocation or command buffer failed")
        }
        for (shared, priv) in staged {
            blitIn.copy(from: shared, sourceOffset: 0, to: priv, destinationOffset: 0, size: shared.length)
        }
        blitIn.fill(buffer: recPriv, range: 0..<recPriv.length, value: RouterSelectionKernel.canaryByte)
        blitIn.endEncoding()
        guard let enc = cb.makeComputeCommandEncoder() else { throw RouterSelectionError("compute encoder failed") }
        var caseCount = UInt32(cases)
        var layerCount = UInt32(offsets.gu.count)
        enc.setComputePipelineState(selectPipeline)
        enc.setBuffer(scoreBuf, offset: 0, index: 0)
        enc.setBuffer(scaleBuf, offset: 0, index: 1)
        enc.setBuffer(layerBuf, offset: 0, index: 2)
        enc.setBuffer(metaBuf, offset: 0, index: 3)
        enc.setBuffer(recPriv, offset: g, index: 4)
        enc.setBytes(&caseCount, length: 4, index: 5)
        enc.setBytes(&layerCount, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: cases, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: ir.threadsPerGroup, height: 1, depth: 1))
        enc.endEncoding()
        guard let blitOut = cb.makeBlitCommandEncoder() else { throw RouterSelectionError("readback encoder failed") }
        blitOut.copy(from: recPriv, sourceOffset: 0, to: recRead, destinationOffset: 0, size: recPriv.length)
        blitOut.endEncoding()
        let done = DispatchSemaphore(value: 0)
        cb.addCompletedHandler { _ in done.signal() }
        cb.commit()
        if done.wait(timeout: .now() + timeoutSeconds) == .timedOut {
            return RouterSelectionGPUOutcome(completion: "NOT_ENDED", terminal: false,
                                             errorDescription: "completion not observed within bound", records: [],
                                             consumed: [], guardsIntact: false, gpuMicros: nil)
        }
        guard cb.status == .completed, cb.error == nil else {
            let why = cb.error.map { String(describing: $0) } ?? ("status " + String(cb.status.rawValue))
            return RouterSelectionGPUOutcome(completion: "ERROR", terminal: true, errorDescription: why, records: [],
                                             consumed: [], guardsIntact: false, gpuMicros: nil)
        }
        let recAll = UnsafeRawBufferPointer(start: recRead.contents(), count: recRead.length)
        var intact = true
        for i in 0..<g where recAll[i] != RouterSelectionKernel.canaryByte
            || recAll[g + recordBytes + i] != RouterSelectionKernel.canaryByte {
            intact = false
        }
        var records: [RouterSelectionGPURecord] = []
        for c in 0..<cases {
            let start = g + c * layout.recordStride
            records.append(RouterSelectionKernel.decodeRecord(
                UnsafeRawBufferPointer(rebasing: recAll[start..<(start + layout.recordStride)]), layout: layout))
        }
        let micros: Double? = cb.gpuEndTime > cb.gpuStartTime ? (cb.gpuEndTime - cb.gpuStartTime) * 1e6 : nil
        return RouterSelectionGPUOutcome(completion: "COMPLETED", terminal: true, errorDescription: nil,
                                         records: records, consumed: [], guardsIntact: intact, gpuMicros: micros)
    }
}

// MARK: - Pure replay helpers

public enum RouterRouteReplay {
    /// Fixed chunk size (<= RouterSelectionKernel.maxCases): 4170 frames give 4 x 1024 + 74.
    public static let chunkSize = 1024

    public static func chunks(total: Int, maxPerChunk: Int) -> [Range<Int>] {
        guard total > 0, maxPerChunk > 0 else { return [] }
        var out = [Range<Int>]()
        var start = 0
        while start < total {
            let end = min(total, start + maxPerChunk)
            out.append(start..<end)
            start = end
        }
        return out
    }

    public static func expectedFrames(fixtures: [RouterOverlapGateFixture], outputs: Int, layers: Int) -> Int {
        fixtures.reduce(0) {
            $0 + RouterRouteCapture.expectedForwards(promptCount: $1.prompt.count, outputs: outputs) * layers
        }
    }

    /// SHA-256 over layer, position, score bits, ids, quotient bits and GU/DN offsets (little endian).
    /// Coefficient bits are deliberately excluded.
    public static func frameDigest(_ frames: [RouterRouteFrame]) -> String {
        var bytes = [UInt8]()
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        func put64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        for f in frames {
            put32(UInt32(truncatingIfNeeded: f.layer))
            put32(UInt32(truncatingIfNeeded: f.position))
            for s in f.scoreBits { put32(s) }
            for i in f.ids { put32(UInt32(truncatingIfNeeded: i)) }
            for q in f.quotientBits { put32(q) }
            for o in f.guOffsets { put64(o) }
            for o in f.dnOffsets { put64(o) }
        }
        return RouterSelectionKernel.sha256Hex(Data(bytes))
    }

    /// Real per-layer GU (packed gate+up) and DN families of the resident mapping: base = tensor offset,
    /// stride = outDim * (inDim / 32) * 18, all 128 experts inside the tensor extent.
    static func realOffsetTable(_ r: Gemma4Runner) throws -> RouterOffsetTable {
        guard r.expertCache == nil, let wb = r.residentWeightsBuffer else {
            throw RouterRouteCaptureError("real offset table needs the resident weight mapping")
        }
        let experts = r.config.expertCount
        var gu = [RouterOffsetFamily]()
        var dn = [RouterOffsetFamily]()
        for layer in 0..<r.config.layerCount {
            let prefix = "blk." + String(layer) + "."
            guard let g = r.tensorRef(prefix + "ffn_gate_up_exps.weight"),
                  let d = r.tensorRef(prefix + "ffn_down_exps.weight") else {
                throw RouterRouteCaptureError("expert tensors missing for layer " + String(layer))
            }
            gu.append(try family(g, experts: experts))
            dn.append(try family(d, experts: experts))
        }
        return RouterOffsetTable(expertCount: experts, alignment: 1, gu: gu, dn: dn,
                                 guBufferLength: UInt64(wb.length), dnBufferLength: UInt64(wb.length))
    }

    static func family(_ t: G4TensorRef, experts: Int) throws -> RouterOffsetFamily {
        guard t.type == .q4_0, t.experts == experts, t.inDim > 0, t.inDim % 32 == 0, t.outDim > 0,
              t.offset >= 0 else {
            throw RouterRouteCaptureError("expert tensor geometry unsupported")
        }
        let (rowBytes, o1) = (t.inDim / 32).multipliedReportingOverflow(by: 18)
        let (stride, o2) = t.outDim.multipliedReportingOverflow(by: rowBytes)
        let (span, o3) = stride.multipliedReportingOverflow(by: experts)
        let (end, o4) = t.offset.addingReportingOverflow(span)
        guard !o1, !o2, !o3, !o4, end <= t.end else {
            throw RouterRouteCaptureError("expert family exceeds its tensor extent")
        }
        return RouterOffsetFamily(base: UInt64(t.offset), stride: UInt64(stride), expertBytes: UInt64(stride))
    }
}

// MARK: - Report

public struct RouterRouteCaptureFixtureRecord: Codable, Equatable, Sendable {
    public let fixture: String
    public let promptCount: Int
    public let expectedFrames: Int
    public let capturedFrames: Int
    public let rejectedFrames: Int
    public let sequenceFailures: [String]
    public let framesSha256: String
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let outputs: Int
    public let allFinite: Bool
    public let comparison: OwnProcessLogitComparison
    public let plainCounts: RouterOverlapGatePhaseCounts?
    public let captureCounts: RouterOverlapGatePhaseCounts?
    public let totalsMismatchesVsPlain: [String]
    public let outputIdentityComparedSteps: Int
    public let outputIdentityBitMismatches: Int
    public let preconditionFailures: [String]
    public let error: String?
    public let checksPassed: Bool
    public let failure: String?
}

public struct RouterRouteReplayFrameRecord: Codable, Equatable, Sendable {
    public let fixture: String
    public let position: Int
    public let layer: Int
    public let ids: [Int]
    public let quotientBits: [UInt32]
    public let selectorFlags: [String]
    public let consumerFlags: [String]
    public let verdict: String
    public let problems: [String]

    public init(fixture: String, position: Int, layer: Int, ids: [Int], quotientBits: [UInt32],
                selectorFlags: [String], consumerFlags: [String], verdict: String, problems: [String]) {
        self.fixture = fixture
        self.position = position
        self.layer = layer
        self.ids = ids
        self.quotientBits = quotientBits
        self.selectorFlags = selectorFlags
        self.consumerFlags = consumerFlags
        self.verdict = verdict
        self.problems = problems
    }
}

/// gpuMicros are command-buffer intervals (gpuEndTime - gpuStartTime): not busy, idle or recoverable time.
public struct RouterRouteReplayChunk: Codable, Equatable, Sendable {
    public let start: Int
    public let count: Int
    public let selectorCompletion: String
    public let selectorGuardsIntact: Bool
    public let selectorGpuMicros: Double?
    public let consumerCompletion: String
    public let consumerGuardsIntact: Bool
    public let consumerGpuMicros: Double?

    public init(start: Int, count: Int, selectorCompletion: String, selectorGuardsIntact: Bool,
                selectorGpuMicros: Double?, consumerCompletion: String, consumerGuardsIntact: Bool,
                consumerGpuMicros: Double?) {
        self.start = start
        self.count = count
        self.selectorCompletion = selectorCompletion
        self.selectorGuardsIntact = selectorGuardsIntact
        self.selectorGpuMicros = selectorGpuMicros
        self.consumerCompletion = consumerCompletion
        self.consumerGuardsIntact = consumerGuardsIntact
        self.consumerGpuMicros = consumerGpuMicros
    }
}

public enum RouterRouteReplayStatus: String, Codable, Sendable {
    case replayExact
    case captureFailed
    case replayMismatch
    case replayRefused
    case gpuError
    case notEnded
    case aborted
    case runnerNotRestored
}

public struct RouterRouteReplayReport: Codable {
    public private(set) var kind = RouterRouteReplayDiagnostics.kind
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var realE4GateComplete = false
    public private(set) var scalesSerialized = false
    public private(set) var notes = RouterRouteReplayDiagnostics.notes
    public private(set) var limitations = RouterRouteReplayDiagnostics.limitations
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var selectionIR = RouterSelectionIR.gemma4
    public var selectionIRSha256: String? = nil
    public var selectionSourceSha256: String? = nil
    public var componentSourceSha256: [String] = []
    public var hardware: RouterSelectionHardware? = nil
    public var identity: RouterOverlapGateIdentity? = nil
    public var identityFailures: [String] = []
    public var fixtures: [RouterRouteCaptureFixtureRecord] = []
    public var expectedTotalFrames = 0
    public var capturedTotalFrames = 0
    public var capturePassed = false
    public var realOffsetsProven: Int? = nil
    public var realOffsetAlignment: UInt64? = nil
    public var hostOffsetMismatches = 0
    public var cpuReferenceMismatches = 0
    public var chunkSize = RouterRouteReplay.chunkSize
    public var chunks: [RouterRouteReplayChunk] = []
    public var frames: [RouterRouteReplayFrameRecord] = []
    public var exactFrames = 0
    public var refusedFrames = 0
    public var mismatchedFrames = 0
    public var offsetMismatchFrames = 0
    public var recordErrorFrames = 0
    public var consumerErrorFrames = 0
    public var replayCompletion: String? = nil
    public var replayTerminal = true
    public var replayError: String? = nil
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: RouterRouteReplayStatus = .aborted

    public init() {}
}

extension RouterRouteReplayReport {
    /// Serialization copy with every always-false field re-asserted.
    func withDiagnosticContract() -> RouterRouteReplayReport {
        var copy = self
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.realE4GateComplete = false
        copy.scalesSerialized = false
        return copy
    }
}

struct RouterRouteCapturedFixture {
    let fixture: String
    let frames: [RouterRouteFrame]
}

// MARK: - Driver

public enum RouterRouteReplayDiagnostics {
    public static let kind = "router-route-capture-real-gpu-replay-diagnostic-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let correctnessContract = 1e-4
    public static let maxReportBytes = 16 << 20
    public static let gpuTimeoutSeconds = 10.0
    public static let usage = "usage: router-route-replay --gguf M.gguf --region R.json --graph G.json "
        + "--fixtures F.json --report NEW_REPORT.json"

    public static let notes: [String] = [
        "Diagnostic only: no artifact, installation, promotion or default change; gainQualified, installable, "
            + "productionPerformanceClaim and realE4GateComplete are always false.",
        "Capture: one optional check in the existing SSA router_topk case, active only inside "
            + "Gemma4Runner.withRouteCapture (refuses the overlap scope, instrumentation and probes; the executor "
            + "refuses raw, non-prepared and overlap-prepared executions while it is active). Math, kernels, "
            + "counters and defaults are unchanged and the scope clears on every exit.",
        "Per fixture inside Q4 width 32: fresh direct reference, then a plain OFF run and a captured OFF run of "
            + "the declared incumbent (v1 E4R32, modern candidate, router launch divisor 1, CPU FP64 routing), "
            + "both 32 frozen IDs fed by the previous actual argmax, exact derived counters with every fallback 0, "
            + "equal kernel and command-buffer totals, captured outputs within 1e-4 of the reference (vector and "
            + "per-element normalized, strict) and bit-identical to the plain run at every logit.",
        "Frames: one per layer per forward in layer order: (21+31) + (23+31) + (2+31) = 139 forwards x 30 = "
            + "4170, validated per fixture.",
        "Replay: every frame, fixed 1024-frame chunks, two GPU passes per chunk: selector-only against the real "
            + "GU/DN base/stride table proven against the resident mapping length, and the stage-1 selector + "
            + "barrier + synthetic payload consumer. Every real frame must be unflagged with ids, quotient bits "
            + "and coefficient bits equal to the captured incumbent and selector offsets equal to the captured "
            + "actual checked offsets; the mirror is never an oracle.",
        "Scales: an in-process copy of the loaded runner's expert scales is handed to the GPU replay only; it is "
            + "never serialized, hashed or reported. Coefficient bits are compared in memory and only counted.",
        "Reuses the phase-53 gate helpers and fixtures; the full gate (ON arm, refusal tests, performance) is not "
            + "rerun. The model SHA-256 is the cached record and is never rehashed.",
    ]

    public static let limitations: [String] = [
        "The synthetic consumer proves record plumbing only; the unchanged packed gate+up / down E4R32 kernels "
            + "are not yet fed GPU-written offsets or coefficients (stage 3).",
        "Exactness is established for the 4170 captured frames of the three pinned fixtures only; it is not a "
            + "general equivalence proof for other inputs.",
        "No timing is claimed: component gpuMicros are command-buffer intervals, not busy, idle or recoverable "
            + "time; no ABBA or performance result is produced.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func encodeReport(_ r: RouterRouteReplayReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    public static func finalStatus(_ r: RouterRouteReplayReport) -> RouterRouteReplayStatus {
        if !r.replayTerminal { return .notEnded }
        if !r.restoredPristine { return .runnerNotRestored }
        if r.abortReason != nil { return .aborted }
        if !r.capturePassed { return .captureFailed }
        guard r.replayCompletion == "COMPLETED" else { return .gpuError }
        if r.refusedFrames > 0 { return .replayRefused }
        let guards = !r.chunks.isEmpty && r.chunks.allSatisfy { $0.selectorGuardsIntact && $0.consumerGuardsIntact }
        let counts = r.mismatchedFrames == 0 && r.offsetMismatchFrames == 0 && r.recordErrorFrames == 0
            && r.consumerErrorFrames == 0 && r.hostOffsetMismatches == 0 && r.cpuReferenceMismatches == 0
        if !guards || !counts || r.expectedTotalFrames <= 0 || r.exactFrames != r.expectedTotalFrames
            || r.frames.count != r.expectedTotalFrames {
            return .replayMismatch
        }
        return .replayExact
    }

    /// Never throws; every failure is recorded. The caller prints END only when the runner is quiescent and
    /// report.replayTerminal holds.
    public static func run(region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                           runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                           emit: (String) -> Void) -> RouterRouteReplayReport {
        let proto = RouterOverlapGateProtocol.required
        var report = RouterRouteReplayReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.expectedTotalFrames = RouterRouteReplay.expectedFrames(
            fixtures: fixtures, outputs: proto.outputsPerRun, layers: runner.config.layerCount)
        let pristineStart = RouterOverlapModelGate.pristine(runner)
        let instrumentationStart = RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner) && !runner.routeCaptureActive
        var setup = [String]()
        if !pristineStart { setup.append("runner is not pristine at start") }
        if !instrumentationStart { setup.append("instrumentation, axis flags or route capture are not off at start") }
        if fixtures != RouterOverlapGateFixtures.pinned { setup.append("fixtures differ from the pinned fixtures") }
        if !moeIsLowercaseHex64(modelSha256) || !moeIsLowercaseHex64(executableDigest) {
            setup.append("model and executable digests must be lowercase 64-hex")
        }
        if runner.config.expertCount != RouterSelectionIR.gemma4.expertCount
            || runner.config.topKExperts != RouterSelectionIR.gemma4.topK {
            setup.append("runner expert count or topK differ from the selection IR")
        }
        do {
            report.selectionIRSha256 = try RouterSelectionKernel.irDigest(RouterSelectionIR.gemma4)
            report.selectionSourceSha256 = RouterSelectionKernel.sourceDigest(
                try RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4))
        } catch {
            setup.append("selection IR/source: " + String(describing: error))
        }
        guard setup.isEmpty else {
            report.abortReason = setup.joined(separator: "; ")
            report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            report.status = finalStatus(report)
            emit("aborted before any GPU work: " + (report.abortReason ?? ""))
            return report
        }
        var captured = [RouterRouteCapturedFixture]()
        do {
            try runner.withQ4DiagnosticReferenceWidth(proto.q4LaunchWidth) { () -> Void in
                try captureAll(&report, captured: &captured, region: region, graph: graph, fixtures: fixtures,
                               runner: runner, modelSha256: modelSha256, executableDigest: executableDigest,
                               pristineStart: pristineStart, instrumentationStart: instrumentationStart,
                               emit: emit)
            }
        } catch {
            report.abortReason = "capture threw: " + String(describing: error)
        }
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            && RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner) && !runner.routeCaptureActive
        report.capturedTotalFrames = captured.reduce(0) { $0 + $1.frames.count }
        report.capturePassed = report.abortReason == nil && report.identity != nil
            && report.identityFailures.isEmpty
            && report.fixtures.map({ $0.fixture }) == RouterOverlapGateFixtures.order
            && report.fixtures.allSatisfy({ $0.checksPassed })
            && captured.map({ $0.fixture }) == RouterOverlapGateFixtures.order
            && report.capturedTotalFrames == report.expectedTotalFrames
        emit("capture " + (report.capturePassed ? "passed" : "FAILED") + ": " + String(report.capturedTotalFrames)
             + " of " + String(report.expectedTotalFrames) + " frames; runner restored " + String(report.restoredPristine))
        if report.capturePassed && report.restoredPristine {
            do {
                let table = try RouterRouteReplay.realOffsetTable(runner)
                report.realOffsetsProven = try table.validate()
                report.realOffsetAlignment = table.alignment
                // Ephemeral in-process copy of the loaded expert scales: GPU replay input only.
                let scales = runner.expertScales
                replay(&report, captured: captured, table: table, scales: scales)
            } catch {
                report.abortReason = "replay setup threw: " + String(describing: error)
            }
        }
        report.status = finalStatus(report)
        emit("status " + report.status.rawValue + "; exact " + String(report.exactFrames) + " of "
             + String(report.expectedTotalFrames) + "; refused " + String(report.refusedFrames) + "; mismatched "
             + String(report.mismatchedFrames) + "; diagnostic only, gainQualified false, realE4GateComplete false")
        return report
    }

    static func captureAll(_ report: inout RouterRouteReplayReport, captured: inout [RouterRouteCapturedFixture],
                           region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                           runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                           pristineStart: Bool, instrumentationStart: Bool, emit: (String) -> Void) throws {
        let proto = RouterOverlapGateProtocol.required
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == proto.q4LaunchWidth else {
            throw RouterRouteCaptureError("Q4 reference width 32 is not in effect inside the width scope")
        }
        var prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
        let derived = RouterOverlapModelGate.derive(on: prepared.on, runner: runner)
        if derived.statisticsGroups > 0 {
            // Allocation only, as in the gate: keeps buffer totals stationary.
            _ = try runner.rmsStatisticsInverse(slots: derived.statisticsGroups)
        }
        var ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
        let identity = try RouterOverlapModelGate.makeIdentity(
            ctx, modelSha256: modelSha256, executableDigest: executableDigest,
            pristineAtStart: pristineStart, instrumentationOffAtStart: instrumentationStart)
        report.identity = identity
        report.identityFailures = RouterOverlapModelGate.identityFailures(identity)
        emit(RouterOverlapModelGate.event("identity", identity))
        guard report.identityFailures.isEmpty else {
            emit("identity FAILED: " + report.identityFailures.joined(separator: "; ") + "; nothing captured")
            return
        }
        for fx in fixtures {
            let expected = RouterRouteCapture.expectedForwards(promptCount: fx.prompt.count,
                                                               outputs: proto.outputsPerRun) * derived.layers
            runner.uninstallMoeGeneratedLowering()
            let ref = RouterOverlapModelGate.qualify(.reference, ctx, fx, reference: [], off: [], offTotals: nil,
                                                     keep: true)
            guard ref.record.passed, ref.kept.count == proto.outputsPerRun else {
                let record = failedFixture(fx, expected: expected,
                                           reason: "fresh direct reference failed: " + (ref.record.failure ?? "unknown"))
                report.fixtures.append(record)
                emit(RouterOverlapModelGate.event("captureFixture", record))
                continue
            }
            prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
            guard RouterOverlapModelGate.derive(on: prepared.on, runner: runner) == derived else {
                throw RouterRouteCaptureError("derived counts changed after re-preparation")
            }
            ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
            let result = captureFixture(ctx, fx, reference: ref.kept)
            report.fixtures.append(result.record)
            captured.append(RouterRouteCapturedFixture(fixture: fx.name, frames: result.frames))
            emit(RouterOverlapModelGate.event("captureFixture", result.record))
        }
    }

    static func failedFixture(_ fx: RouterOverlapGateFixture, expected: Int,
                              reason: String) -> RouterRouteCaptureFixtureRecord {
        RouterRouteCaptureFixtureRecord(
            fixture: fx.name, promptCount: fx.prompt.count, expectedFrames: expected, capturedFrames: 0,
            rejectedFrames: 0, sequenceFailures: [], framesSha256: RouterRouteReplay.frameDigest([]), ids: [],
            idsMatchFrozen: false, outputs: 0, allFinite: false, comparison: OwnProcessLogitComparison(),
            plainCounts: nil, captureCounts: nil, totalsMismatchesVsPlain: [], outputIdentityComparedSteps: 0,
            outputIdentityBitMismatches: 0, preconditionFailures: [], error: nil, checksPassed: false,
            failure: reason)
    }

    /// Plain OFF run, then the same OFF run with the capture scope; both fully validated.
    static func captureFixture(_ c: RouterOverlapGateContext, _ fx: RouterOverlapGateFixture,
                               reference: [[Float]]) -> (record: RouterRouteCaptureFixtureRecord,
                                                         frames: [RouterRouteFrame]) {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let layers = r.config.layerCount
        let forwards = RouterRouteCapture.expectedForwards(promptCount: fx.prompt.count, outputs: proto.outputsPerRun)
        let expected = forwards * layers
        let pre = RouterOverlapModelGate.preconditions(.off, c, prepared: c.off)
        var plain = RouterOverlapGateGreedy()
        var g = RouterOverlapGateGreedy()
        var plainKept = [[Float]]()
        var cmp = OwnProcessLogitComparison()
        var bits = RouterOverlapGateBitComparator()
        var frames = [RouterRouteFrame]()
        var rejected = 0
        var scopeError: String? = nil
        if pre.isEmpty {
            func prefill() throws -> RouterOverlapModelGate.Step {
                try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt, candidate: c.candidate,
                                            prepared: c.off)
            }
            func next(_ token: Int) throws -> RouterOverlapModelGate.Step {
                try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token, candidate: c.candidate,
                                                    prepared: c.off)
            }
            func keepPlain(_ step: Int, _ logits: [Float]) { plainKept.append(logits) }
            func observeCapture(_ step: Int, _ logits: [Float]) {
                cmp.add(step: step, logits: logits, reference: step < reference.count ? reference[step] : [])
                if step < plainKept.count {
                    bits.add(step: step, on: logits, off: plainKept[step])
                } else {
                    bits.missing += 1
                }
            }
            r.reset()
            plain = RouterOverlapModelGate.greedy(r, outputs: proto.outputsPerRun, prefill: prefill, next: next,
                                                  observe: keepPlain)
            r.reset()
            do {
                let capture = try RouterRouteCapture(expertCount: r.config.expertCount, topK: r.config.topKExperts,
                                                     layerCount: layers, maxFrames: expected)
                try r.withRouteCapture(capture) { () -> Void in
                    g = RouterOverlapModelGate.greedy(r, outputs: proto.outputsPerRun, prefill: prefill, next: next,
                                                      observe: observeCapture)
                }
                frames = capture.frames
                rejected = capture.rejected
            } catch {
                scopeError = String(describing: error)
            }
            r.reset()
        }
        let plainCounts = RouterOverlapModelGate.phaseCounts(plain, c.derived, .off, promptCount: fx.prompt.count)
        let captureCounts = RouterOverlapModelGate.phaseCounts(g, c.derived, .off, promptCount: fx.prompt.count)
        let relation = RouterOverlapModelGate.totalsRelation(candidate: g.totals, baseline: plain.totals,
                                                             extraBuffersPerForward: 0,
                                                             promptCount: fx.prompt.count)
        let sequence = RouterRouteCapture.sequenceFailures(frames, forwards: forwards, layerCount: layers)
        var f = pre
        if pre.isEmpty {
            if let e = plain.error { f.append("plain run: " + e) }
            if let e = g.error { f.append("capture run: " + e) }
            if let s = scopeError { f.append("capture scope: " + s) }
            if plain.ids != fx.frozenIds { f.append("plain run IDs differ from the 32 frozen IDs") }
            if g.ids != fx.frozenIds { f.append("capture run IDs differ from the 32 frozen IDs") }
            if !plain.allFinite || plain.outputs != proto.outputsPerRun || !g.allFinite
                || g.outputs != proto.outputsPerRun {
                f.append("not 32 full-vocabulary finite outputs in both runs")
            }
            if !plainCounts.exact { f.append("plain counters: " + plainCounts.mismatches.joined(separator: ", ")) }
            if !captureCounts.exact {
                f.append("capture counters: " + captureCounts.mismatches.joined(separator: ", "))
            }
            f += relation.map { "capture vs plain totals: " + $0 }
            f += cmp.failures(contract: correctnessContract, expectedSteps: proto.outputsPerRun)
            if bits.compared != proto.outputsPerRun || bits.missing != 0 || bits.mismatches != 0 {
                f.append("capture changed outputs: bit-compared " + String(bits.compared) + ", missing "
                         + String(bits.missing) + ", mismatching logits " + String(bits.mismatches))
            }
            if rejected != 0 { f.append(String(rejected) + " route frame(s) rejected as malformed") }
            f += sequence
            if !RouterOverlapModelGate.quiescent(r) { f.append("runner not quiescent after the runs") }
            if r.routeCaptureActive { f.append("route capture scope left active") }
        }
        let record = RouterRouteCaptureFixtureRecord(
            fixture: fx.name, promptCount: fx.prompt.count, expectedFrames: expected, capturedFrames: frames.count,
            rejectedFrames: rejected, sequenceFailures: sequence, framesSha256: RouterRouteReplay.frameDigest(frames),
            ids: g.ids, idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs, allFinite: g.allFinite,
            comparison: cmp, plainCounts: plainCounts, captureCounts: captureCounts,
            totalsMismatchesVsPlain: relation, outputIdentityComparedSteps: bits.compared,
            outputIdentityBitMismatches: bits.mismatches, preconditionFailures: pre, error: g.error ?? plain.error,
            checksPassed: f.isEmpty, failure: f.isEmpty ? nil : f.joined(separator: "; "))
        return (record, frames)
    }

    /// Replays every captured frame (fixed chunks, two GPU passes per chunk) and classifies each frame.
    static func replay(_ report: inout RouterRouteReplayReport, captured: [RouterRouteCapturedFixture],
                       table: RouterOffsetTable, scales: [[Float]]) {
        let ir = RouterSelectionIR.gemma4
        let k = ir.topK
        var fixtureOf = [String]()
        var all = [RouterRouteFrame]()
        for c in captured {
            for f in c.frames {
                fixtureOf.append(c.fixture)
                all.append(f)
            }
        }
        var problems = [[String]](repeating: [], count: all.count)
        for (i, f) in all.enumerated() {
            guard f.layer >= 0, f.layer < table.gu.count, f.layer < scales.count else {
                problems[i].append("RECORD:layer outside the offset table")
                continue
            }
            let wantGu = f.ids.map { table.offset(layer: f.layer, dn: false, expert: $0) }
            let wantDn = f.ids.map { table.offset(layer: f.layer, dn: true, expert: $0) }
            if f.guOffsets != wantGu || f.dnOffsets != wantDn {
                problems[i].append("HOST_OFFSET:captured actual offsets differ from the validated real table")
                report.hostOffsetMismatches += 1
            }
            let cpu = RouterSelectionKernel.cpuReference(scores: f.scores, scales: scales[f.layer], topK: k)
            let same = cpu.map { $0.ids == f.ids && $0.rawBits == f.quotientBits && $0.weightBits == f.coefficientBits }
                ?? false
            if !same {
                problems[i].append("CPU_REFERENCE:public CPU reference differs from the captured incumbent decision")
                report.cpuReferenceMismatches += 1
            }
        }
        let synthetic = RouterOffsetTable.synthetic(layers: table.gu.count, expertCount: ir.expertCount,
                                                    expertBytes: 16)
        let guPayload = synthetic.syntheticPayload(dn: false)
        let dnPayload = synthetic.syntheticPayload(dn: true)
        var sel = [RouterSelectionGPURecord?](repeating: nil, count: all.count)
        var con = [RouterSelectionGPURecord?](repeating: nil, count: all.count)
        var consumed = [[UInt32]?](repeating: nil, count: all.count)
        var completion = "COMPLETED"
        var terminal = true
        for range in RouterRouteReplay.chunks(total: all.count, maxPerChunk: RouterRouteReplay.chunkSize) {
            let scores = range.map { all[$0].scores }
            let layers = range.map { UInt32(all[$0].layer) }
            var s: RouterSelectionGPUOutcome? = nil
            var c: RouterSelectionGPUOutcome? = nil
            var thrown: String? = nil
            do {
                let selector = try RouterSelectionGPUComponent(ir: ir)
                if report.hardware == nil { report.hardware = selector.hardware }
                if !report.componentSourceSha256.contains(selector.sourceSHA256) {
                    report.componentSourceSha256.append(selector.sourceSHA256)
                }
                let so = try selector.runSelector(scores: scores, caseLayers: layers, scales: scales, offsets: table,
                                                  timeoutSeconds: gpuTimeoutSeconds)
                s = so
                if so.terminal && so.completion == "COMPLETED" {
                    let consumer = try RouterSelectionGPUComponent(ir: ir)
                    c = try consumer.run(scores: scores, caseLayers: layers, scales: scales, offsets: synthetic,
                                         guPayload: guPayload, dnPayload: dnPayload,
                                         timeoutSeconds: gpuTimeoutSeconds)
                }
            } catch {
                thrown = String(describing: error)
            }
            report.chunks.append(RouterRouteReplayChunk(
                start: range.lowerBound, count: range.count, selectorCompletion: s?.completion ?? "NOT_RUN",
                selectorGuardsIntact: s?.guardsIntact ?? false, selectorGpuMicros: s?.gpuMicros,
                consumerCompletion: c?.completion ?? "NOT_RUN", consumerGuardsIntact: c?.guardsIntact ?? false,
                consumerGpuMicros: c?.gpuMicros))
            if let thrown {
                // Every component throw precedes its commit: nothing from this chunk is in flight.
                report.replayError = thrown
                completion = "NOT_COMMITTED"
                break
            }
            if s?.terminal == false || c?.terminal == false {
                terminal = false
                completion = "NOT_ENDED"
                report.replayError = "GPU completion not observed within " + String(gpuTimeoutSeconds) + " s"
                break
            }
            guard let so = s, let co = c, so.completion == "COMPLETED", co.completion == "COMPLETED",
                  so.records.count == range.count, co.records.count == range.count,
                  co.consumed.count == range.count else {
                completion = "ERROR"
                report.replayError = s?.errorDescription ?? c?.errorDescription ?? "replay record count mismatch"
                break
            }
            for (j, idx) in range.enumerated() {
                sel[idx] = so.records[j]
                con[idx] = co.records[j]
                consumed[idx] = co.consumed[j]
            }
        }
        report.replayCompletion = completion
        report.replayTerminal = terminal
        let zeros = [UInt32](repeating: 0, count: k)
        for i in 0..<all.count {
            let f = all[i]
            var p = problems[i]
            let ids32 = f.ids.map { UInt32($0) }
            var selectorFlags = [String]()
            var consumerFlags = [String]()
            if let s = sel[i] {
                selectorFlags = s.refusal.labels
                if s.flags != 0 {
                    p.append("REFUSED:selector " + selectorFlags.joined(separator: ","))
                } else {
                    if s.ids != ids32 { p.append("MISMATCH:selector ids") }
                    if s.rawBits != f.quotientBits { p.append("MISMATCH:selector quotient bits") }
                    if s.weightBits != f.coefficientBits { p.append("MISMATCH:selector coefficient bits") }
                    if s.guOffsets != f.guOffsets || s.dnOffsets != f.dnOffsets {
                        p.append("OFFSET:selector GU/DN offsets differ from the captured actual offsets")
                    }
                }
                if s.magic != RouterSelectionKernel.recordMagic || s.layer != UInt32(f.layer) || !s.paddingClean {
                    p.append("RECORD:selector record magic, layer or padding")
                }
            } else {
                p.append("RECORD:not replayed by the selector pass")
            }
            if let c = con[i], let cs = consumed[i], cs.count == 4 * k {
                consumerFlags = c.refusal.labels
                let refused = c.flags != 0
                if refused {
                    p.append("REFUSED:consumer pass " + consumerFlags.joined(separator: ","))
                } else if c.ids != ids32 || c.rawBits != f.quotientBits || c.weightBits != f.coefficientBits {
                    p.append("MISMATCH:consumer pass selection")
                }
                if c.magic != RouterSelectionKernel.recordMagic || c.layer != UInt32(f.layer) || !c.paddingClean {
                    p.append("RECORD:consumer pass record magic, layer or padding")
                }
                let wantIds = refused ? [Int](repeating: 0, count: k) : f.ids
                let wantCoef = refused ? zeros : f.coefficientBits
                for slot in 0..<k {
                    let want: [UInt32] = [
                        RouterOffsetTable.syntheticTag(dn: false, layer: f.layer, expert: wantIds[slot]),
                        RouterOffsetTable.syntheticTag(dn: true, layer: f.layer, expert: wantIds[slot]),
                        wantCoef[slot], 0,
                    ]
                    if Array(cs[(4 * slot)..<(4 * slot + 4)]) != want {
                        p.append("CONSUMER:slot " + String(slot))
                        break
                    }
                }
            } else {
                p.append("RECORD:not replayed by the consumer pass")
            }
            let categories = Set(p.map { String($0.split(separator: ":").first ?? "") })
            if categories.contains("REFUSED") { report.refusedFrames += 1 }
            if categories.contains("MISMATCH") || categories.contains("CPU_REFERENCE") { report.mismatchedFrames += 1 }
            if categories.contains("OFFSET") || categories.contains("HOST_OFFSET") { report.offsetMismatchFrames += 1 }
            if categories.contains("RECORD") { report.recordErrorFrames += 1 }
            if categories.contains("CONSUMER") { report.consumerErrorFrames += 1 }
            if p.isEmpty { report.exactFrames += 1 }
            let verdict = p.isEmpty ? "EXACT" : (categories.contains("REFUSED") ? "REFUSED" : "FAIL")
            report.frames.append(RouterRouteReplayFrameRecord(
                fixture: fixtureOf[i], position: f.position, layer: f.layer, ids: f.ids,
                quotientBits: f.quotientBits, selectorFlags: selectorFlags, consumerFlags: consumerFlags,
                verdict: verdict, problems: p))
        }
    }
}
