import Foundation
import Metal

// Phase 57 stage 3 part 1: default-off, prepared-only, scope-only GPU device routing for the incumbent installed
// E4R32 routed MoE region of the SSA executor.
// - The emitted RouterSelectionKernel runtime entry (the SAME shared body as the component entry) reads the F32
//   router projection (bufRouter) and the local resident expert-scale bytes inside the open serial segment and
//   writes one 2816-byte record per layer plus a 128-score log.
// - The unchanged E4R32 GU/DN kernels bind `constant const ulong* offs` (buffer 1) and the unchanged g4_axpy binds
//   `constant const float& c` (buffer 3) by setBuffer into that record: no layer CPU readback, no CPU selection, no
//   CPU fallback.
// - All 30 record flags are checked only after the token's existing final logits drain. Any flag, error or unwritten
//   record poisons the scope; the failed token is never accepted and nothing is re-executed.
// Diagnostic only: no artifact, installation, promotion, default or candidate-schema change. Original expert scales
// and coefficient bits stay in process memory and are never serialized, hashed or reported.

public struct RouterDeviceRoutingError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

// MARK: - Pure geometry

public enum RouterDeviceRoutingLayout {
    public static func recordBufferLength(layers: Int, layout: RouterSelectionLayout) -> Int {
        layers * layout.recordStride
    }

    public static func recordBase(layer: Int, layout: RouterSelectionLayout) -> Int {
        layer * layout.recordStride
    }

    public static func guOffsetsBinding(layer: Int, layout: RouterSelectionLayout) -> Int {
        recordBase(layer: layer, layout: layout) + layout.guOffsetsOffset
    }

    public static func dnOffsetsBinding(layer: Int, layout: RouterSelectionLayout) -> Int {
        recordBase(layer: layer, layout: layout) + layout.dnOffsetsOffset
    }

    public static func coefficientBinding(layer: Int, slot: Int, layout: RouterSelectionLayout) -> Int {
        recordBase(layer: layer, layout: layout) + layout.coefficientOffset + slot * layout.coefficientStride
    }

    public static func scoreLogBase(layer: Int, experts: Int) -> Int { layer * experts * 4 }

    public static func scoreLogLength(layers: Int, experts: Int) -> Int { layers * experts * 4 }

    /// Every record base, GU/DN offset array and coefficient slot must be binding aligned and inside the buffer.
    public static func bindingFailures(layers: Int, layout: RouterSelectionLayout) -> [String] {
        guard layers > 0, layers <= 128 else { return ["layer count out of range"] }
        guard (try? layout.validate()) != nil else { return ["selection layout invalid"] }
        var f = [String]()
        let total = recordBufferLength(layers: layers, layout: layout)
        for layer in 0..<layers {
            var offsets = [recordBase(layer: layer, layout: layout), guOffsetsBinding(layer: layer, layout: layout),
                           dnOffsetsBinding(layer: layer, layout: layout)]
            for slot in 0..<layout.topK { offsets.append(coefficientBinding(layer: layer, slot: slot, layout: layout)) }
            for o in offsets where o < 0 || o >= total || o % layout.bindingAlignment != 0 {
                f.append("layer " + String(layer) + " binding offset " + String(o) + " misaligned or outside the record buffer")
            }
            if recordBase(layer: layer, layout: layout) + layout.recordStride > total {
                f.append("layer " + String(layer) + " record exceeds the record buffer")
            }
        }
        return f
    }
}

/// Per-forward deltas of a device-routed forward relative to the control forward of the same prepared SSA.
public struct RouterDeviceRoutingExpectation: Codable, Equatable, Sendable {
    public let layers: Int
    public let extraKernelsPerForward: Int
    public let fewerCommandBuffersPerForward: Int
    public let gpuSelectionsPerForward: Int
    public let recordChainsPerForward: Int
    public let cpuSelectionsPerForwardControl: Int
    public let cpuSelectionsPerForwardDeviceRouted: Int

    /// One selector dispatch per layer is added; the per-layer router-score readback flush is removed.
    public static func derive(layers: Int) -> RouterDeviceRoutingExpectation {
        RouterDeviceRoutingExpectation(layers: layers, extraKernelsPerForward: layers,
                                       fewerCommandBuffersPerForward: layers, gpuSelectionsPerForward: layers,
                                       recordChainsPerForward: layers, cpuSelectionsPerForwardControl: layers,
                                       cpuSelectionsPerForwardDeviceRouted: 0)
    }

    public func deviceRoutedKernels(controlKernels: Int) -> Int { controlKernels + extraKernelsPerForward }

    public func deviceRoutedCommandBuffers(controlCommandBuffers: Int) -> Int? {
        let v = controlCommandBuffers - fewerCommandBuffersPerForward
        return v >= 1 ? v : nil
    }
}

// MARK: - Prepared binding

public struct RouterDeviceRoutingSSAOp: Equatable, Sendable {
    public let op: String
    public let layer: Int?
    public let topK: Int?

    public init(op: String, layer: Int?, topK: Int?) {
        self.op = op
        self.layer = layer
        self.topK = topK
    }
}

public enum RouterDeviceRoutingPlan {
    /// Exactly one router_topk (with the model topK) and one later expert_matvec_chain per layer.
    public static func validate(ops: [RouterDeviceRoutingSSAOp], layers: Int,
                                topK: Int) throws -> (router: [Int], chain: [Int]) {
        guard layers > 0, topK > 0 else { throw RouterDeviceRoutingError("plan: layers and topK must be positive") }
        var router = [Int?](repeating: nil, count: layers)
        var chain = [Int?](repeating: nil, count: layers)
        for (i, op) in ops.enumerated() where op.op == "router_topk" || op.op == "expert_matvec_chain" {
            guard let layer = op.layer, layer >= 0, layer < layers else {
                throw RouterDeviceRoutingError("plan: " + op.op + " at " + String(i) + " has no valid layer")
            }
            if op.op == "router_topk" {
                guard router[layer] == nil else {
                    throw RouterDeviceRoutingError("plan: duplicate router_topk for layer " + String(layer))
                }
                guard op.topK == topK else {
                    throw RouterDeviceRoutingError("plan: router_topk topK differs for layer " + String(layer))
                }
                router[layer] = i
            } else {
                guard chain[layer] == nil else {
                    throw RouterDeviceRoutingError("plan: duplicate expert chain for layer " + String(layer))
                }
                guard router[layer] != nil else {
                    throw RouterDeviceRoutingError("plan: expert chain precedes router_topk for layer " + String(layer))
                }
                chain[layer] = i
            }
        }
        let r = router.compactMap { $0 }
        let c = chain.compactMap { $0 }
        guard r.count == layers, c.count == layers else {
            throw RouterDeviceRoutingError("plan: every layer needs one router_topk and one expert chain")
        }
        return (r, c)
    }
}

struct RouterDeviceRoutingPreparedBinding {
    let layers: Int
    let topK: Int
    let routerIndices: [Int]
    let chainIndices: [Int]

    static func make(ssa: GemmaSSAIR, config: Gemma4Config) throws -> RouterDeviceRoutingPreparedBinding {
        guard config.expertCount == RouterSelectionIR.gemma4.expertCount,
              config.topKExperts == RouterSelectionIR.gemma4.topK else {
            throw RouterDeviceRoutingError("model expert count / topK differ from the selection IR")
        }
        let ops = ssa.instructions.map {
            RouterDeviceRoutingSSAOp(op: $0.op, layer: $0.layer,
                                     topK: $0.op == "router_topk" ? $0.attrs["topK"] : nil)
        }
        let v = try RouterDeviceRoutingPlan.validate(ops: ops, layers: config.layerCount, topK: config.topKExperts)
        return RouterDeviceRoutingPreparedBinding(layers: config.layerCount, topK: config.topKExperts,
                                                  routerIndices: v.router, chainIndices: v.chain)
    }
}

// MARK: - Counters, record helpers, verification

public struct RouterDeviceRoutingCounters: Codable, Equatable, Sendable {
    public var cpuSelections = 0
    public var gpuSelections = 0
    public var recordChains = 0
    public var acceptedTokens = 0
    public var failedTokens = 0
    public var flaggedTokens = 0

    public init() {}

    public func minus(_ o: RouterDeviceRoutingCounters) -> RouterDeviceRoutingCounters {
        var d = RouterDeviceRoutingCounters()
        d.cpuSelections = cpuSelections - o.cpuSelections
        d.gpuSelections = gpuSelections - o.gpuSelections
        d.recordChains = recordChains - o.recordChains
        d.acceptedTokens = acceptedTokens - o.acceptedTokens
        d.failedTokens = failedTokens - o.failedTokens
        d.flaggedTokens = flaggedTokens - o.flaggedTokens
        return d
    }
}

/// Host layout of the emitted MSL `struct rs_meta { ulong guB; ulong guS; ulong dnB; ulong dnS; }` (32 bytes).
struct RouterDeviceRoutingMeta {
    var guB: UInt64
    var guS: UInt64
    var dnB: UInt64
    var dnS: UInt64

    init(table: RouterOffsetTable, layer: Int) {
        guB = table.gu[layer].base
        guS = table.gu[layer].stride
        dnB = table.dn[layer].base
        dnS = table.dn[layer].stride
    }
}

/// CPU writer of the record layout (tests and diagnostics only; the runtime record is written by the GPU).
public enum RouterDeviceRoutingRecordEncoder {
    public static func words(ids: [UInt32], flags: UInt32, layer: UInt32, rawBits: [UInt32], weightBits: [UInt32],
                             gu: [UInt64], dn: [UInt64], layout: RouterSelectionLayout) -> [UInt32] {
        var w = [UInt32](repeating: 0, count: layout.recordWords)
        let k = layout.topK
        for s in 0..<min(k, ids.count) { w[layout.idsOffset / 4 + s] = ids[s] }
        w[layout.flagsOffset / 4] = flags
        w[layout.magicOffset / 4] = RouterSelectionKernel.recordMagic
        w[layout.layerOffset / 4] = layer
        for s in 0..<min(k, rawBits.count) { w[layout.rawCoefficientOffset / 4 + s] = rawBits[s] }
        for s in 0..<min(k, gu.count) {
            w[layout.guOffsetsOffset / 4 + 2 * s] = UInt32(truncatingIfNeeded: gu[s])
            w[layout.guOffsetsOffset / 4 + 2 * s + 1] = UInt32(truncatingIfNeeded: gu[s] >> 32)
        }
        for s in 0..<min(k, dn.count) {
            w[layout.dnOffsetsOffset / 4 + 2 * s] = UInt32(truncatingIfNeeded: dn[s])
            w[layout.dnOffsetsOffset / 4 + 2 * s + 1] = UInt32(truncatingIfNeeded: dn[s] >> 32)
        }
        for s in 0..<min(k, weightBits.count) {
            w[(layout.coefficientOffset + s * layout.coefficientStride) / 4] = weightBits[s]
        }
        return w
    }
}

/// Per-token CPU check result: layer indices only (never coefficient or scale bits).
public struct RouterDeviceRoutingTokenCheck: Codable, Equatable, Sendable {
    public let position: Int
    public let layers: Int
    public let flaggedLayers: [Int]
    public let recordErrorLayers: [Int]
    public let scoreErrorLayers: [Int]
    public let idMismatchLayers: [Int]
    public let quotientMismatchLayers: [Int]
    public let coefficientMismatchLayers: [Int]
    public let offsetMismatchLayers: [Int]

    public var exact: Bool {
        flaggedLayers.isEmpty && recordErrorLayers.isEmpty && scoreErrorLayers.isEmpty && idMismatchLayers.isEmpty
            && quotientMismatchLayers.isEmpty && coefficientMismatchLayers.isEmpty && offsetMismatchLayers.isEmpty
    }
}

public enum RouterDeviceRoutingVerifier {
    /// Recomputes the existing CPU FP64 decision from the logged GPU router scores and local scales and compares
    /// it with every layer's GPU record (ids, quotient bits, coefficient bits, real-table GU/DN offsets).
    public static func check(position: Int, records: [RouterSelectionGPURecord], scoreBits: [[UInt32]],
                             scales: [[Float]], table: RouterOffsetTable, topK: Int) -> RouterDeviceRoutingTokenCheck {
        let layers = table.gu.count
        var flagged = [Int]()
        var recordErr = [Int]()
        var scoreErr = [Int]()
        var idMis = [Int]()
        var qMis = [Int]()
        var cMis = [Int]()
        var offMis = [Int]()
        guard records.count == layers, scoreBits.count == layers, scales.count >= layers else {
            return RouterDeviceRoutingTokenCheck(position: position, layers: layers, flaggedLayers: [],
                                                 recordErrorLayers: Array(0..<max(layers, 0)), scoreErrorLayers: [],
                                                 idMismatchLayers: [], quotientMismatchLayers: [],
                                                 coefficientMismatchLayers: [], offsetMismatchLayers: [])
        }
        for layer in 0..<layers {
            let rec = records[layer]
            guard rec.magic == RouterSelectionKernel.recordMagic, rec.layer == UInt32(layer), rec.paddingClean,
                  rec.ids.count == topK, rec.rawBits.count == topK, rec.weightBits.count == topK,
                  rec.guOffsets.count == topK, rec.dnOffsets.count == topK else {
                recordErr.append(layer)
                continue
            }
            if rec.flags != 0 {
                flagged.append(layer)
                continue
            }
            let scores = scoreBits[layer].map { Float(bitPattern: $0) }
            guard scores.count == table.expertCount, scales[layer].count == table.expertCount,
                  let cpu = RouterSelectionKernel.cpuReference(scores: scores, scales: scales[layer], topK: topK) else {
                scoreErr.append(layer)
                continue
            }
            let ids = rec.ids.map { Int($0) }
            if ids != cpu.ids { idMis.append(layer) }
            if rec.rawBits != cpu.rawBits { qMis.append(layer) }
            if rec.weightBits != cpu.weightBits { cMis.append(layer) }
            if ids.allSatisfy({ $0 >= 0 && $0 < table.expertCount }) {
                let wantGu = ids.map { table.offset(layer: layer, dn: false, expert: $0) }
                let wantDn = ids.map { table.offset(layer: layer, dn: true, expert: $0) }
                if rec.guOffsets != wantGu || rec.dnOffsets != wantDn { offMis.append(layer) }
            } else {
                offMis.append(layer)
            }
        }
        return RouterDeviceRoutingTokenCheck(position: position, layers: layers, flaggedLayers: flagged,
                                             recordErrorLayers: recordErr, scoreErrorLayers: scoreErr,
                                             idMismatchLayers: idMis, quotientMismatchLayers: qMis,
                                             coefficientMismatchLayers: cMis, offsetMismatchLayers: offMis)
    }
}

/// In-memory per-token snapshot (records + logged scores). Not Codable: contains coefficient bits.
struct RouterDeviceRoutingTokenSnapshot {
    let position: Int
    let records: [RouterSelectionGPURecord]
    let scoreBits: [[UInt32]]
}

// MARK: - Scope state

final class RouterDeviceRoutingState {
    let layout: RouterSelectionLayout
    let layers: Int
    let experts: Int
    let records: MTLBuffer
    let scoreLog: MTLBuffer
    let pipeline: MTLComputePipelineState
    let sourceSha256: String
    var active = false
    var poisoned = false
    var poisonReason: String? = nil
    var tokenInFlight = false
    var selected: [Bool]
    var chained: [Bool]
    var table: RouterOffsetTable? = nil
    var scaleOffsets: [Int] = []
    var installDigest: String? = nil
    var captureSnapshots = false
    var snapshots: [RouterDeviceRoutingTokenSnapshot] = []
    var gpuSelections = 0
    var recordChains = 0
    var acceptedTokens = 0
    var failedTokens = 0
    var flaggedTokens = 0

    init(layout: RouterSelectionLayout, layers: Int, experts: Int, records: MTLBuffer, scoreLog: MTLBuffer,
         pipeline: MTLComputePipelineState, sourceSha256: String) {
        self.layout = layout
        self.layers = layers
        self.experts = experts
        self.records = records
        self.scoreLog = scoreLog
        self.pipeline = pipeline
        self.sourceSha256 = sourceSha256
        self.selected = [Bool](repeating: false, count: layers)
        self.chained = [Bool](repeating: false, count: layers)
    }

    func begin(table: RouterOffsetTable, scaleOffsets: [Int], installDigest: String, captureSnapshots: Bool) {
        self.table = table
        self.scaleOffsets = scaleOffsets
        self.installDigest = installDigest
        self.captureSnapshots = captureSnapshots
        poisoned = false
        poisonReason = nil
        tokenInFlight = false
        snapshots = []
        selected = [Bool](repeating: false, count: layers)
        chained = [Bool](repeating: false, count: layers)
        active = true
    }

    func poison(_ reason: String) {
        if !poisoned {
            poisoned = true
            poisonReason = reason
        }
    }
}

public enum RouterDeviceRouting {
    public static let kind = "router-device-routing-diagnostic-scope-v1"
    public static let maxSnapshots = 512

    struct SelectorPipeline {
        let pipeline: MTLComputePipelineState
        let sourceSha256: String
    }

    /// Compiles the emitted selector library with the component's exact options on the given device and returns
    /// the runtime entry pipeline after checking the 32-wide SIMD / 128-lane policy and threadgroup memory.
    static func compileSelector(device: MTLDevice) throws -> SelectorPipeline {
        let ir = RouterSelectionIR.gemma4
        let source = try RouterSelectionKernel.emitSource(ir)
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
        } else {
            options.fastMathEnabled = false
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: options) } catch {
            throw RouterDeviceRoutingError("selector library compile: " + String(describing: error))
        }
        guard let fn = library.makeFunction(name: RouterSelectionKernel.selectRuntimeFunction) else {
            throw RouterDeviceRoutingError("selector runtime entry missing")
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        guard pipeline.threadExecutionWidth == ir.simdWidth,
              pipeline.maxTotalThreadsPerThreadgroup >= ir.threadsPerGroup,
              pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
            throw RouterDeviceRoutingError("selector runtime pipeline violates the 32-wide / 128-lane policy")
        }
        return SelectorPipeline(pipeline: pipeline, sourceSha256: RouterSelectionKernel.sourceDigest(source))
    }

    /// Byte offsets of each layer's ffn_down_exps.scale tensor in the resident mapping, checked for type, shape,
    /// 4-byte alignment and extent, and compared bit for bit with the scales the runner read at init (CPU compare
    /// only: never hashed, uploaded, copied out or reported).
    static func scaleOffsets(_ r: Gemma4Runner) throws -> [Int] {
        guard r.expertCache == nil, let wb = r.residentWeightsBuffer else {
            throw RouterDeviceRoutingError("expert scales need the resident weight mapping")
        }
        let experts = r.config.expertCount
        var out = [Int]()
        for layer in 0..<r.config.layerCount {
            guard let t = r.tensorRef("blk." + String(layer) + ".ffn_down_exps.scale"), t.type == .float32,
                  t.inDim == experts, t.outDim == 1, t.experts == 1, t.offset >= 0, t.offset % 4 == 0,
                  t.end - t.offset == experts * 4, t.end <= wb.length,
                  layer < r.expertScales.count, r.expertScales[layer].count == experts else {
                throw RouterDeviceRoutingError("expert scale tensor geometry invalid for layer " + String(layer))
            }
            let p = wb.contents().advanced(by: t.offset).assumingMemoryBound(to: UInt32.self)
            for e in 0..<experts where p[e] != r.expertScales[layer][e].bitPattern {
                throw RouterDeviceRoutingError("resident expert scale bytes differ from the loaded scales")
            }
            out.append(t.offset)
        }
        return out
    }
}

// MARK: - Runner scope API

extension Gemma4Runner {
    public var routerDeviceRoutingActive: Bool { deviceRoutingState?.active == true }

    public var routerDeviceRoutingPoisonReason: String? { deviceRoutingState?.poisonReason }

    /// Refuses a direct (non-prepared) runner entry while the device-routing scope is active. Thrown before any
    /// validation, reset or GPU work and without poisoning; outside an active scope it is a no-op.
    func routerDeviceRoutingRefuseInScope(_ what: String) throws {
        guard let s = deviceRoutingState, s.active else { return }
        throw RouterDeviceRoutingError("device routing: " + what + " refused inside the scope")
    }

    public func routerDeviceRoutingCounters() -> RouterDeviceRoutingCounters {
        var c = RouterDeviceRoutingCounters()
        c.cpuSelections = routerCPUSelectionCount
        if let s = deviceRoutingState {
            c.gpuSelections = s.gpuSelections
            c.recordChains = s.recordChains
            c.acceptedTokens = s.acceptedTokens
            c.failedTokens = s.failedTokens
            c.flaggedTokens = s.flaggedTokens
        }
        return c
    }

    func routerDeviceRoutingEntryFailures(region: MoERegionIR) -> [String] {
        var f = [String]()
        if routerOverlapScopeActive { f.append("router overlap scope active") }
        if routeCaptureActive { f.append("route capture active") }
        if kernelStageProfiler != nil { f.append("kernel stage sampling scope active") }
        if !RouterOverlapModelGate.instrumentationOff(self) {
            f.append("probe/trace/breakdown/timeline/diagnostic/generated kernel/open work/expert cache present")
        }
        if !RouterOverlapModelGate.axisFlagsClean(self) { f.append("runner axis flags are not at their defaults") }
        if q4LaunchWidth != RouterOverlapGateProtocol.required.q4LaunchWidth { f.append("Q4 launch width is not 32") }
        f += RouterOverlapModelGate.installFailures(self, region: region)
        if residentWeightsBuffer == nil { f.append("no resident weight mapping") }
        if config.expertCount != RouterSelectionIR.gemma4.expertCount || config.topKExperts != RouterSelectionIR.gemma4.topK {
            f.append("expert count or topK differ from the selection IR")
        }
        return f
    }

    func routerDeviceRoutingStateEnsure() throws -> RouterDeviceRoutingState {
        if let s = deviceRoutingState { return s }
        let layout = RouterSelectionIR.gemma4.layout
        let layers = config.layerCount
        let experts = config.expertCount
        let failures = RouterDeviceRoutingLayout.bindingFailures(layers: layers, layout: layout)
        guard failures.isEmpty else { throw RouterDeviceRoutingError(failures.joined(separator: "; ")) }
        let compiled = try RouterDeviceRouting.compileSelector(device: q4RunnerDevice)
        guard let records = q4RunnerDevice.makeBuffer(
                  length: RouterDeviceRoutingLayout.recordBufferLength(layers: layers, layout: layout),
                  options: .storageModeShared),
              let log = q4RunnerDevice.makeBuffer(
                  length: RouterDeviceRoutingLayout.scoreLogLength(layers: layers, experts: experts),
                  options: .storageModeShared) else {
            throw RouterDeviceRoutingError("record / score-log allocation failed")
        }
        let s = RouterDeviceRoutingState(layout: layout, layers: layers, experts: experts, records: records,
                                         scoreLog: log, pipeline: compiled.pipeline,
                                         sourceSha256: compiled.sourceSha256)
        deviceRoutingState = s
        return s
    }

    func routerDeviceRoutingAliasFailures(_ s: RouterDeviceRoutingState) -> [String] {
        var others: [MTLBuffer?] = [bufX, bufH, bufT1, bufT2, bufT3, bufKV, bufGU, bufAct, bufExpertGU, bufExpertAct,
                                    bufExpertAcc, bufShared, bufRouter, bufEmbed, bufLogits, localFreqBuffer,
                                    globalFreqBuffer, rmsStatsInverseBuffer, moeScratch?.gu, moeScratch?.act,
                                    moeScratch?.dn, routerOverlapState?.routerIn, residentWeightsBuffer]
        for b in kCaches { others.append(b) }
        for b in vCaches { others.append(b) }
        var f = [String]()
        if s.records === s.scoreLog { f.append("record and score-log buffers alias") }
        for case let b? in others where b === s.records || b === s.scoreLog {
            f.append("device routing buffer aliases a runner weight/KV/activation/scratch/statistics buffer")
            break
        }
        return f
    }

    /// DIAGNOSTIC ONLY. Runs body with GPU device routing armed for device-routing preparations. Refuses nesting,
    /// the overlap scope, route capture, instrumentation, non-pinned installs, width != 32, expert-cache mode,
    /// an unproven offset table and any alias. On exit the open segment (if any) is discarded and the scope
    /// deactivated; a poisoned scope, a token still in flight, a changed install or dirty axis flags throw.
    public func withRouterDeviceRoutingScope<T>(region: MoERegionIR, captureSnapshots: Bool,
                                                _ body: () throws -> T) throws -> T {
        guard !routerDeviceRoutingActive else {
            throw RouterDeviceRoutingError("device routing scope refused (already active)")
        }
        let entry = routerDeviceRoutingEntryFailures(region: region)
        guard entry.isEmpty else {
            throw RouterDeviceRoutingError("device routing scope refused: " + entry.joined(separator: "; "))
        }
        guard let installed = moeGenerated else { throw RouterDeviceRoutingError("no installed routed lowering") }
        let table = try RouterRouteReplay.realOffsetTable(self)
        let proven = try table.validate()
        guard proven == 2 * config.layerCount * config.expertCount else {
            throw RouterDeviceRoutingError("real offset table proof count " + String(proven) + " is not complete")
        }
        let b = installed.binding
        for layer in 0..<config.layerCount {
            guard layer < b.guTensorStarts.count, layer < b.dnTensorStarts.count,
                  UInt64(b.guTensorStarts[layer]) == table.gu[layer].base,
                  UInt64(b.dnTensorStarts[layer]) == table.dn[layer].base,
                  UInt64(b.guStride) == table.gu[layer].stride,
                  UInt64(b.dnStride) == table.dn[layer].stride else {
                throw RouterDeviceRoutingError("installed binding offsets differ from the real table at layer " + String(layer))
            }
        }
        let scales = try RouterDeviceRouting.scaleOffsets(self)
        let state = try routerDeviceRoutingStateEnsure()
        let aliases = routerDeviceRoutingAliasFailures(state)
        guard aliases.isEmpty else { throw RouterDeviceRoutingError(aliases.joined(separator: "; ")) }
        state.begin(table: table, scaleOffsets: scales, installDigest: installed.artifactDigest,
                    captureSnapshots: captureSnapshots)
        defer { routerDeviceRoutingExit(state) }
        let result = try body()
        if state.poisoned || state.tokenInFlight {
            throw RouterDeviceRoutingError("device routing scope poisoned: "
                                           + (state.poisonReason ?? "token not accepted"))
        }
        guard RouterOverlapModelGate.axisFlagsClean(self), moeGenerated?.artifactDigest == state.installDigest,
              openEnc == nil, openCmd == nil else {
            throw RouterDeviceRoutingError("device routing scope did not restore axes, install or segment state")
        }
        return result
    }

    private func routerDeviceRoutingExit(_ s: RouterDeviceRoutingState) {
        if openEnc != nil || openCmd != nil {
            discardOpenSegment()
            s.poison("segment still open at scope exit")
        }
        if s.tokenInFlight {
            s.poison("token in flight at scope exit")
            s.tokenInFlight = false
        }
        s.active = false
    }

    /// In-memory snapshots captured since the last drain (qualification only; never serialized).
    func routerDeviceRoutingDrainSnapshots() -> [RouterDeviceRoutingTokenSnapshot] {
        guard let s = deviceRoutingState else { return [] }
        let out = s.snapshots
        s.snapshots.removeAll()
        return out
    }

    /// Called from routerOverlapPreflight before any reset or GPU work.
    func routerDeviceRoutingPreflight(prepared: GemmaPreparedExecution?, candidate: GemmaForgeCandidate) throws {
        let routed = prepared?.routerDeviceRouting
        guard let s = deviceRoutingState, s.active else {
            if routed != nil {
                throw GemmaForgeError.executionFailed(
                    "device routing: a device-routing preparation executes only inside its scope")
            }
            return
        }
        guard let binding = routed else {
            throw GemmaForgeError.executionFailed(
                "device routing: raw, non-prepared, control or overlap execution refused inside the scope")
        }
        guard prepared?.routerOverlap == nil else {
            throw GemmaForgeError.executionFailed("device routing: overlap preparation refused inside the scope")
        }
        guard candidate == RouterOverlapModelGate.modernCandidate else {
            throw GemmaForgeError.executionFailed("device routing: candidate is not the declared modern candidate")
        }
        guard !s.poisoned else {
            throw GemmaForgeError.executionFailed("device routing: scope poisoned: " + (s.poisonReason ?? "unknown"))
        }
        guard !s.tokenInFlight else {
            throw GemmaForgeError.executionFailed("device routing: previous token was never accepted")
        }
        guard binding.layers == s.layers, binding.topK == s.layout.topK else {
            throw GemmaForgeError.executionFailed("device routing: preparation geometry differs from the scope")
        }
        guard moeGenerated?.artifactDigest == s.installDigest, expertCache == nil,
              q4LaunchWidth == RouterOverlapGateProtocol.required.q4LaunchWidth,
              !routerOverlapScopeActive, !routeCaptureActive else {
            throw GemmaForgeError.executionFailed("device routing: install, width, residency or scope state changed")
        }
    }

    func routerDeviceRoutingBeginToken(binding: RouterDeviceRoutingPreparedBinding) throws {
        guard let s = deviceRoutingState, s.active else {
            throw RouterDeviceRoutingError("device routing token outside the scope")
        }
        guard !s.poisoned, !s.tokenInFlight else {
            s.poison("token begin while poisoned or in flight")
            throw RouterDeviceRoutingError("device routing token refused: scope poisoned or token in flight")
        }
        guard openEnc == nil, openCmd == nil, binding.layers == s.layers else {
            s.poison("token begin with an open segment or a foreign binding")
            throw RouterDeviceRoutingError("device routing token refused: open segment or foreign binding")
        }
        // CPU writes while the GPU is idle: every record must be rewritten by its selector this token.
        let words = s.records.contents().bindMemory(to: UInt32.self, capacity: s.records.length / 4)
        for layer in 0..<s.layers {
            let base = RouterDeviceRoutingLayout.recordBase(layer: layer, layout: s.layout)
            words[(base + s.layout.magicOffset) / 4] = 0
            words[(base + s.layout.flagsOffset) / 4] = 0xFFFF_FFFF
        }
        s.selected = [Bool](repeating: false, count: s.layers)
        s.chained = [Bool](repeating: false, count: s.layers)
        s.tokenInFlight = true
    }

    /// Encodes the shared-body runtime selector for one layer into the open segment (or opens one).
    func encodeRouterSelectRuntime(pipeline: MTLComputePipelineState, router: MTLBuffer, scaleOffset: Int,
                                   meta: RouterDeviceRoutingMeta, records: MTLBuffer, recordOffset: Int, layer: Int,
                                   scoreLog: MTLBuffer, scoreLogOffset: Int) throws {
        let ir = RouterSelectionIR.gemma4
        let layout = ir.layout
        let scoreBytes = ir.expertCount * 4
        guard let wb = residentWeightsBuffer else { throw RouterDeviceRoutingError("no resident weight mapping") }
        guard layer >= 0, layer < config.layerCount, scaleOffset >= 0, scaleOffset % 4 == 0,
              scaleOffset + scoreBytes <= wb.length, recordOffset >= 0,
              recordOffset % layout.bindingAlignment == 0, recordOffset + layout.recordStride <= records.length,
              scoreLogOffset >= 0, scoreLogOffset % 4 == 0, scoreLogOffset + scoreBytes <= scoreLog.length,
              router.length >= scoreBytes, router !== records, router !== scoreLog, records !== scoreLog,
              records !== wb, scoreLog !== wb else {
            throw RouterDeviceRoutingError("selector binding geometry invalid")
        }
        var m = meta
        var l = UInt32(layer)
        try dispatchPipeline(pipeline, name: RouterSelectionKernel.selectRuntimeFunction, groups: 1,
                             threads: ir.threadsPerGroup) { enc in
            enc.setBuffer(router, offset: 0, index: 0)
            enc.setBuffer(wb, offset: scaleOffset, index: 1)
            enc.setBytes(&m, length: MemoryLayout<RouterDeviceRoutingMeta>.stride, index: 2)
            enc.setBuffer(records, offset: recordOffset, index: 3)
            enc.setBytes(&l, length: 4, index: 4)
            enc.setBuffer(scoreLog, offset: scoreLogOffset, index: 5)
        }
    }

    func routerDeviceRoutingSelect(binding: RouterDeviceRoutingPreparedBinding, layer: Int, topK: Int,
                                   router: MTLBuffer) throws {
        guard let s = deviceRoutingState, s.active, s.tokenInFlight, !s.poisoned, let table = s.table else {
            throw RouterDeviceRoutingError("device routing select outside an accepted token")
        }
        do {
            guard layer >= 0, layer < s.layers, !s.selected[layer], topK == s.layout.topK,
                  binding.layers == s.layers, layer < s.scaleOffsets.count else {
                throw RouterDeviceRoutingError("selector refused for layer " + String(layer))
            }
            try encodeRouterSelectRuntime(
                pipeline: s.pipeline, router: router, scaleOffset: s.scaleOffsets[layer],
                meta: RouterDeviceRoutingMeta(table: table, layer: layer), records: s.records,
                recordOffset: RouterDeviceRoutingLayout.recordBase(layer: layer, layout: s.layout), layer: layer,
                scoreLog: s.scoreLog,
                scoreLogOffset: RouterDeviceRoutingLayout.scoreLogBase(layer: layer, experts: s.experts))
            s.selected[layer] = true
            s.gpuSelections += 1
        } catch {
            s.poison(String(describing: error))
            throw error
        }
    }

    func routerDeviceRoutingExpertChain(binding: RouterDeviceRoutingPreparedBinding, layer: Int, tensors: [String],
                                        x: MTLBuffer, xCount: Int, y: MTLBuffer, n: Int) throws {
        guard let s = deviceRoutingState, s.active, s.tokenInFlight, !s.poisoned else {
            throw RouterDeviceRoutingError("device routing expert chain outside an accepted token")
        }
        do {
            guard layer >= 0, layer < s.layers, s.selected[layer], !s.chained[layer] else {
                throw RouterDeviceRoutingError("expert chain without its selector for layer " + String(layer))
            }
            guard let installed = moeGenerated, expertCache == nil, installed.artifactDigest == s.installDigest else {
                throw RouterDeviceRoutingError("installed routed lowering changed or expert-cache mode")
            }
            let prefix = "blk." + String(layer) + "."
            guard tensors.count >= 2, tensors[0] == prefix + "ffn_gate_up_exps.weight",
                  tensors[1] == prefix + "ffn_down_exps.weight", n == config.hiddenSize,
                  xCount == config.hiddenSize, x !== y else {
                throw RouterDeviceRoutingError("expert chain geometry or tensors disagree with the installed region")
            }
            guard try gpuRouterResidentZero(y: y, n: n) else {
                throw RouterDeviceRoutingError("resident zero declined; no CPU fallback in scope")
            }
            try runMoeBatchedRecordRouted(x: x, y: y, layer: layer, records: s.records,
                                          recordBase: RouterDeviceRoutingLayout.recordBase(layer: layer, layout: s.layout),
                                          layout: s.layout, installed: installed)
            s.chained[layer] = true
            s.recordChains += 1
        } catch {
            s.poison(String(describing: error))
            throw error
        }
    }

    /// After the token's final logits drain: every layer selected and chained, every record rewritten (magic and
    /// layer) and every flag word zero. Otherwise the token is failed and the scope poisoned.
    func routerDeviceRoutingVerifyToken(binding: RouterDeviceRoutingPreparedBinding, position: Int) throws {
        guard let s = deviceRoutingState, s.active, s.tokenInFlight else {
            throw RouterDeviceRoutingError("device routing verify outside an in-flight token")
        }
        do {
            guard openEnc == nil, openCmd == nil else {
                throw RouterDeviceRoutingError("segment still open at token end")
            }
            guard binding.layers == s.layers, s.selected.allSatisfy({ $0 }), s.chained.allSatisfy({ $0 }) else {
                throw RouterDeviceRoutingError("not every layer was selected and chained")
            }
            let raw = UnsafeRawBufferPointer(start: s.records.contents(), count: s.records.length)
            var flagged = [Int]()
            var unwritten = [Int]()
            for layer in 0..<s.layers {
                let base = RouterDeviceRoutingLayout.recordBase(layer: layer, layout: s.layout)
                let flags = raw.load(fromByteOffset: base + s.layout.flagsOffset, as: UInt32.self)
                let magic = raw.load(fromByteOffset: base + s.layout.magicOffset, as: UInt32.self)
                let recLayer = raw.load(fromByteOffset: base + s.layout.layerOffset, as: UInt32.self)
                if magic != RouterSelectionKernel.recordMagic || recLayer != UInt32(layer) {
                    unwritten.append(layer)
                } else if flags != 0 {
                    flagged.append(layer)
                }
            }
            guard unwritten.isEmpty, flagged.isEmpty else {
                if !flagged.isEmpty { s.flaggedTokens += 1 }
                throw RouterDeviceRoutingError("token at position " + String(position) + ": flagged layers "
                    + flagged.map { String($0) }.joined(separator: ",") + "; unwritten layers "
                    + unwritten.map { String($0) }.joined(separator: ","))
            }
            if s.captureSnapshots {
                guard s.snapshots.count < RouterDeviceRouting.maxSnapshots else {
                    throw RouterDeviceRoutingError("snapshot bound exceeded")
                }
                var recs = [RouterSelectionGPURecord]()
                for layer in 0..<s.layers {
                    let base = RouterDeviceRoutingLayout.recordBase(layer: layer, layout: s.layout)
                    recs.append(RouterSelectionKernel.decodeRecord(
                        UnsafeRawBufferPointer(rebasing: raw[base..<(base + s.layout.recordStride)]), layout: s.layout))
                }
                let logPtr = s.scoreLog.contents().assumingMemoryBound(to: UInt32.self)
                var scores = [[UInt32]]()
                for layer in 0..<s.layers {
                    scores.append(Array(UnsafeBufferPointer(start: logPtr + layer * s.experts, count: s.experts)))
                }
                s.snapshots.append(RouterDeviceRoutingTokenSnapshot(position: position, records: recs, scoreBits: scores))
            }
            s.tokenInFlight = false
            s.acceptedTokens += 1
        } catch {
            s.failedTokens += 1
            s.poison(String(describing: error))
            throw error
        }
    }
}

// MARK: - Small GPU component (entry identity, constant consumer, real E4 layer)

public struct RouterDeviceRoutingEntryCase: Sendable {
    public let name: String
    public let scores: [Float]
    public let scales: [Float]
    public let declared: RouterSelectionFlags
}

public struct RouterDeviceRoutingEntryResult: Codable, Equatable, Sendable {
    public let completion: String
    public let terminal: Bool
    public let componentCompletion: String
    public let cases: Int
    public let ordinaryCases: Int
    public let declaredRefusalCases: Int
    public let entryIdenticalRecords: Int
    public let entryMismatchCases: [String]
    public let ordinaryCPUExact: Int
    public let ordinaryUnexpectedFlagCases: [String]
    public let refusalMissingDeclaredCases: [String]
    public let hostMirrorFlaggedOrdinaryCases: [String]
    public let constantConsumerExactSlots: Int
    public let constantConsumerMismatchCases: [String]
    public let scoreLogExactCases: Int
    public let paddingCleanCases: Int
    public let guardsIntact: Bool
    public let gpuMicros: Double?
    public let error: String?

    public var passed: Bool {
        terminal && completion == "COMPLETED" && componentCompletion == "COMPLETED" && error == nil
            && cases > 0 && entryIdenticalRecords == cases && entryMismatchCases.isEmpty
            && ordinaryCPUExact == ordinaryCases && ordinaryUnexpectedFlagCases.isEmpty
            && refusalMissingDeclaredCases.isEmpty && constantConsumerExactSlots == cases * RouterSelectionIR.gemma4.topK
            && constantConsumerMismatchCases.isEmpty && scoreLogExactCases == cases && paddingCleanCases == cases
            && guardsIntact
    }

    static func incomplete(completion: String, terminal: Bool, componentCompletion: String, cases: Int,
                           ordinary: Int, error: String?) -> RouterDeviceRoutingEntryResult {
        RouterDeviceRoutingEntryResult(
            completion: completion, terminal: terminal, componentCompletion: componentCompletion, cases: cases,
            ordinaryCases: ordinary, declaredRefusalCases: cases - ordinary, entryIdenticalRecords: 0,
            entryMismatchCases: [], ordinaryCPUExact: 0, ordinaryUnexpectedFlagCases: [], refusalMissingDeclaredCases: [],
            hostMirrorFlaggedOrdinaryCases: [], constantConsumerExactSlots: 0, constantConsumerMismatchCases: [],
            scoreLogExactCases: 0, paddingCleanCases: 0, guardsIntact: false, gpuMicros: nil, error: error)
    }
}

/// Real E4 layer result. Booleans only: no scores, scales, coefficients or weights.
public struct RouterDeviceRoutingLayerResult: Codable, Equatable, Sendable {
    public let layer: Int
    public let hostDomainSupported: Bool
    public let recordFlags: [String]
    public let recordWellFormed: Bool
    public let idsMatchCPU: Bool
    public let quotientBitsMatchCPU: Bool
    public let coefficientBitsMatchCPU: Bool
    public let offsetsMatchRealTable: Bool
    public let scoreLogExact: Bool
    public let guardsIntact: Bool
    public let controlFinite: Bool
    public let guScratchBitIdentical: Bool
    public let actScratchBitIdentical: Bool
    public let dnScratchBitIdentical: Bool
    public let accumulatorBitIdentical: Bool

    public var passed: Bool {
        hostDomainSupported && recordFlags.isEmpty && recordWellFormed && idsMatchCPU && quotientBitsMatchCPU
            && coefficientBitsMatchCPU && offsetsMatchRealTable && scoreLogExact && guardsIntact && controlFinite
            && guScratchBitIdentical && actScratchBitIdentical && dnScratchBitIdentical && accumulatorBitIdentical
    }
}

public struct RouterDeviceRoutingRealLayersResult: Codable, Equatable, Sendable {
    public let device: String
    public let realOffsetsProven: Int
    public let layers: [RouterDeviceRoutingLayerResult]
    public let restoredPristine: Bool

    public var passed: Bool {
        realOffsetsProven == 7680 && layers.count == RouterDeviceRoutingComponent.realLayers.count
            && layers.allSatisfy({ $0.passed }) && restoredPristine
    }
}

public enum RouterDeviceRoutingComponentStatus: String, Codable, Sendable {
    case componentPassed
    case entryFailed
    case realLayersFailed
    case gpuError
    case notEnded
    case aborted
}

public struct RouterDeviceRoutingComponentReport: Codable {
    public private(set) var kind = RouterDeviceRoutingComponent.kind
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var realE4GateComplete = false
    public private(set) var scalesSerialized = false
    public private(set) var notes = RouterDeviceRoutingComponent.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var selectorSourceSha256: String? = nil
    public var selectorIRSha256: String? = nil
    public var entry: RouterDeviceRoutingEntryResult? = nil
    public var real: RouterDeviceRoutingRealLayersResult? = nil
    public var restoredPristine = false
    public var error: String? = nil
    public var status: RouterDeviceRoutingComponentStatus = .aborted

    public init() {}
}

extension RouterDeviceRoutingComponentReport {
    func withDiagnosticContract() -> RouterDeviceRoutingComponentReport {
        var copy = self
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.realE4GateComplete = false
        copy.scalesSerialized = false
        return copy
    }
}

public struct RouterDeviceRoutingComponentOptions: Equatable, Sendable {
    public let gguf: String
    public let region: String
    public let report: String
}

struct RouterDeviceRoutingSplitMix {
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

public enum RouterDeviceRoutingComponent {
    public static let kind = "router-device-routing-component-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 1 << 20
    public static let realLayers = [0, 5, 29]
    public static let constantConsumerFunction = "leviathan_record_constant_consume"
    public static let usage = "usage: router-device-routing-component --gguf M.gguf --region R.json --report NEW_REPORT.json"

    public static let notes: [String] = [
        "Diagnostic component only: no artifact, installation, promotion or default change; gainQualified, installable, "
            + "productionPerformanceClaim and realE4GateComplete are false (the full model gate is a separate part).",
        "Part A (standalone device): the component entry and the runtime entry of the SAME emitted shared selection "
            + "body must produce decoded-equal, padding-clean records (byte identity) on 94 ordinary and 8 declared "
            + "refusal cases; a tiny consumer reads the record through constant const ulong* and constant const float& "
            + "setBuffer bindings in the same serial encoder; canaries and score logs are checked; completion bounded.",
        "Part B (runner's own resident mapping, layers 0, 5, 29): control = CPU decision + host setBytes offsets and "
            + "coefficients through the unchanged E4R32 kernels; candidate = runtime selector record + setBuffer "
            + "bindings; GU/ACT/DN scratch and accumulator must be bit-identical. Synthetic scores and activations only; "
            + "real offsets and local expert scales stay in process memory and are never reported.",
        "Part B uses the runner's existing synchronous segment flush; the external supervisor owns the deadline. "
            + "The host mirror is a reported domain note, never an oracle.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func parseOptions(_ argv: [String]) throws -> RouterDeviceRoutingComponentOptions {
        let keys = ["--gguf", "--region", "--report"]
        var values = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard keys.contains(key), i + 1 < argv.count, values[key] == nil else {
                throw RouterDeviceRoutingError(usage)
            }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw RouterDeviceRoutingError(usage) }
            values[key] = value
            i += 2
        }
        guard let gguf = values["--gguf"], let region = values["--region"], let report = values["--report"] else {
            throw RouterDeviceRoutingError(usage)
        }
        return RouterDeviceRoutingComponentOptions(gguf: gguf, region: region, report: report)
    }

    public static func encodeReport(_ r: RouterDeviceRoutingComponentReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    public static func status(_ r: RouterDeviceRoutingComponentReport) -> RouterDeviceRoutingComponentStatus {
        guard let e = r.entry else { return r.error == nil ? .aborted : .gpuError }
        if !e.terminal { return .notEnded }
        if !e.passed {
            return (e.error != nil || e.completion != "COMPLETED" || e.componentCompletion != "COMPLETED")
                ? .gpuError : .entryFailed
        }
        guard let real = r.real else { return .realLayersFailed }
        if r.error != nil || !real.passed || !r.restoredPristine { return .realLayersFailed }
        return .componentPassed
    }

    /// Fixed deterministic case set: 94 ordinary cases and 8 declared refusals (no adaptive exclusion).
    public static func entryCases() -> [RouterDeviceRoutingEntryCase] {
        var rng = RouterDeviceRoutingSplitMix(state: 0x5253_4C31_0057_0001)
        var out = [RouterDeviceRoutingEntryCase]()
        func grid(_ span: UInt64, _ offset: Int64, _ den: Float) -> [Float] {
            var s = [Float]()
            for _ in 0..<128 { s.append(Float(Int64(rng.below(span)) + offset) / den) }
            return s
        }
        func scales() -> [Float] {
            var s = [Float]()
            for _ in 0..<128 {
                let m = Float(Int64(rng.below(3840)) + 64) / 256
                s.append(rng.below(2) == 0 ? m : -m)
            }
            return s
        }
        func add(_ family: String, _ scores: [Float], _ sc: [Float], _ declared: RouterSelectionFlags) {
            out.append(RouterDeviceRoutingEntryCase(name: family + "-" + String(out.count), scores: scores, scales: sc,
                                                    declared: declared))
        }
        for v in [Float(0), 1.5, -3.25, 7] { add("allEqual", [Float](repeating: v, count: 128), scales(), []) }
        for _ in 0..<48 { add("uniform", grid(32768, -16384, 2048), scales(), []) }
        for _ in 0..<16 { add("ties", grid(33, -16, 4), scales(), []) }
        for _ in 0..<16 { add("wideRange", grid(286_720, -245_760, 4096), scales(), []) }
        for _ in 0..<8 {
            var s = grid(12544, -20224, 256)
            s[Int(rng.below(128))] = 0
            add("dominant", s, scales(), [])
        }
        var e1 = [Float](repeating: -80, count: 128)
        for i in 0..<7 { e1[i] = 0 }
        add("rangeEdge", e1, scales(), [])
        var e2 = [Float](repeating: -76.5, count: 128)
        e2[64] = 3.5
        add("rangeEdge", e2, scales(), [])
        var s = grid(32768, -16384, 2048)
        s[3] = .nan
        add("nanScore", s, scales(), .nonFiniteInput)
        s = grid(32768, -16384, 2048)
        s[50] = .infinity
        add("infScore", s, scales(), .nonFiniteInput)
        s = grid(32768, -16384, 2048)
        s[9] = Float(sign: .plus, exponent: -110, significand: 1)
        add("tinyScore", s, scales(), .tinyInput)
        s = grid(32768, -16384, 2048)
        s[5] = .leastNonzeroMagnitude
        add("subnormalScore", s, scales(), .tinyInput)
        s = grid(32768, -16384, 2048)
        s[0] = 9
        s[100] = -81
        add("rangeBelow", s, scales(), .rangeUnsupported)
        s = [Float](repeating: -1, count: 128)
        s[1] = Float(sign: .plus, exponent: -30, significand: 1)
        s[0] = s[1].nextDown
        add("nearTie", s, scales(), .nearTieCollapse)
        s = grid(2048, -2048, 256)
        s[0] = 2
        var sc = scales()
        sc[0] = .nan
        add("scaleNaN", s, sc, .scaleDomain)
        s = grid(2048, -2048, 256)
        s[1] = 2
        sc = scales()
        sc[1] = .leastNonzeroMagnitude
        add("scaleSubnormal", s, sc, .scaleDomain)
        return out
    }

    static func constantConsumerSource() -> String {
        let lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "kernel void leviathan_record_constant_consume(",
            "    constant const ulong* guOffs [[buffer(0)]],",
            "    constant const ulong* dnOffs [[buffer(1)]],",
            "    constant const float& c [[buffer(2)]],",
            "    constant uint& slot [[buffer(3)]],",
            "    device uint* out [[buffer(4)]],",
            "    uint tid [[thread_position_in_grid]])",
            "{",
            "    if (tid != 0u) { return; }",
            "    ulong g = guOffs[slot];",
            "    ulong d = dnOffs[slot];",
            "    out[0] = uint(g);",
            "    out[1] = uint(g >> 32);",
            "    out[2] = uint(d);",
            "    out[3] = uint(d >> 32);",
            "    out[4] = as_type<uint>(c);",
            "}",
        ]
        let newline = String(Character(Unicode.Scalar(UInt8(10))))
        return lines.joined(separator: newline) + newline
    }

    static func compileConstantConsumer(device: MTLDevice) throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(source: constantConsumerSource(), options: nil)
        guard let fn = library.makeFunction(name: constantConsumerFunction) else {
            throw RouterDeviceRoutingError("constant consumer function missing")
        }
        return try device.makeComputePipelineState(function: fn)
    }

    /// Part A: component entry vs runtime entry record identity plus the constant-address-space consumer, all after
    /// GO, on a standalone device, bounded completion (no waitUntilCompleted).
    public static func runEntryIdentity(timeoutSeconds: Double) throws -> RouterDeviceRoutingEntryResult {
        let ir = RouterSelectionIR.gemma4
        let layout = ir.layout
        let k = ir.topK
        let experts = ir.expertCount
        let cases = entryCases()
        let count = cases.count
        let ordinary = cases.filter { $0.declared.isEmpty }.count
        guard timeoutSeconds > 0, timeoutSeconds <= 60 else { throw RouterDeviceRoutingError("timeout out of range") }
        let table = RouterOffsetTable.synthetic(layers: count, expertCount: experts, expertBytes: 16)
        let component = try RouterSelectionGPUComponent(ir: ir)
        let comp = try component.runSelector(scores: cases.map { $0.scores }, caseLayers: (0..<count).map { UInt32($0) },
                                             scales: cases.map { $0.scales }, offsets: table,
                                             timeoutSeconds: timeoutSeconds)
        guard comp.terminal else {
            return .incomplete(completion: "NOT_RUN", terminal: false, componentCompletion: comp.completion,
                               cases: count, ordinary: ordinary, error: comp.errorDescription)
        }
        guard comp.completion == "COMPLETED", comp.records.count == count else {
            return .incomplete(completion: "NOT_RUN", terminal: true, componentCompletion: comp.completion,
                               cases: count, ordinary: ordinary,
                               error: comp.errorDescription ?? "component record count mismatch")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw RouterDeviceRoutingError("no Metal device or queue")
        }
        let selector = try RouterDeviceRouting.compileSelector(device: device)
        let consumer = try compileConstantConsumer(device: device)
        let g = RouterSelectionKernel.guardBytes
        let scoreWords = cases.flatMap { $0.scores.map { $0.bitPattern } }
        let scaleWords = cases.flatMap { $0.scales.map { $0.bitPattern } }
        func shared(_ words: [UInt32]) -> MTLBuffer? {
            words.withUnsafeBytes { raw in
                raw.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: raw.count, options: .storageModeShared) }
            }
        }
        let recordBytes = count * layout.recordStride
        guard let scoreBuf = shared(scoreWords), let scaleBuf = shared(scaleWords),
              let recBuf = device.makeBuffer(length: 2 * g + recordBytes, options: .storageModeShared),
              let logBuf = device.makeBuffer(length: count * experts * 4, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: count * k * 20, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw RouterDeviceRoutingError("entry identity allocation failed")
        }
        memset(recBuf.contents(), Int32(RouterSelectionKernel.canaryByte), recBuf.length)
        memset(logBuf.contents(), 0, logBuf.length)
        memset(outBuf.contents(), 0xFF, outBuf.length)
        enc.setComputePipelineState(selector.pipeline)
        for i in 0..<count {
            var meta = RouterDeviceRoutingMeta(table: table, layer: i)
            var layer = UInt32(i)
            enc.setBuffer(scoreBuf, offset: i * experts * 4, index: 0)
            enc.setBuffer(scaleBuf, offset: i * experts * 4, index: 1)
            enc.setBytes(&meta, length: MemoryLayout<RouterDeviceRoutingMeta>.stride, index: 2)
            enc.setBuffer(recBuf, offset: g + i * layout.recordStride, index: 3)
            enc.setBytes(&layer, length: 4, index: 4)
            enc.setBuffer(logBuf, offset: i * experts * 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: ir.threadsPerGroup, height: 1, depth: 1))
        }
        enc.setComputePipelineState(consumer)
        for i in 0..<count {
            let base = g + i * layout.recordStride
            for slot in 0..<k {
                var s = UInt32(slot)
                enc.setBuffer(recBuf, offset: base + layout.guOffsetsOffset, index: 0)
                enc.setBuffer(recBuf, offset: base + layout.dnOffsetsOffset, index: 1)
                enc.setBuffer(recBuf, offset: base + layout.coefficientOffset + slot * layout.coefficientStride, index: 2)
                enc.setBytes(&s, length: 4, index: 3)
                enc.setBuffer(outBuf, offset: (i * k + slot) * 20, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        let done = DispatchSemaphore(value: 0)
        cb.addCompletedHandler { _ in done.signal() }
        cb.commit()
        if done.wait(timeout: .now() + timeoutSeconds) == .timedOut {
            return .incomplete(completion: "NOT_ENDED", terminal: false, componentCompletion: comp.completion,
                               cases: count, ordinary: ordinary,
                               error: "runtime entry completion not observed within the bound")
        }
        guard cb.status == .completed, cb.error == nil else {
            let why = cb.error.map { String(describing: $0) } ?? ("status " + String(cb.status.rawValue))
            return .incomplete(completion: "ERROR", terminal: true, componentCompletion: comp.completion,
                               cases: count, ordinary: ordinary, error: why)
        }
        let recAll = UnsafeRawBufferPointer(start: recBuf.contents(), count: recBuf.length)
        var guards = true
        for i in 0..<g where recAll[i] != RouterSelectionKernel.canaryByte
            || recAll[g + recordBytes + i] != RouterSelectionKernel.canaryByte {
            guards = false
        }
        let outPtr = outBuf.contents().assumingMemoryBound(to: UInt32.self)
        let logPtr = logBuf.contents().assumingMemoryBound(to: UInt32.self)
        var identical = 0
        var mismatch = [String]()
        var cpuExact = 0
        var unexpected = [String]()
        var missing = [String]()
        var mirrorFlagged = [String]()
        var exactSlots = 0
        var consumerMismatch = [String]()
        var logExact = 0
        var clean = 0
        for (i, c) in cases.enumerated() {
            let start = g + i * layout.recordStride
            let rec = RouterSelectionKernel.decodeRecord(
                UnsafeRawBufferPointer(rebasing: recAll[start..<(start + layout.recordStride)]), layout: layout)
            if rec.paddingClean { clean += 1 }
            if rec == comp.records[i], rec.paddingClean, comp.records[i].paddingClean {
                identical += 1
            } else {
                mismatch.append(c.name)
            }
            var logOK = true
            for e in 0..<experts where logPtr[i * experts + e] != c.scores[e].bitPattern { logOK = false }
            if logOK { logExact += 1 }
            var consumerOK = true
            for slot in 0..<k {
                let o = (i * k + slot) * 5
                let gu = UInt64(outPtr[o]) | (UInt64(outPtr[o + 1]) << 32)
                let dn = UInt64(outPtr[o + 2]) | (UInt64(outPtr[o + 3]) << 32)
                if gu == rec.guOffsets[slot], dn == rec.dnOffsets[slot], outPtr[o + 4] == rec.weightBits[slot] {
                    exactSlots += 1
                } else {
                    consumerOK = false
                }
            }
            if !consumerOK { consumerMismatch.append(c.name) }
            if c.declared.isEmpty {
                if let m = try? RouterSelectionKernel.mirror(ir, scores: c.scores, scales: c.scales), !m.flags.isEmpty {
                    mirrorFlagged.append(c.name)
                }
                if rec.flags != 0 {
                    unexpected.append(c.name)
                } else if let cpu = RouterSelectionKernel.cpuReference(scores: c.scores, scales: c.scales, topK: k),
                          rec.ids.map({ Int($0) }) == cpu.ids, rec.rawBits == cpu.rawBits,
                          rec.weightBits == cpu.weightBits {
                    cpuExact += 1
                }
            } else if !rec.refusal.isSuperset(of: c.declared) || rec.ids.contains(where: { $0 != 0 })
                        || rec.weightBits.contains(where: { $0 != 0 }) {
                missing.append(c.name)
            }
        }
        let micros: Double? = cb.gpuEndTime > cb.gpuStartTime ? (cb.gpuEndTime - cb.gpuStartTime) * 1e6 : nil
        return RouterDeviceRoutingEntryResult(
            completion: "COMPLETED", terminal: true, componentCompletion: comp.completion, cases: count,
            ordinaryCases: ordinary, declaredRefusalCases: count - ordinary, entryIdenticalRecords: identical,
            entryMismatchCases: mismatch, ordinaryCPUExact: cpuExact, ordinaryUnexpectedFlagCases: unexpected,
            refusalMissingDeclaredCases: missing, hostMirrorFlaggedOrdinaryCases: mirrorFlagged,
            constantConsumerExactSlots: exactSlots, constantConsumerMismatchCases: consumerMismatch,
            scoreLogExactCases: logExact, paddingCleanCases: clean, guardsIntact: guards, gpuMicros: micros,
            error: nil)
    }

    /// Part B: real E4 layers on the runner's own resident mapping. Control = CPU decision + host setBytes offsets
    /// and coefficients (runMoeBatchedRouted); candidate = runtime selector record + setBuffer bindings
    /// (runMoeBatchedRecordRouted). Deterministic synthetic scores and activations; real offsets and local scales.
    public static func runRealLayers(runner r: Gemma4Runner, region: MoERegionIR) throws -> RouterDeviceRoutingRealLayersResult {
        let ir = RouterSelectionIR.gemma4
        let layout = ir.layout
        let k = ir.topK
        let experts = r.config.expertCount
        let hidden = r.config.hiddenSize
        guard RouterOverlapModelGate.pristine(r), RouterOverlapModelGate.instrumentationOff(r),
              RouterOverlapModelGate.axisFlagsClean(r), !r.routerDeviceRoutingActive, !r.routeCaptureActive,
              experts == ir.expertCount, r.config.topKExperts == k,
              realLayers.allSatisfy({ $0 >= 0 && $0 < r.config.layerCount }) else {
            throw RouterDeviceRoutingError("real E4 component requires a pristine runner with instrumentation off")
        }
        defer {
            r.uninstallMoeGeneratedLowering()
            r.reset()
        }
        try r.installMoeBaselineForVerification(
            region: region, schedule: MoESchedule(batch: 4, rowsPerGroup: 32),
            expectedSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256)
        guard let installed = r.moeGenerated else { throw RouterDeviceRoutingError("E4R32 install missing") }
        let table = try RouterRouteReplay.realOffsetTable(r)
        let proven = try table.validate()
        let scaleOffsets = try RouterDeviceRouting.scaleOffsets(r)
        let selector = try RouterDeviceRouting.compileSelector(device: r.q4RunnerDevice)
        let dev = r.q4RunnerDevice
        let g = RouterSelectionKernel.guardBytes
        let logBytes = experts * 4
        guard let xBuf = dev.makeBuffer(length: hidden * 4, options: .storageModeShared),
              let yCtrl = dev.makeBuffer(length: hidden * 4, options: .storageModeShared),
              let yCand = dev.makeBuffer(length: hidden * 4, options: .storageModeShared),
              let scoreBuf = dev.makeBuffer(length: logBytes, options: .storageModeShared),
              let records = dev.makeBuffer(length: 2 * g + layout.recordStride, options: .storageModeShared),
              let scoreLog = dev.makeBuffer(length: 2 * g + logBytes, options: .storageModeShared) else {
            throw RouterDeviceRoutingError("real E4 component allocation failed")
        }
        func words(_ b: MTLBuffer, _ count: Int) -> [UInt32] {
            Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt32.self), count: count))
        }
        func canariesIntact(_ b: MTLBuffer, body: Int) -> Bool {
            let all = UnsafeRawBufferPointer(start: b.contents(), count: b.length)
            for i in 0..<g where all[i] != RouterSelectionKernel.canaryByte
                || all[g + body + i] != RouterSelectionKernel.canaryByte {
                return false
            }
            return true
        }
        var results = [RouterDeviceRoutingLayerResult]()
        for layer in realLayers {
            var scores = [Float]()
            for i in 0..<experts { scores.append(Float(((i * 73 + layer * 29) % 128) - 64) / 16) }
            var x = [Float]()
            for i in 0..<hidden { x.append(Float(((i * 37 + layer * 11) % 97) - 48) / 64) }
            let scales = r.expertScales[layer]
            let domain = (try? RouterSelectionKernel.mirror(ir, scores: scores, scales: scales))?.flags.isEmpty ?? false
            guard let cpu = RouterSelectionKernel.cpuReference(scores: scores, scales: scales, topK: k) else {
                throw RouterDeviceRoutingError("CPU reference failed on the synthetic scores")
            }
            x.withUnsafeBytes { raw in _ = memcpy(xBuf.contents(), raw.baseAddress!, raw.count) }
            scores.withUnsafeBytes { raw in _ = memcpy(scoreBuf.contents(), raw.baseAddress!, raw.count) }
            memset(yCtrl.contents(), 0, yCtrl.length)
            try r.beginSegment()
            try r.runMoeBatchedRouted(x: xBuf, y: yCtrl, layer: layer, topIdx: cpu.ids, weights: cpu.weights,
                                      installed: installed)
            try r.flushOpenSegment()
            guard let scratch = r.moeScratch else { throw RouterDeviceRoutingError("MoE scratch missing") }
            let cGu = words(scratch.gu, scratch.gu.length / 4)
            let cAct = words(scratch.act, scratch.act.length / 4)
            let cDn = words(scratch.dn, scratch.dn.length / 4)
            let cY = words(yCtrl, hidden)
            memset(records.contents(), Int32(RouterSelectionKernel.canaryByte), records.length)
            memset(scoreLog.contents(), Int32(RouterSelectionKernel.canaryByte), scoreLog.length)
            memset(yCand.contents(), 0, yCand.length)
            try r.beginSegment()
            try r.encodeRouterSelectRuntime(pipeline: selector.pipeline, router: scoreBuf,
                                            scaleOffset: scaleOffsets[layer],
                                            meta: RouterDeviceRoutingMeta(table: table, layer: layer),
                                            records: records, recordOffset: g, layer: layer,
                                            scoreLog: scoreLog, scoreLogOffset: g)
            try r.runMoeBatchedRecordRouted(x: xBuf, y: yCand, layer: layer, records: records, recordBase: g,
                                            layout: layout, installed: installed)
            try r.flushOpenSegment()
            let dGu = words(scratch.gu, scratch.gu.length / 4)
            let dAct = words(scratch.act, scratch.act.length / 4)
            let dDn = words(scratch.dn, scratch.dn.length / 4)
            let dY = words(yCand, hidden)
            let recAll = UnsafeRawBufferPointer(start: records.contents(), count: records.length)
            let rec = RouterSelectionKernel.decodeRecord(
                UnsafeRawBufferPointer(rebasing: recAll[g..<(g + layout.recordStride)]), layout: layout)
            let lp = scoreLog.contents().advanced(by: g).assumingMemoryBound(to: UInt32.self)
            var logOK = true
            for e in 0..<experts where lp[e] != scores[e].bitPattern { logOK = false }
            let ids = rec.ids.map { Int($0) }
            var offsetsOK = ids.count == k && ids.allSatisfy({ $0 >= 0 && $0 < experts })
            if offsetsOK {
                offsetsOK = rec.guOffsets == ids.map { table.offset(layer: layer, dn: false, expert: $0) }
                    && rec.dnOffsets == ids.map { table.offset(layer: layer, dn: true, expert: $0) }
            }
            let finite = cY.allSatisfy { Float(bitPattern: $0).isFinite }
            results.append(RouterDeviceRoutingLayerResult(
                layer: layer, hostDomainSupported: domain, recordFlags: rec.refusal.labels,
                recordWellFormed: rec.magic == RouterSelectionKernel.recordMagic && rec.layer == UInt32(layer)
                    && rec.paddingClean,
                idsMatchCPU: ids == cpu.ids, quotientBitsMatchCPU: rec.rawBits == cpu.rawBits,
                coefficientBitsMatchCPU: rec.weightBits == cpu.weightBits, offsetsMatchRealTable: offsetsOK,
                scoreLogExact: logOK,
                guardsIntact: canariesIntact(records, body: layout.recordStride)
                    && canariesIntact(scoreLog, body: logBytes),
                controlFinite: finite, guScratchBitIdentical: cGu == dGu, actScratchBitIdentical: cAct == dAct,
                dnScratchBitIdentical: cDn == dDn, accumulatorBitIdentical: cY == dY))
        }
        r.uninstallMoeGeneratedLowering()
        r.reset()
        let restored = RouterOverlapModelGate.pristine(r) && RouterOverlapModelGate.instrumentationOff(r)
            && RouterOverlapModelGate.axisFlagsClean(r) && !r.routerDeviceRoutingActive
        return RouterDeviceRoutingRealLayersResult(device: dev.name, realOffsetsProven: proven, layers: results,
                                                   restoredPristine: restored)
    }
}
