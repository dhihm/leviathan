import Foundation
import Metal

// Phase 59: default-off GPU kernel stage attribution of the incumbent OFF path (control preparation, declared modern
// candidate with par64 and router launch divisor 1, installed v1 E4R32, Q4 width 32, CPU FP64 routing) using ACTUAL
// encoder-stage timestamp sampling at stage boundaries (dispatch-boundary sampling is unsupported on this device).
// While a sampling scope is active, every deferred dispatch gets its own serial compute encoder with start/end samples
// inside the SAME command buffer: kernels, order, bindings, command buffers, commits and waits are unchanged; only the
// encoder topology changes (one encoder per dispatch instead of one per command buffer). Sampled intervals are raw stage
// ticks of that changed topology: not GPU busy time, utilization, bandwidth or recoverable time, and not the cost of the
// unmodified baseline. Inactive, every runner hook is one optional check: nothing is allocated, sampled or reordered.
// Diagnostic only: no artifact, installation, promotion, default or candidate change; no weights, scales or coefficients
// enter the report.

public struct KernelStageError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public enum KernelStageStatus: String, Codable, Sendable {
    case profiledChecksPassed
    case profiledChecksFailed
    case unsupported
    case aborted
    case runnerNotRestored
}

/// Actual device query and sample-buffer creation result (no timing).
public struct KernelStageSupport: Codable, Equatable, Sendable {
    public let device: String
    public let supportsAtStageBoundary: Bool
    public let supportsAtDispatchBoundary: Bool
    public let timestampCounterSetFound: Bool
    public let sampleBufferCreated: Bool
    public let sampleCapacity: Int
    public let error: String?

    public init(device: String, supportsAtStageBoundary: Bool, supportsAtDispatchBoundary: Bool,
                timestampCounterSetFound: Bool, sampleBufferCreated: Bool, sampleCapacity: Int, error: String?) {
        self.device = device
        self.supportsAtStageBoundary = supportsAtStageBoundary
        self.supportsAtDispatchBoundary = supportsAtDispatchBoundary
        self.timestampCounterSetFound = timestampCounterSetFound
        self.sampleBufferCreated = sampleBufferCreated
        self.sampleCapacity = sampleCapacity
        self.error = error
    }

    public var supported: Bool {
        supportsAtStageBoundary && timestampCounterSetFound && sampleBufferCreated && error == nil
    }
}

/// Checked raw stage samples of one sampled continuation call. Intervals are reported only when every check passed;
/// nothing is ever fabricated as zero.
public struct KernelStageTokenAnalysis: Codable, Equatable, Sendable {
    public let encoders: Int
    public let samples: Int
    public let firstStartTicks: UInt64?
    public let lastEndTicks: UInt64?
    public let spanTicks: UInt64?
    public let intervalTicks: [UInt64]
    public let sumEncoderTicks: UInt64?
    public let overlappingAdjacentEncoders: Int
    public let failures: [String]

    public var disjoint: Bool { failures.isEmpty && overlappingAdjacentEncoders == 0 }
}

/// Pure aggregation input: the kernel labels of one sampled token and their checked intervals.
public struct KernelStageTokenInput: Sendable {
    public let labels: [String]
    public let intervalTicks: [UInt64]
    public let disjoint: Bool

    public init(labels: [String], intervalTicks: [UInt64], disjoint: Bool) {
        self.labels = labels
        self.intervalTicks = intervalTicks
        self.disjoint = disjoint
    }
}

/// Raw per-token tick sums of one kernel or family across the sampled continuations of a fixture. The share is a
/// ratio of disjoint sampled stage intervals (nil if any adjacent encoders overlapped), never busy time.
public struct KernelStageLabelStat: Codable, Equatable, Sendable {
    public let label: String
    public let family: String?
    public let encodersPerToken: Int?
    public let tokens: Int
    public let ticksPerToken: [UInt64]
    public let medianTicksPerToken: Double?
    public let minTicksPerToken: UInt64?
    public let maxTicksPerToken: UInt64?
    public let medianShareOfSampledEncoderTicks: Double?
}

public struct KernelStageToken: Codable, Equatable, Sendable {
    public let continuation: Int
    public let encoders: Int
    public let kernelsDelta: Int
    public let commandBuffersDelta: Int
    public let encodersPerCommandBuffer: [Int]
    /// gpuEndTime - gpuStartTime per sampled command buffer: interval coverage, not busy time.
    public let commandBufferIntervalMicros: [Double]
    /// Index into KernelStageReport.kernelNames per encoder, in encoding order.
    public let kernelIndex: [Int]
    public let analysis: KernelStageTokenAnalysis?
    public let counterCrossCheckFailures: [String]
    public let wallMicros: Double
    public let failures: [String]
}

public struct KernelStageRun: Codable, Equatable, Sendable {
    public let stageSampled: Bool
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let outputs: Int
    public let allFinite: Bool
    public let comparison: OwnProcessLogitComparison
    public let totals: RouterOverlapGateTotals
    public let counts: RouterOverlapGatePhaseCounts
    public let prefillWallMicros: Double
    public let continuationWallMicros: [Double]
    public let continuationWallMedianMicros: Double?
    public let processBefore: OwnProcessTaskSnapshot?
    public let processAfter: OwnProcessTaskSnapshot?
    public let processDelta: OwnProcessSnapshotDelta?
    public let preconditionFailures: [String]
    public let error: String?
    public let passed: Bool
    public let failure: String?
}

public struct KernelStageFixture: Codable, Equatable, Sendable {
    public let fixture: String
    public let reference: RouterOverlapGateQualifiedArm
    public let plain: KernelStageRun?
    public let stageSampled: KernelStageRun?
    public let bitComparedSteps: Int
    public let bitMismatchCount: Int
    public let firstBitMismatch: RouterOverlapGateBitMismatch?
    public let sampledVsPlainTotalsMismatches: [String]
    public let tokens: [KernelStageToken]
    public let kernelStats: [KernelStageLabelStat]
    public let familyStats: [KernelStageLabelStat]
    /// Median sampled continuation wall minus median plain continuation wall: profile overhead of the changed
    /// encoder topology plus sampling, not TPOT and not a performance result.
    public let sampledMinusPlainWallMedianMicros: Double?
    public let passed: Bool
    public let failures: [String]
}

public struct KernelStageReport: Codable {
    public private(set) var kind = KernelStageAttribution.kind
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var notes = KernelStageAttribution.notes
    public private(set) var encoderTopology = KernelStageAttribution.encoderTopology
    public private(set) var timestampUnits = KernelStageAttribution.timestampUnits
    public private(set) var sampleCapacity = KernelStageAttribution.sampleCapacity
    public private(set) var pinnedEncodersPerContinuation = KernelStageAttribution.pinnedEncodersPerContinuation
    public private(set) var pinnedCommandBuffersPerContinuation = KernelStageAttribution.pinnedCommandBuffersPerContinuation
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var ownProcessSnapshots = false
    public var identity: RouterOverlapGateIdentity? = nil
    public var identityFailures: [String] = []
    public var support: KernelStageSupport? = nil
    public var kernelNames: [String] = []
    public var kernelFamilies: [String: String] = [:]
    public var refusalTests: [RouterOverlapGateRefusalTest] = []
    public var refusalPassed = false
    public var fixtures: [KernelStageFixture] = []
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: KernelStageStatus = .aborted

    public init() {}
}

extension KernelStageReport {
    /// Contract normalization at every encode: the four claims are always false.
    func withDiagnosticContract() -> KernelStageReport {
        var copy = self
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

public struct KernelStageOptions: Equatable, Sendable {
    public let gguf: String
    public let region: String
    public let graph: String
    public let fixtures: String
    public let report: String
    public let ownProcessSnapshots: Bool
}

// MARK: - Runner state and hooks

/// One reusable counter sample buffer plus the per-token encoder ledger. Poisoned on any failure and never reused.
final class KernelStageProfilerState {
    let device: MTLDevice
    let sampleBuffer: MTLCounterSampleBuffer
    let capacity: Int
    let families: [String: String]
    var active = false
    var poisonReason: String? = nil
    var kernels: [String?] = []
    var encoderSegment: [Int] = []
    var segmentsOpened = 0
    var segmentsCompleted: [Int] = []
    var segmentIntervalsMicros: [Double] = []
    var lastTokenEndTick: UInt64? = nil

    init(device: MTLDevice, sampleBuffer: MTLCounterSampleBuffer, capacity: Int, families: [String: String]) {
        self.device = device
        self.sampleBuffer = sampleBuffer
        self.capacity = capacity
        self.families = families
    }

    func poison(_ reason: String) {
        if poisonReason == nil { poisonReason = reason }
    }
}

struct KernelStageTokenSampling {
    let kernels: [String]
    let encodersPerCommandBuffer: [Int]
    let commandBufferIntervalMicros: [Double]
    let analysis: KernelStageTokenAnalysis?
    let failures: [String]
}

extension Gemma4Runner {
    public var kernelStageProfileActive: Bool { kernelStageProfiler?.active == true }

    /// beginSegment hook (scope active only): the new command buffer's first sampled encoder.
    func kernelStageOpenSegmentEncoder(_ s: KernelStageProfilerState,
                                       cmd: MTLCommandBuffer) throws -> MTLComputeCommandEncoder {
        if let p = s.poisonReason { throw KernelStageError("stage sampling poisoned: " + p) }
        s.segmentsOpened += 1
        return try kernelStageEncoder(s, cmd: cmd)
    }

    private func kernelStageEncoder(_ s: KernelStageProfilerState,
                                    cmd: MTLCommandBuffer) throws -> MTLComputeCommandEncoder {
        let index = s.kernels.count
        guard 2 * index + 2 <= s.capacity else {
            s.poison("sample capacity " + String(s.capacity) + " exceeded at encoder " + String(index))
            throw KernelStageError("stage sampling refused: sample capacity exceeded")
        }
        let pass = MTLComputePassDescriptor()
        pass.dispatchType = .serial
        guard let attachment = pass.sampleBufferAttachments[0] else {
            s.poison("compute pass sample attachment 0 unavailable")
            throw KernelStageError("stage sampling refused: compute pass sample attachment 0 unavailable")
        }
        attachment.sampleBuffer = s.sampleBuffer
        attachment.startOfEncoderSampleIndex = 2 * index
        attachment.endOfEncoderSampleIndex = 2 * index + 1
        guard let enc = cmd.makeComputeCommandEncoder(descriptor: pass) else {
            s.poison("sampled compute encoder could not be created")
            throw KernelStageError("stage sampling refused: sampled compute encoder could not be created")
        }
        s.kernels.append(nil)
        s.encoderSegment.append(s.segmentsOpened - 1)
        return enc
    }

    /// dispatch() / dispatchPipeline() hook (scope active only), before any encoding of this dispatch: one sampled
    /// encoder per dispatch in the open command buffer; direct submissions, unknown tags, capacity overflow and
    /// forbidden scopes are refused and poison the sample buffer.
    func kernelStagePrepareDispatch(_ s: KernelStageProfilerState, kernel: String) throws {
        if let p = s.poisonReason { throw KernelStageError("stage sampling poisoned: " + p) }
        guard !routerOverlapScopeActive, routerOverlapPendingCount == 0, !routerDeviceRoutingActive,
              !routeCaptureActive else {
            s.poison("dispatch of " + kernel + " inside a router overlap, device routing or route capture scope")
            throw KernelStageError("stage sampling refused: forbidden scope active")
        }
        guard let cmd = openCmd, openEnc != nil else {
            s.poison("direct (unsegmented) submission of " + kernel)
            throw KernelStageError("stage sampling refused: direct (unsegmented) submission of " + kernel)
        }
        guard s.families[kernel] != nil else {
            s.poison("unknown kernel tag " + kernel)
            throw KernelStageError("stage sampling refused: unknown kernel tag " + kernel)
        }
        guard let last = s.kernels.indices.last, s.encoderSegment[last] == s.segmentsOpened - 1 else {
            s.poison("open command buffer has no sampled encoder")
            throw KernelStageError("stage sampling refused: open command buffer has no sampled encoder")
        }
        if s.kernels[last] != nil {
            openEnc?.endEncoding()
            openEnc = nil
            openEnc = try kernelStageEncoder(s, cmd: cmd)
        }
        s.kernels[s.kernels.count - 1] = kernel
    }

    /// flushOpenSegment hook after a completed, error-free command buffer.
    func kernelStageSegmentCompleted(_ s: KernelStageProfilerState, cmd: MTLCommandBuffer) {
        s.segmentsCompleted.append(s.segmentsOpened - 1)
        s.segmentIntervalsMicros.append((cmd.gpuEndTime - cmd.gpuStartTime) * 1e6)
    }

    /// discardOpenSegment hook: an uncommitted sampled command buffer never writes its samples.
    func kernelStageNoteDiscard(_ s: KernelStageProfilerState) {
        if openEnc != nil || openCmd != nil { s.poison("sampled command buffer discarded without commit") }
    }

    /// DIAGNOSTIC ONLY. Runs body (one executor call) with stage sampling armed, then resolves the samples after every
    /// sampled command buffer completed. Refused before any GPU work inside the router overlap, device routing or route
    /// capture scopes, with any instrumentation or open segment, when nested, for a poisoned or foreign buffer and when
    /// the capacity cannot hold two samples for each expected encoder. Never reuses a buffer after a failure.
    func withKernelStageSampling<T>(_ s: KernelStageProfilerState, expectedEncoders: Int,
                                    _ body: () throws -> T) throws -> (T, KernelStageTokenSampling) {
        var f = [String]()
        if kernelStageProfiler != nil || s.active { f.append("a stage sampling scope is already active") }
        if s.poisonReason != nil { f.append("sample buffer poisoned by an earlier failure; never reused") }
        if routerOverlapScopeActive || routerOverlapPendingCount != 0 {
            f.append("router overlap scope active or tickets pending")
        }
        if routeCaptureActive { f.append("route capture active") }
        if routerDeviceRoutingActive { f.append("device routing scope active") }
        if !RouterOverlapModelGate.instrumentationOff(self) {
            f.append("probe/trace/breakdown/timeline/diagnostic/generated kernel/open segment/expert cache present")
        }
        if s.device !== q4RunnerDevice { f.append("sample buffer belongs to another device") }
        if expectedEncoders <= 0 || expectedEncoders > s.capacity / 2 {
            f.append("sample capacity " + String(s.capacity) + " cannot hold two samples for each of "
                     + String(expectedEncoders) + " expected encoders")
        }
        guard f.isEmpty else { throw KernelStageError("stage sampling refused: " + f.joined(separator: "; ")) }
        s.kernels = []
        s.encoderSegment = []
        s.segmentsOpened = 0
        s.segmentsCompleted = []
        s.segmentIntervalsMicros = []
        s.active = true
        kernelStageProfiler = s
        let result: T
        do {
            result = try body()
        } catch {
            s.poison("sampled call threw: " + String(describing: error))
            kernelStageExit(s)
            throw error
        }
        kernelStageExit(s)
        return (result, kernelStageCollect(s))
    }

    private func kernelStageExit(_ s: KernelStageProfilerState) {
        if openEnc != nil || openCmd != nil {
            s.poison("segment still open at scope exit; discarded uncommitted")
            discardOpenSegment()
        }
        s.active = false
        kernelStageProfiler = nil
    }

    private func kernelStageCollect(_ s: KernelStageProfilerState) -> KernelStageTokenSampling {
        var f = [String]()
        if let p = s.poisonReason { f.append("poisoned: " + p) }
        let n = s.kernels.count
        let labels = s.kernels.compactMap { $0 }
        if n == 0 { f.append("no sampled encoder") }
        if labels.count != n { f.append(String(n - labels.count) + " sampled encoder(s) without a dispatch") }
        if s.segmentsCompleted != Array(0..<s.segmentsOpened) {
            f.append("not every sampled command buffer completed")
        }
        var perSegment = [Int](repeating: 0, count: s.segmentsOpened)
        for seg in s.encoderSegment where seg >= 0 && seg < perSegment.count { perSegment[seg] += 1 }
        var analysis: KernelStageTokenAnalysis? = nil
        if f.isEmpty {
            let count = 2 * n
            do {
                if let first = try s.sampleBuffer.resolveCounterRange(0..<count),
                   let second = try s.sampleBuffer.resolveCounterRange(0..<count) {
                    if first.count != count * 8 {
                        f.append("resolved " + String(first.count) + " bytes, expected " + String(count * 8))
                    } else if first != second {
                        f.append("repeated resolve of the same range differs")
                    } else {
                        var raw = [UInt64]()
                        raw.reserveCapacity(count)
                        first.withUnsafeBytes { bytes in
                            for i in 0..<count { raw.append(bytes.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self)) }
                        }
                        let a = KernelStageAttribution.analyze(raw: raw, encoders: n, previousLastTick: s.lastTokenEndTick)
                        analysis = a
                        if a.failures.isEmpty { s.lastTokenEndTick = a.lastEndTicks }
                    }
                } else {
                    f.append("counter resolve returned no data")
                }
            } catch {
                f.append("counter resolve threw: " + String(describing: error))
            }
        }
        if !f.isEmpty || !(analysis?.failures.isEmpty ?? false) {
            s.poison("token sampling failed: " + (f + (analysis?.failures ?? [])).joined(separator: "; "))
        }
        return KernelStageTokenSampling(kernels: labels, encodersPerCommandBuffer: perSegment,
                                        commandBufferIntervalMicros: s.segmentIntervalsMicros, analysis: analysis,
                                        failures: f)
    }
}

// MARK: - Driver

public enum KernelStageAttribution {
    public static let kind = "kernel-stage-attribution-off-stage-boundary-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 16 << 20
    public static let correctnessContract = 1e-4
    /// Hard bound on samples per sampled token (two per encoder).
    public static let maxSamples = 4096
    /// Counter error value (the SDK macro is not importable into Swift).
    public static let counterErrorValue = UInt64.max
    public static let pinnedEncodersPerContinuation = RouterDeviceRoutingPerfProtocol.required.offContinuationKernels
    public static let pinnedCommandBuffersPerContinuation =
        RouterDeviceRoutingPerfProtocol.required.offContinuationCommandBuffers
    /// Derived capacity: two samples per pinned OFF continuation kernel (2 x 1349 = 2698 <= 4096).
    public static let sampleCapacity = 2 * pinnedEncodersPerContinuation
    public static let usage = "usage: kernel-stage-attribution --gguf M.gguf --region R.json --graph G.json "
        + "--fixtures F.json --report NEW_REPORT.json [--own-process-snapshots]"
    public static let encoderTopology = "while sampling: one serial compute encoder per dispatch, with start/end "
        + "stage samples, inside the unchanged command buffers (37 per continuation); kernels, order, bindings, "
        + "commits and waits unchanged; the unsampled runner uses one encoder per command buffer"
    public static let timestampUnits = "raw GPU timestamp ticks as resolved; the installed SDK declarations state no "
        + "unit; nothing is converted"

    public static let familyNames = [
        "q4Matvec", "q6LmHead", "routerF32Matvec", "embedding", "rmsnorm", "attention", "rope", "kvAppend",
        "geluUp", "axpy", "elementwise", "softcap", "routerScale", "residentZero", "moeGateUp", "moeAct",
        "moeDown",
    ]

    public static let refusalTestNames = [
        "stage-sampling-entry-inside-router-overlap-scope",
        "stage-sampling-entry-inside-device-routing-scope",
        "stage-sampling-entry-with-breakdown-instrumentation",
        "stage-sampling-entry-with-timeline-instrumentation",
        "stage-sampling-entry-capacity-below-expected-encoders",
        "nested-stage-sampling-scope",
        "unsegmented-submission-inside-stage-sampling",
    ]

    public static let notes: [String] = [
        "Diagnostic only: gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false; "
            + "no artifact, installation, promotion, default or candidate change.",
        "Sampled path: the incumbent OFF arm only (control preparation, declared modern candidate with par64 and router "
            + "launch divisor 1, installed v1 E4R32, Q4 width 32, CPU FP64 routing). Stage sampling refuses to start or "
            + "dispatch inside the router overlap, device routing or route capture scopes; no existing guard was changed.",
        "Encoder topology changes while sampling: every deferred dispatch gets its own serial compute encoder inside the "
            + "SAME command buffer; kernels, order, bindings, command buffers, commits and waits are unchanged. Sampled "
            + "intervals and the sampled-minus-plain wall overhead describe this changed topology, not the unmodified "
            + "baseline cost.",
        "Per fixture: fresh direct reference, a plain OFF run, then the same OFF run with stage sampling on the 31 "
            + "continuation calls (prefill is not sampled). Both runs need the 32 frozen IDs on the previous actual "
            + "argmax, 32 full-vocabulary finite outputs, exact derived counters, pinned 1349 kernels and 37 command "
            + "buffers per continuation and vector and per-element normalized < 1e-4 against the reference; the sampled "
            + "run must equal the plain run in Float.bitPattern at every logit with identical totals.",
        "Per sampled continuation: one encoder per counted kernel, one sampled command buffer per submitted command "
            + "buffer, exact kernel-name counts against the runner's own counters, two resolved samples per encoder with "
            + "no counter error value, no zero, no end before start, all later than the previous token, and two identical "
            + "resolves. Any failure poisons the sample buffer, which is then never reused.",
        "Encoder intervals are raw stage ticks: not GPU busy time, utilization, bandwidth or recoverable time. Shares "
            + "are ratios of disjoint sampled stage intervals and are omitted when adjacent encoders overlap. CPU wall "
            + "times are profile overhead diagnostics, not TPOT.",
        "Model SHA-256 is the cached explicit-refresh record; nothing here hashes, re-verifies or downloads the model. "
            + "No weights, scales or coefficients enter the report.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func parseOptions(_ argv: [String]) throws -> KernelStageOptions {
        let keys = ["--gguf", "--region", "--graph", "--fixtures", "--report"]
        var v = [String: String]()
        var snapshots = false
        var i = 0
        while i < argv.count {
            let key = argv[i]
            if key == "--own-process-snapshots" {
                guard !snapshots else { throw KernelStageError(usage) }
                snapshots = true
                i += 1
                continue
            }
            guard keys.contains(key), i + 1 < argv.count, v[key] == nil else { throw KernelStageError(usage) }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw KernelStageError(usage) }
            v[key] = value
            i += 2
        }
        guard let gguf = v["--gguf"], let region = v["--region"], let graph = v["--graph"],
              let fixtures = v["--fixtures"], let report = v["--report"] else {
            throw KernelStageError(usage)
        }
        return KernelStageOptions(gguf: gguf, region: region, graph: graph, fixtures: fixtures, report: report,
                                  ownProcessSnapshots: snapshots)
    }

    public static func encodeReport(_ r: KernelStageReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    // MARK: Pure helpers

    /// Exact dispatch names of the OFF path to families; the three MoE names are the installed schedule's entries.
    public static func familyTable(moeGateUp: String, moeAct: String, moeDown: String) -> [String: String] {
        let entries: [(String, String)] = [
            ("g4_q4_matvec", "q4Matvec"),
            ("g4_q6_matvec", "q6LmHead"), (Q6Cooperative.kernelName, "q6LmHead"),
            ("g4_f32_matvec", "routerF32Matvec"),
            ("g4_q6_embed_row", "embedding"),
            ("g4_rmsnorm", "rmsnorm"), ("g4_rmsnorm_vl", "rmsnorm"), (RMSCooperative.kernelName, "rmsnorm"),
            (RMSStatisticsKernels.computeKernelName, "rmsnorm"), (RMSStatisticsKernels.applyKernelName, "rmsnorm"),
            ("g4_attn", "attention"), (AttentionDiagnostic.kernelName, "attention"),
            ("g4_rope", "rope"), ("g4_cache_append", "kvAppend"), ("g4_gelu_up", "geluUp"), ("g4_axpy", "axpy"),
            ("g4_add", "elementwise"), ("g4_scale", "elementwise"), ("g4_softcap", "softcap"),
            (RouterResidencyKernel.kernelName, "routerScale"), (RouterResidencyKernel.zeroKernelName, "residentZero"),
            (moeGateUp, "moeGateUp"), (moeAct, "moeAct"), (moeDown, "moeDown"),
        ]
        var table = [String: String]()
        for (name, family) in entries where table[name] == nil { table[name] = family }
        return table
    }

    /// Exact dispatch names to the runner counter incremented once per dispatch of that name.
    public static func counterKeys(moeGateUp: String, moeAct: String, moeDown: String) -> [String: String] {
        let entries: [(String, String)] = [
            ("g4_q4_matvec", "q4Generic"), (moeGateUp, "moeGu"), (moeAct, "moeAct"), (moeDown, "moeDn"),
            (AttentionDiagnostic.kernelName, "parallelAttentionApplied"),
            (RMSCooperative.kernelName, "rmsCooperativeApplied"),
            (RMSStatisticsKernels.computeKernelName, "rmsStatsComputed"),
            (RMSStatisticsKernels.applyKernelName, "rmsStatsReused"),
            (Q6Cooperative.kernelName, "q6CooperativeApplied"),
            (RouterResidencyKernel.kernelName, "routerScaleApplied"),
            (RouterResidencyKernel.zeroKernelName, "residentZeroApplied"),
            ("g4_f32_matvec", "routerLaunchApplied"),
        ]
        var table = [String: String]()
        for (name, key) in entries where table[name] == nil { table[name] = key }
        return table
    }

    public static func counterCrossCheck(kernelCounts: [String: Int], encoders: Int,
                                         counterDelta: RouterOverlapGateCounters,
                                         counterKeys: [String: String]) -> [String] {
        var f = [String]()
        if encoders != counterDelta["kernels"] {
            f.append("encoders " + String(encoders) + " != counted kernels " + String(counterDelta["kernels"]))
        }
        var byKey = [String: Int]()
        for (kernel, key) in counterKeys { byKey[key, default: 0] += kernelCounts[kernel] ?? 0 }
        for key in byKey.keys.sorted() where byKey[key] != counterDelta[key] {
            f.append("sampled " + key + " encoders " + String(byKey[key] ?? 0) + " != runner counter "
                     + String(counterDelta[key]))
        }
        return f
    }

    /// Checks resolved raw samples (start, end per encoder): count, counter error value, zero, end before start and
    /// staleness against the previous sampled token. Overlap of adjacent encoders is counted, not a failure.
    public static func analyze(raw: [UInt64], encoders: Int, previousLastTick: UInt64?) -> KernelStageTokenAnalysis {
        func failed(_ f: [String]) -> KernelStageTokenAnalysis {
            KernelStageTokenAnalysis(encoders: encoders, samples: raw.count, firstStartTicks: nil, lastEndTicks: nil,
                                     spanTicks: nil, intervalTicks: [], sumEncoderTicks: nil,
                                     overlappingAdjacentEncoders: 0, failures: f)
        }
        guard encoders > 0, encoders <= maxSamples / 2, raw.count == 2 * encoders else {
            return failed(["resolved " + String(raw.count) + " samples for " + String(encoders) + " encoders"])
        }
        var f = [String]()
        let errors = raw.filter { $0 == counterErrorValue }.count
        if errors > 0 { f.append(String(errors) + " sample(s) carry the counter error value") }
        let zeros = raw.filter { $0 == 0 }.count
        if zeros > 0 { f.append(String(zeros) + " zero sample(s)") }
        var reversed = 0
        for k in 0..<encoders where raw[2 * k + 1] < raw[2 * k] { reversed += 1 }
        if reversed > 0 { f.append(String(reversed) + " encoder(s) with end before start") }
        if let previous = previousLastTick, let lowest = raw.min(), lowest <= previous {
            f.append("samples not later than the previous sampled token (stale or non-monotonic)")
        }
        guard f.isEmpty else { return failed(f) }
        var intervals = [UInt64]()
        intervals.reserveCapacity(encoders)
        var sum: UInt64 = 0
        var overlaps = 0
        var lastEnd = raw[1]
        for k in 0..<encoders {
            let d = raw[2 * k + 1] - raw[2 * k]
            intervals.append(d)
            let (s, overflow) = sum.addingReportingOverflow(d)
            if overflow { return failed(["encoder tick sum overflows"]) }
            sum = s
            if k > 0, raw[2 * k] < raw[2 * k - 1] { overlaps += 1 }
            if raw[2 * k + 1] > lastEnd { lastEnd = raw[2 * k + 1] }
        }
        let first = raw[0]
        return KernelStageTokenAnalysis(encoders: encoders, samples: raw.count, firstStartTicks: first,
                                        lastEndTicks: lastEnd, spanTicks: lastEnd >= first ? lastEnd - first : nil,
                                        intervalTicks: intervals, sumEncoderTicks: sum,
                                        overlappingAdjacentEncoders: overlaps, failures: [])
    }

    /// Per-label tick sums per token. Inputs come from analyze(), whose per-token sum is overflow-checked, so the
    /// wrapping additions below cannot wrap for real data.
    public static func aggregate(_ tokens: [KernelStageTokenInput], label: (String) -> String,
                                 family: (String) -> String?) -> [KernelStageLabelStat] {
        let valid = tokens.filter { $0.labels.count == $0.intervalTicks.count && !$0.labels.isEmpty }
        var perToken = [[String: (count: Int, ticks: UInt64)]]()
        var totals = [UInt64]()
        var familyOf = [String: String]()
        var keys = Set<String>()
        for t in valid {
            var m = [String: (count: Int, ticks: UInt64)]()
            var total: UInt64 = 0
            for (i, raw) in t.labels.enumerated() {
                let k = label(raw)
                keys.insert(k)
                if familyOf[k] == nil, let fam = family(raw) { familyOf[k] = fam }
                let e = m[k] ?? (count: 0, ticks: 0)
                m[k] = (count: e.count + 1, ticks: e.ticks &+ t.intervalTicks[i])
                total = total &+ t.intervalTicks[i]
            }
            perToken.append(m)
            totals.append(total)
        }
        let disjoint = !valid.isEmpty && valid.allSatisfy { $0.disjoint }
        var out = [KernelStageLabelStat]()
        for k in keys.sorted() {
            let counts = perToken.map { $0[k]?.count ?? 0 }
            let ticks = perToken.map { $0[k]?.ticks ?? 0 }
            let constant: Int? = Set(counts).count == 1 ? counts[0] : nil
            var share: Double? = nil
            if disjoint, totals.allSatisfy({ $0 > 0 }) {
                share = RouterOverlapModelGate.median(zip(ticks, totals).map { Double($0.0) / Double($0.1) })
            }
            out.append(KernelStageLabelStat(
                label: k, family: familyOf[k], encodersPerToken: constant, tokens: ticks.count, ticksPerToken: ticks,
                medianTicksPerToken: RouterOverlapModelGate.median(ticks.map { Double($0) }),
                minTicksPerToken: ticks.min(), maxTicksPerToken: ticks.max(), medianShareOfSampledEncoderTicks: share))
        }
        return out
    }

    public static func finalStatus(_ r: KernelStageReport) -> KernelStageStatus {
        if !r.restoredPristine { return .runnerNotRestored }
        if r.abortReason != nil { return .aborted }
        if let s = r.support, !s.supported { return .unsupported }
        let passed = r.identity != nil && r.identityFailures.isEmpty && r.support?.supported == true
            && r.refusalPassed && r.fixtures.map({ $0.fixture }) == RouterOverlapGateFixtures.order
            && r.fixtures.allSatisfy({ $0.passed })
        return passed ? .profiledChecksPassed : .profiledChecksFailed
    }

    static func finalize(_ r: inout KernelStageReport) {
        r.status = finalStatus(r)
        r = r.withDiagnosticContract()
    }

    /// Actual device query, then one shared counter sample buffer of `capacity` samples on the runner's own device.
    static func makeProfilerState(device: MTLDevice, capacity: Int,
                                  families: [String: String]) -> (state: KernelStageProfilerState?,
                                                                  support: KernelStageSupport) {
        let atStage = device.supportsCounterSampling(.atStageBoundary)
        let atDispatch = device.supportsCounterSampling(.atDispatchBoundary)
        var timestampSet: MTLCounterSet? = nil
        for set in device.counterSets ?? [] where set.name == MTLCommonCounterSet.timestamp.rawValue {
            if set.counters.contains(where: { $0.name == MTLCommonCounter.timestamp.rawValue }) { timestampSet = set }
        }
        let found = timestampSet != nil
        func support(_ created: Bool, _ error: String?) -> KernelStageSupport {
            KernelStageSupport(device: device.name, supportsAtStageBoundary: atStage,
                               supportsAtDispatchBoundary: atDispatch, timestampCounterSetFound: found,
                               sampleBufferCreated: created, sampleCapacity: capacity, error: error)
        }
        guard capacity >= 2, capacity % 2 == 0, capacity <= maxSamples else {
            return (nil, support(false, "sample capacity " + String(capacity) + " outside 2..." + String(maxSamples)))
        }
        guard atStage, let counterSet = timestampSet else {
            return (nil, support(false, "stage-boundary sampling or the timestamp counter set is unavailable"))
        }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counterSet
        descriptor.label = "kernel-stage-attribution"
        descriptor.storageMode = .shared
        descriptor.sampleCount = capacity
        do {
            let buffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
            return (KernelStageProfilerState(device: device, sampleBuffer: buffer, capacity: capacity,
                                             families: families), support(true, nil))
        } catch {
            return (nil, support(false, "counter sample buffer creation failed: " + String(describing: error)))
        }
    }

    // MARK: Entry point

    /// Never throws; every failure is recorded. The caller prints END only when the runner is quiescent, restored and
    /// no sampling scope is active.
    public static func run(region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                           runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                           processSnapshots: Bool, emit: (String) -> Void) -> KernelStageReport {
        let proto = RouterOverlapGateProtocol.required
        var report = KernelStageReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.ownProcessSnapshots = processSnapshots
        let pristineStart = RouterOverlapModelGate.pristine(runner)
        let instrumentationStart = RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner) && !runner.routerDeviceRoutingActive
            && !runner.routeCaptureActive && runner.kernelStageProfiler == nil
        var setup = [String]()
        if !pristineStart { setup.append("runner is not pristine at start") }
        if !instrumentationStart { setup.append("instrumentation, axis flags or scopes are not off at start") }
        if fixtures != RouterOverlapGateFixtures.pinned { setup.append("fixtures differ from the pinned fixtures") }
        if !moeIsLowercaseHex64(modelSha256) || !moeIsLowercaseHex64(executableDigest) {
            setup.append("model and executable digests must be lowercase 64-hex")
        }
        if sampleCapacity < 2 || sampleCapacity > maxSamples { setup.append("derived sample capacity out of bounds") }
        guard setup.isEmpty else {
            report.abortReason = setup.joined(separator: "; ")
            report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            finalize(&report)
            emit("aborted before any GPU work: " + (report.abortReason ?? ""))
            return report
        }
        do {
            try runner.withQ4DiagnosticReferenceWidth(proto.q4LaunchWidth) { () -> Void in
                try scoped(&report, region: region, graph: graph, fixtures: fixtures, runner: runner,
                           modelSha256: modelSha256, executableDigest: executableDigest,
                           pristineStart: pristineStart, instrumentationStart: instrumentationStart,
                           processSnapshots: processSnapshots, emit: emit)
            }
        } catch {
            report.abortReason = "attribution threw: " + String(describing: error)
        }
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            && RouterOverlapModelGate.instrumentationOff(runner) && RouterOverlapModelGate.axisFlagsClean(runner)
            && !runner.routerDeviceRoutingActive && runner.kernelStageProfiler == nil
        finalize(&report)
        emit("status " + report.status.rawValue + "; restored pristine " + String(report.restoredPristine)
             + "; encoder intervals are raw stage ticks, not busy time; no gain or production claim")
        return report
    }

    static func scoped(_ report: inout KernelStageReport, region: MoERegionIR, graph: GemmaGraphIR,
                       fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner, modelSha256: String,
                       executableDigest: String, pristineStart: Bool, instrumentationStart: Bool,
                       processSnapshots: Bool, emit: (String) -> Void) throws {
        let proto = RouterOverlapGateProtocol.required
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == proto.q4LaunchWidth else {
            throw KernelStageError("Q4 reference width 32 is not in effect inside the width scope")
        }
        var prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
        let derived = RouterOverlapModelGate.derive(on: prepared.on, runner: runner)
        if derived.statisticsGroups > 0 {
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
            emit("identity FAILED; nothing sampled")
            return
        }
        guard let installed = runner.moeGenerated else {
            throw KernelStageError("installed routed lowering missing")
        }
        let gu = installed.schedule.guKernelName
        let act = installed.schedule.actKernelName
        let dn = installed.schedule.dnKernelName
        let families = familyTable(moeGateUp: gu, moeAct: act, moeDown: dn)
        let keys = counterKeys(moeGateUp: gu, moeAct: act, moeDown: dn)
        let names = families.keys.sorted()
        var index = [String: Int]()
        for (i, n) in names.enumerated() { index[n] = i }
        report.kernelNames = names
        report.kernelFamilies = families
        let made = makeProfilerState(device: runner.q4RunnerDevice, capacity: sampleCapacity, families: families)
        report.support = made.support
        emit(RouterOverlapModelGate.event("support", made.support))
        guard let state = made.state else {
            emit("stage sampling unsupported; nothing sampled")
            return
        }
        let refusals = refusalTests(ctx, state: state, region: region)
        report.refusalTests = refusals
        report.refusalPassed = refusals.map({ $0.name }) == refusalTestNames && refusals.allSatisfy({ $0.passed })
        emit(RouterOverlapModelGate.event("refusalTests", refusals))
        guard report.refusalPassed else {
            emit("refusal tests FAILED; nothing sampled")
            return
        }
        for fx in fixtures {
            runner.uninstallMoeGeneratedLowering()
            let ref = RouterOverlapModelGate.qualify(.reference, ctx, fx, reference: [], off: [], offTotals: nil,
                                                     keep: true)
            prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
            guard RouterOverlapModelGate.derive(on: prepared.on, runner: runner) == derived,
                  runner.moeGenerated?.schedule.guKernelName == gu else {
                throw KernelStageError("derived counts or installed kernels changed after re-preparation")
            }
            ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
            var plain: RunOutcome? = nil
            if ref.record.passed, ref.kept.count == proto.outputsPerRun {
                let p = timedRun(ctx, fx, profile: nil, kernelIndex: index, counterKeys: keys, reference: ref.kept,
                                 plain: [], snapshots: processSnapshots)
                plain = p
                emit(RouterOverlapModelGate.event("plainRun", p.run))
            }
            var sampled: RunOutcome? = nil
            if let p = plain, p.run.passed {
                let q = timedRun(ctx, fx, profile: state, kernelIndex: index, counterKeys: keys, reference: ref.kept,
                                 plain: p.kept, snapshots: processSnapshots)
                sampled = q
                emit(RouterOverlapModelGate.event("stageSampledRun", q.run))
            }
            let record = makeFixture(fx, reference: ref.record, plain: plain, sampled: sampled, names: names,
                                     families: families)
            report.fixtures.append(record)
            emit(RouterOverlapModelGate.event("familyStats", record.familyStats))
            emit("fixture " + fx.name + ": passed " + String(record.passed))
            if !RouterOverlapModelGate.quiescent(runner) || runner.kernelStageProfiler != nil {
                throw KernelStageError("runner not quiescent or a sampling scope left active after " + fx.name)
            }
        }
    }

    // MARK: Refusal tests

    static func refusalTests(_ c: RouterOverlapGateContext, state: KernelStageProfilerState,
                             region: MoERegionIR) -> [RouterOverlapGateRefusalTest] {
        let r = c.runner
        let expected = pinnedEncodersPerContinuation
        var tests = [RouterOverlapGateRefusalTest]()
        r.reset()
        func fresh() -> KernelStageProfilerState? {
            KernelStageAttribution.makeProfilerState(device: r.q4RunnerDevice, capacity: state.capacity,
                                                     families: state.families).state
        }
        func probe(_ name: String, inside: Bool, expect: String = "stage sampling refused",
                   _ action: () throws -> Void) {
            let before = RouterOverlapGateGuardState(r)
            let poisonBefore = state.poisonReason
            var refused = false
            var message: String? = nil
            do { try action() } catch {
                refused = true
                message = String(describing: error)
            }
            let after = RouterOverlapGateGuardState(r)
            let noGpu = before.kernels == after.kernels && before.buffers == after.buffers
                && before.gpuBits == after.gpuBits && before.position == after.position
            let intact = !after.open && after.pending == 0 && before.scopeActive == after.scopeActive
                && before.moeDigest == after.moeDigest && before.q4Width == after.q4Width
                && r.kernelStageProfiler == nil && !state.active && state.poisonReason == poisonBefore
                && !r.routerDeviceRoutingActive && !r.breakdownInstrumentation && !r.commandTimelineEnabled
            let attributed = message?.contains(expect) ?? false
            tests.append(RouterOverlapGateRefusalTest(
                name: name, insideScope: inside, refused: refused, error: message, noGpuWork: noGpu,
                stateIntact: intact, passed: refused && attributed && noGpu && intact))
        }
        probe(refusalTestNames[0], inside: true) {
            try r.withRouterOverlapDiagnosticScope { () -> Void in
                _ = try r.withKernelStageSampling(state, expectedEncoders: expected) { () -> Int in 0 }
            }
        }
        probe(refusalTestNames[1], inside: true) {
            try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Void in
                _ = try r.withKernelStageSampling(state, expectedEncoders: expected) { () -> Int in 0 }
            }
        }
        probe(refusalTestNames[2], inside: false) {
            r.breakdownInstrumentation = true
            defer { r.breakdownInstrumentation = false }
            _ = try r.withKernelStageSampling(state, expectedEncoders: expected) { () -> Int in 0 }
        }
        probe(refusalTestNames[3], inside: false) {
            r.commandTimelineEnabled = true
            defer { r.commandTimelineEnabled = false }
            _ = try r.withKernelStageSampling(state, expectedEncoders: expected) { () -> Int in 0 }
        }
        probe(refusalTestNames[4], inside: false) {
            _ = try r.withKernelStageSampling(state, expectedEncoders: state.capacity / 2 + 1) { () -> Int in 0 }
        }
        probe(refusalTestNames[5], inside: true) {
            guard let s = fresh() else { return }
            _ = try r.withKernelStageSampling(s, expectedEncoders: expected) { () -> Int in
                try r.withKernelStageSampling(s, expectedEncoders: expected) { () -> Int in 0 }.0
            }
        }
        probe(refusalTestNames[6], inside: true) {
            guard let s = fresh() else { return }
            let unsegmented = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
            _ = try r.withKernelStageSampling(s, expectedEncoders: expected) { () -> RouterOverlapModelGate.Step in
                try GemmaForge.executeGraphContinue(c.graph, runner: r, token: 2, candidate: unsegmented,
                                                    prepared: c.off)
            }
        }
        return tests
    }

    // MARK: Runs

    struct RunOutcome {
        let run: KernelStageRun
        let tokens: [KernelStageToken]
        let kept: [[Float]]
        let bits: RouterOverlapGateBitComparator
    }

    /// One OFF 32-output run (prefill plus 31 continuations on the previous actual argmax). With a profile state each
    /// continuation call runs inside one stage sampling scope; the wall timer brackets only the executor call, while
    /// scope entry, resolve, counter snapshots and checks run outside it.
    static func timedRun(_ c: RouterOverlapGateContext, _ fx: RouterOverlapGateFixture,
                         profile: KernelStageProfilerState?, kernelIndex: [String: Int],
                         counterKeys: [String: String], reference: [[Float]], plain: [[Float]],
                         snapshots: Bool) -> RunOutcome {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let vocab = r.config.vocabSize
        let tag = fx.name + (profile == nil ? " plain" : " stage-sampled")
        var pre = RouterOverlapModelGate.preconditions(.off, c, prepared: c.off)
        if r.kernelStageProfiler != nil { pre.append("a stage sampling scope is already active") }
        if r.routerDeviceRoutingActive || r.routeCaptureActive { pre.append("device routing or route capture active") }
        var g = RouterOverlapGateGreedy()
        var cmp = OwnProcessLogitComparison()
        var bits = RouterOverlapGateBitComparator()
        var kept = [[Float]]()
        var walls = [Double]()
        var prefillWall = -1.0
        var tokens = [KernelStageToken]()
        func observe(_ step: Int, _ logits: [Float]) {
            cmp.add(step: step, logits: logits, reference: step < reference.count ? reference[step] : [])
            if profile == nil {
                kept.append(logits)
            } else if step < plain.count {
                bits.add(step: step, on: logits, off: plain[step])
            } else {
                bits.missing += 1
            }
        }
        let before = snapshots
            ? OwnProcessInfo.snapshot(label: "run-start " + tag, metalAllocatedBytes: r.metalAllocatedSample()) : nil
        if pre.isEmpty {
            r.reset()
            g.s0 = RouterOverlapGateCounters(r)
            g.s1 = g.s0
            do {
                let t0 = RouterOverlapModelGate.now()
                let first = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt, candidate: c.candidate,
                                                        prepared: c.off)
                prefillWall = RouterOverlapModelGate.micros(t0, RouterOverlapModelGate.now())
                g.s1 = RouterOverlapGateCounters(r)
                g.totals.prefillKernels = first.kernels
                g.totals.prefillBuffers = first.buffers
                g.totals.prefillGpuElapsedMicros = first.gpuMicros
                if RouterOverlapModelGate.fullFinite(first.logits, vocab) {
                    observe(0, first.logits)
                    g.ids.append(first.argmax)
                    g.outputs = 1
                    while g.outputs < proto.outputsPerRun {
                        let token = g.ids[g.ids.count - 1]
                        let k0 = RouterOverlapGateCounters(r)
                        var wall = -1.0
                        func call() throws -> RouterOverlapModelGate.Step {
                            let a = RouterOverlapModelGate.now()
                            let out = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token,
                                                                          candidate: c.candidate, prepared: c.off)
                            wall = RouterOverlapModelGate.micros(a, RouterOverlapModelGate.now())
                            return out
                        }
                        var sampling: KernelStageTokenSampling? = nil
                        let step: RouterOverlapModelGate.Step
                        if let s = profile {
                            let res = try r.withKernelStageSampling(s, expectedEncoders: pinnedEncodersPerContinuation,
                                                                    call)
                            step = res.0
                            sampling = res.1
                        } else {
                            step = try call()
                        }
                        let k1 = RouterOverlapGateCounters(r)
                        walls.append(wall)
                        g.totals.continuationKernels.append(step.kernels)
                        g.totals.continuationBuffers.append(step.buffers)
                        g.totals.continuationGpuElapsedMicros.append(step.gpuMicros)
                        guard RouterOverlapModelGate.fullFinite(step.logits, vocab) else {
                            g.allFinite = false
                            g.error = "output " + String(g.outputs) + " not full-vocabulary finite; run stopped"
                            break
                        }
                        let index = g.outputs
                        observe(index, step.logits)
                        g.ids.append(step.argmax)
                        g.outputs += 1
                        if let smp = sampling {
                            let tok = makeToken(continuation: index - 1, sampling: smp, kernels: step.kernels,
                                                buffers: step.buffers, counterDelta: k1.minus(k0),
                                                counterKeys: counterKeys, kernelIndex: kernelIndex, wall: wall)
                            tokens.append(tok)
                            if !tok.failures.isEmpty {
                                g.error = "stage sampling checks failed at continuation " + String(index - 1)
                                break
                            }
                        }
                    }
                } else {
                    g.allFinite = false
                    g.error = "output 0 not full-vocabulary finite; run stopped"
                }
            } catch {
                g.error = "threw: " + String(describing: error)
            }
            g.s2 = RouterOverlapGateCounters(r)
            r.reset()
        }
        let after = snapshots
            ? OwnProcessInfo.snapshot(label: "run-end " + tag, metalAllocatedBytes: r.metalAllocatedSample()) : nil
        var processDelta: OwnProcessSnapshotDelta? = nil
        if let b = before, let a = after { processDelta = OwnProcessInfo.delta(b, a) }
        let counts = RouterOverlapModelGate.phaseCounts(g, c.derived, .off, promptCount: fx.prompt.count)
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if g.ids != fx.frozenIds { f.append("output IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != proto.outputsPerRun { f.append("not 32 full-vocabulary finite outputs") }
            if !(prefillWall.isFinite && prefillWall > 0) { f.append("prefill wall invalid") }
            if RouterOverlapModelGate.runTPOT(walls) == nil { f.append("continuation walls are not 31 positive values") }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            f += RouterDeviceRoutingPerformance.pinnedTotalsFailures(g.totals, arm: .off)
            f += cmp.failures(contract: correctnessContract, expectedSteps: proto.outputsPerRun)
            if profile != nil {
                if tokens.count != proto.continuationsPerRun {
                    f.append("sampled " + String(tokens.count) + " of 31 continuations")
                }
                let bad = tokens.filter { !$0.failures.isEmpty }.count
                if bad > 0 { f.append(String(bad) + " sampled continuation(s) failed their checks") }
            }
            if !RouterOverlapModelGate.quiescent(r) || r.kernelStageProfiler != nil {
                f.append("runner not quiescent or a sampling scope left active")
            }
        }
        let run = KernelStageRun(
            stageSampled: profile != nil, ids: g.ids, idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs,
            allFinite: g.allFinite, comparison: cmp, totals: g.totals, counts: counts, prefillWallMicros: prefillWall,
            continuationWallMicros: walls, continuationWallMedianMicros: RouterOverlapModelGate.median(walls),
            processBefore: before, processAfter: after, processDelta: processDelta, preconditionFailures: pre,
            error: g.error, passed: f.isEmpty, failure: f.isEmpty ? nil : f.joined(separator: "; "))
        return RunOutcome(run: run, tokens: tokens, kept: kept, bits: bits)
    }

    static func makeToken(continuation: Int, sampling s: KernelStageTokenSampling, kernels: Int, buffers: Int,
                          counterDelta: RouterOverlapGateCounters, counterKeys: [String: String],
                          kernelIndex: [String: Int], wall: Double) -> KernelStageToken {
        var f = s.failures
        if let a = s.analysis {
            f += a.failures
        } else if f.isEmpty {
            f.append("no resolved samples")
        }
        if s.kernels.count != kernels {
            f.append("sampled encoders " + String(s.kernels.count) + " != dispatched kernels " + String(kernels))
        }
        if s.encodersPerCommandBuffer.count != buffers {
            f.append("sampled command buffers " + String(s.encodersPerCommandBuffer.count)
                     + " != submitted command buffers " + String(buffers))
        }
        if s.encodersPerCommandBuffer.contains(0) { f.append("a sampled command buffer carried no encoder") }
        if kernels != pinnedEncodersPerContinuation || buffers != pinnedCommandBuffersPerContinuation {
            f.append("continuation is not the pinned " + String(pinnedEncodersPerContinuation) + " kernels / "
                     + String(pinnedCommandBuffersPerContinuation) + " command buffers")
        }
        var counts = [String: Int]()
        for k in s.kernels { counts[k, default: 0] += 1 }
        let cross = counterCrossCheck(kernelCounts: counts, encoders: s.kernels.count, counterDelta: counterDelta,
                                      counterKeys: counterKeys)
        f += cross
        let idx = s.kernels.map { kernelIndex[$0] ?? -1 }
        if idx.contains(-1) { f.append("sampled kernel label outside the declared table") }
        return KernelStageToken(
            continuation: continuation, encoders: s.kernels.count, kernelsDelta: kernels, commandBuffersDelta: buffers,
            encodersPerCommandBuffer: s.encodersPerCommandBuffer,
            commandBufferIntervalMicros: s.commandBufferIntervalMicros, kernelIndex: idx, analysis: s.analysis,
            counterCrossCheckFailures: cross, wallMicros: wall, failures: f)
    }

    static func makeFixture(_ fx: RouterOverlapGateFixture, reference: RouterOverlapGateQualifiedArm,
                            plain: RunOutcome?, sampled: RunOutcome?, names: [String],
                            families: [String: String]) -> KernelStageFixture {
        let proto = RouterOverlapGateProtocol.required
        var f = [String]()
        if !reference.passed { f.append("fresh direct reference failed: " + (reference.failure ?? "unknown")) }
        if let p = plain, !p.run.passed { f.append("plain OFF run failed: " + (p.run.failure ?? "unknown")) }
        if plain == nil { f.append("plain OFF run not executed") }
        var relation = [String]()
        var kernelStats = [KernelStageLabelStat]()
        var familyStats = [KernelStageLabelStat]()
        var overhead: Double? = nil
        if let q = sampled, let p = plain {
            if !q.run.passed { f.append("stage-sampled OFF run failed: " + (q.run.failure ?? "unknown")) }
            if q.bits.compared != proto.outputsPerRun || q.bits.missing != 0 || q.bits.mismatches != 0 {
                f.append("stage-sampled vs plain bit comparison: compared " + String(q.bits.compared) + ", missing "
                         + String(q.bits.missing) + ", mismatching logits " + String(q.bits.mismatches))
            }
            relation = RouterOverlapModelGate.totalsRelation(candidate: q.run.totals, baseline: p.run.totals,
                                                             extraBuffersPerForward: 0, promptCount: fx.prompt.count)
            f += relation
            let inputs: [KernelStageTokenInput] = q.tokens.compactMap { t -> KernelStageTokenInput? in
                guard let a = t.analysis, a.failures.isEmpty, t.kernelIndex.count == a.intervalTicks.count,
                      t.kernelIndex.allSatisfy({ $0 >= 0 && $0 < names.count }) else { return nil }
                return KernelStageTokenInput(labels: t.kernelIndex.map { names[$0] }, intervalTicks: a.intervalTicks,
                                             disjoint: a.disjoint)
            }
            kernelStats = aggregate(inputs, label: { $0 }, family: { families[$0] })
            familyStats = aggregate(inputs, label: { families[$0] ?? "untagged" }, family: { _ in nil })
            if let a = q.run.continuationWallMedianMicros, let b = p.run.continuationWallMedianMicros {
                overhead = a - b
            }
        } else {
            f.append("stage-sampled OFF run not executed")
        }
        return KernelStageFixture(
            fixture: fx.name, reference: reference, plain: plain?.run, stageSampled: sampled?.run,
            bitComparedSteps: sampled?.bits.compared ?? 0, bitMismatchCount: sampled?.bits.mismatches ?? 0,
            firstBitMismatch: sampled?.bits.first, sampledVsPlainTotalsMismatches: relation,
            tokens: sampled?.tokens ?? [], kernelStats: kernelStats, familyStats: familyStats,
            sampledMinusPlainWallMedianMicros: overhead, passed: f.isEmpty, failures: f)
    }
}
