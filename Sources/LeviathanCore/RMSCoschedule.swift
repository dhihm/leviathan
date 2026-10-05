import Foundation
import Metal

// Phase 63: default-off component diagnostic for co-scheduling strictly independent production RMSNorm instances.
// Both arms dispatch one pipeline compiled from the same RMSCooperative source and nil options as the runner for the same
// instances, inputs, bindings, thread counts and dynamic threadgroup memory, in the forward's instance order. Only the
// encoder dispatch type differs: serial (the production topology) or concurrent. Instances rewrite disjoint buffers in
// place, so the concurrent arm needs no barrier and changes no kernel, arithmetic or per-output order. Gates: Float bit
// identity of every element between arms, finiteness, canary guards and the FP64 contract; then raw stage-boundary
// ticks only. Component only: no runner, SSA, default or candidate change; no weights or outputs enter the report.

public struct RMSCoscheduleError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

/// One production RMSNorm call: `vectors` contiguous vectors of `n` floats normalized in place, scaled by an F32 norm
/// weight or scale-free (weightTensor nil).
public struct RMSCoscheduleInstance: Codable, Equatable, Sendable {
    public let label: String
    public let n: Int
    public let vectors: Int
    public let weightTensor: String?

    public init(label: String, n: Int, vectors: Int, weightTensor: String?) {
        self.label = label
        self.n = n
        self.vectors = vectors
        self.weightTensor = weightTensor
    }
}

/// A group of RMSNorm instances co-scheduled in one encoder, listed in the forward's order.
public struct RMSCoscheduleCase: Codable, Equatable, Sendable {
    public let name: String
    /// True only when no instance of the group consumes another's output in the forward data flow.
    public let modelIndependent: Bool
    public let dataFlow: String
    public let instances: [RMSCoscheduleInstance]

    public init(name: String, modelIndependent: Bool, dataFlow: String, instances: [RMSCoscheduleInstance]) {
        self.name = name
        self.modelIndependent = modelIndependent
        self.dataFlow = dataFlow
        self.instances = instances
    }
}

/// Launch and buffer plan for one instance against the actual pipeline and device limits.
public struct RMSCoschedulePlan: Codable, Equatable, Sendable {
    public let n: Int
    public let vectors: Int
    public let scaled: Bool
    public let threads: Int
    public let threadgroupMemoryLength: Int
    public let floats: Int
    public let spanBytes: Int
    public let bufferBytes: Int
}

/// One timed pair: raw encoder ticks of each arm and the concurrent/serial ratio, or the failed sample checks.
public struct RMSCoschedulePairTiming: Codable, Equatable, Sendable {
    public let pair: Int
    public let serialFirst: Bool
    public let serialTicks: UInt64?
    public let concurrentTicks: UInt64?
    public let concurrentOverSerial: Double?
    public let failures: [String]
}

public struct RMSCoscheduleEvidenceInput: Sendable {
    public let modelIndependent: Bool
    public let timingComplete: Bool
    public let medianSerialFirst: Double?
    public let medianConcurrentFirst: Double?

    public init(modelIndependent: Bool, timingComplete: Bool, medianSerialFirst: Double?,
                medianConcurrentFirst: Double?) {
        self.modelIndependent = modelIndependent
        self.timingComplete = timingComplete
        self.medianSerialFirst = medianSerialFirst
        self.medianConcurrentFirst = medianConcurrentFirst
    }
}

public struct RMSCoscheduleHardware: Codable, Equatable, Sendable {
    public let device: String
    public let pipelineThreadExecutionWidth: Int
    public let pipelineMaxTotalThreads: Int
    public let pipelineStaticThreadgroupBytes: Int
    public let deviceMaxThreadgroupMemoryLength: Int
}

public struct RMSCoscheduleCaseResult: Codable, Equatable, Sendable {
    public var name: String
    public var modelIndependent: Bool
    public var instances: [RMSCoscheduleInstance]
    public var plans: [RMSCoschedulePlan] = []
    public var declined: String? = nil
    public var serialFinite = false
    public var concurrentFinite = false
    public var serialFP64WorstRel: Double? = nil
    public var concurrentFP64WorstRel: Double? = nil
    public var fp64Passed = false
    public var bitMismatchElements: Int? = nil
    public var bitIdentical = false
    public var guardsIntact = false
    public var correctnessPassed = false
    public var timing: [RMSCoschedulePairTiming] = []
    public var medianSerialFirst: Double? = nil
    public var medianConcurrentFirst: Double? = nil
    public var timingComplete = false
    /// Warm and timed arms whose post-completion outputs and canaries were checked against the gated serial outputs.
    public var timedArmsChecked = 0
    /// First warm or timed arm that failed bit identity with the gated serial outputs or changed a canary; timing is
    /// withheld from that arm on.
    public var timedArmIntegrityFailure: String? = nil
    public var failures: [String] = []

    public init(_ c: RMSCoscheduleCase) {
        name = c.name
        modelIndependent = c.modelIndependent
        instances = c.instances
    }
}

public enum RMSCoscheduleStatus: String, Codable, Sendable {
    case passed
    case correctnessFailed
    case timingFailed
    case unsupported
    case commandBufferFailed
    case aborted
}

public struct RMSCoscheduleReport: Codable {
    public private(set) var kind = RMSCoschedule.kind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var notes = RMSCoschedule.notes
    public private(set) var timestampUnits = RMSCoschedule.timestampUnits
    public private(set) var evidenceRatioThreshold = RMSCoschedule.evidenceRatioThreshold
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var kernelName: String? = nil
    public var kernelSourceSha256: String? = nil
    public var hardware: RMSCoscheduleHardware? = nil
    public var support: KernelStageSupport? = nil
    public var compileError: String? = nil
    public var cases: [RMSCoscheduleCaseResult] = []
    public var integrationEvidenceThresholdMet = false
    public var commandBufferFailure: String? = nil
    public var abortReason: String? = nil
    public var terminal = true
    public var status: RMSCoscheduleStatus = .aborted

    public init() {}
}

extension RMSCoscheduleReport {
    /// Contract normalization at every encode: the five claims are always false.
    func withDiagnosticContract() -> RMSCoscheduleReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

public struct RMSCoscheduleOptions: Equatable, Sendable {
    public let gguf: String
    public let report: String
    public let modelSha256: String
    /// Opt-in diagnostic: perform the same setup after GO, write a setup-only report, never call run or commit.
    public let setupOnly: Bool

    public init(gguf: String, report: String, modelSha256: String, setupOnly: Bool = false) {
        self.gguf = gguf
        self.report = report
        self.modelSha256 = modelSha256
        self.setupOnly = setupOnly
    }
}

/// Setup-only diagnostic report: records which setup boundaries were reached and the compiled pipeline facts.
/// Nothing was encoded, committed, gated or timed; every correctness, performance, gain and install claim is false.
public struct RMSCoscheduleSetupReport: Codable {
    public private(set) var kind = RMSCoschedule.setupKind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var correctnessEvaluated = false
    public private(set) var timingEvaluated = false
    public private(set) var integrationEvidenceThresholdMet = false
    public private(set) var commandBuffersCommitted = 0
    public private(set) var notes = RMSCoschedule.setupNotes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var kernelName: String? = nil
    public var kernelSourceSha256: String? = nil
    public var hardware: RMSCoscheduleHardware? = nil
    public var compileError: String? = nil
    public var normTensorsPacked: Int? = nil
    public var packedWeightBytes: Int? = nil
    public var stagesReached: [String] = []
    public var setupError: String? = nil
    public var setupCompleted = false

    public init() {}
}

extension RMSCoscheduleSetupReport {
    func withDiagnosticContract() -> RMSCoscheduleSetupReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        copy.correctnessEvaluated = false
        copy.timingEvaluated = false
        copy.integrationEvidenceThresholdMet = false
        copy.commandBuffersCommitted = 0
        return copy
    }
}

public enum RMSCoschedule {
    public static let kind = "rms-independent-coschedule-component-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 16 << 20
    public static let correctnessContract = 1e-4
    /// Pre-declared: both per-order median concurrent/serial ratios of every model-independent case must be <= this.
    public static let evidenceRatioThreshold = 0.85
    public static let guardBytes = 256
    public static let canary: UInt8 = 0xA5
    public static let maxVectors = 64
    public static let warmupPairs = 2
    public static let timedPairs = 6
    public static let timeoutSeconds = 10.0
    public static let seedBase: UInt64 = 0x6A09E667F3BCC908
    public static let setupKind = "rms-independent-coschedule-setup-diagnostic-v1"
    public static let setupNotes: [String] = [
        "Setup-only diagnostic: the same setup as the component (GGUF virtual mapping and parse, config, tensor "
            + "contract, small norm packing, Metal device and queue, weight buffer, library compile, pipeline "
            + "creation) was performed after GO and then the process ended. No command buffer was encoded or "
            + "committed; no FP64, bit-identity, guard, timing or evidence gate ran. Reaching context-ready is not a "
            + "correctness or performance pass of anything.",
        "Model SHA-256 is the caller-supplied cached record; nothing is hashed, uploaded or downloaded; no weights, "
            + "inputs, outputs or credentials enter the report or the log lines.",
    ]
    public static let usage = "usage: rms-coschedule-component --gguf M.gguf --report NEW_REPORT.json "
        + "--model-sha256 CACHED_LOWERCASE_HEX64 [--setup-only]"
    public static let timestampUnits = "raw GPU timestamp ticks as resolved at stage boundaries; the installed SDK "
        + "declarations state no unit; nothing is converted; intervals are not busy time"

    public static let notes: [String] = [
        "Component only: fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim "
            + "are always false; no runner, SSA, kernel, default, guard, artifact or model change.",
        "Both arms dispatch one pipeline compiled from the same RMSCooperative source with the same nil options as the "
            + "runner's lazy compile, on the system default device, for "
            + "the same instances, inputs, bindings, thread counts and dynamic threadgroup memory in the forward's order; "
            + "only the encoder dispatch type differs (serial = production topology, concurrent = co-scheduled). "
            + "Instances rewrite disjoint buffers in place, so the concurrent arm needs no barrier.",
        "Groups come from Gemma4Runner.forwardLayer data flow. The SSA graph was not inspected in this phase and its "
            + "actual ordering is not asserted.",
        "Gates on every element: Float bit identity between arms, finiteness, 256-byte canary guards and the "
            + "vector-normalized 1e-4 FP64 contract for both arms. Timing runs only after every gate passed.",
        "Timing: 2 warmup + 6 timed pairs, two sequential command buffers per pair (serial first on even pairs), each "
            + "committed only after the previous was observed terminal (10 s), encoder samples 0/1 serial and 2/3 "
            + "concurrent, resolved twice after both are terminal. After every warm and timed arm is terminal, its "
            + "outputs must be Float-bit identical to the gated serial outputs (transitively finite and within the FP64 "
            + "contract) and its canaries intact, checked on the CPU outside the sampled encoder; any failure withholds "
            + "timing for the case. Raw ticks and per-order ratios only; no busy time and no model-level implication.",
        "Pre-declared rule: integrationEvidenceThresholdMet only if every model-independent case has complete timing "
            + "and both per-order median concurrent/serial ratios are <= 0.85. Meeting it authorizes only authoring a "
            + "reviewed integration proposal.",
        "Model SHA-256 is the caller-supplied cached explicit-refresh record value; nothing here hashes or downloads "
            + "the model. GGUFReader maps the whole file read-only (a virtual mapping; tensorData is a zero-copy slice; "
            + "no resident-memory claim); only the frozen cases' small F32 norm tensors are copied into one small "
            + "Metal buffer, for binding and the FP64 oracle; no runner, whole-weight Metal buffer, KV or activation "
            + "buffers are created; no weights, inputs or outputs enter the report.",
        "Setup and stage boundary lines are emitted on the CPU side only (setup stages, counter sampling setup, "
            + "untimed arm commit and terminal per case, timing begin per case); nothing is emitted per warm or "
            + "timed arm and nothing is added inside a sampled encoder.",
    ]

    public static let cases: [RMSCoscheduleCase] = [
        RMSCoscheduleCase(
            name: "local-qkv-norms-blk0", modelIndependent: true,
            dataFlow: "q_norm rewrites bufT1, k_norm bufT2 and v_norm bufKV in place; none reads another's buffer; "
                + "their producers (Q, K, V projections) can precede and their consumers (rope q, rope k, K/V append, "
                + "attention) can follow all three",
            instances: [
                RMSCoscheduleInstance(label: "q_norm", n: 256, vectors: 16, weightTensor: "blk.0.attn_q_norm.weight"),
                RMSCoscheduleInstance(label: "k_norm", n: 256, vectors: 8, weightTensor: "blk.0.attn_k_norm.weight"),
                RMSCoscheduleInstance(label: "v_norm", n: 256, vectors: 8, weightTensor: nil),
            ]),
        RMSCoscheduleCase(
            name: "global-qkv-norms-blk5", modelIndependent: true,
            dataFlow: "as the local trio; v is the raw K projection copied into bufKV before k_norm, so the copy "
                + "precedes the group",
            instances: [
                RMSCoscheduleInstance(label: "q_norm", n: 512, vectors: 16, weightTensor: "blk.5.attn_q_norm.weight"),
                RMSCoscheduleInstance(label: "k_norm", n: 512, vectors: 2, weightTensor: "blk.5.attn_k_norm.weight"),
                RMSCoscheduleInstance(label: "v_norm", n: 512, vectors: 2, weightTensor: nil),
            ]),
        RMSCoscheduleCase(
            name: "post-ffw-norm-pair-blk0", modelIndependent: true,
            dataFlow: "post_ffw_norm_1 rewrites bufShared and post_ffw_norm_2 rewrites bufT3 in place; the routed region "
                + "never reads or writes bufShared, and both outputs are first consumed by the same g4_add",
            instances: [
                RMSCoscheduleInstance(label: "post_ffw_norm_1", n: 2816, vectors: 1,
                                      weightTensor: "blk.0.post_ffw_norm_1.weight"),
                RMSCoscheduleInstance(label: "post_ffw_norm_2", n: 2816, vectors: 1,
                                      weightTensor: "blk.0.post_ffw_norm_2.weight"),
            ]),
    ]

    // MARK: Pure helpers

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func isHex64(_ s: String) -> Bool {
        s.utf8.count == 64 && s.utf8.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
    }

    public static func parseOptions(_ argv: [String]) throws -> RMSCoscheduleOptions {
        let keys = ["--gguf", "--report", "--model-sha256"]
        var v = [String: String]()
        var setupOnly = false
        var i = 0
        while i < argv.count {
            let key = argv[i]
            if key == "--setup-only" {
                guard !setupOnly else { throw RMSCoscheduleError(usage) }
                setupOnly = true
                i += 1
                continue
            }
            guard keys.contains(key), i + 1 < argv.count, v[key] == nil else { throw RMSCoscheduleError(usage) }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw RMSCoscheduleError(usage) }
            v[key] = value
            i += 2
        }
        guard let gguf = v["--gguf"], let report = v["--report"], let sha = v["--model-sha256"], isHex64(sha) else {
            throw RMSCoscheduleError(usage)
        }
        return RMSCoscheduleOptions(gguf: gguf, report: report, modelSha256: sha, setupOnly: setupOnly)
    }

    public static func encodeReport(_ r: RMSCoscheduleReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    public static func encodeSetupReport(_ r: RMSCoscheduleSetupReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    /// Builds the setup-only report from whatever setup produced; never dispatches, encodes or commits anything.
    public static func setupDiagnostic(context: RMSCoscheduleContext?, setupError: String?, stagesReached: [String],
                                       modelSha256: String, executableDigest: String) -> RMSCoscheduleSetupReport {
        var r = RMSCoscheduleSetupReport()
        r.modelSha256 = modelSha256
        r.executableDigest = executableDigest
        r.kernelName = RMSCooperative.kernelName
        r.kernelSourceSha256 = RMSCooperative.sourceSha256()
        r.stagesReached = stagesReached
        r.setupError = setupError
        if let c = context {
            r.normTensorsPacked = c.weightTable.count
            r.packedWeightBytes = c.weights.length
            switch c.pipeline {
            case .success(let p):
                r.hardware = RMSCoscheduleHardware(
                    device: c.device.name, pipelineThreadExecutionWidth: p.threadExecutionWidth,
                    pipelineMaxTotalThreads: p.maxTotalThreadsPerThreadgroup,
                    pipelineStaticThreadgroupBytes: p.staticThreadgroupMemoryLength,
                    deviceMaxThreadgroupMemoryLength: c.device.maxThreadgroupMemoryLength)
            case .failure(let error):
                r.compileError = String(describing: error)
            }
        }
        r.setupCompleted = context != nil && setupError == nil && r.compileError == nil
            && RMSCoscheduleSetup.stagesComplete(stagesReached)
        return r.withDiagnosticContract()
    }

    /// Fail-closed plan against the actual pipeline and device. The same eligibility, alignment and threadgroup-memory
    /// budget decisions as the production rmsnormCooperativeDispatch, plus exact weight geometry and checked spans.
    public static func plan(_ i: RMSCoscheduleInstance, executionWidth: Int, pipelineMaxThreads: Int,
                            staticThreadgroupMemory: Int, deviceMaxThreadgroupMemory: Int,
                            weightOffset: Int?, weightBytes: Int?) -> (plan: RMSCoschedulePlan?, decline: String?) {
        func no(_ reason: String) -> (plan: RMSCoschedulePlan?, decline: String?) { (nil, i.label + ": " + reason) }
        guard RMSCooperative.eligible(n: i.n) else { return no("n " + String(i.n) + " not cooperative-eligible") }
        guard i.vectors >= 1, i.vectors <= maxVectors else { return no("vectors outside 1..." + String(maxVectors)) }
        let scaled = i.weightTensor != nil
        if scaled {
            guard let off = weightOffset, let bytes = weightBytes, off >= 0, bytes == i.n * 4 else {
                return no("weight is not an F32 tensor of n floats")
            }
            guard Gemma4Runner.rmsVectorLoadGuards(n: i.n, scaled: true, wOff: UInt64(off)) else {
                return no("weight offset not 16-byte aligned")
            }
        } else {
            guard weightOffset == nil, weightBytes == nil else { return no("scale-free instance given a weight") }
            guard Gemma4Runner.rmsVectorLoadGuards(n: i.n, scaled: false, wOff: 0) else { return no("width guard") }
        }
        let threads = RMSCooperative.threadCount(n: i.n, executionWidth: executionWidth, maxThreads: pipelineMaxThreads)
        let tgLen = RMSCooperative.threadgroupMemoryLength(n: i.n)
        guard threads > 0, threads <= pipelineMaxThreads, tgLen > 0, staticThreadgroupMemory >= 0,
              staticThreadgroupMemory <= deviceMaxThreadgroupMemory,
              tgLen <= deviceMaxThreadgroupMemory - staticThreadgroupMemory else {
            return no("thread count or threadgroup memory budget")
        }
        let (floats, o1) = i.n.multipliedReportingOverflow(by: i.vectors)
        let (span, o2) = o1 ? (0, true) : floats.multipliedReportingOverflow(by: 4)
        let (total, o3) = o2 ? (0, true) : span.addingReportingOverflow(2 * guardBytes)
        guard !o1, !o2, !o3, floats <= Int(UInt32.max) else { return no("span overflow") }
        return (RMSCoschedulePlan(n: i.n, vectors: i.vectors, scaled: scaled, threads: threads,
                                  threadgroupMemoryLength: tgLen, floats: floats, spanBytes: span,
                                  bufferBytes: total), nil)
    }

    public static func seed(caseIndex: Int, instance: Int) -> UInt64 {
        seedBase &+ UInt64(truncatingIfNeeded: caseIndex &* 16 &+ instance)
    }

    /// Frozen splitmix64 inputs: 23-bit integers scaled by 2^-21, exact in Float, in [-2, 2).
    public static func inputs(seed: UInt64, count: Int) -> [Float] {
        var s = seed
        var out = [Float]()
        out.reserveCapacity(max(0, count))
        for _ in 0..<max(0, count) {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            let k = Int(z >> 41) - (1 << 22)
            out.append(Float(k) / Float(1 << 21))
        }
        return out
    }

    /// Independent FP64 RMSNorm oracle: inv = 1 / sqrt(sum(x^2) / n + eps), y = x * inv * w.
    public static func oracle(x: [Float], weight: [Float]?, n: Int, vectors: Int, eps: Float) -> [Double]? {
        guard n > 0, vectors > 0, x.count == n * vectors else { return nil }
        if let w = weight, w.count != n { return nil }
        var y = [Double](repeating: 0, count: x.count)
        for v in 0..<vectors {
            var ss = 0.0
            for i in 0..<n {
                let d = Double(x[v * n + i])
                ss += d * d
            }
            let inv = 1.0 / (ss / Double(n) + Double(eps)).squareRoot()
            for i in 0..<n {
                let w = weight.map { Double($0[i]) } ?? 1.0
                y[v * n + i] = Double(x[v * n + i]) * inv * w
            }
        }
        return y
    }

    /// Worst per-vector max |got - want| / max |want|; infinity on a non-finite element; nil on bad geometry.
    public static func worstVectorRel(got: [Float], want: [Double], n: Int) -> Double? {
        guard n > 0, !got.isEmpty, got.count == want.count, got.count % n == 0 else { return nil }
        var worst = 0.0
        for v in 0..<(got.count / n) {
            var maxWant = 0.0
            var maxDiff = 0.0
            for i in (v * n)..<((v + 1) * n) {
                guard got[i].isFinite, want[i].isFinite else { return .infinity }
                maxWant = max(maxWant, abs(want[i]))
                maxDiff = max(maxDiff, abs(Double(got[i]) - want[i]))
            }
            worst = max(worst, maxDiff / max(maxWant, 1e-30))
        }
        return worst
    }

    /// Elements whose Float bit patterns differ; nil for empty or unequal-length inputs.
    public static func bitMismatches(_ a: [Float], _ b: [Float]) -> Int? {
        guard !a.isEmpty, a.count == b.count else { return nil }
        var m = 0
        for i in a.indices where a[i].bitPattern != b[i].bitPattern { m += 1 }
        return m
    }

    /// Checks one pair's resolved samples [serialStart, serialEnd, concurrentStart, concurrentEnd].
    public static func analyzePair(raw: [UInt64], pair: Int, serialFirst: Bool) -> RMSCoschedulePairTiming {
        var f = [String]()
        if raw.count != 4 {
            f.append("resolved " + String(raw.count) + " samples, expected 4")
        } else {
            if raw.contains(KernelStageAttribution.counterErrorValue) { f.append("counter error value") }
            if raw.contains(0) { f.append("zero sample") }
            if raw[1] <= raw[0] || raw[3] <= raw[2] { f.append("encoder end not after start") }
            if f.isEmpty, !(serialFirst ? raw[1] <= raw[2] : raw[3] <= raw[0]) {
                f.append("second command buffer started before the first ended")
            }
        }
        guard f.isEmpty else {
            return RMSCoschedulePairTiming(pair: pair, serialFirst: serialFirst, serialTicks: nil, concurrentTicks: nil,
                                           concurrentOverSerial: nil, failures: f)
        }
        let s = raw[1] - raw[0]
        let c = raw[3] - raw[2]
        return RMSCoschedulePairTiming(pair: pair, serialFirst: serialFirst, serialTicks: s, concurrentTicks: c,
                                       concurrentOverSerial: Double(c) / Double(s), failures: [])
    }

    public static func evidenceThresholdMet(_ inputs: [RMSCoscheduleEvidenceInput]) -> Bool {
        let model = inputs.filter { $0.modelIndependent }
        guard !model.isEmpty else { return false }
        return model.allSatisfy { i in
            guard i.timingComplete, let a = i.medianSerialFirst, let b = i.medianConcurrentFirst else { return false }
            return a <= evidenceRatioThreshold && b <= evidenceRatioThreshold
        }
    }

    public static func finalStatus(_ r: RMSCoscheduleReport) -> RMSCoscheduleStatus {
        if r.commandBufferFailure != nil { return .commandBufferFailed }
        if r.abortReason != nil { return .aborted }
        if r.compileError != nil { return .unsupported }
        if r.cases.map({ $0.name }) != cases.map({ $0.name }) { return .aborted }
        if !r.cases.allSatisfy({ $0.correctnessPassed }) { return .correctnessFailed }
        guard let s = r.support, s.supported else { return .unsupported }
        if !r.cases.allSatisfy({ $0.timingComplete }) { return .timingFailed }
        return .passed
    }

    public static func exitCode(status: RMSCoscheduleStatus) -> Int32 {
        switch status {
        case .passed: return 0
        case .timingFailed: return 6
        case .correctnessFailed: return 3
        case .unsupported: return 2
        case .commandBufferFailed: return 4
        case .aborted: return 1
        }
    }

    public static func exitCode(_ r: RMSCoscheduleReport) -> Int32 { r.terminal ? exitCode(status: r.status) : 5 }

    // MARK: GPU arms

    struct Slot {
        let plan: RMSCoschedulePlan
        let buffer: MTLBuffer
        let wOff: UInt64
        let input: [Float]
        let weight: [Float]?
    }

    static func reset(_ slots: [Slot]) {
        for s in slots {
            memset(s.buffer.contents(), Int32(canary), s.plan.bufferBytes)
            s.input.withUnsafeBytes { raw in
                if let base = raw.baseAddress { memcpy(s.buffer.contents() + guardBytes, base, raw.count) }
            }
        }
    }

    static func guardsIntact(_ s: Slot) -> Bool {
        let p = s.buffer.contents()
        for i in 0..<guardBytes where p.load(fromByteOffset: i, as: UInt8.self) != canary { return false }
        for i in (guardBytes + s.plan.spanBytes)..<s.plan.bufferBytes
        where p.load(fromByteOffset: i, as: UInt8.self) != canary { return false }
        return true
    }

    static func output(_ s: Slot) -> [Float] {
        Array(UnsafeBufferPointer(start: (s.buffer.contents() + guardBytes).assumingMemoryBound(to: Float.self),
                                  count: s.plan.floats))
    }

    /// One encoder for the whole group: the production RMS ABI 0...6 plus dynamic threadgroup memory at index 0, one
    /// threadgroup per vector, instances in the forward's order. Only the dispatch type differs between arms.
    static func encode(_ cmd: MTLCommandBuffer, concurrent: Bool, pipeline: MTLComputePipelineState,
                       weights: MTLBuffer, eps: Float, slots: [Slot],
                       sampleBuffer: MTLCounterSampleBuffer?) throws {
        let pass = MTLComputePassDescriptor()
        pass.dispatchType = concurrent ? .concurrent : .serial
        if let sb = sampleBuffer {
            // The installed SDK declares the attachment subscript as optional; fail closed, never force-unwrap.
            guard let a = pass.sampleBufferAttachments[0] else {
                throw RMSCoscheduleError("sample buffer attachment 0 unavailable")
            }
            a.sampleBuffer = sb
            a.startOfEncoderSampleIndex = concurrent ? 2 : 0
            a.endOfEncoderSampleIndex = concurrent ? 3 : 1
        }
        guard let enc = cmd.makeComputeCommandEncoder(descriptor: pass) else {
            throw RMSCoscheduleError("compute encoder could not be created")
        }
        for s in slots {
            var dims = SIMD2<UInt32>(UInt32(s.plan.n), UInt32(s.plan.vectors))
            var e = eps
            var withScale: Int32 = s.plan.scaled ? 1 : 0
            var off = s.wOff
            enc.setComputePipelineState(pipeline)
            enc.setThreadgroupMemoryLength(s.plan.threadgroupMemoryLength, index: 0)
            enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&e, length: MemoryLayout<Float>.stride, index: 1)
            enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 3)
            enc.setBuffer(weights, offset: 0, index: 4)
            enc.setBuffer(s.buffer, offset: guardBytes, index: 5)
            enc.setBuffer(s.buffer, offset: guardBytes, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: s.plan.vectors, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: s.plan.threads, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// Commits and waits on the completion handler only (bounded); no waitUntilCompleted.
    static func submit(_ cmd: MTLCommandBuffer) -> (terminal: Bool, failure: String?) {
        let done = DispatchSemaphore(value: 0)
        cmd.addCompletedHandler { _ in done.signal() }
        cmd.commit()
        guard done.wait(timeout: .now() + timeoutSeconds) == .success else {
            return (false, "command buffer not terminal within " + String(timeoutSeconds) + " s")
        }
        guard cmd.status == .completed, cmd.error == nil else {
            return (true, "command buffer status " + String(cmd.status.rawValue) + ": "
                    + String(describing: cmd.error))
        }
        return (true, nil)
    }

    static func resolve(_ sb: MTLCounterSampleBuffer) -> (raw: [UInt64]?, failure: String?) {
        do {
            guard let a = try sb.resolveCounterRange(0..<4), let b = try sb.resolveCounterRange(0..<4) else {
                return (nil, "counter resolve returned no data")
            }
            guard a.count == 32 else { return (nil, "resolved " + String(a.count) + " bytes, expected 32") }
            guard a == b else { return (nil, "repeated resolve of the same range differs") }
            var raw = [UInt64]()
            a.withUnsafeBytes { bytes in
                for i in 0..<4 { raw.append(bytes.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self)) }
            }
            return (raw, nil)
        } catch {
            return (nil, "counter resolve threw: " + String(describing: error))
        }
    }

    // MARK: Entry point

    /// Never throws; every failure is recorded. terminal is false only when a command buffer did not become terminal.
    public static func run(context: RMSCoscheduleContext, modelSha256: String, executableDigest: String,
                           emit: (String) -> Void) -> RMSCoscheduleReport {
        var report = RMSCoscheduleReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.kernelName = RMSCooperative.kernelName
        report.kernelSourceSha256 = RMSCooperative.sourceSha256()
        execute(&report, context: context, emit: emit)
        report.integrationEvidenceThresholdMet = evidenceThresholdMet(report.cases.map {
            RMSCoscheduleEvidenceInput(modelIndependent: $0.modelIndependent, timingComplete: $0.timingComplete,
                                       medianSerialFirst: $0.medianSerialFirst,
                                       medianConcurrentFirst: $0.medianConcurrentFirst)
        })
        report.status = finalStatus(report)
        report = report.withDiagnosticContract()
        emit("status " + report.status.rawValue + "; terminal " + String(report.terminal)
             + "; evidence threshold met " + String(report.integrationEvidenceThresholdMet)
             + "; raw stage ticks only, not busy time; no gain or production claim")
        return report
    }

    static func describe(_ d: Double?) -> String { d.map { String($0) } ?? "none" }

    static func execute(_ report: inout RMSCoscheduleReport, context: RMSCoscheduleContext, emit: (String) -> Void) {
        guard isHex64(report.modelSha256 ?? ""), isHex64(report.executableDigest ?? "") else {
            report.abortReason = "model and executable digests must be lowercase 64-hex"
            return
        }
        let weights = context.weights
        let device = context.device
        let queue = context.queue
        let pipeline: MTLComputePipelineState
        switch context.pipeline {
        case .success(let p):
            pipeline = p
        case .failure(let error):
            report.compileError = String(describing: error)
            emit("cooperative RMS pipeline unavailable; nothing dispatched")
            return
        }
        report.hardware = RMSCoscheduleHardware(
            device: device.name, pipelineThreadExecutionWidth: pipeline.threadExecutionWidth,
            pipelineMaxTotalThreads: pipeline.maxTotalThreadsPerThreadgroup,
            pipelineStaticThreadgroupBytes: pipeline.staticThreadgroupMemoryLength,
            deviceMaxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength)
        emit("stage: counter sampling setup begin")
        let made = KernelStageAttribution.makeProfilerState(device: device, capacity: 4, families: [:])
        report.support = made.support
        emit("stage: counter sampling setup done; supported " + String(made.support.supported))
        let eps = context.eps
        for (ci, c) in cases.enumerated() {
            var result = RMSCoscheduleCaseResult(c)
            var slots = [Slot]()
            for (ii, inst) in c.instances.enumerated() {
                var offset: Int? = nil
                var bytes: Int? = nil
                if let name = inst.weightTensor {
                    guard let r = context.weightTable[name], r.bytes == inst.n * 4 else {
                        result.declined = inst.label + ": weight tensor " + name + " missing or not F32 of n floats"
                        break
                    }
                    offset = r.offset
                    bytes = r.bytes
                }
                let planned = plan(inst, executionWidth: pipeline.threadExecutionWidth,
                                   pipelineMaxThreads: pipeline.maxTotalThreadsPerThreadgroup,
                                   staticThreadgroupMemory: pipeline.staticThreadgroupMemoryLength,
                                   deviceMaxThreadgroupMemory: device.maxThreadgroupMemoryLength,
                                   weightOffset: offset, weightBytes: bytes)
                guard let p = planned.plan else {
                    result.declined = planned.decline ?? (inst.label + ": declined")
                    break
                }
                var weight: [Float]? = nil
                if let off = offset {
                    let (end, overflow) = off.addingReportingOverflow(p.n * 4)
                    guard !overflow, end <= weights.length else {
                        result.declined = inst.label + ": weight span beyond the resident mapping"
                        break
                    }
                    let w = Array(UnsafeBufferPointer(
                        start: (weights.contents() + off).assumingMemoryBound(to: Float.self), count: p.n))
                    guard w.allSatisfy({ $0.isFinite }) else {
                        result.declined = inst.label + ": non-finite norm weight"
                        break
                    }
                    weight = w
                }
                guard let buffer = device.makeBuffer(length: p.bufferBytes, options: .storageModeShared) else {
                    result.declined = inst.label + ": buffer allocation failed"
                    break
                }
                slots.append(Slot(plan: p, buffer: buffer, wOff: UInt64(offset ?? 0),
                                  input: inputs(seed: seed(caseIndex: ci, instance: ii), count: p.floats),
                                  weight: weight))
                result.plans.append(p)
            }
            if let d = result.declined {
                result.failures.append("declined: " + d)
                report.cases.append(result)
                emit("case " + c.name + ": declined: " + d)
                continue
            }
            var aliasFree = slots.allSatisfy { $0.buffer !== weights }
            for i in slots.indices {
                for j in slots.indices where j != i && slots[i].buffer === slots[j].buffer { aliasFree = false }
            }
            guard aliasFree else {
                result.failures.append("declined: instance buffers alias each other or the weight buffer")
                report.cases.append(result)
                emit("case " + c.name + ": declined: buffer aliasing")
                continue
            }
            var outputs = [[[Float]]]()
            var guardsOK = true
            emit("stage: case " + c.name + ": buffers ready; untimed arms begin")
            for concurrent in [false, true] {
                reset(slots)
                guard let cmd = queue.makeCommandBuffer() else {
                    report.abortReason = "command buffer unavailable"
                    report.cases.append(result)
                    return
                }
                do {
                    try encode(cmd, concurrent: concurrent, pipeline: pipeline, weights: weights, eps: eps,
                               slots: slots, sampleBuffer: nil)
                } catch {
                    report.abortReason = "encode failed: " + String(describing: error)
                    report.cases.append(result)
                    return
                }
                emit("stage: case " + c.name + ": untimed " + (concurrent ? "concurrent" : "serial") + " arm commit")
                let s = submit(cmd)
                emit("stage: case " + c.name + ": untimed " + (concurrent ? "concurrent" : "serial")
                     + " arm terminal " + String(s.terminal))
                if let f = s.failure {
                    report.commandBufferFailure = c.name + ": " + f
                    report.terminal = s.terminal
                    report.cases.append(result)
                    return
                }
                guardsOK = guardsOK && slots.allSatisfy { guardsIntact($0) }
                outputs.append(slots.map { output($0) })
            }
            let serialOut = outputs[0].flatMap { $0 }
            let concurrentOut = outputs[1].flatMap { $0 }
            result.serialFinite = !serialOut.isEmpty && serialOut.allSatisfy { $0.isFinite }
            result.concurrentFinite = !concurrentOut.isEmpty && concurrentOut.allSatisfy { $0.isFinite }
            var worstSerial = 0.0
            var worstConcurrent = 0.0
            var oracleOK = true
            for (k, s) in slots.enumerated() {
                guard let want = oracle(x: s.input, weight: s.weight, n: s.plan.n, vectors: s.plan.vectors, eps: eps),
                      let a = worstVectorRel(got: outputs[0][k], want: want, n: s.plan.n),
                      let b = worstVectorRel(got: outputs[1][k], want: want, n: s.plan.n) else {
                    oracleOK = false
                    continue
                }
                worstSerial = max(worstSerial, a)
                worstConcurrent = max(worstConcurrent, b)
            }
            result.serialFP64WorstRel = oracleOK && worstSerial.isFinite ? worstSerial : nil
            result.concurrentFP64WorstRel = oracleOK && worstConcurrent.isFinite ? worstConcurrent : nil
            result.fp64Passed = oracleOK && worstSerial < correctnessContract && worstConcurrent < correctnessContract
            result.bitMismatchElements = bitMismatches(serialOut, concurrentOut)
            result.bitIdentical = result.bitMismatchElements == 0
            result.guardsIntact = guardsOK
            if !result.serialFinite || !result.concurrentFinite { result.failures.append("non-finite output") }
            if !result.fp64Passed { result.failures.append("FP64 vector-normalized 1e-4 contract not met by both arms") }
            if !result.bitIdentical {
                result.failures.append(String(result.bitMismatchElements ?? -1)
                                       + " element(s) differ in Float bits between arms")
            }
            if !guardsOK { result.failures.append("canary guard bytes changed") }
            result.correctnessPassed = result.failures.isEmpty
            emit("case " + c.name + ": correctness " + String(result.correctnessPassed))
            guard result.correctnessPassed else {
                report.cases.append(result)
                continue
            }
            guard made.support.supported, let sampleBuffer = made.state?.sampleBuffer else {
                report.cases.append(result)
                emit("case " + c.name + ": stage sampling unsupported; timing not run")
                continue
            }
            var timed = [RMSCoschedulePairTiming]()
            var warmupOK = true
            var lastRaw: [UInt64]? = nil
            var armIntegrityOK = true
            emit("stage: case " + c.name + ": timing begin; " + String(warmupPairs + timedPairs)
                 + " pairs; no per-arm lines")
            for k in 0..<(warmupPairs + timedPairs) {
                let serialFirst = k % 2 == 0
                for concurrent in (serialFirst ? [false, true] : [true, false]) {
                    reset(slots)
                    guard let cmd = queue.makeCommandBuffer() else {
                        report.abortReason = "command buffer unavailable"
                        report.cases.append(result)
                        return
                    }
                    do {
                        try encode(cmd, concurrent: concurrent, pipeline: pipeline, weights: weights, eps: eps,
                                   slots: slots, sampleBuffer: sampleBuffer)
                    } catch {
                        report.abortReason = "encode failed: " + String(describing: error)
                        report.cases.append(result)
                        return
                    }
                    let s = submit(cmd)
                    if let f = s.failure {
                        report.commandBufferFailure = c.name + ": " + f
                        report.terminal = s.terminal
                        report.cases.append(result)
                        return
                    }
                    // Post-completion integrity of this warm/timed arm against the already-gated frozen serial outputs:
                    // intact canaries and Float bit identity on every element (transitively finite and within the FP64
                    // contract, since the serial outputs passed those gates). CPU-side only, after the command buffer is
                    // terminal; nothing is added inside the sampled encoder.
                    var armFailure: String? = nil
                    for (idx, slot) in slots.enumerated() {
                        guard guardsIntact(slot) else {
                            armFailure = slot.plan.scaled ? "canary guard bytes changed" : "canary guard bytes changed"
                            break
                        }
                        let mismatches = bitMismatches(output(slot), outputs[0][idx])
                        guard mismatches == 0 else {
                            armFailure = String(mismatches ?? -1) + " element(s) of instance " + String(idx)
                                + " differ in Float bits from the gated serial outputs"
                            break
                        }
                    }
                    result.timedArmsChecked += 1
                    if let af = armFailure {
                        result.timedArmIntegrityFailure = "pair " + String(k) + (concurrent ? " concurrent" : " serial")
                            + " arm: " + af
                        result.failures.append("timing withheld: " + result.timedArmIntegrityFailure!)
                        armIntegrityOK = false
                        break
                    }
                }
                if !armIntegrityOK { break }
                let resolved = resolve(sampleBuffer)
                let pair: RMSCoschedulePairTiming
                if let raw = resolved.raw, let prior = lastRaw, raw.count == 4, prior.count == 4,
                   raw.min()! <= prior.max()! {
                    lastRaw = raw
                    pair = RMSCoschedulePairTiming(pair: k - warmupPairs, serialFirst: serialFirst, serialTicks: nil,
                                                   concurrentTicks: nil, concurrentOverSerial: nil,
                                                   failures: ["samples not strictly after the previous pair (stale)"])
                } else if let raw = resolved.raw {
                    lastRaw = raw
                    pair = analyzePair(raw: raw, pair: k - warmupPairs, serialFirst: serialFirst)
                } else {
                    pair = RMSCoschedulePairTiming(pair: k - warmupPairs, serialFirst: serialFirst, serialTicks: nil,
                                                   concurrentTicks: nil, concurrentOverSerial: nil,
                                                   failures: [resolved.failure ?? "counter resolve failed"])
                }
                if k < warmupPairs {
                    if !pair.failures.isEmpty { warmupOK = false }
                } else {
                    timed.append(pair)
                }
            }
            guard armIntegrityOK else {
                // Fail closed: no ticks, no medians, timingComplete stays false; the evidence rule cannot be met.
                result.timingComplete = false
                report.cases.append(result)
                emit("case " + c.name + ": timing withheld: "
                     + (result.timedArmIntegrityFailure ?? "arm integrity failure"))
                continue
            }
            result.timing = timed
            result.medianSerialFirst = RouterOverlapModelGate.median(
                timed.filter { $0.serialFirst }.compactMap { $0.concurrentOverSerial })
            result.medianConcurrentFirst = RouterOverlapModelGate.median(
                timed.filter { !$0.serialFirst }.compactMap { $0.concurrentOverSerial })
            result.timingComplete = warmupOK && timed.count == timedPairs && timed.allSatisfy { $0.failures.isEmpty }
            if !warmupOK { result.failures.append("a warmup pair failed its sample checks") }
            if warmupOK && !result.timingComplete { result.failures.append("a timed pair failed its sample checks") }
            report.cases.append(result)
            emit("case " + c.name + ": timing complete " + String(result.timingComplete)
                 + "; median concurrent/serial serial-first " + describe(result.medianSerialFirst)
                 + ", concurrent-first " + describe(result.medianConcurrentFirst) + " (raw tick ratios)")
        }
    }
}
