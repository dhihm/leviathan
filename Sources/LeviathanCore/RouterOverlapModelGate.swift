// Phase 53 router overlap MODEL gate: bounded full-model correctness
// qualification plus a NORMAL-path matched performance harness for the
// default-off router overlap diagnostic runtime. Diagnostic only: no
// artifact, no winner, no activation, no promotion; gainQualified is always
// false (a numerical or performance result is not resource eligibility).
//
// Every arm runs on ONE runner and ONE executable inside
// Gemma4Runner.withQ4DiagnosticReferenceWidth(32):
// - reference: fresh direct serial Gemma4Runner.forward, default axes,
//   routed lowering uninstalled (install nil), width 32;
// - OFF (baseline): installed v1 E4R32 (installMoeBaselineForVerification),
//   GemmaPreparedExecution.prepare, the declared full modern candidate (par64
//   included), normal executeGraph / executeGraphContinue, no scope;
// - ON (candidate): same install, width and candidate,
//   prepareRouterOverlapDiagnostic, one withRouterOverlapDiagnosticScope
//   around each complete 32-output run.
// CPU FP64 routing, kernels, MSL, plans, schema and hash policy are untouched;
// this file only calls existing APIs.
//
// Qualification per fixture (en-chat, kr-chat, raw-hello): reference, then
// OFF, then ON. Every output vector must be 262144 finite floats, all 32 IDs
// must equal the frozen IDs, the normalized max diff
// max|delta| / max(1, max|reference|) must be strictly < 1e-4 against the
// same fresh reference, and ON must equal OFF in Float.bitPattern at every
// logit of every output (signed zero included). Counter deltas are checked
// EXACTLY per phase (prefill = prompt forwards, continuation = 31 forwards)
// against counts derived from the prepared SSA, statistics plan, overlap
// plan and config. Kernel totals ON == OFF and command buffers
// ON == OFF + layers per forward are the dispatch-multiset proxy.
//
// Performance (only after every qualification passed): per fixture one
// declared warmup per arm (OFF then ON), then seven fixed blocks (even AB,
// odd BA). Timers bracket ONLY executeGraph(prompt) and each
// executeGraphContinue(previous argmax); scope entry, install/binding
// checks, counter snapshots and finiteness/ID checks run between timers.

import Foundation

public struct RouterOverlapGateError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public enum RouterOverlapModelGateMode: String, Codable, Sendable {
    case qualificationOnly = "qualification-only"
    case performance
}

/// reference = fresh direct serial runner; off = baseline (control
/// preparation, no scope); on = candidate (diagnostic preparation in scope).
public enum RouterOverlapGateArm: String, Codable, Sendable {
    case reference = "reference-direct-serial-installnil-q4w32"
    case off = "off-control-prepared-modern-v1E4R32-q4w32"
    case on = "on-router-overlap-scope-modern-v1E4R32-q4w32"
}

public enum RouterOverlapGateStatus: String, Codable, Sendable {
    case qualificationPassed
    case qualificationFailed
    case performanceRunCheckFailed
    case performanceUnstable
    case performanceNoGain
    case performanceInternalCriteriaMetExternalEligibilityPending
    case runnerNotRestored
    case aborted
}

// MARK: - Pinned fixtures

public struct RouterOverlapGateFixture: Codable, Equatable, Sendable {
    public let name: String
    public let prompt: [Int]
    public let frozenIds: [Int]

    public init(name: String, prompt: [Int], frozenIds: [Int]) {
        self.name = name
        self.prompt = prompt
        self.frozenIds = frozenIds
    }
}

public enum RouterOverlapGateFixtures {
    public static let order = ["en-chat", "kr-chat", "raw-hello"]

    public static let pinned: [RouterOverlapGateFixture] = [
        RouterOverlapGateFixture(
            name: "en-chat",
            prompt: [2, 105, 2364, 107, 9259, 236764, 740, 611, 1601, 786, 3124, 236881, 106,
                     107, 105, 4368, 107, 100, 45518, 107, 101],
            frozenIds: [9259, 236888, 564, 1093, 577, 5293, 531, 1601, 611, 236761, 236743, 108,
                        3689, 563, 580, 822, 3666, 236881, 1599, 740, 2679, 786, 496, 2934,
                        236764, 2679, 573, 1601, 607, 496, 4209, 568]),
        RouterOverlapGateFixture(
            name: "kr-chat",
            prompt: [2, 105, 2364, 107, 61659, 236764, 31694, 57517, 240482, 237170, 51955,
                     237272, 237586, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101],
            frozenIds: [61659, 236888, 113648, 22478, 63949, 9554, 237774, 61487, 86595, 237223,
                        94070, 94106, 54846, 5386, 50860, 18229, 77094, 236764, 75115, 25121,
                        238144, 71160, 237281, 143474, 237384, 57517, 240482, 237482, 66609,
                        88440, 190710, 74221]),
        RouterOverlapGateFixture(
            name: "raw-hello",
            prompt: [2, 9259],
            frozenIds: [3324, 236772, 1437, 236772, 236751, 2964, 236772, 1340, 236772, 2289,
                        236772, 1340, 236772, 1437, 236772, 236751, 2964, 236772, 1340, 236772,
                        236751, 2964, 236772, 1340, 236772, 1437, 236772, 236751, 2964, 236772,
                        1340, 236772]),
    ]

    private struct Entry: Decodable {
        let prompt: [Int]
        let frozenIds: [Int]
    }

    /// Parses the supplied fixture JSON and requires it to equal the pinned
    /// fixtures exactly; returned in the fixed order en-chat, kr-chat, raw-hello.
    public static func parse(_ data: Data) throws -> [RouterOverlapGateFixture] {
        guard data.count <= 1 << 16 else {
            throw RouterOverlapGateError("fixture JSON exceeds 64 KiB")
        }
        let decoded: [String: Entry]
        do { decoded = try JSONDecoder().decode([String: Entry].self, from: data) }
        catch { throw RouterOverlapGateError("fixture JSON decode: " + String(describing: error)) }
        guard decoded.count == order.count, Set(decoded.keys) == Set(order) else {
            throw RouterOverlapGateError("fixture names must be exactly " + order.joined(separator: ", "))
        }
        var list = [RouterOverlapGateFixture]()
        for name in order {
            guard let e = decoded[name] else { throw RouterOverlapGateError("fixture " + name + " missing") }
            list.append(RouterOverlapGateFixture(name: name, prompt: e.prompt, frozenIds: e.frozenIds))
        }
        guard list == pinned else {
            throw RouterOverlapGateError("fixture prompts/frozenIds differ from the pinned fixtures")
        }
        return list
    }
}

// MARK: - Fixed protocol

public struct RouterOverlapGateProtocol: Codable, Equatable, Sendable {
    public let fixtures: [String]
    public let outputsPerRun: Int
    public let continuationsPerRun: Int
    public let q4LaunchWidth: Int
    public let moeBatch: Int
    public let moeRowsPerGroup: Int
    public let normalizedContract: Double
    public let normalization: String
    public let warmupsPerArmPerFixture: Int
    public let blocksPerFixture: Int
    public let blockOrders: [[RouterOverlapGateArm]]
    public let spreadLimit: Double
    public let gainThreshold: Double
    public let statistic: String
    public let outlierPolicy: String

    public static let required = RouterOverlapGateProtocol(
        fixtures: RouterOverlapGateFixtures.order, outputsPerRun: 32, continuationsPerRun: 31,
        q4LaunchWidth: 32, moeBatch: 4, moeRowsPerGroup: 32, normalizedContract: 1e-4,
        normalization: "per output: max_i |logit_i - reference_i| / max(1, max_i |reference_i|), "
            + "strictly < 1e-4 against the same fresh direct reference",
        warmupsPerArmPerFixture: 1, blocksPerFixture: 7,
        blockOrders: (0..<7).map { RouterOverlapModelGate.blockOrder($0) },
        spreadLimit: 0.05, gainThreshold: 0.02,
        statistic: "run TPOT = median of its 31 continuation walls; arm TPOT = median of its 7 "
            + "run TPOTs; spread = (max - min) / median per arm per fixture; gain = "
            + "(baseline - candidate) / baseline per fixture",
        outlierPolicy: "none: every raw latency is reported and used; no retry, rescue, "
            + "adaptive stop or exclusion")
}

// MARK: - Counters

/// Named runtime counters; deltas between snapshots are compared by name.
public struct RouterOverlapGateCounters: Codable, Equatable, Sendable {
    public var values: [String: Int]

    public static let keys: [String] = [
        "kernels", "buffers", "q4Generic", "q4Applied", "q4Fallback", "q4Generated",
        "moeGu", "moeAct", "moeDn", "moeMerges", "moeGenerated", "moeResidentFallback",
        "parallelAttentionApplied", "parallelAttentionFallback",
        "rmsCooperativeApplied", "rmsCooperativeFallback", "rmsGenerated",
        "q6CooperativeApplied", "q6CooperativeFallback",
        "rmsStatsComputed", "rmsStatsReused", "rmsStatsFallback",
        "routerScaleApplied", "routerScaleFallback", "residentZeroApplied",
        "residentZeroFallback", "routerLaunchApplied", "routerLaunchFallback",
        "overlapApplied", "overlapFallback", "overlapAsyncCommitted", "overlapDrained",
        "overlapFailedDrains", "overlapPending",
    ]

    /// Relative totals, checked by the ON/OFF and qualified-OFF relations
    /// instead of the per-forward derived expectation.
    public static let totalKeys: Set<String> = ["kernels", "buffers"]

    public init(values: [String: Int]) { self.values = values }

    public static var zero: RouterOverlapGateCounters {
        var v = [String: Int]()
        for k in keys { v[k] = 0 }
        return RouterOverlapGateCounters(values: v)
    }

    public subscript(_ key: String) -> Int { values[key] ?? 0 }

    public func minus(_ o: RouterOverlapGateCounters) -> RouterOverlapGateCounters {
        var v = [String: Int]()
        for k in Self.keys { v[k] = self[k] - o[k] }
        return RouterOverlapGateCounters(values: v)
    }

    public func scaled(_ n: Int) -> RouterOverlapGateCounters {
        var v = [String: Int]()
        for k in Self.keys { v[k] = self[k] * n }
        return RouterOverlapGateCounters(values: v)
    }

    public static func mismatches(actual a: RouterOverlapGateCounters,
                                  expected e: RouterOverlapGateCounters,
                                  label: String) -> [String] {
        var f = [String]()
        for k in keys where !totalKeys.contains(k) && a[k] != e[k] {
            f.append(label + " " + k + " " + String(a[k]) + " != " + String(e[k]))
        }
        return f
    }
}

extension RouterOverlapGateCounters {
    init(_ r: Gemma4Runner) {
        let s = r.routerOverlapState
        let v: [String: Int] = [
            "kernels": r.dispatchTotal,
            "buffers": r.submittedBufferTotal,
            "q4Generic": r.q4MatvecGenericDispatchCount,
            "q4Applied": r.q4LaunchAppliedCount,
            "q4Fallback": r.q4LaunchFallbackCount,
            "q4Generated": r.q4GeneratedDispatchCount,
            "moeGu": r.moeGuDispatchCount,
            "moeAct": r.moeActDispatchCount,
            "moeDn": r.moeDnDispatchCount,
            "moeMerges": r.moeAxpyMergeCount,
            "moeGenerated": r.moeGeneratedDispatchCount,
            "moeResidentFallback": r.moeResidentFallbackCount,
            "parallelAttentionApplied": r.parallelAttentionAppliedCount,
            "parallelAttentionFallback": r.parallelAttentionFallbackCount,
            "rmsCooperativeApplied": r.rmsCooperativeAppliedCount,
            "rmsCooperativeFallback": r.rmsCooperativeFallbackCount,
            "rmsGenerated": r.rmsGeneratedDispatchCount,
            "q6CooperativeApplied": r.q6CooperativeAppliedCount,
            "q6CooperativeFallback": r.q6CooperativeFallbackCount,
            "rmsStatsComputed": r.rmsStatsComputedCount,
            "rmsStatsReused": r.rmsStatsReusedCount,
            "rmsStatsFallback": r.rmsStatsFallbackCount,
            "routerScaleApplied": r.routerScaleApplied,
            "routerScaleFallback": r.routerScaleFallback,
            "residentZeroApplied": r.residentZeroApplied,
            "residentZeroFallback": r.residentZeroFallback,
            "routerLaunchApplied": r.routerLaunchApplied,
            "routerLaunchFallback": r.routerLaunchFallback,
            "overlapApplied": s?.appliedWindows ?? 0,
            "overlapFallback": s?.fallbackEvents ?? 0,
            "overlapAsyncCommitted": s?.ledger.asyncCommitted ?? 0,
            "overlapDrained": s?.ledger.drained ?? 0,
            "overlapFailedDrains": s?.ledger.failedDrains ?? 0,
            "overlapPending": s?.ledger.pending.count ?? 0,
        ]
        self.init(values: v)
    }
}

/// Per-forward counts derived once from the prepared diagnostic SSA, its
/// statistics plan, its overlap plan and the config (actual source
/// accounting, never hardcoded).
public struct RouterOverlapGateDerived: Codable, Equatable, Sendable {
    public let layers: Int
    public let topK: Int
    public let moeChunks: Int
    public let q4Matvecs: Int
    public let q6Matvecs: Int
    public let attentionOps: Int
    public let rmsnormOps: Int
    public let statisticsGroups: Int
    public let statisticsReuses: Int
    public let routerScaleOps: Int
    public let expertChains: Int
    public let overlapWindows: Int

    /// RMSNorm instructions not served by the statistics plan: each goes
    /// through Gemma4Runner.rmsnorm and its cooperative hook.
    public var rmsCooperativeCalls: Int { rmsnormOps - statisticsGroups - statisticsReuses }

    public var failures: [String] {
        var f = [String]()
        if layers <= 0 || topK <= 0 { f.append("derived: config layers/topK not positive") }
        if moeChunks != (topK + RouterOverlapGateProtocol.required.moeBatch - 1)
            / RouterOverlapGateProtocol.required.moeBatch {
            f.append("derived: MoE chunk count inconsistent")
        }
        if q4Matvecs <= 0 { f.append("derived: no Q4 matvec instructions") }
        if q6Matvecs != 1 { f.append("derived: Q6 LM-head matvecs " + String(q6Matvecs) + " != 1") }
        if attentionOps != layers { f.append("derived: attention ops != layers") }
        if statisticsGroups != layers { f.append("derived: statistics groups != layers") }
        if statisticsReuses != 2 * layers { f.append("derived: statistics reuses != 2 * layers") }
        if routerScaleOps != layers { f.append("derived: router scale ops != layers") }
        if expertChains != layers { f.append("derived: expert chains != layers") }
        if overlapWindows != layers { f.append("derived: overlap windows != layers") }
        if rmsCooperativeCalls <= 0 { f.append("derived: no cooperative RMSNorm calls") }
        return f
    }

    public func perForward(_ arm: RouterOverlapGateArm) -> RouterOverlapGateCounters {
        var c = RouterOverlapGateCounters.zero
        switch arm {
        case .reference:
            // Direct path, install nil: routed experts are 2 * topK generic
            // Q4 matvecs per layer on top of the SSA Q4 matvec set.
            c.values["q4Generic"] = q4Matvecs + 2 * topK * layers
            c.values["q4Applied"] = q4Matvecs + 2 * topK * layers
        case .off, .on:
            c.values["q4Generic"] = q4Matvecs
            c.values["q4Applied"] = q4Matvecs
            c.values["moeGu"] = moeChunks * layers
            c.values["moeAct"] = moeChunks * layers
            c.values["moeDn"] = moeChunks * layers
            c.values["moeMerges"] = topK * layers
            c.values["moeGenerated"] = 3 * moeChunks * layers
            c.values["parallelAttentionApplied"] = attentionOps
            c.values["rmsCooperativeApplied"] = rmsCooperativeCalls
            c.values["q6CooperativeApplied"] = q6Matvecs
            c.values["rmsStatsComputed"] = statisticsGroups
            c.values["rmsStatsReused"] = statisticsReuses
            c.values["routerScaleApplied"] = routerScaleOps
            c.values["residentZeroApplied"] = expertChains
            if arm == .on {
                c.values["overlapApplied"] = overlapWindows
                c.values["overlapAsyncCommitted"] = 2 * overlapWindows
                c.values["overlapDrained"] = 2 * overlapWindows
            }
        }
        return c
    }
}

// MARK: - Records

public struct RouterOverlapGateTotals: Codable, Equatable, Sendable {
    public var prefillKernels = 0
    public var prefillBuffers = 0
    /// Command-buffer elapsed (gpuEnd - gpuStart), not busy time.
    public var prefillGpuElapsedMicros = 0.0
    public var continuationKernels: [Int] = []
    public var continuationBuffers: [Int] = []
    public var continuationGpuElapsedMicros: [Double] = []
}

public struct RouterOverlapGatePhaseCounts: Codable, Equatable, Sendable {
    public let prefillForwards: Int
    public let continuationForwards: Int
    public let prefillActual: RouterOverlapGateCounters
    public let continuationActual: RouterOverlapGateCounters
    public let prefillExpected: RouterOverlapGateCounters
    public let continuationExpected: RouterOverlapGateCounters
    public let mismatches: [String]

    public var exact: Bool { mismatches.isEmpty }
}

public struct RouterOverlapGateDelta: Codable, Equatable, Sendable {
    public let step: Int
    public let index: Int
    public let value: Float
    public let reference: Float
    public let absDelta: Double
    public let normalized: Double
}

public struct RouterOverlapGateBitMismatch: Codable, Equatable, Sendable {
    public let step: Int
    public let index: Int
    public let onBits: UInt32
    public let offBits: UInt32
    public let on: Float
    public let off: Float
}

public struct RouterOverlapGateQualifiedArm: Codable, Equatable, Sendable {
    public let arm: RouterOverlapGateArm
    public let ran: Bool
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let firstIdDriftStep: Int?
    public let outputs: Int
    public let allFinite: Bool
    public let comparedSteps: Int
    public let perStepNormalizedMaxDiff: [Double]
    public let maxNormalizedDiff: Double?
    public let worstStep: RouterOverlapGateDelta?
    public let highestDelta: RouterOverlapGateDelta?
    public let bitComparedSteps: Int?
    public let bitMismatchCount: Int?
    public let firstBitMismatch: RouterOverlapGateBitMismatch?
    public let totals: RouterOverlapGateTotals
    public let counts: RouterOverlapGatePhaseCounts?
    public let totalsRelationMismatches: [String]
    public let preconditionFailures: [String]
    public let scopeError: String?
    public let error: String?
    public let passed: Bool
    public let failure: String?

    static func notRun(_ arm: RouterOverlapGateArm, _ reason: String) -> RouterOverlapGateQualifiedArm {
        RouterOverlapGateQualifiedArm(
            arm: arm, ran: false, ids: [], idsMatchFrozen: false, firstIdDriftStep: nil,
            outputs: 0, allFinite: false, comparedSteps: 0, perStepNormalizedMaxDiff: [],
            maxNormalizedDiff: nil, worstStep: nil, highestDelta: nil, bitComparedSteps: nil,
            bitMismatchCount: nil, firstBitMismatch: nil, totals: RouterOverlapGateTotals(),
            counts: nil, totalsRelationMismatches: [], preconditionFailures: [], scopeError: nil,
            error: nil, passed: false, failure: "not run: " + reason)
    }
}

public struct RouterOverlapGateQualificationFixture: Codable, Equatable, Sendable {
    public let fixture: String
    public let reference: RouterOverlapGateQualifiedArm
    public let off: RouterOverlapGateQualifiedArm
    public let on: RouterOverlapGateQualifiedArm
    public let passed: Bool
}

public struct RouterOverlapGateRefusalTest: Codable, Equatable, Sendable {
    public let name: String
    public let insideScope: Bool
    public let refused: Bool
    public let error: String?
    public let noGpuWork: Bool
    public let stateIntact: Bool
    public let passed: Bool
}

public struct RouterOverlapGateIdentity: Codable, Equatable {
    public let device: String
    public let executableDigest: String
    public let toolchain: String
    public let modelSha256: String
    public let modelSha256Source: String
    public let modelFileVerification: ModelFileVerificationPolicy
    public let graphDigest: String
    public let regionGraphSha256: String
    public let scheduleDigest: String
    public let overlapPlanAnalysis: Int?
    public let overlapPlanDigestVerified: Bool
    public let overlapCanonicalScheduleDigest: String?
    public let overlapStatisticsPlanDigest: String?
    public let overlapWindows: Int?
    public let overlapPermutationCount: Int?
    public let overlapRemappedValues: Int?
    public let moeArtifactDigest: String?
    public let moeBatch: Int?
    public let moeRowsPerGroup: Int?
    public let moeSourceSha256: String?
    public let recordedBaselineMoeSourceSha256: String
    public let moeRegionGraphSha256: String?
    public let q4PipelineMaxThreads: Int
    public let q4LaunchWidthInScope: Int?
    public let engineIdentityDigest: String
    public let candidateFlags: [String: Int]
    public let candidateLegal: Bool
    public let candidateOverlapRejection: String?
    public let sameRunnerBinding: Bool
    public let sameGraphBinding: Bool
    public let preparedInstallBindingOk: Bool
    public let controlHasNoPlan: Bool
    public let diagnosticHasPlan: Bool
    public let controlAndDiagnosticSameSSA: Bool
    public let pristineAtStart: Bool
    public let instrumentationOffAtStart: Bool
    public let derived: RouterOverlapGateDerived
}

public struct RouterOverlapGatePerfRun: Codable, Equatable, Sendable {
    public let fixture: String
    public let arm: RouterOverlapGateArm
    public let warmup: Bool
    public let block: Int?
    public let orderInBlock: Int?
    public let prefillWallMicros: Double
    public let continuationWallMicros: [Double]
    public let tpotMedianWallMicros: Double?
    public let totals: RouterOverlapGateTotals
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let outputs: Int
    public let allFinite: Bool
    public let counts: RouterOverlapGatePhaseCounts
    public let totalsMismatches: [String]
    public let preconditionFailures: [String]
    public let scopeError: String?
    public let error: String?
    public let checksPassed: Bool
    public let failure: String?
}

public struct RouterOverlapGateFixtureOutcome: Codable, Equatable, Sendable {
    public let fixture: String
    public let baselineRunTPOTMicros: [Double]
    public let candidateRunTPOTMicros: [Double]
    public let baselineMedianTPOTMicros: Double?
    public let candidateMedianTPOTMicros: Double?
    public let baselineSpread: Double?
    public let candidateSpread: Double?
    public let baselineStable: Bool
    public let candidateStable: Bool
    public let gain: Double?
    public let gainMeetsThreshold: Bool
    public let baselineMedianPrefillWallMicros: Double?
    public let candidateMedianPrefillWallMicros: Double?
    public let allRunChecksPassed: Bool
    public let outcome: String
}

public struct RouterOverlapGateReport: Codable, Equatable {
    public private(set) var kind = RouterOverlapModelGate.kind
    public var mode: RouterOverlapModelGateMode
    public private(set) var protocolSpec = RouterOverlapGateProtocol.required
    public private(set) var candidate = RouterOverlapModelGate.modernCandidate
    public var identity: RouterOverlapGateIdentity? = nil
    public var identityFailures: [String] = []
    public var refusalTests: [RouterOverlapGateRefusalTest] = []
    public var refusalScopeError: String? = nil
    public var refusalPassed = false
    public var qualification: [RouterOverlapGateQualificationFixture] = []
    public var qualificationPassed = false
    public var qualificationStartUptimeNanos: UInt64? = nil
    public var qualificationEndUptimeNanos: UInt64? = nil
    /// Qualification elapsed time; never counted as TPOT.
    public var qualificationElapsedMicros: Double? = nil
    public var performanceStartUptimeNanos: UInt64? = nil
    public var performanceEndUptimeNanos: UInt64? = nil
    public var runs: [RouterOverlapGatePerfRun] = []
    public var fixtures: [RouterOverlapGateFixtureOutcome] = []
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: RouterOverlapGateStatus = .aborted
    public private(set) var gainQualified = false
    public private(set) var externalResourceEligibility = RouterOverlapModelGate.externalResourceEligibility
    public private(set) var notes = RouterOverlapModelGate.notes

    public init(mode: RouterOverlapModelGateMode) { self.mode = mode }
}

// MARK: - Internal helpers

struct RouterOverlapGateContext {
    let graph: GemmaGraphIR
    let region: MoERegionIR
    let runner: Gemma4Runner
    let candidate: GemmaForgeCandidate
    let derived: RouterOverlapGateDerived
    let off: GemmaPreparedExecution
    let on: GemmaPreparedExecution
}

struct RouterOverlapGateGreedy {
    var ids = [Int]()
    var outputs = 0
    var allFinite = true
    var totals = RouterOverlapGateTotals()
    var s0 = RouterOverlapGateCounters.zero
    var s1 = RouterOverlapGateCounters.zero
    var s2 = RouterOverlapGateCounters.zero
    var error: String? = nil
}

struct RouterOverlapGateArmResult {
    let record: RouterOverlapGateQualifiedArm
    let kept: [[Float]]
    let totals: RouterOverlapGateTotals
}

struct RouterOverlapGateComparator {
    var perStep = [Double]()
    var worst: RouterOverlapGateDelta? = nil
    var highest: RouterOverlapGateDelta? = nil

    mutating func add(step: Int, logits: [Float], reference: [Float]) {
        guard logits.count == reference.count, !logits.isEmpty else {
            perStep.append(.infinity)
            return
        }
        var maxDelta = -1.0
        var index = 0
        var maxRef = 0.0
        for i in 0..<logits.count {
            let d = abs(Double(logits[i]) - Double(reference[i]))
            if d > maxDelta {
                maxDelta = d
                index = i
            }
            let a = abs(Double(reference[i]))
            if a > maxRef { maxRef = a }
        }
        let normalized = maxDelta / max(1.0, maxRef)
        let rec = RouterOverlapGateDelta(step: step, index: index, value: logits[index],
                                         reference: reference[index], absDelta: maxDelta,
                                         normalized: normalized)
        perStep.append(normalized)
        if worst.map({ normalized > $0.normalized }) ?? true { worst = rec }
        if highest.map({ maxDelta > $0.absDelta }) ?? true { highest = rec }
    }
}

struct RouterOverlapGateBitComparator {
    var compared = 0
    var missing = 0
    var mismatches = 0
    var first: RouterOverlapGateBitMismatch? = nil

    mutating func add(step: Int, on: [Float], off: [Float]) {
        compared += 1
        guard on.count == off.count else {
            mismatches += max(on.count, off.count)
            return
        }
        for i in 0..<on.count where on[i].bitPattern != off[i].bitPattern {
            mismatches += 1
            if first == nil {
                first = RouterOverlapGateBitMismatch(step: step, index: i, onBits: on[i].bitPattern,
                                                     offBits: off[i].bitPattern, on: on[i],
                                                     off: off[i])
            }
        }
    }
}

struct RouterOverlapGateGuardState: Equatable {
    let kernels: Int
    let buffers: Int
    let gpuBits: UInt64
    let position: Int
    let asyncCommitted: Int
    let drained: Int
    let pending: Int
    let open: Bool
    let moeDigest: String?
    let q4Width: Int?
    let scopeActive: Bool

    init(_ r: Gemma4Runner) {
        kernels = r.dispatchTotal
        buffers = r.submittedBufferTotal
        gpuBits = r.totalGpuMicros.bitPattern
        position = r.contextPosition()
        asyncCommitted = r.routerOverlapAsyncCommittedCount
        drained = r.routerOverlapDrainedCount
        pending = r.routerOverlapPendingCount
        open = r.openEnc != nil || r.openCmd != nil
        moeDigest = r.moeGenerated?.artifactDigest
        q4Width = r.q4LaunchWidth
        scopeActive = r.routerOverlapScopeActive
    }
}

struct RouterOverlapGateEvent<Payload: Encodable>: Encodable {
    let event: String
    let payload: Payload
}

// MARK: - Gate

public enum RouterOverlapModelGate {
    public static let kind = "router-overlap-model-gate-normal-path-q4w32-v1E4R32"
    public static let readyToken = "BENCHMARK_READY"
    public static let goToken = "GO"
    public static let qualificationPassedToken = "QUALIFICATION_PASSED"
    public static let qualificationFailedToken = "QUALIFICATION_FAILED"
    public static let endToken = "TIMED_WINDOW_END"
    public static let outcomeRunCheckFailed = "runCheckFailed"
    public static let outcomeUnstable = "unstable"
    public static let outcomeNoGain = "noGain"
    public static let outcomeInternalCriteriaMet = "internalCriteriaMet"
    public static let modelSha256Source =
        "ModelDigestCache cached record (cached-explicit-refresh); never hashed, re-verified or "
        + "refreshed by this tool"
    public static let externalResourceEligibility =
        "PENDING: the supervisor's global resource receipts before GO and after TIMED_WINDOW_END "
        + "decide resource eligibility; no internal resource pass is assumed"

    /// The declared full modern candidate for BOTH arms (fourteen flags).
    public static let modernCandidate = GemmaForgeCandidate(
        segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
        gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
        matvecCooperativeQ6: true, rmsStatisticsReuse: true, routerGPUResidency: true,
        routerLaunchDivisor: 0)

    public static let expectedCandidateFlags: [String: Int] = [
        "segmentBatching": 1, "top5Diagnostic": 0, "matvecTiledQ4": 0, "matvecTiledQ6": 0,
        "rmsnormTiled": 0, "matvecTiledF32": 0, "vectorizedLoads": 1, "gpuCacheAppend": 1,
        "parallelAttention": 1, "rmsnormCooperative": 1, "matvecCooperativeQ6": 1,
        "rmsStatisticsReuse": 1, "routerGPUResidency": 1, "routerLaunchDivisor": 0,
    ]

    public static let refusalTestNames = [
        "raw-graph-execution-inside-scope",
        "raw-schedule-execution-inside-scope",
        "control-prepared-execution-inside-scope",
        "control-prepared-continue-inside-scope",
        "diagnostic-prepared-non-modern-candidate-inside-scope",
        "nested-scope-inside-scope",
        "engine-identity-inside-scope",
        "forge-inside-scope",
        "diagnostic-prepared-execution-outside-scope",
        "diagnostic-prepared-continue-outside-scope",
    ]

    public static let notes: [String] = [
        "Diagnostic only: no artifact, no winner, no activation, no promotion; gainQualified is "
            + "always false and a numerical or performance result is not resource eligibility.",
        "Both arms share one executable, one runner, the installed v1 E4R32 routed lowering "
            + "(installMoeBaselineForVerification, hard-pinned recorded source SHA), Q4 reference "
            + "width 32 and the declared full modern candidate (par64 parallel attention on); "
            + "only the router overlap preparation and scope differ. CPU FP64 routing is unchanged.",
        "Counter expectations follow actual source accounting per forward: v1 GU = ACT = DN = "
            + "ceil(topK / 4) * layers, ordered merges = topK * layers, router scale and resident "
            + "zero = layers, statistics compute = plan groups (= layers) and reuse = plan reuses "
            + "(= 2 * layers), cooperative RMS = unplanned RMSNorm instructions, cooperative Q6 = "
            + "Q6 matvecs, parallel attention = attention ops; ON overlap applied = layers, async "
            + "commits = drained = 2 * layers, pending 0; every fallback 0.",
        "Dispatch multiset proxy: per-name kernel attribution does not exist in segment-batched "
            + "mode, so equivalence is established by the exact per-family counters plus total "
            + "kernels ON == OFF and command buffers ON == OFF + layers per forward, checked for "
            + "prefill (prompt forwards) and each continuation separately.",
        "The existing reusable-statistics inverse buffer is pre-sized once before any dispatch "
            + "(allocation only) so its lazy first growth cannot flush an open segment inside the "
            + "first OFF token and perturb buffer totals.",
        "Command-buffer elapsed (gpuEndTime - gpuStartTime) is recorded but is not busy time or "
            + "utilization, may include event waits under overlap, and is never summed or compared "
            + "between arms as gain.",
        "Wall timers bracket only executeGraph(prompt) and each executeGraphContinue(previous "
            + "argmax), including the normal CPU argmax and logits readback; candidate calls include "
            + "the runtime's unavoidable per-token overlap guards. Scope entry, install/source/"
            + "binding checks, counter snapshots and finiteness/ID checks run outside the timers.",
        "The supervisor's raw-line and state disk journaling of the stdout progress lines adds "
            + "nonzero observation overhead between timed calls.",
        "Model SHA-256 is the cached explicit-refresh record; this tool never hashes, re-verifies "
            + "or refreshes the model file.",
    ]

    typealias Step = (argmax: Int, logits: [Float], gpuMicros: Double, kernels: Int, buffers: Int)

    // MARK: Pure helpers

    /// Fixed block order: even blocks AB (OFF, ON), odd blocks BA.
    public static func blockOrder(_ block: Int) -> [RouterOverlapGateArm] {
        block % 2 == 0 ? [.off, .on] : [.on, .off]
    }

    public static func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty else { return nil }
        let s = v.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    public static func runTPOT(_ walls: [Double]) -> Double? {
        guard walls.count == RouterOverlapGateProtocol.required.continuationsPerRun,
              walls.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        return median(walls)
    }

    public static func spread(_ v: [Double]) -> Double? {
        guard v.count == RouterOverlapGateProtocol.required.blocksPerFixture,
              v.allSatisfy({ $0.isFinite && $0 > 0 }),
              let m = median(v), m > 0, let lo = v.min(), let hi = v.max() else { return nil }
        return (hi - lo) / m
    }

    public static func gain(baseline: Double?, candidate: Double?) -> Double? {
        guard let b = baseline, let c = candidate, b.isFinite, c.isFinite, b > 0, c > 0 else {
            return nil
        }
        return (b - c) / b
    }

    public static func candidateFlags(_ c: GemmaForgeCandidate) -> [String: Int] {
        func b(_ v: Bool) -> Int { v ? 1 : 0 }
        let d: [String: Int] = [
            "segmentBatching": b(c.segmentBatching), "top5Diagnostic": b(c.top5Diagnostic),
            "matvecTiledQ4": b(c.matvecTiledQ4), "matvecTiledQ6": b(c.matvecTiledQ6),
            "rmsnormTiled": b(c.rmsnormTiled), "matvecTiledF32": b(c.matvecTiledF32),
            "vectorizedLoads": b(c.vectorizedLoads), "gpuCacheAppend": b(c.gpuCacheAppend),
            "parallelAttention": b(c.parallelAttention),
            "rmsnormCooperative": b(c.rmsnormCooperative),
            "matvecCooperativeQ6": b(c.matvecCooperativeQ6),
            "rmsStatisticsReuse": b(c.rmsStatisticsReuse),
            "routerGPUResidency": b(c.routerGPUResidency),
            "routerLaunchDivisor": c.routerLaunchDivisor,
        ]
        return d
    }

    public static func fixtureOutcome(fixture: String,
                                      runs: [RouterOverlapGatePerfRun]) -> RouterOverlapGateFixtureOutcome {
        let p = RouterOverlapGateProtocol.required
        let mine = runs.filter { $0.fixture == fixture }
        func timed(_ arm: RouterOverlapGateArm) -> [RouterOverlapGatePerfRun] {
            mine.filter { $0.arm == arm && !$0.warmup }.sorted { ($0.block ?? -1) < ($1.block ?? -1) }
        }
        let b = timed(.off)
        let c = timed(.on)
        let warmups = mine.filter { $0.warmup }
        let blocks = Array(0..<p.blocksPerFixture)
        let orderOk = (b + c).allSatisfy { run in
            guard let blk = run.block else { return false }
            return run.orderInBlock == blockOrder(blk).firstIndex(of: run.arm)
        }
        let shapeOk = b.count == p.blocksPerFixture && c.count == p.blocksPerFixture
            && b.compactMap({ $0.block }) == blocks && c.compactMap({ $0.block }) == blocks
            && orderOk && warmups.map({ $0.arm }) == [.off, .on]
        let checks = shapeOk && mine.allSatisfy({ $0.checksPassed })
            && (b + c).allSatisfy({ $0.tpotMedianWallMicros != nil })
        let bT = b.compactMap { $0.tpotMedianWallMicros }
        let cT = c.compactMap { $0.tpotMedianWallMicros }
        let bMed = median(bT)
        let cMed = median(cT)
        let bSpread = spread(bT)
        let cSpread = spread(cT)
        let bStable = bSpread.map { $0 <= p.spreadLimit } ?? false
        let cStable = cSpread.map { $0 <= p.spreadLimit } ?? false
        let g = gain(baseline: bMed, candidate: cMed)
        let meets = g.map { $0 >= p.gainThreshold } ?? false
        let outcome: String
        if !checks {
            outcome = outcomeRunCheckFailed
        } else if !(bStable && cStable) {
            outcome = outcomeUnstable
        } else if !meets {
            outcome = outcomeNoGain
        } else {
            outcome = outcomeInternalCriteriaMet
        }
        return RouterOverlapGateFixtureOutcome(
            fixture: fixture, baselineRunTPOTMicros: bT, candidateRunTPOTMicros: cT,
            baselineMedianTPOTMicros: bMed, candidateMedianTPOTMicros: cMed,
            baselineSpread: bSpread, candidateSpread: cSpread, baselineStable: bStable,
            candidateStable: cStable, gain: g, gainMeetsThreshold: meets,
            baselineMedianPrefillWallMicros: median(b.map { $0.prefillWallMicros }),
            candidateMedianPrefillWallMicros: median(c.map { $0.prefillWallMicros }),
            allRunChecksPassed: checks, outcome: outcome)
    }

    public static func performanceStatus(_ outcomes: [RouterOverlapGateFixtureOutcome]) -> RouterOverlapGateStatus {
        guard outcomes.map({ $0.fixture }) == RouterOverlapGateProtocol.required.fixtures,
              !outcomes.contains(where: { $0.outcome == outcomeRunCheckFailed }) else {
            return .performanceRunCheckFailed
        }
        if outcomes.contains(where: { $0.outcome == outcomeUnstable }) { return .performanceUnstable }
        if outcomes.contains(where: { $0.outcome == outcomeNoGain }) { return .performanceNoGain }
        return .performanceInternalCriteriaMetExternalEligibilityPending
    }

    static func finalStatus(_ r: RouterOverlapGateReport) -> RouterOverlapGateStatus {
        if !r.restoredPristine { return .runnerNotRestored }
        if r.abortReason != nil { return .aborted }
        if !r.qualificationPassed { return .qualificationFailed }
        if r.mode == .qualificationOnly { return .qualificationPassed }
        return performanceStatus(r.fixtures)
    }

    public static func aborted(mode: RouterOverlapModelGateMode, reason: String) -> RouterOverlapGateReport {
        var r = RouterOverlapGateReport(mode: mode)
        r.abortReason = reason
        r.status = .aborted
        return r
    }

    static func jsonEncoder(pretty: Bool) -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys]
        e.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return e
    }

    public static func encodeReport(_ r: RouterOverlapGateReport, pretty: Bool) throws -> Data {
        try jsonEncoder(pretty: pretty).encode(r)
    }

    static func event<T: Encodable>(_ name: String, _ payload: T) -> String {
        let wrapped = RouterOverlapGateEvent(event: name, payload: payload)
        guard let d = try? jsonEncoder(pretty: false).encode(wrapped),
              let s = String(data: d, encoding: .utf8) else {
            return "PROGRESS {" + "\"event\":\"encodeError\"}"
        }
        return "PROGRESS " + s
    }

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    static func micros(_ a: UInt64, _ b: UInt64) -> Double {
        b >= a ? Double(b - a) / 1000 : -1
    }

    static func fullFinite(_ v: [Float], _ vocab: Int) -> Bool {
        v.count == vocab && v.allSatisfy { $0.isFinite }
    }

    // MARK: Runner state predicates

    /// No lowering, default Q4 width, no cache, no generated kernel, no
    /// scope, no open or pending work.
    public static func pristine(_ r: Gemma4Runner) -> Bool {
        r.moeGenerated == nil && r.q4LaunchWidth == nil && r.expertCache == nil
            && r.rmsGeneratedKernel == nil && r.q4GeneratedKernel == nil
            && quiescent(r)
    }

    /// Every committed command buffer completed: no open encoder, no pending
    /// overlap ticket, no active scope.
    public static func quiescent(_ r: Gemma4Runner) -> Bool {
        r.openEnc == nil && r.openCmd == nil && r.routerOverlapPendingCount == 0
            && !r.routerOverlapScopeActive
    }

    static func instrumentationOff(_ r: Gemma4Runner) -> Bool {
        r.layerProbe == nil && r.moeInputProbe == nil && !r.traceLayers
            && !r.breakdownInstrumentation && !r.commandTimelineEnabled
            && !r.attentionDiagnosticActive && r.rmsGeneratedKernel == nil
            && r.q4GeneratedKernel == nil && r.openEnc == nil && r.openCmd == nil
            && r.routerOverlapPendingCount == 0 && r.expertCache == nil
    }

    static func axisFlagsClean(_ r: Gemma4Runner) -> Bool {
        let flags = [r.autoSegment, r.q4TiledKernel, r.q6TiledKernel, r.rmsTiledKernel,
                     r.f32TiledKernel, r.vectorizedLoads, r.gpuCacheAppendKernel,
                     r.parallelAttentionKernel, r.rmsCooperativeKernel,
                     r.matvecCooperativeQ6Kernel, r.rmsStatisticsReuse, r.routerGPUResidency]
        return !flags.contains(true) && r.routerLaunchDivisor == 0
    }

    static func installFailures(_ r: Gemma4Runner, region: MoERegionIR) -> [String] {
        let p = RouterOverlapGateProtocol.required
        guard let m = r.moeGenerated else { return ["v1 E4R32 routed lowering not installed"] }
        var f = [String]()
        if m.schedule.batch != p.moeBatch || m.schedule.rowsPerGroup != p.moeRowsPerGroup {
            f.append("installed routed schedule is not E4R32")
        }
        if MoEKernelEmitter.sourcesSha256([m.kernels.source]) != Q4LaunchArtifact.baselineMoeSourceSha256 {
            f.append("installed routed source SHA differs from the recorded E4R32 baseline")
        }
        if m.region.graphSha256 != region.graphSha256 { f.append("installed region differs") }
        if r.expertCache != nil { f.append("expert-cache mode") }
        return f
    }

    static func bindingFailures(_ p: GemmaPreparedExecution, _ c: RouterOverlapGateContext) -> [String] {
        var f = [String]()
        if p.runner !== c.runner { f.append("prepared execution bound to a different runner") }
        if p.boundGraph != c.graph { f.append("prepared execution bound to a different graph") }
        if p.boundMoeArtifactDigest != c.runner.moeGenerated?.artifactDigest
            || p.boundQ4LaunchWidth != c.runner.q4LaunchWidth {
            f.append("prepared install binding (MoE digest / Q4 width) differs from the runner")
        }
        return f
    }

    static func preconditions(_ arm: RouterOverlapGateArm, _ c: RouterOverlapGateContext,
                              prepared: GemmaPreparedExecution?) -> [String] {
        let r = c.runner
        var f = [String]()
        if r.q4LaunchWidth != RouterOverlapGateProtocol.required.q4LaunchWidth {
            f.append("Q4 launch width is not 32")
        }
        if !instrumentationOff(r) {
            f.append("probe/trace/breakdown/timeline/attention diagnostic/generated kernel/open work present")
        }
        if !axisFlagsClean(r) { f.append("runner axis flags are not at their defaults") }
        if r.routerOverlapScopeActive { f.append("router overlap scope already active") }
        switch arm {
        case .reference:
            if r.moeGenerated != nil { f.append("routed lowering installed for the direct reference") }
        case .off, .on:
            f += installFailures(r, region: c.region)
            if let p = prepared {
                f += bindingFailures(p, c)
                if arm == .off, p.routerOverlap != nil { f.append("OFF preparation carries an overlap plan") }
                if arm == .on, p.routerOverlap == nil { f.append("ON preparation carries no overlap plan") }
            } else {
                f.append("no prepared execution")
            }
        }
        return f
    }

    // MARK: Preparation and identity

    static func installAndPrepare(_ r: Gemma4Runner, region: MoERegionIR, graph: GemmaGraphIR) throws
        -> (off: GemmaPreparedExecution, on: GemmaPreparedExecution) {
        let p = RouterOverlapGateProtocol.required
        r.uninstallMoeGeneratedLowering()
        try r.installMoeBaselineForVerification(
            region: region, schedule: MoESchedule(batch: p.moeBatch, rowsPerGroup: p.moeRowsPerGroup),
            expectedSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256)
        let off = try GemmaPreparedExecution.prepare(graph: graph, runner: r)
        let on = try GemmaPreparedExecution.prepareRouterOverlapDiagnostic(graph: graph, runner: r)
        return (off, on)
    }

    static func derive(on: GemmaPreparedExecution, runner r: Gemma4Runner) -> RouterOverlapGateDerived {
        var q4 = 0
        var q6 = 0
        var attention = 0
        var rms = 0
        var routerScale = 0
        var chains = 0
        for inst in on.ssa.instructions {
            switch inst.op {
            case "matvec":
                if let t = inst.tensors.first, let ref = r.ref[t] {
                    if ref.type == .q4_0 { q4 += 1 } else if ref.type == .q6_K { q6 += 1 }
                }
            case "attention": attention += 1
            case "rmsnorm": rms += 1
            case "scale": if inst.attrs["routerScale"] == 1 { routerScale += 1 }
            case "expert_matvec_chain": chains += 1
            default: break
            }
        }
        let topK = r.config.topKExperts
        let batch = RouterOverlapGateProtocol.required.moeBatch
        return RouterOverlapGateDerived(
            layers: r.config.layerCount, topK: topK, moeChunks: (topK + batch - 1) / batch,
            q4Matvecs: q4, q6Matvecs: q6, attentionOps: attention, rmsnormOps: rms,
            statisticsGroups: on.statisticsPlan.groupCount,
            statisticsReuses: on.statisticsPlan.reuseCount, routerScaleOps: routerScale,
            expertChains: chains, overlapWindows: on.routerOverlapPlan?.windows.count ?? 0)
    }

    static func makeIdentity(_ c: RouterOverlapGateContext, modelSha256: String,
                             executableDigest: String, pristineAtStart: Bool,
                             instrumentationOffAtStart: Bool) throws -> RouterOverlapGateIdentity {
        let r = c.runner
        let m = r.moeGenerated
        let plan = c.on.routerOverlapPlan
        var planVerified = false
        if let plan { planVerified = (try? RouterOverlapPlan.verifyDigest(plan)) != nil }
        let legal = GemmaForge.candidates().contains(c.candidate)
            && (try? GemmaForge.validateCandidate(c.candidate)) != nil
        let engine = GemmaForge.engineDigest(try r.engineIdentity())
        let graphDigest = try GemmaGraph.digest(c.graph)
        let scheduleDigest = try GemmaSSA.digest(c.on.ssa)
        let q4Max = try r.q4MatvecPipelineMaxThreads()
        return RouterOverlapGateIdentity(
            device: r.q4RunnerDevice.name, executableDigest: executableDigest,
            toolchain: Forge.toolchainIdentifier, modelSha256: modelSha256,
            modelSha256Source: modelSha256Source,
            modelFileVerification: ModelFileIdentity.policy, graphDigest: graphDigest,
            regionGraphSha256: c.region.graphSha256, scheduleDigest: scheduleDigest,
            overlapPlanAnalysis: plan?.analysis, overlapPlanDigestVerified: planVerified,
            overlapCanonicalScheduleDigest: plan?.canonicalScheduleDigest,
            overlapStatisticsPlanDigest: plan?.statisticsPlanDigest,
            overlapWindows: plan?.windows.count, overlapPermutationCount: plan?.permutation.count,
            overlapRemappedValues: plan?.remappedValueIDs.count,
            moeArtifactDigest: m?.artifactDigest, moeBatch: m?.schedule.batch,
            moeRowsPerGroup: m?.schedule.rowsPerGroup,
            moeSourceSha256: m.map { MoEKernelEmitter.sourcesSha256([$0.kernels.source]) },
            recordedBaselineMoeSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256,
            moeRegionGraphSha256: m?.region.graphSha256, q4PipelineMaxThreads: q4Max,
            q4LaunchWidthInScope: r.q4LaunchWidth, engineIdentityDigest: engine,
            candidateFlags: candidateFlags(c.candidate), candidateLegal: legal,
            candidateOverlapRejection: RouterOverlapScope.candidateRejection(c.candidate),
            sameRunnerBinding: c.off.runner === r && c.on.runner === r,
            sameGraphBinding: c.off.boundGraph == c.graph && c.on.boundGraph == c.graph,
            preparedInstallBindingOk: bindingFailures(c.off, c).isEmpty && bindingFailures(c.on, c).isEmpty,
            controlHasNoPlan: c.off.routerOverlap == nil, diagnosticHasPlan: c.on.routerOverlap != nil,
            controlAndDiagnosticSameSSA: c.off.ssa == c.on.ssa,
            pristineAtStart: pristineAtStart, instrumentationOffAtStart: instrumentationOffAtStart,
            derived: c.derived)
    }

    static func identityFailures(_ i: RouterOverlapGateIdentity) -> [String] {
        let p = RouterOverlapGateProtocol.required
        var f = [String]()
        if i.candidateFlags.count != 14 || i.candidateFlags != expectedCandidateFlags {
            f.append("candidate is not the declared fourteen-flag full modern candidate")
        }
        if !i.candidateLegal { f.append("candidate is not legal/valid") }
        if let reason = i.candidateOverlapRejection { f.append("overlap scope rejects the candidate: " + reason) }
        if i.moeBatch != p.moeBatch || i.moeRowsPerGroup != p.moeRowsPerGroup {
            f.append("installed routed schedule is not E4R32")
        }
        if i.moeSourceSha256 != i.recordedBaselineMoeSourceSha256 { f.append("routed source SHA mismatch") }
        if i.moeRegionGraphSha256 != i.regionGraphSha256 { f.append("routed region mismatch") }
        if i.q4LaunchWidthInScope != p.q4LaunchWidth { f.append("Q4 width in scope is not 32") }
        if !i.sameRunnerBinding || !i.sameGraphBinding || !i.preparedInstallBindingOk {
            f.append("prepared executions are not bound to this runner/graph/install")
        }
        if !i.controlHasNoPlan || !i.diagnosticHasPlan || !i.controlAndDiagnosticSameSSA {
            f.append("control/diagnostic preparations are not the same SSA with nil/non-nil plan")
        }
        if !i.overlapPlanDigestVerified || i.overlapCanonicalScheduleDigest != i.scheduleDigest {
            f.append("overlap plan digest/schedule binding not verified")
        }
        if !i.pristineAtStart || !i.instrumentationOffAtStart {
            f.append("runner was not pristine with instrumentation off at start")
        }
        f += i.derived.failures
        return f
    }

    // MARK: Execution

    static func directForward(_ r: Gemma4Runner, _ tokens: [Int]) throws -> Step {
        let b0 = r.submittedBufferTotal
        let ev = try r.forward(tokenIds: tokens)
        let logits = try r.currentLogits()
        return (ev.argmaxToken, logits, ev.gpuMicros, ev.dispatchedKernels, r.submittedBufferTotal - b0)
    }

    /// Greedy 32-output run feeding the previous actual argmax. Stops only on
    /// a thrown error or an invalid (non-finite / short) output; an ID drift
    /// does not stop the run. Snapshots are taken between calls.
    static func greedy(_ r: Gemma4Runner, outputs: Int, prefill: () throws -> Step,
                       next: (Int) throws -> Step,
                       observe: (Int, [Float]) -> Void) -> RouterOverlapGateGreedy {
        let vocab = r.config.vocabSize
        var g = RouterOverlapGateGreedy()
        g.s0 = RouterOverlapGateCounters(r)
        g.s1 = g.s0
        do {
            let first = try prefill()
            g.s1 = RouterOverlapGateCounters(r)
            g.totals.prefillKernels = first.kernels
            g.totals.prefillBuffers = first.buffers
            g.totals.prefillGpuElapsedMicros = first.gpuMicros
            if fullFinite(first.logits, vocab) {
                observe(0, first.logits)
                g.ids.append(first.argmax)
                g.outputs = 1
                while g.outputs < outputs {
                    let c = try next(g.ids[g.ids.count - 1])
                    g.totals.continuationKernels.append(c.kernels)
                    g.totals.continuationBuffers.append(c.buffers)
                    g.totals.continuationGpuElapsedMicros.append(c.gpuMicros)
                    guard fullFinite(c.logits, vocab) else {
                        g.allFinite = false
                        g.error = "output " + String(g.outputs) + " not full-vocabulary finite; arm stopped"
                        break
                    }
                    observe(g.outputs, c.logits)
                    g.ids.append(c.argmax)
                    g.outputs += 1
                }
            } else {
                g.allFinite = false
                g.error = "output 0 not full-vocabulary finite; arm stopped"
            }
        } catch {
            g.error = "threw: " + String(describing: error)
        }
        g.s2 = RouterOverlapGateCounters(r)
        return g
    }

    static func phaseCounts(_ g: RouterOverlapGateGreedy, _ d: RouterOverlapGateDerived,
                            _ arm: RouterOverlapGateArm, promptCount: Int) -> RouterOverlapGatePhaseCounts {
        let continuations = RouterOverlapGateProtocol.required.continuationsPerRun
        let per = d.perForward(arm)
        let ep = per.scaled(promptCount)
        let ec = per.scaled(continuations)
        let ap = g.s1.minus(g.s0)
        let ac = g.s2.minus(g.s1)
        var m = RouterOverlapGateCounters.mismatches(actual: ap, expected: ep, label: "prefill")
        m += RouterOverlapGateCounters.mismatches(actual: ac, expected: ec, label: "continuation")
        return RouterOverlapGatePhaseCounts(
            prefillForwards: promptCount, continuationForwards: continuations,
            prefillActual: ap, continuationActual: ac, prefillExpected: ep,
            continuationExpected: ec, mismatches: m)
    }

    /// candidate == baseline kernels; candidate buffers == baseline +
    /// extra per forward (prefill: per prompt token; continuation: per step).
    static func totalsRelation(candidate a: RouterOverlapGateTotals, baseline b: RouterOverlapGateTotals,
                               extraBuffersPerForward extra: Int, promptCount: Int) -> [String] {
        var f = [String]()
        if a.prefillKernels != b.prefillKernels {
            f.append("prefill kernels " + String(a.prefillKernels) + " != " + String(b.prefillKernels))
        }
        let wantPrefill = b.prefillBuffers + extra * promptCount
        if a.prefillBuffers != wantPrefill {
            f.append("prefill command buffers " + String(a.prefillBuffers) + " != " + String(wantPrefill))
        }
        let n = RouterOverlapGateProtocol.required.continuationsPerRun
        guard a.continuationKernels.count == n, b.continuationKernels.count == n,
              a.continuationBuffers.count == n, b.continuationBuffers.count == n else {
            f.append("continuation step counts are not 31 in both runs")
            return f
        }
        for i in 0..<n {
            if a.continuationKernels[i] != b.continuationKernels[i] {
                f.append("continuation " + String(i) + " kernels " + String(a.continuationKernels[i])
                         + " != " + String(b.continuationKernels[i]))
            }
            if a.continuationBuffers[i] != b.continuationBuffers[i] + extra {
                f.append("continuation " + String(i) + " command buffers "
                         + String(a.continuationBuffers[i]) + " != "
                         + String(b.continuationBuffers[i] + extra))
            }
        }
        return f
    }

    static func notRun(_ arm: RouterOverlapGateArm, _ reason: String) -> RouterOverlapGateArmResult {
        RouterOverlapGateArmResult(record: .notRun(arm, reason), kept: [], totals: RouterOverlapGateTotals())
    }

    /// One qualification arm: reference (direct), OFF or ON (whole run in
    /// one scope). OFF/ON are compared with the fresh reference; ON is also
    /// bit-compared with OFF at every logit.
    static func qualify(_ arm: RouterOverlapGateArm, _ c: RouterOverlapGateContext,
                        _ fx: RouterOverlapGateFixture, reference: [[Float]], off: [[Float]],
                        offTotals: RouterOverlapGateTotals?, keep: Bool) -> RouterOverlapGateArmResult {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let prepared: GemmaPreparedExecution? = arm == .reference ? nil : (arm == .on ? c.on : c.off)
        let pre = preconditions(arm, c, prepared: prepared)
        var kept = [[Float]]()
        var cmp = RouterOverlapGateComparator()
        var bits = RouterOverlapGateBitComparator()
        var g = RouterOverlapGateGreedy()
        var scopeError: String? = nil
        if pre.isEmpty {
            r.reset()
            func observe(_ step: Int, _ lg: [Float]) {
                if arm != .reference, step < reference.count {
                    cmp.add(step: step, logits: lg, reference: reference[step])
                }
                if arm == .on {
                    if step < off.count { bits.add(step: step, on: lg, off: off[step]) } else { bits.missing += 1 }
                }
                if keep { kept.append(lg) }
            }
            func preparedPrefill() throws -> Step {
                try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt,
                                            candidate: c.candidate, prepared: prepared)
            }
            func preparedNext(_ token: Int) throws -> Step {
                try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token,
                                                    candidate: c.candidate, prepared: prepared)
            }
            switch arm {
            case .reference:
                r.applyDiagnosticAxis()
                g = greedy(r, outputs: proto.outputsPerRun,
                           prefill: { try directForward(r, fx.prompt) },
                           next: { try directForward(r, [$0]) }, observe: observe)
            case .off:
                g = greedy(r, outputs: proto.outputsPerRun, prefill: preparedPrefill,
                           next: preparedNext, observe: observe)
            case .on:
                do {
                    try r.withRouterOverlapDiagnosticScope { () -> Void in
                        g = greedy(r, outputs: proto.outputsPerRun, prefill: preparedPrefill,
                                   next: preparedNext, observe: observe)
                    }
                } catch {
                    scopeError = String(describing: error)
                }
            }
            r.reset()
        }
        let counts = phaseCounts(g, c.derived, arm, promptCount: fx.prompt.count)
        var relation = [String]()
        if arm == .on {
            if let base = offTotals {
                relation = totalsRelation(candidate: g.totals, baseline: base,
                                          extraBuffersPerForward: c.derived.layers,
                                          promptCount: fx.prompt.count)
            } else {
                relation = ["no OFF totals to relate to"]
            }
        }
        var drift: Int? = nil
        for (i, id) in g.ids.enumerated() where i < fx.frozenIds.count && id != fx.frozenIds[i] {
            drift = i
            break
        }
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if let s = scopeError { f.append("scope: " + s) }
            if g.ids != fx.frozenIds { f.append("IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != proto.outputsPerRun {
                f.append("not 32 full-vocabulary finite outputs")
            }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            if arm != .reference {
                if cmp.perStep.count != proto.outputsPerRun {
                    f.append("compared " + String(cmp.perStep.count) + " of 32 outputs with the fresh reference")
                }
                if !((cmp.perStep.max() ?? .infinity) < proto.normalizedContract) {
                    f.append("normalized max diff vs fresh reference not < 1e-4")
                }
            }
            if arm == .on {
                if bits.compared != proto.outputsPerRun || bits.missing != 0 {
                    f.append("bit-compared " + String(bits.compared) + " of 32 outputs with OFF")
                }
                if bits.mismatches != 0 {
                    f.append("ON differs from OFF in " + String(bits.mismatches) + " logit bit patterns")
                }
                f += relation
            }
            if !quiescent(r) { f.append("runner not quiescent after the arm") }
        }
        let record = RouterOverlapGateQualifiedArm(
            arm: arm, ran: pre.isEmpty, ids: g.ids, idsMatchFrozen: g.ids == fx.frozenIds,
            firstIdDriftStep: drift, outputs: g.outputs, allFinite: g.allFinite,
            comparedSteps: cmp.perStep.count, perStepNormalizedMaxDiff: cmp.perStep,
            maxNormalizedDiff: cmp.perStep.max(), worstStep: cmp.worst, highestDelta: cmp.highest,
            bitComparedSteps: arm == .on ? bits.compared : nil,
            bitMismatchCount: arm == .on ? bits.mismatches : nil,
            firstBitMismatch: bits.first, totals: g.totals, counts: counts,
            totalsRelationMismatches: relation, preconditionFailures: pre, scopeError: scopeError,
            error: g.error, passed: f.isEmpty,
            failure: f.isEmpty ? nil : f.joined(separator: "; "))
        return RouterOverlapGateArmResult(record: record, kept: kept, totals: g.totals)
    }

    /// Guard refusal tests: every call must be refused before any reset or
    /// GPU work and leave install/scope state intact.
    static func refusalTests(_ c: RouterOverlapGateContext, prompt: [Int])
        -> (tests: [RouterOverlapGateRefusalTest], scopeError: String?) {
        let r = c.runner
        var tests = [RouterOverlapGateRefusalTest]()
        func probe(_ name: String, inside: Bool, _ action: () throws -> Void) {
            let before = RouterOverlapGateGuardState(r)
            var refused = false
            var message: String? = nil
            do { try action() } catch {
                refused = true
                message = String(describing: error)
            }
            let after = RouterOverlapGateGuardState(r)
            let noGpu = before.kernels == after.kernels && before.buffers == after.buffers
                && before.gpuBits == after.gpuBits && before.position == after.position
                && before.asyncCommitted == after.asyncCommitted && before.drained == after.drained
            let intact = after.pending == 0 && !after.open && before.moeDigest == after.moeDigest
                && before.q4Width == after.q4Width && before.scopeActive == after.scopeActive
            tests.append(RouterOverlapGateRefusalTest(
                name: name, insideScope: inside, refused: refused, error: message,
                noGpuWork: noGpu, stateIntact: intact, passed: refused && noGpu && intact))
        }
        let nonModern = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
        let url = r.ggufFile.sourceURL ?? URL(fileURLWithPath: "/dev/null")
        let token = prompt[prompt.count - 1]
        var scopeError: String? = nil
        do {
            try r.withRouterOverlapDiagnosticScope { () -> Void in
                probe("raw-graph-execution-inside-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt,
                                                    candidate: c.candidate)
                }
                probe("raw-schedule-execution-inside-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt,
                                                    candidate: c.candidate, schedule: c.off.ssa)
                }
                probe("control-prepared-execution-inside-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt,
                                                    candidate: c.candidate, prepared: c.off)
                }
                probe("control-prepared-continue-inside-scope", inside: true) {
                    _ = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token,
                                                            candidate: c.candidate, prepared: c.off)
                }
                probe("diagnostic-prepared-non-modern-candidate-inside-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt,
                                                    candidate: nonModern, prepared: c.on)
                }
                probe("nested-scope-inside-scope", inside: true) {
                    _ = try r.withRouterOverlapDiagnosticScope { () -> Int in 0 }
                }
                probe("engine-identity-inside-scope", inside: true) {
                    _ = try r.engineIdentity()
                }
                probe("forge-inside-scope", inside: true) {
                    _ = try GemmaForge.forge(graph: c.graph, runner: r, ggufURL: url, prompt: prompt,
                                             decodeTokens: GemmaForge.warmupTokens + GemmaForge.timedTokens,
                                             hardware: r.hardwareProfile())
                }
            }
        } catch {
            scopeError = String(describing: error)
        }
        if !quiescent(r) {
            scopeError = (scopeError.map { $0 + "; " } ?? "") + "scope did not close cleanly"
        }
        probe("diagnostic-prepared-execution-outside-scope", inside: false) {
            _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt,
                                            candidate: c.candidate, prepared: c.on)
        }
        probe("diagnostic-prepared-continue-outside-scope", inside: false) {
            _ = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token,
                                                    candidate: c.candidate, prepared: c.on)
        }
        return (tests, scopeError)
    }

    /// One normal 32-output run. Only executeGraph(prompt) and each
    /// executeGraphContinue(previous argmax) are timed; the candidate scope
    /// wraps the complete run outside the timers.
    static func measuredRun(_ c: RouterOverlapGateContext, _ fx: RouterOverlapGateFixture,
                            _ arm: RouterOverlapGateArm, warmup: Bool, block: Int?, order: Int?,
                            qualifiedOff: RouterOverlapGateTotals) -> RouterOverlapGatePerfRun {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let prepared: GemmaPreparedExecution = arm == .on ? c.on : c.off
        let pre = preconditions(arm, c, prepared: prepared)
        var prefillWall = -1.0
        var walls = [Double]()
        var g = RouterOverlapGateGreedy()
        var scopeError: String? = nil
        if pre.isEmpty {
            func prefill() throws -> Step {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt,
                                                      candidate: c.candidate, prepared: prepared)
                let t1 = DispatchTime.now().uptimeNanoseconds
                prefillWall = micros(t0, t1)
                return out
            }
            func next(_ token: Int) throws -> Step {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token,
                                                              candidate: c.candidate, prepared: prepared)
                let t1 = DispatchTime.now().uptimeNanoseconds
                walls.append(micros(t0, t1))
                return out
            }
            func ignore(_ step: Int, _ logits: [Float]) {}
            if arm == .on {
                do {
                    try r.withRouterOverlapDiagnosticScope { () -> Void in
                        g = greedy(r, outputs: proto.outputsPerRun, prefill: prefill, next: next,
                                   observe: ignore)
                    }
                } catch {
                    scopeError = String(describing: error)
                }
            } else {
                g = greedy(r, outputs: proto.outputsPerRun, prefill: prefill, next: next, observe: ignore)
            }
            r.reset()
        }
        let counts = phaseCounts(g, c.derived, arm, promptCount: fx.prompt.count)
        let relation = totalsRelation(candidate: g.totals, baseline: qualifiedOff,
                                      extraBuffersPerForward: arm == .on ? c.derived.layers : 0,
                                      promptCount: fx.prompt.count)
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if let s = scopeError { f.append("scope: " + s) }
            if g.ids != fx.frozenIds { f.append("output IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != proto.outputsPerRun {
                f.append("not 32 full-vocabulary finite outputs")
            }
            if !(prefillWall.isFinite && prefillWall > 0) { f.append("prefill wall invalid") }
            if runTPOT(walls) == nil { f.append("continuation walls are not 31 positive values") }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            f += relation
            if !quiescent(r) { f.append("runner not quiescent after the run") }
        }
        return RouterOverlapGatePerfRun(
            fixture: fx.name, arm: arm, warmup: warmup, block: block, orderInBlock: order,
            prefillWallMicros: prefillWall, continuationWallMicros: walls,
            tpotMedianWallMicros: runTPOT(walls), totals: g.totals, ids: g.ids,
            idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs, allFinite: g.allFinite,
            counts: counts, totalsMismatches: relation, preconditionFailures: pre,
            scopeError: scopeError, error: g.error, checksPassed: f.isEmpty,
            failure: f.isEmpty ? nil : f.joined(separator: "; "))
    }

    // MARK: Entry point

    /// Runs the gate on a freshly loaded resident runner. Never throws: any
    /// failure is recorded in the report. The caller prints TIMED_WINDOW_END
    /// after this returns and quiescent(runner) holds.
    public static func run(mode: RouterOverlapModelGateMode, region: MoERegionIR, graph: GemmaGraphIR,
                           fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner,
                           modelSha256: String, executableDigest: String,
                           emit: (String) -> Void) -> RouterOverlapGateReport {
        var report = RouterOverlapGateReport(mode: mode)
        let proto = RouterOverlapGateProtocol.required
        let pristineStart = pristine(runner)
        let instrumentationStart = instrumentationOff(runner) && axisFlagsClean(runner)
        var setup = [String]()
        if !pristineStart { setup.append("runner is not pristine at start") }
        if !instrumentationStart { setup.append("instrumentation or axis flags are not off at start") }
        if fixtures != RouterOverlapGateFixtures.pinned { setup.append("fixtures differ from the pinned fixtures") }
        for fx in fixtures where fx.prompt.isEmpty
            || fx.prompt.count + proto.outputsPerRun > runner.maxContext
            || !(fx.prompt + fx.frozenIds).allSatisfy({ $0 >= 0 && $0 < runner.config.vocabSize }) {
            setup.append("fixture " + fx.name + " outside context/vocabulary bounds")
        }
        if !moeIsLowercaseHex64(modelSha256) || !moeIsLowercaseHex64(executableDigest) {
            setup.append("model and executable digests must be lowercase 64-hex")
        }
        guard setup.isEmpty else {
            report.abortReason = setup.joined(separator: "; ")
            report.restoredPristine = pristine(runner)
            report.status = .aborted
            emit("aborted before any GPU work: " + (report.abortReason ?? ""))
            return report
        }
        do {
            try runner.withQ4DiagnosticReferenceWidth(proto.q4LaunchWidth) { () -> Void in
                try scoped(&report, mode: mode, region: region, graph: graph, fixtures: fixtures,
                           runner: runner, modelSha256: modelSha256,
                           executableDigest: executableDigest, pristineStart: pristineStart,
                           instrumentationStart: instrumentationStart, emit: emit)
            }
        } catch {
            report.abortReason = (report.abortReason.map { $0 + "; " } ?? "")
                + "threw: " + String(describing: error)
        }
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = pristine(runner) && instrumentationOff(runner) && axisFlagsClean(runner)
        report.status = finalStatus(report)
        emit("status " + report.status.rawValue + "; restored pristine " + String(report.restoredPristine)
             + "; gainQualified false; " + externalResourceEligibility)
        return report
    }

    static func scoped(_ report: inout RouterOverlapGateReport, mode: RouterOverlapModelGateMode,
                       region: MoERegionIR, graph: GemmaGraphIR,
                       fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner,
                       modelSha256: String, executableDigest: String, pristineStart: Bool,
                       instrumentationStart: Bool, emit: (String) -> Void) throws {
        let proto = RouterOverlapGateProtocol.required
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == proto.q4LaunchWidth else {
            throw RouterOverlapGateError("Q4 reference width 32 is not in effect inside the width scope")
        }
        var prepared = try installAndPrepare(runner, region: region, graph: graph)
        let derived = derive(on: prepared.on, runner: runner)
        if derived.statisticsGroups > 0 {
            // Allocation only (nothing open, nothing dispatched): pre-sizes the
            // existing statistics buffer so its lazy growth never flushes an
            // open segment mid-token and buffer totals stay stationary.
            _ = try runner.rmsStatisticsInverse(slots: derived.statisticsGroups)
        }
        var ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
        let identity = try makeIdentity(ctx, modelSha256: modelSha256,
                                        executableDigest: executableDigest,
                                        pristineAtStart: pristineStart,
                                        instrumentationOffAtStart: instrumentationStart)
        report.identity = identity
        report.identityFailures = identityFailures(identity)
        emit(event("identity", identity))
        guard report.identityFailures.isEmpty else {
            emit("identity FAILED: " + report.identityFailures.joined(separator: "; ")
                 + "; nothing is executed on the GPU")
            if mode == .performance { emit(qualificationFailedToken) }
            return
        }

        let guards = refusalTests(ctx, prompt: fixtures[0].prompt)
        report.refusalTests = guards.tests
        report.refusalScopeError = guards.scopeError
        report.refusalPassed = guards.scopeError == nil
            && guards.tests.map({ $0.name }) == refusalTestNames
            && guards.tests.allSatisfy({ $0.passed })
        emit(event("refusalTests", guards.tests))
        guard report.refusalPassed else {
            emit("guard refusal tests FAILED; qualification not run")
            if mode == .performance { emit(qualificationFailedToken) }
            return
        }

        let qStart = now()
        report.qualificationStartUptimeNanos = qStart
        var offTotals = [String: RouterOverlapGateTotals]()
        var fatal: String? = nil
        for fx in fixtures {
            runner.uninstallMoeGeneratedLowering()
            let ref = qualify(.reference, ctx, fx, reference: [], off: [], offTotals: nil, keep: true)
            prepared = try installAndPrepare(runner, region: region, graph: graph)
            guard derive(on: prepared.on, runner: runner) == derived else {
                fatal = "derived counts changed after re-preparation"
                break
            }
            ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
            let offResult = ref.record.passed
                ? qualify(.off, ctx, fx, reference: ref.kept, off: [], offTotals: nil, keep: true)
                : notRun(.off, "no valid fresh reference")
            let onResult = (ref.record.passed && offResult.kept.count == proto.outputsPerRun)
                ? qualify(.on, ctx, fx, reference: ref.kept, off: offResult.kept,
                          offTotals: offResult.totals, keep: false)
                : notRun(.on, "no valid fresh reference or OFF did not produce 32 finite outputs")
            let record = RouterOverlapGateQualificationFixture(
                fixture: fx.name, reference: ref.record, off: offResult.record, on: onResult.record,
                passed: ref.record.passed && offResult.record.passed && onResult.record.passed)
            report.qualification.append(record)
            offTotals[fx.name] = offResult.totals
            emit(event("qualificationFixture", record))
            if !quiescent(runner) || !installFailures(runner, region: region).isEmpty {
                fatal = "runner not quiescent or install displaced after fixture " + fx.name
                break
            }
        }
        let qEnd = now()
        report.qualificationEndUptimeNanos = qEnd
        report.qualificationElapsedMicros = micros(qStart, qEnd)
        if let fatal { report.abortReason = fatal }
        report.qualificationPassed = fatal == nil
            && report.qualification.map({ $0.fixture }) == proto.fixtures
            && report.qualification.allSatisfy({ $0.passed })
        guard mode == .performance else { return }
        guard report.qualificationPassed else {
            emit(qualificationFailedToken)
            return
        }
        emit(qualificationPassedToken)

        report.performanceStartUptimeNanos = now()
        for fx in fixtures {
            guard let qOff = offTotals[fx.name] else {
                report.abortReason = "missing qualified OFF totals for " + fx.name
                break
            }
            var runs = [RouterOverlapGatePerfRun]()
            for arm in [RouterOverlapGateArm.off, .on] {
                let run = measuredRun(ctx, fx, arm, warmup: true, block: nil, order: nil, qualifiedOff: qOff)
                emit(event("run", run))
                runs.append(run)
            }
            for b in 0..<proto.blocksPerFixture {
                for (o, arm) in blockOrder(b).enumerated() {
                    let run = measuredRun(ctx, fx, arm, warmup: false, block: b, order: o, qualifiedOff: qOff)
                    emit(event("run", run))
                    runs.append(run)
                }
            }
            report.runs.append(contentsOf: runs)
            let outcome = fixtureOutcome(fixture: fx.name, runs: runs)
            report.fixtures.append(outcome)
            emit(event("fixtureOutcome", outcome))
        }
        report.performanceEndUptimeNanos = now()
    }
}
