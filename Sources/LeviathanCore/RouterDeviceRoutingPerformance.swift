import Foundation

// Phase 58: fixed ABBA performance protocol for the qualified default-off GPU device-routing candidate (ON: device
// routing preparation inside withRouterDeviceRoutingScope, captureSnapshots false) against the OFF incumbent (control
// preparation, CPU FP64 routing). One executable, one runner, one v1 E4R32 install, Q4 width 32 and the declared
// modern candidate for both arms; only the device-routing preparation and its scope differ.
// Performance runs only in a separate invocation of the SAME binary after its own numerical qualification
// (RouterDeviceRoutingModelGate, unchanged) is validated against this executable's digest, the hashed qualified
// executable, the cached model SHA, the qualified selector source and freshly validated prerequisite receipts.
// Per fixture (en-chat, kr-chat, raw-hello): one warmup OFF then ON, then seven fixed blocks (even OFF/ON, odd
// ON/OFF). Timers bracket only executeGraph(prompt) and each of the 31 executeGraphContinue(previous actual argmax)
// calls. No retry, reseed, outlier drop, trimming or count change. Diagnostic prototype only: never installable, no
// promotion, no default or artifact change. No scales, coefficient bits or weights enter the report.

public enum RouterDeviceRoutingPerfArm: String, Codable, Sendable {
    case off = "off-control-prepared-cpu-routing-modern-v1E4R32-q4w32"
    case on = "on-device-routing-scope-modern-v1E4R32-q4w32"
}

public enum RouterDeviceRoutingBenchmarkMode: String, Codable, Sendable {
    case qualificationOnly = "qualification-only"
    case performanceOnly = "performance-only"
}

public enum RouterDeviceRoutingPerfStatus: String, Codable, Sendable {
    case performanceCriteriaMet
    case performanceRunCheckFailed
    case performanceUnstable
    case performanceNoGain
    case qualificationReceiptInvalid
    case aborted
    case runnerNotRestored
}

/// One declared run position: warmups carry no block; timed runs carry their block and their order in the block.
public struct RouterDeviceRoutingPerfSlot: Codable, Equatable, Sendable {
    public let arm: RouterDeviceRoutingPerfArm
    public let warmup: Bool
    public let block: Int?
    public let orderInBlock: Int?

    public init(arm: RouterDeviceRoutingPerfArm, warmup: Bool, block: Int?, orderInBlock: Int?) {
        self.arm = arm
        self.warmup = warmup
        self.block = block
        self.orderInBlock = orderInBlock
    }
}

public struct RouterDeviceRoutingPerfProtocol: Codable, Equatable, Sendable {
    public let fixtures: [String]
    public let outputsPerRun: Int
    public let timedContinuationsPerRun: Int
    public let q4LaunchWidth: Int
    public let layers: Int
    public let qualifiedFrames: Int
    public let warmupOrder: [RouterDeviceRoutingPerfArm]
    public let blocksPerFixture: Int
    public let blockOrders: [[RouterDeviceRoutingPerfArm]]
    public let runsPerFixture: Int
    public let totalRuns: Int
    public let timedRuns: Int
    public let offContinuationKernels: Int
    public let offContinuationCommandBuffers: Int
    public let onContinuationKernels: Int
    public let onContinuationCommandBuffers: Int
    public let captureSnapshotsInTimedRuns: Bool
    public let spreadLimit: Double
    public let gainThreshold: Double
    public let timer: String
    public let statistic: String
    public let outlierPolicy: String
    public let exclusions: String
    public let gpuElapsed: String

    public static let required = RouterDeviceRoutingPerfProtocol(
        fixtures: RouterOverlapGateFixtures.order, outputsPerRun: 32, timedContinuationsPerRun: 31, q4LaunchWidth: 32,
        layers: 30, qualifiedFrames: 4170, warmupOrder: [.off, .on], blocksPerFixture: 7,
        blockOrders: (0..<7).map { RouterDeviceRoutingPerformance.blockOrder($0) }, runsPerFixture: 16, totalRuns: 48,
        timedRuns: 42, offContinuationKernels: 1349, offContinuationCommandBuffers: 37, onContinuationKernels: 1379,
        onContinuationCommandBuffers: 7, captureSnapshotsInTimedRuns: false, spreadLimit: 0.05, gainThreshold: 0.02,
        timer: "wall clock (DispatchTime uptime) around exactly one executeGraphContinue(previous actual argmax) call "
            + "per continuation, 31 timers per run; each call includes encoding, GPU waits, command-buffer completion, "
            + "the full-vocabulary logits readback and argmax, and for ON the runtime selector dispatches, the "
            + "record-bound expert chains and the post-token record flag verification; prefill is timed separately "
            + "around executeGraph(prompt) and is not TPOT",
        statistic: "run TPOT = median of its 31 continuation walls; arm TPOT = median of its 7 timed run TPOTs per "
            + "fixture; spread = (max - min) / median per arm per fixture, required <= 0.05; gain = (OFF - ON) / OFF "
            + "per fixture, required >= 0.02 on every fixture",
        outlierPolicy: "none: every timed run is reported and used; no retry, rescue, reseed, adaptive stop, trimming, "
            + "outlier drop or count change; any failed run check fails the protocol",
        exclusions: "outside the timers only: model load, install, preparation, selector compile, scope entry and "
            + "exit, counter snapshots, finiteness and ID checks, progress lines and report serialization; warmups "
            + "are executed and checked but not used for statistics",
        gpuElapsed: "per-call command-buffer intervals (gpuEndTime - gpuStartTime) are recorded for diagnosis only; "
            + "they are not GPU busy time and are never compared between arms or used by any decision")
}

/// Minimal per-run input of the outcome statistics (pure; constructible in tests).
public struct RouterDeviceRoutingPerfSample: Codable, Equatable, Sendable {
    public let fixture: String
    public let slot: RouterDeviceRoutingPerfSlot
    public let tpotMedianWallMicros: Double?
    public let prefillWallMicros: Double
    public let checksPassed: Bool

    public init(fixture: String, slot: RouterDeviceRoutingPerfSlot, tpotMedianWallMicros: Double?,
                prefillWallMicros: Double, checksPassed: Bool) {
        self.fixture = fixture
        self.slot = slot
        self.tpotMedianWallMicros = tpotMedianWallMicros
        self.prefillWallMicros = prefillWallMicros
        self.checksPassed = checksPassed
    }
}

public struct RouterDeviceRoutingPerfRun: Codable, Equatable, Sendable {
    public let fixture: String
    public let arm: RouterDeviceRoutingPerfArm
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
    public let deviceCounters: RouterDeviceRoutingCounters
    public let deviceCounterFailures: [String]
    public let selectorSourceSha256: String?
    public let preconditionFailures: [String]
    public let scopeError: String?
    public let error: String?
    public let checksPassed: Bool
    public let failure: String?

    public var sample: RouterDeviceRoutingPerfSample {
        RouterDeviceRoutingPerfSample(
            fixture: fixture,
            slot: RouterDeviceRoutingPerfSlot(arm: arm, warmup: warmup, block: block, orderInBlock: orderInBlock),
            tpotMedianWallMicros: tpotMedianWallMicros, prefillWallMicros: prefillWallMicros,
            checksPassed: checksPassed)
    }
}

public struct RouterDeviceRoutingPerfFixtureOutcome: Codable, Equatable, Sendable {
    public let fixture: String
    public let offRunTPOTMicros: [Double]
    public let onRunTPOTMicros: [Double]
    public let offMedianTPOTMicros: Double?
    public let onMedianTPOTMicros: Double?
    public let offSpread: Double?
    public let onSpread: Double?
    public let offStable: Bool
    public let onStable: Bool
    public let gain: Double?
    public let gainMeetsThreshold: Bool
    public let offMedianPrefillWallMicros: Double?
    public let onMedianPrefillWallMicros: Double?
    public let allRunChecksPassed: Bool
    public let outcome: String
}

/// Validated in process from the qualification report of THIS executable; digests and counts only.
public struct RouterDeviceRoutingQualificationReceipt: Codable, Equatable, Sendable {
    public let qualificationReportSha256: String
    public let qualifiedExecutableSha256: String
    public let executableSha256: String
    public let modelSha256: String
    public let reportStatus: String?
    public let reportExecutableDigest: String?
    public let selectorSourceSha256: String?
    public let framesExact: Int
    public let replayIdenticalFrames: Int
    public let prerequisites: RouterDeviceRoutingPrerequisiteReceipt
    public let failures: [String]

    public var valid: Bool { failures.isEmpty && prerequisites.valid }
}

public struct RouterDeviceRoutingBenchmarkOptions: Equatable, Sendable {
    public let mode: RouterDeviceRoutingBenchmarkMode
    public let gguf: String
    public let region: String
    public let graph: String
    public let fixtures: String
    public let componentReport: String
    public let componentExecutable: String
    public let selectionReport: String
    public let selectionExecutable: String
    public let qualifiedReport: String?
    public let qualifiedExecutable: String?
    public let report: String
}

public struct RouterDeviceRoutingPerfReport: Codable {
    public private(set) var kind = RouterDeviceRoutingPerformance.kind
    public private(set) var protocolSpec = RouterDeviceRoutingPerfProtocol.required
    public private(set) var candidateFlags = RouterOverlapModelGate.candidateFlags(RouterOverlapModelGate.modernCandidate)
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var scalesSerialized = false
    public private(set) var realE4GateComplete = false
    public private(set) var externalResourceEligibility = RouterDeviceRoutingPerformance.externalResourceEligibility
    public private(set) var notes = RouterDeviceRoutingPerformance.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var qualification: RouterDeviceRoutingQualificationReceipt? = nil
    public var identity: RouterOverlapGateIdentity? = nil
    public var identityFailures: [String] = []
    public var setupSelectorSourceSha256: String? = nil
    public var performanceStartUptimeNanos: UInt64? = nil
    public var performanceEndUptimeNanos: UInt64? = nil
    public var runs: [RouterDeviceRoutingPerfRun] = []
    public var fixtures: [RouterDeviceRoutingPerfFixtureOutcome] = []
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: RouterDeviceRoutingPerfStatus = .aborted

    public init() {}
}

extension RouterDeviceRoutingPerfReport {
    /// Contract normalization at every encode: installable, productionPerformanceClaim and scalesSerialized are
    /// false; realE4GateComplete comes only from the validated qualification receipt; gainQualified is recomputed
    /// from the raw runs and the fixed protocol, never taken from input.
    func withPerformanceContract() -> RouterDeviceRoutingPerfReport {
        var copy = self
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.scalesSerialized = false
        copy.realE4GateComplete = copy.qualification?.valid == true
        let samples = copy.runs.map { $0.sample }
        let recomputed = RouterDeviceRoutingPerfProtocol.required.fixtures.map {
            RouterDeviceRoutingPerformance.fixtureOutcome(fixture: $0, samples: samples)
        }
        copy.gainQualified = copy.realE4GateComplete && copy.restoredPristine && copy.abortReason == nil
            && copy.identity != nil && copy.identityFailures.isEmpty
            && copy.protocolSpec == RouterDeviceRoutingPerfProtocol.required
            && copy.status == .performanceCriteriaMet && copy.fixtures == recomputed
            && RouterDeviceRoutingPerformance.performanceStatus(recomputed) == .performanceCriteriaMet
        return copy
    }
}

public enum RouterDeviceRoutingPerformance {
    public static let kind = "router-device-routing-fixed-abba-performance-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 16 << 20
    public static let outcomeRunCheckFailed = "runCheckFailed"
    public static let outcomeUnstable = "unstable"
    public static let outcomeNoGain = "noGain"
    public static let outcomeCriteriaMet = "criteriaMet"
    public static let externalResourceEligibility = "PENDING: decided by the supervisor's resource receipts before "
        + "GO and after TIMED_WINDOW_END; gainQualified is a relative internal prototype criterion only"
    public static let usage = "usage: router-device-routing-benchmark --mode qualification-only|performance-only "
        + "--gguf M.gguf --region R.json --graph G.json --fixtures F.json --component-report C.json "
        + "--component-executable C_BIN --selection-report S.json --selection-executable S_BIN "
        + "[--qualified-report Q.json --qualified-executable THIS_BIN (performance-only)] --report NEW_REPORT.json"

    public static let notes: [String] = [
        "Diagnostic prototype only: installable, productionPerformanceClaim and scalesSerialized are always false; "
            + "no artifact, installation, promotion or default change.",
        "realE4GateComplete is recomputed at encode time from the validated qualification receipt of THIS executable. "
            + "gainQualified is recomputed from the raw runs: true only when every run of the fixed sequence passed its "
            + "checks and every fixture met spread <= 5% per arm and gain >= 2%.",
        "Both arms: same binary, runner, v1 E4R32 install, Q4 width 32 and declared modern candidate. OFF = control "
            + "preparation with CPU routing; ON = device-routing preparation in one scope per run with captureSnapshots "
            + "false (no CPU FP64 recompute or component replay in any timed run).",
        "Each run is checked outside the timers: 32 frozen IDs, 32 full-vocabulary finite outputs, exact derived "
            + "counters with every fallback 0, totals equal to the qualified OFF totals (OFF) or OFF + 30 kernels and "
            + "OFF - 30 command buffers per forward (ON), pinned 1349/37 (OFF) and 1379/7 (ON) per continuation, "
            + "CPU/GPU selection, record-chain and accepted-token counters, the qualified selector source and an "
            + "intact install.",
        "Command-buffer intervals are diagnostic only: not GPU busy time and never compared between arms.",
        "Measured under the machine's actual normal concurrency: the tool neither waits for nor accepts a quiet "
            + "acknowledgement; GO and the resource receipts before GO and after TIMED_WINDOW_END belong to the "
            + "external supervisor. Progress lines between runs add journaling overhead outside the timers.",
        "Model SHA-256 is the cached explicit-refresh record; this tool never hashes, re-verifies or downloads the model.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    // MARK: Pure helpers

    /// Fixed block order: even blocks OFF then ON, odd blocks ON then OFF.
    public static func blockOrder(_ block: Int) -> [RouterDeviceRoutingPerfArm] {
        block % 2 == 0 ? [.off, .on] : [.on, .off]
    }

    /// The exact per-fixture run sequence: warmup OFF, warmup ON, then blocks 0..6.
    public static func declaredSequence() -> [RouterDeviceRoutingPerfSlot] {
        let p = RouterDeviceRoutingPerfProtocol.required
        var s = p.warmupOrder.map { RouterDeviceRoutingPerfSlot(arm: $0, warmup: true, block: nil, orderInBlock: nil) }
        for b in 0..<p.blocksPerFixture {
            for (o, arm) in blockOrder(b).enumerated() {
                s.append(RouterDeviceRoutingPerfSlot(arm: arm, warmup: false, block: b, orderInBlock: o))
            }
        }
        return s
    }

    public static func spread(_ v: [Double]) -> Double? {
        guard v.count == RouterDeviceRoutingPerfProtocol.required.blocksPerFixture,
              v.allSatisfy({ $0.isFinite && $0 > 0 }), let m = RouterOverlapModelGate.median(v), m > 0,
              let lo = v.min(), let hi = v.max() else { return nil }
        return (hi - lo) / m
    }

    public static func fixtureOutcome(fixture: String,
                                      samples: [RouterDeviceRoutingPerfSample]) -> RouterDeviceRoutingPerfFixtureOutcome {
        let p = RouterDeviceRoutingPerfProtocol.required
        let mine = samples.filter { $0.fixture == fixture }
        let off = mine.filter { !$0.slot.warmup && $0.slot.arm == .off }
        let on = mine.filter { !$0.slot.warmup && $0.slot.arm == .on }
        let checks = mine.map({ $0.slot }) == declaredSequence() && mine.allSatisfy({ $0.checksPassed })
            && (off + on).allSatisfy({ $0.tpotMedianWallMicros != nil })
        let offT = off.compactMap { $0.tpotMedianWallMicros }
        let onT = on.compactMap { $0.tpotMedianWallMicros }
        let offMed = RouterOverlapModelGate.median(offT)
        let onMed = RouterOverlapModelGate.median(onT)
        let offSpread = spread(offT)
        let onSpread = spread(onT)
        let offStable = offSpread.map { $0 <= p.spreadLimit } ?? false
        let onStable = onSpread.map { $0 <= p.spreadLimit } ?? false
        let g = RouterOverlapModelGate.gain(baseline: offMed, candidate: onMed)
        let meets = g.map { $0 >= p.gainThreshold } ?? false
        let outcome: String
        if !checks {
            outcome = outcomeRunCheckFailed
        } else if !(offStable && onStable) {
            outcome = outcomeUnstable
        } else if !meets {
            outcome = outcomeNoGain
        } else {
            outcome = outcomeCriteriaMet
        }
        return RouterDeviceRoutingPerfFixtureOutcome(
            fixture: fixture, offRunTPOTMicros: offT, onRunTPOTMicros: onT, offMedianTPOTMicros: offMed,
            onMedianTPOTMicros: onMed, offSpread: offSpread, onSpread: onSpread, offStable: offStable,
            onStable: onStable, gain: g, gainMeetsThreshold: meets,
            offMedianPrefillWallMicros: RouterOverlapModelGate.median(off.map { $0.prefillWallMicros }),
            onMedianPrefillWallMicros: RouterOverlapModelGate.median(on.map { $0.prefillWallMicros }),
            allRunChecksPassed: checks, outcome: outcome)
    }

    public static func performanceStatus(_ outcomes: [RouterDeviceRoutingPerfFixtureOutcome]) -> RouterDeviceRoutingPerfStatus {
        guard outcomes.map({ $0.fixture }) == RouterDeviceRoutingPerfProtocol.required.fixtures,
              !outcomes.contains(where: { $0.outcome == outcomeRunCheckFailed }) else {
            return .performanceRunCheckFailed
        }
        if outcomes.contains(where: { $0.outcome == outcomeUnstable }) { return .performanceUnstable }
        if outcomes.contains(where: { $0.outcome == outcomeNoGain }) { return .performanceNoGain }
        return .performanceCriteriaMet
    }

    /// Per-continuation kernels and command buffers pinned by the actual qualification (OFF 1349/37, ON 1379/7).
    public static func pinnedTotalsFailures(_ t: RouterOverlapGateTotals, arm: RouterDeviceRoutingPerfArm) -> [String] {
        let p = RouterDeviceRoutingPerfProtocol.required
        let kernels = arm == .on ? p.onContinuationKernels : p.offContinuationKernels
        let buffers = arm == .on ? p.onContinuationCommandBuffers : p.offContinuationCommandBuffers
        guard t.continuationKernels.count == p.timedContinuationsPerRun,
              t.continuationBuffers.count == p.timedContinuationsPerRun else {
            return [arm.rawValue + ": continuation totals do not have 31 entries"]
        }
        var f = [String]()
        if !t.continuationKernels.allSatisfy({ $0 == kernels }) {
            f.append(arm.rawValue + ": continuation kernels are not " + String(kernels) + " per forward")
        }
        if !t.continuationBuffers.allSatisfy({ $0 == buffers }) {
            f.append(arm.rawValue + ": continuation command buffers are not " + String(buffers) + " per forward")
        }
        return f
    }

    /// OFF: layers CPU selections per forward and no device routing; ON: 0 CPU selections, layers GPU selections and
    /// record chains per forward, every token accepted, none failed or flagged.
    public static func deviceCounterFailures(_ d: RouterDeviceRoutingCounters, arm: RouterDeviceRoutingPerfArm,
                                             layers: Int, forwards: Int) -> [String] {
        let cpu = arm == .on ? 0 : layers * forwards
        let gpu = arm == .on ? layers * forwards : 0
        let accepted = arm == .on ? forwards : 0
        var f = [String]()
        if d.cpuSelections != cpu {
            f.append(arm.rawValue + ": CPU selections " + String(d.cpuSelections) + " != " + String(cpu))
        }
        if d.gpuSelections != gpu {
            f.append(arm.rawValue + ": GPU selections " + String(d.gpuSelections) + " != " + String(gpu))
        }
        if d.recordChains != gpu {
            f.append(arm.rawValue + ": record chains " + String(d.recordChains) + " != " + String(gpu))
        }
        if d.acceptedTokens != accepted {
            f.append(arm.rawValue + ": accepted tokens " + String(d.acceptedTokens) + " != " + String(accepted))
        }
        if d.failedTokens != 0 || d.flaggedTokens != 0 { f.append(arm.rawValue + ": failed or flagged tokens") }
        return f
    }

    public static func parseOptions(_ argv: [String]) throws -> RouterDeviceRoutingBenchmarkOptions {
        let common = ["--mode", "--gguf", "--region", "--graph", "--fixtures", "--component-report",
                      "--component-executable", "--selection-report", "--selection-executable", "--report"]
        let qualifiedKeys = ["--qualified-report", "--qualified-executable"]
        var v = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard common.contains(key) || qualifiedKeys.contains(key), i + 1 < argv.count, v[key] == nil else {
                throw RouterDeviceRoutingError(usage)
            }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw RouterDeviceRoutingError(usage) }
            v[key] = value
            i += 2
        }
        guard let modeText = v["--mode"], let mode = RouterDeviceRoutingBenchmarkMode(rawValue: modeText),
              common.allSatisfy({ v[$0] != nil }) else {
            throw RouterDeviceRoutingError(usage)
        }
        let present = qualifiedKeys.map { v[$0] != nil }
        switch mode {
        case .qualificationOnly:
            guard present == [false, false] else { throw RouterDeviceRoutingError(usage) }
        case .performanceOnly:
            guard present == [true, true] else { throw RouterDeviceRoutingError(usage) }
        }
        return RouterDeviceRoutingBenchmarkOptions(
            mode: mode, gguf: v["--gguf"] ?? "", region: v["--region"] ?? "", graph: v["--graph"] ?? "",
            fixtures: v["--fixtures"] ?? "", componentReport: v["--component-report"] ?? "",
            componentExecutable: v["--component-executable"] ?? "", selectionReport: v["--selection-report"] ?? "",
            selectionExecutable: v["--selection-executable"] ?? "", qualifiedReport: v["--qualified-report"],
            qualifiedExecutable: v["--qualified-executable"], report: v["--report"] ?? "")
    }

    public static func encodeReport(_ r: RouterDeviceRoutingPerfReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withPerformanceContract())
    }

    // MARK: Qualification receipt

    /// Validates the numerical qualification report produced by THIS executable. Pure: the caller reads the small
    /// report file, hashes the qualified executable and supplies the freshly validated prerequisite receipt.
    public static func validateQualification(report data: Data, qualifiedExecutableSha256: String,
                                             executableSha256: String, modelSha256: String,
                                             prerequisites: RouterDeviceRoutingPrerequisiteReceipt)
        -> (receipt: RouterDeviceRoutingQualificationReceipt, report: RouterDeviceRoutingGateReport?) {
        let p = RouterDeviceRoutingPerfProtocol.required
        var f = [String]()
        if !prerequisites.valid { f.append("prerequisite receipt invalid") }
        if !moeIsLowercaseHex64(qualifiedExecutableSha256) || !moeIsLowercaseHex64(executableSha256)
            || !moeIsLowercaseHex64(modelSha256) {
            f.append("executable and model digests must be lowercase 64-hex")
        }
        if qualifiedExecutableSha256 != executableSha256 { f.append("qualified executable differs from this executable") }
        var decoded: RouterDeviceRoutingGateReport? = nil
        if data.count > maxReportBytes {
            f.append("qualification report exceeds 16 MiB")
        } else {
            decoded = try? JSONDecoder().decode(RouterDeviceRoutingGateReport.self, from: data)
            if decoded == nil { f.append("qualification report does not decode") }
        }
        if let q = decoded {
            var recomputed = q
            RouterDeviceRoutingModelGate.finalize(&recomputed)
            if q.kind != RouterDeviceRoutingModelGate.kind { f.append("qualification report kind differs") }
            if q.status != .qualificationPassed || recomputed.status != .qualificationPassed
                || !recomputed.qualificationPassed || !recomputed.realE4GateComplete {
                f.append("numerical qualification did not pass (recomputed)")
            }
            if q.executableDigest != executableSha256 {
                f.append("qualification report was produced by a different executable")
            }
            if q.modelSha256 != modelSha256 { f.append("qualification report model SHA differs") }
            if q.prerequisites != prerequisites {
                f.append("qualification prerequisite receipt differs from the freshly validated receipt")
            }
            if q.selectorSourceSha256 == nil || q.selectorSourceSha256 != prerequisites.expectedSelectorSourceSha256 {
                f.append("qualified selector source differs from this build")
            }
            if q.expectedFrames != p.qualifiedFrames || q.framesChecked != p.qualifiedFrames
                || q.framesExact != p.qualifiedFrames {
                f.append("qualified frames are not 4170 checked and exact")
            }
            if q.replay?.terminal != true || q.replay?.completion != "COMPLETED"
                || q.replay?.identicalFrames != p.qualifiedFrames || q.replay?.mismatchCount != 0 {
                f.append("component replay did not cover 4170 identical frames")
            }
            if !q.refusalPassed || q.refusalTests.map({ $0.name }) != RouterDeviceRoutingModelGate.refusalTestNames {
                f.append("refusal tests did not all pass")
            }
            if !q.restoredPristine || q.abortReason != nil { f.append("qualification runner not restored or aborted") }
            if let id = q.identity {
                if id.executableDigest != executableSha256 || id.modelSha256 != modelSha256
                    || id.derived.layers != p.layers || !RouterOverlapModelGate.identityFailures(id).isEmpty
                    || !q.identityFailures.isEmpty {
                    f.append("qualified identity invalid or bound to another executable or model")
                }
            } else {
                f.append("qualified identity missing")
            }
            f += qualifiedFixtureFailures(q.fixtures)
        }
        let receipt = RouterDeviceRoutingQualificationReceipt(
            qualificationReportSha256: RouterSelectionKernel.sha256Hex(data),
            qualifiedExecutableSha256: qualifiedExecutableSha256, executableSha256: executableSha256,
            modelSha256: modelSha256, reportStatus: decoded?.status.rawValue,
            reportExecutableDigest: decoded.flatMap { $0.executableDigest },
            selectorSourceSha256: decoded.flatMap { $0.selectorSourceSha256 },
            framesExact: decoded?.framesExact ?? 0, replayIdenticalFrames: decoded?.replay?.identicalFrames ?? 0,
            prerequisites: prerequisites, failures: f)
        return (receipt, receipt.valid ? decoded : nil)
    }

    static func qualifiedFixtureFailures(_ fixtures: [RouterDeviceRoutingGateFixture]) -> [String] {
        let p = RouterDeviceRoutingPerfProtocol.required
        var f = [String]()
        if fixtures.map({ $0.fixture }) != p.fixtures {
            f.append("qualified fixtures are not exactly en-chat, kr-chat, raw-hello")
        }
        for fx in fixtures {
            guard let on = fx.on, fx.passed, fx.off.passed, on.passed, fx.offExtraFailures.isEmpty else {
                f.append(fx.fixture + ": qualified fixture did not pass")
                continue
            }
            let prompt = RouterOverlapGateFixtures.pinned.first(where: { $0.name == fx.fixture })?.prompt.count ?? -1
            let forwards = prompt + p.outputsPerRun - 1
            var g = pinnedTotalsFailures(fx.off.totals, arm: .off) + pinnedTotalsFailures(on.totals, arm: .on)
            g += RouterDeviceRoutingModelGate.deviceTotalsRelation(on: on.totals, off: fx.off.totals, layers: p.layers,
                                                                   promptCount: prompt)
            g += deviceCounterFailures(on.deviceCounters, arm: .on, layers: p.layers, forwards: forwards)
            if fx.offCPUSelections != p.layers * forwards { g.append("OFF CPU selections differ") }
            if on.snapshots != forwards || on.frameExactTokens != forwards { g.append("not every token frame exact") }
            f += g.map { fx.fixture + ": " + $0 }
        }
        return f
    }

    // MARK: Status

    public static func finalStatus(_ r: RouterDeviceRoutingPerfReport) -> RouterDeviceRoutingPerfStatus {
        if !r.restoredPristine { return .runnerNotRestored }
        if r.abortReason != nil { return .aborted }
        guard r.qualification?.valid == true else { return .qualificationReceiptInvalid }
        if r.identity == nil || !r.identityFailures.isEmpty { return .aborted }
        return performanceStatus(r.fixtures)
    }

    static func finalize(_ r: inout RouterDeviceRoutingPerfReport) {
        r.status = finalStatus(r)
        r = r.withPerformanceContract()
    }

    public static func loadFailed(_ reason: String, modelSha256: String, executableDigest: String,
                                  qualification: RouterDeviceRoutingQualificationReceipt?) -> RouterDeviceRoutingPerfReport {
        var r = RouterDeviceRoutingPerfReport()
        r.modelSha256 = modelSha256
        r.executableDigest = executableDigest
        r.qualification = qualification
        r.abortReason = reason
        r.restoredPristine = true
        finalize(&r)
        return r
    }

    // MARK: Entry point

    /// Never throws; every failure is recorded. The caller prints END only when the runner is quiescent and restored.
    public static func run(region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                           runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                           qualification: RouterDeviceRoutingQualificationReceipt,
                           qualified: RouterDeviceRoutingGateReport?,
                           emit: (String) -> Void) -> RouterDeviceRoutingPerfReport {
        let p = RouterDeviceRoutingPerfProtocol.required
        var report = RouterDeviceRoutingPerfReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.qualification = qualification
        let pristineStart = RouterOverlapModelGate.pristine(runner)
        let instrumentationStart = RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner) && !runner.routerDeviceRoutingActive
            && !runner.routeCaptureActive
        var setup = [String]()
        if !qualification.valid || qualified == nil { setup.append("qualification receipt invalid") }
        if qualification.executableSha256 != executableDigest || qualification.modelSha256 != modelSha256 {
            setup.append("qualification receipt is bound to another executable or model")
        }
        if !pristineStart { setup.append("runner is not pristine at start") }
        if !instrumentationStart { setup.append("instrumentation, axis flags or scopes are not off at start") }
        if fixtures != RouterOverlapGateFixtures.pinned { setup.append("fixtures differ from the pinned fixtures") }
        if runner.config.layerCount != p.layers { setup.append("model layer count differs from the protocol") }
        if !moeIsLowercaseHex64(modelSha256) || !moeIsLowercaseHex64(executableDigest) {
            setup.append("model and executable digests must be lowercase 64-hex")
        }
        guard setup.isEmpty, let q = qualified else {
            report.abortReason = setup.isEmpty ? "no qualified report" : setup.joined(separator: "; ")
            report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            finalize(&report)
            emit("aborted before any GPU work: " + (report.abortReason ?? ""))
            return report
        }
        do {
            try runner.withQ4DiagnosticReferenceWidth(p.q4LaunchWidth) { () -> Void in
                try scoped(&report, region: region, graph: graph, fixtures: fixtures, runner: runner, qualified: q,
                           modelSha256: modelSha256, executableDigest: executableDigest, pristineStart: pristineStart,
                           instrumentationStart: instrumentationStart, emit: emit)
            }
        } catch {
            report.abortReason = (report.abortReason.map { $0 + "; " } ?? "") + "benchmark threw: "
                + String(describing: error)
        }
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            && RouterOverlapModelGate.instrumentationOff(runner) && RouterOverlapModelGate.axisFlagsClean(runner)
            && !runner.routerDeviceRoutingActive
        finalize(&report)
        emit("status " + report.status.rawValue + "; gainQualified " + String(report.gainQualified)
             + "; realE4GateComplete " + String(report.realE4GateComplete) + "; restored pristine "
             + String(report.restoredPristine) + "; installable false; no production performance claim")
        return report
    }

    static func scoped(_ report: inout RouterDeviceRoutingPerfReport, region: MoERegionIR, graph: GemmaGraphIR,
                       fixtures: [RouterOverlapGateFixture], runner: Gemma4Runner,
                       qualified: RouterDeviceRoutingGateReport, modelSha256: String, executableDigest: String,
                       pristineStart: Bool, instrumentationStart: Bool, emit: (String) -> Void) throws {
        let p = RouterDeviceRoutingPerfProtocol.required
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == p.q4LaunchWidth else {
            throw RouterDeviceRoutingError("Q4 reference width 32 is not in effect inside the width scope")
        }
        let prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
        let derived = RouterOverlapModelGate.derive(on: prepared.on, runner: runner)
        if derived.statisticsGroups > 0 {
            _ = try runner.rmsStatisticsInverse(slots: derived.statisticsGroups)
        }
        let ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
        let device = try GemmaPreparedExecution.prepareRouterDeviceRouting(graph: graph, runner: runner)
        let identity = try RouterOverlapModelGate.makeIdentity(
            ctx, modelSha256: modelSha256, executableDigest: executableDigest,
            pristineAtStart: pristineStart, instrumentationOffAtStart: instrumentationStart)
        report.identity = identity
        var failures = RouterOverlapModelGate.identityFailures(identity)
        if qualified.identity != identity { failures.append("identity differs from the qualified identity of this executable") }
        if derived.layers != p.layers { failures.append("derived layer count differs from the protocol") }
        report.identityFailures = failures
        emit(RouterOverlapModelGate.event("identity", identity))
        guard failures.isEmpty else {
            emit("identity FAILED; no timed run executed")
            return
        }
        // Setup (outside every timer): compiles the selector pipeline and allocates the record buffers once.
        try runner.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Void in }
        let setupSource = runner.deviceRoutingState?.sourceSha256
        report.setupSelectorSourceSha256 = setupSource
        guard let expected = qualified.selectorSourceSha256, setupSource == expected else {
            throw RouterDeviceRoutingError("runtime selector source differs from the qualified selector source")
        }
        guard RouterOverlapModelGate.quiescent(runner), !runner.routerDeviceRoutingActive else {
            throw RouterDeviceRoutingError("runner not quiescent after setup")
        }
        var qualifiedOff = [String: RouterOverlapGateTotals]()
        for f in qualified.fixtures { qualifiedOff[f.fixture] = f.off.totals }
        report.performanceStartUptimeNanos = RouterOverlapModelGate.now()
        for fx in fixtures {
            guard let qOff = qualifiedOff[fx.name] else {
                throw RouterDeviceRoutingError("missing qualified OFF totals for " + fx.name)
            }
            var runs = [RouterDeviceRoutingPerfRun]()
            for slot in declaredSequence() {
                let run = measuredRun(ctx, device: device, region: region, fx, slot: slot, qualifiedOff: qOff,
                                      expectedSource: expected)
                runs.append(run)
                report.runs.append(run)
                emit(RouterOverlapModelGate.event("run", run))
                if !RouterOverlapModelGate.quiescent(runner) || runner.routerDeviceRoutingActive {
                    report.performanceEndUptimeNanos = RouterOverlapModelGate.now()
                    throw RouterDeviceRoutingError("runner not quiescent after a run; benchmark stopped")
                }
            }
            let outcome = fixtureOutcome(fixture: fx.name, samples: runs.map { $0.sample })
            report.fixtures.append(outcome)
            emit(RouterOverlapModelGate.event("fixtureOutcome", outcome))
        }
        report.performanceEndUptimeNanos = RouterOverlapModelGate.now()
    }

    // MARK: One normal 32-output run

    /// Only executeGraph(prompt) and each executeGraphContinue(previous actual argmax) are timed. The ON scope
    /// (captureSnapshots false) wraps the complete run outside the timers; every check runs after the run.
    static func measuredRun(_ c: RouterOverlapGateContext, device: GemmaPreparedExecution, region: MoERegionIR,
                            _ fx: RouterOverlapGateFixture, slot: RouterDeviceRoutingPerfSlot,
                            qualifiedOff: RouterOverlapGateTotals, expectedSource: String) -> RouterDeviceRoutingPerfRun {
        let r = c.runner
        let p = RouterDeviceRoutingPerfProtocol.required
        let arm = slot.arm
        let layers = c.derived.layers
        let forwards = fx.prompt.count + p.outputsPerRun - 1
        let prepared = arm == .on ? device : c.off
        var pre = RouterOverlapModelGate.preconditions(.off, c, prepared: c.off)
        if r.routerDeviceRoutingActive || r.routeCaptureActive {
            pre.append("device routing scope or route capture already active")
        }
        if arm == .on {
            if device.routerDeviceRouting == nil { pre.append("device preparation carries no device-routing binding") }
            if device.routerOverlap != nil { pre.append("device preparation carries an overlap plan") }
            pre += RouterOverlapModelGate.bindingFailures(device, c)
        }
        var prefillWall = -1.0
        var walls = [Double]()
        var g = RouterOverlapGateGreedy()
        var scopeError: String? = nil
        let before = r.routerDeviceRoutingCounters()
        if pre.isEmpty {
            func prefill() throws -> RouterOverlapModelGate.Step {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt, candidate: c.candidate,
                                                      prepared: prepared)
                let t1 = DispatchTime.now().uptimeNanoseconds
                prefillWall = RouterOverlapModelGate.micros(t0, t1)
                return out
            }
            func next(_ token: Int) throws -> RouterOverlapModelGate.Step {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token, candidate: c.candidate,
                                                              prepared: prepared)
                let t1 = DispatchTime.now().uptimeNanoseconds
                walls.append(RouterOverlapModelGate.micros(t0, t1))
                return out
            }
            func ignore(_ step: Int, _ logits: [Float]) {}
            r.reset()
            if arm == .on {
                do {
                    try r.withRouterDeviceRoutingScope(region: region,
                                                       captureSnapshots: p.captureSnapshotsInTimedRuns) { () -> Void in
                        g = RouterOverlapModelGate.greedy(r, outputs: p.outputsPerRun, prefill: prefill, next: next,
                                                          observe: ignore)
                    }
                } catch {
                    scopeError = String(describing: error)
                }
            } else {
                g = RouterOverlapModelGate.greedy(r, outputs: p.outputsPerRun, prefill: prefill, next: next,
                                                  observe: ignore)
            }
            r.reset()
        }
        let delta = r.routerDeviceRoutingCounters().minus(before)
        let source = arm == .on ? r.deviceRoutingState?.sourceSha256 : nil
        let counts = RouterOverlapModelGate.phaseCounts(g, c.derived, .off, promptCount: fx.prompt.count)
        var relation = arm == .on
            ? RouterDeviceRoutingModelGate.deviceTotalsRelation(on: g.totals, off: qualifiedOff, layers: layers,
                                                                promptCount: fx.prompt.count)
            : RouterOverlapModelGate.totalsRelation(candidate: g.totals, baseline: qualifiedOff,
                                                    extraBuffersPerForward: 0, promptCount: fx.prompt.count)
        relation += pinnedTotalsFailures(g.totals, arm: arm)
        let deviceFailures = deviceCounterFailures(delta, arm: arm, layers: layers, forwards: forwards)
        let tpot = RouterOverlapModelGate.runTPOT(walls)
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if let s = scopeError { f.append("scope: " + s) }
            if g.ids != fx.frozenIds { f.append("output IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != p.outputsPerRun { f.append("not 32 full-vocabulary finite outputs") }
            if !(prefillWall.isFinite && prefillWall > 0) { f.append("prefill wall invalid") }
            if tpot == nil || walls.count != p.timedContinuationsPerRun {
                f.append("continuation walls are not 31 positive values")
            }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            f += relation
            f += deviceFailures
            if arm == .on, source != expectedSource { f.append("runtime selector source differs from the qualified source") }
            if !RouterOverlapModelGate.installFailures(r, region: c.region).isEmpty { f.append("routed install displaced") }
            if !RouterOverlapModelGate.quiescent(r) || r.routerDeviceRoutingActive {
                f.append("runner not quiescent after the run")
            }
        }
        return RouterDeviceRoutingPerfRun(
            fixture: fx.name, arm: arm, warmup: slot.warmup, block: slot.block, orderInBlock: slot.orderInBlock,
            prefillWallMicros: prefillWall, continuationWallMicros: walls, tpotMedianWallMicros: tpot, totals: g.totals,
            ids: g.ids, idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs, allFinite: g.allFinite,
            counts: counts, totalsMismatches: relation, deviceCounters: delta, deviceCounterFailures: deviceFailures,
            selectorSourceSha256: source, preconditionFailures: pre, scopeError: scopeError, error: g.error,
            checksPassed: f.isEmpty, failure: f.isEmpty ? nil : f.joined(separator: "; "))
    }
}
