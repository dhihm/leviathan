import Foundation

// Phase 57 stage 3 part 2: full numerical model gate for the default-off GPU device-routing scope.
// Per pinned fixture: fresh direct reference, OFF (control prepared, incumbent CPU routing) and ON (device-routing
// preparation inside one withRouterDeviceRoutingScope). Every output must be full-vocabulary finite, within the
// strict 1e-4 bounds of the same fresh reference, follow the 32 frozen IDs on the previous actual argmax, and ON must
// equal OFF in Float.bitPattern at every logit. Every router frame (prefill included) must match the local CPU FP64
// decision exactly, and every frame is replayed through the component entry for record bit identity.
// Diagnostic only: no artifact, installation, promotion or default change; no performance measurement.
// Original expert scales and coefficient bits stay in process memory and are never serialized or hashed.

public enum RouterDeviceRoutingGateStatus: String, Codable, Sendable {
    case qualificationPassed
    case qualificationFailed
    case prerequisitesFailed
    case replayNotEnded
    case aborted
    case runnerNotRestored
}

/// Prerequisite receipt validated in process from the actual component reports and the hashed component binaries.
public struct RouterDeviceRoutingPrerequisiteReceipt: Codable, Equatable, Sendable {
    public let expectedSelectorSourceSha256: String
    public let expectedSelectorIRSha256: String
    public let componentReportSha256: String
    public let componentExecutableSha256: String
    public let selectionReportSha256: String
    public let selectionExecutableSha256: String
    public let failures: [String]

    public var valid: Bool { failures.isEmpty }
}

public struct RouterDeviceRoutingGateOnArm: Codable, Equatable, Sendable {
    public let ran: Bool
    public let ids: [Int]
    public let idsMatchFrozen: Bool
    public let outputs: Int
    public let allFinite: Bool
    public let comparison: OwnProcessLogitComparison
    public let bitComparedSteps: Int
    public let bitMismatchCount: Int
    public let firstBitMismatch: RouterOverlapGateBitMismatch?
    public let totals: RouterOverlapGateTotals
    public let counts: RouterOverlapGatePhaseCounts
    public let totalsMismatches: [String]
    public let deviceCounters: RouterDeviceRoutingCounters
    public let expectedForwards: Int
    public let snapshots: Int
    public let frameExactTokens: Int
    public let frameFailures: [RouterDeviceRoutingTokenCheck]
    public let preconditionFailures: [String]
    public let scopeError: String?
    public let error: String?
    public let passed: Bool
    public let failure: String?
}

public struct RouterDeviceRoutingGateFixture: Codable, Equatable, Sendable {
    public let fixture: String
    public let reference: RouterOverlapGateQualifiedArm
    public let off: RouterOverlapGateQualifiedArm
    public let offPerElement: OwnProcessLogitComparison
    public let offCPUSelections: Int
    public let offExtraFailures: [String]
    public let on: RouterDeviceRoutingGateOnArm?
    public let passed: Bool
}

public struct RouterDeviceRoutingGateReplay: Codable, Equatable, Sendable {
    public var frames = 0
    public var identicalFrames = 0
    public var mismatchCount = 0
    public var mismatchExamples: [String] = []
    public var chunkCompletions: [String] = []
    public var terminal = true
    public var completion = "NOT_RUN"
    public var error: String? = nil

    public init() {}
}

public struct RouterDeviceRoutingGateReport: Codable {
    public private(set) var kind = RouterDeviceRoutingModelGate.kind
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var scalesSerialized = false
    public private(set) var realE4GateComplete = false
    public private(set) var notes = RouterDeviceRoutingModelGate.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var selectorSourceSha256: String? = nil
    public var prerequisites: RouterDeviceRoutingPrerequisiteReceipt? = nil
    public var identity: RouterOverlapGateIdentity? = nil
    public var identityFailures: [String] = []
    public var refusalTests: [RouterOverlapGateRefusalTest] = []
    public var refusalScopeError: String? = nil
    public var refusalPassed = false
    public var fixtures: [RouterDeviceRoutingGateFixture] = []
    public var expectedFrames = 0
    public var framesChecked = 0
    public var framesExact = 0
    public var replay: RouterDeviceRoutingGateReplay? = nil
    public var qualificationPassed = false
    public var restoredPristine = false
    public var abortReason: String? = nil
    public var status: RouterDeviceRoutingGateStatus = .aborted

    public init() {}
}

extension RouterDeviceRoutingGateReport {
    /// Contract normalization: the always-false fields are re-asserted and realE4GateComplete is recomputed from the
    /// final status, the qualification result and the in-process receipt; it is never taken from input.
    func withDiagnosticContract() -> RouterDeviceRoutingGateReport {
        var copy = self
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.scalesSerialized = false
        copy.realE4GateComplete = copy.status == .qualificationPassed && copy.qualificationPassed
            && (copy.prerequisites?.valid ?? false)
        return copy
    }
}

public struct RouterDeviceRoutingGateOptions: Equatable, Sendable {
    public let gguf: String
    public let region: String
    public let graph: String
    public let fixtures: String
    public let componentReport: String
    public let componentExecutable: String
    public let selectionReport: String
    public let selectionExecutable: String
    public let report: String
}

public enum RouterDeviceRoutingModelGate {
    public static let kind = "router-device-routing-model-gate-numerical-v1"
    public static let readyToken = RouterOverlapModelGate.readyToken
    public static let goToken = RouterOverlapModelGate.goToken
    public static let endToken = RouterOverlapModelGate.endToken
    public static let notEndedToken = "NOT_ENDED"
    public static let maxReportBytes = 16 << 20
    public static let maxReceiptBytes = 1 << 20
    public static let correctnessContract = 1e-4
    public static let replayTimeoutSeconds = 10.0
    public static let usage = "usage: router-device-routing-gate --gguf M.gguf --region R.json --graph G.json "
        + "--fixtures F.json --component-report C.json --component-executable C_BIN --selection-report S.json "
        + "--selection-executable S_BIN --report NEW_REPORT.json"

    public static let refusalTestNames = [
        "device-prepared-execution-outside-scope",
        "device-prepared-continue-outside-scope",
        "raw-graph-execution-inside-device-scope",
        "raw-schedule-execution-inside-device-scope",
        "control-prepared-execution-inside-device-scope",
        "control-prepared-continue-inside-device-scope",
        "overlap-prepared-execution-inside-device-scope",
        "device-prepared-non-modern-candidate-inside-device-scope",
        "nested-device-scope-inside-device-scope",
        "route-capture-inside-device-scope",
        "direct-forward-inside-device-scope",
        "device-scope-entry-inside-overlap-scope",
        "device-scope-entry-inside-route-capture",
        "device-scope-entry-with-breakdown-instrumentation",
        "device-scope-entry-with-timeline-instrumentation",
        "device-scope-entry-without-routed-install",
    ]

    public static let notes: [String] = [
        "Diagnostic numerical gate only: no artifact, installation, promotion, default change or performance claim; "
            + "gainQualified, installable, productionPerformanceClaim and scalesSerialized are always false.",
        "realE4GateComplete is recomputed at encode time: true only when the in-process prerequisite receipt (actual "
            + "component reports matched to hashed component executables and this build's selector source/IR) is "
            + "valid and the full numerical qualification passed.",
        "Per fixture: fresh direct reference; OFF control prepared (CPU routing); ON device-routing preparation in one "
            + "scope. 32 frozen IDs on the previous actual argmax, full-vocabulary finite, vector and per-element "
            + "normalized < 1e-4 against the same reference, ON == OFF bitPattern at every logit, exact counters with "
            + "kernels OFF + 30 and command buffers OFF - 30 per forward, GPU selections and record chains 30 per "
            + "forward, CPU selections 0.",
        "Every router frame including prefill is checked against the local CPU FP64 decision recomputed from the "
            + "logged GPU scores and local scales, and replayed through the component entry for record bit identity.",
        "No scales, coefficient bits or weights are reported; the model SHA-256 is the cached explicit-refresh "
            + "record and is never rehashed.",
    ]

    public static func goAccepted(_ line: String?) -> Bool { line == goToken }

    public static func parseOptions(_ argv: [String]) throws -> RouterDeviceRoutingGateOptions {
        let keys = ["--gguf", "--region", "--graph", "--fixtures", "--component-report", "--component-executable",
                    "--selection-report", "--selection-executable", "--report"]
        var v = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard keys.contains(key), i + 1 < argv.count, v[key] == nil else { throw RouterDeviceRoutingError(usage) }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw RouterDeviceRoutingError(usage) }
            v[key] = value
            i += 2
        }
        guard v.count == keys.count, let gguf = v["--gguf"], let region = v["--region"], let graph = v["--graph"],
              let fixtures = v["--fixtures"], let cr = v["--component-report"], let ce = v["--component-executable"],
              let sr = v["--selection-report"], let se = v["--selection-executable"], let report = v["--report"] else {
            throw RouterDeviceRoutingError(usage)
        }
        return RouterDeviceRoutingGateOptions(gguf: gguf, region: region, graph: graph, fixtures: fixtures,
                                              componentReport: cr, componentExecutable: ce, selectionReport: sr,
                                              selectionExecutable: se, report: report)
    }

    public static func encodeReport(_ r: RouterDeviceRoutingGateReport) throws -> Data {
        try RouterOverlapModelGate.jsonEncoder(pretty: false).encode(r.withDiagnosticContract())
    }

    // MARK: Pure helpers

    /// Validates both prerequisite reports against the hashed executables, this build's emitted selector source and
    /// IR digests and the cached model SHA. Pure: the caller reads the small files and hashes the binaries.
    public static func validateReceipts(componentReport: Data, componentExecutableSha256: String,
                                        selectionReport: Data, selectionExecutableSha256: String,
                                        modelSha256: String) -> RouterDeviceRoutingPrerequisiteReceipt {
        var f = [String]()
        let source = (try? RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4))
            .map { RouterSelectionKernel.sourceDigest($0) } ?? ""
        let ir = (try? RouterSelectionKernel.irDigest(RouterSelectionIR.gemma4)) ?? ""
        if !moeIsLowercaseHex64(source) || !moeIsLowercaseHex64(ir) { f.append("selector source or IR digest unavailable") }
        if !moeIsLowercaseHex64(componentExecutableSha256) || !moeIsLowercaseHex64(selectionExecutableSha256) {
            f.append("component executable digests must be lowercase 64-hex")
        }
        if componentReport.count > maxReceiptBytes || selectionReport.count > maxReceiptBytes {
            f.append("prerequisite report exceeds 1 MiB")
        }
        if let comp = try? JSONDecoder().decode(RouterDeviceRoutingComponentReport.self, from: componentReport) {
            if comp.kind != RouterDeviceRoutingComponent.kind { f.append("component report kind differs") }
            if comp.status != .componentPassed || RouterDeviceRoutingComponent.status(comp) != .componentPassed {
                f.append("component report did not pass (recomputed)")
            }
            if comp.executableDigest != componentExecutableSha256 {
                f.append("component report executable digest differs from the hashed component executable")
            }
            if comp.selectorSourceSha256 != source { f.append("component report selector source differs from this build") }
            if comp.selectorIRSha256 != ir { f.append("component report selector IR differs from this build") }
            if comp.modelSha256 != modelSha256 { f.append("component report model SHA differs") }
            if comp.gainQualified || comp.installable || comp.productionPerformanceClaim || comp.realE4GateComplete
                || comp.scalesSerialized {
                f.append("component report contract flags are not false")
            }
        } else {
            f.append("component report does not decode")
        }
        if let obj = (try? JSONSerialization.jsonObject(with: selectionReport)) as? [String: Any] {
            if (obj["status"] as? String) != "PASS" { f.append("selection component report status is not PASS") }
            if (obj["sourceSHA256"] as? String) != source || (obj["componentSourceSHA256"] as? String) != source {
                f.append("selection component report source differs from this build")
            }
            if (obj["executableSHA256"] as? String) != selectionExecutableSha256 {
                f.append("selection component report executable digest differs from the hashed executable")
            }
            if (obj["unflaggedMismatches"] as? Int) != 0 || (obj["unexpectedFlags"] as? Int) != 0
                || (obj["guardsIntact"] as? Bool) != true || (obj["gainQualified"] as? Bool) != false {
                f.append("selection component report counters, guards or contract invalid")
            }
        } else {
            f.append("selection component report does not decode")
        }
        return RouterDeviceRoutingPrerequisiteReceipt(
            expectedSelectorSourceSha256: source, expectedSelectorIRSha256: ir,
            componentReportSha256: RouterSelectionKernel.sha256Hex(componentReport),
            componentExecutableSha256: componentExecutableSha256,
            selectionReportSha256: RouterSelectionKernel.sha256Hex(selectionReport),
            selectionExecutableSha256: selectionExecutableSha256, failures: f)
    }

    /// ON kernels = OFF + layers and ON command buffers = OFF - layers per forward (prefill and each continuation).
    public static func deviceTotalsRelation(on a: RouterOverlapGateTotals, off b: RouterOverlapGateTotals,
                                            layers: Int, promptCount: Int) -> [String] {
        var f = [String]()
        let n = RouterOverlapGateProtocol.required.continuationsPerRun
        if a.prefillKernels != b.prefillKernels + layers * promptCount {
            f.append("prefill kernels " + String(a.prefillKernels) + " != " + String(b.prefillKernels + layers * promptCount))
        }
        if a.prefillBuffers != b.prefillBuffers - layers * promptCount {
            f.append("prefill command buffers " + String(a.prefillBuffers) + " != "
                     + String(b.prefillBuffers - layers * promptCount))
        }
        guard a.continuationKernels.count == n, b.continuationKernels.count == n,
              a.continuationBuffers.count == n, b.continuationBuffers.count == n else {
            f.append("continuation step counts are not 31 in both runs")
            return f
        }
        for i in 0..<n {
            if a.continuationKernels[i] != b.continuationKernels[i] + layers {
                f.append("continuation " + String(i) + " kernels " + String(a.continuationKernels[i]) + " != "
                         + String(b.continuationKernels[i] + layers))
            }
            if a.continuationBuffers[i] != b.continuationBuffers[i] - layers {
                f.append("continuation " + String(i) + " command buffers " + String(a.continuationBuffers[i]) + " != "
                         + String(b.continuationBuffers[i] - layers))
            }
        }
        return f
    }

    public static func snapshotPositionFailures(_ positions: [Int], forwards: Int) -> [String] {
        guard forwards >= 0 else { return ["forward count negative"] }
        if positions != Array(0..<forwards) {
            return ["snapshot positions are not exactly 0.." + String(forwards - 1) + " (" + String(positions.count) + " snapshots)"]
        }
        return []
    }

    public static func finalStatus(_ r: RouterDeviceRoutingGateReport) -> RouterDeviceRoutingGateStatus {
        if let rp = r.replay, !rp.terminal { return .replayNotEnded }
        if !r.restoredPristine { return .runnerNotRestored }
        if r.abortReason != nil { return .aborted }
        guard r.prerequisites?.valid == true else { return .prerequisitesFailed }
        return r.qualificationPassed ? .qualificationPassed : .qualificationFailed
    }

    static func finalize(_ r: inout RouterDeviceRoutingGateReport) {
        let expected = r.expectedFrames
        let replayOK = r.replay.map {
            $0.terminal && $0.completion == "COMPLETED" && $0.frames == expected && $0.identicalFrames == expected
                && $0.mismatchCount == 0
        } ?? false
        r.qualificationPassed = r.abortReason == nil && r.identity != nil && r.identityFailures.isEmpty && r.refusalPassed
            && r.fixtures.map({ $0.fixture }) == RouterOverlapGateFixtures.order && r.fixtures.allSatisfy({ $0.passed })
            && expected > 0 && r.framesChecked == expected && r.framesExact == expected && replayOK
            && r.selectorSourceSha256 != nil && r.selectorSourceSha256 == r.prerequisites?.expectedSelectorSourceSha256
        r.status = finalStatus(r)
        r = r.withDiagnosticContract()
    }

    // MARK: Entry point

    /// Never throws; every failure is recorded. The caller prints END only when the runner is quiescent and the
    /// replay is terminal.
    public static func run(region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                           runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                           prerequisites: RouterDeviceRoutingPrerequisiteReceipt,
                           emit: (String) -> Void) -> RouterDeviceRoutingGateReport {
        let proto = RouterOverlapGateProtocol.required
        var report = RouterDeviceRoutingGateReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.prerequisites = prerequisites
        report.expectedFrames = RouterRouteReplay.expectedFrames(fixtures: fixtures, outputs: proto.outputsPerRun,
                                                                 layers: runner.config.layerCount)
        let pristineStart = RouterOverlapModelGate.pristine(runner)
        let instrumentationStart = RouterOverlapModelGate.instrumentationOff(runner)
            && RouterOverlapModelGate.axisFlagsClean(runner) && !runner.routerDeviceRoutingActive
            && !runner.routeCaptureActive
        var setup = [String]()
        if !prerequisites.valid { setup.append("prerequisite receipt invalid") }
        if !pristineStart { setup.append("runner is not pristine at start") }
        if !instrumentationStart { setup.append("instrumentation, axis flags or scopes are not off at start") }
        if fixtures != RouterOverlapGateFixtures.pinned { setup.append("fixtures differ from the pinned fixtures") }
        if !moeIsLowercaseHex64(modelSha256) || !moeIsLowercaseHex64(executableDigest) {
            setup.append("model and executable digests must be lowercase 64-hex")
        }
        guard setup.isEmpty else {
            report.abortReason = setup.joined(separator: "; ")
            report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            finalize(&report)
            emit("aborted before any GPU work: " + (report.abortReason ?? ""))
            return report
        }
        var frames = [(String, RouterDeviceRoutingTokenSnapshot)]()
        var table: RouterOffsetTable? = nil
        do {
            try runner.withQ4DiagnosticReferenceWidth(proto.q4LaunchWidth) { () -> Void in
                try scoped(&report, frames: &frames, table: &table, region: region, graph: graph, fixtures: fixtures,
                           runner: runner, modelSha256: modelSha256, executableDigest: executableDigest,
                           pristineStart: pristineStart, instrumentationStart: instrumentationStart, emit: emit)
            }
        } catch {
            report.abortReason = "gate threw: " + String(describing: error)
        }
        runner.uninstallMoeGeneratedLowering()
        runner.reset()
        report.restoredPristine = RouterOverlapModelGate.pristine(runner)
            && RouterOverlapModelGate.instrumentationOff(runner) && RouterOverlapModelGate.axisFlagsClean(runner)
            && !runner.routerDeviceRoutingActive
        if report.abortReason == nil, let t = table, !frames.isEmpty {
            report.replay = replayFrames(frames, table: t, scales: runner.expertScales,
                                         timeoutSeconds: replayTimeoutSeconds)
            emit(RouterOverlapModelGate.event("replay", report.replay))
        }
        finalize(&report)
        emit("status " + report.status.rawValue + "; frames exact " + String(report.framesExact) + " of "
             + String(report.expectedFrames) + "; restored pristine " + String(report.restoredPristine)
             + "; realE4GateComplete " + String(report.realE4GateComplete) + "; no performance claim")
        return report
    }

    static func scoped(_ report: inout RouterDeviceRoutingGateReport,
                       frames: inout [(String, RouterDeviceRoutingTokenSnapshot)], table: inout RouterOffsetTable?,
                       region: MoERegionIR, graph: GemmaGraphIR, fixtures: [RouterOverlapGateFixture],
                       runner: Gemma4Runner, modelSha256: String, executableDigest: String,
                       pristineStart: Bool, instrumentationStart: Bool, emit: (String) -> Void) throws {
        let proto = RouterOverlapGateProtocol.required
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.reset()
        }
        guard runner.q4LaunchWidth == proto.q4LaunchWidth else {
            throw RouterDeviceRoutingError("Q4 reference width 32 is not in effect inside the width scope")
        }
        var prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
        let derived = RouterOverlapModelGate.derive(on: prepared.on, runner: runner)
        if derived.statisticsGroups > 0 {
            _ = try runner.rmsStatisticsInverse(slots: derived.statisticsGroups)
        }
        var ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
        var device = try GemmaPreparedExecution.prepareRouterDeviceRouting(graph: graph, runner: runner)
        let identity = try RouterOverlapModelGate.makeIdentity(
            ctx, modelSha256: modelSha256, executableDigest: executableDigest,
            pristineAtStart: pristineStart, instrumentationOffAtStart: instrumentationStart)
        report.identity = identity
        report.identityFailures = RouterOverlapModelGate.identityFailures(identity)
        emit(RouterOverlapModelGate.event("identity", identity))
        guard report.identityFailures.isEmpty else {
            emit("identity FAILED; nothing executed on the GPU")
            return
        }
        let realTable = try RouterRouteReplay.realOffsetTable(runner)
        guard try realTable.validate() == 2 * derived.layers * runner.config.expertCount else {
            throw RouterDeviceRoutingError("real offset table proof incomplete")
        }
        table = realTable
        let guards = refusalTests(ctx, device: device, region: region, prompt: fixtures[0].prompt)
        report.refusalTests = guards.tests
        report.refusalScopeError = guards.scopeError
        report.refusalPassed = guards.scopeError == nil && guards.tests.map({ $0.name }) == refusalTestNames
            && guards.tests.allSatisfy({ $0.passed })
        emit(RouterOverlapModelGate.event("refusalTests", guards.tests))
        guard report.refusalPassed else {
            emit("refusal tests FAILED; qualification not run")
            return
        }
        for fx in fixtures {
            let forwards = fx.prompt.count + proto.outputsPerRun - 1
            runner.uninstallMoeGeneratedLowering()
            let ref = RouterOverlapModelGate.qualify(.reference, ctx, fx, reference: [], off: [], offTotals: nil, keep: true)
            prepared = try RouterOverlapModelGate.installAndPrepare(runner, region: region, graph: graph)
            guard RouterOverlapModelGate.derive(on: prepared.on, runner: runner) == derived else {
                throw RouterDeviceRoutingError("derived counts changed after re-preparation")
            }
            ctx = RouterOverlapGateContext(graph: graph, region: region, runner: runner,
                                           candidate: RouterOverlapModelGate.modernCandidate, derived: derived,
                                           off: prepared.off, on: prepared.on)
            device = try GemmaPreparedExecution.prepareRouterDeviceRouting(graph: graph, runner: runner)
            guard ref.record.passed, ref.kept.count == proto.outputsPerRun else {
                let record = RouterDeviceRoutingGateFixture(
                    fixture: fx.name, reference: ref.record,
                    off: RouterOverlapModelGate.notRun(.off, "no valid fresh reference").record,
                    offPerElement: OwnProcessLogitComparison(), offCPUSelections: 0, offExtraFailures: [], on: nil,
                    passed: false)
                report.fixtures.append(record)
                emit(RouterOverlapModelGate.event("fixture", record))
                continue
            }
            let c0 = runner.routerDeviceRoutingCounters()
            let offResult = RouterOverlapModelGate.qualify(.off, ctx, fx, reference: ref.kept, off: [], offTotals: nil,
                                                           keep: true)
            let offCPU = runner.routerDeviceRoutingCounters().minus(c0).cpuSelections
            var offCmp = OwnProcessLogitComparison()
            for (i, lg) in offResult.kept.enumerated() {
                offCmp.add(step: i, logits: lg, reference: i < ref.kept.count ? ref.kept[i] : [])
            }
            var offExtra = offCmp.failures(contract: correctnessContract, expectedSteps: proto.outputsPerRun)
            if offCPU != derived.layers * forwards {
                offExtra.append("OFF CPU selections " + String(offCPU) + " != " + String(derived.layers * forwards))
            }
            var on: RouterDeviceRoutingGateOnArm? = nil
            if offResult.record.passed, offResult.kept.count == proto.outputsPerRun {
                let result = deviceRun(ctx, device: device, region: region, fx, reference: ref.kept,
                                       off: offResult.kept, offTotals: offResult.totals, table: realTable)
                on = result.arm
                report.framesChecked += result.snapshots.count * derived.layers
                report.framesExact += result.arm.frameExactTokens * derived.layers
                for s in result.snapshots { frames.append((fx.name, s)) }
            }
            if report.selectorSourceSha256 == nil { report.selectorSourceSha256 = runner.deviceRoutingState?.sourceSha256 }
            let record = RouterDeviceRoutingGateFixture(
                fixture: fx.name, reference: ref.record, off: offResult.record, offPerElement: offCmp,
                offCPUSelections: offCPU, offExtraFailures: offExtra, on: on,
                passed: ref.record.passed && offResult.record.passed && offExtra.isEmpty && (on?.passed ?? false))
            report.fixtures.append(record)
            emit(RouterOverlapModelGate.event("fixture", record))
            if !RouterOverlapModelGate.quiescent(runner) || runner.routerDeviceRoutingActive {
                throw RouterDeviceRoutingError("runner not quiescent or device scope still active after " + fx.name)
            }
        }
    }

    // MARK: ON arm

    static func deviceRun(_ c: RouterOverlapGateContext, device: GemmaPreparedExecution, region: MoERegionIR,
                          _ fx: RouterOverlapGateFixture, reference: [[Float]], off: [[Float]],
                          offTotals: RouterOverlapGateTotals, table: RouterOffsetTable)
        -> (arm: RouterDeviceRoutingGateOnArm, snapshots: [RouterDeviceRoutingTokenSnapshot]) {
        let r = c.runner
        let proto = RouterOverlapGateProtocol.required
        let layers = c.derived.layers
        let forwards = fx.prompt.count + proto.outputsPerRun - 1
        var pre = RouterOverlapModelGate.preconditions(.off, c, prepared: c.off)
        if device.routerDeviceRouting == nil { pre.append("device preparation carries no device-routing binding") }
        if device.routerOverlap != nil { pre.append("device preparation carries an overlap plan") }
        pre += RouterOverlapModelGate.bindingFailures(device, c)
        var g = RouterOverlapGateGreedy()
        var cmp = OwnProcessLogitComparison()
        var bits = RouterOverlapGateBitComparator()
        var scopeError: String? = nil
        var snaps = [RouterDeviceRoutingTokenSnapshot]()
        let before = r.routerDeviceRoutingCounters()
        if pre.isEmpty {
            func prefill() throws -> RouterOverlapModelGate.Step {
                try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: fx.prompt, candidate: c.candidate,
                                            prepared: device)
            }
            func next(_ token: Int) throws -> RouterOverlapModelGate.Step {
                try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token, candidate: c.candidate,
                                                    prepared: device)
            }
            func observe(_ step: Int, _ logits: [Float]) {
                cmp.add(step: step, logits: logits, reference: step < reference.count ? reference[step] : [])
                if step < off.count {
                    bits.add(step: step, on: logits, off: off[step])
                } else {
                    bits.missing += 1
                }
            }
            r.reset()
            do {
                try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: true) { () -> Void in
                    g = RouterOverlapModelGate.greedy(r, outputs: proto.outputsPerRun, prefill: prefill, next: next,
                                                      observe: observe)
                    snaps = r.routerDeviceRoutingDrainSnapshots()
                }
            } catch {
                scopeError = String(describing: error)
            }
            if snaps.isEmpty { snaps = r.routerDeviceRoutingDrainSnapshots() }
            r.reset()
        }
        let delta = r.routerDeviceRoutingCounters().minus(before)
        let counts = RouterOverlapModelGate.phaseCounts(g, c.derived, .off, promptCount: fx.prompt.count)
        let relation = deviceTotalsRelation(on: g.totals, off: offTotals, layers: layers, promptCount: fx.prompt.count)
        var frameFailures = [RouterDeviceRoutingTokenCheck]()
        var exactTokens = 0
        for s in snaps {
            let check = RouterDeviceRoutingVerifier.check(position: s.position, records: s.records,
                                                          scoreBits: s.scoreBits, scales: r.expertScales,
                                                          table: table, topK: r.config.topKExperts)
            if check.exact { exactTokens += 1 } else { frameFailures.append(check) }
        }
        var f = pre
        if pre.isEmpty {
            if let e = g.error { f.append(e) }
            if let s = scopeError { f.append("scope: " + s) }
            if g.ids != fx.frozenIds { f.append("ON IDs differ from the 32 frozen IDs") }
            if !g.allFinite || g.outputs != proto.outputsPerRun { f.append("not 32 full-vocabulary finite outputs") }
            if !counts.exact { f.append("counters: " + counts.mismatches.joined(separator: ", ")) }
            f += relation
            f += cmp.failures(contract: correctnessContract, expectedSteps: proto.outputsPerRun)
            if bits.compared != proto.outputsPerRun || bits.missing != 0 || bits.mismatches != 0 {
                f.append("ON vs OFF bit comparison: compared " + String(bits.compared) + ", missing "
                         + String(bits.missing) + ", mismatching logits " + String(bits.mismatches))
            }
            if delta.cpuSelections != 0 || delta.gpuSelections != layers * forwards
                || delta.recordChains != layers * forwards || delta.acceptedTokens != forwards
                || delta.failedTokens != 0 || delta.flaggedTokens != 0 {
                f.append("device routing counters differ from 0 CPU / 30 GPU selections and 30 record chains per forward")
            }
            f += snapshotPositionFailures(snaps.map { $0.position }, forwards: forwards)
            if exactTokens != forwards || !frameFailures.isEmpty {
                f.append(String(frameFailures.count) + " token(s) with non-exact router frames")
            }
            if !RouterOverlapModelGate.quiescent(r) || r.routerDeviceRoutingActive {
                f.append("runner not quiescent or device scope still active")
            }
        }
        let arm = RouterDeviceRoutingGateOnArm(
            ran: pre.isEmpty, ids: g.ids, idsMatchFrozen: g.ids == fx.frozenIds, outputs: g.outputs,
            allFinite: g.allFinite, comparison: cmp, bitComparedSteps: bits.compared, bitMismatchCount: bits.mismatches,
            firstBitMismatch: bits.first, totals: g.totals, counts: counts, totalsMismatches: relation,
            deviceCounters: delta, expectedForwards: forwards, snapshots: snaps.count, frameExactTokens: exactTokens,
            frameFailures: frameFailures, preconditionFailures: pre, scopeError: scopeError, error: g.error,
            passed: f.isEmpty, failure: f.isEmpty ? nil : f.joined(separator: "; "))
        return (arm, snaps)
    }

    // MARK: Refusal tests

    static func refusalTests(_ c: RouterOverlapGateContext, device: GemmaPreparedExecution, region: MoERegionIR,
                             prompt: [Int]) -> (tests: [RouterOverlapGateRefusalTest], scopeError: String?) {
        let r = c.runner
        var tests = [RouterOverlapGateRefusalTest]()
        func probe(_ name: String, inside: Bool, checkInstall: Bool = true, _ action: () throws -> Void) {
            let before = RouterOverlapGateGuardState(r)
            let c0 = r.routerDeviceRoutingCounters()
            var refused = false
            var message: String? = nil
            do { try action() } catch {
                refused = true
                message = String(describing: error)
            }
            let after = RouterOverlapGateGuardState(r)
            let d = r.routerDeviceRoutingCounters().minus(c0)
            let noGpu = before.kernels == after.kernels && before.buffers == after.buffers
                && before.gpuBits == after.gpuBits && before.position == after.position
                && d.cpuSelections == 0 && d.gpuSelections == 0 && d.recordChains == 0 && d.acceptedTokens == 0
                && d.failedTokens == 0
            let notPoisoned = !(r.deviceRoutingState?.active == true && r.deviceRoutingState?.poisoned == true)
            let intact = !after.open && after.pending == 0 && before.scopeActive == after.scopeActive && notPoisoned
                && (!checkInstall || (before.moeDigest == after.moeDigest && before.q4Width == after.q4Width))
            tests.append(RouterOverlapGateRefusalTest(
                name: name, insideScope: inside, refused: refused, error: message, noGpuWork: noGpu,
                stateIntact: intact, passed: refused && noGpu && intact))
        }
        let nonModern = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
        let token = prompt[prompt.count - 1]
        probe("device-prepared-execution-outside-scope", inside: false) {
            _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: c.candidate, prepared: device)
        }
        probe("device-prepared-continue-outside-scope", inside: false) {
            _ = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token, candidate: c.candidate,
                                                    prepared: device)
        }
        var scopeError: String? = nil
        do {
            try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Void in
                probe("raw-graph-execution-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: c.candidate)
                }
                probe("raw-schedule-execution-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: c.candidate,
                                                    schedule: c.off.ssa)
                }
                probe("control-prepared-execution-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: c.candidate,
                                                    prepared: c.off)
                }
                probe("control-prepared-continue-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraphContinue(c.graph, runner: r, token: token, candidate: c.candidate,
                                                            prepared: c.off)
                }
                probe("overlap-prepared-execution-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: c.candidate,
                                                    prepared: c.on)
                }
                probe("device-prepared-non-modern-candidate-inside-device-scope", inside: true) {
                    _ = try GemmaForge.executeGraph(c.graph, runner: r, tokenIds: prompt, candidate: nonModern,
                                                    prepared: device)
                }
                probe("nested-device-scope-inside-device-scope", inside: true) {
                    _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
                }
                probe("route-capture-inside-device-scope", inside: true) {
                    let capture = try RouterRouteCapture(expertCount: r.config.expertCount, topK: r.config.topKExperts,
                                                         layerCount: r.config.layerCount, maxFrames: 1)
                    _ = try r.withRouteCapture(capture) { () -> Int in 0 }
                }
                probe("direct-forward-inside-device-scope", inside: true) {
                    _ = try r.forward(tokenIds: prompt)
                }
            }
        } catch {
            scopeError = String(describing: error)
        }
        if !RouterOverlapModelGate.quiescent(r) || r.routerDeviceRoutingActive {
            scopeError = (scopeError.map { $0 + "; " } ?? "") + "device scope did not close cleanly"
        }
        probe("device-scope-entry-inside-overlap-scope", inside: false) {
            try r.withRouterOverlapDiagnosticScope { () -> Void in
                _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
            }
        }
        probe("device-scope-entry-inside-route-capture", inside: false) {
            let capture = try RouterRouteCapture(expertCount: r.config.expertCount, topK: r.config.topKExperts,
                                                 layerCount: r.config.layerCount, maxFrames: 1)
            try r.withRouteCapture(capture) { () -> Void in
                _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
            }
        }
        probe("device-scope-entry-with-breakdown-instrumentation", inside: false) {
            r.breakdownInstrumentation = true
            defer { r.breakdownInstrumentation = false }
            _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
        }
        probe("device-scope-entry-with-timeline-instrumentation", inside: false) {
            r.commandTimelineEnabled = true
            defer { r.commandTimelineEnabled = false }
            _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
        }
        probe("device-scope-entry-without-routed-install", inside: false, checkInstall: false) {
            r.uninstallMoeGeneratedLowering()
            _ = try r.withRouterDeviceRoutingScope(region: region, captureSnapshots: false) { () -> Int in 0 }
        }
        return (tests, scopeError)
    }

    // MARK: Component-entry replay of every captured frame

    static func replayFrames(_ frames: [(String, RouterDeviceRoutingTokenSnapshot)], table: RouterOffsetTable,
                             scales: [[Float]], timeoutSeconds: Double) -> RouterDeviceRoutingGateReplay {
        var rp = RouterDeviceRoutingGateReplay()
        var names = [String]()
        var layers = [Int]()
        var scores = [[Float]]()
        var runtime = [RouterSelectionGPURecord]()
        for (fixture, snap) in frames {
            for layer in 0..<min(snap.records.count, snap.scoreBits.count) {
                names.append(fixture + ":" + String(snap.position) + ":" + String(layer))
                layers.append(layer)
                scores.append(snap.scoreBits[layer].map { Float(bitPattern: $0) })
                runtime.append(snap.records[layer])
            }
        }
        rp.frames = runtime.count
        guard !runtime.isEmpty else { return rp }
        rp.completion = "COMPLETED"
        for range in RouterRouteReplay.chunks(total: runtime.count, maxPerChunk: RouterRouteReplay.chunkSize) {
            let out: RouterSelectionGPUOutcome
            do {
                let component = try RouterSelectionGPUComponent(ir: RouterSelectionIR.gemma4)
                out = try component.runSelector(scores: range.map { scores[$0] },
                                                caseLayers: range.map { UInt32(layers[$0]) }, scales: scales,
                                                offsets: table, timeoutSeconds: timeoutSeconds)
            } catch {
                // Every component throw precedes its commit: nothing from this chunk is in flight.
                rp.chunkCompletions.append("NOT_COMMITTED")
                rp.completion = "NOT_COMMITTED"
                rp.error = String(describing: error)
                break
            }
            rp.chunkCompletions.append(out.completion)
            if !out.terminal {
                rp.terminal = false
                rp.completion = "NOT_ENDED"
                rp.error = out.errorDescription
                break
            }
            guard out.completion == "COMPLETED", out.records.count == range.count, out.guardsIntact else {
                rp.completion = "ERROR"
                rp.error = out.errorDescription ?? "component record count or canary failure"
                break
            }
            for (j, idx) in range.enumerated() {
                let a = out.records[j]
                let b = runtime[idx]
                if a == b && a.paddingClean && b.paddingClean {
                    rp.identicalFrames += 1
                } else {
                    rp.mismatchCount += 1
                    if rp.mismatchExamples.count < 64 { rp.mismatchExamples.append(names[idx]) }
                }
            }
        }
        return rp
    }
}
