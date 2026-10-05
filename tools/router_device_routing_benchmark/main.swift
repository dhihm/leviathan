// router-device-routing-benchmark - phase-58 fixed ABBA performance protocol for the qualified GPU device-routing
// candidate against the OFF incumbent (diagnostic prototype only; never installable; no production claim).
// Two invocations of this SAME binary, with no rebuild in between:
//   --mode qualification-only: the unchanged RouterDeviceRoutingModelGate numerical qualification bound to this
//     executable's own digest; writes the standard gate report.
//   --mode performance-only: validates that report against this executable, the hashed --qualified-executable, the
//     cached model SHA and freshly validated prerequisite receipts, then runs 6 warmups and 42 timed runs.
// Before GO: argument parsing, exclusive report path, bounded small-file reads, the cached model SHA record (never
// hashed), executable digests and receipt validation; invalid receipts exit 2 before BENCHMARK_READY. After
// BENCHMARK_READY only an exact GO line proceeds; EOF or anything else exits 1 with no Metal device and no mapping.
// TIMED_WINDOW_END is printed only when the runner is quiescent and restored pristine (qualification-only: and every
// replay chunk terminal); otherwise NOT_ENDED.
// Exit (qualification-only): 0 passed, 3 failed, 5 not ended, 1 otherwise.
// Exit (performance-only): 0 criteria met, 3 run check failed, 4 unstable, 6 no gain, 5 not ended, 1 otherwise.
// Exit 2 (either mode): prerequisite or qualification receipt invalid, before BENCHMARK_READY.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("router-device-routing-benchmark: " + s).utf8))
    FileHandle.standardError.write(Data([10]))
}

func boundedRead(_ url: URL, limit: Int) throws -> Data {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var data = Data()
    while data.count <= limit {
        let chunk = (try handle.read(upToCount: 1 << 16)) ?? Data()
        if chunk.isEmpty { break }
        data.append(chunk)
    }
    guard data.count <= limit else {
        throw RouterDeviceRoutingError(url.path + " exceeds " + String(limit) + " bytes")
    }
    return data
}

@discardableResult
func writeReport(_ encoded: Data?, to url: URL) -> Bool {
    guard let data = encoded else {
        complain("report encode failed; nothing written")
        return false
    }
    guard data.count <= RouterDeviceRoutingPerformance.maxReportBytes else {
        complain("report exceeds 16 MiB; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

struct Inputs {
    let region: MoERegionIR
    let graph: GemmaGraphIR
    let fixtures: [RouterOverlapGateFixture]
    let modelSha256: String
    let executableDigest: String
    let prerequisites: RouterDeviceRoutingPrerequisiteReceipt
    let qualification: RouterDeviceRoutingQualificationReceipt?
    let qualified: RouterDeviceRoutingGateReport?
}

/// Small files only: no Metal device, no model mapping, no model hashing.
func loadInputs(_ o: RouterDeviceRoutingBenchmarkOptions) throws -> Inputs {
    let region = try MoERegionIR.parse(try boundedRead(URL(fileURLWithPath: o.region), limit: 1 << 19))
    let graph = try GemmaGraph.parse(try boundedRead(URL(fileURLWithPath: o.graph), limit: 8 << 20))
    let fixtures = try RouterOverlapGateFixtures.parse(
        try boundedRead(URL(fileURLWithPath: o.fixtures), limit: 1 << 16))
    let ggufURL = URL(fileURLWithPath: o.gguf)
    guard let cached = try ModelDigestCache.shared.cachedRecord(for: ggufURL) else {
        throw RouterDeviceRoutingError("no cached model SHA-256 for " + ggufURL.path
            + "; refresh it with the separate explicit command (this tool never hashes the model)")
    }
    let own = try Gemma4Runner.executableDigest()
    let componentReport = try boundedRead(URL(fileURLWithPath: o.componentReport),
                                          limit: RouterDeviceRoutingModelGate.maxReceiptBytes)
    let selectionReport = try boundedRead(URL(fileURLWithPath: o.selectionReport),
                                          limit: RouterDeviceRoutingModelGate.maxReceiptBytes)
    let componentExe = try GemmaForge.fileDigest(URL(fileURLWithPath: o.componentExecutable))
    let selectionExe = try GemmaForge.fileDigest(URL(fileURLWithPath: o.selectionExecutable))
    let prerequisites = RouterDeviceRoutingModelGate.validateReceipts(
        componentReport: componentReport, componentExecutableSha256: componentExe,
        selectionReport: selectionReport, selectionExecutableSha256: selectionExe, modelSha256: cached.sha256)
    var qualification: RouterDeviceRoutingQualificationReceipt? = nil
    var qualified: RouterDeviceRoutingGateReport? = nil
    if o.mode == .performanceOnly, let qr = o.qualifiedReport, let qe = o.qualifiedExecutable {
        let data = try boundedRead(URL(fileURLWithPath: qr), limit: RouterDeviceRoutingPerformance.maxReportBytes)
        let qualifiedExe = try GemmaForge.fileDigest(URL(fileURLWithPath: qe))
        let v = RouterDeviceRoutingPerformance.validateQualification(
            report: data, qualifiedExecutableSha256: qualifiedExe, executableSha256: own, modelSha256: cached.sha256,
            prerequisites: prerequisites)
        qualification = v.receipt
        qualified = v.report
    }
    return Inputs(region: region, graph: graph, fixtures: fixtures, modelSha256: cached.sha256,
                  executableDigest: own, prerequisites: prerequisites, qualification: qualification,
                  qualified: qualified)
}

func gateExit(_ s: RouterDeviceRoutingGateStatus) -> Int32 {
    switch s {
    case .qualificationPassed: return 0
    case .qualificationFailed: return 3
    case .prerequisitesFailed: return 2
    case .replayNotEnded: return 5
    case .aborted, .runnerNotRestored: return 1
    }
}

func perfExit(_ s: RouterDeviceRoutingPerfStatus) -> Int32 {
    switch s {
    case .performanceCriteriaMet: return 0
    case .performanceRunCheckFailed: return 3
    case .performanceUnstable: return 4
    case .performanceNoGain: return 6
    case .qualificationReceiptInvalid: return 2
    case .aborted, .runnerNotRestored: return 1
    }
}

func runQualification(_ inputs: Inputs, runner: Gemma4Runner?, loadError: String?, reportURL: URL) -> Int32 {
    var report: RouterDeviceRoutingGateReport
    if let r = runner {
        report = RouterDeviceRoutingModelGate.run(
            region: inputs.region, graph: inputs.graph, fixtures: inputs.fixtures, runner: r,
            modelSha256: inputs.modelSha256, executableDigest: inputs.executableDigest,
            prerequisites: inputs.prerequisites, emit: { say($0) })
    } else {
        report = RouterDeviceRoutingGateReport()
        report.prerequisites = inputs.prerequisites
        report.abortReason = loadError ?? "runner load failed"
        report.restoredPristine = true
        report.status = RouterDeviceRoutingModelGate.finalStatus(report)
    }
    let terminal = (runner.map { RouterOverlapModelGate.quiescent($0) } ?? true) && report.status != .replayNotEnded
    let encoded = try? RouterDeviceRoutingModelGate.encodeReport(report)
    guard terminal, report.restoredPristine else {
        if !terminal { say(RouterDeviceRoutingPerformance.notEndedToken) }
        complain("runner not quiescent, replay not terminal or runner not restored; END withheld")
        writeReport(encoded, to: reportURL)
        return terminal ? 1 : 5
    }
    say(RouterDeviceRoutingPerformance.endToken)
    guard writeReport(encoded, to: reportURL) else { return 1 }
    say("qualification report " + reportURL.path + " written: status " + report.status.rawValue + "; executable "
        + inputs.executableDigest + "; no performance measured")
    return gateExit(report.status)
}

func runPerformance(_ inputs: Inputs, qualification: RouterDeviceRoutingQualificationReceipt, runner: Gemma4Runner?,
                    loadError: String?, reportURL: URL) -> Int32 {
    let report: RouterDeviceRoutingPerfReport
    if let r = runner {
        report = RouterDeviceRoutingPerformance.run(
            region: inputs.region, graph: inputs.graph, fixtures: inputs.fixtures, runner: r,
            modelSha256: inputs.modelSha256, executableDigest: inputs.executableDigest, qualification: qualification,
            qualified: inputs.qualified, emit: { say($0) })
    } else {
        report = RouterDeviceRoutingPerformance.loadFailed(
            loadError ?? "runner load failed", modelSha256: inputs.modelSha256,
            executableDigest: inputs.executableDigest, qualification: qualification)
    }
    let quiescent = runner.map { RouterOverlapModelGate.quiescent($0) } ?? true
    let encoded = try? RouterDeviceRoutingPerformance.encodeReport(report)
    guard quiescent, report.restoredPristine else {
        if !quiescent { say(RouterDeviceRoutingPerformance.notEndedToken) }
        complain("runner not quiescent or not restored; END withheld")
        writeReport(encoded, to: reportURL)
        return quiescent ? 1 : 5
    }
    say(RouterDeviceRoutingPerformance.endToken)
    guard writeReport(encoded, to: reportURL) else { return 1 }
    say("performance report " + reportURL.path + " written: status " + report.status.rawValue + "; gainQualified "
        + String(report.gainQualified) + "; installable false; no production performance claim")
    return perfExit(report.status)
}

func runTool() -> Int32 {
    let options: RouterDeviceRoutingBenchmarkOptions
    do { options = try RouterDeviceRoutingPerformance.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(RouterDeviceRoutingPerformance.usage)
        return 1
    }
    let fm = FileManager.default
    let reportURL = URL(fileURLWithPath: options.report)
    var isDirectory: ObjCBool = false
    let parentIsDirectory = fm.fileExists(atPath: reportURL.deletingLastPathComponent().path,
                                          isDirectory: &isDirectory) && isDirectory.boolValue
    if let reason = OwnProcessDiagnostics.reportPathRejection(
        path: reportURL.path, exists: fm.fileExists(atPath: reportURL.path), parentIsDirectory: parentIsDirectory) {
        complain(reason)
        return 1
    }
    let inputs: Inputs
    do { inputs = try loadInputs(options) } catch {
        complain(String(describing: error))
        return 1
    }
    guard inputs.prerequisites.valid else {
        complain("prerequisites invalid (no device, no mapping, no report): "
                 + inputs.prerequisites.failures.joined(separator: "; "))
        return 2
    }
    var qualification: RouterDeviceRoutingQualificationReceipt? = nil
    if options.mode == .performanceOnly {
        guard let q = inputs.qualification, q.valid, inputs.qualified != nil else {
            complain("qualification receipt invalid (no device, no mapping, no report): "
                     + (inputs.qualification?.failures.joined(separator: "; ") ?? "missing"))
            return 2
        }
        qualification = q
    }

    say(RouterDeviceRoutingPerformance.readyToken)
    guard RouterDeviceRoutingPerformance.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; nothing loaded, no device created")
        return 1
    }

    var runner: Gemma4Runner? = nil
    var loadError: String? = nil
    do { runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: options.gguf)) } catch {
        loadError = "runner load failed: " + String(describing: error)
    }
    let code: Int32
    if let q = qualification {
        code = runPerformance(inputs, qualification: q, runner: runner, loadError: loadError, reportURL: reportURL)
    } else {
        code = runQualification(inputs, runner: runner, loadError: loadError, reportURL: reportURL)
    }
    _ = runner
    return code
}

exit(runTool())
