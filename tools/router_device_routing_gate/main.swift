// router-device-routing-gate - phase-57 part 2 full numerical model gate (diagnostic only; never installable; no
// performance claim). See Sources/LeviathanCore/RouterDeviceRoutingModelGate.swift.
// Before GO: argument parsing, exclusive report path, bounded region/graph/fixture/receipt reads, the cached model SHA
// record (never hashed), this executable's digest and the digests of the two prerequisite component executables
// (small files). An invalid receipt exits 2 before BENCHMARK_READY. After BENCHMARK_READY only an exact GO line
// proceeds; EOF or anything else exits 1 with no Metal device and no mapping. TIMED_WINDOW_END is printed only when
// the runner is quiescent and every replay chunk is terminal; otherwise NOT_ENDED and exit 5.
// Exit: 0 qualification passed; 3 qualification failed; 2 prerequisites failed; 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("router-device-routing-gate: " + s).utf8))
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
func writeReport(_ report: RouterDeviceRoutingGateReport, to url: URL) -> Bool {
    let data: Data
    do { data = try RouterDeviceRoutingModelGate.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= RouterDeviceRoutingModelGate.maxReportBytes else {
        complain("report exceeds 16 MiB; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

func exitCode(_ s: RouterDeviceRoutingGateStatus) -> Int32 {
    switch s {
    case .qualificationPassed: return 0
    case .qualificationFailed: return 3
    case .prerequisitesFailed: return 2
    case .replayNotEnded: return 5
    case .aborted, .runnerNotRestored: return 1
    }
}

func runTool() -> Int32 {
    let options: RouterDeviceRoutingGateOptions
    do { options = try RouterDeviceRoutingModelGate.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(RouterDeviceRoutingModelGate.usage)
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
    let ggufURL = URL(fileURLWithPath: options.gguf)
    let region: MoERegionIR
    let graph: GemmaGraphIR
    let fixtures: [RouterOverlapGateFixture]
    let modelSha256: String
    let executableDigest: String
    let receipt: RouterDeviceRoutingPrerequisiteReceipt
    do {
        region = try MoERegionIR.parse(try boundedRead(URL(fileURLWithPath: options.region), limit: 1 << 19))
        graph = try GemmaGraph.parse(try boundedRead(URL(fileURLWithPath: options.graph), limit: 8 << 20))
        fixtures = try RouterOverlapGateFixtures.parse(
            try boundedRead(URL(fileURLWithPath: options.fixtures), limit: 1 << 16))
        guard let cached = try ModelDigestCache.shared.cachedRecord(for: ggufURL) else {
            throw RouterDeviceRoutingError("no cached model SHA-256 for " + ggufURL.path
                + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
        let componentReport = try boundedRead(URL(fileURLWithPath: options.componentReport),
                                              limit: RouterDeviceRoutingModelGate.maxReceiptBytes)
        let selectionReport = try boundedRead(URL(fileURLWithPath: options.selectionReport),
                                              limit: RouterDeviceRoutingModelGate.maxReceiptBytes)
        let componentExe = try GemmaForge.fileDigest(URL(fileURLWithPath: options.componentExecutable))
        let selectionExe = try GemmaForge.fileDigest(URL(fileURLWithPath: options.selectionExecutable))
        receipt = RouterDeviceRoutingModelGate.validateReceipts(
            componentReport: componentReport, componentExecutableSha256: componentExe,
            selectionReport: selectionReport, selectionExecutableSha256: selectionExe, modelSha256: modelSha256)
    } catch {
        complain(String(describing: error))
        return 1
    }
    guard receipt.valid else {
        complain("prerequisites invalid (no device, no mapping, no report): " + receipt.failures.joined(separator: "; "))
        return 2
    }

    say(RouterDeviceRoutingModelGate.readyToken)
    guard RouterDeviceRoutingModelGate.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; nothing loaded, no device created")
        return 1
    }

    var runner: Gemma4Runner? = nil
    var report: RouterDeviceRoutingGateReport
    do {
        let loaded = try Gemma4Runner(fileAt: ggufURL)
        runner = loaded
        report = RouterDeviceRoutingModelGate.run(
            region: region, graph: graph, fixtures: fixtures, runner: loaded, modelSha256: modelSha256,
            executableDigest: executableDigest, prerequisites: receipt, emit: { say($0) })
    } catch {
        report = RouterDeviceRoutingGateReport()
        report.prerequisites = receipt
        report.abortReason = "runner load failed: " + String(describing: error)
        report.restoredPristine = true
        report.status = RouterDeviceRoutingModelGate.finalStatus(report)
    }
    if let r = runner, !RouterOverlapModelGate.quiescent(r) {
        complain("runner not quiescent; END withheld")
        writeReport(report, to: reportURL)
        return 1
    }
    if report.status == .replayNotEnded {
        say(RouterDeviceRoutingModelGate.notEndedToken)
        complain("component replay completion was not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    say(RouterDeviceRoutingModelGate.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; diagnostic only, gainQualified false, not installable, no performance claim")
    _ = runner
    return exitCode(report.status)
}

exit(runTool())
