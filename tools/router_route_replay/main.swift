// router-route-replay - phase-56 stage-2 diagnostic: default-off scoped capture of the incumbent CPU
// router_topk decisions on the three pinned fixtures and a real GPU replay of every captured frame.
// See Sources/LeviathanCore/RouterRouteReplay.swift. Diagnostic only; never installable; no performance claim.
//
// Required: --gguf MODEL.gguf --region REGION.json --graph GRAPH.json --fixtures FIXTURES.json
//           --report NEW_REPORT.json
// Before GO: argument parsing, the exclusive report path policy, bounded region/graph/fixture parsing, the
// cached model SHA-256 record (never hashed here) and the executable digest. Then BENCHMARK_READY is printed
// and the tool waits for exactly GO; EOF or any other line exits 1 with no model mapped and no Metal device.
// TIMED_WINDOW_END is printed only when the runner is quiescent and every committed replay command buffer
// finished; a replay timeout prints NOT_ENDED and exits 5. The supervisor owns wall-time limits and the
// process lifetime; this tool never signals, kills or enumerates processes.
// Exit: 0 replayExact; 2 captureFailed; 3 replayMismatch/replayRefused; 4 gpuError; 5 notEnded;
// 1 error, no GO, abort, runner not restored or report not written.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("router-route-replay: " + s).utf8))
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
        throw RouterRouteCaptureError(url.path + " exceeds " + String(limit) + " bytes")
    }
    return data
}

func exitCode(_ s: RouterRouteReplayStatus) -> Int32 {
    switch s {
    case .replayExact: return 0
    case .captureFailed: return 2
    case .replayMismatch, .replayRefused: return 3
    case .gpuError: return 4
    case .notEnded: return 5
    case .aborted, .runnerNotRestored: return 1
    }
}

@discardableResult
func writeReport(_ report: RouterRouteReplayReport, to url: URL) -> Bool {
    let data: Data
    do { data = try RouterRouteReplayDiagnostics.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= RouterRouteReplayDiagnostics.maxReportBytes else {
        complain("report of " + String(data.count) + " bytes exceeds the bound; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

func runTool() -> Int32 {
    let options: OwnProcessDiagnosticsOptions
    do { options = try OwnProcessDiagnostics.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(RouterRouteReplayDiagnostics.usage)
        return 1
    }
    let fm = FileManager.default
    let reportURL = URL(fileURLWithPath: options.report)
    var isDirectory: ObjCBool = false
    let parentIsDirectory = fm.fileExists(atPath: reportURL.deletingLastPathComponent().path,
                                          isDirectory: &isDirectory) && isDirectory.boolValue
    if let reason = OwnProcessDiagnostics.reportPathRejection(
        path: reportURL.path, exists: fm.fileExists(atPath: reportURL.path),
        parentIsDirectory: parentIsDirectory) {
        complain(reason)
        return 1
    }
    let ggufURL = URL(fileURLWithPath: options.gguf)
    let region: MoERegionIR
    let graph: GemmaGraphIR
    let fixtures: [RouterOverlapGateFixture]
    let modelSha256: String
    let executableDigest: String
    do {
        region = try MoERegionIR.parse(try boundedRead(URL(fileURLWithPath: options.region), limit: 1 << 19))
        graph = try GemmaGraph.parse(try boundedRead(URL(fileURLWithPath: options.graph), limit: 8 << 20))
        fixtures = try RouterOverlapGateFixtures.parse(
            try boundedRead(URL(fileURLWithPath: options.fixtures), limit: 1 << 16))
        guard let cached = try ModelDigestCache.shared.cachedRecord(for: ggufURL) else {
            throw RouterRouteCaptureError("no cached model SHA-256 for " + ggufURL.path
                + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(RouterRouteReplayDiagnostics.readyToken)
    guard RouterRouteReplayDiagnostics.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; nothing loaded, no device created")
        return 1
    }

    var runner: Gemma4Runner? = nil
    let report: RouterRouteReplayReport
    do {
        let loaded = try Gemma4Runner(fileAt: ggufURL)
        runner = loaded
        report = RouterRouteReplayDiagnostics.run(
            region: region, graph: graph, fixtures: fixtures, runner: loaded, modelSha256: modelSha256,
            executableDigest: executableDigest, emit: { say($0) })
    } catch {
        var aborted = RouterRouteReplayReport()
        aborted.abortReason = "runner load failed: " + String(describing: error)
        aborted.restoredPristine = true
        aborted.status = RouterRouteReplayDiagnostics.finalStatus(aborted)
        report = aborted
    }

    if let r = runner, !RouterOverlapModelGate.quiescent(r) {
        complain("runner not quiescent after the run; END withheld")
        writeReport(report, to: reportURL)
        return 1
    }
    if !report.replayTerminal {
        say(RouterRouteReplayDiagnostics.notEndedToken)
        complain("GPU replay completion was not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    say(RouterRouteReplayDiagnostics.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; diagnostic only, gainQualified false, not installable, realE4GateComplete false")
    _ = runner
    return exitCode(report.status)
}

exit(runTool())
