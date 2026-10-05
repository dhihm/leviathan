// owned-process-diagnostics - phase-54 own-process memory, submission and
// readback diagnostics (correctness profile only; not installable; never a
// performance or production TPOT claim). See
// Sources/LeviathanCore/OwnProcessDiagnostics.swift.
//
// Required: --gguf MODEL.gguf --region REGION.json --graph GRAPH.json
//           --fixtures FIXTURES.json --report NEW_REPORT.json
// Before GO: argument parsing, the exclusive report path policy, bounded
// region/graph/fixture parsing, the cached model SHA-256 record (never
// hashed here) and the executable digest. Then READY is printed and flushed
// and the tool waits for exactly GO on stdin before the model is mapped or
// any Metal device exists; EOF or any other line exits 1 with nothing
// loaded. END is printed only after the runner is quiescent (no open
// encoder, no pending ticket), for success and failure alike; the report is
// then written exclusively and size-bounded. The supervisor owns wall-time
// limits and the process lifetime; this tool never signals or kills.
// Exit: 0 profiled checks passed; 2 gate qualification failed (nothing
// profiled); 3 profiled checks failed; 1 error, no GO, abort, runner not
// restored or report not written.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("owned-process-diagnostics: " + s).utf8))
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
        throw OwnProcessDiagnosticsError(url.path + " exceeds " + String(limit) + " bytes")
    }
    return data
}

func exitCode(_ s: OwnProcessDiagnosticsStatus) -> Int32 {
    switch s {
    case .profiledChecksPassed: return 0
    case .qualificationFailed: return 2
    case .profiledChecksFailed: return 3
    case .aborted, .runnerNotRestored: return 1
    }
}

@discardableResult
func writeReport(_ report: OwnProcessDiagnosticsReport, to url: URL) -> Bool {
    let data: Data
    do { data = try OwnProcessDiagnostics.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= OwnProcessDiagnostics.maxReportBytes else {
        complain("report of " + String(data.count) + " bytes exceeds the bound; not written")
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
        complain(String(describing: error))
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
            throw OwnProcessDiagnosticsError("no cached model SHA-256 for " + ggufURL.path
                + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(OwnProcessDiagnostics.readyToken)
    guard OwnProcessDiagnostics.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; nothing loaded, no device created")
        return 1
    }

    let preInit = OwnProcessInfo.snapshot(label: "before-runner-init", metalAllocatedBytes: nil)
    var runner: Gemma4Runner? = nil
    let report: OwnProcessDiagnosticsReport
    do {
        let loaded = try Gemma4Runner(fileAt: ggufURL)
        runner = loaded
        report = OwnProcessDiagnostics.run(
            region: region, graph: graph, fixtures: fixtures, runner: loaded,
            modelSha256: modelSha256, executableDigest: executableDigest,
            preInitSnapshot: preInit, emit: { say($0) })
    } catch {
        report = OwnProcessDiagnostics.aborted(
            reason: "runner load failed: " + String(describing: error), preInitSnapshot: preInit)
    }

    if let r = runner, !RouterOverlapModelGate.quiescent(r) {
        complain("runner not quiescent after the run; END withheld")
        writeReport(report, to: reportURL)
        return 1
    }
    say(OwnProcessDiagnostics.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; diagnostic only, gainQualified false, not installable, no production TPOT claim")
    _ = runner
    return exitCode(report.status)
}

exit(runTool())
