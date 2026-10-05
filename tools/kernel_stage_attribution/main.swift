// kernel-stage-attribution - phase-59 default-off GPU kernel stage attribution of the incumbent OFF path using actual
// encoder-stage timestamp sampling (diagnostic only; never installable; no performance, gain or busy-time claim).
// See Sources/LeviathanCore/KernelStageAttribution.swift.
// Before GO: argument parsing, exclusive report path, bounded region/graph/fixture reads, the cached model SHA record
// (never hashed) and this executable's digest. After BENCHMARK_READY only an exact GO line proceeds; EOF or anything
// else exits 1 with no Metal device and no mapping. TIMED_WINDOW_END is printed only when the runner is quiescent,
// restored and no sampling scope is active; otherwise NOT_ENDED. The owned supervisor owns every deadline: the
// runner's synchronous command-buffer waits have no internal bound.
// Exit: 0 checks passed; 3 checks failed; 2 stage sampling unsupported; 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("kernel-stage-attribution: " + s).utf8))
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
        throw KernelStageError(url.path + " exceeds " + String(limit) + " bytes")
    }
    return data
}

@discardableResult
func writeReport(_ report: KernelStageReport, to url: URL) -> Bool {
    let data: Data
    do { data = try KernelStageAttribution.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= KernelStageAttribution.maxReportBytes else {
        complain("report exceeds 16 MiB; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

func exitCode(_ s: KernelStageStatus) -> Int32 {
    switch s {
    case .profiledChecksPassed: return 0
    case .profiledChecksFailed: return 3
    case .unsupported: return 2
    case .aborted, .runnerNotRestored: return 1
    }
}

func runTool() -> Int32 {
    let options: KernelStageOptions
    do { options = try KernelStageAttribution.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(KernelStageAttribution.usage)
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
    do {
        region = try MoERegionIR.parse(try boundedRead(URL(fileURLWithPath: options.region), limit: 1 << 19))
        graph = try GemmaGraph.parse(try boundedRead(URL(fileURLWithPath: options.graph), limit: 8 << 20))
        fixtures = try RouterOverlapGateFixtures.parse(
            try boundedRead(URL(fileURLWithPath: options.fixtures), limit: 1 << 16))
        guard let cached = try ModelDigestCache.shared.cachedRecord(for: ggufURL) else {
            throw KernelStageError("no cached model SHA-256 for " + ggufURL.path
                + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(KernelStageAttribution.readyToken)
    guard KernelStageAttribution.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; nothing loaded, no device created")
        return 1
    }

    var runner: Gemma4Runner? = nil
    var report: KernelStageReport
    do {
        let loaded = try Gemma4Runner(fileAt: ggufURL)
        runner = loaded
        report = KernelStageAttribution.run(
            region: region, graph: graph, fixtures: fixtures, runner: loaded, modelSha256: modelSha256,
            executableDigest: executableDigest, processSnapshots: options.ownProcessSnapshots, emit: { say($0) })
    } catch {
        report = KernelStageReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.abortReason = "runner load failed: " + String(describing: error)
        report.restoredPristine = true
        report.status = KernelStageAttribution.finalStatus(report)
    }
    let quiescent = runner.map { RouterOverlapModelGate.quiescent($0) && !$0.kernelStageProfileActive } ?? true
    guard quiescent, report.restoredPristine else {
        if !quiescent { say(KernelStageAttribution.notEndedToken) }
        complain("runner not quiescent, not restored or a sampling scope is active; END withheld")
        writeReport(report, to: reportURL)
        return quiescent ? 1 : 5
    }
    say(KernelStageAttribution.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; diagnostic only, encoder intervals are raw stage ticks and not busy time, no gain or production claim")
    _ = runner
    return exitCode(report.status)
}

exit(runTool())
