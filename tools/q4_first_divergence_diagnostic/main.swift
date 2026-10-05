// q4-first-divergence-diagnostic - phase-64 FIRST-DIVERGENCE diagnostic on the frozen min-1x64 synthetic case:
// production g4_q4_matvec versus the actual phase-62 register emitter, both uninstrumented (options nil), then
// instrumented single-row copies for decode operands and per-term accumulation traces. Not a candidate, never
// installable, no timing, no model file. Protocol: parse, exclusive report path and executable digest before
// BENCHMARK_READY; exact GO before any Metal device; 10 s semaphore per command buffer; NOT_ENDED and exit 5 on
// timeout with END withheld; TIMED_WINDOW_END only when every committed command buffer completed. Exit: 0 diagnostic
// completed (the attribution verdict is in the report); 7 mismatch not reproduced; 2 unsupported; 4 command buffer
// error; 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("q4-first-divergence-diagnostic: " + s).utf8))
    FileHandle.standardError.write(Data([10]))
}

@discardableResult
func writeReport(_ report: Q4FirstDivergenceReport, to url: URL) -> Bool {
    let data: Data
    do { data = try Q4FirstDivergenceDiagnostics.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= Q4WeightStageDiagnostics.maxReportBytes else {
        complain("report exceeds 16 MiB; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

func runTool() -> Int32 {
    let options: Q4FirstDivergenceDiagnostics.Options
    do { options = try Q4FirstDivergenceDiagnostics.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(Q4FirstDivergenceDiagnostics.usage)
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
    let executableDigest: String
    do { executableDigest = try Gemma4Runner.executableDigest() } catch {
        complain(String(describing: error))
        return 1
    }

    say(Q4WeightStageDiagnostics.readyToken)
    guard Q4WeightStageDiagnostics.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; no device created")
        return 1
    }

    let report = Q4FirstDivergenceDiagnostics.run(executableDigest: executableDigest, emit: { say($0) })
    guard report.terminal else {
        say(Q4WeightStageDiagnostics.notEndedToken)
        complain("a command buffer completion was not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    say(Q4WeightStageDiagnostics.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue + "; original cause established: "
        + String(report.originalCauseEstablished) + "; diagnostic only, no gain, install or production claim")
    return Q4FirstDivergenceDiagnostics.exitCode(report.status)
}

exit(runTool())
