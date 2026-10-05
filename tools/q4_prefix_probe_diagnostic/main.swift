// q4-prefix-probe-diagnostic - phase-65 ORIGINAL-KERNEL PREFIX PROBE on the frozen min-1x64 case: the unmodified
// production g4_q4_matvec and the unmodified phase-62 register source, compiled once with options nil, run
// uninstrumented on 65 prefix-masked activations in production serial term order. Diagnostic input perturbation
// only; never a candidate, never installable; no timing, no GGUF, no hashing. Protocol: parse, exclusive report
// path and executable digest before BENCHMARK_READY; exact GO before any Metal device; 10 s semaphore per command
// buffer; NOT_ENDED and exit 5 on timeout with END withheld; TIMED_WINDOW_END only when every committed command
// buffer completed. Exit: 0 completed (finding in report); 7 baseline mismatch not reproduced or prefix 64 not
// matching the baseline; 3 a probe failed finite/guard/FP64/oracle checks; 2 unsupported; 4 command buffer error;
// 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("q4-prefix-probe-diagnostic: " + s).utf8))
    FileHandle.standardError.write(Data([10]))
}

@discardableResult
func writeReport(_ report: Q4PrefixProbeReport, to url: URL) -> Bool {
    let data: Data
    do { data = try Q4PrefixProbeDiagnostics.encodeReport(report) } catch {
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
    let options: Q4PrefixProbeDiagnostics.Options
    do { options = try Q4PrefixProbeDiagnostics.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(Q4PrefixProbeDiagnostics.usage)
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

    let report = Q4PrefixProbeDiagnostics.run(executableDigest: executableDigest, emit: { say($0) })
    guard report.terminal else {
        say(Q4WeightStageDiagnostics.notEndedToken)
        complain("a command buffer completion was not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    say(Q4WeightStageDiagnostics.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue + "; first differing prefix "
        + (report.firstDifferingPrefix.map { String($0) } ?? "none")
        + "; diagnostic only with limited causality, no gain, install or production claim")
    return Q4PrefixProbeDiagnostics.exitCode(report.status)
}

exit(runTool())
