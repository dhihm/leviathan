// q4-register-loop-component - phase-62 COMPONENT-ONLY diagnostic: production loop shape with a per-block
// five-word register fetch as the Q4_0 byte source, against the production scalar g4_q4_matvec (both compiled with
// options nil). See Sources/LeviathanCore/Q4RegisterLoop.swift. Runs ONLY this candidate; the phase-60 staging and
// phase-61 unrolled-pair candidates are closed and not rerun. Never installable; no full-model, gain, busy-time or
// production claim. Protocol: parse, exclusive report path, cached model SHA record (never hashed) and executable
// digest before BENCHMARK_READY; exact GO before any device or GGUF mapping; 10 s semaphore per command buffer;
// NOT_ENDED and exit 5 on timeout with END withheld; TIMED_WINDOW_END only when every committed command buffer
// completed. Exit: 0 correctness and timing checks passed; 6 timing checks failed; 3 correctness failed;
// 2 unsupported hardware; 4 command buffer error; 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("q4-register-loop-component: " + s).utf8))
    FileHandle.standardError.write(Data([10]))
}

@discardableResult
func writeReport(_ report: Q4RegisterLoopReport, to url: URL) -> Bool {
    let data: Data
    do { data = try Q4RegisterLoopDiagnostics.encodeReport(report) } catch {
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
    let options: Q4WeightStageDiagnostics.Options
    do { options = try Q4RegisterLoopDiagnostics.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(Q4RegisterLoopDiagnostics.usage)
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
    var ggufURL: URL? = nil
    var modelSha256: String? = nil
    let executableDigest: String
    do {
        if let g = options.gguf {
            let url = URL(fileURLWithPath: g)
            guard fm.fileExists(atPath: url.path) else { throw Q4WeightStageError("GGUF not found: " + url.path) }
            guard let cached = try ModelDigestCache.shared.cachedRecord(for: url) else {
                throw Q4WeightStageError("no cached model SHA-256 for " + url.path
                    + "; refresh it with the separate explicit command (this tool never hashes the model)")
            }
            ggufURL = url
            modelSha256 = cached.sha256
        }
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(Q4WeightStageDiagnostics.readyToken)
    guard Q4WeightStageDiagnostics.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; no device created, no GGUF mapped")
        return 1
    }

    let report = Q4RegisterLoopDiagnostics.run(ggufURL: ggufURL, modelSha256: modelSha256,
                                               executableDigest: executableDigest, emit: { say($0) })
    guard report.terminal else {
        say(Q4WeightStageDiagnostics.notEndedToken)
        complain("a command buffer completion was not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    say(Q4WeightStageDiagnostics.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; component only, raw stage ticks are not busy time, no gain, install or production claim")
    return Q4WeightStageDiagnostics.exitCode(report.status)
}

exit(runTool())
