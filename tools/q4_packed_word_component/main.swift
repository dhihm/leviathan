// q4-packed-word-component - phase-61 COMPONENT-ONLY diagnostic for paired register uint32 packed Q4_0 WEIGHT
// decoding against the production scalar g4_q4_matvec (both compiled with options nil). See
// Sources/LeviathanCore/Q4PackedWord.swift. Runs ONLY the register candidate (the phase-60 threadgroup variants
// are closed and not rerun). Never installable; no full-model, gain, busy-time or production claim.
// Before GO: argument parsing, exclusive report path, the cached model SHA record when --gguf is given (never hashed
// here) and this executable's digest. After BENCHMARK_READY only an exact GO line proceeds; EOF or anything else
// exits 1 with no Metal device and no GGUF mapping. Every command buffer completion is observed through a 10 s
// semaphore (no waitUntilCompleted); a timeout prints NOT_ENDED and exits 5 with END withheld. TIMED_WINDOW_END is
// printed only when every committed command buffer completed. The owner supervisor owns deadlines and signals.
// Exit: 0 correctness and timing checks passed; 6 timing checks failed; 3 correctness failed; 2 unsupported
// hardware; 4 command buffer error; 5 not ended; 1 otherwise.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("q4-packed-word-component: " + s).utf8))
    FileHandle.standardError.write(Data([10]))
}

@discardableResult
func writeReport(_ report: Q4PackedWordReport, to url: URL) -> Bool {
    let data: Data
    do { data = try Q4PackedWordDiagnostics.encodeReport(report) } catch {
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
    do { options = try Q4PackedWordDiagnostics.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(Q4PackedWordDiagnostics.usage)
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

    let report = Q4PackedWordDiagnostics.run(ggufURL: ggufURL, modelSha256: modelSha256,
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
