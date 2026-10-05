import Foundation
import LeviathanCore

// Phase 63 component CLI. Before GO: argument parsing, the exclusive new report path and the executable digest only;
// the model SHA-256 is the caller-supplied cached explicit-refresh record value and is never computed here. EOF or any
// line other than exactly GO exits 1 before any Metal device creation or GGUF read. Exit codes: 0 passed, 6 timing
// failed, 3 correctness failed, 2 unsupported, 4 command-buffer error, 5 not ended, 1 otherwise.
// --setup-only (opt-in diagnostic): after GO, the same setup runs with boundary lines, a setup-only report is written
// and END is printed; RMSCoschedule.run is never called and no command buffer is encoded or committed. Exit 0 means
// only that every setup boundary was reached; it is not a correctness or performance pass. Exit 1 otherwise.

func say(_ line: String) {
    FileHandle.standardOutput.write(Data((line + "\n").utf8))
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let options: RMSCoscheduleOptions
do {
    options = try RMSCoschedule.parseOptions(Array(CommandLine.arguments.dropFirst()))
} catch {
    fail(String(describing: error))
}
let reportDescriptor = open(options.report, O_WRONLY | O_CREAT | O_EXCL, 0o600)
if reportDescriptor < 0 { fail("report path must be new: " + options.report) }
let reportHandle = FileHandle(fileDescriptor: reportDescriptor, closeOnDealloc: true)
let executableDigest: String
do {
    executableDigest = try Gemma4Runner.executableDigest()
} catch {
    fail("executable digest unavailable: " + String(describing: error))
}
say(RMSCoschedule.readyToken)
guard RMSCoschedule.goAccepted(readLine(strippingNewline: true)) else {
    fail("GO not received; nothing started")
}
var setupStagesReached: [String] = []
let maybeContext: RMSCoscheduleContext?
var setupError: String? = nil
do {
    // Non-escaping closure formed in top-level (main actor) code: it is invoked synchronously by make() on the same
    // actor, so it may mutate the main-actor-isolated stage collector under strict Swift 6 checking.
    maybeContext = try RMSCoscheduleSetup.make(gguf: URL(fileURLWithPath: options.gguf)) { line in
        say(line)
        if let stage = RMSCoscheduleSetup.stageName(fromDoneLine: line) { setupStagesReached.append(stage) }
    }
} catch {
    maybeContext = nil
    setupError = String(describing: error)
}
if options.setupOnly {
    let setupReport = RMSCoschedule.setupDiagnostic(
        context: maybeContext, setupError: setupError, stagesReached: setupStagesReached,
        modelSha256: options.modelSha256, executableDigest: executableDigest)
    let setupData: Data
    do {
        setupData = try RMSCoschedule.encodeSetupReport(setupReport)
    } catch {
        fail("setup report encoding failed: " + String(describing: error))
    }
    if setupData.count > RMSCoschedule.maxReportBytes {
        fail("setup report exceeds " + String(RMSCoschedule.maxReportBytes) + " bytes; not trimmed")
    }
    reportHandle.write(setupData)
    say("setup-only diagnostic: completed " + String(setupReport.setupCompleted) + "; stages reached "
        + String(setupStagesReached.count) + " of " + String(RMSCoscheduleSetup.stages.count)
        + "; nothing encoded or committed; no correctness, timing, gain or install claim")
    say(RMSCoschedule.endToken)
    exit(setupReport.setupCompleted ? 0 : 1)
}
guard let context = maybeContext else {
    fail("component setup failed: " + (setupError ?? "unknown"))
}
let report = RMSCoschedule.run(context: context, modelSha256: options.modelSha256,
                               executableDigest: executableDigest, emit: say)
let data: Data
do {
    data = try RMSCoschedule.encodeReport(report)
} catch {
    fail("report encoding failed: " + String(describing: error))
}
if data.count > RMSCoschedule.maxReportBytes {
    fail("report exceeds " + String(RMSCoschedule.maxReportBytes) + " bytes; not trimmed")
}
reportHandle.write(data)
if !report.terminal {
    say(RMSCoschedule.notEndedToken)
    exit(5)
}
say(RMSCoschedule.endToken)
exit(RMSCoschedule.exitCode(report))
