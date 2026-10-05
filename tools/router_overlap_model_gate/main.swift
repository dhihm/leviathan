// router-overlap-model-gate - bounded router overlap MODEL correctness gate
// and NORMAL-path matched performance harness (diagnostic only; see
// Sources/LeviathanCore/RouterOverlapModelGate.swift).
//
// Modes: --qualification-only (default) or --performance. Required:
//   --gguf MODEL.gguf --region REGION.json --graph GRAPH.json
//   --fixtures FIXTURES.json --report NEW_REPORT.json
// Before GO: arguments, exclusive report path, bounded region/graph/fixture
// parsing, the cached model SHA-256 record (never hashed here) and the
// executable digest. Then BENCHMARK_READY is printed and flushed and the
// tool waits for exactly GO on stdin BEFORE the model is loaded or any Metal
// object exists. One window: qualification (and, with --performance,
// QUALIFICATION_PASSED plus the fixed benchmark); TIMED_WINDOW_END is printed
// only after every command buffer completed and no overlap ticket is
// pending; the report is written exclusively afterwards.
// Exit: 0 qualification-only passed / performance internal criteria met
// (gain never qualified); 2 qualification failed; 3 performance run-check
// failure, noise or no gain; 1 error, no GO, abort or runner not restored.

import Foundation
import LeviathanCore

struct ToolError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

enum Tool {
    static let usage = "usage: router-overlap-model-gate [--qualification-only | --performance] "
        + "--gguf M.gguf --region R.json --graph G.json --fixtures F.json --report NEW_REPORT.json"
}

struct Options {
    let mode: RouterOverlapModelGateMode
    let gguf: URL
    let region: URL
    let graph: URL
    let fixtures: URL
    let report: URL
}

func parseOptions(_ argv: [String]) throws -> Options {
    let valued = ["--gguf", "--region", "--graph", "--fixtures", "--report"]
    var values = [String: URL]()
    var mode: RouterOverlapModelGateMode? = nil
    var i = 0
    while i < argv.count {
        let a = argv[i]
        if a == "--qualification-only" || a == "--performance" {
            guard mode == nil else { throw ToolError(Tool.usage) }
            mode = a == "--performance" ? .performance : .qualificationOnly
            i += 1
        } else if valued.contains(a), i + 1 < argv.count, values[a] == nil {
            values[a] = URL(fileURLWithPath: argv[i + 1])
            i += 2
        } else {
            throw ToolError(Tool.usage)
        }
    }
    guard Set(values.keys) == Set(valued), let gguf = values["--gguf"], let region = values["--region"],
          let graph = values["--graph"], let fixtures = values["--fixtures"],
          let report = values["--report"] else {
        throw ToolError(Tool.usage)
    }
    return Options(mode: mode ?? .qualificationOnly, gguf: gguf, region: region, graph: graph,
                   fixtures: fixtures, report: report)
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
        throw ToolError(url.path + " exceeds " + String(limit) + " bytes")
    }
    return data
}

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("router-overlap-model-gate: " + s + "\n").utf8))
}

func exitCode(_ r: RouterOverlapGateReport) -> Int32 {
    switch r.status {
    case .qualificationPassed: return r.mode == .qualificationOnly ? 0 : 1
    case .performanceInternalCriteriaMetExternalEligibilityPending: return 0
    case .qualificationFailed: return 2
    case .performanceRunCheckFailed, .performanceUnstable, .performanceNoGain: return 3
    case .runnerNotRestored, .aborted: return 1
    }
}

func runTool() -> Int32 {
    let options: Options
    do { options = try parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(String(describing: error))
        return 1
    }
    let fm = FileManager.default
    guard !fm.fileExists(atPath: options.report.path) else {
        complain("refusing to overwrite " + options.report.path)
        return 1
    }
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: options.report.deletingLastPathComponent().path,
                        isDirectory: &isDirectory), isDirectory.boolValue else {
        complain("report directory does not exist")
        return 1
    }
    let region: MoERegionIR
    let graph: GemmaGraphIR
    let fixtures: [RouterOverlapGateFixture]
    let modelSha256: String
    let executableDigest: String
    do {
        region = try MoERegionIR.parse(try boundedRead(options.region, limit: 1 << 19))
        graph = try GemmaGraph.parse(try boundedRead(options.graph, limit: 8 << 20))
        fixtures = try RouterOverlapGateFixtures.parse(try boundedRead(options.fixtures, limit: 1 << 16))
        guard let cached = try ModelDigestCache.shared.cachedRecord(for: options.gguf) else {
            throw ToolError("no cached model SHA-256 for " + options.gguf.path
                            + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(RouterOverlapModelGate.readyToken)
    guard let go = readLine(strippingNewline: true), go == RouterOverlapModelGate.goToken else {
        complain("expected exactly GO on stdin; nothing loaded, nothing dispatched")
        return 1
    }

    var runner: Gemma4Runner? = nil
    let report: RouterOverlapGateReport
    do {
        let loaded = try Gemma4Runner(fileAt: options.gguf)
        runner = loaded
        report = RouterOverlapModelGate.run(
            mode: options.mode, region: region, graph: graph, fixtures: fixtures, runner: loaded,
            modelSha256: modelSha256, executableDigest: executableDigest, emit: { say($0) })
    } catch {
        report = RouterOverlapModelGate.aborted(
            mode: options.mode, reason: "runner load failed: " + String(describing: error))
    }

    if let r = runner, !RouterOverlapModelGate.quiescent(r) {
        complain("runner not quiescent after the run; TIMED_WINDOW_END withheld")
        if let data = try? RouterOverlapModelGate.encodeReport(report, pretty: true) {
            try? data.write(to: options.report, options: [.withoutOverwriting])
        }
        return 1
    }
    say(RouterOverlapModelGate.endToken)

    do {
        let data = try RouterOverlapModelGate.encodeReport(report, pretty: true)
        try data.write(to: options.report, options: [.withoutOverwriting])
    } catch {
        complain("report write failed: " + String(describing: error))
        return 1
    }
    say("report " + options.report.path + " written: status " + report.status.rawValue
        + "; gainQualified false; no artifact, no activation, no winner")
    _ = runner
    return exitCode(report)
}

exit(runTool())
