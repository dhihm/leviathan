// router-device-routing-component - phase-57 part 1 GPU component (diagnostic only; never installable; no
// performance claim). Required: --gguf MODEL.gguf --region REGION.json --report NEW_REPORT.json
// Before GO: argument parsing, exclusive report path, bounded region parse, cached model SHA record (never hashed),
// executable digest. BENCHMARK_READY is printed; only an exact GO line proceeds. EOF or anything else exits 1 with
// no Metal device, no mapping and no model. Part A runs on a standalone device with bounded completion; part B
// (only if part A passed) loads the runner and uses its resident mapping. TIMED_WINDOW_END is printed only when
// part A is terminal and the runner is quiescent; a part A timeout prints NOT_ENDED and exits 5.
// Exit: 0 passed; 3 entry or real-layer failure; 4 GPU error; 5 not ended; 1 error, no GO or report failure.

import Foundation
import LeviathanCore

func say(_ s: String) {
    print(s)
    fflush(nil)
}

func complain(_ s: String) {
    FileHandle.standardError.write(Data(("router-device-routing-component: " + s).utf8))
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
func writeReport(_ report: RouterDeviceRoutingComponentReport, to url: URL) -> Bool {
    let data: Data
    do { data = try RouterDeviceRoutingComponent.encodeReport(report) } catch {
        complain("report encode failed: " + String(describing: error))
        return false
    }
    guard data.count <= RouterDeviceRoutingComponent.maxReportBytes else {
        complain("report exceeds the bound; not written, not trimmed")
        return false
    }
    do { try data.write(to: url, options: [.withoutOverwriting]) } catch {
        complain("report write failed: " + String(describing: error))
        return false
    }
    return true
}

func exitCode(_ s: RouterDeviceRoutingComponentStatus) -> Int32 {
    switch s {
    case .componentPassed: return 0
    case .entryFailed, .realLayersFailed: return 3
    case .gpuError: return 4
    case .notEnded: return 5
    case .aborted: return 1
    }
}

func runTool() -> Int32 {
    let options: RouterDeviceRoutingComponentOptions
    do { options = try RouterDeviceRoutingComponent.parseOptions(Array(CommandLine.arguments.dropFirst())) } catch {
        complain(RouterDeviceRoutingComponent.usage)
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
    let modelSha256: String
    let executableDigest: String
    do {
        region = try MoERegionIR.parse(try boundedRead(URL(fileURLWithPath: options.region), limit: 1 << 19))
        guard let cached = try ModelDigestCache.shared.cachedRecord(for: ggufURL) else {
            throw RouterDeviceRoutingError("no cached model SHA-256 for " + ggufURL.path
                + "; refresh it with the separate explicit command (this tool never hashes the model)")
        }
        modelSha256 = cached.sha256
        executableDigest = try Gemma4Runner.executableDigest()
    } catch {
        complain(String(describing: error))
        return 1
    }

    say(RouterDeviceRoutingComponent.readyToken)
    guard RouterDeviceRoutingComponent.goAccepted(readLine(strippingNewline: true)) else {
        complain("expected exactly GO on stdin; no device, no mapping, no model")
        return 1
    }

    var report = RouterDeviceRoutingComponentReport()
    report.modelSha256 = modelSha256
    report.executableDigest = executableDigest
    report.selectorSourceSha256 = (try? RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4))
        .map { RouterSelectionKernel.sourceDigest($0) }
    report.selectorIRSha256 = try? RouterSelectionKernel.irDigest(RouterSelectionIR.gemma4)
    do {
        report.entry = try RouterDeviceRoutingComponent.runEntryIdentity(timeoutSeconds: 10)
    } catch {
        report.error = "entry identity threw before commit: " + String(describing: error)
    }
    if let e = report.entry, !e.terminal {
        report.status = RouterDeviceRoutingComponent.status(report)
        say(RouterDeviceRoutingComponent.notEndedToken)
        complain("GPU completion not observed within the bound; END withheld")
        writeReport(report, to: reportURL)
        return 5
    }
    var runner: Gemma4Runner? = nil
    if report.entry?.passed == true {
        do {
            let loaded = try Gemma4Runner(fileAt: ggufURL)
            runner = loaded
            report.real = try RouterDeviceRoutingComponent.runRealLayers(runner: loaded, region: region)
        } catch {
            report.error = "real E4 component: " + String(describing: error)
        }
        if let r = runner {
            report.restoredPristine = RouterOverlapModelGate.pristine(r) && !r.routerDeviceRoutingActive
        }
    }
    if let r = runner, !RouterOverlapModelGate.quiescent(r) {
        complain("runner not quiescent; END withheld")
        writeReport(report, to: reportURL)
        return 1
    }
    report.status = RouterDeviceRoutingComponent.status(report)
    say(RouterDeviceRoutingComponent.endToken)
    guard writeReport(report, to: reportURL) else { return 1 }
    say("report " + reportURL.path + " written: status " + report.status.rawValue
        + "; diagnostic only, gainQualified false, realE4GateComplete false")
    _ = runner
    return exitCode(report.status)
}

exit(runTool())
