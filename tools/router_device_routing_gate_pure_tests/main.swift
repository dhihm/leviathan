import Foundation
import LeviathanCore

// Phase 57 part 2 pure tests: totals relation, snapshot positions, prerequisite receipts, report contract,
// status priority, options and fixed protocol constants. No Metal device, no GGUF, no model weights.

struct Harness {
    var passed = 0
    var failed = 0

    mutating func check(_ ok: Bool, _ name: String) {
        if ok {
            passed += 1
        } else {
            failed += 1
            print("FAIL " + name)
        }
    }
}

func decode<T: Decodable>(_ type: T.Type, _ object: Any) -> T? {
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return nil }
    return try? JSONDecoder().decode(type, from: data)
}

func totals(_ pk: Int, _ pb: Int, _ ck: [Int], _ cb: [Int]) -> RouterOverlapGateTotals? {
    decode(RouterOverlapGateTotals.self, [
        "prefillKernels": pk, "prefillBuffers": pb, "prefillGpuElapsedMicros": 0.0,
        "continuationKernels": ck, "continuationBuffers": cb,
        "continuationGpuElapsedMicros": [Double](repeating: 0, count: ck.count),
    ] as [String: Any])
}

func checkTotals(_ h: inout Harness) {
    guard let off = totals(1349 * 21, 37 * 21, [Int](repeating: 1349, count: 31), [Int](repeating: 37, count: 31)),
          let on = totals(1379 * 21, 7 * 21, [Int](repeating: 1379, count: 31), [Int](repeating: 7, count: 31)),
          let badK = totals(1379 * 21, 7 * 21, [Int](repeating: 1349, count: 31), [Int](repeating: 7, count: 31)),
          let badB = totals(1379 * 21, 37 * 21, [Int](repeating: 1379, count: 31), [Int](repeating: 7, count: 31)),
          let short = totals(1379 * 21, 7 * 21, [Int](repeating: 1379, count: 30), [Int](repeating: 7, count: 30)) else {
        h.check(false, "totals fixtures decode")
        return
    }
    h.check(RouterDeviceRoutingModelGate.deviceTotalsRelation(on: on, off: off, layers: 30, promptCount: 21).isEmpty,
            "kernels +30 and command buffers -30 per forward accepted (1349 -> 1379, 37 -> 7)")
    h.check(RouterDeviceRoutingModelGate.deviceTotalsRelation(on: badK, off: off, layers: 30, promptCount: 21).count == 31,
            "continuation kernel mismatch detected per step")
    h.check(!RouterDeviceRoutingModelGate.deviceTotalsRelation(on: badB, off: off, layers: 30, promptCount: 21).isEmpty,
            "prefill command-buffer mismatch detected")
    h.check(!RouterDeviceRoutingModelGate.deviceTotalsRelation(on: short, off: off, layers: 30, promptCount: 21).isEmpty,
            "short continuation run detected")
}

func checkPositions(_ h: inout Harness) {
    h.check(RouterDeviceRoutingModelGate.snapshotPositionFailures(Array(0..<52), forwards: 52).isEmpty, "complete positions")
    h.check(!RouterDeviceRoutingModelGate.snapshotPositionFailures(Array(0..<51), forwards: 52).isEmpty, "missing position")
    h.check(!RouterDeviceRoutingModelGate.snapshotPositionFailures([0, 2, 1], forwards: 3).isEmpty, "reordered positions")
    let pinned = RouterOverlapGateFixtures.pinned
    h.check(RouterRouteReplay.expectedFrames(fixtures: pinned, outputs: 32, layers: 30) == 4170, "4170 expected frames")
}

let hexA = String(repeating: "a", count: 64)
let hexB = String(repeating: "b", count: 64)
let hexM = String(repeating: "c", count: 64)

func componentReport(source: String, ir: String, exe: String) -> Data? {
    guard let base = try? RouterDeviceRoutingComponent.encodeReport(RouterDeviceRoutingComponentReport()),
          var obj = (try? JSONSerialization.jsonObject(with: base)) as? [String: Any] else { return nil }
    let entry: [String: Any] = [
        "completion": "COMPLETED", "terminal": true, "componentCompletion": "COMPLETED", "cases": 102,
        "ordinaryCases": 94, "declaredRefusalCases": 8, "entryIdenticalRecords": 102, "entryMismatchCases": [String](),
        "ordinaryCPUExact": 94, "ordinaryUnexpectedFlagCases": [String](), "refusalMissingDeclaredCases": [String](),
        "hostMirrorFlaggedOrdinaryCases": [String](), "constantConsumerExactSlots": 816,
        "constantConsumerMismatchCases": [String](), "scoreLogExactCases": 102, "paddingCleanCases": 102,
        "guardsIntact": true,
    ]
    var layers = [[String: Any]]()
    for layer in RouterDeviceRoutingComponent.realLayers {
        var d: [String: Any] = ["layer": layer, "recordFlags": [String]()]
        for key in ["hostDomainSupported", "recordWellFormed", "idsMatchCPU", "quotientBitsMatchCPU",
                    "coefficientBitsMatchCPU", "offsetsMatchRealTable", "scoreLogExact", "guardsIntact", "controlFinite",
                    "guScratchBitIdentical", "actScratchBitIdentical", "dnScratchBitIdentical",
                    "accumulatorBitIdentical"] {
            d[key] = true
        }
        layers.append(d)
    }
    obj["entry"] = entry
    obj["real"] = ["device": "Apple M1 Max", "realOffsetsProven": 7680, "layers": layers, "restoredPristine": true] as [String: Any]
    obj["restoredPristine"] = true
    obj["status"] = "componentPassed"
    obj["executableDigest"] = exe
    obj["selectorSourceSha256"] = source
    obj["selectorIRSha256"] = ir
    obj["modelSha256"] = hexM
    return try? JSONSerialization.data(withJSONObject: obj)
}

func selectionReport(source: String, exe: String, status: String) -> Data? {
    let obj: [String: Any] = ["status": status, "sourceSHA256": source, "componentSourceSHA256": source,
                              "executableSHA256": exe, "unflaggedMismatches": 0, "unexpectedFlags": 0,
                              "guardsIntact": true, "gainQualified": false]
    return try? JSONSerialization.data(withJSONObject: obj)
}

func checkReceipts(_ h: inout Harness) {
    guard let src = try? RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4),
          let ir = try? RouterSelectionKernel.irDigest(RouterSelectionIR.gemma4) else {
        h.check(false, "selector source and IR emit")
        return
    }
    let source = RouterSelectionKernel.sourceDigest(src)
    guard let comp = componentReport(source: source, ir: ir, exe: hexA),
          let sel = selectionReport(source: source, exe: hexB, status: "PASS") else {
        h.check(false, "receipt fixtures encode")
        return
    }
    let good = RouterDeviceRoutingModelGate.validateReceipts(componentReport: comp, componentExecutableSha256: hexA,
                                                            selectionReport: sel, selectionExecutableSha256: hexB,
                                                            modelSha256: hexM)
    h.check(good.valid && good.expectedSelectorSourceSha256 == source, "matching receipts validate")
    h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: comp, componentExecutableSha256: hexB,
                                                          selectionReport: sel, selectionExecutableSha256: hexB,
                                                          modelSha256: hexM).valid, "component executable mismatch refused")
    h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: comp, componentExecutableSha256: hexA,
                                                          selectionReport: sel, selectionExecutableSha256: hexB,
                                                          modelSha256: hexA).valid, "model SHA mismatch refused")
    if let stale = componentReport(source: hexA, ir: ir, exe: hexA) {
        h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: stale, componentExecutableSha256: hexA,
                                                              selectionReport: sel, selectionExecutableSha256: hexB,
                                                              modelSha256: hexM).valid, "stale selector source refused")
    }
    if let failedSel = selectionReport(source: source, exe: hexB, status: "FAIL") {
        h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: comp, componentExecutableSha256: hexA,
                                                              selectionReport: failedSel, selectionExecutableSha256: hexB,
                                                              modelSha256: hexM).valid, "failed selection report refused")
    }
    if var obj = (try? JSONSerialization.jsonObject(with: comp)) as? [String: Any],
       var real = obj["real"] as? [String: Any], var layers = real["layers"] as? [[String: Any]] {
        layers[1]["accumulatorBitIdentical"] = false
        real["layers"] = layers
        obj["real"] = real
        if let tampered = try? JSONSerialization.data(withJSONObject: obj) {
            h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: tampered, componentExecutableSha256: hexA,
                                                                  selectionReport: sel, selectionExecutableSha256: hexB,
                                                                  modelSha256: hexM).valid,
                    "component status recomputed (a failed real layer behind a passed label is refused)")
        }
    }
    h.check(!RouterDeviceRoutingModelGate.validateReceipts(componentReport: Data(), componentExecutableSha256: hexA,
                                                          selectionReport: sel, selectionExecutableSha256: hexB,
                                                          modelSha256: hexM).valid, "undecodable report refused")
}

func gateReport(_ mutate: (inout [String: Any]) -> Void) -> RouterDeviceRoutingGateReport? {
    guard let base = try? RouterDeviceRoutingModelGate.encodeReport(RouterDeviceRoutingGateReport()),
          var obj = (try? JSONSerialization.jsonObject(with: base)) as? [String: Any] else { return nil }
    mutate(&obj)
    return decode(RouterDeviceRoutingGateReport.self, obj)
}

func receiptObject(valid: Bool) -> [String: Any] {
    ["expectedSelectorSourceSha256": hexA, "expectedSelectorIRSha256": hexA, "componentReportSha256": hexA,
     "componentExecutableSha256": hexA, "selectionReportSha256": hexB, "selectionExecutableSha256": hexB,
     "failures": valid ? [String]() : ["x"]]
}

func encodedFlags(_ r: RouterDeviceRoutingGateReport) -> [String: Any] {
    guard let d = try? RouterDeviceRoutingModelGate.encodeReport(r),
          let obj = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { return [:] }
    return obj
}

func checkContract(_ h: inout Harness) {
    let base = encodedFlags(RouterDeviceRoutingGateReport())
    let flags = ["gainQualified", "installable", "productionPerformanceClaim", "scalesSerialized", "realE4GateComplete"]
    h.check(flags.allSatisfy { (base[$0] as? Bool) == false }, "default report encodes every contract flag false")
    if let tampered = gateReport({ o in
        for f in flags { o[f] = true }
        o["status"] = "qualificationPassed"
        o["qualificationPassed"] = true
    }) {
        let e = encodedFlags(tampered)
        h.check(flags.allSatisfy { (e[$0] as? Bool) == false }, "tampered flags without a valid receipt re-emit false")
    } else {
        h.check(false, "tampered report decodes")
    }
    if let earned = gateReport({ o in
        o["status"] = "qualificationPassed"
        o["qualificationPassed"] = true
        o["prerequisites"] = receiptObject(valid: true)
    }) {
        let e = encodedFlags(earned)
        h.check((e["realE4GateComplete"] as? Bool) == true && (e["gainQualified"] as? Bool) == false
                && (e["installable"] as? Bool) == false, "realE4GateComplete derived only from status, result and receipt")
    } else {
        h.check(false, "earned report decodes")
    }
    if let invalidReceipt = gateReport({ o in
        o["status"] = "qualificationPassed"
        o["qualificationPassed"] = true
        o["prerequisites"] = receiptObject(valid: false)
    }) {
        h.check((encodedFlags(invalidReceipt)["realE4GateComplete"] as? Bool) == false, "invalid receipt never completes the gate")
    }
}

func checkStatus(_ h: inout Harness) {
    typealias G = RouterDeviceRoutingModelGate
    var r = RouterDeviceRoutingGateReport()
    h.check(G.finalStatus(r) == .runnerNotRestored, "unrestored runner")
    r.restoredPristine = true
    r.abortReason = "x"
    h.check(G.finalStatus(r) == .aborted, "abort")
    r.abortReason = nil
    h.check(G.finalStatus(r) == .prerequisitesFailed, "missing receipt")
    if let withReceipt = gateReport({ o in
        o["restoredPristine"] = true
        o["prerequisites"] = receiptObject(valid: true)
    }) {
        var p = withReceipt
        h.check(G.finalStatus(p) == .qualificationFailed, "valid receipt without qualification fails")
        p.qualificationPassed = true
        h.check(G.finalStatus(p) == .qualificationPassed, "qualification passed")
        var replay = RouterDeviceRoutingGateReplay()
        replay.terminal = false
        p.replay = replay
        h.check(G.finalStatus(p) == .replayNotEnded, "replay not ended takes priority (END withheld)")
    } else {
        h.check(false, "receipt report decodes")
    }
}

func checkOptions(_ h: inout Harness) {
    let full = ["--gguf", "m", "--region", "r", "--graph", "g", "--fixtures", "f", "--component-report", "c",
                "--component-executable", "ce", "--selection-report", "s", "--selection-executable", "se",
                "--report", "o.json"]
    let parsed = try? RouterDeviceRoutingModelGate.parseOptions(full)
    h.check(parsed?.componentExecutable == "ce" && parsed?.selectionReport == "s" && parsed?.report == "o.json", "options parse")
    h.check((try? RouterDeviceRoutingModelGate.parseOptions(Array(full.dropLast(2)))) == nil, "missing option refused")
    h.check((try? RouterDeviceRoutingModelGate.parseOptions(full + ["--gguf", "x"])) == nil, "duplicate option refused")
    h.check(RouterDeviceRoutingModelGate.goAccepted("GO") && !RouterDeviceRoutingModelGate.goAccepted(nil)
            && !RouterDeviceRoutingModelGate.goAccepted("go"), "only exact GO accepted; EOF refused")
    h.check(RouterDeviceRoutingModelGate.readyToken == RouterOverlapModelGate.readyToken
            && RouterDeviceRoutingModelGate.endToken == RouterOverlapModelGate.endToken, "protocol tokens are the gate constants")
    h.check(RouterDeviceRoutingModelGate.refusalTestNames.count == 16
            && Set(RouterDeviceRoutingModelGate.refusalTestNames).count == 16, "sixteen distinct refusal tests")
    h.check(RouterDeviceRoutingModelGate.refusalTestNames.contains("direct-forward-inside-device-scope"),
            "direct forward refusal inside the device scope is a predetermined probe")
    h.check(RouterDeviceRoutingModelGate.maxReportBytes == 16 << 20, "report bound 16 MiB")
}

func runAll() -> Int32 {
    var h = Harness()
    checkTotals(&h)
    checkPositions(&h)
    checkReceipts(&h)
    checkContract(&h)
    checkStatus(&h)
    checkOptions(&h)
    print("router-device-routing-gate-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
