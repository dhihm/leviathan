import Foundation
import LeviathanCore

// Phase 58 pure tests: fixed protocol constants and the declared ABBA sequence, outcome statistics and status
// priority, pinned totals and device counters, options, qualification receipt refusals and the report contract.
// No Metal device, no GGUF, no model weights.

typealias Perf = RouterDeviceRoutingPerformance

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

let hexA = String(repeating: "a", count: 64)
let hexB = String(repeating: "b", count: 64)
let hexM = String(repeating: "c", count: 64)
let flat = [Double](repeating: 100, count: 7)
let fast = [95.0, 95.5, 94.8, 95.2, 95.0, 94.9, 95.1]

func checkProtocol(_ h: inout Harness) {
    let p = RouterDeviceRoutingPerfProtocol.required
    let seq = Perf.declaredSequence()
    h.check(seq.count == 16 && p.runsPerFixture == 16 && p.totalRuns == 48 && p.timedRuns == 42
            && p.fixtures.count * seq.count == p.totalRuns, "16 runs per fixture, 48 in total, 42 timed")
    h.check(seq.filter({ $0.warmup }).map({ $0.arm }) == [.off, .on] && seq[0].warmup && seq[1].warmup
            && seq[0].block == nil, "one OFF then one ON warmup first per fixture")
    var alternating = true
    for b in 0..<7 {
        let pair = seq.filter { $0.block == b }
        let want: [RouterDeviceRoutingPerfArm] = b % 2 == 0 ? [.off, .on] : [.on, .off]
        if pair.map({ $0.arm }) != want || pair.map({ $0.orderInBlock ?? -1 }) != [0, 1] { alternating = false }
    }
    h.check(alternating && p.blockOrders.count == 7, "seven fixed blocks alternate OFF/ON and ON/OFF")
    h.check(p.outputsPerRun == 32 && p.timedContinuationsPerRun == 31, "31 continuation timers for 32 outputs")
    h.check(p.offContinuationKernels == 1349 && p.offContinuationCommandBuffers == 37
            && p.onContinuationKernels == 1379 && p.onContinuationCommandBuffers == 7,
            "pinned per-forward kernels and command buffers")
    h.check(p.onContinuationKernels - p.offContinuationKernels == p.layers
            && p.offContinuationCommandBuffers - p.onContinuationCommandBuffers == p.layers,
            "ON adds 30 kernels and removes 30 command buffers per forward")
    h.check(!p.captureSnapshotsInTimedRuns && p.spreadLimit == 0.05 && p.gainThreshold == 0.02,
            "no snapshots in timed runs; 5% spread; 2% gain")
    h.check(RouterRouteReplay.expectedFrames(fixtures: RouterOverlapGateFixtures.pinned, outputs: 32, layers: 30)
            == p.qualifiedFrames, "4170 qualified frames")
    h.check(Perf.readyToken == RouterOverlapModelGate.readyToken && Perf.goToken == RouterOverlapModelGate.goToken
            && Perf.endToken == RouterOverlapModelGate.endToken, "protocol tokens are the gate constants")
    h.check(Perf.goAccepted("GO") && !Perf.goAccepted(nil) && !Perf.goAccepted("GO ") && !Perf.goAccepted("go"),
            "only an exact GO line is accepted; EOF refused")
}

func samples(_ fixture: String, off: [Double], on: [Double], fail: Int? = nil,
             drop: Int? = nil) -> [RouterDeviceRoutingPerfSample] {
    var out = [RouterDeviceRoutingPerfSample]()
    for (i, slot) in Perf.declaredSequence().enumerated() where i != drop {
        var tpot = 500.0
        if let b = slot.block { tpot = slot.arm == .off ? off[b] : on[b] }
        out.append(RouterDeviceRoutingPerfSample(fixture: fixture, slot: slot, tpotMedianWallMicros: tpot,
                                                 prefillWallMicros: 1000, checksPassed: i != fail))
    }
    return out
}

func outcome(_ off: [Double], _ on: [Double]) -> RouterDeviceRoutingPerfFixtureOutcome {
    Perf.fixtureOutcome(fixture: "en-chat", samples: samples("en-chat", off: off, on: on))
}

func checkOutcomes(_ h: inout Harness) {
    let met = outcome(flat, fast)
    h.check(met.outcome == Perf.outcomeCriteriaMet && met.allRunChecksPassed && met.offStable && met.onStable,
            "5% faster with stable arms meets the criteria")
    h.check(abs((met.gain ?? 0) - 0.05) < 1e-12 && met.offMedianTPOTMicros == 100 && met.onMedianTPOTMicros == 95,
            "gain = (OFF - ON) / OFF on the 7-run medians")
    h.check(met.offRunTPOTMicros.count == 7 && met.onRunTPOTMicros.count == 7, "warmups excluded from statistics")
    let edgeGain = outcome(flat, [Double](repeating: 98, count: 7))
    h.check(edgeGain.gainMeetsThreshold && edgeGain.outcome == Perf.outcomeCriteriaMet, "a gain of exactly 2% meets")
    h.check(outcome(flat, [Double](repeating: 99, count: 7)).outcome == Perf.outcomeNoGain, "1% gain is no gain")
    let slower = outcome(flat, [Double](repeating: 101, count: 7))
    h.check(slower.outcome == Perf.outcomeNoGain && (slower.gain ?? 0) < 0, "a slower candidate is no gain")
    var edge = flat
    edge[6] = 105
    let edgeSpread = outcome(edge, fast)
    h.check(edgeSpread.offStable && edgeSpread.outcome == Perf.outcomeCriteriaMet, "a spread of exactly 5% is stable")
    var wide = flat
    wide[3] = 106
    h.check(outcome(wide, fast).outcome == Perf.outcomeUnstable, "6% OFF spread is unstable")
    h.check(outcome(flat, [95.0, 95, 95, 95, 95, 95, 100.5]).outcome == Perf.outcomeUnstable,
            "ON spread above 5% is unstable")
    let failed = Perf.fixtureOutcome(fixture: "en-chat", samples: samples("en-chat", off: flat, on: fast, fail: 5))
    h.check(failed.outcome == Perf.outcomeRunCheckFailed, "one failed timed run fails the fixture")
    let warmFail = Perf.fixtureOutcome(fixture: "en-chat", samples: samples("en-chat", off: flat, on: fast, fail: 0))
    h.check(warmFail.outcome == Perf.outcomeRunCheckFailed, "a failed warmup fails the fixture")
    let dropped = Perf.fixtureOutcome(fixture: "en-chat", samples: samples("en-chat", off: flat, on: fast, drop: 9))
    h.check(dropped.outcome == Perf.outcomeRunCheckFailed, "a missing run is never tolerated")
    var reordered = samples("en-chat", off: flat, on: fast)
    reordered.swapAt(2, 3)
    h.check(Perf.fixtureOutcome(fixture: "en-chat", samples: reordered).outcome == Perf.outcomeRunCheckFailed,
            "a run out of the declared order fails the fixture")
    h.check(Perf.fixtureOutcome(fixture: "kr-chat", samples: samples("en-chat", off: flat, on: fast)).outcome
            == Perf.outcomeRunCheckFailed, "runs of another fixture never count")
}

func checkStatus(_ h: inout Harness) {
    let names = RouterOverlapGateFixtures.order
    let met = names.map { Perf.fixtureOutcome(fixture: $0, samples: samples($0, off: flat, on: fast)) }
    var wideOff = flat
    wideOff[0] = 110
    let unstable = Perf.fixtureOutcome(fixture: names[1], samples: samples(names[1], off: wideOff, on: fast))
    let noGain = Perf.fixtureOutcome(fixture: names[2], samples: samples(names[2], off: flat, on: flat))
    let failed = Perf.fixtureOutcome(fixture: names[0], samples: samples(names[0], off: flat, on: fast, fail: 3))
    h.check(Perf.performanceStatus(met) == .performanceCriteriaMet, "all three fixtures met")
    h.check(Perf.performanceStatus([met[0], unstable, met[2]]) == .performanceUnstable, "one unstable fixture")
    h.check(Perf.performanceStatus([met[0], met[1], noGain]) == .performanceNoGain, "one fixture below 2%")
    h.check(Perf.performanceStatus([met[0], unstable, noGain]) == .performanceUnstable, "unstable outranks no gain")
    h.check(Perf.performanceStatus([failed, unstable, noGain]) == .performanceRunCheckFailed,
            "a run check failure outranks everything")
    h.check(Perf.performanceStatus([met[0], met[1]]) == .performanceRunCheckFailed, "a missing fixture fails")
    h.check(Perf.performanceStatus([met[1], met[0], met[2]]) == .performanceRunCheckFailed, "fixture order is fixed")
    var r = RouterDeviceRoutingPerfReport()
    h.check(Perf.finalStatus(r) == .runnerNotRestored, "an unrestored runner")
    r.restoredPristine = true
    r.abortReason = "x"
    h.check(Perf.finalStatus(r) == .aborted, "abort")
    r.abortReason = nil
    h.check(Perf.finalStatus(r) == .qualificationReceiptInvalid, "a missing qualification receipt")
    if let q = decode(RouterDeviceRoutingQualificationReceipt.self, receiptObject(valid: true)) {
        r.qualification = q
        h.check(Perf.finalStatus(r) == .aborted, "a valid receipt without an identity never reaches a performance status")
    } else {
        h.check(false, "receipt decodes")
    }
}

func totals(_ k: Int, _ b: Int, count: Int = 31) -> RouterOverlapGateTotals? {
    decode(RouterOverlapGateTotals.self, [
        "prefillKernels": k * 21, "prefillBuffers": b * 21, "prefillGpuElapsedMicros": 0.0,
        "continuationKernels": [Int](repeating: k, count: count),
        "continuationBuffers": [Int](repeating: b, count: count),
        "continuationGpuElapsedMicros": [Double](repeating: 0, count: count),
    ] as [String: Any])
}

func checkCounters(_ h: inout Harness) {
    guard let off = totals(1349, 37), let on = totals(1379, 7), let short = totals(1379, 7, count: 30) else {
        h.check(false, "totals fixtures decode")
        return
    }
    h.check(Perf.pinnedTotalsFailures(off, arm: .off).isEmpty && Perf.pinnedTotalsFailures(on, arm: .on).isEmpty,
            "pinned OFF 1349/37 and ON 1379/7 accepted")
    h.check(!Perf.pinnedTotalsFailures(off, arm: .on).isEmpty && !Perf.pinnedTotalsFailures(on, arm: .off).isEmpty,
            "swapped arm totals refused")
    h.check(!Perf.pinnedTotalsFailures(short, arm: .on).isEmpty, "30 continuations refused")
    h.check(RouterDeviceRoutingModelGate.deviceTotalsRelation(on: on, off: off, layers: 30, promptCount: 21).isEmpty,
            "pinned totals satisfy the qualified ON/OFF relation")
    var d = RouterDeviceRoutingCounters()
    d.gpuSelections = 30 * 52
    d.recordChains = 30 * 52
    d.acceptedTokens = 52
    h.check(Perf.deviceCounterFailures(d, arm: .on, layers: 30, forwards: 52).isEmpty,
            "ON: 0 CPU, 30 GPU selections and 30 record chains per forward")
    h.check(!Perf.deviceCounterFailures(d, arm: .off, layers: 30, forwards: 52).isEmpty, "ON counters refused for OFF")
    var c = RouterDeviceRoutingCounters()
    c.cpuSelections = 30 * 52
    h.check(Perf.deviceCounterFailures(c, arm: .off, layers: 30, forwards: 52).isEmpty, "OFF: 30 CPU selections per forward")
    var flagged = d
    flagged.flaggedTokens = 1
    h.check(!Perf.deviceCounterFailures(flagged, arm: .on, layers: 30, forwards: 52).isEmpty, "a flagged token refused")
    var fallback = d
    fallback.cpuSelections = 1
    h.check(!Perf.deviceCounterFailures(fallback, arm: .on, layers: 30, forwards: 52).isEmpty,
            "any ON CPU selection refused")
}

func checkOptions(_ h: inout Harness) {
    let base = ["--gguf", "m", "--region", "r", "--graph", "g", "--fixtures", "f", "--component-report", "c",
                "--component-executable", "ce", "--selection-report", "s", "--selection-executable", "se",
                "--report", "o.json"]
    let qual = ["--mode", "qualification-only"] + base
    let perf = ["--mode", "performance-only"] + base + ["--qualified-report", "q.json", "--qualified-executable", "bin"]
    let a = try? Perf.parseOptions(qual)
    h.check(a?.mode == .qualificationOnly && a?.qualifiedReport == nil && a?.report == "o.json", "qualification-only parses")
    let b = try? Perf.parseOptions(perf)
    h.check(b?.mode == .performanceOnly && b?.qualifiedReport == "q.json" && b?.qualifiedExecutable == "bin",
            "performance-only parses")
    h.check((try? Perf.parseOptions(qual + ["--qualified-report", "q.json"])) == nil,
            "qualification-only refuses a qualified report")
    h.check((try? Perf.parseOptions(Array(perf.dropLast(2)))) == nil, "performance-only needs the qualified executable")
    h.check((try? Perf.parseOptions(["--mode", "both"] + base)) == nil, "unknown mode refused")
    h.check((try? Perf.parseOptions(base)) == nil, "mode required")
    h.check((try? Perf.parseOptions(qual + ["--gguf", "x"])) == nil, "duplicate option refused")
}

func prerequisiteObject(valid: Bool) -> [String: Any] {
    ["expectedSelectorSourceSha256": hexA, "expectedSelectorIRSha256": hexA, "componentReportSha256": hexA,
     "componentExecutableSha256": hexA, "selectionReportSha256": hexB, "selectionExecutableSha256": hexB,
     "failures": valid ? [String]() : ["x"]]
}

func receiptObject(valid: Bool) -> [String: Any] {
    ["qualificationReportSha256": hexA, "qualifiedExecutableSha256": hexA, "executableSha256": hexA,
     "modelSha256": hexM, "framesExact": 4170, "replayIdenticalFrames": 4170,
     "prerequisites": prerequisiteObject(valid: true), "failures": valid ? [String]() : ["x"]]
}

func checkQualification(_ h: inout Harness) {
    guard let prereq = decode(RouterDeviceRoutingPrerequisiteReceipt.self, prerequisiteObject(valid: true)),
          let badPrereq = decode(RouterDeviceRoutingPrerequisiteReceipt.self, prerequisiteObject(valid: false)) else {
        h.check(false, "prerequisite receipts decode")
        return
    }
    let junk = Perf.validateQualification(report: Data("{}".utf8), qualifiedExecutableSha256: hexA,
                                          executableSha256: hexA, modelSha256: hexM, prerequisites: prereq)
    h.check(!junk.receipt.valid && junk.report == nil, "an undecodable qualification report is refused")
    guard let base = try? RouterDeviceRoutingModelGate.encodeReport(RouterDeviceRoutingGateReport()),
          var obj = (try? JSONSerialization.jsonObject(with: base)) as? [String: Any] else {
        h.check(false, "gate report fixture encodes")
        return
    }
    let plain = Perf.validateQualification(report: base, qualifiedExecutableSha256: hexA, executableSha256: hexA,
                                           modelSha256: hexM, prerequisites: prereq)
    h.check(!plain.receipt.valid && plain.report == nil, "an unqualified gate report is refused")
    h.check(plain.receipt.qualificationReportSha256.count == 64, "the qualification report is hashed")
    obj["status"] = "qualificationPassed"
    obj["qualificationPassed"] = true
    obj["realE4GateComplete"] = true
    obj["executableDigest"] = hexA
    obj["modelSha256"] = hexM
    obj["selectorSourceSha256"] = hexA
    obj["prerequisites"] = prerequisiteObject(valid: true)
    obj["expectedFrames"] = 4170
    obj["framesChecked"] = 4170
    obj["framesExact"] = 4170
    obj["restoredPristine"] = true
    obj["refusalPassed"] = true
    guard let tampered = try? JSONSerialization.data(withJSONObject: obj) else {
        h.check(false, "tampered report encodes")
        return
    }
    let t = Perf.validateQualification(report: tampered, qualifiedExecutableSha256: hexA, executableSha256: hexA,
                                       modelSha256: hexM, prerequisites: prereq)
    h.check(!t.receipt.valid && t.report == nil && t.receipt.failures.contains(where: { $0.contains("recomputed") }),
            "passed labels without evidence are recomputed and refused")
    let other = Perf.validateQualification(report: tampered, qualifiedExecutableSha256: hexB, executableSha256: hexA,
                                           modelSha256: hexM, prerequisites: prereq)
    h.check(other.receipt.failures.contains(where: { $0.contains("differs from this executable") }),
            "the qualified executable must be this executable")
    let model = Perf.validateQualification(report: tampered, qualifiedExecutableSha256: hexA, executableSha256: hexA,
                                           modelSha256: hexB, prerequisites: prereq)
    h.check(model.receipt.failures.contains(where: { $0.contains("model SHA differs") }), "a model SHA mismatch is refused")
    let pre = Perf.validateQualification(report: tampered, qualifiedExecutableSha256: hexA, executableSha256: hexA,
                                         modelSha256: hexM, prerequisites: badPrereq)
    h.check(!pre.receipt.valid && pre.receipt.failures.contains(where: { $0.contains("prerequisite receipt") }),
            "an invalid or different prerequisite receipt is refused")
}

func encoded(_ r: RouterDeviceRoutingPerfReport) -> [String: Any] {
    guard let d = try? Perf.encodeReport(r),
          let o = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { return [:] }
    return o
}

func checkContract(_ h: inout Harness) {
    let names = ["gainQualified", "installable", "productionPerformanceClaim", "scalesSerialized", "realE4GateComplete"]
    let base = encoded(RouterDeviceRoutingPerfReport())
    h.check(names.allSatisfy({ (base[$0] as? Bool) == false }) && (base["kind"] as? String) == Perf.kind,
            "the default report encodes every contract flag false")
    let proto = base["protocolSpec"] as? [String: Any]
    h.check((proto?["totalRuns"] as? Int) == 48 && (proto?["captureSnapshotsInTimedRuns"] as? Bool) == false,
            "the fixed protocol is serialized in every report")
    var tampered = base
    for n in names { tampered[n] = true }
    tampered["status"] = "performanceCriteriaMet"
    tampered["restoredPristine"] = true
    if let r = decode(RouterDeviceRoutingPerfReport.self, tampered) {
        let e = encoded(r)
        h.check(names.allSatisfy({ (e[$0] as? Bool) == false }), "tampered flags without evidence re-emit false")
    } else {
        h.check(false, "tampered report decodes")
    }
    var withReceipt = tampered
    withReceipt["qualification"] = receiptObject(valid: true)
    if let r = decode(RouterDeviceRoutingPerfReport.self, withReceipt) {
        let e = encoded(r)
        h.check((e["realE4GateComplete"] as? Bool) == true && (e["gainQualified"] as? Bool) == false
                && (e["installable"] as? Bool) == false && (e["productionPerformanceClaim"] as? Bool) == false,
                "a valid receipt completes realE4 but a gain is never qualified without runs")
    } else {
        h.check(false, "receipt report decodes")
    }
}

func runAll() -> Int32 {
    var h = Harness()
    checkProtocol(&h)
    checkOutcomes(&h)
    checkStatus(&h)
    checkCounters(&h)
    checkOptions(&h)
    checkQualification(&h)
    checkContract(&h)
    print("router-device-routing-benchmark-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
