// owned-process-diagnostics-pure-tests - checks of the PUBLIC phase-54
// own-process diagnostic helpers: native own-process task_info capture and
// its delta rules, the breakdown record's declared nesting, the full-
// vocabulary reference comparison, readback-site expectations, the
// READY/GO/report-path protocol, K/V capacity derivation, status precedence,
// report serialization, and that the phase-53 router overlap scope still
// refuses breakdown instrumentation. Every assertion calls real shared
// code. No Metal device, GPU, GGUF, model weights or hashing.
//
// Exit 0 when every check passed, 1 otherwise; the printed count is the
// actual number of checks executed.

import Foundation
import LeviathanCore

struct Harness {
    private(set) var passed = 0
    private(set) var failed: [String] = []

    mutating func check(_ condition: Bool, _ name: String) {
        if condition {
            passed += 1
            print("PASS " + name)
        } else {
            failed.append(name)
            print("FAIL " + name)
        }
    }
}

// MARK: - task_info

func testTaskInfo(_ h: inout Harness) {
    h.check(OwnProcessInfo.eventsInfoCount == 8,
            "task_info: TASK_EVENTS_INFO count is the 8-word native struct size")
    let needed = OwnProcessInfo.vmInfoCountForPhysFootprint
    h.check(needed > 0 && needed <= OwnProcessInfo.vmInfoCount,
            "task_info: the phys_footprint word bound lies inside the native TASK_VM_INFO struct")
    let a = OwnProcessInfo.snapshot(label: "a", metalAllocatedBytes: nil)
    h.check(a.events.status == "ok" && a.events.faults != nil && a.events.pageins != nil
                && a.events.cowFaults != nil && a.events.contextSwitches != nil,
            "task_info: own-process TASK_EVENTS_INFO succeeds with every counter known")
    h.check(a.vm.status == "ok" && (a.vm.physFootprintBytes ?? 0) > 0
                && (a.vm.residentBytes ?? 0) > 0 && (a.vm.virtualBytes ?? 0) > 0,
            "task_info: own-process TASK_VM_INFO reports positive virtual, resident and phys_footprint bytes")
    h.check(a.metalCurrentAllocatedBytes == nil && a.metalStatus.hasPrefix("UNKNOWN"),
            "task_info: a snapshot without a runner device records Metal allocation as UNKNOWN, not zero")
    var pages = [UInt8](repeating: 0, count: 32 << 20)
    var i = 0
    while i < pages.count {
        pages[i] = 1
        i += 4096
    }
    let b = OwnProcessInfo.snapshot(label: "b", metalAllocatedBytes: 0)
    let d = OwnProcessInfo.delta(a, b)
    let faults = d.counters.first { $0.name == "faults" }
    h.check(faults?.status == .known && (faults?.delta ?? 0) > 0,
            "task_info: touching 32 MiB of fresh pages yields a known positive cumulative fault delta")
    h.check(d.counters.filter { $0.monotonic }.allSatisfy { $0.status == .known && ($0.delta ?? -1) >= 0 },
            "task_info: every own-process cumulative counter delta between two snapshots is known and nonnegative")
    h.check(b.metalStatus == "ok" && b.metalCurrentAllocatedBytes == 0 && d.elapsedNanos != nil
                && d.from == "a" && d.to == "b",
            "task_info: an explicitly supplied Metal sample is recorded and the elapsed interval is known")
    h.check(pages.reduce(0) { $0 + Int($1) } == (32 << 20) / 4096,
            "task_info: the touched buffer stayed live across the second snapshot")
}

func testDeltas(_ h: inout Harness) {
    let up = OwnProcessInfo.counterDelta(name: "faults", before: 10, after: 15, monotonic: true)
    h.check(up.status == .known && up.delta == 5, "delta: an increasing cumulative counter gives a known delta")
    let down = OwnProcessInfo.counterDelta(name: "faults", before: 15, after: 10, monotonic: true)
    h.check(down.status == .nonMonotonic && down.delta == nil,
            "delta: a decreasing cumulative counter is nonMonotonic with no value (wrap or reset is never a number)")
    let gauge = OwnProcessInfo.counterDelta(name: "physFootprintBytes", before: 15, after: 10, monotonic: false)
    h.check(gauge.status == .known && gauge.delta == -5, "delta: a decreasing gauge gives a known negative delta")
    let unknownBefore = OwnProcessInfo.counterDelta(name: "pageins", before: nil, after: 7, monotonic: true)
    let unknownAfter = OwnProcessInfo.counterDelta(name: "pageins", before: 7, after: nil, monotonic: true)
    h.check(unknownBefore.status == .unknown && unknownBefore.delta == nil && unknownBefore.after == 7
                && unknownAfter.status == .unknown && unknownAfter.delta == nil,
            "delta: an unknown side gives an unknown delta, never zero")
    let overflow = OwnProcessInfo.counterDelta(name: "virtualBytes", before: Int64.min, after: Int64.max,
                                               monotonic: false)
    h.check(overflow.status == .unknown && overflow.delta == nil, "delta: an overflowing difference is unknown")
    let failedEvents = OwnProcessTaskEvents(status: "UNKNOWN(test)", faults: nil, pageins: nil,
                                            cowFaults: nil, contextSwitches: nil)
    let okEvents = OwnProcessTaskEvents(status: "ok", faults: 100, pageins: 3, cowFaults: 1,
                                        contextSwitches: 9)
    let vm = OwnProcessVMInfo(status: "ok", virtualBytes: 1000, residentBytes: 500, physFootprintBytes: 400)
    let s1 = OwnProcessTaskSnapshot(label: "s1", uptimeNanos: 2000, events: failedEvents, vm: vm,
                                    metalStatus: "UNKNOWN(test)", metalCurrentAllocatedBytes: nil)
    let s2 = OwnProcessTaskSnapshot(label: "s2", uptimeNanos: 1000, events: okEvents, vm: vm,
                                    metalStatus: "ok", metalCurrentAllocatedBytes: 64)
    let d = OwnProcessInfo.delta(s1, s2)
    h.check(d.counters.filter { $0.monotonic }.allSatisfy { $0.status == .unknown && $0.delta == nil },
            "delta: a failed TASK_EVENTS_INFO side leaves every cumulative delta unknown")
    h.check(d.counters.first { $0.name == "physFootprintBytes" }?.delta == 0
                && d.counters.first { $0.name == "metalCurrentAllocatedBytes" }?.status == .unknown,
            "delta: known gauges compare while an unsampled Metal side stays unknown")
    h.check(d.elapsedNanos == nil, "delta: a snapshot pair out of uptime order has no elapsed interval")
    h.check(d.counters.map { $0.name } == ["faults", "pageins", "cowFaults", "contextSwitches",
                                            "virtualBytes", "residentBytes", "physFootprintBytes",
                                            "metalCurrentAllocatedBytes"],
            "delta: counters are reported in the fixed order, cumulative first")
}

// MARK: - Breakdown nesting

func testBreakdown(_ h: inout Harness) {
    let zero = Gemma4SubmissionBreakdown()
    h.check(zero.violations().isEmpty, "breakdown: an all-zero record has no violations")
    var ok = Gemma4SubmissionBreakdown()
    ok.flushMicros = 100
    ok.flushEndEncodingMicros = 1
    ok.flushCommitMicros = 4
    ok.flushWaitMicros = 90
    ok.flushCount = 2
    ok.flushCommandBufferIntervalMicros = 150
    h.check(ok.violations().isEmpty,
            "breakdown: nested flush phases inside the flush total pass; a GPU interval above the CPU flush wall is not a violation")
    var over = ok
    over.flushWaitMicros = 96
    let overViolations = over.violations()
    h.check(overViolations.count == 1 && overViolations[0].contains("exceed"),
            "breakdown: nested flush phases summing above the flush total are exactly one violation")
    var uncounted = Gemma4SubmissionBreakdown()
    uncounted.readCopyMicros = 5
    h.check(uncounted.violations().contains { $0.contains("readCopy") },
            "breakdown: copy time without a counted copy is a violation")
    var negative = Gemma4SubmissionBreakdown()
    negative.encodeMicros = -1
    h.check(!negative.violations().isEmpty, "breakdown: a negative interval is a violation")
    var nonFinite = Gemma4SubmissionBreakdown()
    nonFinite.routerSelectionMicros = .nan
    nonFinite.routerSelectionCount = 1
    h.check(!nonFinite.violations().isEmpty, "breakdown: a non-finite interval is a violation")
    var sitesOnly = Gemma4SubmissionBreakdown()
    sitesOnly.siteRouterMicros = 50
    sitesOnly.siteRouterCount = 30
    h.check(sitesOnly.violations().isEmpty,
            "breakdown: site intervals are nested containers and never checked additively against flush")
    h.check(ok.minus(zero) == ok && ok.minus(ok) == zero, "breakdown: minus is field-wise and exact")
    let data = try? JSONEncoder().encode(ok)
    let back = data.flatMap { try? JSONDecoder().decode(Gemma4SubmissionBreakdown.self, from: $0) }
    h.check(back == ok, "breakdown: Codable round trip preserves every field")
    var keyCount = -1
    var hasKeys = false
    if let data, let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        keyCount = obj.count
        hasKeys = obj["flushCommandBufferIntervalMicros"] != nil && obj["routerSelectionMicros"] != nil
            && obj["siteLogitsCount"] != nil
    }
    h.check(keyCount == 24 && hasKeys, "breakdown: the report schema has exactly 24 named fields")
    let nesting = Gemma4SubmissionBreakdown.nesting
    h.check(nesting.contains { $0.contains("not busy time") } && nesting.contains { $0.contains("never added") },
            "breakdown: the declared nesting forbids additive and busy-time readings")
}

// MARK: - Reference comparison

func testComparison(_ h: inout Harness) {
    let contract = OwnProcessDiagnostics.correctnessContract
    var same = OwnProcessLogitComparison()
    for s in 0..<32 { same.add(step: s, logits: [1, -2, 30], reference: [1, -2, 30]) }
    h.check(same.comparedSteps == 32 && same.maxAbsDelta == 0 && same.maxPerElementNormalized == 0
                && same.failures(contract: contract, expectedSteps: 32).isEmpty,
            "comparison: 32 identical outputs pass with zero differences")
    var vectorOnly = OwnProcessLogitComparison()
    vectorOnly.add(step: 0, logits: [30, 0.5005], reference: [30, 0.5])
    h.check(vectorOnly.maxVectorNormalized < contract && vectorOnly.maxPerElementNormalized > contract
                && vectorOnly.maxAbsDelta > contract,
            "comparison: the per-element bound is stricter than the gate's vector-normalized bound")
    let fails = vectorOnly.failures(contract: contract, expectedSteps: 1)
    h.check(fails.count == 1 && fails[0].contains("per-element"),
            "comparison: a per-element excess fails even when the vector-normalized metric passes")
    h.check(vectorOnly.worstPerElement?.index == 1 && vectorOnly.worstPerElement?.step == 0,
            "comparison: the worst element records its step and vocabulary index")
    var invalid = OwnProcessLogitComparison()
    invalid.add(step: 0, logits: [1, 2], reference: [1])
    invalid.add(step: 1, logits: [.nan, 2], reference: [1, 2])
    invalid.add(step: 2, logits: [], reference: [])
    h.check(invalid.invalidSteps == 3 && invalid.comparedSteps == 0
                && invalid.failures(contract: contract, expectedSteps: 3).count == 2,
            "comparison: length-mismatched, non-finite and empty outputs are invalid, never silently compared")
    var short = OwnProcessLogitComparison()
    short.add(step: 0, logits: [1], reference: [1])
    h.check(short.failures(contract: contract, expectedSteps: 32).contains { $0.contains("compared 1 of 32") },
            "comparison: fewer than 32 compared outputs fail")
    var steps = OwnProcessLogitComparison()
    steps.add(step: 0, logits: [1.00001], reference: [1])
    steps.add(step: 1, logits: [1.00005], reference: [1])
    steps.add(step: 2, logits: [1.00002], reference: [1])
    h.check(steps.worstPerElement?.step == 1 && steps.perStepMaxPerElementNormalized.count == 3
                && steps.failures(contract: contract, expectedSteps: 3).isEmpty,
            "comparison: the worst element tracks the maximum across steps and small diffs pass")
}

// MARK: - Readback sites

func testSites(_ h: inout Harness) {
    let e = OwnProcessSiteExpectation.perForward(layers: 30, copies: 5)
    h.check(e.embedSites == 1 && e.routerSites == 30 && e.copySites == 5 && e.logitsSites == 1
                && e.routerSelections == 30 && e.readCopies == 38 && e.writeCopies == 6,
            "sites: per-forward readback expectation for 30 layers and 5 copies")
    var b = Gemma4SubmissionBreakdown()
    b.siteEmbedCount = 21
    b.siteRouterCount = 630
    b.siteCopyCount = 105
    b.siteLogitsCount = 21
    b.routerSelectionCount = 630
    b.readCopyCount = 38 * 21
    b.writeCopyCount = 6 * 21
    b.bindingCheckCount = 1
    h.check(e.mismatches(b, forwards: 21, calls: 1, label: "prefill").isEmpty,
            "sites: an exact 21-forward prefill record has no mismatches")
    var extra = b
    extra.readCopyCount += 1
    h.check(e.mismatches(extra, forwards: 21, calls: 1, label: "prefill") == ["prefill readCopyCount 799 != 798"],
            "sites: one extra readback is exactly one named mismatch")
    var twoBindings = b
    twoBindings.bindingCheckCount = 2
    h.check(e.mismatches(twoBindings, forwards: 21, calls: 1, label: "prefill").count == 1,
            "sites: binding checks are counted per executor call, not per forward")
}

// MARK: - Protocol inputs

func testProtocolInputs(_ h: inout Harness) {
    let full = ["--gguf", "m.gguf", "--region", "r.json", "--graph", "g.json",
                "--fixtures", "f.json", "--report", "out.json"]
    let parsed = try? OwnProcessDiagnostics.parseOptions(full)
    h.check(parsed?.gguf == "m.gguf" && parsed?.region == "r.json" && parsed?.report == "out.json",
            "protocol: the five required options parse")
    h.check((try? OwnProcessDiagnostics.parseOptions(Array(full.dropLast(2)))) == nil,
            "protocol: a missing option is refused")
    h.check((try? OwnProcessDiagnostics.parseOptions(full + ["--gguf", "x.gguf"])) == nil,
            "protocol: a duplicated option is refused")
    h.check((try? OwnProcessDiagnostics.parseOptions(full + ["--performance"])) == nil,
            "protocol: an unknown flag is refused")
    h.check((try? OwnProcessDiagnostics.parseOptions(["--gguf", "--region"] + Array(full.dropFirst(2)))) == nil,
            "protocol: an option value that is itself a flag is refused")
    h.check(OwnProcessDiagnostics.goAccepted("GO"), "protocol: exactly GO is accepted")
    let refused: [String?] = [nil, "", "go", "GO ", " GO", "GOO", "READY"]
    h.check(refused.allSatisfy { !OwnProcessDiagnostics.goAccepted($0) },
            "protocol: EOF, empty, lowercase, padded or other lines are refused before any model or device work")
    h.check(OwnProcessDiagnostics.reportPathRejection(path: "/tmp/x/out.json", exists: false,
                                                      parentIsDirectory: true) == nil,
            "protocol: a new .json path in an existing directory is accepted")
    h.check(OwnProcessDiagnostics.reportPathRejection(path: "/tmp/x/out.json", exists: true,
                                                      parentIsDirectory: true) != nil
                && OwnProcessDiagnostics.reportPathRejection(path: "/tmp/x/out.json", exists: false,
                                                             parentIsDirectory: false) != nil
                && OwnProcessDiagnostics.reportPathRejection(path: "/tmp/x/out.txt", exists: false,
                                                             parentIsDirectory: true) != nil,
            "protocol: existing files, missing directories and non-.json paths are refused")
    h.check(OwnProcessDiagnostics.readyToken == RouterOverlapModelGate.readyToken
                && OwnProcessDiagnostics.goToken == RouterOverlapModelGate.goToken
                && OwnProcessDiagnostics.endToken == RouterOverlapModelGate.endToken
                && OwnProcessDiagnostics.readyToken == "BENCHMARK_READY"
                && OwnProcessDiagnostics.goToken == "GO"
                && OwnProcessDiagnostics.endToken == "TIMED_WINDOW_END",
            "protocol: tokens are the existing supervisor protocol BENCHMARK_READY, GO and TIMED_WINDOW_END shared with the phase-53 gate")
}

// MARK: - Memory derivation

func testMemoryDerivation(_ h: inout Harness) {
    let widths = Array(repeating: 2048, count: 25) + Array(repeating: 1024, count: 5)
    h.check(OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: 4096) == 1_845_493_760,
            "memory: 25 local and 5 global layers at 4096 positions derive 1760 MiB of K/V")
    h.check(OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: 1) == 450_560
                && OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: 53) == 23_879_680,
            "memory: per-position and 53-position K/V bytes")
    h.check(OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: [Int.max / 2], positions: 4096) == nil,
            "memory: an overflowing derivation is nil, never wrapped")
    h.check(OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: [2048, 0], positions: 1) == nil
                && OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: widths, positions: -1) == nil
                && OwnProcessMemoryAccounting.derivedKVBytes(layerWidths: [], positions: 1) == nil,
            "memory: invalid widths or positions are refused")
    h.check(OwnProcessMemoryAccounting.standardCaveats.contains { $0.contains("not physical residency") },
            "memory: allocation capacity is declared distinct from physical residency")
}

// MARK: - Status and report

func testStatus(_ h: inout Harness) {
    typealias D = OwnProcessDiagnostics
    let six = Array(repeating: true, count: 6)
    h.check(D.status(restoredPristine: true, aborted: false, gateStatus: .qualificationPassed,
                     runsPassed: six, expectedRuns: 6, profileFailures: []) == .profiledChecksPassed,
            "status: six passing profiled runs after a passing gate give profiledChecksPassed")
    h.check(D.status(restoredPristine: false, aborted: true, gateStatus: .qualificationPassed,
                     runsPassed: six, expectedRuns: 6, profileFailures: []) == .runnerNotRestored,
            "status: an unrestored runner takes precedence over every other outcome")
    h.check(D.status(restoredPristine: true, aborted: true, gateStatus: .qualificationPassed,
                     runsPassed: six, expectedRuns: 6, profileFailures: []) == .aborted,
            "status: an abort takes precedence over profile outcomes")
    let others: [RouterOverlapGateStatus?] = [nil, .qualificationFailed, .performanceNoGain,
                                              .performanceInternalCriteriaMetExternalEligibilityPending]
    h.check(others.allSatisfy {
                D.status(restoredPristine: true, aborted: false, gateStatus: $0, runsPassed: six,
                         expectedRuns: 6, profileFailures: []) == .qualificationFailed },
            "status: only a qualification-only pass of the phase-53 gate allows profiling")
    h.check(D.status(restoredPristine: true, aborted: false, gateStatus: .qualificationPassed,
                     runsPassed: Array(six.prefix(5)), expectedRuns: 6, profileFailures: []) == .profiledChecksFailed
                && D.status(restoredPristine: true, aborted: false, gateStatus: .qualificationPassed,
                            runsPassed: [true, true, false, true, true, true], expectedRuns: 6,
                            profileFailures: []) == .profiledChecksFailed
                && D.status(restoredPristine: true, aborted: false, gateStatus: .qualificationPassed,
                            runsPassed: six, expectedRuns: 6, profileFailures: ["x"]) == .profiledChecksFailed,
            "status: a missing run, a failed run or a profile failure gives profiledChecksFailed")
}

func encodedObject(_ r: OwnProcessDiagnosticsReport) -> [String: Any]? {
    guard let data = try? OwnProcessDiagnostics.encodeReport(r) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

func testReport(_ h: inout Harness) {
    let fresh = OwnProcessDiagnosticsReport()
    let obj = encodedObject(fresh)
    h.check(!fresh.gainQualified && !fresh.installable && !fresh.productionTPOTClaim
                && fresh.status == .aborted
                && (obj?["gainQualified"] as? Bool) == false && (obj?["installable"] as? Bool) == false
                && (obj?["productionTPOTClaim"] as? Bool) == false,
            "report: a fresh report encodes gainQualified, installable and productionTPOTClaim false")
    var reEmitted: [String: Any]? = nil
    if var tampered = obj {
        tampered["gainQualified"] = true
        tampered["installable"] = true
        tampered["productionTPOTClaim"] = true
        if let data = try? JSONSerialization.data(withJSONObject: tampered),
           let decoded = try? JSONDecoder().decode(OwnProcessDiagnosticsReport.self, from: data) {
            reEmitted = encodedObject(decoded)
        }
    }
    h.check(reEmitted != nil && (reEmitted?["gainQualified"] as? Bool) == false
                && (reEmitted?["installable"] as? Bool) == false
                && (reEmitted?["productionTPOTClaim"] as? Bool) == false,
            "report: a decoded report claiming gain, installability or production TPOT is re-emitted false")
    let pre = OwnProcessTaskSnapshot(
        label: "before-runner-init", uptimeNanos: 1,
        events: OwnProcessTaskEvents(status: "ok", faults: 1, pageins: 0, cowFaults: 0, contextSwitches: 1),
        vm: OwnProcessVMInfo(status: "ok", virtualBytes: 1, residentBytes: 1, physFootprintBytes: 1),
        metalStatus: "UNKNOWN(test)", metalCurrentAllocatedBytes: nil)
    let aborted = OwnProcessDiagnostics.aborted(reason: "runner load failed", preInitSnapshot: pre)
    h.check(aborted.status == .aborted && aborted.abortReason == "runner load failed"
                && aborted.snapshots == [pre] && !aborted.gainQualified,
            "report: aborted() keeps the pre-init snapshot and the always-false contract")
    h.check((obj?["breakdownNesting"] as? [String]) == Gemma4SubmissionBreakdown.nesting
                && (obj?["correctnessContract"] as? Double) == OwnProcessDiagnostics.correctnessContract,
            "report: the nesting declaration and the 1e-4 contract are part of every report")
}

// MARK: - Phase-53 scope still refuses instrumentation

func testOverlapScopeRefusesInstrumentation(_ h: inout Harness) {
    func snapshot(active: Bool, breakdown: Bool) -> RouterOverlapScopeSnapshot {
        RouterOverlapScopeSnapshot(
            scopeActive: active, eventAndScratchReady: true, openEncoderPresent: false,
            pendingTickets: 0, expertCacheEnabled: false, residentWeightsMapped: true,
            moeBatch: 4, moeRowsPerGroup: 32, installIdentityVerified: true, q4LaunchWidth: 32,
            rmsGeneratedInstalled: false, persistentAxisFlagsClean: true, layerProbeInstalled: false,
            moeInputProbeInstalled: false, traceLayers: false, breakdownInstrumentation: breakdown,
            commandTimelineEnabled: false, attentionDiagnosticActive: false)
    }
    h.check(RouterOverlapScope.rejection(snapshot(active: false, breakdown: false), phase: .entry) == nil
                && RouterOverlapScope.rejection(snapshot(active: true, breakdown: false), phase: .token) == nil,
            "overlap scope: the clean control snapshot qualifies at entry and per token")
    h.check(RouterOverlapScope.rejection(snapshot(active: false, breakdown: true), phase: .entry)
                == "breakdown instrumentation must be off"
                && RouterOverlapScope.rejection(snapshot(active: true, breakdown: true), phase: .token)
                == "breakdown instrumentation must be off",
            "overlap scope: breakdown instrumentation is still refused at entry and per token")
}

// MARK: - Main

func runAll() -> Int32 {
    var h = Harness()
    testTaskInfo(&h)
    testDeltas(&h)
    testBreakdown(&h)
    testComparison(&h)
    testSites(&h)
    testProtocolInputs(&h)
    testMemoryDerivation(&h)
    testStatus(&h)
    testReport(&h)
    testOverlapScopeRefusesInstrumentation(&h)
    print("owned-process-diagnostics-pure-tests: " + String(h.passed) + " passed, "
          + String(h.failed.count) + " failed")
    for name in h.failed { print("FAILED: " + name) }
    return h.failed.isEmpty ? 0 : 1
}

exit(runAll())
