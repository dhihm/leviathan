import Foundation
import LeviathanCore

// Phase 59 pure tests: kernel family and counter tables, counter cross-check, raw stage-sample checks, aggregation,
// options, protocol constants, report contract and status. No Metal device, no GGUF, no model weights.

typealias KS = KernelStageAttribution

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

func checkTables(_ h: inout Harness) {
    let t = KS.familyTable(moeGateUp: "test_gu", moeAct: "test_act", moeDown: "test_dn")
    h.check(t["g4_q4_matvec"] == "q4Matvec" && t["g4_q6_embed_row"] == "embedding" && t["g4_softcap"] == "softcap"
            && t["g4_cache_append"] == "kvAppend" && t["g4_axpy"] == "axpy",
            "incumbent kernel names map to their families")
    h.check(t["test_gu"] == "moeGateUp" && t["test_act"] == "moeAct" && t["test_dn"] == "moeDown",
            "installed MoE entry names map to the MoE families")
    h.check(t["g4_q4_matvec_tiled"] == nil && t["g4_q4_matvec_vl"] == nil && t["unknown"] == nil,
            "kernels outside the OFF path are not tagged")
    h.check(t.values.allSatisfy { KS.familyNames.contains($0) }, "every family is declared")
    h.check(Set(t.values) == Set(KS.familyNames), "every declared family is reachable")
    let k = KS.counterKeys(moeGateUp: "test_gu", moeAct: "test_act", moeDown: "test_dn")
    h.check(k["g4_q4_matvec"] == "q4Generic" && k["test_gu"] == "moeGu" && k["test_act"] == "moeAct"
            && k["test_dn"] == "moeDn" && k["g4_f32_matvec"] == "routerLaunchApplied",
            "kernel names bind to the runner's own per-dispatch counters")
    h.check(k.keys.allSatisfy { t[$0] != nil }, "every counted kernel is tagged")
    h.check(Set(k.values).count == k.count, "no two kernels share one counter")
}

func checkCrossCheck(_ h: inout Harness) {
    let keys = KS.counterKeys(moeGateUp: "test_gu", moeAct: "test_act", moeDown: "test_dn")
    let delta = RouterOverlapGateCounters(values: ["kernels": 5, "q4Generic": 3, "moeGu": 2])
    let counts = ["g4_q4_matvec": 3, "test_gu": 2]
    h.check(KS.counterCrossCheck(kernelCounts: counts, encoders: 5, counterDelta: delta, counterKeys: keys).isEmpty,
            "exact kernel counts agree with the runner counters")
    h.check(KS.counterCrossCheck(kernelCounts: ["g4_q4_matvec": 2, "test_gu": 2, "g4_axpy": 1], encoders: 5,
                                 counterDelta: delta, counterKeys: keys).count == 1,
            "a mislabeled Q4 matvec is detected")
    h.check(!KS.counterCrossCheck(kernelCounts: counts, encoders: 4, counterDelta: delta, counterKeys: keys).isEmpty,
            "encoders must equal the counted kernels")
}

func checkAnalyze(_ h: inout Harness) {
    let ok = KS.analyze(raw: [10, 20, 25, 30], encoders: 2, previousLastTick: nil)
    h.check(ok.failures.isEmpty && ok.intervalTicks == [10, 5] && ok.sumEncoderTicks == 15 && ok.disjoint
            && ok.firstStartTicks == 10 && ok.lastEndTicks == 30 && ok.spanTicks == 20,
            "valid samples give raw intervals and sums")
    let overlap = KS.analyze(raw: [10, 20, 15, 30], encoders: 2, previousLastTick: nil)
    h.check(overlap.failures.isEmpty && overlap.overlappingAdjacentEncoders == 1 && !overlap.disjoint
            && overlap.intervalTicks == [10, 15], "adjacent overlap is counted, not hidden")
    let errorValue = KS.analyze(raw: [10, KS.counterErrorValue, 25, 30], encoders: 2, previousLastTick: nil)
    h.check(!errorValue.failures.isEmpty && errorValue.intervalTicks.isEmpty && errorValue.sumEncoderTicks == nil,
            "the counter error value is refused and no interval is fabricated")
    h.check(!KS.analyze(raw: [0, 20, 25, 30], encoders: 2, previousLastTick: nil).failures.isEmpty,
            "a zero sample is refused")
    h.check(!KS.analyze(raw: [20, 10, 25, 30], encoders: 2, previousLastTick: nil).failures.isEmpty,
            "end before start is refused")
    h.check(!KS.analyze(raw: [10, 20, 25, 30], encoders: 2, previousLastTick: 25).failures.isEmpty,
            "samples not later than the previous token are refused as stale")
    h.check(KS.analyze(raw: [30, 40, 45, 50], encoders: 2, previousLastTick: 25).failures.isEmpty,
            "samples after the previous token are accepted")
    h.check(!KS.analyze(raw: [10, 20, 25], encoders: 2, previousLastTick: nil).failures.isEmpty,
            "a sample count mismatch is refused")
    h.check(!KS.analyze(raw: [], encoders: 0, previousLastTick: nil).failures.isEmpty, "zero encoders refused")
}

func checkAggregate(_ h: inout Harness) {
    let tokens = [
        KernelStageTokenInput(labels: ["a", "b", "a"], intervalTicks: [1, 2, 3], disjoint: true),
        KernelStageTokenInput(labels: ["a", "b", "a"], intervalTicks: [2, 2, 2], disjoint: true),
        KernelStageTokenInput(labels: ["a"], intervalTicks: [1, 2], disjoint: true),
    ]
    let stats = KS.aggregate(tokens, label: { $0 }, family: { $0 == "a" ? "fa" : nil })
    h.check(stats.map { $0.label } == ["a", "b"], "labels sorted; malformed token ignored")
    if stats.count == 2 {
        let a = stats[0]
        let b = stats[1]
        h.check(a.encodersPerToken == 2 && a.ticksPerToken == [4, 4] && a.medianTicksPerToken == 4 && a.tokens == 2
                && a.family == "fa", "per-token tick sums and counts")
        h.check(abs((a.medianShareOfSampledEncoderTicks ?? 0) - 4.0 / 6.0) < 1e-12
                && abs((b.medianShareOfSampledEncoderTicks ?? 0) - 2.0 / 6.0) < 1e-12,
                "shares of disjoint sampled stage ticks")
    }
    let mixed = KS.aggregate([tokens[0], KernelStageTokenInput(labels: ["a", "b", "a"], intervalTicks: [2, 2, 2],
                                                               disjoint: false)],
                             label: { $0 }, family: { _ in nil })
    h.check(mixed.allSatisfy { $0.medianShareOfSampledEncoderTicks == nil }, "no share when encoders overlapped")
    let families = KS.aggregate(tokens, label: { $0 == "a" ? "fa" : "fb" }, family: { _ in nil })
    h.check(families.map { $0.ticksPerToken } == [[4, 4], [2, 2]], "family aggregation sums its kernels")
}

func checkOptions(_ h: inout Harness) {
    let base = ["--gguf", "m", "--region", "r", "--graph", "g", "--fixtures", "f", "--report", "o.json"]
    let plain = try? KS.parseOptions(base)
    h.check(plain?.ownProcessSnapshots == false && plain?.report == "o.json", "options parse")
    h.check((try? KS.parseOptions(base + ["--own-process-snapshots"]))?.ownProcessSnapshots == true,
            "opt-in own-process snapshots")
    h.check((try? KS.parseOptions(base + ["--own-process-snapshots", "--own-process-snapshots"])) == nil,
            "duplicate flag refused")
    h.check((try? KS.parseOptions(Array(base.dropLast(2)))) == nil, "missing option refused")
    h.check((try? KS.parseOptions(base + ["--gguf", "x"])) == nil, "duplicate option refused")
    h.check((try? KS.parseOptions(base + ["--other", "x"])) == nil, "unknown option refused")
}

func checkConstants(_ h: inout Harness) {
    h.check(KS.pinnedEncodersPerContinuation == 1349 && KS.pinnedCommandBuffersPerContinuation == 37,
            "pinned OFF continuation kernels and command buffers")
    h.check(KS.sampleCapacity == 2698 && KS.sampleCapacity <= KS.maxSamples && KS.maxSamples == 4096,
            "derived capacity 2698 within the 4096 bound")
    h.check(KS.counterErrorValue == UInt64.max, "counter error value")
    h.check(KS.readyToken == RouterOverlapModelGate.readyToken && KS.goToken == RouterOverlapModelGate.goToken
            && KS.endToken == RouterOverlapModelGate.endToken, "protocol tokens are the gate constants")
    h.check(KS.goAccepted("GO") && !KS.goAccepted(nil) && !KS.goAccepted("GO ") && !KS.goAccepted("go"),
            "only an exact GO line is accepted; EOF refused")
    h.check(KS.refusalTestNames.count == 7 && Set(KS.refusalTestNames).count == 7, "seven distinct refusal tests")
    h.check(KS.maxReportBytes == 16 << 20, "report bound 16 MiB")
}

func encoded(_ r: KernelStageReport) -> [String: Any] {
    guard let d = try? KS.encodeReport(r),
          let o = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { return [:] }
    return o
}

func checkContract(_ h: inout Harness) {
    let names = ["gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    let base = encoded(KernelStageReport())
    h.check(names.allSatisfy({ (base[$0] as? Bool) == false }) && (base["kind"] as? String) == KS.kind
            && (base["sampleCapacity"] as? Int) == 2698, "the default report encodes every claim false")
    var tampered = base
    for n in names { tampered[n] = true }
    tampered["status"] = "profiledChecksPassed"
    if let r = decode(KernelStageReport.self, tampered) {
        let e = encoded(r)
        h.check(names.allSatisfy({ (e[$0] as? Bool) == false }), "tampered claims re-emit false")
    } else {
        h.check(false, "tampered report decodes")
    }
}

func checkStatus(_ h: inout Harness) {
    var r = KernelStageReport()
    h.check(KS.finalStatus(r) == .runnerNotRestored, "an unrestored runner")
    r.restoredPristine = true
    r.abortReason = "x"
    h.check(KS.finalStatus(r) == .aborted, "abort")
    r.abortReason = nil
    r.support = KernelStageSupport(device: "d", supportsAtStageBoundary: true, supportsAtDispatchBoundary: false,
                                   timestampCounterSetFound: true, sampleBufferCreated: false, sampleCapacity: 2698,
                                   error: "creation failed")
    h.check(KS.finalStatus(r) == .unsupported, "an uncreated sample buffer is unsupported, never faked")
    r.support = KernelStageSupport(device: "d", supportsAtStageBoundary: true, supportsAtDispatchBoundary: false,
                                   timestampCounterSetFound: true, sampleBufferCreated: true, sampleCapacity: 2698,
                                   error: nil)
    h.check(KS.finalStatus(r) == .profiledChecksFailed, "support alone without identity and fixtures fails")
}

func runAll() -> Int32 {
    var h = Harness()
    checkTables(&h)
    checkCrossCheck(&h)
    checkAnalyze(&h)
    checkAggregate(&h)
    checkOptions(&h)
    checkConstants(&h)
    checkContract(&h)
    checkStatus(&h)
    print("kernel-stage-attribution-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
