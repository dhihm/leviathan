// router-overlap-model-gate-pure-tests - pure policy tests for the PUBLIC
// RouterOverlapModelGate API: fixture parsing, the timing statistics,
// the fixed AB/BA protocol, fixtureOutcome / performanceStatus precedence,
// the named counter set, derived per-forward expectations, the declared
// candidate and report encoding. Every assertion calls the real shared
// gate (or runtime) code; nothing re-implements a policy. Synthetic
// performance runs and derived counts are built through their public
// Codable conformance because their memberwise initializers are internal.
// No Metal device, GPU, GGUF, model weights or hashing.
//
// Exit 0 when every check passed, 1 otherwise. The count printed is the
// actual number of checks executed.

import Foundation
import LeviathanCore

typealias Gate = RouterOverlapModelGate

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

// MARK: - JSON construction helpers (test inputs only)

enum JSONText {
    static let quote = String(Character(Unicode.Scalar(UInt8(34))))

    static func string(_ s: String) -> String { quote + s + quote }

    static func ints(_ v: [Int]) -> String {
        "[" + v.map { String($0) }.joined(separator: ",") + "]"
    }

    /// Fixture JSON with keys in the given list order; bodyOverride replaces
    /// one entry's whole object body.
    static func fixtures(_ list: [RouterOverlapGateFixture], extraEntryKey: Bool = false,
                         bodyOverride: [String: String] = [:]) -> Data {
        let entries = list.map { f -> String in
            var body = string("prompt") + ":" + ints(f.prompt) + ","
                + string("frozenIds") + ":" + ints(f.frozenIds)
            if extraEntryKey { body += "," + string("note") + ":" + string("ignored") }
            if let o = bodyOverride[f.name] { body = o }
            return string(f.name) + ":{" + body + "}"
        }
        return Data(("{" + entries.joined(separator: ",") + "}").utf8)
    }
}

func rejects(_ data: Data) -> Bool {
    do {
        _ = try RouterOverlapGateFixtures.parse(data)
        return false
    } catch {
        return error is RouterOverlapGateError
    }
}

func parsed(_ data: Data) -> [RouterOverlapGateFixture]? {
    try? RouterOverlapGateFixtures.parse(data)
}

func withFixture(_ list: [RouterOverlapGateFixture], _ index: Int, prompt: [Int]? = nil,
                 frozen: [Int]? = nil, name: String? = nil) -> [RouterOverlapGateFixture] {
    var copy = list
    let f = copy[index]
    copy[index] = RouterOverlapGateFixture(name: name ?? f.name, prompt: prompt ?? f.prompt,
                                           frozenIds: frozen ?? f.frozenIds)
    return copy
}

enum Synthetic {
    static func zeroCounters() -> [String: Any] {
        var v = [String: Int]()
        for k in RouterOverlapGateCounters.keys { v[k] = 0 }
        return ["values": v]
    }

    static func counts(mismatches: [String]) -> [String: Any] {
        ["prefillForwards": 21, "continuationForwards": 31,
         "prefillActual": zeroCounters(), "continuationActual": zeroCounters(),
         "prefillExpected": zeroCounters(), "continuationExpected": zeroCounters(),
         "mismatches": mismatches]
    }

    static func totals() -> [String: Any] {
        ["prefillKernels": 0, "prefillBuffers": 0, "prefillGpuElapsedMicros": 0.0,
         "continuationKernels": [Int](), "continuationBuffers": [Int](),
         "continuationGpuElapsedMicros": [Double]()]
    }

    static func frozen(_ fixture: String) -> [Int] {
        RouterOverlapGateFixtures.pinned.first { $0.name == fixture }?.frozenIds ?? []
    }

    /// A passing run record; tpotMedianWallMicros comes from the real
    /// Gate.runTPOT over 31 equal continuation walls.
    static func run(fixture: String, arm: RouterOverlapGateArm, warmup: Bool, block: Int?,
                    order: Int?, tpot: Double) -> [String: Any] {
        let walls = Array(repeating: tpot, count: 31)
        var o: [String: Any] = [
            "fixture": fixture, "arm": arm.rawValue, "warmup": warmup,
            "prefillWallMicros": 1000.0, "continuationWallMicros": walls,
            "totals": totals(), "ids": frozen(fixture), "idsMatchFrozen": true, "outputs": 32,
            "allFinite": true, "counts": counts(mismatches: []),
            "totalsMismatches": [String](), "preconditionFailures": [String](),
            "checksPassed": true,
        ]
        if let block { o["block"] = block }
        if let order { o["orderInBlock"] = order }
        if let t = Gate.runTPOT(walls) { o["tpotMedianWallMicros"] = t }
        return o
    }

    /// Warmups (default OFF then ON) followed by the fixed blocks in
    /// Gate.blockOrder; values index the per-block TPOT arrays modulo count.
    static func protocolRuns(fixture: String, baseline: [Double], candidate: [Double],
                             blocks: Int = 7,
                             warmupArms: [RouterOverlapGateArm] = [.off, .on],
                             warmupTPOT: Double = 100) -> [[String: Any]] {
        var out = [[String: Any]]()
        for arm in warmupArms {
            out.append(run(fixture: fixture, arm: arm, warmup: true, block: nil, order: nil,
                           tpot: warmupTPOT))
        }
        for b in 0..<blocks {
            for (o, arm) in Gate.blockOrder(b).enumerated() {
                let t = arm == .off ? baseline[b % baseline.count] : candidate[b % candidate.count]
                out.append(run(fixture: fixture, arm: arm, warmup: false, block: b, order: o, tpot: t))
            }
        }
        return out
    }

    static func decode(_ objects: [[String: Any]]) throws -> [RouterOverlapGatePerfRun] {
        let data = try JSONSerialization.data(withJSONObject: objects)
        return try JSONDecoder().decode([RouterOverlapGatePerfRun].self, from: data)
    }

    static func derived(_ overrides: [String: Int] = [:]) throws -> RouterOverlapGateDerived {
        var o: [String: Int] = [
            "layers": 30, "topK": 8, "moeChunks": 2, "q4Matvecs": 205, "q6Matvecs": 1,
            "attentionOps": 30, "rmsnormOps": 331, "statisticsGroups": 30,
            "statisticsReuses": 60, "routerScaleOps": 30, "expertChains": 30,
            "overlapWindows": 30, "routerProjections": 30,
        ]
        for (k, v) in overrides { o[k] = v }
        let data = try JSONSerialization.data(withJSONObject: o)
        return try JSONDecoder().decode(RouterOverlapGateDerived.self, from: data)
    }
}

func mutate(_ objects: [[String: Any]], where match: ([String: Any]) -> Bool,
            _ change: (inout [String: Any]) -> Void) -> [[String: Any]] {
    objects.map { o in
        guard match(o) else { return o }
        var c = o
        change(&c)
        return c
    }
}

func isTimed(_ arm: RouterOverlapGateArm, block: Int) -> ([String: Any]) -> Bool {
    { o in
        (o["warmup"] as? Bool) == false && (o["arm"] as? String) == arm.rawValue
            && (o["block"] as? Int) == block
    }
}

func isWarmup(_ arm: RouterOverlapGateArm) -> ([String: Any]) -> Bool {
    { o in (o["warmup"] as? Bool) == true && (o["arm"] as? String) == arm.rawValue }
}

func outcome(_ fixture: String, _ objects: [[String: Any]]) -> RouterOverlapGateFixtureOutcome? {
    guard let runs = try? Synthetic.decode(objects) else { return nil }
    return Gate.fixtureOutcome(fixture: fixture, runs: runs)
}

let flat7: [Double] = Array(repeating: 100, count: 7)

func metRuns(_ fixture: String) -> [[String: Any]] {
    Synthetic.protocolRuns(fixture: fixture, baseline: [100, 100, 100, 100, 100, 100, 100],
                           candidate: [97, 97, 97, 97, 97, 97, 97])
}

// MARK: - Fixture parsing

func testFixtureParse(_ h: inout Harness) {
    let pinned = RouterOverlapGateFixtures.pinned
    let order = RouterOverlapGateFixtures.order
    let proto = RouterOverlapGateProtocol.required
    h.check(pinned.map { $0.name } == order && order == proto.fixtures
                && order == ["en-chat", "kr-chat", "raw-hello"],
            "fixtures: pinned names and protocol fixtures use the fixed order en-chat, kr-chat, raw-hello")
    h.check(pinned.allSatisfy { $0.frozenIds.count == proto.outputsPerRun && !$0.prompt.isEmpty },
            "fixtures: every pinned fixture has a non-empty prompt and exactly 32 frozen IDs")
    let canonical = JSONText.fixtures(pinned)
    h.check(parsed(canonical) == pinned, "fixtures: canonical JSON parses to exactly the pinned fixtures")
    h.check(parsed(JSONText.fixtures(Array(pinned.reversed()))) == pinned,
            "fixtures: JSON with reversed key order is returned in the fixed order")

    for (i, f) in pinned.enumerated() {
        var idRejected = 0
        for p in 0..<f.frozenIds.count {
            var ids = f.frozenIds
            ids[p] += 1
            if rejects(JSONText.fixtures(withFixture(pinned, i, frozen: ids))) { idRejected += 1 }
        }
        h.check(f.frozenIds.count == 32 && idRejected == 32,
                "fixtures: " + f.name + " rejects a single tampered frozen ID at each of the 32 positions")
        var promptRejected = 0
        for p in 0..<f.prompt.count {
            var pr = f.prompt
            pr[p] += 1
            if rejects(JSONText.fixtures(withFixture(pinned, i, prompt: pr))) { promptRejected += 1 }
        }
        h.check(promptRejected == f.prompt.count,
                "fixtures: " + f.name + " rejects a single tampered prompt token at each of its "
                    + String(f.prompt.count) + " positions")
        var swapped = f.frozenIds
        swapped.swapAt(0, 1)
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, frozen: swapped))),
                "fixtures: " + f.name + " rejects two swapped frozen IDs")
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, frozen: Array(f.frozenIds.prefix(31))))),
                "fixtures: " + f.name + " rejects 31 frozen IDs")
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, frozen: f.frozenIds + [f.frozenIds[0]]))),
                "fixtures: " + f.name + " rejects 33 frozen IDs")
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, prompt: []))),
                "fixtures: " + f.name + " rejects an empty prompt")
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i,
                                                      prompt: f.prompt + [f.prompt[f.prompt.count - 1]]))),
                "fixtures: " + f.name + " rejects an extra prompt token")
        var low = f.frozenIds
        low[0] = -1
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, frozen: low))),
                "fixtures: " + f.name + " rejects a negative (out-of-range) frozen ID")
        var high = f.frozenIds
        high[31] = 262_144
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, frozen: high))),
                "fixtures: " + f.name + " rejects a frozen ID equal to the vocabulary size")
        var lowPrompt = f.prompt
        lowPrompt[0] = -1
        h.check(rejects(JSONText.fixtures(withFixture(pinned, i, prompt: lowPrompt))),
                "fixtures: " + f.name + " rejects a negative (out-of-range) prompt token")
    }

    h.check(rejects(JSONText.fixtures(Array(pinned.prefix(2)))),
            "fixtures: a missing fixture (raw-hello) is rejected")
    h.check(rejects(JSONText.fixtures(pinned + [RouterOverlapGateFixture(
                name: "extra", prompt: [2], frozenIds: pinned[2].frozenIds)])),
            "fixtures: an extra fourth fixture is rejected")
    h.check(rejects(JSONText.fixtures(withFixture(pinned, 0, name: "en_chat"))),
            "fixtures: a renamed fixture key is rejected")
    h.check(rejects(JSONText.fixtures(withFixture(pinned, 0, name: "En-chat"))),
            "fixtures: a case-changed fixture key is rejected")
    h.check(rejects(JSONText.fixtures(withFixture(pinned, 0, prompt: pinned[1].prompt,
                                                  frozen: pinned[1].frozenIds))),
            "fixtures: kr-chat contents under the en-chat name are rejected")

    let q = JSONText.self
    h.check(rejects(q.fixtures(pinned, bodyOverride: ["kr-chat": q.string("prompt") + ":" + q.ints(pinned[1].prompt)])),
            "fixtures: an entry without frozenIds is rejected")
    h.check(rejects(q.fixtures(pinned, bodyOverride: ["kr-chat": q.string("frozenIds") + ":" + q.ints(pinned[1].frozenIds)])),
            "fixtures: an entry without prompt is rejected")
    h.check(rejects(q.fixtures(pinned, bodyOverride: ["raw-hello": q.string("prompt") + ":" + q.string("2 9259")
                                                        + "," + q.string("frozenIds") + ":" + q.ints(pinned[2].frozenIds)])),
            "fixtures: a string-typed prompt is rejected")
    h.check(rejects(q.fixtures(pinned, bodyOverride: ["raw-hello": q.string("prompt") + ":null,"
                                                        + q.string("frozenIds") + ":" + q.ints(pinned[2].frozenIds)])),
            "fixtures: a null prompt is rejected")
    h.check(rejects(q.fixtures(pinned, bodyOverride: ["raw-hello": q.string("prompt") + ":[2.5,9259],"
                                                        + q.string("frozenIds") + ":" + q.ints(pinned[2].frozenIds)])),
            "fixtures: a fractional prompt token is rejected")
    h.check(rejects(Data("[]".utf8)), "fixtures: a top-level array is rejected")
    h.check(rejects(Data("{}".utf8)), "fixtures: an empty object is rejected")
    h.check(rejects(Data()), "fixtures: empty input is rejected")
    h.check(rejects(Data(canonical.dropLast(1))), "fixtures: truncated JSON is rejected")

    var atLimit = canonical
    h.check(atLimit.count < 1 << 16, "fixtures: canonical JSON is below the 64 KiB bound")
    atLimit.append(Data(repeating: 32, count: (1 << 16) - atLimit.count))
    h.check(atLimit.count == 1 << 16 && parsed(atLimit) == pinned,
            "fixtures: canonical JSON padded to exactly 65536 bytes is accepted")
    var overLimit = atLimit
    overLimit.append(32)
    h.check(rejects(overLimit), "fixtures: otherwise canonical JSON of 65537 bytes is rejected")

    h.check(parsed(JSONText.fixtures(pinned, extraEntryKey: true)) == pinned,
            "fixtures (current behavior): an unknown key inside an entry is ignored and the pinned fixtures are returned")
}

// MARK: - Statistics

func testStatistics(_ h: inout Harness) {
    h.check(Gate.median([]) == nil, "median: empty is nil")
    h.check(Gate.median([5]) == 5, "median: single value")
    h.check(Gate.median([3, 1, 2]) == 2, "median: odd count sorts before selecting")
    h.check(Gate.median([4, 1, 3, 2]) == 2.5, "median: even count averages the middle pair")
    h.check(Gate.median([1, 2, .infinity]) == 2,
            "median (current behavior): unvalidated; infinity only shifts order")

    let ramp = (1...31).map { Double($0) }
    h.check(Gate.runTPOT(ramp) == 16, "runTPOT: 31 walls 1...31 give the median 16")
    h.check(Gate.runTPOT(Array(ramp.reversed())) == 16, "runTPOT: order of walls does not matter")
    h.check(Gate.runTPOT(Array(ramp.prefix(30))) == nil, "runTPOT: 30 walls rejected")
    h.check(Gate.runTPOT(ramp + [32]) == nil, "runTPOT: 32 walls rejected")
    h.check(Gate.runTPOT([]) == nil, "runTPOT: no walls rejected")
    for (label, bad) in [("NaN", Double.nan), ("+infinity", Double.infinity),
                         ("-infinity", -Double.infinity), ("zero", 0.0), ("negative", -1.0)] {
        var first = ramp
        first[0] = bad
        var last = ramp
        last[30] = bad
        h.check(Gate.runTPOT(first) == nil && Gate.runTPOT(last) == nil,
                "runTPOT: a " + label + " wall at either end is rejected")
    }
    let outlier = Array(repeating: 100.0, count: 30) + [1e12]
    h.check(Gate.runTPOT(outlier) == 100,
            "runTPOT: a huge outlier is kept (31 walls used) and only the median is reported")

    h.check(Gate.spread(flat7) == 0, "spread: seven equal TPOTs give zero spread")
    h.check(Gate.spread([100, 100, 100, 100, 100, 100, 105]) == 0.05,
            "spread: (105 - 100) / 100 is exactly 0.05")
    h.check(Gate.spread([90, 95, 100, 100, 100, 105, 110]) == 0.2, "spread: (max - min) / median")
    h.check(Gate.spread(Array(flat7.prefix(6))) == nil, "spread: 6 run TPOTs rejected")
    h.check(Gate.spread(flat7 + [100]) == nil, "spread: 8 run TPOTs rejected")
    for (label, bad) in [("NaN", Double.nan), ("+infinity", Double.infinity),
                         ("zero", 0.0), ("negative", -100.0)] {
        var v = flat7
        v[3] = bad
        h.check(Gate.spread(v) == nil, "spread: a " + label + " run TPOT is rejected")
    }

    h.check(Gate.gain(baseline: 100, candidate: 98) == 0.02, "gain: (100 - 98) / 100 is exactly 0.02")
    h.check(Gate.gain(baseline: 100, candidate: 102) == -0.02, "gain: a slower candidate is negative")
    h.check(Gate.gain(baseline: 100, candidate: 100) == 0, "gain: equal medians give zero")
    h.check(Gate.gain(baseline: nil, candidate: 98) == nil && Gate.gain(baseline: 100, candidate: nil) == nil,
            "gain: a missing median gives nil")
    for (label, bad) in [("NaN", Double.nan), ("+infinity", Double.infinity),
                         ("zero", 0.0), ("negative", -1.0)] {
        h.check(Gate.gain(baseline: bad, candidate: 98) == nil && Gate.gain(baseline: 100, candidate: bad) == nil,
                "gain: a " + label + " median on either side gives nil")
    }
}

// MARK: - Fixed protocol

func testProtocol(_ h: inout Harness) {
    let proto = RouterOverlapGateProtocol.required
    var ordersOk = true
    for b in 0..<7 {
        let want: [RouterOverlapGateArm] = b % 2 == 0 ? [.off, .on] : [.on, .off]
        if Gate.blockOrder(b) != want { ordersOk = false }
    }
    h.check(ordersOk, "protocol: blocks 0, 2, 4, 6 are AB (OFF, ON) and blocks 1, 3, 5 are BA")
    h.check(proto.blockOrders == (0..<7).map(Gate.blockOrder) && proto.blocksPerFixture == 7
                && proto.warmupsPerArmPerFixture == 1,
            "protocol: seven declared block orders and one warmup per arm per fixture")
    h.check(proto.spreadLimit == 0.05 && proto.gainThreshold == 0.02
                && proto.outputsPerRun == 32 && proto.continuationsPerRun == 31,
            "protocol: 5% spread, 2% gain, 32 outputs and 31 continuations are declared")
}

// MARK: - fixtureOutcome

func testFixtureOutcome(_ h: inout Harness) {
    let fx = "en-chat"
    let met = outcome(fx, metRuns(fx))
    h.check(met?.outcome == Gate.outcomeInternalCriteriaMet && met?.allRunChecksPassed == true
                && met?.baselineMedianTPOTMicros == 100 && met?.candidateMedianTPOTMicros == 97
                && met?.baselineStable == true && met?.candidateStable == true
                && met?.gainMeetsThreshold == true,
            "outcome: 3% gain with zero spread meets the internal criteria")
    h.check(met?.baselineRunTPOTMicros.count == 7 && met?.candidateRunTPOTMicros.count == 7,
            "outcome: exactly the 7 timed runs per arm feed the statistics")

    let heavyWarmup = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7,
                                                         candidate: [97], warmupTPOT: 1e9))
    h.check(heavyWarmup?.outcome == Gate.outcomeInternalCriteriaMet
                && heavyWarmup?.baselineMedianTPOTMicros == 100 && heavyWarmup?.baselineSpread == 0,
            "outcome: warmup runs are excluded from medians and spreads")

    let exact = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: [100, 100, 100, 100, 100, 100, 105],
                                                   candidate: [98, 98, 98, 98, 98, 98, 98]))
    h.check(exact?.gain == 0.02 && exact?.baselineSpread == 0.05
                && exact?.outcome == Gate.outcomeInternalCriteriaMet,
            "outcome: exactly 2% gain with exactly 5% baseline spread meets the criteria")

    let below = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [98.0001]))
    h.check(below?.outcome == Gate.outcomeNoGain && below?.gainMeetsThreshold == false,
            "outcome: gain just below 2% is noGain")
    let slower = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [101]))
    h.check(slower?.outcome == Gate.outcomeNoGain && (slower?.gain ?? 0) < 0,
            "outcome: a slower candidate is noGain with negative gain")

    let baseNoisy = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: [100, 100, 100, 100, 100, 100, 105.01],
                                                       candidate: [95]))
    h.check(baseNoisy?.outcome == Gate.outcomeUnstable && baseNoisy?.baselineStable == false
                && baseNoisy?.candidateStable == true,
            "outcome: baseline spread just above 5% is unstable")
    let candNoisy = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7,
                                                       candidate: [90, 90, 90, 90, 90, 90, 95]))
    h.check(candNoisy?.outcome == Gate.outcomeUnstable && candNoisy?.baselineStable == true
                && candNoisy?.candidateStable == false,
            "outcome: candidate spread above 5% alone is unstable")
    let outlier = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: [100, 100, 100, 100, 100, 100, 1000],
                                                     candidate: [97]))
    h.check(outlier?.outcome == Gate.outcomeUnstable && outlier?.baselineSpread == 9
                && outlier?.baselineRunTPOTMicros.count == 7,
            "outcome: an outlier run is not trimmed and makes the arm unstable")
    let noisyAndSlow = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: [100, 100, 100, 100, 100, 100, 110],
                                                          candidate: [101]))
    h.check(noisyAndSlow?.outcome == Gate.outcomeUnstable, "outcome: unstable takes precedence over noGain")

    let oneWarmup = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [97],
                                                       warmupArms: [.off]))
    h.check(oneWarmup?.outcome == Gate.outcomeRunCheckFailed, "outcome: a missing candidate warmup fails the run checks")
    let reversedWarmup = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [97],
                                                            warmupArms: [.on, .off]))
    h.check(reversedWarmup?.outcome == Gate.outcomeRunCheckFailed, "outcome: warmups in ON, OFF order fail the run checks")
    let sixBlocks = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [97], blocks: 6))
    h.check(sixBlocks?.outcome == Gate.outcomeRunCheckFailed && sixBlocks?.baselineRunTPOTMicros.count == 6,
            "outcome: 6 blocks fail the run checks")
    let eightBlocks = outcome(fx, Synthetic.protocolRuns(fixture: fx, baseline: flat7, candidate: [97], blocks: 8))
    h.check(eightBlocks?.outcome == Gate.outcomeRunCheckFailed, "outcome: 8 blocks fail the run checks")

    let base = metRuns(fx)
    var wrongOrder = mutate(base, where: isTimed(.off, block: 1)) { $0["orderInBlock"] = 0 }
    wrongOrder = mutate(wrongOrder, where: isTimed(.on, block: 1)) { $0["orderInBlock"] = 1 }
    h.check(outcome(fx, wrongOrder)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: block 1 recorded in AB instead of BA order fails the run checks")
    var duplicate = mutate(base, where: isTimed(.off, block: 6)) { $0["block"] = 5 }
    duplicate = mutate(duplicate, where: isTimed(.on, block: 6)) { $0["block"] = 5 }
    h.check(outcome(fx, duplicate)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: a duplicated block (block 6 relabeled 5) fails the run checks")

    let frozen = Synthetic.frozen(fx)
    let badIds = mutate(base, where: isTimed(.on, block: 3)) {
        $0["ids"] = Array(frozen.prefix(31))
        $0["outputs"] = 31
        $0["idsMatchFrozen"] = false
        $0["checksPassed"] = false
        $0["failure"] = "output IDs differ from the 32 frozen IDs"
    }
    h.check(outcome(fx, badIds)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: one timed run with malformed IDs fails the run checks")
    let badCounters = mutate(base, where: isTimed(.off, block: 2)) {
        $0["counts"] = Synthetic.counts(mismatches: ["continuation overlapAsyncCommitted 61 != 60"])
        $0["checksPassed"] = false
        $0["failure"] = "counters: continuation overlapAsyncCommitted 61 != 60"
    }
    h.check(outcome(fx, badCounters)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: one timed run with a counter mismatch fails the run checks")
    let badWarmup = mutate(base, where: isWarmup(.on)) {
        $0["checksPassed"] = false
        $0["failure"] = "scope: refused"
    }
    h.check(outcome(fx, badWarmup)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: a failed warmup is excluded from statistics but still fails the run checks")
    let missingTPOT = mutate(base, where: isTimed(.off, block: 4)) {
        $0["continuationWallMicros"] = Array(repeating: 100.0, count: 30)
        $0.removeValue(forKey: "tpotMedianWallMicros")
    }
    h.check(outcome(fx, missingTPOT)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: a timed run without a run TPOT fails the run checks")
    let noisyAndFailed = mutate(Synthetic.protocolRuns(fixture: fx, baseline: [100, 100, 100, 100, 100, 100, 110],
                                                       candidate: [97]),
                                where: isTimed(.on, block: 0)) { $0["checksPassed"] = false }
    h.check(outcome(fx, noisyAndFailed)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: runCheckFailed takes precedence over unstable")

    let mixed = metRuns("en-chat") + mutate(metRuns("kr-chat"), where: isTimed(.on, block: 0)) {
        $0["checksPassed"] = false
    }
    h.check(outcome("en-chat", mixed)?.outcome == Gate.outcomeInternalCriteriaMet
                && outcome("kr-chat", mixed)?.outcome == Gate.outcomeRunCheckFailed,
            "outcome: runs of other fixtures are ignored per fixture")
    let none = outcome(fx, [])
    h.check(none?.outcome == Gate.outcomeRunCheckFailed && none?.baselineMedianTPOTMicros == nil
                && none?.gain == nil,
            "outcome: no runs fail the run checks with nil statistics")
}

// MARK: - performanceStatus

func testPerformanceStatus(_ h: inout Harness) {
    func o(_ name: String, _ runs: [[String: Any]]) -> RouterOverlapGateFixtureOutcome? { outcome(name, runs) }
    let names = RouterOverlapGateFixtures.order
    let met = names.compactMap { o($0, metRuns($0)) }
    let noGain = names.compactMap { o($0, Synthetic.protocolRuns(fixture: $0, baseline: flat7, candidate: [99])) }
    let unstable = names.compactMap { o($0, Synthetic.protocolRuns(fixture: $0, baseline: [100, 100, 100, 100, 100, 100, 110],
                                                                    candidate: [97])) }
    let failed = names.compactMap { o($0, mutate(metRuns($0), where: isTimed(.on, block: 0)) { $0["checksPassed"] = false }) }
    guard met.count == 3, noGain.count == 3, unstable.count == 3, failed.count == 3 else {
        h.check(false, "status: synthetic outcomes could not be constructed")
        return
    }
    h.check(Gate.performanceStatus(met) == .performanceInternalCriteriaMetExternalEligibilityPending,
            "status: all three fixtures met give internal-criteria-met with eligibility pending")
    h.check(Gate.performanceStatus([met[1], met[0], met[2]]) == .performanceRunCheckFailed,
            "status: outcomes out of the fixed fixture order fail the run checks")
    h.check(Gate.performanceStatus(Array(met.prefix(2))) == .performanceRunCheckFailed,
            "status: a missing fixture outcome fails the run checks")
    h.check(Gate.performanceStatus(met + [met[2]]) == .performanceRunCheckFailed,
            "status: an extra fixture outcome fails the run checks")
    h.check(Gate.performanceStatus([]) == .performanceRunCheckFailed, "status: no outcomes fail the run checks")
    h.check(Gate.performanceStatus([met[0], met[1], noGain[2]]) == .performanceNoGain,
            "status: one fixture below 2% gain gives noGain")
    h.check(Gate.performanceStatus([met[0], unstable[1], noGain[2]]) == .performanceUnstable,
            "status: unstable takes precedence over noGain")
    h.check(Gate.performanceStatus([failed[0], unstable[1], noGain[2]]) == .performanceRunCheckFailed,
            "status: runCheckFailed takes precedence over unstable and noGain")
    h.check(Gate.performanceStatus([unstable[0], met[1], met[2]]) == .performanceUnstable,
            "status: one unstable fixture gives unstable")
}

// MARK: - Counters and derived expectations

func testCounters(_ h: inout Harness) {
    typealias C = RouterOverlapGateCounters
    let keys = C.keys
    h.check(Set(keys).count == keys.count && keys.count == 34, "counters: 34 unique named keys")
    h.check(C.totalKeys == ["kernels", "buffers"] && C.totalKeys.isSubset(of: Set(keys)),
            "counters: kernels and buffers are the only relative-total keys")
    h.check(C.zero.values.count == keys.count && C.zero.values.values.allSatisfy { $0 == 0 },
            "counters: zero holds every key at 0")
    let partial = C(values: ["moeGu": 5, "bogus": 9])
    h.check(partial["overlapApplied"] == 0 && partial["bogus"] == 9 && partial["neverSeen"] == 0,
            "counters: a missing key reads as 0")
    let diff = partial.minus(C(values: ["moeGu": 2]))
    h.check(diff.values.count == keys.count && diff["moeGu"] == 3 && diff.values["bogus"] == nil,
            "counters: minus covers exactly the named keys, drops unknown keys, treats missing as 0")
    let scaled = partial.scaled(31)
    h.check(scaled.values.count == keys.count && scaled["moeGu"] == 155 && scaled.values["bogus"] == nil,
            "counters: scaled covers exactly the named keys and drops unknown keys")

    var a = C.zero
    a.values["kernels"] = 999
    a.values["buffers"] = 7
    a.values["bogus"] = 4
    h.check(C.mismatches(actual: a, expected: C.zero, label: "prefill").isEmpty,
            "counters: kernels, buffers and unknown keys never produce mismatches")
    a.values["moeGu"] = 1
    a.values["overlapPending"] = 2
    h.check(C.mismatches(actual: a, expected: C.zero, label: "prefill")
                == ["prefill moeGu 1 != 0", "prefill overlapPending 2 != 0"],
            "counters: mismatches name each differing key in key order with the label")
    let missing = C(values: [:])
    h.check(C.mismatches(actual: missing, expected: C.zero, label: "x").isEmpty,
            "counters: an empty actual set matches an all-zero expectation")
    var nonzero = C.zero
    nonzero.values["moeGu"] = 60
    h.check(C.mismatches(actual: missing, expected: nonzero, label: "continuation") == ["continuation moeGu 0 != 60"],
            "counters: a missing actual key is a mismatch when its expectation is nonzero")
}

func testDerived(_ h: inout Harness) {
    guard let d = try? Synthetic.derived() else {
        h.check(false, "derived: synthetic derived counts could not be decoded")
        return
    }
    typealias C = RouterOverlapGateCounters
    h.check(d.failures.isEmpty && d.rmsCooperativeCalls == 241,
            "derived: a consistent 30-layer topK-8 set has no failures and 241 cooperative RMS calls")
    let corruptions: [(String, Int)] = [
        ("statisticsReuses", 59), ("statisticsGroups", 29), ("q6Matvecs", 2), ("moeChunks", 3),
        ("overlapWindows", 29), ("attentionOps", 29), ("routerScaleOps", 29), ("expertChains", 29),
        ("q4Matvecs", 0), ("layers", 0), ("rmsnormOps", 90), ("routerProjections", 29),
    ]
    for (key, value) in corruptions {
        let bad = try? Synthetic.derived([key: value])
        h.check(bad.map { !$0.failures.isEmpty } ?? false, "derived: " + key + " = " + String(value) + " is reported as a failure")
    }
    let ref = d.perForward(.reference)
    let off = d.perForward(.off)
    let on = d.perForward(.on)
    h.check(ref["q4Generic"] == 685 && ref["q4Applied"] == 685
                && keys(of: ref).filter { $0 != "q4Generic" && $0 != "q4Applied" }.allSatisfy { ref[$0] == 0 },
            "derived: reference expects 205 + 2 * 8 * 30 generic Q4 dispatches and nothing else")
    h.check(off["moeGu"] == 60 && off["moeAct"] == 60 && off["moeDn"] == 60 && off["moeMerges"] == 240
                && off["moeGenerated"] == 180 && off["q4Generic"] == 205 && off["q4Applied"] == 205
                && off["parallelAttentionApplied"] == 30 && off["rmsCooperativeApplied"] == 241
                && off["q6CooperativeApplied"] == 1 && off["rmsStatsComputed"] == 30
                && off["rmsStatsReused"] == 60 && off["routerScaleApplied"] == 30
                && off["residentZeroApplied"] == 30 && off["routerLaunchApplied"] == 30,
            "derived: OFF per-forward expectations follow the v1 E4R32 and modern-axis accounting")
    let differing = C.keys.filter { on[$0] != off[$0] }
    h.check(differing == ["overlapApplied", "overlapAsyncCommitted", "overlapDrained"]
                && on["overlapApplied"] == 30 && on["overlapAsyncCommitted"] == 60 && on["overlapDrained"] == 60,
            "derived: ON differs from OFF only by 30 applied windows and 60 commits/drains")
    let zeroKeys = C.keys.filter { $0.contains("Fallback") || $0 == "overlapFailedDrains" || $0 == "overlapPending"
        || C.totalKeys.contains($0) }
    h.check([ref, off, on].allSatisfy { c in zeroKeys.allSatisfy { c[$0] == 0 } },
            "derived: every fallback, failed drain, pending and total key is expected 0 in all arms")
    h.check(on.scaled(21)["overlapApplied"] == 630 && on.scaled(31)["moeMerges"] == 7440,
            "derived: phase expectations scale per forward (21 prefill, 31 continuation)")
    let expected = on.scaled(31)
    h.check(C.mismatches(actual: expected, expected: expected, label: "continuation").isEmpty,
            "derived: an exact continuation delta has no mismatches")
    var oneExtra = expected
    oneExtra.values["overlapAsyncCommitted"] = 1861
    h.check(C.mismatches(actual: oneExtra, expected: expected, label: "continuation")
                == ["continuation overlapAsyncCommitted 1861 != 1860"],
            "derived: one extra async commit is exactly one named mismatch")
}

func keys(of c: RouterOverlapGateCounters) -> [String] { RouterOverlapGateCounters.keys }

// MARK: - Candidate

func testCandidate(_ h: inout Harness) {
    let modern = Gate.modernCandidate
    let flags = Gate.candidateFlags(modern)
    h.check(flags.count == 14 && flags == Gate.expectedCandidateFlags,
            "candidate: the declared modern candidate maps to exactly the fourteen expected flags")
    h.check(GemmaForge.candidates().contains(modern) && (try? GemmaForge.validateCandidate(modern)) != nil,
            "candidate: the declared modern candidate is legal and valid")
    h.check(RouterOverlapScope.candidateRejection(modern) == nil,
            "candidate: the router overlap scope accepts the declared modern candidate")
    let plain = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
    h.check(Gate.candidateFlags(plain) != Gate.expectedCandidateFlags
                && RouterOverlapScope.candidateRejection(plain) != nil,
            "candidate: a non-modern candidate differs in flags and is rejected by the scope")
}

// MARK: - Frozen incumbent binding

func testIncumbentBinding(_ h: inout Harness) {
    let modern = Gate.modernCandidate
    let flags = Gate.candidateFlags(modern)
    h.check(modern.routerLaunchDivisor == 1 && flags["routerLaunchDivisor"] == 1
                && Gate.expectedCandidateFlags["routerLaunchDivisor"] == 1
                && flags["routerGPUResidency"] == 1 && flags["parallelAttention"] == 1
                && flags["matvecTiledF32"] == 0,
            "incumbent: the declared candidate binds router launch divisor 1 with GPU residency, par64 and no tiled F32")
    let divisorZero = GemmaForgeCandidate(
        segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
        gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
        matvecCooperativeQ6: true, rmsStatisticsReuse: true, routerGPUResidency: true,
        routerLaunchDivisor: 0)
    h.check(Gate.candidateFlags(divisorZero) != Gate.expectedCandidateFlags,
            "incumbent: the divisor-0 variant does not match the declared fourteen-flag map")
    guard let d = try? Synthetic.derived() else {
        h.check(false, "incumbent: synthetic derived counts could not be decoded")
        return
    }
    let ref = d.perForward(.reference)
    let off = d.perForward(.off)
    let on = d.perForward(.on)
    h.check(d.routerProjections == d.layers && off["routerLaunchApplied"] == 30
                && on["routerLaunchApplied"] == 30 && ref["routerLaunchApplied"] == 0
                && [ref, off, on].allSatisfy { $0["routerLaunchFallback"] == 0 },
            "incumbent: router launch applied is exactly one per router projection (30) in OFF and ON, 0 in the reference, no fallback")
    var neverApplied = off.scaled(31)
    neverApplied.values["routerLaunchApplied"] = 0
    h.check(RouterOverlapGateCounters.mismatches(actual: neverApplied, expected: off.scaled(31),
                                                 label: "continuation")
                == ["continuation routerLaunchApplied 0 != 930"],
            "incumbent: a run that never applied the router launch geometry is exactly one named mismatch")
}

// MARK: - Report

func encodedObject(_ r: RouterOverlapGateReport) -> [String: Any]? {
    guard let data = try? Gate.encodeReport(r, pretty: false) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

func testReport(_ h: inout Harness) {
    for mode in [RouterOverlapModelGateMode.qualificationOnly, .performance] {
        let r = RouterOverlapGateReport(mode: mode)
        let o = encodedObject(r)
        h.check(!r.gainQualified && r.status == .aborted && r.kind == Gate.kind
                    && (o?["gainQualified"] as? Bool) == false
                    && (o?["mode"] as? String) == mode.rawValue,
                "report: a fresh " + mode.rawValue + " report encodes gainQualified false and status aborted")
    }
    let aborted = Gate.aborted(mode: .performance, reason: "runner load failed")
    h.check(aborted.status == .aborted && aborted.abortReason == "runner load failed" && !aborted.gainQualified
                && (encodedObject(aborted)?["gainQualified"] as? Bool) == false,
            "report: aborted() keeps gainQualified false")

    let names = RouterOverlapGateFixtures.order
    let metOutcomes = names.compactMap { outcome($0, metRuns($0)) }
    var source = RouterOverlapGateReport(mode: .performance)
    source.qualificationPassed = true
    source.restoredPristine = true
    source.fixtures = metOutcomes
    source.status = Gate.performanceStatus(metOutcomes)
    let sourceObject = encodedObject(source)
    h.check(metOutcomes.count == 3
                && source.status == .performanceInternalCriteriaMetExternalEligibilityPending
                && !source.gainQualified && (sourceObject?["gainQualified"] as? Bool) == false,
            "report: internal criteria met on every fixture still encodes gainQualified false")
    h.check((sourceObject?["externalResourceEligibility"] as? String) == Gate.externalResourceEligibility
                && Gate.externalResourceEligibility.hasPrefix("PENDING")
                && (sourceObject?["notes"] as? [String]) == Gate.notes,
            "report: external eligibility is encoded as PENDING with the fixed notes")

    var nonFinite = RouterOverlapGateReport(mode: .qualificationOnly)
    nonFinite.qualificationElapsedMicros = .infinity
    h.check((encodedObject(nonFinite)?["qualificationElapsedMicros"] as? String) == "inf",
            "report: encodeReport writes infinity as the string inf instead of throwing")
    nonFinite.qualificationElapsedMicros = .nan
    h.check((encodedObject(nonFinite)?["qualificationElapsedMicros"] as? String) == "nan",
            "report: encodeReport writes NaN as the string nan instead of throwing")

    // A caller-supplied true flag must never be re-emitted by encodeReport.
    var injected: Bool? = nil
    if var obj = sourceObject {
        obj["gainQualified"] = true
        if let tampered = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(RouterOverlapGateReport.self, from: tampered),
           let again = encodedObject(decoded) {
            injected = again["gainQualified"] as? Bool
        }
    }
    h.check(injected != true,
            "report: a decoded report carrying gainQualified=true is not re-emitted as true by encodeReport")

    var otherFieldsKept = false
    var prettyFalse = false
    if var obj = sourceObject {
        obj["gainQualified"] = true
        obj["status"] = RouterOverlapGateStatus.performanceNoGain.rawValue
        if let tampered = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(RouterOverlapGateReport.self, from: tampered) {
            let again = encodedObject(decoded)
            otherFieldsKept = (again?["status"] as? String) == RouterOverlapGateStatus.performanceNoGain.rawValue
                && (again?["qualificationPassed"] as? Bool) == true
                && (again?["gainQualified"] as? Bool) == false
            if let pretty = try? Gate.encodeReport(decoded, pretty: true),
               let prettyObject = (try? JSONSerialization.jsonObject(with: pretty)) as? [String: Any] {
                prettyFalse = (prettyObject["gainQualified"] as? Bool) == false
            }
        }
    }
    h.check(otherFieldsKept,
            "report: re-encoding a tampered report forces only gainQualified false and keeps other decoded fields")
    h.check(prettyFalse, "report: pretty encodeReport also never re-emits gainQualified true")
}

// MARK: - GPU-elapsed coverage

func testCoverage(_ h: inout Harness) {
    let coverage = Gate.gpuElapsedCoverage
    h.check(Set(coverage.keys) == ["reference", "off", "on"],
            "coverage: exactly the reference, off and on arms are described")
    let on = coverage["on"] ?? ""
    h.check(on.hasPrefix("INCOMPLETE") && on.contains("EXCLUDES") && on.contains("R/S ticket"),
            "coverage: ON is marked incomplete and excludes the async R/S ticket command buffers")
    h.check(on.contains("coverage differs from OFF and reference")
                && (coverage["reference"] ?? "").contains("different execution path and coverage")
                && (coverage["off"] ?? "").contains("never compared with ON")
                && on.contains("never compared with OFF"),
            "coverage: arms declare different coverage and are never compared")
    h.check(coverage.values.allSatisfy { $0.contains("not busy time") },
            "coverage: no arm's value is described as busy time")
    let texts = Array(coverage.values) + Gate.notes
    h.check(!texts.contains { $0.lowercased().contains("event wait") },
            "coverage: neither coverage nor notes claim that recorded elapsed includes event waits")
    h.check(Gate.notes.contains { $0.contains("gpuElapsedCoverage") && $0.contains("exclude") },
            "coverage: the notes reference gpuElapsedCoverage and the ON exclusion")

    let fresh = RouterOverlapGateReport(mode: .performance)
    let encoded = encodedObject(fresh)
    h.check(fresh.gpuElapsedCoverage == coverage
                && (encoded?["gpuElapsedCoverage"] as? [String: String]) == coverage,
            "coverage: a new report carries and encodes the per-arm coverage")

    var oldShape = encoded
    oldShape?.removeValue(forKey: "gpuElapsedCoverage")
    var old: RouterOverlapGateReport? = nil
    if let obj = oldShape, let data = try? JSONSerialization.data(withJSONObject: obj) {
        old = try? JSONDecoder().decode(RouterOverlapGateReport.self, from: data)
    }
    h.check(old != nil && old?.gpuElapsedCoverage == nil,
            "coverage: a report written before the field existed still decodes, with nil coverage")
    let reEncoded = old.flatMap { encodedObject($0) }
    h.check(reEncoded != nil && reEncoded?["gpuElapsedCoverage"] == nil
                && (reEncoded?["gainQualified"] as? Bool) == false,
            "coverage: re-encoding that old report omits the key and keeps gainQualified false")
}

// MARK: - Main

func runAll() -> Int32 {
    var h = Harness()
    testFixtureParse(&h)
    testStatistics(&h)
    testProtocol(&h)
    testFixtureOutcome(&h)
    testPerformanceStatus(&h)
    testCounters(&h)
    testDerived(&h)
    testCandidate(&h)
    testIncumbentBinding(&h)
    testReport(&h)
    testCoverage(&h)
    print("router-overlap-model-gate-pure-tests: " + String(h.passed) + " passed, "
          + String(h.failed.count) + " failed")
    for name in h.failed { print("FAILED: " + name) }
    return h.failed.isEmpty ? 0 : 1
}

exit(runAll())
