// CPU-only pure tests for the router overlap runtime bookkeeping. They
// drive the SAME ledger, read guard, scope predicate, binding check, step
// derivation, alias guard and token expectation the runner uses, with fake
// command handles. No Metal device, no GGUF, no model weights, no GPU work.
// Real MTLCommandBuffer failure and MTLSharedEvent semantics are NOT
// exercised here; they need later small GPU tests.

import Foundation
import LeviathanCore

final class Harness {
    private(set) var passed = 0
    private(set) var failed = 0

    func check(_ condition: Bool, _ name: String, _ detail: String = "") {
        let suffix = detail.isEmpty ? "" : ": " + detail
        if condition {
            passed += 1
            print("ok   " + name + suffix)
        } else {
            failed += 1
            print("FAIL " + name + suffix)
        }
    }

    func rejects(_ reason: String?, _ name: String) {
        check(reason != nil, name, reason ?? "accepted (expected a rejection)")
    }

    func accepts(_ reason: String?, _ name: String) {
        check(reason == nil, name, reason ?? "")
    }
}

final class FakeLog {
    var events: [String] = []
}

final class FakeHandle: RouterOverlapCommandHandle {
    let name: String
    let log: FakeLog
    let succeeds: Bool
    let micros: Double
    private(set) var commits = 0
    private(set) var waits = 0

    init(_ name: String, _ log: FakeLog, succeeds: Bool = true, micros: Double = 1) {
        self.name = name
        self.log = log
        self.succeeds = succeeds
        self.micros = micros
    }

    func commitHandle() {
        commits += 1
        log.events.append("commit:" + name)
    }

    func waitHandle() {
        waits += 1
        log.events.append("wait:" + name)
    }

    var handleSucceeded: Bool { succeeds }
    var handleIntervalMicros: Double { micros }
    var handleFailureDescription: String { "fake failure " + name }
}

final class Identity {}

// MARK: - ledger

func testOrderedDrain(_ h: Harness) {
    let log = FakeLog()
    let ledger = RouterOverlapTicketLedger<FakeHandle>()
    let r = FakeHandle("R", log, micros: 10)
    let s = FakeHandle("S", log, micros: 20)
    do {
        let sr = try ledger.commit(r, kind: .router) { log.events.append("signal:\($0)") }
        let ss = try ledger.commit(s, kind: .shared) { log.events.append("signal:\($0)") }
        h.check(sr == 1 && ss == 2, "signals are monotonic from the retained counter", "\(sr), \(ss)")
    } catch {
        h.check(false, "commit R/S", "\(error)")
        return
    }
    h.check(log.events == ["signal:1", "commit:R", "signal:2", "commit:S"],
            "signal encoded before each single commit", "\(log.events)")
    h.check(r.commits == 1 && s.commits == 1, "each ticket committed exactly once")
    h.check(ledger.asyncCommitted == 2 && ledger.pendingRouterWrites, "two pending, router writes pending")
    var unblocks: [UInt64] = []
    let first = ledger.drain(through: 1) { unblocks.append($0) }
    h.check(first.drained == 1 && first.intervalMicros == 10, "waitR drains through R only")
    h.check(s.waits == 0 && ledger.pending.count == 1, "S stays pending after waitR")
    h.check(ledger.lastSucceededRouterSignal == 1 && !ledger.pendingRouterWrites,
            "router readable state after R")
    let again = ledger.drain(through: 1) { unblocks.append($0) }
    h.check(again.drained == 0 && again.intervalMicros == 0, "repeat wait drains nothing")
    h.check(ledger.intervalMicrosAccounted == 10, "repeat wait does not double-sum")
    let rest = ledger.drain(through: nil) { unblocks.append($0) }
    h.check(rest.drained == 1 && ledger.intervalMicrosAccounted == 30, "drain-all accounts S once")
    h.check(!ledger.isPoisoned, "successful drains never poison")
    h.check(ledger.drained == 2 && !ledger.hasPending, "all drained, none pending")
    h.check(r.waits == 1 && s.waits == 1, "each ticket waited exactly once")
    h.check(unblocks.isEmpty, "no unblock without a failure")
    h.accepts(ledger.consistencyRejection, "ledger counters consistent after ordered drains")
}

func testFailedDrainContinues(_ h: Harness) {
    let log = FakeLog()
    let ledger = RouterOverlapTicketLedger<FakeHandle>()
    let r1 = FakeHandle("R1", log, succeeds: false, micros: 100)
    let s1 = FakeHandle("S1", log, micros: 5)
    let r2 = FakeHandle("R2", log, micros: 7)
    do {
        try ledger.commit(r1, kind: .router) { _ in }
        try ledger.commit(s1, kind: .shared) { _ in }
        try ledger.commit(r2, kind: .router) { _ in }
    } catch {
        h.check(false, "commit failure fixture", "\(error)")
        return
    }
    var unblocks: [UInt64] = []
    let outcome = ledger.drain(through: 1) { value in
        unblocks.append(value)
        log.events.append("unblock:\(value)")
    }
    h.check(outcome.firstFailure != nil, "failed ticket reported", outcome.firstFailure ?? "")
    h.check(outcome.drained == 3 && !ledger.hasPending, "failure drains every committed ticket")
    h.check(unblocks == [3, 3] && outcome.unblockedThrough == 3,
            "unblock through the highest committed signal, re-raised after the last wait")
    let waits = log.events.filter { !$0.hasPrefix("commit:") }
    h.check(waits == ["wait:R1", "unblock:3", "wait:S1", "wait:R2", "unblock:3"],
            "commit-order waits with unblock before later waits and after the last", "\(waits)")
    h.check(outcome.intervalMicros == 12 && ledger.intervalMicrosAccounted == 12,
            "failed ticket interval is never accounted")
    h.check(ledger.failedDrains == 1, "failure counted once")
    h.accepts(ledger.consistencyRejection, "ledger counters consistent after failure")
}

func testRouterReadGuard(_ h: Harness) {
    let router = Identity()
    let foreign = Identity()
    let rid = ObjectIdentifier(router)
    let log = FakeLog()
    let ledger = RouterOverlapTicketLedger<FakeHandle>()
    let r = FakeHandle("R", log)
    let signal: UInt64
    do {
        signal = try ledger.commit(r, kind: .router) { _ in }
    } catch {
        h.check(false, "commit router fixture", "\(error)")
        return
    }
    func reason(_ requested: ObjectIdentifier, routerSignal: UInt64, count: Int) -> String? {
        RouterOverlapReadGuard.routerReadRejection(
            requested: requested, router: rid, routerSignal: routerSignal,
            lastSucceededRouterSignal: ledger.lastSucceededRouterSignal,
            pendingRouterWrites: ledger.pendingRouterWrites,
            count: count, bufferLength: 512)
    }
    h.rejects(reason(rid, routerSignal: signal, count: 128), "read before completion rejects")
    _ = ledger.drain(through: signal) { _ in }
    h.accepts(reason(rid, routerSignal: signal, count: 128), "read after successful R accepted")
    h.rejects(reason(ObjectIdentifier(foreign), routerSignal: signal, count: 128),
              "foreign buffer rejects")
    h.rejects(reason(rid, routerSignal: signal + 1, count: 128), "other router signal rejects")
    h.rejects(reason(rid, routerSignal: signal, count: 129), "count beyond buffer rejects")
    h.rejects(reason(rid, routerSignal: signal, count: 0), "zero count rejects")
    h.rejects(reason(rid, routerSignal: signal, count: Int.max), "byte overflow rejects")
    let failing = RouterOverlapTicketLedger<FakeHandle>()
    let bad = FakeHandle("Rbad", log, succeeds: false)
    do {
        let s2 = try failing.commit(bad, kind: .router) { _ in }
        _ = failing.drain(through: s2) { _ in }
        h.rejects(RouterOverlapReadGuard.routerReadRejection(
            requested: rid, router: rid, routerSignal: s2,
            lastSucceededRouterSignal: failing.lastSucceededRouterSignal,
            pendingRouterWrites: failing.pendingRouterWrites,
            count: 128, bufferLength: 512), "failed router ticket is never readable")
    } catch {
        h.check(false, "commit failing router fixture", "\(error)")
    }
}

func testSequenceOverflow(_ h: Harness) {
    let log = FakeLog()
    let ledger = RouterOverlapTicketLedger<FakeHandle>(lastCommittedSignal: UInt64.max - 1)
    let a = FakeHandle("A", log)
    let b = FakeHandle("B", log)
    var encoded: [UInt64] = []
    do {
        let s = try ledger.commit(a, kind: .router) { encoded.append($0) }
        h.check(s == UInt64.max, "last representable signal commits")
    } catch {
        h.check(false, "commit at max-1", "\(error)")
    }
    var threw = false
    do {
        try ledger.commit(b, kind: .shared) { encoded.append($0) }
    } catch {
        threw = true
        print("     -> \(error)")
    }
    h.check(threw && b.commits == 0, "overflow fails before commit")
    h.check(encoded == [UInt64.max], "overflow fails before signal encode")
    h.check(ledger.asyncCommitted == 1 && ledger.lastCommittedSignal == UInt64.max,
            "counter never wraps or resets")
    _ = ledger.drain(through: nil) { _ in }
    h.accepts(ledger.consistencyRejection, "ledger consistent after overflow refusal")
    let dup = RouterOverlapTicketLedger<FakeHandle>()
    let c = FakeHandle("C", log)
    var dupThrew = false
    do {
        try dup.commit(c, kind: .router) { _ in }
        try dup.commit(c, kind: .shared) { _ in }
    } catch {
        dupThrew = true
    }
    h.check(dupThrew && c.commits == 1 && dup.asyncCommitted == 1,
            "double commit of one pending handle rejected")
}

// MARK: - scope predicate and candidates

func qualifiedSnapshot(_ phase: RouterOverlapScopePhase) -> RouterOverlapScopeSnapshot {
    let active = phase == .token
    return RouterOverlapScopeSnapshot(
        scopeActive: active, eventAndScratchReady: active, openEncoderPresent: false,
        pendingTickets: 0, expertCacheEnabled: false, residentWeightsMapped: true,
        moeBatch: 4, moeRowsPerGroup: 32, installIdentityVerified: true,
        q4LaunchWidth: 32, rmsGeneratedInstalled: false, persistentAxisFlagsClean: true,
        layerProbeInstalled: false, moeInputProbeInstalled: false, traceLayers: false,
        breakdownInstrumentation: false, commandTimelineEnabled: false,
        attentionDiagnosticActive: false)
}

func testScopePredicate(_ h: Harness) {
    let mutations: [(String, (inout RouterOverlapScopeSnapshot) -> Void)] = [
        ("open encoder", { $0.openEncoderPresent = true }),
        ("pending ticket", { $0.pendingTickets = 1 }),
        ("expert cache", { $0.expertCacheEnabled = true }),
        ("no resident mapping", { $0.residentWeightsMapped = false }),
        ("no MoE lowering", { $0.moeBatch = nil; $0.moeRowsPerGroup = nil }),
        ("MoE E8", { $0.moeBatch = 8 }),
        ("MoE R64", { $0.moeRowsPerGroup = 64 }),
        ("install identity", { $0.installIdentityVerified = false }),
        ("Q4 width 64", { $0.q4LaunchWidth = 64 }),
        ("generated RMS", { $0.rmsGeneratedInstalled = true }),
        ("axis flags", { $0.persistentAxisFlagsClean = false }),
        ("layerProbe", { $0.layerProbeInstalled = true }),
        ("moeInputProbe", { $0.moeInputProbeInstalled = true }),
        ("traceLayers", { $0.traceLayers = true }),
        ("breakdown", { $0.breakdownInstrumentation = true }),
        ("timeline", { $0.commandTimelineEnabled = true }),
        ("attention diagnostic", { $0.attentionDiagnosticActive = true }),
    ]
    for phase in [RouterOverlapScopePhase.entry, .token] {
        let label = phase == .entry ? "entry" : "token"
        h.accepts(RouterOverlapScope.rejection(qualifiedSnapshot(phase), phase: phase),
                  "\(label): qualified snapshot accepted")
        var noWidth = qualifiedSnapshot(phase)
        noWidth.q4LaunchWidth = nil
        h.accepts(RouterOverlapScope.rejection(noWidth, phase: phase),
                  "\(label): default Q4 width accepted")
        for (name, mutate) in mutations {
            var s = qualifiedSnapshot(phase)
            mutate(&s)
            h.rejects(RouterOverlapScope.rejection(s, phase: phase), "\(label): \(name) rejected")
        }
    }
    var nested = qualifiedSnapshot(.entry)
    nested.scopeActive = true
    h.rejects(RouterOverlapScope.rejection(nested, phase: .entry), "entry: nesting rejected")
    var inactive = qualifiedSnapshot(.token)
    inactive.scopeActive = false
    h.rejects(RouterOverlapScope.rejection(inactive, phase: .token), "token: inactive scope rejected")
    var noEvent = qualifiedSnapshot(.token)
    noEvent.eventAndScratchReady = false
    h.rejects(RouterOverlapScope.rejection(noEvent, phase: .token), "token: missing event rejected")
}

func testCandidates(_ h: Harness) {
    let all = GemmaForge.candidates()
    h.check(all.count == 32 && Set(all).count == 32, "legal candidate set unchanged (32 unique)")
    var allValid = true
    for c in all {
        do { try GemmaForge.validateCandidate(c) } catch { allValid = false }
    }
    h.check(allValid, "every legal candidate still validates")
    let eligible = all.filter { RouterOverlapScope.candidateRejection($0) == nil }
    h.check(eligible.count == 5 && eligible.allSatisfy { $0.routerGPUResidency },
            "exactly the five router-residency candidates are overlap-eligible")
    h.check(Set(eligible.map(\.routerLaunchDivisor)) == [0, 1, 2, 4, 8],
            "every allowed launch divisor is eligible")
    let modern = GemmaForgeCandidate(
        segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
        gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
        matvecCooperativeQ6: true, rmsStatisticsReuse: true,
        routerGPUResidency: true, routerLaunchDivisor: 0)
    h.accepts(RouterOverlapScope.candidateRejection(modern), "full modern v1 candidate accepted")
    let variants: [(String, GemmaForgeCandidate)] = [
        ("top5 diagnostic", GemmaForgeCandidate(
            segmentBatching: true, top5Diagnostic: true, vectorizedLoads: true,
            gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
            matvecCooperativeQ6: true, rmsStatisticsReuse: true,
            routerGPUResidency: true, routerLaunchDivisor: 0)),
        ("no segment batching", GemmaForgeCandidate(
            segmentBatching: false, top5Diagnostic: false, vectorizedLoads: true,
            gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
            matvecCooperativeQ6: true, rmsStatisticsReuse: true,
            routerGPUResidency: true, routerLaunchDivisor: 0)),
        ("tiled Q6", GemmaForgeCandidate(
            segmentBatching: true, top5Diagnostic: false, matvecTiledQ6: true,
            vectorizedLoads: true, gpuCacheAppend: true, parallelAttention: true,
            rmsnormCooperative: true, matvecCooperativeQ6: true,
            rmsStatisticsReuse: true, routerGPUResidency: true, routerLaunchDivisor: 0)),
        ("divisor 3", GemmaForgeCandidate(
            segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
            gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
            matvecCooperativeQ6: true, rmsStatisticsReuse: true,
            routerGPUResidency: true, routerLaunchDivisor: 3)),
        ("no router residency", GemmaForgeCandidate(
            segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
            gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true,
            matvecCooperativeQ6: true, rmsStatisticsReuse: true)),
    ]
    for (name, candidate) in variants {
        h.rejects(RouterOverlapScope.candidateRejection(candidate), "candidate \(name) rejected")
    }
}

// MARK: - binding, steps, alias, expectation

func qualifiedFacts() -> RouterOverlapBindingFacts {
    let eps = Float(1e-6).bitPattern
    return RouterOverlapBindingFacts(
        planAnalysis: RouterOverlapPlan.analysisVersion, digestVerified: true,
        canonicalScheduleMatches: true, scheduleDigestMatches: true,
        statisticsDigestMatches: true, statisticsAnalysisMatches: true,
        planEpsilonBits: eps, statisticsEpsilonBits: eps, runnerEpsilonBits: eps,
        configMatches: true, layerCount: 30, windowCount: 30,
        instructionCount: 934, permutationCount: 934,
        remappedCount: 60, uniqueRemappedCount: 60, remappedMatchesWindows: true,
        controlRejectedIndexMatches: true, statisticsGroups: 30, statisticsReuses: 60)
}

func testBindingFacts(_ h: Harness) {
    h.accepts(RouterOverlapBindingCheck.rejection(qualifiedFacts()), "qualified binding accepted")
    let mutations: [(String, (inout RouterOverlapBindingFacts) -> Void)] = [
        ("analysis version", { $0.planAnalysis += 1 }),
        ("plan digest", { $0.digestVerified = false }),
        ("canonical schedule", { $0.canonicalScheduleMatches = false }),
        ("schedule digest", { $0.scheduleDigestMatches = false }),
        ("statistics digest", { $0.statisticsDigestMatches = false }),
        ("statistics analysis", { $0.statisticsAnalysisMatches = false }),
        ("plan epsilon", { $0.planEpsilonBits ^= 1 }),
        ("statistics epsilon", { $0.statisticsEpsilonBits ^= 1 }),
        ("runner epsilon", { $0.runnerEpsilonBits ^= 1 }),
        ("config", { $0.configMatches = false }),
        ("window count", { $0.windowCount = 29 }),
        ("layer count", { $0.layerCount = 31 }),
        ("permutation length", { $0.permutationCount = 933 }),
        ("remapped count", { $0.remappedCount = 59 }),
        ("remapped duplicates", { $0.uniqueRemappedCount = 59 }),
        ("remapped vs windows", { $0.remappedMatchesWindows = false }),
        ("control index", { $0.controlRejectedIndexMatches = false }),
        ("statistics groups", { $0.statisticsGroups = 29 }),
        ("statistics reuses", { $0.statisticsReuses = 59 }),
    ]
    for (name, mutate) in mutations {
        var f = qualifiedFacts()
        mutate(&f)
        h.rejects(RouterOverlapBindingCheck.rejection(f), "binding \(name) rejected")
    }
}

func overlapOrder(_ i0: Int) -> [Int] {
    [i0, i0 + 6, i0 + 7, i0 + 8, i0 + 1, i0 + 2, i0 + 3, i0 + 4, i0 + 5,
     i0 + 9, i0 + 10, i0 + 11]
}

func testStepDerivation(_ h: Harness) {
    let n = 30
    let windows: [(layer: Int, canonicalIndices: [Int])] = [
        (layer: 0, canonicalIndices: Array(2..<14)),
        (layer: 1, canonicalIndices: Array(15..<27)),
    ]
    var permutation = Array(0..<n)
    for w in windows {
        let i0 = w.canonicalIndices[0]
        permutation.replaceSubrange(i0..<(i0 + 12), with: overlapOrder(i0))
    }
    do {
        let steps = try RouterOverlapPreparedBinding.deriveSteps(
            windows: windows, permutation: permutation, instructionCount: n)
        let planned = [steps[10], steps[7], steps[11], steps[23], steps[20], steps[24]]
        let expected: [RouterOverlapStep] = [
            .commitRouter(layer: 0), .commitShared(layer: 0), .awaitRouter(layer: 0),
            .commitRouter(layer: 1), .commitShared(layer: 1), .awaitRouter(layer: 1),
        ]
        h.check(planned == expected, "steps at i8 (R), i5 (S), i9 (await)")
        h.check(steps.filter { $0 != .plain }.count == 6, "only six planned steps")
    } catch {
        h.check(false, "step derivation", "\(error)")
    }
    func refused(_ w: [(layer: Int, canonicalIndices: [Int])], _ p: [Int], _ name: String) {
        do {
            _ = try RouterOverlapPreparedBinding.deriveSteps(
                windows: w, permutation: p, instructionCount: n)
            h.check(false, name, "accepted (expected a rejection)")
        } catch {
            h.check(true, name, "\(error)")
        }
    }
    refused(windows, Array(0..<n), "canonical order is not the overlap order")
    refused([windows[0], (layer: 0, canonicalIndices: Array(15..<27))], permutation,
            "duplicate layer rejected")
    var gap = windows[0].canonicalIndices
    gap[5] = 28
    refused([(layer: 0, canonicalIndices: gap)], permutation, "non-contiguous window rejected")
    refused([(layer: 0, canonicalIndices: Array(20..<32))], permutation,
            "out-of-range window rejected")
    var notPermutation = permutation
    notPermutation[0] = 1
    refused(windows, notPermutation, "non-permutation rejected")
    var outside = permutation
    outside.swapAt(0, 1)
    refused(windows, outside, "permuted position outside windows rejected")
    refused([windows[0], (layer: 1, canonicalIndices: Array(10..<22))], permutation,
            "overlapping windows rejected")
}

func testAliasGuard(_ h: Harness) {
    let a = Identity()
    let b = Identity()
    let base: UInt = 0x10000
    let dedicated = RouterOverlapBufferExtent(name: "routerIn", identity: ObjectIdentifier(a),
                                              base: base, length: 11264)
    func other(_ identity: ObjectIdentifier, _ start: UInt, _ length: Int) -> RouterOverlapBufferExtent {
        RouterOverlapBufferExtent(name: "h", identity: identity, base: start, length: length)
    }
    let distinct = other(ObjectIdentifier(b), 0x20000, 11264)
    h.accepts(RouterOverlapAliasGuard.rejection(dedicated: dedicated, requiredBytes: 11264,
                                                others: [distinct]), "distinct buffers accepted")
    h.rejects(RouterOverlapAliasGuard.rejection(
        dedicated: dedicated, requiredBytes: 11264,
        others: [other(ObjectIdentifier(a), 0x20000, 16)]), "same object rejected")
    h.rejects(RouterOverlapAliasGuard.rejection(
        dedicated: dedicated, requiredBytes: 11264,
        others: [other(ObjectIdentifier(b), base + 100, 16)]), "overlapping range rejected")
    h.accepts(RouterOverlapAliasGuard.rejection(
        dedicated: dedicated, requiredBytes: 11264,
        others: [other(ObjectIdentifier(b), base + 11264, 16)]), "adjacent range accepted")
    h.rejects(RouterOverlapAliasGuard.rejection(
        dedicated: dedicated, requiredBytes: 11264,
        others: [other(ObjectIdentifier(b), UInt.max - 10, 100)]), "wrapping range rejected")
    h.rejects(RouterOverlapAliasGuard.rejection(dedicated: dedicated, requiredBytes: 11268,
                                                others: [distinct]), "undersized routerIn rejected")
}

func counters(_ applied: Int, _ committed: Int, _ drained: Int,
              _ computed: Int, _ reused: Int) -> RouterOverlapCounters {
    RouterOverlapCounters(
        applied: applied, asyncCommitted: committed, drained: drained, failedDrains: 0,
        pending: 0, statsComputed: computed, statsReused: reused, statsFallback: 0,
        routerScaleFallback: 0, residentZeroFallback: 0, routerLaunchFallback: 0,
        moeFallback: 0)
}

func testTokenExpectation(_ h: Harness) {
    let before = counters(30, 60, 60, 100, 200)
    let good = counters(60, 120, 120, 130, 260)
    h.accepts(RouterOverlapTokenExpectation.rejection(
        before: before, after: good, windows: 30, statisticsGroups: 30, statisticsReuses: 60),
        "one qualified token accepted (deltas over a nonzero baseline)")
    let mutations: [(String, (inout RouterOverlapCounters) -> Void)] = [
        ("applied 29", { $0.applied -= 1 }),
        ("async 119", { $0.asyncCommitted -= 1 }),
        ("drained 119", { $0.drained -= 1 }),
        ("failed drain", { $0.failedDrains = 1 }),
        ("pending left", { $0.pending = 1 }),
        ("statistics compute", { $0.statsComputed += 1 }),
        ("statistics reuse", { $0.statsReused -= 1 }),
        ("statistics fallback", { $0.statsFallback = 1 }),
        ("router scale fallback", { $0.routerScaleFallback = 1 }),
        ("resident zero fallback", { $0.residentZeroFallback = 1 }),
        ("router launch fallback", { $0.routerLaunchFallback = 1 }),
        ("MoE fallback", { $0.moeFallback = 1 }),
    ]
    for (name, mutate) in mutations {
        var after = good
        mutate(&after)
        h.rejects(RouterOverlapTokenExpectation.rejection(
            before: before, after: after, windows: 30, statisticsGroups: 30,
            statisticsReuses: 60), "token \(name) rejected")
    }
    h.rejects(RouterOverlapTokenExpectation.rejection(
        before: before, after: good, windows: 0, statisticsGroups: 30, statisticsReuses: 60),
        "zero windows rejected")
}

// MARK: - poison, state disposition, execution refusal

func testPoisonedLedger(_ h: Harness) {
    let log = FakeLog()
    let ledger = RouterOverlapTicketLedger<FakeHandle>()
    let r = FakeHandle("R", log, succeeds: false)
    let s = FakeHandle("S", log)
    do {
        try ledger.commit(r, kind: .router) { _ in }
        try ledger.commit(s, kind: .shared) { _ in }
    } catch {
        h.check(false, "poison fixture", "\(error)")
        return
    }
    h.check(!ledger.isPoisoned, "not poisoned before any drain")
    var unblocks: [UInt64] = []
    let outcome = ledger.drain(through: 1) { unblocks.append($0) }
    h.check(ledger.isPoisoned && ledger.firstFailure == outcome.firstFailure,
            "failed ticket poisons the ledger")
    h.check(outcome.drained == 2 && s.waits == 1,
            "every already committed ticket drained before reuse")
    h.check(unblocks == [2, 2], "unblock before later waits and after the last", "\(unblocks)")
    let next = FakeHandle("N", log)
    var encoded = false
    var refused = false
    do {
        try ledger.commit(next, kind: .router) { _ in encoded = true }
    } catch {
        refused = true
        print("     -> \(error)")
    }
    h.check(refused && !encoded && next.commits == 0,
            "poisoned ledger refuses commit before encode")
    h.check(ledger.asyncCommitted == 2 && ledger.lastCommittedSignal == 2,
            "refusal leaves the counter unchanged")
    let empty = ledger.drain(through: nil) { unblocks.append($0) }
    h.check(empty.drained == 0 && empty.firstFailure == nil && ledger.isPoisoned,
            "cleanup drain still works while poisoned; poison persists")
    h.check(unblocks.count == 2, "no extra unblock without a new failure")
    h.accepts(ledger.consistencyRejection, "poisoned ledger counters consistent")
}

func testStateDisposition(_ h: Harness) {
    let cases: [(String, Bool, Bool, Int, RouterOverlapScope.StateDisposition)] = [
        ("no state allocates", false, false, 0, .allocate),
        ("no state allocates regardless of the poison flag", false, true, 0, .allocate),
        ("clean drained state is reused", true, false, 0, .reuse),
        ("poisoned drained state is reallocated (old event not reused)", true, true, 0, .reallocate),
        ("pending tickets refuse entry", true, false, 1, .refuse),
        ("poisoned with pending tickets refuses entry", true, true, 2, .refuse),
    ]
    for (name, hasState, poisoned, pending, want) in cases {
        let got = RouterOverlapScope.stateDisposition(
            hasState: hasState, poisoned: poisoned, pendingTickets: pending)
        h.check(got == want, name, "\(got)")
    }
}

func testExecutionRejection(_ h: Harness) {
    let diag = RouterOverlapScope.ExecutionKind.diagnosticPrepared
    h.rejects(RouterOverlapScope.executionRejection(diag, scopeActive: false, poisoned: false),
              "diagnostic preparation outside the scope refused")
    h.rejects(RouterOverlapScope.executionRejection(diag, scopeActive: true, poisoned: true),
              "diagnostic execution refused while poisoned (scope stays active)")
    h.accepts(RouterOverlapScope.executionRejection(diag, scopeActive: true, poisoned: false),
              "diagnostic execution accepted in a clean active scope")
    for what in ["direct forward", "engineIdentity", "forge",
                 "raw, non-prepared or control execution"] {
        let kind = RouterOverlapScope.ExecutionKind.outsideScopeOnly(what)
        h.rejects(RouterOverlapScope.executionRejection(kind, scopeActive: true, poisoned: false),
                  "\(what) refused in an active scope")
        h.rejects(RouterOverlapScope.executionRejection(kind, scopeActive: true, poisoned: true),
                  "\(what) refused in a poisoned active scope")
        h.accepts(RouterOverlapScope.executionRejection(kind, scopeActive: false, poisoned: false),
                  "\(what) allowed outside the scope")
        h.accepts(RouterOverlapScope.executionRejection(kind, scopeActive: false, poisoned: true),
                  "\(what) allowed outside the scope with a retained poisoned state")
    }
}

let harness = Harness()
testOrderedDrain(harness)
testFailedDrainContinues(harness)
testRouterReadGuard(harness)
testSequenceOverflow(harness)
testScopePredicate(harness)
testCandidates(harness)
testBindingFacts(harness)
testStepDerivation(harness)
testAliasGuard(harness)
testTokenExpectation(harness)
testPoisonedLedger(harness)
testStateDisposition(harness)
testExecutionRejection(harness)
print("router overlap runtime pure tests: \(harness.passed) passed, \(harness.failed) failed")
exit(harness.failed == 0 ? 0 : 1)
