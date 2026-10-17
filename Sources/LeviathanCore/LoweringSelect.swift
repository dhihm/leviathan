// Phase 13 — family-agnostic bounded selection helpers.
//
// The small shared machinery every lowering family's select pipeline uses:
// reference-arm generation under the pass-7 incumbent configuration, the
// strict teacher-forced replay gate (full-logit 1e-4 + exact greedy +
// finite, per step), matched AB/BA paired timing, and the 2% policy
// arithmetic. Deliberately minimal — no per-family kernel/schedule logic
// lives here; families pass install/uninstall closures and a dispatch
// counter. The Q4/RMS pipelines predate this file and keep their own
// copies; new families must NOT clone a full selector (phase-13 review
// directive).

import Foundation

public enum LoweringSelect {
    /// One fixture's teacher-forced reference history + per-step logits.
    public struct RefArm {
        public var ids: [Int]
        public var logits: [[Float]]
    }

    /// Re-applies the incumbent reference configuration after a reset
    /// (pass-7 selection: VL RMSNorm + segment batching; the reset clears
    /// autoSegment, so it MUST be re-applied after every runner.reset()).
    public static func resetIncumbentConfig(_ runner: Gemma4Runner) throws {
        runner.reset()
        runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
        guard runner.segmentBatchingActive else {
            throw Q4LoweringError.componentGateFailed(
                "segment batching failed to re-engage after reset")
        }
    }

    /// Execution path adapters: one prefill (many tokens) and one decode
    /// step, both returning (argmax, full logits). All arms of a
    /// selection — reference, gates, timing — must use the SAME adapter.
    public typealias ForwardMany = (_ runner: Gemma4Runner, _ tokens: [Int])
        throws -> (argmax: Int, logits: [Float])
    public typealias ForwardOne = (_ runner: Gemma4Runner, _ token: Int)
        throws -> (argmax: Int, logits: [Float])

    /// The DIRECT runner forward path (phase <13 selections).
    public static func directForwardMany(_ runner: Gemma4Runner,
                                         _ tokens: [Int]) throws
            -> (argmax: Int, logits: [Float]) {
        let prefill = try runner.forward(tokenIds: tokens)
        return (prefill.argmaxToken, try runner.currentLogits())
    }
    public static func directForwardOne(_ runner: Gemma4Runner,
                                        _ token: Int) throws
            -> (argmax: Int, logits: [Float]) {
        let ev = try runner.forward(tokenIds: [token])
        return (ev.argmaxToken, try runner.currentLogits())
    }

    /// The ACTUAL compiled Forge SSA execution path (the serving engine's
    /// interpreter) under the unchanged pass-7 selected candidate. Both
    /// the incumbent and generated arms of a phase-13 selection run this.
    /// `segmentBatching: false` is the DIAGNOSTIC per-kernel-attribution
    /// variant (phase 14 profiling): the ONLY difference is submission
    /// granularity — every kernel runs in its own command buffer so the
    /// existing GPU start/end attribution applies per kernel.
    public static func ssaForwardAdapters(graph: GemmaGraphIR,
                                          segmentBatching: Bool = true)
            -> (ForwardMany, ForwardOne) {
        let candidate = GemmaForgeCandidate(segmentBatching: segmentBatching,
                                            top5Diagnostic: false,
                                            vectorizedLoads: true)
        let many: ForwardMany = { runner, tokens in
            let r = try GemmaForge.executeGraph(graph, runner: runner,
                                                 tokenIds: tokens,
                                                 candidate: candidate)
            return (r.argmax, r.logits)
        }
        let one: ForwardOne = { runner, token in
            let r = try GemmaForge.executeGraphContinue(graph, runner: runner,
                                                        token: token,
                                                        candidate: candidate)
            return (r.argmax, r.logits)
        }
        return (many, one)
    }

    /// The reset semantics per path: the direct path re-applies the
    /// pass-7 axis after every reset; the SSA path resets inside
    /// executeGraph (no axis to re-apply).
    public static func directReset(_ runner: Gemma4Runner) throws {
        try resetIncumbentConfig(runner)
    }
    /// SSA reset does NOT touch an installed lowering — candidate arms
    /// keep the generated kernels across resets; uninstalling happens
    /// ONLY at explicit incumbent/reference boundaries.
    public static func ssaReset(_ runner: Gemma4Runner) throws {
        runner.reset()
    }

    /// Reference greedy histories per fixture under the incumbent config
    /// (both path parameters; the reference arm NEVER has a lowering
    /// installed — the caller passes the incumbent reset).
    /// `requireFullLength` demands EXACTLY `genTokens` steps per fixture —
    /// a production reference that hits EOS early is a FAILED selection,
    /// never a silently shorter gate/timing target (diagnostic smoke runs
    /// may allow truncation).
    public static func generateReferenceArms(
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        genTokens: Int, isEndOfGeneration: (Int) -> Bool,
        forwardMany: ForwardMany, forwardOne: ForwardOne,
        reset: (Gemma4Runner) throws -> Void,
        requireFullLength: Bool = true,
        log: inout [String]) throws -> [String: RefArm] {
        var arms = [String: RefArm]()
        for fixture in fixtures {
            try reset(runner)
            var ids = [Int]()
            var logits = [[Float]]()
            let (am0, lg0) = try forwardMany(runner, fixture.tokens)
            logits.append(lg0)
            ids.append(am0)
            var prev = am0
            while ids.count < genTokens {
                if isEndOfGeneration(prev) { break }
                let (am, lg) = try forwardOne(runner, prev)
                logits.append(lg)
                ids.append(am)
                prev = am
            }
            if requireFullLength, ids.count < genTokens {
                throw Q4LoweringError.componentGateFailed(
                    "fixture \(fixture.name) produced only \(ids.count)/"
                    + "\(genTokens) reference steps (early EOS) — production "
                    + "selection requires full-length references")
            }
            arms[fixture.name] = RefArm(ids: ids, logits: logits)
            log.append("reference [\(fixture.name)]: \(ids.count) greedy steps")
        }
        return arms
    }

    /// Strict teacher-forced replay gate for one fixture under whatever
    /// lowering is currently installed: per step, full-logit
    /// absdiff/max(1,|ref|) < 1e-4, exact greedy argmax equality, all
    /// finite. `dispatchCount` lets the caller prove the generated path
    /// actually executed (and that fallbacks stayed at zero).
    public static func gateReplay(
        runner: Gemma4Runner, fixture: Q4LoweringForge.Q4Fixture,
        ref: RefArm, dispatchCount: () -> Int,
        fallbackCount: () -> Int = { 0 })
        -> (worstRel: Double, fail: String?, generatedDispatches: Int) {
        gateReplay(runner: runner, fixture: fixture, ref: ref,
                   forwardMany: directForwardMany, forwardOne: directForwardOne,
                   reset: directReset,
                   dispatchCount: dispatchCount, fallbackCount: fallbackCount)
    }

    /// Path-parameterized strict replay gate (same contract as above).
    public static func gateReplay(
        runner: Gemma4Runner, fixture: Q4LoweringForge.Q4Fixture,
        ref: RefArm,
        forwardMany: ForwardMany, forwardOne: ForwardOne,
        reset: (Gemma4Runner) throws -> Void,
        dispatchCount: () -> Int,
        fallbackCount: () -> Int)
        -> (worstRel: Double, fail: String?, generatedDispatches: Int) {
        guard !ref.ids.isEmpty, !ref.logits.isEmpty,
              ref.ids.count == ref.logits.count else {
            return (0, "empty reference arm", 0)
        }
        let before = dispatchCount()
        let fallbacksBefore = fallbackCount()
        do {
            try reset(runner)
            // STREAMING replay: compare each step's logits, then force the
            // reference token — fail-fast at the first violation, bounded
            // memory (one logits vector at a time).
            let (_, firstLogits) = try forwardMany(runner, fixture.tokens)
            var worst = 0.0
            var lg = firstLogits
            for s in 0..<ref.ids.count {
                let want = ref.logits[s]
                guard !lg.isEmpty, lg.count == want.count,
                      lg.allSatisfy({ $0.isFinite }),
                      want.allSatisfy({ $0.isFinite }) else {
                    return (worst, "step \(s): shape/nonfinite violation",
                            dispatchCount() - before)
                }
                let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                guard argmax == ref.ids[s] else {
                    return (worst,
                            "step \(s): argmax \(argmax) != reference \(ref.ids[s])",
                            dispatchCount() - before)
                }
                for i in 0..<lg.count {
                    worst = max(worst, abs(Double(lg[i] - want[i]))
                                / max(1.0, abs(Double(want[i]))))
                }
                if worst >= 1e-4 {
                    return (worst,
                            String(format: "step %d outside 1e-4 (worst %.3e)",
                                   s, worst),
                            dispatchCount() - before)
                }
                if s + 1 < ref.ids.count {
                    let (_, next) = try forwardOne(runner, ref.ids[s])
                    lg = next
                }
            }
            guard fallbackCount() == fallbacksBefore else {
                return (worst, "generated path fell back mid-replay",
                        dispatchCount() - before)
            }
            return (worst, nil, dispatchCount() - before)
        } catch {
            return (0, "replay threw: \(error)", dispatchCount() - before)
        }
    }

    /// One matched timing arm over the fixtures (same binary/model/
    /// prompt/context; EOS-truncated histories time fewer tokens at true
    /// positions — recorded, never clamped). THROWS on any forward
    /// failure: a failed run must never be timed.
    public static func timedArm(
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        refArms: [String: RefArm], timedTokens: Int) throws -> [Q4TimingArmFixture] {
        try timedArm(runner: runner, fixtures: fixtures, refArms: refArms,
                     timedTokens: timedTokens,
                     forwardMany: directForwardMany, forwardOne: directForwardOne,
                     reset: directReset)
    }

    /// Path-parameterized matched timing arm.
    public static func timedArm(
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        refArms: [String: RefArm], timedTokens: Int,
        forwardMany: ForwardMany, forwardOne: ForwardOne,
        reset: (Gemma4Runner) throws -> Void) throws -> [Q4TimingArmFixture] {
        var arms = [Q4TimingArmFixture]()
        for fixture in fixtures {
            let ref = refArms[fixture.name]!
            let steps = ref.ids.count
            guard steps >= 2 else {
                throw Q4LoweringError.componentGateFailed(
                    "fixture \(fixture.name) produced only \(steps) gated steps")
            }
            let effTimed = max(1, min(timedTokens, steps - 1))
            try reset(runner)
            _ = try forwardMany(runner, fixture.tokens)
            var idx = 0
            while idx < steps - effTimed - 1 {
                _ = try forwardOne(runner, ref.ids[idx])
                idx += 1
            }
            var w = [Double]()
            var g = [Double]()
            for k in 0..<effTimed {
                let tok = ref.ids[min(idx + k, steps - 2)]
                let gpuBefore = runner.totalGpuMicros
                let t0 = DispatchTime.now()
                _ = try forwardOne(runner, tok)
                w.append(Double(DispatchTime.now().uptimeNanoseconds
                                - t0.uptimeNanoseconds) / 1000)
                g.append(runner.totalGpuMicros - gpuBefore)
            }
            arms.append(Q4TimingArmFixture(walls: w, gpus: g,
                                           startIdx: idx, timedCount: effTimed))
        }
        return arms
    }

    public static func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2 : s[s.count/2]
    }

    public static func armFixtureMedians(_ arm: [Q4TimingArmFixture]) -> [Double] {
        arm.map { median($0.walls) }
    }

    /// 3 PAIRED repetitions, alternating order (AB/BA/AB) so drift hits
    /// both arms equally. Each candidate keeps its OWN paired incumbent
    /// metrics — selection compares candMed against THIS candidate's
    /// incMed, never another candidate's arm.
    public static func pairedTiming(
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        refArms: [String: RefArm], timedTokens: Int,
        install: () throws -> Void, uninstall: () -> Void)
        throws -> (reps: [Q4TimingRepetition], candMed: Double, incMed: Double,
                   wallMedPerFixture: [Double], gpuMedPerFixture: [Double]) {
        try pairedTiming(runner: runner, fixtures: fixtures, refArms: refArms,
                         timedTokens: timedTokens,
                         forwardMany: directForwardMany, forwardOne: directForwardOne,
                         reset: directReset,
                         install: install, uninstall: uninstall)
    }

    /// Path-parameterized paired timing: BOTH arms always use the SAME
    /// adapter — never mix execution paths within a selection.
    public static func pairedTiming(
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        refArms: [String: RefArm], timedTokens: Int,
        forwardMany: ForwardMany, forwardOne: ForwardOne,
        reset: (Gemma4Runner) throws -> Void,
        install: () throws -> Void, uninstall: () -> Void)
        throws -> (reps: [Q4TimingRepetition], candMed: Double, incMed: Double,
                   wallMedPerFixture: [Double], gpuMedPerFixture: [Double]) {
        func candidateArm() throws -> [Q4TimingArmFixture] {
            try install()
            defer { uninstall() }
            return try timedArm(runner: runner, fixtures: fixtures,
                                refArms: refArms, timedTokens: timedTokens,
                                forwardMany: forwardMany, forwardOne: forwardOne,
                                reset: reset)
        }
        func incumbentArm() throws -> [Q4TimingArmFixture] {
            uninstall()
            return try timedArm(runner: runner, fixtures: fixtures,
                                refArms: refArms, timedTokens: timedTokens,
                                forwardMany: forwardMany, forwardOne: forwardOne,
                                reset: reset)
        }
        var reps = [Q4TimingRepetition]()
        for rep in 0..<3 {
            let candidateFirst = rep % 2 == 1
            let (cand, inc): ([Q4TimingArmFixture], [Q4TimingArmFixture])
            if candidateFirst {
                cand = try candidateArm()
                inc = try incumbentArm()
            } else {
                inc = try incumbentArm()
                cand = try candidateArm()
            }
            reps.append(Q4TimingRepetition(candidateFirst: candidateFirst,
                                           candidate: cand, incumbent: inc))
        }
        let candSums = reps.map { armFixtureMedians($0.candidate).reduce(0, +) }
        let incSums = reps.map { armFixtureMedians($0.incumbent).reduce(0, +) }
        // Individually finite samples can still sum to infinity; a
        // non-finite or non-positive arm median must never reach the
        // policy comparison.
        guard candSums.allSatisfy({ $0.isFinite && $0 > 0 }),
              incSums.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw Q4LoweringError.componentGateFailed(
                "paired timing produced non-finite/non-positive arm medians "
                + "(cand \(candSums), inc \(incSums))")
        }
        let wallMed = (0..<fixtures.count).map { f in
            median(reps.map { armFixtureMedians($0.candidate)[f] })
        }
        let gpuMed = (0..<fixtures.count).map { f in
            median(reps.map { median($0.candidate[f].gpus) })
        }
        return (reps, median(candSums), median(incSums), wallMed, gpuMed)
    }

    /// The documented 2% policy: selected only when the candidate beats
    /// its own paired incumbent median by more than GemmaForge.noiseRelWall.
    public static func policySelects(candMed: Double, incMed: Double) -> Bool {
        candMed < incMed * (1 - GemmaForge.noiseRelWall)
    }
}
