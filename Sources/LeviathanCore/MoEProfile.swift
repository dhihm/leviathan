// Phase 14 — bounded DIAGNOSTIC instrumentation for the routed MoE
// lowering (measurements only; NO optimization is implemented here).
//
// `moe-lower profile` compiles the STATIC diagnostic schedule E4_R32
// (the accepted phase-13 winner family) fresh from the SAME deterministic
// emitter, installs it through the INTERNAL trial path only (never the
// public production installer, never artifact approval or rebinding —
// a stale phase-13 artifact is never re-bound to a new binary), and
// reports two views of the SAME canonical raw-hello 32-step history on
// the ACTUAL compiled Forge SSA path:
//
//   1. NON-INTRUSIVE (production settings: segment batching ON,
//      vectorized loads ON, no top-5 diagnostic): per-token decode
//      wall/GPU plus the existing breakdown instrumentation. The GPU
//      per-kernel view is UNAVAILABLE here (deferred segments defer
//      attribution by design), which is why view 2 exists.
//   2. INTRUSIVE per-kernel replay: IDENTICAL E4 kernels, RMS/vectorized
//      settings unchanged — segmentBatching FALSE ONLY, so every kernel
//      runs in its own command buffer and the existing per-kernel GPU
//      start/end attribution applies. The replay is GATED FIRST (all
//      262144 logits per actual step: finite, exact greedy, relative
//      absdiff < 1e-4 vs the view-1 reference); a failed gate makes the
//      whole profile invalid, never merely "noisier numbers".
//
// All cumulative stats (kernel/matvec profiles, breakdown, dispatch
// counters) are snapshotted before/after each window and reported as
// DECODE-ONLY deltas — prefill is excluded from the decode summary but
// recorded. Kernel samples include per-command-buffer start/end effects,
// so the intrusive ranking is DIAGNOSTIC GPU attribution, NOT normal
// batched performance proof and NOT an occupancy claim. A matched
// instrumentation-off/on pair quantifies the timer overhead.
//
// Everything reuses the bounded shared helpers (LoweringSelect adapters,
// existing runner counters); no selector clone.

import Foundation
import Metal

/// Serializable facts of one diagnostic profile run. Pure data — every
/// field is either an identity, a measured number, or a checked boolean;
/// nothing here approves an artifact or changes the default serving path.
public struct MoEProfileReport: Codable, Equatable, Sendable {
    public static let schema = 1

    public let profileSchema: Int
    public let codegenVersion: String
    /// Identity binding (root captures file hashes externally too): the
    /// running executable, the actual GGUF, the measured hardware, the
    /// region graph, and the freshly emitted MSL source.
    public let executableSha256: String
    public let ggufSha256: String
    public let deviceName: String
    public let maxTotalThreadsPerThreadgroup: Int
    public let threadExecutionWidth: Int
    public let recommendedWorkingSetSize: Int
    public let regionGraphSha256: String
    public let schedule: MoESchedule
    /// SHA-256 of the freshly emitted MSL source (deterministic emitter —
    /// equals the accepted phase-13 artifact's sourceSha256 when emission
    /// rules are unchanged; verified against `expectedSourceSha256` when
    /// supplied, else reported for offline comparison).
    public let sourceSha256: String
    public let expectedSourceSha256: String?
    public let sourceMatchesExpected: Bool?
    public let diagnosticTrialOnly: Bool
    public let binding: BindingFacts
    public let reference: ReferenceFacts
    public let decode: DecodeView
    public let intrusive: IntrusiveView?
    public let overhead: OverheadCheck?

    public struct BindingFacts: Codable, Equatable, Sendable {
        public let layerCount: Int
        public let expertCount: Int
        public let topK: Int
        public let guStrideBytes: Int
        public let dnStrideBytes: Int
        public let gpuScratchBytes: Int
    }

    public struct ReferenceFacts: Codable, Equatable, Sendable {
        public let fixtureName: String
        public let promptTokenCount: Int
        /// Actual gated greedy steps (exactly genTokens — early EOS fails
        /// the whole profile; no shortening).
        public let steps: Int
        public let historyDigest: String
        public let vocabSize: Int
    }

    /// NON-INTRUSIVE decode view (production settings, breakdown timers
    /// ON). Explicit aggregate N/sum/median — never an inverse median
    /// pretending to be an aggregate.
    public struct DecodeView: Codable, Equatable, Sendable {
        public let decodeTokens: Int
        public let excludedPrefillForwards: Int
        public let sumWallMicros: Double
        public let medianWallMicros: Double
        public let aggregateDecodeTps: Double
        public let sumGpuMicros: Double
        public let wallsMicros: [Double]
        public let gpusMicros: [Double]
        public let encodeMicros: Double
        public let flushMicros: Double
        /// Host share of flush (flush overlaps GPU busy; clamped at 0).
        public let flushHostMicros: Double
        public let readCopyMicros: Double
        public let writeCopyMicros: Double
        public let residualMicros: Double
        public let submittedBufferDelta: Int
        public let dispatchDelta: Int
        public let stageDeltas: StageCounts
        public let mergeDelta: Int
        public let fallbackDelta: Int
    }

    public struct StageCounts: Codable, Equatable, Sendable {
        public let gu: Int
        public let act: Int
        public let dn: Int
    }

    /// INTRUSIVE per-kernel view (segmentBatching FALSE only, gated
    /// replay of the SAME history). Diagnostic GPU attribution — not
    /// normal batched performance proof, not an occupancy claim.
    public struct IntrusiveView: Codable, Equatable, Sendable {
        public let gateWorstRel: Double
        public let gateSteps: Int
        public let gateLogitsPerStep: Int
        public let decodeGpuMicrosDelta: Double
        public let kernels: [KernelStat]
        public let matvecs: [MatvecStat]
        /// The generated moegen GU/ACT/DN family, separated from the
        /// matvec view (generated expert kernels BYPASS matvec stats —
        /// reported separately so the expert total is never missing).
        public let generatedFamily: KernelStat
        public let kernelSumMicros: Double
        /// sum(kernel stats) must reproduce the measured decode GPU delta
        /// (same doubles, same accumulation — a mismatch means lost or
        /// double-counted attribution).
        public let kernelSumMatchesGpuDelta: Bool
        public let matvecSumMicros: Double
        /// The matvec view is a TENSOR grouping of engine-kernel time (a
        /// subset view); it must stay within the kernel total and never
        /// include the generated family.
        public let matvecWithinKernelSum: Bool
        public let stageDeltas: StageCounts
        public let mergeDelta: Int
        public let fallbackDelta: Int
        public let expectedStagePerKind: Int
        public let expectedMerge: Int
        public let countsExact: Bool
    }

    public struct KernelStat: Codable, Equatable, Sendable {
        public let name: String
        /// "moe-generated" (moegen_*) or "engine" (everything else).
        public let family: String
        public let gpuMicros: Double
        public let calls: Int
    }

    public struct MatvecStat: Codable, Equatable, Sendable {
        public let tensor: String
        public let gpuMicros: Double
        public let calls: Int
    }

    /// ONE sequential off/on pair — a PRELIMINARY diagnostic indicator
    /// of timer cost, NOT a causal overhead proof and NOT a performance
    /// claim; candidate fairness still requires the mandatory 3-paired
    /// x 31 matched selection protocol.
    public struct OverheadCheck: Codable, Equatable, Sendable {
        public let instrumentationOffSumWallMicros: Double
        public let instrumentationOnSumWallMicros: Double
        public let ratioOnOff: Double
    }
}

public enum MoELoweringProfile {

    /// The static diagnostic schedule: the accepted phase-13 winner
    /// (E4_R32) family, freshly compiled — never an artifact rebinding.
    public static func diagnosticSchedule() -> MoESchedule {
        MoESchedule(batch: 4, rowsPerGroup: 32)
    }

    public static func family(ofKernel name: String) -> String {
        name.hasPrefix("moegen_") ? "moe-generated" : "engine"
    }

    /// Decode-only diff of two kernel-profile snapshots: per-name GPU/call
    /// DELTAS (drop unchanged entries; every reported delta must be
    /// finite with nonnegative GPU and positive calls — the caller gates).
    public static func kernelDiff(
        before: [String: (gpu: Double, calls: Int)],
        after: [String: (gpu: Double, calls: Int)]) -> [MoEProfileReport.KernelStat] {
        var out = [MoEProfileReport.KernelStat]()
        for (name, a) in after.sorted(by: { $0.key < $1.key }) {
            let b = before[name] ?? (gpu: 0, calls: 0)
            let dGpu = a.gpu - b.gpu, dCalls = a.calls - b.calls
            if dCalls > 0 {
                out.append(.init(name: name, family: family(ofKernel: name),
                                 gpuMicros: dGpu, calls: dCalls))
            }
        }
        return out.sorted { $0.gpuMicros > $1.gpuMicros }
    }

    /// Bound checks on one profiled decode window's derived numbers:
    /// finite, positive walls/GPU, nonempty matched arrays, nonnegative
    /// breakdown, exact-32 window. Meaningful bounds, not mirrored logic.
    public static func decodeViewIsBounded(_ v: MoEProfileReport.DecodeView,
                                           expectedTokens: Int) -> Bool {
        v.decodeTokens == expectedTokens
            && v.wallsMicros.count == expectedTokens
            && v.gpusMicros.count == expectedTokens
            && v.wallsMicros.allSatisfy({ $0.isFinite && $0 > 0 })
            && v.gpusMicros.allSatisfy({ $0.isFinite && $0 >= 0 })
            && v.sumWallMicros.isFinite && v.sumWallMicros > 0
            && v.medianWallMicros.isFinite && v.medianWallMicros > 0
            && v.aggregateDecodeTps.isFinite && v.aggregateDecodeTps > 0
            && v.sumGpuMicros.isFinite && v.sumGpuMicros >= 0
            && [v.encodeMicros, v.flushMicros, v.flushHostMicros,
                v.readCopyMicros, v.writeCopyMicros, v.residualMicros]
                .allSatisfy({ $0.isFinite && $0 >= 0 })
            && v.fallbackDelta == 0
    }

    // MARK: - run

    /// One bounded diagnostic profile. Throws (no report written) on any
    /// identity, gate, or counter failure — a failed profile is never a
    /// partial success.
    public static func run(
        regionJSON: Data, ggufURL: URL, graph: GemmaGraphIR,
        runner: Gemma4Runner, promptTokens: [Int],
        isEndOfGeneration: (Int) -> Bool,
        expectedSourceSha256: String?,
        genTokens: Int = 32,
        fixtureName: String = "raw-hello") throws -> (report: MoEProfileReport, log: [String]) {

        var log = [String]()
        guard genTokens >= 2, genTokens <= 64, !promptTokens.isEmpty,
              promptTokens.count + genTokens <= 4096 else {
            throw MoELoweringError.invalidSchedule("profile bounds violated")
        }
        let region = try MoERegionIR.parse(regionJSON)
        let schedule = diagnosticSchedule()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MoELoweringError.hardwareUnavailable("Metal unavailable")
        }
        guard schedule.isLegal(region: region, hardware: runner.hardwareProfile()) else {
            throw MoELoweringError.invalidSchedule(
                "diagnostic schedule \(schedule) illegal on this hardware")
        }
        // Fresh compile + fresh binding; the emitted source is compared to
        // the expected phase-13 source SHA when supplied (fail-closed).
        let kernels = try MoEEmittedKernels.compile(region: region,
                                                    schedule: schedule, device: device)
        let sourceSha = MoEKernelEmitter.sourcesSha256([kernels.source])
        var sourceMatches: Bool? = nil
        if let expected = expectedSourceSha256 {
            guard moeIsLowercaseHex64(expected) else {
                throw MoELoweringError.invalidArtifact(
                    "expected source SHA malformed")
            }
            guard expected == sourceSha else {
                throw MoELoweringError.invalidArtifact(
                    "emitted source SHA \(sourceSha) != accepted phase-13 "
                    + "source \(expected) — emitter drifted; profile refused")
            }
            sourceMatches = true
        }
        let bindingFacts: MoEProfileReport.BindingFacts
        do {
            let b = try MoELoweringForge.bindRegion(
                region: region, file: try GGUFReader.read(ggufURL), device: device)
            bindingFacts = .init(layerCount: b.layerCount, expertCount: b.expertCount,
                                 topK: region.topK, guStrideBytes: b.guStride,
                                 dnStrideBytes: b.dnStride,
                                 gpuScratchBytes: b.scratchTotalBytes)
        }
        log.append("diagnostic schedule \(schedule.guKernelName) source sha \(sourceSha.prefix(16))…"
                   + (sourceMatches == true ? " (matches accepted phase-13 source)" : ""))

        // INTERNAL trial install only; always uninstalled on exit. This
        // claims NOTHING about artifact acceptance — it is the same
        // emitted kernel family, freshly compiled and bound.
        try runner.installMoeTrialLowering(region: region, kernels: kernels)
        defer { runner.uninstallMoeGeneratedLowering() }
        // Instrumentation starts OFF and the ORIGINAL value is restored on
        // exit — no inherited timer state leaks into or out of a profile.
        let breakdownWasOn = runner.breakdownInstrumentation
        runner.breakdownInstrumentation = false
        defer { runner.breakdownInstrumentation = breakdownWasOn }

        // Production-settings adapters (segment batching ON) and the
        // intrusive pair (segmentBatching OFF — the ONLY difference).
        let (normalMany, normalOne) = LoweringSelect.ssaForwardAdapters(graph: graph)
        let (intrusiveMany, intrusiveOne) = LoweringSelect.ssaForwardAdapters(
            graph: graph, segmentBatching: false)
        let layerCount = bindingFacts.layerCount
        let topK = region.topK
        let e = schedule.batch

        // ---- 1. canonical reference under the E4 baseline, SAME SSA path
        //         (normal settings); exactly genTokens steps, no EOS cut.
        //         Every logits vector is validated (exact vocab size, all
        //         finite) and every id range-checked BEFORE storing — a
        //         NaN reference could otherwise mask gate maxima, and a
        //         zero-sized vector would trap the argmax.
        let vocab = runner.config.vocabSize
        try LoweringSelect.ssaReset(runner)
        var refIds = [Int]()
        var refLogits = [[Float]]()
        func accept(_ am: Int, _ lg: [Float]) throws {
            guard lg.count == vocab, lg.allSatisfy({ $0.isFinite }),
                  am >= 0, am < vocab else {
                throw MoELoweringError.componentGateFailed(
                    "reference step \(refIds.count): logits nonfinite/wrong-size "
                    + "(\(lg.count) vs \(vocab)) or id \(am) out of range")
            }
            refLogits.append(lg); refIds.append(am)
        }
        let (am0, lg0) = try normalMany(runner, promptTokens)
        try accept(am0, lg0)
        var prev = am0
        while refIds.count < genTokens {
            if isEndOfGeneration(prev) {
                throw MoELoweringError.componentGateFailed(
                    "reference hit EOS at \(refIds.count)/\(genTokens) — profile requires full length")
            }
            let (am, lg) = try normalOne(runner, prev)
            try accept(am, lg)
            prev = am
        }
        var history = promptTokens
        history.append(contentsOf: refIds.dropLast())
        let refFacts = MoEProfileReport.ReferenceFacts(
            fixtureName: fixtureName, promptTokenCount: promptTokens.count,
            steps: refIds.count, historyDigest: GemmaForge.workloadDigest(history),
            vocabSize: vocab)
        log.append("reference [\(fixtureName)]: \(refIds.count) greedy steps, "
                   + "digest \(refFacts.historyDigest.prefix(12))…")

        // Shared fail-closed step gate (used by BOTH the timed normal
        // replay and the intrusive replay): full-size finite logits,
        // exact greedy argmax, relative absdiff < 1e-4 per step.
        func gateStep(_ lg: [Float], _ s: Int, _ worstSoFar: Double) throws -> Double {
            guard lg.count == refLogits[s].count, lg.count == vocab,
                  lg.allSatisfy({ $0.isFinite }) else {
                throw MoELoweringError.componentGateFailed(
                    "step \(s): nonfinite/wrong-size logits")
            }
            let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
            guard argmax == refIds[s] else {
                throw MoELoweringError.componentGateFailed(
                    "step \(s): greedy mismatch \(argmax) != \(refIds[s])")
            }
            var worst = worstSoFar
            for i in 0..<lg.count {
                worst = max(worst, abs(Double(lg[i] - refLogits[s][i]))
                            / max(1.0, abs(Double(refLogits[s][i]))))
            }
            guard worst < 1e-4 else {
                throw MoELoweringError.componentGateFailed(
                    String(format: "step %d outside 1e-4 (%.3e)", s, worst))
            }
            return worst
        }

        // ---- 2. NON-INTRUSIVE decode view (production settings, timers
        //         ON). The replay is GATED against the reference at every
        //         one of the 32 steps (timers do not excuse skipping the
        //         gate). Counters are snapshotted AFTER prefill: every
        //         reported delta below is DECODE-ONLY (31 steps); the
        //         prefill forward is recorded, never mixed in.
        runner.breakdownInstrumentation = true
        try LoweringSelect.ssaReset(runner)
        var worstNormal = 0.0
        let (_, lgP) = try normalMany(runner, promptTokens)
        worstNormal = try gateStep(lgP, 0, worstNormal)
        let guA = runner.moeGuDispatchCount, actA = runner.moeActDispatchCount
        let dnA = runner.moeDnDispatchCount, mergeA = runner.moeAxpyMergeCount
        let fbA = runner.moeResidentFallbackCount
        let bufsA = runner.submittedBufferTotal, dispA = runner.dispatchTotal
        runner.breakdownReset()                     // decode-only counters
        let gpuAfterPrefill = runner.totalGpuMicros
        var walls = [Double]()
        var gpus = [Double]()
        for s in 0..<(refIds.count - 1) {
            let t0 = DispatchTime.now()
            let (am, lg) = try normalOne(runner, refIds[s])
            walls.append(Double(DispatchTime.now().uptimeNanoseconds
                                - t0.uptimeNanoseconds) / 1000)
            gpus.append(runner.totalGpuMicros - gpuAfterPrefill
                        - gpus.reduce(0, +))
            worstNormal = try gateStep(lg, s + 1, worstNormal)
            guard am == refIds[s + 1] else {
                throw MoELoweringError.componentGateFailed(
                    "timed replay diverged at step \(s + 1)")
            }
        }
        let br = runner.breakdownCounters()
        let sumWall = walls.reduce(0, +)
        let sumGpu = gpus.reduce(0, +)
        let flushHost = max(0, br.flush - sumGpu)
        let decodeView = MoEProfileReport.DecodeView(
            decodeTokens: walls.count,
            excludedPrefillForwards: promptTokens.count,
            sumWallMicros: sumWall,
            medianWallMicros: LoweringSelect.median(walls),
            aggregateDecodeTps: Double(walls.count) / max(sumWall / 1e6, 1e-9),
            sumGpuMicros: sumGpu,
            wallsMicros: walls, gpusMicros: gpus,
            encodeMicros: br.encode, flushMicros: br.flush,
            flushHostMicros: flushHost,
            readCopyMicros: br.readCopy, writeCopyMicros: br.writeCopy,
            residualMicros: max(0, sumWall - br.encode - br.flush
                                - br.readCopy - br.writeCopy),
            submittedBufferDelta: runner.submittedBufferTotal - bufsA,
            dispatchDelta: runner.dispatchTotal - dispA,
            stageDeltas: .init(gu: runner.moeGuDispatchCount - guA,
                               act: runner.moeActDispatchCount - actA,
                               dn: runner.moeDnDispatchCount - dnA),
            mergeDelta: runner.moeAxpyMergeCount - mergeA,
            fallbackDelta: runner.moeResidentFallbackCount - fbA)
        guard decodeViewIsBounded(decodeView, expectedTokens: genTokens - 1) else {
            throw MoELoweringError.componentGateFailed(
                "non-intrusive decode window failed bounded checks")
        }
        // DECODE-ONLY expected counts: ceil(topK/E) chunk dispatches per
        // stage kind per layer per DECODE forward (31 of them — the
        // prefill forward is excluded), and topK ordered merges per layer
        // per decode forward. (E4/topK8/30 layers: 1860 per kind, 7440.)
        let chunksPerKind = (topK + e - 1) / e
        let wantStageDecode = chunksPerKind * layerCount * (genTokens - 1)
        let wantMergeDecode = topK * layerCount * (genTokens - 1)
        guard decodeView.stageDeltas.gu == wantStageDecode,
              decodeView.stageDeltas.act == wantStageDecode,
              decodeView.stageDeltas.dn == wantStageDecode,
              decodeView.mergeDelta == wantMergeDecode else {
            throw MoELoweringError.componentGateFailed(
                "decode counter deltas wrong: stages \(decodeView.stageDeltas)/"
                + "\(wantStageDecode), merges \(decodeView.mergeDelta)/\(wantMergeDecode)")
        }
        log.append(String(format: "decode (normal SSA, timers on, gated worst %.3e): N %d, sum %.0f us, aggregate %.3f tok/s, GPU %.0f us, stages %d/%d/%d, merges %d, fallbacks %d",
                          worstNormal, decodeView.decodeTokens,
                          decodeView.sumWallMicros, decodeView.aggregateDecodeTps,
                          decodeView.sumGpuMicros, decodeView.stageDeltas.gu,
                          decodeView.stageDeltas.act, decodeView.stageDeltas.dn,
                          decodeView.mergeDelta, decodeView.fallbackDelta))

        // ---- 3. INTRUSIVE per-kernel replay: segmentBatching FALSE only;
        //         the full-logit gate (all 32 steps) runs BEFORE any
        //         profile is valid, and the reported kernel/matvec stats
        //         and counters are DECODE-ONLY (snapshots after prefill).
        var intrusive: MoEProfileReport.IntrusiveView? = nil
        do {
            runner.breakdownInstrumentation = false
            try LoweringSelect.ssaReset(runner)
            var worst = 0.0
            var gated = 0
            let (_, lgIP) = try intrusiveMany(runner, promptTokens)
            worst = try gateStep(lgIP, 0, worst); gated += 1
            // Post-prefill snapshots: decode-only attribution below.
            let kMid = snapKernels(runner)
            let mMid = snapMatvecs(runner)
            let gpuMid = runner.totalGpuMicros
            let guI = runner.moeGuDispatchCount, actI = runner.moeActDispatchCount
            let dnI = runner.moeDnDispatchCount, mergeI = runner.moeAxpyMergeCount
            let fbI = runner.moeResidentFallbackCount
            for s in 0..<(refIds.count - 1) {
                let (_, lg) = try intrusiveOne(runner, refIds[s])
                worst = try gateStep(lg, s + 1, worst); gated += 1
            }
            let decodeGpuDelta = runner.totalGpuMicros - gpuMid
            let kStats = kernelDiff(before: kMid, after: snapKernels(runner))
            let mStats = matvecDiff(before: mMid, after: snapMatvecs(runner))
            let kernelSum = kStats.reduce(0) { $0 + $1.gpuMicros }
            let matvecSum = mStats.reduce(0) { $0 + $1.gpuMicros }
            let genStats = kStats.filter { $0.family == "moe-generated" }
            let genGpu = genStats.reduce(0) { $0 + $1.gpuMicros }
            let genCalls = genStats.reduce(0) { $0 + $1.calls }
            guard gated == genTokens, decodeGpuDelta.isFinite,
                  decodeGpuDelta > 0, kernelSum.isFinite, kernelSum > 0 else {
                throw MoELoweringError.componentGateFailed(
                    "intrusive view failed bounded checks")
            }
            // Same doubles, same accumulation order: attribution must
            // reproduce the measured GPU delta (tolerance for float sum
            // reassociation only), and the matvec view must remain a
            // subset of kernel time (never overlapping the generated
            // family, which bypasses matvec stats entirely).
            let kernelSumOK = abs(kernelSum - decodeGpuDelta)
                <= 1e-6 * max(1, decodeGpuDelta)
            let matvecOK = matvecSum <= kernelSum + 1e-6 * max(1, kernelSum)
            guard kernelSumOK, matvecOK else {
                throw MoELoweringError.componentGateFailed(
                    String(format: "attribution sums inconsistent: kernels %.1f vs GPU %.1f, matvec %.1f",
                           kernelSum, decodeGpuDelta, matvecSum))
            }
            // DECODE-ONLY expected counts (31 forwards; prefill excluded):
            // ceil(topK/E) dispatches per stage kind per layer per decode
            // forward, topK ordered merges per layer per decode forward.
            let perKind = chunksPerKind * layerCount * (genTokens - 1)
            let wantMerge = topK * layerCount * (genTokens - 1)
            let stages = MoEProfileReport.StageCounts(
                gu: runner.moeGuDispatchCount - guI,
                act: runner.moeActDispatchCount - actI,
                dn: runner.moeDnDispatchCount - dnI)
            let merges = runner.moeAxpyMergeCount - mergeI
            let fallbacks = runner.moeResidentFallbackCount - fbI
            let countsExact = stages.gu == perKind && stages.act == perKind
                && stages.dn == perKind && merges == wantMerge && fallbacks == 0
            guard countsExact else {
                throw MoELoweringError.componentGateFailed(
                    "intrusive counters wrong: stages \(stages)/\(perKind), merges \(merges)/\(wantMerge), fallbacks \(fallbacks)")
            }
            intrusive = MoEProfileReport.IntrusiveView(
                gateWorstRel: worst, gateSteps: gated,
                gateLogitsPerStep: refFacts.vocabSize,
                decodeGpuMicrosDelta: decodeGpuDelta,
                kernels: kStats, matvecs: mStats,
                generatedFamily: .init(name: "moegen_E\(e)_R\(schedule.rowsPerGroup)*",
                                       family: "moe-generated",
                                       gpuMicros: genGpu, calls: genCalls),
                kernelSumMicros: kernelSum, kernelSumMatchesGpuDelta: kernelSumOK,
                matvecSumMicros: matvecSum, matvecWithinKernelSum: matvecOK,
                stageDeltas: stages, mergeDelta: merges, fallbackDelta: fallbacks,
                expectedStagePerKind: perKind, expectedMerge: wantMerge,
                countsExact: countsExact)
            log.append(String(format: "intrusive replay (segmentBatching OFF only): gate %d steps worst %.3e; decode-only GPU %.0f us over %d kernel kinds; moegen family %.0f us (%d calls)",
                              gated, worst, decodeGpuDelta, kStats.count, genGpu, genCalls))
            for k in kStats.prefix(8) {
                log.append(String(format: "  %@ %@ %.1f us (%d calls)",
                                  k.family, k.name, k.gpuMicros, k.calls))
            }
        }

        // ---- 4. matched instrumentation OFF run (overhead of timers).
        var overhead: MoEProfileReport.OverheadCheck? = nil
        do {
            runner.breakdownInstrumentation = false
            try LoweringSelect.ssaReset(runner)
            _ = try normalMany(runner, promptTokens)
            var offWalls = [Double]()
            for s in 0..<(refIds.count - 1) {
                let t0 = DispatchTime.now()
                _ = try normalOne(runner, refIds[s])
                offWalls.append(Double(DispatchTime.now().uptimeNanoseconds
                                       - t0.uptimeNanoseconds) / 1000)
            }
            let offSum = offWalls.reduce(0, +)
            guard offSum.isFinite, offSum > 0, decodeView.sumWallMicros > 0 else {
                throw MoELoweringError.componentGateFailed(
                    "overhead window failed bounded checks")
            }
            overhead = .init(instrumentationOffSumWallMicros: offSum,
                             instrumentationOnSumWallMicros: decodeView.sumWallMicros,
                             ratioOnOff: decodeView.sumWallMicros / offSum)
            log.append(String(format: "timer cost PRELIMINARY off/on pair (not a causal proof): on %.0f us vs off %.0f us (ratio %.3f)",
                              decodeView.sumWallMicros, offSum, overhead!.ratioOnOff))
        }

        let hw = runner.hardwareProfile()
        let report = MoEProfileReport(
            profileSchema: MoEProfileReport.schema,
            codegenVersion: MoEKernelEmitter.codegenVersion,
            executableSha256: try Gemma4Runner.executableDigest(),
            ggufSha256: try GemmaForge.fileDigest(ggufURL),
            deviceName: hw.deviceName,
            maxTotalThreadsPerThreadgroup: hw.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: hw.threadExecutionWidth,
            recommendedWorkingSetSize: hw.recommendedMaxWorkingSetSize,
            regionGraphSha256: region.graphSha256,
            schedule: schedule,
            sourceSha256: sourceSha,
            expectedSourceSha256: expectedSourceSha256,
            sourceMatchesExpected: sourceMatches,
            diagnosticTrialOnly: true,
            binding: bindingFacts,
            reference: refFacts,
            decode: decodeView,
            intrusive: intrusive,
            overhead: overhead)
        return (report, log)
    }

    static func snapKernels(_ runner: Gemma4Runner) -> [String: (gpu: Double, calls: Int)] {
        var d = [String: (gpu: Double, calls: Int)]()
        for k in runner.kernelGpuProfile() { d[k.kernel] = (k.gpuMicros, k.calls) }
        return d
    }

    static func snapMatvecs(_ runner: Gemma4Runner) -> [String: (gpu: Double, calls: Int)] {
        var d = [String: (gpu: Double, calls: Int)]()
        for m in runner.matvecGpuProfile() { d[m.tensor] = (m.gpuMicros, m.calls) }
        return d
    }

    static func matvecDiff(
        before: [String: (gpu: Double, calls: Int)],
        after: [String: (gpu: Double, calls: Int)]) -> [MoEProfileReport.MatvecStat] {
        var out = [MoEProfileReport.MatvecStat]()
        for (tensor, a) in after {
            let b = before[tensor] ?? (gpu: 0, calls: 0)
            if a.calls - b.calls > 0 {
                out.append(.init(tensor: tensor, gpuMicros: a.gpu - b.gpu,
                                 calls: a.calls - b.calls))
            }
        }
        return out.sorted { $0.gpuMicros > $1.gpuMicros }
    }
}
