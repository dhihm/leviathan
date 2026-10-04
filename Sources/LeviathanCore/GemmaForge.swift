// LeviathanForge pipeline for the Gemma 4 quantized graph: captured
// model.py graph → validation against the real GGUF → bounded legal
// candidates → correctness gate (graph executor vs the independently
// validated per-kernel reference path, and raw-logit tolerance) →
// measured GPU-time selection → reusable plan cache keyed on graph +
// the cached model-file SHA-256 (hashed once, reused, refreshed only
// explicitly; see ModelDigestCache) + hardware + toolchain + workload. No agent, no LLM at
// runtime; anything outside the supported op set is rejected at
// validation. The graph executor drives the runner's kernel primitives
// from the validated IR — it is not a call into the fixed hand-written
// forward.

import Foundation
import CryptoKit
import Metal

public enum GemmaForgeError: Error, CustomStringConvertible, Equatable {
    case invalidGraph(String)
    case noValidCandidate([String])
    case invalidPlan(String)
    case executionFailed(String)

    public var description: String {
        switch self {
        case .invalidGraph(let m): return "gemma forge graph invalid: \(m)"
        case .noValidCandidate(let r): return "no valid candidate: \(r)"
        case .invalidPlan(let m): return "gemma plan invalid: \(m)"
        case .executionFailed(let m): return "execution failed: \(m)"
        }
    }
}

public struct GemmaForgeCandidate: Codable, Equatable, Hashable, Sendable {
    /// Batch GPU-only kernel runs into one command buffer per run
    /// (numerically identical order; only submission granularity changes).
    public let segmentBatching: Bool
    /// Compute the full top-5 logit diagnostic (CPU cost) or argmax only.
    public let top5Diagnostic: Bool
    /// Dispatch the tiled cooperative Q4_0 matvec kernel (hardware-derived
    /// threadgroup width). NOT bit-identical: the threadgroup reduction
    /// reassociates the addition order — correctness is gated against the
    /// FP64 oracle and full-logit/greedy equality before selection.
    public let matvecTiledQ4: Bool
    /// Tiled cooperative Q6_K matvec (tied embedding / LM head). Same
    /// non-bit-identical caveat and gates as matvecTiledQ4.
    public let matvecTiledQ6: Bool
    /// Tiled cooperative RMSNorm (measured 44% GPU-time hotspot in the
    /// per-kernel profile). NOT bit-identical (reassociated sum-of-
    /// squares); gated against a Double oracle + full Forge gates.
    public let rmsnormTiled: Bool
    /// Tiled cooperative F32 matvec (the MoE router ffn_gate_inp; the
    /// scalar one-thread-per-row kernel is a measured GPU hotspot).
    /// Same non-bit-identical caveat and gates as matvecTiledQ4.
    public let matvecTiledF32: Bool
    /// Dispatch the vectorized-load plain RMSNorm when the production
    /// alignment/width guards hold: float4 LOADS ONLY, with an
    /// arithmetic sequence identical to the original kernel. The test
    /// suite asserts EXACT bit equality on hardware for the tested
    /// geometries (256/2112/2816, weighted/scale-free, in/out-of-place)
    /// and the measured cross-path logit rel in the Forge was 0.0e+00
    /// on this configuration; bit equality on hardware is a measured
    /// property of the compiled pair, not a proof for all inputs.
    /// (A vectorized-load Q4 matvec was measured ~2.1x per synthetic
    /// call but its compiled outputs differed from the original kernel
    /// — cause not established — and it is rejected from legal
    /// dispatch.)
    public let vectorizedLoads: Bool

    /// GPU-resident KV cache append: copy the K/V words into the cache
    /// buffers with a bit-preserving uint32 GPU copy kernel encoded in
    /// the open compute segment instead of reading them back to the CPU
    /// and memcpy-ing (which forces a segment flush + CPU roundtrip per
    /// layer). Bit-exact by construction (raw word copy, no arithmetic);
    /// the kernel bounds-guards gid so no out-of-range word is touched.
    public let gpuCacheAppend: Bool

    /// PRODUCTION parallel-attention axis (default false): serve eligible
    /// attention geometry with the original 64-lane par64 kernel through
    /// the runner's production parallel-attention pipeline (numerically
    /// unmodified kernel; same ordered compute-segment path). Ineligible
    /// geometry safely falls back to the incumbent g4_attn and counts a
    /// production fallback. Explicit Codable key — a pre-axis artifact
    /// missing this field is rejected, never silently defaulted.
    public let parallelAttention: Bool

    /// Exact-order cooperative RMSNorm (default false): cooperative
    /// staging into dynamic threadgroup memory, lane-0 sum-of-squares in
    /// the EXACT incumbent kernel order (no reassociated tree reduction),
    /// and a parallel per-element apply with the identical x*inv*(scale)
    /// association. Bit equality against the original kernel on hardware
    /// is asserted by the test suite for the tested geometries (256/704/
    /// 2816, weighted/unweighted, in/out-of-place, multivector); the full
    /// Forge gates decide selection, unchanged and fail-closed. Never
    /// combined with rmsnormTiled in any legal candidate. Explicit
    /// Codable key — an artifact missing this field is rejected, never
    /// silently defaulted.
    public let rmsnormCooperative: Bool

    /// Intra-block SIMD-reduction Q6_K matvec (default false): one
    /// 32-lane SIMD group per output row, lane-local accumulation over
    /// the same four dequantized products per half-block as the plain
    /// kernel, and a simd_sum group reduction written by lane 0. The
    /// simd_sum reduction reassociates the floating-point addition
    /// order, so bit-exactness is NOT claimed; qualification is against
    /// the strict FP64 oracle and the full Forge gates, unchanged and
    /// fail-closed. Never combined with matvecTiledQ6 in any legal
    /// candidate. Explicit Codable key — an artifact missing this field
    /// is rejected, never silently defaulted.
    public let matvecCooperativeQ6: Bool

    /// Reusable RMS statistics (default false): the first RMSNorm of a
    /// compatible SSA statistics group computes the inverse statistic
    /// once (stored GPU-side in a tiny persistent buffer) and later
    /// RMSNorms with the SAME live input version reuse it with the
    /// identical per-element arithmetic. Explicit Codable key — an
    /// artifact missing this field is rejected, never silently
    /// defaulted.
    public let rmsStatisticsReuse: Bool

    /// GPU router residency (default false): moves the second per-element
    /// router scale multiply onto a precise-compiled GPU kernel and arms
    /// the resident-expert output reset wrapper. Explicit Codable key —
    /// an artifact missing this field is rejected, never silently
    /// defaulted.
    public let routerGPUResidency: Bool

    /// Router F32 matvec launch-geometry divisor (default 0). 0 keeps the
    /// original single-group geometry (GPU scale/zero only); a nonzero
    /// value in {1, 2, 4, 8} divides the scalar router matvec's
    /// threadgroup width, changing only threadgroup geometry — never the
    /// kernel, bindings, or per-row arithmetic. Explicit Codable key.
    public let routerLaunchDivisor: Int

    public init(segmentBatching: Bool, top5Diagnostic: Bool,
                matvecTiledQ4: Bool = false, matvecTiledQ6: Bool = false,
                rmsnormTiled: Bool = false, matvecTiledF32: Bool = false,
                vectorizedLoads: Bool = false,
                gpuCacheAppend: Bool = false,
                parallelAttention: Bool = false,
                rmsnormCooperative: Bool = false,
                matvecCooperativeQ6: Bool = false,
                rmsStatisticsReuse: Bool = false,
                routerGPUResidency: Bool = false,
                routerLaunchDivisor: Int = 0) {
        self.segmentBatching = segmentBatching
        self.top5Diagnostic = top5Diagnostic
        self.matvecTiledQ4 = matvecTiledQ4
        self.matvecTiledQ6 = matvecTiledQ6
        self.rmsnormTiled = rmsnormTiled
        self.matvecTiledF32 = matvecTiledF32
        self.vectorizedLoads = vectorizedLoads
        self.gpuCacheAppend = gpuCacheAppend
        self.parallelAttention = parallelAttention
        self.rmsnormCooperative = rmsnormCooperative
        self.matvecCooperativeQ6 = matvecCooperativeQ6
        self.rmsStatisticsReuse = rmsStatisticsReuse
        self.routerGPUResidency = routerGPUResidency
        self.routerLaunchDivisor = routerLaunchDivisor
    }

    private enum CodingKeys: String, CodingKey {
        case segmentBatching, top5Diagnostic, matvecTiledQ4, matvecTiledQ6
        case rmsnormTiled, matvecTiledF32, vectorizedLoads, gpuCacheAppend
        case parallelAttention, rmsnormCooperative, matvecCooperativeQ6,
             rmsStatisticsReuse, routerGPUResidency, routerLaunchDivisor
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Explicit identity: an artifact persisted before this axis
        // existed is REJECTED rather than silently interpreted — an old
        // engine identity must not be replayed as if it had decided the
        // new flag.
        guard let gpuCacheAppend = try c.decodeIfPresent(Bool.self, forKey: .gpuCacheAppend) else {
            throw DecodingError.keyNotFound(CodingKeys.gpuCacheAppend, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "gpuCacheAppend missing (pre-axis artifact)"))
        }
        guard let parallelAttention = try c.decodeIfPresent(Bool.self, forKey: .parallelAttention) else {
            throw DecodingError.keyNotFound(CodingKeys.parallelAttention, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "parallelAttention missing (pre-axis artifact)"))
        }
        guard let rmsnormCooperative = try c.decodeIfPresent(Bool.self, forKey: .rmsnormCooperative) else {
            throw DecodingError.keyNotFound(CodingKeys.rmsnormCooperative, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "rmsnormCooperative missing (pre-axis artifact)"))
        }
        guard let matvecCooperativeQ6 = try c.decodeIfPresent(Bool.self, forKey: .matvecCooperativeQ6) else {
            throw DecodingError.keyNotFound(CodingKeys.matvecCooperativeQ6, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "matvecCooperativeQ6 missing (pre-axis artifact)"))
        }
        guard let rmsStatisticsReuse = try c.decodeIfPresent(Bool.self, forKey: .rmsStatisticsReuse) else {
            throw DecodingError.keyNotFound(CodingKeys.rmsStatisticsReuse, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "rmsStatisticsReuse missing (pre-axis artifact)"))
        }
        guard let routerGPUResidency = try c.decodeIfPresent(Bool.self, forKey: .routerGPUResidency) else {
            throw DecodingError.keyNotFound(CodingKeys.routerGPUResidency, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "routerGPUResidency missing (pre-axis artifact)"))
        }
        guard let routerLaunchDivisor = try c.decodeIfPresent(Int.self, forKey: .routerLaunchDivisor) else {
            throw DecodingError.keyNotFound(CodingKeys.routerLaunchDivisor, DecodingError.Context(
                codingPath: c.codingPath, debugDescription: "routerLaunchDivisor missing (pre-axis artifact)"))
        }
        self.segmentBatching = try c.decode(Bool.self, forKey: .segmentBatching)
        self.top5Diagnostic = try c.decode(Bool.self, forKey: .top5Diagnostic)
        self.matvecTiledQ4 = try c.decode(Bool.self, forKey: .matvecTiledQ4)
        self.matvecTiledQ6 = try c.decode(Bool.self, forKey: .matvecTiledQ6)
        self.rmsnormTiled = try c.decode(Bool.self, forKey: .rmsnormTiled)
        self.matvecTiledF32 = try c.decode(Bool.self, forKey: .matvecTiledF32)
        self.vectorizedLoads = try c.decode(Bool.self, forKey: .vectorizedLoads)
        self.gpuCacheAppend = gpuCacheAppend
        self.parallelAttention = parallelAttention
        self.rmsnormCooperative = rmsnormCooperative
        self.matvecCooperativeQ6 = matvecCooperativeQ6
        self.rmsStatisticsReuse = rmsStatisticsReuse
        self.routerGPUResidency = routerGPUResidency
        self.routerLaunchDivisor = routerLaunchDivisor
    }
}

public struct GemmaForgeCandidateResult: Codable, Equatable, Sendable {
    public let candidate: GemmaForgeCandidate
    public let correct: Bool
    /// SELECTION OBJECTIVE: median end-to-end wall time per timed token
    /// (forward + CPU router + argmax + necessary readback; the post-hoc
    /// correctness comparison is excluded from the timed region). GPU-only
    /// time misses CPU launch/driver overhead — the dominant cost of the
    /// per-kernel submission path — so wall time selects.
    public let medianWallMicrosPerToken: Double?
    /// Reported separately: median GPU-busy time per timed token.
    public let medianGpuMicrosPerToken: Double?
    /// Median dispatched kernels per timed token (kernels, not buffers).
    public let kernelsPerToken: Double?
    /// Median submitted command buffers per timed token — makes the
    /// batching effect of a candidate directly observable.
    public let buffersPerToken: Double?
    public let note: String

    public init(candidate: GemmaForgeCandidate, correct: Bool,
         medianWallMicrosPerToken: Double? = nil,
         medianGpuMicrosPerToken: Double?,
         kernelsPerToken: Double? = nil,
         buffersPerToken: Double? = nil, note: String) {
        self.candidate = candidate
        self.correct = correct
        self.medianWallMicrosPerToken = medianWallMicrosPerToken
        self.medianGpuMicrosPerToken = medianGpuMicrosPerToken
        self.kernelsPerToken = kernelsPerToken
        self.buffersPerToken = buffersPerToken
        self.note = note
    }
}

/// Model-configuration fingerprint captured from the preparing
/// runner's Gemma4Config. Equality of this fingerprint plus structural
/// graph equality are the CHEAP reuse gates; they never replace the
/// full one-time validation performed during preparation.
public struct GemmaConfigFingerprint: Equatable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let layerCount: Int
    public let expertCount: Int
    public let expertUsedCount: Int
    public let expertIntermediate: Int
    public let sharedIntermediate: Int
    public let maxContext: Int
    public let globalLayers: Set<Int>

    public init(_ c: Gemma4Config) {
        self.vocabSize = c.vocabSize
        self.hiddenSize = c.hiddenSize
        self.layerCount = c.layerCount
        self.expertCount = c.expertCount
        self.expertUsedCount = c.topKExperts
        self.expertIntermediate = c.expertIntermediate
        self.sharedIntermediate = c.sharedIntermediate
        self.maxContext = c.maxContext
        self.globalLayers = Set(c.globalLayers)
    }
}

/// Immutable PREPARED execution: the canonical SSA lowering of ONE
/// graph validated against the mapped GGUF at prepare time, bound to
/// ONE exact retained runner object with the SSA value table
/// precomputed once.
///
/// All expensive qualification — GemmaGraph.validate against the real
/// GGUF, canonical lowering, SSA dataflow validation — happens ONCE in
/// `prepare`. Fast reuse performs NO hashing and NO re-lowering: it
/// compares the runner by object identity (the runner is RETAINED so a
/// freed-and-reallocated object cannot reuse the address), the graph
/// by structural equality with the validated immutable copy, and the
/// config by fingerprint. A mismatch is a hard rejection before any
/// GPU work; a prepared execution can never run on a different runner,
/// even one with an identical configuration.
///
/// NOT Sendable: it retains a non-Sendable runner. The runner is
/// already serial per request; removing the conformance rather than
/// declaring an unsafe unchecked conformance keeps the compiler from
/// blessing a data race that does not exist in the intended use.
/// Construction is private — the only path to an instance is `prepare`,
/// which throws on any failure, so no partially qualified
/// representation can exist.
public struct GemmaPreparedExecution {
    /// The validated canonical SSA schedule this preparation executes.
    public let ssa: GemmaSSAIR
    /// The EXACT runner object this preparation is bound to, retained
    /// for its lifetime so ObjectIdentifier reuse after deallocation is
    /// impossible — identity comparison is against the live object.
    let runner: Gemma4Runner
    /// Immutable copy of the validated graph; reuse compares structural
    /// equality against this copy (Equatable, no digest hashing).
    let boundGraph: GemmaGraphIR
    /// Config fingerprint captured at prepare time.
    let configFingerprint: GemmaConfigFingerprint
    /// Precomputed immutable value table (id -> value), built once.
    public let byID: [String: GemmaSSAIR.Value]
    /// Read-only reusable RMS statistics plan derived ONCE at prepare
    /// time from the validated SSA and the runner's epsilon — never
    /// re-derived per token.
    public let statisticsPlan: RMSStatisticsPlan
    /// Lightweight composite-install binding captured at prepare time:
    /// the installed MoE lowering's artifact digest and the installed Q4
    /// launch width. assertBound compares these cheap values on every
    /// reuse and rejects a CHANGED install state BEFORE any GPU work —
    /// even for the same runner object and the same graph. The
    /// per-call candidate axis flags (e.g. parallel attention) vary per
    /// call and are deliberately NOT part of this binding; nothing here
    /// hashes the executable or kernel source per token.
    let boundMoeArtifactDigest: String?
    let boundQ4LaunchWidth: Int?

    private init(ssa: GemmaSSAIR, runner: Gemma4Runner, graph: GemmaGraphIR,
                 configFingerprint: GemmaConfigFingerprint,
                 byID: [String: GemmaSSAIR.Value],
                 statisticsPlan: RMSStatisticsPlan) {
        self.ssa = ssa
        self.runner = runner
        self.boundGraph = graph
        self.configFingerprint = configFingerprint
        self.byID = byID
        self.statisticsPlan = statisticsPlan
        self.boundMoeArtifactDigest = runner.moeGenerated?.artifactDigest
        self.boundQ4LaunchWidth = runner.q4LaunchWidth
    }

    /// One-time preparation: validates the graph against the runner's
    /// mapped GGUF + config, lowers canonically, dataflow-validates the
    /// SSA, and captures the binding (exact runner object, immutable
    /// graph copy, config fingerprint) plus the precomputed value
    /// table. Throws on any failure.
    public static func prepare(graph: GemmaGraphIR,
                               runner: Gemma4Runner) throws -> GemmaPreparedExecution {
        try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
        let ssa = try GemmaSSA.lower(graph)
        try GemmaSSA.validate(ssa, config: graph.config)
        let statisticsPlan = try RMSStatisticsPlan.analyze(
            ssa: ssa, epsilon: runner.config.rmsNormEps)
        return GemmaPreparedExecution(
            ssa: ssa, runner: runner, graph: graph,
            configFingerprint: GemmaConfigFingerprint(runner.config),
            byID: valueTable(ssa), statisticsPlan: statisticsPlan)
    }

    /// Builds the id -> value dictionary for a schedule (shared by the
    /// prepared path and the ordinary lowering paths so both dispatch
    /// from identical operand tables).
    public static func valueTable(_ ssa: GemmaSSAIR) -> [String: GemmaSSAIR.Value] {
        var table = [String: GemmaSSAIR.Value]()
        for v in ssa.values { table[v.id] = v }
        return table
    }

    /// Fast reuse gate: runner object identity + config fingerprint +
    /// structural graph equality. No engine-identity hashing, no graph
    /// digesting, no lowering. Rejected before any GPU work on mismatch.
    func assertBound(to runner: Gemma4Runner, graph: GemmaGraphIR) throws {
        guard runner === self.runner else {
            throw GemmaForgeError.invalidPlan(
                "prepared execution is bound to a different runner")
        }
        guard GemmaConfigFingerprint(runner.config) == configFingerprint else {
            throw GemmaForgeError.invalidPlan(
                "prepared execution model configuration mismatch")
        }
        guard graph == boundGraph else {
            throw GemmaForgeError.invalidPlan(
                "prepared execution graph mismatch")
        }
        // Cheap composite-state binding: reject a changed install state
        // (MoE artifact digest / Q4 launch width) before any GPU work,
        // even for the same owner runner and the same graph. These are
        // lightweight value comparisons only.
        guard runner.moeGenerated?.artifactDigest == boundMoeArtifactDigest,
              runner.q4LaunchWidth == boundQ4LaunchWidth else {
            throw GemmaForgeError.invalidPlan(
                "prepared execution composite install state changed — re-prepare required")
        }
    }
}

public struct GemmaForgePlan: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let graphDigest: String
    /// Actual SHA-256 of the model file from the shared ModelDigestCache
    /// (computed once from the bytes, reused without re-reading, refreshed
    /// only explicitly). Historical, not a fresh per-run verification.
    public let ggufDigest: String
    /// Model-file digest policy (schema 3). Required on decode and covered
    /// by planDigest; only cached-explicit-refresh-v1 is representable.
    public let modelFileVerification: ModelFileVerificationPolicy
    /// Pass-engine identity (graph-executor version + compiled Metal
    /// pipeline set): changing kernels or the executor invalidates the
    /// cache even without a toolchain change.
    public let engineDigest: String
    public let hardware: ForgeHardwareProfile
    public let toolchain: String
    public let workloadDigest: String
    public let promptTokenCount: Int
    public let decodeTokens: Int
    public let selected: GemmaForgeCandidate
    public let selectedMedianWallMicrosPerToken: Double
    public let selectedMedianGpuMicrosPerToken: Double
    /// Non-nil when the plan was forged in SSD/streaming expert mode:
    /// the byte budget of the bounded expert cache (an engine identity
    /// input — a different budget re-forges). nil = resident weights.
    public let expertCacheBudgetBytes: Int?
    public let results: [GemmaForgeCandidateResult]
    /// The FULL lowered executable schedule (typed SSA instructions with
    /// operands and buffer-binding value names) saved with the plan —
    /// the runtime executes these artifact operands, not a re-derivation.
    public let schedule: GemmaSSAIR?
    /// SHA-256 of the canonical schedule JSON (must equal
    /// GemmaSSA.digest(lower(graph)) on reuse).
    public let scheduleDigest: String
    /// Full-workload gate record (set when the forge verified the plan's
    /// selection over a longer --gen-tokens generation than the 4-token
    /// measurement workload): the gate length and the measured rejection
    /// reasons of every candidate that failed it. nil on plans forged
    /// before this field existed or when no long gate ran.
    public let fullGateTokens: Int?
    public let fullGateRejections: [String]?
    public let planDigest: String

    init(schemaVersion: Int, graphDigest: String, ggufDigest: String,
         engineDigest: String, hardware: ForgeHardwareProfile,
         toolchain: String, workloadDigest: String,
         promptTokenCount: Int, decodeTokens: Int,
         selected: GemmaForgeCandidate,
         selectedMedianWallMicrosPerToken: Double,
         selectedMedianGpuMicrosPerToken: Double,
         results: [GemmaForgeCandidateResult], schedule: GemmaSSAIR?,
         scheduleDigest: String, planDigest: String) {
        self.init(schemaVersion: schemaVersion, graphDigest: graphDigest,
                  ggufDigest: ggufDigest, engineDigest: engineDigest,
                  hardware: hardware, toolchain: toolchain,
                  workloadDigest: workloadDigest,
                  promptTokenCount: promptTokenCount, decodeTokens: decodeTokens,
                  selected: selected,
                  selectedMedianWallMicrosPerToken: selectedMedianWallMicrosPerToken,
                  selectedMedianGpuMicrosPerToken: selectedMedianGpuMicrosPerToken,
                  expertCacheBudgetBytes: nil, results: results,
                  schedule: schedule, scheduleDigest: scheduleDigest,
                  planDigest: planDigest)
    }

    init(schemaVersion: Int, graphDigest: String, ggufDigest: String,
         engineDigest: String, hardware: ForgeHardwareProfile,
         toolchain: String, workloadDigest: String,
         promptTokenCount: Int, decodeTokens: Int,
         selected: GemmaForgeCandidate,
         selectedMedianWallMicrosPerToken: Double,
         selectedMedianGpuMicrosPerToken: Double,
         expertCacheBudgetBytes: Int?,
         results: [GemmaForgeCandidateResult], schedule: GemmaSSAIR?,
         scheduleDigest: String, planDigest: String,
         fullGateTokens: Int? = nil,
         fullGateRejections: [String]? = nil,
         modelFileVerification: ModelFileVerificationPolicy = ModelFileIdentity.policy) {
        self.schemaVersion = schemaVersion
        self.graphDigest = graphDigest
        self.ggufDigest = ggufDigest
        self.engineDigest = engineDigest
        self.hardware = hardware
        self.toolchain = toolchain
        self.workloadDigest = workloadDigest
        self.promptTokenCount = promptTokenCount
        self.decodeTokens = decodeTokens
        self.selected = selected
        self.selectedMedianWallMicrosPerToken = selectedMedianWallMicrosPerToken
        self.selectedMedianGpuMicrosPerToken = selectedMedianGpuMicrosPerToken
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.results = results
        self.schedule = schedule
        self.scheduleDigest = scheduleDigest
        self.fullGateTokens = fullGateTokens
        self.fullGateRejections = fullGateRejections
        self.modelFileVerification = modelFileVerification
        self.planDigest = planDigest
    }
}

public enum GemmaForge {

    /// 3: cached model-file SHA-256 with the cached-explicit-refresh-v1
    /// policy; schema-1 (full hash) and schema-2 (unverified) plans are
    /// rejected and must be re-forged.
    public static let schemaVersion = 3
    public static let warmupTokens = 1
    public static let timedTokens = 3
    /// Per-candidate GPU-time budget (micros) across all timed tokens.
    public static let budgetMicros: Double = 120_000_000

    /// Bounded legal candidate set for this graph. Segment batching is
    /// legal everywhere (CPU readbacks flush the open segment); the top-5
    /// diagnostic is a reporting choice, both dimensions are gated.
    public static func candidates() -> [GemmaForgeCandidate] {
        // Only real execution-level optimizations; the top-5 logit
        // diagnostic is reporting, not a compiler pass, and is excluded.
        [
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false),
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledQ4: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true),
            // Q6_K tiling (tied embedding / LM head) stacked on the
            // measured winners so far.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true, matvecTiledQ6: true),
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledQ4: true, matvecTiledQ6: true),
            // Tiled cooperative RMSNorm stacked on the current winner.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true, rmsnormTiled: true),
            // Tiled cooperative F32 matvec (MoE router) alone for
            // attribution, and stacked on the current winner.
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledF32: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true, rmsnormTiled: true,
                                matvecTiledF32: true),
            // Vectorized-load plain RMSNorm (identical arithmetic
            // sequence; the test suite asserts exact bit equality
            // against the retained original for the tested geometries)
            // — alone for attribution and stacked on segment batching.
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                vectorizedLoads: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true),
            // Q6_K tiled LM-head matvec stacked on the bit-identical
            // batched VL baseline. Bounded pass justified by the teacher-
            // forced layerwise diagnosis (raw Hello, 32 tokens): q6_tiled
            // ALONE showed worst logit rel 9.3e-06 vs the plain kernel
            // (10x inside the strict 1e-4 gate) while segment batching and
            // the VL RMSNorm are bit-identical. CAVEATS: the FP64
            // attribution sampled the Q4 tiled matvec only (layer-17
            // attn_q, 256 rows) — it does NOT measure Q6 accuracy; the Q6
            // head is ~4.6% of historical per-kernel GPU time (not the
            // dominant cost), so the gain may be modest; the FULL 32-token
            // gate on the composed candidate decides selection, unchanged
            // and fail-closed.
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledQ6: true, vectorizedLoads: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ6: true, vectorizedLoads: true),
            // GPU-resident cache_append (bounded new pass): eliminates the
            // per-layer CPU readback+memcpy roundtrip by encoding a bit-
            // preserving uint32 GPU copy in the compute segment so the
            // following attention reads ordered GPU writes. Alone for
            // attribution (unbatched) and stacked on segment batching.
            // Bit-exact by construction; the CPU incumbent path is
            // untouched and remains the default selector policy.
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                gpuCacheAppend: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true),
            // PRODUCTION parallel attention (bounded new pass): the runner's
            // production par64 pipeline (original 64-lane kernel, numerically
            // unmodified) serves eligible attention geometry; ineligible
            // geometry falls back to the incumbent g4_attn and counts a
            // production fallback. Stacked on the selected bit-identical
            // batched VL baseline, and on batched VL + GPU cache append —
            // all existing numerical gates (<1e-4, exact greedy) unchanged
            // and no forced winner.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, parallelAttention: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true),
            // Exact-order cooperative RMSNorm (bounded new pass): the
            // incumbent RMS kernel is the largest single GPU-time
            // hotspot (41.77% RMS across 331 calls/token — one thread
            // per vector, serial reads and serial apply). The
            // cooperative kernel stages the input into dynamic
            // threadgroup memory, keeps the lane-0 sum-of-squares in the
            // EXACT incumbent order (no reassociated tree), and applies
            // per element in parallel with the identical association —
            // hardware bit equality is asserted by the test suite for
            // the tested geometries; the FULL gates decide selection.
            // The 7 distinct non-empty combinations of {rmsCoop, Q6
            // tiled, F32 tiled} on the ALWAYS batched + VL + GPU-KV +
            // parallel-attention baseline (never rmsnormTiled, never
            // tiled Q4). Total candidate set before the cooperative
            // Q6 matvec pass: 24.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ6: true, vectorizedLoads: true,
                                gpuCacheAppend: true, parallelAttention: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledF32: true, vectorizedLoads: true,
                                gpuCacheAppend: true, parallelAttention: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ6: true, matvecTiledF32: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledF32: true, vectorizedLoads: true,
                                gpuCacheAppend: true, parallelAttention: true,
                                rmsnormCooperative: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ6: true, matvecTiledF32: true,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ6: true, matvecTiledF32: true,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true),
            // PRODUCTION intra-block SIMD-reduction Q6_K matvec (bounded
            // new pass): eligible Q6_K matvecs (tied embedding / LM head)
            // dispatch the cooperative kernel from the public
            // Q6Cooperative enum source; ineligible geometry falls back
            // to the plain g4_q6_matvec and counts a production
            // fallback. The simd_sum reduction reassociates the addition
            // order, so bit-exactness is NOT claimed — the strict FP64
            // oracle and the FULL numerical gates (<1e-4, exact greedy)
            // decide selection, unchanged and fail-closed. Appended on
            // the ALWAYS batched + VL + GPU-KV + parallel-attention
            // baseline WITHOUT the cooperative RMSNorm axis and WITH it
            // (never tiled Q4, never tiled F32, never tiled Q6, never
            // rmsnormTiled). Total candidate set: 26.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, matvecCooperativeQ6: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true),
            // Reusable RMS statistics (bounded new pass): the validated
            // plan turns the measured 30 groups / 60 duplicated statistic
            // reductions per token into one compute + N cached applies,
            // bit-exact by construction against the incumbent arithmetic.
            // Appended alone on the ALWAYS batched + VL + GPU-KV +
            // parallel-attention + cooperative-RMS + cooperative-Q6
            // baseline (all tiled axes false, never rmsnormTiled).
            // Total candidate set: 27.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true),
            // GPU router residency (bounded new pass, Q4/router-owned):
            // the second per-element router scale multiply moves to a
            // precise-compiled GPU kernel (the first g4_scale round keeps
            // its own ordered dispatch; the two multiplies are never
            // fused), and the resident-expert accumulator reset becomes a
            // bit-preserving uint32 GPU zero in the same ordered encoder.
            // routerLaunchDivisor 0 keeps the original scalar router F32
            // matvec geometry (GPU scale/zero only); divisors 1/2/4/8
            // re-dispatch the SAME pipeline and bindings with a derived
            // threadgroup width. Appended on the new-main fast-flag
            // baseline (batched + VL + GPU-KV + parallel attention +
            // cooperative RMS + cooperative Q6 + reusable RMS statistics;
            // all tiled axes false). Total candidate set: 32.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true,
                                routerGPUResidency: true, routerLaunchDivisor: 0),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true,
                                routerGPUResidency: true, routerLaunchDivisor: 1),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true,
                                routerGPUResidency: true, routerLaunchDivisor: 2),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true,
                                routerGPUResidency: true, routerLaunchDivisor: 4),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                vectorizedLoads: true, gpuCacheAppend: true,
                                parallelAttention: true, rmsnormCooperative: true,
                                matvecCooperativeQ6: true,
                                rmsStatisticsReuse: true,
                                routerGPUResidency: true, routerLaunchDivisor: 8),
        ]
    }

    /// PURE device-free candidate legality guard: reusable RMS
    /// statistics reuse the EXACT cooperative + vectorized incumbent
    /// arithmetic, so rmsStatisticsReuse is legal only when
    /// rmsnormCooperative and vectorizedLoads are enabled and
    /// rmsnormTiled is disabled. Both public executors invoke this
    /// BEFORE any runner reset or GPU work — an incompatible artifact
    /// is rejected rather than silently falling back.
    public static func validateCandidate(_ candidate: GemmaForgeCandidate) throws {
        if candidate.rmsStatisticsReuse {
            guard candidate.rmsnormCooperative, candidate.vectorizedLoads,
                  !candidate.rmsnormTiled else {
                throw GemmaForgeError.invalidGraph(
                    "rmsStatisticsReuse requires rmsnormCooperative=true, " +
                    "vectorizedLoads=true, rmsnormTiled=false")
            }
        }
        // Router launch divisor: restricted to the allowed set; a nonzero
        // divisor (derived scalar router threadgroup geometry) requires
        // the residency flag and must not combine with tiled F32.
        guard RouterResidencyKernel.allowedLaunchDivisors.contains(
            candidate.routerLaunchDivisor) else {
            throw GemmaForgeError.invalidGraph(
                "routerLaunchDivisor must be one of " +
                "\(RouterResidencyKernel.allowedLaunchDivisors)")
        }
        if candidate.routerLaunchDivisor != 0 {
            guard candidate.routerGPUResidency, !candidate.matvecTiledF32 else {
                throw GemmaForgeError.invalidGraph(
                    "routerLaunchDivisor != 0 requires routerGPUResidency=true " +
                    "and matvecTiledF32=false")
            }
        }
    }

    /// General streaming SHA-256 utility for executable/source/small
    /// fixture files. Model (GGUF) digests go through ModelDigestCache,
    /// which hashes each model once and reuses the persisted result. The
    /// file is read in
    /// bounded chunks. Each chunk is hashed inside its own autorelease
    /// pool so Foundation's autoreleased read buffers are released every
    /// iteration and memory stays bounded regardless of file size.
    public static func fileDigest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = CryptoKit.SHA256()
        let chunk = 1 << 23
        var done = false
        while !done {
            done = try autoreleasepool {
                let data = try handle.read(upToCount: chunk)
                guard let data, !data.isEmpty else { return true }
                hasher.update(data: data)
                return false
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - graph executor

    /// Executes the validated graph IR on the runner's kernel primitives.
    /// The per-kernel mode is bit-identical in dispatch order to the
    /// reference path; segment batching only groups submissions.
    public static func executeGraph(_ graph: GemmaGraphIR, runner: Gemma4Runner,
                                    tokenIds: [Int],
                                    candidate: GemmaForgeCandidate,
                                    schedule: GemmaSSAIR? = nil,
                                    prepared: GemmaPreparedExecution? = nil) throws
            -> (argmax: Int, logits: [Float], gpuMicros: Double, kernels: Int,
                buffers: Int) {
        guard !tokenIds.isEmpty else { throw GemmaForgeError.executionFailed("empty tokens") }
        // Untrusted graphs never reach execution: full validation
        // against the actual GGUF + config (unknown ops/layers/tensors
        // rejected before any raw cache indexing). The PREPARED path
        // skips this because prepare already validated THIS exact graph
        // — the binding check below establishes (runner object
        // identity + structural graph equality + config fingerprint)
        // that the graph IS the one validated at prepare time. The
        // ordinary paths are unchanged.
        let ssa: GemmaSSAIR
        let byID: [String: GemmaSSAIR.Value]
        let statisticsPlan: RMSStatisticsPlan
        if let prepared {
            guard schedule == nil else {
                throw GemmaForgeError.executionFailed(
                    "pass either a prepared execution or a raw schedule, not both")
            }
            try prepared.assertBound(to: runner, graph: graph)
            ssa = prepared.ssa
            byID = prepared.byID
            statisticsPlan = prepared.statisticsPlan
        } else {
            try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
            if let schedule {
                try GemmaSSA.validate(schedule, config: graph.config)
                // Public schedule execution additionally requires EXACT
                // equality with the canonical lowering of THIS validated
                // graph (tensor bindings and geometry checked against the
                // mapped GGUF via GemmaGraph.validate above) — a hand-built
                // look-alike schedule is rejected before any GPU work.
                let canonical = try GemmaSSA.lower(graph)
                guard schedule == canonical else {
                    throw GemmaForgeError.invalidGraph(
                        "schedule is not the canonical lowering of this graph")
                }
                ssa = schedule
            } else {
                ssa = try GemmaSSA.lower(graph)
            }
            byID = GemmaPreparedExecution.valueTable(ssa)
            statisticsPlan = try RMSStatisticsPlan.analyze(
                ssa: ssa, epsilon: runner.config.rmsNormEps)
        }
        // Axis legality is enforced BEFORE any runner reset or GPU
        // work: an incompatible statistics candidate is rejected here.
        try validateCandidate(candidate)
        let gpuBefore = runner.gpuMicrosTotal
        let dispatchBefore = runner.dispatchCountTotal
        let buffersBefore = runner.submittedBuffers
        runner.reset()
        var argmax = -1
        var logits: [Float] = []
        do {
            for token in tokenIds {
                let pos = runner.contextPosition()
                guard token >= 0, token < runner.config.vocabSize,
                      pos < runner.maxContext else {
                    throw GemmaForgeError.executionFailed("token/position out of bounds")
                }
                argmax = try executeToken(ssa, byID: byID, runner,
                                          token: token, position: pos,
                                          candidate: candidate,
                                          statisticsPlan: statisticsPlan)
                runner.advancePosition()
                logits = try runner.read(runner.bufLogits, runner.config.vocabSize)
            }
        } catch {
            // Cleanup without losing the error: discard any uncommitted
            // deferred encoder, disable auto-segment, and reset the context
            // so no partial KV state survives a failed run.
            runner.discardOpenSegment()
            runner.reset()
            throw error
        }
        return (argmax, logits, runner.gpuMicrosTotal - gpuBefore,
                runner.dispatchCountTotal - dispatchBefore,
                runner.submittedBuffers - buffersBefore)
    }

    /// Executes ONE token continuing the runner's current KV context at
    /// position consumedPosition (no reset) — the real decode workload.
    public static func executeGraphContinue(_ graph: GemmaGraphIR,
                                            runner: Gemma4Runner, token: Int,
                                            candidate: GemmaForgeCandidate,
                                            schedule: GemmaSSAIR? = nil,
                                            prepared: GemmaPreparedExecution? = nil) throws
            -> (argmax: Int, logits: [Float], gpuMicros: Double, kernels: Int,
                buffers: Int) {
        // Prepared decode path: the full GGUF validation, lowering, and
        // value-table build happened once at prepare time; reuse here is
        // the cheap binding check (exact runner object identity, config
        // fingerprint, structural graph equality) — no per-token
        // hashing or lowering. Token/position bounds guards below are
        // unchanged on every path. Ordinary paths validate as before.
        let ssa: GemmaSSAIR
        let byID: [String: GemmaSSAIR.Value]
        let statisticsPlan: RMSStatisticsPlan
        if let prepared {
            guard schedule == nil else {
                throw GemmaForgeError.executionFailed(
                    "pass either a prepared execution or a raw schedule, not both")
            }
            try prepared.assertBound(to: runner, graph: graph)
            ssa = prepared.ssa
            byID = prepared.byID
            statisticsPlan = prepared.statisticsPlan
        } else {
            try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
            if let schedule {
                try GemmaSSA.validate(schedule, config: graph.config)
                let canonical = try GemmaSSA.lower(graph)
                guard schedule == canonical else {
                    throw GemmaForgeError.invalidGraph(
                        "schedule is not the canonical lowering of this graph")
                }
                ssa = schedule
            } else {
                ssa = try GemmaSSA.lower(graph)
            }
            byID = GemmaPreparedExecution.valueTable(ssa)
            statisticsPlan = try RMSStatisticsPlan.analyze(
                ssa: ssa, epsilon: runner.config.rmsNormEps)
        }
        // Axis legality is enforced BEFORE any GPU work: an
        // incompatible statistics candidate is rejected here.
        try validateCandidate(candidate)
        guard token >= 0, token < runner.config.vocabSize,
              runner.contextPosition() < runner.maxContext else {
            throw GemmaForgeError.executionFailed("token/position out of bounds")
        }
        let gpuBefore = runner.gpuMicrosTotal
        let dispatchBefore = runner.dispatchCountTotal
        let buffersBefore = runner.submittedBuffers
        let position = runner.contextPosition()
        do {
            let argmax = try executeToken(ssa, byID: byID, runner,
                                          token: token, position: position,
                                          candidate: candidate,
                                          statisticsPlan: statisticsPlan)
            runner.advancePosition()
            let logits = try runner.read(runner.bufLogits, runner.config.vocabSize)
            return (argmax, logits, runner.gpuMicrosTotal - gpuBefore,
                    runner.dispatchCountTotal - dispatchBefore,
                    runner.submittedBuffers - buffersBefore)
        } catch {
            runner.discardOpenSegment()
            runner.reset()
            throw error
        }
    }

    /// Executes one token from the SSA operands. The value table
    /// (`byID`) is supplied by the caller: either the immutable table
    /// retained by a GemmaPreparedExecution, or one built once per
    /// public request — never rebuilt per token inside this function.
    private static func executeToken(_ ssa: GemmaSSAIR,
                                     byID: [String: GemmaSSAIR.Value],
                                     _ runner: Gemma4Runner,
                                     token: Int, position: Int,
                                     candidate: GemmaForgeCandidate,
                                     statisticsPlan: RMSStatisticsPlan?) throws -> Int {
        let config = runner.config
        // The schedule was lowered, validated, and turned into a value
        // table ONCE by the caller; the executor dispatches kernels from
        // the SSA OPERANDS. Buffers are resolved per operand kind (vector
        // values only — KV/routing state never touches this path),
        // in-place ops are validated by the lowering to alias their
        // input's buffer, and the final logits are read from the FINAL
        // ARTIFACT OUTPUT VALUE, not a fixed buffer.
        func vb(_ id: String) throws -> MTLBuffer {
            guard let value = byID[id], value.kind == .vector else {
                throw GemmaForgeError.executionFailed(
                    "operand \(id) is not a buffer-resident vector value")
            }
            switch value.name {
            case "x": return runner.bufX
            case "h": return runner.bufH
            case "t1": return runner.bufT1
            case "t2": return runner.bufT2
            case "t3": return runner.bufT3
            case "kv": return runner.bufKV
            case "gate", "up": return runner.bufGU   // offset slices, see matvec
            case "act": return runner.bufAct
            case "shared": return runner.bufShared
            case "router": return runner.bufRouter
            case "logits": return runner.bufLogits
            default:
                throw GemmaForgeError.executionFailed(
                    "no physical buffer binding for value \(value.name)")
            }
        }
        func a(_ inst: GemmaSSAIR.Instruction, _ key: String) -> Int {
            inst.attrs[key] ?? 0
        }
        // Batch GPU-only runs into one command buffer per run across ops;
        // CPU readback (read/write) flushes the open segment. Kernel order
        // is identical in both modes.
        runner.autoSegment = candidate.segmentBatching
        runner.q4TiledKernel = candidate.matvecTiledQ4
        runner.q6TiledKernel = candidate.matvecTiledQ6
        runner.rmsTiledKernel = candidate.rmsnormTiled
        runner.f32TiledKernel = candidate.matvecTiledF32
        runner.vectorizedLoads = candidate.vectorizedLoads
        runner.gpuCacheAppendKernel = candidate.gpuCacheAppend
        runner.parallelAttentionKernel = candidate.parallelAttention
        runner.rmsCooperativeKernel = candidate.rmsnormCooperative
        runner.matvecCooperativeQ6Kernel = candidate.matvecCooperativeQ6
        runner.rmsStatisticsReuse = candidate.rmsStatisticsReuse
        runner.routerGPUResidency = candidate.routerGPUResidency
        runner.routerLaunchDivisor = candidate.routerLaunchDivisor
        defer {
            runner.autoSegment = false
            runner.q4TiledKernel = false
            runner.q6TiledKernel = false
            runner.rmsTiledKernel = false
            runner.f32TiledKernel = false
            runner.vectorizedLoads = false
            runner.gpuCacheAppendKernel = false
            runner.parallelAttentionKernel = false
            runner.rmsCooperativeKernel = false
            runner.matvecCooperativeQ6Kernel = false
            runner.rmsStatisticsReuse = false
            runner.routerGPUResidency = false
            runner.routerLaunchDivisor = 0
        }
        var routerState = [Int: (topIdx: [Int], weights: [Float])]()

        // Fresh per-token readiness: a statistic computed in a previous
        // token, before a reset, or across a failed run is NEVER reused.
        var readySlots = [Bool](repeating: false, count: statisticsPlan?.groupCount ?? 0)

        for (index, inst) in ssa.instructions.enumerated() {
            switch inst.op {
            case "embed_row":
                guard inst.tensors.first == "token_embd.weight" else {
                    throw GemmaForgeError.executionFailed(
                        "embed tensor binding is not the embedding table")
                }
                try runner.decodeEmbedding(token)
                let emb = try runner.read(runner.bufEmbed, runner.hidden)
                guard emb.allSatisfy({ $0.isFinite }) else {
                    throw GemmaForgeError.executionFailed("embedding nonfinite")
                }
                try runner.write(emb, to: try vb(inst.output))
            case "rmsnorm":
                // n is PER VECTOR (the primitive's contract): the validated
                // width for vectorized norms, the total length otherwise.
                let vectors = max(1, a(inst, "vectors"))
                let n = vectors == 1 ? byID[inst.output]!.length : a(inst, "width")
                let xBuf = try vb(inst.inputs[0])
                let yBuf = try vb(inst.output)
                // Reusable statistics path. The installed GENERATED RMS
                // kernel family keeps precedence (never mixed reduction
                // arithmetic — count a fallback and use the chain below).
                // The planned action is rechecked against the CURRENT
                // call (input id, width, vectors, epsilon bits) before
                // any trust; unknown/unplanned/ineligible calls and any
                // guard failure use the ordinary incumbent RMS chain.
                // Reusable statistics path: only PLANNED actions touch
                // the production fallback counter, and each planned
                // action that falls back increments EXACTLY once. The
                // installed GENERATED RMS kernel family keeps precedence
                // (never mixed reduction arithmetic); a plan guard
                // mismatch or a not-yet-ready reuse slot counts here in
                // the executor, while the dispatch helpers count their
                // own guard failures themselves — never both. Unplanned
                // RMSNorm calls (no action at this instruction index)
                // never count. Every fallback runs the unchanged
                // ordinary incumbent RMS chain below.
                var statsServed = false
                var statsFallback = false
                if candidate.rmsStatisticsReuse,
                   let plan = statisticsPlan,
                   plan.epsilonBits == config.rmsNormEps.bitPattern,
                   let action = plan.action(at: index) {
                    if runner.rmsGeneratedKernel != nil {
                        statsFallback = true
                    } else if action.inputID != inst.inputs.first
                        || action.width != n || action.vectors != vectors {
                        statsFallback = true
                    } else if action.kind == .compute {
                        if try runner.rmsStatisticsCompute(
                            n: n, vectors: vectors,
                            weightTensor: inst.tensors.first,
                            slots: plan.groupCount, slot: action.slot,
                            x: xBuf, y: yBuf) {
                            if action.slot < readySlots.count {
                                readySlots[action.slot] = true
                            }
                            statsServed = true
                        }
                    } else if action.slot >= readySlots.count
                        || !readySlots[action.slot] {
                        statsFallback = true
                    } else if try runner.rmsStatisticsCachedApply(
                        n: n, vectors: vectors,
                        weightTensor: inst.tensors.first,
                        slots: plan.groupCount, slot: action.slot,
                        x: xBuf, y: yBuf) {
                        statsServed = true
                    }
                }
                if statsFallback {
                    runner.rmsStatisticsCountFallback()
                }
                if !statsServed {
                    try runner.rmsnorm(x: xBuf, y: yBuf,
                                       n: n, vectors: vectors,
                                       weightTensor: inst.tensors.first)
                }
            case "matvec":
                // Physical capacity includes the slice offset (shared_up
                // writes the upper half of the fused gate|up buffer); the
                // LOGICAL output length stays the slice length.
                let rows = a(inst, "rows")
                try runner.matvec(tensor: inst.tensors[0],
                                  x: try vb(inst.inputs[0]),
                                  xCount: byID[inst.inputs[0]]!.length,
                                  y: try vb(inst.output),
                                  yCapacity: a(inst, "offset") + rows,
                                  rows: rows,
                                  yByteOffset: a(inst, "offset") * 4)
            case "rope":
                // Explicit validated phase operand from the lowering.
                try runner.rope(x: try vb(inst.inputs[0]),
                                heads: a(inst, "heads"),
                                half: a(inst, "half"), position: position,
                                freqs: a(inst, "global") == 1
                                    ? runner.globalFreqBuffer
                                    : runner.localFreqBuffer)
            case "copy":
                let src = try runner.read(try vb(inst.inputs[0]),
                                          byID[inst.inputs[0]]!.length)
                try runner.write(src, to: try vb(inst.output))
            case "axpy":
                // inputs[0] is the accumulator (x) — validated to alias the
                // output — inputs[1] the addend (h).
                try runner.axpy(x: try vb(inst.inputs[1]), y: try vb(inst.output),
                                c: 1, n: byID[inst.output]!.length)
            case "add":
                try runner.add(a: try vb(inst.inputs[0]), b: try vb(inst.inputs[1]),
                               y: try vb(inst.output), n: byID[inst.output]!.length)
            case "scale":
                // Guarded layer geometry — a malformed artifact cannot reach
                // raw per-layer vectors.
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("scale layer invalid")
                }
                if inst.attrs["routerScale"] == 1 {
                    let outBuf = try vb(inst.output)
                    let n = byID[inst.output]!.length
                    // Round 1: the original g4_scale dispatch, unconditional.
                    try runner.scaleInPlace(x: outBuf,
                                            c: Float(1.0 / sqrt(Double(n))), n: n)
                    // Round 2: GPU-encoded second scale when armed and
                    // validated (shared helper with the direct forward
                    // path); otherwise the ORIGINAL CPU element multiply
                    // runs unchanged.
                    if !(try runner.gpuRouterSecondScale(layer: layer, n: n, x: outBuf)) {
                        var routerIn = try runner.read(outBuf, n)
                        let scale = runner.routerScales[layer]
                        for i in 0..<n { routerIn[i] *= scale[i] }
                        try runner.write(routerIn, to: outBuf)
                    }
                } else {
                    try runner.scaleInPlace(x: try vb(inst.output),
                                            c: runner.layerScalars[layer],
                                            n: byID[inst.output]!.length)
                }
            case "gelu_up":
                // gate/up alias the fused gate|up buffer at their matvec
                // offsets; the kernel consumes the fused pair in place.
                try runner.geluUp(gu: try vb(inst.inputs[0]), y: try vb(inst.output),
                                  n: a(inst, "width"))
            case "cache_append":
                // Guarded geometry: layer bounds and checked products — a
                // malformed artifact cannot index raw cache storage.
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount,
                      let kvHeads = inst.attrs["kvHeads"],
                      let headDim = inst.attrs["headDim"],
                      kvHeads > 0, headDim > 0,
                      position >= 0 else {
                    throw GemmaForgeError.executionFailed("cache_append geometry invalid")
                }
                let kvWidth = byID[inst.inputs[0]]!.length
                guard kvWidth == kvHeads * headDim,
                      kvWidth == byID[inst.inputs[1]]!.length else {
                    throw GemmaForgeError.executionFailed("cache_append width mismatch")
                }
                let (byteWidth, ov1) = kvWidth.multipliedReportingOverflow(by: 4)
                let (byteOffset, ov2) = byteWidth.multipliedReportingOverflow(by: position)
                guard !ov1, !ov2,
                      byteOffset + byteWidth <= runner.kCaches[layer].length,
                      byteOffset + byteWidth <= runner.vCaches[layer].length else {
                    throw GemmaForgeError.executionFailed("cache_append out of bounds")
                }
                if runner.gpuCacheAppendKernel {
                    try runner.gpuCacheAppendCopy(kSrc: try vb(inst.inputs[0]),
                                                  vSrc: try vb(inst.inputs[1]),
                                                  kDst: runner.kCaches[layer],
                                                  vDst: runner.vCaches[layer],
                                                  kvWidth: kvWidth, position: position)
                } else {
                    try runner.flushOpenSegment()
                    let k = try runner.read(try vb(inst.inputs[0]), kvWidth)
                    let v = try runner.read(try vb(inst.inputs[1]), kvWidth)
                    memcpy(runner.kCaches[layer].contents() + byteOffset, k, byteWidth)
                    memcpy(runner.vCaches[layer].contents() + byteOffset, v, byteWidth)
                }
            case "attention":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("attention layer invalid")
                }
                var p = SIMD4<UInt32>(UInt32(position + 1), UInt32(position),
                                      UInt32(a(inst, "kvHeads")), UInt32(a(inst, "headDim")))
                var q = SIMD2<UInt32>(UInt32(a(inst, "heads")), UInt32(a(inst, "window")))
                let qBuf = try vb(inst.inputs[0])
                let outBuf = try vb(inst.output)
                // DIAGNOSTIC ONLY: the SSA path reaches the shared attention diagnostic hook with the exact incumbent
                // bindings. When no candidate is installed, or the geometry falls back, the unchanged incumbent runs.
                let attentionDiagnosticHandled = try runner.attentionDiagnosticDispatch(
                    qHeads: Int(a(inst, "heads")), kvHeads: Int(a(inst, "kvHeads")),
                    headDim: Int(a(inst, "headDim")), position: Int(position),
                    window: Int(a(inst, "window")),
                    bind: { enc in
                        enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
                        enc.setBytes(&q, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                        enc.setBuffer(qBuf, offset: 0, index: 2)
                        enc.setBuffer(runner.kCaches[layer], offset: 0, index: 3)
                        enc.setBuffer(runner.vCaches[layer], offset: 0, index: 4)
                        enc.setBuffer(outBuf, offset: 0, index: 5)
                    })
                if !attentionDiagnosticHandled {
                    try runner.dispatch("g4_attn", grid: a(inst, "heads")) { enc in
                        enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
                        enc.setBytes(&q, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                        enc.setBuffer(qBuf, offset: 0, index: 2)
                        enc.setBuffer(runner.kCaches[layer], offset: 0, index: 3)
                        enc.setBuffer(runner.vCaches[layer], offset: 0, index: 4)
                        enc.setBuffer(outBuf, offset: 0, index: 5)
                    }
                }
            case "router_topk":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("router layer invalid")
                }
                try runner.flushOpenSegment()
                let scores = try runner.read(try vb(inst.inputs[0]), config.expertCount)
                guard scores.allSatisfy({ $0.isFinite }) else {
                    throw GemmaForgeError.executionFailed("router scores nonfinite")
                }
                let maxScore = scores.max() ?? 0
                let exps = scores.map { exp(Double($0 - maxScore)) }
                let sumExp = exps.reduce(0, +)
                let probs = exps.map { $0 / sumExp }
                var idx = Array(0..<config.expertCount)
                idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
                let topIdx = Array(idx.prefix(a(inst, "topK")))
                let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
                let scales = runner.expertScales[layer]
                let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }
                routerState[layer] = (topIdx, weights)
            case "expert_matvec_chain":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount,
                      let gateUp = runner.ref[inst.tensors[0]],
                      let down = runner.ref[inst.tensors[1]],
                      let (topIdx, weights) = routerState[layer] else {
                    throw GemmaForgeError.executionFailed("missing expert tensors/routing")
                }
                let e = a(inst, "intermediate")
                let guStride = gateUp.outDim * (gateUp.inDim / 32) * 18
                let dnStride = down.outDim * (down.inDim / 32) * 18
                let outBuf = try vb(inst.output)
                let n = byID[inst.output]!.length
                // Resident-expert output reset: GPU zero in the SAME
                // ordered encoder when armed (flag on, no expert cache);
                // the fallback preserves the ORIGINAL CPU flush/prefetch/
                // memset sequence exactly (the flush safely drains any
                // pending GPU work before the CPU memset).
                if !(try runner.gpuRouterResidentZero(y: outBuf, n: n)) {
                    try runner.flushOpenSegment()
                    // Synchronous free-slots-only read-ahead (cache mode;
                    // never evicts, no async overlap).
                    try runner.prefetchExperts(layer: layer, topIdx: topIdx)
                    memset(outBuf.contents(), 0, n * 4)
                }
                // DEDICATED expert scratch, physically separate from all
                // SSA-bound buffers — internal effects of this compound op
                // can never clobber live SSA values (gate/up/act/t2).
                let inBuf = try vb(inst.inputs[0])
                let inLen = byID[inst.inputs[0]]!.length
                // Generated batched lowering (phase 13): the SAME shared
                // implementation as the direct forward path — the serving
                // SSA interpreter must not silently bypass it. Structured
                // else (no return: later norms/layers/logits must run);
                // fail-closed on any guard; cache mode keeps the incumbent
                // serial loop below with the fallback counter.
                var batched = false
                if let installed = runner.moeGenerated, runner.expertCache == nil {
                    guard n == config.hiddenSize, inLen == config.hiddenSize,
                          inBuf !== outBuf,
                          inst.tensors[0] == "blk.\(layer).ffn_gate_up_exps.weight",
                          inst.tensors[1] == "blk.\(layer).ffn_down_exps.weight" else {
                        throw GemmaForgeError.executionFailed(
                            "SSA expert chain geometry/tensors disagree with the "
                                + "installed batched region")
                    }
                    try runner.runMoeBatchedRouted(x: inBuf, y: outBuf,
                                                   layer: layer, topIdx: topIdx,
                                                   weights: weights,
                                                   installed: installed)
                    batched = true
                }
                if runner.moeGenerated != nil, runner.expertCache != nil {
                    runner.recordMoeResidentFallback()
                }
                if !batched {
                for (slot, expert) in topIdx.enumerated() {
                    if runner.expertCache != nil {
                        // SSD/streaming mode: slices stream through the
                        // bounded LRU cache (same kernels, slot-pool
                        // bindings) instead of the resident mapping.
                        try runner.expertMatvec(tensor: inst.tensors[0],
                                                layer: layer, expert: expert,
                                                x: inBuf, xCount: inLen,
                                                y: runner.bufExpertGU,
                                                yCapacity: 2 * config.sharedIntermediate)
                        try runner.geluUp(gu: runner.bufExpertGU, y: runner.bufExpertAct,
                                          n: e)
                        try runner.expertMatvec(tensor: inst.tensors[1],
                                                layer: layer, expert: expert,
                                                x: runner.bufExpertAct, xCount: e,
                                                y: runner.bufExpertAcc, yCapacity: n)
                    } else {
                        try runner.matvec(tensor: inst.tensors[0],
                                          extraOffset: expert * guStride,
                                          x: inBuf, xCount: inLen,
                                          y: runner.bufExpertGU,
                                          yCapacity: 2 * config.sharedIntermediate)
                        try runner.geluUp(gu: runner.bufExpertGU, y: runner.bufExpertAct,
                                          n: e)
                        try runner.matvec(tensor: inst.tensors[1],
                                          extraOffset: expert * dnStride,
                                          x: runner.bufExpertAct, xCount: e,
                                          y: runner.bufExpertAcc, yCapacity: n)
                    }
                    try runner.axpy(x: runner.bufExpertAcc, y: outBuf,
                                    c: weights[slot], n: n)
                    runner.expertCache?.endExpertIteration(
                        segmentPending: runner.openEnc != nil)
                }
                }
            case "softcap":
                var cap = Float(a(inst, "cap"))
                var count = UInt32(byID[inst.output]!.length)
                let outBuf = try vb(inst.output)
                try runner.dispatch("g4_softcap", grid: byID[inst.output]!.length) { enc in
                    enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
                    enc.setBuffer(outBuf, offset: 0, index: 1)
                    enc.setBytes(&cap, length: MemoryLayout<Float>.stride, index: 2)
                }
            default:
                throw GemmaForgeError.invalidGraph(
                    "unsupported ssa op at execution: \(inst.op)")
            }
        }
        // Final logits come from the FINAL ARTIFACT OUTPUT VALUE.
        guard let finalInst = ssa.instructions.last,
              finalInst.op == "softcap",
              let finalValue = byID[finalInst.output],
              finalValue.name == "logits",
              finalValue.length == config.vocabSize else {
            throw GemmaForgeError.executionFailed("schedule terminal is not logits")
        }
        let logits = try runner.read(try vb(finalInst.output), finalValue.length)
        guard logits.allSatisfy({ $0.isFinite }) else {
            throw GemmaForgeError.executionFailed("nonfinite logits")
        }
        if candidate.top5Diagnostic {
            _ = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
        }
        return logits.enumerated().max { $0.element < $1.element }!.offset
    }

    // MARK: - pipeline

    /// Full pipeline: validate the captured graph, gate candidates
    /// against the reference per-kernel executor AND the runner's own
    /// validated forward (raw logits), measure GPU time per token, and
    /// select the best valid candidate.
    public static func forge(graph: GemmaGraphIR, runner: Gemma4Runner,
                             ggufURL: URL, prompt: [Int], decodeTokens: Int,
                             hardware: ForgeHardwareProfile) throws
            -> (plan: GemmaForgePlan, results: [GemmaForgeCandidateResult],
                selected: GemmaForgeCandidate) {
        guard prompt.allSatisfy({ $0 >= 0 && $0 < runner.config.vocabSize }),
              !prompt.isEmpty,
              prompt.count + warmupTokens + timedTokens <= runner.maxContext,
              decodeTokens == warmupTokens + timedTokens else {
            throw GemmaForgeError.invalidGraph("bad workload tokens/size")
        }
        // Identity inputs: the graph digest is computed from the actual graph
        // and the model digest is the actual SHA-256 from the shared
        // ModelDigestCache (hashed once on an initial miss or an explicit
        // --rehash-model refresh; otherwise reused without re-reading). The
        // URL must still BE the file the runner mapped (byteCount cross-check)
        // — an execution binding check, not a cache freshness check.
        guard let mappedURL = runner.ggufFile.sourceURL, mappedURL == ggufURL,
              runner.ggufFile.byteCount == (try? URL(fileURLWithPath: ggufURL.path)
                    .resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1) else {
            throw GemmaForgeError.invalidPlan(
                "forge URL is not the runner's mapped GGUF file")
        }
        let graphDigest = try GemmaGraph.digest(graph)
        let ggufDigest = try ModelDigestCache.shared.digest(for: ggufURL).sha256
        let engineDigest = Self.engineDigest(try runner.engineIdentity())
        let reference = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
        // Reference gate 1: graph executor (per-kernel) must agree with
        // the runner's independently validated forward on raw logits and
        // greedy argmax for the workload prompt.
        runner.reset()
        let refEvidence = try runner.forward(tokenIds: prompt)
        // Independent FULL raw-logit reference captured immediately after
        // the validated runner forward, before any graph execution.
        let refLogitsAll = try runner.read(runner.bufLogits,
                                           runner.config.vocabSize)
        let (refArgmax, refLogits, _, _, _) = try executeGraph(
            graph, runner: runner, tokenIds: prompt, candidate: reference)
        guard refArgmax == refEvidence.argmaxToken else {
            throw GemmaForgeError.noValidCandidate([
                "graph executor argmax \(refArgmax) != reference \(refEvidence.argmaxToken)"])
        }
        // Graph executor vs independent reference over ALL logits.
        guard refLogits.count == refLogitsAll.count,
              refLogits.allSatisfy({ $0.isFinite }),
              refLogitsAll.allSatisfy({ $0.isFinite }) else {
            throw GemmaForgeError.noValidCandidate(["logit count/finiteness gate failed"])
        }
        var refWorst = 0.0
        for i in 0..<refLogits.count {
            refWorst = max(refWorst, abs(Double(refLogits[i] - refLogitsAll[i]))
                           / max(1.0, abs(Double(refLogitsAll[i]))))
        }
        guard refWorst < 1e-4 else {
            throw GemmaForgeError.noValidCandidate([
                String(format: "graph logits vs runner reference worst rel %.3e", refWorst)])
        }

        // Reference greedy decode sequence (warmup + timed tokens)
        // captured ONCE with the reference candidate at true positions.
        guard prompt.count + warmupTokens + timedTokens <= runner.maxContext,
              decodeTokens == warmupTokens + timedTokens else {
            throw GemmaForgeError.invalidGraph(
                "workload exceeds context or decodeTokens != warmup+timed")
        }
        // INDEPENDENT reference: the validated runner.forward itself,
        // continuing the prompt context; per-step raw logits captured.
        runner.reset()
        let prefillEvidence = try runner.forward(tokenIds: prompt)
        guard prefillEvidence.argmaxToken == refArgmax else {
            throw GemmaForgeError.noValidCandidate(["independent prefill argmax mismatch"])
        }
        var refDecodeSeq = [Int]()
        var refDecodeLogits = [[Float]]()
        var prevTok = prefillEvidence.argmaxToken
        for _ in 0..<(warmupTokens + timedTokens) {
            let ev = try runner.forward(tokenIds: [prevTok])
            refDecodeSeq.append(ev.argmaxToken)
            refDecodeLogits.append(try runner.read(runner.bufLogits,
                                                   runner.config.vocabSize))
            prevTok = ev.argmaxToken
        }
        // One PREPARED execution per runner, built ONCE outside the timed
        // candidate loop: full GGUF validation + canonical lowering + SSA
        // dataflow validation + value-table build happen a single time;
        // every candidate measurement then reuses the cheap binding check
        // (exact runner object identity + config fingerprint + structural
        // graph equality). All existing correctness/validation gates are
        // unchanged.
        let prep = try GemmaPreparedExecution.prepare(graph: graph, runner: runner)
        var results: [GemmaForgeCandidateResult] = []
        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2
                                             : s[s.count/2]
        }
        struct Sample { var wall: Double; var gpu: Double; var kernels: Double; var buffers: Double }
        var best: (candidate: GemmaForgeCandidate, wall: Double)?
        outer: for candidate in candidates() {
            // Correctness gate: raw logits within 1e-4 rel of the
            // reference executor and identical greedy argmax.
            let (argmax, logits, _, _, _) = try executeGraph(
                graph, runner: runner, tokenIds: prompt, candidate: candidate,
                prepared: prep)
            guard argmax == refArgmax else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil, note: "argmax mismatch"))
                continue
            }
            guard logits.count == refLogitsAll.count,
                  logits.allSatisfy({ $0.isFinite }) else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil, note: "logit count/finiteness gate failed"))
                continue
            }
            var worst = 0.0
            for i in 0..<logits.count {
                worst = max(worst, abs(Double(logits[i] - refLogitsAll[i]))
                            / max(1.0, abs(Double(refLogitsAll[i]))))
            }
            guard worst < 1e-4 else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil,
                    note: String(format: "logits outside 1e-4 rel (worst %.3e)", worst)))
                continue
            }
            // Measured selection. Objective: end-to-end per-token WALL
            // time (the timed region is the executeGraphContinue call —
            // forward, CPU router work, argmax, and the necessary logits
            // readback; the post-hoc comparison against the reference is
            // outside the timer). GPU-busy time is recorded separately.
            // Every candidate replays the identical reference greedy
            // sequence at the true decode positions on a fresh prompt
            // context, gated per step (exact next token + full logits).
            _ = try executeGraph(graph, runner: runner, tokenIds: prompt,
                                 candidate: candidate, prepared: prep)
            var prev = refArgmax
            for w in 0..<warmupTokens {
                let (n, wl, _, _, _) = try executeGraphContinue(
                    graph, runner: runner, token: prev, candidate: candidate,
                    prepared: prep)
                let want = refDecodeLogits[w]
                guard n == refDecodeSeq[w],
                      wl.count == want.count, wl.allSatisfy({ $0.isFinite }),
                      zip(wl, want).allSatisfy({
                          abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                      }) else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil,
                        note: "warmup step \(w) outside reference"))
                    continue outer
                }
                prev = n
            }
            let replay = refDecodeSeq
            var samples: [Sample] = []
            var spent: Double = 0
            for run in 0..<timedTokens {
                let tok = replay.indices.contains(run) ? replay[run] : prev
                let t0 = DispatchTime.now()
                let (next, stepLogits, gpu, kernels, buffers) = try executeGraphContinue(
                    graph, runner: runner, token: tok, candidate: candidate,
                    prepared: prep)
                let wall = Double(DispatchTime.now().uptimeNanoseconds
                                  - t0.uptimeNanoseconds) / 1000
                // Per-step gate vs the independent reference logits —
                // outside the timed region.
                let want = refDecodeLogits[warmupTokens + run]
                // Full-array worst rel AFTER explicit length/finiteness
                // checks — allSatisfy would short-circuit and understate
                // the worst value (and even report 0 on token mismatch).
                var worstStep = 0.0
                let shapeOk = next == refDecodeSeq[warmupTokens + run]
                    && stepLogits.count == want.count
                    && stepLogits.allSatisfy({ $0.isFinite })
                    && want.allSatisfy({ $0.isFinite })
                if shapeOk {
                    for (a, b) in zip(stepLogits, want) {
                        worstStep = max(worstStep, abs(Double(a - b))
                                        / max(1.0, abs(Double(b))))
                    }
                }
                guard shapeOk && worstStep < 1e-4 else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil,
                        note: String(format: "decode step %d outside 1e-4 of reference (worst %.3e%@)",
                                     run, worstStep,
                                     next == refDecodeSeq[warmupTokens + run]
                                         ? "" : "; shape/token mismatch")))
                    continue outer
                }
                prev = next
                guard gpu.isFinite, gpu > 0, wall.isFinite, wall > 0 else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil, note: "timing unavailable"))
                    continue outer
                }
                samples.append(Sample(wall: wall, gpu: gpu,
                                      kernels: Double(kernels), buffers: Double(buffers)))
                spent += gpu
                if spent > budgetMicros { break }
            }
            guard !samples.isEmpty else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: true,
                    medianGpuMicrosPerToken: nil, note: "budget exhausted"))
                continue
            }
            let wallMed = median(samples.map(\.wall))
            let gpuMed = median(samples.map(\.gpu))
            let note = String(format: "valid; logits rel %.1e; %d timed tokens; wall %.0f us, gpu %.0f us, %.0f kernels, %.0f buffers/token",
                              worst, samples.count, wallMed, gpuMed,
                              median(samples.map(\.kernels)), median(samples.map(\.buffers)))
            results.append(GemmaForgeCandidateResult(
                candidate: candidate, correct: true,
                medianWallMicrosPerToken: wallMed,
                medianGpuMicrosPerToken: gpuMed,
                kernelsPerToken: median(samples.map(\.kernels)),
                buffersPerToken: median(samples.map(\.buffers)), note: note))
            if best == nil || wallMed < best!.wall {
                best = (candidate, wallMed)
            }
        }
        guard let winner = best, !results.isEmpty else {
            throw GemmaForgeError.noValidCandidate(results.map { "\($0.candidate): \($0.note)" })
        }
        // Baseline fallback with a noise threshold: the per-kernel
        // baseline is always in the measured set; an "optimized" winner
        // is selected only if it beats the baseline wall median by more
        // than noiseRelWall — negligible noise must not force an
        // optimization. Otherwise the baseline itself is selected.
        var selected = winner
        if winner.candidate.segmentBatching {
            if let base = results.first(where: {
                $0.correct && !$0.candidate.segmentBatching
                    && $0.medianWallMicrosPerToken != nil }) {
                if winner.wall > base.medianWallMicrosPerToken! * (1 - Self.noiseRelWall) {
                    selected = (candidate: base.candidate,
                                wall: base.medianWallMicrosPerToken!)
                }
            }
        }
        let selectedResult = results.first {
            $0.candidate == selected.candidate && $0.correct
        }!
        _ = decodeTokens
        let plan = try makePlan(graph: graph, graphDigest: graphDigest,
                                ggufDigest: ggufDigest, engineDigest: engineDigest,
                                hardware: hardware,
                                prompt: prompt, decodeTokens: decodeTokens,
                                selected: selected.candidate,
                                medianWall: selectedResult.medianWallMicrosPerToken!,
                                medianGpu: selectedResult.medianGpuMicrosPerToken!,
                                results: results,
                                expertCacheBudgetBytes: runner.expertCache?.budgetBytes)
        return (plan, results, selected.candidate)
    }

    /// Relative wall-time improvement a candidate must exceed over the
    /// baseline to be selected — a CHOSEN policy threshold (2%), not an
    /// empirically calibrated noise floor; the measured wall medians and
    /// per-sample counts are recorded in the plan so the choice can be
    /// audited.
    public static let noiseRelWall = 0.02

    /// SHA-256 hex of the pass-engine identity string.
    public static func engineDigest(_ identity: String) -> String {
        SHA256.hex(SHA256.hash(Array(identity.utf8)))
    }

    private static func isLowercaseHex64(_ s: String) -> Bool {
        s.count == 64 && s.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") }
    }

    /// Builds the plan with identity: graph digest, the cached model-file
    /// SHA-256 (ModelDigestCache; hashed once, refreshed only explicitly),
    /// pass-engine digest, measured hardware, verbatim
    /// toolchain, workload digest, and a planDigest covering all fields
    /// plus the selection. Malformed inputs are rejected here, not
    /// cached. The planDigest is an integrity check, not authentication.
    public static func makePlan(graph: GemmaGraphIR, graphDigest: String,
                                ggufDigest: String, engineDigest: String,
                                hardware: ForgeHardwareProfile,
                                prompt: [Int], decodeTokens: Int,
                                selected: GemmaForgeCandidate,
                                medianWall: Double, medianGpu: Double,
                                results: [GemmaForgeCandidateResult],
                                expertCacheBudgetBytes: Int? = nil) throws
            -> GemmaForgePlan {
        guard isLowercaseHex64(graphDigest), isLowercaseHex64(ggufDigest),
              isLowercaseHex64(engineDigest) else {
            throw GemmaForgeError.invalidPlan("identity digests must be 64 lowercase hex chars")
        }
        guard medianWall.isFinite, medianWall > 0,
              medianGpu.isFinite, medianGpu > 0 else {
            throw GemmaForgeError.invalidPlan("selected medians must be finite and positive")
        }
        let legal = Set(candidates())
        guard results.count == legal.count,
              Set(results.map(\.candidate)).count == results.count,
              results.allSatisfy({ legal.contains($0.candidate) }) else {
            throw GemmaForgeError.invalidPlan(
                "results must be exactly the legal candidate set, no duplicates")
        }
        guard let sel = results.first(where: { $0.candidate == selected }),
              sel.correct,
              let selWall = sel.medianWallMicrosPerToken,
              abs(selWall - medianWall) < 1e-9,
              let selGpu = sel.medianGpuMicrosPerToken,
              abs(selGpu - medianGpu) < 1e-9 else {
            throw GemmaForgeError.invalidPlan("plan selection inconsistent with results")
        }
        // The plan embeds the lowered executable schedule; its digest must
        // equal the deterministic lowering of this exact graph.
        let lowered = try GemmaSSA.lower(graph)
        let scheduleDigest = try GemmaSSA.digest(lowered)
        guard isLowercaseHex64(scheduleDigest) else {
            throw GemmaForgeError.invalidPlan("schedule digest malformed")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let body = GemmaForgePlan(
            schemaVersion: schemaVersion, graphDigest: graphDigest,
            ggufDigest: ggufDigest, engineDigest: engineDigest,
            hardware: hardware, toolchain: Forge.toolchainIdentifier,
            workloadDigest: workloadDigest(prompt),
            promptTokenCount: prompt.count, decodeTokens: decodeTokens,
            selected: selected,
            selectedMedianWallMicrosPerToken: medianWall,
            selectedMedianGpuMicrosPerToken: medianGpu,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            results: results, schedule: lowered,
            scheduleDigest: scheduleDigest, planDigest: "")
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
        return GemmaForgePlan(schemaVersion: body.schemaVersion,
                              graphDigest: body.graphDigest,
                              ggufDigest: body.ggufDigest,
                              engineDigest: body.engineDigest,
                              hardware: body.hardware,
                              toolchain: body.toolchain,
                              workloadDigest: body.workloadDigest,
                              promptTokenCount: body.promptTokenCount,
                              decodeTokens: body.decodeTokens,
                              selected: body.selected,
                              selectedMedianWallMicrosPerToken: medianWall,
                              selectedMedianGpuMicrosPerToken: medianGpu,
                              expertCacheBudgetBytes: expertCacheBudgetBytes,
                              results: body.results, schedule: lowered,
                              scheduleDigest: scheduleDigest, planDigest: digest)
    }

    /// Saves a plan atomically (Data.write .atomic replaces any existing
    /// cache file without leaving partial output).
    public static func savePlan(_ plan: GemmaForgePlan, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(plan)
        try bytes.write(to: url, options: [.atomic])
    }

    /// Rebuilds a plan with a different selected candidate (used by the
    /// full-workload legal-candidate fallback: candidates are tried in
    /// the short-selection timing order and the FIRST that passes the
    /// requested gate length is selected — not necessarily the fastest
    /// overall, since faster ones may be measured-rejected). The
    /// candidate must be a correct result in `plan.results`; its
    /// recorded medians become the plan's selection medians; the
    /// schedule and all identity digests are unchanged. The full-gate
    /// record (gate length + measured rejection reasons for failed
    /// candidates) is persisted so reuse cannot silently forget that a
    /// faster candidate was rejected (same-length re-gates merge with
    /// the prior history); the plan digest is recomputed over the new
    /// body.
    public static func reselect(_ plan: GemmaForgePlan,
                                to candidate: GemmaForgeCandidate,
                                fullGateTokens: Int,
                                fullGateRejections: [String]) throws
        -> GemmaForgePlan {
        guard let result = plan.results.first(where: { $0.candidate == candidate }),
              result.correct,
              let wall = result.medianWallMicrosPerToken,
              let gpu = result.medianGpuMicrosPerToken else {
            throw GemmaForgeError.invalidPlan(
                "reselect target is not a correct measured result")
        }
        // Re-gate history: when a cached plan is re-gated at the SAME
        // length, this run's (possibly empty) rejection list must not
        // erase the prior history — prior reasons are kept first and
        // any new unique reasons appended (stable order). A different
        // gate length starts a fresh record.
        let recordedRejections: [String]
        if let priorTokens = plan.fullGateTokens, priorTokens == fullGateTokens,
           let prior = plan.fullGateRejections {
            var seen = Set(prior)
            recordedRejections = prior
                + fullGateRejections.filter { seen.insert($0).inserted }
        } else {
            recordedRejections = fullGateRejections
        }
        let body = GemmaForgePlan(
            schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
            ggufDigest: plan.ggufDigest, engineDigest: plan.engineDigest,
            hardware: plan.hardware, toolchain: plan.toolchain,
            workloadDigest: plan.workloadDigest,
            promptTokenCount: plan.promptTokenCount,
            decodeTokens: plan.decodeTokens, selected: candidate,
            selectedMedianWallMicrosPerToken: wall,
            selectedMedianGpuMicrosPerToken: gpu,
            expertCacheBudgetBytes: plan.expertCacheBudgetBytes,
            results: plan.results, schedule: plan.schedule,
            scheduleDigest: plan.scheduleDigest, planDigest: "",
            fullGateTokens: fullGateTokens,
            fullGateRejections: recordedRejections,
            modelFileVerification: plan.modelFileVerification)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
        return GemmaForgePlan(schemaVersion: body.schemaVersion,
                              graphDigest: body.graphDigest,
                              ggufDigest: body.ggufDigest,
                              engineDigest: body.engineDigest,
                              hardware: body.hardware,
                              toolchain: body.toolchain,
                              workloadDigest: body.workloadDigest,
                              promptTokenCount: body.promptTokenCount,
                              decodeTokens: body.decodeTokens,
                              selected: body.selected,
                              selectedMedianWallMicrosPerToken: wall,
                              selectedMedianGpuMicrosPerToken: gpu,
                              expertCacheBudgetBytes: body.expertCacheBudgetBytes,
                              results: body.results, schedule: body.schedule,
                              scheduleDigest: body.scheduleDigest,
                              planDigest: digest,
                              fullGateTokens: fullGateTokens,
                              fullGateRejections: recordedRejections,
                              modelFileVerification: body.modelFileVerification)
    }

    /// Bounded-read plan loader: the file is read in chunks up to 1 MiB
    /// BEFORE any full-file allocation.
    public static func loadPlan(from url: URL) throws -> GemmaForgePlan {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let limit = 1 << 20
        var data = Data()
        while data.count <= limit {
            let chunk = try handle.read(upToCount: 1 << 16) ?? Data()
            if chunk.isEmpty { break }
            data.append(chunk)
        }
        return try parsePlan(data)
    }

    /// Parses plan bytes with a bounded size check.
    public static func parsePlan(_ data: Data) throws -> GemmaForgePlan {
        guard data.count <= 1 << 20 else {
            throw GemmaForgeError.invalidPlan("plan exceeds 1 MiB")
        }
        do { return try JSONDecoder().decode(GemmaForgePlan.self, from: data) }
        catch { throw GemmaForgeError.invalidPlan("\(error)") }
    }

    /// Validates a loaded plan against the CURRENT graph, the cached
    /// model-file SHA-256 and its cached-explicit-refresh policy,
    /// pass-engine identity, hardware, toolchain, workload, candidate
    /// legality, result consistency, and its own planDigest. A stale,
    /// foreign, tampered, or inconsistent plan is rejected and re-forged —
    /// never executed. planDigest is integrity, not authentication.
    public static func validatePlan(_ plan: GemmaForgePlan, graph: GemmaGraphIR,
                                    ggufDigest: String, engineDigest: String,
                                    hardware: ForgeHardwareProfile,
                                    prompt: [Int], decodeTokens: Int,
                                    expertCacheBudget: Int? = nil) throws {
        try validatePlanIdentity(plan, graph: graph, ggufDigest: ggufDigest,
                                 engineDigest: engineDigest, hardware: hardware,
                                 expertCacheBudget: expertCacheBudget)
        // ---- workload binding (the profiling workload the plan's
        // medians were measured on) ----
        guard plan.workloadDigest == workloadDigest(prompt),
              plan.promptTokenCount == prompt.count,
              plan.decodeTokens == decodeTokens else {
            throw GemmaForgeError.invalidPlan("workload mismatch")
        }
    }

    /// Identity + structural validation WITHOUT the workload binding —
    /// for serving VARYING prompts from a persisted artifact. Checks
    /// everything `validatePlan` checks except the prompt/decode/workload
    /// fields. The caller MUST disclose that the selected candidate was
    /// measured on the plan's own profiling workload (recorded in
    /// `plan.workloadDigest`/`promptTokenCount`/`decodeTokens`), not on
    /// the served prompt — no fastest-on-all-prompts claim is implied.
    public static func validatePlanIdentity(_ plan: GemmaForgePlan,
                                            graph: GemmaGraphIR,
                                            ggufDigest: String,
                                            engineDigest: String,
                                            hardware: ForgeHardwareProfile,
                                            expertCacheBudget: Int? = nil) throws {
        guard plan.schemaVersion == schemaVersion else {
            throw GemmaForgeError.invalidPlan("schema \(plan.schemaVersion) != \(schemaVersion)")
        }
        // Model-file policy: the plan must record the cached explicit-refresh
        // policy (decoding already rejects a missing or retired value).
        // Full-hash and unverified plans fail the schema gate above and must
        // be re-forged — never re-tagged.
        guard plan.modelFileVerification == ModelFileIdentity.policy else {
            throw GemmaForgeError.invalidPlan("model-file verification policy mismatch")
        }
        // Structural validity of the LOADED plan (not just plans this
        // process authored): digest syntax, positive finite medians,
        // exactly the legal candidate set with no duplicates.
        guard isLowercaseHex64(plan.graphDigest),
              isLowercaseHex64(plan.ggufDigest),
              isLowercaseHex64(plan.engineDigest),
              isLowercaseHex64(plan.planDigest) else {
            throw GemmaForgeError.invalidPlan("plan digests malformed")
        }
        guard plan.selectedMedianWallMicrosPerToken.isFinite,
              plan.selectedMedianWallMicrosPerToken > 0,
              plan.selectedMedianGpuMicrosPerToken.isFinite,
              plan.selectedMedianGpuMicrosPerToken > 0 else {
            throw GemmaForgeError.invalidPlan("plan medians not finite/positive")
        }
        // Recorded PROFILING metadata is validated structurally here
        // (hex syntax, positive counts, checked context bounds) even
        // though the workload BINDING itself is the caller's policy.
        guard isLowercaseHex64(plan.workloadDigest),
              isLowercaseHex64(plan.scheduleDigest) else {
            throw GemmaForgeError.invalidPlan("recorded workload/schedule digests malformed")
        }
        guard plan.promptTokenCount > 0, plan.decodeTokens > 0,
              plan.promptTokenCount <= Int.max - plan.decodeTokens,
              plan.promptTokenCount + plan.decodeTokens <= Self.maxBoundedContext else {
            throw GemmaForgeError.invalidPlan("recorded workload counts invalid")
        }
        let legalSet = Set(candidates())
        guard plan.results.count == legalSet.count,
              Set(plan.results.map(\.candidate)).count == plan.results.count,
              plan.results.allSatisfy({ legalSet.contains($0.candidate) }) else {
            throw GemmaForgeError.invalidPlan(
                "plan results not exactly the legal candidate set")
        }
        let currentGraphDigest = try GemmaGraph.digest(graph)
        guard plan.graphDigest == currentGraphDigest else {
            throw GemmaForgeError.invalidPlan("graph digest mismatch")
        }
        guard !ggufDigest.isEmpty, plan.ggufDigest == ggufDigest else {
            throw GemmaForgeError.invalidPlan("GGUF content digest mismatch")
        }
        guard !engineDigest.isEmpty, plan.engineDigest == engineDigest else {
            throw GemmaForgeError.invalidPlan("pass-engine identity mismatch")
        }
        guard plan.hardware == hardware else {
            throw GemmaForgeError.invalidPlan("hardware profile mismatch")
        }
        guard plan.toolchain == Forge.toolchainIdentifier else {
            throw GemmaForgeError.invalidPlan("toolchain mismatch")
        }
        // The plan's expert-cache budget is an engine identity input
        // (reflected in engineDigest): a plan forged resident must not
        // execute in streaming mode and vice versa.
        guard plan.expertCacheBudgetBytes == expertCacheBudget else {
            throw GemmaForgeError.invalidPlan(
                "expert cache budget mismatch: plan \(plan.expertCacheBudgetBytes.map(String.init) ?? "resident") vs current \(expertCacheBudget.map(String.init) ?? "resident")")
        }
        guard candidates().contains(plan.selected) else {
            throw GemmaForgeError.invalidPlan("selected candidate not legal")
        }
        guard let sel = plan.results.first(where: { $0.candidate == plan.selected }),
              sel.correct,
              let selWall = sel.medianWallMicrosPerToken,
              abs(selWall - plan.selectedMedianWallMicrosPerToken) < 1e-9,
              sel.medianGpuMicrosPerToken != nil,
              abs(sel.medianGpuMicrosPerToken! - plan.selectedMedianGpuMicrosPerToken) < 1e-9
        else {
            throw GemmaForgeError.invalidPlan("selected result inconsistent with plan")
        }
        // Embedded schedule: must be present, dataflow-valid, and EXACTLY
        // the deterministic lowering of the current graph. The runtime
        // executes these artifact operands after this check.
        guard let schedule = plan.schedule else {
            throw GemmaForgeError.invalidPlan("plan carries no lowered schedule")
        }
        let lowered = try GemmaSSA.lower(graph)
        guard schedule == lowered,
              plan.scheduleDigest == (try GemmaSSA.digest(schedule)),
              plan.scheduleDigest == (try GemmaSSA.digest(lowered)) else {
            throw GemmaForgeError.invalidPlan("embedded schedule tampered/mismatched")
        }
        // Full-gate record bounds and pairing: the gate length is a
        // bounded generation count, and the token/rejection fields must
        // be present together (a gate length without its rejection
        // record could silently hide measured rejections).
        if let tokens = plan.fullGateTokens {
            guard tokens >= 2, tokens <= 64,
                  plan.promptTokenCount + tokens <= maxBoundedContext,
                  plan.fullGateRejections != nil else {
                throw GemmaForgeError.invalidPlan(
                    "full-gate record malformed (tokens \(plan.fullGateTokens ?? -1), "
                    + "rejections \(plan.fullGateRejections == nil ? "missing" : "present"))")
            }
        } else if plan.fullGateRejections != nil {
            throw GemmaForgeError.invalidPlan(
                "full-gate rejections present without a gate length")
        }
        // planDigest self-consistency: the digest covers all fields.
        let body = GemmaForgePlan(
            schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
            ggufDigest: plan.ggufDigest, engineDigest: plan.engineDigest,
            hardware: plan.hardware, toolchain: plan.toolchain,
            workloadDigest: plan.workloadDigest,
            promptTokenCount: plan.promptTokenCount, decodeTokens: plan.decodeTokens,
            selected: plan.selected,
            selectedMedianWallMicrosPerToken: plan.selectedMedianWallMicrosPerToken,
            selectedMedianGpuMicrosPerToken: plan.selectedMedianGpuMicrosPerToken,
            expertCacheBudgetBytes: plan.expertCacheBudgetBytes,
            results: plan.results, schedule: plan.schedule,
            scheduleDigest: plan.scheduleDigest, planDigest: "",
            fullGateTokens: plan.fullGateTokens,
            fullGateRejections: plan.fullGateRejections,
            modelFileVerification: plan.modelFileVerification)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
        guard plan.planDigest == digest else {
            throw GemmaForgeError.invalidPlan("planDigest tampered")
        }
    }

    /// The bounded first-milestone context every Gemma plan is forged
    /// under (mirrors Gemma4Config.maxContext's 4096 cap).
    public static let maxBoundedContext = 4096

    /// Recomputes the planDigest over the plan's CURRENT fields (all
    /// fields except planDigest itself). Test tooling uses this to
    /// isolate structural gates from planDigest staleness after a
    /// mutation; it is integrity bookkeeping, not authentication.
    public static func resignPlan(_ plan: GemmaForgePlan) throws -> GemmaForgePlan {
        func rebuilt(digest: String) -> GemmaForgePlan {
            GemmaForgePlan(
                schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
                ggufDigest: plan.ggufDigest, engineDigest: plan.engineDigest,
                hardware: plan.hardware, toolchain: plan.toolchain,
                workloadDigest: plan.workloadDigest,
                promptTokenCount: plan.promptTokenCount,
                decodeTokens: plan.decodeTokens, selected: plan.selected,
                selectedMedianWallMicrosPerToken: plan.selectedMedianWallMicrosPerToken,
                selectedMedianGpuMicrosPerToken: plan.selectedMedianGpuMicrosPerToken,
                expertCacheBudgetBytes: plan.expertCacheBudgetBytes,
                results: plan.results, schedule: plan.schedule,
                scheduleDigest: plan.scheduleDigest, planDigest: digest,
                fullGateTokens: plan.fullGateTokens,
                fullGateRejections: plan.fullGateRejections,
                modelFileVerification: plan.modelFileVerification)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(rebuilt(digest: "")))))
        return rebuilt(digest: digest)
    }

    public static func workloadDigest(_ tokens: [Int]) -> String {
        var material = [UInt8]()
        for t in tokens {
            withUnsafeBytes(of: UInt32(t).littleEndian) { material.append(contentsOf: $0) }
        }
        return SHA256.hex(SHA256.hash(material))
    }
}
