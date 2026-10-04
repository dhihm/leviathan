// Phase 15 — Q4 launch-geometry pass (bit-exact by construction).
//
// The phase-14 measured ranking (root profile: g4_q4_matvec 32.0% of
// decode GPU, 109.7 ms/token over 6355 calls) makes the ORDINARY dense
// g4_q4_matvec launch the next target — NOT more MoE batching. The
// incumbent generic dispatch launches ceil(rows / min(rows, 1024))
// threadgroups (2-8 groups for the hot shapes on a 32-core GPU). This
// pass changes ONLY the launch width of that ONE allowlisted kernel:
// the SAME compiled pipeline, the SAME per-thread MSL arithmetic (one
// thread per row, serial in-row accumulation, no barriers), no weight
// repacking or shadow buffers. Geometry alone does not prove occupancy;
// selection is decided by MEASURED paired timing, never by the geometry
// hypothesis.
//
// Baseline fairness: every arm (reference, gates, timing) runs the
// ACCEPTED E4_R32 MoE lowering (freshly compiled, source SHA pinned to
// the accepted phase-13 emission) plus the fresh all-layer model
// binding, on the SAME binary/settings/contexts — the serial phase-13
// incumbent is NOT the comparator and the prior 21.9% gain is NOT
// recounted. The candidate changes ONLY the Q4 launch width.

import Foundation
import Metal

// MARK: - Schedule

/// One launch-geometry candidate: the threadgroup WIDTH for generic
/// g4_q4_matvec launches (groups = ceil(rows / width); the kernel's
/// `gid >= rows` guard handles tails).
public struct Q4LaunchSchedule: Codable, Equatable, Hashable, Sendable {
    public let width: Int

    public init(width: Int) { self.width = width }

    public var label: String { "g4_q4_matvec@w\(width)" }

    /// Positive, power of two (hardware-derived ladder), within the
    /// ACTUAL pipeline's threadgroup maximum.
    public func isLegal(pipelineMaxThreads: Int) -> Bool {
        width > 0 && width & (width - 1) == 0 && width <= pipelineMaxThreads
    }
}

public enum Q4LaunchDeriver {
    public static let maxCandidates = 24

    /// Hardware-derived bounded ladder: executionWidth x {1,2,4,8} plus
    /// the incumbent control width (the pipeline maximum the default
    /// min(grid, max) path uses for the hot shapes). Deduplicated,
    /// legality-filtered, bounded. The CONTROL is measured and recorded
    /// but never a selection finalist (no slower fake baseline).
    public static func derive(hardware: ForgeHardwareProfile,
                              pipelineMaxThreads: Int) throws -> [Q4LaunchSchedule] {
        guard hardware.threadExecutionWidth > 0,
              pipelineMaxThreads > 0,
              hardware.threadExecutionWidth <= pipelineMaxThreads else {
            throw Q4LoweringError.invalidSchedule(
                "launch-geometry caps invalid: width \(hardware.threadExecutionWidth), "
                + "pipeline max \(pipelineMaxThreads)")
        }
        let w = hardware.threadExecutionWidth
        var widths = Set<Int>()
        for m in [1, 2, 4, 8] {
            let (p, o) = w.multipliedReportingOverflow(by: m)
            if !o { widths.insert(p) }
        }
        widths.insert(pipelineMaxThreads)  // incumbent control
        var out = widths
            .filter { Q4LaunchSchedule(width: $0).isLegal(pipelineMaxThreads: pipelineMaxThreads) }
            .sorted { $0 < $1 }
            .map { Q4LaunchSchedule(width: $0) }
        if out.count > maxCandidates { out = Array(out.prefix(maxCandidates)) }
        guard !out.isEmpty else {
            throw Q4LoweringError.invalidSchedule("no legal launch widths")
        }
        return out
    }

    /// The incumbent control schedule (the default min(grid, max) width
    /// for hot shapes with rows <= max, i.e. the pipeline maximum).
    public static func control(pipelineMaxThreads: Int) -> Q4LaunchSchedule {
        Q4LaunchSchedule(width: pipelineMaxThreads)
    }
}

// MARK: - Hot shapes (derived from the ACTUAL GGUF, never invented)

/// One representative hot 2-D Q4_0 tensor. The component launches the
/// FULL outDim rows against the runner's resident zero-copy weight
/// mapping (no shadow copy); the independent FP64 oracle is bounded to
/// the first min(outDim, 64) rows (a small prefix read for the oracle
/// only).
public struct Q4LaunchHotShape: Codable, Equatable, Sendable {
    public let tensor: String
    public let inDim: Int
    public let outDim: Int
    public let oracleRows: Int
    public let oracleBytes: Int
}

public enum Q4LaunchHotShapes {
    public static let maxOracleRows = 64
    /// Bounded layer sample: one local, one global, one mid layer — the
    /// distinct attention/shared-FF geometries of this architecture.
    public static let sampledLayers = [0, 5, 15]
    public static let tensorKinds = ["attn_q", "attn_k", "attn_v", "attn_output",
                                     "ffn_gate", "ffn_up", "ffn_down"]

    /// One REQUIRED-tensor directory lookup: returns nil ONLY when the
    /// directory has no entry of exactly that name. Any other failure
    /// must throw; it is never silently mapped to "absent".
    public typealias TensorLookup = (String) throws -> (dims: [Int], ggmlType: UInt32)?

    /// Production wrapper: layer globalness comes from the VALIDATED
    /// architecture metadata (Gemma4Config) of the ACTUAL file; only
    /// GGUFError.missingTensor for the exact requested name maps to
    /// "absent", every other directory error propagates.
    public static func derive(file: GGUFFile, layerCount: Int) throws -> [Q4LaunchHotShape] {
        let config = try Gemma4Config(file: file)
        guard config.layerCount == layerCount else {
            throw Q4LoweringError.invalidSchedule(
                "hot shapes: layerCount \(layerCount) disagrees with the file config \(config.layerCount)")
        }
        return try resolve(layerCount: layerCount,
                           isGlobalLayer: { config.isGlobalLayer($0) },
                           lookup: { name in
            do {
                let info = try file.tensor(name)
                return (dims: info.dims, ggmlType: info.ggmlType)
            } catch GGUFError.missingTensor(let missing) where missing == name {
                return nil
            }
        })
    }

    /// Pure metadata resolver (unit-testable with synthetic lookups).
    /// Every sampled layer REQUIRES attn_q, attn_k, attn_output, ffn_gate,
    /// ffn_up and ffn_down. attn_v is REQUIRED on LOCAL layers; GLOBAL
    /// layers share V with K (v_eq_k), so attn_v must be ABSENT there
    /// (the K projection is already sampled) and a present global attn_v
    /// contradicts the validated metadata and fails closed. A missing
    /// LOCAL attn_v is never reinterpreted as shared. Only 2-D Q4_0
    /// tensors with a 32-multiple inDim are eligible.
    public static func resolve(layerCount: Int,
                               isGlobalLayer: (Int) -> Bool,
                               lookup: TensorLookup) throws -> [Q4LaunchHotShape] {
        guard layerCount > 0 else {
            throw Q4LoweringError.invalidSchedule("hot shapes: layerCount \(layerCount) invalid")
        }
        var out = [Q4LaunchHotShape]()
        for layer in sampledLayers {
            guard layer >= 0, layer < layerCount else { continue }
            let global = isGlobalLayer(layer)
            for kind in tensorKinds {
                let name = "blk.\(layer).\(kind).weight"
                let entry = try lookup(name)
                if kind == "attn_v" && global {
                    guard entry == nil else {
                        throw Q4LoweringError.invalidSchedule(
                            "global layer \(layer) carries \(name) although V is shared with K (v_eq_k)")
                    }
                    continue
                }
                guard let info = entry else {
                    throw Q4LoweringError.invalidSchedule(
                        "required hot-shape tensor \(name) missing on "
                        + (global ? "global" : "local") + " layer \(layer)")
                }
                guard GGMLTensorType(rawValue: info.ggmlType) == .q4_0,
                      info.dims.count == 2 else {
                    throw Q4LoweringError.invalidSchedule(
                        "hot shape \(name) is not a 2-D Q4_0 tensor")
                }
                let inDim = info.dims[0], outDim = info.dims[1]
                guard inDim > 0, outDim > 0, inDim % 32 == 0 else {
                    throw Q4LoweringError.invalidSchedule("hot shape \(name) dims invalid")
                }
                let oracleRows = min(outDim, maxOracleRows)
                out.append(.init(tensor: name, inDim: inDim, outDim: outDim,
                                 oracleRows: oracleRows,
                                 oracleBytes: oracleRows * (inDim / 32) * 18))
            }
        }
        guard !out.isEmpty else {
            throw Q4LoweringError.invalidSchedule("no hot shapes derived")
        }
        let total = out.reduce(0) { $0 + $1.oracleBytes }
        guard total <= 64 << 20 else {
            throw Q4LoweringError.invalidSchedule(
                "oracle payload \(total) exceeds the 64 MiB bound")
        }
        return out
    }
}

// MARK: - Component records

public struct Q4LaunchComponentResult: Codable, Equatable, Sendable {
    public let schedule: Q4LaunchSchedule
    public let accepted: Bool
    public let bitIdenticalToIncumbent: Bool?
    public let worstRelToOracle: Double?
    public let screenMicrosPerShape: [Double]?
    public let note: String
}

// MARK: - Gate record + policy (mirrors the phase-13 contract, keyed on
// the launch width; separate small validators, phase-13 behavior
// untouched).

public struct Q4LaunchGateRecord: Codable, Equatable, Sendable {
    public let gatePolicy: String
    public static let policyVersion = "q4launch-strict3x32-3fixture-2pct-v1"
    public static let executionPathSSA = "gemma-forge-ssa-v1"
    public let executionPath: String
    public let diagnostic: Bool
    public let genTokens: Int
    public let timedTokens: Int
    public let fixtures: [Q4FixtureRecord]
    public let results: [Q4LaunchCandidateResult]
    public let baselineWallPerFixture: [Double]
    public let performanceSelected: Bool
    public let policyNote: String
}

public struct Q4LaunchCandidateResult: Codable, Equatable, Sendable {
    public let schedule: Q4LaunchSchedule
    public let fixturesPassed: Int
    public let fixturesFailed: Int
    public let worstRel: Double?
    public let gateFailReason: String?
    /// Persisted measured counter proofs (per gate pass, summed over the
    /// fixture replays): the DERIVED eligible g4_q4_matvec population,
    /// the candidate-width applications (must equal eligible), the MoE
    /// per-stage dispatches, and the ordered merges. Nil when the gate
    /// failed before counters could be proven.
    public var eligibleDispatches: Int?
    public var appliedDispatches: Int?
    public var moeStageDispatches: Int?
    public var mergeDispatches: Int?
    public var timingReps: [Q4TimingRepetition]?
}

public enum Q4LaunchGatePolicy {
    public static let productionFixtures = ["en-chat", "kr-chat", "raw-hello"]
    public static let productionGenTokens = 32
    public static let productionTimedTokens = 31

    public static func shapeValid(_ gate: Q4LaunchGateRecord,
                                  schedule: Q4LaunchSchedule,
                                  binding: MoEModelBinding) -> Bool {
        guard gate.gatePolicy == Q4LaunchGateRecord.policyVersion,
              gate.executionPath == Q4LaunchGateRecord.executionPathSSA,
              !gate.diagnostic,
              gate.genTokens == productionGenTokens,
              gate.timedTokens == productionTimedTokens,
              gate.fixtures.map(\.name) == productionFixtures,
              gate.fixtures.allSatisfy({ $0.gatedSteps == productionGenTokens
                  && $0.promptTokenCount > 0 }),
              !gate.results.isEmpty,
              gate.results.count <= Q4LaunchDeriver.maxCandidates,
              Set(gate.results.map(\.schedule)).count == gate.results.count,
              gate.results.filter({ $0.schedule == schedule }).count == 1,
              let sel = gate.results.first(where: { $0.schedule == schedule }),
              sel.fixturesPassed == gate.fixtures.count,
              sel.fixturesFailed == 0,
              sel.gateFailReason == nil,
              let worst = sel.worstRel, worst.isFinite, worst >= 0, worst < 1e-4,
              // Persisted geometry/MoE counter proofs, checked against the
              // BINDING-derived expected counts (never trusted booleans):
              // nonvacuous eligible population, applied == eligible, and
              // stages/merges equal to the chunk arithmetic over the
              // ACTUAL recorded fixture forwards.
              let eligible = sel.eligibleDispatches, eligible > 0,
              sel.appliedDispatches == eligible,
              let stages = sel.moeStageDispatches,
              let merges = sel.mergeDispatches,
              stages == expectedStageCount(gate: gate, binding: binding),
              merges == expectedMergeCount(gate: gate, binding: binding),
              let reps = sel.timingReps, reps.count == 3,
              reps.enumerated().allSatisfy({
                  $0.offset % 2 == ($0.element.candidateFirst ? 1 : 0) }),
              reps.allSatisfy({ rep in
                  rep.candidate.count == gate.fixtures.count
                      && rep.incumbent.count == gate.fixtures.count
                      && zip(rep.candidate, rep.incumbent).allSatisfy { c, i in
                          c.timedCount == productionTimedTokens
                              && i.timedCount == productionTimedTokens
                              && c.startIdx == 0 && i.startIdx == 0
                              && c.walls.count == productionTimedTokens
                              && i.walls.count == productionTimedTokens
                              && c.gpus.count == productionTimedTokens
                              && i.gpus.count == productionTimedTokens
                              && c.walls.allSatisfy({ $0.isFinite && $0 > 0 })
                              && i.walls.allSatisfy({ $0.isFinite && $0 > 0 })
                              && c.gpus.allSatisfy({ $0.isFinite && $0 >= 0 })
                              && i.gpus.allSatisfy({ $0.isFinite && $0 >= 0 })
                      }
              }) else { return false }
        return true
    }

    /// Binding-derived expected per-stage dispatch count over the
    /// recorded fixture forwards (one dispatch per chunk; E4 baseline).
    /// CHECKED arithmetic throughout — hostile JSON (Int.max prompt
    /// counts or binding fields) yields nil, never a trapped process.
    public static func expectedStageCount(gate: Q4LaunchGateRecord,
                                   binding: MoEModelBinding) -> Int? {
        guard binding.topK > 0, binding.topK <= Q4LaunchBoundCheck.maxTopK,
              binding.layerCount > 0,
              binding.layerCount <= Q4LaunchBoundCheck.maxLayerCount,
              let forwards = expectedForwards(gate) else { return nil }
        let chunks = (binding.topK + 3) / 4
        let (p1, o1) = chunks.multipliedReportingOverflow(by: binding.layerCount)
        let (p2, o2) = p1.multipliedReportingOverflow(by: forwards)
        guard !o1, !o2 else { return nil }
        return p2
    }

    /// Binding-derived expected ordered-merge count (topK per layer per
    /// forward). CHECKED arithmetic; nil on invalid/overflowing inputs.
    public static func expectedMergeCount(gate: Q4LaunchGateRecord,
                                   binding: MoEModelBinding) -> Int? {
        guard binding.topK > 0, binding.topK <= Q4LaunchBoundCheck.maxTopK,
              binding.layerCount > 0,
              binding.layerCount <= Q4LaunchBoundCheck.maxLayerCount,
              let forwards = expectedForwards(gate) else { return nil }
        let (p1, o1) = binding.topK.multipliedReportingOverflow(by: binding.layerCount)
        let (p2, o2) = p1.multipliedReportingOverflow(by: forwards)
        guard !o1, !o2 else { return nil }
        return p2
    }

    /// Checked total forwards over the recorded fixtures (shared by the
    /// two expected-count helpers); nil on invalid/overflowing inputs.
    static func expectedForwards(_ gate: Q4LaunchGateRecord) -> Int? {
        guard gate.fixtures.allSatisfy({
            $0.promptTokenCount > 0
                && $0.promptTokenCount <= Q4LaunchBoundCheck.maxPromptTokens }) else {
            return nil
        }
        var forwards = 0
        for f in gate.fixtures {
            let (s1, o1) = f.promptTokenCount.addingReportingOverflow(
                productionGenTokens - 1)
            let (s2, o2) = forwards.addingReportingOverflow(s1)
            guard !o1, !o2, s1 > 0 else { return nil }
            forwards = s2
        }
        return forwards
    }

    /// Bounded inputs for the checked expected-count arithmetic (a
    /// hostile decoded artifact cannot trap the validator).
    public enum Q4LaunchBoundCheck {
        public static let maxTopK = 64
        /// Aligned with MoEModelBinding's own layer bound (128) — the
        /// checked arithmetic and the binding decoder agree.
        public static let maxLayerCount = 128
        public static let maxPromptTokens = 1 << 20
    }

    /// 2% policy recomputed from the recorded arms; every repetition's
    /// arm sum must be finite and positive (an overflowing rep cannot
    /// hide behind the median).
    public static func recomputedSelection(_ gate: Q4LaunchGateRecord,
                                           schedule: Q4LaunchSchedule,
                                           binding: MoEModelBinding) -> Bool? {
        guard shapeValid(gate, schedule: schedule, binding: binding),
              let sel = gate.results.first(where: { $0.schedule == schedule }),
              let reps = sel.timingReps else { return nil }
        let candSums = reps.map { armSum($0.candidate) }
        let baseSums = reps.map { armSum($0.incumbent) }
        guard candSums.allSatisfy({ $0.isFinite && $0 > 0 }),
              baseSums.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        return LoweringSelect.median(candSums)
            < LoweringSelect.median(baseSums) * (1 - GemmaForge.noiseRelWall)
    }

    public static func productionValid(_ gate: Q4LaunchGateRecord,
                                       schedule: Q4LaunchSchedule,
                                       binding: MoEModelBinding) -> Bool {
        guard let recomputed = recomputedSelection(gate, schedule: schedule,
                                                   binding: binding),
              gate.performanceSelected == recomputed, recomputed else { return false }
        return true
    }

    static func armSum(_ arm: [Q4TimingArmFixture]) -> Double {
        arm.map { LoweringSelect.median($0.walls) }.reduce(0, +)
    }
}

// MARK: - Composite artifact

/// The phase-15 composite: Q4 launch geometry ON TOP of the accepted
/// E4_R32 MoE baseline. Identity binds the cached model-file SHA-256
/// (ModelDigestCache: hashed once, refreshed only explicitly), executable, hardware,
/// toolchain, the primitive MSL source, the launch width, AND the whole
/// E4 baseline identity (region graph, emitted source SHA, schedule,
/// immutable model binding). A stale phase-13 MoE artifact is NEVER
/// re-signed or re-bound: the E4 source is RE-COMPILED here and must
/// hash equal to the recorded accepted emission.
public struct Q4LaunchArtifact: Codable, Equatable, Sendable {
    /// 3: cached model-file SHA-256 with the cached-explicit-refresh-v1
    /// policy; schema-1 and schema-2 artifacts are rejected and must be
    /// re-selected.
    public static let schema = 3
    /// The accepted phase-13 E4_R32 emitted-source SHA (pinned).
    public static let baselineMoeSourceSha256 =
        "c6f7a659cc0254950341d1acf360866e358792bcdb8c46ffb5222ddc24b1b5a8"

    public let artifactSchema: Int
    public let codegenVersion: String
    public let schedule: Q4LaunchSchedule
    public let kernelName: String
    /// SHA-256 of the runner's ACTUAL embedded MSL (the primitive whose
    /// launch geometry changed).
    public let primitiveSourceSha256: String
    public let baselineMoeSourceSha256: String
    public let baselineMoeSchedule: MoESchedule
    public let baselineMoeBinding: MoEModelBinding
    public let regionGraphSha256: String
    public let hardware: ForgeHardwareProfile
    public let toolchain: String
    /// Actual SHA-256 of the model file from the shared ModelDigestCache
    /// (computed once from the bytes, reused without re-reading, refreshed
    /// only explicitly). Historical, not a fresh per-run verification.
    public let ggufDigest: String
    /// Model-file digest policy (schema 3): required on decode, covered by
    /// artifactDigest; only cached-explicit-refresh-v1 is representable.
    public var modelFileVerification: ModelFileVerificationPolicy = ModelFileIdentity.policy
    public let executableDigest: String
    public let graphDigest: String
    public let hotShapes: [Q4LaunchHotShape]
    public let componentResults: [Q4LaunchComponentResult]
    public let incumbentWorstRelToOracle: Double
    public let modelGate: Q4LaunchGateRecord?
    public let artifactDigest: String

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: [.atomic])
    }

    public static func load(from url: URL) throws -> Q4LaunchArtifact {
        guard let data = try? Data(contentsOf: url) else {
            throw Q4LoweringError.invalidArtifact("unreadable artifact \(url.path)")
        }
        guard data.count <= 4 << 20 else {
            throw Q4LoweringError.invalidArtifact("artifact exceeds 4 MiB")
        }
        do { return try JSONDecoder().decode(Q4LaunchArtifact.self, from: data) }
        catch { throw Q4LoweringError.invalidArtifact("decode: \(error)") }
    }

    /// Canonical body digest (every field except artifactDigest itself):
    /// recomputed at validation — a tampered body fails the self-digest
    /// check before anything else runs.
    public static func bodyDigest(_ body: Q4LaunchArtifact) throws -> String {
        let stripped = Q4LaunchArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            schedule: body.schedule, kernelName: body.kernelName,
            primitiveSourceSha256: body.primitiveSourceSha256,
            baselineMoeSourceSha256: body.baselineMoeSourceSha256,
            baselineMoeSchedule: body.baselineMoeSchedule,
            baselineMoeBinding: body.baselineMoeBinding,
            regionGraphSha256: body.regionGraphSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest, graphDigest: body.graphDigest,
            hotShapes: body.hotShapes, componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            modelGate: body.modelGate, artifactDigest: "")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(stripped)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    static func withDigest(_ body: Q4LaunchArtifact) throws -> Q4LaunchArtifact {
        let d = try Q4LaunchArtifact.bodyDigest(body)
        return Q4LaunchArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            schedule: body.schedule, kernelName: body.kernelName,
            primitiveSourceSha256: body.primitiveSourceSha256,
            baselineMoeSourceSha256: body.baselineMoeSourceSha256,
            baselineMoeSchedule: body.baselineMoeSchedule,
            baselineMoeBinding: body.baselineMoeBinding,
            regionGraphSha256: body.regionGraphSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest, graphDigest: body.graphDigest,
            hotShapes: body.hotShapes, componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            modelGate: body.modelGate, artifactDigest: d)
    }

    /// Fail-closed identity validation against the CURRENT runner, file,
    /// and freshly derived facts. The E4 baseline source is re-emitted
    /// and must hash equal; the recorded HARDWARE must equal the actual
    /// measured profile; the recorded graph digest must equal the
    /// SUPPLIED actual graph's digest (not a syntax check); the
    /// artifact's self-digest is RECOMPUTED and must match. The width
    /// must be legal on the actual compiled pipeline; the primitive
    /// source must be the runner's CURRENT shader source.
    public func validate(region currentRegion: MoERegionIR,
                         pipelineMaxThreads: Int,
                         hardware actualHardware: ForgeHardwareProfile,
                         graphDigest actualGraphDigest: String,
                         ggufDigest: String,
                         executableDigest: String) throws {
        try currentRegion.validate()
        guard artifactSchema == Self.schema else {
            throw Q4LoweringError.invalidArtifact("schema \(artifactSchema)")
        }
        // Self-digest first: a tampered body can never pass as a whole.
        guard try Q4LaunchArtifact.bodyDigest(self) == artifactDigest,
              moeIsLowercaseHex64(artifactDigest) else {
            throw Q4LoweringError.invalidArtifact(
                "artifact self-digest mismatch (tampered body)")
        }
        guard hardware == actualHardware else {
            throw Q4LoweringError.invalidArtifact(
                "hardware mismatch: recorded \(hardware.deviceName) "
                + "vs actual \(actualHardware.deviceName)")
        }
        guard graphDigest == actualGraphDigest else {
            throw Q4LoweringError.invalidArtifact(
                "graph digest mismatch — the actual Gemma graph differs "
                + "from the recorded selection")
        }
        guard codegenVersion == Q4LaunchForge.codegenVersion else {
            throw Q4LoweringError.invalidArtifact("codegen stale")
        }
        guard kernelName == Gemma4Runner.q4LaunchKernel,
              schedule.isLegal(pipelineMaxThreads: pipelineMaxThreads) else {
            throw Q4LoweringError.invalidArtifact(
                "launch width \(schedule.width) illegal on this pipeline")
        }
        let msl = SHA256.hex(SHA256.hash(Array(Gemma4Runner.shaderSource.utf8)))
        guard primitiveSourceSha256 == msl else {
            throw Q4LoweringError.invalidArtifact(
                "primitive MSL source changed — fresh gates required")
        }
        guard baselineMoeSourceSha256 == Self.baselineMoeSourceSha256,
              baselineMoeSchedule == MoESchedule(batch: 4, rowsPerGroup: 32),
              regionGraphSha256 == currentRegion.graphSha256 else {
            throw Q4LoweringError.invalidArtifact("baseline E4 identity mismatch")
        }
        let reemitted = try MoEKernelEmitter.emit(region: currentRegion,
                                                  schedule: baselineMoeSchedule)
        guard MoEKernelEmitter.sourcesSha256([reemitted]) == baselineMoeSourceSha256 else {
            throw Q4LoweringError.invalidArtifact("baseline E4 emission drifted")
        }
        guard toolchain == Forge.toolchainIdentifier,
              ggufDigest == self.ggufDigest,
              modelFileVerification == ModelFileIdentity.policy,
              executableDigest == self.executableDigest,
              moeIsLowercaseHex64(ggufDigest), moeIsLowercaseHex64(executableDigest),
              moeIsLowercaseHex64(graphDigest),
              moeIsLowercaseHex64(primitiveSourceSha256) else {
            throw Q4LoweringError.invalidArtifact("identity digest mismatch")
        }
        guard hardware.maxTotalThreadsPerThreadgroup > 0,
              hardware.threadExecutionWidth > 0 else {
            throw Q4LoweringError.invalidArtifact("hardware record invalid")
        }
        guard !componentResults.isEmpty,
              componentResults.count <= Q4LaunchDeriver.maxCandidates,
              Set(componentResults.map(\.schedule)).count == componentResults.count,
              incumbentWorstRelToOracle.isFinite,
              incumbentWorstRelToOracle >= 0,
              incumbentWorstRelToOracle < 1e-4 else {
            throw Q4LoweringError.invalidArtifact("component records inconsistent")
        }
        guard let sel = componentResults.first(where: {
                  $0.schedule == schedule && $0.accepted }),
              sel.bitIdenticalToIncumbent == true,
              let rel = sel.worstRelToOracle, rel.isFinite, rel >= 0, rel < 1e-4 else {
            throw Q4LoweringError.invalidArtifact(
                "selected width not component-accepted (bit-identical, <1e-4)")
        }
        if let mg = modelGate {
            guard mg.results.allSatisfy({ $0.schedule.width > 0
                    && $0.schedule.width <= pipelineMaxThreads }) else {
                throw Q4LoweringError.invalidArtifact("gate schedule illegal")
            }
            if !mg.diagnostic {
                guard Q4LaunchGatePolicy.shapeValid(mg, schedule: schedule, binding: baselineMoeBinding) else {
                    throw Q4LoweringError.invalidArtifact("model-gate shape invalid")
                }
            }
        }
    }
}

// MARK: - Forge (component measurement on REAL GGUF weights)

public enum Q4LaunchForge {
    public static let codegenVersion = "q4launch-v1"

    /// Quiet-NaN bit pattern written to EVERY output row before each
    /// runMatvec launch (sentinel poisoning). Public so qualifiers can
    /// assert that uncovered rows carry exactly this pattern.
    public static let outputSentinelBits: UInt32 = 0x7FC0_5A5A

    /// One g4_q4_matvec dispatch at an EXPLICIT launch width against
    /// prepared buffers — the exact geometry decision the runtime pass
    /// makes. Shared by the component forge and the test suite (tail
    /// shapes, bit identity across widths). ALL bounds are checked
    /// BEFORE any command encoder is allocated; a CANDIDATE width
    /// (requirePowerOfTwo, the default) must be a power of two within
    /// the ACTUAL pipeline maximum, and the row count must fit the
    /// kernel's uint indexing. The incumbent CONTROL width — the
    /// unchanged default `min(rows, pipelineMax)` — may legitimately be
    /// a non-power-of-two and passes requirePowerOfTwo: false.
    public static func runMatvec(
        pipe: MTLComputePipelineState, queue: MTLCommandQueue,
        wb: MTLBuffer, xb: MTLBuffer, yb: MTLBuffer,
        inDim: Int, rows: Int, width: Int,
        weightOffset: Int = 0,
        requirePowerOfTwo: Bool = true) throws -> (out: [Float], gpuMicros: Double) {
        guard inDim > 0, inDim % 32 == 0, inDim <= Int(UInt32.max),
              rows > 0, rows <= Int(UInt32.max),
              width > 0, (!requirePowerOfTwo || width & (width - 1) == 0),
              width <= pipe.maxTotalThreadsPerThreadgroup,
              weightOffset >= 0,
              let groups = groups(grid: rows, width: width) else {
            throw Q4LoweringError.componentGateFailed("runMatvec bounds invalid")
        }
        // Buffer extents (checked arithmetic) and nonaliasing, all BEFORE
        // any command buffer is created — an undersized or aliasing call
        // is rejected without submitting work.
        let blocksPerRow = inDim / 32
        let weightBytes = try MoEModelBinding.sliceStride(
            inDim: inDim, outDim: rows, MoELoweringError.invalidRegion)
        let weightEnd = try MoEModelBinding.add(
            weightOffset, weightBytes, "weight extent",
            MoELoweringError.invalidRegion)
        guard wb.length >= weightEnd,
              xb.length >= inDim * 4, yb.length >= rows * 4,
              wb !== xb, wb !== yb, xb !== yb, blocksPerRow > 0 else {
            throw Q4LoweringError.componentGateFailed(
                String(format: "runMatvec buffers invalid: wb %d < %d, xb %d, yb %d or aliasing",
                       wb.length, weightEnd, xb.length, yb.length))
        }
        // Sentinel poisoning: after ALL bounds/nonalias checks and BEFORE
        // any encode, every output row is overwritten with a quiet-NaN
        // sentinel, so a launch that fails to write a row can never reuse
        // a previous run's (e.g. the incumbent's) value. The finiteness
        // and bit-identity checks then reject any uncovered row.
        guard yb.storageMode == .shared else {
            throw Q4LoweringError.componentGateFailed(
                "runMatvec output must be CPU-visible shared storage (sentinel poisoning)")
        }
        let poisoned = yb.contents().assumingMemoryBound(to: Float.self)
        let sentinel = Float(bitPattern: outputSentinelBits)
        for i in 0..<rows { poisoned[i] = sentinel }
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Q4LoweringError.hardwareUnavailable("runMatvec encoder")
        }
        var params = SIMD2<UInt32>(UInt32(inDim), UInt32(rows))
        var off = UInt64(weightOffset)
        enc.setComputePipelineState(pipe)
        enc.setBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4LoweringError.componentGateFailed(
                "runMatvec command buffer: \(String(describing: cmd.error))")
        }
        let out = [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: rows))
        return (out, (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6)
    }

    /// The bounded, overflow-safe group count the runtime computes
    /// (identical formula; nil on invalid bounds).
    public static func groups(grid: Int, width: Int) -> Int? {
        guard grid > 0, grid <= Int(UInt32.max), width > 0 else { return nil }
        return (grid - 1) / width + 1
    }

    /// Deterministic fixed inputs, computed BEFORE any candidate runs —
    /// identical for incumbent, every width, and the FP64 oracle.
    public static func fixedInput(inDim: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<inDim).map { _ in rng.nextUniform() }
    }

    /// Component evaluation over the REAL hot shapes at FULL row counts
    /// against the runner's OWN resident zero-copy weight mapping (the
    /// exact bytes production dispatches — no shadow copy, no repack):
    ///   - every candidate width launches ALL outDim rows,
    ///   - bitPattern identity to the incumbent (default width) is
    ///     checked on EVERY output row of EVERY shape (the launch
    ///     geometry must be bit-exact by construction — measured, never
    ///     assumed),
    ///   - the independent FP64 oracle stays bounded to the first
    ///     min(outDim,64) rows (small prefix read for the oracle only),
    ///     < 1e-4 for the incumbent AND every candidate, all outputs
    ///     finite,
    /// then screens GPU medians over the FULL shapes to rank the widths
    /// (control measured and recorded, never a finalist).
    public static func component(
        runner: Gemma4Runner, file: GGUFFile, shapes: [Q4LaunchHotShape],
        candidates: [Q4LaunchSchedule],
        warmup: Int = 2, reps: Int = 5) throws
        -> (results: [Q4LaunchComponentResult], incumbentOracle: Double) {

        guard warmup > 0, reps > 0, !shapes.isEmpty, !candidates.isEmpty else {
            throw Q4LoweringError.invalidSchedule("component bounds violated")
        }
        guard let wb = runner.residentWeightsBuffer else {
            throw Q4LoweringError.hardwareUnavailable("runner has no resident mapping")
        }
        // The runner's OWN device, queue and CACHED production pipeline:
        // bit identity is proven on the exact pipeline object the runtime
        // dispatch() launches, against buffers of the same device. No
        // separately compiled copy and no second device object.
        let device = runner.q4RunnerDevice
        let queue = runner.q4RunnerQueue
        let pipe = try runner.q4MatvecPipeline()
        // The incumbent CONTROL is the ACTUAL pipeline maximum, which may
        // legitimately be a non-power-of-two. Only that exact width may
        // bypass the power-of-two candidate rule; any other illegal width
        // is rejected up front (never silently launched or clamped).
        let controlWidth = pipe.maxTotalThreadsPerThreadgroup
        guard Set(candidates).count == candidates.count,
              candidates.allSatisfy({ $0.width == controlWidth
                  || $0.isLegal(pipelineMaxThreads: controlWidth) }) else {
            throw Q4LoweringError.invalidSchedule(
                "component candidates must be distinct legal widths or the exact control \(controlWidth)")
        }

        // Per-shape fixed payloads: resident weight offset/extent, a
        // deterministic fixed x (BEFORE any candidate comparison), the
        // bounded oracle prefix, and small x/y buffers. Weights are NEVER
        // copied — the resident buffer is bound directly.
        struct Payload {
            let shape: Q4LaunchHotShape
            let weightOffset: Int
            let x: [Float]
            let oracle: [Double]
            let xb: MTLBuffer
            let yb: MTLBuffer
        }
        var payloads = [Payload]()
        for shape in shapes {
            guard let r = runner.tensorRef(shape.tensor), r.type == .q4_0,
                  r.inDim == shape.inDim, r.outDim == shape.outDim, r.experts == 1 else {
                throw Q4LoweringError.invalidSchedule(
                    "runner tensor \(shape.tensor) disagrees with the hot shape")
            }
            let stride = try MoEModelBinding.sliceStride(
                inDim: shape.inDim, outDim: shape.outDim,
                MoELoweringError.invalidRegion)
            guard r.offset >= 0, r.offset + stride <= r.end,
                  r.end <= wb.length else {
                throw Q4LoweringError.invalidSchedule(
                    "\(shape.tensor): resident extent disagrees with the mapping")
            }
            let info = try file.tensor(shape.tensor)
            let oracleBlocks = [UInt8](try file.tensorData(info)
                .prefix(shape.oracleBytes))
            let x = fixedInput(inDim: shape.inDim,
                               seed: 0x51_4C_41_55_4E_43_48 &+ UInt64(shape.inDim))
            guard x.allSatisfy({ $0.isFinite }) else {
                throw Q4LoweringError.oracleFailed("fixed input nonfinite")
            }
            let oracle = try Q4Oracle.matvecFP64(
                blocks: oracleBlocks[...], outDim: shape.oracleRows,
                inDim: shape.inDim, x: x)
            guard oracle.allSatisfy({ $0.isFinite }) else {
                throw Q4LoweringError.oracleFailed("oracle nonfinite")
            }
            guard let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                             options: .storageModeShared),
                  let yb = device.makeBuffer(length: shape.outDim * 4,
                                             options: .storageModeShared) else {
                throw Q4LoweringError.hardwareUnavailable("component buffers")
            }
            payloads.append(.init(shape: shape, weightOffset: r.offset, x: x,
                                  oracle: oracle, xb: xb, yb: yb))
        }

        func run(_ p: Payload, width: Int) throws -> (out: [Float], gpuMicros: Double) {
            // Power-of-two rule bypassed ONLY for the exact control width.
            try Q4LaunchForge.runMatvec(
                pipe: pipe, queue: queue, wb: wb, xb: p.xb, yb: p.yb,
                inDim: p.shape.inDim, rows: p.shape.outDim, width: width,
                weightOffset: p.weightOffset,
                requirePowerOfTwo: width != controlWidth)
        }
        func rel(_ out: [Float], _ oracle: [Double]) -> Double {
            var worst = 0.0
            for i in 0..<oracle.count {
                worst = max(worst, abs(Double(out[i]) - oracle[i])
                            / max(1.0, abs(oracle[i])))
            }
            return worst
        }
        func bits(_ a: [Float], _ b: [Float]) -> Bool {
            a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
        }

        // Incumbent (default width) on the FULL shapes — the bit-identity
        // reference, the oracle baseline, and its own control timings.
        var incumbent = [[Float]]()
        var incumbentRel = 0.0
        for p in payloads {
            let defWidth = max(1, min(p.shape.outDim,
                                      pipe.maxTotalThreadsPerThreadgroup))
            let r = try Q4LaunchForge.runMatvec(
                pipe: pipe, queue: queue, wb: wb, xb: p.xb, yb: p.yb,
                inDim: p.shape.inDim, rows: p.shape.outDim, width: defWidth,
                weightOffset: p.weightOffset, requirePowerOfTwo: false)
            guard r.out.allSatisfy({ $0.isFinite }) else {
                throw Q4LoweringError.componentGateFailed("incumbent nonfinite")
            }
            incumbentRel = max(incumbentRel, rel(r.out, p.oracle))
            incumbent.append(r.out)
        }
        guard incumbentRel < 1e-4 else {
            throw Q4LoweringError.componentGateFailed(
                String(format: "incumbent outside oracle contract (%.3e)", incumbentRel))
        }

        var results = [Q4LaunchComponentResult]()
        for cand in candidates {
            let isControl = cand.width == controlWidth
            var allBits = true
            var worst = 0.0
            var screen = [Double]()
            for (i, p) in payloads.enumerated() {
                for _ in 0..<warmup { _ = try run(p, width: cand.width) }
                var times = [Double]()
                var out = [Float]()
                for _ in 0..<reps {
                    let r = try run(p, width: cand.width)
                    times.append(r.gpuMicros)
                    out = r.out
                }
                guard out.allSatisfy({ $0.isFinite }) else {
                    throw Q4LoweringError.componentGateFailed(
                        "width \(cand.width) nonfinite on \(p.shape.tensor)")
                }
                allBits = allBits && bits(out, incumbent[i])
                worst = max(worst, rel(out, p.oracle))
                screen.append(LoweringSelect.median(times))
            }
            guard allBits, worst < 1e-4 else {
                results.append(.init(schedule: cand, accepted: false,
                                     bitIdenticalToIncumbent: allBits,
                                     worstRelToOracle: worst,
                                     screenMicrosPerShape: screen,
                                     note: allBits
                                        ? String(format: "oracle %.3e", worst)
                                        : "NOT bit-identical to incumbent"))
                continue
            }
            results.append(.init(
                schedule: cand, accepted: true, bitIdenticalToIncumbent: true,
                worstRelToOracle: worst, screenMicrosPerShape: screen,
                note: isControl ? "incumbent control (never a finalist)" : "bit-exact"))
        }
        return (results, incumbentRel)
    }

    /// Screen ranking: fastest two ACCEPTED non-control widths by summed
    /// per-shape medians. The control is returned for the record.
    public static func finalists(
        _ results: [Q4LaunchComponentResult],
        controlWidth: Int) -> (finalists: [Q4LaunchSchedule], control: Q4LaunchComponentResult?) {
        let control = results.first { $0.schedule.width == controlWidth }
        let ranked = results
            .filter { $0.accepted && $0.schedule.width != controlWidth }
            .compactMap { r -> (Q4LaunchSchedule, Double)? in
                guard let s = r.screenMicrosPerShape else { return nil }
                return (r.schedule, s.reduce(0, +))
            }
            .sorted { $0.1 < $1.1 }
        return (Array(ranked.prefix(2).map(\.0)), control)
    }
}

// MARK: - Selection (fair: vs the ACCEPTED E4 baseline, same binary)

public enum Q4LaunchSelect {

    /// Full bounded selection. EVERY arm runs the freshly compiled E4_R32
    /// MoE lowering + fresh binding (installed once, uninstalled at
    /// exit); the BASELINE arm is E4 with the default launch width; the
    /// CANDIDATE arm is E4 + one width. Reference = the E4 baseline arm
    /// (old width control), 3 canonical prompts x 32 steps, all-logit
    /// strict gates; 3 paired AB/BA x 31 matched decode steps; fixed 2%
    /// policy recomputed from the recorded arms. An honest no-win keeps
    /// the incumbent (E4 default) active.
    public static func selectAndPersist(
        regionJSON: Data, ggufURL: URL, graph: GemmaGraphIR,
        runner: Gemma4Runner, fixtures: [Q4LoweringForge.Q4Fixture],
        genTokens: Int, timedTokens: Int,
        isEndOfGeneration: (Int) -> Bool,
        diagnostic: Bool,
        saveTo url: URL) throws -> (artifact: Q4LaunchArtifact, log: [String]) {

        var log = [String]()
        if diagnostic {
            guard genTokens >= 2, genTokens <= 64, timedTokens >= 1,
                  timedTokens < genTokens, fixtures.count >= 1, fixtures.count <= 4,
                  Set(fixtures.map(\.name)).count == fixtures.count else {
                throw Q4LoweringError.invalidSchedule("smoke bounds violated")
            }
        } else {
            guard genTokens == Q4LaunchGatePolicy.productionGenTokens,
                  timedTokens == Q4LaunchGatePolicy.productionTimedTokens,
                  fixtures.map(\.name) == Q4LaunchGatePolicy.productionFixtures else {
                throw Q4LoweringError.invalidSchedule(
                    "production launch-select requires 32/31 over the canonical fixtures")
            }
        }
        for f in fixtures {
            guard !f.tokens.isEmpty, f.tokens.count + genTokens <= 4096 else {
                throw Q4LoweringError.invalidSchedule("fixture \(f.name) exceeds context")
            }
        }

        let region = try MoERegionIR.parse(regionJSON)
        let hardware = runner.hardwareProfile()
        // The ACTUAL compiled g4_q4_matvec pipeline maximum — the control
        // width and the legality bound.
        let pipeMax = try runner.q4MatvecPipelineMaxThreads()
        let candidates = try Q4LaunchDeriver.derive(hardware: hardware,
                                                    pipelineMaxThreads: pipeMax)
        let control = Q4LaunchDeriver.control(pipelineMaxThreads: pipeMax)
        log.append("launch-geometry candidates: "
                   + candidates.map { String($0.width) }.joined(separator: "/")
                   + " (control \(control.width))")

        // ---- component gates on the REAL GGUF hot shapes ----
        let file = try GGUFReader.read(ggufURL)
        let shapes = try Q4LaunchHotShapes.derive(file: file,
                                                  layerCount: runner.config.layerCount)
        // The runner's OWN device: the component runs on the runner's
        // cached pipeline/queue, and the E4 baseline below compiles on
        // the same device that owns the resident weights and scratch.
        let device = runner.q4RunnerDevice
        var all = candidates
        if !candidates.contains(control) { all.append(control) }
        let (componentResults, incumbentRel) = try Q4LaunchForge.component(
            runner: runner, file: file, shapes: shapes, candidates: all)
        for r in componentResults {
            log.append(String(format: "component %@: %@ rel %.3e %@",
                              r.schedule.label, r.accepted ? "ACCEPT" : "REJECT",
                              r.worstRelToOracle ?? -1,
                              r.screenMicrosPerShape.map { String(format: "%.1f us", $0.reduce(0, +)) } ?? ""))
        }
        let (finalists, controlRec) = Q4LaunchForge.finalists(componentResults,
                                                              controlWidth: control.width)
        guard !finalists.isEmpty else {
            throw Q4LoweringError.componentGateFailed(
                "no accepted launch-width finalists")
        }

        // ---- E4 baseline installed for EVERY arm from here on ----
        let moeKernels = try MoEEmittedKernels.compile(region: region,
                                                       schedule: MoESchedule(batch: 4, rowsPerGroup: 32),
                                                       device: device)
        let moeSourceSha = MoEKernelEmitter.sourcesSha256([moeKernels.source])
        guard moeSourceSha == Q4LaunchArtifact.baselineMoeSourceSha256 else {
            throw Q4LoweringError.invalidArtifact(
                "E4 emission drifted (\(moeSourceSha)) — baseline identity refused")
        }
        try runner.installMoeTrialLowering(region: region, kernels: moeKernels)
        defer {
            runner.uninstallMoeGeneratedLowering()
            runner.uninstallQ4LaunchGeometry()
        }
        let moeBinding = try Gemma4Runner.moeBindingForProfile(region: region, runner: runner)
        log.append("E4_R32 baseline installed (source \(moeSourceSha.prefix(12))…, "
                   + "binding \(moeBinding.layerCount) layers)")

        // ---- reference arms: E4 baseline (old width control) ----
        let (ssaMany, ssaOne) = LoweringSelect.ssaForwardAdapters(graph: graph)
        let refArms = try LoweringSelect.generateReferenceArms(
            runner: runner, fixtures: fixtures, genTokens: genTokens,
            isEndOfGeneration: isEndOfGeneration,
            forwardMany: ssaMany, forwardOne: ssaOne,
            reset: LoweringSelect.ssaReset,
            requireFullLength: !diagnostic, log: &log)

        // ---- per-finalist strict gates with derived counter proofs ----
        let layerCount = moeBinding.layerCount
        let topK = region.topK
        let moeE = 4
        var modelResults = [Q4LaunchCandidateResult]()
        var baselineWall: [Double]? = nil
        var chosen: (schedule: Q4LaunchSchedule, candMed: Double)? = nil

        for cand in finalists {
            // GATE: install the width, replay every fixture strictly.
            runner.uninstallQ4LaunchGeometry()
            try runner.installQ4LaunchTrial(width: cand.width)
            var passes = 0
            var failures = 0
            var worstAll = 0.0
            var failReason: String? = nil
            var eligibleSum = 0, appliedSum = 0, stageSum = 0, mergeSum = 0
            for fixture in fixtures {
                let ref = refArms[fixture.name]!
                let eligible0 = runner.q4MatvecGenericDispatchCount
                let applied0 = runner.q4LaunchAppliedCount
                let fb0 = runner.q4LaunchFallbackCount
                let gu0 = runner.moeGuDispatchCount, act0 = runner.moeActDispatchCount
                let dn0 = runner.moeDnDispatchCount, m0 = runner.moeAxpyMergeCount
                let mfb0 = runner.moeResidentFallbackCount
                let g = LoweringSelect.gateReplay(
                    runner: runner, fixture: fixture, ref: ref,
                    forwardMany: ssaMany, forwardOne: ssaOne,
                    reset: LoweringSelect.ssaReset,
                    dispatchCount: { runner.q4LaunchAppliedCount },
                    fallbackCount: { runner.q4LaunchFallbackCount
                        + runner.moeResidentFallbackCount })
                let eligible = runner.q4MatvecGenericDispatchCount - eligible0
                let applied = runner.q4LaunchAppliedCount - applied0
                let geoFb = runner.q4LaunchFallbackCount - fb0
                let moeFb = runner.moeResidentFallbackCount - mfb0
                let forwards = fixture.tokens.count + ref.ids.count - 1
                // Each stage kernel dispatches ONCE per chunk: E4 over
                // topK8 = 2 chunks per layer per forward (phase-13
                // measured 3120 = 2*30*52 on en-chat). NO extra factor 2.
                let wantMoeStage = ((topK + moeE - 1) / moeE) * layerCount * forwards
                let wantMerge = topK * layerCount * forwards
                let stagesOK = runner.moeGuDispatchCount - gu0 == wantMoeStage
                    && runner.moeActDispatchCount - act0 == wantMoeStage
                    && runner.moeDnDispatchCount - dn0 == wantMoeStage
                    && runner.moeAxpyMergeCount - m0 == wantMerge
                // Applied must EQUAL the derived eligible population and
                // that population must be NONVACUOUS (eligible > 0) — an
                // empty pass can never prove geometry engagement.
                if g.fail == nil, g.worstRel < 1e-4, eligible > 0,
                    g.generatedDispatches == eligible,
                    geoFb == 0, moeFb == 0, stagesOK {
                    passes += 1
                    worstAll = max(worstAll, g.worstRel)
                    eligibleSum += eligible
                    appliedSum += applied
                    stageSum += wantMoeStage
                    mergeSum += wantMerge
                } else {
                    failures += 1
                    if failReason == nil {
                        failReason = g.fail
                            ?? "counters: applied \(applied)/eligible \(eligible), "
                               + "geoFb \(geoFb), moeFb \(moeFb), stagesOK \(stagesOK)"
                    }
                }
            }
            runner.uninstallQ4LaunchGeometry()
            if failures > 0 {
                modelResults.append(.init(schedule: cand, fixturesPassed: passes,
                                          fixturesFailed: failures, worstRel: nil,
                                          gateFailReason: failReason ?? "gate failed",
                                          eligibleDispatches: nil,
                                          appliedDispatches: nil,
                                          moeStageDispatches: nil,
                                          mergeDispatches: nil,
                                          timingReps: nil))
                log.append("gate FAIL \(cand.label): \(failReason ?? "")")
                continue
            }
            log.append(String(format: "gate PASS %@: %d/%d fixtures, worst %.3e, eligible==applied",
                              cand.label, passes, fixtures.count, worstAll))

            // PAIRED TIMING: candidate (E4+width) vs baseline (E4 only),
            // same adapters, AB/BA x3. Install state managed HERE (the
            // shared timedArm has no install hooks; both arms keep E4).
            var reps = [Q4TimingRepetition]()
            for rep in 0..<3 {
                let candidateFirst = rep % 2 == 1
                func candArm() throws -> [Q4TimingArmFixture] {
                    try runner.installQ4LaunchTrial(width: cand.width)
                    defer { runner.uninstallQ4LaunchGeometry() }
                    return try LoweringSelect.timedArm(
                        runner: runner, fixtures: fixtures, refArms: refArms,
                        timedTokens: timedTokens,
                        forwardMany: ssaMany, forwardOne: ssaOne,
                        reset: LoweringSelect.ssaReset)
                }
                func baseArm() throws -> [Q4TimingArmFixture] {
                    runner.uninstallQ4LaunchGeometry()
                    return try LoweringSelect.timedArm(
                        runner: runner, fixtures: fixtures, refArms: refArms,
                        timedTokens: timedTokens,
                        forwardMany: ssaMany, forwardOne: ssaOne,
                        reset: LoweringSelect.ssaReset)
                }
                let c = try candidateFirst ? candArm() : baseArm()
                let b = try candidateFirst ? baseArm() : candArm()
                reps.append(Q4TimingRepetition(candidateFirst: candidateFirst,
                                               candidate: candidateFirst ? c : b,
                                               incumbent: candidateFirst ? b : c))
            }
            let candSums = reps.map { Q4LaunchGatePolicy.armSum($0.candidate) }
            let baseSums = reps.map { Q4LaunchGatePolicy.armSum($0.incumbent) }
            guard candSums.allSatisfy({ $0.isFinite && $0 > 0 }),
                  baseSums.allSatisfy({ $0.isFinite && $0 > 0 }) else {
                throw Q4LoweringError.componentGateFailed("timing sums nonfinite")
            }
            let candMed = LoweringSelect.median(candSums)
            let baseMed = LoweringSelect.median(baseSums)
            let selects = candMed < baseMed * (1 - GemmaForge.noiseRelWall)
            modelResults.append(.init(schedule: cand, fixturesPassed: passes,
                                      fixturesFailed: 0, worstRel: worstAll,
                                      gateFailReason: nil,
                                      eligibleDispatches: eligibleSum,
                                      appliedDispatches: appliedSum,
                                      moeStageDispatches: stageSum,
                                      mergeDispatches: mergeSum,
                                      timingReps: reps))
            log.append(String(format: "matched timing %@: median %.0f vs E4-baseline %.0f us/token-sum -> %@",
                              cand.label, candMed, baseMed,
                              selects ? "SELECTED (>2%)" : "no win (honest record)"))
            if selects, chosen == nil || candMed < chosen!.candMed {
                chosen = (cand, candMed)
                baselineWall = reps[0].incumbent.map { LoweringSelect.median($0.walls) }
            }
        }

        // Fixture records (workload digests) from the actual references.
        let fixtureRecords = fixtures.map { f -> Q4FixtureRecord in
            let ids = refArms[f.name]!.ids
            var history = f.tokens
            history.append(contentsOf: ids.dropLast())
            return Q4FixtureRecord(name: f.name, promptTokenCount: f.tokens.count,
                                   gatedSteps: ids.count,
                                   historyDigest: GemmaForge.workloadDigest(history))
        }
        let anyGatePassed = modelResults.contains { $0.gateFailReason == nil }
        if !anyGatePassed && !diagnostic {
            throw Q4LoweringError.componentGateFailed("no finalist passed the strict gates")
        }
        guard let chosen else {
            log.append("no 2% win — honest record persisted, incumbent E4 stays active")
            let art = try persist(region: region, runner: runner, ggufURL: ggufURL,
                                  hardware: hardware,
                                  graphDigest: try GemmaGraph.digest(graph),
                                  shapes: shapes,
                                  componentResults: componentResults,
                                  incumbentRel: incumbentRel, moeBinding: moeBinding,
                                  selectedSchedule: nil,
                                  modelResults: modelResults, fixtureRecords: fixtureRecords,
                                  genTokens: genTokens, timedTokens: timedTokens,
                                  baselineWall: baselineWall, performanceSelected: false,
                                  diagnostic: diagnostic, saveTo: url)
            log.append("Q4LAUNCH_SELECT_COMPLETED (no win) -> \(url.path)")
            return (art, log)
        }

        // ---- persisted reload proof through the PUBLIC installer ----
        let artifact = try persist(region: region, runner: runner, ggufURL: ggufURL,
                                   hardware: hardware,
                                   graphDigest: try GemmaGraph.digest(graph),
                                   shapes: shapes,
                                   componentResults: componentResults,
                                   incumbentRel: incumbentRel, moeBinding: moeBinding,
                                   selectedSchedule: chosen.schedule,
                                   modelResults: modelResults, fixtureRecords: fixtureRecords,
                                   genTokens: genTokens, timedTokens: timedTokens,
                                   baselineWall: baselineWall, performanceSelected: true,
                                   diagnostic: false, saveTo: url)
        let reloaded = try Q4LaunchArtifact.load(from: url)
        try runner.installQ4LaunchGenerated(artifact: reloaded,
                                            ggufDigest: artifact.ggufDigest,
                                            region: region,
                                            graphDigest: try GemmaGraph.digest(graph))
        defer {
            runner.uninstallQ4LaunchGeometry()
            runner.uninstallMoeGeneratedLowering()
        }
        let enRef = refArms[fixtures[0].name]!
        let eligible0 = runner.q4MatvecGenericDispatchCount
        let g = LoweringSelect.gateReplay(
            runner: runner, fixture: fixtures[0], ref: enRef,
            forwardMany: ssaMany, forwardOne: ssaOne,
            reset: LoweringSelect.ssaReset,
            dispatchCount: { runner.q4LaunchAppliedCount },
            fallbackCount: { runner.q4LaunchFallbackCount
                + runner.moeResidentFallbackCount })
        let eligible = runner.q4MatvecGenericDispatchCount - eligible0
        guard g.fail == nil, g.generatedDispatches == eligible,
              runner.q4LaunchFallbackCount == 0 else {
            throw Q4LoweringError.componentGateFailed(
                "reload proof failed: \(g.fail ?? "counters")")
        }
        log.append(String(format: "persisted execution: reloaded+validated+installed (public); %@ replay worst %.3e, eligible==applied==%d",
                          artifact.schedule.label, g.worstRel, eligible))
        return (artifact, log)
    }

    private static func persist(region: MoERegionIR, runner: Gemma4Runner,
                                ggufURL: URL, hardware: ForgeHardwareProfile,
                                graphDigest: String,
                                shapes: [Q4LaunchHotShape],
                                componentResults: [Q4LaunchComponentResult],
                                incumbentRel: Double, moeBinding: MoEModelBinding,
                                selectedSchedule: Q4LaunchSchedule?,
                                modelResults: [Q4LaunchCandidateResult],
                                fixtureRecords: [Q4FixtureRecord],
                                genTokens: Int, timedTokens: Int,
                                baselineWall: [Double]?,
                                performanceSelected: Bool,
                                diagnostic: Bool, saveTo url: URL) throws
            -> Q4LaunchArtifact {
        let gate = Q4LaunchGateRecord(
            gatePolicy: Q4LaunchGateRecord.policyVersion,
            executionPath: Q4LaunchGateRecord.executionPathSSA,
            diagnostic: diagnostic,
            genTokens: genTokens,
            timedTokens: timedTokens,
            fixtures: fixtureRecords,
            results: modelResults,
            baselineWallPerFixture: baselineWall ?? [],
            performanceSelected: performanceSelected,
            policyNote: performanceSelected
                ? "selected vs accepted E4_R32 baseline, same binary/settings, 3 paired AB/BA x31, fixed 2%"
                : "honest no-win record: incumbent E4_R32 stays active")
        let body = Q4LaunchArtifact(
            artifactSchema: Q4LaunchArtifact.schema,
            codegenVersion: Q4LaunchForge.codegenVersion,
            schedule: selectedSchedule
                ?? modelResults.first(where: { $0.gateFailReason == nil })?.schedule
                ?? Q4LaunchSchedule(width: 1),
            kernelName: Gemma4Runner.q4LaunchKernel,
            primitiveSourceSha256: SHA256.hex(SHA256.hash(Array(Gemma4Runner.shaderSource.utf8))),
            baselineMoeSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256,
            baselineMoeSchedule: MoESchedule(batch: 4, rowsPerGroup: 32),
            baselineMoeBinding: moeBinding,
            regionGraphSha256: region.graphSha256,
            hardware: hardware,
            toolchain: Forge.toolchainIdentifier,
            // Actual model SHA-256 from the shared cache (hashed once, reused).
            ggufDigest: try ModelDigestCache.shared.digest(for: ggufURL).sha256,
            executableDigest: try Gemma4Runner.executableDigest(),
            graphDigest: graphDigest,
            hotShapes: shapes,
            componentResults: componentResults,
            incumbentWorstRelToOracle: incumbentRel,
            modelGate: gate,
            artifactDigest: "")
        let artifact = try Q4LaunchArtifact.withDigest(body)
        try artifact.save(to: url)
        return artifact
    }
}

// MARK: - Runner integration

extension Gemma4Runner {

    /// Fresh binding for the profile/selection (same crosscheck the
    /// internal installer performs).
    static func moeBindingForProfile(region: MoERegionIR, runner: Gemma4Runner)
            throws -> MoEModelBinding {
        try Gemma4Runner.moeBinding(for: region, runner: runner)
    }

    /// PUBLIC composite install: validates the artifact end-to-end
    /// (self-digest, recorded hardware vs the ACTUAL measured profile,
    /// recorded graph digest vs the SUPPLIED actual graph, fresh E4
    /// emission hash, fresh model binding EQUALITY, width legality on
    /// the actual pipeline, primitive-source/toolchain/model/executable
    /// digests, production gate policy) BEFORE any state change. Every
    /// remaining fallible step is also completed BEFORE mutation where
    /// possible; if a post-mutation step still throws, the PRIOR
    /// lowering state (captured up front) is restored — no partial
    /// success, no lost pre-existing install.
    public func installQ4LaunchGenerated(artifact: Q4LaunchArtifact,
                                         ggufDigest: String,
                                         region currentRegion: MoERegionIR,
                                         graphDigest: String) throws {
        // The runner's OWN device: the composite E4 kernels are compiled
        // for the device that owns the resident weights.
        let device = q4RunnerDevice
        try artifact.validate(region: currentRegion,
                              pipelineMaxThreads: try q4MatvecPipelineMaxThreads(),
                              hardware: hardwareProfile(),
                              graphDigest: graphDigest,
                              ggufDigest: ggufDigest,
                              executableDigest: try Gemma4Runner.executableDigest())
        guard let gate = artifact.modelGate,
              Q4LaunchGatePolicy.productionValid(gate, schedule: artifact.schedule,
                                              binding: artifact.baselineMoeBinding) else {
            throw Q4LoweringError.invalidArtifact(
                "not production-valid (diagnostic/no-win records never install)")
        }
        // Fresh binding equality against the artifact's recorded one.
        let fresh = try Gemma4Runner.moeBinding(for: currentRegion, runner: self)
        guard fresh == artifact.baselineMoeBinding else {
            throw Q4LoweringError.invalidArtifact(
                "recorded E4 binding does not match this file/device")
        }
        // Pre-build and pre-check every fallible piece BEFORE mutating.
        let kernels = try MoEEmittedKernels.compile(
            region: currentRegion, schedule: artifact.baselineMoeSchedule,
            device: device)
        guard MoEKernelEmitter.sourcesSha256([kernels.source])
                == artifact.baselineMoeSourceSha256 else {
            throw Q4LoweringError.invalidArtifact("E4 emission drifted at install")
        }
        let pipeMax = try q4MatvecPipelineMaxThreads()
        guard artifact.schedule.width > 0,
              artifact.schedule.width <= pipeMax else {
            throw Q4LoweringError.invalidArtifact(
                "width \(artifact.schedule.width) illegal on this pipeline (\(pipeMax))")
        }
        // Coherent mutation with full prior-state restore on any throw.
        let saved = snapshotLoweringState()
        do {
            try installMoeTrialLowering(region: currentRegion, kernels: kernels)
            try installQ4LaunchTrial(width: artifact.schedule.width)
        } catch {
            restoreLoweringState(saved)
            throw error
        }
    }
}
