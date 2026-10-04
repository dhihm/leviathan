// Pure, bounded IR-derived opportunity discovery over a validated Gemma
// SSA schedule. The single axis decided here is the existing GPU
// cache-append copy: for every cache_append the STATIC subset of the
// gpuCacheAppendCopy binding contract is proven from the IR (legal
// layer, exactly t2 then kv, whole-buffer vector f32 operands, config
// geometry, checked UInt32 width, matching kvstate output), and the
// axis is supported only when every canonical layer carries exactly one
// proven append.
//
// SHADOW ONLY. Nothing here is wired into GemmaForge.candidates, plan
// creation/validation, engine identity, the executor, the runner, or
// any kernel. No file access, runner mutation, Metal call, or model
// loading. Runtime concerns (actual position, spans, buffer lengths,
// pipeline legality) are NOT certified. A decoded report is diagnostic
// data, never an admissible proof: candidate filtering always re-runs
// discovery from the supplied SSA, config, and profile. No speed claim
// is made or implied.

import Foundation

public enum GemmaOpportunityDiscoveryError: Error, CustomStringConvertible, Equatable {
    case invalidConfig(String)
    case invalidHardware(String)
    case malformedSSA(String)

    public var description: String {
        switch self {
        case .invalidConfig(let m): return "opportunity discovery config invalid: \(m)"
        case .invalidHardware(let m): return "opportunity discovery hardware profile invalid: \(m)"
        case .malformedSSA(let m): return "opportunity discovery ssa invalid: \(m)"
        }
    }
}

/// Deterministic, Codable discovery report. `body` carries every analysis
/// input identity (analysis version, config, hardware profile, SSA
/// digest) and the ordered result; `reportDigest` is SHA-256 over the
/// canonical sorted-key JSON of `body` ALONE, so the digest never covers
/// itself. The digest is integrity bookkeeping, not authentication.
public struct GemmaOpportunityReport: Codable, Equatable, Sendable {
    /// Stable typed reasons for a withheld axis.
    public enum ReasonCode: String, Codable, Equatable, Sendable {
        /// cache_append without a legal layer index.
        case layerAbsent
        /// Not exactly two operands.
        case operandCount
        /// Operands are not exactly t2 then kv (the executor's K/V order).
        case operandBinding
        /// An operand is not a vector f32 value.
        case operandKind
        /// An operand aliases a buffer at a nonzero slice offset; the copy
        /// binds whole buffers at offset zero.
        case operandSlice
        /// kvHeads/headDim attributes differ from the layer's config geometry.
        case geometryMismatch
        /// The config geometry width is not a positive UInt32.
        case widthExceedsUInt32
        /// An operand length differs from the config geometry width.
        case operandLength
        /// The output is not the layer's kvstate value of that width.
        case outputMismatch
        /// More than one cache_append for one layer.
        case duplicateLayer
        /// No cache_append for a canonical layer.
        case missingLayer
    }

    public struct Reason: Codable, Equatable, Sendable {
        public let code: ReasonCode
        public let layer: Int?
        /// Instruction indices involved (empty for a missing layer).
        public let instructionIndices: [Int]
        /// SSA value IDs involved, in operand order.
        public let valueIDs: [String]

        public init(code: ReasonCode, layer: Int?, instructionIndices: [Int],
                    valueIDs: [String]) {
            self.code = code
            self.layer = layer
            self.instructionIndices = instructionIndices
            self.valueIDs = valueIDs
        }
    }

    /// Static evidence for one proven cache_append.
    public struct CacheAppendProof: Codable, Equatable, Sendable {
        public let instructionIndex: Int
        public let layer: Int
        public let global: Bool
        public let keyInputID: String
        public let valueInputID: String
        public let outputID: String
        public let kvHeads: Int
        public let headDim: Int
        /// Words copied per operand (kvHeads * headDim), proven <= UInt32.max.
        public let wordWidth: Int
    }

    public struct Body: Codable, Equatable, Sendable {
        public let analysisVersion: Int
        public let axis: String
        public let config: GemmaGraphIR.GraphConfig
        public let hardware: ForgeHardwareProfile
        public let ssaDigest: String
        public let supported: Bool
        /// Proven appends in instruction order.
        public let proofs: [CacheAppendProof]
        /// Per-instruction reasons in instruction order, then coverage
        /// reasons in ascending layer order.
        public let reasons: [Reason]
    }

    public let body: Body
    public let reportDigest: String

    init(body: Body) throws {
        self.body = body
        self.reportDigest = try Self.digest(of: body)
    }

    /// SHA-256 hex of the canonical sorted-key JSON of a body.
    public static func digest(of body: Body) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
    }

    /// Recomputes the body digest. A report that verifies is still only
    /// diagnostic data, never an admissible input for candidate generation.
    public func verifyDigest() throws -> Bool {
        try Self.digest(of: body) == reportDigest
    }
}

/// Pure opportunity discovery for the gpuCacheAppend axis.
public enum GemmaOpportunityDiscovery {

    /// Bumped whenever the proof rules or the report shape change.
    public static let analysisVersion = 1
    /// The single axis this analysis decides.
    public static let axis = "gpuCacheAppend"
    /// Diagnostic allocation bound on the layer count (coverage reasons
    /// are emitted per layer). An input-sanity bound, not a model fact.
    public static let maxLayerCount = 1 << 12

    /// Validates config and profile sanity, runs the existing pure
    /// GemmaSSA.validate, then proves each cache_append. Invalid config,
    /// invalid profile, or malformed SSA throws; valid but unsupported
    /// schedules return a report with stable reasons.
    public static func discover(ssa: GemmaSSAIR,
                                config: GemmaGraphIR.GraphConfig,
                                hardware: ForgeHardwareProfile) throws
            -> GemmaOpportunityReport {
        // Sanity runs BEFORE the existing validator: GemmaSSA.validate
        // computes 8 * hidden unchecked, so an overflowing config must
        // never reach it.
        try validateConfig(config)
        try validateHardware(hardware)
        let ssaDigest: String
        do {
            try GemmaSSA.validate(ssa, config: config)
            ssaDigest = try GemmaSSA.digest(ssa)
        } catch {
            throw GemmaOpportunityDiscoveryError.malformedSSA("\(error)")
        }
        var byID = [String: GemmaSSAIR.Value]()
        for value in ssa.values { byID[value.id] = value }
        let globalSet = Set(config.globalLayers)
        var proofs = [GemmaOpportunityReport.CacheAppendProof]()
        var reasons = [GemmaOpportunityReport.Reason]()
        // Layer -> appends (instruction index, output id), in instruction order.
        var coverage = [Int: [(index: Int, output: String)]]()

        for (index, inst) in ssa.instructions.enumerated() where inst.op == "cache_append" {
            guard let layer = inst.layer, layer >= 0, layer < config.layerCount else {
                reasons.append(GemmaOpportunityReport.Reason(
                    code: .layerAbsent, layer: inst.layer,
                    instructionIndices: [index], valueIDs: [inst.output]))
                continue
            }
            coverage[layer, default: []].append((index: index, output: inst.output))
            var withheld = [GemmaOpportunityReport.Reason]()
            func withhold(_ code: GemmaOpportunityReport.ReasonCode, _ ids: [String]) {
                withheld.append(GemmaOpportunityReport.Reason(
                    code: code, layer: layer, instructionIndices: [index], valueIDs: ids))
            }
            guard inst.inputs.count == 2 else {
                withhold(.operandCount, inst.inputs)
                reasons += withheld
                continue
            }
            guard let key = byID[inst.inputs[0]], let value = byID[inst.inputs[1]],
                  let output = byID[inst.output] else {
                // Unreachable after validation; never a silent pass.
                throw GemmaOpportunityDiscoveryError.malformedSSA(
                    "cache_append at \(index) references an undefined value")
            }
            let operandIDs = [key.id, value.id]
            // Exact executor binding order: K from t2, V from kv.
            if key.name != "t2" || value.name != "kv" {
                withhold(.operandBinding, operandIDs)
            }
            if key.kind != .vector || value.kind != .vector
                || key.dtype != .f32 || value.dtype != .f32 {
                withhold(.operandKind, operandIDs)
            }
            // Whole-buffer ABI: the copy binds its sources at offset zero.
            if (key.sliceOffset ?? 0) != 0 || (value.sliceOffset ?? 0) != 0 {
                withhold(.operandSlice, operandIDs)
            }
            // Geometry must equal the supplied layer's config geometry.
            let global = globalSet.contains(layer)
            let heads = global ? config.globalKVHeads : config.localKVHeads
            let dim = global ? config.globalHeadDim : config.localHeadDim
            if inst.attrs["kvHeads"] != heads || inst.attrs["headDim"] != dim {
                withhold(.geometryMismatch, operandIDs)
            }
            // The copy passes its width (and a word offset) as UInt32.
            let (width, overflow) = heads.multipliedReportingOverflow(by: dim)
            let fitsWord = !overflow && width > 0 && width <= Int(UInt32.max)
            if !fitsWord {
                withhold(.widthExceedsUInt32, operandIDs)
            } else if key.length != width || value.length != width {
                withhold(.operandLength, operandIDs)
            }
            if output.kind != .kvstate || output.name != "kvstate_\(layer)"
                || (fitsWord && output.length != width) {
                withhold(.outputMismatch, [output.id])
            }
            if withheld.isEmpty {
                proofs.append(GemmaOpportunityReport.CacheAppendProof(
                    instructionIndex: index, layer: layer, global: global,
                    keyInputID: key.id, valueInputID: value.id, outputID: output.id,
                    kvHeads: heads, headDim: dim, wordWidth: width))
            } else {
                reasons += withheld
            }
        }
        // Coverage: exactly one append per canonical layer, reported in
        // ascending layer order (never dictionary iteration order).
        for layer in 0..<config.layerCount {
            let entries = coverage[layer] ?? []
            if entries.isEmpty {
                reasons.append(GemmaOpportunityReport.Reason(
                    code: .missingLayer, layer: layer,
                    instructionIndices: [], valueIDs: []))
            } else if entries.count > 1 {
                reasons.append(GemmaOpportunityReport.Reason(
                    code: .duplicateLayer, layer: layer,
                    instructionIndices: entries.map { $0.index },
                    valueIDs: entries.map { $0.output }))
            }
        }
        let body = GemmaOpportunityReport.Body(
            analysisVersion: analysisVersion, axis: axis, config: config,
            hardware: hardware, ssaDigest: ssaDigest,
            supported: reasons.isEmpty && proofs.count == config.layerCount,
            proofs: proofs, reasons: reasons)
        return try GemmaOpportunityReport(body: body)
    }

    /// Shadow-only consistency helper: re-runs FRESH discovery from the
    /// supplied inputs (a report is never accepted as input), then keeps
    /// the manual baseline (index 0) unconditionally and admits or
    /// withholds ONLY the later manual tuples carrying gpuCacheAppend,
    /// preserving manual order. Every other tuple passes through
    /// untouched. Not called by the forge.
    public static func shadowCandidates(ssa: GemmaSSAIR,
                                        config: GemmaGraphIR.GraphConfig,
                                        hardware: ForgeHardwareProfile) throws
            -> [GemmaForgeCandidate] {
        let supported = try discover(ssa: ssa, config: config,
                                     hardware: hardware).body.supported
        // Index 0 is pinned explicitly, whatever its flags.
        return GemmaForge.candidates().enumerated()
            .filter { $0.offset == 0 || !$0.element.gpuCacheAppend || supported }
            .map(\.element)
    }

    // MARK: - input sanity (discovery-owned invariants)

    static func validateConfig(_ c: GemmaGraphIR.GraphConfig) throws {
        func reject(_ message: String) -> GemmaOpportunityDiscoveryError {
            .invalidConfig(message)
        }
        guard c.layerCount > 0, c.layerCount <= maxLayerCount else {
            throw reject("layerCount \(c.layerCount) outside 1...\(maxLayerCount)")
        }
        guard c.hidden > 0, c.vocabSize > 0, c.qHeads > 0 else {
            throw reject("hidden/vocabSize/qHeads must be positive")
        }
        // GemmaSSA.validate computes this product unchecked.
        guard !c.hidden.multipliedReportingOverflow(by: 8).overflow else {
            throw reject("8 * hidden overflows")
        }
        guard c.localKVHeads > 0, c.globalKVHeads > 0,
              c.localHeadDim > 0, c.globalHeadDim > 0 else {
            throw reject("KV heads and head dims must be positive")
        }
        guard !c.localKVHeads.multipliedReportingOverflow(by: c.localHeadDim).overflow,
              !c.globalKVHeads.multipliedReportingOverflow(by: c.globalHeadDim).overflow else {
            throw reject("KV geometry product overflows")
        }
        guard c.slidingWindow >= 0, c.expertCount > 0, c.expertUsedCount > 0,
              c.expertUsedCount <= c.expertCount, c.expertIntermediate > 0,
              c.sharedIntermediate > 0 else {
            throw reject("window/expert/intermediate dimensions invalid")
        }
        // Finite is required twice over: the report digest encodes the
        // config as canonical JSON and JSONEncoder rejects non-finite
        // floats, and lowering converts the cap with Int(...), so it must
        // also be positive and well inside the integer range.
        guard c.finalSoftcap.isFinite, c.finalSoftcap > 0,
              c.finalSoftcap < Float(Int32.max) else {
            throw reject("finalSoftcap \(c.finalSoftcap) not finite/positive/bounded")
        }
        guard Set(c.globalLayers).count == c.globalLayers.count,
              c.globalLayers.allSatisfy({ $0 >= 0 && $0 < c.layerCount }) else {
            throw reject("globalLayers must be unique legal layer indices")
        }
    }

    static func validateHardware(_ h: ForgeHardwareProfile) throws {
        guard !h.deviceName.isEmpty else {
            throw GemmaOpportunityDiscoveryError.invalidHardware("empty device name")
        }
        guard h.maxTotalThreadsPerThreadgroup > 0, h.threadExecutionWidth > 0,
              h.recommendedMaxWorkingSetSize > 0 else {
            throw GemmaOpportunityDiscoveryError.invalidHardware(
                "thread and working-set capacities must be positive")
        }
        guard h.threadExecutionWidth <= h.maxTotalThreadsPerThreadgroup else {
            throw GemmaOpportunityDiscoveryError.invalidHardware(
                "execution width exceeds max threads per threadgroup")
        }
    }
}
