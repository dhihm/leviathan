// Pure two-axis SHADOW composition over a validated Gemma SSA schedule.
// It composes two EXISTING static analyses, unchanged, in fixed order:
//
//   gpuCacheAppend     -> GemmaOpportunityDiscovery.discover (validates
//                         config, profile, and SSA itself; its errors
//                         propagate unwrapped)
//   rmsStatisticsReuse -> RMSStatisticsPlan.analyze with the caller's
//                         explicit epsilon (no default exists here)
//
// No grouping, liveness, alias, or effect rule is re-derived: the cache
// verdict is the existing report's own `supported`, and the statistics
// axis is supported exactly when the existing plan has at least one
// group and at least one reuse.
//
// SHADOW ONLY. Nothing here is wired into GemmaForge.candidates, plan
// creation/validation, engine identity, the executor, the runner, or any
// kernel. No file access, runner mutation, Metal call, or model loading.
// A decoded report is diagnostic data, never an admissible proof:
// candidate filtering always re-runs both analyses from the supplied
// SSA, config, profile, and epsilon. No speed claim is made or implied.

import Foundation

public enum GemmaShadowCompositionError: Error, CustomStringConvertible, Equatable {
    /// The epsilon is not finite and positive. Carries the exact bit
    /// pattern so NaN payloads compare by bits, never by Float equality.
    case invalidEpsilon(UInt32)

    public var description: String {
        switch self {
        case .invalidEpsilon(let bits):
            return "shadow composition epsilon invalid (bits 0x\(String(bits, radix: 16)))"
        }
    }
}

/// Deterministic, Codable two-axis report. `body` carries the complete
/// evidence of both analyses (the existing cache report, which records
/// its own validated config, hardware profile, and SSA digest, and the
/// FULL RMS plan with its groups and dense actions), the exact epsilon
/// bits, and the ordered verdicts. `reportDigest` is SHA-256 over the
/// canonical sorted-key JSON of `body` ALONE, so the digest never covers
/// itself. The digest is integrity bookkeeping, not authentication.
public struct GemmaComposedOpportunityReport: Codable, Equatable, Sendable {
    /// The composed axes, in their fixed report order.
    public enum Axis: String, Codable, Equatable, Sendable, CaseIterable {
        case gpuCacheAppend
        case rmsStatisticsReuse
    }

    /// Stable typed reasons for a withheld axis.
    public enum ReasonCode: String, Codable, Equatable, Sendable {
        /// The existing cache report withheld the axis; its own ordered
        /// reasons (cacheReport.body.reasons) carry the IR evidence.
        case cacheAppendUnsupported
        /// The existing RMS plan has no group with a reuse
        /// (groupCount == 0 or reuseCount == 0).
        case noReusableGroups
    }

    public struct Verdict: Codable, Equatable, Sendable {
        public let axis: Axis
        public let supported: Bool
        /// nil exactly when the axis is supported.
        public let reason: ReasonCode?
    }

    public struct Body: Codable, Equatable, Sendable {
        public let compositionVersion: Int
        /// Exact Float bit pattern of the caller's epsilon.
        public let epsilonBits: UInt32
        /// RMSStatisticsPlan analysis version of `rmsPlan`.
        public let rmsAnalysisVersion: Int
        /// The existing single-axis report, verbatim.
        public let cacheReport: GemmaOpportunityReport
        /// The FULL existing statistics plan: groups and dense actions.
        public let rmsPlan: RMSStatisticsPlan
        /// One verdict per axis, in Axis.allCases order.
        public let verdicts: [Verdict]
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

    /// The recorded verdict for an axis, if present.
    public func verdict(for axis: Axis) -> Verdict? {
        body.verdicts.first { $0.axis == axis }
    }
}

/// Pure shadow composition of the gpuCacheAppend and rmsStatisticsReuse axes.
public enum GemmaShadowComposition {

    private typealias Report = GemmaComposedOpportunityReport

    /// Bumped whenever the composition rules or the report shape change.
    public static let compositionVersion = 1

    /// Validates the explicit epsilon, then runs the existing cache
    /// discovery and the existing RMS statistics analysis exactly as-is.
    /// An invalid epsilon throws before any analysis or report; every
    /// error from the wrapped analyses propagates unchanged.
    public static func discover(ssa: GemmaSSAIR,
                                config: GemmaGraphIR.GraphConfig,
                                hardware: ForgeHardwareProfile,
                                epsilon: Float) throws
            -> GemmaComposedOpportunityReport {
        guard epsilon.isFinite, epsilon > 0 else {
            throw GemmaShadowCompositionError.invalidEpsilon(epsilon.bitPattern)
        }
        let cacheReport = try GemmaOpportunityDiscovery.discover(
            ssa: ssa, config: config, hardware: hardware)
        let plan = try RMSStatisticsPlan.analyze(ssa: ssa, epsilon: epsilon)
        let cacheSupported = cacheReport.body.supported
        let rmsSupported = plan.groupCount > 0 && plan.reuseCount > 0
        let verdicts = [
            Report.Verdict(
                axis: .gpuCacheAppend, supported: cacheSupported,
                reason: cacheSupported ? nil : Report.ReasonCode.cacheAppendUnsupported),
            Report.Verdict(
                axis: .rmsStatisticsReuse, supported: rmsSupported,
                reason: rmsSupported ? nil : Report.ReasonCode.noReusableGroups),
        ]
        let body = Report.Body(
            compositionVersion: compositionVersion,
            epsilonBits: epsilon.bitPattern,
            rmsAnalysisVersion: plan.analysis,
            cacheReport: cacheReport, rmsPlan: plan, verdicts: verdicts)
        return try GemmaComposedOpportunityReport(body: body)
    }

    /// Shadow-only consistency helper: re-runs FRESH composition from the
    /// supplied inputs (a report is never accepted as input), then keeps
    /// the manual baseline (index 0) unconditionally and keeps every later
    /// manual tuple, in manual order with all flags untouched, only when
    /// each composed axis it enables is supported. Not called by the forge.
    public static func shadowCandidates(ssa: GemmaSSAIR,
                                        config: GemmaGraphIR.GraphConfig,
                                        hardware: ForgeHardwareProfile,
                                        epsilon: Float) throws
            -> [GemmaForgeCandidate] {
        let report = try discover(ssa: ssa, config: config, hardware: hardware,
                                  epsilon: epsilon)
        let cache = report.verdict(for: .gpuCacheAppend)?.supported == true
        let rms = report.verdict(for: .rmsStatisticsReuse)?.supported == true
        // Index 0 is pinned explicitly, whatever its flags.
        return GemmaForge.candidates().enumerated()
            .filter { $0.offset == 0
                || ((!$0.element.gpuCacheAppend || cache)
                    && (!$0.element.rmsStatisticsReuse || rms)) }
            .map(\.element)
    }
}
