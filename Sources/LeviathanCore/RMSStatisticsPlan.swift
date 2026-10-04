// Graph-driven reusable RMS statistics plan.
//
// Measured evidence: 331 RMSNorm operations per token, 30 groups of 3
// identical immutable input versions + attrs across the shared FFW /
// routing / expert branch, i.e. 60 duplicated statistic reductions
// (published GPU diagnostic: RMS 28.74%, Q4 24.36%, F32 router 17.62%,
// Q6 4.53%, normal host flush 32.89%). This analyzer turns that static
// SSA evidence into a reusable plan: the FIRST RMSNorm of a compatible
// group computes the inverse statistic once, later RMSNorms with the
// SAME live input version reuse it.
//
// Design rules enforced here:
// - Graph-driven only: every decision comes from the validated SSA
//   operands (input value ID, dtype, length, layout, attrs, epsilon
//   bits). No cache keyed only by physical buffer, no hard-coded layer
//   or tensor-name rules.
// - No reuse across changed inputs / epsilon / dtype / shape / layout:
//   all of them are part of the group key, epsilon compared by exact
//   float bit pattern.
// - No statistical equivalence based on algebraic approximations: two
//   norms share a statistic only when their keys are exactly equal.
// - Physical aliasing: canonical gate/up share the GU backing buffer;
//   every other known vector name owns its backing. ANY write to a
//   backing buffer conservatively seals every open group on that whole
//   buffer (even for disjoint slices) — no unsafe alias sharing. An
//   in-place first norm writes its own input backing, so it can never
//   persist a statistic for later reuse.
// - Known effects only: matvec / rmsnorm / gelu_up / scale write their
//   declared vector output; router_topk reads router scores and changes
//   HOST routing state only, it does NOT mutate activation buffers, so
//   open groups survive it. Any other op (attention,
//   expert_matvec_chain, cache_append, embed_row, ...) is opaque and
//   seals ALL open groups.
// - Logical live versions are tracked; a stale SSA reference fails
//   closed with an error (no trap).
// - The plan is fully precomputed: a dense [Action?] indexed by
//   instruction index, consumed by the executor with no per-token
//   scanning or hashing. No caches across token invocations, positions
//   or model resets live here; the compiler plan is reused at prepare
//   time and later integration will additionally enforce epsilon bits,
//   scratch epochs and physical liveness.

import Foundation

public struct RMSStatisticsPlan: Codable, Equatable, Sendable {

    /// Metadata for one reusable-statistic group: >= 2 RMSNorm
    /// instructions sharing one immutable input version and key.
    public struct GroupMetadata: Codable, Equatable, Sendable {
        /// Exact SSA value ID (e.g. "x@1") whose statistic is shared.
        public let inputID: String
        public let width: Int
        public let vectors: Int
        /// Exact Float bit pattern of the epsilon used for the key.
        public let epsilonBits: UInt32
        /// Instruction index of the FIRST use: the one compute.
        public let firstInstructionIndex: Int
        /// Instruction indices of the later uses: the reuses.
        public let useIndices: [Int]
    }

    public enum ActionKind: String, Codable, Equatable, Sendable {
        case compute
        case reuse
    }

    /// Per-instruction action: compute the statistic (and store the
    /// inverse) or reuse a previously computed inverse from `slot`.
    public struct Action: Codable, Equatable, Sendable {
        public let kind: ActionKind
        public let slot: Int
        public let width: Int
        public let vectors: Int
        public let inputID: String
    }

    /// Version of the analysis (grouping key, effect model, backing
    /// map). Bumped whenever the analysis semantics change so compiled
    /// plans can never be interchangeably consumed across versions.
    public static let analysisVersion = 2

    public let analysis: Int
    public let epsilonBits: UInt32
    public let groups: [GroupMetadata]
    /// Dense per-instruction-index actions (nil where no RMS statistic
    /// compute/reuse applies). Constructed only by `analyze`.
    public let actions: [Action?]

    public var groupCount: Int { groups.count }
    public var reuseCount: Int { groups.reduce(0) { $0 + $1.useIndices.count } }

    /// Checked accessor; out-of-range indices return nil, never trap.
    public func action(at instructionIndex: Int) -> Action? {
        guard instructionIndex >= 0, instructionIndex < actions.count else { return nil }
        return actions[instructionIndex]
    }

    /// Exact grouping key: all attrs plus exact input ID, dtype, length,
    /// layout and epsilon bits. Any difference prevents sharing.
    private struct GroupKey: Hashable {
        let inputID: String
        let dtype: GemmaSSAIR.DType
        let length: Int
        let inputSlice: Int
        let outputSlice: Int
        let attrs: [String: Int]
        let epsilonBits: UInt32
    }

    private final class OpenGroup {
        var members: [Int] = []
    }

    /// Ops with KNOWN effects: each writes exactly its declared vector
    /// output and nothing else.
    private static let knownWriters: Set<String> = ["matvec", "rmsnorm", "gelu_up", "scale"]

    /// Canonical physical backing map. gate/up alias one shared GU
    /// buffer; every other known vector name owns its backing. Unknown
    /// names are given a unique per-name state backing so they can never
    /// accidentally collide with a vector backing.
    private static func backing(of name: String) -> String {
        if name == "gate" || name == "up" { return "GU" }
        return GemmaSSA.vectorNames.contains(name) ? name : "state:" + name
    }

    /// Analyze a validated canonical SSA and precompute the reusable RMS
    /// statistics plan. Fail-closed on malformed metadata, duplicate
    /// value IDs, unsupported schema, invalid epsilon and stale SSA
    /// references; unsupported RMSNorm shapes/attrs are SKIPPED (never
    /// throw, never share).
    public static func analyze(ssa: GemmaSSAIR, epsilon: Float) throws -> RMSStatisticsPlan {
        guard ssa.schema == GemmaSSA.schemaVersion else {
            throw GemmaSSAError.badLowering("unsupported schema \(ssa.schema)")
        }
        guard epsilon.isFinite, epsilon > 0 else {
            throw GemmaSSAError.badLowering("epsilon must be finite and positive")
        }
        let epsBits = epsilon.bitPattern

        var byID: [String: GemmaSSAIR.Value] = [:]
        for v in ssa.values {
            guard byID[v.id] == nil else {
                throw GemmaSSAError.badDataflow("duplicate value id \(v.id)")
            }
            byID[v.id] = v
        }

        var written = Set<String>()
        for inst in ssa.instructions { written.insert(inst.output) }

        // Initial logical live versions: for each name, the lowest
        // version NOT produced inside this IR (the entering value).
        var live: [String: Int] = [:]
        var seenNames = Set<String>()
        for v in ssa.values where !seenNames.contains(v.name) {
            seenNames.insert(v.name)
            if let initial = ssa.values
                .filter({ $0.name == v.name && !written.contains($0.id) })
                .min(by: { $0.version < $1.version }) {
                live[v.name] = initial.version
            }
        }

        var open: [GroupKey: OpenGroup] = [:]
        var sealed: [(key: GroupKey, group: OpenGroup)] = []

        for (index, inst) in ssa.instructions.enumerated() {
            // Every input must reference the CURRENTLY-LIVE version.
            for ref in inst.inputs {
                guard let v = byID[ref] else {
                    throw GemmaSSAError.badDataflow("unknown input value \(ref)")
                }
                guard live[v.name] == v.version else {
                    throw GemmaSSAError.badDataflow("stale input reference \(ref)")
                }
            }
            guard let out = byID[inst.output] else {
                throw GemmaSSAError.badDataflow("unknown output value \(inst.output)")
            }
            if let current = live[out.name], out.version <= current {
                throw GemmaSSAError.badDataflow("output \(inst.output) is not a fresh version")
            }

            if knownWriters.contains(inst.op) {
                // Known writer: seal only groups whose input lives in the
                // SAME backing buffer (whole buffer, even for disjoint
                // slices — conservative).
                let writtenBacking = backing(of: out.name)
                for (key, group) in open
                where backing(of: byID[key.inputID]!.name) == writtenBacking {
                    sealed.append((key, group))
                    open.removeValue(forKey: key)
                }
            } else if inst.op == "router_topk" {
                // Reads router scores and mutates HOST routing state
                // only; activation buffers untouched — open groups
                // survive (inputs were already liveness-checked above).
            } else {
                // Unknown / opaque compound op (attention,
                // expert_matvec_chain, cache_append, embed_row, ...):
                // may touch any fixed buffer; seal EVERYTHING.
                for (key, group) in open { sealed.append((key, group)) }
                open.removeAll()
            }

            live[out.name] = out.version

            guard inst.op == "rmsnorm" else { continue }
            guard let inputRef = inst.inputs.first else {
                throw GemmaSSAError.badDataflow("rmsnorm missing input")
            }
            let input = byID[inputRef]!

            // v1 eligibility: f32 vector in/out, attrs limited to
            // width/vectors, vectors == 1, positive width (multiple of
            // 4, bounded by the cooperative kernel cap), input length
            // == output length (width taken from the output length —
            // vectorized multi-norm is skipped), whole-buffer slices
            // (nil or 0). Everything unsupported is skipped, not shared.
            let eligible = input.kind == .vector
                && input.dtype == .f32
                && out.kind == .vector
                && out.dtype == .f32
                && GemmaSSA.vectorNames.contains(input.name)
                && GemmaSSA.vectorNames.contains(out.name)
                && inst.attrs.allSatisfy({ $0.key == "width" || $0.key == "vectors" })
                && (inst.attrs["vectors"] ?? 1) == 1
                && (inst.attrs["width"] == nil || inst.attrs["width"] == out.length)
                && out.length > 0
                && out.length % 4 == 0
                && out.length <= RMSCooperative.maxN
                && input.length == out.length
                && (input.sliceOffset == nil || input.sliceOffset == 0)
                && (out.sliceOffset == nil || out.sliceOffset == 0)

            // An in-place first norm compares PHYSICAL backing, not
            // names: any output aliasing the input's backing buffer
            // (e.g. gate/up on the shared GU buffer) writes its own
            // input backing — it cannot persist a statistic for later
            // reuse, so it never opens a group. The write barrier above
            // has already sealed any group on that backing.
            guard eligible,
                  backing(of: out.name) != backing(of: input.name) else { continue }

            let key = GroupKey(
                inputID: input.id,
                dtype: input.dtype,
                length: input.length,
                inputSlice: 0,
                outputSlice: 0,
                attrs: inst.attrs,
                epsilonBits: epsBits
            )
            let group = open[key] ?? OpenGroup()
            group.members.append(index)
            open[key] = group
        }

        for (key, group) in open { sealed.append((key, group)) }

        // Only segments with >= 2 compatible RMSNorms are reusable.
        let valid = sealed
            .filter { $0.group.members.count >= 2 }
            .sorted { $0.group.members[0] < $1.group.members[0] }

        var groups: [GroupMetadata] = []
        var actions = [Action?](repeating: nil, count: ssa.instructions.count)
        for (slot, entry) in valid.enumerated() {
            let first = entry.group.members[0]
            let uses = Array(entry.group.members.dropFirst())
            let vectors = entry.key.attrs["vectors"] ?? 1
            groups.append(GroupMetadata(
                inputID: entry.key.inputID,
                width: entry.key.length,
                vectors: vectors,
                epsilonBits: entry.key.epsilonBits,
                firstInstructionIndex: first,
                useIndices: uses
            ))
            actions[first] = Action(
                kind: .compute, slot: slot, width: entry.key.length,
                vectors: vectors, inputID: entry.key.inputID
            )
            for use in uses {
                actions[use] = Action(
                    kind: .reuse, slot: slot, width: entry.key.length,
                    vectors: vectors, inputID: entry.key.inputID
                )
            }
        }

        return RMSStatisticsPlan(
            analysis: analysisVersion,
            epsilonBits: epsBits,
            groups: groups,
            actions: actions
        )
    }
}
