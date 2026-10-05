// Pure legality/proof layer for the router/shared FFN overlap schedule.
//
// Nothing here executes, dispatches, or touches Metal, GGUF bytes, or a
// runner. The planner derives a reordered schedule from the VALIDATED
// canonical SSA lowering, the reusable RMS statistics plan, and the graph
// config, and proves it legal by replaying the physical effects of every
// instruction as GemmaForge.executeToken binds them.
//
// Canonical per-layer FFN window (actual lowering order):
//   i0  ffn_norm      rmsnorm  x -> h            statistics compute
//   i1  shared_gate   matvec   h -> gate         GU bytes [0, 4S)
//   i2  shared_up     matvec   h -> up           GU bytes [4S, 8S)
//   i3  shared_gelu   gelu_up  gate, up -> act   reads fused GU [0, 8S)
//   i4  shared_down   matvec   act -> shared
//   i5  shared_post1  rmsnorm  shared -> shared  in place
//   i6  router_norm   rmsnorm  x -> h            statistics reuse
//   i7  router_scale  scale    h -> h            in place
//   i8  router_proj   matvec   h -> router
//   i9  router_topk   router -> rout_L           host routing state
//   i10 moe_norm2     rmsnorm  x -> h            statistics reuse
//   i11 expert chain  h, rout_L -> t3
// Overlap order: i0, i6, i7, i8, i1..i5, i9, i10, ... . Moving the router
// segment ahead of the shared branch would clobber the h version from i0
// (still read by i1/i2), so ONLY the two h versions produced by i6/i7 (and
// their uses in i7/i8) are remapped to a synthetic routerIn resource.
// Every other value, tensor binding, attribute, geometry, statistics
// action and slot is the canonical one; i0 still computes the statistic
// before the moved i6 reuses it, so no arithmetic changes.
//
// Proof scope: the replay models exactly the buffer bindings, byte
// ranges, in-place aliasing, KV/routing state, statistics slots and
// declared scratch that executeToken uses per op, and rejects any op it
// does not model. Effects internal to the runner primitives (Gemma4Runner,
// the batched MoE region, router residency kernels) are outside this
// source and are NOT proven here; runtime integration must bind routerIn
// to a dedicated buffer and re-establish those properties separately.

import Foundation

public enum RouterOverlapPlanError: Error, CustomStringConvertible, Equatable {
    /// Structural, statistics, or proof rejection of the planner input.
    case rejected(String)
    /// Physical replay rejection at a canonical instruction index.
    case replayRejected(index: Int, String)

    public var description: String {
        switch self {
        case .rejected(let m): return "router overlap rejected: \(m)"
        case .replayRejected(let index, let m):
            return "router overlap replay rejected at instruction \(index): \(m)"
        }
    }
}

/// Physical resource touched by an instruction in the replay model.
public enum RouterOverlapResource: Codable, Hashable, Sendable, CustomStringConvertible {
    /// A canonical runner vector buffer (shared mapping with the executor).
    case buffer(GemmaPhysicalBuffer)
    /// Synthetic dedicated router input (planner only; no runtime binding).
    case routerIn
    /// Per-layer K/V cache storage (logical kvstate_L).
    case kvCache(Int)
    /// Per-layer host routing state (logical rout_L).
    case routing(Int)
    /// Persistent reusable-statistics slot buffer (4 bytes per slot).
    case statisticsSlots
    /// Dedicated expert-chain scratch (internal effect of the compound op).
    case expertScratch
    /// Embedding decode scratch (internal effect of embed_row).
    case embedScratch

    public var description: String {
        switch self {
        case .buffer(let b): return "buffer:\(b.rawValue)"
        case .routerIn: return "routerIn"
        case .kvCache(let l): return "kvcache:\(l)"
        case .routing(let l): return "routing:\(l)"
        case .statisticsSlots: return "statisticsSlots"
        case .expertScratch: return "expertScratch"
        case .embedScratch: return "embedScratch"
        }
    }
}

/// Byte interval of one resource holding one logical value.
public struct RouterOverlapPlacement: Equatable, Sendable, CustomStringConvertible {
    public let resource: RouterOverlapResource
    public let lowerByte: Int
    public let upperByte: Int

    public func overlaps(_ other: RouterOverlapPlacement) -> Bool {
        resource == other.resource && lowerByte < other.upperByte
            && other.lowerByte < upperByte
    }

    public var description: String { "\(resource)[\(lowerByte), \(upperByte))" }
}

/// Counters of one successful physical replay.
public struct RouterOverlapReplayReport: Codable, Equatable, Sendable {
    public let instructions: Int
    public let logicalReads: Int
    public let physicalWrites: Int
    public let statisticComputes: Int
    public let statisticReuses: Int
}

private struct Resident: Equatable {
    let lo: Int
    let hi: Int
    let id: String
}

/// Immutable, digest-stamped router overlap plan. Constructed only by
/// `analyze`; a decoded plan must pass `reverify` before any use.
public struct RouterOverlapPlan: Codable, Equatable, Sendable {

    /// Version of the analysis (window shape, remap rule, effect model).
    public static let analysisVersion = 1
    /// Ops whose physical effects the replay models; anything else rejects.
    public static let modeledOps: [String] = [
        "embed_row", "rmsnorm", "matvec", "rope", "copy", "axpy", "add",
        "scale", "gelu_up", "cache_append", "attention", "router_topk",
        "expert_matvec_chain", "softcap",
    ]

    /// One per-layer FFN window, in canonical instruction indices.
    public struct Window: Codable, Equatable, Sendable {
        public let layer: Int
        /// i0 ... i11 (ffn_norm .. expert chain), contiguous.
        public let canonicalIndices: [Int]
        /// Execution order replacing positions i0 ... i9:
        /// i0, i6, i7, i8, i1, i2, i3, i4, i5, i9.
        public let executionOrder: [Int]
        /// Reusable-statistics slot (index into the ORIGINAL plan groups).
        public let statisticsSlot: Int
        public let statisticsInputID: String
        public let computeIndex: Int
        public let reuseIndices: [Int]
        /// h version from i0, read by shared gate/up and kept stable.
        public let sharedNormOutputID: String
        /// h versions produced by i6/i7, bound to routerIn.
        public let remappedValueIDs: [String]
        public let routerScoresID: String
        public let routingStateID: String
    }

    public let analysis: Int
    public let config: GemmaGraphIR.GraphConfig
    /// The original canonical schedule, unchanged.
    public let canonicalSchedule: GemmaSSAIR
    public let canonicalScheduleDigest: String
    public let statisticsPlanDigest: String
    public let statisticsAnalysis: Int
    public let epsilonBits: UInt32
    public let syntheticResource: String
    public let windows: [Window]
    /// Execution position -> canonical instruction index.
    public let permutation: [Int]
    public let remappedValueIDs: [String]
    public let canonicalReplay: RouterOverlapReplayReport
    public let permutedReplay: RouterOverlapReplayReport
    /// Where the control replay (permuted, no remap) was rejected: the
    /// first shared gate, proving the hazard model detects the clobber.
    public let controlRejectedIndex: Int
    /// SHA-256 of the sorted-key JSON of this plan with an empty digest.
    public let digest: String

    /// Remap consumed by the replay (every remapped id -> routerIn).
    public var remap: [String: RouterOverlapResource] {
        Dictionary(uniqueKeysWithValues: remappedValueIDs.map { ($0, .routerIn) })
    }

    public func window(forLayer layer: Int) -> Window? {
        windows.first { $0.layer == layer }
    }

    public func isRemapped(_ valueID: String) -> Bool {
        remappedValueIDs.contains(valueID)
    }

    private static func reject(_ m: String) -> RouterOverlapPlanError {
        .rejected(m)
    }

    private static func encodedDigest<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return SHA256.hex(SHA256.hash(Array(try encoder.encode(value))))
    }

    /// Additive prepare-time helper: the digest form `analyze` stamps into
    /// statisticsPlanDigest, so a runtime binding can compare it without
    /// re-running the analysis.
    public static func digestOfStatisticsPlan(_ statisticsPlan: RMSStatisticsPlan) throws -> String {
        try encodedDigest(statisticsPlan)
    }

    private func withDigest(_ digest: String) -> RouterOverlapPlan {
        RouterOverlapPlan(
            analysis: analysis, config: config, canonicalSchedule: canonicalSchedule,
            canonicalScheduleDigest: canonicalScheduleDigest,
            statisticsPlanDigest: statisticsPlanDigest,
            statisticsAnalysis: statisticsAnalysis, epsilonBits: epsilonBits,
            syntheticResource: syntheticResource, windows: windows,
            permutation: permutation, remappedValueIDs: remappedValueIDs,
            canonicalReplay: canonicalReplay, permutedReplay: permutedReplay,
            controlRejectedIndex: controlRejectedIndex, digest: digest)
    }

    /// Integrity check of the stamped digest (not authentication).
    public static func verifyDigest(_ plan: RouterOverlapPlan) throws {
        guard try encodedDigest(plan.withDigest("")) == plan.digest else {
            throw reject("plan digest mismatch")
        }
    }

    /// Digest check plus equality with a fresh analysis of the inputs.
    public static func reverify(_ plan: RouterOverlapPlan, ssa: GemmaSSAIR,
                                config: GemmaGraphIR.GraphConfig,
                                statisticsPlan: RMSStatisticsPlan,
                                epsilon: Float) throws {
        try verifyDigest(plan)
        let fresh = try analyze(ssa: ssa, config: config,
                                statisticsPlan: statisticsPlan, epsilon: epsilon)
        guard fresh == plan else {
            throw reject("plan does not match a fresh analysis of this schedule")
        }
    }

    /// Graph entry: the schedule must be EXACTLY the canonical lowering of
    /// a graph that structurally matches the expected op sequence.
    public static func analyze(graph: GemmaGraphIR, ssa: GemmaSSAIR,
                               statisticsPlan: RMSStatisticsPlan,
                               epsilon: Float) throws -> RouterOverlapPlan {
        guard graph.schema == 1, graph.model == "gemma4-26b-a4b" else {
            throw reject("graph schema/model tag unsupported")
        }
        guard graph.ops.allSatisfy({ GemmaGraph.supportedOps.contains($0.op) }),
              graph.ops == GemmaGraph.expectedSequence(graph.config) else {
            throw reject("graph is not the expected op sequence for its config")
        }
        let canonical: GemmaSSAIR
        do {
            canonical = try GemmaSSA.lower(graph)
        } catch {
            throw reject("canonical lowering failed: \(error)")
        }
        guard ssa == canonical else {
            throw reject("schedule is not the canonical lowering of this graph")
        }
        return try analyze(ssa: ssa, config: graph.config,
                           statisticsPlan: statisticsPlan, epsilon: epsilon)
    }

    /// Canonical per-layer SSA op sequence (local: v_proj matvec; global:
    /// v_eq_k copy). The FFN window starts at offset 14.
    private static func layerOps(global: Bool) -> [String] {
        ["rmsnorm", "matvec", "rmsnorm", "rope", "matvec",
         global ? "copy" : "matvec", "rmsnorm", "rope", "rmsnorm",
         "cache_append", "attention", "matvec", "rmsnorm", "axpy",
         "rmsnorm", "matvec", "matvec", "gelu_up", "matvec", "rmsnorm",
         "rmsnorm", "scale", "matvec", "router_topk", "rmsnorm",
         "expert_matvec_chain",
         "rmsnorm", "add", "rmsnorm", "axpy", "scale"]
    }

    /// Fail-closed planner over a validated SSA, its statistics plan and
    /// the graph config. Order of checks: dataflow, window structure,
    /// value scope / x identity, statistics, local/global layer pattern,
    /// physical replays (canonical, permuted+remap, control).
    public static func analyze(ssa: GemmaSSAIR, config c: GemmaGraphIR.GraphConfig,
                               statisticsPlan: RMSStatisticsPlan,
                               epsilon: Float) throws -> RouterOverlapPlan {
        guard epsilon.isFinite, epsilon > 0 else {
            throw reject("epsilon must be finite and positive")
        }
        guard c.layerCount > 0, c.hidden > 0, c.sharedIntermediate > 0,
              c.expertCount > 0, c.expertUsedCount > 0,
              c.expertUsedCount <= c.expertCount else {
            throw reject("graph config geometry invalid")
        }
        do {
            try GemmaSSA.validate(ssa, config: c)
        } catch {
            throw reject("canonical SSA dataflow invalid: \(error)")
        }
        let insts = ssa.instructions
        let n = insts.count
        var byID = [String: GemmaSSAIR.Value]()
        for v in ssa.values { byID[v.id] = v }
        var producer = [String: Int]()
        var consumers = [String: [Int]]()
        for (j, inst) in insts.enumerated() {
            guard producer[inst.output] == nil else {
                throw reject("value \(inst.output) produced twice")
            }
            producer[inst.output] = j
            for id in inst.inputs { consumers[id, default: []].append(j) }
        }
        let H = c.hidden, S = c.sharedIntermediate, E = c.expertCount
        let K = c.expertUsedCount, EI = c.expertIntermediate

        struct Spec {
            let op: String
            let tensors: [String]
            let attrs: [String: Int]
            let name: String
            let length: Int
            let kind: GemmaSSAIR.Kind
            let slice: Int?
        }

        // ---- window structure, value scope, x identity ----
        var located: [(layer: Int, indices: [Int], xID: String)] = []
        for L in 0..<c.layerCount {
            let p = "blk.\(L)."
            let candidates = insts.indices.filter {
                insts[$0].layer == L && insts[$0].op == "rmsnorm"
                    && insts[$0].tensors == [p + "ffn_norm.weight"]
            }
            guard candidates.count == 1 else {
                throw reject("missing or ambiguous FFN window for layer \(L) "
                    + "(\(candidates.count) ffn_norm candidates)")
            }
            let i0 = candidates[0]
            guard i0 + 11 < n else {
                throw reject("missing FFN window tail for layer \(L)")
            }
            let idx = Array(i0...(i0 + 11))
            let w = idx.map { insts[$0] }
            let specs: [Spec] = [
                Spec(op: "rmsnorm", tensors: [p + "ffn_norm.weight"], attrs: [:],
                     name: "h", length: H, kind: .vector, slice: nil),
                Spec(op: "matvec", tensors: [p + "ffn_gate.weight"],
                     attrs: ["rows": S, "offset": 0],
                     name: "gate", length: S, kind: .vector, slice: 0),
                Spec(op: "matvec", tensors: [p + "ffn_up.weight"],
                     attrs: ["rows": S, "offset": S],
                     name: "up", length: S, kind: .vector, slice: S),
                Spec(op: "gelu_up", tensors: [], attrs: ["width": S],
                     name: "act", length: S, kind: .vector, slice: nil),
                Spec(op: "matvec", tensors: [p + "ffn_down.weight"], attrs: ["rows": H],
                     name: "shared", length: H, kind: .vector, slice: nil),
                Spec(op: "rmsnorm", tensors: [p + "post_ffw_norm_1.weight"], attrs: [:],
                     name: "shared", length: H, kind: .vector, slice: nil),
                Spec(op: "rmsnorm", tensors: [], attrs: [:],
                     name: "h", length: H, kind: .vector, slice: nil),
                Spec(op: "scale", tensors: [], attrs: ["routerScale": 1],
                     name: "h", length: H, kind: .vector, slice: nil),
                Spec(op: "matvec", tensors: [p + "ffn_gate_inp.weight"], attrs: ["rows": E],
                     name: "router", length: E, kind: .vector, slice: nil),
                Spec(op: "router_topk", tensors: [], attrs: ["topK": K, "experts": E],
                     name: "rout_\(L)", length: K, kind: .routing, slice: nil),
                Spec(op: "rmsnorm", tensors: [p + "pre_ffw_norm_2.weight"], attrs: [:],
                     name: "h", length: H, kind: .vector, slice: nil),
                Spec(op: "expert_matvec_chain",
                     tensors: [p + "ffn_gate_up_exps.weight", p + "ffn_down_exps.weight"],
                     attrs: ["experts": K, "intermediate": EI],
                     name: "t3", length: H, kind: .vector, slice: nil),
            ]
            for k in 0..<12 {
                let inst = w[k]
                let s = specs[k]
                func bad(_ m: String) -> RouterOverlapPlanError {
                    reject("window layer \(L) i\(k) (\(inst.op)): \(m)")
                }
                guard inst.layer == L else {
                    throw bad("layer \(String(describing: inst.layer)) != \(L)")
                }
                guard inst.op == s.op else { throw bad("op is not \(s.op)") }
                guard inst.tensors == s.tensors else {
                    throw bad("tensor binding \(inst.tensors) != \(s.tensors)")
                }
                guard inst.attrs == s.attrs else {
                    throw bad("attrs/geometry \(inst.attrs) != \(s.attrs)")
                }
                guard let out = byID[inst.output], out.name == s.name,
                      out.length == s.length, out.kind == s.kind,
                      out.sliceOffset == s.slice, out.dtype == .f32 else {
                    throw bad("output \(inst.output) name/length/kind/layout differ from \(s.name)")
                }
            }
            let o = w.map(\.output)
            let edges: [Int: [String]] = [
                1: [o[0]], 2: [o[0]], 3: [o[1], o[2]], 4: [o[3]], 5: [o[4]],
                7: [o[6]], 8: [o[7]], 9: [o[8]], 11: [o[10], o[9]],
            ]
            for k in edges.keys.sorted() where w[k].inputs != edges[k]! {
                throw reject("window layer \(L) i\(k) (\(w[k].op)): inputs "
                    + "\(w[k].inputs) != \(edges[k]!)")
            }
            for k in [0, 6, 10] where w[k].inputs.count != 1 {
                throw reject("window layer \(L) i\(k): exactly one input expected")
            }
            func only(_ id: String, _ allowed: [Int], _ what: String) throws {
                let uses = consumers[id] ?? []
                guard uses == allowed else {
                    throw reject("window layer \(L): \(what) \(id) used at \(uses), "
                        + "allowed only at \(allowed)")
                }
            }
            try only(o[6], [idx[7]], "remapped router_norm value escapes its segment:")
            try only(o[7], [idx[8]], "remapped router_scale value escapes its segment:")
            try only(o[8], [idx[9]], "router scores not consumed only by router_topk:")
            try only(o[9], [idx[11]], "routing state not consumed only by the expert chain:")
            try only(o[0], [idx[1], idx[2]], "shared FFN norm output read outside shared gate/up:")
            let xID = w[0].inputs[0]
            guard let x = byID[xID], x.name == "x", x.kind == .vector,
                  x.length == H, x.sliceOffset == nil,
                  w[6].inputs == [xID], w[10].inputs == [xID] else {
                throw reject("window layer \(L): ffn_norm, router_norm and moe_norm2 "
                    + "do not read one identical x version")
            }
            located.append((layer: L, indices: idx, xID: xID))
        }

        // ---- statistics: every cached action, slot, epsilon, version ----
        guard statisticsPlan.analysis == RMSStatisticsPlan.analysisVersion else {
            throw reject("statistics plan analysis \(statisticsPlan.analysis) != "
                + "\(RMSStatisticsPlan.analysisVersion)")
        }
        guard statisticsPlan.epsilonBits == epsilon.bitPattern else {
            throw reject("statistics epsilon bits differ from the planner epsilon")
        }
        guard statisticsPlan.actions.count == n,
              statisticsPlan.groups.count == c.layerCount,
              statisticsPlan.reuseCount == 2 * c.layerCount else {
            throw reject("statistics plan shape is not \(c.layerCount) groups / "
                + "\(2 * c.layerCount) reuses over \(n) instructions")
        }
        var windows: [Window] = []
        var slots = Set<Int>()
        var planned = Set<Int>()
        for entry in located {
            let idx = entry.indices
            let L = entry.layer
            guard let a0 = statisticsPlan.action(at: idx[0]),
                  let a6 = statisticsPlan.action(at: idx[6]),
                  let a10 = statisticsPlan.action(at: idx[10]) else {
                throw reject("statistics: layer \(L) window lacks compute/reuse actions")
            }
            guard a0.kind == .compute, a6.kind == .reuse, a10.kind == .reuse else {
                throw reject("statistics: layer \(L) actions are not compute/reuse/reuse")
            }
            let slot = a0.slot
            guard a6.slot == slot, a10.slot == slot, slot >= 0,
                  slot < statisticsPlan.groups.count,
                  slots.insert(slot).inserted else {
                throw reject("statistics: layer \(L) slot \(slot) not shared or not unique")
            }
            for a in [a0, a6, a10] {
                guard a.inputID == entry.xID, a.width == H, a.vectors == 1 else {
                    throw reject("statistics: layer \(L) action input/width/vectors mismatch")
                }
            }
            let g = statisticsPlan.groups[slot]
            guard g.inputID == entry.xID, g.width == H, g.vectors == 1,
                  g.epsilonBits == epsilon.bitPattern,
                  g.firstInstructionIndex == idx[0],
                  g.useIndices == [idx[6], idx[10]] else {
                throw reject("statistics: layer \(L) group metadata mismatch")
            }
            planned.formUnion([idx[0], idx[6], idx[10]])
            let o = idx.map { insts[$0].output }
            windows.append(Window(
                layer: L, canonicalIndices: idx,
                executionOrder: [idx[0], idx[6], idx[7], idx[8], idx[1], idx[2],
                                 idx[3], idx[4], idx[5], idx[9]],
                statisticsSlot: slot, statisticsInputID: entry.xID,
                computeIndex: idx[0], reuseIndices: [idx[6], idx[10]],
                sharedNormOutputID: o[0], remappedValueIDs: [o[6], o[7]],
                routerScoresID: o[8], routingStateID: o[9]))
        }
        for (i, a) in statisticsPlan.actions.enumerated()
        where a != nil && !planned.contains(i) {
            throw reject("statistics action at instruction \(i) lies outside every FFN window")
        }
        let derived: RMSStatisticsPlan
        do {
            derived = try RMSStatisticsPlan.analyze(ssa: ssa, epsilon: epsilon)
        } catch {
            throw reject("statistics re-derivation failed: \(error)")
        }
        guard derived == statisticsPlan else {
            throw reject("statistics plan is not the analysis of this schedule")
        }

        // ---- local/global layer pattern of the whole schedule ----
        let perLayer = layerOps(global: false).count
        let (body, overflow) = perLayer.multipliedReportingOverflow(by: c.layerCount)
        guard !overflow, n == 1 + body + 3 else {
            throw reject("layer pattern: schedule length \(n) is not 1 + \(perLayer) x "
                + "\(c.layerCount) + 3")
        }
        guard insts[0].op == "embed_row", insts[0].layer == nil,
              insts[n - 3].op == "rmsnorm", insts[n - 2].op == "matvec",
              insts[n - 1].op == "softcap",
              insts[(n - 3)...].allSatisfy({ $0.layer == nil }) else {
            throw reject("layer pattern: embedding/final tail mismatch")
        }
        for L in 0..<c.layerCount {
            let isGlobal = c.globalLayers.contains(L)
            let start = 1 + perLayer * L
            let block = Array(insts[start..<(start + perLayer)])
            let headDim = isGlobal ? c.globalHeadDim : c.localHeadDim
            let kvHeads = isGlobal ? c.globalKVHeads : c.localKVHeads
            let g = isGlobal ? 1 : 0
            guard block.allSatisfy({ $0.layer == L }),
                  block.map(\.op) == layerOps(global: isGlobal) else {
                throw reject("layer pattern: layer \(L) (\(isGlobal ? "global" : "local")) "
                    + "does not match the canonical op sequence")
            }
            guard block[3].attrs == ["heads": c.qHeads, "half": headDim / 2, "global": g],
                  block[7].attrs == ["heads": kvHeads, "half": headDim / 2, "global": g],
                  block[9].attrs == ["kvHeads": kvHeads, "headDim": headDim],
                  block[10].attrs == ["heads": c.qHeads, "kvHeads": kvHeads,
                                      "headDim": headDim,
                                      "window": isGlobal ? 0 : c.slidingWindow],
                  block[5].tensors == (isGlobal ? [] : ["blk.\(L).attn_v.weight"]),
                  located[L].indices[0] == start + 14 else {
                throw reject("layer pattern: layer \(L) local/global geometry or window "
                    + "position mismatch")
            }
        }

        // ---- permutation ----
        var permutation = Array(0..<n)
        for w in windows {
            for (k, j) in w.executionOrder.enumerated() {
                permutation[w.canonicalIndices[0] + k] = j
            }
        }
        guard Set(permutation).count == n else {
            throw reject("derived order is not a permutation")
        }
        let remappedIDs = windows.flatMap(\.remappedValueIDs)
        let remap = Dictionary(uniqueKeysWithValues:
            remappedIDs.map { ($0, RouterOverlapResource.routerIn) })

        // ---- physical replays ----
        let canonicalReport: RouterOverlapReplayReport
        do {
            canonicalReport = try physicalReplay(
                ssa: ssa, layerCount: c.layerCount, order: Array(0..<n),
                remap: [:], statisticsPlan: statisticsPlan)
        } catch {
            throw reject("canonical physical replay failed: \(error)")
        }
        let permutedReport: RouterOverlapReplayReport
        do {
            permutedReport = try physicalReplay(
                ssa: ssa, layerCount: c.layerCount, order: permutation,
                remap: remap, statisticsPlan: statisticsPlan)
        } catch {
            throw reject("permuted physical replay with routerIn remap failed: \(error)")
        }
        var controlIndex = -1
        do {
            _ = try physicalReplay(
                ssa: ssa, layerCount: c.layerCount, order: permutation,
                remap: [:], statisticsPlan: statisticsPlan)
        } catch RouterOverlapPlanError.replayRejected(let index, _) {
            controlIndex = index
        }
        guard controlIndex == windows[0].canonicalIndices[1] else {
            throw reject("control replay without the routerIn remap did not reject at the "
                + "first shared gate (got \(controlIndex)); the hazard model is not trusted")
        }

        let scheduleDigest: String
        do {
            scheduleDigest = try GemmaSSA.digest(ssa)
        } catch {
            throw reject("schedule digest failed: \(error)")
        }
        let unsigned = RouterOverlapPlan(
            analysis: analysisVersion, config: c, canonicalSchedule: ssa,
            canonicalScheduleDigest: scheduleDigest,
            statisticsPlanDigest: try encodedDigest(statisticsPlan),
            statisticsAnalysis: statisticsPlan.analysis,
            epsilonBits: epsilon.bitPattern,
            syntheticResource: RouterOverlapResource.routerIn.description,
            windows: windows, permutation: permutation,
            remappedValueIDs: remappedIDs,
            canonicalReplay: canonicalReport, permutedReplay: permutedReport,
            controlRejectedIndex: controlIndex, digest: "")
        return unsigned.withDigest(try encodedDigest(unsigned))
    }

    /// Replays `order` over the physical effects GemmaForge.executeToken
    /// gives each op. A read succeeds only if the exact byte interval of
    /// the value's placement still holds that value (any overlapping write
    /// since evicts it); every write evicts all overlapping residents.
    /// Statistics reuse requires its slot to hold the statistic of the same
    /// input version. Unknown ops are rejected.
    public static func physicalReplay(ssa: GemmaSSAIR, layerCount: Int,
                                      order: [Int],
                                      remap: [String: RouterOverlapResource],
                                      statisticsPlan: RMSStatisticsPlan?) throws
            -> RouterOverlapReplayReport {
        let insts = ssa.instructions
        let n = insts.count
        guard n > 0, order.count == n, Set(order).count == n,
              order.allSatisfy({ $0 >= 0 && $0 < n }) else {
            throw reject("replay order is not a permutation of the schedule")
        }
        guard order.last == n - 1 else {
            throw reject("replay order must keep the terminal instruction last")
        }
        var byID = [String: GemmaSSAIR.Value]()
        for v in ssa.values {
            guard byID[v.id] == nil else { throw reject("duplicate value \(v.id)") }
            byID[v.id] = v
        }
        for id in remap.keys.sorted() {
            guard let v = byID[id], v.kind == .vector, (v.sliceOffset ?? 0) == 0 else {
                throw reject("remap of unknown, non-vector or sliced value \(id)")
            }
        }
        var residents = [RouterOverlapResource: [Resident]]()
        var reads = 0, writes = 0, computes = 0, reuses = 0

        func value(_ id: String, _ i: Int) throws -> GemmaSSAIR.Value {
            guard let v = byID[id] else {
                throw RouterOverlapPlanError.replayRejected(index: i, "undefined value \(id)")
            }
            return v
        }
        func stateLayer(_ name: String, _ prefix: String, _ i: Int) throws -> Int {
            guard name.hasPrefix(prefix), let l = Int(name.dropFirst(prefix.count)),
                  l >= 0, l < layerCount else {
                throw RouterOverlapPlanError.replayRejected(
                    index: i, "state value \(name) has no valid layer")
            }
            return l
        }
        func place(_ v: GemmaSSAIR.Value, _ i: Int) throws -> RouterOverlapPlacement {
            let (bytes, ov) = v.length.multipliedReportingOverflow(by: 4)
            guard v.length > 0, !ov else {
                throw RouterOverlapPlanError.replayRejected(index: i, "bad length for \(v.id)")
            }
            if let r = remap[v.id] {
                return RouterOverlapPlacement(resource: r, lowerByte: 0, upperByte: bytes)
            }
            switch v.kind {
            case .vector:
                guard let key = GemmaPhysicalBuffer.of(valueName: v.name) else {
                    throw RouterOverlapPlanError.replayRejected(
                        index: i, "no physical binding for \(v.id)")
                }
                let slice = v.sliceOffset ?? 0
                guard slice >= 0, slice == 0 || key == .gu else {
                    throw RouterOverlapPlanError.replayRejected(
                        index: i, "slice offset on a non-fused value \(v.id)")
                }
                let (lo, o1) = slice.multipliedReportingOverflow(by: 4)
                let (hi, o2) = lo.addingReportingOverflow(bytes)
                guard !o1, !o2 else {
                    throw RouterOverlapPlanError.replayRejected(
                        index: i, "byte range overflow for \(v.id)")
                }
                return RouterOverlapPlacement(resource: .buffer(key), lowerByte: lo, upperByte: hi)
            case .kvstate:
                let l = try stateLayer(v.name, "kvstate_", i)
                return RouterOverlapPlacement(resource: .kvCache(l), lowerByte: 0, upperByte: bytes)
            case .routing:
                let l = try stateLayer(v.name, "rout_", i)
                return RouterOverlapPlacement(resource: .routing(l), lowerByte: 0, upperByte: bytes)
            }
        }
        func holds(_ p: RouterOverlapPlacement, _ token: String) -> Bool {
            (residents[p.resource] ?? []).contains(
                Resident(lo: p.lowerByte, hi: p.upperByte, id: token))
        }
        func read(_ id: String, _ i: Int) throws -> RouterOverlapPlacement {
            let p = try place(try value(id, i), i)
            guard holds(p, id) else {
                throw RouterOverlapPlanError.replayRejected(
                    index: i, "\(id) is not intact at \(p) (clobbered by an overlapping "
                        + "write or not yet produced)")
            }
            reads += 1
            return p
        }
        func write(_ p: RouterOverlapPlacement, _ token: String) {
            var list = residents[p.resource] ?? []
            list.removeAll { $0.lo < p.upperByte && p.lowerByte < $0.hi }
            list.append(Resident(lo: p.lowerByte, hi: p.upperByte, id: token))
            residents[p.resource] = list
            writes += 1
        }

        for i in order {
            let inst = insts[i]
            func bad(_ m: String) -> RouterOverlapPlanError {
                .replayRejected(index: i, "\(inst.op): \(m)")
            }
            func requireLayer() throws -> Int {
                guard let l = inst.layer, l >= 0, l < layerCount else {
                    throw bad("layer missing or out of range")
                }
                return l
            }
            let out = try value(inst.output, i)
            let outP = try place(out, i)
            switch inst.op {
            case "embed_row":
                guard inst.inputs.isEmpty, outP.lowerByte == 0 else {
                    throw bad("embedding has no inputs and writes its output at byte 0")
                }
                write(RouterOverlapPlacement(resource: .embedScratch, lowerByte: 0,
                                             upperByte: outP.upperByte), "scratch@\(i)")
                write(outP, out.id)
            case "rmsnorm":
                guard inst.inputs.count == 1 else { throw bad("one input expected") }
                let inP = try read(inst.inputs[0], i)
                guard inP.lowerByte == 0, outP.lowerByte == 0 else {
                    throw bad("rmsnorm binds whole buffers at byte 0")
                }
                if let plan = statisticsPlan, let action = plan.action(at: i) {
                    guard action.inputID == inst.inputs[0], action.slot >= 0,
                          action.slot < plan.groupCount else {
                        throw bad("statistics action input/slot invalid")
                    }
                    let slotP = RouterOverlapPlacement(
                        resource: .statisticsSlots, lowerByte: action.slot * 4,
                        upperByte: action.slot * 4 + 4)
                    let token = "stat:" + action.inputID
                    switch action.kind {
                    case .compute:
                        write(slotP, token)
                        computes += 1
                    case .reuse:
                        guard holds(slotP, token) else {
                            throw bad("statistic slot \(action.slot) not ready for \(action.inputID)")
                        }
                        reuses += 1
                    }
                }
                write(outP, out.id)
            case "matvec":
                guard inst.inputs.count == 1, inst.tensors.count == 1,
                      let rows = inst.attrs["rows"], rows == out.length else {
                    throw bad("operands/rows invalid")
                }
                let inP = try read(inst.inputs[0], i)
                guard inP.lowerByte == 0 else {
                    throw bad("matvec reads its input from byte 0, but it is placed at \(inP)")
                }
                let offset = inst.attrs["offset"] ?? 0
                guard offset >= 0, offset < (1 << 40),
                      offset * 4 == outP.lowerByte,
                      (offset + rows) * 4 == outP.upperByte else {
                    throw bad("physical write range from offset \(offset) x \(rows) rows "
                        + "disagrees with output placement \(outP)")
                }
                guard !inP.overlaps(outP) else { throw bad("input and output overlap") }
                write(outP, out.id)
            case "rope", "softcap":
                guard inst.inputs.count == 1 else { throw bad("one input expected") }
                let inP = try read(inst.inputs[0], i)
                guard inP == outP, outP.lowerByte == 0 else {
                    throw bad("in-place op requires input and output on one placement")
                }
                write(outP, out.id)
            case "copy":
                guard inst.inputs.count == 1 else { throw bad("one input expected") }
                let inP = try read(inst.inputs[0], i)
                guard inP.lowerByte == 0, outP.lowerByte == 0, !inP.overlaps(outP) else {
                    throw bad("copy binds disjoint whole buffers at byte 0")
                }
                write(outP, out.id)
            case "scale":
                guard inst.inputs.count == 1,
                      (inst.attrs["routerScale"] == 1) != (inst.attrs["layerScale"] == 1) else {
                    throw bad("exactly one of routerScale/layerScale expected")
                }
                _ = try requireLayer()
                let inP = try read(inst.inputs[0], i)
                guard inP == outP, outP.lowerByte == 0 else {
                    throw bad("scale runs in place on its output binding; input at \(inP) "
                        + "is not the output placement \(outP)")
                }
                write(outP, out.id)
            case "axpy":
                guard inst.inputs.count == 2 else { throw bad("two inputs expected") }
                let acc = try read(inst.inputs[0], i)
                let addend = try read(inst.inputs[1], i)
                guard acc == outP, outP.lowerByte == 0, addend.lowerByte == 0,
                      !addend.overlaps(outP) else {
                    throw bad("accumulator must alias the output and the addend must not")
                }
                write(outP, out.id)
            case "add":
                guard inst.inputs.count == 2 else { throw bad("two inputs expected") }
                let a = try read(inst.inputs[0], i)
                let b = try read(inst.inputs[1], i)
                guard a.lowerByte == 0, b.lowerByte == 0, outP.lowerByte == 0 else {
                    throw bad("add binds whole buffers at byte 0")
                }
                write(outP, out.id)
            case "gelu_up":
                guard inst.inputs.count == 2, let width = inst.attrs["width"],
                      width > 0, width < (1 << 40) else {
                    throw bad("operands/width invalid")
                }
                let g = try read(inst.inputs[0], i)
                let u = try read(inst.inputs[1], i)
                guard g.resource == u.resource, g.lowerByte == 0,
                      g.upperByte == width * 4, u.lowerByte == width * 4,
                      u.upperByte == width * 8 else {
                    throw bad("the kernel reads the fused pair [0, \(width * 8)) from the gate "
                        + "binding; gate \(g) / up \(u) are not that contiguous pair")
                }
                guard outP.lowerByte == 0, !outP.overlaps(g), !outP.overlaps(u) else {
                    throw bad("output aliases the fused gate|up pair")
                }
                write(outP, out.id)
            case "cache_append":
                let l = try requireLayer()
                guard inst.inputs.count == 2 else { throw bad("two inputs expected") }
                let k = try read(inst.inputs[0], i)
                let v = try read(inst.inputs[1], i)
                guard k.lowerByte == 0, v.lowerByte == 0, outP.resource == .kvCache(l) else {
                    throw bad("K/V sources or layer cache binding invalid")
                }
                write(outP, out.id)
            case "attention":
                let l = try requireLayer()
                guard inst.inputs.count == 2 else { throw bad("two inputs expected") }
                let q = try read(inst.inputs[0], i)
                let kv = try read(inst.inputs[1], i)
                guard kv.resource == .kvCache(l), q.lowerByte == 0,
                      outP.lowerByte == 0, !outP.overlaps(q) else {
                    throw bad("query/cache/output bindings invalid")
                }
                write(outP, out.id)
            case "router_topk":
                let l = try requireLayer()
                guard inst.inputs.count == 1, let experts = inst.attrs["experts"],
                      experts > 0, experts < (1 << 40) else {
                    throw bad("operands/experts invalid")
                }
                let scores = try read(inst.inputs[0], i)
                guard scores.lowerByte == 0, scores.upperByte == experts * 4,
                      outP.resource == .routing(l) else {
                    throw bad("score readback range or routing state binding invalid")
                }
                write(outP, out.id)
            case "expert_matvec_chain":
                let l = try requireLayer()
                guard inst.inputs.count == 2 else { throw bad("two inputs expected") }
                let x = try read(inst.inputs[0], i)
                let r = try read(inst.inputs[1], i)
                guard r.resource == .routing(l), x.lowerByte == 0, outP.lowerByte == 0,
                      x.resource != outP.resource else {
                    throw bad("expert chain needs distinct input/output buffers and its "
                        + "own layer routing state")
                }
                write(RouterOverlapPlacement(resource: .expertScratch, lowerByte: 0,
                                             upperByte: 1), "scratch@\(i)")
                write(outP, out.id)
            default:
                throw bad("no modeled physical effect; unknown ops are rejected")
            }
        }
        let terminal = try value(insts[n - 1].output, n - 1)
        guard holds(try place(terminal, n - 1), terminal.id) else {
            throw RouterOverlapPlanError.replayRejected(
                index: n - 1, "terminal output \(terminal.id) is not intact")
        }
        return RouterOverlapReplayReport(
            instructions: n, logicalReads: reads, physicalWrites: writes,
            statisticComputes: computes, statisticReuses: reuses)
    }
}
