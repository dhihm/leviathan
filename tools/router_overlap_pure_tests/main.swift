// Standalone PURE tests for RouterOverlapPlan (router/shared FFN overlap
// legality). Builds the canonical Gemma graph IR from the actual config,
// or decodes an existing graph JSON passed with --graph PATH (metadata
// only), lowers it, and checks the plan, its permutation, its physical
// replay proof and its fail-closed rejections. No GGUF, no Metal device,
// no model evaluation, no weight hashing. Exits nonzero on any failure.

import Foundation
import LeviathanCore

enum TestSetupError: Error, CustomStringConvertible {
    case missing(String)

    var description: String {
        switch self {
        case .missing(let what): return "test setup: \(what) missing"
        }
    }
}

/// The SSA, statistics plan and overlap plan initializers are internal
/// to LeviathanCore, so mutations go through their Codable JSON form.
func mutated<T: Codable>(_ value: T,
                         _ edit: (inout [String: Any]) throws -> Void) throws -> T {
    let data = try JSONEncoder().encode(value)
    guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw TestSetupError.missing("JSON object")
    }
    try edit(&object)
    let edited = try JSONSerialization.data(withJSONObject: object)
    return try JSONDecoder().decode(T.self, from: edited)
}

func editInstruction(_ object: inout [String: Any], _ index: Int,
                     _ edit: (inout [String: Any]) -> Void) throws {
    guard var list = object["instructions"] as? [[String: Any]],
          index >= 0, index < list.count else {
        throw TestSetupError.missing("instruction \(index)")
    }
    edit(&list[index])
    object["instructions"] = list
}

func editValue(_ object: inout [String: Any], _ id: String,
               _ edit: (inout [String: Any]) -> Void) throws {
    let parts = id.split(separator: "@")
    guard parts.count == 2, let version = Int(parts[1]),
          var values = object["values"] as? [[String: Any]],
          let at = values.firstIndex(where: { candidate in
              (candidate["name"] as? String) == String(parts[0])
                  && (candidate["version"] as? Int) == version
          }) else {
        throw TestSetupError.missing("value \(id)")
    }
    edit(&values[at])
    object["values"] = values
}

func editAction(_ object: inout [String: Any], _ index: Int,
                _ edit: (inout [String: Any]) -> Void) throws {
    guard var actions = object["actions"] as? [Any],
          index >= 0, index < actions.count else {
        throw TestSetupError.missing("action \(index)")
    }
    var action = actions[index] as? [String: Any] ?? [:]
    edit(&action)
    actions[index] = action
    object["actions"] = actions
}

func setAttr(_ instruction: inout [String: Any], _ key: String, _ value: Int) {
    var attrs = instruction["attrs"] as? [String: Any] ?? [:]
    attrs[key] = value
    instruction["attrs"] = attrs
}

/// Read-only actual model config (no weights).
func actualConfig(layers: Int = 30) -> GemmaGraphIR.GraphConfig {
    GemmaGraphIR.GraphConfig(
        layerCount: layers, hidden: 2816, qHeads: 16, localKVHeads: 8,
        globalKVHeads: 2, localHeadDim: 256, globalHeadDim: 512,
        slidingWindow: 1024, vocabSize: 262144, expertCount: 128,
        expertUsedCount: 8, expertIntermediate: 704, sharedIntermediate: 2112,
        finalSoftcap: 30, globalLayers: [5, 11, 17, 23, 29])
}

final class Harness {
    private(set) var passed = 0
    private(set) var failed = 0

    func check(_ condition: Bool, _ name: String) {
        if condition {
            passed += 1
        } else {
            failed += 1
            print("FAIL \(name)")
        }
    }

    func expectNoThrow(_ name: String, _ body: () throws -> Void) {
        do {
            try body()
            passed += 1
        } catch {
            failed += 1
            print("FAIL \(name): \(error)")
        }
    }

    func expectRejection(_ name: String, containing needle: String? = nil,
                         _ body: () throws -> Void) {
        do {
            try body()
            failed += 1
            print("FAIL \(name): accepted, expected a rejection")
        } catch {
            let text = "\(error)"
            if let needle, !text.contains(needle) {
                failed += 1
                print("FAIL \(name): rejected for an unexpected reason: \(text)")
            } else {
                passed += 1
                print("ok   \(name): \(text)")
            }
        }
    }

    func replayRejectionIndex(_ body: () throws -> Void) -> Int? {
        do {
            try body()
            return nil
        } catch RouterOverlapPlanError.replayRejected(let index, _) {
            return index
        } catch {
            return -1
        }
    }

    func recordUnexpected(_ error: Error) {
        failed += 1
        print("FAIL unexpected error: \(error)")
    }

    func run(graphPath: String?) throws {
        let synthetic = GemmaGraphIR(schema: 1, model: "gemma4-26b-a4b",
                                     config: actualConfig(),
                                     ops: GemmaGraph.expectedSequence(actualConfig()))
        var graph = synthetic
        if let graphPath {
            let decoded = try GemmaGraph.parse(
                try Data(contentsOf: URL(fileURLWithPath: graphPath)))
            check(decoded == synthetic,
                  "decoded --graph JSON equals the canonical graph for the actual config")
            graph = decoded
        }
        let config = graph.config
        let epsilon: Float = 1e-6
        let ssa = try GemmaSSA.lower(graph)
        let originalDigest = try GemmaSSA.digest(ssa)
        let stats = try RMSStatisticsPlan.analyze(ssa: ssa, epsilon: epsilon)
        let n = ssa.instructions.count
        check(stats.groupCount == 30 && stats.reuseCount == 60,
              "statistics plan has 30 groups / 60 reuses")

        // ---- accepted plan ----
        let plan = try RouterOverlapPlan.analyze(graph: graph, ssa: ssa,
                                                 statisticsPlan: stats, epsilon: epsilon)
        check(plan.analysis == RouterOverlapPlan.analysisVersion, "analysis version 1")
        check(plan.windows.count == 30, "30 FFN windows")
        let stages = ["ffn_norm", "shared_gate", "shared_up", "shared_gelu",
                      "shared_down", "shared_post1", "router_norm", "router_scale",
                      "router_proj", "router_topk", "moe_norm2", "expert_loop"]
        var slots = Set<Int>()
        for L in 0..<min(30, plan.windows.count) {
            guard let i0 = graph.ops.firstIndex(where: { $0.op == "ffn_norm" && $0.layer == L }) else {
                check(false, "graph ffn_norm for layer \(L)")
                continue
            }
            let w = plan.windows[L]
            let idx = Array(i0...(i0 + 11))
            check(w.layer == L && w.canonicalIndices == idx, "layer \(L) window indices")
            check(idx.map { graph.ops[$0].op } == stages && idx.allSatisfy { graph.ops[$0].layer == L },
                  "layer \(L) window stages")
            let order = [idx[0], idx[6], idx[7], idx[8], idx[1], idx[2],
                         idx[3], idx[4], idx[5], idx[9]]
            check(w.executionOrder == order && Array(plan.permutation[i0...(i0 + 9)]) == order,
                  "layer \(L) permutation i0,i6,i7,i8,i1..i5,i9")
            check(plan.permutation[i0 + 10] == idx[10] && plan.permutation[i0 + 11] == idx[11],
                  "layer \(L) i10/i11 stay in place")
            check(w.computeIndex == idx[0] && w.reuseIndices == [idx[6], idx[10]],
                  "layer \(L) compute/reuse indices")
            check(stats.action(at: idx[0])?.kind == .compute
                    && stats.action(at: idx[0])?.slot == w.statisticsSlot
                    && stats.action(at: idx[6])?.kind == .reuse
                    && stats.action(at: idx[6])?.slot == w.statisticsSlot
                    && stats.action(at: idx[10])?.kind == .reuse
                    && stats.action(at: idx[10])?.slot == w.statisticsSlot,
                  "layer \(L) original statistics compute + 2 reuses on one slot")
            slots.insert(w.statisticsSlot)
            check(w.remappedValueIDs == [ssa.instructions[idx[6]].output,
                                         ssa.instructions[idx[7]].output]
                    && w.remappedValueIDs.allSatisfy { $0.hasPrefix("h@") },
                  "layer \(L) remaps exactly the router_norm/router_scale h versions")
            check(w.statisticsInputID == ssa.instructions[idx[0]].inputs[0]
                    && w.statisticsInputID.hasPrefix("x@"),
                  "layer \(L) statistics input is the residual x version")
            check(w.sharedNormOutputID == ssa.instructions[idx[0]].output,
                  "layer \(L) shared norm output id")
        }
        check(slots == Set(0..<30), "30 unique statistics slots")
        check(plan.remappedValueIDs.count == 60 && Set(plan.remappedValueIDs).count == 60,
              "60 distinct remapped ids")
        let moved = Set(plan.windows.flatMap { Array($0.canonicalIndices[0...9]) })
        check(plan.permutation.indices.allSatisfy { moved.contains($0) || plan.permutation[$0] == $0 },
              "permutation is the identity outside the windows")
        check(Set(plan.permutation) == Set(0..<n) && plan.permutation.count == n,
              "permutation is a bijection")
        check(plan.canonicalReplay.statisticComputes == 30
                && plan.canonicalReplay.statisticReuses == 60
                && plan.permutedReplay.statisticComputes == 30
                && plan.permutedReplay.statisticReuses == 60
                && plan.canonicalReplay.logicalReads == plan.permutedReplay.logicalReads,
              "replay reports: 30 computes / 60 reuses in both orders")

        // ---- original schedule unchanged, stable digest ----
        let relowered = try GemmaSSA.lower(graph)
        let redigested = try GemmaSSA.digest(ssa)
        check(relowered == ssa && plan.canonicalSchedule == ssa
                && plan.canonicalScheduleDigest == originalDigest
                && redigested == originalDigest,
              "original canonical schedule and digest unchanged")
        let again = try RouterOverlapPlan.analyze(graph: graph, ssa: ssa,
                                                  statisticsPlan: stats, epsilon: epsilon)
        check(again == plan && again.digest == plan.digest, "plan and digest are deterministic")
        check(plan.digest.count == 64
                && plan.digest.allSatisfy { "0123456789abcdef".contains($0) },
              "digest is 64 lowercase hex")
        let roundTrip = try JSONDecoder().decode(RouterOverlapPlan.self,
                                                 from: JSONEncoder().encode(plan))
        check(roundTrip == plan, "plan Codable round trip")
        expectNoThrow("digest verifies") { try RouterOverlapPlan.verifyDigest(plan) }
        expectNoThrow("reverify against fresh analysis") {
            try RouterOverlapPlan.reverify(plan, ssa: ssa, config: config,
                                           statisticsPlan: stats, epsilon: epsilon)
        }
        let w0 = plan.windows[0]
        let tampered = try mutated(plan) { object in
            guard var p = object["permutation"] as? [Int] else {
                throw TestSetupError.missing("permutation")
            }
            p.swapAt(w0.canonicalIndices[0] + 1, w0.canonicalIndices[0] + 4)
            object["permutation"] = p
        }
        expectRejection("tampered permutation fails the digest", containing: "digest") {
            try RouterOverlapPlan.verifyDigest(tampered)
        }
        expectRejection("tampered permutation fails reverify") {
            try RouterOverlapPlan.reverify(tampered, ssa: ssa, config: config,
                                           statisticsPlan: stats, epsilon: epsilon)
        }

        // ---- physical replay ----
        let identity = Array(0..<n)
        expectNoThrow("canonical replay") {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: identity, remap: [:], statisticsPlan: stats)
        }
        expectNoThrow("permuted replay with routerIn remap") {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: plan.permutation, remap: plan.remap,
                statisticsPlan: stats)
        }
        let control = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: plan.permutation, remap: [:],
                statisticsPlan: stats)
        }
        check(control == w0.canonicalIndices[1] && plan.controlRejectedIndex == control,
              "control replay without remap rejects at the first shared gate "
                + "(got \(String(describing: control)))")
        let partial = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: plan.permutation,
                remap: [w0.remappedValueIDs[0]: .routerIn], statisticsPlan: stats)
        }
        check(partial == w0.canonicalIndices[7],
              "partial remap rejects at the in-place router scale (got \(String(describing: partial)))")
        var broad = plan.remap
        broad[w0.sharedNormOutputID] = .routerIn
        let broadIndex = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: plan.permutation, remap: broad,
                statisticsPlan: stats)
        }
        check(broadIndex == w0.canonicalIndices[1],
              "remapping the shared norm output too rejects at the shared gate "
                + "(got \(String(describing: broadIndex)))")
        let xAlias = Dictionary(uniqueKeysWithValues:
            w0.remappedValueIDs.map { ($0, RouterOverlapResource.buffer(.x)) })
        let xIndex = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: plan.permutation, remap: xAlias,
                statisticsPlan: stats)
        }
        check(xIndex == w0.canonicalIndices[10],
              "aliasing the router input onto x clobbers the residual read by moe_norm2 "
                + "(got \(String(describing: xIndex)))")
        let sharedAlias = Dictionary(uniqueKeysWithValues:
            w0.remappedValueIDs.map { ($0, RouterOverlapResource.buffer(.shared)) })
        let sharedIndex = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: identity, remap: sharedAlias,
                statisticsPlan: stats)
        }
        check(sharedIndex == w0.canonicalIndices[0] + 13,
              "aliasing the router input onto shared clobbers the shared branch before combine "
                + "(got \(String(describing: sharedIndex)))")
        var early = plan.permutation
        early.swapAt(w0.canonicalIndices[0], w0.canonicalIndices[0] + 1)
        let earlyIndex = replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: ssa, layerCount: 30, order: early, remap: plan.remap,
                statisticsPlan: stats)
        }
        check(earlyIndex == w0.canonicalIndices[6],
              "statistics reuse before its compute rejects (got \(String(describing: earlyIndex)))")

        // ---- SSA mutations: dataflow-valid, rejected by the planner ----
        func rejectsSSA(_ name: String, _ needle: String, _ bad: GemmaSSAIR,
                        config badConfig: GemmaGraphIR.GraphConfig? = nil) {
            do {
                try GemmaSSA.validate(bad, config: badConfig ?? config)
                check(true, "\(name): mutation passes dataflow validation")
            } catch {
                check(false, "\(name): mutation unexpectedly fails dataflow validation: \(error)")
            }
            expectRejection(name, containing: needle) {
                _ = try RouterOverlapPlan.analyze(ssa: bad, config: badConfig ?? config,
                                                  statisticsPlan: stats, epsilon: epsilon)
            }
            expectRejection("\(name) (graph entry)", containing: "canonical lowering") {
                _ = try RouterOverlapPlan.analyze(graph: graph, ssa: bad,
                                                  statisticsPlan: stats, epsilon: epsilon)
            }
        }
        let w3 = plan.windows[3]
        let w29 = plan.windows[29]
        rejectsSSA("wrong shared gate tensor", "tensor binding", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[1]) { inst in
                inst["tensors"] = ["blk.3.ffn_up.weight"]
            }
        })
        let wrongOffset = try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[2]) { inst in
                setAttr(&inst, "offset", 0)
            }
        }
        rejectsSSA("wrong shared_up GU offset", "attrs/geometry", wrongOffset)
        check(replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: wrongOffset, layerCount: 30, order: identity, remap: [:],
                statisticsPlan: nil)
        } == w3.canonicalIndices[2], "replay rejects the wrong GU write range at shared_up")
        rejectsSSA("wrong ffn_norm geometry attrs", "attrs/geometry", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[0]) { inst in
                setAttr(&inst, "width", 2816)
            }
        })
        rejectsSSA("wrong router scale attrs", "attrs/geometry", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[7]) { inst in
                setAttr(&inst, "layerScale", 1)
            }
        })
        rejectsSSA("wrong topK", "attrs/geometry", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[9]) { inst in
                setAttr(&inst, "topK", 7)
            }
            try editInstruction(&object, w3.canonicalIndices[11]) { inst in
                setAttr(&inst, "experts", 7)
            }
            try editValue(&object, w3.routingStateID) { value in value["length"] = 7 }
        })
        rejectsSSA("escaped remapped router input", "escapes", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[10]) { inst in
                inst["inputs"] = [w3.remappedValueIDs[1]]
            }
        })
        rejectsSSA("extra router score consumer", "router scores", try mutated(ssa) { object in
            guard var list = object["instructions"] as? [[String: Any]],
                  var values = object["values"] as? [[String: Any]] else {
                throw TestSetupError.missing("instructions/values")
            }
            let extra: [String: Any] = [
                "op": "copy", "layer": 29, "inputs": [w29.routerScoresID],
                "output": "router@1000", "tensors": [String](), "attrs": [String: Int](),
            ]
            list.insert(extra, at: w29.canonicalIndices[11] + 1)
            values.append(["name": "router", "version": 1000, "dtype": "f32",
                           "length": 128, "kind": "vector"])
            object["instructions"] = list
            object["values"] = values
        })
        rejectsSSA("missing FFN window", "missing", try mutated(ssa) { object in
            try editInstruction(&object, w3.canonicalIndices[0]) { inst in inst["layer"] = 4 }
        })
        rejectsSSA("wrong global rope pattern", "layer pattern", try mutated(ssa) { object in
            try editInstruction(&object, plan.windows[5].canonicalIndices[0] - 11) { inst in
                setAttr(&inst, "global", 0)
            }
        })
        expectRejection("missing layer (config claims 31 layers)", containing: "missing") {
            _ = try RouterOverlapPlan.analyze(ssa: ssa, config: actualConfig(layers: 31),
                                              statisticsPlan: stats, epsilon: epsilon)
        }
        let unknown = try mutated(ssa) { object in
            try editInstruction(&object, 1) { inst in inst["op"] = "mystery_op" }
        }
        expectRejection("unknown op rejected by the planner", containing: "dataflow") {
            _ = try RouterOverlapPlan.analyze(ssa: unknown, config: config,
                                              statisticsPlan: stats, epsilon: epsilon)
        }
        check(replayRejectionIndex {
            _ = try RouterOverlapPlan.physicalReplay(
                ssa: unknown, layerCount: 30, order: identity, remap: [:], statisticsPlan: nil)
        } == 1, "unknown op rejected by the replay effect model")

        // ---- statistics plan mutations ----
        func rejectsStats(_ name: String, _ bad: RMSStatisticsPlan) {
            expectRejection(name, containing: "statistics") {
                _ = try RouterOverlapPlan.analyze(ssa: ssa, config: config,
                                                  statisticsPlan: bad, epsilon: epsilon)
            }
        }
        rejectsStats("statistics reuse turned into compute", try mutated(stats) { object in
            try editAction(&object, w3.canonicalIndices[6]) { action in action["kind"] = "compute" }
        })
        rejectsStats("statistics slot swapped", try mutated(stats) { object in
            try editAction(&object, w3.canonicalIndices[10]) { action in
                action["slot"] = plan.windows[4].statisticsSlot
            }
        })
        rejectsStats("statistics input version changed", try mutated(stats) { object in
            try editAction(&object, w3.canonicalIndices[10]) { action in action["inputID"] = "x@0" }
        })
        rejectsStats("statistics action outside any window", try mutated(stats) { object in
            try editAction(&object, 1) { action in
                action["kind"] = "reuse"
                action["slot"] = 0
                action["width"] = 2816
                action["vectors"] = 1
                action["inputID"] = "x@0"
            }
        })
        rejectsStats("statistics analysis version changed", try mutated(stats) { object in
            object["analysis"] = 1
        })
        let otherBits = Int(Float(1e-5).bitPattern)
        rejectsStats("statistics epsilon bits changed", try mutated(stats) { object in
            guard var groups = object["groups"] as? [[String: Any]] else {
                throw TestSetupError.missing("groups")
            }
            for g in groups.indices { groups[g]["epsilonBits"] = otherBits }
            object["groups"] = groups
            object["epsilonBits"] = otherBits
        })
        rejectsStats("statistics group lost a reuse", try mutated(stats) { object in
            guard var groups = object["groups"] as? [[String: Any]] else {
                throw TestSetupError.missing("groups")
            }
            groups[w3.statisticsSlot]["useIndices"] = [w3.canonicalIndices[6]]
            object["groups"] = groups
        })
        expectRejection("planner epsilon differs from the statistics epsilon",
                        containing: "statistics") {
            _ = try RouterOverlapPlan.analyze(ssa: ssa, config: config,
                                              statisticsPlan: stats, epsilon: 1e-5)
        }
    }
}

let harness = Harness()
let arguments = ProcessInfo.processInfo.arguments
var graphPath: String?
if let flag = arguments.firstIndex(of: "--graph") {
    guard flag + 1 < arguments.count else {
        print("--graph requires a PATH")
        exit(2)
    }
    graphPath = arguments[flag + 1]
}
do {
    try harness.run(graphPath: graphPath)
} catch {
    harness.recordUnexpected(error)
}
print("router overlap pure tests: \(harness.passed) passed, \(harness.failed) failed")
exit(harness.failed == 0 ? 0 : 1)
