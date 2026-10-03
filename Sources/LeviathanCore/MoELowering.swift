// Phase 13 — routed MoE region lowering (ONE region family).
//
// Pipeline: REAL torch.export(strict=True) region capture (Python frontend,
// tools/moe_lower/export_moe_region.py) -> MoERegionIR validation INCLUDING
// independent Swift re-validation of the normalized ATen graph (canonical
// digest + structural re-derivation of every dimension and the slot order —
// never trusting the frontend's declared fields) -> shape+hardware-derived
// batch schedules E in {1,2,4,8} x row-tiles R -> Metal source EMISSION
// (three kernels per schedule: batched GU matvec, batched GELU(tanh)*up,
// batched DN matvec; per-row serial arithmetic EXACTLY the incumbent's) ->
// component bit-identity gates against the ACTUAL incumbent kernels on real
// GGUF expert bytes + independent CPU FP64 oracle -> persisted artifact
// with full identity binding -> teacher-forced full-model gates + matched
// paired timing + 2% policy -> production install/reload proof.
//
// The 8 original ordered axpy merges stay OUTSIDE the region, unchanged.
// Resident-GGUF mode only; the expert-cache/SSD fallback keeps the
// incumbent serial loop. Bounded to the one routed-region dataflow — this
// is not a general arbitrary-Python compiler. See
// docs/MOE_BATCH_LOWERING_DESIGN.md.

import Foundation
import CryptoKit
import Metal

public func moeIsLowercaseHex64(_ s: String) -> Bool {
    s.count == 64 && s.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") }
}

public enum MoELoweringError: Error, CustomStringConvertible, Equatable {
    case invalidRegion(String)
    case invalidSchedule(String)
    case emissionFailed(String)
    case oracleFailed(String)
    case componentGateFailed(String)
    case invalidArtifact(String)
    case hardwareUnavailable(String)

    public var description: String {
        switch self {
        case .invalidRegion(let m): return "moe region invalid: \(m)"
        case .invalidSchedule(let m): return "moe schedule invalid: \(m)"
        case .emissionFailed(let m): return "moe emission failed: \(m)"
        case .oracleFailed(let m): return "moe oracle failed: \(m)"
        case .componentGateFailed(let m): return "moe component gate failed: \(m)"
        case .invalidArtifact(let m): return "moe artifact invalid: \(m)"
        case .hardwareUnavailable(let m): return "moe hardware unavailable: \(m)"
        }
    }
}

// MARK: - Normalized graph values (mirror of the Python frontend's JSON)

/// One operand in the normalized ATen graph. Node references are the
/// {"__node__": name} marker; everything else is a literal the frontend
/// already restricted (no floats, no nested dicts beyond node refs).
public enum MoEGVal: Equatable, Sendable {
    case nodeRef(String)
    case int(Int)
    case str(String)
    case bool(Bool)
    case none
    case list([MoEGVal])

    /// Canonical string — MUST match export_moe_region.py canon() exactly.
    public func canon() -> String {
        switch self {
        case .nodeRef(let n): return "@" + n
        case .int(let v): return "i:\(v)"
        case .str(let v): return "s:" + v
        case .bool(let v): return v ? "b:1" : "b:0"
        case .none: return "n"
        case .list(let v):
            return "l<" + v.map { $0.canon() }.joined(separator: ",") + ">"
        }
    }
}

extension MoEGVal: Codable {
    private enum CodingError: Error { case badOperand(String) }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .none; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .int(i); return }
        if let s = try? c.decode(String.self) { self = .str(s); return }
        if let dict = try? c.decode([String: String].self),
           dict.count == 1, let name = dict["__node__"] {
            self = .nodeRef(name); return
        }
        if let arr = try? c.decode([MoEGVal].self) { self = .list(arr); return }
        throw CodingError.badOperand("unsupported operand kind")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .nodeRef(let n): try c.encode(["__node__": n])
        case .int(let v): try c.encode(v)
        case .str(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .none: try c.encodeNil()
        case .list(let v): try c.encode(v)
        }
    }
}

/// Normalized node metadata: scalar tensors carry dtype+shape; sequence
/// outputs (split's tuple) carry item dtypes/shapes derived by the
/// frontend from the exported FakeTensors.
public struct MoEGraphMeta: Codable, Equatable, Sendable {
    public let dtype: String
    public let shape: [Int]?
    public let items: [MoEGraphMeta]?

    /// Canonical string — sorted RAW keys (no prefix) with canonical
    /// values, exactly the Python canon() dict form: keys dtype, items,
    /// shape in sorted order; a sequence meta always carries shape as
    /// "n" and its full item metas.
    public func canon() -> String {
        var parts = ["dtype:s:" + dtype]
        if let items {
            parts.append("items:l<" +
                items.map { $0.canon() }.joined(separator: ",") + ">")
        }
        parts.append("shape:" + (shape.map {
            "l<" + $0.map { "i:\($0)" }.joined(separator: ",") + ">"
        } ?? "n"))
        return "d<" + parts.joined(separator: ",") + ">"
    }
}

public struct MoEGraphNode: Codable, Equatable, Sendable {
    public let name: String
    public let op: String
    public let target: String
    public let args: [MoEGVal]
    public let kwargs: [String: MoEGVal]
    public let meta: MoEGraphMeta?

    func canonLine() -> String {
        // Python canon() emits dict entries as RAW key + ":" + value.
        let kwargsCanon = "d<" + kwargs.keys.sorted().map {
            $0 + ":" + kwargs[$0]!.canon()
        }.joined(separator: ",") + ">"
        let argsCanon = "l<" + args.map { $0.canon() }.joined(separator: ",") + ">"
        let metaCanon = meta?.canon() ?? "n"
        return [name, op, target, argsCanon, kwargsCanon, metaCanon]
            .joined(separator: "|")
    }
}

public struct MoENormalizedGraph: Codable, Equatable, Sendable {
    public let nodes: [MoEGraphNode]
    public let inputs: [String]
    public let outputs: [String]

    public func canonicalPayload() -> String {
        var lines = nodes.map { $0.canonLine() }
        lines.append("inputs|l<" + inputs.map { "s:\($0)" }.joined(separator: ",") + ">")
        lines.append("outputs|l<" + outputs.map { "s:\($0)" }.joined(separator: ",") + ">")
        return lines.joined(separator: "\n")
    }

    public func digest() -> String {
        SHA256.hex(SHA256.hash(Array(canonicalPayload().utf8)))
    }
}

// MARK: - Region IR

/// The validated routed MoE region: the torch-exported, Swift-revalidated
/// top-K slot dataflow (no-bias linear / split / GELU(tanh) / mul / stack)
/// bound to the two 3-D Q4_0 expert families of the actual GGUF.
public struct MoERegionIR: Codable, Equatable, Sendable {
    public let schema: Int
    public let region: String
    public let hidden: Int
    public let expertIntermediate: Int
    public let topK: Int
    public let gateUpOut: Int
    public let geluApprox: String
    public let stackOutputs: Bool
    public let slotOps: [String]
    public let guTensor: String
    public let dnTensor: String
    public let graph: MoENormalizedGraph
    public let graphSha256: String

    struct TorchAdvisory: Codable, Equatable, Sendable {
        let advisoryOnly: Bool?
        let strict: Bool?
        let version: String?
        let atenOps: [String]?
    }
    let torch: TorchAdvisory?

    public static let supportedSchema = 1
    public static let regionName = "gemma4.moe.routed-region"
    public static let maxGraphNodes = 512
    public static let maxDim = 8192
    public static let maxTopK = 64

    public static func parse(_ data: Data) throws -> MoERegionIR {
        guard data.count <= 1 << 19 else {
            throw MoELoweringError.invalidRegion("region JSON exceeds 512 KiB")
        }
        let region: MoERegionIR
        do { region = try JSONDecoder().decode(MoERegionIR.self, from: data) }
        catch { throw MoELoweringError.invalidRegion("region JSON decode: \(error)") }
        try region.validate()
        return region
    }

    /// Independent Swift re-validation of the NORMALIZED ATen GRAPH: the
    /// canonical digest is recomputed, the supported dataflow is re-derived
    /// node-by-node (placeholders, per-slot chains, slot order, census),
    /// and every declared semantic field must equal what the graph itself
    /// says. A frontend that lies (or an attacker recomputing the digest)
    /// still fails closed here.
    public func validate() throws {
        guard schema == Self.supportedSchema else {
            throw MoELoweringError.invalidRegion("schema \(schema) != \(Self.supportedSchema)")
        }
        guard region == Self.regionName else {
            throw MoELoweringError.invalidRegion("unsupported region \(region)")
        }
        guard geluApprox == "tanh", stackOutputs else {
            throw MoELoweringError.invalidRegion(
                "gelu must be tanh and outputs stacked, got \(geluApprox)/\(stackOutputs)")
        }
        guard slotOps == ["q4_0.expert-slice-linear-gate-up",
                          "gelu-tanh-split-mul",
                          "q4_0.expert-slice-linear-down"] else {
            throw MoELoweringError.invalidRegion("slotOps mismatch: \(slotOps)")
        }
        guard graphSha256 == graph.digest() else {
            throw MoELoweringError.invalidRegion(
                "normalized graph digest mismatch (graph tampered or stale frontend)")
        }
        guard graph.nodes.count <= Self.maxGraphNodes,
              moeIsLowercaseHex64(graphSha256) else {
            throw MoELoweringError.invalidRegion("graph bounds/digest form invalid")
        }
        guard graph.inputs.count == 3, graph.outputs.count == 1 else {
            throw MoELoweringError.invalidRegion("graph must have 3 inputs and 1 output")
        }

        // Plain-identifier node names keep the canonical serialization
        // collision-free (mirrors the frontend check).
        let nameOk = graph.nodes.allSatisfy { !$0.name.isEmpty &&
            $0.name.allSatisfy { ($0.isLetter && $0.isASCII) || $0 == "_" ||
                ($0.isNumber && $0.isASCII) || $0 == "." } }
        guard nameOk else {
            throw MoELoweringError.invalidRegion("node name not a plain identifier")
        }
        let names = Set(graph.nodes.map(\.name))
        guard names.count == graph.nodes.count else {
            throw MoELoweringError.invalidRegion("duplicate node names")
        }

        // Re-derive the region from the graph itself.
        let dims = try MoERegionGraphWalk.walk(graph: graph)
        guard dims.hidden == hidden, dims.inter == expertIntermediate,
              dims.topK == topK, dims.gateUpOut == gateUpOut else {
            throw MoELoweringError.invalidRegion(
                "declared dims (\(hidden), \(expertIntermediate), \(topK), \(gateUpOut)) "
                    + "disagree with the graph-derived \(dims)")
        }
        for (label, v) in [("hidden", hidden), ("inter", expertIntermediate),
                           ("gateUpOut", gateUpOut)] {
            guard v > 0, v <= Self.maxDim, v % Q4_0.valuesPerBlock == 0 else {
                throw MoELoweringError.invalidRegion("\(label)=\(v) out of bounds")
            }
        }
        guard topK > 0, topK <= Self.maxTopK else {
            throw MoELoweringError.invalidRegion("topK \(topK) out of bounds")
        }
        guard 2 * expertIntermediate == gateUpOut else {
            throw MoELoweringError.invalidRegion("gateUpOut != 2*expertIntermediate")
        }
    }

    // Model binding (all-layer spans, exact metadata dims, checked
    // scratch, ordered route offsets) lives in MoEModelBinding.swift
    // (separately authored, same module): MoEModelBinding.bind(region:...).

    /// SHA-256 over the canonical (sorted-keys) JSON encoding.
    public func digest() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(self)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }
}

/// Structural re-derivation of the supported region from the normalized
/// graph — the Swift-side authority (the Python walk is only an early gate).
enum MoERegionGraphWalk {
    struct Dims: Equatable { let hidden, inter, topK, gateUpOut: Int }

    static let select = "aten.select.int"
    static let linear = "aten.linear.default"
    static let split = "aten.split.Tensor"
    static let gelu = "aten.gelu.default"
    static let mul = "aten.mul.Tensor"
    static let stack = "aten.stack.default"
    static let getitem = "operator.getitem"

    static func walk(graph: MoENormalizedGraph) throws -> Dims {
        let byName = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.name, $0) })
        func node(_ ref: MoEGVal, _ what: String) throws -> MoEGraphNode {
            guard case let .nodeRef(name) = ref, let n = byName[name] else {
                throw MoELoweringError.invalidRegion("\(what): expected a node reference, got \(ref)")
            }
            return n
        }
        func intConst(_ v: MoEGVal, _ what: String) throws -> Int {
            guard case let .int(i) = v, i >= 0, i <= MoERegionIR.maxDim * 4 else {
                throw MoELoweringError.invalidRegion("\(what): constant must be a bounded int, got \(v)")
            }
            return i
        }
        func meta(_ n: MoEGraphNode, _ what: String, rank: Int? = nil) throws -> (dtype: String, shape: [Int]) {
            guard let m = n.meta, let shape = m.shape, m.dtype != "seq" else {
                throw MoELoweringError.invalidRegion("\(what) (\(n.name)): missing static scalar meta")
            }
            // Bounded BEFORE any arithmetic elsewhere uses the dims.
            guard shape.allSatisfy({ $0 >= 0 && $0 <= 4 * MoERegionIR.maxDim }) else {
                throw MoELoweringError.invalidRegion("\(what) (\(n.name)): dims \(shape) out of bounds")
            }
            if let rank { guard shape.count == rank else {
                throw MoELoweringError.invalidRegion("\(what) (\(n.name)): rank \(shape.count) != \(rank)")
            } }
            return (m.dtype, shape)
        }
        for n in graph.nodes {
            guard ["placeholder", "call_function", "output"].contains(n.op) else {
                throw MoELoweringError.invalidRegion(
                    "unsupported node kind \(n.op) (\(n.name))")
            }
            if n.op != "call_function", !n.kwargs.isEmpty {
                throw MoELoweringError.invalidRegion(
                    "\(n.op) node \(n.name) must carry no kwargs")
            }
        }

        let placeholders = graph.nodes.filter { $0.op == "placeholder" }
        guard placeholders.count == 3,
              placeholders.map(\.name) == graph.inputs else {
            throw MoELoweringError.invalidRegion("graph inputs must be the 3 placeholders")
        }
        let xN = placeholders[0], guN = placeholders[1], dnN = placeholders[2]
        let xd = try meta(xN, "input x", rank: 2)
        let gd = try meta(guN, "input gu_weights", rank: 3)
        let dd = try meta(dnN, "input down_weights", rank: 3)
        for (what, m) in [("x", xd), ("gu_weights", gd), ("down_weights", dd)] {
            guard m.dtype == "float32" else {
                throw MoELoweringError.invalidRegion("\(what) dtype \(m.dtype) != float32")
            }
        }
        guard xd.shape[0] == 1 else {
            throw MoELoweringError.invalidRegion("x batch dim must be 1")
        }
        let hidden = xd.shape[1]
        let topK = gd.shape[0], gateUpOut = gd.shape[1], guIn = gd.shape[2]
        let dnOut = dd.shape[1], dnIn = dd.shape[2]
        guard guIn == hidden, dd.shape[0] == topK, dnOut == hidden else {
            throw MoELoweringError.invalidRegion("input dims inconsistent with region")
        }

        let outputNodes = graph.nodes.filter { $0.op == "output" }
        guard outputNodes.count == 1, outputNodes[0].name == graph.outputs[0],
              outputNodes[0].args.count == 1,
              case let .list(packed) = outputNodes[0].args[0], packed.count == 1 else {
            throw MoELoweringError.invalidRegion("output must pack exactly one value")
        }
        let stackN = try node(packed[0], "output")
        guard stackN.target == stack else {
            throw MoELoweringError.invalidRegion(
                "output must be \(stack), got \(stackN.target) (\(stackN.name))")
        }
        // stack(TensorList, dim=0): dim omitted (default), explicit arg, or kwarg.
        var slotRefs: [MoEGVal]
        switch (stackN.args.count, stackN.kwargs.count) {
        case (1, 0):
            guard case let .list(l) = stackN.args[0] else {
                throw MoELoweringError.invalidRegion("stack inputs must be a list")
            }
            slotRefs = l
        case (2, 0):
            guard case let .list(l) = stackN.args[0],
                  try intConst(stackN.args[1], "stack dim") == 0 else {
                throw MoELoweringError.invalidRegion("stack args malformed")
            }
            slotRefs = l
        case (1, 1):
            guard case let .list(l) = stackN.args[0],
                  stackN.kwargs.keys.sorted() == ["dim"],
                  try intConst(stackN.kwargs["dim"]!, "stack dim") == 0 else {
                throw MoELoweringError.invalidRegion("stack kwargs malformed")
            }
            slotRefs = l
        default:
            throw MoELoweringError.invalidRegion("stack arity malformed")
        }
        guard slotRefs.count == topK else {
            throw MoELoweringError.invalidRegion("stack has \(slotRefs.count) inputs != topK \(topK)")
        }
        let sm = try meta(stackN, "stack output", rank: 3)
        guard sm.dtype == "float32", sm.shape == [topK, 1, hidden] else {
            throw MoELoweringError.invalidRegion("stack output meta \(sm) malformed")
        }

        var inter: Int? = nil
        // USED nodes: placeholders + output + stack + every per-slot
        // pattern node INSERTED as it is validated (illegal reuse across
        // slots is rejected — a node may be consumed by exactly one slot).
        var used = Set([xN.name, guN.name, dnN.name, stackN.name,
                        outputNodes[0].name])
        func markUsed(_ n: MoEGraphNode, _ what: String) throws {
            guard used.insert(n.name).inserted else {
                throw MoELoweringError.invalidRegion(
                    "\(what) (\(n.name)) is reused or shared across slots")
            }
        }
        for (slot, ref) in slotRefs.enumerated() {
            let final = try node(ref, "slot \(slot) final")
            guard final.target == linear else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): final op must be \(linear), got \(final.target)")
            }
            var fargs = final.args
            if fargs.count == 3, case .none = fargs[2] { fargs = Array(fargs.prefix(2)) }
            guard fargs.count == 2, final.kwargs.isEmpty else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): down linear must be (input, weight) without bias")
            }
            let actN = try node(fargs[0], "slot \(slot) activation")
            let dnSel = try node(fargs[1], "slot \(slot) down select")
            guard dnSel.target == select, dnSel.args.count == 3, dnSel.kwargs.isEmpty,
                  dnSel.args[0] == .nodeRef(dnN.name),
                  try intConst(dnSel.args[1], "slot \(slot) down dim") == 0,
                  try intConst(dnSel.args[2], "slot \(slot) down idx") == slot else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): down weight select must be \(dnN.name)[0, \(slot)]")
            }
            let yd = try meta(final, "slot \(slot) down output", rank: 2)
            guard yd.dtype == "float32", yd.shape == [1, hidden] else {
                throw MoELoweringError.invalidRegion("slot \(slot): down output \(yd) malformed")
            }
            let dsd = try meta(dnSel, "slot \(slot) down weight", rank: 2)
            guard dsd.dtype == "float32", dsd.shape == [hidden, dnIn] else {
                throw MoELoweringError.invalidRegion("slot \(slot): down slice \(dsd) malformed")
            }

            guard actN.target == mul, actN.args.count == 2, actN.kwargs.isEmpty else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): activation must be \(mul), got \(actN.target)")
            }
            let geluN = try node(actN.args[0], "slot \(slot) gelu")
            let upN = try node(actN.args[1], "slot \(slot) up")
            guard geluN.target == gelu, geluN.args.count == 1,
                  geluN.kwargs["approximate"] == .str("tanh"),
                  geluN.kwargs.count == 1 else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): gelu must be approximate='tanh' only, got \(geluN.kwargs)")
            }
            let gateN = try node(geluN.args[0], "slot \(slot) gate")
            guard gateN.target == getitem, gateN.args.count == 2, gateN.kwargs.isEmpty,
                  upN.target == getitem, upN.args.count == 2, upN.kwargs.isEmpty,
                  try intConst(gateN.args[1], "slot \(slot) gate idx") == 0,
                  try intConst(upN.args[1], "slot \(slot) up idx") == 1,
                  gateN.args[0] == upN.args[0] else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): gate/up must be getitems 0/1 of the SAME split")
            }
            let splitN = try node(gateN.args[0], "slot \(slot) split")
            guard splitN.target == split, splitN.args.count == 3,
                  splitN.kwargs.isEmpty,
                  case let .int(splitDim) = splitN.args[2], splitDim == -1 else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): split must be (tensor, size, -1) with no kwargs")
            }
            let splitSize = try intConst(splitN.args[1], "slot \(slot) split size")
            if let i = inter { guard i == splitSize else {
                throw MoELoweringError.invalidRegion("slot \(slot): split size \(splitSize) != \(i)")
            } } else { inter = splitSize }
            guard splitSize > 0 else {
                throw MoELoweringError.invalidRegion("slot \(slot): split size must be positive")
            }
            // split sequence item metadata: exactly 2 static f32 [1, inter].
            guard let smeta = splitN.meta, smeta.dtype == "seq",
                  let items = smeta.items, items.count == 2,
                  items.allSatisfy({ $0.dtype == "float32" && $0.shape == [1, splitSize] }) else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): split item metas must be 2x f32 [1, \(splitSize)]")
            }
            for (what, n) in [("gate", gateN), ("up", upN)] {
                let md = try meta(n, "slot \(slot) \(what)", rank: 2)
                guard md.dtype == "float32", md.shape == [1, splitSize] else {
                    throw MoELoweringError.invalidRegion("slot \(slot): \(what) meta \(md) malformed")
                }
            }
            let amd = try meta(actN, "slot \(slot) activation", rank: 2)
            guard amd.dtype == "float32", amd.shape == [1, splitSize] else {
                throw MoELoweringError.invalidRegion("slot \(slot): activation meta \(amd) malformed")
            }

            let linN = try node(splitN.args[0], "slot \(slot) GU linear")
            var largs = linN.args
            if largs.count == 3, case .none = largs[2] { largs = Array(largs.prefix(2)) }
            guard linN.target == linear, largs.count == 2, linN.kwargs.isEmpty,
                  largs[0] == .nodeRef(xN.name) else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): GU linear must take the SHARED x placeholder, no bias")
            }
            let guSel = try node(largs[1], "slot \(slot) GU select")
            guard guSel.target == select, guSel.args.count == 3, guSel.kwargs.isEmpty,
                  guSel.args[0] == .nodeRef(guN.name),
                  try intConst(guSel.args[1], "slot \(slot) GU dim") == 0,
                  try intConst(guSel.args[2], "slot \(slot) GU idx") == slot else {
                throw MoELoweringError.invalidRegion(
                    "slot \(slot): GU weight select must be \(guN.name)[0, \(slot)]")
            }
            let lud = try meta(linN, "slot \(slot) GU output", rank: 2)
            guard lud.dtype == "float32", lud.shape == [1, gateUpOut] else {
                throw MoELoweringError.invalidRegion("slot \(slot): GU output \(lud) malformed")
            }
            let gsd = try meta(guSel, "slot \(slot) GU weight", rank: 2)
            guard gsd.dtype == "float32", gsd.shape == [gateUpOut, hidden] else {
                throw MoELoweringError.invalidRegion("slot \(slot): GU slice \(gsd) malformed")
            }
            try markUsed(final, "slot \(slot) down linear")
            try markUsed(dnSel, "slot \(slot) down select")
            try markUsed(actN, "slot \(slot) activation")
            try markUsed(geluN, "slot \(slot) gelu")
            try markUsed(gateN, "slot \(slot) gate")
            try markUsed(upN, "slot \(slot) up")
            try markUsed(splitN, "slot \(slot) split")
            try markUsed(linN, "slot \(slot) GU linear")
            try markUsed(guSel, "slot \(slot) GU select")
        }

        // Exact census: every call_function node is one of the pattern's,
        // with the exact expected counts — no strays, no missing.
        var census: [String: Int] = [:]
        for n in graph.nodes where n.op == "call_function" {
            census[n.target, default: 0] += 1
            guard used.contains(n.name) else {
                throw MoELoweringError.invalidRegion(
                    "unconsumed/extra graph node \(n.target) (\(n.name))")
            }
        }
        let expect: [String: Int] = [select: 2 * topK, linear: 2 * topK,
                                     split: topK, gelu: topK, mul: topK,
                                     stack: 1, getitem: 2 * topK]
        guard census == expect else {
            throw MoELoweringError.invalidRegion("ATen census \(census) != \(expect)")
        }

        guard let interFinal = inter else {
            throw MoELoweringError.invalidRegion("no slots")
        }
        guard 2 * interFinal == gateUpOut, dnIn == interFinal else {
            throw MoELoweringError.invalidRegion("inter \(interFinal) inconsistent with GGUF families")
        }
        return Dims(hidden: hidden, inter: interFinal, topK: topK, gateUpOut: gateUpOut)
    }
}

// MARK: - Schedule

/// One batched lowering schedule for the routed region. Every field has a
/// REAL effect on the emitted Metal source: the batch E (experts per
/// chunk: GU/ACT/DN kernels each compute E experts' worth of rows per
/// dispatch) and the row tile R (consecutive rows per threadgroup —
/// E*R*rowsPerExpert mapping changes the actual grid/resource geometry).
public struct MoESchedule: Codable, Equatable, Hashable, Sendable {
    public let batch: Int
    public let rowsPerGroup: Int

    public init(batch: Int, rowsPerGroup: Int) {
        self.batch = batch
        self.rowsPerGroup = rowsPerGroup
    }

    public var guKernelName: String { "moegen_gu_E\(batch)_R\(rowsPerGroup)" }
    public var actKernelName: String { "moegen_act_E\(batch)" }
    public var dnKernelName: String { "moegen_dn_E\(batch)_R\(rowsPerGroup)" }

    /// Legal iff: E in {1,2,4,8} (INDEPENDENT of topK divisibility —
    /// ceil-chunk schedules carry dynamic expert TAILS through the runtime
    /// p.z chunkSlots and host-checked bounds; the actual Gemma topK=8
    /// divides every candidate E so nothing changes there), R a power of
    /// two bounded by the measured threadgroup maximum.
    public func isLegal(region: MoERegionIR,
                        hardware: ForgeHardwareProfile) -> Bool {
        let e = batch, r = rowsPerGroup
        guard [1, 2, 4, 8].contains(e), region.topK > 0,
              r > 0, r & (r - 1) == 0, r <= 1024,
              r <= hardware.maxTotalThreadsPerThreadgroup else { return false }
        return true
    }
}

public enum MoEScheduleDeriver {
    public static let maxCandidates = 24

    /// E in {1,2,4,8} (E=1 is the sanity anchor, excluded from finalist
    /// eligibility) x row-tiles R derived from the measured execution
    /// width ladder (w, 2w, 4w, 8w + the incumbent cap-width control),
    /// subject to the GLOBAL scratch budget <= 160 KiB and the device's
    /// actual buffer limits. Legality only — no performance claim.
    public static func derive(region: MoERegionIR,
                              hardware: ForgeHardwareProfile) throws -> [MoESchedule] {
        let scratchBytes = try MoEScratch.budgetBytes(region: region)  // global, not per-E
        // Non-positive measured caps are INVALID inputs, not a width-1
        // fallback — a tampered/absent profile must fail closed, and the
        // w*2/4/8 ladder products are overflow-checked. The working set
        // must actually hold the scratch and the width must respect the
        // threadgroup cap. (maxBufferLength is NOT part of this profile;
        // the immutable binding and kernel compile check the real device.)
        guard hardware.threadExecutionWidth > 0,
              hardware.maxTotalThreadsPerThreadgroup > 0,
              hardware.threadExecutionWidth
                  <= hardware.maxTotalThreadsPerThreadgroup,
              hardware.recommendedMaxWorkingSetSize >= scratchBytes else {
            throw MoELoweringError.invalidSchedule(
                "hardware caps invalid: width \(hardware.threadExecutionWidth), "
                + "maxThreads \(hardware.maxTotalThreadsPerThreadgroup), "
                + "working set \(hardware.recommendedMaxWorkingSetSize)")
        }
        let w = hardware.threadExecutionWidth
        var rowTiles = [Int]()
        for m in [1, 2, 4, 8] {
            let (p, o) = w.multipliedReportingOverflow(by: m)
            if !o { rowTiles.append(p) }
        }
        rowTiles.append(min(1024, hardware.maxTotalThreadsPerThreadgroup))
        rowTiles = Array(rowTiles.filter { $0 > 0 }.reduce(into: Set<Int>()) {
            $0.insert($1) }.sorted())
        var result = [MoESchedule]()
        for e in [1, 2, 4, 8] {
            for r in rowTiles {
                let s = MoESchedule(batch: e, rowsPerGroup: r)
                if s.isLegal(region: region, hardware: hardware),
                   !result.contains(s) {
                    result.append(s)
                }
            }
        }
        return Array(result.prefix(maxCandidates))
    }
}

/// Engine-owned batched scratch (three separate allocations, sized ONCE
/// for the full topK — the runtime chunks E reuses slot-major layout).
public struct MoEScratch {
    public static let budget = 160 * 1024

    /// GPU scratch ONLY (the three MTLBuffers: gu/act/dn payloads). The
    /// expert-id/offset values ride host-side setBytes snapshots and are
    /// NOT GPU scratch. Checked arithmetic is the SAME shared helper the
    /// immutable binding uses — overflow on tampered dims throws, never
    /// wraps into a false "fits" answer.
    public static func budgetBytes(region: MoERegionIR) throws -> Int {
        // Positive dims first: the shared mul helper permits zero, and a
        // zero-dim region would produce a zero-byte "fits" answer.
        guard region.topK > 0, region.gateUpOut > 0,
              region.expertIntermediate > 0, region.hidden > 0 else {
            throw MoELoweringError.invalidSchedule(
                "scratch dims non-positive (topK \(region.topK), gu "
                + "\(region.gateUpOut), inter \(region.expertIntermediate), "
                + "hidden \(region.hidden))")
        }
        let s = try MoEModelBinding.scratchBytes(
            topK: region.topK, gateUpOut: region.gateUpOut,
            inter: region.expertIntermediate, hidden: region.hidden)
        guard s.total <= budget else {
            throw MoELoweringError.invalidSchedule(
                "scratch \(s.total) bytes exceeds the \(budget) byte budget")
        }
        return s.total
    }
}

// MARK: - Metal source emission

public enum MoEKernelEmitter {
    /// Bumped whenever emission rules change; part of artifact identity.
    public static let codegenVersion = "moegen-v1"

    static let helperBegin = "// [g4-activation-helpers-begin]"
    static let helperEnd = "// [g4-activation-helpers-end]"

    /// The LITERAL incumbent activation helpers (g4_tanh +
    /// g4_gelu_up_value) extracted verbatim from the runner's own shader
    /// source — the batched kernels compile the SAME source text, never a
    /// re-typed copy. Fail-closed if the markers move.
    public static func literalActivationHelpers() throws -> String {
        let src = Gemma4Runner.shaderSource
        guard let b = src.range(of: helperBegin),
              let e = src.range(of: helperEnd), e.lowerBound > b.upperBound else {
            throw MoELoweringError.emissionFailed(
                "activation helper markers missing from incumbent shader source")
        }
        return String(src[b.lowerBound..<e.upperBound])
    }

    /// Emits the three kernels for one schedule. Per-row serial arithmetic
    /// is EXACTLY the incumbent g4_q4_matvec expression in the same term
    /// order, and the ACT kernel calls the LITERAL incumbent
    /// g4_gelu_up_value helper — bit-equality to the ACTUAL incumbent
    /// kernels is MEASURED by the component gate, never assumed. The batch
    /// changes only the grid mapping: constexpr E is the compile-time
    /// maximum chunk; the runtime p.z/p.w (chunkSlots, slotBase) carry the
    /// ACTUAL chunk (<= E, slotBase + chunkSlots <= topK checked
    /// host-side), so dynamic expert tails stay correct.
    public static func emit(region: MoERegionIR, schedule: MoESchedule) throws -> String {
        let e = schedule.batch, r = schedule.rowsPerGroup
        let shapeOnly = ForgeHardwareProfile(
            deviceName: "emit", maxTotalThreadsPerThreadgroup: .max,
            threadExecutionWidth: 1, recommendedMaxWorkingSetSize: 0)
        guard schedule.isLegal(region: region, hardware: shapeOnly) else {
            throw MoELoweringError.emissionFailed("schedule \(schedule) illegal for region")
        }
        let helpers = try literalActivationHelpers()
        let inter = region.expertIntermediate
        let header = """
        #include <metal_stdlib>
        using namespace metal;

        \(helpers)

        // \(schedule.guKernelName)/\(schedule.actKernelName)/\(schedule.dnKernelName)
        // — EMITTED by LeviathanForge MoE region lowering (\(codegenVersion))
        // from the torch-exported, Swift-revalidated region \(region.region)
        // hidden=\(region.hidden) inter=\(inter) topK=\(region.topK),
        // schedule E=\(e) R=\(r). Per-row serial arithmetic is the incumbent
        // g4_q4_matvec term order; the ACT kernel uses the LITERAL incumbent
        // g4_gelu_up_value above. E is the compile-time chunk maximum; the
        // runtime p.z/p.w carry the actual chunk (tails legal, host-checked).
        // The 8 ordered axpy merges stay OUTSIDE this region, unchanged.

        """
        let gu = """
        kernel void \(schedule.guKernelName)(
                constant const uint4& p [[buffer(0)]],   // inDim, rowsPerExpert, chunkSlots, slotBase
                constant const ulong* offs [[buffer(1)]],  // per-ABSOLUTE-slot expert slice byte offsets
                device const uchar* wbase [[buffer(2)]],
                device const float* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                uint lid [[thread_position_in_threadgroup]],
                uint wid [[threadgroup_position_in_grid]]) {
            constexpr uint EMAX = \(e)u;
            constexpr uint R = \(r)u;
            uint chunkSlots = p.z;
            uint row = wid * R + lid;                 // row across the chunk
            uint rowsPerExpert = p.y;
            uint chunkRows = chunkSlots * rowsPerExpert;
            if (row >= chunkRows) { return; }
            uint slot = p.w + row / rowsPerExpert;    // absolute slot index
            uint rowLocal = row % rowsPerExpert;
            uint blocksPerRow = p.x / 32u;
            device const uchar* rowBytes = wbase + offs[slot]
                + (ulong)rowLocal * blocksPerRow * 18u;
            float sum = 0.0f;
            for (uint b = 0; b < blocksPerRow; b++) {
                device const uchar* blk = rowBytes + b * 18u;
                ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
                float d = float(as_type<half>(dbits));
                for (uint l = 0; l < 16u; l++) {
                    uchar q = blk[2u + l];
                    sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];
                    sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];
                }
            }
            y[(ulong)slot * rowsPerExpert + rowLocal] = sum;
        }

        """
        let act = """
        kernel void \(schedule.actKernelName)(
                constant const uint3& p [[buffer(0)]],  // inter, chunkSlots, slotBase
                device const float* gu [[buffer(1)]],
                device float* y [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
            uint i = gid;                              // element across chunk
            if (i >= p.y * p.x) { return; }
            uint slot = p.z + i / p.x;
            uint lane = i % p.x;
            uint gateUpOut = 2u * p.x;
            float g = gu[(ulong)slot * gateUpOut + lane];
            float u = gu[(ulong)slot * gateUpOut + p.x + lane];
            y[(ulong)slot * p.x + lane] = g4_gelu_up_value(g, u);
        }

        """
        let dn = """
        kernel void \(schedule.dnKernelName)(
                constant const uint4& p [[buffer(0)]],   // inDim(inter), rowsPerExpert(hidden), chunkSlots, slotBase
                constant const ulong* offs [[buffer(1)]],
                device const uchar* wbase [[buffer(2)]],
                device const float* x [[buffer(3)]],     // ACT scratch (slot-major)
                device float* y [[buffer(4)]],           // DN scratch (slot-major)
                uint lid [[thread_position_in_threadgroup]],
                uint wid [[threadgroup_position_in_grid]]) {
            constexpr uint EMAX = \(e)u;
            constexpr uint R = \(r)u;
            uint chunkSlots = p.z;
            uint row = wid * R + lid;
            uint rowsPerExpert = p.y;
            uint chunkRows = chunkSlots * rowsPerExpert;
            if (row >= chunkRows) { return; }
            uint slot = p.w + row / rowsPerExpert;
            uint rowLocal = row % rowsPerExpert;
            uint blocksPerRow = p.x / 32u;
            device const float* xv = x + (ulong)slot * p.x;
            device const uchar* rowBytes = wbase + offs[slot]
                + (ulong)rowLocal * blocksPerRow * 18u;
            float sum = 0.0f;
            for (uint b = 0; b < blocksPerRow; b++) {
                device const uchar* blk = rowBytes + b * 18u;
                ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
                float d = float(as_type<half>(dbits));
                for (uint l = 0; l < 16u; l++) {
                    uchar q = blk[2u + l];
                    sum += d * (float(q & 0x0Fu) - 8.0f) * xv[b * 32u + l];
                    sum += d * (float(q >> 4) - 8.0f) * xv[b * 32u + 16u + l];
                }
            }
            y[(ulong)slot * rowsPerExpert + rowLocal] = sum;
        }
        """
        // EMAX documents the compile-time chunk bound; the runtime chunk
        // slots are host-checked <= EMAX (unused-warning-free reference).
        _ = EMAXDoc
        return header + gu + act + dn
    }

    static let EMAXDoc = 0

    /// SHA-256 over the three emitted kernels' complete source (including
    /// the embedded literal helper block — the artifact binds the FULL
    /// compiled source).
    public static func sourcesSha256(_ sources: [String]) -> String {
        SHA256.hex(SHA256.hash(Array(sources.joined(separator: "\n").utf8)))
    }
}

// MARK: - Compiled batched kernels

/// One compiled schedule: three pipelines + the schedule/region they were
/// emitted from. Bindings match the emitted kernels exactly.
public struct MoEEmittedKernels: Sendable {
    public let schedule: MoESchedule
    public let source: String
    public let guPipeline: MTLComputePipelineState
    public let actPipeline: MTLComputePipelineState
    public let dnPipeline: MTLComputePipelineState
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public static func compile(region: MoERegionIR, schedule: MoESchedule,
                               device: MTLDevice) throws -> MoEEmittedKernels {
        let source = try MoEKernelEmitter.emit(region: region, schedule: schedule)
        guard let queue = device.makeCommandQueue() else {
            throw MoELoweringError.hardwareUnavailable("no command queue")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MoELoweringError.emissionFailed("MSL compile: \(error)") }
        func pipe(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name),
                  let p = try? device.makeComputePipelineState(function: fn) else {
                throw MoELoweringError.emissionFailed("pipeline for \(name)")
            }
            return p
        }
        let gu = try pipe(schedule.guKernelName)
        let act = try pipe(schedule.actKernelName)
        let dn = try pipe(schedule.dnKernelName)
        guard schedule.rowsPerGroup <= gu.maxTotalThreadsPerThreadgroup,
              schedule.rowsPerGroup <= dn.maxTotalThreadsPerThreadgroup else {
            throw MoELoweringError.emissionFailed(
                "\(schedule): R exceeds compiled threadgroup maximum")
        }
        return MoEEmittedKernels(schedule: schedule, source: source,
                                 guPipeline: gu, actPipeline: act,
                                 dnPipeline: dn, device: device, queue: queue)
    }

    // MARK: standalone component evaluation (real expert bytes)

    /// ONE shared fail-closed prelude for every public component/timing
    /// entry: validated region, schedule legality on THIS compiled
    /// pipeline + device, positive reps/warmup, exact expected Q4_0
    /// strides, finite x of the region's hidden size, packed-bytes and
    /// GLOBAL scratch budget vs the device's actual caps — all BEFORE any
    /// division-derived use, copy, allocation, or enqueue. The two entry
    /// points cannot diverge because they share this helper.
    func validatedComponentInputs(region: MoERegionIR,
                                  guBlocks: [UInt8], dnBlocks: [UInt8],
                                  x: [Float], warmup: Int, reps: Int) throws
            -> (topK: Int, guStride: Int, dnStride: Int,
                guOffs: [UInt64], dnOffs: [UInt64]) {
        try region.validate()
        let caps = ForgeHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: guPipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: guPipeline.threadExecutionWidth,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
        guard schedule.isLegal(region: region, hardware: caps),
              warmup > 0, reps > 0 else {
            throw MoELoweringError.componentGateFailed(
                "schedule \(schedule) illegal on this device, or warmup/reps "
                    + "\(warmup)/\(reps) not positive")
        }
        let topK = region.topK
        let wantGu = try MoEModelBinding.sliceStride(
            inDim: region.hidden, outDim: region.gateUpOut,
            MoELoweringError.invalidSchedule)
        let wantDn = try MoEModelBinding.sliceStride(
            inDim: region.expertIntermediate, outDim: region.hidden,
            MoELoweringError.invalidSchedule)
        guard topK > 0, guBlocks.count % topK == 0, dnBlocks.count % topK == 0 else {
            throw MoELoweringError.componentGateFailed("component geometry mismatch")
        }
        let guStride = guBlocks.count / topK
        let dnStride = dnBlocks.count / topK
        guard guStride == wantGu, dnStride == wantDn,
              x.count == region.hidden,
              x.allSatisfy({ $0.isFinite }) else {
            throw MoELoweringError.componentGateFailed(
                "component geometry mismatch: guStride \(guStride) (want \(wantGu)), "
                    + "dnStride \(dnStride) (want \(wantDn)), x \(x.count)")
        }
        // Packed component weights must fit one buffer; the GLOBAL engine
        // scratch (GU/ACT/DN) must fit the 160 KiB budget AND the device's
        // actual caps (checked via the same Opus utilities).
        let packed = try MoEModelBinding.add(guBlocks.count, dnBlocks.count,
                                             "packed component bytes",
                                             MoELoweringError.invalidSchedule)
        guard packed <= device.maxBufferLength else {
            throw MoELoweringError.componentGateFailed(
                "packed component bytes \(packed) exceed maxBufferLength")
        }
        let scratch = try MoEModelBinding.scratchBytes(
            topK: topK, gateUpOut: region.gateUpOut,
            inter: region.expertIntermediate, hidden: region.hidden)
        guard scratch.total <= MoEModelBinding.scratchBudgetBytes,
              scratch.total <= caps.recommendedMaxWorkingSetSize,
              scratch.gu <= device.maxBufferLength,
              scratch.act <= device.maxBufferLength,
              scratch.dn <= device.maxBufferLength else {
            throw MoELoweringError.componentGateFailed(
                "scratch \(scratch.total) B exceeds budget/device caps")
        }
        // GU slices first, DN slices second: DN offsets are BASE-SHIFTED
        // past the GU bytes in the packed component buffer.
        var guOffs = [UInt64]()
        var dnOffs = [UInt64]()
        for s in 0..<topK {
            guOffs.append(UInt64(s * guStride))
            dnOffs.append(UInt64(guBlocks.count + s * dnStride))
        }
        return (topK, guStride, dnStride, guOffs, dnOffs)
    }

    /// Runs the batched sequence over the given per-slot expert byte
    /// offsets on private device buffers, returning slot-major GU/ACT/DN
    /// arrays (exactly what the component bit gate compares).
    public func runComponent(region: MoERegionIR,
                             guBlocks: [UInt8], dnBlocks: [UInt8],
                             x: [Float]) throws
            -> (gu: [Float], act: [Float], dn: [Float]) {
        let checked = try validatedComponentInputs(
            region: region, guBlocks: guBlocks, dnBlocks: dnBlocks,
            x: x, warmup: 1, reps: 1)
        let topK = checked.topK
        let guOffs = checked.guOffs
        let dnOffs = checked.dnOffs
        var weights = guBlocks
        weights.append(contentsOf: dnBlocks)
        let e = schedule.batch
        let chunks = (topK + e - 1) / e   // ceil: tails carry fewer slots
        guard let wb = device.makeBuffer(bytes: weights, length: weights.count,
                                         options: .storageModeShared),
              let offsB = device.makeBuffer(bytes: guOffs + dnOffs,
                                            length: 2 * topK * 8,
                                            options: .storageModeShared),
              let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let guB = device.makeBuffer(length: topK * region.gateUpOut * 4,
                                          options: .storageModeShared),
              let actB = device.makeBuffer(length: topK * region.expertIntermediate * 4,
                                           options: .storageModeShared),
              let dnB = device.makeBuffer(length: topK * region.hidden * 4,
                                          options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MoELoweringError.hardwareUnavailable("component buffers")
        }
        let guOffBase = 0
        let dnOffBase = topK * 8   // dn offsets live after gu offsets in offsB
        for c in 0..<chunks {
            let slotBase = c * e
            let count = min(e, topK - slotBase)
            // Host-checked tail bounds BEFORE any encode.
            guard count > 0, count <= e, slotBase + count <= topK else {
                throw MoELoweringError.componentGateFailed("chunk bounds violated")
            }
            var gp = SIMD4<UInt32>(UInt32(region.hidden), UInt32(region.gateUpOut),
                                   UInt32(count), UInt32(slotBase))
            enc.setComputePipelineState(guPipeline)
            enc.setBytes(&gp, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
            enc.setBuffer(offsB, offset: guOffBase, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3)
            enc.setBuffer(guB, offset: 0, index: 4)
            let guRows = count * region.gateUpOut
            let guGroups = (guRows + schedule.rowsPerGroup - 1) / schedule.rowsPerGroup
            enc.dispatchThreadgroups(
                MTLSize(width: guGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: schedule.rowsPerGroup,
                                               height: 1, depth: 1))

            var ap = SIMD3<UInt32>(UInt32(region.expertIntermediate),
                                   UInt32(count), UInt32(slotBase))
            enc.setComputePipelineState(actPipeline)
            enc.setBytes(&ap, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
            enc.setBuffer(guB, offset: 0, index: 1)
            enc.setBuffer(actB, offset: 0, index: 2)
            enc.dispatchThreads(
                MTLSize(width: count * region.expertIntermediate, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(1024, actPipeline.maxTotalThreadsPerThreadgroup),
                                               height: 1, depth: 1))

            var dp = SIMD4<UInt32>(UInt32(region.expertIntermediate),
                                   UInt32(region.hidden),
                                   UInt32(count), UInt32(slotBase))
            enc.setComputePipelineState(dnPipeline)
            enc.setBytes(&dp, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
            enc.setBuffer(offsB, offset: dnOffBase, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(actB, offset: 0, index: 3)
            enc.setBuffer(dnB, offset: 0, index: 4)
            let dnRows = count * region.hidden
            let dnGroups = (dnRows + schedule.rowsPerGroup - 1) / schedule.rowsPerGroup
            enc.dispatchThreadgroups(
                MTLSize(width: dnGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: schedule.rowsPerGroup,
                                               height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw MoELoweringError.componentGateFailed(
                "component status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        func readF(_ b: MTLBuffer, _ count: Int) -> [Float] {
            [Float](UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: Float.self),
                                        count: count))
        }
        return (readF(guB, topK * region.gateUpOut),
                readF(actB, topK * region.expertIntermediate),
                readF(dnB, topK * region.hidden))
    }

    /// Component performance screen: builds the buffers once, then times
    /// whole chunked sequences submit->complete (median of reps).
    public func timedLayerMedianUs(region: MoERegionIR,
                                   guBlocks: [UInt8], dnBlocks: [UInt8],
                                   x: [Float], warmup: Int = 5,
                                   reps: Int = 20) throws -> Double {
        let checked = try validatedComponentInputs(
            region: region, guBlocks: guBlocks, dnBlocks: dnBlocks,
            x: x, warmup: warmup, reps: reps)
        let topK = checked.topK
        let guOffs = checked.guOffs
        let dnOffs = checked.dnOffs
        var weights = guBlocks
        weights.append(contentsOf: dnBlocks)
        let e = schedule.batch
        let chunks = (topK + e - 1) / e   // ceil: tails carry fewer slots
        guard let wb = device.makeBuffer(bytes: weights, length: weights.count,
                                         options: .storageModeShared),
              let offsB = device.makeBuffer(bytes: guOffs + dnOffs,
                                            length: 2 * topK * 8,
                                            options: .storageModeShared),
              let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let guB = device.makeBuffer(length: topK * region.gateUpOut * 4,
                                          options: .storageModeShared),
              let actB = device.makeBuffer(length: topK * region.expertIntermediate * 4,
                                           options: .storageModeShared),
              let dnB = device.makeBuffer(length: topK * region.hidden * 4,
                                          options: .storageModeShared) else {
            throw MoELoweringError.hardwareUnavailable("screen buffers")
        }
        func oneLayer() throws {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                throw MoELoweringError.hardwareUnavailable("screen encoder")
            }
            for c in 0..<chunks {
                let slotBase = c * e
                let count = min(e, topK - slotBase)
                guard count > 0, count <= e, slotBase + count <= topK else {
                    throw MoELoweringError.componentGateFailed("chunk bounds violated")
                }
                var gp = SIMD4<UInt32>(UInt32(region.hidden), UInt32(region.gateUpOut),
                                       UInt32(count), UInt32(slotBase))
                enc.setComputePipelineState(guPipeline)
                enc.setBytes(&gp, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
                enc.setBuffer(offsB, offset: 0, index: 1)
                enc.setBuffer(wb, offset: 0, index: 2)
                enc.setBuffer(xb, offset: 0, index: 3)
                enc.setBuffer(guB, offset: 0, index: 4)
                let guRows = count * region.gateUpOut
                let guGroups = (guRows + schedule.rowsPerGroup - 1) / schedule.rowsPerGroup
                enc.dispatchThreadgroups(
                    MTLSize(width: guGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: schedule.rowsPerGroup,
                                                   height: 1, depth: 1))
                var ap = SIMD3<UInt32>(UInt32(region.expertIntermediate),
                                       UInt32(count), UInt32(slotBase))
                enc.setComputePipelineState(actPipeline)
                enc.setBytes(&ap, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
                enc.setBuffer(guB, offset: 0, index: 1)
                enc.setBuffer(actB, offset: 0, index: 2)
                enc.dispatchThreads(
                    MTLSize(width: count * region.expertIntermediate, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(1024, actPipeline.maxTotalThreadsPerThreadgroup),
                                                   height: 1, depth: 1))
                var dp = SIMD4<UInt32>(UInt32(region.expertIntermediate),
                                       UInt32(region.hidden),
                                       UInt32(count), UInt32(slotBase))
                enc.setComputePipelineState(dnPipeline)
                enc.setBytes(&dp, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
                enc.setBuffer(offsB, offset: topK * 8, index: 1)
                enc.setBuffer(wb, offset: 0, index: 2)
                enc.setBuffer(actB, offset: 0, index: 3)
                enc.setBuffer(dnB, offset: 0, index: 4)
                let dnRows = count * region.hidden
                let dnGroups = (dnRows + schedule.rowsPerGroup - 1) / schedule.rowsPerGroup
                enc.dispatchThreadgroups(
                    MTLSize(width: dnGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: schedule.rowsPerGroup,
                                                   height: 1, depth: 1))
            }
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw MoELoweringError.componentGateFailed(
                    "screen status \(cmd.status.rawValue), \(String(describing: cmd.error))")
            }
        }
        for _ in 0..<warmup { try oneLayer() }
        var samples = [Double]()
        samples.reserveCapacity(reps)
        for _ in 0..<reps {
            let t0 = DispatchTime.now()
            try oneLayer()
            samples.append(Double(DispatchTime.now().uptimeNanoseconds
                                  - t0.uptimeNanoseconds) / 1000.0)
        }
        let s = samples.sorted()
        return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2 : s[s.count/2]
    }
}

// MARK: - Installed lowering handle (runner state)

/// The runner's installed batched lowering: the validated region, its
/// schedule, the compiled kernels, and the immutable GGUF binding that
/// supplies per-route offsets. No mutable shared route-offset buffers —
/// every dispatch passes offsets as setBytes snapshots.
public struct MoEInstalledLowering: Sendable {
    public let region: MoERegionIR
    public let schedule: MoESchedule
    public let kernels: MoEEmittedKernels
    public let binding: MoEModelBinding
    public let artifactDigest: String
}

// MARK: - Component inputs (real GGUF expert slices, bounded)

public enum MoEComponentInputs {
    public static let componentCopyBudgetBytes = 64 * 1024 * 1024

    /// Distinct arbitrary routed-expert ids for gates/screens. Bounds are
    /// checked BEFORE any arithmetic (count <= 64, experts in 1...4096).
    public static func expertIds(count: Int, experts: Int) throws -> [Int] {
        guard count > 0, count <= MoERegionIR.maxTopK,
              experts > 0, experts <= 4096, experts >= count else {
            throw MoELoweringError.invalidRegion(
                "need 1...\(MoERegionIR.maxTopK) ids over 1...4096 experts "
                    + "(got count \(count), experts \(experts))")
        }
        let seeds = [0, 127, 5, 63, 42, 100, 17, 121, 3, 90, 33, 71]
        var result = [Int]()
        var i = 0
        while result.count < count {
            let id = (seeds[i % seeds.count] + (i / seeds.count) * 7) % experts
            if !result.contains(id) { result.append(id) }
            i += 1
            if i > 8 * experts {
                throw MoELoweringError.invalidRegion(
                    "could not derive \(count) distinct expert ids")
            }
        }
        return result
    }

    /// Full per-expert byte slices for one layer's family, pread from the
    /// real GGUF: the tensor's ACTUAL 3-D Q4_0 dims must equal the
    /// requested [inDim, outDim, experts], ids bounded/unique/nonempty,
    /// and the total copy checked against the 64 MiB component budget —
    /// all BEFORE any preallocation or read.
    public static func expertBlocks(file: GGUFFile, tensor: String,
                                    ids: [Int], inDim: Int, outDim: Int) throws -> [UInt8] {
        guard !ids.isEmpty,
              Set(ids).count == ids.count,
              ids.allSatisfy({ $0 >= 0 }) else {
            throw MoELoweringError.invalidRegion("expert ids must be unique and >= 0")
        }
        let info: GGUFTensorInfo
        do { info = try file.tensor(tensor) } catch {
            throw MoELoweringError.invalidRegion("tensor \(tensor) not in GGUF")
        }
        guard info.ggmlType == GGMLTensorType.q4_0.rawValue,
              info.dims.count == 3,
              info.dims[0] == inDim, info.dims[1] == outDim,
              let experts = info.dims.last,
              ids.allSatisfy({ $0 < experts }) else {
            throw MoELoweringError.invalidRegion(
                "\(tensor): dims \(info.dims) type \(info.ggmlType) != "
                    + "Q4_0 [\(inDim), \(outDim), experts]")
        }
        let stride = try MoEModelBinding.sliceStride(
            inDim: inDim, outDim: outDim, MoELoweringError.invalidRegion)
        let total = try MoEModelBinding.mul(stride, ids.count, "component copy",
                                            MoELoweringError.invalidRegion)
        guard total <= componentCopyBudgetBytes else {
            throw MoELoweringError.invalidRegion(
                "component copy \(total) B exceeds \(componentCopyBudgetBytes) B budget")
        }
        let range = try file.tensorRange(info)
        var out = [UInt8]()
        out.reserveCapacity(total)
        guard let url = file.sourceURL else {
            throw MoELoweringError.invalidRegion("GGUF has no source URL")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        for id in ids {
            let start = try MoEModelBinding.add(
                range.lowerBound, MoEModelBinding.mul(
                    id, stride, "expert offset", MoELoweringError.invalidRegion),
                "expert offset", MoELoweringError.invalidRegion)
            let end = try MoEModelBinding.add(start, stride, "expert end",
                                              MoELoweringError.invalidRegion)
            guard start >= range.lowerBound, end <= range.upperBound else {
                throw MoELoweringError.invalidRegion("expert slice beyond tensor extent")
            }
            try handle.seek(toOffset: UInt64(start))
            guard let data = try handle.read(upToCount: stride),
                  data.count == stride else {
                throw MoELoweringError.invalidRegion("expert slice read failed")
            }
            out.append(contentsOf: data)
        }
        return out
    }

    public static func activation(region: MoERegionIR, seed: UInt64 = 71_007) throws -> [Float] {
        try region.validate()
        var rng = SplitMix64(seed: seed)
        return (0..<region.hidden).map { _ in rng.nextUniform() }
    }
}

// MARK: - Component result records

public struct MoEComponentResult: Codable, Equatable, Sendable {
    public let schedule: MoESchedule
    public let accepted: Bool
    /// MEASURED bitPattern equality of the batched GU/ACT/DN outputs vs
    /// the ACTUAL incumbent kernels on the same real expert bytes and
    /// activation. Required (true) for acceptance.
    public let bitIdenticalToIncumbent: Bool?
    public let worstRelToOracle: Double?
    public let note: String
}

// MARK: - Model-level records

public struct MoEModelCandidateResult: Codable, Equatable, Sendable {
    public let schedule: MoESchedule
    public let fixturesPassed: Int
    public let fixturesFailed: Int
    public let worstRel: Double?
    public let gateFailReason: String?
    public var wallMedPerFixture: [Double]?
    public var gpuMedPerFixture: [Double]?
    public var timingReps: [Q4TimingRepetition]?
}

public struct MoEModelGateRecord: Codable, Equatable, Sendable {
    public let gatePolicy: String
    /// The EXACT execution path every arm (reference, gates, timing) ran
    /// on — phase 13 production selections use the compiled Forge SSA
    /// interpreter for BOTH the incumbent and generated arms.
    public let executionPath: String
    /// true = bounded --smoke-test plumbing record (NOT production
    /// quality/performance evidence; the public installer rejects it).
    public let diagnostic: Bool
    public let genTokens: Int
    public let timedTokens: Int
    public let fixtures: [Q4FixtureRecord]
    public let results: [MoEModelCandidateResult]
    public let incumbentWallPerFixture: [Double]
    public let incumbentGpuPerFixture: [Double]
    public let performanceSelected: Bool
    public let policyNote: String
}

/// The production gate contract: strict 3-fixture x 32-step full-logit
/// gates, 3 paired AB/BA repetitions x 31 timed decode steps, fixed 2%
/// policy RECOMPUTED from the paired arms (performanceSelected alone is
/// not evidence). A public production install requires exactly this
/// shape for the artifact's SELECTED schedule; a diagnostic (smoke)
/// record is structurally valid for plumbing only.
public enum MoEGatePolicy {
    public static let version = "strict3x32-3fixture-2pct-v1"
    /// Production selections run on the ACTUAL compiled Forge SSA path.
    public static let executionPathSSA = "gemma-forge-ssa-v1"
    public static let productionFixtures = ["en-chat", "kr-chat", "raw-hello"]
    public static let productionGenTokens = 32
    public static let productionTimedTokens = 31

    /// Shape/numerical validity of the ARTIFACT-SELECTED schedule's
    /// record: policy version + SSA execution path, non-diagnostic, exact
    /// fixture set with 32 gated steps each, that schedule fully
    /// gate-passed (no failure reason, worst finite in [0, 1e-4)), 3
    /// alternating paired repetitions with 31 finite positive wall samples
    /// and finite gpu samples per fixture arm at MATCHED windows (same
    /// startIdx). Says NOTHING about the performance outcome.
    public static func shapeValid(_ gate: MoEModelGateRecord,
                                  schedule: MoESchedule) -> Bool {
        guard gate.gatePolicy == version,
              gate.executionPath == executionPathSSA,
              !gate.diagnostic,
              gate.genTokens == productionGenTokens,
              gate.timedTokens == productionTimedTokens,
              gate.fixtures.map(\.name) == productionFixtures,
              gate.fixtures.allSatisfy({ $0.gatedSteps == productionGenTokens
                  && $0.promptTokenCount > 0 }),
              let sel = gate.results.first(where: { $0.schedule == schedule }),
              sel.fixturesPassed == gate.fixtures.count,
              sel.fixturesFailed == 0,
              sel.gateFailReason == nil,
              let worst = sel.worstRel, worst.isFinite, worst >= 0, worst < 1e-4,
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

    /// The 2% policy RECOMPUTED from the paired arms themselves (nil when
    /// the record's shape is invalid). A recorded performanceSelected that
    /// disagrees with this recompute is dishonest and fails validation.
    public static func recomputedSelection(_ gate: MoEModelGateRecord,
                                           schedule: MoESchedule) -> Bool? {
        guard shapeValid(gate, schedule: schedule),
              let sel = gate.results.first(where: { $0.schedule == schedule }),
              let reps = sel.timingReps else { return nil }
        // EVERY repetition's arm sum must be finite and positive — a
        // single overflowing repetition must not hide behind the median.
        let candSums = reps.map { armSum($0.candidate) }
        let incSums = reps.map { armSum($0.incumbent) }
        guard candSums.allSatisfy({ $0.isFinite && $0 > 0 }),
              incSums.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        let candMed = LoweringSelect.median(candSums)
        let incMed = LoweringSelect.median(incSums)
        return candMed < incMed * (1 - GemmaForge.noiseRelWall)
    }

    /// Production validity: valid shape AND the recorded selection equals
    /// the recomputed policy AND the candidate actually WON. An honest
    /// no-win record (performanceSelected == false == recompute) is
    /// structurally valid for inspection but never installs publicly.
    public static func productionValid(_ gate: MoEModelGateRecord,
                                       schedule: MoESchedule) -> Bool {
        guard let recomputed = recomputedSelection(gate, schedule: schedule),
              gate.performanceSelected == recomputed, recomputed else {
            return false
        }
        return true
    }

    static func armSum(_ arm: [Q4TimingArmFixture]) -> Double {
        arm.map { LoweringSelect.median($0.walls) }.reduce(0, +)
    }
}

// MARK: - Artifact

public struct MoELoweringArtifact: Codable, Equatable, Sendable {
    public static let schema = 1

    public let artifactSchema: Int
    public let codegenVersion: String
    public let regionDigest: String
    public let region: MoERegionIR
    /// The immutable GGUF/model binding at forge time: every layer's
    /// expert tensors, actual device caps, checked strides and scratch.
    public let binding: MoEModelBinding
    public let schedule: MoESchedule
    public let guKernelName: String
    public let actKernelName: String
    public let dnKernelName: String
    public let emittedSource: String
    public let sourceSha256: String
    public let hardware: ForgeHardwareProfile
    public let toolchain: String
    public let ggufDigest: String
    public let executableDigest: String
    public let componentResults: [MoEComponentResult]
    public let incumbentWorstRelToOracle: Double
    public let expertIds: [Int]
    public let modelGate: MoEModelGateRecord?
    public let artifactDigest: String

    static func bodyDigest(_ body: MoELoweringArtifact) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stripped = MoELoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            regionDigest: body.regionDigest, region: body.region,
            binding: body.binding, schedule: body.schedule,
            guKernelName: body.guKernelName, actKernelName: body.actKernelName,
            dnKernelName: body.dnKernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            expertIds: body.expertIds, modelGate: body.modelGate,
            artifactDigest: "")
        let bytes = try encoder.encode(stripped)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: [.atomic])
    }

    public static func load(from url: URL) throws -> MoELoweringArtifact {
        guard let data = try? Data(contentsOf: url) else {
            throw MoELoweringError.invalidArtifact("unreadable artifact \(url.path)")
        }
        guard data.count <= 2 << 20 else {
            throw MoELoweringError.invalidArtifact("artifact exceeds 2 MiB")
        }
        do { return try JSONDecoder().decode(MoELoweringArtifact.self, from: data) }
        catch { throw MoELoweringError.invalidArtifact("decode: \(error)") }
    }

    /// Structural identity validation, fail-closed: region revalidation
    /// (graph digest + structural re-derivation), region digest vs the
    /// CURRENT region, binding/region dimension agreement, schedule
    /// legality against the ACTUAL hardware argument, component-record
    /// consistency (unique bounded schedules, exactly topK unique legal
    /// expert ids, finite incumbent oracle < 1e-4, finite nonnegative
    /// rels), codegen, re-EMITTED source bytes (with the literal
    /// incumbent helper block — the FULL compiled source is bound), source
    /// SHA, hardware, toolchain, GGUF, executable, model-gate consistency
    /// (non-diagnostic records must satisfy the FULL production policy for
    /// the selected schedule; diagnostic records validate structurally but
    /// can never pass the public installer), self digest. Trusted-local
    /// integrity only — NOT an authentication attestation.
    public func validate(region currentRegion: MoERegionIR,
                         hardware: ForgeHardwareProfile,
                         ggufDigest: String,
                         executableDigest: String) throws {
        try currentRegion.validate()
        try region.validate()
        guard artifactSchema == Self.schema else {
            throw MoELoweringError.invalidArtifact("schema \(artifactSchema)")
        }
        guard codegenVersion == MoEKernelEmitter.codegenVersion else {
            throw MoELoweringError.invalidArtifact("codegen \(codegenVersion) stale")
        }
        guard region == currentRegion,
              regionDigest == (try currentRegion.digest()),
              moeIsLowercaseHex64(regionDigest) else {
            throw MoELoweringError.invalidArtifact("region mismatch vs current region")
        }
        guard binding.graphSha256 == region.graphSha256,
              binding.topK == region.topK, binding.hidden == region.hidden,
              binding.inter == region.expertIntermediate,
              binding.gateUpOut == region.gateUpOut else {
            throw MoELoweringError.invalidArtifact("binding disagrees with region")
        }
        guard schedule.isLegal(region: region, hardware: hardware) else {
            throw MoELoweringError.invalidArtifact("schedule illegal on actual hardware")
        }
        guard guKernelName == schedule.guKernelName,
              actKernelName == schedule.actKernelName,
              dnKernelName == schedule.dnKernelName else {
            throw MoELoweringError.invalidArtifact("kernel names inconsistent with schedule")
        }
        let reemitted = try MoEKernelEmitter.emit(region: region, schedule: schedule)
        guard reemitted == emittedSource else {
            throw MoELoweringError.invalidArtifact("emitted source tampered/rule drift")
        }
        guard sourceSha256 == MoEKernelEmitter.sourcesSha256([emittedSource]),
              moeIsLowercaseHex64(sourceSha256) else {
            throw MoELoweringError.invalidArtifact("source SHA mismatch")
        }
        guard hardware.maxTotalThreadsPerThreadgroup > 0,
              hardware.threadExecutionWidth > 0,
              hardware.recommendedMaxWorkingSetSize > 0,
              hardware == self.hardware else {
            throw MoELoweringError.invalidArtifact("hardware mismatch")
        }
        guard toolchain == Forge.toolchainIdentifier else {
            throw MoELoweringError.invalidArtifact("toolchain mismatch")
        }
        guard ggufDigest == self.ggufDigest, moeIsLowercaseHex64(ggufDigest),
              executableDigest == self.executableDigest,
              moeIsLowercaseHex64(executableDigest) else {
            throw MoELoweringError.invalidArtifact("GGUF/executable digest mismatch")
        }
        guard !componentResults.isEmpty,
              componentResults.count <= MoEScheduleDeriver.maxCandidates,
              Set(componentResults.map(\.schedule)).count == componentResults.count,
              expertIds.count == region.topK,
              Set(expertIds).count == region.topK,
              expertIds.allSatisfy({ $0 >= 0 && $0 < binding.expertCount }),
              incumbentWorstRelToOracle.isFinite,
              incumbentWorstRelToOracle >= 0,
              incumbentWorstRelToOracle < 1e-4 else {
            throw MoELoweringError.invalidArtifact("component records inconsistent")
        }
        for r in componentResults {
            if let rel = r.worstRelToOracle {
                guard rel.isFinite, rel >= 0 else {
                    throw MoELoweringError.invalidArtifact("component rel invalid")
                }
            }
        }
        guard let sel = componentResults.first(where: {
                  $0.schedule == schedule && $0.accepted }),
              sel.bitIdenticalToIncumbent == true,
              let rel = sel.worstRelToOracle, rel.isFinite, rel >= 0, rel < 1e-4 else {
            throw MoELoweringError.invalidArtifact(
                "selected schedule not component-accepted (bit-identical, <1e-4)")
        }
        if let mg = modelGate {
            guard mg.gatePolicy == MoEGatePolicy.version,
                  mg.executionPath == MoEGatePolicy.executionPathSSA,
                  mg.genTokens >= 2, mg.timedTokens >= 1,
                  mg.timedTokens < mg.genTokens,
                  !mg.fixtures.isEmpty,
                  mg.fixtures.allSatisfy({
                      $0.gatedSteps >= 2 && $0.gatedSteps <= mg.genTokens
                          && $0.promptTokenCount > 0
                          && q4IsLowercaseHex64($0.historyDigest) }),
                  Set(mg.fixtures.map(\.name)).count == mg.fixtures.count,
                  // Bounded, UNIQUE candidate schedules — duplicates or an
                  // unbounded results array are artifact tampering.
                  !mg.results.isEmpty,
                  mg.results.count <= MoEScheduleDeriver.maxCandidates,
                  Set(mg.results.map(\.schedule)).count == mg.results.count,
                  mg.results.allSatisfy({ r in
                      r.schedule.isLegal(region: region, hardware: hardware)
                          && componentResults.contains { $0.schedule == r.schedule } }),
                  // The artifact's chosen schedule appears EXACTLY once.
                  mg.results.filter({ $0.schedule == schedule }).count == 1,
                  mg.results.contains(where: { $0.schedule == schedule }),
                  let selr = mg.results.first(where: { $0.schedule == schedule }),
                  selr.fixturesPassed == mg.fixtures.count,
                  selr.fixturesFailed == 0,
                  selr.gateFailReason == nil,
                  let worst = selr.worstRel, worst.isFinite, worst >= 0, worst < 1e-4 else {
                throw MoELoweringError.invalidArtifact(
                    "model-gate record inconsistent or schedule not fully gate-passed")
            }
            if !mg.diagnostic {
                guard MoEGatePolicy.shapeValid(mg, schedule: schedule),
                      let recomputed = MoEGatePolicy.recomputedSelection(
                          mg, schedule: schedule),
                      mg.performanceSelected == recomputed else {
                    throw MoELoweringError.invalidArtifact(
                        "non-diagnostic gate record fails the production shape "
                            + "or disagrees with the recomputed 2% policy")
                }
            }
        }
        guard artifactDigest == (try Self.bodyDigest(self)),
              moeIsLowercaseHex64(artifactDigest) else {
            throw MoELoweringError.invalidArtifact("artifact digest tampered")
        }
    }
}

// MARK: - Hardware profile source

public enum MoECurrentHardware {
    /// Measured hardware profile from the compiled incumbent pipeline —
    /// the same source the Q4 lowering uses (execution width, threadgroup
    /// maximum). Buffer caps come from the device via MoEModelBinding.
    public static func profile() throws -> ForgeHardwareProfile {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MoELoweringError.hardwareUnavailable("Metal unavailable")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil) }
        catch { throw MoELoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let fn = library.makeFunction(name: "g4_q4_matvec"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw MoELoweringError.emissionFailed("incumbent pipeline")
        }
        return ForgeHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }
}

// MARK: - Component forge (incumbent reference, oracle, bit gates)

public enum MoELoweringForge {

    /// Shared region binding used by forge/verify/install: the resident
    /// length is the runner's page-rounded mmap length (the runner's
    /// ACTUAL hw.pagesize through the shared checked formula), NOT the
    /// raw file size, so exact fresh-binding equality holds against the
    /// real runner.
    public static func bindRegion(region: MoERegionIR, file: GGUFFile,
                                  device: MTLDevice) throws -> MoEModelBinding {
        guard let mapLen = MoEModelBinding.residentMappingLength(
            fileBytes: file.byteCount,
            pageSize: Gemma4Runner.residentPageSize) else {
            throw MoELoweringError.invalidRegion(
                "file bytes \(file.byteCount) cannot yield a legal mapping length")
        }
        return try MoEModelBinding.bind(
            region: region, file: file,
            residentBufferLength: mapLen,
            maxBufferLength: device.maxBufferLength,
            recommendedWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }

    /// Incumbent reference outputs for the component gates: the ACTUAL
    /// production kernels (g4_q4_matvec / g4_gelu_up compiled from the
    /// runner's own embedded MSL) run per slot over the same real expert
    /// bytes and activation the batched kernels see — including the DN
    /// matvec reading THAT SLOT's activation (checked input offset).
    public static func runIncumbent(device: MTLDevice, queue: MTLCommandQueue,
                             region: MoERegionIR, guBlocks: [UInt8],
                             dnBlocks: [UInt8], x: [Float])
            throws -> (gu: [Float], act: [Float], dn: [Float]) {
        try region.validate()
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil) }
        catch { throw MoELoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let matFn = library.makeFunction(name: "g4_q4_matvec"),
              let matPipe = try? device.makeComputePipelineState(function: matFn),
              let geluFn = library.makeFunction(name: "g4_gelu_up"),
              let geluPipe = try? device.makeComputePipelineState(function: geluFn) else {
            throw MoELoweringError.emissionFailed("incumbent pipelines")
        }
        let topK = region.topK
        let guStride = try MoEModelBinding.sliceStride(
            inDim: region.hidden, outDim: region.gateUpOut,
            MoELoweringError.invalidRegion)
        let dnStride = try MoEModelBinding.sliceStride(
            inDim: region.expertIntermediate, outDim: region.hidden,
            MoELoweringError.invalidRegion)
        guard guStride * topK == guBlocks.count, dnStride * topK == dnBlocks.count,
              x.count == region.hidden,
              x.allSatisfy({ $0.isFinite }) else {
            throw MoELoweringError.componentGateFailed("incumbent geometry/input mismatch")
        }
        guard let wbGu = device.makeBuffer(bytes: guBlocks, length: guBlocks.count,
                                           options: .storageModeShared),
              let wbDn = device.makeBuffer(bytes: dnBlocks, length: dnBlocks.count,
                                           options: .storageModeShared),
              let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let guB = device.makeBuffer(length: topK * region.gateUpOut * 4,
                                          options: .storageModeShared),
              let actB = device.makeBuffer(length: topK * region.expertIntermediate * 4,
                                           options: .storageModeShared),
              let dnB = device.makeBuffer(length: topK * region.hidden * 4,
                                          options: .storageModeShared) else {
            throw MoELoweringError.hardwareUnavailable("incumbent buffers")
        }
        func matvec(_ pipe: MTLComputePipelineState, _ wb: MTLBuffer, _ wOff: Int,
                    inDim: Int, rows: Int, xin: MTLBuffer, xinOff: Int,
                    y: MTLBuffer, yOff: Int) throws {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                throw MoELoweringError.hardwareUnavailable("incumbent encoder")
            }
            var p = SIMD2<UInt32>(UInt32(inDim), UInt32(rows))
            var off = UInt64(wOff)
            enc.setComputePipelineState(pipe)
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(xin, offset: xinOff, index: 3)
            enc.setBuffer(y, offset: yOff, index: 4)
            let width = max(1, min(rows, pipe.maxTotalThreadsPerThreadgroup))
            let groups = (rows + width - 1) / width
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw MoELoweringError.componentGateFailed("incumbent command buffer")
            }
        }
        func geluUp(_ n: Int, src: MTLBuffer, srcOff: Int, y: MTLBuffer, yOff: Int) throws {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                throw MoELoweringError.hardwareUnavailable("incumbent encoder")
            }
            var count = UInt32(n)
            enc.setComputePipelineState(geluPipe)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(src, offset: srcOff, index: 1)
            enc.setBuffer(y, offset: yOff, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(1024, geluPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw MoELoweringError.componentGateFailed("incumbent command buffer")
            }
        }
        for slot in 0..<topK {
            try matvec(matPipe, wbGu, slot * guStride, inDim: region.hidden,
                       rows: region.gateUpOut, xin: xb, xinOff: 0, y: guB,
                       yOff: slot * region.gateUpOut * 4)
            try geluUp(region.expertIntermediate, src: guB,
                       srcOff: slot * region.gateUpOut * 4, y: actB,
                       yOff: slot * region.expertIntermediate * 4)
            // DN reads THIS slot's activation (checked offset).
            try matvec(matPipe, wbDn, slot * dnStride, inDim: region.expertIntermediate,
                       rows: region.hidden, xin: actB,
                       xinOff: slot * region.expertIntermediate * 4, y: dnB,
                       yOff: slot * region.hidden * 4)
        }
        func readF(_ b: MTLBuffer, _ count: Int) -> [Float] {
            [Float](UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: Float.self),
                                        count: count))
        }
        let gu = readF(guB, topK * region.gateUpOut)
        let act = readF(actB, topK * region.expertIntermediate)
        let dn = readF(dnB, topK * region.hidden)
        guard gu.allSatisfy({ $0.isFinite }),
              act.allSatisfy({ $0.isFinite }),
              dn.allSatisfy({ $0.isFinite }) else {
            throw MoELoweringError.componentGateFailed(
                "incumbent outputs non-finite (geometry \(region.hidden)/\(region.expertIntermediate)/\(topK))")
        }
        return (gu, act, dn)
    }

    /// Independent CPU FP64 oracle over the REAL expert bytes: bounded
    /// rows of the first two slots' GU and DN matvecs (Q4Oracle consumes
    /// EXACTLY the bounded-row byte prefix of each slice), plus a Double
    /// gelu-tanh reference on the gate|up outputs. All arrays are
    /// finite-checked before any max()/rel arithmetic.
    public static func oracleRels(region: MoERegionIR, guBlocks: [UInt8],
                           dnBlocks: [UInt8], guGpu: [Float], actGpu: [Float],
                           dnGpu: [Float], x: [Float], rows: Int = 64) throws -> Double {
        try region.validate()
        let topK = region.topK
        let guStride = try MoEModelBinding.sliceStride(
            inDim: region.hidden, outDim: region.gateUpOut,
            MoELoweringError.oracleFailed)
        let dnStride = try MoEModelBinding.sliceStride(
            inDim: region.expertIntermediate, outDim: region.hidden,
            MoELoweringError.oracleFailed)
        guard guStride * topK == guBlocks.count, dnStride * topK == dnBlocks.count,
              guGpu.count == topK * region.gateUpOut,
              actGpu.count == topK * region.expertIntermediate,
              dnGpu.count == topK * region.hidden,
              x.count == region.hidden,
              guGpu.allSatisfy({ $0.isFinite }),
              actGpu.allSatisfy({ $0.isFinite }),
              dnGpu.allSatisfy({ $0.isFinite }),
              x.allSatisfy({ $0.isFinite }) else {
            throw MoELoweringError.oracleFailed("oracle geometry/finiteness mismatch")
        }
        let boundedRows = min(rows, region.gateUpOut, region.hidden)
        let guRowBytes = try MoEModelBinding.mul(
            boundedRows, region.hidden / 32 * 18, "GU oracle prefix",
            MoELoweringError.oracleFailed)
        let dnRowBytes = try MoEModelBinding.mul(
            boundedRows, region.expertIntermediate / 32 * 18, "DN oracle prefix",
            MoELoweringError.oracleFailed)
        var worst = 0.0
        func rel(_ y: [Float], _ ref: [Double]) -> Double {
            guard y.count == ref.count else { return .infinity }
            var w = 0.0
            for i in 0..<y.count {
                w = max(w, abs(Double(y[i]) - ref[i]) / max(1.0, abs(ref[i])))
            }
            return w
        }
        for slot in 0..<min(2, topK) {
            let guByteStart = slot * guStride
            let guOutStart = slot * region.gateUpOut
            let actStart = slot * region.expertIntermediate
            let dnByteStart = slot * dnStride
            let dnOutStart = slot * region.hidden
            let guRef = try Q4Oracle.matvecFP64(
                blocks: guBlocks[guByteStart..<(guByteStart + guRowBytes)],
                outDim: boundedRows, inDim: region.hidden, x: x)
            let guRows = Array(guGpu[guOutStart..<(guOutStart + boundedRows)])
            worst = max(worst, rel(guRows, guRef))
            let actSlot = Array(actGpu[actStart..<(actStart + region.expertIntermediate)])
            let dnRef = try Q4Oracle.matvecFP64(
                blocks: dnBlocks[dnByteStart..<(dnByteStart + dnRowBytes)],
                outDim: boundedRows, inDim: region.expertIntermediate, x: actSlot)
            let dnRows = Array(dnGpu[dnOutStart..<(dnOutStart + boundedRows)])
            worst = max(worst, rel(dnRows, dnRef))
            var actRef = [Double]()
            actRef.reserveCapacity(region.expertIntermediate)
            for lane in 0..<region.expertIntermediate {
                let g = Double(guGpu[slot * region.gateUpOut + lane])
                let u = Double(guGpu[slot * region.gateUpOut + region.expertIntermediate + lane])
                let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
                actRef.append(0.5 * g * (1.0 + tanh(inner)) * u)
            }
            worst = max(worst, rel(actSlot, actRef))
        }
        guard worst.isFinite, worst < 1e-4 else {
            throw MoELoweringError.oracleFailed(
                String(format: "oracle rel %.3e outside the 1e-4 contract", worst))
        }
        return worst
    }

    /// Component forge: parse + revalidate the region, bind the model
    /// file, derive legal schedules, emit + compile every candidate, and
    /// gate each against the ACTUAL incumbent kernels (bitPattern
    /// equality on ALL slots, required) and the independent FP64 oracle
    /// (the incumbent is measured against the same oracle). Numerical
    /// selection only — no performance claim.
    public static func forge(regionJSON: Data, ggufURL: URL) throws
            -> (artifact: MoELoweringArtifact, results: [MoEComponentResult],
                incumbentRel: Double, region: MoERegionIR) {
        let region = try MoERegionIR.parse(regionJSON)
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw MoELoweringError.hardwareUnavailable("Metal unavailable")
        }
        let file = try GGUFReader.read(ggufURL)
        let binding = try MoELoweringForge.bindRegion(region: region, file: file,
                                                      device: device)
        let hardware = try MoECurrentHardware.profile()
        let ids = try MoEComponentInputs.expertIds(count: region.topK,
                                                   experts: binding.expertCount)
        let guBlocks = try MoEComponentInputs.expertBlocks(
            file: file, tensor: region.guTensor, ids: ids,
            inDim: region.hidden, outDim: region.gateUpOut)
        let dnBlocks = try MoEComponentInputs.expertBlocks(
            file: file, tensor: region.dnTensor, ids: ids,
            inDim: region.expertIntermediate, outDim: region.hidden)
        let x = try MoEComponentInputs.activation(region: region)
        let inc = try runIncumbent(device: device, queue: queue, region: region,
                                   guBlocks: guBlocks, dnBlocks: dnBlocks, x: x)
        let incumbentRel = try oracleRels(region: region, guBlocks: guBlocks,
                                          dnBlocks: dnBlocks, guGpu: inc.gu,
                                          actGpu: inc.act, dnGpu: inc.dn, x: x)

        var results = [MoEComponentResult]()
        var best: (schedule: MoESchedule, source: String, rel: Double)? = nil
        for schedule in try MoEScheduleDeriver.derive(region: region, hardware: hardware) {
            do {
                let kernels = try MoEEmittedKernels.compile(region: region,
                                                            schedule: schedule,
                                                            device: device)
                let out = try kernels.runComponent(region: region,
                                                   guBlocks: guBlocks,
                                                   dnBlocks: dnBlocks, x: x)
                func bitsEqual(_ a: [Float], _ b: [Float]) -> Bool {
                    a.count == b.count && zip(a, b).allSatisfy {
                        $0.bitPattern == $1.bitPattern }
                }
                let bitIdentical = bitsEqual(out.gu, inc.gu)
                    && bitsEqual(out.act, inc.act) && bitsEqual(out.dn, inc.dn)
                let rel = try oracleRels(region: region, guBlocks: guBlocks,
                                         dnBlocks: dnBlocks, guGpu: out.gu,
                                         actGpu: out.act, dnGpu: out.dn, x: x)
                let accepted = bitIdentical && rel < 1e-4
                results.append(MoEComponentResult(
                    schedule: schedule, accepted: accepted,
                    bitIdenticalToIncumbent: bitIdentical,
                    worstRelToOracle: rel,
                    note: accepted
                        ? String(format: "bits identical to incumbent; rel64 %.3e", rel)
                        : bitIdentical
                            ? String(format: "outside 1e-4 FP64 contract (rel64 %.3e)", rel)
                            : "NOT bit-identical to the actual incumbent kernels"))
                if accepted, best == nil || rel < best!.rel {
                    best = (schedule, kernels.source, rel)
                }
            } catch {
                results.append(MoEComponentResult(
                    schedule: schedule, accepted: false,
                    bitIdenticalToIncumbent: nil, worstRelToOracle: nil,
                    note: "rejected: \(error)"))
            }
        }
        guard let winner = best else {
            throw MoELoweringError.componentGateFailed(
                results.map { "\($0.schedule): \($0.note)" }.joined(separator: "; "))
        }
        let ggufDigest = try GemmaForge.fileDigest(ggufURL)
        let artifact = try MoELoweringArtifact.withDigest(MoELoweringArtifact(
            artifactSchema: MoELoweringArtifact.schema,
            codegenVersion: MoEKernelEmitter.codegenVersion,
            regionDigest: try region.digest(), region: region,
            binding: binding, schedule: winner.schedule,
            guKernelName: winner.schedule.guKernelName,
            actKernelName: winner.schedule.actKernelName,
            dnKernelName: winner.schedule.dnKernelName,
            emittedSource: winner.source,
            sourceSha256: MoEKernelEmitter.sourcesSha256([winner.source]),
            hardware: hardware, toolchain: Forge.toolchainIdentifier,
            ggufDigest: ggufDigest,
            executableDigest: try Gemma4Runner.executableDigest(),
            componentResults: results,
            incumbentWorstRelToOracle: incumbentRel,
            expertIds: ids, modelGate: nil, artifactDigest: ""))
        return (artifact, results, incumbentRel, region)
    }

    // MARK: bounded model-level selection (composed from LoweringSelect)

    /// Full bounded selection in ONE process, composed from the shared
    /// family-agnostic helpers (LoweringSelect) — NOT a per-family
    /// selector clone. Protocol: component gates; component screen at the
    /// real hot shape (E=1 sanity anchor excluded; 2 fastest finalists);
    /// per-finalist teacher-forced strict gates with dispatch/fallback
    /// DELTA proof; matched 3-rep AB/BA paired timing; fixed 2% policy;
    /// persist + reload proof (production installs publicly only when the
    /// policy SELECTED the candidate; an honest no-win records
    /// performanceSelected=false, stays incumbent-active, and still
    /// proves the kernels executed via the internal trial path).
    public static func selectAndPersist(regionJSON: Data, ggufURL: URL,
                                        graph: GemmaGraphIR,
                                        runner: Gemma4Runner,
                                        fixtures: [Q4LoweringForge.Q4Fixture],
                                        genTokens: Int, timedTokens: Int,
                                        isEndOfGeneration: (Int) -> Bool,
                                        diagnostic: Bool,
                                        saveTo url: URL) throws
            -> (artifact: MoELoweringArtifact, log: [String]) {
        if diagnostic {
            guard genTokens >= 2, genTokens <= 64, timedTokens >= 1,
                  timedTokens < genTokens, fixtures.count >= 1, fixtures.count <= 4,
                  Set(fixtures.map(\.name)).count == fixtures.count else {
                throw MoELoweringError.invalidSchedule("smoke bounds violated")
            }
        } else {
            guard genTokens == MoEGatePolicy.productionGenTokens,
                  timedTokens == MoEGatePolicy.productionTimedTokens,
                  fixtures.map(\.name) == MoEGatePolicy.productionFixtures else {
                throw MoELoweringError.invalidSchedule(
                    "production select requires 32/31 over en-chat/kr-chat/raw-hello")
            }
        }
        for f in fixtures {
            guard !f.tokens.isEmpty, f.tokens.count + genTokens <= 4096 else {
                throw MoELoweringError.invalidSchedule("fixture \(f.name) exceeds context")
            }
        }
        var log = [String]()
        let region = try MoERegionIR.parse(regionJSON)
        // EVERY arm (reference, gates, timing, reload proof) runs on the
        // SAME compiled Forge SSA execution path — never mixed with the
        // direct runner forward.
        let (ssaMany, ssaOne) = LoweringSelect.ssaForwardAdapters(graph: graph)
        // Selection NEVER leaks an install into another phase.
        defer { runner.uninstallMoeGeneratedLowering() }
        runner.uninstallMoeGeneratedLowering()

        // ---- 1. component phase ----
        let (componentArtifact, componentResults, incumbentRel, _) = try forge(
            regionJSON: regionJSON, ggufURL: ggufURL)
        log.append(String(format: "component: incumbent rel-to-FP64 %.3e; %d/%d candidates accepted",
                          incumbentRel,
                          componentResults.filter(\.accepted).count,
                          componentResults.count))

        // ---- 2. component screen (real hot shape; E=1 sanity anchor) ----
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MoELoweringError.hardwareUnavailable("Metal unavailable")
        }
        let file = try GGUFReader.read(ggufURL)
        let ids = try MoEComponentInputs.expertIds(count: region.topK,
                                                   experts: componentArtifact.binding.expertCount)
        let guBlocks = try MoEComponentInputs.expertBlocks(
            file: file, tensor: region.guTensor, ids: ids,
            inDim: region.hidden, outDim: region.gateUpOut)
        let dnBlocks = try MoEComponentInputs.expertBlocks(
            file: file, tensor: region.dnTensor, ids: ids,
            inDim: region.expertIntermediate, outDim: region.hidden)
        let screenX = try MoEComponentInputs.activation(region: region)
        var compiled = [MoESchedule: MoEEmittedKernels]()
        var screenMed = [MoESchedule: Double]()
        for result in componentResults where result.accepted {
            do {
                let kernels = try MoEEmittedKernels.compile(
                    region: region, schedule: result.schedule, device: device)
                compiled[result.schedule] = kernels
                screenMed[result.schedule] = result.schedule.batch == 1
                    ? .infinity   // E1 sanity anchor: recorded, never finalist
                    : try kernels.timedLayerMedianUs(region: region,
                                                     guBlocks: guBlocks,
                                                     dnBlocks: dnBlocks, x: screenX)
            } catch {
                screenMed[result.schedule] = .infinity
                log.append("screen \(result.schedule): compile/screen failed: \(error)")
            }
        }
        let finalists = Set(
            screenMed.filter { $0.key.batch != 1 && $0.value.isFinite }
                .sorted {
                    $0.value < $1.value
                        || ($0.value == $1.value && $0.key.guKernelName < $1.key.guKernelName)
                }.prefix(2).map(\.key))
        if finalists.isEmpty {
            throw MoELoweringError.componentGateFailed(
                "component screen left no eligible finalists (accepted: "
                + "\(componentResults.filter(\.accepted).count))")
        }
        for result in componentResults where result.accepted {
            let med = screenMed[result.schedule] ?? .infinity
            log.append(String(format: "screen %@: %@ %.1f us/layer-equivalent%@",
                              result.schedule.guKernelName,
                              finalists.contains(result.schedule) ? "finalist" : "pruned",
                              med,
                              result.schedule.batch == 1 ? " (E1 sanity anchor)" : ""))
        }
        if let maxE = finalists.map(\.batch).max() {
            let batchedPerLayer = 3 * ((region.topK + maxE - 1) / maxE)
            log.append("dispatch arithmetic at E=\(maxE): \(batchedPerLayer) batched "
                + "math dispatches per layer/token vs 24 incumbent (\(24 - batchedPerLayer) "
                + "fewer; count claim, not a speed claim)")
        }

        // ---- 3. reference arms (shared helper; incumbent kernels only) ----
        let refArms = try LoweringSelect.generateReferenceArms(
            runner: runner, fixtures: fixtures, genTokens: genTokens,
            isEndOfGeneration: isEndOfGeneration,
            forwardMany: ssaMany, forwardOne: ssaOne,
            reset: LoweringSelect.ssaReset,
            requireFullLength: !diagnostic, log: &log)

        // ---- 4. per-finalist strict gates with dispatch/fallback DELTAS ----
        var modelResults = [MoEModelCandidateResult]()
        var passingKernels = [MoESchedule: MoEEmittedKernels]()
        for result in componentResults where result.accepted {
            guard finalists.contains(result.schedule) else {
                modelResults.append(MoEModelCandidateResult(
                    schedule: result.schedule, fixturesPassed: 0, fixturesFailed: 0,
                    worstRel: nil,
                    gateFailReason: "component screen prune (not a gate failure)",
                    wallMedPerFixture: nil, gpuMedPerFixture: nil))
                continue
            }
            guard let kernels = compiled[result.schedule] else {
                modelResults.append(MoEModelCandidateResult(
                    schedule: result.schedule, fixturesPassed: 0,
                    fixturesFailed: fixtures.count, worstRel: nil,
                    gateFailReason: "compile: missing after screen",
                    wallMedPerFixture: nil, gpuMedPerFixture: nil))
                continue
            }
            do { try runner.installMoeTrialLowering(region: region, kernels: kernels) }
            catch {
                modelResults.append(MoEModelCandidateResult(
                    schedule: result.schedule, fixturesPassed: 0,
                    fixturesFailed: fixtures.count, worstRel: nil,
                    gateFailReason: "install: \(error)",
                    wallMedPerFixture: nil, gpuMedPerFixture: nil))
                continue
            }
            var passed = 0
            var failed = 0
            var worstAll = 0.0
            var firstFail: String? = nil
            var totalGenerated = 0
            var fallbackDelta = 0
            for fixture in fixtures {
                let ref = refArms[fixture.name]!
                let (worst, fail, dispatches) = LoweringSelect.gateReplay(
                    runner: runner, fixture: fixture, ref: ref,
                    forwardMany: ssaMany, forwardOne: ssaOne,
                    reset: LoweringSelect.ssaReset,
                    dispatchCount: { runner.moeGeneratedDispatchCount },
                    fallbackCount: { runner.moeResidentFallbackCount })
                totalGenerated += dispatches
                if let fail {
                    failed += 1
                    if firstFail == nil { firstFail = "[\(fixture.name)] \(fail)" }
                } else {
                    passed += 1
                    worstAll = max(worstAll, worst)
                }
            }
            fallbackDelta = runner.moeResidentFallbackCount
            runner.uninstallMoeGeneratedLowering()
            var fullyPassed = failed == 0
            if fullyPassed, totalGenerated <= 0 {
                firstFail = "no batched dispatches recorded (generated path never ran)"
                fullyPassed = false
            }
            if fullyPassed, fallbackDelta != 0 {
                firstFail = "generated path fell back \(fallbackDelta) times"
                fullyPassed = false
            }
            if fullyPassed { passingKernels[result.schedule] = kernels }
            modelResults.append(MoEModelCandidateResult(
                schedule: result.schedule, fixturesPassed: passed,
                fixturesFailed: fullyPassed ? 0 : max(failed, 1),
                worstRel: fullyPassed ? worstAll : nil,
                gateFailReason: firstFail,
                wallMedPerFixture: nil, gpuMedPerFixture: nil))
            log.append(String(format: "gate %@ %@: %d/%d fixtures, worst rel %.3e, batched dispatches %d, fallbacks %d",
                              fullyPassed ? "PASS" : "FAIL",
                              result.schedule.guKernelName, passed, fixtures.count,
                              worstAll, totalGenerated, fallbackDelta))
            if !fullyPassed, let f = firstFail {
                log.append("  first failure: \(f)")
            }
        }
        guard modelResults.contains(where: { $0.fixturesFailed == 0 && $0.worstRel != nil })
        else {
            throw MoELoweringError.componentGateFailed(
                "no generated candidate passed all fixture gates: "
                + modelResults.compactMap { r -> String? in
                    r.fixturesFailed > 0 ? "\(r.schedule.guKernelName): \(r.gateFailReason ?? "?")" : nil
                }.joined(separator: "; "))
        }

        // ---- 5. matched paired timing (shared helper) ----
        var best: (schedule: MoESchedule, candMed: Double, incMed: Double)? = nil
        var incWalls: [Double] = []
        var incGpus: [Double] = []
        for i in modelResults.indices where modelResults[i].fixturesFailed == 0 {
            guard let schedule = passingKernels.keys.first(where: {
                $0 == modelResults[i].schedule }) else { continue }
            let timing = try LoweringSelect.pairedTiming(
                runner: runner, fixtures: fixtures, refArms: refArms,
                timedTokens: timedTokens,
                forwardMany: ssaMany, forwardOne: ssaOne,
                reset: LoweringSelect.ssaReset,
                install: {
                    try runner.installMoeTrialLowering(region: region,
                                                       kernels: passingKernels[schedule]!)
                },
                uninstall: { runner.uninstallMoeGeneratedLowering() })
            modelResults[i].timingReps = timing.reps
            modelResults[i].wallMedPerFixture = timing.wallMedPerFixture
            modelResults[i].gpuMedPerFixture = timing.gpuMedPerFixture
            log.append(String(format: "matched timing %@: median %.0f vs paired incumbent %.0f us/token-sum",
                              schedule.guKernelName, timing.candMed, timing.incMed))
            if best == nil
                || timing.candMed / timing.incMed < best!.candMed / best!.incMed
                || (timing.candMed / timing.incMed == best!.candMed / best!.incMed
                    && schedule.guKernelName < best!.schedule.guKernelName) {
                best = (schedule, timing.candMed, timing.incMed)
                incWalls = (0..<fixtures.count).map { f in
                    LoweringSelect.median(timing.reps.map {
                        LoweringSelect.armFixtureMedians($0.incumbent)[f] })
                }
                incGpus = (0..<fixtures.count).map { f in
                    LoweringSelect.median(timing.reps.map {
                        LoweringSelect.median($0.incumbent[f].gpus) })
                }
            }
        }

        // ---- 6. fixed 2% policy ----
        let performanceSelected: Bool
        let selectedSchedule: MoESchedule
        if let b = best {
            performanceSelected = LoweringSelect.policySelects(candMed: b.candMed,
                                                               incMed: b.incMed)
            selectedSchedule = b.schedule
        } else {
            performanceSelected = false
            selectedSchedule = componentArtifact.schedule
        }
        let policyNote = performanceSelected
            ? String(format: "selected %@ beats its own paired incumbent matched-repeat median summed wall by %.1f%% (> %.0f%% policy threshold)",
                     selectedSchedule.guKernelName,
                     (1 - best!.candMed / best!.incMed) * 100,
                     GemmaForge.noiseRelWall * 100)
            : "incumbent fallback: no gate-passing candidate beat its own paired incumbent beyond the policy threshold on matched-repeat medians"
        log.append("policy: \(policyNote)")

        // ---- persist ----
        let selectedSource = try MoEKernelEmitter.emit(region: region,
                                                       schedule: selectedSchedule)
        let fixtureRecords = fixtures.map { fixture -> Q4FixtureRecord in
            let ref = refArms[fixture.name]!
            var history = fixture.tokens
            history.append(contentsOf: ref.ids.dropLast())
            return Q4FixtureRecord(
                name: fixture.name, promptTokenCount: fixture.tokens.count,
                gatedSteps: ref.ids.count,
                historyDigest: GemmaForge.workloadDigest(history))
        }
        let gateRecord = MoEModelGateRecord(
            gatePolicy: MoEGatePolicy.version,
            executionPath: MoEGatePolicy.executionPathSSA,
            diagnostic: diagnostic,
            genTokens: genTokens, timedTokens: timedTokens,
            fixtures: fixtureRecords, results: modelResults,
            incumbentWallPerFixture: incWalls, incumbentGpuPerFixture: incGpus,
            performanceSelected: performanceSelected, policyNote: policyNote)
        let ggufDigest = try GemmaForge.fileDigest(ggufURL)
        let artifact = try MoELoweringArtifact.withDigest(MoELoweringArtifact(
            artifactSchema: componentArtifact.artifactSchema,
            codegenVersion: componentArtifact.codegenVersion,
            regionDigest: componentArtifact.regionDigest, region: region,
            binding: componentArtifact.binding, schedule: selectedSchedule,
            guKernelName: selectedSchedule.guKernelName,
            actKernelName: selectedSchedule.actKernelName,
            dnKernelName: selectedSchedule.dnKernelName,
            emittedSource: selectedSource,
            sourceSha256: MoEKernelEmitter.sourcesSha256([selectedSource]),
            hardware: componentArtifact.hardware,
            toolchain: Forge.toolchainIdentifier,
            ggufDigest: ggufDigest,
            executableDigest: try Gemma4Runner.executableDigest(),
            componentResults: componentArtifact.componentResults,
            incumbentWorstRelToOracle: componentArtifact.incumbentWorstRelToOracle,
            expertIds: componentArtifact.expertIds,
            modelGate: gateRecord, artifactDigest: ""))
        try artifact.save(to: url)

        // ---- 7. persisted reload proof with per-stage DELTA counters.
        // Expected batched dispatches = 3*((topK+E-1)/E) per layer per
        // forward; merges = topK per layer per forward; the replay's
        // forward count is the ACTUAL prompt length + gated steps - 1. ----
        let reloaded = try MoELoweringArtifact.load(from: url)
        try reloaded.validate(region: region,
                              hardware: componentArtifact.hardware,
                              ggufDigest: ggufDigest,
                              executableDigest: try Gemma4Runner.executableDigest())
        if diagnostic || !performanceSelected {
            // Smoke records and honest no-win selections prove execution
            // through the INTERNAL trial path; a public production install
            // remains reserved for policy-selected candidates.
            var reloadKernels = compiled[selectedSchedule]
            if reloadKernels == nil {
                reloadKernels = try MoEEmittedKernels.compile(
                    region: region, schedule: selectedSchedule, device: device)
            }
            try runner.installMoeTrialLowering(region: region, kernels: reloadKernels!)
        } else {
            try runner.installMoeGeneratedLowering(artifact: reloaded,
                                                   ggufDigest: ggufDigest)
        }
        let first = fixtures[0]
        let gu0 = runner.moeGuDispatchCount, act0 = runner.moeActDispatchCount
        let dn0 = runner.moeDnDispatchCount, merge0 = runner.moeAxpyMergeCount
        let fb0 = runner.moeResidentFallbackCount
        let (worst, fail, _) = LoweringSelect.gateReplay(
            runner: runner, fixture: first, ref: refArms[first.name]!,
            forwardMany: ssaMany, forwardOne: ssaOne,
            reset: LoweringSelect.ssaReset,
            dispatchCount: { runner.moeGeneratedDispatchCount },
            fallbackCount: { runner.moeResidentFallbackCount })
        let dGu = runner.moeGuDispatchCount - gu0
        let dAct = runner.moeActDispatchCount - act0
        let dDn = runner.moeDnDispatchCount - dn0
        let dMerge = runner.moeAxpyMergeCount - merge0
        let dFallback = runner.moeResidentFallbackCount - fb0
        runner.uninstallMoeGeneratedLowering()
        let forwards = first.tokens.count + refArms[first.name]!.ids.count - 1
        let layers = componentArtifact.binding.layerCount
        let eSel = selectedSchedule.batch
        let wantBatched = 3 * ((region.topK + eSel - 1) / eSel) * layers * forwards
        let wantMerges = region.topK * layers * forwards
        guard fail == nil, dGu == wantBatched / 3, dAct == wantBatched / 3,
              dDn == wantBatched / 3, dMerge == wantMerges, dFallback == 0 else {
            throw MoELoweringError.componentGateFailed(
                "persisted execution failed: \(fail ?? "counter mismatch") "
                    + "(GU \(dGu)/\(wantBatched / 3), ACT \(dAct), DN \(dDn), "
                    + "merges \(dMerge)/\(wantMerges), fallbacks \(dFallback))")
        }
        log.append(String(format: "persisted execution: reloaded+validated+installed from disk (%@); %@ replay worst rel %.3e; batched GU/ACT/DN deltas %d/%d/%d == expected %d each; ordered axpy merges %d == %d; fallbacks 0",
                          (diagnostic || !performanceSelected)
                              ? "internal trial" : "production",
                          first.name, worst, dGu, dAct, dDn, wantBatched / 3,
                          dMerge, wantMerges))
        return (artifact, log)
    }
}

extension MoELoweringArtifact {
    /// Convenience: body digest + re-wrap (used by the forge/select).
    static func withDigest(_ body: MoELoweringArtifact) throws -> MoELoweringArtifact {
        let d = try MoELoweringArtifact.bodyDigest(body)
        return MoELoweringArtifact(
            artifactSchema: body.artifactSchema,
            codegenVersion: body.codegenVersion,
            regionDigest: body.regionDigest, region: body.region,
            binding: body.binding, schedule: body.schedule,
            guKernelName: body.guKernelName, actKernelName: body.actKernelName,
            dnKernelName: body.dnKernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            expertIds: body.expertIds, modelGate: body.modelGate,
            artifactDigest: d)
    }
}
