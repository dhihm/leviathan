// Typed SSA dataflow IR lowered from a VALIDATED captured Gemma graph.
// Every write produces a new value version; every instruction's inputs
// reference specific previous versions with explicit dtype, length, and
// kind (vector / KV-cache state / routing state). The executor dispatches
// kernels from these OPERANDS, not from stage-name implicit state.
//
// Liveness discipline that makes fixed-buffer aliasing safe: every read
// references the CURRENTLY-LIVE version of its name at that program
// point (validated below). A buffer therefore has exactly one live
// logical value at any time, and all reads precede the next write of
// that name — so alias allocation onto the runner's fixed buffers after
// validation cannot clobber a live value. Expert-loop scratch buffers
// are internal effects of the compound op, not logical inputs.

import Foundation

public enum GemmaSSAError: Error, CustomStringConvertible, Equatable {
    case badLowering(String)
    case badDataflow(String)

    public var description: String {
        switch self {
        case .badLowering(let m): return "ssa lowering invalid: \(m)"
        case .badDataflow(let m): return "ssa dataflow invalid: \(m)"
        }
    }
}

public struct GemmaSSAIR: Codable, Equatable, Sendable {
    public enum DType: String, Codable, Equatable, Sendable { case f32 }

    /// Kind of an SSA value: an activation vector, per-layer KV-cache
    /// state (length = kvHeads*headDim for that layer), or per-layer
    /// routing state (length = topK; the expert indices/weights).
    public enum Kind: String, Codable, Equatable, Sendable {
        case vector, kvstate, routing
    }

    public struct Value: Codable, Equatable, Sendable {
        public let name: String
        public let version: Int
        public let dtype: DType
        public let length: Int
        public let kind: Kind
        /// Physical slice offset (elements) for values that alias a shared
        /// buffer at an offset (gate=0, up=sharedIntermediate in the fused
        /// gate|up buffer); nil for whole-buffer values. Recorded in the
        /// artifact so buffer interference is explicit, not a hidden
        /// executor assumption.
        public let sliceOffset: Int?

        public var id: String { "\(name)@\(version)" }
    }

    public struct Instruction: Codable, Equatable, Sendable {
        /// Generic math op (see validate for each signature).
        public let op: String
        public let layer: Int?
        public let inputs: [String]
        public let output: String
        /// Quantized weight tensor bindings (matvec: exactly 1;
        /// expert_matvec_chain: gateUp then down).
        public let tensors: [String]
        /// Scalar/geometry attributes (offset, heads, headDim, ...).
        public let attrs: [String: Int]
    }

    public let schema: Int
    public let values: [Value]
    public let instructions: [Instruction]

    public init(schema: Int, values: [Value], instructions: [Instruction]) {
        self.schema = schema
        self.values = values
        self.instructions = instructions
    }
}

public enum GemmaSSA {

    public static let schemaVersion = 1
    /// Logical activation-vector value names (KV/routing state uses the
    /// prefixed names kvstate_<L> / rout_<L> with matching kinds).
    public static let vectorNames: Set<String> = [
        "x", "h", "t1", "t2", "t3", "kv", "gate", "up", "act", "shared",
        "router", "logits",
    ]

    // MARK: - lowering (validated stage graph -> typed SSA)

    /// Lowers a graph that has already passed `GemmaGraph.validate`.
    /// Deterministic: same graph in, same SSA out. Layer "globalness" is
    /// derived from the CONFIG's global layer set; a present stage flag
    /// is cross-checked against it.
    public static func lower(_ graph: GemmaGraphIR) throws -> GemmaSSAIR {
        let c = graph.config
        var versions = [String: Int]()
        var values = [GemmaSSAIR.Value]()
        var instructions = [GemmaSSAIR.Instruction]()
        func def(_ name: String, _ length: Int,
                 _ kind: GemmaSSAIR.Kind = .vector,
                 sliceOffset: Int? = nil) -> String {
            let v = (versions[name] ?? -1) + 1
            versions[name] = v
            values.append(GemmaSSAIR.Value(name: name, version: v, dtype: .f32,
                                           length: length, kind: kind,
                                           sliceOffset: sliceOffset))
            return "\(name)@\(v)"
        }
        func use(_ name: String) throws -> String {
            guard let v = versions[name] else {
                throw GemmaSSAError.badLowering("use of undefined value \(name)")
            }
            return "\(name)@\(v)"
        }
        func emit(_ op: String, _ layer: Int?, _ tensors: [String],
                  _ inputs: [String], _ output: String,
                  _ attrs: [String: Int] = [:]) {
            instructions.append(GemmaSSAIR.Instruction(
                op: op, layer: layer, inputs: inputs, output: output,
                tensors: tensors, attrs: attrs))
        }

        for stage in graph.ops {
            let L = stage.layer
            // Globalness from config (authoritative); cross-check flag.
            let isGlobal = c.globalLayers.contains(L ?? -1)
            if let flag = stage.global, flag != isGlobal {
                throw GemmaSSAError.badLowering(
                    "stage global flag disagrees with config at layer \(L ?? -1)")
            }
            let headDim = isGlobal ? c.globalHeadDim : c.localHeadDim
            let kvHeads = isGlobal ? c.globalKVHeads : c.localKVHeads
            let kvWidth = kvHeads * headDim
            let qWidth = c.qHeads * headDim
            let t = stage.tensor.map { [$0] } ?? []
            switch stage.op {
            case "embed":
                // The decoded embedding row lands directly in x (the
                // residual base), matching the reference forward.
                emit("embed_row", nil, t, [], def("x", c.hidden))
            case "attn_norm":
                emit("rmsnorm", L, t, [try use("x")], def("h", c.hidden))
            case "q_proj":
                emit("matvec", L, t, [try use("h")], def("t1", qWidth),
                     ["rows": qWidth])
            case "q_norm":
                emit("rmsnorm", L, t, [try use("t1")], def("t1", qWidth),
                     ["vectors": c.qHeads, "width": headDim])
            case "q_rope":
                emit("rope", L, [], [try use("t1")], def("t1", qWidth),
                     ["heads": c.qHeads, "half": headDim / 2, "global": isGlobal ? 1 : 0])
            case "k_proj":
                emit("matvec", L, t, [try use("h")], def("t2", kvWidth),
                     ["rows": kvWidth])
            case "v_eq_k":
                emit("copy", L, [], [try use("t2")], def("kv", kvWidth))
            case "v_proj":
                emit("matvec", L, t, [try use("h")], def("kv", kvWidth),
                     ["rows": kvWidth])
            case "k_norm":
                emit("rmsnorm", L, t, [try use("t2")], def("t2", kvWidth),
                     ["vectors": kvHeads, "width": headDim])
            case "k_rope":
                emit("rope", L, [], [try use("t2")], def("t2", kvWidth),
                     ["heads": kvHeads, "half": headDim / 2, "global": isGlobal ? 1 : 0])
            case "v_norm":
                emit("rmsnorm", L, [], [try use("kv")], def("kv", kvWidth),
                     ["vectors": kvHeads, "width": headDim])
            case "cache_append":
                emit("cache_append", L, [],
                     [try use("t2"), try use("kv")],
                     def("kvstate_\(L ?? -1)", kvWidth, .kvstate),
                     ["kvHeads": kvHeads, "headDim": headDim])
            case "attention":
                emit("attention", L, [],
                     [try use("t1"), try use("kvstate_\(L ?? -1)")],
                     def("t3", qWidth),
                     ["heads": c.qHeads, "kvHeads": kvHeads, "headDim": headDim,
                      "window": isGlobal ? 0 : c.slidingWindow])
            case "o_proj":
                emit("matvec", L, t, [try use("t3")], def("t2", c.hidden),
                     ["rows": c.hidden])
            case "post_attn_norm":
                emit("rmsnorm", L, t, [try use("t2")], def("h", c.hidden))
            case "residual_add":
                // Accumulate h into x: BOTH operands are explicit inputs.
                emit("axpy", L, [], [try use("x"), try use("h")], def("x", c.hidden))
            case "ffn_norm":
                emit("rmsnorm", L, t, [try use("x")], def("h", c.hidden))
            case "shared_gate":
                // gate/up are DISTINCT logical values (offset slices of the
                // fused gate|up buffer — offsets RECORDED in the value, so
                // buffer interference is explicit in the artifact).
                emit("matvec", L, t, [try use("h")], def("gate", c.sharedIntermediate,
                                                         sliceOffset: 0),
                     ["rows": c.sharedIntermediate, "offset": 0])
            case "shared_up":
                emit("matvec", L, t, [try use("h")],
                     def("up", c.sharedIntermediate, sliceOffset: c.sharedIntermediate),
                     ["rows": c.sharedIntermediate, "offset": c.sharedIntermediate])
            case "shared_gelu":
                emit("gelu_up", L, [], [try use("gate"), try use("up")],
                     def("act", c.sharedIntermediate), ["width": c.sharedIntermediate])
            case "shared_down":
                emit("matvec", L, t, [try use("act")], def("shared", c.hidden),
                     ["rows": c.hidden])
            case "shared_post1":
                emit("rmsnorm", L, t, [try use("shared")], def("shared", c.hidden))
            case "router_norm":
                emit("rmsnorm", L, [], [try use("x")], def("h", c.hidden))
            case "router_scale":
                emit("scale", L, [], [try use("h")], def("h", c.hidden),
                     ["routerScale": 1])
            case "router_proj":
                emit("matvec", L, t, [try use("h")], def("router", c.expertCount),
                     ["rows": c.expertCount])
            case "router_topk":
                emit("router_topk", L, [], [try use("router")],
                     def("rout_\(L ?? -1)", c.expertUsedCount, .routing),
                     ["topK": c.expertUsedCount, "experts": c.expertCount])
            case "moe_norm2":
                emit("rmsnorm", L, t, [try use("x")], def("h", c.hidden))
            case "expert_loop":
                // Compound op: logical inputs are the normalized input and
                // the routing state ONLY. The gu/act/accumulator scratch
                // buffers are internal effects (attrs), not stale logical
                // inputs.
                emit("expert_matvec_chain", L,
                     [stage.gateUp ?? "", stage.down ?? ""],
                     [try use("h"), try use("rout_\(L ?? -1)")],
                     def("t3", c.hidden),
                     ["experts": c.expertUsedCount, "intermediate": c.expertIntermediate])
            case "moe_post2":
                emit("rmsnorm", L, t, [try use("t3")], def("t3", c.hidden))
            case "combine":
                emit("add", L, [], [try use("shared"), try use("t3")], def("h", c.hidden))
            case "post_ffw_norm":
                emit("rmsnorm", L, t, [try use("h")], def("h", c.hidden))
            case "layer_scale":
                emit("scale", L, [], [try use("x")], def("x", c.hidden),
                     ["layerScale": 1])
            case "final_norm":
                emit("rmsnorm", nil, t, [try use("x")], def("h", c.hidden))
            case "lm_head":
                emit("matvec", nil, t, [try use("h")], def("logits", c.vocabSize),
                     ["rows": c.vocabSize])
            case "softcap":
                emit("softcap", nil, [], [try use("logits")], def("logits", c.vocabSize),
                     ["cap": Int(c.finalSoftcap)])
            default:
                throw GemmaSSAError.badLowering("unsupported stage op \(stage.op)")
            }
        }
        let ssa = GemmaSSAIR(schema: schemaVersion, values: values,
                             instructions: instructions)
        try validate(ssa, config: c)
        return ssa
    }

    // MARK: - dataflow validation

    /// Checks the SSA on its own terms: schema, name/kind legality, dtype
    /// uniformity, def-before-use, the LIVE-VERSION discipline (every
    /// input references the currently-live version of its name — the
    /// property that makes post-validation buffer aliasing safe), per-op
    /// operand signatures and length compatibility, state-kind geometry,
    /// tensor-binding arity, and a logits-producing terminal.
    public static func validate(_ ssa: GemmaSSAIR, config c: GemmaGraphIR.GraphConfig) throws {
        guard ssa.schema == schemaVersion else {
            throw GemmaSSAError.badDataflow("schema \(ssa.schema) != \(schemaVersion)")
        }
        // Bounded length ceiling: no value may claim an absurd allocation.
        let maxLen = max(c.vocabSize, 8 * c.hidden)
        var byID = [String: GemmaSSAIR.Value]()
        for v in ssa.values {
            let nameOk = (vectorNames.contains(v.name) && v.kind == .vector)
                || (v.name.hasPrefix("kvstate_") && v.kind == .kvstate)
                || (v.name.hasPrefix("rout_") && v.kind == .routing)
            guard nameOk else {
                throw GemmaSSAError.badDataflow("value \(v.id): name/kind mismatch")
            }
            guard v.dtype == .f32, v.length > 0, v.length <= maxLen,
                  v.version >= 0,
                  v.sliceOffset == nil || v.sliceOffset! >= 0 else {
                throw GemmaSSAError.badDataflow("bad value \(v.id)")
            }
            guard byID[v.id] == nil else {
                throw GemmaSSAError.badDataflow("duplicate value \(v.id)")
            }
            byID[v.id] = v
        }
        func checkedMul(_ a: Int, _ b: Int) -> Int? {
            let (p, o) = a.multipliedReportingOverflow(by: b)
            return o ? nil : p
        }
        // Live version per name; reads must reference exactly it.
        var live = [String: Int]()
        func viv(_ id: String, _ i: Int, _ what: String) throws -> GemmaSSAIR.Value {
            guard let v = byID[id] else {
                throw GemmaSSAError.badDataflow(
                    "instruction \(i) (\(what)) references undefined value \(id)")
            }
            guard live[v.name] == v.version else {
                throw GemmaSSAError.badDataflow(
                    "instruction \(i) (\(what)) reads non-live version \(id) "
                    + "(live is \(v.name)@\(live[v.name].map(String.init) ?? "none"))")
            }
            return v
        }
        for (i, inst) in ssa.instructions.enumerated() {
            // Layer, where present, must be a legal layer index (or nil
            // for global instructions).
            if let layer = inst.layer {
                guard layer >= 0, layer < c.layerCount else {
                    throw GemmaSSAError.badDataflow("layer \(layer) out of range at \(i)")
                }
            }
            let inputs = try inst.inputs.map { try viv($0, i, inst.op) }
            guard let out = byID[inst.output] else {
                throw GemmaSSAError.badDataflow(
                    "instruction \(i) (\(inst.op)) writes undefined value \(inst.output)")
            }
            if let seen = live[out.name], out.version <= seen {
                throw GemmaSSAError.badDataflow(
                    "non-fresh output version \(inst.output) at instruction \(i)")
            }
            live[out.name] = out.version
            switch inst.op {
            case "embed_row":
                guard inst.inputs.isEmpty, inst.tensors.count == 1,
                      out.length == c.hidden else {
                    throw GemmaSSAError.badDataflow("embed_row operands invalid at \(i)")
                }
            case "gelu_up":
                guard inputs.count == 2,
                      let width = inst.attrs["width"],
                      width > 0,
                      inputs[0].length == width, inputs[1].length == width,
                      out.length == width,
                      inputs[0].name == "gate", inputs[1].name == "up",
                      // Explicit slice interference: up starts exactly
                      // where gate ends in the fused buffer.
                      inputs[0].sliceOffset == 0,
                      inputs[1].sliceOffset == width else {
                    throw GemmaSSAError.badDataflow("gelu_up operands invalid at \(i)")
                }
            case "softcap":
                guard inputs.count == 1, inputs[0].length == out.length,
                      inst.attrs["cap"] != nil,
                      out.name == inputs[0].name else {
                    throw GemmaSSAError.badDataflow("softcap operands invalid at \(i)")
                }
            case "rmsnorm", "copy":
                guard inputs.count == 1, inputs[0].length == out.length,
                      inst.tensors.count <= 1 else {
                    throw GemmaSSAError.badDataflow(
                        "\(inst.op) operand/length mismatch at \(i)")
                }
                if let vectors = inst.attrs["vectors"], let width = inst.attrs["width"] {
                    guard vectors > 0, width > 0, vectors * width == out.length else {
                        throw GemmaSSAError.badDataflow(
                            "rmsnorm vectors/width geometry invalid at \(i)")
                    }
                }
            case "rope":
                guard inputs.count == 1, inst.tensors.isEmpty,
                      let heads = inst.attrs["heads"], let half = inst.attrs["half"],
                      heads > 0, half > 0, inputs[0].length == out.length,
                      out.length == heads * 2 * half,
                      inst.attrs["global"] != nil,
                      out.name == inputs[0].name else {
                    throw GemmaSSAError.badDataflow("rope operands invalid at \(i)")
                }
            case "scale":
                guard inputs.count == 1, inputs[0].length == out.length,
                      inst.tensors.isEmpty,
                      inst.attrs["routerScale"] != nil || inst.attrs["layerScale"] != nil,
                      out.name == inputs[0].name
                else {
                    throw GemmaSSAError.badDataflow("scale operands invalid at \(i)")
                }
            case "matvec":
                guard inputs.count == 1, inst.tensors.count == 1,
                      let rows = inst.attrs["rows"], rows == out.length else {
                    throw GemmaSSAError.badDataflow("matvec operand/rows mismatch at \(i)")
                }
            case "axpy":
                // BOTH the accumulator (x) and the addend (h) are inputs;
                // accumulation is in place on the accumulator's buffer.
                guard inputs.count == 2,
                      inputs.allSatisfy({ $0.length == out.length }),
                      out.name == inputs[0].name else {
                    throw GemmaSSAError.badDataflow("axpy operand/length mismatch at \(i)")
                }
            case "add":
                guard inputs.count == 2,
                      inputs.allSatisfy({ $0.length == out.length }) else {
                    throw GemmaSSAError.badDataflow("add operand/length mismatch at \(i)")
                }
            case "attention":
                guard inputs.count == 2,
                      inputs[0].kind == .vector, inputs[1].kind == .kvstate,
                      inputs[1].name == "kvstate_\(inst.layer ?? -1)",
                      let headDim = inst.attrs["headDim"],
                      let kvHeads = inst.attrs["kvHeads"],
                      let heads = inst.attrs["heads"],
                      headDim > 0, kvHeads > 0, heads > 0,
                      let qLen = checkedMul(heads, headDim),
                      let kvLen = checkedMul(kvHeads, headDim),
                      inputs[0].length == qLen,
                      inputs[1].length == kvLen,
                      out.length == qLen else {
                    throw GemmaSSAError.badDataflow("attention operands invalid at \(i)")
                }
            case "cache_append":
                guard inputs.count == 2,
                      inputs[0].length == inputs[1].length,
                      let kvHeads = inst.attrs["kvHeads"],
                      let headDim = inst.attrs["headDim"],
                      kvHeads > 0, headDim > 0,
                      let kvLen = checkedMul(kvHeads, headDim),
                      inputs[0].length == kvLen,
                      out.kind == .kvstate, out.name == "kvstate_\(inst.layer ?? -1)",
                      out.length == inputs[0].length else {
                    throw GemmaSSAError.badDataflow("cache_append operands invalid at \(i)")
                }
            case "router_topk":
                guard inputs.count == 1, inputs[0].length == inst.attrs["experts"],
                      out.kind == .routing, out.length == inst.attrs["topK"] else {
                    throw GemmaSSAError.badDataflow("router_topk operands invalid at \(i)")
                }
            case "expert_matvec_chain":
                guard inputs.count == 2, inputs[1].kind == .routing,
                      inputs[1].name == "rout_\(inst.layer ?? -1)",
                      inputs[1].length == inst.attrs["experts"],
                      inputs[0].length == c.hidden, out.length == c.hidden,
                      inst.tensors.count == 2 else {
                    throw GemmaSSAError.badDataflow("expert chain operands invalid at \(i)")
                }
            default:
                throw GemmaSSAError.badDataflow("unknown ssa op \(inst.op) at \(i)")
            }
        }
        guard let final = ssa.instructions.last,
              final.op == "softcap",
              final.output.hasPrefix("logits@") else {
            throw GemmaSSAError.badDataflow("schedule does not end at logits")
        }
    }

    /// Canonical digest of the lowered schedule (a stable identity input
    /// for the plan artifact). Serialization failure is an error, never
    /// a silent empty digest.
    public static func digest(_ ssa: GemmaSSAIR) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(ssa)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }
}
