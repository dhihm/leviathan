// Gemma graph capture IR: the restricted op set a captured model.py may
// express, validated against the REAL GGUF (config + 658-tensor contract)
// before anything executes. This is the model input path for the Gemma
// Forge pipeline — unknown ops, wrong sequences, or missing tensor
// bindings are rejected here.

import Foundation

public enum GemmaGraphError: Error, CustomStringConvertible, Equatable {
    case badJSON(String)
    case unsupportedOp(String)
    case configMismatch(String)
    case badSequence(String)
    case missingTensor(String)

    public var description: String {
        switch self {
        case .badJSON(let m): return "graph JSON invalid: \(m)"
        case .unsupportedOp(let o): return "unsupported graph op: \(o)"
        case .configMismatch(let m): return "graph config mismatch: \(m)"
        case .badSequence(let m): return "graph op sequence invalid: \(m)"
        case .missingTensor(let t): return "graph references missing tensor: \(t)"
        }
    }
}

/// The captured graph as Codable IR. `ops` is the exact execution order.
public struct GemmaGraphIR: Codable, Equatable, Sendable {
    public struct GraphConfig: Codable, Equatable, Sendable {
        public let layerCount: Int
        public let hidden: Int
        public let qHeads: Int
        public let localKVHeads: Int
        public let globalKVHeads: Int
        public let localHeadDim: Int
        public let globalHeadDim: Int
        public let slidingWindow: Int
        public let vocabSize: Int
        public let expertCount: Int
        public let expertUsedCount: Int
        public let expertIntermediate: Int
        public let sharedIntermediate: Int
        public let finalSoftcap: Float
        public let globalLayers: [Int]

        public init(layerCount: Int, hidden: Int, qHeads: Int,
                    localKVHeads: Int, globalKVHeads: Int, localHeadDim: Int,
                    globalHeadDim: Int, slidingWindow: Int, vocabSize: Int,
                    expertCount: Int, expertUsedCount: Int, expertIntermediate: Int,
                    sharedIntermediate: Int, finalSoftcap: Float,
                    globalLayers: [Int]) {
            self.layerCount = layerCount; self.hidden = hidden
            self.qHeads = qHeads; self.localKVHeads = localKVHeads
            self.globalKVHeads = globalKVHeads; self.localHeadDim = localHeadDim
            self.globalHeadDim = globalHeadDim
            self.slidingWindow = slidingWindow; self.vocabSize = vocabSize
            self.expertCount = expertCount; self.expertUsedCount = expertUsedCount
            self.expertIntermediate = expertIntermediate
            self.sharedIntermediate = sharedIntermediate
            self.finalSoftcap = finalSoftcap; self.globalLayers = globalLayers
        }
    }

    public struct Op: Codable, Equatable, Sendable {
        public let op: String
        public let layer: Int?
        public let tensor: String?
        public let global: Bool?
        public let topK: Int?
        public let gateUp: String?
        public let down: String?
        public let cap: Float?

        public init(op: String, layer: Int? = nil, tensor: String? = nil,
                    global: Bool? = nil, topK: Int? = nil, gateUp: String? = nil,
                    down: String? = nil, cap: Float? = nil) {
            self.op = op; self.layer = layer; self.tensor = tensor
            self.global = global; self.topK = topK; self.gateUp = gateUp
            self.down = down; self.cap = cap
        }
    }

    public let schema: Int
    public let model: String
    public let config: GraphConfig
    public let ops: [Op]

    public init(schema: Int, model: String, config: GraphConfig, ops: [Op]) {
        self.schema = schema; self.model = model; self.config = config
        self.ops = ops
    }
}

public enum GemmaGraph {

    public static let supportedOps: Set<String> = [
        "embed", "attn_norm", "q_proj", "q_norm", "q_rope", "k_proj",
        "v_eq_k", "v_proj", "k_norm", "k_rope", "v_norm", "cache_append",
        "attention", "o_proj", "post_attn_norm", "residual_add", "ffn_norm",
        "shared_gate", "shared_up", "shared_gelu", "shared_down", "shared_post1",
        "router_norm", "router_scale", "router_proj", "router_topk", "moe_norm2",
        "expert_loop", "moe_post2", "combine", "post_ffw_norm", "layer_scale",
        "final_norm", "lm_head", "softcap",
    ]

    /// Parses bounded graph JSON (input is untrusted user output).
    public static func parse(_ data: Data) throws -> GemmaGraphIR {
        guard data.count <= 8 << 20 else {
            throw GemmaGraphError.badJSON("graph exceeds 8 MiB")
        }
        do {
            return try JSONDecoder().decode(GemmaGraphIR.self, from: data)
        } catch {
            throw GemmaGraphError.badJSON("\(error)")
        }
    }

    /// The exact expected op sequence for a config — the graph is legal
    /// only if it matches this sequence op-for-op (same type, layer,
    /// tensor binding, and flags). Anything else is rejected.
    public static func expectedSequence(_ c: GemmaGraphIR.GraphConfig) -> [GemmaGraphIR.Op] {
        var ops = [GemmaGraphIR.Op]()
        func add(_ op: String, _ layer: Int? = nil, _ tensor: String? = nil,
                 _ global: Bool? = nil, _ topK: Int? = nil,
                 _ gateUp: String? = nil, _ down: String? = nil,
                 _ cap: Float? = nil) {
            ops.append(GemmaGraphIR.Op(op: op, layer: layer, tensor: tensor,
                                       global: global, topK: topK,
                                       gateUp: gateUp, down: down, cap: cap))
        }
        add("embed", nil, "token_embd.weight")
        for layer in 0..<c.layerCount {
            let isGlobal = c.globalLayers.contains(layer)
            let p = "blk.\(layer)."
            add("attn_norm", layer, p + "attn_norm.weight")
            add("q_proj", layer, p + "attn_q.weight")
            add("q_norm", layer, p + "attn_q_norm.weight")
            add("q_rope", layer, nil, isGlobal)
            add("k_proj", layer, p + "attn_k.weight")
            if isGlobal { add("v_eq_k", layer) }
            else { add("v_proj", layer, p + "attn_v.weight") }
            add("k_norm", layer, p + "attn_k_norm.weight")
            add("k_rope", layer, nil, isGlobal)
            add("v_norm", layer)
            add("cache_append", layer, nil, isGlobal)
            add("attention", layer, nil, isGlobal)
            add("o_proj", layer, p + "attn_output.weight")
            add("post_attn_norm", layer, p + "post_attention_norm.weight")
            add("residual_add", layer)
            add("ffn_norm", layer, p + "ffn_norm.weight")
            add("shared_gate", layer, p + "ffn_gate.weight")
            add("shared_up", layer, p + "ffn_up.weight")
            add("shared_gelu", layer)
            add("shared_down", layer, p + "ffn_down.weight")
            add("shared_post1", layer, p + "post_ffw_norm_1.weight")
            add("router_norm", layer)
            add("router_scale", layer)
            add("router_proj", layer, p + "ffn_gate_inp.weight")
            add("router_topk", layer, nil, nil, c.expertUsedCount)
            add("moe_norm2", layer, p + "pre_ffw_norm_2.weight")
            add("expert_loop", layer, nil, nil, nil,
                p + "ffn_gate_up_exps.weight", p + "ffn_down_exps.weight")
            add("moe_post2", layer, p + "post_ffw_norm_2.weight")
            add("combine", layer)
            add("post_ffw_norm", layer, p + "post_ffw_norm.weight")
            add("residual_add", layer)
            add("layer_scale", layer)
        }
        add("final_norm", nil, "output_norm.weight")
        add("lm_head", nil, "token_embd.weight")
        add("softcap", nil, nil, nil, nil, nil, nil, c.finalSoftcap)
        return ops
    }

    /// Full validation against the real GGUF: schema, model tag, config
    /// cross-checked with the file's parsed architecture config, op
    /// vocabulary, exact sequence, and every bound tensor present.
    public static func validate(_ graph: GemmaGraphIR, config: Gemma4Config,
                                file: GGUFFile) throws {
        guard graph.schema == 1 else {
            throw GemmaGraphError.badJSON("schema \(graph.schema) != 1")
        }
        guard graph.model == "gemma4-26b-a4b" else {
            throw GemmaGraphError.badJSON("unknown model tag \(graph.model)")
        }
        let c = graph.config
        guard c.layerCount == config.layerCount, c.hidden == config.hiddenSize,
              c.qHeads == config.qHeads, c.localKVHeads == config.localKVHeads,
              c.globalKVHeads == config.globalKVHeads,
              c.localHeadDim == config.localHeadDim,
              c.globalHeadDim == config.globalHeadDim,
              c.slidingWindow == config.slidingWindow,
              c.vocabSize == config.vocabSize,
              c.expertCount == config.expertCount,
              c.expertUsedCount == config.topKExperts,
              c.expertIntermediate == config.expertIntermediate,
              c.sharedIntermediate == config.sharedIntermediate,
              abs(c.finalSoftcap - config.finalSoftcap) < 1e-6,
              c.globalLayers == config.globalLayers.sorted() else {
            throw GemmaGraphError.configMismatch(
                "graph config does not match the GGUF architecture")
        }
        let expected = expectedSequence(c)
        guard graph.ops.count == expected.count else {
            throw GemmaGraphError.badSequence(
                "op count \(graph.ops.count) != expected \(expected.count)")
        }
        // Structural pass FIRST, in full: every op must be supported and
        // match the expected sequence exactly (type, layer, tensor
        // binding, flags) — only then are bound tensors checked for
        // existence, so a structural mutation anywhere in the graph is
        // rejected as such, not masked by an earlier tensor miss.
        for (i, (got, want)) in zip(graph.ops, expected).enumerated() {
            guard supportedOps.contains(got.op) else {
                throw GemmaGraphError.unsupportedOp(got.op)
            }
            guard got == want else {
                throw GemmaGraphError.badSequence(
                    "op \(i) \(got.op) does not match expected \(want.op)")
            }
        }
        for got in graph.ops {
            if let t = got.tensor {
                guard file.tensor(named: t) != nil else {
                    throw GemmaGraphError.missingTensor(t)
                }
            }
            if let g = got.gateUp {
                guard file.tensor(named: g) != nil else {
                    throw GemmaGraphError.missingTensor(g)
                }
            }
            if let d = got.down {
                guard file.tensor(named: d) != nil else {
                    throw GemmaGraphError.missingTensor(d)
                }
            }
        }
    }

    /// Canonical digest of the validated graph.
    public static func digest(_ graph: GemmaGraphIR) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(graph)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }
}

extension GGUFFile {
    func tensor(named name: String) -> GGUFTensorInfo? {
        tensors.first { $0.name == name }
    }
}
