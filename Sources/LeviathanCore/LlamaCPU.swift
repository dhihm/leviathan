import Foundation

public enum LlamaCPUError: Error {
    case unsupported(String)
    case invalid(String)
    case cancelled
}

/// Scalar reference inference for dense, bias-free Llama GGUF models with
/// standard RoPE and F32/F16 weights. Matrices follow GGML dimension order.
public struct LlamaCPU {
    private struct Layer {
        let attentionNorm, query, key, value, attentionOutput: [Float]
        let ffnNorm, gate, up, down: [Float]
    }

    private let width, heads, kvHeads, headWidth, context: Int
    private let epsilon, ropeBase: Float
    private let embedding, outputNorm, output: [Float]
    private let layers: [Layer]
    private let vocabulary: Int

    public init(file: GGUFFile) throws {
        guard try file.metadataString("general.architecture") == "llama" else {
            throw LlamaCPUError.unsupported("expected Llama GGUF")
        }
        func dimension(_ key: String, limit: Int) throws -> Int {
            let value = try file.metadataUInt("llama." + key)
            guard value > 0, value <= UInt64(limit) else { throw LlamaCPUError.invalid(key) }
            return Int(value)
        }
        func number(_ key: String, default fallback: Float) throws -> Float {
            switch file.metadata["llama." + key] {
            case .none: return fallback
            case let .float32(value) where value.isFinite && value > 0: return value
            default: throw LlamaCPUError.unsupported(key)
            }
        }

        width = try dimension("embedding_length", limit: 8192)
        heads = try dimension("attention.head_count", limit: 256)
        kvHeads = file.metadata["llama.attention.head_count_kv"] == nil
            ? heads : try dimension("attention.head_count_kv", limit: 256)
        context = try dimension("context_length", limit: 1 << 20)
        let count = try dimension("block_count", limit: 64)
        let hidden = try dimension("feed_forward_length", limit: 32768)
        headWidth = width / heads
        guard width % heads == 0, heads % kvHeads == 0,
              headWidth > 0, headWidth % 2 == 0 else {
            throw LlamaCPUError.invalid("attention geometry")
        }
        for key in ["rope.dimension_count", "attention.key_length", "attention.value_length"] {
            if file.metadata["llama." + key] != nil,
               try dimension(key, limit: width) != headWidth {
                throw LlamaCPUError.unsupported(key)
            }
        }
        guard file.metadata["llama.rope.scaling.type"] == nil
                || file.metadata["llama.rope.scaling.type"] == .string("none"),
              file.metadata["llama.rope.scale_linear"] == nil,
              file.metadata["llama.rope.scaling.factor"] == nil
                || file.metadata["llama.rope.scaling.factor"] == .float32(1),
              file.metadata["llama.attention.sliding_window"] == nil else {
            throw LlamaCPUError.unsupported("scaled RoPE or sliding attention")
        }
        epsilon = try number("attention.layer_norm_rms_epsilon", default: 1e-5)
        ropeBase = try number("rope.freq_base", default: 10000)

        let shape = try file.tensor("token_embd.weight").dims
        guard shape.count == 2, shape[1] <= 262144 else {
            throw LlamaCPUError.invalid("token embedding shape")
        }
        vocabulary = shape[1]
        var seen = Set<String>()
        let decodedLimitMiB = 520
        var decodedBytes = 0
        func load(_ name: String, _ dims: [Int]) throws -> [Float] {
            let tensor = try file.tensor(name)
            guard tensor.dims == dims else { throw LlamaCPUError.invalid("shape of " + name) }
            guard tensor.ggmlType == 0 || tensor.ggmlType == 1 else {
                throw LlamaCPUError.unsupported("expected F32/F16: " + name)
            }
            let count = dims.reduce(1, *)
            // The scalar engine stores decoded weights as Float32.
            guard count <= ((decodedLimitMiB << 20) - decodedBytes) / 4 else {
                throw LlamaCPUError.unsupported("CPU weight limit is \(decodedLimitMiB) MiB decoded")
            }
            decodedBytes += count * 4
            seen.insert(name)
            let data = try file.tensorData(tensor)
            let values = data.withUnsafeBytes { bytes -> [Float] in
                if tensor.ggmlType == 0 {
                    return (0..<count).map {
                        Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(
                            fromByteOffset: $0 * 4, as: UInt32.self)))
                    }
                }
                return (0..<count).map {
                    Float(Float16(bitPattern: UInt16(littleEndian: bytes.loadUnaligned(
                        fromByteOffset: $0 * 2, as: UInt16.self))))
                }
            }
            guard values.allSatisfy(\.isFinite) else { throw LlamaCPUError.invalid(name) }
            return values
        }

        let width = self.width
        let kvWidth = kvHeads * headWidth
        embedding = try load("token_embd.weight", [width, vocabulary])
        outputNorm = try load("output_norm.weight", [width])
        if file.tensors.contains(where: { $0.name == "output.weight" }) {
            output = try load("output.weight", [width, vocabulary])
        } else {
            output = embedding
        }
        layers = try (0..<count).map { index in
            let prefix = "blk.\(index)."
            return try Layer(
                attentionNorm: load(prefix + "attn_norm.weight", [width]),
                query: load(prefix + "attn_q.weight", [width, width]),
                key: load(prefix + "attn_k.weight", [width, kvWidth]),
                value: load(prefix + "attn_v.weight", [width, kvWidth]),
                attentionOutput: load(prefix + "attn_output.weight", [width, width]),
                ffnNorm: load(prefix + "ffn_norm.weight", [width]),
                gate: load(prefix + "ffn_gate.weight", [width, hidden]),
                up: load(prefix + "ffn_up.weight", [width, hidden]),
                down: load(prefix + "ffn_down.weight", [hidden, width]))
        }
        if let extra = file.tensors.first(where: { !seen.contains($0.name) }) {
            throw LlamaCPUError.unsupported("unexpected tensor " + extra.name)
        }
    }

    private func project(_ x: [Float], _ weights: [Float]) -> [Float] {
        (0..<(weights.count / x.count)).map { row in
            let offset = row * x.count
            var sum: Float = 0
            for i in x.indices { sum += x[i] * weights[offset + i] }
            return sum
        }
    }

    private func normalize(_ x: [Float], _ weights: [Float]) -> [Float] {
        let meanSquare = x.reduce(Float(0)) { $0 + $1 * $1 } / Float(x.count)
        let scale = 1 / sqrt(meanSquare + epsilon)
        return zip(x, weights).map { $0 * scale * $1 }
    }

    private func rotate(_ x: inout [Float], position: Int) {
        let half = headWidth / 2
        for i in 0..<half {
            let angle = Float(position) / pow(ropeBase, Float(i) / Float(half))
            let c = cos(angle), s = sin(angle)
            for start in stride(from: 0, to: x.count, by: headWidth) {
                let a = start + 2 * i, b = a + 1
                let first = x[a], second = x[b]
                x[a] = first * c - second * s
                x[b] = first * s + second * c
            }
        }
    }

    private struct Cache {
        var keys: [[Float]] = []
        var values: [[Float]] = []
    }

    private func attention(_ query: [Float], cache: Cache) -> [Float] {
        let scale = 1 / sqrt(Float(headWidth))
        var result = [Float](repeating: 0, count: width)
        for head in 0..<heads {
            let q = head * headWidth
            let kv = (head / (heads / kvHeads)) * headWidth
            let scores = cache.keys.map { key -> Float in
                var dot: Float = 0
                for i in 0..<headWidth { dot += query[q + i] * key[kv + i] }
                return dot * scale
            }
            let peak = scores.max()!
            let weights = scores.map { exp($0 - peak) }
            let total = weights.reduce(0, +)
            for t in weights.indices {
                let weight = weights[t] / total
                for i in 0..<headWidth { result[q + i] += weight * cache.values[t][kv + i] }
            }
        }
        return result
    }

    public struct GenerationTiming {
        public fileprivate(set) var setupMillis = 0.0
        public fileprivate(set) var prefillMillis = 0.0
        public fileprivate(set) var firstTokenMillis = 0.0
        public fileprivate(set) var decodeMillis: [Double] = []
    }

    /// Returns at most `count` continuation IDs, including EOS when emitted.
    public func generate(prompt: [Int], count: Int, eos: Int? = nil) throws -> [Int] {
        try generate(prompt: prompt, count: count, eos: eos, timing: nil)
    }

    public func generate(prompt: [Int], count: Int, eos: Int? = nil,
                         timing: ((GenerationTiming) -> Void)?,
                         cancelAfterTokens: Int? = nil,
                         onToken: ((Int, Int) throws -> Void)? = nil) throws -> [Int] {
        let setupStart = timing == nil ? 0 : ProcessInfo.processInfo.systemUptime
        let limit = min(context, 512)
        guard count > 0, count <= 128, !prompt.isEmpty, prompt.count <= limit - count + 1,
              prompt.allSatisfy({ (0..<vocabulary).contains($0) }),
              eos.map({ (0..<vocabulary).contains($0) }) ?? true else {
            throw LlamaCPUError.invalid("token IDs, generation count or context length")
        }
        let steps = prompt.count + count - 1
        let cacheBytes = steps * layers.count * kvHeads * headWidth * 2 * MemoryLayout<Float>.stride
        guard cacheBytes <= 128 << 20 else {
            throw LlamaCPUError.unsupported("requested context exceeds the 128 MiB KV cache limit")
        }
        var caches = [Cache](repeating: Cache(), count: layers.count)
        var generated: [Int] = []
        var measurements = GenerationTiming()
        let prefillStart = timing == nil ? 0 : ProcessInfo.processInfo.systemUptime
        if timing != nil { measurements.setupMillis = (prefillStart - setupStart) * 1000 }
        for position in 0..<steps {
            let decodeStart = timing != nil && position >= prompt.count
                ? ProcessInfo.processInfo.systemUptime : 0
            let token = position < prompt.count ? prompt[position] : generated[generated.count - 1]
            var x = Array(embedding[(token * width)..<((token + 1) * width)])
            for (index, layer) in layers.enumerated() {
                let input = normalize(x, layer.attentionNorm)
                var query = project(input, layer.query)
                var key = project(input, layer.key)
                rotate(&query, position: position)
                rotate(&key, position: position)
                caches[index].keys.append(key)
                caches[index].values.append(project(input, layer.value))
                let mixed = project(attention(query, cache: caches[index]), layer.attentionOutput)
                for i in x.indices { x[i] += mixed[i] }
                let normalized = normalize(x, layer.ffnNorm)
                let gated = zip(project(normalized, layer.gate), project(normalized, layer.up))
                    .map { ($0 / (1 + exp(-$0))) * $1 }
                let delta = project(gated, layer.down)
                for i in x.indices { x[i] += delta[i] }
            }
            guard position >= prompt.count - 1 else { continue }
            var firstStart = 0.0
            if timing != nil && position == prompt.count - 1 {
                firstStart = ProcessInfo.processInfo.systemUptime
                measurements.prefillMillis = (firstStart - prefillStart) * 1000
            }
            let logits = project(normalize(x, outputNorm), output)
            guard logits.allSatisfy(\.isFinite) else { throw LlamaCPUError.invalid("non-finite logits") }
            let next = logits.indices.max { logits[$0] < logits[$1] }!
            generated.append(next)
            if timing != nil {
                let ended = ProcessInfo.processInfo.systemUptime
                if position == prompt.count - 1 {
                    measurements.firstTokenMillis = (ended - firstStart) * 1000
                } else {
                    measurements.decodeMillis.append((ended - decodeStart) * 1000)
                }
            }
            try onToken?(next, generated.count - 1)
            if generated.count == cancelAfterTokens { throw LlamaCPUError.cancelled }
            if next == eos { break }
        }
        timing?(measurements)
        return generated
    }
}
