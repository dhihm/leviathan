import Foundation
import Metal

/// Errors from the Gemma 4 (26B-A4B, QAT Q4_0) original runtime.
public enum Gemma4Error: Error, Equatable, Sendable {
    case missingMetadata(String)
    case badMetadata(String)
    case badTensorGeometry(String)
    case incompatibleModel(String)
    case metalUnavailable
    case shaderCompileFailed(String)
    case functionMissing(String)
    case commandBufferFailed(String)
    case nonFiniteActivation(String)
    case exceedsMaxBufferLength(needed: Int, max: Int)
    case mmapFailed(errno: Int32)
    case badTokenId(Int)
    case contextTooLong(Int, max: Int)
}

/// Architecture facts for the exact downloaded GGUF, parsed from its own
/// metadata and cross-checked against the full tensor directory. Nothing is
/// assumed from generic Llama layouts.
public struct Gemma4Config: Sendable {
    public let hiddenSize = 2816
    public let layerCount = 30
    public let qHeads = 16
    public let localKVHeads = 8
    public let globalKVHeads = 2
    public let localHeadDim = 256
    public let globalHeadDim = 512
    public let slidingWindow = 1024
    public let vocabSize = 262_144
    public let expertCount = 128
    public let topKExperts = 8
    public let expertIntermediate = 704
    public let sharedIntermediate = 2112
    public let rmsNormEps: Float = 1e-6
    public let finalSoftcap: Float = 30
    public let localRopeTheta: Double = 10_000
    public let globalRopeTheta: Double = 1_000_000
    /// Global layers rotate a 0.25 proportion of head_dim with zero
    /// frequencies for the rest ("proportional" RoPE).
    public let globalPartialRotaryFactor = 0.25
    /// Layers using full attention with head_dim 512 and k_eq_v (5, 11, 17,
    /// 23, 29 in this checkpoint).
    public let globalLayers: [Int]
    public let maxContext: Int

    public func isGlobalLayer(_ i: Int) -> Bool { globalLayers.contains(i) }

    /// Global-layer effective inverse frequencies. The checkpoint ships a
    /// `rope_freqs.weight` factor table (first 64 entries 1.0, the rest
    /// ~1e30): the effective frequency for pair i is
    /// `theta^(-2i/512) / rf[i]`, which zeroes rotation beyond the 0.25
    /// partial-rotary cut, matching transformers' proportional RoPE zeros.
    public func globalInvFreq(factors: [Float]? = nil) -> [Float] {
        let half = globalHeadDim / 2
        var freqs = (0..<half).map {
            Float(pow(globalRopeTheta, -Double(2 * $0) / Double(globalHeadDim)))
        }
        if let factors {
            for i in 0..<half { freqs[i] /= factors[i] }
        } else {
            let ropeAngles = Int(Double(globalPartialRotaryFactor) * Double(globalHeadDim) / 2)
            for i in ropeAngles..<half { freqs[i] = 0 }
        }
        return freqs
    }

    /// Local-layer inverse frequencies: theta 1e4 over head_dim 256.
    public func localInvFreq() -> [Float] {
        let half = localHeadDim / 2
        return (0..<half).map {
            Float(pow(localRopeTheta, -Double(2 * $0) / Double(localHeadDim)))
        }
    }

    /// Parses the architecture keys from the GGUF metadata. The
    /// global/local classification comes from the per-layer KV-head-count
    /// array and is cross-checked against tensor geometry in
    /// `TensorContract.validate`.
    public init(file: GGUFFile) throws {
        guard file.metadata["general.architecture"] == .string("gemma4") else {
            throw Gemma4Error.incompatibleModel("general.architecture is not gemma4")
        }
        guard case let .uint32(blocks)? = file.metadata["gemma4.block_count"],
              blocks == layerCount else {
            throw Gemma4Error.badMetadata("gemma4.block_count")
        }
        guard case let .uint32(hidden)? = file.metadata["gemma4.embedding_length"],
              hidden == hiddenSize else {
            throw Gemma4Error.badMetadata("gemma4.embedding_length")
        }
        guard case let .uint32(experts)? = file.metadata["gemma4.expert_count"],
              experts == expertCount,
              case let .uint32(used)? = file.metadata["gemma4.expert_used_count"],
              used == topKExperts,
              case let .uint32(eInter)? = file.metadata["gemma4.expert_feed_forward_length"],
              eInter == expertIntermediate,
              case let .uint32(sInter)? = file.metadata["gemma4.feed_forward_length"],
              sInter == sharedIntermediate,
              case let .float32(softcap)? = file.metadata["gemma4.final_logit_softcapping"],
              softcap == finalSoftcap else {
            throw Gemma4Error.badMetadata("MoE/softcap metadata mismatch")
        }
        guard case let .array(elementType, count, uints, ints, _, _, _)? =
                file.metadata["gemma4.attention.head_count_kv"],
              count == layerCount else {
            throw Gemma4Error.badMetadata("gemma4.attention.head_count_kv array")
        }
        // GGUF array readers keep uint32 elements in `uints`, int32 in
        // `ints`; accept the signed spelling (this checkpoint uses int32).
        let kvRaw: [Int]
        if !uints.isEmpty {
            kvRaw = uints.map(Int.init)
        } else if elementType == .int32 || elementType == .int64, ints.count == count {
            kvRaw = ints.map(Int.init)
        } else {
            throw Gemma4Error.badMetadata("head_count_kv array is not integer-typed")
        }
        var global: [Int] = []
        for (i, raw) in kvRaw.enumerated() {
            let kv = Int(raw)
            if kv == globalKVHeads {
                global.append(i)
            } else if kv != localKVHeads {
                throw Gemma4Error.badMetadata("layer \(i) has \(kv) kv heads")
            }
        }
        guard global == [5, 11, 17, 23, 29] else {
            throw Gemma4Error.incompatibleModel("global layers are \(global)")
        }
        self.globalLayers = global
        guard case let .uint32(ctx)? = file.metadata["gemma4.context_length"] else {
            throw Gemma4Error.badMetadata("gemma4.context_length")
        }
        self.maxContext = min(Int(ctx), 4096)  // bounded first-milestone context
    }
}

/// Full 658-tensor contract validation, run against the parsed GGUF before
/// any GPU allocation or dispatch. Every expected tensor name, type, and
/// dimension list is checked exactly; unknown or missing tensors are
/// errors. A crafted GGUF with wrong geometry is rejected here rather than
/// becoming an out-of-bounds GPU read later.
public enum Gemma4TensorContract {

    /// The exact expected tensor table (name, dims, type) for the config;
    /// public so tests can verify the contract itself.
    public struct ExpectedTensor: Sendable, Equatable {
        public let name: String
        public let dims: [Int]
        public let type: GGMLTensorType
    }

    public static func expectations(config: Gemma4Config) -> [ExpectedTensor] {
        var e: [ExpectedTensor] = [
            ExpectedTensor(name: "token_embd.weight",
                        dims: [config.hiddenSize, config.vocabSize], type: .q6_K),
            ExpectedTensor(name: "output_norm.weight", dims: [config.hiddenSize], type: .float32),
            ExpectedTensor(name: "rope_freqs.weight",
                        dims: [config.globalHeadDim / 2], type: .float32),
        ]
        for layer in 0..<config.layerCount {
            let g = config.isGlobalLayer(layer)
            let headDim = g ? config.globalHeadDim : config.localHeadDim
            let kvWidth = (g ? config.globalKVHeads : config.localKVHeads) * headDim
            let qWidth = config.qHeads * headDim
            let p = "blk.\(layer)."
            e.append(ExpectedTensor(name: p + "attn_norm.weight",
                                 dims: [config.hiddenSize], type: .float32))
            e.append(ExpectedTensor(name: p + "attn_q.weight",
                                 dims: [config.hiddenSize, qWidth], type: .q4_0))
            e.append(ExpectedTensor(name: p + "attn_k.weight",
                                 dims: [config.hiddenSize, kvWidth], type: .q4_0))
            if !g {
                e.append(ExpectedTensor(name: p + "attn_v.weight",
                                     dims: [config.hiddenSize, kvWidth], type: .q4_0))
            }
            e.append(ExpectedTensor(name: p + "attn_q_norm.weight", dims: [headDim], type: .float32))
            e.append(ExpectedTensor(name: p + "attn_k_norm.weight", dims: [headDim], type: .float32))
            e.append(ExpectedTensor(name: p + "attn_output.weight",
                                 dims: [qWidth, config.hiddenSize], type: .q4_0))
            for n in ["post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight",
                      "post_ffw_norm_1.weight", "pre_ffw_norm_2.weight",
                      "post_ffw_norm_2.weight"] {
                e.append(ExpectedTensor(name: p + n, dims: [config.hiddenSize], type: .float32))
            }
            e.append(ExpectedTensor(name: p + "ffn_gate.weight",
                                 dims: [config.hiddenSize, config.sharedIntermediate],
                                 type: .q4_0))
            e.append(ExpectedTensor(name: p + "ffn_up.weight",
                                 dims: [config.hiddenSize, config.sharedIntermediate],
                                 type: .q4_0))
            e.append(ExpectedTensor(name: p + "ffn_down.weight",
                                 dims: [config.sharedIntermediate, config.hiddenSize],
                                 type: .q4_0))
            e.append(ExpectedTensor(name: p + "ffn_gate_inp.weight",
                                 dims: [config.hiddenSize, config.expertCount], type: .float32))
            e.append(ExpectedTensor(name: p + "ffn_gate_inp.scale",
                                 dims: [config.hiddenSize], type: .float32))
            e.append(ExpectedTensor(name: p + "ffn_gate_up_exps.weight",
                                 dims: [config.hiddenSize, 2 * config.expertIntermediate,
                                        config.expertCount],
                                 type: .q4_0))
            e.append(ExpectedTensor(name: p + "ffn_down_exps.weight",
                                 dims: [config.expertIntermediate, config.hiddenSize,
                                        config.expertCount],
                                 type: .q4_0))
            e.append(ExpectedTensor(name: p + "ffn_down_exps.scale",
                                 dims: [config.expertCount], type: .float32))
            e.append(ExpectedTensor(name: p + "layer_output_scale.weight",
                                 dims: [1], type: .float32))
        }
        return e
    }

    /// Validates the complete tensor directory: exact name set, exact dims,
    /// exact types, and in-file byte bounds (already enforced by
    /// `GGUFReader.parse`, re-stated here for the contract).
    public static func validate(file: GGUFFile, config: Gemma4Config) throws {
        let expected = expectations(config: config)
        let expectedNames = Set(expected.map(\.name))
        let actualNames = Set(file.tensors.map(\.name))
        let missing = expectedNames.subtracting(actualNames).sorted()
        let extra = actualNames.subtracting(expectedNames).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            throw Gemma4Error.incompatibleModel(
                "tensor set mismatch: missing \(missing.prefix(4))…, extra \(extra.prefix(4))…")
        }
        for exp in expected {
            guard let info = file.tensors.first(where: { $0.name == exp.name }) else {
                throw Gemma4Error.incompatibleModel("tensor \(exp.name) vanished")
            }
            guard info.dims == exp.dims else {
                throw Gemma4Error.badTensorGeometry(
                    "\(exp.name): dims \(info.dims), expected \(exp.dims)")
            }
            guard let type = GGMLTensorType(rawValue: info.ggmlType), type == exp.type else {
                throw Gemma4Error.badTensorGeometry(
                    "\(exp.name): type \(info.ggmlType), expected \(exp.type)")
            }
            _ = try file.tensorRange(info)   // bounds check (parse already did)
        }
    }
}

/// Evidence from one full-model GPU forward on the real 14.44 GB checkpoint.
/// All counters are deltas for this forward only; footprint is sampled at
/// token boundaries (not a kernel-level trace).
public struct Gemma4RunEvidence: Sendable {
    public let deviceName: String
    public let deviceMaxBufferLength: Int
    public let recommendedMaxWorkingSetSize: Int
    /// Bytes of quantized weights mapped read-only into the GPU address
    /// space with no copy (the page-rounded whole file).
    public let weightsMappedBytes: Int
    public let metalAllocatedBytes: Int
    public let tokensProcessed: Int
    public let contextLength: Int
    public let dispatchedKernels: Int
    public let gpuMicros: Double
    /// Maximum sampled physical footprint of this process (bytes).
    public let peakFootprintBytes: Int
    public let argmaxToken: Int
    public let topLogits: [(token: Int, logit: Float)]
}

/// One weight tensor's GPU addressing: absolute byte offsets into the
/// zero-copy mapped file buffer plus its validated geometry.
struct G4TensorRef {
    let offset: Int              // absolute file offset (byte)
    let end: Int                 // absolute file end offset (exclusive)
    let inDim: Int               // ne[0]
    let outDim: Int              // ne[1]
    let experts: Int             // ne[2] or 1
    let type: GGMLTensorType

    var elementsPerSlice: Int { inDim * outDim }
}

/// GPU primitive probes for the Gemma 4 math kernels (used by tests to
/// compare the overflow-safe MSL tanh and friends against scalar
/// references, including the large-argument region where Metal's built-in
/// tanh returns NaN on the tested driver).
public enum Gemma4Math {

    /// Evaluates the RUNTIME's overflow-safe tanh (the exact `g4_tanh`
    /// helper compiled from `Gemma4Runner.shaderSource`, not a duplicate)
    /// on the GPU for each finite argument and returns the results.
    public static func stableTanhGPU(_ args: [Float]) throws -> [Float] {
        guard !args.isEmpty, args.count <= 1 << 20,
              args.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.badTensorGeometry("tanh probe needs 1...2^20 finite args")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw Gemma4Error.metalUnavailable
        }
        // The probe kernel is appended to the REAL runtime shader source so
        // a regression in the runtime helper fails this probe too.
        let source = Gemma4Runner.shaderSource + """

        kernel void g4_tanh_probe(constant const uint& n [[buffer(0)]],
                                  device const float* x [[buffer(1)]],
                                  device float* y [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
            if (gid >= n) { return; }
            y[gid] = g4_tanh(x[gid]);
        }
        """
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw Gemma4Error.shaderCompileFailed("\(error)") }
        guard let fn = library.makeFunction(name: "g4_tanh_probe"),
              let pipeline = try? device.makeComputePipelineState(function: fn),
              let xb = device.makeBuffer(bytes: args, length: args.count * 4,
                                          options: .storageModeShared),
              let yb = device.makeBuffer(length: args.count * 4,
                                          options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Gemma4Error.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        var n = UInt32(args.count)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 0)
        enc.setBuffer(xb, offset: 0, index: 1)
        enc.setBuffer(yb, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: args.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: max(1, min(args.count,
                                                  pipeline.maxTotalThreadsPerThreadgroup)),
                                height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Gemma4Error.commandBufferFailed("tanh probe")
        }
        return [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self),
            count: args.count))
    }
}

/// Persistent-GPU Gemma 4 runtime. The checkpoint is mmap'd read-only once
/// and wrapped in a no-copy shared-storage MTLBuffer; quantized weights are
/// dequantized block-by-block inside the kernels and are NEVER expanded to
/// F32/BF16 in memory. Device, queue, shader library, pipelines, activation
/// buffers, and KV caches are created once and reused across tokens.
///
/// The runner is stateful across `forward` calls (KV caches and the
/// consumed position continue); call `reset()` to start a fresh context.
public final class Gemma4Runner {

    private let config: Gemma4Config
    private let ggufFile: GGUFFile
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var dispatchCountTotal = 0
    private var gpuMicrosTotal: Double = 0

    // Weight storage. The mapping's lifetime is owned by `weightsBuffer`'s
    // deallocator (RAII); no manual munmap outside init failure paths.
    private var mapLength = 0
    private var weightsBuffer: MTLBuffer?
    private var mapEnd = 0    // exclusive absolute offset covered by the buffer

    // Activation buffers (F32, tiny compared with weights).
    private var bufX, bufH, bufT1, bufT2, bufT3, bufKV, bufGU, bufAct: MTLBuffer!
    private var bufShared, bufRouter, bufEmbed, bufLogits: MTLBuffer!
    private var localFreqBuffer, globalFreqBuffer: MTLBuffer!
    // KV caches per layer: [maxContext * kvHeads * headDim] F32 each.
    private var kCaches: [MTLBuffer] = []
    private var vCaches: [MTLBuffer] = []

    // Tensor addressing table.
    private var ref: [String: G4TensorRef] = [:]
    // Small F32 parameters read into CPU arrays at init (finite-checked).
    private var expertScales: [[Float]] = []   // [layer][128] router per-expert scale
    private var layerScalars: [Float] = []     // layer_output_scale
    private var routerScales: [[Float]] = []   // [layer][2816] router input scale

    private let hidden: Int
    private let maxContext: Int
    private var consumedPosition = 0
    private var peakFootprint = 0
    /// Diagnostic: print per-layer residual max |x| (and non-finite counts)
    /// to localize divergence during bring-up.
    public var traceLayers = false

    public init(fileAt url: URL) throws {
        let gguf = try GGUFReader.read(url)
        self.config = try Gemma4Config(file: gguf)
        self.ggufFile = gguf
        self.hidden = config.hiddenSize
        self.maxContext = config.maxContext

        // ---- FULL tensor-contract validation before any GPU resource.
        try Gemma4TensorContract.validate(file: gguf, config: config)
        for info in gguf.tensors {
            let range = try gguf.tensorRange(info)
            guard let type = GGMLTensorType(rawValue: info.ggmlType) else {
                throw Gemma4Error.badTensorGeometry("tensor \(info.name) type \(info.ggmlType)")
            }
            ref[info.name] = G4TensorRef(offset: range.lowerBound, end: range.upperBound,
                                         inDim: info.dims[0],
                                         outDim: info.dims.count > 1 ? info.dims[1] : 1,
                                         experts: info.dims.count > 2 ? info.dims[2] : 1,
                                         type: type)
        }

        // ---- small F32 parameters (finite-checked CPU reads from mmap).
        // The file mapping for these comes second; mmap is established
        // below, so read via a bounded Data slice of the GGUF mapping.
        func readF32FromGGUF(_ name: String, count: Int) throws -> [Float] {
            let info = try gguf.tensor(name)
            let data = try gguf.tensorData(info)
            guard data.count == count * 4 else {
                throw Gemma4Error.badTensorGeometry("\(name): \(data.count) bytes, want \(count * 4)")
            }
            let values = data.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
            guard values.allSatisfy({ $0.isFinite }) else {
                throw Gemma4Error.nonFiniteActivation("\(name) parameter")
            }
            return values
        }
        for layer in 0..<config.layerCount {
            expertScales.append(try readF32FromGGUF("blk.\(layer).ffn_down_exps.scale",
                                                    count: config.expertCount))
            let scal = try readF32FromGGUF("blk.\(layer).layer_output_scale.weight", count: 1)
            layerScalars.append(scal[0])
            routerScales.append(try readF32FromGGUF("blk.\(layer).ffn_gate_inp.scale",
                                                    count: config.hiddenSize))
        }
        // rope_freqs is a divisor factor table, not raw inverse frequencies
        // (measured on the real file: first 64 entries 1.0, the rest ~1e30).
        // Validate its shape/values, then derive the effective global
        // frequencies from it to preserve exact GGUF RoPE behavior.
        let storedFactors = try readF32FromGGUF("rope_freqs.weight",
                                                count: config.globalHeadDim / 2)
        let ropeAngles = Int(Double(config.globalPartialRotaryFactor)
                             * Double(config.globalHeadDim) / 2)
        for (i, f) in storedFactors.enumerated() {
            guard f.isFinite, f > 0 else {
                throw Gemma4Error.incompatibleModel(
                    "rope_freqs[\(i)] not a finite positive factor: \(f)")
            }
            if i < ropeAngles, abs(f - 1) > 1e-6 {
                throw Gemma4Error.incompatibleModel(
                    "rope_freqs[\(i)] \(f) in the active rotary range != 1.0")
            }
            if i >= ropeAngles, f < 1e20 {
                throw Gemma4Error.incompatibleModel(
                    "rope_freqs[\(i)] \(f) does not disable rotation beyond the partial cut")
            }
        }

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw Gemma4Error.metalUnavailable
        }
        self.device = device
        self.queue = queue

        // ---- mmap the whole file read-only for the zero-copy weight buffer.
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Gemma4Error.mmapFailed(errno: errno) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 0,
              st.st_size <= Int(48 << 30) else {
            throw Gemma4Error.mmapFailed(errno: errno)
        }
        let pageSize = GemmaRunner.pageSize
        let (mapLenRaw, of) = Int(st.st_size).addingReportingOverflow(pageSize - 1)
        guard !of else { throw Gemma4Error.mmapFailed(errno: 0) }
        let mapLen = mapLenRaw / pageSize * pageSize
        guard let raw = mmap(nil, mapLen, PROT_READ, MAP_SHARED, fd, 0), raw != MAP_FAILED else {
            throw Gemma4Error.mmapFailed(errno: errno)
        }
        self.mapLength = mapLen
        self.mapEnd = mapLen

        // Preflight: the mapped region must fit within device buffer limits.
        guard mapLen <= device.maxBufferLength else {
            _ = munmap(raw, mapLen)
            throw Gemma4Error.exceedsMaxBufferLength(needed: mapLen, max: device.maxBufferLength)
        }
        // Ownership handoff: the MTLBuffer's deallocator owns the munmap
        // from here on. Before this point failures munmap above; after it,
        // any init failure releases the buffer (and thus the mapping)
        // exactly once — no manual unmap remains.
        guard let wb = device.makeBuffer(bytesNoCopy: raw, length: mapLen,
                                         options: .storageModeShared,
                                         deallocator: { ptr, len in
                                             _ = munmap(ptr, len)
                                         }) else {
            _ = munmap(raw, mapLen)
            throw Gemma4Error.metalUnavailable
        }
        self.weightsBuffer = wb

        // ---- shaders + pipelines (compiled once).
        do { library = try device.makeLibrary(source: Self.shaderSource, options: nil) }
        catch { throw Gemma4Error.shaderCompileFailed("\(error)") }
        for name in ["g4_q4_matvec", "g4_q6_matvec", "g4_f32_matvec", "g4_q6_embed_row",
                     "g4_rmsnorm", "g4_gelu_up", "g4_rope", "g4_attn",
                     "g4_axpy", "g4_add", "g4_scale", "g4_softcap"] {
            guard let fn = library.makeFunction(name: name),
                  let p = try? device.makeComputePipelineState(function: fn) else {
                throw Gemma4Error.functionMissing(name)
            }
            pipelines[name] = p
        }

        // ---- activation buffers. Widths are preflighted for every use:
        // bufT1/bufT3 hold qHeads*globalHeadDim (8192); bufT2 must hold
        // both KV staging (max 2048) AND 2816-wide projections (attn_output,
        // ffn_down, expert down) -> 2816; bufKV holds the wider of the raw
        // k copy / v projection (2048).
        let f4 = MemoryLayout<Float>.stride
        func alloc(_ count: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: count * f4, options: .storageModeShared) else {
                throw Gemma4Error.metalUnavailable
            }
            return b
        }
        let widestKV = max(config.localKVHeads * config.localHeadDim,
                           config.globalKVHeads * config.globalHeadDim)
        bufX = try alloc(hidden)
        bufH = try alloc(hidden)
        bufT1 = try alloc(config.qHeads * config.globalHeadDim)
        bufT2 = try alloc(max(hidden, widestKV))
        bufT3 = try alloc(config.qHeads * config.globalHeadDim)
        bufKV = try alloc(widestKV)
        bufGU = try alloc(2 * config.sharedIntermediate)
        bufAct = try alloc(config.sharedIntermediate)
        bufShared = try alloc(hidden)
        bufRouter = try alloc(config.expertCount)
        bufEmbed = try alloc(hidden)
        bufLogits = try alloc(config.vocabSize)
        localFreqBuffer = try upload(config.localInvFreq())
        globalFreqBuffer = try upload(config.globalInvFreq(factors: storedFactors))
        for layer in 0..<config.layerCount {
            let width = config.isGlobalLayer(layer)
                ? config.globalKVHeads * config.globalHeadDim
                : config.localKVHeads * config.localHeadDim
            kCaches.append(try alloc(maxContext * width))
            vCaches.append(try alloc(maxContext * width))
        }
        peakFootprint = Gemma4Runner.physFootprint()
    }

    fileprivate enum GemmaRunner {
        static let pageSize: Int = {
            var size: Int = 0
            var len = MemoryLayout<Int>.size
            sysctlbyname("hw.pagesize", &size, &len, nil, 0)
            return max(size, 4096)
        }()
    }

    private func upload(_ values: [Float]) throws -> MTLBuffer {
        guard let b = device.makeBuffer(bytes: values, length: values.count * 4,
                                        options: .storageModeShared) else {
            throw Gemma4Error.metalUnavailable
        }
        return b
    }

    // MARK: - kernel dispatch

    /// Encodes one kernel in its own command buffer, commits, and waits.
    /// Correct and simple; batching is a later, measured optimization.
    private func dispatch(_ name: String, grid: Int,
                          _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard let pipeline = pipelines[name], let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Gemma4Error.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        bind(enc)
        let width = max(1, min(grid, pipeline.maxTotalThreadsPerThreadgroup))
        let groups = (grid + width - 1) / width
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Gemma4Error.commandBufferFailed(
                "kernel \(name): status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        gpuMicrosTotal += (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
        dispatchCountTotal += 1
    }

    // MARK: - GPU primitives (each preflights byte bounds before dispatch)

    /// y[yOff...] = W·x for a quantized/F32 2-D weight tensor slice
    /// (`rows` rows of `inDim` values, row-major blocks). `extraOffset`
    /// selects an expert slice of a 3-D tensor. Byte bounds are checked
    /// against the tensor's own file extent before anything is encoded.
    private func matvec(tensor: String, extraOffset: Int = 0,
                        x: MTLBuffer, xCount: Int,
                        y: MTLBuffer, yCapacity: Int,
                        rows: Int? = nil,
                        yByteOffset: Int = 0) throws {
        guard let r = ref[tensor] else { throw Gemma4Error.badTensorGeometry("missing \(tensor)") }
        let outRows = rows ?? r.outDim
        guard outRows > 0, r.inDim > 0, yByteOffset >= 0,
              r.inDim % r.type.elementsPerBlock == 0 else {
            throw Gemma4Error.badTensorGeometry("bad dims for \(tensor)")
        }
        // Actual buffer extents (not caller claims): output rows must fit
        // the shifted output binding; the input must carry inDim floats.
        let (yEnd, ofY) = yByteOffset.addingReportingOverflow(outRows * 4)
        guard !ofY, yByteOffset % 256 == 0,
              y.length >= yEnd, yCapacity * 4 >= yEnd,
              x.length >= r.inDim * 4 else {
            throw Gemma4Error.badTensorGeometry(
                "buffer extents violated for \(tensor): y needs \(yEnd) bytes (has \(y.length)), x needs \(r.inDim * 4) (has \(x.length))")
        }
        // Input element count.
        guard xCount == r.inDim else {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor): input \(xCount) != inDim \(r.inDim)")
        }
        // Weight byte bounds within this tensor's file extent.
        let blocksPerRow = r.inDim / r.type.elementsPerBlock
        let (bytesPerRow, of1) = blocksPerRow.multipliedReportingOverflow(by: r.type.bytesPerBlock)
        let (weightBytes, of2) = of1 ? (0, true) : outRows.multipliedReportingOverflow(by: bytesPerRow)
        let (wStart, of3) = r.offset.addingReportingOverflow(extraOffset)
        let (wEnd, of4) = of3 ? (0, true) : wStart.addingReportingOverflow(weightBytes)
        guard !of1, !of2, !of3, !of4, wStart >= r.offset, wEnd <= r.end, wEnd <= mapEnd else {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor)+\(extraOffset): weight bytes [\(wStart), \(wEnd)) beyond tensor extent")
        }
        guard let wb = weightsBuffer else { throw Gemma4Error.metalUnavailable }
        let kernel: String
        switch r.type {
        case .q4_0: kernel = "g4_q4_matvec"
        case .q6_K: kernel = "g4_q6_matvec"
        case .float32: kernel = "g4_f32_matvec"
        case .float16: throw Gemma4Error.badTensorGeometry("F16 unsupported for \(tensor)")
        }
        var p = SIMD2<UInt32>(UInt32(r.inDim), UInt32(outRows))
        var off = UInt64(wStart)
        try dispatch(kernel, grid: outRows) { enc in
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(x, offset: 0, index: 3)
            enc.setBuffer(y, offset: yByteOffset, index: 4)
        }
    }

    /// Per-vector RMSNorm over `vectors` contiguous vectors of length `n`
    /// (1 thread per vector; in-place x==y is safe). `weightTensor` nil =
    /// scale-free (router norm, v_norm).
    private func rmsnorm(x: MTLBuffer, y: MTLBuffer, n: Int, vectors: Int,
                         weightTensor: String?) throws {
        var wOff: UInt64 = 0
        if let name = weightTensor {
            guard let r = ref[name], r.type == .float32,
                  r.inDim == n, r.end - r.offset == n * 4 else {
                throw Gemma4Error.badTensorGeometry("bad norm weight \(name)")
            }
            wOff = UInt64(r.offset)
        }
        var eps = config.rmsNormEps
        var dims = SIMD2<UInt32>(UInt32(n), UInt32(vectors))
        var withScale: Int32 = weightTensor == nil ? 0 : 1
        guard let wb = weightsBuffer else { throw Gemma4Error.metalUnavailable }
        let (span, of) = n.multipliedReportingOverflow(by: vectors * 4)
        guard !of, n > 0, vectors > 0,
              x.length >= span, y.length >= span else {
            throw Gemma4Error.badTensorGeometry(
                "rmsnorm extents: \(vectors)x\(n) floats need \(span) bytes (x \(x.length), y \(y.length))")
        }
        try dispatch("g4_rmsnorm", grid: vectors) { enc in
            enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 1)
            enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
            enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
            enc.setBuffer(wb, offset: 0, index: 4)
            enc.setBuffer(x, offset: 0, index: 5)
            enc.setBuffer(y, offset: 0, index: 6)
        }
    }

    /// In-place RoPE over `heads` vectors of headDim = 2*half values
    /// (1 thread per head; pairs (i, i+half) rotated by inv[i] * position).
    private func rope(x: MTLBuffer, heads: Int, half: Int, position: Int,
                      freqs: MTLBuffer) throws {
        guard half > 0, heads > 0,
              freqs.length >= half * 4,
              x.length >= heads * 2 * half * 4 else {
            throw Gemma4Error.badTensorGeometry(
                "rope extents: \(heads) heads x \(2 * half) values exceed buffers")
        }
        var pos = UInt32(position)
        var hh = SIMD2<UInt32>(UInt32(heads), UInt32(half))
        try dispatch("g4_rope", grid: heads) { enc in
            enc.setBytes(&hh, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
            enc.setBuffer(x, offset: 0, index: 2)
            enc.setBuffer(freqs, offset: 0, index: 3)
        }
    }

    /// y[i] = gelu_tanh(gu[i]) * gu[n+i]  (fused gate|up layout).
    private func geluUp(gu: MTLBuffer, y: MTLBuffer, n: Int) throws {
        guard gu.length >= 2 * n * 4, y.length >= n * 4 else {
            throw Gemma4Error.badTensorGeometry("gelu_up buffer too small for n=\(n)")
        }
        var count = UInt32(n)
        try dispatch("g4_gelu_up", grid: n) { enc in
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(gu, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
        }
    }

    private func axpy(x: MTLBuffer, y: MTLBuffer, c: Float, n: Int) throws {
        guard x.length >= n * 4, y.length >= n * 4 else {
            throw Gemma4Error.badTensorGeometry("axpy buffer too small for n=\(n)")
        }
        var count = UInt32(n)
        var cc = c
        try dispatch("g4_axpy", grid: n) { enc in
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&cc, length: MemoryLayout<Float>.stride, index: 3)
        }
    }

    private func add(a: MTLBuffer, b: MTLBuffer, y: MTLBuffer, n: Int) throws {
        guard a.length >= n * 4, b.length >= n * 4, y.length >= n * 4 else {
            throw Gemma4Error.badTensorGeometry("add buffer too small for n=\(n)")
        }
        var count = UInt32(n)
        try dispatch("g4_add", grid: n) { enc in
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(a, offset: 0, index: 1)
            enc.setBuffer(b, offset: 0, index: 2)
            enc.setBuffer(y, offset: 0, index: 3)
        }
    }

    private func scaleInPlace(x: MTLBuffer, c: Float, n: Int) throws {
        guard x.length >= n * 4 else {
            throw Gemma4Error.badTensorGeometry("scale buffer too small for n=\(n)")
        }
        var count = UInt32(n)
        var cc = c
        try dispatch("g4_scale", grid: n) { enc in
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            enc.setBytes(&cc, length: MemoryLayout<Float>.stride, index: 2)
        }
    }

    // MARK: - CPU access to shared buffers

    private func read(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        [Float](UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self),
                                    count: count))
    }

    private func write(_ values: [Float], to buffer: MTLBuffer) {
        _ = values.withUnsafeBytes { raw in
            memcpy(buffer.contents(), raw.baseAddress!, values.count * 4)
        }
    }

    // MARK: - forward pass

    /// Resets the KV caches' consumed position; the next `forward` starts a
    /// fresh context at position 0. (Cache bytes need no clearing: only
    /// positions < consumedPosition are ever read.)
    public func reset() {
        consumedPosition = 0
    }

    /// Full forward over `tokenIds`, continuing the current context (call
    /// `reset()` first for a fresh one). Returns evidence with the greedy
    /// argmax and top-5 logits at the final position.
    public func forward(tokenIds: [Int]) throws -> Gemma4RunEvidence {
        guard !tokenIds.isEmpty else { throw Gemma4Error.badTokenId(-1) }
        guard tokenIds.allSatisfy({ $0 >= 0 && $0 < config.vocabSize }) else {
            throw Gemma4Error.badTokenId(tokenIds.first { $0 < 0 || $0 >= config.vocabSize } ?? -1)
        }
        guard consumedPosition + tokenIds.count <= maxContext else {
            throw Gemma4Error.contextTooLong(consumedPosition + tokenIds.count, max: maxContext)
        }
        let dispatchBefore = dispatchCountTotal
        let gpuBefore = gpuMicrosTotal
        for (i, token) in tokenIds.enumerated() {
            _ = try forwardToken(token, position: consumedPosition + i)
            peakFootprint = max(peakFootprint, Gemma4Runner.physFootprint())
        }
        consumedPosition += tokenIds.count
        let logits = read(bufLogits, config.vocabSize)
        guard logits.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.nonFiniteActivation("final logits")
        }
        let argmax = logits.enumerated().max { $0.element < $1.element }!.offset
        let top = logits.enumerated().sorted { $0.element > $1.element }
            .prefix(5).map { ($0.offset, $0.element) }
        return Gemma4RunEvidence(
            deviceName: device.name,
            deviceMaxBufferLength: device.maxBufferLength,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize),
            weightsMappedBytes: mapLength,
            metalAllocatedBytes: Int(clamping: device.currentAllocatedSize),
            tokensProcessed: tokenIds.count,
            contextLength: consumedPosition,
            dispatchedKernels: dispatchCountTotal - dispatchBefore,
            gpuMicros: gpuMicrosTotal - gpuBefore,
            peakFootprintBytes: peakFootprint,
            argmaxToken: argmax,
            topLogits: top.map { (token: $0.0, logit: $0.1) })
    }

    /// Dequantizes one Q6_K embedding row on the GPU (2816 values = 11
    /// blocks), scaled by sqrt(hidden) per the official Gemma4 embedding.
    private func decodeEmbedding(_ token: Int) throws {
        guard let emb = ref["token_embd.weight"] else {
            throw Gemma4Error.badTensorGeometry("missing token_embd.weight")
        }
        guard let wb = weightsBuffer else { throw Gemma4Error.metalUnavailable }
        var dims = UInt32(hidden)
        var off = UInt64(emb.offset)
        var row = UInt32(token)
        var scale = sqrt(Float(hidden))
        try dispatch("g4_q6_embed_row", grid: hidden) { enc in
            enc.setBytes(&dims, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBytes(&row, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 3)
            enc.setBuffer(wb, offset: 0, index: 4)
            enc.setBuffer(bufEmbed, offset: 0, index: 5)
        }
    }

    /// Independent parity gate for the GPU embedding-row decoder: the full
    /// 2816-value row decoded by the kernel is compared value-by-value
    /// against the pre-existing, separately validated scalar CPU Q6_K
    /// `decodeBlock` oracle over all 11 blocks (times the embedding scale).
    /// Returns the worst relative difference.
    @discardableResult
    public func verifyEmbeddingRow(token: Int) throws -> Float {
        try decodeEmbedding(token)
        let gpu = read(bufEmbed, hidden)
        guard gpu.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.nonFiniteActivation("embedding row \(token)")
        }
        let info = try ggufFile.tensor("token_embd.weight")
        let data = try ggufFile.tensorData(info)
        let blocksPerRow = hidden / Q6_K.valuesPerBlock
        let scale = sqrt(Float(hidden))
        var worst: Float = 0
        for b in 0..<blocksPerRow {
            let base = data.startIndex + token * blocksPerRow * Q6_K.bytesPerBlock
                + b * Q6_K.bytesPerBlock
            let blockBytes = Array(data[base..<(base + Q6_K.bytesPerBlock)])
            let expected = try Q6_K.decodeBlock(blockBytes[0..<Q6_K.bytesPerBlock])
            for j in 0..<Q6_K.valuesPerBlock {
                let want = expected[j] * scale
                let got = gpu[b * Q6_K.valuesPerBlock + j]
                let diff = abs(got - want) / max(abs(want), 1e-6)
                worst = max(worst, diff)
            }
        }
        return worst
    }

    /// One token at `position`; writes into the KV caches and returns the
    /// greedy argmax of the (already softcapped) logits.
    private func forwardToken(_ token: Int, position: Int) throws -> Int {
        try decodeEmbedding(token)
        let embVals = read(bufEmbed, hidden)
        guard embVals.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.nonFiniteActivation("embedding row \(token)")
        }
        write(embVals, to: bufX)

        for layer in 0..<config.layerCount {
            try forwardLayer(layer, position: position)
        }

        // Final norm + tied LM head (Q6_K embedding rows dotted with h) +
        // softcap 30.
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1,
                    weightTensor: "output_norm.weight")
        try matvec(tensor: "token_embd.weight", x: bufH, xCount: hidden,
                   y: bufLogits, yCapacity: config.vocabSize,
                   rows: config.vocabSize)
        var cap = config.finalSoftcap
        var count = UInt32(config.vocabSize)
        try dispatch("g4_softcap", grid: config.vocabSize) { enc in
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
            enc.setBuffer(bufLogits, offset: 0, index: 1)
            enc.setBytes(&cap, length: MemoryLayout<Float>.stride, index: 2)
        }
        let logits = read(bufLogits, config.vocabSize)
        guard let bad = logits.firstIndex(where: { !$0.isFinite }) else {
            return logits.enumerated().max { $0.element < $1.element }!.offset
        }
        throw Gemma4Error.nonFiniteActivation("logits[\(bad)]")
    }

    private func forwardLayer(_ layer: Int, position: Int) throws {
        let isGlobal = config.isGlobalLayer(layer)
        let headDim = isGlobal ? config.globalHeadDim : config.localHeadDim
        let kvHeads = isGlobal ? config.globalKVHeads : config.localKVHeads
        let kvWidth = kvHeads * headDim
        let qWidth = config.qHeads * headDim
        let prefix = "blk.\(layer)."

        // ---- attention block ----
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1,
                    weightTensor: prefix + "attn_norm.weight")
        try matvec(tensor: prefix + "attn_q.weight", x: bufH, xCount: hidden,
                   y: bufT1, yCapacity: config.qHeads * config.globalHeadDim)
        // QK-norm per head (1 thread/head), then RoPE on q (1 thread/head).
        try rmsnorm(x: bufT1, y: bufT1, n: headDim, vectors: config.qHeads,
                    weightTensor: prefix + "attn_q_norm.weight")
        try rope(x: bufT1, heads: config.qHeads, half: headDim / 2, position: position,
                 freqs: isGlobal ? globalFreqBuffer : localFreqBuffer)
        try matvec(tensor: prefix + "attn_k.weight", x: bufH, xCount: hidden,
                   y: bufT2, yCapacity: max(hidden, 2048))
        if isGlobal {
            // k_eq_v: v IS the raw k_proj output (pre k_norm, pre RoPE),
            // then scale-free per-head RMSNorm (v_norm).
            let kRaw = read(bufT2, kvWidth)
            write(kRaw, to: bufKV)
            try rmsnorm(x: bufT2, y: bufT2, n: headDim, vectors: kvHeads,
                        weightTensor: prefix + "attn_k_norm.weight")
            try rope(x: bufT2, heads: kvHeads, half: headDim / 2, position: position,
                     freqs: globalFreqBuffer)
            try rmsnorm(x: bufKV, y: bufKV, n: headDim, vectors: kvHeads, weightTensor: nil)
        } else {
            // local: v from its own projection, then scale-free v_norm.
            try matvec(tensor: prefix + "attn_v.weight", x: bufH, xCount: hidden,
                       y: bufKV, yCapacity: 2048)
            try rmsnorm(x: bufT2, y: bufT2, n: headDim, vectors: kvHeads,
                        weightTensor: prefix + "attn_k_norm.weight")
            try rope(x: bufT2, heads: kvHeads, half: headDim / 2, position: position,
                     freqs: localFreqBuffer)
            try rmsnorm(x: bufKV, y: bufKV, n: headDim, vectors: kvHeads, weightTensor: nil)
        }
        let kNormed = read(bufT2, kvWidth)
        let vNormed = read(bufKV, kvWidth)
        memcpy(kCaches[layer].contents() + position * kvWidth * 4, kNormed, kvWidth * 4)
        memcpy(vCaches[layer].contents() + position * kvWidth * 4, vNormed, kvWidth * 4)

        // Attention: one thread per query head; sliding window 1024 on local
        // layers, full causal on global; scaling is 1.0 (QK-normed heads).
        var p = SIMD4<UInt32>(UInt32(position + 1), UInt32(position),
                              UInt32(kvHeads), UInt32(headDim))
        var q = SIMD2<UInt32>(UInt32(config.qHeads),
                              UInt32(isGlobal ? 0 : config.slidingWindow))
        try dispatch("g4_attn", grid: config.qHeads) { enc in
            enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
            enc.setBytes(&q, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
            enc.setBuffer(bufT1, offset: 0, index: 2)
            enc.setBuffer(kCaches[layer], offset: 0, index: 3)
            enc.setBuffer(vCaches[layer], offset: 0, index: 4)
            enc.setBuffer(bufT3, offset: 0, index: 5)
        }
        try matvec(tensor: prefix + "attn_output.weight", x: bufT3, xCount: qWidth,
                   y: bufT2, yCapacity: max(hidden, 2048))
        try rmsnorm(x: bufT2, y: bufH, n: hidden, vectors: 1,
                    weightTensor: prefix + "post_attention_norm.weight")
        try axpy(x: bufH, y: bufX, c: 1, n: hidden)   // x += postnorm(attn_out)

        // ---- FFN block: shared expert + routed MoE ----
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1,
                    weightTensor: prefix + "ffn_norm.weight")
        // Shared expert. gate|up fused into bufGU ([0..2112) gate,
        // [2112..4224) up via a 256-aligned binding offset); its normed
        // output lives in bufShared and must survive the routed loop.
        try matvec(tensor: prefix + "ffn_gate.weight", x: bufH, xCount: hidden,
                   y: bufGU, yCapacity: 2 * config.sharedIntermediate)
        try matvec(tensor: prefix + "ffn_up.weight", x: bufH, xCount: hidden,
                   y: bufGU, yCapacity: 2 * config.sharedIntermediate,
                   yByteOffset: config.sharedIntermediate * 4)
        try geluUp(gu: bufGU, y: bufAct, n: config.sharedIntermediate)
        try matvec(tensor: prefix + "ffn_down.weight", x: bufAct,
                   xCount: config.sharedIntermediate,
                   y: bufShared, yCapacity: hidden)
        if traceLayers {
            let gu = read(bufGU, 2 * config.sharedIntermediate)
            let act = read(bufAct, config.sharedIntermediate)
            let dn = read(bufShared, hidden)
            func stat(_ v: [Float]) -> (Int, Float) {
                (v.lazy.filter { !$0.isFinite }.count,
                 v.lazy.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0)
            }
            let (g1, g2) = (stat(Array(gu[0..<2112])), stat(Array(gu[2112..<4224])))
            let (a1, a2) = (stat(act), stat(dn))
            print("  layer \(layer) gate(\(g1.0) nf, max \(g1.1)) up(\(g2.0) nf, max \(g2.1)) " +
                  "act(\(a1.0) nf, max \(a1.1)) down(\(a2.0) nf, max \(a2.1))")
            for i in act.indices where !act[i].isFinite {
                print("    act[\(i)] = \(act[i]), gate \(gu[i]), up \(gu[2112 + i])")
            }
        }
        try rmsnorm(x: bufShared, y: bufShared, n: hidden, vectors: 1,
                    weightTensor: prefix + "post_ffw_norm_1.weight")
        if traceLayers {
            let s = read(bufShared, hidden)
            print("  layer \(layer) shared mlp: nonfinite \(s.lazy.filter { !$0.isFinite }.count), " +
                  "max \(s.lazy.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0)")
        }

        // Router on the residual (pre-norm) stream: scale-free RMSNorm,
        // * scale * hidden^-0.5, F32 router matvec, fp64 CPU softmax,
        // top-8 (ties -> lower index), renormalize, * per-expert QAT scale.
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1, weightTensor: nil)
        try scaleInPlace(x: bufH, c: Float(1.0 / sqrt(Double(hidden))), n: hidden)
        var routerIn = read(bufH, hidden)
        let routerScale = routerScales[layer]
        for i in 0..<hidden { routerIn[i] *= routerScale[i] }
        write(routerIn, to: bufH)
        try matvec(tensor: prefix + "ffn_gate_inp.weight", x: bufH, xCount: hidden,
                   y: bufRouter, yCapacity: config.expertCount)
        let scores = read(bufRouter, config.expertCount)
        guard scores.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.nonFiniteActivation("layer \(layer) router scores")
        }
        let maxScore = scores.max() ?? 0
        let exps = scores.map { exp(Double($0 - maxScore)) }
        let sumExp = exps.reduce(0, +)
        let probs = exps.map { $0 / sumExp }
        var idx = Array(0..<config.expertCount)
        idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
        let topIdx = Array(idx.prefix(config.topKExperts))
        let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
        let scales = expertScales[layer]
        let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }

        // Routed experts on the residual stream normed by pre_ffw_norm_2.
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1,
                    weightTensor: prefix + "pre_ffw_norm_2.weight")
        guard let gateUp = ref[prefix + "ffn_gate_up_exps.weight"],
              let down = ref[prefix + "ffn_down_exps.weight"] else {
            throw Gemma4Error.badTensorGeometry("missing expert weights layer \(layer)")
        }
        let e = config.expertIntermediate
        // Per-expert slice strides in bytes (validated inside matvec bounds).
        let guBlocksPerRow = gateUp.inDim / 32
        let guStride = gateUp.outDim * guBlocksPerRow * 18
        let dnBlocksPerRow = down.inDim / 32
        let dnStride = down.outDim * dnBlocksPerRow * 18
        memset(bufT3.contents(), 0, hidden * 4)   // MoE accumulator
        for (slot, expert) in topIdx.enumerated() {
            try matvec(tensor: prefix + "ffn_gate_up_exps.weight",
                       extraOffset: expert * guStride,
                       x: bufH, xCount: hidden,
                       y: bufGU, yCapacity: 2 * config.sharedIntermediate)
            try geluUp(gu: bufGU, y: bufAct, n: e)
            try matvec(tensor: prefix + "ffn_down_exps.weight",
                       extraOffset: expert * dnStride,
                       x: bufAct, xCount: e,
                       y: bufT2, yCapacity: max(hidden, 2048))
            try axpy(x: bufT2, y: bufT3, c: weights[slot], n: hidden)
        }
        try rmsnorm(x: bufT3, y: bufT3, n: hidden, vectors: 1,
                    weightTensor: prefix + "post_ffw_norm_2.weight")
        if traceLayers {
            let m = read(bufT3, hidden)
            print("  layer \(layer) moe sum: nonfinite \(m.lazy.filter { !$0.isFinite }.count), " +
                  "max \(m.lazy.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0), " +
                  "topIdx \(topIdx), weights \(weights.map { String(format: "%.3f", $0) })")
        }
        // h = post_ffw_norm(post_ffw_norm_1(sharedMLP) + moe2); x = residual + h.
        try add(a: bufShared, b: bufT3, y: bufH, n: hidden)
        try rmsnorm(x: bufH, y: bufH, n: hidden, vectors: 1,
                    weightTensor: prefix + "post_ffw_norm.weight")
        try axpy(x: bufH, y: bufX, c: 1, n: hidden)
        try scaleInPlace(x: bufX, c: layerScalars[layer], n: hidden)
        if traceLayers {
            let x = read(bufX, hidden)
            let bad = x.lazy.filter { !$0.isFinite }.count
            let maxAbs = x.lazy.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0
            print("layer \(layer): nonfinite \(bad), max|x| \(maxAbs), scalar \(layerScalars[layer])")
        }
    }

    // MARK: - footprint

    /// Physical footprint of this process (bytes), via task_info.
    public static func physFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO),
                          $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    // MARK: - MSL

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // ---- quantized matvec kernels; weights addressed as base + wOff ----
    // p = (inDim, outDim); row-major weight rows of inDim values.
    kernel void g4_q4_matvec(constant const uint2& p [[buffer(0)]],
                             constant const ulong& wOff [[buffer(1)]],
                             device const uchar* wbase [[buffer(2)]],
                             device const float* x [[buffer(3)]],
                             device float* y [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
        uint row = gid;
        if (row >= p.y) { return; }
        uint blocksPerRow = p.x / 32u;
        device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 18u;
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
        y[row] = sum;
    }

    kernel void g4_q6_matvec(constant const uint2& p [[buffer(0)]],
                             constant const ulong& wOff [[buffer(1)]],
                             device const uchar* wbase [[buffer(2)]],
                             device const float* x [[buffer(3)]],
                             device float* y [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
        uint row = gid;
        if (row >= p.y) { return; }
        uint blocksPerRow = p.x / 256u;
        device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 210u;
        float sum = 0.0f;
        for (uint b = 0; b < blocksPerRow; b++) {
            device const uchar* blk = rowBytes + b * 210u;
            ushort dbits = ushort(blk[208]) | (ushort(blk[209]) << 8u);
            float d = float(as_type<half>(dbits));
            for (uint c = 0; c < 2u; c++) {
                uint ql = c * 64u;
                uint qh = 128u + c * 32u;
                uint sc = c * 8u;
                uint xbase = b * 256u + c * 128u;
                for (uint l = 0; l < 32u; l++) {
                    uint isub = l / 16u;
                    uchar h = blk[qh + l];
                    float s0 = float((char)blk[192u + sc + isub]);
                    float s1 = float((char)blk[192u + sc + isub + 2u]);
                    float s2 = float((char)blk[192u + sc + isub + 4u]);
                    float s3 = float((char)blk[192u + sc + isub + 6u]);
                    float q1 = float((int)((blk[ql + l] & 0x0Fu) | ((h & 3u) << 4u)) - 32);
                    float q2 = float((int)((blk[ql + 32u + l] & 0x0Fu) | (((h >> 2u) & 3u) << 4u)) - 32);
                    float q3 = float((int)((blk[ql + l] >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                    float q4 = float((int)((blk[ql + 32u + l] >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                    sum += d * s0 * q1 * x[xbase + l];
                    sum += d * s1 * q2 * x[xbase + 32u + l];
                    sum += d * s2 * q3 * x[xbase + 64u + l];
                    sum += d * s3 * q4 * x[xbase + 96u + l];
                }
            }
        }
        y[row] = sum;
    }

    kernel void g4_f32_matvec(constant const uint2& p [[buffer(0)]],
                              constant const ulong& wOff [[buffer(1)]],
                              device const uchar* wbase [[buffer(2)]],
                              device const float* x [[buffer(3)]],
                              device float* y [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
        uint row = gid;
        if (row >= p.y) { return; }
        device const float* w = (device const float*)(wbase + wOff) + (ulong)row * p.x;
        float sum = 0.0f;
        for (uint i = 0; i < p.x; i++) { sum += w[i] * x[i]; }
        y[row] = sum;
    }

    // Dequantize one Q6_K embedding row (valuesPerBlock 256, bytes 210).
    // Canonical layout per value j: c = j/128 half-chunk, q = (j%128)/32
    // quarter, lane = j%32, subgroup = lane/16; ql byte = c*64 + lane +
    // (q odd ? 32 : 0); low nibble for q<2, high for q>=2; top 2 bits from
    // qh[c*32 + lane] >> (2q); scale = sc[c*8 + 2q + subgroup].
    kernel void g4_q6_embed_row(constant const uint& dim [[buffer(0)]],
                                constant const ulong& wOff [[buffer(1)]],
                                constant const uint& row [[buffer(2)]],
                                constant const float& scale [[buffer(3)]],
                                device const uchar* wbase [[buffer(4)]],
                                device float* y [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
        uint v = gid;
        if (v >= dim) { return; }
        uint blocksPerRow = dim / 256u;
        uint block = v / 256u;
        uint j = v % 256u;
        device const uchar* blk = wbase + wOff
            + ((ulong)row * blocksPerRow + block) * 210u;
        ushort dbits = ushort(blk[208]) | (ushort(blk[209]) << 8u);
        float d = float(as_type<half>(dbits));
        uint c = j / 128u;
        uint l0 = j % 128u;
        uint quarter = l0 / 32u;
        uint lane = l0 % 32u;
        uint subgroup = lane / 16u;
        uint qlIndex = c * 64u + lane + ((quarter & 1u) != 0u ? 32u : 0u);
        uchar h = blk[128u + c * 32u + lane];
        float s = float((char)blk[192u + c * 8u + 2u * quarter + subgroup]);
        uint lo = (quarter < 2u) ? (blk[qlIndex] & 0x0Fu) : (blk[qlIndex] >> 4u);
        uint two = (h >> (2u * quarter)) & 3u;
        float level = float(int(lo | (two << 4u)) - 32);
        y[v] = d * s * level * scale;
    }

    // RMSNorm over p.y contiguous vectors of length p.x (1 thread/vector).
    kernel void g4_rmsnorm(constant const uint2& p [[buffer(0)]],
                           constant const float& eps [[buffer(1)]],
                           constant const int& withScale [[buffer(2)]],
                           constant const ulong& wOff [[buffer(3)]],
                           device const uchar* wbase [[buffer(4)]],
                           device const float* x [[buffer(5)]],
                           device float* y [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
        uint vec = gid;
        if (vec >= p.y) { return; }
        uint n = p.x;
        device const float* xv = x + (ulong)vec * n;
        float ms = 0.0f;
        for (uint i = 0; i < n; i++) { ms += xv[i] * xv[i]; }
        ms = ms / float(n) + eps;
        float inv = 1.0f / sqrt(ms);
        device const float* w = (device const float*)(wbase + wOff);
        device float* yv = y + (ulong)vec * n;
        for (uint i = 0; i < n; i++) {
            yv[i] = xv[i] * inv * (withScale != 0 ? w[i] : 1.0f);
        }
    }

    // Overflow-safe tanh: Metal's tanh() returns NaN for |x| > ~49 on this
    // driver (verified by a standalone probe, 193/400 NaN); this form uses
    // exp(-2|x|) which never overflows and is the same real-valued math.
    static float g4_tanh(float x) {
        float e = exp(-2.0f * fabs(x));
        return copysign((1.0f - e) / (1.0f + e), x);
    }

    // y[i] = gelu_tanh(gu[i]) * gu[n+i]
    kernel void g4_gelu_up(constant const uint& n [[buffer(0)]],
                           device const float* gu [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
        uint i = gid;
        if (i >= n) { return; }
        float g = gu[i];
        float u = gu[n + i];
        float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        float gelu = 0.5f * g * (1.0f + g4_tanh(inner));
        y[i] = gelu * u;
    }

    // In-place RoPE: pairs (i, i+half) with frequency inv[i] * pos
    // (1 thread/head; half frequencies cover headDim = 2*half values).
    kernel void g4_rope(constant const uint2& hh [[buffer(0)]],
                        constant const uint& pos [[buffer(1)]],
                        device float* x [[buffer(2)]],
                        device const float* inv [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
        uint head = gid;
        if (head >= hh.x) { return; }
        uint halfLen = hh.y;
        device float* v = x + (ulong)head * 2u * halfLen;
        for (uint i = 0; i < halfLen; i++) {
            float f = inv[i] * float(pos);
            float cs = cos(f);
            float sn = sin(f);
            float a = v[i];
            float b = v[i + halfLen];
            v[i] = a * cs - b * sn;
            v[i + halfLen] = a * sn + b * cs;
        }
    }

    // Single-query attention over a KV cache. p = (T, qpos, kvHeads, headDim)
    // q = (qHeads, window; window 0 = full). Scaling is 1.0 (QK-normed).
    // Three passes over the attended range: stable max, then exp-sum, then
    // the weighted V accumulation (exp rescored per pass).
    kernel void g4_attn(constant const uint4& p [[buffer(0)]],
                        constant const uint2& q [[buffer(1)]],
                        device const float* qv [[buffer(2)]],
                        device const float* kc [[buffer(3)]],
                        device const float* vc [[buffer(4)]],
                        device float* out [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
        uint head = gid;
        if (head >= q.x) { return; }
        uint qpos = p.y, kvHeads = p.z, hd = p.w;
        uint window = q.y;
        uint group = q.x / kvHeads;
        uint kvh = head / group;
        device const float* qh = qv + (ulong)head * hd;
        device const float* khead = kc + (ulong)kvh * hd;
        device const float* vhead = vc + (ulong)kvh * hd;
        uint lo = 0u;
        if (window != 0u && qpos + 1u > window) { lo = qpos + 1u - window; }
        float m = -INFINITY;
        for (uint t = lo; t <= qpos; t++) {
            device const float* kt = khead + (ulong)t * kvHeads * hd;
            float s = 0.0f;
            for (uint i = 0; i < hd; i++) { s += qh[i] * kt[i]; }
            m = max(m, s);
        }
        float l = 0.0f;
        for (uint t = lo; t <= qpos; t++) {
            device const float* kt = khead + (ulong)t * kvHeads * hd;
            float s = 0.0f;
            for (uint i = 0; i < hd; i++) { s += qh[i] * kt[i]; }
            l += exp(s - m);
        }
        for (uint i = 0; i < hd; i++) { out[(ulong)head * hd + i] = 0.0f; }
        for (uint t = lo; t <= qpos; t++) {
            device const float* kt = khead + (ulong)t * kvHeads * hd;
            device const float* vt = vhead + (ulong)t * kvHeads * hd;
            float s = 0.0f;
            for (uint i = 0; i < hd; i++) { s += qh[i] * kt[i]; }
            float wgt = exp(s - m) / l;
            for (uint i = 0; i < hd; i++) { out[(ulong)head * hd + i] += wgt * vt[i]; }
        }
    }

    kernel void g4_axpy(constant const uint& n [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant const float& c [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        y[gid] += c * x[gid];
    }

    kernel void g4_add(constant const uint& n [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       device float* y [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        y[gid] = a[gid] + b[gid];
    }

    kernel void g4_scale(constant const uint& n [[buffer(0)]],
                         device float* x [[buffer(1)]],
                         constant const float& c [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        x[gid] *= c;
    }

    // In-place final-logit softcap: x = cap * tanh(x / cap).
    kernel void g4_softcap(constant const uint& n [[buffer(0)]],
                           device float* x [[buffer(1)]],
                           constant const float& cap [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        x[gid] = cap * g4_tanh(x[gid] / cap);
    }
    """
}
