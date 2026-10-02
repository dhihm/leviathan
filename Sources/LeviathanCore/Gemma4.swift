import Foundation
import CryptoKit
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

    public let config: Gemma4Config
    /// The parsed GGUF file (tokenizer metadata is read from it).
    public let ggufFile: GGUFFile
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]
    internal var dispatchCountTotal = 0
    internal var gpuMicrosTotal: Double = 0

    // Weight storage. The mapping's lifetime is owned by `weightsBuffer`'s
    // deallocator (RAII); no manual munmap outside init failure paths.
    private var mapLength = 0
    private var weightsBuffer: MTLBuffer?
    private var mapEnd = 0    // exclusive absolute offset covered by the buffer

    // Activation buffers (F32, tiny compared with weights).
    internal var bufX, bufH, bufT1, bufT2, bufT3, bufKV, bufGU, bufAct: MTLBuffer!
    /// Dedicated expert-loop scratch (never aliases SSA-bound buffers).
    internal var bufExpertGU, bufExpertAct, bufExpertAcc: MTLBuffer!
    internal var bufShared, bufRouter, bufEmbed, bufLogits: MTLBuffer!
    internal var localFreqBuffer, globalFreqBuffer: MTLBuffer!
    // KV caches per layer: [maxContext * kvHeads * headDim] F32 each.
    internal var kCaches: [MTLBuffer] = []
    internal var vCaches: [MTLBuffer] = []

    // Tensor addressing table.
    internal var ref: [String: G4TensorRef] = [:]
    // Small F32 parameters read into CPU arrays at init (finite-checked).
    internal var expertScales: [[Float]] = []   // [layer][128] router per-expert scale
    internal var layerScalars: [Float] = []     // layer_output_scale
    internal var routerScales: [[Float]] = []   // [layer][2816] router input scale

    internal let hidden: Int
    internal let maxContext: Int
    internal var consumedPosition = 0
    internal var peakFootprint = 0
    /// Diagnostic: print per-layer residual max |x| (and non-finite counts)
    /// to localize divergence during bring-up.
    public var traceLayers = false

    /// Resident-mode runner (whole-file GPU mapping). For the bounded
    /// SSD expert mode prefer `init(fileAt:expertCacheBudgetBytes:)` —
    /// it never creates the whole-file mapping, so startup memory is
    /// the compact buffer + KV + activations + slot pools only.
    public convenience init(fileAt url: URL) throws {
        try self.init(fileAt: url, expertCacheBudgetBytes: nil)
    }

    /// Cache-mode runner: with a non-nil budget, the 14.44 GB whole-
    /// file GPU mapping is NEVER created — non-expert weights are
    /// compacted directly from the file into a dedicated GPU buffer
    /// (~1.6 GB) and expert slices stream through the bounded cache.
    /// A nil budget is exactly the resident init.
    public init(fileAt url: URL, expertCacheBudgetBytes budget: Int?) throws {
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

        // ---- weight storage. Resident mode: mmap the whole file read-
        // only for the zero-copy weight buffer. Cache mode: SKIP the
        // whole-file mapping entirely (bounded startup) — the switch
        // below compacts non-expert weights straight from the file.
        if budget == nil {
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

            // Preflight: the mapped region must fit within device limits.
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
        }

        // ---- shaders + pipelines (compiled once). The runtime Metal
        // library compile + pipeline construction is clocked SEPARATELY
        // from the rest of the runner load (weight mapping etc.) so the
        // CLI can report the split; monotonic DispatchTime, bounded to
        // this one section, kernels and error paths unchanged.
        let metalCompileT0 = DispatchTime.now().uptimeNanoseconds
        do { library = try device.makeLibrary(source: Self.shaderSource, options: nil) }
        catch { throw Gemma4Error.shaderCompileFailed("\(error)") }
        for name in ["g4_q4_matvec", "g4_q4_matvec_tiled", "g4_q6_matvec", "g4_q6_matvec_tiled", "g4_f32_matvec", "g4_f32_matvec_tiled", "g4_q6_embed_row",
                     "g4_rmsnorm", "g4_rmsnorm_vl", "g4_rmsnorm_tiled", "g4_gelu_up", "g4_rope", "g4_attn",
                     "g4_axpy", "g4_add", "g4_scale", "g4_softcap"] {
            guard let fn = library.makeFunction(name: name),
                  let p = try? device.makeComputePipelineState(function: fn) else {
                throw Gemma4Error.functionMissing(name)
            }
            pipelines[name] = p
        }
        metalCompileMicros = Double(DispatchTime.now().uptimeNanoseconds - metalCompileT0) / 1000
        // Hardware-derived tile widths: the largest legal power of two
        // (tree reduction, kernel partials cap 128) that is at most 2× the
        // pipeline's execution width and the device maximum. A missing
        // pipeline or a ceiling below 32 is a hard error — no silent
        // fallback width.
        func derivedTileWidth(_ name: String) throws -> Int {
            guard let tiled = pipelines[name],
                  tiled.maxTotalThreadsPerThreadgroup >= 32 else {
                throw Gemma4Error.metalUnavailable
            }
            let ceiling = min(128, tiled.maxTotalThreadsPerThreadgroup,
                              max(32, 2 * tiled.threadExecutionWidth))
            var w = 128
            while w > ceiling { w /= 2 }
            return w
        }
        q4TileWidth = try derivedTileWidth("g4_q4_matvec_tiled")
        q6TileWidth = try derivedTileWidth("g4_q6_matvec_tiled")
        rmsTileWidth = try derivedTileWidth("g4_rmsnorm_tiled")
        f32TileWidth = try derivedTileWidth("g4_f32_matvec_tiled")

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
        // DEDICATED expert-loop scratch, physically separate from every
        // SSA-bound activation buffer: the compound expert op's internal
        // effects can never clobber still-live SSA values (gate/up/act/t2
        // remain valid across the expert chain).
        bufExpertGU = try alloc(2 * config.sharedIntermediate)
        bufExpertAct = try alloc(max(config.sharedIntermediate,
                                     config.expertIntermediate))
        bufExpertAcc = try alloc(hidden)
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
        // Cache-mode construction: switch AFTER all standard setup, with
        // no whole-file mapping ever created (bounded startup peak).
        if let budget {
            try switchToExpertCache(budgetBytes: budget)
            expertCache?.noteInitAllocated()
        }
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

    // Deferred command-buffer encoding (graph-executor segment batching):
    // when a segment is open, dispatch() only encodes into it; any CPU
    // readback must flush first. Numerically identical to per-kernel
    // commit (same kernels, same order) — only submission granularity
    // changes. GPU time is accounted at flush for the whole segment.
    internal var openCmd: MTLCommandBuffer? = nil
    internal var openEnc: MTLComputeCommandEncoder? = nil
    /// When true, dispatch() lazily opens segments (see above).
    internal var autoSegment = false
    /// Submitted command buffers (separate from kernel count) — makes
    /// the batching effect observable.
    internal var submittedBuffers = 0
    /// Profiling attribution (per-kernel mode only): accumulated GPU
    /// microseconds per weight-tensor name across matvec calls.
    internal var matvecGpuMicrosByTensor: [String: Double] = [:]
    internal var matvecCallsByTensor: [String: Int] = [:]
    internal var kernelGpuMicrosByName: [String: (Double, Int)] = [:]

    /// Batched wall/GPU breakdown instrumentation (OFF by default; no
    /// timer reads when off — one branch remains). Categories are
    /// measured at the runner primitives WITHOUT changing submission
    /// or synchronization — unlike the per-kernel profile, the batched
    /// execution path is untouched (same kernels, same segments, same
    /// flush points):
    ///   encode   — CPU time encoding kernels into the open deferred
    ///              segment (dispatch/dispatchExplicit deferred branch);
    ///              GPU is idle at this point in batched mode, so this
    ///              does not overlap GPU busy time,
    ///   flush    — wall time inside flushOpenSegment (endEncoding +
    ///              commit + waitUntilCompleted + status checks); this
    ///              OVERLAPS GPU busy time (the wait spans execution),
    ///              so the host-side share is flush - gpu (reported
    ///              clamped at 0),
    ///   readCopy — CPU time copying bytes out of a shared buffer
    ///              (excluding the flush it triggers),
    ///   writeCopy — CPU time memcpy-ing into a shared buffer.
    /// GPU busy time comes from gpuMicrosTotal as usual. Everything else
    /// (router softmax/top-k on CPU, operand resolution, argmax, CLI
    /// overhead) is the reported residual. Monotonic DispatchTime.
    /// Runtime Metal library compile + compute-pipeline construction
    /// time (microseconds), measured inside the runner constructor so
    /// load-time reporting can separate shader compilation from weight
    /// mapping/allocation. Zero only if somehow skipped; set once at init.
    public internal(set) var metalCompileMicros: Double = 0

    public var breakdownInstrumentation = false
    internal var brEncodeMicros = 0.0
    internal var brFlushMicros = 0.0
    internal var brReadCopyMicros = 0.0
    internal var brWriteCopyMicros = 0.0

    /// Zeroes the breakdown counters (call right before the measured
    /// window; GPU time is read as a gpuMicrosTotal delta by the caller).
    public func breakdownReset() {
        brEncodeMicros = 0
        brFlushMicros = 0
        brReadCopyMicros = 0
        brWriteCopyMicros = 0
    }

    /// Point sample of the counters (microseconds) since the last reset.
    public func breakdownCounters() -> (encode: Double, flush: Double,
                                        readCopy: Double, writeCopy: Double) {
        (brEncodeMicros, brFlushMicros, brReadCopyMicros, brWriteCopyMicros)
    }

    private static func brNow() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1000
    }

    /// When true, Q4_0 matvecs dispatch the tiled cooperative kernel
    /// (a Forge candidate — gated against the FP64 oracle and measured
    /// on real hardware before selection). Public so the profiling CLI
    /// can measure the selected kernels with exact per-kernel attribution.
    public var q4TiledKernel = false
    /// Same for Q6_K matvecs (tied embedding / LM head).
    public var q6TiledKernel = false
    /// Q6_K tiled width, derived from the Q6 pipeline's OWN limits.
    internal var q6TileWidth = 64
    /// Tiled cooperative RMSNorm (a Forge candidate; the per-vector
    /// single-thread reduce is the measured 44% GPU-time hotspot).
    public var rmsTiledKernel = false
    internal var rmsTileWidth = 64
    /// Tiled cooperative F32 matvec (MoE router ffn_gate_inp — the
    /// one-thread-per-row scalar kernel is a measured GPU hotspot).
    public var f32TiledKernel = false
    internal var f32TileWidth = 64
    /// Vectorized-load plain RMSNorm dispatch (a Forge candidate;
    /// float4 loads with an identical arithmetic sequence — EXACT bit
    /// equality against the retained original kernel is asserted on
    /// hardware by the test suite). A vectorized-load Q4 matvec variant
    /// was also written and measured 2.1x per call, but its compiled
    /// outputs differed from the original kernel (cause not established)
    /// — it is REJECTED from legal dispatch and kept only as a test
    /// diagnostic. The runner's reference forward path keeps the
    /// original kernels unless this is set.
    public var vectorizedLoads = false
    /// Hardware-derived tile width (set at init from the compiled
    /// pipeline): 2× execution width, clamped to the device maximum and
    /// the kernel's 128-slot partials array.
    internal var q4TileWidth = 64

    /// Commits and waits for the open deferred segment, if any.
    internal func flushOpenSegment() throws {
        guard let cmd = openCmd, let enc = openEnc else { return }
        let brT0 = breakdownInstrumentation ? Self.brNow() : 0
        defer { if breakdownInstrumentation { brFlushMicros += Self.brNow() - brT0 } }
        openCmd = nil
        openEnc = nil
        enc.endEncoding()
        cmd.commit()
        submittedBuffers += 1
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Gemma4Error.commandBufferFailed(
                "deferred segment: status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        gpuMicrosTotal += (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
        expertCache?.noteSegmentFlushed()
    }

    /// Error cleanup: ends (without committing) any open deferred encoder,
    /// drops the uncommitted command buffer, and disables auto-segment.
    /// Used on execution failure so no encoder leaks and no later flush can
    /// commit kernels from a failed partial run. The thrown error is the
    /// caller's to preserve.
    internal func discardOpenSegment() {
        openEnc?.endEncoding()
        openEnc = nil
        openCmd = nil
        autoSegment = false
        // The discarded segment never executes — its expert-slice pins
        // cannot be waiting on GPU reads and are released wholesale.
        expertCache?.discardAllPins()
    }

    /// Opens a deferred segment: subsequent dispatches encode into one
    /// command buffer until `flushOpenSegment()`.
    internal func beginSegment() throws {
        try flushOpenSegment()
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Gemma4Error.metalUnavailable
        }
        openCmd = cmd
        openEnc = enc
    }

    /// Encodes one kernel in its own command buffer, commits, and waits —
    /// unless a deferred segment is open, in which case the kernel is
    /// encoded into it (same kernels, same order).
    internal func dispatch(_ name: String, grid: Int,
                          _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard let pipeline = pipelines[name] else { throw Gemma4Error.metalUnavailable }
        // Auto-segment mode (graph executor batching candidate): open a
        // segment lazily so consecutive GPU-only kernels across ops share
        // one command buffer; CPU readback flushes it.
        if openEnc == nil, autoSegment { try beginSegment() }
        if let deferredEnc = openEnc {
            let brT0 = breakdownInstrumentation ? Self.brNow() : 0
            defer { if breakdownInstrumentation { brEncodeMicros += Self.brNow() - brT0 } }
            deferredEnc.setComputePipelineState(pipeline)
            bind(deferredEnc)
            let width = max(1, min(grid, pipeline.maxTotalThreadsPerThreadgroup))
            let groups = (grid + width - 1) / width
            deferredEnc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            dispatchCountTotal += 1   // count kernels, not buffers
            return
        }
        guard let cmd = queue.makeCommandBuffer(),
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
        submittedBuffers += 1
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Gemma4Error.commandBufferFailed(
                "kernel \(name): status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        let delta = (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
        gpuMicrosTotal += delta
        kernelGpuMicrosByName[name, default: (0, 0)].0 += delta
        kernelGpuMicrosByName[name]!.1 += 1
        dispatchCountTotal += 1
    }

    /// Shared PRODUCTION validator for cooperative kernel widths: a
    /// positive power of two (tree reduction), at most 128 (the kernels'
    /// partials arrays), and within the pipeline's device maximum. Used
    /// by dispatchExplicit AND by numerical tests — one decision, no
    /// mirrored guards.
    /// PRODUCTION guard for the vectorized-load RMSNorm dispatch: the
    /// width must be a positive float4 multiple, and a scaled weight
    /// tensor must sit at a 16-byte aligned offset in the packed weight
    /// buffer. Used by rmsnorm() and directly by the test suite — one
    /// decision, no mirrored guards.
    public static func rmsVectorLoadGuards(n: Int, scaled: Bool, wOff: UInt64) -> Bool {
        guard n > 0, n % 4 == 0 else { return false }
        if scaled && wOff % 16 != 0 { return false }
        return true
    }

    public static func legalCooperativeWidth(_ w: Int, pipelineMax: Int) -> Bool {
        w > 0 && w & (w - 1) == 0 && w <= 128 && w <= pipelineMax
    }

    /// Like dispatch(), but with an EXPLICIT threadgroup shape (groups ×
    /// threads) for cooperative kernels. The width must pass
    /// legalCooperativeWidth — illegal widths are REJECTED, never
    /// silently clamped.
    internal func dispatchExplicit(_ name: String, groups: Int, threads: Int,
                                   _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard let pipeline = pipelines[name] else { throw Gemma4Error.metalUnavailable }
        guard Self.legalCooperativeWidth(threads,
                                         pipelineMax: pipeline.maxTotalThreadsPerThreadgroup)
        else {
            throw Gemma4Error.badTensorGeometry(
                "illegal cooperative tile width \(threads) for \(name)")
        }
        let width = threads
        if openEnc == nil, autoSegment { try beginSegment() }
        if let deferredEnc = openEnc {
            let brT0 = breakdownInstrumentation ? Self.brNow() : 0
            defer { if breakdownInstrumentation { brEncodeMicros += Self.brNow() - brT0 } }
            deferredEnc.setComputePipelineState(pipeline)
            bind(deferredEnc)
            deferredEnc.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            dispatchCountTotal += 1
            return
        }
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Gemma4Error.metalUnavailable
        }
        enc.setComputePipelineState(pipeline)
        bind(enc)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        submittedBuffers += 1
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Gemma4Error.commandBufferFailed(
                "kernel \(name): status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        let delta = (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
        gpuMicrosTotal += delta
        kernelGpuMicrosByName[name, default: (0, 0)].0 += delta
        kernelGpuMicrosByName[name]!.1 += 1
        dispatchCountTotal += 1
    }

    // MARK: - GPU primitives (each preflights byte bounds before dispatch)

    /// y[yOff...] = W·x for a quantized/F32 2-D weight tensor slice
    /// (`rows` rows of `inDim` values, row-major blocks). `extraOffset`
    /// selects an expert slice of a 3-D tensor. Byte bounds are checked
    /// against the tensor's own file extent before anything is encoded.
    internal func matvec(tensor: String, extraOffset: Int = 0,
                        x: MTLBuffer, xCount: Int,
                        y: MTLBuffer, yCapacity: Int,
                        rows: Int? = nil,
                        yByteOffset: Int = 0) throws {
        guard let r = ref[tensor] else { throw Gemma4Error.badTensorGeometry("missing \(tensor)") }
        // In cache mode, expert weights stream exclusively through
        // expertMatvec — an explicit name rejection here (the file-
        // coordinate mapEnd bound alone could numerically overlap the
        // compact length for early-layer experts and read wrong bytes).
        if expertCache != nil,
           tensor.hasSuffix(".ffn_gate_up_exps.weight")
                || tensor.hasSuffix(".ffn_down_exps.weight") {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor) must route through expertMatvec in cache mode")
        }
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
        try matvecDispatch(tensor: tensor, type: r.type, wbase: wb, wStart: wStart,
                           inDim: r.inDim, outRows: outRows,
                           x: x, y: y, yByteOffset: yByteOffset)
    }

    /// Kernel selection + dispatch shared by the resident matvec and the
    /// expert-cache slice matvec. `wbase` + `wStart` address the weight
    /// bytes (the mapped file / compacted buffer, or an expert slot
    /// pool); extent checks are the CALLER's job.
    internal func matvecDispatch(tensor: String, type: GGMLTensorType,
                                 wbase: MTLBuffer, wStart: Int, inDim: Int,
                                 outRows: Int, x: MTLBuffer, y: MTLBuffer,
                                 yByteOffset: Int) throws {
        let kernel: String
        switch type {
        case .q4_0: kernel = "g4_q4_matvec"
        case .q6_K: kernel = "g4_q6_matvec"
        case .float32: kernel = "g4_f32_matvec"
        case .float16: throw Gemma4Error.badTensorGeometry("F16 unsupported for \(tensor)")
        }
        var p = SIMD2<UInt32>(UInt32(inDim), UInt32(outRows))
        var off = UInt64(wStart)
        let gpuBefore = gpuMicrosTotal
        func bind(_ enc: MTLComputeCommandEncoder) {
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wbase, offset: 0, index: 2)
            enc.setBuffer(x, offset: 0, index: 3)
            enc.setBuffer(y, offset: yByteOffset, index: 4)
        }
        if type == .q4_0 && q4TiledKernel {
            // Tiled cooperative dispatch: one threadgroup per row with an
            // explicit hardware-derived width.
            try dispatchExplicit("g4_q4_matvec_tiled", groups: outRows,
                                 threads: q4TileWidth, bind)
        } else if type == .q6_K && q6TiledKernel {
            try dispatchExplicit("g4_q6_matvec_tiled", groups: outRows,
                                 threads: q6TileWidth, bind)
        } else if type == .float32 && f32TiledKernel {
            try dispatchExplicit("g4_f32_matvec_tiled", groups: outRows,
                                 threads: f32TileWidth, bind)
        } else {
            try dispatch(kernel, grid: outRows, bind)
        }
        // Per-tensor GPU attribution (exact in per-kernel mode; in
        // deferred-segment mode the whole segment's time lands on the
        // tensor that triggered the flush — documented limitation).
        matvecGpuMicrosByTensor[tensor, default: 0] += gpuMicrosTotal - gpuBefore
        matvecCallsByTensor[tensor, default: 0] += 1
    }

    // MARK: - bounded expert cache (SSD/streaming mode)

    /// Bounded LRU cache of expert weight slices. Non-nil after
    /// `enableExpertCache`: the whole-file mapping is dropped, non-
    /// expert weights live in a compacted GPU buffer, and expert
    /// slices stream through the cache with a byte budget.
    internal var expertCache: Gemma4ExpertCache?
    /// Synchronous free-slots-only read-ahead of the current layer's
    /// routed expert slices (cache mode; never evicts). Disable for A/B
    /// measurement.
    public var expertPrefetch = true

    /// Read-ahead for the current layer's routed experts into FREE
    /// cache slots only (synchronous blocking preads between expert
    /// iterations — no async I/O/GPU overlap is claimed; never evicts,
    /// so it cannot change any computation). No-op in resident mode or
    /// when disabled.
    internal func prefetchExperts(layer: Int, topIdx: [Int]) throws {
        guard let cache = expertCache, expertPrefetch else { return }
        guard layer >= 0, layer < config.layerCount,
              let gu = ref["blk.\(layer).ffn_gate_up_exps.weight"],
              let dn = ref["blk.\(layer).ffn_down_exps.weight"] else {
            throw Gemma4Error.badTensorGeometry("prefetch refs missing layer \(layer)")
        }
        for e in topIdx {
            guard e >= 0, e < config.expertCount else {
                throw Gemma4Error.badTensorGeometry("prefetch expert \(e) out of bounds")
            }
            // Checked extents within each tensor's FILE range.
            guard let guMul = Checked.mul(e, cache.guStrideBytes),
                  let guStart = Checked.add(gu.offset, guMul),
                  let guEnd = Checked.add(guStart, cache.guStrideBytes),
                  guEnd <= gu.end,
                  let dnMul = Checked.mul(e, cache.dnStrideBytes),
                  let dnStart = Checked.add(dn.offset, dnMul),
                  let dnEnd = Checked.add(dnStart, cache.dnStrideBytes),
                  dnEnd <= dn.end else {
                throw Gemma4Error.badTensorGeometry("prefetch extent layer \(layer)")
            }
            _ = try cache.prefetch(key: layer * config.expertCount + e,
                                   family: .gateUp, fileOffset: guStart)
            _ = try cache.prefetch(key: layer * config.expertCount + e,
                                   family: .down, fileOffset: dnStart)
        }
    }

    /// Expert-slice matvec through the bounded cache: resolves the
    /// (layer, expert) slice — hit, or a checked pread miss that may
    /// evict the family's LRU unpinned slot — then dispatches the SAME
    /// quantized kernels against the slot pool bytes.
    internal func expertMatvec(tensor: String, layer: Int, expert: Int,
                               x: MTLBuffer, xCount: Int,
                               y: MTLBuffer, yCapacity: Int) throws {
        guard let cache = expertCache else {
            throw Gemma4Error.badTensorGeometry("expert cache not enabled for \(tensor)")
        }
        // Exact name contract: only the two expert families, addressed
        // by their own layer prefix (the tensor name IS the key's layer
        // coordinate — a mismatched layer argument is rejected).
        let family: Gemma4ExpertCache.Family
        if tensor == "blk.\(layer).ffn_gate_up_exps.weight" {
            family = .gateUp
        } else if tensor == "blk.\(layer).ffn_down_exps.weight" {
            family = .down
        } else {
            throw Gemma4Error.badTensorGeometry(
                "expertMatvec requires blk.<layer>.ffn_{gate_up,down}_exps.weight, got \(tensor)")
        }
        guard let r = ref[tensor], r.type == .q4_0 else {
            throw Gemma4Error.badTensorGeometry("missing/mistyped expert tensor \(tensor)")
        }
        guard expert >= 0, expert < r.experts, layer >= 0,
              layer < config.layerCount else {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor): expert \(expert) layer \(layer) out of bounds")
        }
        let blocksPerRow = r.inDim / r.type.elementsPerBlock
        let (bytesPerRow, of1) = blocksPerRow.multipliedReportingOverflow(by: r.type.bytesPerBlock)
        let (stride, of2) = of1 ? (0, true) : r.outDim.multipliedReportingOverflow(by: bytesPerRow)
        guard !of1, !of2, stride > 0,
              stride == (family == .gateUp ? cache.guStrideBytes
                                           : cache.dnStrideBytes),
              xCount == r.inDim, x.length >= r.inDim * 4,
              y.length >= r.outDim * 4, yCapacity >= r.outDim else {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor) expert \(expert): slice/buffer geometry mismatch")
        }
        let (prod, ofMul) = expert.multipliedReportingOverflow(by: stride)
        let (wStart, of3) = ofMul ? (0, true) : r.offset.addingReportingOverflow(prod)
        let (wEnd, of4) = of3 ? (0, true) : wStart.addingReportingOverflow(stride)
        guard !ofMul, !of3, !of4, wStart >= r.offset, wEnd <= r.end else {
            throw Gemma4Error.badTensorGeometry(
                "\(tensor) expert \(expert): slice [\(wStart), \(wEnd)) beyond tensor extent")
        }
        let slice = try cache.slice(key: layer * r.experts + expert,
                                    family: family, fileOffset: wStart) { [self] in
            try flushOpenSegment()
        }
        try matvecDispatch(tensor: tensor, type: r.type, wbase: slice.buffer,
                           wStart: slice.offset, inDim: r.inDim, outRows: r.outDim,
                           x: x, y: y, yByteOffset: 0)
    }

    /// Post-construction switch to SSD/streaming expert mode. NOTE: on
    /// a RESIDENT runner this transiently holds the whole-file mapping
    /// AND the compact buffer until the atomic switch below; for a
    /// bounded startup peak use `init(fileAt:expertCacheBudgetBytes:)`
    /// which never creates the whole-file mapping.
    public func enableExpertCache(budgetBytes: Int) throws {
        guard expertCache == nil else {
            throw Gemma4Error.badMetadata("expert cache already enabled")
        }
        try switchToExpertCache(budgetBytes: budgetBytes)
    }

    /// Shared switch used by the bounded-startup init and the post-hoc
    /// `enableExpertCache`:
    /// 1. every NON-expert tensor is compacted into a dedicated GPU
    ///    buffer with checked chunked preads;
    /// 2. the two expert tensor families stream through the bounded
    ///    LRU cache within `budgetBytes` (pool allocation is the
    ///    budget; pinned/in-flight slots are inside those pools).
    /// Fail-closed throughout: nothing observable is mutated until
    /// every step succeeded; a resident runner keeps its whole-file
    /// mapping intact on error.
    private func switchToExpertCache(budgetBytes: Int) throws {
        guard expertCache == nil else {
            throw Gemma4Error.badMetadata("expert cache already enabled")
        }
        guard let url = ggufFile.sourceURL else {
            throw Gemma4Error.badMetadata("GGUF source URL unknown — cannot stream expert slices")
        }
        guard budgetBytes > 0 else {
            throw Gemma4Error.badTensorGeometry("expert cache budget must be positive")
        }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Gemma4Error.mmapFailed(errno: errno) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 0 else {
            throw Gemma4Error.mmapFailed(errno: errno)
        }
        let fileSize = Int(st.st_size)
        // Uniform per-layer expert slice strides (single slot pool pair).
        func q4Stride(_ r: G4TensorRef) -> Int? {
            guard r.type == .q4_0, r.inDim % 32 == 0 else { return nil }
            let (bytesPerRow, o1) = (r.inDim / 32).multipliedReportingOverflow(by: 18)
            guard !o1 else { return nil }
            let (s, o2) = r.outDim.multipliedReportingOverflow(by: bytesPerRow)
            return o2 ? nil : s
        }
        guard let gu0 = ref["blk.0.ffn_gate_up_exps.weight"],
              let dn0 = ref["blk.0.ffn_down_exps.weight"],
              let guStride = q4Stride(gu0), guStride > 0,
              let dnStride = q4Stride(dn0), dnStride > 0,
              gu0.experts == config.expertCount,
              dn0.experts == config.expertCount else {
            throw Gemma4Error.badTensorGeometry("expert tensor geometry invalid")
        }
        for layer in 0..<config.layerCount {
            guard let g = ref["blk.\(layer).ffn_gate_up_exps.weight"],
                  let d = ref["blk.\(layer).ffn_down_exps.weight"],
                  q4Stride(g) == guStride, q4Stride(d) == dnStride else {
                throw Gemma4Error.badTensorGeometry("expert strides differ across layers")
            }
        }
        // ---- budget/device preflight BEFORE any large allocation.
        guard ExpertCacheGeometry.slotCounts(budgetBytes: budgetBytes,
                                             guStrideBytes: guStride,
                                             dnStrideBytes: dnStride) != nil,
              guStride <= device.maxBufferLength, dnStride <= device.maxBufferLength,
              budgetBytes <= device.maxBufferLength else {
            throw Gemma4Error.exceedsMaxBufferLength(
                needed: max(budgetBytes, max(guStride, dnStride)),
                max: device.maxBufferLength)
        }
        // ---- compact every non-expert tensor into one GPU buffer.
        func isExpertWeight(_ name: String) -> Bool {
            name.hasSuffix(".ffn_gate_up_exps.weight")
                || name.hasSuffix(".ffn_down_exps.weight")
        }
        var ranges: [(name: String, start: Int, end: Int)] = []
        var total = 0
        for info in ggufFile.tensors where !isExpertWeight(info.name) {
            let range = try ggufFile.tensorRange(info)
            let (t, of) = total.addingReportingOverflow(range.upperBound - range.lowerBound)
            guard !of else { throw Gemma4Error.badTensorGeometry("compaction overflow") }
            ranges.append((info.name, range.lowerBound, range.upperBound))
            total = t
        }
        guard total <= device.maxBufferLength else {
            throw Gemma4Error.exceedsMaxBufferLength(needed: total, max: device.maxBufferLength)
        }
        guard let compact = device.makeBuffer(length: total, options: .storageModeShared) else {
            throw Gemma4Error.metalUnavailable
        }
        var base = 0
        let chunk = 1 << 24
        for r in ranges {
            let len = r.end - r.start
            var done = 0
            while done < len {
                let want = min(chunk, len - done)
                let got = pread(fd, compact.contents() + base + done, want,
                                off_t(r.start + done))
                guard got > 0 else { throw Gemma4Error.mmapFailed(errno: errno) }
                done += got
            }
            base += len
        }
        // Remap non-expert refs into a LOCAL table first — nothing is
        // mutated until every step succeeded (fail-closed: an error
        // above leaves the whole-file resident mode fully intact).
        // Expert refs KEEP their file coordinates (the cache preads
        // from them) and are rejected by name in `matvec`.
        var newRef = ref
        var cumulative = 0
        for r in ranges {
            let len = r.end - r.start
            let old = newRef[r.name]!
            newRef[r.name] = G4TensorRef(offset: cumulative, end: cumulative + len,
                                         inDim: old.inDim, outDim: old.outDim,
                                         experts: old.experts, type: old.type)
            cumulative += len
        }
        let cache = try Gemma4ExpertCache(device: device, fileURL: url,
                                          ggufSize: fileSize,
                                          guStrideBytes: guStride,
                                          dnStrideBytes: dnStride,
                                          budgetBytes: budgetBytes)
        cache.nonExpertResidentBytes = total
        // Complete any pending encoded GPU work against the OLD weight
        // mapping before it is dropped.
        try flushOpenSegment()
        // ---- atomic switch (no throw past this point).
        ref = newRef
        expertCache = cache
        // The whole-file mapping is released by the MTLBuffer
        // deallocator — expert weights leave the GPU address space.
        weightsBuffer = compact
        mapLength = total
        mapEnd = total
    }

    /// Snapshot of the expert-cache counters (nil in resident mode).
    public func expertCacheStats() -> Gemma4ExpertCacheStats? {
        expertCache?.stats()
    }

    /// Point sample of MTLDevice.currentAllocatedSize. NOT a peak or
    /// residency measurement: the Metal allocator may retain freed bytes
    /// and lazy allocations may not have happened yet.
    public func metalAllocatedSample() -> Int {
        Int(clamping: device.currentAllocatedSize)
    }

    /// Per-vector RMSNorm over `vectors` contiguous vectors of length `n`
    /// (1 thread per vector; in-place x==y is safe). `weightTensor` nil =
    /// scale-free (router norm, v_norm).
    internal func rmsnorm(x: MTLBuffer, y: MTLBuffer, n: Int, vectors: Int,
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
        let bindNorm: (MTLComputeCommandEncoder) -> Void = { enc in
            enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 1)
            enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
            enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
            enc.setBuffer(wb, offset: 0, index: 4)
            enc.setBuffer(x, offset: 0, index: 5)
            enc.setBuffer(y, offset: 0, index: 6)
        }
        if rmsTiledKernel {
            // Tiled cooperative variant (Forge candidate; the single-
            // thread-per-vector reduce is the measured 44% GPU hotspot).
            try dispatchExplicit("g4_rmsnorm_tiled", groups: vectors,
                                 threads: rmsTileWidth, bindNorm)
        } else if vectorizedLoads && Self.rmsVectorLoadGuards(
                n: n, scaled: weightTensor != nil, wOff: wOff) {
            // Vectorized-load variant: identical arithmetic order, float4
            // loads. Guards (one production decision point, also tested):
            // the width must be a float4 multiple, and a scaled weight
            // tensor must sit at a 16-byte aligned offset in the packed
            // weight buffer (x/y offsets are vec*n floats, so n % 4 == 0
            // already aligns them). Otherwise: original kernel.
            try dispatch("g4_rmsnorm_vl", grid: vectors, bindNorm)
        } else {
            try dispatch("g4_rmsnorm", grid: vectors, bindNorm)
        }
    }

    /// In-place RoPE over `heads` vectors of headDim = 2*half values
    /// (1 thread per head; pairs (i, i+half) rotated by inv[i] * position).
    internal func rope(x: MTLBuffer, heads: Int, half: Int, position: Int,
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
    internal func geluUp(gu: MTLBuffer, y: MTLBuffer, n: Int) throws {
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

    internal func axpy(x: MTLBuffer, y: MTLBuffer, c: Float, n: Int) throws {
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

    internal func add(a: MTLBuffer, b: MTLBuffer, y: MTLBuffer, n: Int) throws {
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

    internal func scaleInPlace(x: MTLBuffer, c: Float, n: Int) throws {
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

    internal func read(_ buffer: MTLBuffer, _ count: Int) throws -> [Float] {
        // A deferred segment must be committed (and verified) before CPU
        // readback — a failed segment must surface, not vanish.
        try flushOpenSegment()
        let brT0 = breakdownInstrumentation ? Self.brNow() : 0
        defer { if breakdownInstrumentation { brReadCopyMicros += Self.brNow() - brT0 } }
        return [Float](UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self),
                                           count: count))
    }

    internal func write(_ values: [Float], to buffer: MTLBuffer) throws {
        try flushOpenSegment()
        let brT0 = breakdownInstrumentation ? Self.brNow() : 0
        defer { if breakdownInstrumentation { brWriteCopyMicros += Self.brNow() - brT0 } }
        _ = values.withUnsafeBytes { raw in
            memcpy(buffer.contents(), raw.baseAddress!, values.count * 4)
        }
    }

    // MARK: - forward pass

    /// Current context length (tokens whose KV is populated).
    internal func contextPosition() -> Int { consumedPosition }

    /// Read-only tile-width metadata for reporting/identity (the
    /// The tiled-kernel flags are public for the profiling CLI (which
    /// measures selected kernels with exact per-kernel attribution);
    /// Forge candidates set them only through the gated executor.
    public var tileWidths: (q4: Int, q6: Int, rms: Int, f32: Int) {
        (q4TileWidth, q6TileWidth, rmsTileWidth, f32TileWidth)
    }

    /// Public bounded accessor for the final logits vector (flushes any
    /// open deferred segment first).
    public func currentLogits() throws -> [Float] {
        try read(bufLogits, config.vocabSize)
    }

    /// Snapshot of the per-KERNEL GPU-time attribution (per-kernel mode).
    public func kernelGpuProfile() -> [(kernel: String, gpuMicros: Double, calls: Int)] {
        kernelGpuMicrosByName
            .map { (kernel: $0.key, gpuMicros: $0.value.0, calls: $0.value.1) }
            .sorted { $0.gpuMicros > $1.gpuMicros }
    }

    /// Snapshot of the matvec GPU-time attribution (per-kernel mode),
    /// sorted by descending GPU time.
    public func matvecGpuProfile() -> [(tensor: String, gpuMicros: Double, calls: Int)] {
        matvecGpuMicrosByTensor
            .map { (tensor: $0.key, gpuMicros: $0.value,
                    calls: matvecCallsByTensor[$0.key] ?? 0) }
            .sorted { $0.gpuMicros > $1.gpuMicros }
    }

    /// SHA-256 of the RUNNING EXECUTABLE's bytes (bounded streaming, read
    /// once per process): any rebuilt binary — Swift executor, lowering,
    /// pass policy, allocation — changes this digest, so plan caches
    /// invalidate automatically on engine rebuilds. FAILS CLOSED: if the
    /// real binary bytes cannot be resolved/hashed, an error propagates
    /// and no plan identity is produced (no placeholder digest).
    static func executableDigest() throws -> String {
        guard let exec = Bundle.main.executableURL else {
            throw Gemma4Error.metalUnavailable
        }
        let digest = try GemmaForge.fileDigest(exec)
        guard digest.count == 64,
              digest.allSatisfy({ ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") })
        else {
            throw Gemma4Error.badMetadata("executable digest malformed")
        }
        return digest
    }

    /// Identity of the pass engine: the running executable's content
    /// digest (automatic build identity), the SHA-256 of the ACTUAL
    /// embedded MSL source (compiled via makeLibrary(source:, options:
    /// nil) — DEFAULT compiler options, recorded here as options=default;
    /// we make no claim about what the default set does or does not
    /// enable), the compiled pipeline set, the SSA schedule schema, and
    /// the candidate-set signature.
    public func engineIdentity() throws -> String {
        let msl = SHA256.hex(SHA256.hash(Array(Self.shaderSource.utf8)))
        let candidates = GemmaForge.candidates()
            .map { "\($0.segmentBatching)/\($0.matvecTiledQ4)/\($0.matvecTiledQ6)/\($0.rmsnormTiled)/\($0.matvecTiledF32)/\($0.vectorizedLoads)" }
            .sorted().joined(separator: ",")
        return "gemma-graph-executor-v1|exe=" + (try Self.executableDigest())
            + "|msl=" + msl
            + "|pipelines=" + pipelines.keys.sorted().joined(separator: ",")
            + "|ssaSchema=\(GemmaSSA.schemaVersion)"
            + "|candidates=" + candidates
            + "|q4TileWidth=\(q4TileWidth)"
            + "|q6TileWidth=\(q6TileWidth)"
            + "|rmsTileWidth=\(rmsTileWidth)"
            + "|f32TileWidth=\(f32TileWidth)"
            + "|mslOptions=default"
            + "|expertCacheBudget=\(expertCache?.budgetBytes ?? 0)"
            + "|expertPrefetch=\(expertCache != nil && expertPrefetch ? "free-only" : "none")"
    }

    /// Measured hardware profile of the device this runner holds.
    public func hardwareProfile() -> ForgeHardwareProfile {
        ForgeHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: pipelines.values.map { $0.maxTotalThreadsPerThreadgroup }.min() ?? 0,
            threadExecutionWidth: pipelines.values.map { $0.threadExecutionWidth }.min() ?? 0,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }

    /// Advances the context by one externally-executed token (graph
    /// executor decode continuation).
    internal func advancePosition() { consumedPosition += 1 }

    /// Resets the KV caches' consumed position; the next `forward` starts a
    /// fresh context at position 0. (Cache bytes need no clearing: only
    /// positions < consumedPosition are ever read.)
    public func reset() {
        discardOpenSegment()
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
        let logits = try read(bufLogits, config.vocabSize)
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
    internal func decodeEmbedding(_ token: Int) throws {
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
        let gpu = try read(bufEmbed, hidden)
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
        let embVals = try read(bufEmbed, hidden)
        guard embVals.allSatisfy({ $0.isFinite }) else {
            throw Gemma4Error.nonFiniteActivation("embedding row \(token)")
        }
        try write(embVals, to: bufX)

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
        let logits = try read(bufLogits, config.vocabSize)
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
            let kRaw = try read(bufT2, kvWidth)
            try write(kRaw, to: bufKV)
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
        let kNormed = try read(bufT2, kvWidth)
        let vNormed = try read(bufKV, kvWidth)
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
            let gu = try read(bufGU, 2 * config.sharedIntermediate)
            let act = try read(bufAct, config.sharedIntermediate)
            let dn = try read(bufShared, hidden)
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
            let s = try read(bufShared, hidden)
            print("  layer \(layer) shared mlp: nonfinite \(s.lazy.filter { !$0.isFinite }.count), " +
                  "max \(s.lazy.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0)")
        }

        // Router on the residual (pre-norm) stream: scale-free RMSNorm,
        // * scale * hidden^-0.5, F32 router matvec, fp64 CPU softmax,
        // top-8 (ties -> lower index), renormalize, * per-expert QAT scale.
        try rmsnorm(x: bufX, y: bufH, n: hidden, vectors: 1, weightTensor: nil)
        try scaleInPlace(x: bufH, c: Float(1.0 / sqrt(Double(hidden))), n: hidden)
        var routerIn = try read(bufH, hidden)
        let routerScale = routerScales[layer]
        for i in 0..<hidden { routerIn[i] *= routerScale[i] }
        try write(routerIn, to: bufH)
        try matvec(tensor: prefix + "ffn_gate_inp.weight", x: bufH, xCount: hidden,
                   y: bufRouter, yCapacity: config.expertCount)
        let scores = try read(bufRouter, config.expertCount)
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
        // Synchronous free-slots-only read-ahead of this layer's routed
        // experts (cache mode only; never evicts, no async overlap).
        try prefetchExperts(layer: layer, topIdx: topIdx)
        memset(bufT3.contents(), 0, hidden * 4)   // MoE accumulator
        for (slot, expert) in topIdx.enumerated() {
            if expertCache != nil {
                try expertMatvec(tensor: prefix + "ffn_gate_up_exps.weight",
                                 layer: layer, expert: expert,
                                 x: bufH, xCount: hidden,
                                 y: bufGU, yCapacity: 2 * config.sharedIntermediate)
                try geluUp(gu: bufGU, y: bufAct, n: e)
                try expertMatvec(tensor: prefix + "ffn_down_exps.weight",
                                 layer: layer, expert: expert,
                                 x: bufAct, xCount: e,
                                 y: bufT2, yCapacity: max(hidden, 2048))
            } else {
                try matvec(tensor: prefix + "ffn_gate_up_exps.weight",
                           extraOffset: expert * guStride,
                           x: bufH, xCount: hidden,
                           y: bufGU, yCapacity: 2 * config.sharedIntermediate)
                try geluUp(gu: bufGU, y: bufAct, n: e)
                try matvec(tensor: prefix + "ffn_down_exps.weight",
                           extraOffset: expert * dnStride,
                           x: bufAct, xCount: e,
                           y: bufT2, yCapacity: max(hidden, 2048))
            }
            try axpy(x: bufT2, y: bufT3, c: weights[slot], n: hidden)
            expertCache?.endExpertIteration(segmentPending: openEnc != nil)
        }
        try rmsnorm(x: bufT3, y: bufT3, n: hidden, vectors: 1,
                    weightTensor: prefix + "post_ffw_norm_2.weight")
        if traceLayers {
            let m = try read(bufT3, hidden)
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
            let x = try read(bufX, hidden)
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

    public static let shaderSource = """
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

    // MEASURED-REJECTED vectorized-load variant of g4_q4_matvec (kept
    // for the test diagnostic ONLY — never in the production pipeline
    // list or legal dispatch). The 18-byte Q4_0 block stride forbids
    // aligned vector loads of the quant bytes, so the q loads stay
    // scalar; the 32 activations per block load as eight float4s. The
    // source arithmetic sequence is identical to g4_q4_matvec, yet the
    // COMPILED kernels produced different outputs on real hardware
    // (measured, worst rel printed by the test suite; cause not
    // established), so bit-equality could not be proven and the variant
    // was rejected despite a measured ~2.1x per-call speedup.
    kernel void g4_q4_matvec_vl(constant const uint2& p [[buffer(0)]],
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
            float4 xa = *(device const float4*)(x + b * 32u);
            float4 xb2 = *(device const float4*)(x + b * 32u + 4u);
            float4 xc = *(device const float4*)(x + b * 32u + 8u);
            float4 xd = *(device const float4*)(x + b * 32u + 12u);
            float4 xe = *(device const float4*)(x + b * 32u + 16u);
            float4 xf = *(device const float4*)(x + b * 32u + 20u);
            float4 xg = *(device const float4*)(x + b * 32u + 24u);
            float4 xh = *(device const float4*)(x + b * 32u + 28u);
            sum += d * (float(blk[2u] & 0x0Fu) - 8.0f) * xa.x;
            sum += d * (float(blk[2u] >> 4) - 8.0f) * xe.x;
            sum += d * (float(blk[3u] & 0x0Fu) - 8.0f) * xa.y;
            sum += d * (float(blk[3u] >> 4) - 8.0f) * xe.y;
            sum += d * (float(blk[4u] & 0x0Fu) - 8.0f) * xa.z;
            sum += d * (float(blk[4u] >> 4) - 8.0f) * xe.z;
            sum += d * (float(blk[5u] & 0x0Fu) - 8.0f) * xa.w;
            sum += d * (float(blk[5u] >> 4) - 8.0f) * xe.w;
            sum += d * (float(blk[6u] & 0x0Fu) - 8.0f) * xb2.x;
            sum += d * (float(blk[6u] >> 4) - 8.0f) * xf.x;
            sum += d * (float(blk[7u] & 0x0Fu) - 8.0f) * xb2.y;
            sum += d * (float(blk[7u] >> 4) - 8.0f) * xf.y;
            sum += d * (float(blk[8u] & 0x0Fu) - 8.0f) * xb2.z;
            sum += d * (float(blk[8u] >> 4) - 8.0f) * xf.z;
            sum += d * (float(blk[9u] & 0x0Fu) - 8.0f) * xb2.w;
            sum += d * (float(blk[9u] >> 4) - 8.0f) * xf.w;
            sum += d * (float(blk[10u] & 0x0Fu) - 8.0f) * xc.x;
            sum += d * (float(blk[10u] >> 4) - 8.0f) * xg.x;
            sum += d * (float(blk[11u] & 0x0Fu) - 8.0f) * xc.y;
            sum += d * (float(blk[11u] >> 4) - 8.0f) * xg.y;
            sum += d * (float(blk[12u] & 0x0Fu) - 8.0f) * xc.z;
            sum += d * (float(blk[12u] >> 4) - 8.0f) * xg.z;
            sum += d * (float(blk[13u] & 0x0Fu) - 8.0f) * xc.w;
            sum += d * (float(blk[13u] >> 4) - 8.0f) * xg.w;
            sum += d * (float(blk[14u] & 0x0Fu) - 8.0f) * xd.x;
            sum += d * (float(blk[14u] >> 4) - 8.0f) * xh.x;
            sum += d * (float(blk[15u] & 0x0Fu) - 8.0f) * xd.y;
            sum += d * (float(blk[15u] >> 4) - 8.0f) * xh.y;
            sum += d * (float(blk[16u] & 0x0Fu) - 8.0f) * xd.z;
            sum += d * (float(blk[16u] >> 4) - 8.0f) * xh.z;
            sum += d * (float(blk[17u] & 0x0Fu) - 8.0f) * xd.w;
            sum += d * (float(blk[17u] >> 4) - 8.0f) * xh.w;
        }
        y[row] = sum;
    }

    // Tiled cooperative variant: one threadgroup computes ONE row; each
    // thread accumulates a strided subset of blocks, then a threadgroup
    // reduction combines the partials. NOTE: the reduction reassociates
    // the addition order, so results are NOT bit-identical to the
    // single-thread kernel — correctness is gated against the
    // independent FP64 oracle with a justified tolerance, and greedy
    // equality is enforced end-to-end by the Forge gates.
    kernel void g4_q4_matvec_tiled(constant const uint2& p [[buffer(0)]],
                                   constant const ulong& wOff [[buffer(1)]],
                                   device const uchar* wbase [[buffer(2)]],
                                   device const float* x [[buffer(3)]],
                                   device float* y [[buffer(4)]],
                                   uint lid [[thread_position_in_threadgroup]],
                                   uint lsizeUInt [[threads_per_threadgroup]],
                                   uint wid [[threadgroup_position_in_grid]]) {
        uint row = wid;
        uint lsize = lsizeUInt;
        if (row >= p.y) { return; }
        uint blocksPerRow = p.x / 32u;
        device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 18u;
        float sum = 0.0f;
        for (uint b = lid; b < blocksPerRow; b += lsize) {
            device const uchar* blk = rowBytes + b * 18u;
            ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
            float d = float(as_type<half>(dbits));
            for (uint l = 0; l < 16u; l++) {
                uchar q = blk[2u + l];
                sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];
                sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];
            }
        }
        threadgroup float partials[128];
        partials[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = lsize / 2u; s > 0u; s >>= 1u) {
            if (lid < s) { partials[lid] += partials[lid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) { y[row] = partials[0]; }
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

    // Tiled cooperative Q6_K variant: same reduction caveat as the Q4_0
    // tiled kernel (reassociated addition — NOT bit-identical; gated
    // against the FP64 oracle and by full-logit/greedy Forge gates).
    kernel void g4_q6_matvec_tiled(constant const uint2& p [[buffer(0)]],
                                   constant const ulong& wOff [[buffer(1)]],
                                   device const uchar* wbase [[buffer(2)]],
                                   device const float* x [[buffer(3)]],
                                   device float* y [[buffer(4)]],
                                   uint lid [[thread_position_in_threadgroup]],
                                   uint lsizeUInt [[threads_per_threadgroup]],
                                   uint wid [[threadgroup_position_in_grid]]) {
        uint row = wid;
        uint lsize = lsizeUInt;
        if (row >= p.y) { return; }
        uint blocksPerRow = p.x / 256u;
        device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 210u;
        float sum = 0.0f;
        for (uint b = lid; b < blocksPerRow; b += lsize) {
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
        threadgroup float partials[128];
        partials[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = lsize / 2u; s > 0u; s >>= 1u) {
            if (lid < s) { partials[lid] += partials[lid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) { y[row] = partials[0]; }
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

    // Tiled cooperative F32 matvec (one threadgroup per row, strided
    // input, threadgroup partials tree reduction). The scalar kernel uses
    // one thread per row, which underparallelizes the MoE router
    // (ffn_gate_inp: outDim = expert count) — a measured GPU-time hotspot.
    // NOT bit-identical to g4_f32_matvec: the reduction reassociates the
    // addition order; gated against the FP64 oracle + full Forge gates.
    // Measured outcome: valid as a standalone candidate, but stacked with
    // the tiled Q4/RMS candidates the full-logit drift reaches 1.066e-04,
    // just over the 1e-4 gate — the Forge therefore does NOT select the
    // stacked combination. A Kahan-compensated variant measured the SAME
    // worst value (and default Metal fast-math may cancel compensation),
    // so the plain reduction is kept; no claim is made about which term
    // dominates the stacked drift.
    kernel void g4_f32_matvec_tiled(constant const uint2& p [[buffer(0)]],
                                    constant const ulong& wOff [[buffer(1)]],
                                    device const uchar* wbase [[buffer(2)]],
                                    device const float* x [[buffer(3)]],
                                    device float* y [[buffer(4)]],
                                    uint lid [[thread_position_in_threadgroup]],
                                    uint lsizeUInt [[threads_per_threadgroup]],
                                    uint wid [[threadgroup_position_in_grid]]) {
        uint row = wid;
        uint lsize = lsizeUInt;
        if (row >= p.y) { return; }
        device const float* w = (device const float*)(wbase + wOff) + (ulong)row * p.x;
        float sum = 0.0f;
        for (uint i = lid; i < p.x; i += lsize) { sum += w[i] * x[i]; }
        threadgroup float partials[128];
        partials[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = lsize / 2u; s > 0u; s >>= 1u) {
            if (lid < s) { partials[lid] += partials[lid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) { y[row] = partials[0]; }
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

    // Vectorized-load variant of g4_rmsnorm: the sum-of-squares and the
    // apply loop consume x (and w) in float4 chunks, accumulating the
    // components in EXACTLY the original scalar order (i, i+1, i+2,
    // i+3). Requires n % 4 == 0 and, when withScale != 0, a 16-byte
    // aligned weight offset — the dispatcher checks both and falls back
    // to g4_rmsnorm otherwise. Bit-identity to the original kernel is
    // asserted on hardware by the test suite.
    kernel void g4_rmsnorm_vl(constant const uint2& p [[buffer(0)]],
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
        for (uint i = 0; i < n; i += 4u) {
            float4 v = *(device const float4*)(xv + i);
            ms += v.x * v.x;
            ms += v.y * v.y;
            ms += v.z * v.z;
            ms += v.w * v.w;
        }
        ms = ms / float(n) + eps;
        float inv = 1.0f / sqrt(ms);
        device const float* w = (device const float*)(wbase + wOff);
        device float* yv = y + (ulong)vec * n;
        if (withScale != 0) {
            for (uint i = 0; i < n; i += 4u) {
                float4 v = *(device const float4*)(xv + i);
                float4 wv = *(device const float4*)(w + i);
                yv[i] = v.x * inv * wv.x;
                yv[i + 1u] = v.y * inv * wv.y;
                yv[i + 2u] = v.z * inv * wv.z;
                yv[i + 3u] = v.w * inv * wv.w;
            }
        } else {
            for (uint i = 0; i < n; i += 4u) {
                float4 v = *(device const float4*)(xv + i);
                yv[i] = v.x * inv;
                yv[i + 1u] = v.y * inv;
                yv[i + 2u] = v.z * inv;
                yv[i + 3u] = v.w * inv;
            }
        }
    }

    // Tiled cooperative RMSNorm: one threadgroup per vector; strided
    // sum-of-squares partials + tree reduction, then strided apply. The
    // reduction reassociates the addition order — NOT bit-identical to
    // the single-thread kernel; gated against a Double oracle and by the
    // full-logit/greedy Forge gates. All threads read the vector fully
    // (phase 1) before any in-place write (phase 2): the barrier between
    // phases makes in-place x == y safe; distinct vectors are disjoint.
    kernel void g4_rmsnorm_tiled(constant const uint2& p [[buffer(0)]],
                                 constant const float& eps [[buffer(1)]],
                                 constant const int& withScale [[buffer(2)]],
                                 constant const ulong& wOff [[buffer(3)]],
                                 device const uchar* wbase [[buffer(4)]],
                                 device const float* x [[buffer(5)]],
                                 device float* y [[buffer(6)]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint lsizeUInt [[threads_per_threadgroup]],
                                 uint wid [[threadgroup_position_in_grid]]) {
        uint vec = wid;
        uint lsize = lsizeUInt;
        if (vec >= p.y) { return; }
        uint n = p.x;
        device const float* xv = x + (ulong)vec * n;
        threadgroup float partials[128];
        float ms = 0.0f;
        for (uint i = lid; i < n; i += lsize) { float v = xv[i]; ms += v * v; }
        partials[lid] = ms;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = lsize / 2u; s > 0u; s >>= 1u) {
            if (lid < s) { partials[lid] += partials[lid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = 1.0f / sqrt(partials[0] / float(n) + eps);
        device const float* w = (device const float*)(wbase + wOff);
        device float* yv = y + (ulong)vec * n;
        for (uint i = lid; i < n; i += lsize) {
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
