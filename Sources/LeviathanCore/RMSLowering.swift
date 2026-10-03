// Phase 10 — parameterized serial-order-preserving cooperative RMSNorm
// lowering (ONE operator family).
//
// Pipeline: Python op input (.py with semantics/shape/layout/precision
// metadata) -> RMSOpIR validation -> shape+hardware-derived schedule
// enumeration (T from the measured execution width, staging granularity from
// the shape) -> Metal source EMISSION from the schedule parameters ->
// independent CPU FP64 oracle gate (existing 1e-4 component contract) AND a
// hardware BIT-EQUALITY gate against the incumbent production kernel
// (g4_rmsnorm_vl) -> RMSLoweringArtifact with full identity binding ->
// persisted execution that genuinely dispatches the emitted, compiled
// pipeline, failing closed to the incumbent kernel.
//
// The reduction preserves the incumbent's serial summation order exactly:
// cooperative threads only STAGE exact copies of the inputs into threadgroup
// memory; lane 0 performs the sum of squares in the original scalar order;
// the output apply is parallel but per-element independent (identical
// arithmetic). See docs/RMS_LOWERING_DESIGN.md. Bounded to the RMSNorm
// family; not a general compiler.

import Foundation
import CryptoKit
import Metal

func rmsIsLowercaseHex64(_ s: String) -> Bool {
    s.count == 64 && s.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") }
}

/// TRUE bit-level equality of two Float arrays: every element's bitPattern
/// must match, so -0.0 / +0.0 (and any other distinguished bit pattern)
/// compare as DIFFERENT — numerical `==` would silently accept a sign flip.
public func rmsBitEqual(_ a: [Float], _ b: [Float]) -> Bool {
    guard a.count == b.count else { return false }
    for i in a.indices where a[i].bitPattern != b[i].bitPattern { return false }
    return true
}

public enum RMSLoweringError: Error, CustomStringConvertible, Equatable {
    case invalidOp(String)
    case invalidSchedule(String)
    case emissionFailed(String)
    case oracleFailed(String)
    case componentGateFailed(String)
    case invalidArtifact(String)
    case hardwareUnavailable(String)

    public var description: String {
        switch self {
        case .invalidOp(let m): return "rms op invalid: \(m)"
        case .invalidSchedule(let m): return "rms schedule invalid: \(m)"
        case .emissionFailed(let m): return "rms emission failed: \(m)"
        case .oracleFailed(let m): return "rms oracle failed: \(m)"
        case .componentGateFailed(let m): return "rms component gate failed: \(m)"
        case .invalidArtifact(let m): return "rms artifact invalid: \(m)"
        case .hardwareUnavailable(let m): return "rms hardware unavailable: \(m)"
        }
    }
}

// MARK: - Operation IR

/// The validated RMSNorm operation: y[i] = x[i] * inv_rms(x) * w[i] over
/// contiguous vectors of n float32 values. `vectors` is the CAPTURE-TIME
/// component batch (bounding representativeVectors); the runtime dispatch
/// is per-call — the installed generated kernel serves whatever vector
/// count rmsnorm() passes for the bound (n, scaled) family. Parsed from
/// the Python-captured op JSON; every field is checked and anything
/// outside the supported operation is rejected before any lowering happens.
public struct RMSOpIR: Codable, Equatable, Sendable {
    public let schema: Int
    public let op: String
    public let semantics: String
    public let n: Int
    public let vectors: Int
    /// Whether the per-channel weight w is applied (false: w == 1).
    public let scaled: Bool
    public let eps: Double
    public let layout: String
    public let quant: String
    public let reduction: String
    public let inputPrecision: String
    /// Optional real GGUF tensor binding for representative weight bytes;
    /// when present the tensor MUST be F32 with dims == [n].
    public let tensor: String?
    /// Bounded representative vector count for component validation
    /// (1...vectors, bounded to 64).
    public let representativeVectors: Int?

    public static let supportedSchema = 1
    public static let maxN = 1 << 14

    public static func parse(_ data: Data) throws -> RMSOpIR {
        guard data.count <= 1 << 16 else {
            throw RMSLoweringError.invalidOp("op JSON exceeds 64 KiB")
        }
        let op: RMSOpIR
        do { op = try JSONDecoder().decode(RMSOpIR.self, from: data) }
        catch { throw RMSLoweringError.invalidOp("op JSON decode: \(error)") }
        try op.validate()
        return op
    }

    public func validate() throws {
        guard schema == Self.supportedSchema else {
            throw RMSLoweringError.invalidOp("schema \(schema) != \(Self.supportedSchema)")
        }
        guard op == "rmsnorm",
              semantics == "y[i] = x[i] * inv_rms(x) * w[i]" else {
            throw RMSLoweringError.invalidOp("unsupported op/semantics \(op)/\(semantics)")
        }
        guard layout == "row-major-f32-vectors", quant == "f32",
              reduction == "serial-order-i0..n", inputPrecision == "f32" else {
            throw RMSLoweringError.invalidOp(
                "unsupported layout/quant/reduction/precision \(layout)/\(quant)/\(reduction)/\(inputPrecision)")
        }
        guard n > 0, n % 4 == 0, n <= Self.maxN, vectors > 0, vectors <= 4096 else {
            throw RMSLoweringError.invalidOp("bad shape n \(n) vectors \(vectors)")
        }
        guard eps.isFinite, eps >= 0, eps <= 1e-3 else {
            throw RMSLoweringError.invalidOp("bad eps \(eps)")
        }
        if let reps = representativeVectors {
            guard reps > 0, reps <= vectors, reps <= 64 else {
                throw RMSLoweringError.invalidOp("representativeVectors \(reps) out of bounds")
            }
        }
    }

    /// Cross-checks the declared shape against the BOUND tensor's actual
    /// GGUF dimensions and type. Fail-closed on any mismatch.
    public func validateAgainst(file: GGUFFile) throws {
        guard let tensor else { return }
        guard let info = try? file.tensor(tensor) else {
            throw RMSLoweringError.invalidOp("bound tensor \(tensor) not in GGUF")
        }
        guard info.ggmlType == 0 else {
            throw RMSLoweringError.invalidOp(
                "bound tensor \(tensor) ggmlType \(info.ggmlType) is not F32")
        }
        guard info.dims == [n] else {
            throw RMSLoweringError.invalidOp(
                "bound tensor \(tensor) dims \(info.dims) != declared [\(n)]")
        }
    }

    /// SHA-256 over the canonical (sorted-keys) JSON encoding.
    public func digest() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(self)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    /// The bounded vector count actually evaluated at the component level.
    public var componentVectors: Int { min(representativeVectors ?? 8, vectors, 64) }
}

// MARK: - Hardware profile (RMS-specific: includes threadgroup memory cap)

public struct RMSHardwareProfile: Codable, Equatable, Sendable {
    public let deviceName: String
    public let maxTotalThreadsPerThreadgroup: Int
    public let threadExecutionWidth: Int
    public let maxThreadgroupMemoryLength: Int
    public let recommendedMaxWorkingSetSize: Int

    public init(deviceName: String, maxTotalThreadsPerThreadgroup: Int,
                threadExecutionWidth: Int, maxThreadgroupMemoryLength: Int,
                recommendedMaxWorkingSetSize: Int) {
        self.deviceName = deviceName
        self.maxTotalThreadsPerThreadgroup = maxTotalThreadsPerThreadgroup
        self.threadExecutionWidth = threadExecutionWidth
        self.maxThreadgroupMemoryLength = maxThreadgroupMemoryLength
        self.recommendedMaxWorkingSetSize = recommendedMaxWorkingSetSize
    }
}

// MARK: - Schedule

/// One parameterized lowering schedule for the RMSNorm family. Every field
/// has a REAL effect on the emitted Metal source: the cooperative thread
/// mapping (threadsPerVector drives the staging and apply strides) and the
/// memory access width (staging granularity).
public struct RMSSchedule: Codable, Equatable, Hashable, Sendable {
    public enum Staging: String, Codable, Sendable {
        case scalar
        case float4
    }

    public let threadsPerVector: Int
    public let staging: Staging

    public init(threadsPerVector: Int, staging: RMSSchedule.Staging) {
        self.threadsPerVector = threadsPerVector
        self.staging = staging
    }

    public var kernelName: String {
        "rmsgen_rms_T\(threadsPerVector)_\(staging.rawValue)"
    }

    /// Legality against the operation shape and the measured device: power-
    /// of-two T within the threadgroup maximum and the vector length, and
    /// the emitted kernel's ACTUAL static threadgroup memory — the staged
    /// array (n * 4 bytes) PLUS the 4-byte invShared slot — within the
    /// threadgroup memory cap.
    public func isLegal(n: Int, hardware: RMSHardwareProfile) -> Bool {
        let t = threadsPerVector
        guard t > 0, t & (t - 1) == 0,
              t <= hardware.maxTotalThreadsPerThreadgroup,
              t <= n,
              n * 4 + 4 <= hardware.maxThreadgroupMemoryLength else { return false }
        if staging == .float4, n % 4 != 0 { return false }
        return true
    }
}

// MARK: - Schedule derivation (shape × measured hardware)

public enum RMSScheduleDeriver {
    public static let maxCandidates = 6

    /// Derives the bounded legal schedule set from the operation SHAPE and
    /// the MEASURED hardware. Candidate 0 is always the incumbent-mapping
    /// baseline (T=1, scalar staging — the same one-thread serial
    /// summation order as the reference kernel); cooperative T values are
    /// DERIVED from the device: the legal powers of two are ranked by
    /// proximity of log2(T) to log2(threadExecutionWidth) (the SIMD lane
    /// count — the occupancy-relevant measured quantity), and the two
    /// nearest survive. Microbenchmarks never select anything here.
    public static func derive(op: RMSOpIR, hardware: RMSHardwareProfile) -> [RMSSchedule] {
        let baseline = RMSSchedule(threadsPerVector: 1, staging: .scalar)
        // The baseline stages the same threadgroup array — if even it is
        // illegal (threadgroup memory cap), NOTHING is legal for this
        // shape on this hardware and the derived set is EMPTY (never an
        // illegal entry presented as legal).
        guard baseline.isLegal(n: op.n, hardware: hardware) else { return [] }
        var result = [baseline]
        let w = max(1, hardware.threadExecutionWidth)
        var ladder = [Int]()
        var t = 2
        while t <= hardware.maxTotalThreadsPerThreadgroup && t <= op.n {
            let probe = RMSSchedule(threadsPerVector: t, staging: .scalar)
            if probe.isLegal(n: op.n, hardware: hardware) {
                ladder.append(t)
            }
            t <<= 1
        }
        let logw = Foundation.log2(Double(w))
        ladder.sort {
            abs(Foundation.log2(Double($0)) - logw) < abs(Foundation.log2(Double($1)) - logw)
                || (abs(Foundation.log2(Double($0)) - logw) == abs(Foundation.log2(Double($1)) - logw)
                    && $0 < $1)
        }
        for t in ladder.prefix(2) {
            result.append(RMSSchedule(threadsPerVector: t, staging: .scalar))
            result.append(RMSSchedule(threadsPerVector: t, staging: .float4))
        }
        return Array(result.prefix(maxCandidates))
    }
}

// MARK: - Metal source emission

public enum RMSKernelEmitter {
    /// Bumped whenever emission rules change; part of artifact identity.
    public static let codegenVersion = "rmsgen-v1"

    /// Emits the complete MSL source for one schedule. The thread mapping,
    /// staging width, and apply stride are generated FROM the schedule
    /// parameters and the op shape.
    public static func emit(op: RMSOpIR, schedule: RMSSchedule) throws -> String {
        // Emission-time legality is SHAPE-only (the deriver and the
        // artifact gates re-check against the measured hardware).
        let shapeOnly = RMSHardwareProfile(
            deviceName: "emit", maxTotalThreadsPerThreadgroup: .max,
            threadExecutionWidth: 1, maxThreadgroupMemoryLength: .max,
            recommendedMaxWorkingSetSize: 0)
        guard schedule.isLegal(n: op.n, hardware: shapeOnly) else {
            throw RMSLoweringError.emissionFailed(
                "schedule \(schedule) illegal for n \(op.n)")
        }
        let t = schedule.threadsPerVector
        let stagingLoop: String
        switch schedule.staging {
        case .scalar:
            stagingLoop = """
            for (uint i = lid; i < N; i += T) { staged[i] = xv[i]; }
            """
        case .float4:
            stagingLoop = """
            for (uint i = lid * 4u; i < N; i += T * 4u) {
                *(threadgroup float4*)(staged + i) = *(device const float4*)(xv + i);
            }
            """
        }
        return """
        #include <metal_stdlib>
        using namespace metal;

        // \(schedule.kernelName) — EMITTED by LeviathanForge RMS lowering
        // (\(codegenVersion)) from schedule \(schedule) and op shape n=\(op.n).
        // T=\(t) threads cooperate per vector: PARALLEL staging of exact
        // input copies into threadgroup memory, a SERIAL lane-0 sum of
        // squares in the ORIGINAL scalar order (i = 0..n-1, plain float
        // accumulation — the incumbent's exact reduction), then a PARALLEL
        // strided output apply whose per-element arithmetic is identical to
        // the incumbent kernel's. In-place x == y is safe by construction:
        // the whole vector is staged (and the reduction consumed the staged
        // copy) before any device write; the apply never re-reads x.
        kernel void \(schedule.kernelName)(constant const uint2& p [[buffer(0)]],
                                            constant const float& eps [[buffer(1)]],
                                            constant const int& withScale [[buffer(2)]],
                                            constant const ulong& wOff [[buffer(3)]],
                                            device const uchar* wbase [[buffer(4)]],
                                            device const float* x [[buffer(5)]],
                                            device float* y [[buffer(6)]],
                                            uint lid [[thread_position_in_threadgroup]],
                                            uint wid [[threadgroup_position_in_grid]]) {
            constexpr uint N = \(op.n)u;
            constexpr uint T = \(t)u;
            uint vec = wid;
            if (vec >= p.y || p.x != N) { return; }
            device const float* xv = x + (ulong)vec * N;
            threadgroup float staged[N];
            threadgroup float invShared[1];
        \(stagingLoop)
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0u) {
                float ms = 0.0f;
                for (uint i = 0u; i < N; i++) { ms += staged[i] * staged[i]; }
                invShared[0] = 1.0f / sqrt(ms / float(N) + eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = invShared[0];
            device const float* w = (device const float*)(wbase + wOff);
            device float* yv = y + (ulong)vec * N;
            if (withScale != 0) {
                for (uint i = lid; i < N; i += T) { yv[i] = staged[i] * inv * w[i]; }
            } else {
                for (uint i = lid; i < N; i += T) { yv[i] = staged[i] * inv; }
            }
        }
        """
    }
}

// MARK: - Independent CPU FP64 oracle

public enum RMSOracle {
    /// Independent Double-precision RMSNorm: the sum of squares accumulates
    /// in Double in the same serial order — the ground truth the 1e-4
    /// component contract measures against (the incumbent kernels are
    /// measured against it too, never assumed exact).
    public static func rmsnormFP64(x: [Float], w: [Float]?, eps: Double,
                                   vectors: Int) throws -> [Double] {
        let n = x.count / vectors
        guard n > 0, x.count == n * vectors,
              x.allSatisfy({ $0.isFinite }),
              w == nil || w!.count == n,
              w?.allSatisfy({ $0.isFinite }) ?? true else {
            throw RMSLoweringError.oracleFailed(
                "oracle geometry: x \(x.count), vectors \(vectors)")
        }
        var y = [Double](repeating: 0, count: x.count)
        for v in 0..<vectors {
            var ms = 0.0
            for i in 0..<n {
                let d = Double(x[v * n + i])
                ms += d * d
            }
            let inv = 1.0 / (ms / Double(n) + eps).squareRoot()
            guard inv.isFinite else {
                throw RMSLoweringError.oracleFailed("nonfinite inv at vector \(v)")
            }
            for i in 0..<n {
                let out = Double(x[v * n + i]) * inv * (w.map { Double($0[i]) } ?? 1.0)
                guard out.isFinite else {
                    throw RMSLoweringError.oracleFailed("nonfinite output vector \(v) elem \(i)")
                }
                y[v * n + i] = out
            }
        }
        return y
    }
}

// MARK: - Component candidate evaluation

public struct RMSComponentResult: Codable, Equatable, Sendable {
    public let schedule: RMSSchedule
    public let accepted: Bool
    /// Worst |gpu - ref64| / max(1, |ref64|) over the representative vectors.
    public let worstRelToOracle: Double?
    /// BIT-LEVEL equality vs the incumbent production kernel's outputs on
    /// the same inputs (bitPatterns compared; nil when the kernel failed to
    /// run). The primary contract of this family.
    public let bitIdenticalToIncumbent: Bool?
    /// Worst |gpu - incumbentGpu| / max(1, |incumbentGpu|) when not bit-identical.
    public let worstRelToIncumbent: Double?
    public let note: String
}

/// Bounded representative inputs for the op: real GGUF weight bytes when the
/// op binds a tensor (pread of exactly n*4 bytes), else a deterministic
/// synthetic F32 weight; activations are always deterministic synthetic.
public enum RMSComponentInputs {
    public static func weightBytes(op: RMSOpIR, file: GGUFFile?) throws -> [UInt8] {
        if let file, let tensor = op.tensor {
            try op.validateAgainst(file: file)
            let info: GGUFTensorInfo
            do { info = try file.tensor(tensor) } catch {
                throw RMSLoweringError.invalidOp("bound tensor \(tensor): \(error)")
            }
            let range = try file.tensorRange(info)
            let needed = op.n * 4
            guard needed <= range.count else {
                throw RMSLoweringError.invalidOp(
                    "tensor extent \(range.count) smaller than n*4 \(needed)")
            }
            guard let url = file.sourceURL else {
                throw RMSLoweringError.invalidOp("GGUF has no source URL for bounded read")
            }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(range.lowerBound))
            guard let data = try handle.read(upToCount: needed), data.count == needed else {
                throw RMSLoweringError.invalidOp("bounded tensor read failed")
            }
            return Array(data)
        }
        var rng = SplitMix64(seed: 0x5A4C_524D)
        var bytes = [UInt8](repeating: 0, count: op.n * 4)
        for i in 0..<op.n {
            let f = 0.5 + rng.nextUniform()
            withUnsafeBytes(of: f.bitPattern.littleEndian) { raw in
                bytes.replaceSubrange((i * 4)..<(i * 4 + 4), with: raw)
            }
        }
        return bytes
    }

    public static func activation(op: RMSOpIR, seed: UInt64 = 70_010) -> [Float] {
        var rng = SplitMix64(seed: seed)
        let count = op.n * op.componentVectors
        return (0..<count).map { _ in
            // Norm inputs are post-projection activations: center them
            // around zero like real inputs, scale 1.
            rng.nextUniform() * 2.0 - 1.0
        }
    }
}

/// Standalone GPU evaluation of one emitted kernel (component level). The
/// kernel bindings match the incumbent rmsnorm contract exactly (p, eps,
/// withScale, wOff, wbase, x, y), so the same inputs drive both.
public struct RMSEmittedKernel: Sendable {
    public let schedule: RMSSchedule
    public let source: String
    public let pipeline: MTLComputePipelineState
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public static func compile(op: RMSOpIR, schedule: RMSSchedule,
                               device: MTLDevice) throws -> RMSEmittedKernel {
        let source = try RMSKernelEmitter.emit(op: op, schedule: schedule)
        guard let queue = device.makeCommandQueue() else {
            throw RMSLoweringError.hardwareUnavailable("no command queue")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw RMSLoweringError.emissionFailed("MSL compile: \(error)") }
        guard let fn = library.makeFunction(name: schedule.kernelName),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw RMSLoweringError.emissionFailed("pipeline for \(schedule.kernelName)")
        }
        // Occupancy legality on the ACTUAL compiled pipeline: the emitted
        // threadgroup (T threads plus the staged array and the invShared
        // slot) must fit.
        guard schedule.threadsPerVector <= pipeline.maxTotalThreadsPerThreadgroup,
              op.n * 4 + 4 <= device.maxThreadgroupMemoryLength else {
            throw RMSLoweringError.emissionFailed(
                "\(schedule.kernelName): threadgroup (T=\(schedule.threadsPerVector), staged \(op.n * 4)B) exceeds device limits")
        }
        return RMSEmittedKernel(schedule: schedule, source: source,
                                pipeline: pipeline, device: device, queue: queue)
    }

    /// Runs the kernel on the given inputs; w is nil for unscaled ops.
    /// inPlace runs y aliased to x (the in-place model sites) — the barrier
    /// construction makes that safe, and the component gate exercises it.
    public func run(op: RMSOpIR, x: [Float], w: [Float]?, vectors: Int,
                    eps: Double, inPlace: Bool = false) throws -> [Float] {
        let n = op.n
        guard x.count == n * vectors, x.allSatisfy({ $0.isFinite }),
              w == nil || w!.count == n else {
            throw RMSLoweringError.componentGateFailed("component geometry mismatch")
        }
        let wBytes: [UInt8]
        if let w {
            wBytes = w.withUnsafeBytes { Array($0) }
        } else {
            wBytes = [UInt8](repeating: 0, count: 0)
        }
        guard let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let wb = device.makeBuffer(bytes: wBytes.isEmpty ? [UInt8(0)] : wBytes,
                                         length: max(1, wBytes.count),
                                         options: .storageModeShared),
              let yb = device.makeBuffer(length: x.count * 4, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw RMSLoweringError.hardwareUnavailable("buffer/encoder allocation")
        }
        if inPlace {
            memcpy(yb.contents(), xb.contents(), x.count * 4)
        }
        var p = SIMD2<UInt32>(UInt32(n), UInt32(vectors))
        var epsF = Float(eps)
        var withScale: Int32 = w == nil ? 0 : 1
        var off = UInt64(0)
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&epsF, length: MemoryLayout<Float>.stride, index: 1)
        enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 3)
        enc.setBuffer(wb, offset: 0, index: 4)
        enc.setBuffer(inPlace ? yb : xb, offset: 0, index: 5)
        enc.setBuffer(yb, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: vectors, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: schedule.threadsPerVector,
                                                                height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw RMSLoweringError.componentGateFailed(
                "status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: x.count))
    }

    /// Warmed component GPU timing screen: median GPU-busy microseconds per
    /// call over `reps` calls after `warmup` warmups (command-buffer
    /// gpuStartTime..gpuEndTime, one commit per call, buffers allocated
    /// once). RANKING/PRUNING ONLY — this number never selects a winner and
    /// is never reported as a performance claim.
    public func screenMicros(op: RMSOpIR, x: [Float], w: [Float]?, vectors: Int,
                             eps: Double, warmup: Int = 5, reps: Int = 30) throws -> Double {
        guard x.count == op.n * vectors else {
            throw RMSLoweringError.componentGateFailed("screen geometry mismatch")
        }
        let wBytes: [UInt8]
        if let w { wBytes = w.withUnsafeBytes { Array($0) } }
        else { wBytes = [] }
        guard let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let wb = device.makeBuffer(bytes: wBytes.isEmpty ? [UInt8(0)] : wBytes,
                                         length: max(1, wBytes.count),
                                         options: .storageModeShared),
              let yb = device.makeBuffer(length: x.count * 4, options: .storageModeShared)
        else {
            throw RMSLoweringError.hardwareUnavailable("screen buffers")
        }
        var p = SIMD2<UInt32>(UInt32(op.n), UInt32(vectors))
        var epsF = Float(eps)
        var withScale: Int32 = w == nil ? 0 : 1
        var off = UInt64(0)
        var samples = [Double]()
        samples.reserveCapacity(reps)
        for iter in 0..<(warmup + reps) {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                throw RMSLoweringError.hardwareUnavailable("screen encoder")
            }
            enc.setComputePipelineState(pipeline)
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&epsF, length: MemoryLayout<Float>.stride, index: 1)
            enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 3)
            enc.setBuffer(wb, offset: 0, index: 4)
            enc.setBuffer(xb, offset: 0, index: 5)
            enc.setBuffer(yb, offset: 0, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: vectors, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(
                                        width: schedule.threadsPerVector, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw RMSLoweringError.componentGateFailed("screen command buffer")
            }
            if iter >= warmup {
                samples.append((cmd.gpuEndTime - cmd.gpuStartTime) * 1e6)
            }
        }
        let s = samples.sorted()
        return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2 : s[s.count/2]
    }
}

/// Standalone incumbent-kernel runner for the component bit-equality gate:
/// compiles g4_rmsnorm and g4_rmsnorm_vl from the runner's own embedded MSL
/// (the same sources the production path uses) and evaluates them with one
/// thread per vector.
public struct RMSIncumbentKernel: Sendable {
    public let scalar: MTLComputePipelineState
    public let vl: MTLComputePipelineState
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public static func compile(device: MTLDevice) throws -> RMSIncumbentKernel {
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil) }
        catch { throw RMSLoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let scalarFn = library.makeFunction(name: "g4_rmsnorm"),
              let scalarPipe = try? device.makeComputePipelineState(function: scalarFn),
              let vlFn = library.makeFunction(name: "g4_rmsnorm_vl"),
              let vlPipe = try? device.makeComputePipelineState(function: vlFn),
              let queue = device.makeCommandQueue() else {
            throw RMSLoweringError.emissionFailed("incumbent pipelines")
        }
        return RMSIncumbentKernel(scalar: scalarPipe, vl: vlPipe,
                                  device: device, queue: queue)
    }

    /// The production incumbent for a call shape: g4_rmsnorm_vl when the
    /// production guards hold (they do for every actual norm width here),
    /// else the scalar original.
    public func productionPipeline(n: Int, scaled: Bool) -> MTLComputePipelineState {
        let wOff: UInt64 = scaled ? 0 : 0
        return Gemma4Runner.rmsVectorLoadGuards(n: n, scaled: scaled, wOff: wOff)
            ? vl : scalar
    }

    public func run(pipeline: MTLComputePipelineState, n: Int, vectors: Int,
                    x: [Float], w: [Float]?, eps: Double,
                    inPlace: Bool = false) throws -> [Float] {
        guard x.count == n * vectors, w == nil || w!.count == n else {
            throw RMSLoweringError.componentGateFailed("incumbent geometry mismatch")
        }
        var wBytes = [UInt8]()
        if let w {
            wBytes = w.withUnsafeBytes { Array($0) }
        }
        guard let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let wb = device.makeBuffer(bytes: wBytes.isEmpty ? [UInt8(0)] : wBytes,
                                         length: max(1, wBytes.count),
                                         options: .storageModeShared),
              let yb = device.makeBuffer(length: x.count * 4, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw RMSLoweringError.hardwareUnavailable("incumbent buffers")
        }
        if inPlace {
            memcpy(yb.contents(), xb.contents(), x.count * 4)
        }
        var p = SIMD2<UInt32>(UInt32(n), UInt32(vectors))
        var epsF = Float(eps)
        var withScale: Int32 = w == nil ? 0 : 1
        var off = UInt64(0)
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&epsF, length: MemoryLayout<Float>.stride, index: 1)
        enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 3)
        enc.setBuffer(wb, offset: 0, index: 4)
        enc.setBuffer(inPlace ? yb : xb, offset: 0, index: 5)
        enc.setBuffer(yb, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: vectors, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw RMSLoweringError.componentGateFailed("incumbent command buffer")
        }
        return [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: x.count))
    }
}

// MARK: - Artifact

/// One bounded chat fixture used by the model-level gate.
public struct RMSFixtureRecord: Codable, Equatable, Sendable {
    public let name: String
    public let promptTokenCount: Int
    public let gatedSteps: Int
    /// Workload digest of the EXACT teacher-forced history.
    public let historyDigest: String
}

/// Raw samples of ONE timing arm on ONE fixture.
public struct RMSTimingArmFixture: Codable, Equatable, Sendable {
    public let walls: [Double]
    public let gpus: [Double]
    public let startIdx: Int
    public let timedCount: Int
}

/// One PAIRED repetition (alternating order across repetitions).
public struct RMSTimingRepetition: Codable, Equatable, Sendable {
    public let candidateFirst: Bool
    public let candidate: [RMSTimingArmFixture]
    public let incumbent: [RMSTimingArmFixture]
}

/// Model-level gate + matched-timing record for ONE generated candidate.
public struct RMSModelCandidateResult: Codable, Equatable, Sendable {
    public let schedule: RMSSchedule
    public let fixturesPassed: Int
    public let fixturesFailed: Int
    public let worstRel: Double?
    public let gateFailReason: String?
    public var wallMedPerFixture: [Double]?
    public var gpuMedPerFixture: [Double]?
    public var timingReps: [RMSTimingRepetition]?
}

/// The full model-level selection record persisted with the artifact.
public struct RMSModelGateRecord: Codable, Equatable, Sendable {
    public let genTokens: Int
    public let timedTokens: Int
    public let fixtures: [RMSFixtureRecord]
    public let results: [RMSModelCandidateResult]
    public let incumbentWallPerFixture: [Double]
    public let incumbentGpuPerFixture: [Double]
    public let performanceSelected: Bool
    public let policyNote: String
}

public struct RMSLoweringArtifact: Codable, Equatable, Sendable {
    public static let schema = 1

    public let artifactSchema: Int
    public let codegenVersion: String
    public let opDigest: String
    public let op: RMSOpIR
    public let schedule: RMSSchedule
    public let kernelName: String
    public let emittedSource: String
    public let sourceSha256: String
    public let hardware: RMSHardwareProfile
    public let toolchain: String
    public let ggufDigest: String
    public let executableDigest: String
    public let componentResults: [RMSComponentResult]
    public let incumbentWorstRelToOracle: Double
    public let representativeVectors: Int
    public let modelGate: RMSModelGateRecord?
    public let artifactDigest: String

    /// Canonical-body digest (all fields except artifactDigest).
    static func bodyDigest(_ body: RMSLoweringArtifact) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stripped = RMSLoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeVectors: body.representativeVectors,
            modelGate: body.modelGate, artifactDigest: "")
        let bytes = try encoder.encode(stripped)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: [.atomic])
    }

    public static func load(from url: URL) throws -> RMSLoweringArtifact {
        guard let data = try? Data(contentsOf: url) else {
            throw RMSLoweringError.invalidArtifact("unreadable artifact \(url.path)")
        }
        guard data.count <= 1 << 20 else {
            throw RMSLoweringError.invalidArtifact("artifact exceeds 1 MiB")
        }
        do { return try JSONDecoder().decode(RMSLoweringArtifact.self, from: data) }
        catch { throw RMSLoweringError.invalidArtifact("decode: \(error)") }
    }

    /// Full identity validation. Fail-closed: any mismatch (schema, op
    /// digest vs the CURRENT op, codegen version, re-EMITTED source bytes
    /// vs the persisted source, source SHA, hardware, toolchain, GGUF,
    /// executable, component-gate consistency, self digest) rejects the
    /// artifact — the caller falls back to the incumbent kernel.
    public func validate(op currentOp: RMSOpIR, hardware: RMSHardwareProfile,
                         ggufDigest: String, executableDigest: String) throws {
        try currentOp.validate()
        try op.validate()
        guard artifactSchema == Self.schema else {
            throw RMSLoweringError.invalidArtifact("schema \(artifactSchema)")
        }
        guard codegenVersion == RMSKernelEmitter.codegenVersion else {
            throw RMSLoweringError.invalidArtifact("codegen \(codegenVersion) stale")
        }
        guard op == currentOp, opDigest == (try currentOp.digest()) else {
            throw RMSLoweringError.invalidArtifact("op mismatch vs current operation")
        }
        guard kernelName == schedule.kernelName else {
            throw RMSLoweringError.invalidArtifact("kernel name inconsistent with schedule")
        }
        let reemitted = try RMSKernelEmitter.emit(op: op, schedule: schedule)
        guard reemitted == emittedSource else {
            throw RMSLoweringError.invalidArtifact("emitted source tampered/rule drift")
        }
        guard sourceSha256 == SHA256.hex(SHA256.hash(Array(emittedSource.utf8))) else {
            throw RMSLoweringError.invalidArtifact("source SHA mismatch")
        }
        guard hardware.maxTotalThreadsPerThreadgroup > 0,
              hardware.threadExecutionWidth > 0,
              hardware.maxThreadgroupMemoryLength > 0,
              hardware.recommendedMaxWorkingSetSize > 0,
              hardware == self.hardware else {
            throw RMSLoweringError.invalidArtifact("hardware mismatch")
        }
        guard toolchain == Forge.toolchainIdentifier else {
            throw RMSLoweringError.invalidArtifact("toolchain mismatch")
        }
        guard ggufDigest == self.ggufDigest else {
            throw RMSLoweringError.invalidArtifact("GGUF digest mismatch")
        }
        guard executableDigest == self.executableDigest else {
            throw RMSLoweringError.invalidArtifact("executable digest mismatch")
        }
        guard !componentResults.isEmpty,
              componentResults.contains(where: { $0.schedule == schedule && $0.accepted }) else {
            throw RMSLoweringError.invalidArtifact("selected schedule not gate-accepted")
        }
        if let sel = componentResults.first(where: { $0.schedule == schedule && $0.accepted }) {
            guard let rel = sel.worstRelToOracle, rel.isFinite, rel >= 0, rel < 1e-4,
                  sel.bitIdenticalToIncumbent == true else {
                throw RMSLoweringError.invalidArtifact(
                    "component record for selected schedule not within the 1e-4 contract or not bit-identical to the incumbent")
            }
        }
        if let mg = modelGate {
            guard mg.genTokens >= 2, mg.timedTokens >= 1,
                  mg.timedTokens < mg.genTokens,
                  !mg.fixtures.isEmpty,
                  mg.fixtures.allSatisfy({
                      $0.gatedSteps >= 2 && $0.gatedSteps <= mg.genTokens
                          && $0.promptTokenCount > 0
                          && rmsIsLowercaseHex64($0.historyDigest) }),
                  Set(mg.fixtures.map(\.name)).count == mg.fixtures.count,
                  let sel = mg.results.first(where: { $0.schedule == schedule }),
                  sel.fixturesPassed == mg.fixtures.count,
                  sel.fixturesFailed == 0,
                  let worst = sel.worstRel, worst < 1e-4 else {
                throw RMSLoweringError.invalidArtifact(
                    "model-gate record inconsistent or schedule not fully gate-passed")
            }
        }
        guard artifactDigest == (try Self.bodyDigest(self)) else {
            throw RMSLoweringError.invalidArtifact("artifact digest tampered")
        }
    }
}

// MARK: - Hardware profile source

public enum RMSCurrentHardware {
    /// The measured hardware profile used for artifact identity: taken from
    /// the compiled incumbent pipeline (execution width, threadgroup
    /// maximum) and the live device (threadgroup memory cap, working set)
    /// exactly as at forge time.
    public static func profile() throws -> RMSHardwareProfile {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RMSLoweringError.hardwareUnavailable("Metal unavailable")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil) }
        catch { throw RMSLoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let fn = library.makeFunction(name: "g4_rmsnorm"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw RMSLoweringError.emissionFailed("incumbent pipeline")
        }
        return RMSHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            maxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }
}

// MARK: - Component forge

/// Renders the FINAL persisted-execution proof log line. Centralized in
/// ONE production helper (called by selectAndPersist and directly by the
/// regression test) so the tested formatter IS the production formatter:
/// the real 32-step select run once crashed (SIGSEGV via
/// objc_opt_respondsToSelector) because a Swift Bool crossed
/// String(format:) %@ as a C vararg, and a duplicated safe copy in the
/// test would not have caught it. Bool and optional-tensor arguments are
/// converted to objects BEFORE the format call; never pass raw Bool here.
public func rmsPersistedExecutionLogLine(fixtureName: String, worstRel: Double,
                                         dispatches: Int, n: Int, scaled: Bool,
                                         tensor: String?) -> String {
    String(format: "persisted execution: reloaded+validated+installed from disk; %@ replay worst rel %.3e; generated dispatches %d (n=%d, scaled=%@%@)",
           fixtureName, worstRel, dispatches, n,
           scaled ? "true" : "false",
           tensor.map { ", weight tensor \($0)" } ?? "")
}

public enum RMSLoweringForge {

    /// Forges the operator lowering artifact: validate the op against the
    /// GGUF, derive legal schedules from shape + measured hardware, emit
    /// and compile every candidate, gate each against the independent CPU
    /// FP64 oracle under the EXISTING 1e-4 component contract AND against
    /// BIT-IDENTITY with the incumbent production kernel on the same real
    /// inputs (both out-of-place and the in-place x == y mode the model
    /// actually uses at the head-norm sites). SELECTION IS NUMERICAL ONLY
    /// (bit-identical first, then smallest worst rel-to-FP64); no
    /// performance claim is made here — speed is decided only by the
    /// matched model-level measurement.
    public static func forge(opJSON: Data, ggufURL: URL?) throws
            -> (artifact: RMSLoweringArtifact, results: [RMSComponentResult],
                incumbentRel: Double) {
        let op = try RMSOpIR.parse(opJSON)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RMSLoweringError.hardwareUnavailable("Metal unavailable")
        }
        var file: GGUFFile? = nil
        var ggufDigest = ""
        if op.tensor != nil {
            guard let ggufURL else {
                throw RMSLoweringError.invalidOp("op binds a tensor; --gguf is required")
            }
            let f = try GGUFReader.read(ggufURL)
            try op.validateAgainst(file: f)
            file = f
            ggufDigest = try GemmaForge.fileDigest(ggufURL)
        }

        let incumbent = try RMSIncumbentKernel.compile(device: device)
        let hardware = RMSHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: incumbent.vl.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: incumbent.vl.threadExecutionWidth,
            maxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))

        let vectors = op.componentVectors
        let wBytes = try RMSComponentInputs.weightBytes(op: op, file: file)
        let w: [Float]? = op.scaled
            ? wBytes.withUnsafeBytes { raw in Array(raw.assumingMemoryBound(to: Float.self)) }
            : nil
        let x = RMSComponentInputs.activation(op: op)
        let ref64 = try RMSOracle.rmsnormFP64(x: x, w: w, eps: op.eps, vectors: vectors)
        let production = incumbent.productionPipeline(n: op.n, scaled: op.scaled)

        func rel(_ y: [Float], _ ref: [Double]) throws -> Double {
            guard y.count == ref.count, y.allSatisfy({ $0.isFinite }) else {
                throw RMSLoweringError.componentGateFailed("output count/nonfinite")
            }
            var worst = 0.0
            for i in 0..<y.count {
                worst = max(worst, abs(Double(y[i]) - ref[i]) / max(1.0, abs(ref[i])))
            }
            return worst
        }

        // Incumbent production kernel vs the FP64 oracle, in BOTH the
        // out-of-place and the in-place mode (measured, not assumed).
        let incOut = try incumbent.run(pipeline: production, n: op.n, vectors: vectors,
                                       x: x, w: w, eps: op.eps)
        let incInPlace = try incumbent.run(pipeline: production, n: op.n, vectors: vectors,
                                           x: x, w: w, eps: op.eps, inPlace: true)
        guard rmsBitEqual(incOut, incInPlace) else {
            throw RMSLoweringError.componentGateFailed(
                "incumbent kernel is not in-place stable — cannot define the reference")
        }
        let incumbentRel = try rel(incOut, ref64)

        var results = [RMSComponentResult]()
        var best: (schedule: RMSSchedule, source: String, rel: Double)? = nil
        for schedule in RMSScheduleDeriver.derive(op: op, hardware: hardware) {
            do {
                let kernel = try RMSEmittedKernel.compile(op: op, schedule: schedule,
                                                          device: device)
                let out = try kernel.run(op: op, x: x, w: w, vectors: vectors, eps: op.eps)
                // In-place mode must ALSO be bit-identical (the model's
                // head-norm sites are in-place; the barrier construction
                // must prove itself on hardware, not on paper).
                let inPlace = try kernel.run(op: op, x: x, w: w, vectors: vectors,
                                             eps: op.eps, inPlace: true)
                let worstOracle = try rel(out, ref64)
                let bitIdentical = rmsBitEqual(out, incOut) && rmsBitEqual(inPlace, incOut)
                var worstInc = 0.0
                if !bitIdentical {
                    for i in 0..<out.count {
                        worstInc = max(worstInc, abs(Double(out[i] - incOut[i]))
                                       / max(1.0, abs(Double(incOut[i]))))
                    }
                }
                // Contract: bit-identity to the incumbent production kernel
                // (both modes) AND the existing 1e-4 FP64 distance.
                let accepted = bitIdentical && worstOracle < 1e-4
                results.append(RMSComponentResult(
                    schedule: schedule, accepted: accepted,
                    worstRelToOracle: worstOracle,
                    bitIdenticalToIncumbent: bitIdentical,
                    worstRelToIncumbent: bitIdentical ? nil : worstInc,
                    note: accepted
                        ? String(format: "bit-identical, rel64 %.3e", worstOracle)
                        : (bitIdentical
                           ? String(format: "bit-identical but outside 1e-4 FP64 contract (rel64 %.3e)", worstOracle)
                           : String(format: "NOT bit-identical (relInc %.3e)", worstInc))))
                if accepted, best == nil || worstOracle < best!.rel {
                    best = (schedule, kernel.source, worstOracle)
                }
            } catch {
                results.append(RMSComponentResult(
                    schedule: schedule, accepted: false, worstRelToOracle: nil,
                    bitIdenticalToIncumbent: nil, worstRelToIncumbent: nil,
                    note: "rejected: \(error)"))
            }
        }
        guard let winner = best else {
            throw RMSLoweringError.componentGateFailed(
                results.map { "\($0.schedule): \($0.note)" }.joined(separator: "; "))
        }
        let body = RMSLoweringArtifact(
            artifactSchema: RMSLoweringArtifact.schema,
            codegenVersion: RMSKernelEmitter.codegenVersion,
            opDigest: try op.digest(), op: op, schedule: winner.schedule,
            kernelName: winner.schedule.kernelName, emittedSource: winner.source,
            sourceSha256: SHA256.hex(SHA256.hash(Array(winner.source.utf8))),
            hardware: hardware, toolchain: Forge.toolchainIdentifier,
            ggufDigest: ggufDigest,
            executableDigest: try Gemma4Runner.executableDigest(),
            componentResults: results, incumbentWorstRelToOracle: incumbentRel,
            representativeVectors: vectors, modelGate: nil, artifactDigest: "")
        let digest = try RMSLoweringArtifact.bodyDigest(body)
        let artifact = RMSLoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeVectors: body.representativeVectors,
            modelGate: nil, artifactDigest: digest)
        return (artifact, results, incumbentRel)
    }

    // MARK: bounded model-level orchestration (per-candidate gates,
    // matched timing, policy selection, persisted execution)

    /// One bounded chat fixture: name + pre-tokenized prompt history.
    public struct RMSFixture {
        public let name: String
        public let tokens: [Int]
        public init(name: String, tokens: [Int]) {
            self.name = name
            self.tokens = tokens
        }
    }

    /// Full bounded selection orchestration, all in ONE process — the exact
    /// phase-9 protocol applied to the RMS family: component phase; per-
    /// fixture incumbent reference histories; per component-accepted
    /// candidate teacher-forced same-history replay gates (full logits,
    /// UNCHANGED strict 1e-4 + exact argmax + finiteness at every step);
    /// 3 paired matched-timing repetitions with alternating order; the
    /// documented 2% policy; persisted artifact + reload/validate/install
    /// with dispatch-counter proof. See Q4LoweringForge.selectAndPersist
    /// for the full protocol commentary — the structure is identical.
    public static func selectAndPersist(opJSON: Data, ggufURL: URL,
                                        runner: Gemma4Runner,
                                        fixtures: [RMSFixture],
                                        genTokens: Int, timedTokens: Int,
                                        isEndOfGeneration: (Int) -> Bool,
                                        saveTo url: URL) throws
            -> (artifact: RMSLoweringArtifact, log: [String]) {
        guard genTokens >= 2, genTokens <= 64, timedTokens >= 1,
              timedTokens < genTokens, fixtures.count >= 1, fixtures.count <= 4 else {
            throw RMSLoweringError.invalidSchedule(
                "orchestration bounds: genTokens \(genTokens), timedTokens \(timedTokens), fixtures \(fixtures.count)")
        }
        let names = Set(fixtures.map(\.name))
        guard names.count == fixtures.count else {
            throw RMSLoweringError.invalidSchedule("fixture names must be unique")
        }
        for f in fixtures {
            guard !f.tokens.isEmpty, f.tokens.count + genTokens <= 4096 else {
                throw RMSLoweringError.invalidSchedule("fixture \(f.name) exceeds context")
            }
        }
        var log = [String]()
        // Incumbent reference configuration (pass-7 selection), re-applied
        // AFTER every reset (reset clears autoSegment — the reset-order
        // bug root measured; see the phase-9 commentary).
        func resetIncumbentConfig() throws {
            runner.reset()
            runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
            guard runner.segmentBatchingActive else {
                throw RMSLoweringError.componentGateFailed(
                    "segment batching failed to re-engage after reset")
            }
        }
        let op = try RMSOpIR.parse(opJSON)
        try op.validateAgainst(file: runner.ggufFile)
        let ggufDigest = try GemmaForge.fileDigest(ggufURL)

        // ---- 1. component phase ----
        let (componentArtifact, componentResults, incumbentRel) = try forge(
            opJSON: opJSON, ggufURL: ggufURL)
        log.append(String(format: "component: incumbent rel-to-FP64 %.3e; %d/%d candidates accepted",
                          incumbentRel,
                          componentResults.filter(\.accepted).count, componentResults.count))

        // ---- 2. reference arm per fixture ----
        struct RefArm {
            var ids: [Int]
            var logits: [[Float]]
        }
        var refArms = [String: RefArm]()
        for fixture in fixtures {
            try resetIncumbentConfig()
            var ids = [Int]()
            var logits = [[Float]]()
            let prefill = try runner.forward(tokenIds: fixture.tokens)
            logits.append(try runner.currentLogits())
            ids.append(prefill.argmaxToken)
            var prev = prefill.argmaxToken
            while ids.count < genTokens {
                if isEndOfGeneration(prev) { break }
                let ev = try runner.forward(tokenIds: [prev])
                logits.append(try runner.currentLogits())
                ids.append(ev.argmaxToken)
                prev = ev.argmaxToken
            }
            refArms[fixture.name] = RefArm(ids: ids, logits: logits)
            log.append("reference [\(fixture.name)]: \(ids.count) greedy steps")
        }

        func gateReplay(_ fixture: RMSFixture, _ ref: RefArm)
                -> (worstRel: Double, fail: String?, generatedDispatches: Int) {
            let dispatchBefore = runner.rmsGeneratedDispatchCount
            do {
                try resetIncumbentConfig()
                for t in fixture.tokens {
                    _ = try runner.forward(tokenIds: [t])
                }
                var worst = 0.0
                for s in 0..<ref.ids.count {
                    let lg = try runner.currentLogits()
                    let want = ref.logits[s]
                    guard lg.count == want.count,
                          lg.allSatisfy({ $0.isFinite }),
                          want.allSatisfy({ $0.isFinite }) else {
                        return (worst, "step \(s): shape/nonfinite violation",
                                runner.rmsGeneratedDispatchCount - dispatchBefore)
                    }
                    let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                    guard argmax == ref.ids[s] else {
                        return (worst, "step \(s): argmax \(argmax) != reference \(ref.ids[s])",
                                runner.rmsGeneratedDispatchCount - dispatchBefore)
                    }
                    for i in 0..<lg.count {
                        worst = max(worst, abs(Double(lg[i] - want[i]))
                                    / max(1.0, abs(Double(want[i]))))
                    }
                    if worst >= 1e-4 {
                        return (worst, String(format: "step %d outside 1e-4 (worst %.3e)",
                                              s, worst),
                                runner.rmsGeneratedDispatchCount - dispatchBefore)
                    }
                    if s + 1 < ref.ids.count {
                        _ = try runner.forward(tokenIds: [ref.ids[s]])
                    }
                }
                return (worst, nil, runner.rmsGeneratedDispatchCount - dispatchBefore)
            } catch {
                return (0, "replay threw: \(error)",
                        runner.rmsGeneratedDispatchCount - dispatchBefore)
            }
        }

        // ---- 3. per-candidate model gates. Component-first funnel: every
        // component-accepted candidate runs a WARMED component GPU timing
        // screen (median GPU-busy µs/call; ranking/pruning only — never a
        // win claim), and only the TWO FASTEST proceed to the expensive
        // full-model gates and matched timing. ----
        let device = MTLCreateSystemDefaultDevice()
        var modelResults = [RMSModelCandidateResult]()
        var passingKernels = [RMSSchedule: RMSEmittedKernel]()
        let accepted = componentResults.filter(\.accepted)
        var funnel = [RMSSchedule]()
        if let device {
            let wBytes = try RMSComponentInputs.weightBytes(op: op, file: runner.ggufFile)
            let w: [Float]? = op.scaled
                ? wBytes.withUnsafeBytes { raw in Array(raw.assumingMemoryBound(to: Float.self)) }
                : nil
            let x = RMSComponentInputs.activation(op: op)
            var screened = [(schedule: RMSSchedule, med: Double)]()
            // Screen at the PRODUCTION vector count for this family: the
            // hot model sites call the bound scaled n=2816 norm with
            // vectors=1, so ranking at the capture batch (8) would rank a
            // different operating point than the one being selected for.
            let screenVectors = 1
            let screenX = Array(x.prefix(op.n * screenVectors))
            let screenW = w.map { Array($0.prefix(op.n)) }
            for result in accepted {
                // The T=1 baseline is the serial-order sanity arm; it is
                // never promoted to the costly model gates.
                guard result.schedule.threadsPerVector > 1 else { continue }
                guard let kernel = try? RMSEmittedKernel.compile(op: op,
                                                                 schedule: result.schedule,
                                                                 device: device),
                      let med = try? kernel.screenMicros(op: op, x: screenX, w: screenW,
                                                         vectors: screenVectors,
                                                         eps: op.eps) else { continue }
                screened.append((result.schedule, med))
            }
            // Deterministic: fastest median first; ties by kernel name.
            screened.sort { $0.med < $1.med || ($0.med == $1.med
                                                && $0.schedule.kernelName < $1.schedule.kernelName) }
            for entry in screened {
                log.append(String(format: "component screen %@: %.1f us/call (warmed median, pruning only)",
                                  entry.schedule.kernelName, entry.med))
            }
            funnel = screened.prefix(2).map(\.schedule)
        }
        for result in funnel.map({ schedule in
            RMSComponentResult(schedule: schedule, accepted: true,
                               worstRelToOracle: nil, bitIdenticalToIncumbent: nil,
                               worstRelToIncumbent: nil, note: "screen-selected") }) {
            guard let device else { break }
            let kernel: RMSEmittedKernel
            do {
                kernel = try RMSEmittedKernel.compile(op: op, schedule: result.schedule,
                                                       device: device)
            } catch {
                modelResults.append(RMSModelCandidateResult(
                    schedule: result.schedule, fixturesPassed: 0,
                    fixturesFailed: fixtures.count, worstRel: nil,
                    gateFailReason: "compile: \(error)",
                    wallMedPerFixture: nil, gpuMedPerFixture: nil))
                continue
            }
            do { try runner.installRmsTrialKernel(op: op, kernel: kernel) }
            catch {
                modelResults.append(RMSModelCandidateResult(
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
            for fixture in fixtures {
                let ref = refArms[fixture.name]!
                let (worst, fail, dispatches) = gateReplay(fixture, ref)
                totalGenerated += dispatches
                if let fail {
                    failed += 1
                    if firstFail == nil { firstFail = "[\(fixture.name)] \(fail)" }
                } else {
                    passed += 1
                    worstAll = max(worstAll, worst)
                }
            }
            runner.uninstallRmsGeneratedKernel()
            let fullyPassed = failed == 0 && totalGenerated > 0
            if fullyPassed { passingKernels[result.schedule] = kernel }
            modelResults.append(RMSModelCandidateResult(
                schedule: result.schedule, fixturesPassed: passed, fixturesFailed: failed,
                worstRel: failed == 0 ? worstAll : nil,
                gateFailReason: firstFail,
                wallMedPerFixture: nil, gpuMedPerFixture: nil))
            log.append(String(format: "gate %@ %@: %d/%d fixtures, worst rel %.3e",
                              fullyPassed ? "PASS" : "FAIL",
                              result.schedule.kernelName, passed, fixtures.count, worstAll))
            if !fullyPassed, let f = firstFail {
                log.append("  first failure: \(f)")
            }
        }

        guard modelResults.contains(where: { $0.fixturesFailed == 0 && $0.worstRel != nil })
        else {
            throw RMSLoweringError.componentGateFailed(
                "no generated candidate passed all fixture gates: "
                + modelResults.map { "\($0.schedule.kernelName): \($0.gateFailReason ?? "?")" }
                    .joined(separator: "; "))
        }

        // ---- 4. matched timing: 3 PAIRED repetitions per fully passing
        // candidate, alternating order. THROWS on any forward failure. ----
        func timedArm() throws -> [RMSTimingArmFixture] {
            var arms = [RMSTimingArmFixture]()
            for fixture in fixtures {
                let ref = refArms[fixture.name]!
                let steps = ref.ids.count
                guard steps >= 2 else {
                    throw RMSLoweringError.componentGateFailed(
                        "fixture \(fixture.name) produced only \(steps) gated steps")
                }
                let effTimed = max(1, min(timedTokens, steps - 1))
                try resetIncumbentConfig()
                for t in fixture.tokens {
                    _ = try runner.forward(tokenIds: [t])
                }
                var idx = 0
                while idx < steps - effTimed - 1 {
                    _ = try runner.forward(tokenIds: [ref.ids[idx]])
                    idx += 1
                }
                var w = [Double]()
                var g = [Double]()
                for k in 0..<effTimed {
                    let tok = ref.ids[min(idx + k, steps - 2)]
                    let gpuBefore = runner.totalGpuMicros
                    let t0 = DispatchTime.now()
                    _ = try runner.forward(tokenIds: [tok])
                    w.append(Double(DispatchTime.now().uptimeNanoseconds
                                    - t0.uptimeNanoseconds) / 1000)
                    g.append(runner.totalGpuMicros - gpuBefore)
                }
                arms.append(RMSTimingArmFixture(walls: w, gpus: g,
                                                startIdx: idx, timedCount: effTimed))
            }
            return arms
        }
        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2 : s[s.count/2]
        }
        func armFixtureMedians(_ arm: [RMSTimingArmFixture]) -> [Double] {
            arm.map { median($0.walls) }
        }
        let repetitionCount = 3
        struct PairedTiming {
            var schedule: RMSSchedule
            var candMed: Double
            var incMed: Double
            var reps: [RMSTimingRepetition]
        }
        var paired = [PairedTiming]()
        var bestTiming: PairedTiming? = nil
        for i in modelResults.indices where modelResults[i].fixturesFailed == 0 {
            guard let schedule = passingKernels.keys.first(where: {
                $0 == modelResults[i].schedule }) else { continue }
            var reps = [RMSTimingRepetition]()
            for rep in 0..<repetitionCount {
                let candidateFirst = rep % 2 == 1
                func candidateArm() throws -> [RMSTimingArmFixture] {
                    try runner.installRmsTrialKernel(op: op, kernel: passingKernels[schedule]!)
                    defer { runner.uninstallRmsGeneratedKernel() }
                    return try timedArm()
                }
                func incumbentArm() throws -> [RMSTimingArmFixture] {
                    runner.uninstallRmsGeneratedKernel()
                    return try timedArm()
                }
                let (cand, inc): ([RMSTimingArmFixture], [RMSTimingArmFixture])
                if candidateFirst {
                    cand = try candidateArm()
                    inc = try incumbentArm()
                } else {
                    inc = try incumbentArm()
                    cand = try candidateArm()
                }
                reps.append(RMSTimingRepetition(candidateFirst: candidateFirst,
                                                candidate: cand, incumbent: inc))
            }
            modelResults[i].timingReps = reps
            let candSums = reps.map { armFixtureMedians($0.candidate).reduce(0, +) }
            let incSums = reps.map { armFixtureMedians($0.incumbent).reduce(0, +) }
            let candMed = median(candSums)
            let incMed = median(incSums)
            paired.append(PairedTiming(schedule: schedule, candMed: candMed,
                                       incMed: incMed, reps: reps))
            modelResults[i].wallMedPerFixture = (0..<fixtures.count).map { f in
                median(reps.map { armFixtureMedians($0.candidate)[f] })
            }
            modelResults[i].gpuMedPerFixture = (0..<fixtures.count).map { f in
                median(reps.map { median($0.candidate[f].gpus) })
            }
            log.append(String(format: "matched timing %@: rep sums %@ vs paired incumbent %@ (median %.0f vs %.0f us/token)",
                              schedule.kernelName,
                              candSums.map { String(format: "%.0f", $0) }
                                .joined(separator: ","),
                              incSums.map { String(format: "%.0f", $0) }
                                .joined(separator: ","),
                              candMed, incMed))
            if bestTiming == nil
                || candMed / incMed < bestTiming!.candMed / bestTiming!.incMed
                || (candMed / incMed == bestTiming!.candMed / bestTiming!.incMed
                    && schedule.kernelName < bestTiming!.schedule.kernelName) {
                bestTiming = paired.last
            }
        }

        let incWalls: [Double]
        let incGpus: [Double]
        if let best = bestTiming {
            incWalls = (0..<fixtures.count).map { f in
                median(best.reps.map { armFixtureMedians($0.incumbent)[f] })
            }
            incGpus = (0..<fixtures.count).map { f in
                median(best.reps.map { median($0.incumbent[f].gpus) }) }
        } else {
            incWalls = []
            incGpus = []
        }

        // ---- 5. policy ----
        let performanceSelected: Bool
        let selectedSchedule: RMSSchedule
        if let best = bestTiming,
           best.candMed < best.incMed * (1 - GemmaForge.noiseRelWall) {
            performanceSelected = true
            selectedSchedule = best.schedule
        } else if let best = bestTiming {
            performanceSelected = false
            selectedSchedule = best.schedule
        } else {
            performanceSelected = false
            selectedSchedule = componentArtifact.schedule
        }
        let policyNote = performanceSelected
            ? String(format: "selected %@ beats its own paired incumbent matched-repeat median summed wall by %.1f%% (> %.0f%% policy threshold)",
                     selectedSchedule.kernelName,
                     (1 - bestTiming!.candMed / bestTiming!.incMed) * 100,
                     GemmaForge.noiseRelWall * 100)
            : "incumbent fallback: no gate-passing candidate beat its own paired incumbent beyond the policy threshold on matched-repeat medians"
        log.append("policy: \(policyNote)")

        // ---- persist the full artifact ----
        let selectedSource = try RMSKernelEmitter.emit(op: op, schedule: selectedSchedule)
        let fixtureRecords = fixtures.map { fixture -> RMSFixtureRecord in
            let ref = refArms[fixture.name]!
            var history = fixture.tokens
            history.append(contentsOf: ref.ids.dropLast())
            return RMSFixtureRecord(
                name: fixture.name, promptTokenCount: fixture.tokens.count,
                gatedSteps: ref.ids.count,
                historyDigest: GemmaForge.workloadDigest(history))
        }
        let gateRecord = RMSModelGateRecord(
            genTokens: genTokens, timedTokens: timedTokens, fixtures: fixtureRecords,
            results: modelResults, incumbentWallPerFixture: incWalls,
            incumbentGpuPerFixture: incGpus,
            performanceSelected: performanceSelected, policyNote: policyNote)
        let body = RMSLoweringArtifact(
            artifactSchema: componentArtifact.artifactSchema,
            codegenVersion: componentArtifact.codegenVersion,
            opDigest: componentArtifact.opDigest, op: op, schedule: selectedSchedule,
            kernelName: selectedSchedule.kernelName, emittedSource: selectedSource,
            sourceSha256: SHA256.hex(SHA256.hash(Array(selectedSource.utf8))),
            hardware: componentArtifact.hardware, toolchain: Forge.toolchainIdentifier,
            ggufDigest: ggufDigest,
            executableDigest: try Gemma4Runner.executableDigest(),
            componentResults: componentArtifact.componentResults,
            incumbentWorstRelToOracle: componentArtifact.incumbentWorstRelToOracle,
            representativeVectors: componentArtifact.representativeVectors,
            modelGate: gateRecord, artifactDigest: "")
        let digest = try RMSLoweringArtifact.bodyDigest(body)
        let artifact = RMSLoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeVectors: body.representativeVectors,
            modelGate: gateRecord, artifactDigest: digest)
        try artifact.save(to: url)

        // ---- 6. persisted-execution proof ----
        let reloaded = try RMSLoweringArtifact.load(from: url)
        try reloaded.validate(op: op, hardware: componentArtifact.hardware,
                              ggufDigest: ggufDigest,
                              executableDigest: try Gemma4Runner.executableDigest())
        try runner.installRmsGeneratedKernel(artifact: reloaded, ggufDigest: ggufDigest)
        let first = fixtures[0]
        let (worst, fail, dispatches) = gateReplay(first, refArms[first.name]!)
        runner.uninstallRmsGeneratedKernel()
        guard fail == nil, dispatches > 0 else {
            throw RMSLoweringError.componentGateFailed(
                "persisted execution failed: \(fail ?? "no generated dispatches")")
        }
        // Bool must never cross String(format:) %@ as a C vararg: it is not
        // an Objective-C object and crashes objc_opt_respondsToSelector.
        log.append(rmsPersistedExecutionLogLine(fixtureName: first.name, worstRel: worst,
                                                dispatches: dispatches, n: op.n,
                                                scaled: op.scaled, tensor: op.tensor))
        return (artifact, log)
    }
}
