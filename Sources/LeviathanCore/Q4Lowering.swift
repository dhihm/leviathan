// Phase 9 — parameterized Q4_0 operator lowering (ONE operator family).
//
// Pipeline: Python op input (.py with semantics/shape/layout/quant/
// precision metadata) -> Q4OpIR validation -> shape+hardware-derived
// schedule enumeration -> Metal source EMISSION from the schedule
// parameters -> independent CPU FP64 oracle gate (existing 1e-4
// component contract) -> Q4LoweringArtifact with full identity binding
// -> persisted execution that genuinely dispatches the emitted, compiled
// pipeline, failing closed to the incumbent kernel.
//
// This is deliberately bounded to the Q4_0 matvec family; it is not a
// general arbitrary-Python compiler. See docs/Q4_LOWERING_DESIGN.md.

import Foundation
import CryptoKit
import Metal

func q4IsLowercaseHex64(_ s: String) -> Bool {
    s.count == 64 && s.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") }
}

public enum Q4LoweringError: Error, CustomStringConvertible, Equatable {
    case invalidOp(String)
    case invalidSchedule(String)
    case emissionFailed(String)
    case oracleFailed(String)
    case componentGateFailed(String)
    case invalidArtifact(String)
    case hardwareUnavailable(String)

    public var description: String {
        switch self {
        case .invalidOp(let m): return "q4 op invalid: \(m)"
        case .invalidSchedule(let m): return "q4 schedule invalid: \(m)"
        case .emissionFailed(let m): return "q4 emission failed: \(m)"
        case .oracleFailed(let m): return "q4 oracle failed: \(m)"
        case .componentGateFailed(let m): return "q4 component gate failed: \(m)"
        case .invalidArtifact(let m): return "q4 artifact invalid: \(m)"
        case .hardwareUnavailable(let m): return "q4 hardware unavailable: \(m)"
        }
    }
}

// MARK: - Operation IR

/// The validated Q4_0 matvec operation: y[row] = dot(W_row, x) with W in
/// row-major Q4_0 blocks. Parsed from the Python-captured op JSON; every
/// field is checked and anything outside the supported operation is
/// rejected before any lowering happens.
public struct Q4OpIR: Codable, Equatable, Sendable {
    public let schema: Int
    public let op: String
    public let semantics: String
    public let inDim: Int
    public let outDim: Int
    public let layout: String
    public let quant: String
    public let accumulate: String
    public let inputPrecision: String
    /// Optional real GGUF tensor binding for representative weight bytes;
    /// when present, the declared shape MUST match the tensor's actual
    /// GGUF dimensions (dims[0]=inDim, dims[1]=outDim) or lowering is
    /// rejected.
    public let tensor: String?
    /// Bounded representative row count for component validation reads
    /// (1...outDim; defaults to a bounded slice when nil).
    public let representativeRows: Int?

    public static let supportedSchema = 1

    public static func parse(_ data: Data) throws -> Q4OpIR {
        guard data.count <= 1 << 16 else {
            throw Q4LoweringError.invalidOp("op JSON exceeds 64 KiB")
        }
        let op: Q4OpIR
        do { op = try JSONDecoder().decode(Q4OpIR.self, from: data) }
        catch { throw Q4LoweringError.invalidOp("op JSON decode: \(error)") }
        try op.validate()
        return op
    }

    public func validate() throws {
        guard schema == Self.supportedSchema else {
            throw Q4LoweringError.invalidOp("schema \(schema) != \(Self.supportedSchema)")
        }
        guard op == "q4_0.matvec", semantics == "dot(W_row, x)" else {
            throw Q4LoweringError.invalidOp("unsupported op/semantics \(op)/\(semantics)")
        }
        guard layout == "row-major-q4_0-blocks", quant == "q4_0" else {
            throw Q4LoweringError.invalidOp("unsupported layout/quant \(layout)/\(quant)")
        }
        guard accumulate == "f32", inputPrecision == "f32" else {
            throw Q4LoweringError.invalidOp("unsupported accumulate/inputPrecision")
        }
        guard inDim > 0, outDim > 0, inDim % Q4_0.valuesPerBlock == 0,
              inDim <= ModelWorkloadLimits.maxQ4InputDim,
              outDim <= ModelWorkloadLimits.maxQ4OutputDim else {
            throw Q4LoweringError.invalidOp("bad dims inDim \(inDim) outDim \(outDim)")
        }
        if let rows = representativeRows {
            guard rows > 0, rows <= outDim, rows <= 4096 else {
                throw Q4LoweringError.invalidOp("representativeRows \(rows) out of bounds")
            }
        }
    }

    /// Cross-checks the declared shape against the BOUND tensor's actual
    /// GGUF dimensions and quant type. Fail-closed on any mismatch.
    public func validateAgainst(file: GGUFFile) throws {
        guard let tensor else { return }
        guard let info = try? file.tensor(tensor) else {
            throw Q4LoweringError.invalidOp("bound tensor \(tensor) not in GGUF")
        }
        guard info.ggmlType == 2 else {
            throw Q4LoweringError.invalidOp(
                "bound tensor \(tensor) ggmlType \(info.ggmlType) is not Q4_0")
        }
        guard info.dims.count == 2, info.dims[0] == inDim, info.dims[1] == outDim else {
            throw Q4LoweringError.invalidOp(
                "bound tensor \(tensor) dims \(info.dims) != declared [\(inDim), \(outDim)]")
        }
    }

    /// SHA-256 over the canonical (sorted-keys) JSON encoding.
    public func digest() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(self)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    /// The bounded row count actually read for component validation.
    public var componentRows: Int { min(representativeRows ?? 256, outDim, 4096) }
}

// MARK: - Schedule

/// One parameterized lowering schedule for the Q4_0 matvec. Every field
/// has a REAL effect on the emitted Metal source: thread mapping
/// (threadsPerRow), row/column work decomposition (rowsPerGroup, the
/// block-chunk split), reduction strategy (accumulation), and memory
/// access (xLoads).
public struct Q4Schedule: Codable, Equatable, Hashable, Sendable {
    public enum Accumulation: String, Codable, Sendable {
        /// Plain float32 partials and ascending-order combine (an
        /// explicitly reassociated deterministic order for T>1; the T=1
        /// member reproduces the incumbent serial order).
        case f32
        /// Float32 partials carrying a Neumaier compensation term through
        /// the per-thread accumulation and the ordered combine — a
        /// deterministic error-compensated reduction, still float32
        /// arithmetic on the GPU (Apple GPUs have no usable FP64).
        case f32compensated
    }

    public enum XLoads: String, Codable, Sendable {
        case scalar
        case float4
    }

    public let threadsPerRow: Int
    public let rowsPerGroup: Int
    public let accumulation: Accumulation
    public let xLoads: XLoads

    public init(threadsPerRow: Int, rowsPerGroup: Int,
                accumulation: Q4Schedule.Accumulation, xLoads: Q4Schedule.XLoads) {
        self.threadsPerRow = threadsPerRow
        self.rowsPerGroup = rowsPerGroup
        self.accumulation = accumulation
        self.xLoads = xLoads
    }

    public var kernelName: String {
        "q4gen_matvec_T\(threadsPerRow)_R\(rowsPerGroup)_\(accumulation.rawValue)_\(xLoads.rawValue)"
    }

    /// Legality against the operation shape and the measured device:
    /// power-of-two T, bounded threadgroup occupancy, and — the SHAPE
    /// input — at least `minWorkBlocks` Q4 blocks per thread so small
    /// geometries legally admit only small T.
    public func isLegal(blocksPerRow: Int, hardware: ForgeHardwareProfile,
                        minWorkBlocks: Int = 4) -> Bool {
        let t = threadsPerRow
        guard t > 0, t & (t - 1) == 0, t <= 64,
              rowsPerGroup > 0, [1, 2, 4].contains(rowsPerGroup),
              t * rowsPerGroup <= hardware.maxTotalThreadsPerThreadgroup,
              blocksPerRow / t >= minWorkBlocks else { return false }
        return true
    }
}

// MARK: - Schedule derivation (shape × measured hardware)

public enum Q4ScheduleDeriver {
    public static let maxCandidates = 8

    /// Derives the bounded legal schedule set from the operation SHAPE and
    /// the MEASURED hardware. Candidate 0 is always the incumbent-mapping
    /// baseline (T=1, R=1, plain f32, scalar loads — the same one-thread-
    /// per-row serial summation order as the reference kernel); the rest
    /// are ranked by proximity of T to the measured execution width.
    /// Microbenchmarks never select anything here — this list only defines
    /// what is LEGAL to emit and gate.
    public static func derive(op: Q4OpIR, hardware: ForgeHardwareProfile) -> [Q4Schedule] {
        let blocksPerRow = op.inDim / Q4_0.valuesPerBlock
        let baseline = Q4Schedule(threadsPerRow: 1, rowsPerGroup: 1,
                                  accumulation: .f32, xLoads: .scalar)
        var result = [baseline]
        guard baseline.isLegal(blocksPerRow: blocksPerRow, hardware: hardware) else {
            return result
        }
        // Shape-derived T ladder, ranked by proximity to the execution
        // width; only the two nearest legal values survive the bound.
        let w = max(1, hardware.threadExecutionWidth)
        var ladder = [Int]()
        for t in [2, 4, 8, 16, 32] {
            let probe = Q4Schedule(threadsPerRow: t, rowsPerGroup: 1,
                                   accumulation: .f32, xLoads: .scalar)
            if probe.isLegal(blocksPerRow: blocksPerRow, hardware: hardware) {
                ladder.append(t)
            }
        }
        ladder.sort { abs($0 - w) < abs($1 - w) || (abs($0 - w) == abs($1 - w) && $0 < $1) }
        let ts = Array(ladder.prefix(2))
        for t in ts {
            result.append(Q4Schedule(threadsPerRow: t, rowsPerGroup: 1,
                                     accumulation: .f32, xLoads: .scalar))
            result.append(Q4Schedule(threadsPerRow: t, rowsPerGroup: 1,
                                     accumulation: .f32compensated, xLoads: .scalar))
            result.append(Q4Schedule(threadsPerRow: t, rowsPerGroup: 1,
                                     accumulation: .f32compensated, xLoads: .float4))
        }
        // One cooperative row-group entry on the width-nearest T.
        if let t = ts.first {
            let s = Q4Schedule(threadsPerRow: t, rowsPerGroup: 2,
                               accumulation: .f32compensated, xLoads: .float4)
            if s.isLegal(blocksPerRow: blocksPerRow, hardware: hardware) {
                result.append(s)
            }
        }
        return Array(result.prefix(maxCandidates))
    }
}

// MARK: - Metal source emission

public enum Q4KernelEmitter {
    /// Bumped whenever emission rules change; part of artifact identity.
    public static let codegenVersion = "q4gen-v1"

    /// Emits the complete MSL source for one schedule. The thread mapping,
    /// block-chunk decomposition, reduction structure, and load width are
    /// generated FROM the schedule parameters and the op shape — authored
    /// rules, parameterized derivation, not a table of prewritten kernels.
    public static func emit(op: Q4OpIR, schedule: Q4Schedule) throws -> String {
        let blocksPerRow = op.inDim / Q4_0.valuesPerBlock
        guard blocksPerRow > 0 else {
            throw Q4LoweringError.emissionFailed("inDim \(op.inDim) has no Q4 blocks")
        }
        // Emission-time legality is SHAPE-only (the deriver and the
        // artifact gates re-check against the measured hardware).
        let shapeOnly = ForgeHardwareProfile(
            deviceName: "emit", maxTotalThreadsPerThreadgroup: .max,
            threadExecutionWidth: 1, recommendedMaxWorkingSetSize: 0)
        guard schedule.isLegal(blocksPerRow: blocksPerRow, hardware: shapeOnly) else {
            throw Q4LoweringError.emissionFailed(
                "schedule \(schedule) illegal for inDim \(op.inDim)")
        }
        let t = schedule.threadsPerRow
        let r = schedule.rowsPerGroup
        let kahan = schedule.accumulation == .f32compensated
        let inner: String
        switch schedule.xLoads {
        case .scalar:
            inner = """
            for (uint l = 0; l < 16u; l++) {
                uchar q = blk[2u + l];
                term = d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];
                ADD(term);
                term = d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];
                ADD(term);
            }
            """
        case .float4:
            inner = """
            float4 xa = *(device const float4*)(x + b * 32u);
            float4 xb = *(device const float4*)(x + b * 32u + 4u);
            float4 xc = *(device const float4*)(x + b * 32u + 8u);
            float4 xd = *(device const float4*)(x + b * 32u + 12u);
            float4 xe = *(device const float4*)(x + b * 32u + 16u);
            float4 xf = *(device const float4*)(x + b * 32u + 20u);
            float4 xg = *(device const float4*)(x + b * 32u + 24u);
            float4 xh = *(device const float4*)(x + b * 32u + 28u);
            ADD(d * (float(blk[2u] & 0x0Fu) - 8.0f) * xa.x);
            ADD(d * (float(blk[2u] >> 4) - 8.0f) * xe.x);
            ADD(d * (float(blk[3u] & 0x0Fu) - 8.0f) * xa.y);
            ADD(d * (float(blk[3u] >> 4) - 8.0f) * xe.y);
            ADD(d * (float(blk[4u] & 0x0Fu) - 8.0f) * xa.z);
            ADD(d * (float(blk[4u] >> 4) - 8.0f) * xe.z);
            ADD(d * (float(blk[5u] & 0x0Fu) - 8.0f) * xa.w);
            ADD(d * (float(blk[5u] >> 4) - 8.0f) * xe.w);
            ADD(d * (float(blk[6u] & 0x0Fu) - 8.0f) * xb.x);
            ADD(d * (float(blk[6u] >> 4) - 8.0f) * xf.x);
            ADD(d * (float(blk[7u] & 0x0Fu) - 8.0f) * xb.y);
            ADD(d * (float(blk[7u] >> 4) - 8.0f) * xf.y);
            ADD(d * (float(blk[8u] & 0x0Fu) - 8.0f) * xb.z);
            ADD(d * (float(blk[8u] >> 4) - 8.0f) * xf.z);
            ADD(d * (float(blk[9u] & 0x0Fu) - 8.0f) * xb.w);
            ADD(d * (float(blk[9u] >> 4) - 8.0f) * xf.w);
            ADD(d * (float(blk[10u] & 0x0Fu) - 8.0f) * xc.x);
            ADD(d * (float(blk[10u] >> 4) - 8.0f) * xg.x);
            ADD(d * (float(blk[11u] & 0x0Fu) - 8.0f) * xc.y);
            ADD(d * (float(blk[11u] >> 4) - 8.0f) * xg.y);
            ADD(d * (float(blk[12u] & 0x0Fu) - 8.0f) * xc.z);
            ADD(d * (float(blk[12u] >> 4) - 8.0f) * xg.z);
            ADD(d * (float(blk[13u] & 0x0Fu) - 8.0f) * xc.w);
            ADD(d * (float(blk[13u] >> 4) - 8.0f) * xg.w);
            ADD(d * (float(blk[14u] & 0x0Fu) - 8.0f) * xd.x);
            ADD(d * (float(blk[14u] >> 4) - 8.0f) * xh.x);
            ADD(d * (float(blk[15u] & 0x0Fu) - 8.0f) * xd.y);
            ADD(d * (float(blk[15u] >> 4) - 8.0f) * xh.y);
            ADD(d * (float(blk[16u] & 0x0Fu) - 8.0f) * xd.z);
            ADD(d * (float(blk[16u] >> 4) - 8.0f) * xh.z);
            ADD(d * (float(blk[17u] & 0x0Fu) - 8.0f) * xd.w);
            ADD(d * (float(blk[17u] >> 4) - 8.0f) * xh.w);
            """
        }
        // Reduction macros: plain f32 or Neumaier-compensated f32. The
        // per-thread partial and the ascending-thread combine share the
        // strategy; everything stays float32 on the GPU.
        let accumulationDecls: String
        let combineBody: String
        if kahan {
            accumulationDecls = """
            float sum = 0.0f;
            float comp = 0.0f;
            float term = 0.0f;
            #define ADD(termv) do { \
                float tv = (termv); \
                float yv = sum + tv; \
                if (fabs(sum) >= fabs(tv)) { comp += (sum - yv) + tv; } \
                else { comp += (tv - yv) + sum; } \
                sum = yv; \
            } while (0)
            """
            combineBody = """
            float acc = 0.0f;
            float accc = 0.0f;
            for (uint i = 0u; i < T; i++) {
                float tv = partials[r * T + i];
                float yv = acc + tv;
                if (fabs(acc) >= fabs(tv)) { accc += (acc - yv) + tv; }
                else { accc += (tv - yv) + acc; }
                acc = yv;
            }
            y[row] = acc + accc;
            """
        } else {
            accumulationDecls = """
            float sum = 0.0f;
            float comp = 0.0f;
            float term = 0.0f;
            #define ADD(termv) do { sum += (termv); } while (0)
            """
            combineBody = """
            float acc = 0.0f;
            for (uint i = 0u; i < T; i++) { acc += partials[r * T + i]; }
            y[row] = acc;
            """
        }
        return """
        #include <metal_stdlib>
        using namespace metal;

        // \(schedule.kernelName) — EMITTED by LeviathanForge Q4 lowering
        // (\(codegenVersion)) from schedule \(schedule) and op shape
        // inDim=\(op.inDim) (blocksPerRow=\(blocksPerRow)).
        // T=\(t) threads cooperate per row over contiguous block chunks;
        // R=\(r) rows per threadgroup; accumulation=\(schedule.accumulation.rawValue);
        // xLoads=\(schedule.xLoads.rawValue). Partials combine in ascending
        // thread order — an explicitly reassociated deterministic order
        // (NOT bit-identical to the one-thread-per-row serial kernel for
        // T>1); gated against the independent CPU FP64 oracle and the
        // unchanged model-level 1e-4 gate.
        kernel void \(schedule.kernelName)(constant const uint2& p [[buffer(0)]],
                                            constant const ulong& wOff [[buffer(1)]],
                                            device const uchar* wbase [[buffer(2)]],
                                            device const float* x [[buffer(3)]],
                                            device float* y [[buffer(4)]],
                                            uint lid [[thread_position_in_threadgroup]],
                                            uint wid [[threadgroup_position_in_grid]]) {
            constexpr uint T = \(t)u;
            constexpr uint R = \(r)u;
            uint row = wid * R + (lid / T);
            uint r = lid / T;
            uint t = lid % T;
            bool active = row < p.y;
            uint blocksPerRow = p.x / 32u;
            uint chunk = (blocksPerRow + T - 1u) / T;
            uint b0 = t * chunk;
            uint b1 = min(b0 + chunk, blocksPerRow);
            device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 18u;
        \(accumulationDecls)
            if (active) {
                for (uint b = b0; b < b1; b++) {
                    device const uchar* blk = rowBytes + b * 18u;
                    ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
                    float d = float(as_type<half>(dbits));
        \(inner)
                }
            }
            threadgroup float partials[\(t * r)];
            partials[r * T + t] = sum + comp;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active && t == 0u) {
        \(combineBody)
            }
        }
        """
    }
}

// MARK: - Independent CPU FP64 oracle

public enum Q4Oracle {
    /// Independent Double-precision Q4_0 matvec. Decodes the FP16 scale
    /// and the biased nibble levels in Double and accumulates the dot in
    /// Double — a genuinely independent reference for diagnosing float32
    /// reduction differences (the incumbent serial kernel is measured
    /// against it too, never assumed exact).
    public static func matvecFP64(blocks: ArraySlice<UInt8>, outDim: Int, inDim: Int,
                                  x: [Float]) throws -> [Double] {
        guard inDim > 0, outDim > 0, inDim % Q4_0.valuesPerBlock == 0 else {
            throw Q4LoweringError.oracleFailed("bad dims \(inDim) x \(outDim)")
        }
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        let needed = outDim * blocksPerRow * Q4_0.bytesPerBlock
        guard blocks.count == needed, x.count == inDim,
              x.allSatisfy({ $0.isFinite }) else {
            throw Q4LoweringError.oracleFailed(
                "oracle geometry: \(blocks.count) bytes (need \(needed)), x \(x.count)")
        }
        let base = blocks.startIndex
        var y = [Double](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum = 0.0
            for b in 0..<blocksPerRow {
                let o = base + (row * blocksPerRow + b) * Q4_0.bytesPerBlock
                let bits = UInt16(blocks[o]) | UInt16(blocks[o + 1]) << 8
                guard bits & 0x7C00 != 0x7C00 else {
                    throw Q4LoweringError.oracleFailed("nonfinite scale block \((o - base) / Q4_0.bytesPerBlock)")
                }
                let d = Double(Float16(bitPattern: bits))
                for l in 0..<16 {
                    let q = blocks[o + 2 + l]
                    sum += d * (Double(q & 0xF) - 8.0) * Double(x[b * 32 + l])
                    sum += d * (Double(q >> 4) - 8.0) * Double(x[b * 32 + 16 + l])
                }
            }
            guard sum.isFinite else {
                throw Q4LoweringError.oracleFailed("nonfinite output row \(row)")
            }
            y[row] = sum
        }
        return y
    }
}

// MARK: - Component candidate evaluation

public struct Q4ComponentResult: Codable, Equatable, Sendable {
    public let schedule: Q4Schedule
    public let accepted: Bool
    /// Worst |gpu - ref64| / max(1, |ref64|) over the representative rows.
    public let worstRelToOracle: Double?
    /// Worst |gpu - incumbentGpu| / max(1, |incumbentGpu|): the emitted
    /// kernel's distance from the incumbent kernel's own outputs.
    public let worstRelToIncumbent: Double?
    /// Sum of |products| at the worst-oracle row (FP64) —
    /// cancellation context for a small result with large terms.
    public let worstRowSumAbsProducts: Double?
    public let note: String
}

/// Bounded representative weight bytes for the op: real GGUF tensor rows
/// when the op binds a tensor (pread, never the whole tensor set), or a
/// deterministic synthetic Q4_0 matrix when unbound.
public enum Q4ComponentInputs {
    public static func weightBytes(op: Q4OpIR, file: GGUFFile?) throws -> [UInt8] {
        let rows = op.componentRows
        let blocksPerRow = op.inDim / Q4_0.valuesPerBlock
        if let file, let tensor = op.tensor {
            try op.validateAgainst(file: file)
            let info: GGUFTensorInfo
            do { info = try file.tensor(tensor) } catch {
                throw Q4LoweringError.invalidOp("bound tensor \(tensor): \(error)")
            }
            let range = try file.tensorRange(info)
            let needed = rows * blocksPerRow * Q4_0.bytesPerBlock
            guard needed <= range.count else {
                throw Q4LoweringError.invalidOp(
                    "representative rows \(rows) exceed tensor extent \(range.count)")
            }
            guard let url = file.sourceURL else {
                throw Q4LoweringError.invalidOp("GGUF has no source URL for bounded read")
            }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(range.lowerBound))
            guard let data = try handle.read(upToCount: needed), data.count == needed else {
                throw Q4LoweringError.invalidOp("bounded tensor read failed")
            }
            return Array(data)
        }
        // Deterministic synthetic blocks with guaranteed-finite FP16
        // scales: any drawn exponent-all-ones pattern is replaced by the
        // finite FP16 encoding of 1.0 (0x3C00).
        var rng = SplitMix64(seed: 0x5A4C_4F57)
        var blocks = [UInt8]()
        blocks.reserveCapacity(rows * blocksPerRow * Q4_0.bytesPerBlock)
        for _ in 0..<(rows * blocksPerRow) {
            blocks += [UInt8(rng.next() & 0xFF), UInt8(rng.next() & 0xFF)]
            let n = blocks.count
            let expBits = (UInt16(blocks[n - 2]) | UInt16(blocks[n - 1]) << 8) & 0x7C00
            if expBits == 0x7C00 {
                blocks[n - 2] = 0x00
                blocks[n - 1] = 0x3C
            }
            for _ in 0..<16 { blocks.append(UInt8(rng.next() & 0xFF)) }
        }
        return blocks
    }

    public static func activation(op: Q4OpIR, seed: UInt64 = 70_004) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<op.inDim).map { _ in rng.nextUniform() }
    }
}

/// Standalone GPU evaluation of one emitted kernel (component level). The
/// kernel bindings match the incumbent g4_q4_matvec contract exactly
/// (p, wOff, wbase, x, y), so the same call site drives both.
public struct Q4EmittedKernel: Sendable {
    public let schedule: Q4Schedule
    public let source: String
    public let pipeline: MTLComputePipelineState
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public static func compile(op: Q4OpIR, schedule: Q4Schedule,
                               device: MTLDevice) throws -> Q4EmittedKernel {
        let source = try Q4KernelEmitter.emit(op: op, schedule: schedule)
        guard let queue = device.makeCommandQueue() else {
            throw Q4LoweringError.hardwareUnavailable("no command queue")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw Q4LoweringError.emissionFailed("MSL compile: \(error)") }
        guard let fn = library.makeFunction(name: schedule.kernelName),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw Q4LoweringError.emissionFailed("pipeline for \(schedule.kernelName)")
        }
        // Occupancy legality on the ACTUAL compiled pipeline: the emitted
        // threadgroup (T*R threads plus the partials array) must fit.
        guard schedule.threadsPerRow * schedule.rowsPerGroup
                <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw Q4LoweringError.emissionFailed(
                "\(schedule.kernelName): T*R exceeds compiled threadgroup maximum")
        }
        return Q4EmittedKernel(schedule: schedule, source: source,
                               pipeline: pipeline, device: device, queue: queue)
    }

    public func matvec(blocks: [UInt8], outDim: Int, inDim: Int,
                       x: [Float]) throws -> [Float] {
        let rows = outDim
        guard blocks.count == rows * (inDim / 32) * 18, x.count == inDim else {
            throw Q4LoweringError.componentGateFailed("component geometry mismatch")
        }
        guard let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                         options: .storageModeShared),
              let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                         options: .storageModeShared),
              let yb = device.makeBuffer(length: rows * 4, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw Q4LoweringError.hardwareUnavailable("buffer/encoder allocation")
        }
        var p = SIMD2<UInt32>(UInt32(inDim), UInt32(rows))
        var off = UInt64(0)
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        let threads = schedule.threadsPerRow * schedule.rowsPerGroup
        let groups = (rows + schedule.rowsPerGroup - 1) / schedule.rowsPerGroup
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4LoweringError.componentGateFailed(
                "status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        let out = [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: rows))
        return out
    }
}

// MARK: - Artifact

/// One bounded chat fixture used by the model-level gate.
public struct Q4FixtureRecord: Codable, Equatable, Sendable {
    public let name: String
    public let promptTokenCount: Int
    /// Actual gated greedy steps for this fixture (may be < genTokens when
    /// the reference history hit end-of-generation).
    public let gatedSteps: Int
    /// Workload digest of the EXACT teacher-forced history: prompt tokens
    /// plus every reference greedy id the replay forced (the full history
    /// the gate compared against — prompt digest alone would be weaker).
    public let historyDigest: String
}

/// Raw samples of ONE timing arm on ONE fixture: per-timed-token wall and
/// GPU-busy samples plus the actual history position window (EOS-truncated
/// histories time fewer tokens at true positions — recorded, not clamped).
public struct Q4TimingArmFixture: Codable, Equatable, Sendable {
    public let walls: [Double]
    public let gpus: [Double]
    public let startIdx: Int
    public let timedCount: Int
}

/// One PAIRED repetition: the candidate arm and the incumbent arm run
/// back-to-back with alternating order across repetitions (AB/BA/AB) so
/// drift affects both arms equally.
public struct Q4TimingRepetition: Codable, Equatable, Sendable {
    public let candidateFirst: Bool
    public let candidate: [Q4TimingArmFixture]
    public let incumbent: [Q4TimingArmFixture]
}

/// Model-level gate + matched-timing record for ONE generated candidate.
public struct Q4ModelCandidateResult: Codable, Equatable, Sendable {
    public let schedule: Q4Schedule
    /// Fixtures where EVERY step passed: full-logit rel < 1e-4, exact
    /// greedy argmax equality, all-finite.
    public let fixturesPassed: Int
    public let fixturesFailed: Int
    /// Worst full-logit rel over all steps of all passing fixtures.
    public let worstRel: Double?
    /// First failure reason (nil when all fixtures passed).
    public let gateFailReason: String?
    /// Per-fixture median WALL us/token over the timed tokens (matched
    /// arms: same binary/model/prompt/context, sequential residency).
    /// var: the orchestrator fills these only after the gate pass.
    public var wallMedPerFixture: [Double]?
    /// Per-fixture median GPU-busy us/token (reported separately; no
    /// per-kernel attribution is claimed in segment-batched mode).
    public var gpuMedPerFixture: [Double]?
    /// Raw paired timing repetitions (candidate + incumbent arms,
    /// alternating order); nil when the candidate failed the gates.
    public var timingReps: [Q4TimingRepetition]?
}

/// The full model-level selection record persisted with the artifact.
public struct Q4ModelGateRecord: Codable, Equatable, Sendable {
    public let genTokens: Int
    public let timedTokens: Int
    public let fixtures: [Q4FixtureRecord]
    public let results: [Q4ModelCandidateResult]
    /// Incumbent arm per-fixture medians (same matched protocol).
    public let incumbentWallPerFixture: [Double]
    public let incumbentGpuPerFixture: [Double]
    /// Policy outcome: true only when a generated candidate passed ALL
    /// fixture gates AND beat the incumbent summed wall medians by more
    /// than GemmaForge.noiseRelWall (the same documented 2% policy
    /// threshold the whole-model forge uses). false = incumbent fallback.
    public let performanceSelected: Bool
    public let policyNote: String
}

public struct Q4LoweringArtifact: Codable, Equatable, Sendable {
    public static let schema = 1

    public let artifactSchema: Int
    public let codegenVersion: String
    /// SHA-256 of the canonical op JSON (semantic IR identity).
    public let opDigest: String
    public let op: Q4OpIR
    public let schedule: Q4Schedule
    public let kernelName: String
    /// The emitted Metal source itself, persisted with the artifact.
    public let emittedSource: String
    /// SHA-256 of the emitted source bytes.
    public let sourceSha256: String
    /// Hardware the schedule was derived and gated on.
    public let hardware: ForgeHardwareProfile
    public let toolchain: String
    /// GGUF content digest at forge time (empty when the op is unbound).
    public let ggufDigest: String
    /// Digest of the forging executable — a rebuilt binary re-forges.
    public let executableDigest: String
    /// Component gate record: per-candidate oracle results and the
    /// incumbent's own measured distance to the FP64 oracle.
    public let componentResults: [Q4ComponentResult]
    public let incumbentWorstRelToOracle: Double
    public let representativeRows: Int
    /// Model-level gate/timing/selection record (present when the
    /// orchestration ran the bounded fixtures; nil for component-only
    /// artifacts).
    public let modelGate: Q4ModelGateRecord?
    public let artifactDigest: String

    /// Canonical-body digest (all fields except artifactDigest).
    static func bodyDigest(_ body: Q4LoweringArtifact) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stripped = Q4LoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeRows: body.representativeRows,
            modelGate: body.modelGate, artifactDigest: "")
        let bytes = try encoder.encode(stripped)
        return SHA256.hex(SHA256.hash(Array(bytes)))
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: [.atomic])
    }

    public static func load(from url: URL) throws -> Q4LoweringArtifact {
        guard let data = try? Data(contentsOf: url) else {
            throw Q4LoweringError.invalidArtifact("unreadable artifact \(url.path)")
        }
        guard data.count <= 1 << 20 else {
            throw Q4LoweringError.invalidArtifact("artifact exceeds 1 MiB")
        }
        do { return try JSONDecoder().decode(Q4LoweringArtifact.self, from: data) }
        catch { throw Q4LoweringError.invalidArtifact("decode: \(error)") }
    }

    /// Full identity validation. Fail-closed: any mismatch (schema, op
    /// digest vs the CURRENT op, codegen version, re-EMITTED source bytes
    /// vs the persisted source, source SHA, hardware, toolchain, GGUF,
    /// executable, component-gate consistency, self digest) rejects the
    /// artifact — the caller falls back to the incumbent kernel.
    public func validate(op currentOp: Q4OpIR, hardware: ForgeHardwareProfile,
                         ggufDigest: String, executableDigest: String) throws {
        // The passed op is re-validated here (a decoded-but-unvalidated op
        // must not slide through on the strength of `op == currentOp`
        // alone — callers often pass artifact.op itself).
        try currentOp.validate()
        try op.validate()
        guard artifactSchema == Self.schema else {
            throw Q4LoweringError.invalidArtifact("schema \(artifactSchema)")
        }
        guard codegenVersion == Q4KernelEmitter.codegenVersion else {
            throw Q4LoweringError.invalidArtifact("codegen \(codegenVersion) stale")
        }
        guard op == currentOp, opDigest == (try currentOp.digest()) else {
            throw Q4LoweringError.invalidArtifact("op mismatch vs current operation")
        }
        guard kernelName == schedule.kernelName else {
            throw Q4LoweringError.invalidArtifact("kernel name inconsistent with schedule")
        }
        // The persisted source must be exactly what the CURRENT emission
        // rules produce for this (op, schedule) — tampering or rule drift
        // is rejected, not silently executed.
        let reemitted = try Q4KernelEmitter.emit(op: op, schedule: schedule)
        guard reemitted == emittedSource else {
            throw Q4LoweringError.invalidArtifact("emitted source tampered/rule drift")
        }
        guard sourceSha256 == SHA256.hex(SHA256.hash(Array(emittedSource.utf8))) else {
            throw Q4LoweringError.invalidArtifact("source SHA mismatch")
        }
        guard hardware.maxTotalThreadsPerThreadgroup > 0,
              hardware.threadExecutionWidth > 0,
              hardware.recommendedMaxWorkingSetSize > 0,
              hardware == self.hardware else {
            throw Q4LoweringError.invalidArtifact("hardware mismatch")
        }
        guard toolchain == Forge.toolchainIdentifier else {
            throw Q4LoweringError.invalidArtifact("toolchain mismatch")
        }
        guard ggufDigest == self.ggufDigest else {
            throw Q4LoweringError.invalidArtifact("GGUF digest mismatch")
        }
        guard executableDigest == self.executableDigest else {
            throw Q4LoweringError.invalidArtifact("executable digest mismatch")
        }
        guard !componentResults.isEmpty,
              componentResults.contains(where: { $0.schedule == schedule && $0.accepted }) else {
            throw Q4LoweringError.invalidArtifact("selected schedule not gate-accepted")
        }
        // The component-accepted entry for the persisted schedule must
        // carry a finite, in-contract oracle distance.
        if let sel = componentResults.first(where: { $0.schedule == schedule && $0.accepted }) {
            guard let rel = sel.worstRelToOracle, rel.isFinite, rel >= 0, rel < 1e-4 else {
                throw Q4LoweringError.invalidArtifact(
                    "component record for selected schedule not within the 1e-4 contract")
            }
        }
        // Model-gate record consistency (when present): the persisted
        // schedule must have passed EVERY fixture gate — even when the
        // policy selected the incumbent for performance
        // (performanceSelected == false), a gate-failing candidate must
        // never activate.
        if let mg = modelGate {
            guard mg.genTokens >= 2, mg.timedTokens >= 1,
                  mg.timedTokens < mg.genTokens,
                  !mg.fixtures.isEmpty,
                  mg.fixtures.allSatisfy({
                      $0.gatedSteps >= 2 && $0.gatedSteps <= mg.genTokens
                          && $0.promptTokenCount > 0
                          && q4IsLowercaseHex64($0.historyDigest) }),
                  Set(mg.fixtures.map(\.name)).count == mg.fixtures.count,
                  let sel = mg.results.first(where: { $0.schedule == schedule }),
                  sel.fixturesPassed == mg.fixtures.count,
                  sel.fixturesFailed == 0,
                  let worst = sel.worstRel, worst < 1e-4 else {
                throw Q4LoweringError.invalidArtifact(
                    "model-gate record inconsistent or schedule not fully gate-passed")
            }
        }
        guard artifactDigest == (try Self.bodyDigest(self)) else {
            throw Q4LoweringError.invalidArtifact("artifact digest tampered")
        }
    }
}

// MARK: - Component forge (the q4-lower pipeline, no model run)

public enum Q4CurrentHardware {
    /// The measured hardware profile used for artifact identity: taken
    /// from the compiled incumbent pipeline (execution width, threadgroup
    /// maximum) exactly as at forge time.
    public static func profile() throws -> ForgeHardwareProfile {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Q4LoweringError.hardwareUnavailable("Metal unavailable")
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil) }
        catch { throw Q4LoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let fn = library.makeFunction(name: "g4_q4_matvec"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            throw Q4LoweringError.emissionFailed("incumbent pipeline")
        }
        return ForgeHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }
}

public enum Q4LoweringForge {

    /// Forges the operator lowering artifact: validate the op against the
    /// GGUF, derive legal schedules from shape + measured hardware, emit
    /// and compile every candidate, gate each against the independent CPU
    /// FP64 oracle under the EXISTING 1e-4 component contract (the
    /// incumbent kernel is gated and measured against the same oracle),
    /// and persist the numerically best accepted candidate. SELECTION IS
    /// NUMERICAL ONLY (smallest worst rel-to-FP64; ties prefer the
    /// baseline order) — no performance claim is made here; speed is
    /// decided only by the matched model-level measurement.
    public static func forge(opJSON: Data, ggufURL: URL?) throws
            -> (artifact: Q4LoweringArtifact, results: [Q4ComponentResult],
                incumbentRel: Double) {
        let op = try Q4OpIR.parse(opJSON)
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw Q4LoweringError.hardwareUnavailable("Metal unavailable")
        }
        var file: GGUFFile? = nil
        var ggufDigest = ""
        if op.tensor != nil {
            guard let ggufURL else {
                throw Q4LoweringError.invalidOp("op binds a tensor; --gguf is required")
            }
            let f = try GGUFReader.read(ggufURL)
            try op.validateAgainst(file: f)
            file = f
            ggufDigest = try GemmaForge.fileDigest(ggufURL)
        }
        let workingSet = Int(clamping: device.recommendedMaxWorkingSetSize)
        let blocksPerRow = op.inDim / Q4_0.valuesPerBlock

        // The incumbent reference kernel, compiled from the runner's own
        // embedded MSL — the same g4_q4_matvec the production path uses.
        // Its compiled pipeline provides the MEASURED hardware inputs the
        // deriver uses (execution width, threadgroup maximum).
        let incumbentLibrary: MTLLibrary
        do { incumbentLibrary = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                                       options: nil) }
        catch { throw Q4LoweringError.emissionFailed("incumbent MSL compile: \(error)") }
        guard let incumbentFn = incumbentLibrary.makeFunction(name: "g4_q4_matvec"),
              let incumbentPipeline = try? device.makeComputePipelineState(function: incumbentFn)
        else {
            throw Q4LoweringError.emissionFailed("incumbent pipeline")
        }
        let hardware = ForgeHardwareProfile(
            deviceName: device.name,
            maxTotalThreadsPerThreadgroup: incumbentPipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: incumbentPipeline.threadExecutionWidth,
            recommendedMaxWorkingSetSize: workingSet)

        let rows = op.componentRows
        let blocks = try Q4ComponentInputs.weightBytes(op: op, file: file)
        let x = Q4ComponentInputs.activation(op: op)
        let ref64 = try Q4Oracle.matvecFP64(blocks: blocks[...], outDim: rows,
                                            inDim: op.inDim, x: x)

        func rel(_ y: [Float], _ ref: [Double]) throws -> (worst: Double, sumAbs: Double) {
            guard y.count == ref.count, y.allSatisfy({ $0.isFinite }) else {
                throw Q4LoweringError.componentGateFailed("output count/nonfinite")
            }
            var worst = 0.0
            var worstRow = 0
            for i in 0..<y.count {
                let r = abs(Double(y[i]) - ref[i]) / max(1.0, abs(ref[i]))
                if r > worst { worst = r; worstRow = i }
            }
            // Cancellation context at the worst row: sum |products| in FP64.
            var sumAbs = 0.0
            for b in 0..<blocksPerRow {
                let o = (worstRow * blocksPerRow + b) * Q4_0.bytesPerBlock
                let bits = UInt16(blocks[o]) | UInt16(blocks[o + 1]) << 8
                let d = Double(Float16(bitPattern: bits))
                for l in 0..<16 {
                    let q = blocks[o + 2 + l]
                    sumAbs += abs(d * (Double(q & 0xF) - 8.0) * Double(x[b * 32 + l]))
                    sumAbs += abs(d * (Double(q >> 4) - 8.0) * Double(x[b * 32 + 16 + l]))
                }
            }
            return (worst, sumAbs)
        }

        // Incumbent kernel vs the FP64 oracle (measured, not assumed).
        func runIncumbent() throws -> [Float] {
            guard let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                             options: .storageModeShared),
                  let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                             options: .storageModeShared),
                  let yb = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                  let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                throw Q4LoweringError.hardwareUnavailable("incumbent buffers")
            }
            var p = SIMD2<UInt32>(UInt32(op.inDim), UInt32(rows))
            var off = UInt64(0)
            enc.setComputePipelineState(incumbentPipeline)
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3)
            enc.setBuffer(yb, offset: 0, index: 4)
            let width = max(1, min(rows, incumbentPipeline.maxTotalThreadsPerThreadgroup))
            let groups = (rows + width - 1) / width
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw Q4LoweringError.componentGateFailed("incumbent command buffer")
            }
            return [Float](UnsafeBufferPointer(
                start: yb.contents().assumingMemoryBound(to: Float.self), count: rows))
        }
        let incumbentOut = try runIncumbent()
        let incumbentRel = try rel(incumbentOut, ref64).worst

        var results = [Q4ComponentResult]()
        var best: (schedule: Q4Schedule, source: String, rel: Double)?
        for schedule in Q4ScheduleDeriver.derive(op: op, hardware: hardware) {
            do {
                let kernel = try Q4EmittedKernel.compile(op: op, schedule: schedule,
                                                         device: device)
                let out = try kernel.matvec(blocks: blocks, outDim: rows,
                                            inDim: op.inDim, x: x)
                let (worstOracle, sumAbs) = try rel(out, ref64)
                var worstInc = 0.0
                for i in 0..<out.count {
                    worstInc = max(worstInc, abs(Double(out[i] - incumbentOut[i]))
                                   / max(1.0, abs(Double(incumbentOut[i]))))
                }
                // EXISTING component contract, unchanged: finite outputs
                // and rel-to-FP64 < 1e-4 (same tolerance the Q4/Q6 GPU
                // kernels are held to in the test suite).
                let accepted = worstOracle < 1e-4
                results.append(Q4ComponentResult(
                    schedule: schedule, accepted: accepted,
                    worstRelToOracle: worstOracle, worstRelToIncumbent: worstInc,
                    worstRowSumAbsProducts: sumAbs,
                    note: accepted
                        ? String(format: "rel64 %.3e relInc %.3e", worstOracle, worstInc)
                        : String(format: "outside 1e-4 FP64 contract (rel64 %.3e)", worstOracle)))
                if accepted, best == nil || worstOracle < best!.rel {
                    best = (schedule, kernel.source, worstOracle)
                }
            } catch {
                results.append(Q4ComponentResult(
                    schedule: schedule, accepted: false, worstRelToOracle: nil,
                    worstRelToIncumbent: nil, worstRowSumAbsProducts: nil,
                    note: "rejected: \(error)"))
            }
        }
        guard let winner = best else {
            throw Q4LoweringError.componentGateFailed(
                results.map { "\($0.schedule): \($0.note)" }.joined(separator: "; "))
        }
        let body = Q4LoweringArtifact(
            artifactSchema: Q4LoweringArtifact.schema,
            codegenVersion: Q4KernelEmitter.codegenVersion,
            opDigest: try op.digest(), op: op, schedule: winner.schedule,
            kernelName: winner.schedule.kernelName, emittedSource: winner.source,
            sourceSha256: SHA256.hex(SHA256.hash(Array(winner.source.utf8))),
            hardware: hardware, toolchain: Forge.toolchainIdentifier,
            ggufDigest: ggufDigest,
            executableDigest: try Gemma4Runner.executableDigest(),
            componentResults: results, incumbentWorstRelToOracle: incumbentRel,
            representativeRows: rows, modelGate: nil, artifactDigest: "")
        let digest = try Q4LoweringArtifact.bodyDigest(body)
        let artifact = Q4LoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeRows: body.representativeRows, modelGate: nil, artifactDigest: digest)
        return (artifact, results, incumbentRel)
    }

    // MARK: bounded model-level orchestration (per-candidate gates,
    // matched timing, policy selection, persisted execution)

    /// One bounded chat fixture: name + pre-tokenized prompt history.
    public struct Q4Fixture {
        public let name: String
        public let tokens: [Int]
        public init(name: String, tokens: [Int]) {
            self.name = name
            self.tokens = tokens
        }
    }

    /// Full bounded selection orchestration, all in ONE process (one
    /// executable digest — rebuilding the binary invalidates the artifact,
    /// so nothing is re-derived after a rebuild):
    ///
    /// 1. component phase (forge): derive, emit, compile, FP64-gate every
    ///    candidate; the incumbent kernel is measured against the same
    ///    oracle;
    /// 2. reference arm: incumbent kernels under the pass-7 selected
    ///    configuration (segment batching + VL RMSNorm) generate greedy
    ///    histories per fixture;
    /// 3. per component-accepted candidate: trial-install, teacher-forced
    ///    same-history replay on EVERY fixture, gating full logits at the
    ///    UNCHANGED strict 1e-4 relative formula + exact greedy argmax +
    ///    finiteness at every step;
    /// 4. matched timing (incumbent vs every fully gate-passing candidate,
    ///    same binary/model/prompt/context, sequential residency);
    /// 5. policy: a generated candidate is performance-selected only if it
    ///    beats the incumbent summed wall medians by more than
    ///    GemmaForge.noiseRelWall (the documented 2% threshold); otherwise
    ///    the incumbent is the recorded selection and the artifact states
    ///    performanceSelected = false (the gate-passing generated schedule
    ///    is still persisted for the record, but validation guarantees it
    ///    passed every fixture);
    /// 6. persisted-execution proof: the artifact is saved, RELOADED from
    ///    disk, identity-validated, installed through the production path,
    ///    and one gated fixture replay runs with the dedicated dispatch
    ///    counters proving the emitted kernel executed.
    public static func selectAndPersist(opJSON: Data, ggufURL: URL,
                                        runner: Gemma4Runner,
                                        fixtures: [Q4Fixture],
                                        genTokens: Int, timedTokens: Int,
                                        isEndOfGeneration: (Int) -> Bool,
                                        saveTo url: URL) throws
            -> (artifact: Q4LoweringArtifact, log: [String]) {
        guard genTokens >= 2, genTokens <= 64, timedTokens >= 1,
              timedTokens < genTokens, fixtures.count >= 1, fixtures.count <= 4 else {
            throw Q4LoweringError.invalidSchedule(
                "orchestration bounds: genTokens \(genTokens), timedTokens \(timedTokens), fixtures \(fixtures.count)")
        }
        let names = Set(fixtures.map(\.name))
        guard names.count == fixtures.count else {
            throw Q4LoweringError.invalidSchedule("fixture names must be unique")
        }
        for f in fixtures {
            guard !f.tokens.isEmpty, f.tokens.count + genTokens <= 4096 else {
                throw Q4LoweringError.invalidSchedule("fixture \(f.name) exceeds context")
            }
        }
        var log = [String]()
        // The incumbent reference configuration (pass-7 selection):
        // segment batching + vectorized-load RMSNorm. IMPORTANT: it is
        // re-applied AFTER every reset — runner.reset() ->
        // discardOpenSegment() clears autoSegment, so applying the axis
        // only once up front would silently disable batching for every
        // arm (measured by root as per-kernel submissions).
        func resetIncumbentConfig() throws {
            runner.reset()
            runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
            // Fail closed: if batching ever fails to re-engage, the
            // matched arms would silently measure the per-kernel
            // submission mode (the reset-order bug root measured).
            guard runner.segmentBatchingActive else {
                throw Q4LoweringError.componentGateFailed(
                    "segment batching failed to re-engage after reset")
            }
        }
        let op = try Q4OpIR.parse(opJSON)
        guard let boundTensor = op.tensor else {
            throw Q4LoweringError.invalidOp("orchestration requires a bound tensor")
        }
        try op.validateAgainst(file: runner.ggufFile)
        let ggufDigest = try GemmaForge.fileDigest(ggufURL)

        // ---- 1. component phase ----
        let (componentArtifact, componentResults, incumbentRel) = try forge(
            opJSON: opJSON, ggufURL: ggufURL)
        log.append(String(format: "component: incumbent rel-to-FP64 %.3e; %d/%d candidates accepted",
                          incumbentRel,
                          componentResults.filter(\.accepted).count, componentResults.count))

        // ---- 2. reference arm per fixture (incumbent, pass-7 config,
        // re-applied after each reset — see resetIncumbentConfig) ----
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

        /// Teacher-forced replay gate for one fixture under whatever kernel
        /// is currently installed. Returns (worstRel, failReason).
        func gateReplay(_ fixture: Q4Fixture, _ ref: RefArm)
                -> (worstRel: Double, fail: String?, generatedDispatches: Int) {
            let dispatchBefore = runner.q4GeneratedDispatchCount
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
                                runner.q4GeneratedDispatchCount - dispatchBefore)
                    }
                    let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                    guard argmax == ref.ids[s] else {
                        return (worst, "step \(s): argmax \(argmax) != reference \(ref.ids[s])",
                                runner.q4GeneratedDispatchCount - dispatchBefore)
                    }
                    for i in 0..<lg.count {
                        worst = max(worst, abs(Double(lg[i] - want[i]))
                                    / max(1.0, abs(Double(want[i]))))
                    }
                    if worst >= 1e-4 {
                        return (worst, String(format: "step %d outside 1e-4 (worst %.3e)",
                                              s, worst),
                                runner.q4GeneratedDispatchCount - dispatchBefore)
                    }
                    if s + 1 < ref.ids.count {
                        _ = try runner.forward(tokenIds: [ref.ids[s]])
                    }
                }
                return (worst, nil, runner.q4GeneratedDispatchCount - dispatchBefore)
            } catch {
                return (0, "replay threw: \(error)",
                        runner.q4GeneratedDispatchCount - dispatchBefore)
            }
        }

        // ---- 3. per-candidate model gates ----
        let device = MTLCreateSystemDefaultDevice()
        var modelResults = [Q4ModelCandidateResult]()
        var passingKernels = [Q4Schedule: Q4EmittedKernel]()
        for result in componentResults where result.accepted {
            guard let device else { break }
            let kernel: Q4EmittedKernel
            do {
                kernel = try Q4EmittedKernel.compile(op: op, schedule: result.schedule,
                                                     device: device)
            } catch {
                modelResults.append(Q4ModelCandidateResult(
                    schedule: result.schedule, fixturesPassed: 0,
                    fixturesFailed: fixtures.count, worstRel: nil,
                    gateFailReason: "compile: \(error)",
                    wallMedPerFixture: nil, gpuMedPerFixture: nil))
                continue
            }
            do { try runner.installQ4TrialKernel(op: op, kernel: kernel) }
            catch {
                modelResults.append(Q4ModelCandidateResult(
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
            runner.uninstallQ4GeneratedKernel()
            let fullyPassed = failed == 0 && totalGenerated > 0
            if fullyPassed { passingKernels[result.schedule] = kernel }
            modelResults.append(Q4ModelCandidateResult(
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

        // Fail closed when NO candidate passed every fixture gate: an
        // artifact whose schedule failed a gate must never be persisted
        // as installable (production install would reject it anyway).
        guard modelResults.contains(where: { $0.fixturesFailed == 0 && $0.worstRel != nil })
        else {
            throw Q4LoweringError.componentGateFailed(
                "no generated candidate passed all fixture gates: "
                + modelResults.map { "\($0.schedule.kernelName): \($0.gateFailReason ?? "?")" }
                    .joined(separator: "; "))
        }

        // ---- 4. matched timing: 3 PAIRED repetitions per fully passing
        // candidate, alternating order (incumbent-first / candidate-first
        // / incumbent-first) so drift affects both arms equally. Raw
        // per-arm samples and the actual history window are kept in the
        // record. THROWS on any forward failure — a failed run must never
        // be timed (a swallowed error would measure a fast no-op). The
        // timed count adapts to EOS-truncated histories:
        /// effTimed = min(timedTokens, steps - 1), floor 1.
        func timedArm() throws -> [Q4TimingArmFixture] {
            var arms = [Q4TimingArmFixture]()
            for fixture in fixtures {
                let ref = refArms[fixture.name]!
                let steps = ref.ids.count
                guard steps >= 2 else {
                    throw Q4LoweringError.componentGateFailed(
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
                arms.append(Q4TimingArmFixture(walls: w, gpus: g,
                                               startIdx: idx, timedCount: effTimed))
            }
            return arms
        }
        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2 : s[s.count/2]
        }
        /// Per-fixture median walls of one recorded arm.
        func armFixtureMedians(_ arm: [Q4TimingArmFixture]) -> [Double] {
            arm.map { median($0.walls) }
        }
        let repetitionCount = 3
        // Each candidate keeps its OWN paired incumbent metrics — the
        // selection compares candMed against THIS candidate's paired
        // incMed (normalized ratio, deterministic tie-break), never
        // against another candidate's incumbent arm.
        struct PairedTiming {
            var schedule: Q4Schedule
            var candMed: Double
            var incMed: Double
            var reps: [Q4TimingRepetition]
        }
        var paired = [PairedTiming]()
        var bestTiming: PairedTiming? = nil
        for i in modelResults.indices where modelResults[i].fixturesFailed == 0 {
            guard let schedule = passingKernels.keys.first(where: {
                $0 == modelResults[i].schedule }) else { continue }
            var reps = [Q4TimingRepetition]()
            for rep in 0..<repetitionCount {
                let candidateFirst = rep % 2 == 1
                // Candidate arm (installed only for the arm's duration).
                func candidateArm() throws -> [Q4TimingArmFixture] {
                    try runner.installQ4TrialKernel(op: op, kernel: passingKernels[schedule]!)
                    defer { runner.uninstallQ4GeneratedKernel() }
                    return try timedArm()
                }
                // Incumbent arm (generated kernel explicitly removed).
                func incumbentArm() throws -> [Q4TimingArmFixture] {
                    runner.uninstallQ4GeneratedKernel()
                    return try timedArm()
                }
                let (cand, inc): ([Q4TimingArmFixture], [Q4TimingArmFixture])
                if candidateFirst {
                    cand = try candidateArm()
                    inc = try incumbentArm()
                } else {
                    inc = try incumbentArm()
                    cand = try candidateArm()
                }
                reps.append(Q4TimingRepetition(candidateFirst: candidateFirst,
                                               candidate: cand, incumbent: inc))
            }
            modelResults[i].timingReps = reps
            let candSums = reps.map { armFixtureMedians($0.candidate).reduce(0, +) }
            let incSums = reps.map { armFixtureMedians($0.incumbent).reduce(0, +) }
            let candMed = median(candSums)
            let incMed = median(incSums)
            paired.append(PairedTiming(schedule: schedule, candMed: candMed,
                                       incMed: incMed, reps: reps))
            // Reported per-fixture medians: median across repetitions.
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
            // Selection metric: PAIRED normalized ratio candMed/incMed
            // (deterministic tie-break by kernel name).
            if bestTiming == nil
                || candMed / incMed < bestTiming!.candMed / bestTiming!.incMed
                || (candMed / incMed == bestTiming!.candMed / bestTiming!.incMed
                    && schedule.kernelName < bestTiming!.schedule.kernelName) {
                bestTiming = paired.last
            }
        }

        // Top-level incumbent metrics come from the SELECTED candidate's
        // own paired repetitions (its incumbent arms), never from an
        // aggregation over unmatched arms.
        let incWalls: [Double]
        let incGpus: [Double]
        if let best = bestTiming {
            incWalls = (0..<fixtures.count).map { f in
                median(best.reps.map { armFixtureMedians($0.incumbent)[f] })
            }
            incGpus = (0..<fixtures.count).map { f in
                median(best.reps.map { median($0.incumbent[f].gpus) })
            }
        } else {
            incWalls = []
            incGpus = []
        }

        // ---- 5. policy ----
        let performanceSelected: Bool
        let selectedSchedule: Q4Schedule
        if let best = bestTiming,
           best.candMed < best.incMed * (1 - GemmaForge.noiseRelWall) {
            performanceSelected = true
            selectedSchedule = best.schedule
        } else if let best = bestTiming {
            // Gate-passing but no win beyond the noise threshold: the
            // incumbent stays the production selection; the artifact
            // records the passing schedule for the record only.
            performanceSelected = false
            selectedSchedule = best.schedule
        } else {
            // Unreachable: the no-passing guard above threw already.
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
        let selectedSource = try Q4KernelEmitter.emit(op: op, schedule: selectedSchedule)
        let fixtureRecords = fixtures.map { fixture -> Q4FixtureRecord in
            let ref = refArms[fixture.name]!
            // Full teacher-forced history: prompt + forced reference ids.
            var history = fixture.tokens
            history.append(contentsOf: ref.ids.dropLast())
            return Q4FixtureRecord(
                name: fixture.name, promptTokenCount: fixture.tokens.count,
                gatedSteps: ref.ids.count,
                historyDigest: GemmaForge.workloadDigest(history))
        }
        let gateRecord = Q4ModelGateRecord(
            genTokens: genTokens, timedTokens: timedTokens, fixtures: fixtureRecords,
            results: modelResults, incumbentWallPerFixture: incWalls,
            incumbentGpuPerFixture: incGpus,
            performanceSelected: performanceSelected, policyNote: policyNote)
        let body = Q4LoweringArtifact(
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
            representativeRows: componentArtifact.representativeRows,
            modelGate: gateRecord, artifactDigest: "")
        let digest = try Q4LoweringArtifact.bodyDigest(body)
        let artifact = Q4LoweringArtifact(
            artifactSchema: body.artifactSchema, codegenVersion: body.codegenVersion,
            opDigest: body.opDigest, op: body.op, schedule: body.schedule,
            kernelName: body.kernelName, emittedSource: body.emittedSource,
            sourceSha256: body.sourceSha256, hardware: body.hardware,
            toolchain: body.toolchain, ggufDigest: body.ggufDigest,
            executableDigest: body.executableDigest,
            componentResults: body.componentResults,
            incumbentWorstRelToOracle: body.incumbentWorstRelToOracle,
            representativeRows: body.representativeRows,
            modelGate: gateRecord, artifactDigest: digest)
        try artifact.save(to: url)

        // ---- 6. persisted-execution proof: reload from disk, validate,
        // install through the PRODUCTION path, one gated fixture replay ----
        let reloaded = try Q4LoweringArtifact.load(from: url)
        try reloaded.validate(op: op, hardware: componentArtifact.hardware,
                              ggufDigest: ggufDigest,
                              executableDigest: try Gemma4Runner.executableDigest())
        try runner.installQ4GeneratedKernel(artifact: reloaded, ggufDigest: ggufDigest)
        let first = fixtures[0]
        let (worst, fail, dispatches) = gateReplay(first, refArms[first.name]!)
        runner.uninstallQ4GeneratedKernel()
        guard fail == nil, dispatches > 0 else {
            throw Q4LoweringError.componentGateFailed(
                "persisted execution failed: \(fail ?? "no generated dispatches")")
        }
        log.append(String(format: "persisted execution: reloaded+validated+installed from disk; %@ replay worst rel %.3e; generated dispatches %d (bound tensor %@)",
                          first.name, worst, dispatches, boundTensor))
        return (artifact, log)
    }
}
