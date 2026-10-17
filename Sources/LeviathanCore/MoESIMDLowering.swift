// Phase 49 — cooperative SIMD routed-MoE lowering: CORRECTNESS PROTOTYPE.
//
// A separately versioned kernel family for the validated routed MoE region
// (MoERegionIR): a fused gate/up dot + GELU(tanh)*up kernel and a
// cooperative down kernel. L lanes of one SIMD group compute one logical
// row; each lane owns WHOLE 32-value Q4_0 blocks (lane-strided:
// b = lane, lane + L, ...), the slot's input vector is staged once per
// threadgroup, and the L partials combine through a fixed
// simd_shuffle_xor butterfly. The mapping is distinct from the v1
// one-thread-per-row emission (MoEKernelEmitter, untouched), from the Q4
// simdLanePlain mapping (one lane per quantization component), and from
// the measured-rejected g4_q4_matvec_vl variant (never used here).
//
// Numerical class "simd-reassociated-oracle": the per-row dot is
// reassociated across lanes, so NO bit identity with v1 is claimed. Every
// output is gated against an independent full CPU FP64 oracle at the
// unchanged normalized 1e-4 contract. This file produces component
// evidence and a NON-INSTALLABLE report only: no runner dispatch, no
// artifact class, no installer, no flag. Schedules derive from the
// validated region shape and actual hardware caps; this is not an
// arbitrary Python compiler. No model hashing, network or process calls.

import Foundation
import CryptoKit
import Metal

// MARK: - Errors and checked arithmetic

public enum MoESIMDError: Error, Equatable, CustomStringConvertible {
    case invalidShape(String)
    case illegalSchedule(String)
    case emissionFailed(String)
    case compileFailed(String)
    case componentFailed(String)
    case oracleFailed(String)
    case hardwareUnavailable(String)
    case invalidReport(String)

    public var description: String {
        switch self {
        case .invalidShape(let m): return "moe simd shape invalid: \(m)"
        case .illegalSchedule(let m): return "moe simd schedule illegal: \(m)"
        case .emissionFailed(let m): return "moe simd emission failed: \(m)"
        case .compileFailed(let m): return "moe simd compile failed: \(m)"
        case .componentFailed(let m): return "moe simd component failed: \(m)"
        case .oracleFailed(let m): return "moe simd oracle failed: \(m)"
        case .hardwareUnavailable(let m): return "moe simd hardware unavailable: \(m)"
        case .invalidReport(let m): return "moe simd report invalid: \(m)"
        }
    }
}

/// Checked integer arithmetic for extents, offsets and budgets.
enum MoESIMDChecked {
    static func mul(_ a: Int, _ b: Int, _ what: String) throws -> Int {
        guard a >= 0, b >= 0 else {
            throw MoESIMDError.invalidShape("\(what): negative operand")
        }
        let (r, overflow) = a.multipliedReportingOverflow(by: b)
        guard !overflow else { throw MoESIMDError.invalidShape("\(what): overflow") }
        return r
    }

    static func add(_ a: Int, _ b: Int, _ what: String) throws -> Int {
        guard a >= 0, b >= 0 else {
            throw MoESIMDError.invalidShape("\(what): negative operand")
        }
        let (r, overflow) = a.addingReportingOverflow(b)
        guard !overflow else { throw MoESIMDError.invalidShape("\(what): overflow") }
        return r
    }
}

// MARK: - Shape facts

/// Shape facts of the routed region. Production derivation goes through
/// init(region:) (full MoERegionIR revalidation, including the graph
/// re-derivation); the direct initializer exists for small synthetic
/// component tests and never produces a report.
public struct MoESIMDShapeFacts: Codable, Equatable, Hashable, Sendable {
    public let hidden: Int
    public let inter: Int
    public let topK: Int

    public init(hidden: Int, inter: Int, topK: Int) throws {
        guard hidden > 0, inter > 0, hidden % 32 == 0, inter % 32 == 0,
              hidden <= MoERegionIR.maxDim, inter <= MoERegionIR.maxDim,
              topK > 0, topK <= MoERegionIR.maxTopK else {
            throw MoESIMDError.invalidShape(
                "hidden \(hidden), inter \(inter), topK \(topK) outside the supported region bounds")
        }
        self.hidden = hidden
        self.inter = inter
        self.topK = topK
    }

    public init(region: MoERegionIR) throws {
        try region.validate()
        guard region.gateUpOut == 2 * region.expertIntermediate else {
            throw MoESIMDError.invalidShape("gateUpOut \(region.gateUpOut) != 2 * inter")
        }
        try self.init(hidden: region.hidden, inter: region.expertIntermediate,
                      topK: region.topK)
    }

    /// Fused gate|up rows per expert slice (gate i, then up inter + i).
    public var gateUpRows: Int { 2 * inter }
    public var guBlocksPerRow: Int { hidden / 32 }
    public var dnBlocksPerRow: Int { inter / 32 }

    public func guStrideBytes() throws -> Int {
        let rowBytes = try MoESIMDChecked.mul(guBlocksPerRow, 18, "GU row bytes")
        return try MoESIMDChecked.mul(gateUpRows, rowBytes, "GU slice stride")
    }

    public func dnStrideBytes() throws -> Int {
        let rowBytes = try MoESIMDChecked.mul(dnBlocksPerRow, 18, "DN row bytes")
        return try MoESIMDChecked.mul(hidden, rowBytes, "DN slice stride")
    }
}

// MARK: - Schedule

public enum MoESIMDAccumulation: String, Codable, CaseIterable, Sendable {
    /// Plain float32 lane partials and butterfly combine; default compile
    /// options (nil).
    case plain
    /// Neumaier-compensated float32 partials and combine, compiled with
    /// fastMathEnabled = false so the compensation is not algebraically
    /// cancelled.
    case neumaierPrecise

    public var compileOptionsLabel: String {
        self == .plain ? "default" : "fastMathEnabled=false"
    }

    var nameTag: String { self == .plain ? "plain" : "neum" }
}

public enum MoESIMDFamily: String, Codable, Sendable {
    /// Fused dot of the row pair (gate i, up inter + i) followed by the
    /// literal incumbent g4_gelu_up_value, writing the activation directly.
    case fusedGateUpAct
    /// Cooperative down projection over the slot's staged activation.
    case down
}

public struct MoESIMDSubschedule: Codable, Equatable, Hashable, Sendable {
    public let family: MoESIMDFamily
    /// L: lanes cooperating on one logical row (8, 16 or 32).
    public let lanes: Int
    /// RT: logical rows per threadgroup, derived as T / L, T in {128, 256}.
    public let rowsPerGroup: Int
    public let accumulation: MoESIMDAccumulation

    public init(family: MoESIMDFamily, lanes: Int, rowsPerGroup: Int,
                accumulation: MoESIMDAccumulation) {
        self.family = family
        self.lanes = lanes
        self.rowsPerGroup = rowsPerGroup
        self.accumulation = accumulation
    }

    public var kernelName: String {
        let base = family == .fusedGateUpAct ? "moesimd_gu" : "moesimd_dn"
        return base + "_L\(lanes)_RT\(rowsPerGroup)_" + accumulation.nameTag
    }
}

public struct MoESIMDSchedule: Codable, Equatable, Hashable, Sendable {
    /// E: experts per dispatch chunk. Dispatch granularity only — the
    /// per-row arithmetic does not depend on it, which the batch-invariance
    /// check verifies bitwise.
    public let batch: Int
    public let gu: MoESIMDSubschedule
    public let dn: MoESIMDSubschedule

    public init(batch: Int, gu: MoESIMDSubschedule, dn: MoESIMDSubschedule) {
        self.batch = batch
        self.gu = gu
        self.dn = dn
    }

    public var label: String { "E\(batch)|" + batchFreeLabel }
    public var batchFreeLabel: String { gu.kernelName + "|" + dn.kernelName }
}

// MARK: - Hardware profile

public struct MoESIMDHardwareProfile: Codable, Equatable, Sendable {
    public let deviceName: String
    public let threadExecutionWidth: Int
    public let maxTotalThreadsPerThreadgroup: Int
    public let maxThreadgroupMemoryLength: Int
    public let staticThreadgroupMemoryLength: Int
    public let maxBufferLength: Int
    public let recommendedMaxWorkingSetSize: Int

    public init(deviceName: String, threadExecutionWidth: Int,
                maxTotalThreadsPerThreadgroup: Int, maxThreadgroupMemoryLength: Int,
                staticThreadgroupMemoryLength: Int, maxBufferLength: Int,
                recommendedMaxWorkingSetSize: Int) {
        self.deviceName = deviceName
        self.threadExecutionWidth = threadExecutionWidth
        self.maxTotalThreadsPerThreadgroup = maxTotalThreadsPerThreadgroup
        self.maxThreadgroupMemoryLength = maxThreadgroupMemoryLength
        self.staticThreadgroupMemoryLength = staticThreadgroupMemoryLength
        self.maxBufferLength = maxBufferLength
        self.recommendedMaxWorkingSetSize = recommendedMaxWorkingSetSize
    }

    /// The ACTUAL limits of one compiled pipeline on its device.
    public static func actual(pipeline: MTLComputePipelineState,
                              device: MTLDevice) -> MoESIMDHardwareProfile {
        MoESIMDHardwareProfile(
            deviceName: device.name,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
            maxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            staticThreadgroupMemoryLength: pipeline.staticThreadgroupMemoryLength,
            maxBufferLength: device.maxBufferLength,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
    }

    /// Pre-derivation profile measured from a compiled SIMD probe pipeline.
    /// Every candidate is re-checked against its OWN compiled pipeline
    /// (MoESIMDCompiledKernel.compile), which is authoritative.
    public static func probe(device: MTLDevice) throws -> MoESIMDHardwareProfile {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void moesimd_hardware_probe(device float* y [[buffer(0)]],
                                           uint tid [[thread_position_in_grid]]) {
            y[tid] = simd_shuffle_xor(float(tid), ushort(1));
        }
        """
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) }
        catch { throw MoESIMDError.hardwareUnavailable("probe library: \(error)") }
        guard let fn = library.makeFunction(name: "moesimd_hardware_probe") else {
            throw MoESIMDError.hardwareUnavailable("probe function missing")
        }
        let pipeline: MTLComputePipelineState
        do { pipeline = try device.makeComputePipelineState(function: fn) }
        catch { throw MoESIMDError.hardwareUnavailable("probe pipeline: \(error)") }
        return actual(pipeline: pipeline, device: device)
    }
}

// MARK: - Legality (one pure decision function)

public enum MoESIMDLegality {
    public static let laneOptions = [8, 16, 32]
    public static let threadOptions = [128, 256]
    public static let batchOptions = [4, 8]
    public static let requiredExecutionWidth = 32

    /// Logical rows per expert: inter row PAIRS for the fused kernel,
    /// hidden rows for the down kernel.
    public static func logicalRows(_ family: MoESIMDFamily, _ facts: MoESIMDShapeFacts) -> Int {
        family == .fusedGateUpAct ? facts.inter : facts.hidden
    }

    public static func inputDim(_ family: MoESIMDFamily, _ facts: MoESIMDShapeFacts) -> Int {
        family == .fusedGateUpAct ? facts.hidden : facts.inter
    }

    public static func blocksPerRow(_ family: MoESIMDFamily, _ facts: MoESIMDShapeFacts) -> Int {
        inputDim(family, facts) / 32
    }

    /// Staged input bytes: hidden * 4 (fused GU) or inter * 4 (down).
    public static func stageBytes(_ family: MoESIMDFamily, _ facts: MoESIMDShapeFacts) throws -> Int {
        try MoESIMDChecked.mul(inputDim(family, facts), 4, "staged input bytes")
    }

    /// nil = legal; otherwise the rejection reason. Used for derivation
    /// (measured profile), emission (shape-only profile) and the compiled
    /// pipeline guard (actual pipeline profile) — one decision.
    public static func reason(_ s: MoESIMDSubschedule, facts: MoESIMDShapeFacts,
                              hardware: MoESIMDHardwareProfile) -> String? {
        let w = hardware.threadExecutionWidth
        guard w == requiredExecutionWidth else {
            return "execution width \(w) != \(requiredExecutionWidth) required by the lane mapping"
        }
        guard laneOptions.contains(s.lanes), w % s.lanes == 0 else {
            return "lanes per row \(s.lanes) not in \(laneOptions) or not dividing execution width \(w)"
        }
        guard s.rowsPerGroup > 0 else { return "rowsPerGroup must be positive" }
        let (threads, threadsOverflow) = s.lanes.multipliedReportingOverflow(by: s.rowsPerGroup)
        guard !threadsOverflow, threadOptions.contains(threads) else {
            return "threads per group L*RT must be one of \(threadOptions) (RT is derived as T/L)"
        }
        guard threads <= hardware.maxTotalThreadsPerThreadgroup else {
            return "threads per group \(threads) exceed the pipeline maximum \(hardware.maxTotalThreadsPerThreadgroup)"
        }
        guard threads % w == 0 else {
            return "threads per group \(threads) not a multiple of execution width \(w)"
        }
        let rows = logicalRows(s.family, facts)
        guard rows % s.rowsPerGroup == 0 else {
            return "rowsPerGroup \(s.rowsPerGroup) does not divide \(rows) logical rows per expert (a threadgroup would cross slots)"
        }
        let blocks = blocksPerRow(s.family, facts)
        guard blocks * 2 >= s.lanes else {
            return "only \(blocks) Q4 blocks per row: fewer than half of the \(s.lanes) lanes would own a block"
        }
        guard let bytes = try? stageBytes(s.family, facts), bytes % 16 == 0 else {
            return "staged input bytes invalid or not a 16-byte multiple"
        }
        guard hardware.staticThreadgroupMemoryLength >= 0,
              hardware.maxThreadgroupMemoryLength > 0 else {
            return "threadgroup memory caps invalid"
        }
        let (tg, tgOverflow) = bytes.addingReportingOverflow(hardware.staticThreadgroupMemoryLength)
        guard !tgOverflow, tg <= hardware.maxThreadgroupMemoryLength else {
            return "threadgroup memory \(bytes) staged + \(hardware.staticThreadgroupMemoryLength) static exceeds the device cap \(hardware.maxThreadgroupMemoryLength)"
        }
        let (groups, groupsOverflow) = (rows / s.rowsPerGroup).multipliedReportingOverflow(by: facts.topK)
        guard !groupsOverflow, groups <= Int(UInt32.max) else {
            return "threadgroup count exceeds the uint grid range"
        }
        return nil
    }

    /// E legality. topK tails are legal: chunks carry ceil(topK / E)
    /// dispatches with the runtime chunkSlots <= E (host-checked).
    public static func batchReason(_ batch: Int, facts: MoESIMDShapeFacts) -> String? {
        guard batchOptions.contains(batch) else {
            return "batch E \(batch) not in \(batchOptions)"
        }
        guard facts.topK > 0 else { return "topK must be positive" }
        return nil
    }
}

// MARK: - Schedule derivation

public struct MoESIMDRejection: Codable, Equatable, Sendable {
    public let candidate: String
    public let reason: String
}

public struct MoESIMDDerivation: Codable, Equatable, Sendable {
    public let facts: MoESIMDShapeFacts
    public let hardware: MoESIMDHardwareProfile
    public let guLegal: [MoESIMDSubschedule]
    public let dnLegal: [MoESIMDSubschedule]
    public let schedules: [MoESIMDSchedule]
    public let rejections: [MoESIMDRejection]
}

public enum MoESIMDScheduleDeriver {
    public static let maxCatalog = 48

    /// Independent subschedule derivation for one family: every (L, T,
    /// accumulation) candidate with RT = T / L is checked; rejections are
    /// recorded with their reasons.
    public static func subschedules(family: MoESIMDFamily, facts: MoESIMDShapeFacts,
                                    hardware: MoESIMDHardwareProfile)
            -> (legal: [MoESIMDSubschedule], rejected: [MoESIMDRejection]) {
        var legal = [MoESIMDSubschedule]()
        var rejected = [MoESIMDRejection]()
        for accumulation in MoESIMDAccumulation.allCases {
            for lanes in MoESIMDLegality.laneOptions {
                for threads in MoESIMDLegality.threadOptions {
                    let s = MoESIMDSubschedule(family: family, lanes: lanes,
                                               rowsPerGroup: threads / lanes,
                                               accumulation: accumulation)
                    if let why = MoESIMDLegality.reason(s, facts: facts, hardware: hardware) {
                        rejected.append(MoESIMDRejection(candidate: s.kernelName, reason: why))
                    } else {
                        legal.append(s)
                    }
                }
            }
        }
        return (legal, rejected)
    }

    /// Production derivation from the VALIDATED region.
    public static func derive(region: MoERegionIR,
                              hardware: MoESIMDHardwareProfile) throws -> MoESIMDDerivation {
        derive(facts: try MoESIMDShapeFacts(region: region), hardware: hardware)
    }

    /// Bounded composition: GU and DN subschedules with the SAME lanes and
    /// accumulation, for every legal E — at most maxCatalog entries (the
    /// real shape yields exactly 48). Unpaired subschedules are recorded.
    public static func derive(facts: MoESIMDShapeFacts,
                              hardware: MoESIMDHardwareProfile) -> MoESIMDDerivation {
        let gu = subschedules(family: .fusedGateUpAct, facts: facts, hardware: hardware)
        let dn = subschedules(family: .down, facts: facts, hardware: hardware)
        var rejections = gu.rejected + dn.rejected
        var schedules = [MoESIMDSchedule]()
        for batch in MoESIMDLegality.batchOptions {
            if let why = MoESIMDLegality.batchReason(batch, facts: facts) {
                rejections.append(MoESIMDRejection(candidate: "E\(batch)", reason: why))
                continue
            }
            for g in gu.legal {
                for d in dn.legal where d.lanes == g.lanes && d.accumulation == g.accumulation {
                    let s = MoESIMDSchedule(batch: batch, gu: g, dn: d)
                    if schedules.count < maxCatalog {
                        schedules.append(s)
                    } else {
                        rejections.append(MoESIMDRejection(
                            candidate: s.label, reason: "catalog bound \(maxCatalog) reached"))
                    }
                }
            }
        }
        for g in gu.legal where !dn.legal.contains(where: {
            $0.lanes == g.lanes && $0.accumulation == g.accumulation }) {
            rejections.append(MoESIMDRejection(
                candidate: g.kernelName,
                reason: "no legal down subschedule with the same lanes/accumulation (bounded same-L composition)"))
        }
        for d in dn.legal where !gu.legal.contains(where: {
            $0.lanes == d.lanes && $0.accumulation == d.accumulation }) {
            rejections.append(MoESIMDRejection(
                candidate: d.kernelName,
                reason: "no legal fused GU subschedule with the same lanes/accumulation (bounded same-L composition)"))
        }
        return MoESIMDDerivation(facts: facts, hardware: hardware, guLegal: gu.legal,
                                 dnLegal: dn.legal, schedules: schedules,
                                 rejections: rejections)
    }
}

// MARK: - Metal source emission

public enum MoESIMDKernelEmitter {
    /// Bumped whenever emission rules change; part of the report identity.
    public static let codegenVersion = "moesimd-v1"

    /// Shape-only legality profile for emission; the measured hardware and
    /// each compiled pipeline are re-checked separately.
    static func shapeOnlyHardware() -> MoESIMDHardwareProfile {
        MoESIMDHardwareProfile(deviceName: "emit", threadExecutionWidth: 32,
                               maxTotalThreadsPerThreadgroup: 1024,
                               maxThreadgroupMemoryLength: 1 << 40,
                               staticThreadgroupMemoryLength: 0,
                               maxBufferLength: 1 << 40,
                               recommendedMaxWorkingSetSize: 1 << 40)
    }

    /// Emits one complete standalone MSL source for one subschedule. The
    /// fused GU source embeds the LITERAL incumbent activation helpers.
    public static func emit(_ s: MoESIMDSubschedule, facts: MoESIMDShapeFacts) throws -> String {
        if let why = MoESIMDLegality.reason(s, facts: facts, hardware: shapeOnlyHardware()) {
            throw MoESIMDError.emissionFailed("\(s.kernelName): \(why)")
        }
        let header = """
        #include <metal_stdlib>
        using namespace metal;

        // \(s.kernelName) — EMITTED by the cooperative SIMD routed-MoE
        // lowering (\(codegenVersion)); numerical class simd-reassociated-oracle.
        // family=\(s.family.rawValue) L=\(s.lanes) RT=\(s.rowsPerGroup)
        // accumulation=\(s.accumulation.rawValue) compileOptions=\(s.accumulation.compileOptionsLabel)
        // region shape hidden=\(facts.hidden) inter=\(facts.inter) topK=\(facts.topK).
        // Lane-strided whole Q4_0 blocks, staged input, fixed xor butterfly.

        """
        let accumulator: String
        switch s.accumulation {
        case .plain:
            accumulator = """
            static inline void moesimd_acc(thread float& s, thread float& c, float t) {
                (void)c;
                s += t;
            }

            """
        case .neumaierPrecise:
            accumulator = """
            static inline void moesimd_acc(thread float& s, thread float& c, float t) {
                float y = s + t;
                if (fabs(s) >= fabs(t)) { c += (s - y) + t; } else { c += (t - y) + s; }
                s = y;
            }

            """
        }
        switch s.family {
        case .fusedGateUpAct:
            let helpers: String
            do { helpers = try MoEKernelEmitter.literalActivationHelpers() }
            catch { throw MoESIMDError.emissionFailed("activation helpers: \(error)") }
            return header + helpers + "\n\n" + accumulator + fusedKernel(s)
        case .down:
            return header + accumulator + downKernel(s)
        }
    }

    static func fusedKernel(_ s: MoESIMDSubschedule) -> String {
        """
        kernel void \(s.kernelName)(
                constant const uint4& p [[buffer(0)]],
                constant const ulong* offs [[buffer(1)]],
                device const uchar* wbase [[buffer(2)]],
                device const float* x [[buffer(3)]],
                device float* act [[buffer(4)]],
                threadgroup float* xs [[threadgroup(0)]],
                uint tid [[thread_index_in_threadgroup]],
                uint wid [[threadgroup_position_in_grid]]) {
            // p = (hidden, inter row pairs per expert, chunkSlots, slotBase);
            // offs = per-ABSOLUTE-slot GU slice byte offsets (even, host-checked).
            constexpr uint L = \(s.lanes)u;
            constexpr uint RT = \(s.rowsPerGroup)u;
            constexpr uint TG = L * RT;
            uint inDim = p.x;
            uint rows = p.y;
            uint groupsPerSlot = rows / RT;
            uint slotLocal = wid / groupsPerSlot;
            // Uniform per threadgroup (depends on wid only): the whole group
            // leaves together, before any barrier or SIMD reduction.
            if (slotLocal >= p.z) { return; }
            uint slot = p.w + slotLocal;
            for (uint i = tid; i < inDim; i += TG) { xs[i] = x[i]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint lane = tid % L;
            uint row = (wid % groupsPerSlot) * RT + tid / L;
            uint blocksPerRow = inDim / 32u;
            ulong rowStride = (ulong)blocksPerRow * 18ul;
            device const uchar* gateRow = wbase + offs[slot] + (ulong)row * rowStride;
            device const uchar* upRow = wbase + offs[slot] + (ulong)(row + rows) * rowStride;
            float gs = 0.0f;
            float gc = 0.0f;
            float us = 0.0f;
            float uc = 0.0f;
            // Lane-strided whole blocks: lanes past the block tail keep zero
            // partials and still join the reduction.
            for (uint b = lane; b < blocksPerRow; b += L) {
                device const ushort* gq = (device const ushort*)(gateRow + (ulong)b * 18ul);
                device const ushort* uq = (device const ushort*)(upRow + (ulong)b * 18ul);
                float gd = float(as_type<half>(gq[0]));
                float ud = float(as_type<half>(uq[0]));
                threadgroup const float* xb = xs + b * 32u;
                for (uint k = 0u; k < 8u; k++) {
                    uint gw = uint(gq[1u + k]);
                    uint uw = uint(uq[1u + k]);
                    uint l0 = 2u * k;
                    float x0 = xb[l0];
                    float x1 = xb[l0 + 1u];
                    float x16 = xb[16u + l0];
                    float x17 = xb[17u + l0];
                    moesimd_acc(gs, gc, gd * (float(gw & 0x0Fu) - 8.0f) * x0);
                    moesimd_acc(gs, gc, gd * (float((gw >> 4u) & 0x0Fu) - 8.0f) * x16);
                    moesimd_acc(gs, gc, gd * (float((gw >> 8u) & 0x0Fu) - 8.0f) * x1);
                    moesimd_acc(gs, gc, gd * (float((gw >> 12u) & 0x0Fu) - 8.0f) * x17);
                    moesimd_acc(us, uc, ud * (float(uw & 0x0Fu) - 8.0f) * x0);
                    moesimd_acc(us, uc, ud * (float((uw >> 4u) & 0x0Fu) - 8.0f) * x16);
                    moesimd_acc(us, uc, ud * (float((uw >> 8u) & 0x0Fu) - 8.0f) * x1);
                    moesimd_acc(us, uc, ud * (float((uw >> 12u) & 0x0Fu) - 8.0f) * x17);
                }
            }
            // Fixed xor butterfly over the L lanes of this logical row: masks
            // below L never leave the L-aligned lane group (32 % L == 0).
            for (ushort m = ushort(L / 2u); m > 0; m >>= 1) {
                float gos = simd_shuffle_xor(gs, m);
                float goc = simd_shuffle_xor(gc, m);
                float uos = simd_shuffle_xor(us, m);
                float uoc = simd_shuffle_xor(uc, m);
                moesimd_acc(gs, gc, gos);
                gc += goc;
                moesimd_acc(us, uc, uos);
                uc += uoc;
            }
            if (lane == 0u) {
                act[(ulong)slot * rows + row] = g4_gelu_up_value(gs + gc, us + uc);
            }
        }

        """
    }

    static func downKernel(_ s: MoESIMDSubschedule) -> String {
        """
        kernel void \(s.kernelName)(
                constant const uint4& p [[buffer(0)]],
                constant const ulong* offs [[buffer(1)]],
                device const uchar* wbase [[buffer(2)]],
                device const float* act [[buffer(3)]],
                device float* y [[buffer(4)]],
                threadgroup float* xs [[threadgroup(0)]],
                uint tid [[thread_index_in_threadgroup]],
                uint wid [[threadgroup_position_in_grid]]) {
            // p = (inter, hidden rows per expert, chunkSlots, slotBase);
            // act and y are slot-major; offs are even (host-checked).
            constexpr uint L = \(s.lanes)u;
            constexpr uint RT = \(s.rowsPerGroup)u;
            constexpr uint TG = L * RT;
            uint inDim = p.x;
            uint rows = p.y;
            uint groupsPerSlot = rows / RT;
            uint slotLocal = wid / groupsPerSlot;
            // Uniform per threadgroup: the whole group leaves together.
            if (slotLocal >= p.z) { return; }
            uint slot = p.w + slotLocal;
            for (uint i = tid; i < inDim; i += TG) { xs[i] = act[(ulong)slot * inDim + i]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint lane = tid % L;
            uint row = (wid % groupsPerSlot) * RT + tid / L;
            uint blocksPerRow = inDim / 32u;
            ulong rowStride = (ulong)blocksPerRow * 18ul;
            device const uchar* rowBytes = wbase + offs[slot] + (ulong)row * rowStride;
            float s = 0.0f;
            float c = 0.0f;
            for (uint b = lane; b < blocksPerRow; b += L) {
                device const ushort* q = (device const ushort*)(rowBytes + (ulong)b * 18ul);
                float d = float(as_type<half>(q[0]));
                threadgroup const float* xb = xs + b * 32u;
                for (uint k = 0u; k < 8u; k++) {
                    uint w = uint(q[1u + k]);
                    uint l0 = 2u * k;
                    moesimd_acc(s, c, d * (float(w & 0x0Fu) - 8.0f) * xb[l0]);
                    moesimd_acc(s, c, d * (float((w >> 4u) & 0x0Fu) - 8.0f) * xb[16u + l0]);
                    moesimd_acc(s, c, d * (float((w >> 8u) & 0x0Fu) - 8.0f) * xb[l0 + 1u]);
                    moesimd_acc(s, c, d * (float((w >> 12u) & 0x0Fu) - 8.0f) * xb[17u + l0]);
                }
            }
            for (ushort m = ushort(L / 2u); m > 0; m >>= 1) {
                float os = simd_shuffle_xor(s, m);
                float oc = simd_shuffle_xor(c, m);
                moesimd_acc(s, c, os);
                c += oc;
            }
            if (lane == 0u) {
                y[(ulong)slot * rows + row] = s + c;
            }
        }

        """
    }
}

// MARK: - Compiled kernels

public struct MoESIMDCompiledKernel {
    public let subschedule: MoESIMDSubschedule
    public let facts: MoESIMDShapeFacts
    public let source: String
    public let sourceSha256: String
    public let compileOptions: String
    public let stageBytes: Int
    public let pipeline: MTLComputePipelineState
    public let device: MTLDevice

    /// Emits, compiles (plain: nil default options; neumaierPrecise:
    /// fastMathEnabled = false) and re-checks legality against the ACTUAL
    /// compiled pipeline — that guard is authoritative.
    public static func compile(_ s: MoESIMDSubschedule, facts: MoESIMDShapeFacts,
                               device: MTLDevice) throws -> MoESIMDCompiledKernel {
        let source = try MoESIMDKernelEmitter.emit(s, facts: facts)
        let options: MTLCompileOptions?
        switch s.accumulation {
        case .plain:
            options = nil
        case .neumaierPrecise:
            let o = MTLCompileOptions()
            o.fastMathEnabled = false
            options = o
        }
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: options) }
        catch { throw MoESIMDError.compileFailed("\(s.kernelName) library: \(error)") }
        guard let fn = library.makeFunction(name: s.kernelName) else {
            throw MoESIMDError.compileFailed("\(s.kernelName): function missing")
        }
        let pipeline: MTLComputePipelineState
        do { pipeline = try device.makeComputePipelineState(function: fn) }
        catch { throw MoESIMDError.compileFailed("\(s.kernelName) pipeline: \(error)") }
        let actual = MoESIMDHardwareProfile.actual(pipeline: pipeline, device: device)
        if let why = MoESIMDLegality.reason(s, facts: facts, hardware: actual) {
            throw MoESIMDError.illegalSchedule("\(s.kernelName) on the compiled pipeline: \(why)")
        }
        let stage = try MoESIMDLegality.stageBytes(s.family, facts)
        return MoESIMDCompiledKernel(
            subschedule: s, facts: facts, source: source,
            sourceSha256: SHA256.hex(SHA256.hash(Array(source.utf8))),
            compileOptions: s.accumulation.compileOptionsLabel, stageBytes: stage,
            pipeline: pipeline, device: device)
    }
}

// MARK: - Component bench (bounded packed real expert slices)

public struct MoESIMDComponentOutput {
    public let act: [Float]
    public let dn: [Float]
    public let guGpuSeconds: Double
    public let dnGpuSeconds: Double
}

/// Standalone component runner over packed per-slot expert slices (GU
/// slices first, DN slices base-shifted after them). Every output buffer
/// is NaN-filled before each dispatch and every output is finite-checked;
/// completion and positive ordered GPU timestamps are required.
public final class MoESIMDComponentBench {
    public static let weightBudgetBytes = 64 * 1024 * 1024

    public let facts: MoESIMDShapeFacts
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let weights: MTLBuffer
    private let guOffsets: [UInt64]
    private let dnOffsets: [UInt64]
    private let xBuffer: MTLBuffer
    private let actBuffer: MTLBuffer
    private let dnBuffer: MTLBuffer
    private let actCount: Int
    private let dnCount: Int

    public init(device: MTLDevice, facts: MoESIMDShapeFacts,
                guBlocks: [UInt8], dnBlocks: [UInt8]) throws {
        let guStride = try facts.guStrideBytes()
        let dnStride = try facts.dnStrideBytes()
        let guTotal = try MoESIMDChecked.mul(guStride, facts.topK, "packed GU bytes")
        let dnTotal = try MoESIMDChecked.mul(dnStride, facts.topK, "packed DN bytes")
        guard guBlocks.count == guTotal, dnBlocks.count == dnTotal else {
            throw MoESIMDError.componentFailed(
                "packed slices \(guBlocks.count)/\(dnBlocks.count) bytes != expected \(guTotal)/\(dnTotal)")
        }
        let total = try MoESIMDChecked.add(guTotal, dnTotal, "packed weight bytes")
        guard total <= Self.weightBudgetBytes, total <= device.maxBufferLength else {
            throw MoESIMDError.componentFailed(
                "packed weights \(total) bytes exceed the component budget or the device buffer cap")
        }
        // Q4_0 blocks are 18 bytes, so slice strides are even; offsets are
        // re-checked even because the kernels read ushort quant pairs.
        guard guStride % 2 == 0, dnStride % 2 == 0 else {
            throw MoESIMDError.componentFailed("slice strides not 2-byte aligned")
        }
        var guOffs = [UInt64]()
        var dnOffs = [UInt64]()
        for slot in 0..<facts.topK {
            let g = try MoESIMDChecked.mul(slot, guStride, "GU offset")
            let d = try MoESIMDChecked.add(guTotal, try MoESIMDChecked.mul(slot, dnStride, "DN offset"),
                                           "DN offset")
            let gEnd = try MoESIMDChecked.add(g, guStride, "GU end")
            let dEnd = try MoESIMDChecked.add(d, dnStride, "DN end")
            guard g % 2 == 0, d % 2 == 0, gEnd <= guTotal, dEnd <= total else {
                throw MoESIMDError.componentFailed("slot \(slot) offsets misaligned or out of extent")
            }
            guOffs.append(UInt64(g))
            dnOffs.append(UInt64(d))
        }
        let actCount = try MoESIMDChecked.mul(facts.topK, facts.inter, "activation floats")
        let dnCount = try MoESIMDChecked.mul(facts.topK, facts.hidden, "down floats")
        var packed = guBlocks
        packed.append(contentsOf: dnBlocks)
        guard let queue = device.makeCommandQueue(),
              let weights = device.makeBuffer(bytes: packed, length: total,
                                              options: .storageModeShared),
              let xBuffer = device.makeBuffer(length: facts.hidden * 4,
                                              options: .storageModeShared),
              let actBuffer = device.makeBuffer(length: actCount * 4,
                                                options: .storageModeShared),
              let dnBuffer = device.makeBuffer(length: dnCount * 4,
                                               options: .storageModeShared) else {
            throw MoESIMDError.hardwareUnavailable("component buffers")
        }
        self.facts = facts
        self.device = device
        self.queue = queue
        self.weights = weights
        self.guOffsets = guOffs
        self.dnOffsets = dnOffs
        self.xBuffer = xBuffer
        self.actBuffer = actBuffer
        self.dnBuffer = dnBuffer
        self.actCount = actCount
        self.dnCount = dnCount
    }

    /// End-to-end component: fused GU writes the activation on the GPU, and
    /// the down kernel reads that GPU-produced activation unchanged.
    public func runComponent(gu: MoESIMDCompiledKernel, dn: MoESIMDCompiledKernel,
                             batch: Int, x: [Float]) throws -> MoESIMDComponentOutput {
        guard dn.subschedule.family == .down else {
            throw MoESIMDError.componentFailed("\(dn.subschedule.kernelName) is not a down kernel")
        }
        let g = try runFusedGateUp(gu, batch: batch, x: x)
        let d = try runDownOnResidentActivation(dn, batch: batch)
        return MoESIMDComponentOutput(act: g.act, dn: d.dn,
                                      guGpuSeconds: g.gpuSeconds, dnGpuSeconds: d.gpuSeconds)
    }

    public func runFusedGateUp(_ kernel: MoESIMDCompiledKernel, batch: Int,
                               x: [Float]) throws -> (act: [Float], gpuSeconds: Double) {
        guard kernel.subschedule.family == .fusedGateUpAct else {
            throw MoESIMDError.componentFailed("\(kernel.subschedule.kernelName) is not a fused GU kernel")
        }
        guard x.count == facts.hidden, x.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.componentFailed("input must carry \(facts.hidden) finite floats")
        }
        x.withUnsafeBytes { raw in
            xBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        fillNaN(actBuffer, count: actCount)
        let seconds = try encode(kernel, batch: batch, input: xBuffer, output: actBuffer)
        let act = readFloats(actBuffer, count: actCount)
        guard act.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.componentFailed("non-finite activation output (NaN-filled before dispatch)")
        }
        return (act, seconds)
    }

    /// Down kernel over an explicit host activation (isolated screening).
    public func runDown(_ kernel: MoESIMDCompiledKernel, batch: Int,
                        act: [Float]) throws -> (dn: [Float], gpuSeconds: Double) {
        guard kernel.subschedule.family == .down else {
            throw MoESIMDError.componentFailed("\(kernel.subschedule.kernelName) is not a down kernel")
        }
        guard act.count == actCount, act.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.componentFailed("activation must carry \(actCount) finite floats")
        }
        act.withUnsafeBytes { raw in
            actBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        return try runDownOnResidentActivation(kernel, batch: batch)
    }

    private func runDownOnResidentActivation(_ kernel: MoESIMDCompiledKernel,
                                             batch: Int) throws -> (dn: [Float], gpuSeconds: Double) {
        guard kernel.subschedule.family == .down else {
            throw MoESIMDError.componentFailed("\(kernel.subschedule.kernelName) is not a down kernel")
        }
        fillNaN(dnBuffer, count: dnCount)
        let seconds = try encode(kernel, batch: batch, input: actBuffer, output: dnBuffer)
        let dn = readFloats(dnBuffer, count: dnCount)
        guard dn.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.componentFailed("non-finite down output (NaN-filled before dispatch)")
        }
        return (dn, seconds)
    }

    private func fillNaN(_ buffer: MTLBuffer, count: Int) {
        let p = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { p[i] = .nan }
    }

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self),
                                  count: count))
    }

    private func encode(_ kernel: MoESIMDCompiledKernel, batch: Int,
                        input: MTLBuffer, output: MTLBuffer) throws -> Double {
        let s = kernel.subschedule
        guard kernel.device === device, kernel.facts == facts else {
            throw MoESIMDError.componentFailed("kernel compiled for a different device or shape")
        }
        let actual = MoESIMDHardwareProfile.actual(pipeline: kernel.pipeline, device: device)
        if let why = MoESIMDLegality.reason(s, facts: facts, hardware: actual) {
            throw MoESIMDError.illegalSchedule("\(s.kernelName): \(why)")
        }
        if let why = MoESIMDLegality.batchReason(batch, facts: facts) {
            throw MoESIMDError.illegalSchedule(why)
        }
        let rows = MoESIMDLegality.logicalRows(s.family, facts)
        let inDim = MoESIMDLegality.inputDim(s.family, facts)
        let groupsPerSlot = rows / s.rowsPerGroup
        let threads = s.lanes * s.rowsPerGroup
        let offsets = s.family == .fusedGateUpAct ? guOffsets : dnOffsets
        // Chunk plan, host-checked BEFORE any encoder exists.
        var chunks = [(start: Int, count: Int)]()
        var start = 0
        while start < facts.topK {
            let count = min(batch, facts.topK - start)
            guard count > 0, count <= batch, start + count <= facts.topK,
                  count * groupsPerSlot <= Int(UInt32.max) else {
                throw MoESIMDError.componentFailed("chunk bounds violated")
            }
            chunks.append((start: start, count: count))
            start += count
        }
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MoESIMDError.hardwareUnavailable("command buffer/encoder")
        }
        enc.setComputePipelineState(kernel.pipeline)
        enc.setThreadgroupMemoryLength(kernel.stageBytes, index: 0)
        offsets.withUnsafeBytes { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 1)
        }
        enc.setBuffer(weights, offset: 0, index: 2)
        enc.setBuffer(input, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)
        for chunk in chunks {
            var p = SIMD4<UInt32>(UInt32(inDim), UInt32(rows),
                                  UInt32(chunk.count), UInt32(chunk.start))
            enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
            enc.dispatchThreadgroups(
                MTLSize(width: chunk.count * groupsPerSlot, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw MoESIMDError.componentFailed(
                "\(s.kernelName): status \(cmd.status.rawValue), \(String(describing: cmd.error))")
        }
        let gpuStart = cmd.gpuStartTime
        let gpuEnd = cmd.gpuEndTime
        guard gpuStart > 0, gpuEnd > gpuStart, (gpuEnd - gpuStart).isFinite else {
            throw MoESIMDError.componentFailed(
                "\(s.kernelName): invalid GPU timestamps start \(gpuStart) end \(gpuEnd)")
        }
        return gpuEnd - gpuStart
    }
}

// MARK: - Independent full CPU FP64 oracle

public struct MoESIMDReference: Sendable {
    /// Slot-major [topK][inter] gate rows, up rows and activations, and
    /// [topK][hidden] down outputs — all computed in Double.
    public let gate: [Double]
    public let up: [Double]
    public let act: [Double]
    public let dn: [Double]
}

public enum MoESIMDOracle {
    /// The unchanged normalized component contract.
    public static let contract = 1e-4

    /// Independent Double-input Q4_0 matvec over a bounded byte extent:
    /// FP16 scale and biased nibbles decoded in Double, Double accumulation.
    public static func q4MatvecFP64(blocks: [UInt8], byteOffset: Int, outDim: Int,
                                    inDim: Int, x: [Double]) throws -> [Double] {
        guard outDim > 0, inDim > 0, inDim % 32 == 0, byteOffset >= 0,
              x.count == inDim, x.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.oracleFailed(
                "oracle geometry/input invalid (\(outDim)x\(inDim), x \(x.count))")
        }
        let blocksPerRow = inDim / 32
        let (rowBytes, o1) = blocksPerRow.multipliedReportingOverflow(by: 18)
        let (needed, o2) = outDim.multipliedReportingOverflow(by: rowBytes)
        let (end, o3) = byteOffset.addingReportingOverflow(needed)
        guard !o1, !o2, !o3, end <= blocks.count else {
            throw MoESIMDError.oracleFailed(
                "oracle extent at \(byteOffset) (+\(needed)) beyond \(blocks.count) bytes")
        }
        var y = [Double](repeating: 0, count: outDim)
        try blocks.withUnsafeBufferPointer { w in
            try x.withUnsafeBufferPointer { xv in
                for row in 0..<outDim {
                    let rowBase = byteOffset + row * rowBytes
                    var sum = 0.0
                    for b in 0..<blocksPerRow {
                        let o = rowBase + b * 18
                        let bits = UInt16(w[o]) | (UInt16(w[o + 1]) << 8)
                        guard bits & 0x7C00 != 0x7C00 else {
                            throw MoESIMDError.oracleFailed("non-finite FP16 scale at byte \(o)")
                        }
                        let d = Double(Float16(bitPattern: bits))
                        let xb = b * 32
                        for l in 0..<16 {
                            let q = w[o + 2 + l]
                            sum += d * Double(Int(q & 0x0F) - 8) * xv[xb + l]
                            sum += d * Double(Int(q >> 4) - 8) * xv[xb + 16 + l]
                        }
                    }
                    guard sum.isFinite else {
                        throw MoESIMDError.oracleFailed("non-finite FP64 output row \(row)")
                    }
                    y[row] = sum
                }
            }
        }
        return y
    }

    /// Double GELU(tanh)(gate) * up — the reference formula, evaluated in
    /// Double (never through the GPU helper).
    public static func geluTanhUp(gate g: Double, up u: Double) -> Double {
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        return 0.5 * g * (1.0 + tanh(inner)) * u
    }

    /// Full end-to-end reference for ALL slots: every gate/up row, every
    /// activation from the Double GU outputs, and every down row from the
    /// Double activation (never from GPU values or Float-cast references).
    public static func reference(facts: MoESIMDShapeFacts, guBlocks: [UInt8],
                                 dnBlocks: [UInt8], x: [Float]) throws -> MoESIMDReference {
        guard x.count == facts.hidden, x.allSatisfy({ $0.isFinite }) else {
            throw MoESIMDError.oracleFailed("x must carry \(facts.hidden) finite floats")
        }
        let guStride = try facts.guStrideBytes()
        let dnStride = try facts.dnStrideBytes()
        let guTotal = try MoESIMDChecked.mul(guStride, facts.topK, "GU bytes")
        let dnTotal = try MoESIMDChecked.mul(dnStride, facts.topK, "DN bytes")
        guard guBlocks.count == guTotal, dnBlocks.count == dnTotal else {
            throw MoESIMDError.oracleFailed("packed slice sizes do not match the shape")
        }
        let xd = x.map { Double($0) }
        var gate = [Double]()
        var up = [Double]()
        var act = [Double]()
        var dn = [Double]()
        gate.reserveCapacity(facts.topK * facts.inter)
        up.reserveCapacity(facts.topK * facts.inter)
        act.reserveCapacity(facts.topK * facts.inter)
        dn.reserveCapacity(facts.topK * facts.hidden)
        for slot in 0..<facts.topK {
            let gu = try q4MatvecFP64(blocks: guBlocks, byteOffset: slot * guStride,
                                      outDim: facts.gateUpRows, inDim: facts.hidden, x: xd)
            let g = Array(gu[0..<facts.inter])
            let u = Array(gu[facts.inter..<facts.gateUpRows])
            var a = [Double](repeating: 0, count: facts.inter)
            for i in 0..<facts.inter {
                a[i] = geluTanhUp(gate: g[i], up: u[i])
                guard a[i].isFinite else {
                    throw MoESIMDError.oracleFailed("non-finite FP64 activation slot \(slot) lane \(i)")
                }
            }
            let d = try q4MatvecFP64(blocks: dnBlocks, byteOffset: slot * dnStride,
                                     outDim: facts.hidden, inDim: facts.inter, x: a)
            gate.append(contentsOf: g)
            up.append(contentsOf: u)
            act.append(contentsOf: a)
            dn.append(contentsOf: d)
        }
        return MoESIMDReference(gate: gate, up: up, act: act, dn: dn)
    }

    /// Worst abs(delta) / max(1, abs(ref)); infinity on any count mismatch
    /// or non-finite value.
    public static func worstNormalizedRel(_ gpu: [Float], _ ref: [Double]) -> Double {
        guard gpu.count == ref.count, !gpu.isEmpty else { return .infinity }
        var worst = 0.0
        for i in 0..<gpu.count {
            let g = Double(gpu[i])
            guard g.isFinite, ref[i].isFinite else { return .infinity }
            worst = max(worst, abs(g - ref[i]) / max(1.0, abs(ref[i])))
        }
        return worst
    }

    /// JSON-safe value: non-finite measurements are reported as nil.
    public static func reportable(_ v: Double) -> Double? { v.isFinite ? v : nil }
}

public enum MoESIMDBits {
    public static func patterns(_ v: [Float]) -> [UInt32] { v.map { $0.bitPattern } }

    public static func identical(_ a: [Float], _ b: [Float]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
    }
}

// MARK: - Qualification (component screening; no selection)

public struct MoESIMDNamedInput: Sendable {
    public let name: String
    public let values: [Float]
    public init(name: String, values: [Float]) {
        self.name = name
        self.values = values
    }
}

public struct MoESIMDInputRecord: Codable, Equatable, Sendable {
    public let name: String
    public let elementCount: Int
    public let provenance: String
    public init(name: String, elementCount: Int, provenance: String) {
        self.name = name
        self.elementCount = elementCount
        self.provenance = provenance
    }
}

public struct MoESIMDInputResult: Codable, Equatable, Sendable {
    public let input: String
    public let actWorstRel: Double?
    public let dnWorstRel: Double?
    /// Down kernel fed the Float-rounded FP64 activation, compared with the
    /// FP64 down reference (attributes error to the down kernel alone).
    public let dnIsolatedWorstRel: Double?
    public let repeatBitIdentical: Bool
    public let passed: Bool
    /// Diagnostic GPU intervals only (not a speed qualification).
    public let guGpuMicros: Double?
    public let dnGpuMicros: Double?
    public let failure: String?
}

public struct MoESIMDScheduleResult: Codable, Equatable, Sendable {
    public let schedule: MoESIMDSchedule
    public let guSourceSha256: String?
    public let dnSourceSha256: String?
    public let guCompileOptions: String
    public let dnCompileOptions: String
    public let inputs: [MoESIMDInputResult]
    /// Bitwise equality of first-input outputs across E for identical
    /// kernels (nil when no E peer was evaluated).
    public var batchInvariantBits: Bool?
    public var componentAccepted: Bool
    public var rejection: String?
}

public struct MoESIMDIncumbentResult: Codable, Equatable, Sendable {
    public let input: String
    public let schedule: String
    public let gateWorstRel: Double?
    public let upWorstRel: Double?
    public let actWorstRel: Double?
    public let dnWorstRel: Double?
    public let withinContract: Bool
    public let failure: String?
}

public enum MoESIMDQualification {
    /// Screens every derived schedule on every input against the shared
    /// FP64 references: end-to-end act/down, isolated down, repeat-run
    /// bitPattern determinism, and E-batch bitwise invariance. Every
    /// rejection is recorded; nothing is selected or activated.
    public static func screen(derivation: MoESIMDDerivation,
                              bench: MoESIMDComponentBench,
                              inputs: [MoESIMDNamedInput],
                              references: [MoESIMDReference],
                              log: (String) -> Void) throws -> [MoESIMDScheduleResult] {
        guard derivation.facts == bench.facts else {
            throw MoESIMDError.componentFailed("derivation facts differ from the bench facts")
        }
        guard !inputs.isEmpty, inputs.count == references.count,
              Set(inputs.map { $0.name }).count == inputs.count else {
            throw MoESIMDError.componentFailed(
                "inputs must be non-empty, uniquely named, with one reference each")
        }
        let facts = bench.facts
        let contract = MoESIMDOracle.contract
        var compiled = [MoESIMDSubschedule: Result<MoESIMDCompiledKernel, Error>]()
        func kernel(_ s: MoESIMDSubschedule) -> Result<MoESIMDCompiledKernel, Error> {
            if let cached = compiled[s] { return cached }
            let r = Result<MoESIMDCompiledKernel, Error> {
                try MoESIMDCompiledKernel.compile(s, facts: facts, device: bench.device)
            }
            compiled[s] = r
            return r
        }
        var results = [MoESIMDScheduleResult]()
        var firstInputBits = [MoESIMDSchedule: [UInt32]]()
        for schedule in derivation.schedules {
            let guResult = kernel(schedule.gu)
            let dnResult = kernel(schedule.dn)
            guard case .success(let gk) = guResult, case .success(let dk) = dnResult else {
                var why = [String]()
                if case .failure(let e) = guResult { why.append("GU: \(e)") }
                if case .failure(let e) = dnResult { why.append("DN: \(e)") }
                let reason = why.joined(separator: "; ")
                log("rejected \(schedule.label): \(reason)")
                results.append(MoESIMDScheduleResult(
                    schedule: schedule, guSourceSha256: nil, dnSourceSha256: nil,
                    guCompileOptions: schedule.gu.accumulation.compileOptionsLabel,
                    dnCompileOptions: schedule.dn.accumulation.compileOptionsLabel,
                    inputs: [], batchInvariantBits: nil, componentAccepted: false,
                    rejection: reason))
                continue
            }
            var inputResults = [MoESIMDInputResult]()
            for (index, input) in inputs.enumerated() {
                let reference = references[index]
                do {
                    let first = try bench.runComponent(gu: gk, dn: dk, batch: schedule.batch,
                                                       x: input.values)
                    let second = try bench.runComponent(gu: gk, dn: dk, batch: schedule.batch,
                                                        x: input.values)
                    let repeatBits = MoESIMDBits.identical(first.act, second.act)
                        && MoESIMDBits.identical(first.dn, second.dn)
                    let actRel = MoESIMDOracle.worstNormalizedRel(first.act, reference.act)
                    let dnRel = MoESIMDOracle.worstNormalizedRel(first.dn, reference.dn)
                    let isolated = try bench.runDown(dk, batch: schedule.batch,
                                                     act: reference.act.map { Float($0) })
                    let isoRel = MoESIMDOracle.worstNormalizedRel(isolated.dn, reference.dn)
                    var reasons = [String]()
                    if !repeatBits { reasons.append("repeat run not bit-identical") }
                    if !(actRel < contract) {
                        reasons.append(String(format: "act worst rel %.3e", actRel))
                    }
                    if !(dnRel < contract) {
                        reasons.append(String(format: "end-to-end down worst rel %.3e", dnRel))
                    }
                    if !(isoRel < contract) {
                        reasons.append(String(format: "isolated down worst rel %.3e", isoRel))
                    }
                    if index == 0 {
                        firstInputBits[schedule] = MoESIMDBits.patterns(first.act)
                            + MoESIMDBits.patterns(first.dn)
                    }
                    inputResults.append(MoESIMDInputResult(
                        input: input.name,
                        actWorstRel: MoESIMDOracle.reportable(actRel),
                        dnWorstRel: MoESIMDOracle.reportable(dnRel),
                        dnIsolatedWorstRel: MoESIMDOracle.reportable(isoRel),
                        repeatBitIdentical: repeatBits, passed: reasons.isEmpty,
                        guGpuMicros: first.guGpuSeconds * 1e6,
                        dnGpuMicros: first.dnGpuSeconds * 1e6,
                        failure: reasons.isEmpty ? nil : reasons.joined(separator: "; ")))
                } catch {
                    inputResults.append(MoESIMDInputResult(
                        input: input.name, actWorstRel: nil, dnWorstRel: nil,
                        dnIsolatedWorstRel: nil, repeatBitIdentical: false, passed: false,
                        guGpuMicros: nil, dnGpuMicros: nil, failure: "\(error)"))
                }
            }
            let passedAll = inputResults.allSatisfy { $0.passed }
            log("\(passedAll ? "accepted" : "FAILED") \(schedule.label)")
            results.append(MoESIMDScheduleResult(
                schedule: schedule, guSourceSha256: gk.sourceSha256,
                dnSourceSha256: dk.sourceSha256, guCompileOptions: gk.compileOptions,
                dnCompileOptions: dk.compileOptions, inputs: inputResults,
                batchInvariantBits: nil, componentAccepted: passedAll,
                rejection: passedAll ? nil : "one or more inputs failed"))
        }
        for i in results.indices {
            let schedule = results[i].schedule
            guard let mine = firstInputBits[schedule] else { continue }
            let peers = firstInputBits.filter {
                $0.key != schedule && $0.key.batchFreeLabel == schedule.batchFreeLabel
            }
            guard !peers.isEmpty else { continue }
            let invariant = peers.values.allSatisfy { $0 == mine }
            results[i].batchInvariantBits = invariant
            if !invariant {
                results[i].componentAccepted = false
                results[i].rejection = "E-batch variants of identical kernels differ bitwise"
            }
        }
        return results
    }

    /// The EXISTING v1 E4R32 kernels (untouched emission) measured against
    /// the SAME full FP64 references; failures are recorded honestly.
    public static func incumbentV1(region: MoERegionIR, device: MTLDevice,
                                   guBlocks: [UInt8], dnBlocks: [UInt8],
                                   inputs: [MoESIMDNamedInput],
                                   references: [MoESIMDReference]) -> [MoESIMDIncumbentResult] {
        let schedule = MoESchedule(batch: 4, rowsPerGroup: 32)
        let label = "v1 " + schedule.guKernelName + "/" + schedule.actKernelName
            + "/" + schedule.dnKernelName
        func failedResult(_ name: String, _ why: String) -> MoESIMDIncumbentResult {
            MoESIMDIncumbentResult(input: name, schedule: label, gateWorstRel: nil,
                                   upWorstRel: nil, actWorstRel: nil, dnWorstRel: nil,
                                   withinContract: false, failure: why)
        }
        guard inputs.count == references.count else {
            return inputs.map { failedResult($0.name, "inputs/references mismatch") }
        }
        let kernels: MoEEmittedKernels
        do {
            kernels = try MoEEmittedKernels.compile(region: region, schedule: schedule,
                                                    device: device)
        } catch {
            return inputs.map { failedResult($0.name, "compile: \(error)") }
        }
        let inter = region.expertIntermediate
        let gateUpOut = region.gateUpOut
        var out = [MoESIMDIncumbentResult]()
        for (index, input) in inputs.enumerated() {
            do {
                let r = try kernels.runComponent(region: region, guBlocks: guBlocks,
                                                 dnBlocks: dnBlocks, x: input.values)
                var gate = [Float]()
                var up = [Float]()
                for slot in 0..<region.topK {
                    let base = slot * gateUpOut
                    guard base + gateUpOut <= r.gu.count else {
                        throw MoESIMDError.componentFailed("v1 GU output too short")
                    }
                    gate.append(contentsOf: r.gu[base..<(base + inter)])
                    up.append(contentsOf: r.gu[(base + inter)..<(base + gateUpOut)])
                }
                let ref = references[index]
                let g = MoESIMDOracle.worstNormalizedRel(gate, ref.gate)
                let u = MoESIMDOracle.worstNormalizedRel(up, ref.up)
                let a = MoESIMDOracle.worstNormalizedRel(r.act, ref.act)
                let d = MoESIMDOracle.worstNormalizedRel(r.dn, ref.dn)
                let contract = MoESIMDOracle.contract
                let within = g < contract && u < contract && a < contract && d < contract
                out.append(MoESIMDIncumbentResult(
                    input: input.name, schedule: label,
                    gateWorstRel: MoESIMDOracle.reportable(g),
                    upWorstRel: MoESIMDOracle.reportable(u),
                    actWorstRel: MoESIMDOracle.reportable(a),
                    dnWorstRel: MoESIMDOracle.reportable(d),
                    withinContract: within,
                    failure: within ? nil : "outside the full FP64 contract"))
            } catch {
                out.append(failedResult(input.name, "\(error)"))
            }
        }
        return out
    }
}

// MARK: - Non-installable component report

/// Component qualification report. NOT a production artifact: there is no
/// installer for it, installable/productionArtifact/winnerSelected are
/// always false, and decodeValidated rejects any record claiming
/// otherwise or any other numerical class.
public struct MoESIMDComponentReport: Codable, Equatable {
    public static let currentSchema = 1
    public static let kindValue = "moe-simd-component-qualification"
    public static let numericalClassValue = "simd-reassociated-oracle"
    public static let standingNotes: [String] = [
        "Correctness prototype only: NOT a production artifact; installable=false; no runner integration.",
        "Numerical class simd-reassociated-oracle: per-row dots are reassociated across lanes; gated only against the independent full FP64 oracle at the unchanged normalized 1e-4 contract; no bit identity with v1 is claimed.",
        "GPU intervals are diagnostic only: the A/A reproducibility gate failed and global paging was active; no paired comparison, no winner, no activation.",
        "Model SHA-256 is the cached historical record (explicit-refresh policy); it was not re-verified in this run.",
        "Catalog bounded by the same-L GU/DN composition rule; E affects dispatch only (verified by bitwise batch invariance).",
        "Direct-dispatch diagnostic precondition recorded: generic g4_q4_matvec exceeded routed MoE GU+DN+ACT under a different synchronization; this package stays MoE-scoped."
    ]

    public let reportSchema: Int
    public let kind: String
    public let numericalClass: String
    public let installable: Bool
    public let productionArtifact: Bool
    public let winnerSelected: Bool
    public let diagnosticTimingOnly: Bool
    public let codegenVersion: String
    public let regionGraphSha256: String
    public let regionDigest: String
    public let facts: MoESIMDShapeFacts
    public let hardware: MoESIMDHardwareProfile
    public let toolchain: String
    public let modelSha256: String
    public let modelFileVerification: ModelFileVerificationPolicy
    public let executableDigest: String
    public let guTensor: String
    public let dnTensor: String
    public let expertIds: [Int]
    public let oracleContract: Double
    public let inputs: [MoESIMDInputRecord]
    public let derivationRejections: [MoESIMDRejection]
    public let catalogSize: Int
    public let acceptedCount: Int
    public let results: [MoESIMDScheduleResult]
    public let incumbent: [MoESIMDIncumbentResult]
    public let notes: [String]

    public static func make(region: MoERegionIR, hardware: MoESIMDHardwareProfile,
                            derivation: MoESIMDDerivation,
                            results: [MoESIMDScheduleResult],
                            incumbent: [MoESIMDIncumbentResult],
                            inputs: [MoESIMDInputRecord],
                            expertIds: [Int],
                            modelSha256: String,
                            executableDigest: String) throws -> MoESIMDComponentReport {
        let facts = try MoESIMDShapeFacts(region: region)
        guard derivation.facts == facts, derivation.hardware == hardware else {
            throw MoESIMDError.invalidReport("derivation was not derived from this region and hardware")
        }
        guard moeIsLowercaseHex64(modelSha256), moeIsLowercaseHex64(executableDigest) else {
            throw MoESIMDError.invalidReport("model/executable digests must be lowercase 64-hex")
        }
        guard expertIds.count == facts.topK, Set(expertIds).count == expertIds.count,
              expertIds.allSatisfy({ $0 >= 0 }) else {
            throw MoESIMDError.invalidReport("expert ids must be topK unique non-negative values")
        }
        let report = MoESIMDComponentReport(
            reportSchema: currentSchema, kind: kindValue,
            numericalClass: numericalClassValue,
            installable: false, productionArtifact: false, winnerSelected: false,
            diagnosticTimingOnly: true,
            codegenVersion: MoESIMDKernelEmitter.codegenVersion,
            regionGraphSha256: region.graphSha256, regionDigest: try region.digest(),
            facts: facts, hardware: hardware, toolchain: Forge.toolchainIdentifier,
            modelSha256: modelSha256, modelFileVerification: ModelFileIdentity.policy,
            executableDigest: executableDigest, guTensor: region.guTensor,
            dnTensor: region.dnTensor, expertIds: expertIds,
            oracleContract: MoESIMDOracle.contract, inputs: inputs,
            derivationRejections: derivation.rejections,
            catalogSize: derivation.schedules.count,
            acceptedCount: results.filter { $0.componentAccepted }.count,
            results: results, incumbent: incumbent, notes: standingNotes)
        try report.validateNonInstallable()
        return report
    }

    /// Fail-closed identity check: this record can never describe an
    /// installable or production artifact, a selected winner, or another
    /// numerical class.
    public func validateNonInstallable() throws {
        guard reportSchema == Self.currentSchema, kind == Self.kindValue,
              numericalClass == Self.numericalClassValue,
              !installable, !productionArtifact, !winnerSelected, diagnosticTimingOnly,
              codegenVersion == MoESIMDKernelEmitter.codegenVersion,
              oracleContract == MoESIMDOracle.contract,
              moeIsLowercaseHex64(regionGraphSha256), moeIsLowercaseHex64(modelSha256),
              moeIsLowercaseHex64(executableDigest) else {
            throw MoESIMDError.invalidReport(
                "record is not a valid non-installable simd-reassociated-oracle component report")
        }
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try encoder.encode(self)
    }

    public static func decodeValidated(_ data: Data) throws -> MoESIMDComponentReport {
        let report: MoESIMDComponentReport
        do { report = try JSONDecoder().decode(MoESIMDComponentReport.self, from: data) }
        catch { throw MoESIMDError.invalidReport("decode: \(error)") }
        try report.validateNonInstallable()
        return report
    }
}
