// DIAGNOSTIC ONLY: one bounded-parallel single-query attention candidate.
// It is never production, never artifact-bound, never a serving default, and
// productionQualified is always false.
//
// Hypothesis (unmeasured):
// - The incumbent g4_attn runs one thread per query head, i.e. 16 threads
//   per decode step.
// - Each thread computes every q.k dot product three times, then runs the
//   weighted V sum serially for every output dimension.
//
// Candidate `g4_attn_par64_diag`: one 64-lane threadgroup per query head.
//   1. Scores. Attended tokens are distributed across lanes. Each score is
//      ONE serial dot product in the incumbent's i order, computed once and
//      cached in threadgroup memory: 4096 floats = 16 KiB, the engine
//      context length.
//   2. Lane 0 computes the max, then exp(s - m) and the exp-sum, both in
//      chronological token order. Each exp is cached in place.
//   3. Each token's weight exp / l (the incumbent's division) is formed
//      once, in parallel.
//   4. Output dimensions are distributed across lanes. Each lane
//      accumulates its weighted V sums in chronological order, starting
//      from 0.0f.
//
// Arithmetic guarantees and caveats:
// - No dot product or reduction is reassociated.
// - exp and the normalization are not changed.
// - A cached score or exp equals a recomputed one only if the compiler emits
//   identical arithmetic for each recomputation. FMA contraction may still
//   differ from the incumbent, so the ACTUAL cached-incumbent component gate
//   and the full gate decide.
//
// Safety:
// - Barriers are never inside lane-dependent branches.
// - Early returns are threadgroup-uniform only: head bound, invalid
//   geometry, capacity.
// - An attended range larger than the cache writes NaN; it is never
//   silently truncated.
// - Host eligibility routes such calls to the incumbent and counts them as
//   fallbacks.

import Foundation
import Metal

/// Thread-safe dispatch counters for the attention diagnostic. They are
/// reset on entry to `withAttentionDiagnostic` and updated only by the
/// diagnostic branch of the attention dispatch site.
public final class AttentionDiagnosticCounters: @unchecked Sendable {
    private let lock = NSLock()
    private var attempts = 0
    private var applied = 0
    private var fallback = 0

    public init() {}

    public func reset() {
        lock.lock()
        attempts = 0
        applied = 0
        fallback = 0
        lock.unlock()
    }

    public func recordApplied() {
        lock.lock()
        attempts += 1
        applied += 1
        lock.unlock()
    }

    public func recordFallback() {
        lock.lock()
        attempts += 1
        fallback += 1
        lock.unlock()
    }

    public func snapshot() -> (attempts: Int, applied: Int, fallback: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (attempts, applied, fallback)
    }
}

public enum AttentionDiagnosticError: Error, CustomStringConvertible {
    case gate(String)
    case hardware(String)

    public var description: String {
        switch self {
        case .gate(let s): return "attention gate: " + s
        case .hardware(let s): return "attention hardware: " + s
        }
    }
}

public struct AttentionComponentRecord: Sendable {
    public let label: String
    public let tokens: Int
    public let queryPosition: Int
    public let attended: Int
    public let headDim: Int
    public let qHeads: Int
    public let kvHeads: Int
    public let window: Int
    public let bitIdenticalToReference: Bool
    public let maxAbsCandidateVsReference: Double
    public let relCandidateVsReference: Double
    public let maxAbsCandidateVsFP64: Double
    public let relCandidateVsFP64: Double
    public let maxAbsReferenceVsFP64: Double
    public let relReferenceVsFP64: Double
    public let maxAbsReferenceOutput: Double
    public let allFinite: Bool
    public let deterministic: Bool
    public let referenceGpuMicros: Double
    public let candidateGpuMicros: Double
    public let pass: Bool
}

public enum AttentionDiagnostic {
    public static let kernelName = "g4_attn_par64_diag"
    public static let canaryKernelName = "g4_attn_canary_zero_diag"
    public static let labelPrefix = "attn-diagnostic"
    /// Separate pipeline-cache key. The cached g4_attn entry is untouched.
    public static let slotKey = "diagnostic.attention.candidate"
    public static let threads = 64
    public static let maxContext = 4096
    public static let maxHeadDim = 512
    public static let localHeadDim = 256
    public static let localKVHeads = 8
    public static let globalHeadDim = 512
    public static let globalKVHeads = 2
    public static let productionQualified = false
    public static let relBound = 1e-4
    public static let guardFloats = 64
    public static let sentinelBits: UInt32 = 0x7FC0_A77E
    public static let tokenCases = [1, 2, 17, 33, 65, 257, 1024, 1025, 4096]
    public static let counters = AttentionDiagnosticCounters()

    /// Candidate source. It has the same ABI as g4_attn:
    /// - p = (T, qpos, kvHeads, headDim) at 0;
    /// - q = (qHeads, window) at 1;
    /// - query at 2, K cache at 3, V cache at 4, output at 5.
    public static let candidateSource = """
    #include <metal_stdlib>
    using namespace metal;

    #define ATTN_CAP 4096

    // DIAGNOSTIC candidate (never production): one 64-lane threadgroup per
    // query head, scores computed once, chronological reductions.
    kernel void g4_attn_par64_diag(constant const uint4& p [[buffer(0)]],
                                   constant const uint2& q [[buffer(1)]],
                                   device const float* qv [[buffer(2)]],
                                   device const float* kc [[buffer(3)]],
                                   device const float* vc [[buffer(4)]],
                                   device float* out [[buffer(5)]],
                                   uint head [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]],
                                   uint lanes [[threads_per_threadgroup]]) {
        threadgroup float sc[ATTN_CAP];
        threadgroup float stat[1];
        if (head >= q.x) { return; }
        uint qpos = p.y, kvHeads = p.z, hd = p.w;
        uint window = q.y;
        device float* oh = out + (ulong)head * hd;
        if (kvHeads == 0u || q.x < kvHeads) {
            for (uint i = lane; i < hd; i += lanes) { oh[i] = as_type<float>(0x7FC00000u); }
            return;
        }
        uint group = q.x / kvHeads;
        uint kvh = head / group;
        uint lo = 0u;
        if (window != 0u && qpos + 1u > window) { lo = qpos + 1u - window; }
        uint n = qpos + 1u - lo;
        if (n > ATTN_CAP) {
            for (uint i = lane; i < hd; i += lanes) { oh[i] = as_type<float>(0x7FC00000u); }
            return;
        }
        device const float* qh = qv + (ulong)head * hd;
        device const float* khead = kc + (ulong)kvh * hd;
        device const float* vhead = vc + (ulong)kvh * hd;
        for (uint t = lo + lane; t <= qpos; t += lanes) {
            device const float* kt = khead + (ulong)t * kvHeads * hd;
            float s = 0.0f;
            for (uint i = 0; i < hd; i++) { s += qh[i] * kt[i]; }
            sc[t - lo] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float m = -INFINITY;
            for (uint j = 0; j < n; j++) { m = max(m, sc[j]); }
            float l = 0.0f;
            for (uint j = 0; j < n; j++) {
                float e = exp(sc[j] - m);
                l += e;
                sc[j] = e;
            }
            stat[0] = l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float l = stat[0];
        for (uint j = lane; j < n; j += lanes) { sc[j] = sc[j] / l; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < hd; i += lanes) {
            float acc = 0.0f;
            for (uint j = 0; j < n; j++) {
                device const float* vt = vhead + (ulong)(lo + j) * kvHeads * hd;
                acc += sc[j] * vt[i];
            }
            oh[i] = acc;
        }
    }
    """

    /// Dispatch-reachability canary with the same ABI. It writes exact zeros.
    /// A logit change proves only that the diagnostic slot is reached; it is
    /// never a quality result.
    public static let canarySource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void g4_attn_canary_zero_diag(constant const uint4& p [[buffer(0)]],
                                         constant const uint2& q [[buffer(1)]],
                                         device const float* qv [[buffer(2)]],
                                         device const float* kc [[buffer(3)]],
                                         device const float* vc [[buffer(4)]],
                                         device float* out [[buffer(5)]],
                                         uint head [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]],
                                         uint lanes [[threads_per_threadgroup]]) {
        if (head >= q.x) { return; }
        for (uint i = lane; i < p.w; i += lanes) { out[(ulong)head * p.w + i] = 0.0f; }
    }
    """

    public static func candidateSourceSha256() -> String {
        MoEKernelEmitter.sourcesSha256([candidateSource])
    }

    /// Actual runner configuration fields used by the qualifier.
    public static func configSummary(runner: Gemma4Runner) -> [String: Int] {
        return ["qHeads": Int(runner.config.qHeads),
                "slidingWindow": Int(runner.config.slidingWindow)]
    }

    /// Compiles the candidate (or the canary) on the runner's OWN device with
    /// default options (nil), labelled with `labelPrefix`.
    public static func compile(runner: Gemma4Runner, canary: Bool = false) throws -> MTLComputePipelineState {
        let device = runner.q4RunnerDevice
        let library = try device.makeLibrary(source: canary ? canarySource : candidateSource, options: nil)
        guard let fn = library.makeFunction(name: canary ? canaryKernelName : kernelName) else {
            throw AttentionDiagnosticError.hardware("attention diagnostic function missing")
        }
        let desc = MTLComputePipelineDescriptor()
        desc.computeFunction = fn
        desc.label = labelPrefix + (canary ? "-canary" : "-candidate")
        let pipe = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        guard pipe.maxTotalThreadsPerThreadgroup >= threads,
              pipe.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
            throw AttentionDiagnosticError.gate(
                "attention diagnostic illegal (max threads \(pipe.maxTotalThreadsPerThreadgroup), "
                + "threadgroup memory \(pipe.staticThreadgroupMemoryLength))")
        }
        return pipe
    }

    /// Host eligibility for the candidate. Calls outside this geometry fall
    /// back to the incumbent and are counted; they never pass silently.
    public static func eligible(qHeads: Int, kvHeads: Int, headDim: Int, position: Int,
                                window: Int, maxThreads: Int) -> Bool {
        return qHeads > 0 && kvHeads > 0 && qHeads % kvHeads == 0
            && headDim > 0 && headDim <= maxHeadDim
            && position >= 0 && position < maxContext
            && window >= 0 && threads <= maxThreads
    }

    static func nextBits(_ s: inout UInt64) -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    static func unitFloat(_ s: inout UInt64) -> Float {
        Float(nextBits(&s) >> 40) / Float(1 << 24) * 2 - 1
    }

    static func sameBits(_ a: [Float], _ b: [Float]) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern { return false }
        return true
    }

    static func sharedBuffer(_ device: MTLDevice, _ a: [Float]) throws -> MTLBuffer {
        guard !a.isEmpty,
              let b = device.makeBuffer(bytes: a, length: a.count * 4, options: .storageModeShared) else {
            throw AttentionDiagnosticError.hardware("attention input buffer")
        }
        return b
    }

    /// One launch with sentinel-poisoned output plus a guard region past the
    /// last output float that must survive.
    static func run(pipe: MTLComputePipelineState, queue: MTLCommandQueue, groups: Int, threads tg: Int,
                    p: SIMD4<UInt32>, q: SIMD2<UInt32>, qb: MTLBuffer, kb: MTLBuffer, vb: MTLBuffer,
                    ob: MTLBuffer, outCount: Int) throws -> (out: [Float], gpuMicros: Double) {
        guard groups > 0, tg > 0, tg <= pipe.maxTotalThreadsPerThreadgroup, outCount > 0,
              ob.storageMode == .shared, ob.length >= (outCount + guardFloats) * 4,
              qb !== ob, kb !== ob, vb !== ob else {
            throw AttentionDiagnosticError.gate("attention run geometry or buffers invalid")
        }
        let raw = ob.contents()
        for i in 0..<(outCount + guardFloats) {
            raw.storeBytes(of: sentinelBits, toByteOffset: i * 4, as: UInt32.self)
        }
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw AttentionDiagnosticError.hardware("attention encoder")
        }
        var pv = p
        var qv = q
        enc.setComputePipelineState(pipe)
        enc.setBytes(&pv, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
        enc.setBytes(&qv, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        enc.setBuffer(qb, offset: 0, index: 2)
        enc.setBuffer(kb, offset: 0, index: 3)
        enc.setBuffer(vb, offset: 0, index: 4)
        enc.setBuffer(ob, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw AttentionDiagnosticError.hardware("attention command buffer: \(String(describing: cmd.error))")
        }
        var out = [Float](repeating: 0, count: outCount)
        for i in 0..<outCount {
            out[i] = Float(bitPattern: raw.load(fromByteOffset: i * 4, as: UInt32.self))
        }
        for i in outCount..<(outCount + guardFloats)
        where raw.load(fromByteOffset: i * 4, as: UInt32.self) != sentinelBits {
            throw AttentionDiagnosticError.gate("attention launch wrote past the output (guard \(i - outCount))")
        }
        return (out, (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6)
    }

    /// Independent FP64 stable-softmax attention oracle (bounded by the
    /// component token cases).
    static func fp64Attention(q: [Float], k: [Float], v: [Float], qHeads: Int, kvHeads: Int,
                              headDim hd: Int, tokens: Int, window: Int) -> [Double] {
        let qpos = tokens - 1
        let lo = (window != 0 && qpos + 1 > window) ? qpos + 1 - window : 0
        let group = qHeads / kvHeads
        var out = [Double](repeating: 0, count: qHeads * hd)
        var s = [Double](repeating: 0, count: tokens)
        for h in 0..<qHeads {
            let kvh = h / group
            let qBase = h * hd
            var m = -Double.infinity
            for t in lo...qpos {
                let kBase = t * kvHeads * hd + kvh * hd
                var d = 0.0
                for i in 0..<hd { d += Double(q[qBase + i]) * Double(k[kBase + i]) }
                s[t] = d
                m = max(m, d)
            }
            var l = 0.0
            for t in lo...qpos {
                s[t] = exp(s[t] - m)
                l += s[t]
            }
            for t in lo...qpos {
                let w = s[t] / l
                let vBase = t * kvHeads * hd + kvh * hd
                for i in 0..<hd { out[qBase + i] += w * Double(v[vBase + i]) }
            }
        }
        return out
    }

    /// One component case against the ACTUAL cached incumbent g4_attn.
    /// - Inputs: deterministic synthetic Q, K and V.
    /// - Query position: qpos = T - 1.
    /// - Relative metrics: max |diff| / max(1, max |baseline|).
    /// - Bit identity: reported, not required.
    public static func componentCase(runner: Gemma4Runner, candidate: MTLComputePipelineState,
                                     label: String, tokens: Int, headDim: Int, kvHeads: Int,
                                     window: Int, seed: UInt64) throws -> AttentionComponentRecord {
        guard !runner.attentionDiagnosticActive else {
            throw AttentionDiagnosticError.gate("component reference requested while the attention diagnostic is installed")
        }
        let qHeads = Int(runner.config.qHeads)
        guard tokens >= 1,
              eligible(qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, position: tokens - 1,
                       window: window, maxThreads: candidate.maxTotalThreadsPerThreadgroup) else {
            throw AttentionDiagnosticError.gate("component case \(label) is outside the validated candidate geometry")
        }
        let device = runner.q4RunnerDevice
        let queue = runner.q4RunnerQueue
        let reference = try runner.attentionIncumbentPipeline()
        guard reference !== candidate, candidate.label?.hasPrefix(labelPrefix) == true,
              candidate.device === device else {
            throw AttentionDiagnosticError.gate("attention component inputs invalid (identity, label or device)")
        }
        var state = seed
        let qCount = qHeads * headDim
        let kvCount = tokens * kvHeads * headDim
        var qs = [Float](repeating: 0, count: qCount)
        var ks = [Float](repeating: 0, count: kvCount)
        var vs = [Float](repeating: 0, count: kvCount)
        for i in 0..<qCount { qs[i] = 0.25 * unitFloat(&state) }
        for i in 0..<kvCount { ks[i] = unitFloat(&state) }
        for i in 0..<kvCount { vs[i] = unitFloat(&state) }
        let qb = try sharedBuffer(device, qs)
        let kb = try sharedBuffer(device, ks)
        let vb = try sharedBuffer(device, vs)
        guard let ob = device.makeBuffer(length: (qCount + guardFloats) * 4, options: .storageModeShared) else {
            throw AttentionDiagnosticError.hardware("attention output buffer")
        }
        let p = SIMD4<UInt32>(UInt32(tokens), UInt32(tokens - 1), UInt32(kvHeads), UInt32(headDim))
        let q = SIMD2<UInt32>(UInt32(qHeads), UInt32(window))
        let refWidth = min(qHeads, reference.maxTotalThreadsPerThreadgroup)
        let refGroups = (qHeads + refWidth - 1) / refWidth
        let r = try run(pipe: reference, queue: queue, groups: refGroups, threads: refWidth,
                        p: p, q: q, qb: qb, kb: kb, vb: vb, ob: ob, outCount: qCount)
        let c1 = try run(pipe: candidate, queue: queue, groups: qHeads, threads: threads,
                         p: p, q: q, qb: qb, kb: kb, vb: vb, ob: ob, outCount: qCount)
        let c2 = try run(pipe: candidate, queue: queue, groups: qHeads, threads: threads,
                         p: p, q: q, qb: qb, kb: kb, vb: vb, ob: ob, outCount: qCount)
        let oracle = fp64Attention(q: qs, k: ks, v: vs, qHeads: qHeads, kvHeads: kvHeads,
                                   headDim: headDim, tokens: tokens, window: window)
        var finite = true
        var maxAbsRef = 0.0
        var maxAbsOracle = 0.0
        var dCR = 0.0
        var dCO = 0.0
        var dRO = 0.0
        for i in 0..<qCount {
            let rv = Double(r.out[i])
            let cv = Double(c1.out[i])
            let ov = oracle[i]
            if !rv.isFinite || !cv.isFinite || !ov.isFinite { finite = false; continue }
            maxAbsRef = max(maxAbsRef, abs(rv))
            maxAbsOracle = max(maxAbsOracle, abs(ov))
            dCR = max(dCR, abs(cv - rv))
            dCO = max(dCO, abs(cv - ov))
            dRO = max(dRO, abs(rv - ov))
        }
        let relCR = dCR / max(1.0, maxAbsRef)
        let relCO = dCO / max(1.0, maxAbsOracle)
        let relRO = dRO / max(1.0, maxAbsOracle)
        let deterministic = sameBits(c1.out, c2.out)
        let qpos = tokens - 1
        let lo = (window != 0 && qpos + 1 > window) ? qpos + 1 - window : 0
        let pass = finite && deterministic && relCR < relBound && relCO < relBound && relRO < relBound
        return AttentionComponentRecord(
            label: label, tokens: tokens, queryPosition: qpos, attended: qpos + 1 - lo,
            headDim: headDim, qHeads: qHeads, kvHeads: kvHeads, window: window,
            bitIdenticalToReference: sameBits(c1.out, r.out),
            maxAbsCandidateVsReference: dCR, relCandidateVsReference: relCR,
            maxAbsCandidateVsFP64: dCO, relCandidateVsFP64: relCO,
            maxAbsReferenceVsFP64: dRO, relReferenceVsFP64: relRO,
            maxAbsReferenceOutput: maxAbsRef, allFinite: finite, deterministic: deterministic,
            referenceGpuMicros: r.gpuMicros, candidateGpuMicros: c1.gpuMicros, pass: pass)
    }

    /// Capacity guard at T = maxContext + 1 in global geometry. Two checks:
    /// - Host eligibility must reject the call.
    /// - A direct candidate launch must write NaN to every output float. The
    ///   NaN must be written by the kernel, which is distinguished from an
    ///   unwritten sentinel.
    public static func capacityGuardCase(runner: Gemma4Runner,
                                         candidate: MTLComputePipelineState) throws -> (ineligible: Bool, wroteNaN: Bool) {
        let qHeads = Int(runner.config.qHeads)
        let headDim = globalHeadDim
        let kvHeads = globalKVHeads
        let tokens = maxContext + 1
        guard kvHeads > 0, qHeads > 0, qHeads % kvHeads == 0 else {
            throw AttentionDiagnosticError.gate("capacity guard geometry invalid for this config")
        }
        let ineligible = !eligible(qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
                                   position: tokens - 1, window: 0,
                                   maxThreads: candidate.maxTotalThreadsPerThreadgroup)
        let device = runner.q4RunnerDevice
        let qCount = qHeads * headDim
        let qb = try sharedBuffer(device, [Float](repeating: 0.01, count: qCount))
        let kv = [Float](repeating: 0.01, count: tokens * kvHeads * headDim)
        let kb = try sharedBuffer(device, kv)
        let vb = try sharedBuffer(device, kv)
        guard let ob = device.makeBuffer(length: (qCount + guardFloats) * 4, options: .storageModeShared) else {
            throw AttentionDiagnosticError.hardware("attention output buffer")
        }
        let c = try run(pipe: candidate, queue: runner.q4RunnerQueue, groups: qHeads, threads: threads,
                        p: SIMD4<UInt32>(UInt32(tokens), UInt32(tokens - 1), UInt32(kvHeads), UInt32(headDim)),
                        q: SIMD2<UInt32>(UInt32(qHeads), 0),
                        qb: qb, kb: kb, vb: vb, ob: ob, outCount: qCount)
        let wroteNaN = c.out.allSatisfy { !$0.isFinite && $0.bitPattern != sentinelBits }
        return (ineligible, wroteNaN)
    }

    /// Synthetic component matrix.
    /// - Token counts: T in `tokenCases`.
    /// - Local geometry: head dim 256, 8 KV heads, the actual config sliding
    ///   window.
    /// - Global geometry: head dim 512, 2 KV heads, window 0.
    /// - Query heads come from the actual config.
    /// - Stops at the first failing case.
    public static func syntheticComponent(runner: Gemma4Runner,
                                          candidate: MTLComputePipelineState) throws -> [AttentionComponentRecord] {
        let local = (name: "local", headDim: localHeadDim, kvHeads: localKVHeads,
                     window: Int(runner.config.slidingWindow))
        let global = (name: "global", headDim: globalHeadDim, kvHeads: globalKVHeads, window: 0)
        var out: [AttentionComponentRecord] = []
        var k: UInt64 = 0
        for g in [local, global] {
            for t in tokenCases {
                let rec = try componentCase(runner: runner, candidate: candidate,
                                            label: "\(g.name)_hd\(g.headDim)_kv\(g.kvHeads)_w\(g.window)_T\(t)",
                                            tokens: t, headDim: g.headDim, kvHeads: g.kvHeads,
                                            window: g.window, seed: 0xA77E_0000 &+ k)
                out.append(rec)
                k += 1
                if !rec.pass { return out }
            }
        }
        return out
    }
}
