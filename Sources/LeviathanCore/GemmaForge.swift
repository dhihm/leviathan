// LeviathanForge pipeline for the Gemma 4 quantized graph: captured
// model.py graph → validation against the real GGUF → bounded legal
// candidates → correctness gate (graph executor vs the independently
// validated per-kernel reference path, and raw-logit tolerance) →
// measured GPU-time selection → reusable plan cache keyed on graph +
// GGUF content + hardware + toolchain + workload. No agent, no LLM at
// runtime; anything outside the supported op set is rejected at
// validation. The graph executor drives the runner's kernel primitives
// from the validated IR — it is not a call into the fixed hand-written
// forward.

import Foundation
import CryptoKit
import Metal

public enum GemmaForgeError: Error, CustomStringConvertible, Equatable {
    case invalidGraph(String)
    case noValidCandidate([String])
    case invalidPlan(String)
    case executionFailed(String)

    public var description: String {
        switch self {
        case .invalidGraph(let m): return "gemma forge graph invalid: \(m)"
        case .noValidCandidate(let r): return "no valid candidate: \(r)"
        case .invalidPlan(let m): return "gemma plan invalid: \(m)"
        case .executionFailed(let m): return "execution failed: \(m)"
        }
    }
}

public struct GemmaForgeCandidate: Codable, Equatable, Hashable, Sendable {
    /// Batch GPU-only kernel runs into one command buffer per run
    /// (numerically identical order; only submission granularity changes).
    public let segmentBatching: Bool
    /// Compute the full top-5 logit diagnostic (CPU cost) or argmax only.
    public let top5Diagnostic: Bool
    /// Dispatch the tiled cooperative Q4_0 matvec kernel (hardware-derived
    /// threadgroup width). NOT bit-identical: the threadgroup reduction
    /// reassociates the addition order — correctness is gated against the
    /// FP64 oracle and full-logit/greedy equality before selection.
    public let matvecTiledQ4: Bool
    /// Tiled cooperative Q6_K matvec (tied embedding / LM head). Same
    /// non-bit-identical caveat and gates as matvecTiledQ4.
    public let matvecTiledQ6: Bool

    public init(segmentBatching: Bool, top5Diagnostic: Bool,
                matvecTiledQ4: Bool = false, matvecTiledQ6: Bool = false) {
        self.segmentBatching = segmentBatching
        self.top5Diagnostic = top5Diagnostic
        self.matvecTiledQ4 = matvecTiledQ4
        self.matvecTiledQ6 = matvecTiledQ6
    }
}

public struct GemmaForgeCandidateResult: Codable, Equatable, Sendable {
    public let candidate: GemmaForgeCandidate
    public let correct: Bool
    /// SELECTION OBJECTIVE: median end-to-end wall time per timed token
    /// (forward + CPU router + argmax + necessary readback; the post-hoc
    /// correctness comparison is excluded from the timed region). GPU-only
    /// time misses CPU launch/driver overhead — the dominant cost of the
    /// per-kernel submission path — so wall time selects.
    public let medianWallMicrosPerToken: Double?
    /// Reported separately: median GPU-busy time per timed token.
    public let medianGpuMicrosPerToken: Double?
    /// Median dispatched kernels per timed token (kernels, not buffers).
    public let kernelsPerToken: Double?
    /// Median submitted command buffers per timed token — makes the
    /// batching effect of a candidate directly observable.
    public let buffersPerToken: Double?
    public let note: String

    init(candidate: GemmaForgeCandidate, correct: Bool,
         medianWallMicrosPerToken: Double? = nil,
         medianGpuMicrosPerToken: Double?,
         kernelsPerToken: Double? = nil,
         buffersPerToken: Double? = nil, note: String) {
        self.candidate = candidate
        self.correct = correct
        self.medianWallMicrosPerToken = medianWallMicrosPerToken
        self.medianGpuMicrosPerToken = medianGpuMicrosPerToken
        self.kernelsPerToken = kernelsPerToken
        self.buffersPerToken = buffersPerToken
        self.note = note
    }
}

public struct GemmaForgePlan: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let graphDigest: String
    public let ggufDigest: String
    /// Pass-engine identity (graph-executor version + compiled Metal
    /// pipeline set): changing kernels or the executor invalidates the
    /// cache even without a toolchain change.
    public let engineDigest: String
    public let hardware: ForgeHardwareProfile
    public let toolchain: String
    public let workloadDigest: String
    public let promptTokenCount: Int
    public let decodeTokens: Int
    public let selected: GemmaForgeCandidate
    public let selectedMedianWallMicrosPerToken: Double
    public let selectedMedianGpuMicrosPerToken: Double
    public let results: [GemmaForgeCandidateResult]
    /// The FULL lowered executable schedule (typed SSA instructions with
    /// operands and buffer-binding value names) saved with the plan —
    /// the runtime executes these artifact operands, not a re-derivation.
    public let schedule: GemmaSSAIR?
    /// SHA-256 of the canonical schedule JSON (must equal
    /// GemmaSSA.digest(lower(graph)) on reuse).
    public let scheduleDigest: String
    public let planDigest: String

    init(schemaVersion: Int, graphDigest: String, ggufDigest: String,
         engineDigest: String, hardware: ForgeHardwareProfile,
         toolchain: String, workloadDigest: String,
         promptTokenCount: Int, decodeTokens: Int,
         selected: GemmaForgeCandidate,
         selectedMedianWallMicrosPerToken: Double,
         selectedMedianGpuMicrosPerToken: Double,
         results: [GemmaForgeCandidateResult], schedule: GemmaSSAIR?,
         scheduleDigest: String, planDigest: String) {
        self.schemaVersion = schemaVersion
        self.graphDigest = graphDigest
        self.ggufDigest = ggufDigest
        self.engineDigest = engineDigest
        self.hardware = hardware
        self.toolchain = toolchain
        self.workloadDigest = workloadDigest
        self.promptTokenCount = promptTokenCount
        self.decodeTokens = decodeTokens
        self.selected = selected
        self.selectedMedianWallMicrosPerToken = selectedMedianWallMicrosPerToken
        self.selectedMedianGpuMicrosPerToken = selectedMedianGpuMicrosPerToken
        self.results = results
        self.schedule = schedule
        self.scheduleDigest = scheduleDigest
        self.planDigest = planDigest
    }
}

public enum GemmaForge {

    public static let schemaVersion = 1
    public static let warmupTokens = 1
    public static let timedTokens = 3
    /// Per-candidate GPU-time budget (micros) across all timed tokens.
    public static let budgetMicros: Double = 120_000_000

    /// Bounded legal candidate set for this graph. Segment batching is
    /// legal everywhere (CPU readbacks flush the open segment); the top-5
    /// diagnostic is a reporting choice, both dimensions are gated.
    public static func candidates() -> [GemmaForgeCandidate] {
        // Only real execution-level optimizations; the top-5 logit
        // diagnostic is reporting, not a compiler pass, and is excluded.
        [
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false),
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledQ4: true),
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true),
            // Q6_K tiling (tied embedding / LM head) stacked on the
            // measured winners so far.
            GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                matvecTiledQ4: true, matvecTiledQ6: true),
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false,
                                matvecTiledQ4: true, matvecTiledQ6: true),
        ]
    }

    /// Streaming SHA-256 of a large file (the 14.4 GB GGUF), read in
    /// bounded chunks.
    public static func fileDigest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = CryptoKit.SHA256()
        let chunk = 1 << 23
        while true {
            let data = try handle.read(upToCount: chunk)
            guard let data, !data.isEmpty else { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - graph executor

    /// Executes the validated graph IR on the runner's kernel primitives.
    /// The per-kernel mode is bit-identical in dispatch order to the
    /// reference path; segment batching only groups submissions.
    public static func executeGraph(_ graph: GemmaGraphIR, runner: Gemma4Runner,
                                    tokenIds: [Int],
                                    candidate: GemmaForgeCandidate,
                                    schedule: GemmaSSAIR? = nil) throws
            -> (argmax: Int, logits: [Float], gpuMicros: Double, kernels: Int,
                buffers: Int) {
        guard !tokenIds.isEmpty else { throw GemmaForgeError.executionFailed("empty tokens") }
        // Untrusted graphs never reach execution: full validation against
        // the actual GGUF + config first (unknown ops/layers/tensors are
        // rejected before any raw cache indexing).
        try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
        // Lower + validate the typed SSA schedule ONCE per request — or
        // execute the plan artifact's embedded schedule operands directly
        // (dataflow-validated here against the same config).
        let ssa: GemmaSSAIR
        if let schedule {
            try GemmaSSA.validate(schedule, config: graph.config)
            // Public schedule execution additionally requires EXACT
            // equality with the canonical lowering of THIS validated
            // graph (tensor bindings and geometry checked against the
            // mapped GGUF via GemmaGraph.validate above) — a hand-built
            // look-alike schedule is rejected before any GPU work.
            let canonical = try GemmaSSA.lower(graph)
            guard schedule == canonical else {
                throw GemmaForgeError.invalidGraph(
                    "schedule is not the canonical lowering of this graph")
            }
            ssa = schedule
        } else {
            ssa = try GemmaSSA.lower(graph)
        }
        let gpuBefore = runner.gpuMicrosTotal
        let dispatchBefore = runner.dispatchCountTotal
        let buffersBefore = runner.submittedBuffers
        runner.reset()
        var argmax = -1
        var logits: [Float] = []
        do {
            for token in tokenIds {
                let pos = runner.contextPosition()
                guard token >= 0, token < runner.config.vocabSize,
                      pos < runner.maxContext else {
                    throw GemmaForgeError.executionFailed("token/position out of bounds")
                }
                argmax = try executeToken(ssa, runner, token: token, position: pos,
                                          candidate: candidate)
                runner.advancePosition()
                logits = try runner.read(runner.bufLogits, runner.config.vocabSize)
            }
        } catch {
            // Cleanup without losing the error: discard any uncommitted
            // deferred encoder, disable auto-segment, and reset the context
            // so no partial KV state survives a failed run.
            runner.discardOpenSegment()
            runner.reset()
            throw error
        }
        return (argmax, logits, runner.gpuMicrosTotal - gpuBefore,
                runner.dispatchCountTotal - dispatchBefore,
                runner.submittedBuffers - buffersBefore)
    }

    /// Executes ONE token continuing the runner's current KV context at
    /// position consumedPosition (no reset) — the real decode workload.
    public static func executeGraphContinue(_ graph: GemmaGraphIR,
                                            runner: Gemma4Runner, token: Int,
                                            candidate: GemmaForgeCandidate,
                                            schedule: GemmaSSAIR? = nil) throws
            -> (argmax: Int, logits: [Float], gpuMicros: Double, kernels: Int,
                buffers: Int) {
        try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
        guard token >= 0, token < runner.config.vocabSize,
              runner.contextPosition() < runner.maxContext else {
            throw GemmaForgeError.executionFailed("token/position out of bounds")
        }
        let ssa: GemmaSSAIR
        if let schedule {
            try GemmaSSA.validate(schedule, config: graph.config)
            let canonical = try GemmaSSA.lower(graph)
            guard schedule == canonical else {
                throw GemmaForgeError.invalidGraph(
                    "schedule is not the canonical lowering of this graph")
            }
            ssa = schedule
        } else {
            ssa = try GemmaSSA.lower(graph)
        }
        let gpuBefore = runner.gpuMicrosTotal
        let dispatchBefore = runner.dispatchCountTotal
        let buffersBefore = runner.submittedBuffers
        let position = runner.contextPosition()
        do {
            let argmax = try executeToken(ssa, runner, token: token, position: position,
                                          candidate: candidate)
            runner.advancePosition()
            let logits = try runner.read(runner.bufLogits, runner.config.vocabSize)
            return (argmax, logits, runner.gpuMicrosTotal - gpuBefore,
                    runner.dispatchCountTotal - dispatchBefore,
                    runner.submittedBuffers - buffersBefore)
        } catch {
            runner.discardOpenSegment()
            runner.reset()
            throw error
        }
    }

    private static func executeToken(_ ssa: GemmaSSAIR, _ runner: Gemma4Runner,
                                     token: Int, position: Int,
                                     candidate: GemmaForgeCandidate) throws -> Int {
        let config = runner.config
        // The schedule was lowered (and validated) ONCE by the caller; the
        // executor dispatches kernels from the SSA OPERANDS. Buffers are
        // resolved per operand kind (vector values only — KV/routing state
        // never touches this path), in-place ops are validated by the
        // lowering to alias their input's buffer, and the final logits are
        // read from the FINAL ARTIFACT OUTPUT VALUE, not a fixed buffer.
        var byID = [String: GemmaSSAIR.Value]()
        for v in ssa.values { byID[v.id] = v }
        func vb(_ id: String) throws -> MTLBuffer {
            guard let value = byID[id], value.kind == .vector else {
                throw GemmaForgeError.executionFailed(
                    "operand \(id) is not a buffer-resident vector value")
            }
            switch value.name {
            case "x": return runner.bufX
            case "h": return runner.bufH
            case "t1": return runner.bufT1
            case "t2": return runner.bufT2
            case "t3": return runner.bufT3
            case "kv": return runner.bufKV
            case "gate", "up": return runner.bufGU   // offset slices, see matvec
            case "act": return runner.bufAct
            case "shared": return runner.bufShared
            case "router": return runner.bufRouter
            case "logits": return runner.bufLogits
            default:
                throw GemmaForgeError.executionFailed(
                    "no physical buffer binding for value \(value.name)")
            }
        }
        func a(_ inst: GemmaSSAIR.Instruction, _ key: String) -> Int {
            inst.attrs[key] ?? 0
        }
        // Batch GPU-only runs into one command buffer per run across ops;
        // CPU readback (read/write) flushes the open segment. Kernel order
        // is identical in both modes.
        runner.autoSegment = candidate.segmentBatching
        runner.q4TiledKernel = candidate.matvecTiledQ4
        runner.q6TiledKernel = candidate.matvecTiledQ6
        defer {
            runner.autoSegment = false
            runner.q4TiledKernel = false
            runner.q6TiledKernel = false
        }
        var routerState = [Int: (topIdx: [Int], weights: [Float])]()

        for inst in ssa.instructions {
            switch inst.op {
            case "embed_row":
                guard inst.tensors.first == "token_embd.weight" else {
                    throw GemmaForgeError.executionFailed(
                        "embed tensor binding is not the embedding table")
                }
                try runner.decodeEmbedding(token)
                let emb = try runner.read(runner.bufEmbed, runner.hidden)
                guard emb.allSatisfy({ $0.isFinite }) else {
                    throw GemmaForgeError.executionFailed("embedding nonfinite")
                }
                try runner.write(emb, to: try vb(inst.output))
            case "rmsnorm":
                // n is PER VECTOR (the primitive's contract): the validated
                // width for vectorized norms, the total length otherwise.
                let vectors = max(1, a(inst, "vectors"))
                let n = vectors == 1 ? byID[inst.output]!.length : a(inst, "width")
                try runner.rmsnorm(x: try vb(inst.inputs[0]), y: try vb(inst.output),
                                   n: n, vectors: vectors,
                                   weightTensor: inst.tensors.first)
            case "matvec":
                // Physical capacity includes the slice offset (shared_up
                // writes the upper half of the fused gate|up buffer); the
                // LOGICAL output length stays the slice length.
                let rows = a(inst, "rows")
                try runner.matvec(tensor: inst.tensors[0],
                                  x: try vb(inst.inputs[0]),
                                  xCount: byID[inst.inputs[0]]!.length,
                                  y: try vb(inst.output),
                                  yCapacity: a(inst, "offset") + rows,
                                  rows: rows,
                                  yByteOffset: a(inst, "offset") * 4)
            case "rope":
                // Explicit validated phase operand from the lowering.
                try runner.rope(x: try vb(inst.inputs[0]),
                                heads: a(inst, "heads"),
                                half: a(inst, "half"), position: position,
                                freqs: a(inst, "global") == 1
                                    ? runner.globalFreqBuffer
                                    : runner.localFreqBuffer)
            case "copy":
                let src = try runner.read(try vb(inst.inputs[0]),
                                          byID[inst.inputs[0]]!.length)
                try runner.write(src, to: try vb(inst.output))
            case "axpy":
                // inputs[0] is the accumulator (x) — validated to alias the
                // output — inputs[1] the addend (h).
                try runner.axpy(x: try vb(inst.inputs[1]), y: try vb(inst.output),
                                c: 1, n: byID[inst.output]!.length)
            case "add":
                try runner.add(a: try vb(inst.inputs[0]), b: try vb(inst.inputs[1]),
                               y: try vb(inst.output), n: byID[inst.output]!.length)
            case "scale":
                // Guarded layer geometry — a malformed artifact cannot reach
                // raw per-layer vectors.
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("scale layer invalid")
                }
                if inst.attrs["routerScale"] == 1 {
                    let outBuf = try vb(inst.output)
                    let n = byID[inst.output]!.length
                    try runner.scaleInPlace(x: outBuf,
                                            c: Float(1.0 / sqrt(Double(n))), n: n)
                    var routerIn = try runner.read(outBuf, n)
                    let scale = runner.routerScales[layer]
                    for i in 0..<n { routerIn[i] *= scale[i] }
                    try runner.write(routerIn, to: outBuf)
                } else {
                    try runner.scaleInPlace(x: try vb(inst.output),
                                            c: runner.layerScalars[layer],
                                            n: byID[inst.output]!.length)
                }
            case "gelu_up":
                // gate/up alias the fused gate|up buffer at their matvec
                // offsets; the kernel consumes the fused pair in place.
                try runner.geluUp(gu: try vb(inst.inputs[0]), y: try vb(inst.output),
                                  n: a(inst, "width"))
            case "cache_append":
                // Guarded geometry: layer bounds and checked products — a
                // malformed artifact cannot index raw cache storage.
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount,
                      let kvHeads = inst.attrs["kvHeads"],
                      let headDim = inst.attrs["headDim"],
                      kvHeads > 0, headDim > 0,
                      position >= 0 else {
                    throw GemmaForgeError.executionFailed("cache_append geometry invalid")
                }
                let kvWidth = byID[inst.inputs[0]]!.length
                guard kvWidth == kvHeads * headDim,
                      kvWidth == byID[inst.inputs[1]]!.length else {
                    throw GemmaForgeError.executionFailed("cache_append width mismatch")
                }
                let (byteWidth, ov1) = kvWidth.multipliedReportingOverflow(by: 4)
                let (byteOffset, ov2) = byteWidth.multipliedReportingOverflow(by: position)
                guard !ov1, !ov2,
                      byteOffset + byteWidth <= runner.kCaches[layer].length,
                      byteOffset + byteWidth <= runner.vCaches[layer].length else {
                    throw GemmaForgeError.executionFailed("cache_append out of bounds")
                }
                try runner.flushOpenSegment()
                let k = try runner.read(try vb(inst.inputs[0]), kvWidth)
                let v = try runner.read(try vb(inst.inputs[1]), kvWidth)
                memcpy(runner.kCaches[layer].contents() + byteOffset, k, byteWidth)
                memcpy(runner.vCaches[layer].contents() + byteOffset, v, byteWidth)
            case "attention":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("attention layer invalid")
                }
                var p = SIMD4<UInt32>(UInt32(position + 1), UInt32(position),
                                      UInt32(a(inst, "kvHeads")), UInt32(a(inst, "headDim")))
                var q = SIMD2<UInt32>(UInt32(a(inst, "heads")), UInt32(a(inst, "window")))
                let qBuf = try vb(inst.inputs[0])
                let outBuf = try vb(inst.output)
                try runner.dispatch("g4_attn", grid: a(inst, "heads")) { enc in
                    enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
                    enc.setBytes(&q, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                    enc.setBuffer(qBuf, offset: 0, index: 2)
                    enc.setBuffer(runner.kCaches[layer], offset: 0, index: 3)
                    enc.setBuffer(runner.vCaches[layer], offset: 0, index: 4)
                    enc.setBuffer(outBuf, offset: 0, index: 5)
                }
            case "router_topk":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount else {
                    throw GemmaForgeError.executionFailed("router layer invalid")
                }
                try runner.flushOpenSegment()
                let scores = try runner.read(try vb(inst.inputs[0]), config.expertCount)
                guard scores.allSatisfy({ $0.isFinite }) else {
                    throw GemmaForgeError.executionFailed("router scores nonfinite")
                }
                let maxScore = scores.max() ?? 0
                let exps = scores.map { exp(Double($0 - maxScore)) }
                let sumExp = exps.reduce(0, +)
                let probs = exps.map { $0 / sumExp }
                var idx = Array(0..<config.expertCount)
                idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
                let topIdx = Array(idx.prefix(a(inst, "topK")))
                let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
                let scales = runner.expertScales[layer]
                let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }
                routerState[layer] = (topIdx, weights)
            case "expert_matvec_chain":
                guard let layer = inst.layer, layer >= 0,
                      layer < config.layerCount,
                      let gateUp = runner.ref[inst.tensors[0]],
                      let down = runner.ref[inst.tensors[1]],
                      let (topIdx, weights) = routerState[layer] else {
                    throw GemmaForgeError.executionFailed("missing expert tensors/routing")
                }
                let e = a(inst, "intermediate")
                let guStride = gateUp.outDim * (gateUp.inDim / 32) * 18
                let dnStride = down.outDim * (down.inDim / 32) * 18
                try runner.flushOpenSegment()
                let outBuf = try vb(inst.output)
                let n = byID[inst.output]!.length
                memset(outBuf.contents(), 0, n * 4)
                // DEDICATED expert scratch, physically separate from all
                // SSA-bound buffers — internal effects of this compound op
                // can never clobber live SSA values (gate/up/act/t2).
                let inBuf = try vb(inst.inputs[0])
                let inLen = byID[inst.inputs[0]]!.length
                for (slot, expert) in topIdx.enumerated() {
                    try runner.matvec(tensor: inst.tensors[0],
                                      extraOffset: expert * guStride,
                                      x: inBuf, xCount: inLen,
                                      y: runner.bufExpertGU,
                                      yCapacity: 2 * config.sharedIntermediate)
                    try runner.geluUp(gu: runner.bufExpertGU, y: runner.bufExpertAct,
                                      n: e)
                    try runner.matvec(tensor: inst.tensors[1],
                                      extraOffset: expert * dnStride,
                                      x: runner.bufExpertAct, xCount: e,
                                      y: runner.bufExpertAcc, yCapacity: n)
                    try runner.axpy(x: runner.bufExpertAcc, y: outBuf,
                                    c: weights[slot], n: n)
                }
            case "softcap":
                var cap = Float(a(inst, "cap"))
                var count = UInt32(byID[inst.output]!.length)
                let outBuf = try vb(inst.output)
                try runner.dispatch("g4_softcap", grid: byID[inst.output]!.length) { enc in
                    enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
                    enc.setBuffer(outBuf, offset: 0, index: 1)
                    enc.setBytes(&cap, length: MemoryLayout<Float>.stride, index: 2)
                }
            default:
                throw GemmaForgeError.invalidGraph(
                    "unsupported ssa op at execution: \(inst.op)")
            }
        }
        // Final logits come from the FINAL ARTIFACT OUTPUT VALUE.
        guard let finalInst = ssa.instructions.last,
              finalInst.op == "softcap",
              let finalValue = byID[finalInst.output],
              finalValue.name == "logits",
              finalValue.length == config.vocabSize else {
            throw GemmaForgeError.executionFailed("schedule terminal is not logits")
        }
        let logits = try runner.read(try vb(finalInst.output), finalValue.length)
        guard logits.allSatisfy({ $0.isFinite }) else {
            throw GemmaForgeError.executionFailed("nonfinite logits")
        }
        if candidate.top5Diagnostic {
            _ = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
        }
        return logits.enumerated().max { $0.element < $1.element }!.offset
    }

    // MARK: - pipeline

    /// Full pipeline: validate the captured graph, gate candidates
    /// against the reference per-kernel executor AND the runner's own
    /// validated forward (raw logits), measure GPU time per token, and
    /// select the best valid candidate.
    public static func forge(graph: GemmaGraphIR, runner: Gemma4Runner,
                             ggufURL: URL, prompt: [Int], decodeTokens: Int,
                             hardware: ForgeHardwareProfile) throws
            -> (plan: GemmaForgePlan, results: [GemmaForgeCandidateResult],
                selected: GemmaForgeCandidate) {
        guard prompt.allSatisfy({ $0 >= 0 && $0 < runner.config.vocabSize }),
              !prompt.isEmpty,
              prompt.count + warmupTokens + timedTokens <= runner.maxContext,
              decodeTokens == warmupTokens + timedTokens else {
            throw GemmaForgeError.invalidGraph("bad workload tokens/size")
        }
        // Real identity inputs: the graph digest and the GGUF content
        // digest are computed from actual bytes — no placeholder identity.
        // The digest URL must BE the file the runner mapped (byteCount is
        // also cross-checked), so an unrelated file is rejected rather
        // than silently hashed.
        guard let mappedURL = runner.ggufFile.sourceURL, mappedURL == ggufURL,
              runner.ggufFile.byteCount == (try? URL(fileURLWithPath: ggufURL.path)
                    .resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1) else {
            throw GemmaForgeError.invalidPlan(
                "forge URL is not the runner's mapped GGUF file")
        }
        let graphDigest = try GemmaGraph.digest(graph)
        let ggufDigest = try fileDigest(ggufURL)
        let engineDigest = Self.engineDigest(try runner.engineIdentity())
        let reference = GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)
        // Reference gate 1: graph executor (per-kernel) must agree with
        // the runner's independently validated forward on raw logits and
        // greedy argmax for the workload prompt.
        runner.reset()
        let refEvidence = try runner.forward(tokenIds: prompt)
        // Independent FULL raw-logit reference captured immediately after
        // the validated runner forward, before any graph execution.
        let refLogitsAll = try runner.read(runner.bufLogits,
                                           runner.config.vocabSize)
        let (refArgmax, refLogits, _, _, _) = try executeGraph(
            graph, runner: runner, tokenIds: prompt, candidate: reference)
        guard refArgmax == refEvidence.argmaxToken else {
            throw GemmaForgeError.noValidCandidate([
                "graph executor argmax \(refArgmax) != reference \(refEvidence.argmaxToken)"])
        }
        // Graph executor vs independent reference over ALL logits.
        guard refLogits.count == refLogitsAll.count,
              refLogits.allSatisfy({ $0.isFinite }),
              refLogitsAll.allSatisfy({ $0.isFinite }) else {
            throw GemmaForgeError.noValidCandidate(["logit count/finiteness gate failed"])
        }
        var refWorst = 0.0
        for i in 0..<refLogits.count {
            refWorst = max(refWorst, abs(Double(refLogits[i] - refLogitsAll[i]))
                           / max(1.0, abs(Double(refLogitsAll[i]))))
        }
        guard refWorst < 1e-4 else {
            throw GemmaForgeError.noValidCandidate([
                String(format: "graph logits vs runner reference worst rel %.3e", refWorst)])
        }

        // Reference greedy decode sequence (warmup + timed tokens)
        // captured ONCE with the reference candidate at true positions.
        guard prompt.count + warmupTokens + timedTokens <= runner.maxContext,
              decodeTokens == warmupTokens + timedTokens else {
            throw GemmaForgeError.invalidGraph(
                "workload exceeds context or decodeTokens != warmup+timed")
        }
        // INDEPENDENT reference: the validated runner.forward itself,
        // continuing the prompt context; per-step raw logits captured.
        runner.reset()
        let prefillEvidence = try runner.forward(tokenIds: prompt)
        guard prefillEvidence.argmaxToken == refArgmax else {
            throw GemmaForgeError.noValidCandidate(["independent prefill argmax mismatch"])
        }
        var refDecodeSeq = [Int]()
        var refDecodeLogits = [[Float]]()
        var prevTok = prefillEvidence.argmaxToken
        for _ in 0..<(warmupTokens + timedTokens) {
            let ev = try runner.forward(tokenIds: [prevTok])
            refDecodeSeq.append(ev.argmaxToken)
            refDecodeLogits.append(try runner.read(runner.bufLogits,
                                                   runner.config.vocabSize))
            prevTok = ev.argmaxToken
        }
        var results: [GemmaForgeCandidateResult] = []
        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            return s.count.isMultiple(of: 2) ? (s[s.count/2 - 1] + s[s.count/2]) / 2
                                             : s[s.count/2]
        }
        struct Sample { var wall: Double; var gpu: Double; var kernels: Double; var buffers: Double }
        var best: (candidate: GemmaForgeCandidate, wall: Double)?
        outer: for candidate in candidates() {
            // Correctness gate: raw logits within 1e-4 rel of the
            // reference executor and identical greedy argmax.
            let (argmax, logits, _, _, _) = try executeGraph(
                graph, runner: runner, tokenIds: prompt, candidate: candidate)
            guard argmax == refArgmax else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil, note: "argmax mismatch"))
                continue
            }
            guard logits.count == refLogitsAll.count,
                  logits.allSatisfy({ $0.isFinite }) else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil, note: "logit count/finiteness gate failed"))
                continue
            }
            var worst = 0.0
            for i in 0..<logits.count {
                worst = max(worst, abs(Double(logits[i] - refLogitsAll[i]))
                            / max(1.0, abs(Double(refLogitsAll[i]))))
            }
            guard worst < 1e-4 else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: false,
                    medianGpuMicrosPerToken: nil,
                    note: String(format: "logits outside 1e-4 rel (worst %.3e)", worst)))
                continue
            }
            // Measured selection. Objective: end-to-end per-token WALL
            // time (the timed region is the executeGraphContinue call —
            // forward, CPU router work, argmax, and the necessary logits
            // readback; the post-hoc comparison against the reference is
            // outside the timer). GPU-busy time is recorded separately.
            // Every candidate replays the identical reference greedy
            // sequence at the true decode positions on a fresh prompt
            // context, gated per step (exact next token + full logits).
            _ = try executeGraph(graph, runner: runner, tokenIds: prompt,
                                 candidate: candidate)
            var prev = refArgmax
            for w in 0..<warmupTokens {
                let (n, wl, _, _, _) = try executeGraphContinue(
                    graph, runner: runner, token: prev, candidate: candidate)
                let want = refDecodeLogits[w]
                guard n == refDecodeSeq[w],
                      wl.count == want.count, wl.allSatisfy({ $0.isFinite }),
                      zip(wl, want).allSatisfy({
                          abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                      }) else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil,
                        note: "warmup step \(w) outside reference"))
                    continue outer
                }
                prev = n
            }
            let replay = refDecodeSeq
            var samples: [Sample] = []
            var spent: Double = 0
            for run in 0..<timedTokens {
                let tok = replay.indices.contains(run) ? replay[run] : prev
                let t0 = DispatchTime.now()
                let (next, stepLogits, gpu, kernels, buffers) = try executeGraphContinue(
                    graph, runner: runner, token: tok, candidate: candidate)
                let wall = Double(DispatchTime.now().uptimeNanoseconds
                                  - t0.uptimeNanoseconds) / 1000
                // Per-step gate vs the independent reference logits —
                // outside the timed region.
                let want = refDecodeLogits[warmupTokens + run]
                guard next == refDecodeSeq[warmupTokens + run],
                      stepLogits.count == want.count,
                      stepLogits.allSatisfy({ $0.isFinite }),
                      zip(stepLogits, want).allSatisfy({
                          abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                      }) else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil,
                        note: "decode step \(run) outside 1e-4 of reference"))
                    continue outer
                }
                prev = next
                guard gpu.isFinite, gpu > 0, wall.isFinite, wall > 0 else {
                    results.append(GemmaForgeCandidateResult(
                        candidate: candidate, correct: false,
                        medianGpuMicrosPerToken: nil, note: "timing unavailable"))
                    continue outer
                }
                samples.append(Sample(wall: wall, gpu: gpu,
                                      kernels: Double(kernels), buffers: Double(buffers)))
                spent += gpu
                if spent > budgetMicros { break }
            }
            guard !samples.isEmpty else {
                results.append(GemmaForgeCandidateResult(
                    candidate: candidate, correct: true,
                    medianGpuMicrosPerToken: nil, note: "budget exhausted"))
                continue
            }
            let wallMed = median(samples.map(\.wall))
            let gpuMed = median(samples.map(\.gpu))
            let note = String(format: "valid; logits rel %.1e; %d timed tokens; wall %.0f us, gpu %.0f us, %.0f kernels, %.0f buffers/token",
                              worst, samples.count, wallMed, gpuMed,
                              median(samples.map(\.kernels)), median(samples.map(\.buffers)))
            results.append(GemmaForgeCandidateResult(
                candidate: candidate, correct: true,
                medianWallMicrosPerToken: wallMed,
                medianGpuMicrosPerToken: gpuMed,
                kernelsPerToken: median(samples.map(\.kernels)),
                buffersPerToken: median(samples.map(\.buffers)), note: note))
            if best == nil || wallMed < best!.wall {
                best = (candidate, wallMed)
            }
        }
        guard let winner = best, !results.isEmpty else {
            throw GemmaForgeError.noValidCandidate(results.map { "\($0.candidate): \($0.note)" })
        }
        // Baseline fallback with a noise threshold: the per-kernel
        // baseline is always in the measured set; an "optimized" winner
        // is selected only if it beats the baseline wall median by more
        // than noiseRelWall — negligible noise must not force an
        // optimization. Otherwise the baseline itself is selected.
        var selected = winner
        if winner.candidate.segmentBatching {
            if let base = results.first(where: {
                $0.correct && !$0.candidate.segmentBatching
                    && $0.medianWallMicrosPerToken != nil }) {
                if winner.wall > base.medianWallMicrosPerToken! * (1 - Self.noiseRelWall) {
                    selected = (candidate: base.candidate,
                                wall: base.medianWallMicrosPerToken!)
                }
            }
        }
        let selectedResult = results.first {
            $0.candidate == selected.candidate && $0.correct
        }!
        _ = decodeTokens
        let plan = try makePlan(graph: graph, graphDigest: graphDigest,
                                ggufDigest: ggufDigest, engineDigest: engineDigest,
                                hardware: hardware,
                                prompt: prompt, decodeTokens: decodeTokens,
                                selected: selected.candidate,
                                medianWall: selectedResult.medianWallMicrosPerToken!,
                                medianGpu: selectedResult.medianGpuMicrosPerToken!,
                                results: results)
        return (plan, results, selected.candidate)
    }

    /// Relative wall-time improvement a candidate must exceed over the
    /// baseline to be selected — a CHOSEN policy threshold (2%), not an
    /// empirically calibrated noise floor; the measured wall medians and
    /// per-sample counts are recorded in the plan so the choice can be
    /// audited.
    public static let noiseRelWall = 0.02

    /// SHA-256 hex of the pass-engine identity string.
    public static func engineDigest(_ identity: String) -> String {
        SHA256.hex(SHA256.hash(Array(identity.utf8)))
    }

    private static func isLowercaseHex64(_ s: String) -> Bool {
        s.count == 64 && s.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") }
    }

    /// Builds the plan with REAL identity: graph digest, streaming GGUF
    /// content digest, pass-engine digest, measured hardware, verbatim
    /// toolchain, workload digest, and a planDigest covering all fields
    /// plus the selection. Malformed inputs are rejected here, not
    /// cached. The planDigest is an integrity check, not authentication.
    public static func makePlan(graph: GemmaGraphIR, graphDigest: String,
                                ggufDigest: String, engineDigest: String,
                                hardware: ForgeHardwareProfile,
                                prompt: [Int], decodeTokens: Int,
                                selected: GemmaForgeCandidate,
                                medianWall: Double, medianGpu: Double,
                                results: [GemmaForgeCandidateResult]) throws
            -> GemmaForgePlan {
        guard isLowercaseHex64(graphDigest), isLowercaseHex64(ggufDigest),
              isLowercaseHex64(engineDigest) else {
            throw GemmaForgeError.invalidPlan("identity digests must be 64 lowercase hex chars")
        }
        guard medianWall.isFinite, medianWall > 0,
              medianGpu.isFinite, medianGpu > 0 else {
            throw GemmaForgeError.invalidPlan("selected medians must be finite and positive")
        }
        let legal = Set(candidates())
        guard results.count == legal.count,
              Set(results.map(\.candidate)).count == results.count,
              results.allSatisfy({ legal.contains($0.candidate) }) else {
            throw GemmaForgeError.invalidPlan(
                "results must be exactly the legal candidate set, no duplicates")
        }
        guard let sel = results.first(where: { $0.candidate == selected }),
              sel.correct,
              let selWall = sel.medianWallMicrosPerToken,
              abs(selWall - medianWall) < 1e-9,
              let selGpu = sel.medianGpuMicrosPerToken,
              abs(selGpu - medianGpu) < 1e-9 else {
            throw GemmaForgeError.invalidPlan("plan selection inconsistent with results")
        }
        // The plan embeds the lowered executable schedule; its digest must
        // equal the deterministic lowering of this exact graph.
        let lowered = try GemmaSSA.lower(graph)
        let scheduleDigest = try GemmaSSA.digest(lowered)
        guard isLowercaseHex64(scheduleDigest) else {
            throw GemmaForgeError.invalidPlan("schedule digest malformed")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let body = GemmaForgePlan(
            schemaVersion: schemaVersion, graphDigest: graphDigest,
            ggufDigest: ggufDigest, engineDigest: engineDigest,
            hardware: hardware, toolchain: Forge.toolchainIdentifier,
            workloadDigest: workloadDigest(prompt),
            promptTokenCount: prompt.count, decodeTokens: decodeTokens,
            selected: selected,
            selectedMedianWallMicrosPerToken: medianWall,
            selectedMedianGpuMicrosPerToken: medianGpu,
            results: results, schedule: lowered,
            scheduleDigest: scheduleDigest, planDigest: "")
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
        return GemmaForgePlan(schemaVersion: body.schemaVersion,
                              graphDigest: body.graphDigest,
                              ggufDigest: body.ggufDigest,
                              engineDigest: body.engineDigest,
                              hardware: body.hardware,
                              toolchain: body.toolchain,
                              workloadDigest: body.workloadDigest,
                              promptTokenCount: body.promptTokenCount,
                              decodeTokens: body.decodeTokens,
                              selected: body.selected,
                              selectedMedianWallMicrosPerToken: medianWall,
                              selectedMedianGpuMicrosPerToken: medianGpu,
                              results: body.results, schedule: lowered,
                              scheduleDigest: scheduleDigest, planDigest: digest)
    }

    /// Saves a plan atomically (Data.write .atomic replaces any existing
    /// cache file without leaving partial output).
    public static func savePlan(_ plan: GemmaForgePlan, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(plan)
        try bytes.write(to: url, options: [.atomic])
    }

    /// Bounded-read plan loader: the file is read in chunks up to 1 MiB
    /// BEFORE any full-file allocation.
    public static func loadPlan(from url: URL) throws -> GemmaForgePlan {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let limit = 1 << 20
        var data = Data()
        while data.count <= limit {
            let chunk = try handle.read(upToCount: 1 << 16) ?? Data()
            if chunk.isEmpty { break }
            data.append(chunk)
        }
        return try parsePlan(data)
    }

    /// Parses plan bytes with a bounded size check.
    public static func parsePlan(_ data: Data) throws -> GemmaForgePlan {
        guard data.count <= 1 << 20 else {
            throw GemmaForgeError.invalidPlan("plan exceeds 1 MiB")
        }
        do { return try JSONDecoder().decode(GemmaForgePlan.self, from: data) }
        catch { throw GemmaForgeError.invalidPlan("\(error)") }
    }

    /// Validates a loaded plan against the CURRENT graph, GGUF content,
    /// pass-engine identity, hardware, toolchain, workload, candidate
    /// legality, result consistency, and its own planDigest. A stale,
    /// foreign, tampered, or inconsistent plan is rejected and re-forged —
    /// never executed. planDigest is integrity, not authentication.
    public static func validatePlan(_ plan: GemmaForgePlan, graph: GemmaGraphIR,
                                    ggufDigest: String, engineDigest: String,
                                    hardware: ForgeHardwareProfile,
                                    prompt: [Int], decodeTokens: Int) throws {
        guard plan.schemaVersion == schemaVersion else {
            throw GemmaForgeError.invalidPlan("schema \(plan.schemaVersion) != \(schemaVersion)")
        }
        // Structural validity of the LOADED plan (not just plans this
        // process authored): digest syntax, positive finite medians,
        // exactly the legal candidate set with no duplicates.
        guard isLowercaseHex64(plan.graphDigest),
              isLowercaseHex64(plan.ggufDigest),
              isLowercaseHex64(plan.engineDigest),
              isLowercaseHex64(plan.planDigest) else {
            throw GemmaForgeError.invalidPlan("plan digests malformed")
        }
        guard plan.selectedMedianWallMicrosPerToken.isFinite,
              plan.selectedMedianWallMicrosPerToken > 0,
              plan.selectedMedianGpuMicrosPerToken.isFinite,
              plan.selectedMedianGpuMicrosPerToken > 0 else {
            throw GemmaForgeError.invalidPlan("plan medians not finite/positive")
        }
        let legalSet = Set(candidates())
        guard plan.results.count == legalSet.count,
              Set(plan.results.map(\.candidate)).count == plan.results.count,
              plan.results.allSatisfy({ legalSet.contains($0.candidate) }) else {
            throw GemmaForgeError.invalidPlan(
                "plan results not exactly the legal candidate set")
        }
        let currentGraphDigest = try GemmaGraph.digest(graph)
        guard plan.graphDigest == currentGraphDigest else {
            throw GemmaForgeError.invalidPlan("graph digest mismatch")
        }
        guard !ggufDigest.isEmpty, plan.ggufDigest == ggufDigest else {
            throw GemmaForgeError.invalidPlan("GGUF content digest mismatch")
        }
        guard !engineDigest.isEmpty, plan.engineDigest == engineDigest else {
            throw GemmaForgeError.invalidPlan("pass-engine identity mismatch")
        }
        guard plan.hardware == hardware else {
            throw GemmaForgeError.invalidPlan("hardware profile mismatch")
        }
        guard plan.toolchain == Forge.toolchainIdentifier else {
            throw GemmaForgeError.invalidPlan("toolchain mismatch")
        }
        guard plan.workloadDigest == workloadDigest(prompt),
              plan.promptTokenCount == prompt.count,
              plan.decodeTokens == decodeTokens else {
            throw GemmaForgeError.invalidPlan("workload mismatch")
        }
        guard candidates().contains(plan.selected) else {
            throw GemmaForgeError.invalidPlan("selected candidate not legal")
        }
        guard let sel = plan.results.first(where: { $0.candidate == plan.selected }),
              sel.correct,
              let selWall = sel.medianWallMicrosPerToken,
              abs(selWall - plan.selectedMedianWallMicrosPerToken) < 1e-9,
              sel.medianGpuMicrosPerToken != nil,
              abs(sel.medianGpuMicrosPerToken! - plan.selectedMedianGpuMicrosPerToken) < 1e-9
        else {
            throw GemmaForgeError.invalidPlan("selected result inconsistent with plan")
        }
        // Embedded schedule: must be present, dataflow-valid, and EXACTLY
        // the deterministic lowering of the current graph. The runtime
        // executes these artifact operands after this check.
        guard let schedule = plan.schedule else {
            throw GemmaForgeError.invalidPlan("plan carries no lowered schedule")
        }
        let lowered = try GemmaSSA.lower(graph)
        guard schedule == lowered,
              plan.scheduleDigest == (try GemmaSSA.digest(schedule)),
              plan.scheduleDigest == (try GemmaSSA.digest(lowered)) else {
            throw GemmaForgeError.invalidPlan("embedded schedule tampered/mismatched")
        }
        // planDigest self-consistency: the digest covers all fields.
        let body = GemmaForgePlan(
            schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
            ggufDigest: plan.ggufDigest, engineDigest: plan.engineDigest,
            hardware: plan.hardware, toolchain: plan.toolchain,
            workloadDigest: plan.workloadDigest,
            promptTokenCount: plan.promptTokenCount, decodeTokens: plan.decodeTokens,
            selected: plan.selected,
            selectedMedianWallMicrosPerToken: plan.selectedMedianWallMicrosPerToken,
            selectedMedianGpuMicrosPerToken: plan.selectedMedianGpuMicrosPerToken,
            results: plan.results, schedule: plan.schedule,
            scheduleDigest: plan.scheduleDigest, planDigest: "")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let digest = SHA256.hex(SHA256.hash(Array(try encoder.encode(body))))
        guard plan.planDigest == digest else {
            throw GemmaForgeError.invalidPlan("planDigest tampered")
        }
    }

    public static func workloadDigest(_ tokens: [Int]) -> String {
        var material = [UInt8]()
        for t in tokens {
            withUnsafeBytes(of: UInt32(t).littleEndian) { material.append(contentsOf: $0) }
        }
        return SHA256.hex(SHA256.hash(material))
    }
}
