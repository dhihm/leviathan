import Foundation
import Metal

// Phase 62: COMPONENT-ONLY candidate "register word fetch inside the production loop shape".
// Evidence carried forward (no instruction-level cause was measured): phase 61 (fully unrolled 36-byte pair body,
// constant x indices) stayed finite, in FP64 contract and guard-intact on all 17 supported cases but failed Float
// bit identity on 16 of them (every multi-row case; 1368 of 2447 rows; only the single-row min-1x64 case matched),
// so its timing was withheld; phase 60 (production block loop + inner l loop with a different byte SOURCE) was
// bit-identical everywhere. This candidate therefore keeps the compiler-visible production
// loop shape exactly: for b over blocksPerRow, for l over 16, the same dbits decode, the same two sum lines with the
// same x[b * 32u + l] / x[b * 32u + 16u + l] indices. Only the byte source changes: per block, five aligned uint words
// are loaded into registers (an even block is pair words 0..4; an odd block starts 18 bytes into the pair, so its
// five words are funnel-shifted by 16 bits from pair words 4..8, integer ops only) and byte 2 + l is extracted by
// word select and shift inside the inner loop. No pair unrolling, threadgroup memory, barriers, SIMD, float4,
// reassociation or explicit fma. Compiled bits remain empirical. Reuses the phase-60 corpus, oracle, guards, GPU
// class, tick checks and timing protocol, and the phase-61 alignment plan and launch, unchanged.
// fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false.

// MARK: - Descriptor

public struct Q4RegisterLoopDescriptor: Codable, Equatable, Hashable, Sendable {
    public let executionWidth: Int
    public let wordsPerBlockFetch: Int
    public let wordsPerPair: Int
    public let oddBlockShiftBits: Int

    public var kernelName: String { "q4rl_matvec_W" + String(executionWidth) + "_fetch" + String(wordsPerBlockFetch) }

    /// The alignment contract is the phase-61 pair layout (even block count, 4-aligned offset, 9 words per pair).
    public var pairLayout: Q4PackedWordDescriptor? { try? Q4PackedWordDescriptor.derive(executionWidth: executionWidth) }

    public static func derive(executionWidth: Int) throws -> Q4RegisterLoopDescriptor {
        let d = Q4RegisterLoopDescriptor(executionWidth: executionWidth, wordsPerBlockFetch: 5, wordsPerPair: 9,
                                         oddBlockShiftBits: 16)
        try d.validate()
        return d
    }

    public func validate() throws {
        guard pairLayout != nil else {
            throw Q4WeightStageError("register-loop descriptor requires the width-32 pair layout, got width "
                                     + String(executionWidth))
        }
        guard wordsPerBlockFetch == 5, wordsPerPair == 9, oddBlockShiftBits == 16 else {
            throw Q4WeightStageError("register-loop descriptor is not the canonical 5-word block fetch over 9-word pairs")
        }
    }
}

// MARK: - Plan and host mirror of the per-block word fetch

public enum Q4RegisterLoopPlan {
    public static func declineReason(inDim: Int, outDim: Int, weightOffset: Int, weightLength: Int,
                                     descriptor: Q4RegisterLoopDescriptor, executionWidth: Int,
                                     pipelineMaxThreads: Int, staticThreadgroupBytes: Int) -> String? {
        guard (try? descriptor.validate()) != nil, let pair = descriptor.pairLayout else { return "descriptor invalid" }
        return Q4PackedWordPlan.declineReason(inDim: inDim, outDim: outDim, weightOffset: weightOffset,
                                              weightLength: weightLength, descriptor: pair,
                                              executionWidth: executionWidth, pipelineMaxThreads: pipelineMaxThreads,
                                              staticThreadgroupBytes: staticThreadgroupBytes)
    }

    public static func launch(outDim: Int, descriptor: Q4RegisterLoopDescriptor) -> Q4WeightStageLaunch {
        Q4WeightStageLaunch(threadgroups: (outDim + descriptor.executionWidth - 1) / descriptor.executionWidth,
                            threadsPerThreadgroup: descriptor.executionWidth)
    }

    /// The five block words the kernel holds in registers for block parity `odd` of one 9-word pair: even blocks
    /// take pair words 0..4 unchanged; odd blocks funnel pair words 4..8 by 16 bits so that block byte i lives at
    /// word i / 4, shift 8 * (i % 4), exactly like an even block.
    public static func blockWords(pair w: [UInt32], odd: Bool) -> [UInt32]? {
        guard w.count == 9 else { return nil }
        if !odd { return Array(w[0..<5]) }
        return [(w[4] >> 16) | (w[5] << 16), (w[5] >> 16) | (w[6] << 16), (w[6] >> 16) | (w[7] << 16),
                (w[7] >> 16) | (w[8] << 16), w[8] >> 16]
    }

    /// Host mirror of the in-loop extraction: the 18 block bytes from the five block words.
    public static func blockBytes(_ bw: [UInt32]) -> [UInt8]? {
        guard bw.count == 5 else { return nil }
        return (0..<18).map { i in UInt8((bw[i / 4] >> UInt32(8 * (i % 4))) & 0xFF) }
    }
}

// MARK: - Emitter

public enum Q4RegisterLoopEmitter {
    public static let codegenVersion = "q4rl-v1"

    /// Deterministic MSL: the production loop shape with a per-block five-word register fetch as the byte source.
    public static func emit(_ d: Q4RegisterLoopDescriptor) throws -> String {
        try d.validate()
        let lines: [String] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "// " + d.kernelName + " - EMITTED by LeviathanForge Q4 register-loop decoding (" + codegenVersion + ").",
            "// Production block loop and 16-term inner loop; each block's 18 bytes are fetched as five aligned uint",
            "// words into registers (odd blocks funnel-shifted by 16 bits from pair words 4..8); byte 2 + l is",
            "// extracted inside the inner loop. No on-chip shared staging, synchronization, vector load or reassociation.",
            "kernel void " + d.kernelName + "(constant const uint2& p [[buffer(0)]],",
            "        constant const ulong& wOff [[buffer(1)]],",
            "        device const uchar* wbase [[buffer(2)]],",
            "        device const float* x [[buffer(3)]],",
            "        device float* y [[buffer(4)]],",
            "        uint gid [[thread_position_in_grid]]) {",
            "    uint row = gid;",
            "    if (row >= p.y) { return; }",
            "    uint blocksPerRow = p.x / 32u;",
            "    uint pairsPerRow = blocksPerRow / 2u;",
            "    device const uint* words = (device const uint*)(wbase + wOff) + (ulong)row * (ulong)pairsPerRow * 9u;",
            "    float sum = 0.0f;",
            "    for (uint b = 0; b < blocksPerRow; b++) {",
            "        device const uint* pw = words + (ulong)(b >> 1u) * 9u;",
            "        uint b0; uint b1; uint b2; uint b3; uint b4;",
            "        if ((b & 1u) == 0u) {",
            "            b0 = pw[0]; b1 = pw[1]; b2 = pw[2]; b3 = pw[3]; b4 = pw[4];",
            "        } else {",
            "            uint t4 = pw[4]; uint t5 = pw[5]; uint t6 = pw[6]; uint t7 = pw[7]; uint t8 = pw[8];",
            "            b0 = (t4 >> 16u) | (t5 << 16u);",
            "            b1 = (t5 >> 16u) | (t6 << 16u);",
            "            b2 = (t6 >> 16u) | (t7 << 16u);",
            "            b3 = (t7 >> 16u) | (t8 << 16u);",
            "            b4 = t8 >> 16u;",
            "        }",
            "        ushort dbits = ushort(b0 & 0xFFFFu);",
            "        float d = float(as_type<half>(dbits));",
            "        for (uint l = 0; l < 16u; l++) {",
            "            uint bi = 2u + l;",
            "            uint wsel = bi >> 2u;",
            "            uint word = (wsel == 0u) ? b0 : (wsel == 1u) ? b1 : (wsel == 2u) ? b2 : (wsel == 3u) ? b3 : b4;",
            "            uchar q = uchar((word >> ((bi & 3u) * 8u)) & 0xFFu);",
            "            sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];",
            "            sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];",
            "        }",
            "    }",
            "    y[row] = sum;",
            "}",
        ]
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Report

public struct Q4RegisterLoopReport: Codable {
    public private(set) var kind = Q4RegisterLoopDiagnostics.kind
    public private(set) var closedPredecessors = Q4RegisterLoopDiagnostics.closedPredecessors
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var timestampUnits = Q4WeightStageDiagnostics.timestampUnits
    public private(set) var notes = Q4RegisterLoopDiagnostics.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var controlFunction = Q4WeightStageEmitter.controlFunction
    public var controlSourceSha256: String? = nil
    public var controlLaunchWidth = Q4WeightStagePlan.controlLaunchWidth
    public var hardware: Q4WeightStageHardware? = nil
    public var descriptor: Q4RegisterLoopDescriptor? = nil
    public var kernelName: String? = nil
    public var candidateSourceSha256: String? = nil
    public var compileError: String? = nil
    public var pipelineThreadExecutionWidth: Int? = nil
    public var pipelineMaxTotalThreads: Int? = nil
    public var staticThreadgroupBytes: Int? = nil
    public var corpusCases = 0
    public var realCases = 0
    public var cases: [Q4WeightStageCaseResult] = []
    public var correctnessPassed = false
    public var timing: [Q4WeightStageTimingShape] = []
    public var timingPassed: Bool? = nil
    public var commandBuffersCommitted = 0
    public var terminal = true
    public var error: String? = nil
    public var status: Q4WeightStageStatus = .aborted

    public init() {}
}

extension Q4RegisterLoopReport {
    func withContract() -> Q4RegisterLoopReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

// MARK: - Driver

public enum Q4RegisterLoopDiagnostics {
    public static let kind = "q4-register-loop-decode-component-v1"
    public static let usage = "usage: q4-register-loop-component --report NEW_REPORT.json [--gguf M.gguf]"
    /// Closed candidates this one must not be confused with; their records stand.
    public static let closedPredecessors: [String] = [
        "q4-packed-weight-staging-component-v1: bit-identical, performance-rejected (2.48x to 2.86x slower)",
        "q4-packed-word-register-decode-component-v1: finite, FP64 in contract and guards intact, correctness-rejected (Float bits differ on 16 of 17 supported cases, 1368 of 2447 rows; timing withheld)",
    ]

    public static let notes: [String] = [
        "Component only: fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim "
            + "are always false; no default, kernel, guard, artifact or model change.",
        "Candidate: the production block loop and 16-term inner loop are kept exactly (same dbits decode, same two sum "
            + "lines, same x[b * 32u + l] indices); only the byte source changes to a per-block five-word register fetch "
            + "(odd blocks funnel-shifted by 16 bits from pair words 4..8) with in-loop byte extraction. No pair "
            + "unrolling, threadgroup memory, barriers, SIMD, float4, reassociation or explicit fma.",
        "Rationale: phase 60 kept this loop shape with a different byte source and was bit-identical everywhere; phase 61 "
            + "unrolled the pair body and lost bit identity on every multi-row case (16 of 17 supported cases, 1368 of "
            + "2447 rows). No instruction-level cause was measured; compiled bits remain empirical.",
        "Control is the production g4_q4_matvec from Gemma4Runner.shaderSource; both compiled with options nil; bit "
            + "identity, the 1e-4 FP64 contract and guards are measured on every row of the unchanged corpus before timing.",
        "Timing reuses the phase-60 protocol unchanged (2 warmup + 6 timed pairs, two sequential command buffers per "
            + "pair, samples 0/1 then 2/3 resolved twice after both are terminal); raw ticks and per-order ratios only, "
            + "no busy time, no full-model implication.",
    ]

    public static func parseOptions(_ argv: [String]) throws -> Q4WeightStageDiagnostics.Options {
        var v = [String: String]()
        var i = 0
        while i < argv.count {
            let key = argv[i]
            guard key == "--report" || key == "--gguf", i + 1 < argv.count, v[key] == nil else {
                throw Q4WeightStageError(usage)
            }
            let value = argv[i + 1]
            guard !value.isEmpty, !value.hasPrefix("--") else { throw Q4WeightStageError(usage) }
            v[key] = value
            i += 2
        }
        guard let report = v["--report"] else { throw Q4WeightStageError(usage) }
        return Q4WeightStageDiagnostics.Options(report: report, gguf: v["--gguf"])
    }

    public static func encodeReport(_ r: Q4RegisterLoopReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(r.withContract())
    }

    public static func finalStatus(_ r: Q4RegisterLoopReport) -> Q4WeightStageStatus {
        if !r.terminal { return .notEnded }
        if r.error != nil { return .aborted }
        guard r.hardware != nil, r.descriptor != nil, r.compileError == nil else { return .unsupported }
        let cbError = r.cases.contains { $0.controlCompletion == "ERROR" || $0.candidateCompletion == "ERROR" }
            || r.timing.contains { $0.pairs.contains { $0.completion == "ERROR" } }
        if cbError { return .commandBufferError }
        guard r.correctnessPassed else { return .correctnessFailed }
        if r.timingPassed == false { return .timingChecksFailed }
        if r.timingPassed == nil { return .unsupported }
        return .correctnessPassed
    }

    public static func run(ggufURL: URL?, modelSha256: String?, executableDigest: String,
                           emit: (String) -> Void) -> Q4RegisterLoopReport {
        let counter = Q4WeightStageDiagnostics.CommitCounter()
        var report = runCore(ggufURL: ggufURL, modelSha256: modelSha256, executableDigest: executableDigest,
                             counter: counter, emit: emit)
        report.commandBuffersCommitted = counter.committed
        return report
    }

    static func runCore(ggufURL: URL?, modelSha256: String?, executableDigest: String,
                        counter: Q4WeightStageDiagnostics.CommitCounter,
                        emit: (String) -> Void) -> Q4RegisterLoopReport {
        var report = Q4RegisterLoopReport()
        report.modelSha256 = modelSha256
        report.executableDigest = executableDigest
        report.controlSourceSha256 = Q4WeightStageEmitter.sourceDigest(Gemma4Runner.shaderSource)
        guard let device = MTLCreateSystemDefaultDevice() else {
            report.error = "no Metal device"
            report.status = finalStatus(report)
            return report
        }
        let gpu: Q4WeightStageGPU
        do { gpu = try Q4WeightStageGPU(device: device) } catch {
            report.error = String(describing: error)
            report.status = finalStatus(report)
            return report
        }
        defer { counter.committed = gpu.commandBuffers.count }
        let atStage = device.supportsCounterSampling(.atStageBoundary)
        let atDispatch = device.supportsCounterSampling(.atDispatchBoundary)
        var timestampSet: MTLCounterSet? = nil
        for set in device.counterSets ?? [] where set.name == MTLCommonCounterSet.timestamp.rawValue {
            if set.counters.contains(where: { $0.name == MTLCommonCounter.timestamp.rawValue }) { timestampSet = set }
        }
        var sampleBuffer: MTLCounterSampleBuffer? = nil
        if atStage, let set = timestampSet {
            let d = MTLCounterSampleBufferDescriptor()
            d.counterSet = set
            d.label = "q4-register-loop-component"
            d.storageMode = .shared
            d.sampleCount = Q4WeightStageTicks.sampleCount
            sampleBuffer = try? device.makeCounterSampleBuffer(descriptor: d)
        }
        report.hardware = Q4WeightStageHardware(
            device: device.name, controlThreadExecutionWidth: gpu.control.threadExecutionWidth,
            controlMaxTotalThreads: gpu.control.maxTotalThreadsPerThreadgroup,
            deviceMaxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            deviceMaxThreadsPerThreadgroupWidth: device.maxThreadsPerThreadgroup.width,
            supportsAtStageBoundary: atStage, supportsAtDispatchBoundary: atDispatch,
            timestampCounterSetFound: timestampSet != nil, sampleBufferCreated: sampleBuffer != nil)
        emit("device " + device.name + ": control width " + String(gpu.control.threadExecutionWidth)
             + ", stage sampling " + String(sampleBuffer != nil))

        let descriptor: Q4RegisterLoopDescriptor
        do { descriptor = try Q4RegisterLoopDescriptor.derive(executionWidth: gpu.control.threadExecutionWidth) } catch {
            report.compileError = "descriptor: " + String(describing: error)
            report.status = finalStatus(report)
            return report
        }
        report.descriptor = descriptor
        report.kernelName = descriptor.kernelName
        let pipe: MTLComputePipelineState
        do {
            let source = try Q4RegisterLoopEmitter.emit(descriptor)
            report.candidateSourceSha256 = Q4WeightStageEmitter.sourceDigest(source)
            let library = try device.makeLibrary(source: source, options: nil)
            guard let fn = library.makeFunction(name: descriptor.kernelName) else {
                throw Q4WeightStageError("candidate function missing")
            }
            pipe = try device.makeComputePipelineState(function: fn)
        } catch {
            report.compileError = "candidate compile (options nil): " + String(describing: error)
            report.status = finalStatus(report)
            return report
        }
        report.pipelineThreadExecutionWidth = pipe.threadExecutionWidth
        report.pipelineMaxTotalThreads = pipe.maxTotalThreadsPerThreadgroup
        report.staticThreadgroupBytes = pipe.staticThreadgroupMemoryLength

        var cases = Q4WeightStageCorpus.synthetic()
        var timingCases = Q4WeightStageCorpus.syntheticTiming()
        if let url = ggufURL {
            do {
                let file = try GGUFReader.read(url)
                for (i, t) in Q4WeightStageCorpus.realTensors.enumerated() {
                    cases.append(try Q4WeightStageCorpus.real(file: file, tensor: t, rows: Q4WeightStageCorpus.realRows,
                                                              name: "real-" + String(i) + "-" + t))
                    report.realCases += 1
                }
                timingCases.append(try Q4WeightStageCorpus.real(
                    file: file, tensor: Q4WeightStageCorpus.realTimingTensor,
                    rows: Q4WeightStageCorpus.realTimingRows, name: "timing-real-" + Q4WeightStageCorpus.realTimingTensor))
            } catch {
                report.error = "real corpus: " + String(describing: error)
                report.status = finalStatus(report)
                return report
            }
        }
        report.corpusCases = cases.count

        var allPassed = true
        for c in cases {
            let b: Q4WeightStageBuffers
            let yControl: MTLBuffer
            do {
                b = try gpu.makeBuffers(c)
                yControl = try gpu.makeOutput(outDim: c.outDim)
            } catch {
                report.error = "allocation: " + String(describing: error)
                report.status = finalStatus(report)
                return report
            }
            let controlOutcome = gpu.runSingle(
                pipeline: gpu.control,
                launch: Q4WeightStagePlan.controlLaunch(outDim: c.outDim,
                                                       pipelineMaxThreads: gpu.control.maxTotalThreadsPerThreadgroup),
                b: b, y: yControl)
            if !controlOutcome.terminal {
                report.terminal = false
                report.error = "control " + c.name + ": " + (controlOutcome.error ?? "not ended")
                report.status = finalStatus(report)
                return report
            }
            var ref: [Double]? = nil
            var refError: String? = nil
            do { ref = try Q4Oracle.matvecFP64(blocks: c.blocks[...], outDim: c.outDim, inDim: c.inDim, x: c.x) } catch {
                refError = String(describing: error)
            }
            let ctlBits = controlOutcome.completion == "COMPLETED" ? gpu.outputBits(yControl, outDim: c.outDim) : []
            let ctlGuards = gpu.guardsIntact(b.w, payloadBytes: b.weightBytes)
                && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4) && gpu.guardsIntact(yControl, payloadBytes: c.outDim * 4)
            let (ctlWorst, ctlFinite) = ref.map { Q4WeightStageDiagnostics.worstRel(ctlBits, $0) } ?? (nil, false)
            let ctlPass = ctlFinite && controlOutcome.completion == "COMPLETED" && ctlGuards
                && (ctlWorst.map { $0 < Q4WeightStageDiagnostics.correctnessContract } ?? false)
            var f = [String]()
            if controlOutcome.completion != "COMPLETED" { f.append("control " + controlOutcome.completion) }
            if let e = refError { f.append("FP64 oracle: " + e) }
            if !ctlPass { f.append("control outside the FP64 contract, non-finite or guard violation") }
            let decline = Q4RegisterLoopPlan.declineReason(
                inDim: c.inDim, outDim: c.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
                weightLength: Q4WeightStageGPU.guardBytes + c.blocks.count, descriptor: descriptor,
                executionWidth: pipe.threadExecutionWidth, pipelineMaxThreads: pipe.maxTotalThreadsPerThreadgroup,
                staticThreadgroupBytes: pipe.staticThreadgroupMemoryLength)
            if let reason = decline {
                if !c.expectedDecline { f.append("unexpected decline: " + reason) }
                report.cases.append(Q4WeightStageCaseResult(
                    name: c.name, origin: c.origin, inDim: c.inDim, outDim: c.outDim, blocksPerRow: c.blocksPerRow,
                    expectedDecline: c.expectedDecline, declined: reason, controlCompletion: controlOutcome.completion,
                    candidateCompletion: nil, controlFinite: ctlFinite, candidateFinite: false,
                    controlFP64WorstRel: ctlWorst, candidateFP64WorstRel: nil, controlFP64Pass: ctlPass,
                    candidateFP64Pass: false, bitIdentical: false, mismatchRows: 0, firstMismatchRow: nil,
                    guardsIntact: ctlGuards, passed: f.isEmpty, failure: f.isEmpty ? nil : f.joined(separator: "; ")))
                if !f.isEmpty { allPassed = false }
                continue
            }
            if c.expectedDecline { f.append("predeclared unsupported shape was not declined") }
            let y: MTLBuffer
            do { y = try gpu.makeOutput(outDim: c.outDim) } catch {
                report.error = "allocation: " + String(describing: error)
                report.status = finalStatus(report)
                return report
            }
            let outcome = gpu.runSingle(pipeline: pipe, launch: Q4RegisterLoopPlan.launch(outDim: c.outDim, descriptor: descriptor),
                                        b: b, y: y)
            if !outcome.terminal {
                report.terminal = false
                report.error = "candidate " + c.name + ": " + (outcome.error ?? "not ended")
                report.status = finalStatus(report)
                return report
            }
            let bits = outcome.completion == "COMPLETED" ? gpu.outputBits(y, outDim: c.outDim) : []
            let guards = ctlGuards && gpu.guardsIntact(b.w, payloadBytes: b.weightBytes)
                && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4) && gpu.guardsIntact(y, payloadBytes: c.outDim * 4)
            let (candWorst, candFinite) = ref.map { Q4WeightStageDiagnostics.worstRel(bits, $0) } ?? (nil, false)
            let candPass = candFinite && outcome.completion == "COMPLETED"
                && (candWorst.map { $0 < Q4WeightStageDiagnostics.correctnessContract } ?? false)
            var mismatches = 0
            var firstMismatch: Int? = nil
            if bits.count == ctlBits.count {
                for r in 0..<bits.count where bits[r] != ctlBits[r] {
                    mismatches += 1
                    if firstMismatch == nil { firstMismatch = r }
                }
            } else {
                mismatches = max(bits.count, ctlBits.count)
            }
            let identical = outcome.completion == "COMPLETED" && controlOutcome.completion == "COMPLETED"
                && bits.count == ctlBits.count && mismatches == 0
            if outcome.completion != "COMPLETED" { f.append("candidate " + outcome.completion) }
            if !candFinite { f.append("candidate output not finite") }
            if !candPass { f.append("candidate outside the FP64 contract") }
            if !identical { f.append(String(mismatches) + " row(s) differ in Float bits from the control") }
            if !guards { f.append("guard bytes violated") }
            report.cases.append(Q4WeightStageCaseResult(
                name: c.name, origin: c.origin, inDim: c.inDim, outDim: c.outDim, blocksPerRow: c.blocksPerRow,
                expectedDecline: c.expectedDecline, declined: nil, controlCompletion: controlOutcome.completion,
                candidateCompletion: outcome.completion, controlFinite: ctlFinite, candidateFinite: candFinite,
                controlFP64WorstRel: ctlWorst, candidateFP64WorstRel: candWorst, controlFP64Pass: ctlPass,
                candidateFP64Pass: candPass, bitIdentical: identical, mismatchRows: mismatches,
                firstMismatchRow: firstMismatch, guardsIntact: guards, passed: f.isEmpty,
                failure: f.isEmpty ? nil : f.joined(separator: "; ")))
            if !f.isEmpty { allPassed = false }
        }
        report.correctnessPassed = allPassed
        emit("candidate " + descriptor.kernelName + ": " + String(report.cases.filter { $0.passed }.count) + " of "
             + String(report.cases.count) + " cases passed")

        if allPassed, let sb = sampleBuffer {
            var okAll = true
            for tc in timingCases {
                let decline = Q4RegisterLoopPlan.declineReason(
                    inDim: tc.inDim, outDim: tc.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
                    weightLength: Q4WeightStageGPU.guardBytes + tc.blocks.count, descriptor: descriptor,
                    executionWidth: pipe.threadExecutionWidth, pipelineMaxThreads: pipe.maxTotalThreadsPerThreadgroup,
                    staticThreadgroupBytes: pipe.staticThreadgroupMemoryLength)
                let shape = Q4WeightStageDiagnostics.timedShapeCore(
                    gpu: gpu, candidateDecline: decline,
                    candidateLaunch: Q4RegisterLoopPlan.launch(outDim: tc.outDim, descriptor: descriptor), pipeline: pipe,
                    sampleBuffer: sb, tc: tc)
                guard let s = shape.shape else {
                    report.terminal = false
                    report.error = "timing " + tc.name + ": completion not observed"
                    report.status = finalStatus(report)
                    return report
                }
                report.timing.append(s)
                if !s.passed { okAll = false }
                emit("timing " + tc.name + ": raw ratio " + (s.rawRatioCandidateOverControl.map { String($0) } ?? "n/a")
                     + " (raw ticks, not busy time)")
            }
            report.timingPassed = okAll
        }
        report.status = finalStatus(report)
        return report
    }
}
