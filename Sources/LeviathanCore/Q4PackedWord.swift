import Foundation
import Metal

// Phase 61: COMPONENT-ONLY candidate "paired register uint32 packed Q4_0 WEIGHT decoding".
// One thread per output row at the production launch width, NO threadgroup memory, NO barriers. Two consecutive
// Q4_0 blocks are 36 bytes = nine aligned 32-bit words (even block count, 4-aligned weight offset). Each pair is
// loaded as nine words into registers; the FP16 scale of block 0 is the low half of word 0, the scale of block 1 the
// high half of word 4; block 0's sixteen quant bytes are the high halves of words 0..3 and the low half of word 4,
// block 1's are words 5..8. Block 0's sixteen low/high term pairs are evaluated first in the production order,
// then block 1's, with the production expression d * (float(q & 0x0Fu) - 8.0f) * x[...] and device scalar x reads.
// No input vector loads, no input staging, no launch-width change, no SIMD or tree reassociation, no FP64 and no
// threadgroup staging: distinct from every closed candidate. Whether nine word loads beat thirty-six byte loads is
// not claimed; it is measured, and compiled Float bits are empirical (identical source never guarantees them).
// Reuses the phase-60 corpus, oracle, guards, GPU class, tick checks and timing protocol unchanged.
// fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false.

// MARK: - Descriptor

public struct Q4PackedWordDescriptor: Codable, Equatable, Hashable, Sendable {
    public let executionWidth: Int
    public let blocksPerPair: Int
    public let bytesPerPair: Int
    public let wordsPerPair: Int

    public static let requiredExecutionWidth = 32

    public var kernelName: String { "q4pw_matvec_W" + String(executionWidth) + "_pair" + String(wordsPerPair) }

    public static func derive(executionWidth: Int) throws -> Q4PackedWordDescriptor {
        let d = Q4PackedWordDescriptor(executionWidth: executionWidth, blocksPerPair: 2,
                                       bytesPerPair: 2 * Q4_0.bytesPerBlock, wordsPerPair: 2 * Q4_0.bytesPerBlock / 4)
        try d.validate()
        return d
    }

    /// Canonical layout identity: width 32 (the measured production width this scope is bound to), two 18-byte
    /// blocks per pair, 36 bytes = 9 whole words.
    public func validate() throws {
        guard executionWidth == Self.requiredExecutionWidth else {
            throw Q4WeightStageError("packed-word descriptor is bound to execution width "
                                     + String(Self.requiredExecutionWidth) + ", got " + String(executionWidth))
        }
        guard blocksPerPair == 2, bytesPerPair == 2 * Q4_0.bytesPerBlock, bytesPerPair % 4 == 0,
              wordsPerPair == bytesPerPair / 4, wordsPerPair == 9 else {
            throw Q4WeightStageError("packed-word descriptor is not the canonical 36-byte / 9-word pair layout")
        }
    }
}

// MARK: - Plan and host mirror of the pair layout

public struct Q4PackedWordPairDecode: Equatable, Sendable {
    public let d0Bits: UInt16
    public let q0: [UInt8]
    public let d1Bits: UInt16
    public let q1: [UInt8]
}

public enum Q4PackedWordPlan {
    /// nil when supported; otherwise the exact reason the candidate is declined BEFORE any dispatch.
    public static func declineReason(inDim: Int, outDim: Int, weightOffset: Int, weightLength: Int,
                                     descriptor: Q4PackedWordDescriptor, executionWidth: Int,
                                     pipelineMaxThreads: Int, staticThreadgroupBytes: Int) -> String? {
        if (try? descriptor.validate()) == nil { return "descriptor invalid" }
        guard inDim > 0, inDim % Q4_0.valuesPerBlock == 0, inDim <= ModelWorkloadLimits.maxQ4InputDim,
              inDim <= Int(UInt32.max) else {
            return "inDim " + String(inDim) + " is not a positive bounded multiple of 32"
        }
        guard outDim > 0, outDim <= ModelWorkloadLimits.maxQ4OutputDim, outDim <= Int(UInt32.max) else {
            return "outDim " + String(outDim) + " is not positive and bounded"
        }
        let blocksPerRow = inDim / Q4_0.valuesPerBlock
        guard blocksPerRow % descriptor.blocksPerPair == 0 else {
            return "odd block count " + String(blocksPerRow) + ": rows are not whole 36-byte pairs"
        }
        let (rowBytes, o1) = blocksPerRow.multipliedReportingOverflow(by: Q4_0.bytesPerBlock)
        guard !o1, rowBytes % 4 == 0 else { return "row byte count invalid" }
        guard weightOffset >= 0, weightOffset % 4 == 0 else {
            return "weight offset " + String(weightOffset) + " is not 4-byte aligned"
        }
        let (span, o2) = rowBytes.multipliedReportingOverflow(by: outDim)
        let (end, o3) = o2 ? (0, true) : weightOffset.addingReportingOverflow(span)
        guard !o2, !o3, end <= weightLength else { return "weight span exceeds the bound buffer" }
        guard executionWidth == descriptor.executionWidth else {
            return "compiled execution width " + String(executionWidth) + " != descriptor width "
                + String(descriptor.executionWidth)
        }
        guard descriptor.executionWidth <= pipelineMaxThreads else {
            return "launch width exceeds the compiled pipeline thread maximum " + String(pipelineMaxThreads)
        }
        guard staticThreadgroupBytes == 0 else {
            return "register candidate reports " + String(staticThreadgroupBytes)
                + " bytes of static threadgroup memory; expected 0"
        }
        let groups = (outDim + descriptor.executionWidth - 1) / descriptor.executionWidth
        guard groups > 0, groups <= Int(UInt32.max) else { return "threadgroup count invalid" }
        return nil
    }

    /// One thread per row at the descriptor width: the production control geometry.
    public static func launch(outDim: Int, descriptor: Q4PackedWordDescriptor) -> Q4WeightStageLaunch {
        Q4WeightStageLaunch(threadgroups: (outDim + descriptor.executionWidth - 1) / descriptor.executionWidth,
                            threadsPerThreadgroup: descriptor.executionWidth)
    }

    /// Byte index 0..<36 of a pair -> (word index, bit shift) in little-endian words.
    public static func byteSource(_ byte: Int) -> (word: Int, shift: Int) { (byte / 4, 8 * (byte % 4)) }

    /// The nine little-endian words of one 36-byte pair (nil for any other length).
    public static func pairWords(_ bytes: ArraySlice<UInt8>) -> [UInt32]? {
        guard bytes.count == 2 * Q4_0.bytesPerBlock else { return nil }
        let b = Array(bytes)
        return (0..<9).map { w in
            UInt32(b[4 * w]) | UInt32(b[4 * w + 1]) << 8 | UInt32(b[4 * w + 2]) << 16 | UInt32(b[4 * w + 3]) << 24
        }
    }

    /// Host mirror of the kernel's register decode: scales and quant bytes of both blocks from the nine words.
    public static func decodePair(_ w: [UInt32]) -> Q4PackedWordPairDecode? {
        guard w.count == 9 else { return nil }
        func byte(_ i: Int) -> UInt8 {
            let (word, shift) = byteSource(i)
            return UInt8((w[word] >> UInt32(shift)) & 0xFF)
        }
        return Q4PackedWordPairDecode(d0Bits: UInt16(w[0] & 0xFFFF), q0: (2..<18).map(byte),
                                      d1Bits: UInt16(w[4] >> 16), q1: (20..<36).map(byte))
    }
}

// MARK: - Emitter

public enum Q4PackedWordEmitter {
    public static let codegenVersion = "q4pw-v1"

    /// Deterministic MSL: nine word loads per pair into registers, block 0 then block 1, each in the production
    /// term order with the production expression text; x stays device scalar loads at the production indices.
    public static func emit(_ d: Q4PackedWordDescriptor) throws -> String {
        try d.validate()
        var lines: [String] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "// " + d.kernelName + " - EMITTED by LeviathanForge Q4 packed-word decoding (" + codegenVersion + ").",
            "// One thread per row; each 36-byte block pair is loaded as 9 aligned uint words into registers; the",
            "// FP16 scales and quant bytes are recovered from word halves; the per-block term order and expressions",
            "// are the production scalar kernel's. No on-chip shared staging, no synchronization, no input staging.",
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
            "    for (uint pr = 0u; pr < pairsPerRow; pr++) {",
            "        device const uint* pw = words + pr * 9u;",
            "        uint w0 = pw[0]; uint w1 = pw[1]; uint w2 = pw[2]; uint w3 = pw[3]; uint w4 = pw[4];",
            "        uint w5 = pw[5]; uint w6 = pw[6]; uint w7 = pw[7]; uint w8 = pw[8];",
            "        uint xb = pr * 64u;",
            "        uchar q;",
            "        ushort dbits = ushort(w0 & 0xFFFFu);",
            "        float d = float(as_type<half>(dbits));",
        ]
        for l in 0..<16 {
            let (word, shift) = Q4PackedWordPlan.byteSource(2 + l)
            lines.append("        q = uchar((w" + String(word) + " >> " + String(shift) + "u) & 0xFFu);")
            lines.append("        sum += d * (float(q & 0x0Fu) - 8.0f) * x[xb + " + String(l) + "u];")
            lines.append("        sum += d * (float(q >> 4) - 8.0f) * x[xb + 16u + " + String(l) + "u];")
        }
        lines.append("        dbits = ushort(w4 >> 16u);")
        lines.append("        d = float(as_type<half>(dbits));")
        for l in 0..<16 {
            let (word, shift) = Q4PackedWordPlan.byteSource(20 + l)
            lines.append("        q = uchar((w" + String(word) + " >> " + String(shift) + "u) & 0xFFu);")
            lines.append("        sum += d * (float(q & 0x0Fu) - 8.0f) * x[xb + 32u + " + String(l) + "u];")
            lines.append("        sum += d * (float(q >> 4) - 8.0f) * x[xb + 48u + " + String(l) + "u];")
        }
        lines += ["    }", "    y[row] = sum;", "}"]
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Report

public struct Q4PackedWordReport: Codable {
    public private(set) var kind = Q4PackedWordDiagnostics.kind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var timestampUnits = Q4WeightStageDiagnostics.timestampUnits
    public private(set) var notes = Q4PackedWordDiagnostics.notes
    public var modelSha256: String? = nil
    public var executableDigest: String? = nil
    public var controlFunction = Q4WeightStageEmitter.controlFunction
    public var controlSourceSha256: String? = nil
    public var controlLaunchWidth = Q4WeightStagePlan.controlLaunchWidth
    public var hardware: Q4WeightStageHardware? = nil
    public var descriptor: Q4PackedWordDescriptor? = nil
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

extension Q4PackedWordReport {
    func withContract() -> Q4PackedWordReport {
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

public enum Q4PackedWordDiagnostics {
    public static let kind = "q4-packed-word-register-decode-component-v1"
    public static let usage = "usage: q4-packed-word-component --report NEW_REPORT.json [--gguf M.gguf]"

    public static let notes: [String] = [
        "Component only: fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim "
            + "are always false; no default, kernel, guard, artifact or model change.",
        "Candidate: one thread per row, each 36-byte Q4_0 block pair loaded as nine aligned uint words into registers; "
            + "scales and quant bytes recovered from word halves; block 0 then block 1 in the production term order "
            + "with the production expressions; x stays device scalar loads. No threadgroup memory, barriers, vector "
            + "loads, input staging, launch-width change or reassociation.",
        "Control is the production g4_q4_matvec from Gemma4Runner.shaderSource; both compiled with options nil. "
            + "Identical arithmetic source does not guarantee identical compiled bits; bit identity is measured on every "
            + "row of every case of the unchanged phase-60 corpus with the same 1e-4 FP64 contract and guards.",
        "Timing reuses the phase-60 protocol unchanged: per shape 2 warmup pairs then 6 timed pairs in alternating "
            + "order, two sequential command buffers per pair, samples 0/1 then 2/3 resolved twice after both are "
            + "terminal, raw ticks and per-order ratios only. A lower ratio implies no full-model gain and licenses no "
            + "integration; a full-model gate would still be required.",
        "Nine word loads replacing thirty-six byte loads is the hypothesis under test, not a claim about instruction "
            + "count or bandwidth; phase 59 ticks cannot attribute cause and phase 60 threadgroup staging was 2.5x slower.",
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

    public static func encodeReport(_ r: Q4PackedWordReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(r.withContract())
    }

    public static func finalStatus(_ r: Q4PackedWordReport) -> Q4WeightStageStatus {
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
                           emit: (String) -> Void) -> Q4PackedWordReport {
        let counter = Q4WeightStageDiagnostics.CommitCounter()
        var report = runCore(ggufURL: ggufURL, modelSha256: modelSha256, executableDigest: executableDigest,
                             counter: counter, emit: emit)
        report.commandBuffersCommitted = counter.committed
        return report
    }

    static func runCore(ggufURL: URL?, modelSha256: String?, executableDigest: String,
                        counter: Q4WeightStageDiagnostics.CommitCounter,
                        emit: (String) -> Void) -> Q4PackedWordReport {
        var report = Q4PackedWordReport()
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
            d.label = "q4-packed-word-component"
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

        let descriptor: Q4PackedWordDescriptor
        do { descriptor = try Q4PackedWordDescriptor.derive(executionWidth: gpu.control.threadExecutionWidth) } catch {
            report.compileError = "descriptor: " + String(describing: error)
            report.status = finalStatus(report)
            return report
        }
        report.descriptor = descriptor
        report.kernelName = descriptor.kernelName
        let pipe: MTLComputePipelineState
        do {
            let source = try Q4PackedWordEmitter.emit(descriptor)
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
            let decline = Q4PackedWordPlan.declineReason(
                inDim: c.inDim, outDim: c.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
                weightLength: Q4WeightStageGPU.guardBytes + c.blocks.count,
                descriptor: descriptor, executionWidth: pipe.threadExecutionWidth,
                pipelineMaxThreads: pipe.maxTotalThreadsPerThreadgroup,
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
            let outcome = gpu.runSingle(pipeline: pipe, launch: Q4PackedWordPlan.launch(outDim: c.outDim, descriptor: descriptor),
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
                let decline = Q4PackedWordPlan.declineReason(
                    inDim: tc.inDim, outDim: tc.outDim, weightOffset: Q4WeightStageGPU.guardBytes,
                    weightLength: Q4WeightStageGPU.guardBytes + tc.blocks.count, descriptor: descriptor,
                    executionWidth: pipe.threadExecutionWidth, pipelineMaxThreads: pipe.maxTotalThreadsPerThreadgroup,
                    staticThreadgroupBytes: pipe.staticThreadgroupMemoryLength)
                let shape = Q4WeightStageDiagnostics.timedShapeCore(
                    gpu: gpu, candidateDecline: decline,
                    candidateLaunch: Q4PackedWordPlan.launch(outDim: tc.outDim, descriptor: descriptor), pipeline: pipe,
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
