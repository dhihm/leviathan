import Foundation
import Metal

// Phase 65: ORIGINAL-KERNEL PREFIX PROBE on the frozen min-1x64 case. Diagnostic input perturbation only: the
// unmodified production g4_q4_matvec and the unmodified phase-62 register source are compiled once (options nil) and
// run, uninstrumented, on 65 predeclared derived activation vectors. Prefix k keeps the original x value at the x
// indices of the first k terms in the production serial term order (block b, byte l, low nibble then high nibble)
// and 0.0f elsewhere. Weights, pipelines, launch geometry and buffers are the frozen ones. Nothing here is a
// candidate, a corpus change, a gate change or a production change; phases 61, 62 and 64 stay frozen.
//
// Causality is limited by construction: zero-suffix terms are exact no-ops under serial accumulation, so under any
// order-preserving contraction the k-prefix output is the running sum after k terms and the first differing prefix
// names the term whose inclusion first exposes a compiled difference. Under legal reassociation the masked suffix
// changes which terms meet, the prefix output is not the original running sum, and a prefix difference does not
// identify the original instruction sequence or prove the original accumulator first differs at that term.
// fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim are always false.

// MARK: - Serial term order and derived inputs

public enum Q4PrefixProbeOrder {
    /// The x index read by serial term t of the production kernel (block t / 32, byte (t % 32) / 2, nibble t % 2).
    public static func xIndex(term t: Int) -> Int {
        let b = t / 32
        let r = t % 32
        let l = r / 2
        return b * 32 + (r % 2 == 0 ? l : 16 + l)
    }

    public static func termOrder(inDim: Int) -> [Int] { (0..<inDim).map { xIndex(term: $0) } }

    /// Prefix k of the original activation: original values at the first k terms' x indices, 0.0f elsewhere.
    public static func maskedInput(_ x: [Float], prefix k: Int) -> [Float]? {
        guard x.count > 0, x.count % 32 == 0, k >= 0, k <= x.count else { return nil }
        var out = [Float](repeating: 0, count: x.count)
        for t in 0..<k {
            let i = xIndex(term: t)
            out[i] = x[i]
        }
        return out
    }

    /// FP64 serial sum of the first k ORIGINAL terms in production order (exact half decode, double products).
    public static func prefixOracle(blocks: [UInt8], inDim: Int, x: [Float], prefix k: Int) -> Double? {
        guard inDim > 0, inDim % 32 == 0, k >= 0, k <= inDim, blocks.count >= inDim / 32 * 18, x.count >= inDim else {
            return nil
        }
        var sum = 0.0
        for t in 0..<k {
            let b = t / 32
            let r = t % 32
            let l = r / 2
            let o = b * 18
            let dbits = UInt16(blocks[o]) | (UInt16(blocks[o + 1]) << 8)
            let d = Double(Q4FirstDivergenceCPU.halfToFloat(dbits))
            let q = blocks[o + 2 + l]
            let nibble = r % 2 == 0 ? (q & 0x0F) : (q >> 4)
            let w = Double(nibble) - 8
            sum += d * w * Double(x[xIndex(term: t)])
        }
        return sum
    }
}

// MARK: - Report

public enum Q4PrefixProbeStatus: String, Codable, Sendable {
    case completed
    case notReproduced
    case probeContractFailed
    case unsupported
    case commandBufferError
    case notEnded
    case aborted
}

public struct Q4PrefixProbeResult: Codable, Equatable, Sendable {
    public let prefix: Int
    public let termIndex: Int?
    public let xIndexAdded: Int?
    public let locator: String?
    public let controlCompletion: String
    public let candidateCompletion: String
    public let controlHex: String?
    public let candidateHex: String?
    public let bitIdentical: Bool
    public let finite: Bool
    public let guardsIntact: Bool
    public let prefixOracle: Double?
    public let oracleAgreesWithMaskedOracle: Bool
    public let controlFP64Rel: Double?
    public let candidateFP64Rel: Double?
    public let fp64Pass: Bool
    public let passed: Bool

    public init(prefix: Int, termIndex: Int?, xIndexAdded: Int?, locator: String?, controlCompletion: String,
                candidateCompletion: String, controlHex: String?, candidateHex: String?, bitIdentical: Bool,
                finite: Bool, guardsIntact: Bool, prefixOracle: Double?, oracleAgreesWithMaskedOracle: Bool,
                controlFP64Rel: Double?, candidateFP64Rel: Double?, fp64Pass: Bool, passed: Bool) {
        self.prefix = prefix
        self.termIndex = termIndex
        self.xIndexAdded = xIndexAdded
        self.locator = locator
        self.controlCompletion = controlCompletion
        self.candidateCompletion = candidateCompletion
        self.controlHex = controlHex
        self.candidateHex = candidateHex
        self.bitIdentical = bitIdentical
        self.finite = finite
        self.guardsIntact = guardsIntact
        self.prefixOracle = prefixOracle
        self.oracleAgreesWithMaskedOracle = oracleAgreesWithMaskedOracle
        self.controlFP64Rel = controlFP64Rel
        self.candidateFP64Rel = candidateFP64Rel
        self.fp64Pass = fp64Pass
        self.passed = passed
    }
}

public struct Q4PrefixProbeModelResult: Codable, Equatable, Sendable {
    public let model: String
    public let prefixesMatchingControl: Int
    public let prefixesMatchingCandidate: Int
    public let firstPrefixDifferingFromControl: Int?
    public let firstPrefixDifferingFromCandidate: Int?
}

public struct Q4PrefixProbeReport: Codable {
    public private(set) var kind = Q4PrefixProbeDiagnostics.kind
    public private(set) var fullModelQualified = false
    public private(set) var gainQualified = false
    public private(set) var installable = false
    public private(set) var productionPerformanceClaim = false
    public private(set) var busyTimeClaim = false
    public private(set) var notes = Q4PrefixProbeDiagnostics.notes
    public private(set) var closedRecords = Q4PrefixProbeDiagnostics.closedRecords
    public private(set) var causalityStatement = Q4PrefixProbeDiagnostics.causalityStatement
    public var executableDigest: String? = nil
    public var controlFunction = Q4WeightStageEmitter.controlFunction
    public var controlSourceSha256: String? = nil
    public var registerKernelName: String? = nil
    public var registerSourceSha256: String? = nil
    public var registerCompileError: String? = nil
    public var declined: String? = nil
    public var hardware: Q4WeightStageHardware? = nil
    public var caseName = Q4FirstDivergenceDiagnostics.caseName
    public var inDim: Int? = nil
    public var outDim: Int? = nil
    public var blocksPerRow: Int? = nil
    public var termOrderXIndices: [Int] = []
    public var baselineControlHex: String? = nil
    public var baselineCandidateHex: String? = nil
    public var baselineControlCompletion: String? = nil
    public var baselineCandidateCompletion: String? = nil
    public var baselineGuardsIntact: Bool? = nil
    public var reproducedMismatch: Bool? = nil
    public var baselineControlMatchesPhase64Record: Bool? = nil
    public var baselineCandidateMatchesPhase64Record: Bool? = nil
    public var probes: [Q4PrefixProbeResult] = []
    public var allProbesPassed: Bool? = nil
    public var fullPrefixMatchesBaseline: Bool? = nil
    public var differingPrefixCount: Int? = nil
    public var firstDifferingPrefix: Int? = nil
    public var firstDifferingTermIndex: Int? = nil
    public var firstDifferingLocator: String? = nil
    public var lastIdenticalPrefixHex: String? = nil
    public var firstDifferingControlHex: String? = nil
    public var firstDifferingCandidateHex: String? = nil
    /// First differing prefix whose two outputs are not both signed zeros (00000000 / 80000000). The default math
    /// mode may drop the sign of a zero sum, so a zero-only difference stays in the probes but is never the headline.
    public var firstDifferingPrefixExcludingZeroPairs: Int? = nil
    public var cpuModels: [Q4PrefixProbeModelResult] = []
    public var commandBuffersCommitted = 0
    public var terminal = true
    public var error: String? = nil
    public var status: Q4PrefixProbeStatus = .aborted

    public init() {}
}

extension Q4PrefixProbeReport {
    func withContract() -> Q4PrefixProbeReport {
        var copy = self
        copy.fullModelQualified = false
        copy.gainQualified = false
        copy.installable = false
        copy.productionPerformanceClaim = false
        copy.busyTimeClaim = false
        return copy
    }
}

// MARK: - GPU helper: a derived activation buffer sharing the frozen weight buffer (originals untouched)

extension Q4WeightStageGPU {
    func makeProbeInput(base: Q4WeightStageBuffers, x: [Float]) throws -> Q4WeightStageBuffers {
        let g = Self.guardBytes
        guard x.count == base.inDim,
              let xb = device.makeBuffer(length: 2 * g + x.count * 4, options: .storageModeShared) else {
            throw Q4WeightStageError("probe input allocation failed")
        }
        memset(xb.contents(), Int32(Self.canaryByte), xb.length)
        x.withUnsafeBytes { raw in _ = memcpy(xb.contents().advanced(by: g), raw.baseAddress!, raw.count) }
        retained.append(xb)
        return Q4WeightStageBuffers(w: base.w, x: xb, weightBytes: base.weightBytes, inDim: base.inDim,
                                    outDim: base.outDim)
    }
}

// MARK: - Driver

public enum Q4PrefixProbeDiagnostics {
    public static let kind = "q4-prefix-probe-diagnostic-v1"
    public static let usage = "usage: q4-prefix-probe-diagnostic --report NEW_REPORT.json"
    /// Phase-64 recorded uninstrumented finals on the frozen row (comparison only; the run's own baseline governs).
    public static let phase64ControlHex = "bd6b6117"
    public static let phase64CandidateHex = "bd6b6116"
    public static let closedRecords: [String] = [
        "q4-packed-word-register-decode-component-v1 (phase 61): correctness-rejected, frozen",
        "q4-register-loop-decode-component-v1 (phase 62): correctness-rejected on 17 of 17 supported cases, frozen",
        "q4-first-divergence-diagnostic-v1 (phase 64): instrumented decoded-operand probes identical, uninstrumented "
            + "control bd6b6117 vs register bd6b6116, both production-shaped trace finals bd6b6116 failed original "
            + "calibration, original cause NOT ESTABLISHED, frozen",
    ]
    public static let causalityStatement = "A prefix output difference at prefix k shows that the two unmodified "
        + "compiled kernels compute the first k terms (with zero suffix) differently. Under order-preserving "
        + "contraction the k-prefix output is the running sum after k terms, so k names the term whose inclusion "
        + "first exposes the difference. Under legal compiler reassociation the masked suffix changes which terms "
        + "meet, the prefix output is not the original running sum, and this result does not prove where the "
        + "original runtime accumulator first differs nor identify either kernel's instruction sequence."
    public static let notes: [String] = [
        "Diagnostic input perturbation only: no candidate, no corpus, gate, flag, source or default change; "
            + "fullModelQualified, gainQualified, installable, productionPerformanceClaim and busyTimeClaim are false.",
        "Both pipelines are compiled once from their unmodified sources with options nil and never instrumented; the "
            + "frozen min-1x64 weights, launch geometry and buffers are reused; only the activation vector is masked.",
        "Prefix k keeps the original x at the x indices of the first k production serial terms (block b, byte l, low "
            + "nibble then high nibble) and 0.0f elsewhere; prefix 64 is the original input and must bit-match the "
            + "in-session baseline of the same pipelines.",
        "Every probe is checked for completion, finite outputs, intact 256-byte guards, the existing 1e-4 FP64 "
            + "contract on the masked input, and agreement between the explicit FP64 prefix oracle over the original "
            + "terms and the masked-input oracle.",
        "CPU serial accumulation models are compared per prefix as consistent-with hypotheses only.",
        "One-hot inputs are omitted by predeclaration to keep the probe set at the 65 fixed prefixes; a single-term "
            + "output would test product association and rounding, not accumulation order. Differences where both "
            + "outputs are signed zeros are reported separately because the default math mode may drop the sign of a "
            + "zero sum. No timing, no GGUF, no hashing, no instrumentation.",
    ]

    public struct Options: Equatable, Sendable {
        public let report: String
    }

    public static func parseOptions(_ argv: [String]) throws -> Options {
        guard argv.count == 2, argv[0] == "--report", !argv[1].isEmpty, !argv[1].hasPrefix("--") else {
            throw Q4WeightStageError(usage)
        }
        return Options(report: argv[1])
    }

    public static func encodeReport(_ r: Q4PrefixProbeReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(r.withContract())
    }

    public static func exitCode(_ s: Q4PrefixProbeStatus) -> Int32 {
        switch s {
        case .completed: return 0
        case .notReproduced: return 7
        case .probeContractFailed: return 3
        case .unsupported: return 2
        case .commandBufferError: return 4
        case .notEnded: return 5
        case .aborted: return 1
        }
    }

    /// Run integrity only; the first-difference finding is a report field.
    public static func finalStatus(_ r: Q4PrefixProbeReport) -> Q4PrefixProbeStatus {
        if !r.terminal { return .notEnded }
        if r.error != nil { return .aborted }
        guard r.hardware != nil, r.registerCompileError == nil, r.declined == nil else { return .unsupported }
        if r.baselineControlCompletion == "ERROR" || r.baselineCandidateCompletion == "ERROR"
            || r.probes.contains(where: { $0.controlCompletion == "ERROR" || $0.candidateCompletion == "ERROR" }) {
            return .commandBufferError
        }
        // A step that was never committed is an aborted run, never a reproduction or probe finding.
        let completions = [r.baselineControlCompletion, r.baselineCandidateCompletion].compactMap { $0 }
            + r.probes.flatMap { [$0.controlCompletion, $0.candidateCompletion] }
        if completions.contains(where: { $0 != "COMPLETED" }) { return .aborted }
        guard r.reproducedMismatch == true, r.fullPrefixMatchesBaseline == true else { return .notReproduced }
        guard r.allProbesPassed == true else { return .probeContractFailed }
        return .completed
    }

    /// Pure summary: the first prefix whose control and candidate bits differ, among completed probes.
    public static func summarize(_ probes: [Q4PrefixProbeResult]) -> (first: Q4PrefixProbeResult?, count: Int) {
        let differing = probes.filter {
            $0.controlCompletion == "COMPLETED" && $0.candidateCompletion == "COMPLETED" && !$0.bitIdentical
        }
        return (differing.min(by: { $0.prefix < $1.prefix }), differing.count)
    }

    public static func run(executableDigest: String, emit: (String) -> Void) -> Q4PrefixProbeReport {
        let counter = Q4WeightStageDiagnostics.CommitCounter()
        var report = runCore(executableDigest: executableDigest, counter: counter, emit: emit)
        report.commandBuffersCommitted = counter.committed
        return report
    }

    static func runCore(executableDigest: String, counter: Q4WeightStageDiagnostics.CommitCounter,
                        emit: (String) -> Void) -> Q4PrefixProbeReport {
        var report = Q4PrefixProbeReport()
        report.executableDigest = executableDigest
        report.controlSourceSha256 = Q4WeightStageEmitter.sourceDigest(Gemma4Runner.shaderSource)
        func finish(_ r: inout Q4PrefixProbeReport) -> Q4PrefixProbeReport {
            r.status = finalStatus(r)
            return r
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            report.error = "no Metal device"
            return finish(&report)
        }
        let gpu: Q4WeightStageGPU
        do { gpu = try Q4WeightStageGPU(device: device) } catch {
            report.error = String(describing: error)
            return finish(&report)
        }
        defer { counter.committed = gpu.commandBuffers.count }
        report.hardware = Q4WeightStageHardware(
            device: device.name, controlThreadExecutionWidth: gpu.control.threadExecutionWidth,
            controlMaxTotalThreads: gpu.control.maxTotalThreadsPerThreadgroup,
            deviceMaxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength,
            deviceMaxThreadsPerThreadgroupWidth: device.maxThreadsPerThreadgroup.width,
            supportsAtStageBoundary: device.supportsCounterSampling(.atStageBoundary),
            supportsAtDispatchBoundary: device.supportsCounterSampling(.atDispatchBoundary),
            timestampCounterSetFound: false, sampleBufferCreated: false)
        emit("device " + device.name + ": control width " + String(gpu.control.threadExecutionWidth) + ", no sampling")

        guard let c = Q4FirstDivergenceDiagnostics.frozenCase(), c.outDim == 1 else {
            report.error = "frozen case not found or not single-row"
            return finish(&report)
        }
        report.inDim = c.inDim
        report.outDim = c.outDim
        report.blocksPerRow = c.blocksPerRow
        report.termOrderXIndices = Q4PrefixProbeOrder.termOrder(inDim: c.inDim)

        let descriptor: Q4RegisterLoopDescriptor
        let registerPipe: MTLComputePipelineState
        do {
            descriptor = try Q4RegisterLoopDescriptor.derive(executionWidth: gpu.control.threadExecutionWidth)
            let source = try Q4RegisterLoopEmitter.emit(descriptor)
            report.registerKernelName = descriptor.kernelName
            report.registerSourceSha256 = Q4WeightStageEmitter.sourceDigest(source)
            let library = try device.makeLibrary(source: source, options: nil)
            guard let fn = library.makeFunction(name: descriptor.kernelName) else {
                throw Q4WeightStageError("register function missing")
            }
            registerPipe = try device.makeComputePipelineState(function: fn)
        } catch {
            report.registerCompileError = "phase-62 register source (options nil): " + String(describing: error)
            return finish(&report)
        }
        let g = Q4WeightStageGPU.guardBytes
        if let reason = Q4RegisterLoopPlan.declineReason(
            inDim: c.inDim, outDim: c.outDim, weightOffset: g, weightLength: g + c.blocks.count,
            descriptor: descriptor, executionWidth: registerPipe.threadExecutionWidth,
            pipelineMaxThreads: registerPipe.maxTotalThreadsPerThreadgroup,
            staticThreadgroupBytes: registerPipe.staticThreadgroupMemoryLength) {
            report.declined = reason
            return finish(&report)
        }
        let controlLaunch = Q4WeightStagePlan.controlLaunch(outDim: c.outDim,
                                                            pipelineMaxThreads: gpu.control.maxTotalThreadsPerThreadgroup)
        let registerLaunch = Q4RegisterLoopPlan.launch(outDim: c.outDim, descriptor: descriptor)

        /// One control-then-candidate pair on one input; two sequential command buffers, bounded by submit.
        struct Pair {
            let ctl: Q4WeightStageOutcome
            let cand: Q4WeightStageOutcome?
            let ctlBits: [UInt32]
            let candBits: [UInt32]
            let guards: Bool
            var terminal: Bool { ctl.terminal && (cand?.terminal ?? true) }
        }
        func runPair(_ b: Q4WeightStageBuffers) -> Pair? {
            guard let yC = try? gpu.makeOutput(outDim: 1), let yK = try? gpu.makeOutput(outDim: 1) else { return nil }
            let ctl = gpu.runSingle(pipeline: gpu.control, launch: controlLaunch, b: b, y: yC)
            if !ctl.terminal { return Pair(ctl: ctl, cand: nil, ctlBits: [], candBits: [], guards: false) }
            let cand = gpu.runSingle(pipeline: registerPipe, launch: registerLaunch, b: b, y: yK)
            // A candidate whose completion was not observed may still be executing: read nothing.
            if !cand.terminal { return Pair(ctl: ctl, cand: cand, ctlBits: [], candBits: [], guards: false) }
            let ctlBits = ctl.completion == "COMPLETED" ? gpu.outputBits(yC, outDim: 1) : []
            let candBits = cand.completion == "COMPLETED" ? gpu.outputBits(yK, outDim: 1) : []
            let guards = gpu.guardsIntact(b.w, payloadBytes: b.weightBytes) && gpu.guardsIntact(b.x, payloadBytes: c.inDim * 4)
                && gpu.guardsIntact(yC, payloadBytes: 4) && gpu.guardsIntact(yK, payloadBytes: 4)
            return Pair(ctl: ctl, cand: cand, ctlBits: ctlBits, candBits: candBits, guards: guards)
        }

        // Baseline: the frozen input on the same two pipelines.
        let base: Q4WeightStageBuffers
        do { base = try gpu.makeBuffers(c) } catch {
            report.error = "allocation: " + String(describing: error)
            return finish(&report)
        }
        guard let baseline = runPair(base) else {
            report.error = "baseline output allocation failed"
            return finish(&report)
        }
        if !baseline.terminal {
            report.terminal = false
            report.error = "baseline: completion not observed"
            return finish(&report)
        }
        report.baselineControlCompletion = baseline.ctl.completion
        report.baselineCandidateCompletion = baseline.cand?.completion
        report.baselineGuardsIntact = baseline.guards
        report.baselineControlHex = baseline.ctlBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
        report.baselineCandidateHex = baseline.candBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
        if baseline.ctlBits.count == 1, baseline.candBits.count == 1 {
            let finite = Float(bitPattern: baseline.ctlBits[0]).isFinite && Float(bitPattern: baseline.candBits[0]).isFinite
            if !finite || !baseline.guards {
                report.error = "baseline not trustworthy: finite=" + String(finite) + " guards=" + String(baseline.guards)
                return finish(&report)
            }
            report.reproducedMismatch = baseline.ctlBits[0] != baseline.candBits[0]
            report.baselineControlMatchesPhase64Record = report.baselineControlHex == phase64ControlHex
            report.baselineCandidateMatchesPhase64Record = report.baselineCandidateHex == phase64CandidateHex
        }
        emit("baseline: control " + (report.baselineControlHex ?? "n/a") + " candidate "
             + (report.baselineCandidateHex ?? "n/a") + " reproduced="
             + (report.reproducedMismatch.map { String($0) } ?? "n/a"))
        guard report.reproducedMismatch == true else { return finish(&report) }

        // Prefix probes k = 0...inDim on the same pipelines; only the activation is masked.
        var allPassed = true
        var controlByPrefix = [UInt32?]()
        var candidateByPrefix = [UInt32?]()
        for k in 0...c.inDim {
            guard let masked = Q4PrefixProbeOrder.maskedInput(c.x, prefix: k) else {
                report.error = "mask construction failed at prefix " + String(k)
                return finish(&report)
            }
            let b: Q4WeightStageBuffers
            do { b = try gpu.makeProbeInput(base: base, x: masked) } catch {
                report.error = "allocation: " + String(describing: error)
                return finish(&report)
            }
            guard let pair = runPair(b) else {
                report.error = "probe output allocation failed at prefix " + String(k)
                return finish(&report)
            }
            if !pair.terminal {
                report.terminal = false
                report.error = "prefix " + String(k) + ": completion not observed"
                return finish(&report)
            }
            let ctlHex = pair.ctlBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
            let candHex = pair.candBits.first.map { Q4FirstDivergenceLayout.hex8($0) }
            let completed = pair.ctl.completion == "COMPLETED" && pair.cand?.completion == "COMPLETED"
            let finite = completed && pair.ctlBits.count == 1 && pair.candBits.count == 1
                && Float(bitPattern: pair.ctlBits[0]).isFinite && Float(bitPattern: pair.candBits[0]).isFinite
            let identical = finite && pair.ctlBits[0] == pair.candBits[0]
            let prefixOracle = Q4PrefixProbeOrder.prefixOracle(blocks: c.blocks, inDim: c.inDim, x: c.x, prefix: k)
            var ctlRel: Double? = nil
            var candRel: Double? = nil
            var agrees = false
            if let ref = try? Q4Oracle.matvecFP64(blocks: c.blocks[...], outDim: 1, inDim: c.inDim, x: masked),
               ref.count == 1 {
                ctlRel = Q4WeightStageDiagnostics.worstRel(pair.ctlBits, ref).worst
                candRel = Q4WeightStageDiagnostics.worstRel(pair.candBits, ref).worst
                if let p = prefixOracle { agrees = abs(p - ref[0]) <= 1e-12 * max(1.0, abs(ref[0])) }
            }
            let fp64Pass = (ctlRel.map { $0 < Q4WeightStageDiagnostics.correctnessContract } ?? false)
                && (candRel.map { $0 < Q4WeightStageDiagnostics.correctnessContract } ?? false)
            let passed = completed && finite && pair.guards && fp64Pass && agrees
            if !passed { allPassed = false }
            controlByPrefix.append(finite ? pair.ctlBits[0] : nil)
            candidateByPrefix.append(finite ? pair.candBits[0] : nil)
            report.probes.append(Q4PrefixProbeResult(
                prefix: k, termIndex: k > 0 ? k - 1 : nil, xIndexAdded: k > 0 ? Q4PrefixProbeOrder.xIndex(term: k - 1) : nil,
                locator: k > 0 ? Q4FirstDivergenceLayout.locator(stage: "sum", index: k - 1) : nil,
                controlCompletion: pair.ctl.completion, candidateCompletion: pair.cand?.completion ?? "NOT_COMMITTED",
                controlHex: ctlHex, candidateHex: candHex, bitIdentical: identical, finite: finite,
                guardsIntact: pair.guards, prefixOracle: prefixOracle, oracleAgreesWithMaskedOracle: agrees,
                controlFP64Rel: ctlRel, candidateFP64Rel: candRel, fp64Pass: fp64Pass, passed: passed))
            if k % 16 == 0 || !identical {
                emit("prefix " + String(k) + ": control " + (ctlHex ?? "n/a") + " candidate " + (candHex ?? "n/a")
                     + (identical ? " identical" : " DIFFER"))
            }
        }
        report.allProbesPassed = allPassed
        if let full = report.probes.last, full.prefix == c.inDim {
            report.fullPrefixMatchesBaseline = full.controlHex == report.baselineControlHex
                && full.candidateHex == report.baselineCandidateHex && full.controlHex != nil
        }
        let summary = summarize(report.probes)
        report.differingPrefixCount = summary.count
        report.firstDifferingPrefixExcludingZeroPairs = report.probes.first(where: { p in
            let zero: (String?) -> Bool = { $0 == "00000000" || $0 == "80000000" }
            return p.controlCompletion == "COMPLETED" && p.candidateCompletion == "COMPLETED" && !p.bitIdentical
                && !(zero(p.controlHex) && zero(p.candidateHex))
        })?.prefix
        if let f = summary.first {
            report.firstDifferingPrefix = f.prefix
            report.firstDifferingTermIndex = f.termIndex
            report.firstDifferingLocator = f.locator
            report.firstDifferingControlHex = f.controlHex
            report.firstDifferingCandidateHex = f.candidateHex
            if f.prefix > 0, let prev = report.probes.first(where: { $0.prefix == f.prefix - 1 }), prev.bitIdentical {
                report.lastIdenticalPrefixHex = prev.controlHex
            }
        }

        // CPU serial hypotheses per prefix (running sum after k terms; prefix 0 is +0).
        for model in Q4FirstDivergenceCPU.Model.allCases {
            guard let sums = Q4FirstDivergenceCPU.runningSums(model: model, blocks: c.blocks, inDim: c.inDim, x: c.x) else {
                continue
            }
            var mc = 0
            var mk = 0
            var fc: Int? = nil
            var fk: Int? = nil
            for k in 0...c.inDim where k < controlByPrefix.count {
                let predicted: UInt32 = k == 0 ? 0 : sums[k - 1]
                if controlByPrefix[k] == predicted { mc += 1 } else if fc == nil { fc = k }
                if candidateByPrefix[k] == predicted { mk += 1 } else if fk == nil { fk = k }
            }
            report.cpuModels.append(Q4PrefixProbeModelResult(
                model: model.rawValue, prefixesMatchingControl: mc, prefixesMatchingCandidate: mk,
                firstPrefixDifferingFromControl: fc, firstPrefixDifferingFromCandidate: fk))
        }
        emit("first differing prefix: " + (report.firstDifferingPrefix.map { String($0) } ?? "none") + "; "
             + String(summary.count) + " of " + String(report.probes.count) + " prefixes differ; limited causality applies")
        return finish(&report)
    }
}
