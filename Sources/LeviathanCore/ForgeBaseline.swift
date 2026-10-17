import Foundation

/// One evaluated candidate of a baseline-first selection (real GPU runs of
/// the existing Metal kernels). role: "baseline" or "optimization".
public struct ForgeQualifiedResult: Codable, Equatable, Sendable {
    public let candidate: ForgeCandidate
    public let role: String
    /// Passed qualification on every input/reference AND was measured.
    public let qualified: Bool
    public let maxAbsDiff: Double?
    /// Median command-buffer GPU time (us), warmup excluded.
    public let medianGpuMicros: Double?
    public let timedRuns: Int
    /// Set only when the diagnostic switch perturbed this optimization.
    public let diagnosticForcedFailure: Bool
    public let note: String
}

/// Persisted baseline-first selection, re-validated before every reuse.
public struct ForgeBaselineSelection: Codable, Equatable, Sendable {
    public let baseline: ForgeQualifiedResult
    public let optimizations: [ForgeQualifiedResult]
    public let selected: ForgeCandidate
    public let selectedRole: String
    public let selectedMedianGpuMicros: Double
    public let minRelativeGain: Double
    public let qualificationInputs: Int
    public let diagnosticFailOptimizations: Bool
    public let decision: String
}

/// Narrow baseline-first extension of Forge (Forge.forge is unchanged):
/// 1. the unfused baseline is verified FIRST against scalar + Accelerate
///    (+ source framework on input 0) references; failure = no selection;
/// 2. every other legal candidate (Forge.candidates, device-measured cap)
///    is qualified the same way, then benchmarked like Forge.forge;
/// 3. an optimization is selected only if qualified and its median beats
///    the baseline by >= minRelativeGain, else explicit baseline fallback.
/// diagnosticFailOptimizations perturbs output[0] of OPTIMIZATION runs
/// (never the baseline) after the real GPU run, to exercise fallback.
public enum ForgeBaseline {
    public static let baselineRole = "baseline"
    public static let optimizationRole = "optimization"
    public static let minRelativeGain = 0.02
    public static let absTolerance: Float = 1e-4
    public static let relTolerance: Float = 1e-4
    public static let maxQualificationInputs = 8
    static let maxNoteLength = 256

    public static func minDenseOutDim(_ program: LoweredProgram) -> Int {
        var smallest = Int.max
        for step in program.steps {
            if case let .dense(_, _, outDim, _, _, _, _, _) = step { smallest = min(smallest, outDim) }
        }
        return smallest == Int.max ? 1 : smallest
    }

    public static func baselineCandidate(_ program: LoweredProgram,
                                         maxThreadsPerThreadgroup: Int) -> ForgeCandidate {
        ForgeCandidate(threadgroupWidth: max(1, min(32, maxThreadsPerThreadgroup, minDenseOutDim(program))),
                       fuseDenseRelu: false)
    }

    public static func legalCandidates(_ program: LoweredProgram,
                                       maxThreadsPerThreadgroup: Int) -> [ForgeCandidate] {
        Forge.candidates(fusable: MetalBackend.isDenseReluFusable(program),
                         maxThreadsPerThreadgroup: maxThreadsPerThreadgroup,
                         minDenseOutDim: minDenseOutDim(program))
    }

    /// Kernels one run dispatches (mirrors MetalBackend.Prepared.run).
    public static func kernelSequence(_ program: LoweredProgram, candidate: ForgeCandidate) -> [String] {
        var names: [String] = []
        let steps = program.steps
        var index = 0
        while index < steps.count {
            switch steps[index] {
            case let .dense(_, _, _, _, outBuf, _, _, _):
                if candidate.fuseDenseRelu, index + 1 < steps.count,
                   case let .relu(_, buffer) = steps[index + 1], buffer == outBuf {
                    names.append("leviathan_dense_relu")
                    index += 2
                    continue
                }
                names.append("leviathan_dense")
            case .relu:
                names.append("leviathan_relu")
            }
            index += 1
        }
        return names
    }

    public static func select(program: LoweredProgram, qualificationInputs: [[Float]],
                              sourceReference: [Float]?, diagnosticFailOptimizations: Bool) throws
            -> (selection: ForgeBaselineSelection, hardware: ForgeHardwareProfile) {
        guard !qualificationInputs.isEmpty, qualificationInputs.count <= maxQualificationInputs else {
            throw ForgeError.invalidBatch("qualification input count outside 1...\(maxQualificationInputs)")
        }
        var preparedRuns: [MetalBackend.Prepared] = []
        var referenceSets: [[[Float]]] = []
        for (index, input) in qualificationInputs.enumerated() {
            let preparedRun = try MetalBackend.Prepared(program: program, batch: 1, input: input)
            preparedRuns.append(preparedRun)
            let scalar = try ScalarBackend.run(program, batch: 1, input: input)
            let accelerate = try AccelerateBackend.run(program, batch: 1, input: input)
            var references = [scalar, accelerate]
            if index == 0, let source = sourceReference {
                guard source.count == program.outputDim, source.allSatisfy({ $0.isFinite }) else {
                    throw ForgeError.invalidGraph("source reference has wrong size or non-finite values")
                }
                references.append(source)
            }
            referenceSets.append(references)
        }
        let prepared = preparedRuns
        let allReferences = referenceSets
        let bench = prepared[0]
        let hardware = ForgeHardwareProfile(
            deviceName: bench.deviceName,
            maxTotalThreadsPerThreadgroup: bench.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: bench.threadExecutionWidth,
            recommendedMaxWorkingSetSize: bench.recommendedMaxWorkingSetSize)
        let cap = hardware.maxTotalThreadsPerThreadgroup
        let baseline = baselineCandidate(program, maxThreadsPerThreadgroup: cap)

        func qualify(_ candidate: ForgeCandidate, forceFailure: Bool) -> (ok: Bool, maxAbsDiff: Double?, note: String) {
            var worst: Float = 0
            for (k, preparedRun) in prepared.enumerated() {
                var output: [Float]
                do {
                    output = try preparedRun.run(threadgroupWidth: candidate.threadgroupWidth,
                                                 fuseDenseRelu: candidate.fuseDenseRelu).output
                } catch {
                    return (false, nil, clip("execution failed: \(error)"))
                }
                if forceFailure, !output.isEmpty { output[0] += max(1, abs(output[0])) }
                for reference in allReferences[k] {
                    guard output.count == reference.count else { return (false, nil, "output size mismatch") }
                    for i in output.indices {
                        guard output[i].isFinite, reference[i].isFinite else {
                            return (false, nil, "non-finite value on qualification input \(k)")
                        }
                        let diff = abs(output[i] - reference[i])
                        worst = max(worst, diff)
                        if diff > absTolerance + relTolerance * abs(reference[i]) {
                            return (false, Double(diff), "qualification input \(k) element \(i): diff \(diff) exceeds tolerance")
                        }
                    }
                }
            }
            return (true, Double(worst), "qualified on \(prepared.count) inputs")
        }

        func measure(_ candidate: ForgeCandidate) -> (median: Double?, runs: Int, note: String) {
            var times: [Double] = []
            var spent = 0.0
            for iteration in 0..<(Forge.warmupRuns + Forge.timedRuns) {
                let micros: Double
                do {
                    micros = try bench.run(threadgroupWidth: candidate.threadgroupWidth,
                                           fuseDenseRelu: candidate.fuseDenseRelu).gpuMicros
                } catch {
                    return (nil, 0, clip("benchmark failed: \(error)"))
                }
                guard micros.isFinite, micros > 0 else { return (nil, 0, "GPU timestamps unavailable") }
                if iteration >= Forge.warmupRuns {
                    times.append(micros)
                    spent += micros
                    if spent > Forge.budgetMicros { break }
                }
            }
            guard !times.isEmpty else { return (nil, 0, "no timed runs") }
            let sorted = times.sorted()
            let mid = sorted.count / 2
            let median = sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
            return (median, times.count, "\(times.count) timed runs")
        }

        let baselineCheck = qualify(baseline, forceFailure: false)
        guard baselineCheck.ok else {
            throw ForgeError.noValidCandidate(rejected: ["baseline \(label(baseline)): \(baselineCheck.note)"])
        }
        let baselineTiming = measure(baseline)
        guard let baselineMedian = baselineTiming.median else {
            throw ForgeError.noValidCandidate(rejected: ["baseline \(label(baseline)): \(baselineTiming.note)"])
        }
        let baselineResult = ForgeQualifiedResult(
            candidate: baseline, role: baselineRole, qualified: true, maxAbsDiff: baselineCheck.maxAbsDiff,
            medianGpuMicros: baselineMedian, timedRuns: baselineTiming.runs, diagnosticForcedFailure: false,
            note: clip("verified baseline: \(baselineCheck.note); \(baselineTiming.note)"))

        var optimizations: [ForgeQualifiedResult] = []
        for candidate in legalCandidates(program, maxThreadsPerThreadgroup: cap) where candidate != baseline {
            let check = qualify(candidate, forceFailure: diagnosticFailOptimizations)
            guard check.ok else {
                let prefix = diagnosticFailOptimizations ? "diagnostic forced failure: " : "rejected: "
                optimizations.append(ForgeQualifiedResult(
                    candidate: candidate, role: optimizationRole, qualified: false, maxAbsDiff: check.maxAbsDiff,
                    medianGpuMicros: nil, timedRuns: 0, diagnosticForcedFailure: diagnosticFailOptimizations,
                    note: clip(prefix + check.note)))
                continue
            }
            let timing = measure(candidate)
            optimizations.append(ForgeQualifiedResult(
                candidate: candidate, role: optimizationRole, qualified: timing.median != nil,
                maxAbsDiff: check.maxAbsDiff, medianGpuMicros: timing.median, timedRuns: timing.runs,
                diagnosticForcedFailure: false, note: clip("\(check.note); \(timing.note)")))
        }

        let best = optimizations.filter { $0.qualified }
            .min { ($0.medianGpuMicros ?? .infinity) < ($1.medianGpuMicros ?? .infinity) }
        let threshold = baselineMedian * (1 - minRelativeGain)
        var selected = baseline
        var selectedRole = baselineRole
        var selectedMedian = baselineMedian
        var decision: String
        if let best, let bestMedian = best.medianGpuMicros, bestMedian < threshold {
            selected = best.candidate
            selectedRole = optimizationRole
            selectedMedian = bestMedian
            decision = "optimization selected: \(label(best.candidate)) \(fmt(bestMedian)) us vs verified baseline \(fmt(baselineMedian)) us"
        } else if optimizations.isEmpty {
            decision = "baseline: no optimization candidate is legal for this graph and device"
        } else if let best, let bestMedian = best.medianGpuMicros {
            decision = "baseline fallback: best qualified optimization \(label(best.candidate)) \(fmt(bestMedian)) us is not at least \(fmt(minRelativeGain * 100))% faster than verified baseline \(fmt(baselineMedian)) us"
        } else {
            decision = "baseline fallback: none of \(optimizations.count) optimization candidate(s) passed qualification" + (diagnosticFailOptimizations ? " (diagnostic forced failure)" : "")
        }
        decision = String(decision.prefix(512))
        let selection = ForgeBaselineSelection(
            baseline: baselineResult, optimizations: optimizations, selected: selected,
            selectedRole: selectedRole, selectedMedianGpuMicros: selectedMedian,
            minRelativeGain: minRelativeGain, qualificationInputs: prepared.count,
            diagnosticFailOptimizations: diagnosticFailOptimizations, decision: decision)
        return (selection, hardware)
    }

    /// Re-validates a persisted selection against the regenerated legal set
    /// for this program on this device. Never re-tunes.
    public static func validateSelection(_ selection: ForgeBaselineSelection, program: LoweredProgram,
                                         hardware: ForgeHardwareProfile) throws {
        func bad(_ message: String) -> ForgeError { .invalidPlan("selection: " + message) }
        func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-9 * max(1, abs(b)) }
        let cap = hardware.maxTotalThreadsPerThreadgroup
        guard cap > 0, cap <= 1 << 16 else { throw bad("device threadgroup cap out of bounds") }
        let baseline = baselineCandidate(program, maxThreadsPerThreadgroup: cap)
        let expected = Set(legalCandidates(program, maxThreadsPerThreadgroup: cap)).subtracting([baseline])
        guard selection.minRelativeGain == minRelativeGain,
              selection.qualificationInputs >= 1, selection.qualificationInputs <= maxQualificationInputs,
              selection.decision.count <= 512 else { throw bad("malformed selection header") }
        let record = selection.baseline
        guard record.candidate == baseline, record.role == baselineRole, record.qualified,
              !record.diagnosticForcedFailure, let baselineMedian = record.medianGpuMicros,
              baselineMedian.isFinite, baselineMedian > 0 else {
            throw bad("baseline record is not the verified unfused baseline")
        }
        let candidates = selection.optimizations.map { $0.candidate }
        guard candidates.count == expected.count, Set(candidates) == expected,
              selection.optimizations.allSatisfy({ $0.role == optimizationRole }) else {
            throw bad("optimization records do not match the legal candidate set")
        }
        for entry in [record] + selection.optimizations {
            guard entry.note.count <= maxNoteLength, entry.timedRuns >= 0, entry.timedRuns <= Forge.timedRuns else {
                throw bad("malformed record")
            }
            if let m = entry.medianGpuMicros, !(m.isFinite && m > 0) { throw bad("invalid median") }
            if let d = entry.maxAbsDiff, !(d.isFinite && d >= 0) { throw bad("invalid max abs diff") }
            if entry.qualified, entry.medianGpuMicros == nil || entry.diagnosticForcedFailure {
                throw bad("qualified record without measurement")
            }
            if entry.diagnosticForcedFailure, !selection.diagnosticFailOptimizations || entry.qualified {
                throw bad("inconsistent diagnostic marker")
            }
        }
        if selection.diagnosticFailOptimizations, selection.optimizations.contains(where: { $0.qualified }) {
            throw bad("diagnostic bundle claims a qualified optimization")
        }
        let threshold = baselineMedian * (1 - minRelativeGain)
        let bestMedian = selection.optimizations.compactMap { $0.qualified ? $0.medianGpuMicros : nil }.min()
        switch selection.selectedRole {
        case baselineRole:
            guard selection.selected == baseline, close(selection.selectedMedianGpuMicros, baselineMedian) else {
                throw bad("baseline selection does not match the baseline record")
            }
            if let best = bestMedian, best < threshold * (1 - 1e-9) {
                throw bad("baseline recorded although a qualified optimization won")
            }
        case optimizationRole:
            guard !selection.diagnosticFailOptimizations, selection.selected != baseline,
                  let chosen = selection.optimizations.first(where: { $0.candidate == selection.selected }),
                  chosen.qualified, let median = chosen.medianGpuMicros,
                  close(selection.selectedMedianGpuMicros, median), median < threshold * (1 + 1e-9),
                  let best = bestMedian, median <= best * (1 + 1e-9) else {
                throw bad("optimization selection is not the best qualified measured win over the baseline")
            }
        default:
            throw bad("unknown selected role")
        }
    }

    static func clip(_ text: String) -> String { String(text.prefix(maxNoteLength)) }
    static func label(_ c: ForgeCandidate) -> String {
        "width \(c.threadgroupWidth) " + (c.fuseDenseRelu ? "fused dense+relu" : "unfused")
    }
    static func fmt(_ value: Double) -> String { String(format: "%.3f", value) }
}
