import Foundation

/// Baseline-first pass selection for a DAGProgram on DAGMetalBackend. Every candidate must pass
/// correctness checks before it is measured. It follows the ForgeBaseline policy with stricter inputs:
/// - Exactly 4 qualification inputs (the example plus 3 seeded inputs).
/// - A required, finite source-framework reference for the example. It is checked first.
/// - Scalar and Accelerate references on all 4 inputs.
/// - The unfused baseline is verified and measured first. If it fails, selection stops.
/// - An optimization is chosen only if it is qualified, measured, and more than 2% faster.
/// Candidates come only from the validated graph and the measured device, never from a model
/// name. kernelSequence is what the host expects from safeFusionPairs. It is not a hardware trace.
public enum DAGForge {
    public static let qualificationInputCount = 4
    public static let maxDecisionLength = 512
    public static let baselineRole = ForgeBaseline.baselineRole
    public static let optimizationRole = ForgeBaseline.optimizationRole
    public static let minRelativeGain = ForgeBaseline.minRelativeGain

    private struct RunMismatch: Error, CustomStringConvertible { let description: String }
    static func clip(_ text: String) -> String { ForgeBaseline.clip(text) }

    public static func minDenseOutDim(_ program: DAGProgram) -> Int {
        program.layouts.values.map { $0.outDim }.min() ?? 1
    }

    public static func baselineCandidate(_ program: DAGProgram, maxThreadsPerThreadgroup: Int) -> ForgeCandidate {
        ForgeCandidate(threadgroupWidth: max(1, min(32, maxThreadsPerThreadgroup, minDenseOutDim(program))),
                       fuseDenseRelu: false)
    }

    public static func legalCandidates(_ program: DAGProgram, maxThreadsPerThreadgroup: Int) -> [ForgeCandidate] {
        Forge.candidates(fusable: !program.safeFusionPairs.isEmpty,
                         maxThreadsPerThreadgroup: maxThreadsPerThreadgroup,
                         minDenseOutDim: minDenseOutDim(program))
    }

    /// The kernels one run should dispatch, in node order. Computed on the host, not traced.
    public static func kernelSequence(_ program: DAGProgram, candidate: ForgeCandidate) -> [String] {
        let pairs = candidate.fuseDenseRelu ? program.safeFusionPairs : []
        let fusedDense = Set(pairs.map { $0.dense }), fusedRelu = Set(pairs.map { $0.relu })
        return program.graph.nodes.compactMap { node -> String? in
            switch node.op {
            case .dense: return fusedDense.contains(node.id) ? "leviathan_dense_relu" : "leviathan_dense"
            case .relu: return fusedRelu.contains(node.id) ? nil : "leviathan_dag_relu"
            case .add: return "leviathan_dag_add"
            }
        }
    }

    /// Returns nil when every value is finite and within abs 1e-4 + rel 1e-4 of the reference.
    static func mismatch(_ output: [Float], _ reference: [Float], worst: inout Float) -> String? {
        guard output.count == reference.count else { return "output size mismatch" }
        for i in output.indices {
            guard output[i].isFinite, reference[i].isFinite else { return "non-finite value at element \(i)" }
            let diff = abs(output[i] - reference[i])
            worst = max(worst, diff)
            if diff > ForgeBaseline.absTolerance + ForgeBaseline.relTolerance * abs(reference[i]) {
                return "element \(i): diff \(diff) exceeds tolerance"
            }
        }
        return nil
    }

    public static func select(program: DAGProgram, qualificationInputs: [[Float]], sourceReference: [Float]?,
                              diagnosticFailOptimizations: Bool) throws
            -> (selection: ForgeBaselineSelection, hardware: ForgeHardwareProfile) {
        guard qualificationInputs.count == qualificationInputCount else {
            throw ForgeError.invalidBatch("exactly \(qualificationInputCount) qualification inputs required (example + 3 seeded), got \(qualificationInputs.count)")
        }
        for (k, input) in qualificationInputs.enumerated() {
            guard input.count == program.graph.inputDim, input.allSatisfy({ $0.isFinite }) else {
                throw ForgeError.invalidBatch("qualification input \(k) has wrong size or non-finite values")
            }
        }
        guard let source = sourceReference, source.count == program.outputDim,
              source.allSatisfy({ $0.isFinite }) else {
            throw ForgeError.invalidGraph("source reference required: \(program.outputDim) finite values for the example input")
        }
        var references: [[[Float]]] = []
        for (k, input) in qualificationInputs.enumerated() {
            let scalar = try DAGScalarBackend.run(program, input: input)
            let accelerate = try DAGAccelerateBackend.run(program, input: input)
            if k == 0 {
                var unused: Float = 0
                for cpu in [scalar, accelerate] {
                    if let why = mismatch(source, cpu, worst: &unused) {
                        throw ForgeError.invalidGraph(clip("source reference disagrees with CPU references on the example input: \(why)"))
                    }
                }
                references.append([source, scalar, accelerate])
            } else {
                references.append([scalar, accelerate])
            }
        }
        let allReferences = references
        let prepared = try qualificationInputs.map { try DAGMetalBackend.Prepared(program: program, input: $0) }
        let bench = prepared[0]
        let hardware = ForgeHardwareProfile(
            deviceName: bench.deviceName,
            maxTotalThreadsPerThreadgroup: bench.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: bench.threadExecutionWidth,
            recommendedMaxWorkingSetSize: bench.recommendedMaxWorkingSetSize)
        let cap = hardware.maxTotalThreadsPerThreadgroup
        guard cap > 0, cap <= 1 << 16, prepared.allSatisfy({
            $0.deviceName == hardware.deviceName && $0.maxTotalThreadsPerThreadgroup == cap }) else {
            throw ForgeError.invalidPlan("dag selection: device profile out of bounds or inconsistent across inputs")
        }
        let baseline = baselineCandidate(program, maxThreadsPerThreadgroup: cap)

        func execute(_ p: DAGMetalBackend.Prepared, _ candidate: ForgeCandidate) throws -> (output: [Float], gpuMicros: Double) {
            let result = try p.run(threadgroupWidth: candidate.threadgroupWidth, fuseDenseRelu: candidate.fuseDenseRelu)
            guard result.deviceName == hardware.deviceName else { throw RunMismatch(description: "run reported another device") }
            guard result.kernels == kernelSequence(program, candidate: candidate) else {
                throw RunMismatch(description: "dispatched kernels differ from the host-computed sequence")
            }
            return (result.output, result.gpuMicros)
        }

        func qualify(_ candidate: ForgeCandidate, forceFailure: Bool) -> (ok: Bool, maxAbsDiff: Double?, note: String) {
            var worst: Float = 0
            for (k, p) in prepared.enumerated() {
                var output: [Float]
                do { output = try execute(p, candidate).output }
                catch { return (false, nil, clip("execution failed: \(error)")) }
                // Diagnostic only: perturbs the real GPU readback of an optimization run.
                if forceFailure, !output.isEmpty { output[0] += max(1, abs(output[0])) }
                for reference in allReferences[k] {
                    if let why = mismatch(output, reference, worst: &worst) {
                        return (false, worst.isFinite ? Double(worst) : nil, clip("qualification input \(k): \(why)"))
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
                do { micros = try execute(bench, candidate).gpuMicros }
                catch { return (nil, 0, clip("benchmark failed: \(error)")) }
                guard micros.isFinite, micros > 0 else { return (nil, 0, "GPU timestamps unavailable") }
                if iteration >= Forge.warmupRuns {
                    times.append(micros)
                    spent += micros
                    if spent > Forge.budgetMicros { break }
                }
            }
            guard !times.isEmpty else { return (nil, 0, "no timed runs") }
            let sorted = times.sorted(), mid = sorted.count / 2
            let median = sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
            guard median.isFinite, median > 0 else { return (nil, 0, "invalid median") }
            return (median, times.count, "\(times.count) timed runs")
        }

        let baselineCheck = qualify(baseline, forceFailure: false)
        guard baselineCheck.ok else {
            throw ForgeError.noValidCandidate(rejected: ["baseline \(ForgeBaseline.label(baseline)): \(baselineCheck.note)"])
        }
        let baselineTiming = measure(baseline)
        guard let baselineMedian = baselineTiming.median else {
            throw ForgeError.noValidCandidate(rejected: ["baseline \(ForgeBaseline.label(baseline)): \(baselineTiming.note)"])
        }
        let baselineResult = ForgeQualifiedResult(
            candidate: baseline, role: baselineRole, qualified: true, maxAbsDiff: baselineCheck.maxAbsDiff,
            medianGpuMicros: baselineMedian, timedRuns: baselineTiming.runs, diagnosticForcedFailure: false,
            note: clip("verified baseline: \(baselineCheck.note); \(baselineTiming.note)"))

        var optimizations: [ForgeQualifiedResult] = []
        for candidate in legalCandidates(program, maxThreadsPerThreadgroup: cap) where candidate != baseline {
            let check = qualify(candidate, forceFailure: diagnosticFailOptimizations)
            if diagnosticFailOptimizations || !check.ok {
                let note = diagnosticFailOptimizations
                    ? "diagnostic forced failure: " + (check.ok ? "perturbed output unexpectedly passed; rejected" : check.note)
                    : "rejected: " + check.note
                optimizations.append(ForgeQualifiedResult(
                    candidate: candidate, role: optimizationRole, qualified: false, maxAbsDiff: check.maxAbsDiff,
                    medianGpuMicros: nil, timedRuns: 0, diagnosticForcedFailure: diagnosticFailOptimizations,
                    note: clip(note)))
                continue
            }
            let timing = measure(candidate)
            optimizations.append(ForgeQualifiedResult(
                candidate: candidate, role: optimizationRole, qualified: timing.median != nil,
                maxAbsDiff: check.maxAbsDiff, medianGpuMicros: timing.median,
                timedRuns: timing.median == nil ? 0 : timing.runs,
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
            decision = "optimization selected: \(ForgeBaseline.label(best.candidate)) \(ForgeBaseline.fmt(bestMedian)) us vs verified baseline \(ForgeBaseline.fmt(baselineMedian)) us"
        } else if optimizations.isEmpty {
            decision = "baseline: no optimization candidate is legal for this graph and device"
        } else if let best, let bestMedian = best.medianGpuMicros {
            decision = "baseline fallback: best qualified optimization \(ForgeBaseline.label(best.candidate)) \(ForgeBaseline.fmt(bestMedian)) us is not more than \(ForgeBaseline.fmt(minRelativeGain * 100))% faster than verified baseline \(ForgeBaseline.fmt(baselineMedian)) us"
        } else {
            decision = "baseline fallback: none of \(optimizations.count) optimization candidate(s) passed qualification" + (diagnosticFailOptimizations ? " (diagnostic forced failure)" : "")
        }
        decision = String(decision.prefix(maxDecisionLength))
        let selection = ForgeBaselineSelection(
            baseline: baselineResult, optimizations: optimizations, selected: selected,
            selectedRole: selectedRole, selectedMedianGpuMicros: selectedMedian,
            minRelativeGain: minRelativeGain, qualificationInputs: prepared.count,
            diagnosticFailOptimizations: diagnosticFailOptimizations, decision: decision)
        try validateSelection(selection, program: program, hardware: hardware)
        return (selection, hardware)
    }

    /// Checks a persisted selection again against the legal set rebuilt for this program on this
    /// device. It never re-tunes. If any check fails, it throws invalidPlan.
    public static func validateSelection(_ selection: ForgeBaselineSelection, program: DAGProgram,
                                         hardware: ForgeHardwareProfile) throws {
        func bad(_ message: String) -> ForgeError { .invalidPlan("dag selection: " + message) }
        func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= 1e-9 * max(1, abs(b)) }
        let cap = hardware.maxTotalThreadsPerThreadgroup
        guard cap > 0, cap <= 1 << 16 else { throw bad("device threadgroup cap out of bounds") }
        let baseline = baselineCandidate(program, maxThreadsPerThreadgroup: cap)
        let legal = legalCandidates(program, maxThreadsPerThreadgroup: cap)
        let legalSet = Set(legal)
        guard legalSet.count == legal.count, legalSet.contains(baseline), !baseline.fuseDenseRelu else {
            throw bad("regenerated legal set is inconsistent")
        }
        let expected = legalSet.subtracting([baseline])
        guard selection.minRelativeGain == minRelativeGain,
              selection.qualificationInputs == qualificationInputCount,
              !selection.decision.isEmpty, selection.decision.count <= maxDecisionLength else {
            throw bad("malformed selection header")
        }
        let record = selection.baseline
        guard record.candidate == baseline, record.role == baselineRole, record.qualified,
              !record.diagnosticForcedFailure, let baselineMedian = record.medianGpuMicros,
              baselineMedian.isFinite, baselineMedian > 0, record.maxAbsDiff != nil,
              (1...Forge.timedRuns).contains(record.timedRuns) else {
            throw bad("baseline record is not the verified, measured unfused baseline")
        }
        let candidates = selection.optimizations.map { $0.candidate }
        guard candidates.count == expected.count, Set(candidates).count == candidates.count,
              Set(candidates) == expected,
              selection.optimizations.allSatisfy({ $0.role == optimizationRole }) else {
            throw bad("optimization records do not exactly match the legal candidate set")
        }
        for entry in [record] + selection.optimizations {
            guard entry.note.count <= ForgeBaseline.maxNoteLength,
                  (0...Forge.timedRuns).contains(entry.timedRuns),
                  entry.candidate.threadgroupWidth >= 1, entry.candidate.threadgroupWidth <= cap else {
                throw bad("malformed record")
            }
            if let m = entry.medianGpuMicros, !(m.isFinite && m > 0) { throw bad("invalid median") }
            if let d = entry.maxAbsDiff, !(d.isFinite && d >= 0) { throw bad("invalid max abs diff") }
            if entry.qualified {
                guard entry.medianGpuMicros != nil, entry.maxAbsDiff != nil, entry.timedRuns >= 1,
                      !entry.diagnosticForcedFailure else {
                    throw bad("qualified record without valid measurement")
                }
            } else {
                guard entry.medianGpuMicros == nil, entry.timedRuns == 0 else {
                    throw bad("unqualified record carries timings")
                }
            }
            guard entry.diagnosticForcedFailure == (selection.diagnosticFailOptimizations && entry.role == optimizationRole) else {
                throw bad("inconsistent diagnostic marker")
            }
        }
        guard legalSet.contains(selection.selected), selection.selectedMedianGpuMicros.isFinite,
              selection.selectedMedianGpuMicros > 0 else {
            throw bad("selected candidate is not legal or has no valid median")
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
}
