import Foundation

/// Errors from the Forge automatic pass-selection pipeline.
public enum ForgeError: Error, Equatable, Sendable {
    /// The graph failed validation; nothing is generated or measured.
    case invalidGraph(String)
    /// The batch is outside the validated range or overflows allocation.
    case invalidBatch(String)
    /// No candidate passed the correctness gate, so no selection is made.
    case noValidCandidate(rejected: [String])
    /// A plan file is malformed, oversized, or does not match the current
    /// model/hardware/toolchain/workload — it is never executed.
    case invalidPlan(String)
}

/// Measured hardware profile captured from the actual device used.
public struct ForgeHardwareProfile: Codable, Equatable, Sendable {
    public let deviceName: String
    public let maxTotalThreadsPerThreadgroup: Int
    public let threadExecutionWidth: Int
    public let recommendedMaxWorkingSetSize: Int

    public init(deviceName: String, maxTotalThreadsPerThreadgroup: Int,
                threadExecutionWidth: Int, recommendedMaxWorkingSetSize: Int) {
        self.deviceName = deviceName
        self.maxTotalThreadsPerThreadgroup = maxTotalThreadsPerThreadgroup
        self.threadExecutionWidth = threadExecutionWidth
        self.recommendedMaxWorkingSetSize = recommendedMaxWorkingSetSize
    }
}

/// One automatically generated optimization candidate. The bounded
/// candidate set is derived from the graph and the measured device
/// profile — never searched at runtime by an agent or an unbounded
/// optimizer.
public struct ForgeCandidate: Codable, Equatable, Hashable, Sendable {
    /// Dense-kernel threadgroup width actually dispatched (already
    /// deduplicated against the legal device maximum when generated).
    public let threadgroupWidth: Int
    /// Run the fused dense+ReLU pass where the graph allows it.
    public let fuseDenseRelu: Bool

    public init(threadgroupWidth: Int, fuseDenseRelu: Bool) {
        self.threadgroupWidth = threadgroupWidth
        self.fuseDenseRelu = fuseDenseRelu
    }
}

/// Result of measuring one candidate.
public struct ForgeCandidateResult: Codable, Equatable, Sendable {
    public let candidate: ForgeCandidate
    public let correct: Bool
    /// Median measured GPU time in microseconds (command-buffer
    /// GPUStartTime..GPUEndTime, warmup excluded); nil if rejected.
    public let medianGpuMicros: Double?
    public let note: String
}

/// Cache/reuse schema version; bumped whenever plan semantics change.
public let forgeSchemaVersion = 1

/// The selected, reusable plan. The digest keys the plan to the exact
/// graph (including the synthetic-weight seed), hardware profile, toolchain
/// schema version, and workload, so a rerun on the same model + hardware +
/// workload reuses the measured decision instead of re-benchmarking.
public struct ForgePlan: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let graphDigest: String
    public let hardware: ForgeHardwareProfile
    public let batch: Int
    public let seed: UInt64
    /// Verbatim toolchain identity, compared on load — a plan built by a
    /// different toolchain is stale even if the schema number matches.
    public let toolchain: String
    /// SHA-256 of the real weight content when the plan was forged from
    /// exported parameters; nil for seeded synthetic weights.
    public let weightsDigest: String?
    public let selected: ForgeCandidate
    public let selectedMedianGpuMicros: Double
    public let results: [ForgeCandidateResult]
    public let planDigest: String

    public init(schemaVersion: Int, graphDigest: String, hardware: ForgeHardwareProfile,
                batch: Int, seed: UInt64, toolchain: String,
                weightsDigest: String? = nil,
                selected: ForgeCandidate,
                selectedMedianGpuMicros: Double, results: [ForgeCandidateResult],
                planDigest: String) {
        self.schemaVersion = schemaVersion
        self.graphDigest = graphDigest
        self.hardware = hardware
        self.batch = batch
        self.seed = seed
        self.toolchain = toolchain
        self.weightsDigest = weightsDigest
        self.selected = selected
        self.selectedMedianGpuMicros = selectedMedianGpuMicros
        self.results = results
        self.planDigest = planDigest
    }

    public static let fileExtension = "forge.json"
    public static let maxPlanJSONBytes = 1 << 20

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url)
    }

    /// Loads and fully validates an untrusted plan file: bounded size
    /// (re-checked on the actual decoded data), schema version, positive
    /// bounded integers everywhere, exact current-hardware match, digest
    /// self-consistency, selected-candidate legality against the
    /// regenerated candidate set, and result/digest agreement. A stale,
    /// foreign, or corrupt plan throws `invalidPlan` and is never executed.
    public static func loadValidated(from url: URL, graph: GraphIR, seed: UInt64,
                                     batch: Int,
                                     hardware: ForgeHardwareProfile,
                                     weights: [Float]? = nil) throws -> ForgePlan {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? -1
        guard size >= 0, size <= Int64(maxPlanJSONBytes) else {
            throw ForgeError.invalidPlan("plan file exceeds \(maxPlanJSONBytes) bytes")
        }
        let rawData = try Data(contentsOf: url)
        // Re-check the actual decoded size (attributes can race).
        guard rawData.count <= maxPlanJSONBytes else {
            throw ForgeError.invalidPlan("plan data exceeds \(maxPlanJSONBytes) bytes")
        }
        let plan: ForgePlan
        do { plan = try JSONDecoder().decode(ForgePlan.self, from: rawData) }
        catch { throw ForgeError.invalidPlan("plan JSON decode failed: \(error)") }
        guard plan.schemaVersion == forgeSchemaVersion else {
            throw ForgeError.invalidPlan("schema \(plan.schemaVersion) != \(forgeSchemaVersion)")
        }
        // All integers that feed digest material must be positive and
        // bounded BEFORE any UInt64 conversion. Workload identity has its
        // own specific rejection message, checked separately below.
        guard !plan.toolchain.isEmpty, plan.toolchain.count <= 512,
              plan.batch > 0, plan.batch <= Limits.maxBatch,
              plan.hardware.maxTotalThreadsPerThreadgroup > 0,
              plan.hardware.maxTotalThreadsPerThreadgroup <= 1 << 16,
              plan.hardware.threadExecutionWidth > 0,
              plan.hardware.threadExecutionWidth <= 1 << 16,
              plan.hardware.recommendedMaxWorkingSetSize > 0,
              plan.selected.threadgroupWidth > 0,
              plan.selected.threadgroupWidth <= 1 << 16,
              plan.results.count > 0, plan.results.count <= 16 else {
            throw ForgeError.invalidPlan("plan fields outside validated bounds")
        }
        guard plan.seed == seed, plan.batch == batch else {
            throw ForgeError.invalidPlan("workload mismatch (batch/seed)")
        }
        guard plan.toolchain == Forge.toolchainIdentifier else {
            throw ForgeError.invalidPlan("toolchain mismatch — plan built by another toolchain")
        }
        guard plan.hardware == hardware else {
            throw ForgeError.invalidPlan("hardware mismatch — plan is for another machine")
        }
        var expectedDigest = try Forge.graphDigest(graph, seed: seed)
        var expectedWeightsDigest: String? = nil
        if let supplied = weights {
            guard supplied.count <= Limits.maxWeightBytes / 4,
                  supplied.allSatisfy({ $0.isFinite }) else {
                throw ForgeError.invalidPlan("supplied weights invalid (size or finiteness)")
            }
            var bytes = [UInt8]()
            bytes.reserveCapacity(supplied.count * 4)
            for value in supplied {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes.append(contentsOf: $0) }
            }
            expectedWeightsDigest = SHA256.hex(SHA256.hash(bytes))
            expectedDigest = SHA256.hex(SHA256.hash(
                Array(expectedDigest.utf8) + Array(expectedWeightsDigest!.utf8)))
        }
        guard plan.graphDigest == expectedDigest else {
            throw ForgeError.invalidPlan("graph digest mismatch — plan is for another model")
        }
        // The selected candidate must be one this graph + device would
        // actually generate, with a finite positive median that matches
        // its own recorded valid result.
        // Regenerate the EXACT candidate set this graph + hardware would
        // generate (fusion availability and per-graph width cap included),
        // so a 16-wide graph's own width-16 selection validates, fusion
        // flags on a non-fusable graph reject, and every recorded result
        // is a legal candidate.
        let artifact = try CompiledArtifact(graph: graph, seed: seed)
        let program = artifact.program
        let fusable = MetalBackend.isDenseReluFusable(program)
        var minDenseOutDim = 1 << 30
        for step in program.steps {
            if case let .dense(_, _, outDim, _, _, _, _, _) = step {
                minDenseOutDim = min(minDenseOutDim, outDim)
            }
        }
        if minDenseOutDim == 1 << 30 { minDenseOutDim = 1 }
        let legal = Set(Forge.candidates(
            fusable: fusable,
            maxThreadsPerThreadgroup: hardware.maxTotalThreadsPerThreadgroup,
            minDenseOutDim: minDenseOutDim))
        guard legal.contains(plan.selected) else {
            throw ForgeError.invalidPlan("selected candidate not in the generated set")
        }
        guard let selectedResult = plan.results.first(where: {
                    $0.candidate == plan.selected && $0.correct }),
              let median = selectedResult.medianGpuMicros,
              median.isFinite, median > 0,
              abs(median - plan.selectedMedianGpuMicros) < 1e-9,
              plan.selectedMedianGpuMicros.isFinite, plan.selectedMedianGpuMicros > 0 else {
            throw ForgeError.invalidPlan("selected candidate has no valid measured result")
        }
        // Every result must be a legal candidate, unique, with bounded
        // notes and finite positive medians exactly when measured.
        let resultCandidates = plan.results.map { $0.candidate }
        guard Set(resultCandidates).count == resultCandidates.count,
              resultCandidates.allSatisfy({ legal.contains($0) }),
              plan.results.allSatisfy({ $0.note.count <= 256 }),
              plan.results.allSatisfy({ r in
                  if r.correct, let m = r.medianGpuMicros { return m.isFinite && m > 0 }
                  return r.medianGpuMicros == nil || !r.correct
              }) else {
            throw ForgeError.invalidPlan("results contain illegal/duplicate/malformed entries")
        }
        // The digest covers graph/hardware/workload/toolchain/selection —
        // NOT the recorded measurement list, so it proves plan identity,
        // not integrity of every stored number.
        guard plan.planDigest == plan.recomputedDigest else {
            throw ForgeError.invalidPlan("plan digest mismatch — wrong model, hardware, toolchain, or workload")
        }
        return plan
    }

    var recomputedDigest: String {
        var material = Array(graphDigest.utf8)
        material += Array(hardware.deviceName.utf8)
        withUnsafeBytes(of: UInt64(hardware.maxTotalThreadsPerThreadgroup).littleEndian) {
            material += Array($0) }
        withUnsafeBytes(of: UInt64(hardware.threadExecutionWidth).littleEndian) {
            material += Array($0) }
        withUnsafeBytes(of: UInt64(hardware.recommendedMaxWorkingSetSize).littleEndian) {
            material += Array($0) }
        material += Array(toolchain.utf8)
        material += Array((weightsDigest ?? "synthetic").utf8)
        withUnsafeBytes(of: UInt64(batch).littleEndian) { material += Array($0) }
        withUnsafeBytes(of: seed.littleEndian) { material += Array($0) }
        withUnsafeBytes(of: UInt64(schemaVersion).littleEndian) { material += Array($0) }
        withUnsafeBytes(of: UInt64(selected.threadgroupWidth).littleEndian) {
            material += Array($0) }
        material += selected.fuseDenseRelu ? [1] : [0]
        return SHA256.hex(SHA256.hash(material))
    }
}

/// The automatic pass-generation pipeline: validate the model graph,
/// measure the hardware profile, generate a bounded set of legal
/// candidates (kernel-width variants plus the fused dense+ReLU pass),
/// gate every candidate on numerical correctness against two independent
/// CPU references, measure real GPU time (command-buffer device time,
/// warmup excluded, finite budget) with prepared resources so compilation
/// and uploads are never timed, select the best valid candidate, and emit
/// a reusable, validated plan. All of it runs without any coding agent or
/// LLM at runtime.
///
/// Currently supported graph ops: `dense` and `relu` (the validated
/// synthetic-graph IR). Unsupported operations are rejected at graph
/// validation before anything is generated — this is not a generic
/// compiler and does not pretend to be one.
public enum Forge {

    /// Requested widths; generation clamps to the measured device legal
    /// maximum and deduplicates, so every candidate is a distinct actual
    /// dispatch.
    public static let requestedWidths = [32, 64, 128, 256]
    public static let warmupRuns = 3
    public static let timedRuns = 11
    /// Total GPU-time budget per candidate in microseconds.
    public static let budgetMicros: Double = 2_000_000

    public static func graphDigest(_ graph: GraphIR, seed: UInt64) throws -> String {
        _ = try validate(graph)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(graph)
        var material = [UInt8](bytes)
        withUnsafeBytes(of: seed.littleEndian) { material.append(contentsOf: $0) }
        // NOTE: for synthetic seeded graphs the seed fully determines the
        // weights. Real imported weights must hash their actual content.
        return SHA256.hex(SHA256.hash(material))
    }

    /// Generates the bounded candidate set for a graph on a device: all
    /// distinct ACTUALLY-DISPATCHED widths (requested width capped by the
    /// device legal maximum and the smallest dense outDim, then
    /// deduplicated — for a 16-wide output every requested width collapses
    /// to the same dispatch), each with and without fusion when (and only
    /// when) the graph contains a fusable dense→relu pair.
    public static func candidates(fusable: Bool, maxThreadsPerThreadgroup: Int,
                                  minDenseOutDim: Int = 1 << 30) -> [ForgeCandidate] {
        guard maxThreadsPerThreadgroup > 0, minDenseOutDim > 0 else { return [] }
        var widths = Set<Int>()
        for w in requestedWidths {
            widths.insert(min(w, maxThreadsPerThreadgroup, minDenseOutDim))
        }
        widths.insert(min(32, maxThreadsPerThreadgroup, minDenseOutDim))
        var candidates = widths.sorted().flatMap { w in
            [ForgeCandidate(threadgroupWidth: w, fuseDenseRelu: false)]
                + (fusable ? [ForgeCandidate(threadgroupWidth: w, fuseDenseRelu: true)] : [])
        }
        candidates.removeAll { $0.threadgroupWidth < 1 }
        return candidates
    }

    /// Toolchain identity in the plan digest: schema version alone is not
    /// a toolchain. Captured once per process via `swift --version`
    /// (fixed executable, bounded output) plus the OS version; persisted
    /// as a plan field and compared verbatim on load.
    public static let toolchainIdentifier: String = {
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        var swiftVersion = "unknown"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
        process.arguments = ["--version"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        if (try? process.run()) != nil {
            // The read is bounded, but waitUntilExit has no wall-clock
            // timeout: a pathologically hanging toolchain process could
            // stall this once-per-process probe. Accepted, documented
            // limitation for a fixed system executable.
            let data = pipe.fileHandleForReading.readData(ofLength: 256)
            if process.isRunning { process.waitUntilExit() }
            if let first = String(data: data, encoding: .utf8)?
                .split(separator: "\n").first {
                swiftVersion = String(first)
            }
        }
        return "swift(\(swiftVersion)) os(\(os)) schema(\(forgeSchemaVersion)) msl(\(MetalBackend.shaderFingerprint))"
    }()

    /// Runs the full pipeline for a validated graph and deterministic
    /// seeded input. Throws `noValidCandidate` if every candidate fails
    /// the correctness gate.
    ///
    /// With `weights` supplied (real exported parameters, not the seeded
    /// synthetic stream), the weights replace the synthetic blob after
    /// count/finiteness validation, and the plan identity includes the
    /// weights' content hash — a plan forged for one set of parameter
    /// values never validates against another.
    public static func forge(graph: GraphIR, batch: Int, seed: UInt64,
                             weights: [Float]? = nil) throws
            -> (plan: ForgePlan, output: [Float], hardware: ForgeHardwareProfile,
                results: [ForgeCandidateResult]) {
        let artifact: CompiledArtifact
        do { artifact = try CompiledArtifact(graph: graph, seed: seed) }
        catch { throw ForgeError.invalidGraph("\(error)") }
        var program = artifact.program
        var weightsDigest: String? = nil
        if let supplied = weights {
            let allFinite = supplied.allSatisfy({ $0.isFinite })
            guard supplied.count <= Limits.maxWeightBytes / 4,
                  supplied.count == program.weights.count,
                  allFinite else {
                throw ForgeError.invalidGraph(
                    "supplied weights invalid: \(supplied.count) floats (expected \(program.weights.count)), finite \(allFinite)")
            }
            var bytes = [UInt8]()
            bytes.reserveCapacity(supplied.count * 4)
            for value in supplied {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes.append(contentsOf: $0) }
            }
            weightsDigest = SHA256.hex(SHA256.hash(bytes))
            program = LoweredProgram(inputDim: program.inputDim,
                                     outputDim: program.outputDim,
                                     bufferCount: program.bufferCount,
                                     steps: program.steps,
                                     weights: supplied)
        }

        // Batch bounds are checked before any allocation arithmetic.
        guard batch > 0, batch <= Limits.maxBatch else {
            throw ForgeError.invalidBatch("batch \(batch) outside 1...\(Limits.maxBatch)")
        }
        let (count, overflow) = batch.multipliedReportingOverflow(by: program.inputDim)
        guard !overflow, count <= Limits.maxTensorElements else {
            throw ForgeError.invalidBatch("batch x inputDim overflows tensor budget")
        }

        var input: [Float] = []
        input.reserveCapacity(count)
        var rng = SplitMix64(seed: seed)
        for _ in 0..<count { input.append(rng.nextUniform()) }

        // Prepare once: correctness runs and every timed run dispatch the
        // same compiled pipelines and buffers — compilation and upload are
        // never part of a measurement.
        let prepared = try MetalBackend.Prepared(program: program, batch: batch, input: input)
        let hardware = ForgeHardwareProfile(
            deviceName: prepared.deviceName,
            maxTotalThreadsPerThreadgroup: prepared.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: prepared.threadExecutionWidth,
            recommendedMaxWorkingSetSize: prepared.recommendedMaxWorkingSetSize)

        // Independent references computed once, not per candidate.
        let scalar = try ScalarBackend.run(program, batch: batch, input: input)
        let accelerate = try AccelerateBackend.run(program, batch: batch, input: input)

        let fusable = MetalBackend.isDenseReluFusable(program)
        var digest = try graphDigest(graph, seed: seed)
        if let weightsDigest {
            digest = SHA256.hex(SHA256.hash(Array(digest.utf8) + Array(weightsDigest.utf8)))
        }
        var results: [ForgeCandidateResult] = []
        var best: (candidate: ForgeCandidate, median: Double, output: [Float])?
        let minDenseOutDim = program.steps.reduce(into: 1 << 30) { acc, step in
            if case let .dense(_, _, outDim, _, _, _, _, _) = step { acc = min(acc, outDim) }
        }
        outer: for candidate in candidates(
            fusable: fusable,
            maxThreadsPerThreadgroup: hardware.maxTotalThreadsPerThreadgroup,
            minDenseOutDim: minDenseOutDim == 1 << 30 ? 1 : minDenseOutDim) {
            // Correctness gate first: any candidate outside 1e-4 of both
            // references is rejected and never measured.
            do {
                let (output, _) = try prepared.run(
                    threadgroupWidth: candidate.threadgroupWidth,
                    fuseDenseRelu: candidate.fuseDenseRelu)
                guard output.count == scalar.count,
                      zip(output, scalar).allSatisfy({ abs($0.0 - $0.1) < 1e-4 }),
                      zip(output, accelerate).allSatisfy({ abs($0.0 - $0.1) < 1e-4 }) else {
                    results.append(ForgeCandidateResult(
                        candidate: candidate, correct: false, medianGpuMicros: nil,
                        note: "output outside 1e-4 of CPU references"))
                    continue
                }
                // Measured selection: warmup, then median of command-buffer
                // GPU times under a finite GPU-time budget.
                var gpuTimes: [Double] = []
                var spent: Double = 0
                for run in 0..<(warmupRuns + timedRuns) {
                    let (_, micros) = try prepared.run(
                        threadgroupWidth: candidate.threadgroupWidth,
                        fuseDenseRelu: candidate.fuseDenseRelu)
                    // GPU timestamps unavailable or non-positive on this
                    // run: this candidate cannot be measured — reject it
                    // and continue with the remaining candidates.
                    guard micros.isFinite, micros > 0 else {
                        results.append(ForgeCandidateResult(
                            candidate: candidate, correct: false, medianGpuMicros: nil,
                            note: "GPU timestamps unavailable (\(micros) us)"))
                        continue outer
                    }
                    if run >= warmupRuns {
                        gpuTimes.append(micros)
                        spent += micros
                    }
                    if spent > budgetMicros, run >= warmupRuns { break }
                }
                guard !gpuTimes.isEmpty else {
                    results.append(ForgeCandidateResult(
                        candidate: candidate, correct: true, medianGpuMicros: nil,
                        note: "measurement budget exhausted before any timed run"))
                    continue
                }
                let sorted = gpuTimes.sorted()
                let median = sorted.count.isMultiple(of: 2)
                    ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
                    : sorted[sorted.count / 2]
                results.append(ForgeCandidateResult(
                    candidate: candidate, correct: true, medianGpuMicros: median,
                    note: "valid; \(gpuTimes.count) timed runs"))
                if best == nil || median < best!.median {
                    best = (candidate, median, output)
                }
            } catch let e as ForgeError {
                throw e
            } catch {
                results.append(ForgeCandidateResult(
                    candidate: candidate, correct: false, medianGpuMicros: nil,
                    note: "execution failed: \(error)"))
            }
        }
        guard let winner = best else {
            throw ForgeError.noValidCandidate(
                rejected: results.map { "\($0.candidate): \($0.note)" })
        }
        let plan = ForgePlan(
            schemaVersion: forgeSchemaVersion,
            graphDigest: digest,
            hardware: hardware,
            batch: batch,
            seed: seed,
            toolchain: toolchainIdentifier,
            weightsDigest: weightsDigest,
            selected: winner.candidate,
            selectedMedianGpuMicros: winner.median,
            results: results,
            planDigest: "")
        // Recompute with the digest material and refill.
        let planFinal = ForgePlan(
            schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
            hardware: plan.hardware, batch: plan.batch, seed: plan.seed,
            toolchain: plan.toolchain,
            weightsDigest: plan.weightsDigest,
            selected: plan.selected, selectedMedianGpuMicros: plan.selectedMedianGpuMicros,
            results: plan.results, planDigest: plan.recomputedDigest)
        return (planFinal, winner.output, hardware, results)
    }
}
