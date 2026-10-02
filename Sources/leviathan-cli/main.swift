import Foundation
import LeviathanCore

let programName = "leviathan"

func usage() -> Never {
    print("""
    USAGE: \(programName) <command> [options]

    Commands:
      compile --graph <file.json> --out <file.lvi> [--seed N]
          Validate a graph JSON file, lower it, and write an
          integrity-checked artifact.
      run --artifact <file.lvi> [--batch N] [--seed N] [--backend B]
          Load and integrity-check an artifact, run it with a deterministic
          seeded input, print the output. Backend B is one of:
            metal          Apple Silicon GPU via Metal (default)
            cpu-accelerate CPU reference using Apple Accelerate (vDSP)
            cpu-scalar     CPU scalar reference in plain Swift
      check [--backend B]
          Compile the built-in synthetic demo graph in memory and compare
          the selected primary backend (default: metal) against the
          independent CPU references (Accelerate and scalar) across
          several batch sizes.
    """)
    exit(64)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("error: \(message)\n".data(using: .utf8)!)
    exit(1)
}

func requireValue(_ flag: String, _ args: [String]) -> String {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
        fail("missing value for \(flag)")
    }
    return args[index + 1]
}

/// Parses an optional non-negative integer flag value; rejects malformed
/// and negative values instead of trapping or silently using them.
func optionalNonNegativeInt(_ flag: String, _ args: [String],
                            default defaultValue: Int) -> Int {
    guard let index = args.firstIndex(of: flag) else {
        return defaultValue
    }
    guard index + 1 < args.count else {
        fail("\(flag) requires a non-negative integer value")
    }
    guard let parsed = Int(args[index + 1]), parsed >= 0 else {
        fail("value for \(flag) must be a non-negative integer, got '\(args[index + 1])'")
    }
    return parsed
}

/// Converts a non-negative Int to UInt64 without trapping.
func toUInt64(_ value: Int, _ flag: String) -> UInt64 {
    guard value >= 0, let u = UInt64(exactly: value) else {
        fail("value for \(flag) is out of range")
    }
    return u
}

/// The built-in synthetic demo network: a small fully connected MLP with
/// ReLU activations. Weights are generated deterministically by the
/// compiler from a seed; this is NOT a trained model.
let demoGraph = GraphIR(inputDim: 16, ops: [
    .dense(outDim: 32),
    .relu,
    .dense(outDim: 32),
    .relu,
    .dense(outDim: 8),
])

func graphJSON(_ graph: GraphIR) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(graph)
}

func demoInput(dim: Int, batch: Int, seed: UInt64) -> [Float] {
    var rng = SplitMix64(seed: seed &+ 0xD1CE_B0BA)
    return (0..<(batch * dim)).map { _ in rng.nextUniform() }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { usage() }

/// Parses the optional --backend flag into a backend runner. Valid values:
/// metal (default), cpu-accelerate, cpu-scalar. Anything else is an error,
/// never a silent fallback.
func backendRunner(_ args: [String],
                   primary: @escaping (LoweredProgram, Int, [Float]) throws -> [Float])
                   -> (String, (LoweredProgram, Int, [Float]) throws -> [Float]) {
    guard let index = args.firstIndex(of: "--backend") else {
        return ("metal", primary)
    }
    guard index + 1 < args.count else {
        fail("--backend requires a value: metal, cpu-accelerate, or cpu-scalar")
    }
    switch args[index + 1] {
    case "metal": return ("metal", primary)
    case "cpu-accelerate": return ("cpu-accelerate", AccelerateBackend.run)
    case "cpu-scalar": return ("cpu-scalar", ScalarBackend.run)
    default: fail("value for --backend must be metal, cpu-accelerate, or cpu-scalar, "
                  + "got '\(args[index + 1])'")
    }
}

/// Wraps MetalBackend.runReporting so it matches the plain runner signature
/// while printing evidence captured from the actual GPU run.
func metalRunWithEvidence(_ program: LoweredProgram, batch: Int, input: [Float]) throws -> [Float] {
    let (output, evidence) = try MetalBackend.runReporting(program, batch: batch, input: input)
    print("metal device: \(evidence.deviceName)")
    print("shader library compiled at runtime; \(evidence.dispatchedSteps) step(s) dispatched; "
          + "command buffer \(evidence.commandStatus); \(evidence.readbackElements) value(s) read back")
    return output
}

switch command {
case "compile":
    let graphPath = requireValue("--graph", args)
    let outPath = requireValue("--out", args)
    let seed = toUInt64(optionalNonNegativeInt("--seed", args, default: 42), "--seed")
    let graph: GraphIR
    do {
        let graphURL = URL(fileURLWithPath: graphPath)
        let attributes = try FileManager.default.attributesOfItem(atPath: graphURL.path)
        let fileSize = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard fileSize >= 0, fileSize <= Int64(Limits.maxGraphJSONBytes) else {
            fail("graph file '\(graphPath)' exceeds \(Limits.maxGraphJSONBytes) bytes")
        }
        graph = try JSONDecoder().decode(GraphIR.self, from: Data(contentsOf: graphURL))
    } catch {
        fail("cannot read graph '\(graphPath)': \(error)")
    }
    do {
        _ = try LeviathanCore.validate(graph)
        let artifact = try CompiledArtifact(graph: graph, seed: seed)
        try artifact.save(to: URL(fileURLWithPath: outPath))
        let fileBytes = (try? Data(contentsOf: URL(fileURLWithPath: outPath)).count) ?? 0
        print("compiled \(graph.ops.count) ops -> \(outPath)")
        print("digest: \(artifact.digest)")
        print("weights: \(artifact.weights.count) float32, artifact bytes: \(fileBytes)")
    } catch {
        fail("compile failed: \(error)")
    }

case "run":
    let artifactPath = requireValue("--artifact", args)
    let batch = optionalNonNegativeInt("--batch", args, default: 1)
    let seed = toUInt64(optionalNonNegativeInt("--seed", args, default: 7), "--seed")
    let (backendName, runner) = backendRunner(args, primary: metalRunWithEvidence)
    guard batch >= 1, batch <= Limits.maxBatch else {
        fail("batch must be between 1 and \(Limits.maxBatch), got \(batch)")
    }
    do {
        let artifact = try CompiledArtifact.load(from: URL(fileURLWithPath: artifactPath))
        print("artifact OK, digest \(artifact.digest.prefix(16))…")
        let input = demoInput(dim: artifact.header.inputDim, batch: batch, seed: seed)
        let start = ContinuousClock.now
        let output = try runner(artifact.program, batch, input)
        let elapsed = ContinuousClock.now - start
        print(String(format: "output[%d]: %@", output.count,
                     output.map { String(format: "%.6f", $0) }.joined(separator: " ")))
        print(String(format: "ran %d step(s), batch %d, backend %@, in %.3f ms",
                     artifact.header.steps.count, batch, backendName,
                     Double(elapsed.components.attoseconds) / 1e15 + Double(elapsed.components.seconds) * 1e3))
    } catch {
        fail("run failed: \(error)")
    }

case "check":
    let (backendName, runner) = backendRunner(args, primary: metalRunWithEvidence)
    do {
        _ = try LeviathanCore.validate(demoGraph)
        let artifact = try CompiledArtifact(graph: demoGraph, seed: 42)
        print("digest: \(artifact.digest)")
        var worstDiff: Float = 0
        for batch in [1, 4, 16, 64] {
            let input = demoInput(dim: demoGraph.inputDim, batch: batch, seed: 7)
            let start = ContinuousClock.now
            let primary = try runner(artifact.program, batch, input)
            let elapsed = ContinuousClock.now - start
            let reference = try ScalarBackend.run(artifact.program, batch: batch, input: input)
            let accelerate = try AccelerateBackend.run(artifact.program, batch: batch, input: input)
            guard let diffScalar = maxAbsDiff(primary, reference),
                  let diffAccel = maxAbsDiff(primary, accelerate),
                  diffScalar < 1e-4, diffAccel < 1e-4 else {
                fail("backends disagree at batch \(batch): out of tolerance")
            }
            let worstForBatch = max(diffScalar, diffAccel)
            worstDiff = max(worstDiff, worstForBatch)
            print(String(format: "batch %3d: max |%@ - reference| = %.3e (%.3f ms)",
                         batch, backendName, worstForBatch,
                         Double(elapsed.components.attoseconds) / 1e15
                             + Double(elapsed.components.seconds) * 1e3))
        }
        print(String(format: "all batches agree within 1e-4 (worst %.3e)", worstDiff))
        print("PASS")
    } catch {
        fail("check failed: \(error)")
    }

    case "gguf-inspect":
        // Metadata-only inspection of a GGUF file: the file is memory-mapped
        // read-only and never copied; no weights are loaded or executed.
        let path = requireValue("--file", args)
        let url = URL(fileURLWithPath: path)
        // Optional full tensor-directory dump (name, dims, type, bytes,
        // file offset) for pinning the exact tensor contract of a model.
        let tensorsOutPath = args.firstIndex(of: "--tensors-out").map { _ in
            requireValue("--tensors-out", args)
        }
        do {
            let file = try GGUFReader.read(url)
            if let tensorsOutPath {
                var lines = [String]()
                lines.append("# name\tdims\ttype\tbytes\tfileOffset")
                for info in file.tensors {
                    let byteCount = try file.tensorRange(info).count
                    lines.append("\(info.name)\t\(info.dims)\t\(info.ggmlType)\t\(byteCount)\t\(try file.tensorRange(info).lowerBound)")
                }
                try lines.joined(separator: "\n").appending("\n")
                    .write(toFile: tensorsOutPath, atomically: true, encoding: .utf8)
                print("tensor directory (\(file.tensors.count) tensors) written to \(tensorsOutPath)")
            }
        print("path: \(path)")
        print("version: \(file.version)")
        print("file bytes: \(file.byteCount)")
        print("data offset: \(file.dataOffset) (alignment \(file.alignment))")
        print("metadata entries: \(file.metadata.count)")
        if args.contains("--metadata-all") {
            for (key, value) in file.metadata.sorted(by: { $0.key < $1.key }) {
                if case let .array(_, count, _, _, _, _, _) = value {
                    print("  \(key): <array of \(count)>")
                } else {
                    print("  \(key): \(value)")
                }
            }
        }
        let interesting = ["general.architecture", "general.name", "general.license",
                           "gemma4.context_length", "gemma4.block_count",
                           "gemma4.embedding_length", "gemma4.attention.head_count",
                           "gemma4.expert_count", "gemma4.expert_used_count",
                           "gemma4.expert_feed_forward_length",
                           "gemma4.feed_forward_length",
                           "gemma4.attention.key_length",
                           "gemma4.attention.value_length",
                           "gemma4.attention.sliding_window",
                           "tokenizer.ggml.model"]
        for key in interesting {
            if let value = file.metadata[key] {
                if case let .array(_, count, _, _, _, _, _) = value {
                    print("  \(key): <array of \(count)>")
                } else {
                    print("  \(key): \(value)")
                }
            }
        }
        print("tensors: \(file.tensors.count)")
        var typeCounts: [UInt32: (tensors: Int, bytes: Int)] = [:]
        for info in file.tensors {
            // Unknown types are reported explicitly with byte size -1 —
            // never silently folded into a 0-byte total.
            let byteCount: Int
            do { byteCount = try file.tensorRange(info).count }
            catch GGUFError.unsupportedTensorType { byteCount = -1 }
            let entry = typeCounts[info.ggmlType] ?? (tensors: 0, bytes: 0)
            typeCounts[info.ggmlType] = (tensors: entry.tensors + 1,
                                         bytes: entry.bytes + byteCount)
        }
        for (type, entry) in typeCounts.sorted(by: { $0.key < $1.key }) {
            let name = GGMLTensorType(rawValue: type).map { "\($0)" }
                ?? "unsupported(\(type))"
            print("  type \(type) (\(name)): \(entry.tensors) tensors, \(entry.bytes) bytes")
        }
        for info in file.tensors.prefix(12) {
            let byteCount: Int
            do { byteCount = try file.tensorRange(info).count }
            catch GGUFError.unsupportedTensorType { byteCount = -1 }
            print("  \(info.name): dims \(info.dims) type \(info.ggmlType) bytes \(byteCount)\(byteCount == -1 ? " (UNSUPPORTED TYPE)" : "")")
        }
        if file.tensors.count > 12 { print("  … \(file.tensors.count - 12) more") }
    } catch {
        fail("gguf-inspect failed: \(error)")
    }

case "forge":
    // Automatic bounded pass generation: measured hardware profile,
    // candidate generation, correctness gates, measured GPU-time
    // selection, and a reusable plan artifact. No agent or LLM at
    // runtime. Supported ops: dense, relu — anything else is rejected at
    // graph validation. Input is either a graph JSON (--graph) or a
    // validated PyTorch export prefix (--export-prefix, which supplies
    // BOTH the architecture and the real parameter weights plus a
    // PyTorch reference for parity checking).
    let exportPrefixIndex = args.firstIndex(of: "--export-prefix")
    let exportPrefix = exportPrefixIndex.map { _ in requireValue("--export-prefix", args) }
    let graphPath: String
    if let exportPrefix {
        if args.contains("--graph") { fail("--graph and --export-prefix are mutually exclusive") }
        graphPath = exportPrefix + ".graph.json"
    } else {
        graphPath = requireValue("--graph", args)
    }
    let batch = optionalNonNegativeInt("--batch", args, default: 4)
    let seed = toUInt64(optionalNonNegativeInt("--seed", args, default: 42), "--seed")
    let planURL = URL(fileURLWithPath: requireValue("--plan", args))
    // One validated export loaded ONCE: architecture, real weights, and
    // the PyTorch reference always come from the same validated triple —
    // no re-load that could race with a rewritten file.
    let weightExport: WeightExport?
    do {
        if let exportPrefix {
            weightExport = try WeightExport.load(prefix: exportPrefix)
        } else {
            weightExport = nil
        }
    } catch {
        fail("cannot load export '\(exportPrefix ?? "")': \(error)")
    }
    let graph: GraphIR
    let weights: [Float]?
    let reference: [Float]?   // PyTorch refOutput, only with --export-prefix
    do {
        if let weightExport {
            graph = weightExport.graph
            weights = weightExport.weights
            reference = weightExport.refOutput
        } else {
            weights = nil
            reference = nil
            let graphURL = URL(fileURLWithPath: graphPath)
            let attributes = try FileManager.default.attributesOfItem(atPath: graphURL.path)
            let fileSize = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            guard fileSize >= 0, fileSize <= Int64(Limits.maxGraphJSONBytes) else {
                fail("graph file '\(graphPath)' exceeds \(Limits.maxGraphJSONBytes) bytes")
            }
            let graphData = try Data(contentsOf: graphURL)
            guard graphData.count <= Limits.maxGraphJSONBytes else {
                fail("graph file '\(graphPath)' exceeds \(Limits.maxGraphJSONBytes) bytes")
            }
            graph = try JSONDecoder().decode(GraphIR.self, from: graphData)
        }
    } catch {
        fail("cannot read model '\(graphPath)': \(error)")
    }
    // Batch bounds are validated before ANY allocation, in both the
    // cache-reuse path and the fresh-forge path.
    guard batch > 0, batch <= Limits.maxBatch else {
        fail("batch \(batch) outside 1...\(Limits.maxBatch)")
    }
    // Helper: the program with whichever weights this run uses.
    func makeProgram() throws -> LoweredProgram {
        let base = try CompiledArtifact(graph: graph, seed: seed).program
        guard let weights else { return base }
        return LoweredProgram(inputDim: base.inputDim, outputDim: base.outputDim,
                              bufferCount: base.bufferCount, steps: base.steps,
                              weights: weights)
    }
    // Shared PyTorch parity gate: the selected candidate, the REAL
    // weights, and the manifest's deterministic input — identical logic
    // for fresh forges and cache reuses.
    func runParity(_ plan: ForgePlan, _ label: String) throws {
        guard let weightExport else { return }
        let program = try makeProgram()
        let prepared = try MetalBackend.Prepared(
            program: program, batch: 1, input: weightExport.refInput)
        let (output, _) = try prepared.run(
            threadgroupWidth: plan.selected.threadgroupWidth,
            fuseDenseRelu: plan.selected.fuseDenseRelu)
        guard output.count == weightExport.refOutput.count,
              output.allSatisfy({ $0.isFinite }) else {
            fail("\(label): parity vector wrong size or non-finite")
        }
        var worst: Float = 0
        for i in 0..<output.count {
            worst = max(worst, abs(output[i] - weightExport.refOutput[i])
                        / max(1, abs(weightExport.refOutput[i])))
        }
        print(String(format: "pytorch parity (%@): worst rel diff %.3e", label, worst))
        guard worst < 1e-4 else { fail("\(label): GPU output disagrees with PyTorch reference") }
    }
    do {
        // Cache reuse: if a plan for this exact graph/weights/seed/batch/
        // hardware/toolchain exists, validate it and run its selected
        // candidate once through the correctness gate — no re-benchmarking.
        if FileManager.default.fileExists(atPath: planURL.path) {
            let program = try makeProgram()
            var probeInput: [Float] = []
            var rng = SplitMix64(seed: seed)
            for _ in 0..<(batch * program.inputDim) {
                probeInput.append(rng.nextUniform())
            }
            let prepared = try MetalBackend.Prepared(
                program: program, batch: batch, input: probeInput)
            let hardware = ForgeHardwareProfile(
                deviceName: prepared.deviceName,
                maxTotalThreadsPerThreadgroup: prepared.maxTotalThreadsPerThreadgroup,
                threadExecutionWidth: prepared.threadExecutionWidth,
                recommendedMaxWorkingSetSize: prepared.recommendedMaxWorkingSetSize)
            if let plan = try? ForgePlan.loadValidated(
                    from: planURL, graph: graph, seed: seed, batch: batch,
                    hardware: hardware, weights: weights) {
                let (output, gpuMicros) = try prepared.run(
                    threadgroupWidth: plan.selected.threadgroupWidth,
                    fuseDenseRelu: plan.selected.fuseDenseRelu)
                let scalar = try ScalarBackend.run(program, batch: batch, input: probeInput)
                guard output.count == scalar.count,
                      output.allSatisfy({ $0.isFinite }),
                      zip(output, scalar).allSatisfy({ abs($0.0 - $0.1) < 1e-4 }) else {
                    fail("cached plan \(plan.planDigest) failed the correctness gate")
                }
                print("cache hit: plan \(plan.planDigest)")
                print("selected: width \(plan.selected.threadgroupWidth)"
                      + " fused \(plan.selected.fuseDenseRelu)")
                print(String(format: "recorded median gpu %.1f us, this run %.1f us",
                             plan.selectedMedianGpuMicros, gpuMicros))
                try runParity(plan, "reused")
                print("PASS (reused)")
                exit(0)
            }
            print("stale plan at \(planURL.path) — re-forging")
        }
        let (plan, _, hardware, results) = try Forge.forge(
            graph: graph, batch: batch, seed: seed, weights: weights)
        print("hardware: \(hardware.deviceName), max threads/threadgroup \(hardware.maxTotalThreadsPerThreadgroup), exec width \(hardware.threadExecutionWidth), working set \(hardware.recommendedMaxWorkingSetSize) bytes")
        if let weights { print("weights: real parameters, \(weights.count) floats") }
        for r in results {
            let median = r.medianGpuMicros.map { String(format: "%.1f us", $0) } ?? "-"
            print("  width \(r.candidate.threadgroupWidth) fused \(r.candidate.fuseDenseRelu ? 1 : 0): correct \(r.correct), median \(median) — \(r.note)")
        }
        print("selected: width \(plan.selected.threadgroupWidth)"
              + " fused \(plan.selected.fuseDenseRelu), median "
              + String(format: "%.1f", plan.selectedMedianGpuMicros) + " us")
        // PyTorch parity when an export prefix was supplied: the selected
        // candidate, real weights, manifest's deterministic input — from
        // the SAME validated export loaded once at startup.
        try runParity(plan, "fresh")
        try plan.save(to: planURL)
        print("plan saved: \(planURL.path), digest \(plan.planDigest)")
        print("PASS")
    } catch {
        fail("forge failed: \(error)")
    }

    case "gemma-forward":
        // Full-model forward on the real Gemma 4 26B-A4B QAT Q4_0 GGUF:
        // quantized weights stay quantized (mmap'd, zero-copy GPU buffer);
        // returns the greedy argmax token plus GPU evidence. Token IDs in,
        // token ID out — tokenizer wiring is a separate milestone.
        let path = requireValue("--file", args)
        let tokensArg = requireValue("--tokens", args)
        let tokenIds = tokensArg.split(separator: ",").map {
            guard let id = Int($0.trimmingCharacters(in: .whitespaces)), id >= 0 else {
                fail("invalid token id '\($0)'")
            }
            return id
        }
        guard !tokenIds.isEmpty, tokenIds.count <= 4096 else {
            fail("--tokens must contain 1...4096 ids")
        }
        do {
            let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: path))
            runner.traceLayers = args.contains("--trace")
            // Kernel-level parity gate before any correctness claim: the GPU
            // embedding-row decoder against the independent scalar Q6_K oracle.
            let embedWorst = try runner.verifyEmbeddingRow(token: tokenIds[0])
            guard embedWorst < 1e-4 else {
                fail(String(format: "embedding parity failed: worst rel diff %.3e", embedWorst))
            }
            print(String(format: "embedding parity (GPU vs scalar Q6_K oracle, %d values): worst rel diff %.3e",
                         2816, embedWorst))
            let start = ContinuousClock.now
            let evidence = try runner.forward(tokenIds: tokenIds)
            let elapsed = ContinuousClock.now - start
            print("device: \(evidence.deviceName)")
            print("maxBufferLength: \(evidence.deviceMaxBufferLength) bytes, " +
                  "recommendedWorkingSet: \(evidence.recommendedMaxWorkingSetSize) bytes")
            print("weights mapped (zero-copy, quantized): \(evidence.weightsMappedBytes) bytes")
            print("metal currentAllocatedSize (includes mapped weights): " +
                  "\(evidence.metalAllocatedBytes) bytes")
            print("max sampled process footprint: \(evidence.peakFootprintBytes) bytes")
            print("tokens: \(evidence.tokensProcessed), kernels dispatched: \(evidence.dispatchedKernels)")
            print(String(format: "gpu time: %.1f ms, wall: %.1f ms",
                         evidence.gpuMicros / 1000,
                         Double(elapsed.components.attoseconds) / 1e15
                             + Double(elapsed.components.seconds) * 1e3))
            print("greedy next token: \(evidence.argmaxToken)")
            for t in evidence.topLogits {
                print(String(format: "  token %6d  logit %+.6f", t.token, t.logit))
            }
            print("FORWARD_COMPLETED (model correctness vs external baseline not yet claimed)")
        } catch {
            fail("gemma-forward failed: \(error)")
        }

    case "gemma-token-lookup":
        // Exact token-piece lookup in the GGUF vocab (tokenizer.ggml.tokens
        // + token_type). Diagnostic for pinning exact token IDs; NOT a
        // general tokenizer implementation.
        let path = requireValue("--file", args)
        do {
            let file = try GGUFReader.read(URL(fileURLWithPath: path))
            if let idArg = args.firstIndex(of: "--id").map({ _ in requireValue("--id", args) }),
               let id = Int(idArg), id >= 0 {
                guard case let .array(_, count, _, _, _, strings, _)? = file.metadata["tokenizer.ggml.tokens"],
                      id < count else { fail("id \(id) out of range") }
                print("token id \(id): '\(strings[id])'")
                exit(0)
            }
            let piece = requireValue("--piece", args)
            guard case let .array(_, count, _, _, _, strings, _)? = file.metadata["tokenizer.ggml.tokens"],
                  case let .array(_, tcount, _, _, _, _, _)? = file.metadata["tokenizer.ggml.token_type"],
                  count == tcount else {
                fail("missing tokenizer.ggml.tokens/token_type arrays")
            }
            var hits = 0
            for (i, t) in strings.enumerated() where t == piece {
                print("token id \(i): exact match")
                hits += 1
                if hits >= 8 { break }
            }
            if hits == 0 { print("no exact match for piece '\(piece)'") }
        } catch {
            fail("gemma-token-lookup failed: \(error)")
        }

    default:
        usage()
}
