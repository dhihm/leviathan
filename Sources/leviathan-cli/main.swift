import Foundation
import Metal
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
      gemma-forge --file <model.gguf> --graph <g.json> --plan <p.json>
                  [--prompt-ids 2,9259 | --text T [--chat]] [--gen-tokens 8]
                  [--expert-cache-budget BYTES]
          Validate the captured Gemma graph against the real GGUF, gate
          and measure all legal engine candidates, select by per-token
          wall median, save a reusable plan artifact (typed SSA schedule
          + identity digests), and run a verified gated generation.
          --expert-cache-budget switches to the bounded SSD expert mode
          (non-expert weights compacted to a GPU buffer; expert slices
          stream through an LRU slot cache within the byte budget).
      gemma-expert-cache-verify --file <model.gguf> --graph <g.json>
                  [--prompt-ids 2,9259] [--steps 4]
                  [--small-budget 8000000] [--large-budget 1000000000]
          Forced eviction/miss/hit/in-flight verification of the expert
          cache through the actual SSA path vs a resident snapshot.
      gemma-serve --file <model.gguf> --graph <g.json> --plan <p.json>
                  [--port 8377] [--max-tokens 16] [--max-prompt-chars 512]
                  [--max-requests 64] [--expert-cache-budget BYTES]
                  [--gate-first-request]
          Bounded localhost (127.0.0.1) serving of the validated plan
          artifact: one POST /generate at a time plus GET /health, no
          runtime agents; strict bounds and fail-closed HTTP parsing.
      gemma-profile --file <model.gguf> --prompt-ids 2,9259
          Per-tensor and per-kernel GPU-time attribution.
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

    case "gguf-tokenizer-export":
        // Export the GGUF tokenizer metadata (tokens, scores, token types,
        // merges, chat template, and key scalars) as one JSON file so the
        // tokenizer semantics can be pinned and cross-verified offline.
        // Reads metadata only; no weights are loaded or executed.
        let path = requireValue("--file", args)
        let outPath = requireValue("--out", args)
        do {
            let file = try GGUFReader.read(URL(fileURLWithPath: path))
            guard case let .array(_, tokenCount, _, _, _, tokens, _)? =
                    file.metadata["tokenizer.ggml.tokens"] else {
                fail("missing tokenizer.ggml.tokens")
            }
            func intArray(_ key: String) -> [Int]? {
                guard case let .array(_, count, _, ints, _, _, _)? = file.metadata[key],
                      count == tokenCount else { return nil }
                return ints.map(Int.init)
            }
            func strArray(_ key: String) -> [String]? {
                guard case let .array(_, _, _, _, _, strings, _)? = file.metadata[key] else {
                    return nil
                }
                return strings
            }
            var out: [String: Any] = [
                "tokens": tokens,
            ]
            if let scores = file.metadata["tokenizer.ggml.scores"],
               case let .array(_, scoreCount, _, _, floats, _, _) = scores,
               scoreCount == tokenCount {
                out["scores"] = floats
            }
            if let types = intArray("tokenizer.ggml.token_type") { out["types"] = types }
            if let merges = strArray("tokenizer.ggml.merges") { out["merges"] = merges }
            if case let .string(template)? = file.metadata["tokenizer.chat_template"] {
                out["chat_template"] = template
            }
            for (jsonKey, metaKey) in [("bos_token_id", "tokenizer.ggml.bos_token_id"),
                                       ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                                       ("unknown_token_id", "tokenizer.ggml.unknown_token_id"),
                                       ("padding_token_id", "tokenizer.ggml.padding_token_id"),
                                       ("mask_token_id", "tokenizer.ggml.mask_token_id")] {
                if case let .uint32(v)? = file.metadata[metaKey] { out[jsonKey] = Int(v) }
            }
            if case let .bool(v)? = file.metadata["tokenizer.ggml.add_bos_token"] {
                out["add_bos_token"] = v
            }
            if case let .bool(v)? = file.metadata["tokenizer.ggml.add_space_prefix"] {
                out["add_space_prefix"] = v
            }
            if case let .string(v)? = file.metadata["tokenizer.ggml.model"] {
                out["ggml_model"] = v
            }
            let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
            try data.write(to: URL(fileURLWithPath: outPath))
            print("exported \(tokenCount) tokens to \(outPath) (\(data.count) bytes)")
        } catch {
            fail("gguf-tokenizer-export failed: \(error)")
        }

    case "gemma-tokenize":
        // Exact Gemma 4 tokenization of a text prompt (SPM-style BPE over
        // the GGUF vocab, space→▁ escape, newline pre-split, ranked merges,
        // <0xXX> byte fallback, forced BOS). --chat applies the model's
        // simple user→model chat path (bos + <|turn>user\n…<turn|>\n +
        // <|turn>model\n<|channel>thought\n<channel|>) derived from the
        // GGUF's embedded Jinja template (no tools/thinking path).
        // Prints IDs and the decode round-trip for verification.
        let path = requireValue("--file", args)
        let text = requireValue("--text", args)
        let chat = args.contains("--chat")
        do {
            let file = try GGUFReader.read(URL(fileURLWithPath: path))
            let tokenizer = try GemmaTokenizer(gguf: file)
            let input = chat
                ? "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                    + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                : text
            let ids = try tokenizer.encode(input)
            print(ids.map(String.init).joined(separator: " "))
            print("count: \(ids.count)")
            print("decoded: \(try tokenizer.decode(ids))")
        } catch {
            fail("gemma-tokenize failed: \(error)")
        }

    case "gemma-generate":
        // Bounded greedy text generation on the real GGUF. Input is EITHER
        // exact token IDs (--tokens, authoritative) or text (--text /
        // --chat) encoded by the exact in-repo tokenizer (BPE verified
        // against an independent Rust oracle). The KV cache continues
        // incrementally (each token dispatches one command buffer per
        // kernel, ~1799/token currently); generation stops at --max-tokens
        // or end-of-generation. Prints token IDs for exact baseline
        // comparison, the decoded text, and separately measured load /
        // prefill (TTFT) / per-token decode durations.
        let path = requireValue("--file", args)
        let maxTokens = args.firstIndex(of: "--max-tokens").map { _ in
            Int(requireValue("--max-tokens", args)) ?? 0
        } ?? 8
        guard maxTokens > 0, maxTokens <= 256 else { fail("--max-tokens must be 1...256") }
        // --repeat runs the SAME greedy generation N times in ONE process
        // (KV reset between repetitions) so repetition 1 measures a load +
        // first generation while later repetitions measure warm in-process
        // decode without re-paying model load. Determinism across
        // repetitions is asserted (identical token IDs) — greedy Metal
        // execution is expected to be bit-stable here.
        let repeatCount = args.firstIndex(of: "--repeat").map { _ in
            Int(requireValue("--repeat", args)) ?? 0
        } ?? 1
        guard repeatCount > 0, repeatCount <= 32 else { fail("--repeat must be 1...32") }
        do {
            // Parse and validate the prompt BEFORE any GPU allocation.
            let fileURL = URL(fileURLWithPath: path)
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
            var promptIDs: [Int]
            if let tokensArg = args.firstIndex(of: "--tokens").map({ _ in
                requireValue("--tokens", args) }) {
                promptIDs = tokensArg.split(separator: ",").map {
                    guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                          id >= 0, id < tokenizer.vocabSize else {
                        fail("invalid token id '\($0)'")
                    }
                    return id
                }
            } else {
                let text = requireValue("--text", args)
                let input = args.contains("--chat")
                    ? "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                    : text
                promptIDs = try tokenizer.encode(input)
            }
            guard !promptIDs.isEmpty, promptIDs.count + maxTokens <= 4096 else {
                fail("prompt + max-tokens must fit the 4096 context")
            }
            print("prompt ids: \(promptIDs.map(String.init).joined(separator: " "))")

            let loadStart = ContinuousClock.now
            let runner = try Gemma4Runner(fileAt: fileURL)
            func ms(_ t: ContinuousClock.Duration) -> Double {
                Double(t.components.seconds) * 1e3 + Double(t.components.attoseconds) / 1e15
            }
            let loadMs = ms(ContinuousClock.now - loadStart)

            // Optional stored-plan mode: the PRIMARY acceptance path. The
            // timed repetitions below execute the persisted validated SSA
            // schedule + selected candidate ONLY. Before them, ONE untimed
            // correctness gate replays the runner reference path (a
            // separate execution path sharing the same validated kernel
            // primitives, NOT a mathematically independent implementation)
            // and requires exact greedy token ids with rel < 1e-4 on every
            // logit array (both arrays finiteness-checked explicitly).
            var planOpt: GemmaForgePlan? = nil
            var graphOpt: GemmaGraphIR? = nil
            var memorySample = 0
            var gatedIds: [Int]? = nil
            if args.contains("--plan") {
                let prepStart = ContinuousClock.now
                let planPath = requireValue("--plan", args)
                let graphPath = requireValue("--graph", args)
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                var graphData = Data()
                while graphData.count <= 8 << 20 {
                    let chunk = try graphHandle.read(upToCount: 1 << 16) ?? Data()
                    if chunk.isEmpty { break }
                    graphData.append(chunk)
                }
                try? graphHandle.close()
                guard graphData.count <= 8 << 20 else {
                    fail("graph file > 8 MiB: \(graphPath)")
                }
                graphOpt = try GemmaGraph.parse(graphData)
                let engine = GemmaForge.engineDigest(try runner.engineIdentity())
                print("hashing GGUF for plan identity (streaming, one pass)…")
                let ggufDigest = try GemmaForge.fileDigest(fileURL)
                planOpt = try GemmaForge.loadPlan(
                    from: URL(fileURLWithPath: planPath))
                try GemmaForge.validatePlanIdentity(
                    planOpt!, graph: graphOpt!, ggufDigest: ggufDigest,
                    engineDigest: engine,
                    hardware: runner.hardwareProfile(),
                    expertCacheBudget: nil)
                print("plan identity validated: \(planOpt!.planDigest) "
                      + "(selected \(planOpt!.selected))")
                print(String(format: "identity validation prep: %.1f ms (graph parse + GGUF hash + plan load)",
                             ms(ContinuousClock.now - prepStart)))
            }
            if let plan = planOpt, let graph = graphOpt, args.contains("--gate") {
                // --gate is OPTIONAL and OFF for measurements: this untimed
                // reference replay warms every model resource, so a first
                // repetition after it cannot be called cold. Run it as a
                // separate verification invocation.
                runner.reset()
                let gateStart = ContinuousClock.now
                let prefill = try runner.forward(tokenIds: promptIDs)
                let prefillLogits = try runner.currentLogits()
                memorySample = prefill.metalAllocatedBytes
                var refIds = [prefill.argmaxToken]
                var stepLogits = [[Float]]()
                var prev = prefill.argmaxToken
                while refIds.count < maxTokens {
                    if tokenizer.isEndOfGeneration(prev) { break }
                    let ev = try runner.forward(tokenIds: [prev])
                    stepLogits.append(try runner.currentLogits())
                    refIds.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
                runner.reset()
                let (am, lg, _, _, _) = try GemmaForge.executeGraph(
                    graph, runner: runner, tokenIds: promptIDs,
                    candidate: plan.selected, schedule: plan.schedule)
                guard am == refIds[0],
                      lg.count == prefillLogits.count,
                      lg.allSatisfy({ $0.isFinite }),
                      prefillLogits.allSatisfy({ $0.isFinite }),
                      zip(lg, prefillLogits).allSatisfy({
                          abs(Double($0.0 - $0.1))
                              / max(1.0, abs(Double($0.1))) < 1e-4
                      }) else {
                    fail("plan gate: prefill diverged from reference")
                }
                var gateGen = [am]
                for step in 1..<min(refIds.count, maxTokens) {
                    let (next, sl, _, _, _) = try GemmaForge.executeGraphContinue(
                        graph, runner: runner, token: gateGen.last!,
                        candidate: plan.selected, schedule: plan.schedule)
                    let want = stepLogits[step - 1]
                    guard next == refIds[step],
                          sl.count == want.count,
                          sl.allSatisfy({ $0.isFinite }),
                          want.allSatisfy({ $0.isFinite }),
                          zip(sl, want).allSatisfy({
                              abs(Double($0.0 - $0.1))
                                  / max(1.0, abs(Double($0.1))) < 1e-4
                          }) else {
                        fail("plan gate: step \(step) diverged from reference")
                    }
                    gateGen.append(next)
                }
                guard gateGen == refIds else {
                    fail("plan gate: sequence differs from reference")
                }
                print(String(format: "plan gate passed (untimed reference replay, rel < 1e-4, exact ids): %.1f ms",
                             ms(ContinuousClock.now - gateStart)))
                // Timed repetitions must reproduce the gated sequence.
                gatedIds = gateGen
            }

            var referenceIds = gatedIds
            for repetition in 1...repeatCount {
                if repetition > 1 { runner.reset() }
                let prefillStart = ContinuousClock.now
                var gpuMicrosTotal = 0.0
                var firstToken = 0
                if let plan = planOpt, let graph = graphOpt {
                    let (am, _, gpuMicros, _, _) = try GemmaForge.executeGraph(
                        graph, runner: runner, tokenIds: promptIDs,
                        candidate: plan.selected, schedule: plan.schedule)
                    firstToken = am
                    gpuMicrosTotal = gpuMicros
                    if memorySample == 0 {
                        memorySample = runner.metalAllocatedSample()
                    }
                } else {
                    let evidence = try runner.forward(tokenIds: promptIDs)
                    firstToken = evidence.argmaxToken
                    gpuMicrosTotal = evidence.gpuMicros
                    memorySample = evidence.metalAllocatedBytes
                }
                let ttftMs = ms(ContinuousClock.now - prefillStart)

                var generated = [firstToken]
                var decodeMs = [Double]()
                while generated.count < maxTokens {
                    if tokenizer.isEndOfGeneration(generated.last!) {
                        print("eos: token \(generated.last!)")
                        break
                    }
                    let stepStart = ContinuousClock.now
                    if let plan = planOpt, let graph = graphOpt {
                        let (next, _, gpuMicros, _, _) =
                            try GemmaForge.executeGraphContinue(
                                graph, runner: runner, token: generated.last!,
                                candidate: plan.selected, schedule: plan.schedule)
                        gpuMicrosTotal += gpuMicros
                        generated.append(next)
                    } else {
                        let evidence = try runner.forward(
                            tokenIds: [generated.last!])
                        gpuMicrosTotal += evidence.gpuMicros
                        generated.append(evidence.argmaxToken)
                    }
                    decodeMs.append(ms(ContinuousClock.now - stepStart))
                }
                if let ref = referenceIds {
                    guard generated == ref else {
                        fail("repetition \(repetition) diverged: \(generated) != \(ref)")
                    }
                } else {
                    referenceIds = generated
                }
                print("generated ids: \(generated.map(String.init).joined(separator: " "))")
                print("generated text: \(try tokenizer.decode(generated))")
                let decodeSum = decodeMs.reduce(0, +)
                let medianText: String
                if decodeMs.isEmpty {
                    medianText = "n/a"
                } else {
                    let sorted = decodeMs.sorted()
                    let mid = sorted.count / 2
                    let median = sorted.count % 2 == 1
                        ? sorted[mid]
                        : (sorted[mid - 1] + sorted[mid]) / 2
                    medianText = String(format: "%.2f", median)
                }
                print(String(format: "repetition %d/%d: prefill/TTFT: %.1f ms (%d prompt tokens) | decode: %.1f ms total (%d steps), median ms/token %@ | gpu total: %.1f ms | metal device allocated sample: %.3f GiB",
                             repetition, repeatCount, ttftMs, promptIDs.count,
                             decodeSum, decodeMs.count, medianText,
                             gpuMicrosTotal / 1000,
                             Double(memorySample) / 1073741824.0))
            }
            print(String(format: "load (once): %.1f ms", loadMs))
            print("GENERATION_COMPLETED")
        } catch {
            fail("gemma-generate failed: \(error)")
        }

    case "gemma-config-export":
        // Exports the DSL config JSON for tools/forge_gemma capture from
        // the authorized GGUF (validated reader; no GPU allocation).
        do {
            let cfg = try Gemma4Config(file: GGUFReader.read(
                URL(fileURLWithPath: requireValue("--file", args))))
            let config: [String: Any] = [
                "layerCount": cfg.layerCount, "hidden": cfg.hiddenSize,
                "qHeads": cfg.qHeads, "localKVHeads": cfg.localKVHeads,
                "globalKVHeads": cfg.globalKVHeads,
                "localHeadDim": cfg.localHeadDim, "globalHeadDim": cfg.globalHeadDim,
                "slidingWindow": cfg.slidingWindow, "vocabSize": cfg.vocabSize,
                "expertCount": cfg.expertCount, "expertUsedCount": cfg.topKExperts,
                "expertIntermediate": cfg.expertIntermediate,
                "sharedIntermediate": cfg.sharedIntermediate,
                "finalSoftcap": cfg.finalSoftcap,
                "globalLayers": cfg.globalLayers,
            ]
            let out = requireValue("--out", args)
            let data = try JSONSerialization.data(withJSONObject: config,
                                                   options: [.sortedKeys, .prettyPrinted])
            try data.write(to: URL(fileURLWithPath: out))
            print("config exported: \(out)")
        } catch {
            fail("gemma-config-export failed: \(error)")
        }

    case "gemma-forge":
        // Gemma through the ACTUAL Forge pipeline: captured model.py graph
        // (restricted stage-list DSL, honestly labeled — typed operand IR
        // is the next milestone) → validation against the real GGUF →
        // full-logit correctness gates vs the independently validated
        // runner forward → measured candidate selection (wall median,
        // GPU median reported separately, baseline fallback with a
        // chosen 2% noise threshold) → plan cache with digest-based
        // integrity → verified graph-executed greedy generation (fixed
        // step count or EOS, a correctness benchmark, not general text
        // generation).
        let path = requireValue("--file", args)
        let graphPath = requireValue("--graph", args)
        let planPath = requireValue("--plan", args)
        let promptArg = args.firstIndex(of: "--prompt-ids").map { _ in
            requireValue("--prompt-ids", args) } ?? "2,9259"
        let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
            Int(requireValue("--gen-tokens", args)) ?? 0 } ?? 8
        guard genTokens >= 2, genTokens <= 64 else { fail("--gen-tokens must be 2...64") }
        do {
            // Optional chat/text workload: prompt IDs from the ORIGINAL
            // GGUF-backed tokenizer with the canonical chat template —
            // the exact same encode path as gemma-generate.
            var chatPromptIDs: [Int]? = nil
            if args.contains("--text") {
                let fileURL0 = URL(fileURLWithPath: path)
                let text = requireValue("--text", args)
                let input = args.contains("--chat")
                    ? "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                    : text
                chatPromptIDs = try GemmaTokenizer(
                    gguf: GGUFReader.read(fileURL0)).encode(input)
            }
            let fileURL = URL(fileURLWithPath: path)
            // Bounded graph read (never a full-file allocation first).
            let graphHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: graphPath))
            var graphData = Data()
            while graphData.count <= 8 << 20 {
                let chunk = try graphHandle.read(upToCount: 1 << 16) ?? Data()
                if chunk.isEmpty { break }
                graphData.append(chunk)
            }
            try? graphHandle.close()
            guard graphData.count <= 8 << 20 else {
                fail("graph file > 8 MiB: \(graphPath)")
            }
            let graph = try GemmaGraph.parse(graphData)
            // Validate graph + prompt against the GGUF-parsed config and
            // the real tensor directory BEFORE the runner (and GPU) is
            // allocated.
            let preFile = try GGUFReader.read(fileURL)
            let preConfig = try Gemma4Config(file: preFile)
            let prompt: [Int]
            if let chat = chatPromptIDs {
                prompt = chat
            } else {
                prompt = promptArg.split(separator: ",").map {
                    guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                          id >= 0, id < preConfig.vocabSize else {
                        fail("invalid prompt id '\($0)'"); return 0
                    }
                    return id
                }
            }
            let decodeTokens = GemmaForge.warmupTokens + GemmaForge.timedTokens
            guard !prompt.isEmpty,
                  prompt.count + max(genTokens, decodeTokens) <= 4096 else {
                fail("bad prompt workload")
            }
            try GemmaGraph.validate(graph, config: preConfig, file: preFile)

            // Optional SSD/streaming expert mode: non-expert weights are
            // compacted to a dedicated GPU buffer and expert slices
            // stream through a bounded LRU cache within this budget.
            // The budget/geometry is preflighted from the ALREADY
            // validated tensor directory BEFORE any runner allocation —
            // a malformed budget fails before the large KV caches are
            // built — and the runner uses the bounded-startup cache
            // constructor (no whole-file GPU mapping is ever created).
            var expertBudget: Int? = nil
            if args.contains("--expert-cache-budget") {
                guard let b = Int(requireValue("--expert-cache-budget", args)), b > 0 else {
                    fail("--expert-cache-budget must be a positive integer (bytes)")
                }
                func strideOf(_ name: String) -> Int? {
                    guard let info = try? preFile.tensor(name),
                          info.ggmlType == GGMLTensorType.q4_0.rawValue,
                          info.dims.count == 3, info.dims[0] % 32 == 0 else { return nil }
                    return info.dims[0] / 32 * 18 * info.dims[1]
                }
                guard let gu = strideOf("blk.0.ffn_gate_up_exps.weight"),
                      let dn = strideOf("blk.0.ffn_down_exps.weight"),
                      gu > 0, dn > 0,
                      ExpertCacheGeometry.slotCounts(budgetBytes: b,
                                                     guStrideBytes: gu,
                                                     dnStrideBytes: dn) != nil else {
                    fail("--expert-cache-budget \(b) cannot hold one gate-up + one down expert slice")
                }
                expertBudget = b
            }
            let runner = try Gemma4Runner(fileAt: fileURL,
                                          expertCacheBudgetBytes: expertBudget)
            if let b = expertBudget {
                print("expert cache mode: budget \(b) bytes (bounded startup, no whole-file mapping)")
            }
            let hardware = runner.hardwareProfile()
            let engine = GemmaForge.engineDigest(try runner.engineIdentity())
            print("graph validated: \(graph.ops.count) ops, model \(graph.model)")
            print("hardware: \(hardware.deviceName), max threads/threadgroup \(hardware.maxTotalThreadsPerThreadgroup), exec width \(hardware.threadExecutionWidth), working set \(hardware.recommendedMaxWorkingSetSize) bytes")
            print("engine digest: \(engine)")

            var plan: GemmaForgePlan
            var forgedFresh = false
            let planURL = URL(fileURLWithPath: planPath)
            if FileManager.default.fileExists(atPath: planPath) {
                print("hashing GGUF for plan identity (streaming, one pass)…")
                let ggufDigest = try GemmaForge.fileDigest(fileURL)
                if let loaded = try? GemmaForge.loadPlan(from: planURL),
                   (try? GemmaForge.validatePlan(loaded, graph: graph,
                                                 ggufDigest: ggufDigest,
                                                 engineDigest: engine,
                                                 hardware: hardware,
                                                 prompt: prompt,
                                                 decodeTokens: decodeTokens,
                                                 expertCacheBudget: expertBudget)) != nil {
                    // Cache reuse still passes the correctness gate before
                    // any output is trusted.
                    runner.reset()
                    let ref = try runner.forward(tokenIds: prompt)
                    let refLogits = try runner.currentLogits()
                    runner.reset()
                    let (am, lg, _, _, _) = try GemmaForge.executeGraph(
                        graph, runner: runner, tokenIds: prompt,
                        candidate: loaded.selected, schedule: loaded.schedule)
                    guard am == ref.argmaxToken, lg.count == refLogits.count,
                          zip(lg, refLogits).allSatisfy({
                              abs(Double($0.0 - $0.1))
                                  / max(1.0, abs(Double($0.1))) < 1e-4
                          }) else {
                        fail("cached plan \(loaded.planDigest) failed the correctness gate")
                    }
                    print("cache hit: plan \(loaded.planDigest)")
                    plan = loaded
                } else {
                    print("stale/foreign plan at \(planPath) — re-forging")
                    let forged = try GemmaForge.forge(graph: graph, runner: runner,
                                                      ggufURL: fileURL, prompt: prompt,
                                                      decodeTokens: decodeTokens,
                                                      hardware: hardware)
                    plan = forged.plan
                    forgedFresh = true
                    try GemmaForge.savePlan(plan, to: planURL)
                }
            } else {
                let forged = try GemmaForge.forge(graph: graph, runner: runner,
                                                  ggufURL: fileURL, prompt: prompt,
                                                  decodeTokens: decodeTokens,
                                                  hardware: hardware)
                plan = forged.plan
                forgedFresh = true
                try GemmaForge.savePlan(plan, to: planURL)
            }
            for r in plan.results {
                print(String(format: "  candidate batching=%d tiledQ4=%d tiledQ6=%d rmsTiled=%d f32Tiled=%d: correct %d, wall %@/token, gpu %@/token, %@ kernels/token, %@ buffers/token — %@",
                             r.candidate.segmentBatching ? 1 : 0,
                             r.candidate.matvecTiledQ4 ? 1 : 0,
                             r.candidate.matvecTiledQ6 ? 1 : 0,
                             r.candidate.rmsnormTiled ? 1 : 0,
                             r.candidate.matvecTiledF32 ? 1 : 0,
                             r.correct ? 1 : 0,
                             r.medianWallMicrosPerToken.map { String(format: "%.0f us", $0) } ?? "-",
                             r.medianGpuMicrosPerToken.map { String(format: "%.0f us", $0) } ?? "-",
                             r.kernelsPerToken.map { String(format: "%.0f", $0) } ?? "-",
                             r.buffersPerToken.map { String(format: "%.0f", $0) } ?? "-",
                             r.note))
            }
            print(String(format: "selected: batching=%d tiledQ4=%d tiledQ6=%d rmsTiled=%d f32Tiled=%d (tile widths q4=%d q6=%d rms=%d f32=%d), wall %.0f us/token, gpu %.0f us/token",
                         plan.selected.segmentBatching ? 1 : 0,
                         plan.selected.matvecTiledQ4 ? 1 : 0,
                         plan.selected.matvecTiledQ6 ? 1 : 0,
                         plan.selected.rmsnormTiled ? 1 : 0,
                         plan.selected.matvecTiledF32 ? 1 : 0,
                         runner.tileWidths.q4, runner.tileWidths.q6,
                         runner.tileWidths.rms, runner.tileWidths.f32,
                         plan.selectedMedianWallMicrosPerToken,
                         plan.selectedMedianGpuMicrosPerToken))
            if forgedFresh {
                print("plan saved: \(planPath), digest \(plan.planDigest)")
            } else {
                print("plan reused from cache (not rewritten): digest \(plan.planDigest)")
            }

            // Final verified generation: capture the reference greedy
            // sequence ONCE with the independently validated runner
            // forward, then replay it through the graph executor, gating
            // prefill AND every step (exact next token + full 262144
            // logits < 1e-4). Fixed step count or EOS — a correctness
            // benchmark, not general text generation.
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
            runner.reset()
            let prefill = try runner.forward(tokenIds: prompt)
            let prefillLogits = try runner.currentLogits()
            var refSeq = [prefill.argmaxToken]
            var refStepLogits = [[Float]]()
            var prev = prefill.argmaxToken
            while refSeq.count < genTokens {
                let ev = try runner.forward(tokenIds: [prev])
                refStepLogits.append(try runner.currentLogits())
                refSeq.append(ev.argmaxToken)
                prev = ev.argmaxToken
            }
            runner.reset()
            let (graphArgmax, graphPrefillLogits, pgpu, _, _) = try GemmaForge.executeGraph(
                graph, runner: runner, tokenIds: prompt, candidate: plan.selected,
                schedule: plan.schedule)
            // Gate the graph prefill itself against the reference prefill.
            guard graphArgmax == prefill.argmaxToken,
                  graphPrefillLogits.count == prefillLogits.count,
                  graphPrefillLogits.allSatisfy({ $0.isFinite }),
                  zip(graphPrefillLogits, prefillLogits).allSatisfy({
                      abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                  }) else {
                fail("graph prefill diverged from runner reference")
            }
            var gen = [graphArgmax]
            var wallSamples: [Double] = []
            var gpuSum = pgpu
            for step in 1..<genTokens {
                if tokenizer.isEndOfGeneration(gen.last!) {
                    print("eos at step \(step): token \(gen.last!) — stopping early")
                    refSeq = Array(refSeq.prefix(step))
                    break
                }
                let tok = refSeq[step - 1]
                let t0 = DispatchTime.now()
                let (next, lg, gpu, kernels, buffers) = try GemmaForge.executeGraphContinue(
                    graph, runner: runner, token: tok, candidate: plan.selected,
                    schedule: plan.schedule)
                let wall = Double(DispatchTime.now().uptimeNanoseconds
                                  - t0.uptimeNanoseconds) / 1000
                let want = refStepLogits[step - 1]
                guard next == refSeq[step], lg.count == want.count,
                      lg.allSatisfy({ $0.isFinite }),
                      zip(lg, want).allSatisfy({
                          abs(Double($0.0 - $0.1))
                              / max(1.0, abs(Double($0.1))) < 1e-4
                      }) else {
                    fail("graph step \(step) diverged from runner reference")
                }
                gen.append(next)
                wallSamples.append(wall)
                gpuSum += gpu
                print(String(format: "  step %d: token %d, wall %.0f us, gpu %.0f us, %d kernels, %d buffers",
                             step, next, wall, gpu, kernels, buffers))
            }
            let sortedWall = wallSamples.sorted()
            let mid = sortedWall.count / 2
            let medianWall = sortedWall.count % 2 == 1
                ? sortedWall[mid] : (sortedWall[mid - 1] + sortedWall[mid]) / 2
            print("prompt ids: \(prompt.map(String.init).joined(separator: " "))")
            print("greedy ids (reference == graph): \(refSeq.map(String.init).joined(separator: " "))")
            print("text: \(try tokenizer.decode(refSeq))")
            print(String(format: "verified generation: %d steps, median wall %.0f us/token, gpu total %.1f ms",
                         wallSamples.count, medianWall, gpuSum / 1000))
            if let st = runner.expertCacheStats() {
                print("expert cache stats: budget \(st.budgetBytes) B, slot pools \(st.residentBytes) B (gu \(st.guSlots) x \(st.guStrideBytes) + dn \(st.dnSlots) x \(st.dnStrideBytes)), non-expert compact buffer \(st.nonExpertResidentBytes) B")
                print("  misses \(st.misses), hits \(st.hits) (prefetch-served \(st.prefetchHits), prefetch installed \(st.prefetchSlices) slices / \(st.prefetchBytes) B), evictions \(st.evictions), bytes read \(st.sliceBytesRead) (pread may hit OS page cache — no cold-SSD claim), flushes-before-reuse \(st.flushesBeforeReuse)")
                print("  peak pinned \(st.peakPinnedBytes) B, device allocated sampled at end of construction \(st.initDeviceAllocatedBytes) B (sample, not a guaranteed startup peak), peak sampled \(st.peakDeviceAllocatedBytes) B")
                let enc = JSONEncoder()
                enc.outputFormatting = [.sortedKeys]
                if let js = try? enc.encode(st),
                   let line = String(data: js, encoding: .utf8) {
                    print("EXPERT_CACHE_STATS_JSON \(line)")
                }
            }
            print("FORGE_PASS")
        } catch {
            fail("gemma-forge failed: \(error)")
        }

    case "gemma-expert-cache-verify":
        // Forced eviction / miss / hit / in-flight verification of the
        // bounded SSD expert cache THROUGH THE ACTUAL SSA EXECUTION
        // PATH, against a resident-weights reference snapshot:
        //   Phase 1: resident runner — a FAILED enable attempt (budget
        //   too small) must throw and leave resident mode intact; then
        //   full-logit snapshot of prefill + decode steps; the runner
        //   is RELEASED before any cache runner is built.
        //   Phase 2: cache runner with a tiny budget (forced thrash),
        //   in BOTH per-kernel and segment-batching modes — the
        //   batching pass exercises in-flight pinning and must show
        //   flushes-before-reuse > 0; exact greedy tokens + full-logit
        //   parity gate every pass before anything is trusted.
        //   Phase 3: fresh cache runner with a larger budget — the
        //   same prefill replayed twice; the observed hit count is
        //   reported (whatever it is — prompt and decode working sets
        //   differ, so no full-replay-hit assumption is made), gated
        //   only on hits > 0 with identical logits.
        // Reads are preads that may hit the OS page cache — no
        // cold-SSD latency claim is made or measured here.
        let path = requireValue("--file", args)
        let graphPath = requireValue("--graph", args)
        let promptArg = args.firstIndex(of: "--prompt-ids").map { _ in
            requireValue("--prompt-ids", args) } ?? "2,9259"
        func intArg(_ name: String, _ def: Int) -> Int {
            guard let i = args.firstIndex(of: name) else { return def }
            guard let v = Int(requireValue(name, args)) else {
                fail("\(name) must be an integer"); return def }
            return v
        }
        let stepsArg = intArg("--steps", 4)
        let smallBudget = intArg("--small-budget", 8_000_000)
        let largeBudget = intArg("--large-budget", 1_000_000_000)
        guard stepsArg >= 1, stepsArg <= 16,
              smallBudget > 0, largeBudget > smallBudget else {
            fail("bad --steps/--small-budget/--large-budget")
        }
        do {
            let fileURL = URL(fileURLWithPath: path)
            // Bounded graph read (same policy as gemma-forge).
            let graphHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: graphPath))
            var graphData = Data()
            while graphData.count <= 8 << 20 {
                let chunk = try graphHandle.read(upToCount: 1 << 16) ?? Data()
                if chunk.isEmpty { break }
                graphData.append(chunk)
            }
            try? graphHandle.close()
            guard graphData.count <= 8 << 20 else { fail("graph file > 8 MiB") }
            let graph = try GemmaGraph.parse(graphData)
            let preFile = try GGUFReader.read(fileURL)
            let preConfig = try Gemma4Config(file: preFile)
            let prompt = promptArg.split(separator: ",").map {
                guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                      id >= 0, id < preConfig.vocabSize else {
                    fail("invalid prompt id '\($0)')"); return 0 }
                return id
            }
            guard !prompt.isEmpty, prompt.count + stepsArg <= 4096 else {
                fail("bad prompt workload")
            }
            try GemmaGraph.validate(graph, config: preConfig, file: preFile)
            let reference = GemmaForgeCandidate(segmentBatching: false,
                                                top5Diagnostic: false)
            let batching = GemmaForgeCandidate(segmentBatching: true,
                                               top5Diagnostic: false)

            // Strict logit comparison: empty/count-mismatch/non-finite
            // on EITHER side is a hard failure before any max-error
            // (Swift's max can hide NaN).
            func worstRel(_ a: [Float], _ b: [Float]) throws -> Double {
                guard !a.isEmpty, a.count == b.count,
                      a.allSatisfy({ $0.isFinite }),
                      b.allSatisfy({ $0.isFinite }) else {
                    throw GemmaForgeError.executionFailed(
                        "logit vectors empty/count-mismatch/non-finite")
                }
                var worst = 0.0
                for i in 0..<a.count {
                    let d = abs(Double(a[i] - b[i])) / max(1.0, abs(Double(b[i])))
                    if d.isNaN { throw GemmaForgeError.executionFailed("NaN rel error") }
                    worst = max(worst, d)
                }
                return worst
            }

            // ---- Phase 1: failed-enable rollback + resident snapshot.
            func residentReference() throws
                    -> (promptLogits: [Float], tokens: [Int],
                        stepLogits: [[Float]]) {
                let runner = try Gemma4Runner(fileAt: fileURL)
                // Rollback regression: an impossible budget must throw
                // and leave the resident whole-file mode fully intact.
                guard (try? runner.enableExpertCache(budgetBytes: 1)) == nil else {
                    throw GemmaForgeError.executionFailed(
                        "budget-1 enable unexpectedly succeeded")
                }
                runner.reset()
                let prefill = try runner.forward(tokenIds: prompt)
                let promptLogits = try runner.currentLogits()
                var tokens = [prefill.argmaxToken]
                var stepLogits = [[Float]]()
                var prev = prefill.argmaxToken
                while tokens.count <= stepsArg {
                    let ev = try runner.forward(tokenIds: [prev])
                    stepLogits.append(try runner.currentLogits())
                    tokens.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
                return (promptLogits, tokens, stepLogits)
            }
            let snapshot = try autoreleasepool { try residentReference() }
            if let dev = MTLCreateSystemDefaultDevice() {
                print("device allocated after resident release, before cache init (sample): \(dev.currentAllocatedSize) B")
            }
            print("phase 1: failed-enable rejected; resident reference \(prompt.count) prompt tokens + \(stepsArg) steps captured; runner released")

            // ---- Phase 2: tiny-budget cache runner (forced eviction),
            // per-kernel AND segment-batching (in-flight pinning) modes.
            func cachePass(budget: Int, candidate: GemmaForgeCandidate,
                           replays: Int, prefetch: Bool = true) throws
                    -> (worstRel: Double, tokensMatch: Bool,
                        stats: Gemma4ExpertCacheStats) {
                let runner = try Gemma4Runner(fileAt: fileURL,
                                              expertCacheBudgetBytes: budget)
                runner.expertPrefetch = prefetch
                guard let st0 = runner.expertCacheStats() else {
                    throw GemmaForgeError.executionFailed("cache stats missing")
                }
                var stepWorst = 0.0
                var tokensMatch = true
                for _ in 0..<replays {
                    runner.reset()
                    let (am, lg, _, _, _) = try GemmaForge.executeGraph(
                        graph, runner: runner, tokenIds: prompt,
                        candidate: candidate)
                    tokensMatch = tokensMatch && am == snapshot.tokens[0]
                    stepWorst = max(stepWorst, try worstRel(lg, snapshot.promptLogits))
                    var prev = am
                    for s in 0..<stepsArg {
                        let (next, sl, _, _, _) = try GemmaForge.executeGraphContinue(
                            graph, runner: runner, token: prev, candidate: candidate)
                        tokensMatch = tokensMatch && next == snapshot.tokens[s + 1]
                        stepWorst = max(stepWorst, try worstRel(sl, snapshot.stepLogits[s]))
                        prev = next
                    }
                }
                guard let st = runner.expertCacheStats() else {
                    throw GemmaForgeError.executionFailed("cache stats missing")
                }
                _ = st0
                return (stepWorst, tokensMatch, st)
            }
            func reportStats(_ tag: String, budget: Int, _ st: Gemma4ExpertCacheStats) {
                print("\(tag)buffer-length ledger (budget \(budget) B): slot pools \(st.residentBytes) B (gu \(st.guSlots) x \(st.guStrideBytes) + dn \(st.dnSlots) x \(st.dnStrideBytes)), non-expert compact buffer \(st.nonExpertResidentBytes) B")
                print("\(tag)measured samples: device allocated sampled at end of construction \(st.initDeviceAllocatedBytes) B (a SAMPLE — not a guaranteed startup peak), peak sampled \(st.peakDeviceAllocatedBytes) B, peak pinned \(st.peakPinnedBytes) B")
                print("\(tag)io counters: misses \(st.misses), hits \(st.hits) (of which prefetch-served \(st.prefetchHits); prefetch installed \(st.prefetchSlices) slices / \(st.prefetchBytes) B), evictions \(st.evictions), bytes read \(st.sliceBytesRead) (pread may hit OS page cache — no cold-SSD claim), flushes-before-reuse \(st.flushesBeforeReuse)")
                let enc = JSONEncoder()
                enc.outputFormatting = [.sortedKeys]
                if let js = try? enc.encode(st),
                   let line = String(data: js, encoding: .utf8) {
                    print("\(tag)EXPERT_CACHE_STATS_JSON \(line)")
                }
            }

            let smallPerKernel = try autoreleasepool {
                try cachePass(budget: smallBudget, candidate: reference,
                              replays: 1, prefetch: false)
            }
            guard smallPerKernel.tokensMatch, smallPerKernel.worstRel < 1e-4,
                  smallPerKernel.stats.misses > 0,
                  smallPerKernel.stats.evictions > 0,
                  smallPerKernel.stats.hits == 0 else {
                fail(String(format: "small-budget per-kernel pass failed: tokensMatch %@, worstRel %.3e, misses %d, evictions %d, hits %d",
                            smallPerKernel.tokensMatch ? "yes" : "NO",
                            smallPerKernel.worstRel,
                            smallPerKernel.stats.misses,
                            smallPerKernel.stats.evictions,
                            smallPerKernel.stats.hits))
            }
            print(String(format: "phase 2a per-kernel (budget %d B): exact tokens, worst rel %.3e",
                         smallBudget, smallPerKernel.worstRel))
            reportStats("  ", budget: smallBudget, smallPerKernel.stats)

            // Segment batching with an insufficient slot count is what
            // actually exercises in-flight pins: encoded expert matvecs
            // hold slots in flight until a flush-before-reuse.
            let smallBatched = try autoreleasepool {
                try cachePass(budget: smallBudget, candidate: batching,
                              replays: 1, prefetch: false)
            }
            guard smallBatched.tokensMatch, smallBatched.worstRel < 1e-4,
                  smallBatched.stats.misses > 0,
                  smallBatched.stats.evictions > 0,
                  smallBatched.stats.flushesBeforeReuse > 0,
                  smallBatched.stats.peakPinnedBytes <= smallBatched.stats.residentBytes,
                  smallBatched.stats.peakPinnedBytes <= smallBudget else {
                fail(String(format: "small-budget batching pass failed: tokensMatch %@, worstRel %.3e, flushesBeforeReuse %d, peakPinned %d (resident %d, budget %d)",
                            smallBatched.tokensMatch ? "yes" : "NO",
                            smallBatched.worstRel,
                            smallBatched.stats.flushesBeforeReuse,
                            smallBatched.stats.peakPinnedBytes,
                            smallBatched.stats.residentBytes, smallBudget))
            }
            print(String(format: "phase 2b batching (budget %d B): exact tokens, worst rel %.3e, in-flight flushes-before-reuse %d observed",
                         smallBudget, smallBatched.worstRel,
                         smallBatched.stats.flushesBeforeReuse))
            reportStats("  ", budget: smallBudget, smallBatched.stats)

            // ---- Phase 3: larger-budget cache runner, replayed twice —
            // reported hits are the OBSERVED count (prompt and decode
            // working sets differ; no full-replay-hit assumption).
            let large = try autoreleasepool {
                try cachePass(budget: largeBudget, candidate: reference, replays: 2)
            }
            guard large.tokensMatch, large.worstRel < 1e-4,
                  large.stats.hits > 0 else {
                fail(String(format: "large-budget pass failed: tokensMatch %@, worstRel %.3e, hits %d",
                            large.tokensMatch ? "yes" : "NO",
                            large.worstRel, large.stats.hits))
            }
            let totalLoads = large.stats.hits + large.stats.misses
            print(String(format: "phase 3 (budget %d B, 2 replays, prefetch on): exact tokens, worst rel %.3e; observed hits %d of %d loads (%.1f%%, prefetch-served %d)",
                         largeBudget, large.worstRel, large.stats.hits,
                         totalLoads,
                         100.0 * Double(large.stats.hits) / Double(totalLoads),
                         large.stats.prefetchHits))
            reportStats("  ", budget: largeBudget, large.stats)
            // Prefetch A/B: identical workload with prefetch disabled —
            // logits and greedy tokens must be unchanged (prefetch never
            // evicts and loads byte-identical slices); hit/eviction
            // counters are reported, whatever they are.
            let noPf = try autoreleasepool {
                try cachePass(budget: largeBudget, candidate: reference,
                              replays: 2, prefetch: false)
            }
            guard noPf.tokensMatch, noPf.worstRel < 1e-4 else {
                fail(String(format: "prefetch-off pass failed: tokensMatch %@, worstRel %.3e",
                            noPf.tokensMatch ? "yes" : "NO", noPf.worstRel))
            }
            let pfLoads = noPf.stats.hits + noPf.stats.misses
            print(String(format: "phase 3b (same workload, prefetch OFF): exact tokens, worst rel %.3e; hits %d of %d loads (%.1f%%), evictions %d vs %d with prefetch",
                         noPf.worstRel, noPf.stats.hits, pfLoads,
                         100.0 * Double(noPf.stats.hits) / Double(pfLoads),
                         noPf.stats.evictions, large.stats.evictions))
            reportStats("  ", budget: largeBudget, noPf.stats)
            print("EXPERT_CACHE_PASS")
        } catch {
            fail("gemma-expert-cache-verify failed: \(error)")
        }

    case "gemma-serve":
        // Bounded LOCALHOST serving from a validated persisted SSA plan
        // artifact. No model.py, no agents, no Python at runtime — the
        // server loads the plan (typed SSA schedule + selected
        // candidate), checks its identity (graph/GGUF content/engine/
        // hardware/toolchain/schedule/planDigest WITHOUT the workload
        // binding — the selected candidate was measured on the plan's
        // own profiling workload, disclosed at startup; no
        // fastest-on-all-prompts claim), and serves a bounded greedy
        // subset: one POST /generate at a time (serial GPU), plus GET
        // /health (no body). Runtime executes the STORED schedule and
        // selected flags only; by default no per-request reference
        // replay (correctness gates ran at forge time) —
        // --gate-first-request additionally gates the first served
        // generation against the runner forward path (a SEPARATE
        // execution path sharing the same validated kernel primitives,
        // NOT a mathematically independent implementation): exact greedy
        // tokens and full-logit rel < 1e-4 at every step, finiteness on
        // both arrays, and final exact sequence equality. Bounded
        // everywhere: request count, prompt length, output tokens,
        // header/body sizes, one absolute per-request read deadline
        // enforced on EVERY recv (socket timeouts shrink to the
        // remaining budget); duplicate Content-Length, Transfer-Encoding,
        // malformed header lines, fractional/boolean-typed max_tokens and
        // non-boolean chat are rejected; Content-Type base media type
        // must be exactly application/json (parameters allowed).
        signal(SIGPIPE, SIG_IGN)
        let path = requireValue("--file", args)
        let graphPath = requireValue("--graph", args)
        let planPath = requireValue("--plan", args)
        func intArgServe(_ name: String, _ def: Int) -> Int {
            guard let i = args.firstIndex(of: name) else { return def }
            guard let v = Int(requireValue(name, args)) else {
                fail("\(name) must be an integer"); return def }
            return v
        }
        let port = intArgServe("--port", 8377)
        let maxTokens = intArgServe("--max-tokens", 16)
        let maxPromptChars = intArgServe("--max-prompt-chars", 512)
        let maxRequests = intArgServe("--max-requests", 64)
        let gateFirstRequest = args.contains("--gate-first-request")
        guard port >= 1024, port <= 65535,
              maxTokens >= 1, maxTokens <= 64,
              maxPromptChars >= 1, maxPromptChars <= 2048,
              maxRequests >= 1, maxRequests <= 1024 else {
            fail("--port must be 1024...65535; --max-tokens 1...64; --max-prompt-chars 1...2048; --max-requests 1...1024")
        }
        do {
            let fileURL = URL(fileURLWithPath: path)
            let graphHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: graphPath))
            var graphData = Data()
            while graphData.count <= 8 << 20 {
                let chunk = try graphHandle.read(upToCount: 1 << 16) ?? Data()
                if chunk.isEmpty { break }
                graphData.append(chunk)
            }
            try? graphHandle.close()
            guard graphData.count <= 8 << 20 else { fail("graph file > 8 MiB") }
            let graph = try GemmaGraph.parse(graphData)
            let preFile = try GGUFReader.read(fileURL)
            let preConfig = try Gemma4Config(file: preFile)
            try GemmaGraph.validate(graph, config: preConfig, file: preFile)
            var expertBudget: Int? = nil
            if args.contains("--expert-cache-budget") {
                guard let b = Int(requireValue("--expert-cache-budget", args)), b > 0 else {
                    fail("--expert-cache-budget must be a positive integer (bytes)")
                }
                func strideOf(_ name: String) -> Int? {
                    guard let info = try? preFile.tensor(name),
                          info.ggmlType == GGMLTensorType.q4_0.rawValue,
                          info.dims.count == 3, info.dims[0] % 32 == 0 else { return nil }
                    return info.dims[0] / 32 * 18 * info.dims[1]
                }
                guard let gu = strideOf("blk.0.ffn_gate_up_exps.weight"),
                      let dn = strideOf("blk.0.ffn_down_exps.weight"),
                      gu > 0, dn > 0,
                      ExpertCacheGeometry.slotCounts(budgetBytes: b,
                                                     guStrideBytes: gu,
                                                     dnStrideBytes: dn) != nil else {
                    fail("--expert-cache-budget \(b) cannot hold one gate-up + one down expert slice")
                }
                expertBudget = b
            }
            let runner = try Gemma4Runner(fileAt: fileURL,
                                          expertCacheBudgetBytes: expertBudget)
            let hardware = runner.hardwareProfile()
            let engine = GemmaForge.engineDigest(try runner.engineIdentity())
            print("hashing GGUF for plan identity (streaming, one pass)…")
            let ggufDigest = try GemmaForge.fileDigest(fileURL)
            let plan = try GemmaForge.loadPlan(from: URL(fileURLWithPath: planPath))
            try GemmaForge.validatePlanIdentity(plan, graph: graph,
                                                ggufDigest: ggufDigest,
                                                engineDigest: engine,
                                                hardware: hardware,
                                                expertCacheBudget: expertBudget)
            print("plan identity validated: \(plan.planDigest)")
            print("disclosure: selected candidate \(plan.selected) was measured on the plan's profiling workload (\(plan.promptTokenCount) prompt tokens, \(plan.decodeTokens) decode tokens, workload digest \(plan.workloadDigest)); serving uses it as the engine configuration for a bounded greedy subset — no fastest-on-all-prompts claim")
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))

            // Gate-mode reference (runner forward path — shares the
            // validated kernel primitives, not independent math).
            func referenceGeneration(promptIDs: [Int], tokens: Int) throws
                    -> (ids: [Int], stepLogits: [[Float]], prefillLogits: [Float]) {
                runner.reset()
                let prefill = try runner.forward(tokenIds: promptIDs)
                let prefillLogits = try runner.currentLogits()
                var ids = [prefill.argmaxToken]
                var stepLogits = [[Float]]()
                var prev = prefill.argmaxToken
                while ids.count < tokens {
                    if tokenizer.isEndOfGeneration(prev) { break }
                    let ev = try runner.forward(tokenIds: [prev])
                    stepLogits.append(try runner.currentLogits())
                    ids.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
                return (ids, stepLogits, prefillLogits)
            }

            // Production path: execute the STORED schedule + selected
            // flags only. Exactly `tokens` output tokens (EOS stops
            // early); wall times measured for the whole request, the
            // prefill, and each decode step.
            func generate(promptText: String, chat: Bool, tokens: Int,
                          reference: (ids: [Int], stepLogits: [[Float]],
                                      prefillLogits: [Float])? = nil) throws
                    -> (ids: [Int], text: String, medianWall: Double,
                        requestWall: Double, prefillWall: Double) {
                let requestStart = DispatchTime.now()
                let input = chat
                    ? "<|turn>user\n" + promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                    : promptText
                let promptIDs = try tokenizer.encode(input)
                guard promptIDs.count + tokens <= 4096 else {
                    throw GemmaForgeError.executionFailed("prompt+output exceeds context")
                }
                runner.reset()
                let p0 = DispatchTime.now()
                let (am, lg, _, _, _) = try GemmaForge.executeGraph(
                    graph, runner: runner, tokenIds: promptIDs,
                    candidate: plan.selected, schedule: plan.schedule)
                let prefillWall = Double(DispatchTime.now().uptimeNanoseconds
                                         - p0.uptimeNanoseconds) / 1000
                if let ref = reference {
                    guard am == ref.ids[0],
                          lg.count == ref.prefillLogits.count,
                          lg.allSatisfy({ $0.isFinite }),
                          ref.prefillLogits.allSatisfy({ $0.isFinite }),
                          zip(lg, ref.prefillLogits).allSatisfy({
                              abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                          }) else {
                        throw GemmaForgeError.executionFailed(
                            "gated prefill diverged from reference")
                    }
                }
                var gen = [am]
                var wallSamples: [Double] = []
                for step in 1..<tokens {
                    if tokenizer.isEndOfGeneration(gen.last!) { break }
                    let t0 = DispatchTime.now()
                    let (next, sl, _, _, _) = try GemmaForge.executeGraphContinue(
                        graph, runner: runner, token: gen.last!,
                        candidate: plan.selected, schedule: plan.schedule)
                    let wall = Double(DispatchTime.now().uptimeNanoseconds
                                      - t0.uptimeNanoseconds) / 1000
                    if let ref = reference, step - 1 < ref.stepLogits.count {
                        let want = ref.stepLogits[step - 1]
                        // Both arrays finiteness-checked explicitly
                        // before any comparison (max can hide NaN).
                        guard next == ref.ids[step],
                              sl.count == want.count,
                              sl.allSatisfy({ $0.isFinite }),
                              want.allSatisfy({ $0.isFinite }),
                              zip(sl, want).allSatisfy({
                                  abs(Double($0.0 - $0.1)) / max(1.0, abs(Double($0.1))) < 1e-4
                              }) else {
                            throw GemmaForgeError.executionFailed(
                                "gated step \(step) diverged from reference")
                        }
                    }
                    gen.append(next)
                    wallSamples.append(wall)
                }
                // No skipped comparisons: the ENTIRE gated sequence
                // must equal the reference sequence exactly.
                if let ref = reference {
                    guard gen == ref.ids else {
                        throw GemmaForgeError.executionFailed(
                            "gated sequence differs from reference")
                    }
                }
                let sorted = wallSamples.sorted()
                let mid = sorted.count / 2
                let median = sorted.isEmpty ? 0
                    : (sorted.count % 2 == 1 ? sorted[mid]
                                             : (sorted[mid - 1] + sorted[mid]) / 2)
                let requestWall = Double(DispatchTime.now().uptimeNanoseconds
                                         - requestStart.uptimeNanoseconds) / 1000
                return (gen, try tokenizer.decode(gen), median, requestWall,
                        prefillWall)
            }

            // ---- minimal bounded HTTP server on 127.0.0.1 ----
            let maxHeaderBytes = 8192
            let maxBodyBytes = 4096
            let absoluteDeadlineNs: UInt64 = 15_000_000_000   // 15 s per request
            let serverFD = socket(AF_INET, Int32(SOCK_STREAM), 0)
            guard serverFD >= 0 else { fail("socket() failed: errno \(errno)") }
            var one: Int32 = 1
            setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR, &one,
                       socklen_t(MemoryLayout<Int32>.size))
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = UInt16(port).bigEndian
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let bindRC = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(serverFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindRC == 0 else { fail("bind(127.0.0.1:\(port)) failed: errno \(errno)") }
            guard listen(serverFD, 1) == 0 else { fail("listen() failed: errno \(errno)") }
            print("serving http://127.0.0.1:\(port)/generate (serial, \(maxRequests)-request bound, prompt ≤ \(maxPromptChars) chars, ≤ \(maxTokens) tokens, body ≤ \(maxBodyBytes) B, 15 s absolute read deadline)")
            print("SERVE_READY")

            func sendHTTP(_ fd: Int32, _ status: String, _ body: Data) {
                let head = "HTTP/1.1 \(status)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
                let payload = head.data(using: .utf8)! + body
                payload.withUnsafeBytes { raw in
                    var sent = 0
                    while sent < raw.count {
                        let n = Darwin.send(fd, raw.baseAddress!.advanced(by: sent),
                                            raw.count - sent, 0)
                        if n <= 0 { break }
                        sent += n
                    }
                }
            }
            func jsonError(_ message: String) -> Data {
                (try? JSONSerialization.data(withJSONObject: ["error": message])) ?? Data()
            }
            // Strict JSON type helpers: a JSON true/false is a
            // CFBoolean; a JSON number is not.
            func isJSONBoolean(_ v: Any) -> Bool {
                guard let n = v as? NSNumber else { return false }
                return CFGetTypeID(n) == CFBooleanGetTypeID()
            }
            func isJSONInteger(_ v: Any) -> Bool {
                guard let n = v as? NSNumber else { return false }
                if CFGetTypeID(n) == CFBooleanGetTypeID() { return false }
                return n.doubleValue.isFinite
                    && n.doubleValue >= 1
                    && n.doubleValue <= Double(maxTokens)
                    && (n.doubleValue.truncatingRemainder(dividingBy: 1)) == 0
            }
            // Set SO_RCVTIMEO/SO_SNDTIMEO; false on failure (the
            // connection is closed — bounded behavior).
            func setSocketTimeouts(_ fd: Int32, seconds: Double) -> Bool {
                guard seconds > 0 else { return false }
                var tv = timeval(tv_sec: time_t(seconds),
                                 tv_usec: suseconds_t((seconds.truncatingRemainder(dividingBy: 1)) * 1_000_000))
                // {0,0} would DISABLE the socket timeout — round up to
                // a nonzero 1 us so a near-deadline read cannot block.
                if tv.tv_sec == 0 && tv.tv_usec == 0 { tv.tv_usec = 1 }
                let rc1 = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv,
                                     socklen_t(MemoryLayout<timeval>.size))
                let rc2 = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv,
                                     socklen_t(MemoryLayout<timeval>.size))
                return rc1 == 0 && rc2 == 0
            }
            // One bounded recv with the socket timeout shrunk to the
            // REMAINING absolute deadline (a single read can never
            // overshoot it by a full timeout). -2 = timeout.
            func boundedRecv(_ fd: Int32, deadline: UInt64,
                             into buf: UnsafeMutablePointer<UInt8>, count: Int) -> Int {
                let now = DispatchTime.now().uptimeNanoseconds
                guard now < deadline else { return -2 }
                let remainingNs = deadline - now
                guard setSocketTimeouts(fd, seconds: Double(remainingNs) / 1e9)
                else { return -2 }
                let n = recv(fd, buf, count, 0)
                // The read itself must also finish inside the deadline;
                // a socket-level EAGAIN/EWOULDBLOCK is a timeout too.
                guard DispatchTime.now().uptimeNanoseconds <= deadline else { return -2 }
                if n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) { return -2 }
                return n
            }
            // Strict header parse. The recv may deliver header AND body
            // in one packet, so the body PREFIX read past the header
            // terminator is returned (never discarded). Returns nil
            // after an error response.
            func readHeaders(_ fd: Int32, deadline: UInt64)
                    -> (method: String, target: String, contentLength: Int?,
                        contentType: String?, bodyPrefix: Data)? {
                let cap = maxHeaderBytes + maxBodyBytes
                var raw = Data()
                var headerEnd = -1
                while raw.count < cap {
                    var chunk = [UInt8](repeating: 0, count: min(4096, cap - raw.count))
                    let n = boundedRecv(fd, deadline: deadline, into: &chunk,
                                        count: chunk.count)
                    if n == -2 {
                        sendHTTP(fd, "408 Request Timeout",
                             jsonError("absolute read deadline exceeded"))
                        return nil
                    }
                    guard n > 0 else { break }
                    raw.append(contentsOf: chunk[0..<n])
                    if let r = raw.range(of: Data("\r\n\r\n".utf8)) {
                        headerEnd = r.lowerBound
                        break
                    }
                }
                if headerEnd < 0 || headerEnd + 4 > maxHeaderBytes {
                    let status = headerEnd + 4 > maxHeaderBytes || raw.count >= cap
                        ? "431 Request Header Fields Too Large"
                        : "400 Bad Request"
                    sendHTTP(fd, status,
                             jsonError("headers exceed \(maxHeaderBytes) bytes or missing terminator"))
                    return nil
                }
                // Body prefix: bytes already read past the terminator
                // (curl commonly sends headers+body in one segment).
                let prefix = raw.count > headerEnd + 4
                    ? raw.subdata(in: (headerEnd + 4)..<raw.count) : Data()
                guard prefix.count <= maxBodyBytes else {
                    sendHTTP(fd, "413 Payload Too Large",
                         jsonError("body exceeds \(maxBodyBytes) bytes"))
                    return nil
                }
                // Header text EXCLUDES the empty terminator lines: the
                // split below sees only real header lines (no arbitrary
                // interior empty lines).
                guard let headerText = String(data: raw.subdata(in: 0..<headerEnd),
                                              encoding: .utf8) else {
                    sendHTTP(fd, "400 Bad Request", jsonError("headers not UTF-8"))
                    return nil
                }
                let lines = headerText.components(separatedBy: "\r\n")
                let requestLine = lines.first?.components(separatedBy: " ") ?? []
                guard requestLine.count == 3,
                      requestLine[2] == "HTTP/1.0" || requestLine[2] == "HTTP/1.1" else {
                    sendHTTP(fd, "400 Bad Request",
                         jsonError("malformed request line or protocol (need HTTP/1.0 or HTTP/1.1)"))
                    return nil
                }
                var contentLength: Int? = nil
                var sawContentLength = false
                var contentType: String? = nil
                for line in lines.dropFirst() {
                    let parts = line.split(separator: ":", maxSplits: 1)
                    guard parts.count == 2, !parts[0].isEmpty else {
                        sendHTTP(fd, "400 Bad Request", jsonError("malformed header line"))
                        return nil
                    }
                    let name = parts[0].trimmingCharacters(in: .whitespaces).lowercased()
                    let value = parts[1].trimmingCharacters(in: .whitespaces)
                    if name == "transfer-encoding" {
                        sendHTTP(fd, "400 Bad Request",
                             jsonError("transfer-encoding is not supported"))
                        return nil
                    }
                    if name == "content-length" {
                        if sawContentLength {
                            sendHTTP(fd, "400 Bad Request",
                                 jsonError("duplicate content-length"))
                            return nil
                        }
                        sawContentLength = true
                        guard let v = Int(value), v >= 0, v <= maxBodyBytes else {
                            sendHTTP(fd, "413 Payload Too Large",
                                 jsonError("content-length exceeds \(maxBodyBytes) bytes"))
                            return nil
                        }
                        contentLength = v
                    }
                    if name == "content-type" { contentType = value }
                }
                return (requestLine[0], requestLine[1], contentLength, contentType,
                        prefix)
            }
            func readBody(_ fd: Int32, deadline: UInt64, bodyLength: Int,
                          prefix: Data) -> Data? {
                var body = prefix
                if body.count > bodyLength {
                    sendHTTP(fd, "413 Payload Too Large",
                         jsonError("body exceeds content-length bound"))
                    return nil
                }
                var chunk = [UInt8](repeating: 0, count: min(4096, max(1, bodyLength)))
                while body.count < bodyLength {
                    let n = boundedRecv(fd, deadline: deadline, into: &chunk,
                                        count: min(chunk.count, bodyLength - body.count))
                    if n <= 0 {
                        sendHTTP(fd, "408 Request Timeout",
                             jsonError("body read timeout/truncation"))
                        return nil
                    }
                    body.append(contentsOf: chunk[0..<n])
                }
                return body
            }

            var handled = 0
            var firstRequest = true
            while handled < maxRequests {
                let clientFD = accept(serverFD, nil, nil)
                guard clientFD >= 0 else {
                    print("accept failed (errno \(errno)) — stopping")
                    break
                }
                defer { close(clientFD) }
                handled += 1
                // Initial generous timeouts (bounded by the absolute
                // deadline in every read anyway).
                guard setSocketTimeouts(clientFD, seconds: 10) else {
                    sendHTTP(clientFD, "500 Internal Server Error",
                         jsonError("socket timeout configuration failed"))
                    continue
                }
                let deadline = DispatchTime.now().uptimeNanoseconds &+ absoluteDeadlineNs
                guard let headers = readHeaders(clientFD, deadline: deadline) else { continue }
                // GET /health: no body required or allowed.
                if headers.method == "GET" {
                    guard headers.target == "/health" else {
                        sendHTTP(clientFD, "404 Not Found",
                             jsonError("unknown path \(headers.target)"))
                        continue
                    }
                    guard (headers.contentLength ?? 0) == 0,
                          headers.bodyPrefix.isEmpty else {
                        sendHTTP(clientFD, "400 Bad Request",
                             jsonError("/health takes no body"))
                        continue
                    }
                    sendHTTP(clientFD, "200 OK",
                         (try? JSONSerialization.data(withJSONObject: [
                             "status": "ok", "plan_digest": plan.planDigest
                         ])) ?? Data())
                    continue
                }
                guard headers.method == "POST" else {
                    sendHTTP(clientFD, "405 Method Not Allowed",
                         jsonError("only POST /generate and GET /health are supported"))
                    continue
                }
                guard headers.target == "/generate" else {
                    sendHTTP(clientFD, "404 Not Found",
                         jsonError("unknown path \(headers.target)"))
                    continue
                }
                // Strict base media type equality (parameters allowed).
                let mediaType = (headers.contentType ?? "")
                    .split(separator: ";").first?
                    .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
                guard mediaType == "application/json" else {
                    sendHTTP(clientFD, "415 Unsupported Media Type",
                         jsonError("content-type must be application/json"))
                    continue
                }
                guard let bodyLength = headers.contentLength else {
                    sendHTTP(clientFD, "411 Length Required",
                         jsonError("content-length required"))
                    continue
                }
                guard let body = readBody(clientFD, deadline: deadline,
                                          bodyLength: bodyLength,
                                          prefix: headers.bodyPrefix) else { continue }
                guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
                else {
                    sendHTTP(clientFD, "400 Bad Request",
                         jsonError("body is not a JSON object"))
                    continue
                }
                let allowed: Set<String> = ["text", "chat", "max_tokens"]
                guard Set(obj.keys).isSubset(of: allowed) else {
                    sendHTTP(clientFD, "400 Bad Request",
                         jsonError("unsupported body keys"))
                    continue
                }
                if let chat = obj["chat"], !isJSONBoolean(chat) {
                    sendHTTP(clientFD, "400 Bad Request",
                         jsonError("chat must be a JSON boolean"))
                    continue
                }
                var tokens = maxTokens
                if let tAny = obj["max_tokens"] {
                    guard isJSONInteger(tAny), let n = tAny as? NSNumber else {
                        sendHTTP(clientFD, "400 Bad Request",
                             jsonError("max_tokens must be a JSON integer 1...\(maxTokens)"))
                        continue
                    }
                    tokens = Int(n.int64Value)
                }
                guard let text = obj["text"] as? String,
                      !text.isEmpty, text.count <= maxPromptChars else {
                    sendHTTP(clientFD, "400 Bad Request",
                         jsonError("text must be a non-empty string of at most \(maxPromptChars) chars"))
                    continue
                }
                let chat = (obj["chat"] as? NSNumber)?.boolValue ?? false
                let handlerStart = DispatchTime.now()
                var gated = false
                do {
                    var reference: (ids: [Int], stepLogits: [[Float]],
                                    prefillLogits: [Float])? = nil
                    if gateFirstRequest, firstRequest {
                        let input = chat
                            ? "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                                + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                            : text
                        let promptIDs = try tokenizer.encode(input)
                        guard promptIDs.count + tokens <= 4096 else {
                            throw GemmaForgeError.executionFailed(
                                "prompt+output exceeds context")
                        }
                        reference = try referenceGeneration(promptIDs: promptIDs,
                                                            tokens: tokens)
                        gated = true
                    }
                    let result = try generate(promptText: text, chat: chat,
                                              tokens: tokens, reference: reference)
                    firstRequest = false
                    let handlerWall = Double(DispatchTime.now().uptimeNanoseconds
                                             - handlerStart.uptimeNanoseconds) / 1000
                    let resp = try JSONSerialization.data(withJSONObject: [
                        "text": result.text,
                        "token_ids": result.ids,
                        "output_tokens": result.ids.count,
                        "steps": result.ids.count - 1,
                        "plan_digest": plan.planDigest,
                        "gated": gated,
                        "median_wall_us_per_token": result.medianWall,
                        "prefill_wall_us": result.prefillWall,
                        "generation_wall_us": result.requestWall,
                        "handler_wall_us": handlerWall
                    ])
                    sendHTTP(clientFD, "200 OK", resp)
                    print("served request \(handled)/\(maxRequests): \(result.ids.count - 1) steps, ids \(result.ids)")
                } catch {
                    let handlerWall = Double(DispatchTime.now().uptimeNanoseconds
                                             - handlerStart.uptimeNanoseconds) / 1000
                    sendHTTP(clientFD, "500 Internal Server Error",
                         jsonError("generation failed or diverged (handler \(Int(handlerWall)) us): \(error)"))
                }
            }
            close(serverFD)
            print("request bound (\(maxRequests)) reached — server stopped")
            print("SERVE_DONE")
        } catch {
            fail("gemma-serve failed: \(error)")
        }

    case "gemma-profile":
        // Measures where per-token GPU time goes: one reference
        // (per-kernel) forward, then matvec GPU time aggregated per
        // tensor and per tensor FAMILY (attn/expert/shared/lm_head...).
        // Honest attribution: exact in per-kernel mode.
        let path = requireValue("--file", args)
        let promptArg = args.firstIndex(of: "--prompt-ids").map { _ in
            requireValue("--prompt-ids", args) } ?? "2,9259"
        do {
            let prompt = promptArg.split(separator: ",").map { Int($0) ?? -1 }
            let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: path))
            // Optional kernel flags to profile the SELECTED kernels (e.g.
            // --q4-tiled --rms-tiled) while keeping per-kernel dispatch so
            // attribution stays exact; batching stays off for that reason.
            if args.contains("--q4-tiled") { runner.q4TiledKernel = true }
            if args.contains("--q6-tiled") { runner.q6TiledKernel = true }
            if args.contains("--rms-tiled") { runner.rmsTiledKernel = true }
            if args.contains("--f32-tiled") { runner.f32TiledKernel = true }
            let flagText = "q4Tiled=\(runner.q4TiledKernel) "
                + "q6Tiled=\(runner.q6TiledKernel) "
                + "rmsTiled=\(runner.rmsTiledKernel) "
                + "f32Tiled=\(runner.f32TiledKernel)"
            guard prompt.allSatisfy({ $0 >= 0 && $0 < runner.config.vocabSize }) else {
                fail("bad prompt ids")
            }
            runner.reset()
            let evidence = try runner.forward(tokenIds: prompt)
            let total = evidence.gpuMicros
            print(String(format: "per-kernel forward (%@): %d tokens, total GPU %.1f ms, %d kernels",
                         flagText, prompt.count, total / 1000, evidence.dispatchedKernels))
            let profile = runner.matvecGpuProfile()
            let matvecSum = profile.reduce(0.0) { $0 + $1.gpuMicros }
            var byFamily = [String: (gpu: Double, calls: Int)]()
            for entry in profile {
                let comps = entry.tensor.split(separator: ".")
                let family = comps.count > 2 && comps[0] == "blk"
                    ? comps.dropFirst(2).joined(separator: ".") : entry.tensor
                byFamily[family, default: (0, 0)].gpu += entry.gpuMicros
                byFamily[family, default: (0, 0)].calls += entry.calls
            }
            print(String(format: "matvec total %.1f ms (%.0f%% of GPU), non-matvec %.1f ms",
                         matvecSum / 1000, 100 * matvecSum / max(total, 1),
                         (total - matvecSum) / 1000))
            print("top tensors by GPU time:")
            for entry in profile.prefix(12) {
                print(String(format: "  %-34@ %8.1f ms  %5.1f%%  %4d calls",
                             entry.tensor, entry.gpuMicros / 1000,
                             100 * entry.gpuMicros / max(matvecSum, 1), entry.calls))
            }
            print("by tensor family:")
            for (family, stat) in byFamily.sorted(by: { $0.value.gpu > $1.value.gpu }).prefix(14) {
                print(String(format: "  %-34@ %8.1f ms  %5.1f%%  %4d calls",
                             family, stat.gpu / 1000,
                             100 * stat.gpu / max(matvecSum, 1), stat.calls))
            }
            print("by kernel name (all dispatches, per-kernel mode):")
            for k in runner.kernelGpuProfile() {
                print(String(format: "  %-22@ %8.1f ms  %5.1f%%  %5d calls  %6.0f us/call",
                             k.kernel, k.gpuMicros / 1000,
                             100 * k.gpuMicros / max(total, 1), k.calls,
                             k.gpuMicros / Double(max(k.calls, 1))))
            }
            print("PROFILE_COMPLETED")
        } catch {
            fail("gemma-profile failed: \(error)")
        }

    default:
        usage()
}
