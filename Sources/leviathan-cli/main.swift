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
                  [--composite-plan <c.json> --moe-region <r.json>]
          Paired optional composite flags (both or neither): install the
          production composite Q4 launch-geometry + MoE lowering artifact
          (existing artifact required; invalid artifacts fail closed)
          before forge identity/selection. Requires --gen-tokens 32 and
          resident mode (no --expert-cache-budget).
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
      model-digest --file <model.gguf> [--cache-only] [--rehash-model]
          Shared model SHA-256 cache report (no GPU). A cache hit reads no
          model bytes; a miss hashes the file once; --cache-only never
          hashes; --rehash-model refreshes explicitly.
      --rehash-model (model commands, with --file or --gguf)
          Re-hash the model file once before the workflow; otherwise model
          digests are reused from the cache and never re-read automatically.
    """)
    exit(64)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("error: \(message)\n".data(using: .utf8)!)
    exit(1)
}

/// Resolves the model file's SHA-256 through the shared persisted digest
/// cache and prints exactly how it was obtained (cache hit with its
/// historical timestamp, initial computation, or explicit refresh). A hit
/// never reads the model bytes and is never presented as fresh.
func cachedModelDigest(_ url: URL) throws -> String {
    let lookup = try ModelDigestCache.shared.digest(for: url)
    print(lookup.disclosure)
    return lookup.sha256
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

// Explicit model digest refresh. `--rehash-model` re-hashes the model file
// named by --file or --gguf exactly once, before the requested workflow,
// and stores the result in the shared ModelDigestCache; every nested lookup
// in the workflow then reuses that refreshed record. Without it, model
// digests come from the cache (hashed only on an initial miss) and are
// never re-read automatically. The option is rejected for commands that do
// not use a model digest instead of being silently ignored, and help never
// hashes anything.
let modelDigestCommands: Set<String> = [
    "gemma-generate", "gemma-forge", "gemma-serve", "q4-lower", "moe-lower",
    "rms-lower", "gemma-q4gen-gate", "gemma-rmsgen-gate", "model-digest",
]
if args.contains("--rehash-model") {
    guard modelDigestCommands.contains(command) else {
        fail("--rehash-model is not supported by command '" + command + "'")
    }
    let fileFlag = args.contains("--file")
    let ggufFlag = args.contains("--gguf")
    guard fileFlag != ggufFlag else {
        fail("--rehash-model requires exactly one of --file or --gguf")
    }
    let modelPath = requireValue(fileFlag ? "--file" : "--gguf", args)
    do {
        let refreshed = try ModelDigestCache.shared.digest(
            for: URL(fileURLWithPath: modelPath), refresh: true)
        print(refreshed.disclosure)
    } catch {
        fail("--rehash-model failed: " + String(describing: error))
    }
}

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
        // --bos-align matches Ollama's warm KV-prefix policy as proven in
        // the 11498 debug daemon's cache-manager log (slot 0, prompt 2,
        // used 1, remaining 1: only BOS is reused, the non-BOS prompt
        // token is freshly evaluated). Each timed repetition prefills
        // [BOS] untimed, then times the one-token continuation and the
        // decode steps — both engines then freshly evaluate exactly one
        // prompt token per round. Requires --plan (SSA execution path)
        // and exactly 2 prompt ids.
        // --unprepared opts OUT of the default prepared hot path for the
        // qualified plan (feature toggle; prepared is the DEFAULT).
        let unpreparedOptOut = args.contains("--unprepared")
        guard !unpreparedOptOut || args.contains("--plan") else {
            fail("--unprepared requires --plan (it selects the plan schedule path)")
        }
        let bosAlign = args.contains("--bos-align")
        guard !bosAlign || args.contains("--plan") else {
            fail("--bos-align requires --plan (SSA execution path)")
        }
        // --breakdown prints a per-decode-step wall/GPU decomposition from
        // the runner's opt-in primitive counters (encode / flush-wait /
        // read-copy / write-copy; GPU busy time overlaps the flush wait).
        // The instrumentation adds two clock reads per primitive call; the
        // overhead control is the SAME workload run without --breakdown.
        let breakdown = args.contains("--breakdown")
        // Paired composite flags (both or neither), validated BEFORE any
        // GPU allocation. Composite here is inference / timed generation
        // ONLY: an existing composite artifact is required and selection is
        // NEVER allowed (the helper runs with allowSelection false).
        let compositeOpts = parseCompositeOptions(args)
        if let comp = compositeOpts {
            guard args.contains("--plan") else {
                fail("--composite-plan/--moe-region require --plan")
            }
            guard FileManager.default.fileExists(atPath: comp.compositePlanPath) else {
                fail("composite artifact not found: \(comp.compositePlanPath)")
            }
            guard !args.contains("--expert-cache-budget") else {
                fail("--composite-plan/--moe-region require resident weights and cannot "
                      + "be combined with --expert-cache-budget (streaming expert cache)")
            }
        }
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
            if bosAlign {
                guard promptIDs.count == 2 else {
                    fail("--bos-align requires a 2-token prompt (BOS + one token)")
                }
            }
            print("prompt ids: \(promptIDs.map(String.init).joined(separator: " "))")

            let loadStart = ContinuousClock.now
            let runner = try Gemma4Runner(fileAt: fileURL)
            func ms(_ t: ContinuousClock.Duration) -> Double {
                Double(t.components.seconds) * 1e3 + Double(t.components.attoseconds) / 1e15
            }
            let loadMs = ms(ContinuousClock.now - loadStart)
            if breakdown { runner.breakdownInstrumentation = true }

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
            var preparedExec: GemmaPreparedExecution? = nil
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
                // Composite state binding: after the standard graph/plan
                // reads and the cached model digest lookup, BEFORE the engine digest and
                // plan-identity validation. allowSelection false — this is
                // inference / timed generation, never selection. One-shot
                // preparatory work only (no per-token IO or hashing).
                let ggufDigest = try cachedModelDigest(fileURL)
                if let comp = compositeOpts {
                    let regionJSON = try readMoERegionBounded(comp.moeRegionPath)
                    _ = try MoERegionIR.parse(regionJSON)
                    let result = try GemmaCompositeForge.prepare(
                        regionJSON: regionJSON, ggufURL: fileURL, graph: graphOpt!,
                        runner: runner, ggufDigest: ggufDigest,
                        artifactURL: URL(fileURLWithPath: comp.compositePlanPath),
                        fixtures: try canonicalCompositeFixtures(tokenizer: tokenizer),
                        isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                        allowSelection: false)
                    for line in result.log { print(line) }
                    print("composite: enabled, installed \(result.installed), "
                          + "cache hit \(result.cacheHit), artifact digest "
                          + result.artifactDigest.prefix(16) + "…")
                }
                let engine = GemmaForge.engineDigest(try runner.engineIdentity())
                planOpt = try GemmaForge.loadPlan(
                    from: URL(fileURLWithPath: planPath))
                try GemmaForge.validatePlanIdentity(
                    planOpt!, graph: graphOpt!, ggufDigest: ggufDigest,
                    engineDigest: engine,
                    hardware: runner.hardwareProfile(),
                    expertCacheBudget: nil)
                print("plan identity validated: \(planOpt!.planDigest) "
                      + "(selected \(planOpt!.selected))")
                // DEFAULT PREPARED hot path: one-time prepare (validate +
                // lower + value table) after identity validation, with the
                // prepared SSA asserted EXACTLY equal to the plan's embedded
                // schedule — a mismatch is a hard rejection. --unprepared
                // opts out to the persisted-schedule path.
                if unpreparedOptOut {
                    preparedExec = nil
                    print("prepared execution: OFF (--unprepared opt-out; plan schedule path)")
                } else {
                    let prep = try GemmaPreparedExecution.prepare(
                        graph: graphOpt!, runner: runner)
                    guard prep.ssa == planOpt!.schedule else {
                        fail("prepared SSA schedule differs from the plan's embedded schedule — rejecting")
                    }
                    preparedExec = prep
                    print("prepared execution: ON (prepared.ssa asserted equal to plan.schedule)")
                }
                print("selected candidate flags: batching=\(planOpt!.selected.segmentBatching ? 1 : 0) tiledQ4=\(planOpt!.selected.matvecTiledQ4 ? 1 : 0) tiledQ6=\(planOpt!.selected.matvecTiledQ6 ? 1 : 0) rmsTiled=\(planOpt!.selected.rmsnormTiled ? 1 : 0) f32Tiled=\(planOpt!.selected.matvecTiledF32 ? 1 : 0) vecLoads=\(planOpt!.selected.vectorizedLoads ? 1 : 0) gpuCacheAppend=\(planOpt!.selected.gpuCacheAppend ? 1 : 0) parallelAttention=\(planOpt!.selected.parallelAttention ? 1 : 0) rmsCoop=\(planOpt!.selected.rmsnormCooperative ? 1 : 0) q6Coop=\(planOpt!.selected.matvecCooperativeQ6 ? 1 : 0) rmsStats=\(planOpt!.selected.rmsStatisticsReuse ? 1 : 0) routerGPU=\(planOpt!.selected.routerGPUResidency ? 1 : 0) routerDiv=\(planOpt!.selected.routerLaunchDivisor)")
                print(String(format: "identity validation prep: %.1f ms (graph parse + cached model digest lookup + plan load)",
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
                let (am, lg, _, _, _) = try (preparedExec != nil
                    ? GemmaForge.executeGraph(graph, runner: runner, tokenIds: promptIDs,
                                              candidate: plan.selected, prepared: preparedExec)
                    : GemmaForge.executeGraph(graph, runner: runner, tokenIds: promptIDs,
                                              candidate: plan.selected, schedule: plan.schedule))
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
                    let (next, sl, _, _, _) = try (preparedExec != nil
                        ? GemmaForge.executeGraphContinue(graph, runner: runner, token: gateGen.last!,
                                                          candidate: plan.selected, prepared: preparedExec)
                        : GemmaForge.executeGraphContinue(graph, runner: runner, token: gateGen.last!,
                                                          candidate: plan.selected, schedule: plan.schedule))
                    let want = stepLogits[step - 1]
                    // Explicit shape/finiteness checks first, then the
                    // full-array worst relative difference (no
                    // short-circuit: the reported worst covers every
                    // logit, and a shape mismatch is labeled as such
                    // instead of reporting 0.0).
                    let shapeOk = next == refIds[step] && sl.count == want.count
                    let finiteOk = sl.allSatisfy({ $0.isFinite })
                        && want.allSatisfy({ $0.isFinite })
                    var worstRel = 0.0
                    if sl.count == want.count {
                        for (a, b) in zip(sl, want) {
                            let rel = abs(Double(a - b))
                                / max(1.0, abs(Double(b)))
                            if rel > worstRel { worstRel = rel }
                        }
                    }
                    guard shapeOk, finiteOk, worstRel < 1e-4 else {
                        let shapeNote = shapeOk ? "" : " (shape/token mismatch)"
                        fail(String(format: "plan gate: step %d outside 1e-4 of reference (worst %.3e%@)",
                                    step, worstRel, shapeNote))
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
                var ttftMs = 0.0
                if let plan = planOpt, let graph = graphOpt {
                    if bosAlign {
                        // Untimed BOS-only prefill (the prefix Ollama
                        // keeps warm per the 11498 cache-manager log);
                        // the timed window starts right before the
                        // freshly evaluated non-BOS prompt token.
                        _ = try (preparedExec != nil
                            ? GemmaForge.executeGraph(graph, runner: runner, tokenIds: [promptIDs[0]],
                                                      candidate: plan.selected, prepared: preparedExec)
                            : GemmaForge.executeGraph(graph, runner: runner, tokenIds: [promptIDs[0]],
                                                      candidate: plan.selected, schedule: plan.schedule))
                        let timedStart = ContinuousClock.now
                        let (am, _, gpuMicros, _, _) =
                            try (preparedExec != nil
                                ? GemmaForge.executeGraphContinue(graph, runner: runner, token: promptIDs[1],
                                                                  candidate: plan.selected, prepared: preparedExec)
                                : GemmaForge.executeGraphContinue(graph, runner: runner, token: promptIDs[1],
                                                                  candidate: plan.selected, schedule: plan.schedule))
                        firstToken = am
                        gpuMicrosTotal = gpuMicros
                        ttftMs = ms(ContinuousClock.now - timedStart)
                    } else {
                        let (am, _, gpuMicros, _, _) = try (preparedExec != nil
                            ? GemmaForge.executeGraph(graph, runner: runner, tokenIds: promptIDs,
                                                      candidate: plan.selected, prepared: preparedExec)
                            : GemmaForge.executeGraph(graph, runner: runner, tokenIds: promptIDs,
                                                      candidate: plan.selected, schedule: plan.schedule))
                        firstToken = am
                        gpuMicrosTotal = gpuMicros
                        ttftMs = ms(ContinuousClock.now - prefillStart)
                    }
                    if memorySample == 0 {
                        memorySample = runner.metalAllocatedSample()
                    }
                } else {
                    let evidence = try runner.forward(tokenIds: promptIDs)
                    firstToken = evidence.argmaxToken
                    gpuMicrosTotal = evidence.gpuMicros
                    memorySample = evidence.metalAllocatedBytes
                    ttftMs = ms(ContinuousClock.now - prefillStart)
                }

                var generated = [firstToken]
                var decodeMs = [Double]()
                var bdTotals = (wall: 0.0, gpu: 0.0, flush: 0.0, encode: 0.0,
                                readCopy: 0.0, writeCopy: 0.0, residual: 0.0)
                while generated.count < maxTokens {
                    if tokenizer.isEndOfGeneration(generated.last!) {
                        print("eos: token \(generated.last!)")
                        break
                    }
                    if breakdown { runner.breakdownReset() }
                    let stepStart = ContinuousClock.now
                    if let plan = planOpt, let graph = graphOpt {
                        let (next, _, gpuMicros, _, _) =
                            try (preparedExec != nil
                                ? GemmaForge.executeGraphContinue(graph, runner: runner, token: generated.last!,
                                                                  candidate: plan.selected, prepared: preparedExec)
                                : GemmaForge.executeGraphContinue(graph, runner: runner, token: generated.last!,
                                                                  candidate: plan.selected, schedule: plan.schedule))
                        gpuMicrosTotal += gpuMicros
                        generated.append(next)
                        if breakdown {
                            let wall = ms(ContinuousClock.now - stepStart)
                            let c = runner.breakdownCounters()
                            let residual = wall - c.flush / 1000 - c.encode / 1000
                                - c.readCopy / 1000 - c.writeCopy / 1000
                            bdTotals.wall += wall
                            bdTotals.gpu += gpuMicros / 1000
                            bdTotals.flush += c.flush / 1000
                            bdTotals.encode += c.encode / 1000
                            bdTotals.readCopy += c.readCopy / 1000
                            bdTotals.writeCopy += c.writeCopy / 1000
                            bdTotals.residual += residual
                            let hostFlush = max(0.0, c.flush / 1000 - gpuMicros / 1000)
                            print(String(format: "breakdown step %d: wall %.2f ms | gpu %.2f | flushWait %.2f (host share %.2f) | encode %.2f | readCopy %.2f | writeCopy %.2f | residual %.2f",
                                         generated.count - 1, wall, gpuMicros / 1000,
                                         c.flush / 1000, hostFlush, c.encode / 1000,
                                         c.readCopy / 1000, c.writeCopy / 1000,
                                         residual))
                        }
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
                if breakdown, bdTotals.wall > 0 {
                    let flushHost = max(0.0, bdTotals.flush - bdTotals.gpu)
                    func pct(_ v: Double) -> Double { 100 * v / bdTotals.wall }
                    print(String(format: "breakdown totals (%d steps, decode only): wall %.1f ms | gpu %.1f (%.1f%%) | flush-host %.1f (%.1f%%) | encode %.1f (%.1f%%) | readCopy %.1f (%.1f%%) | writeCopy %.1f (%.1f%%) | residual %.1f (%.1f%%)",
                                 decodeMs.count, bdTotals.wall, bdTotals.gpu, pct(bdTotals.gpu),
                                 flushHost, pct(flushHost), bdTotals.encode, pct(bdTotals.encode),
                                 bdTotals.readCopy, pct(bdTotals.readCopy),
                                 bdTotals.writeCopy, pct(bdTotals.writeCopy),
                                 bdTotals.residual, pct(bdTotals.residual)))
                }
            }
            print("parallel attention totals: applied \(runner.parallelAttentionAppliedCount), fallback \(runner.parallelAttentionFallbackCount)")
            print("rms cooperative totals: applied \(runner.rmsCooperativeAppliedCount), fallback \(runner.rmsCooperativeFallbackCount)")
            print("q6 cooperative totals: applied \(runner.q6CooperativeAppliedCount), fallback \(runner.q6CooperativeFallbackCount)")
            print("rms statistics totals: computed \(runner.rmsStatsComputedCount), reused \(runner.rmsStatsReusedCount), fallback \(runner.rmsStatsFallbackCount)")
            print("router residency totals: scale applied \(runner.routerScaleApplied), scale fallback \(runner.routerScaleFallback), zero applied \(runner.residentZeroApplied), zero fallback \(runner.residentZeroFallback), launch applied \(runner.routerLaunchApplied), launch fallback \(runner.routerLaunchFallback)")
            if compositeOpts != nil {
                // Actual-run evidence that the composite lowering executed:
                // batched MoE stage dispatches (GU/ACT/DN), ordered axpy
                // merges, resident fallbacks, and the Q4 launch-width
                // applications — counters read from the public runner state.
                print("composite moe counters: GU \(runner.moeGuDispatchCount), ACT \(runner.moeActDispatchCount), DN \(runner.moeDnDispatchCount), merges \(runner.moeAxpyMergeCount), resident fallback \(runner.moeResidentFallbackCount)")
                print("composite q4 launch width: applied \(runner.q4LaunchAppliedCount), fallback \(runner.q4LaunchFallbackCount)")
            }
            print(String(format: "load (once): %.1f ms (runtime Metal library + pipeline compile within it: %.1f ms)",
                         loadMs, runner.metalCompileMicros / 1000))
            print("GENERATION_COMPLETED")
        } catch {
            fail("gemma-generate failed: \(error)")
        }

    case "gemma-diagnose":
        // Bounded opt-in teacher-forced layerwise numerical diagnosis.
        // Runs the plain runner reference path ONCE to generate a greedy
        // history, then REPLAYS the EXACT SAME token history (teacher
        // forcing: no autoregressive feedback) under each kernel/
        // scheduling axis separately — the same flags the forge executor
        // sets per candidate — comparing per-layer residual checkpoints
        // and full logits against the reference. Also reports gate-
        // criterion context (top1-top2 margin, softmax KL, L2) so a tiny
        // logit difference can be judged against the argmax margin, and
        // runs an independent FP64 CPU oracle over the attention-Q matvec
        // at the worst step/layer to attribute the error (which kernel is
        // closer to exact math). Diagnostic ONLY: no gate is loosened,
        // nothing here feeds candidate selection.
        let path = requireValue("--file", args)
        let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
            Int(requireValue("--gen-tokens", args)) ?? 0 } ?? 32
        guard genTokens >= 2, genTokens <= 64 else {
            fail("--gen-tokens must be 2...64")
        }
        let rowProbe = args.firstIndex(of: "--row-probe").map { _ in
            Int(requireValue("--row-probe", args)) ?? 0 } ?? 128
        guard rowProbe >= 16, rowProbe <= 1024 else {
            fail("--row-probe must be 16...1024")
        }
        do {
            let fileURL = URL(fileURLWithPath: path)
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
            var history: [Int]
            if let tokensArg = args.firstIndex(of: "--tokens").map({ _ in
                requireValue("--tokens", args) }) {
                history = tokensArg.split(separator: ",").map {
                    guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                          id >= 0, id < tokenizer.vocabSize else {
                        fail("invalid token id '\($0)'")
                    }
                    return id
                }
            } else {
                let text = requireValue("--text", args)
                history = try tokenizer.encode(text)
            }
            guard !history.isEmpty, history.count + genTokens <= 4096 else {
                fail("prompt + gen-tokens must fit the 4096 context")
            }
            print("prompt ids: \(history.map(String.init).joined(separator: " "))")
            let runner = try Gemma4Runner(fileAt: fileURL)

            // ---- reference pass A (probe ON): greedy history + snapshots.
            // Snapshots keyed (step, layer, stage); steps index the
            // teacher-forced forward of history[i] (step 0 = first prompt
            // token's forward is folded: forward(history) then logits at
            // each position). Step s logits predict generated[s].
            var refSnap = [String: [Float]]()
            var refIds = [Int]()
            var refLogitsA = [[Float]]()
            runner.layerProbe = { stage, layer, position, x in
                refSnap["\(position):\(layer):\(stage)"] = x
            }
            runner.reset()
            do {
                let prefill = try runner.forward(tokenIds: history)
                refLogitsA.append(try runner.currentLogits())
                refIds.append(prefill.argmaxToken)
                var prev = prefill.argmaxToken
                while refIds.count < genTokens {
                    if tokenizer.isEndOfGeneration(prev) { break }
                    let ev = try runner.forward(tokenIds: [prev])
                    refLogitsA.append(try runner.currentLogits())
                    refIds.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
            } catch { fail("reference pass A failed: \(error)") }
            runner.layerProbe = nil
            let steps = refLogitsA.count
            print("reference greedy ids (\(steps)): \(refIds.map(String.init).joined(separator: " "))")

            // Teacher-forced replay helper: feeds prompt then the ref ids
            // (all but the last — the last id's own forward is not needed),
            // ONE token per forward so probe positions align exactly with
            // the reference pass (step s logits come from position
            // history.count - 1 + s in both).
            // The diagnostic axis currently in force — updated at EVERY
            // applyDiagnosticAxis site so teacherForced can restore it
            // AFTER reset (reset() -> discardOpenSegment() clears
            // autoSegment; without the restore a segmentBatching axis
            // would silently replay per-kernel submissions).
            var currentAxis = (q4: false, q6: false, rms: false,
                               f32: false, vl: false, seg: false)
            func applyAxis(_ a: (q4: Bool, q6: Bool, rms: Bool,
                                 f32: Bool, vl: Bool, seg: Bool)) {
                currentAxis = a
                runner.applyDiagnosticAxis(q4Tiled: a.q4, q6Tiled: a.q6,
                                           rmsTiled: a.rms, f32Tiled: a.f32,
                                           vectorizedLoads: a.vl,
                                           segmentBatching: a.seg)
            }
            func teacherForced(_ afterStep: (Int, [Float]) throws -> Void) throws {
                runner.reset()
                applyAxis(currentAxis)
                guard currentAxis.seg == runner.segmentBatchingActive else {
                    fail("segment batching axis lost across reset")
                }
                // reset() -> discardOpenSegment() clears autoSegment, so
                // the CURRENT diagnostic axis is re-applied AFTER every
                // reset — otherwise a segmentBatching axis would silently
                // replay per-kernel submissions (numerically identical,
                // but not the intended measurement mode).
                runner.applyDiagnosticAxis(q4Tiled: currentAxis.q4, q6Tiled: currentAxis.q6,
                                           rmsTiled: currentAxis.rms, f32Tiled: currentAxis.f32,
                                           vectorizedLoads: currentAxis.vl,
                                           segmentBatching: currentAxis.seg)
                guard currentAxis.seg == runner.segmentBatchingActive else {
                    fail("segment batching axis lost across reset")
                }
                for t in history {
                    try runner.forward(tokenIds: [t])
                }
                try afterStep(0, try runner.currentLogits())
                for s in 1..<steps {
                    try runner.forward(tokenIds: [refIds[s]])
                    try afterStep(s, try runner.currentLogits())
                }
            }

            // ---- reference pass B (probe OFF): pure per-kernel logits.
            // Fail closed: the probe's readbacks only flush deferred
            // segments (same kernels, same order), so the reference must
            // be BIT-IDENTICAL between the probe-on and probe-off passes.
            var refLogitsB = [[Float]]()
            try teacherForced { s, lg in
                guard refLogitsA[s].count == lg.count,
                      zip(refLogitsA[s], lg).allSatisfy({
                          $0.bitPattern == $1.bitPattern }) else {
                    fail("reference probe-on/off logits differ at step \(s) — probe changed numerics; diagnosis invalid")
                }
                refLogitsB.append(lg)
            }

            // Logit metrics of candidate logits `lg` vs reference `ref` at
            // one step: gate-formula worst rel, argmax agreement, reference
            // top1-top2 margin, softmax KL (Double, full vocab), relative L2.
            // Fail-closed: shape mismatch or any non-finite value is an
            // error, never a silently truncated zip or a NaN-driven zero.
            func logitMetrics(_ lg: [Float], _ ref: [Float])
                    -> (worstRel: Double, argmaxEq: Bool, margin: Float,
                        kl: Double, l2rel: Double) {
                guard lg.count == ref.count, !lg.isEmpty,
                      lg.allSatisfy({ $0.isFinite }),
                      ref.allSatisfy({ $0.isFinite }) else {
                    fail("logit metrics: shape/nonfinite violation "
                         + "(\(lg.count) vs \(ref.count))")
                }
                var worst = 0.0
                var l2 = 0.0
                var refN = 0.0
                for (a, b) in zip(lg, ref) {
                    let rel = abs(Double(a - b)) / max(1.0, abs(Double(b)))
                    if rel > worst { worst = rel }
                    let dd = Double(a - b)
                    l2 += dd * dd
                    refN += Double(b) * Double(b)
                }
                let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                let refArg = ref.enumerated().max { $0.element < $1.element }!.offset
                let sorted = ref.sorted(by: >)
                let maxRef = Double(sorted[0])
                let mExp = ref.map { exp(Double($0) - maxRef) }
                let mSum = mExp.reduce(0, +)
                let cExp = lg.map { exp(Double($0) - maxRef) }
                let cSum = cExp.reduce(0, +)
                var kl = 0.0
                for (p, q) in zip(mExp, cExp) {
                    let pi = p / mSum, qi = q / cSum
                    if pi > 0 { kl += pi * log(pi / max(qi, 1e-300)) }
                }
                return (worst, argmax == refArg, sorted[0] - sorted[1],
                        kl, sqrt(l2 / max(refN, 1e-300)))
            }

            struct AxisConfig {
                let name: String
                let q4: Bool, q6: Bool, rms: Bool, f32: Bool, vl: Bool, seg: Bool
            }
            let axes = [
                AxisConfig(name: "segment_only", q4: false, q6: false, rms: false,
                           f32: false, vl: false, seg: true),
                AxisConfig(name: "q4_tiled", q4: true, q6: false, rms: false,
                           f32: false, vl: false, seg: false),
                AxisConfig(name: "q6_tiled", q4: false, q6: true, rms: false,
                           f32: false, vl: false, seg: false),
                AxisConfig(name: "rms_tiled", q4: false, q6: false, rms: true,
                           f32: false, vl: false, seg: false),
                AxisConfig(name: "f32_tiled", q4: false, q6: false, rms: false,
                           f32: true, vl: false, seg: false),
                AxisConfig(name: "tiled_all+segment", q4: true, q6: true, rms: true,
                           f32: true, vl: false, seg: true),
                AxisConfig(name: "vl+segment", q4: false, q6: false, rms: false,
                           f32: false, vl: true, seg: true),
            ]

            // (axis, position) layerwise results: per-position worst rel at
            // each checkpoint + the layer attaining the post_attn/post_layer
            // max, and whether ANY snapshot differed at bit level (rel 0 is
            // numerical equality, not proof of bit identity).
            struct StepResult {
                var postAttn = 0.0; var postAttnLayer = -1
                var postLayer = 0.0; var postLayerLayer = -1
                var attnNormH = 0.0
                var anyBitsDiffer = false
                // Per-layer rel at each checkpoint (index = layer) for the
                // first-nonzero / propagation table at the worst position.
                var perLayerAttn = [Double](repeating: 0, count: 30)
                var perLayerPost = [Double](repeating: 0, count: 30)
            }
            for axis in axes {
                applyAxis((q4: axis.q4, q6: axis.q6, rms: axis.rms,
                           f32: axis.f32, vl: axis.vl, seg: axis.seg))
                // Pass A: probe ON, layerwise comparison against refSnap.
                // Indexed by teacher-forced POSITION (0 = first prompt
                // token); the teacher-forced history covers exactly
                // positions 0..<history.count + steps - 1.
                let positionCount = history.count + steps - 1
                var layerResults = [StepResult](repeating: StepResult(), count: positionCount)
                runner.layerProbe = { stage, layer, position, x in
                    guard position >= 0, position < positionCount else {
                        fail("probe position \(position) outside teacher-forced history")
                    }
                    guard let want = refSnap["\(position):\(layer):\(stage)"],
                          want.count == x.count,
                          x.allSatisfy({ $0.isFinite }),
                          want.allSatisfy({ $0.isFinite }) else {
                        fail("probe key/shape/finite mismatch: position \(position) layer \(layer) \(stage)")
                    }
                    var worst = 0.0
                    var bitsDiffer = false
                    for (a, b) in zip(x, want) {
                        if a.bitPattern != b.bitPattern { bitsDiffer = true }
                        let rel = abs(Double(a - b)) / max(1.0, abs(Double(b)))
                        if rel > worst { worst = rel }
                    }
                    var r = layerResults[position]
                    r.anyBitsDiffer = r.anyBitsDiffer || bitsDiffer
                    switch stage {
                    case "post_attn":
                        r.perLayerAttn[layer] = worst
                        if worst > r.postAttn { r.postAttn = worst; r.postAttnLayer = layer }
                    case "post_layer":
                        r.perLayerPost[layer] = worst
                        if worst > r.postLayer { r.postLayer = worst; r.postLayerLayer = layer }
                    case "attn_norm_h": r.attnNormH = max(r.attnNormH, worst)
                    default: break
                    }
                    layerResults[position] = r
                }
                // RUN pass A (probe ON): teacher-forced replay with the
                // layerwise capture comparing against refSnap as it fires;
                // the full logits of THIS pass are compared bitwise with
                // the probe-OFF pass below (probe-invariance per axis).
                var passALogits = [[Float]]()
                try teacherForced { _, lg in passALogits.append(lg) }
                runner.layerProbe = nil
                // Pass B: probe OFF, pure full-batching logits metrics
                // (gate-formula worst rel, argmax, margin, KL, L2, and
                // bit equality against the reference logits).
                var pure = [(worstRel: Double, argmaxEq: Bool, margin: Float,
                             kl: Double, l2rel: Double, bitsEq: Bool)]()
                var stepWorst = [Double](repeating: 0, count: steps)
                var passBLogits = [[Float]]()
                try teacherForced { s, lg in
                    let m = logitMetrics(lg, refLogitsB[s])
                    let bitsEq = zip(lg, refLogitsB[s]).allSatisfy {
                        $0.bitPattern == $1.bitPattern }
                    pure.append((m.worstRel, m.argmaxEq, m.margin, m.kl, m.l2rel, bitsEq))
                    stepWorst[s] = m.worstRel
                    passBLogits.append(lg)
                }
                applyAxis((q4: false, q6: false, rms: false, f32: false,
                           vl: false, seg: false))

                // Probe-invariance for THIS axis: the probe-on pass logits
                // must be bit-identical to the probe-off pass logits (the
                // probe only adds segment flushes; same kernels, order).
                guard passALogits.count == passBLogits.count,
                      zip(passALogits, passBLogits).allSatisfy({
                          $0.count == $1.count && zip($0, $1).allSatisfy {
                              $0.bitPattern == $1.bitPattern } }) else {
                    fail("axis \(axis.name): probe-on/off logits differ bitwise — probe changed numerics under this axis; diagnosis invalid")
                }

                // Teacher-forced POSITION p maps to output step
                // p - (history.count - 1); report both.
                let firstOut = history.count - 1
                let argmaxMismatches = pure.filter { !$0.argmaxEq }.count
                let maxLayerStep = layerResults.enumerated().max {
                    $0.element.postLayer < $1.element.postLayer }
                let maxPureStep = stepWorst.enumerated().max { $0.element < $1.element }
                print("axis \(axis.name): max post_layer rel \(String(format: "%.3e", maxLayerStep?.element.postLayer ?? 0)) at position \(maxLayerStep?.offset ?? -1) (output step \((maxLayerStep?.offset ?? -1) - firstOut)) layer \(maxLayerStep?.element.postLayerLayer ?? -1); max post_attn rel \(String(format: "%.3e", layerResults.map(\.postAttn).max() ?? 0)); pure logits worst rel \(String(format: "%.3e", maxPureStep?.element ?? 0)) at output step \(maxPureStep?.offset ?? -1); argmax mismatch steps \(argmaxMismatches)/\(steps)")
                if axis.name == "segment_only" {
                    let allBits = pure.allSatisfy(\.bitsEq)
                    let snapBits = !layerResults.contains(where: \.anyBitsDiffer)
                    print("axis segment_only: logits BIT-IDENTICAL to reference: \(allBits); layer snapshots bit-identical: \(snapBits)")
                }
                // Gate-criterion context for the worst pure step.
                if let ws = maxPureStep, ws.element > 0 {
                    let m = pure[ws.offset]
                    print(String(format: "  step %d: worstRel %.3e | argmax match %@ | ref margin(top1-top2) %.4f | KL %.3e | relL2 %.3e",
                                 ws.offset, m.worstRel,
                                 m.argmaxEq ? "yes" : "NO", m.margin, m.kl, m.l2rel))
                }
                if axis.name == "tiled_all+segment" || axis.name == "q4_tiled" {
                    // Per-step pure worst rel growth table (compact).
                    let line = stepWorst.enumerated().map {
                        String(format: "%d:%.1e", $0.offset, $0.element)
                    }.joined(separator: " ")
                    print("  per-step pure worstRel: \(line)")
                    // Per-layer propagation table at the worst layerwise
                    // position: first nonzero layer and the rel trajectory.
                    if let worstPos = maxLayerStep {
                        let attnLine = worstPos.element.perLayerAttn.map {
                            String(format: "%.1e", $0) }.joined(separator: " ")
                        let postLine = worstPos.element.perLayerPost.map {
                            String(format: "%.1e", $0) }.joined(separator: " ")
                        let firstNonzero = worstPos.element.perLayerPost.firstIndex {
                            $0 > 0 } ?? -1
                        print("  per-layer post_attn rel at position \(worstPos.offset): \(attnLine)")
                        print("  per-layer post_layer rel at position \(worstPos.offset): \(postLine)")
                        print("  first nonzero post_layer layer: \(firstNonzero)")
                    }
                }
            }

            // ---- SSA graph executor linkage (optional --graph): teacher-
            // force the ACTUAL failing forge candidate (and the selected
            // vectorized-loads one) through executeGraphContinue and
            // compare full logits to the runner reference — connecting the
            // layerwise axis diagnosis to the real 32-token gate failure.
            if let graphArg = args.firstIndex(of: "--graph").map({ _ in
                requireValue("--graph", args) }) {
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphArg))
                var graphData = Data()
                while graphData.count <= 8 << 20 {
                    let chunk = try graphHandle.read(upToCount: 1 << 16) ?? Data()
                    if chunk.isEmpty { break }
                    graphData.append(chunk)
                }
                try? graphHandle.close()
                guard graphData.count <= 8 << 20 else { fail("graph file > 8 MiB") }
                let graph = try GemmaGraph.parse(graphData)
                let graphCandidates: [(String, GemmaForgeCandidate)] = [
                    // The EXACT three pass-7 32-token-gate rejections from
                    // the persisted plan record, then the selected VL
                    // candidate; "tiled_all" (with Q6/F32 too) is an extra
                    // extreme diagnostic, labeled as such.
                    ("ssa batch+q4+rms (rejected step25 1.171e-04)",
                     GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                         matvecTiledQ4: true, rmsnormTiled: true)),
                    ("ssa batch+q4 (rejected step28 1.032e-04)",
                     GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                         matvecTiledQ4: true)),
                    ("ssa batch+q4+q6 (rejected step28 1.041e-04)",
                     GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                         matvecTiledQ4: true, matvecTiledQ6: true)),
                    ("ssa batch+rmsVL (selected)",
                     GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                         vectorizedLoads: true)),
                    ("ssa tiled_all+segment (EXTREME extra, not a pass-7 candidate)",
                     GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                         matvecTiledQ4: true, matvecTiledQ6: true,
                                         rmsnormTiled: true, matvecTiledF32: true)),
                ]
                for (name, cand) in graphCandidates {
                    runner.reset()
                    let (_, lg0, _, _, _) = try GemmaForge.executeGraph(
                        graph, runner: runner, tokenIds: history, candidate: cand)
                    var worstAll = logitMetrics(lg0, refLogitsB[0]).worstRel
                    var worstStep = 0
                    var argmaxMismatchSteps = 0
                    if !logitMetrics(lg0, refLogitsB[0]).argmaxEq { argmaxMismatchSteps += 1 }
                    for s in 1..<steps {
                        let (_, lg, _, _, _) = try GemmaForge.executeGraphContinue(
                            graph, runner: runner, token: refIds[s - 1], candidate: cand)
                        let m = logitMetrics(lg, refLogitsB[s])
                        if m.worstRel > worstAll { worstAll = m.worstRel; worstStep = s }
                        if !m.argmaxEq { argmaxMismatchSteps += 1 }
                    }
                    print(String(format: "axis %@: logits worst rel %.3e at output step %d | argmax mismatch steps %d/%d",
                                 name, worstAll, worstStep, argmaxMismatchSteps, steps))
                }
            }

            // ---- independent FP64 attribution at the attention-Q matvec:
            // at the reference's attn_norm_h input for the layer with the
            // largest post_attn error in a chosen step, evaluate plain and
            // tiled kernels and an FP64 CPU oracle over the first rowProbe
            // rows; whichever kernel is closer to FP64 is the numerically
            // better reduction for this invocation.
            do {
                // Re-run the q4_tiled axis probe ON to find the worst
                // (step, layer) post_attn checkpoint (cheap: reuse logic).
                applyAxis((q4: true, q6: false, rms: false, f32: false,
                           vl: false, seg: false))
                var best = (rel: 0.0, step: 0, layer: 0)
                runner.layerProbe = { stage, layer, position, x in
                    guard stage == "post_attn",
                          let want = refSnap["\(position):\(layer):post_attn"] else { return }
                    var worst = 0.0
                    for (a, b) in zip(x, want) {
                        let rel = abs(Double(a - b)) / max(1.0, abs(Double(b)))
                        if rel > worst { worst = rel }
                    }
                    if worst > best.rel { best = (worst, position, layer) }
                }
                try teacherForced { _, _ in }
                runner.layerProbe = nil
                applyAxis((q4: false, q6: false, rms: false, f32: false,
                           vl: false, seg: false))
                let (stepL, layerL) = (best.step, best.layer)
                guard let xRef = refSnap["\(stepL):\(layerL):attn_norm_h"] else {
                    fail("missing reference attn_norm_h for attribution")
                }
                let tensor = "blk.\(layerL).attn_q.weight"
                let plain = try runner.diagnosticMatvec(tensor: tensor, x: xRef,
                                                        rows: rowProbe, tiled: false)
                let tiled = try runner.diagnosticMatvec(tensor: tensor, x: xRef,
                                                        rows: rowProbe, tiled: true)
                // FP64 oracle over the same rows, decoded from GGUF bytes.
                let info = try runner.ggufFile.tensor(tensor)
                let data = try runner.ggufFile.tensorData(info)
                let inDim = runner.config.hiddenSize
                let blocksPerRow = inDim / Q4_0.valuesPerBlock
                let rowBytes = blocksPerRow * Q4_0.bytesPerBlock
                var worstPlain = 0.0, worstTiled = 0.0
                for r in 0..<rowProbe {
                    let rowStart = data.startIndex + r * rowBytes
                    var acc = 0.0
                    for b in 0..<blocksPerRow {
                        let blk = data[rowStart + b * 18..<rowStart + b * 18 + 18]
                        let dbits = UInt16(blk[blk.startIndex]) | (UInt16(blk[blk.startIndex + 1]) << 8)
                        let d = Double(Float16(bitPattern: dbits))
                        for l in 0..<16 {
                            let q = blk[blk.startIndex + 2 + l]
                            let lo = Double(Int(q & 0x0F) - 8)
                            let hi = Double(Int(q >> 4) - 8)
                            acc += d * lo * Double(xRef[b * 32 + l])
                            acc += d * hi * Double(xRef[b * 32 + 16 + l])
                        }
                    }
                    worstPlain = max(worstPlain, abs(Double(plain[r]) - acc) / max(1.0, abs(acc)))
                    worstTiled = max(worstTiled, abs(Double(tiled[r]) - acc) / max(1.0, abs(acc)))
                }
                print(String(format: "fp64 attribution at position %d (output step %d) layer %d (%@, %d rows): rel-to-FP64 worst plain %.3e | tiled %.3e",
                             stepL, stepL - (history.count - 1), layerL, tensor,
                             rowProbe, worstPlain, worstTiled))
            } catch { fail("fp64 attribution failed: \(error)") }
            print("DIAGNOSIS_COMPLETED")
        } catch {
            fail("gemma-diagnose failed: \(error)")
        }

    case "q4-lower":
        // Phase 9 parameterized Q4_0 OPERATOR lowering (one family).
        // Python op input (.py with semantics/shape/layout/quant/precision
        // metadata, captured by tools/q4_lower/export_q4_op.py) -> op IR
        // validation -> shape+hardware-derived schedule enumeration ->
        // Metal source EMISSION -> independent CPU FP64 oracle gate under
        // the existing 1e-4 component contract -> persisted artifact with
        // full identity binding. No model run here; no performance claim.
        // args[0] is the command itself; the subcommand follows it.
        let rest = Array(args.dropFirst())
        let mode = rest.first ?? ""
        switch mode {
        case "forge":
            let opPath = requireValue("--op", args)
            let outPath = requireValue("--out", args)
            let ggufPath = args.firstIndex(of: "--gguf").map { _ in
                requireValue("--gguf", args) }
            do {
                let opJSON = try Data(contentsOf: URL(fileURLWithPath: opPath))
                let (artifact, results, incumbentRel) = try Q4LoweringForge.forge(
                    opJSON: opJSON,
                    ggufURL: ggufPath.map { URL(fileURLWithPath: $0) })
                print(String(format: "incumbent g4_q4_matvec rel-to-FP64: %.3e", incumbentRel))
                for r in results {
                    print("  \(r.accepted ? "ACCEPT" : "REJECT") \(r.schedule.kernelName): \(r.note)")
                }
                try artifact.save(to: URL(fileURLWithPath: outPath))
                print("selected: \(artifact.schedule.kernelName)")
                print("artifact saved: \(outPath) (digest \(artifact.artifactDigest.prefix(16))…)")
            } catch {
                fail("q4-lower forge failed: \(error)")
            }
        case "verify":
            let artifactPath = requireValue("--artifact", args)
            let ggufPath = requireValue("--gguf", args)
            do {
                let artifact = try Q4LoweringArtifact.load(
                    from: URL(fileURLWithPath: artifactPath))
                let ggufURL = URL(fileURLWithPath: ggufPath)
                // The op is ALSO re-validated against the actual GGUF
                // tensor directory here — not just the artifact's own copy.
                try artifact.op.validateAgainst(file: try GGUFReader.read(ggufURL))
                // Model SHA-256 from the shared digest cache (hashed once;
                // reused without re-reading unless --rehash-model).
                let digest = try cachedModelDigest(ggufURL)
                try artifact.validate(op: artifact.op,
                                      hardware: Q4CurrentHardware.profile(),
                                      ggufDigest: digest,
                                      executableDigest: try Gemma4Runner.executableDigest())
                print("artifact VALID: \(artifact.schedule.kernelName) "
                      + "bound to \(artifact.op.tensor ?? "<synthetic>") "
                      + "[\(artifact.op.inDim), \(artifact.op.outDim)]")
                print("codegen \(artifact.codegenVersion), rows \(artifact.representativeRows), "
                      + "source sha \(artifact.sourceSha256.prefix(16))…")
            } catch {
                fail("q4-lower verify failed: \(error)")
            }
        case "select":
            // Full bounded orchestration in ONE process: component gates
            // for every derived candidate, component performance screen
            // at the real hot shape (phase 11), per-finalist teacher-
            // forced model gates on the EN/KR chat fixtures, matched
            // timing vs the incumbent, documented policy selection,
            // persistence, and a persisted-execution proof (reload from
            // disk -> validate -> production install -> gated replay with
            // dispatch counters).
            // --smoke-test: bounded END-TO-END plumbing check (raw-hello
            // only, gen 2 / timed 1 by default) — validates that the new
            // pipeline paths run; NEVER a numerical-quality or
            // performance claim. The regular path retains all 3 fixtures.
            let opPath = requireValue("--op", args)
            let ggufPath = requireValue("--gguf", args)
            let outPath = requireValue("--out", args)
            let smokeTest = args.contains("--smoke-test")
            let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
                Int(requireValue("--gen-tokens", args)) ?? 0 } ?? (smokeTest ? 2 : 32)
            let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
                Int(requireValue("--timed-tokens", args)) ?? 0 } ?? (smokeTest ? 1 : 31)
            guard genTokens >= 2, genTokens <= 64, timedTokens >= 1, timedTokens < genTokens else {
                fail("--gen-tokens must be 2...64 and --timed-tokens 1...(gen-1)")
            }
            if smokeTest {
                print("SMOKE MODE: raw-hello only, gen \(genTokens)/timed \(timedTokens) — "
                      + "plumbing check, NOT a quality/performance proof")
            }
            let enText = args.firstIndex(of: "--text-en").map { _ in
                requireValue("--text-en", args) } ?? "Hello, can you help me today?"
            let krText = args.firstIndex(of: "--text-kr").map { _ in
                requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
            do {
                let fileURL = URL(fileURLWithPath: ggufPath)
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                // Canonical chat fixtures (the SAME user→model template the
                // existing --chat path derived from the GGUF's embedded
                // Jinja template) for BOTH languages, plus the raw Hello
                // fixture for continuity with the phase-7/8 evidence.
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                var fixtures = smokeTest
                    ? [Q4LoweringForge.Q4Fixture(name: "raw-hello",
                                                 tokens: try tokenizer.encode("Hello"))]
                    : [
                    Q4LoweringForge.Q4Fixture(name: "en-chat", tokens: try chatIds(enText)),
                    Q4LoweringForge.Q4Fixture(name: "kr-chat", tokens: try chatIds(krText)),
                    Q4LoweringForge.Q4Fixture(name: "raw-hello", tokens: try tokenizer.encode("Hello")),
                ]
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (_, log) = try Q4LoweringForge.selectAndPersist(
                    opJSON: try Data(contentsOf: URL(fileURLWithPath: opPath)),
                    ggufURL: fileURL, runner: runner, fixtures: fixtures,
                    genTokens: genTokens, timedTokens: timedTokens,
                    isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                    saveTo: URL(fileURLWithPath: outPath))
                for line in log { print(line) }
                print("Q4_SELECT_COMPLETED -> \(outPath)")
            } catch {
                fail("q4-lower select failed: \(error)")
            }
        case "launch-select":
            // Phase 15 Q4 LAUNCH-GEOMETRY selection (bit-exact by
            // construction: same pipeline, same per-thread arithmetic,
            // only the g4_q4_matvec launch width changes). Fair
            // comparator: the ACCEPTED E4_R32 baseline in THIS binary —
            // both arms run E4 + fresh binding; the reference is the E4
            // old-width control over the 3 canonical prompts x 32 steps
            // (all-logit strict gates); 3 paired AB/BA x 31 matched
            // decode steps at the fixed 2% policy. --smoke-test is an
            // internal diagnostic record that can never install.
            let regionPath = requireValue("--region", args)
            let ggufPath = requireValue("--gguf", args)
            let outPath = requireValue("--out", args)
            let graphPath = requireValue("--graph", args)
            let smokeTest = args.contains("--smoke-test")
            let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
                Int(requireValue("--gen-tokens", args)) ?? 0 } ?? (smokeTest ? 2 : 32)
            let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
                Int(requireValue("--timed-tokens", args)) ?? 0 } ?? (smokeTest ? 1 : 31)
            if smokeTest {
                print("SMOKE MODE: raw-hello only, gen \(genTokens)/timed \(timedTokens) — "
                      + "plumbing check, NOT a quality/performance proof")
            } else {
                guard genTokens == 32, timedTokens == 31 else {
                    fail("production launch-select requires --gen-tokens 32 and --timed-tokens 31")
                }
            }
            let enText = args.firstIndex(of: "--text-en").map { _ in
                requireValue("--text-en", args) } ?? "Hello, can you help me today?"
            let krText = args.firstIndex(of: "--text-kr").map { _ in
                requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
            do {
                let fileURL = URL(fileURLWithPath: ggufPath)
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                defer { try? graphHandle.close() }
                guard let graphData = try graphHandle.read(upToCount: 8 << 20),
                      !graphData.isEmpty else {
                    fail("graph file unreadable: \(graphPath)")
                }
                let graph = try GemmaGraph.parse(graphData)
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                let fixtures = smokeTest
                    ? [Q4LoweringForge.Q4Fixture(name: "raw-hello",
                                                 tokens: try tokenizer.encode("Hello"))]
                    : [
                    Q4LoweringForge.Q4Fixture(name: "en-chat", tokens: try chatIds(enText)),
                    Q4LoweringForge.Q4Fixture(name: "kr-chat", tokens: try chatIds(krText)),
                    Q4LoweringForge.Q4Fixture(name: "raw-hello", tokens: try tokenizer.encode("Hello")),
                ]
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (_, log) = try Q4LaunchSelect.selectAndPersist(
                    regionJSON: try Data(contentsOf: URL(fileURLWithPath: regionPath)),
                    ggufURL: fileURL, graph: graph, runner: runner, fixtures: fixtures,
                    genTokens: genTokens, timedTokens: timedTokens,
                    isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                    diagnostic: smokeTest,
                    saveTo: URL(fileURLWithPath: outPath))
                for line in log { print(line) }
                print("Q4LAUNCH_SELECT_COMPLETED -> \(outPath)")
            } catch {
                fail("q4-lower launch-select failed: \(error)")
            }
        case "launch-verify-production":
            // FRESH-PROCESS verification of the composite Q4 launch
            // artifact (separately launched): identity first, then an
            // E4-baseline (default width) 3x32 reference with workload
            // digest checks, PUBLIC composite install (E4 + width
            // coherently), and strict 3x32 candidate gates with
            // derived eligible-geometry and MoE counter deltas.
            let artifactPath = requireValue("--artifact", args)
            let ggufPath = requireValue("--gguf", args)
            let regionPath = requireValue("--region", args)
            let graphPath = requireValue("--graph", args)
            do {
                let artifact = try Q4LaunchArtifact.load(
                    from: URL(fileURLWithPath: artifactPath))
                let fileURL = URL(fileURLWithPath: ggufPath)
                let region = try MoERegionIR.parse(
                    try Data(contentsOf: URL(fileURLWithPath: regionPath)))
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                defer { try? graphHandle.close() }
                guard let graphData = try graphHandle.read(upToCount: 8 << 20),
                      !graphData.isEmpty else {
                    fail("graph file unreadable: \(graphPath)")
                }
                let graph = try GemmaGraph.parse(graphData)
                let ggufDigest = try cachedModelDigest(fileURL)
                let graphDigest = try GemmaGraph.digest(graph)
                let runner = try Gemma4Runner(fileAt: fileURL)
                try artifact.validate(region: region,
                                      pipelineMaxThreads: try runner.q4MatvecPipelineMaxThreads(),
                                      hardware: runner.hardwareProfile(),
                                      graphDigest: graphDigest,
                                      ggufDigest: ggufDigest,
                                      executableDigest: try Gemma4Runner.executableDigest())
                guard let gate = artifact.modelGate,
                      Q4LaunchGatePolicy.productionValid(gate, schedule: artifact.schedule,
                                              binding: artifact.baselineMoeBinding) else {
                    fail("artifact is not production-valid (diagnostic/no-win records "
                         + "never verify publicly)")
                }
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                let (ssaMany, ssaOne) = LoweringSelect.ssaForwardAdapters(graph: graph)
                let enText = args.firstIndex(of: "--text-en").map { _ in
                    requireValue("--text-en", args) } ?? "Hello, can you help me today?"
                let krText = args.firstIndex(of: "--text-kr").map { _ in
                    requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                let prompts: [(String, [Int])] = [
                    ("en-chat", try chatIds(enText)),
                    ("kr-chat", try chatIds(krText)),
                    ("raw-hello", try tokenizer.encode("Hello")),
                ]
                guard prompts.map(\.0) == gate.fixtures.map(\.name) else {
                    fail("fixture names/order differ from the recorded selection")
                }
                let genTokens = Q4LaunchGatePolicy.productionGenTokens

                // E4 baseline reference (default launch width, E4
                // installed, nothing else) — exact 32 steps per fixture.
                runner.uninstallQ4LaunchGeometry()
                runner.uninstallMoeGeneratedLowering()
                guard MTLCreateSystemDefaultDevice() != nil else {
                    fail("Metal unavailable")
                }
                try runner.installMoeBaselineForVerification(
                    region: region, schedule: artifact.baselineMoeSchedule,
                    expectedSourceSha256: artifact.baselineMoeSourceSha256)
                defer {
                    runner.uninstallQ4LaunchGeometry()
                    runner.uninstallMoeGeneratedLowering()
                }
                var refs: [(name: String, prompt: [Int], ids: [Int],
                            logits: [[Float]], baselineEligible: Int)] = []
                for (name, prompt) in prompts {
                    try LoweringSelect.ssaReset(runner)
                    // Baseline eligible population: actual g4_q4_matvec
                    // dispatches for THIS fixture's whole history — the
                    // candidate replay's derived population must equal it.
                    let baseEligible0 = runner.q4MatvecGenericDispatchCount
                    let (am0, lg0) = try ssaMany(runner, prompt)
                    guard lg0.count == runner.config.vocabSize,
                          lg0.allSatisfy({ $0.isFinite }) else {
                        fail("E4-baseline reference [\(name)]: nonfinite/wrong-size logits")
                    }
                    var ids = [am0]
                    var logits = [lg0]
                    var prev = am0
                    while ids.count < genTokens {
                        if tokenizer.isEndOfGeneration(prev) { break }
                        let (am, lg) = try ssaOne(runner, prev)
                        // EVERY decode step's logits are validated (full
                        // vocab, finite) and the id range-checked before
                        // anything is gated against them.
                        guard lg.count == runner.config.vocabSize,
                              lg.allSatisfy({ $0.isFinite }),
                              am >= 0, am < runner.config.vocabSize else {
                            fail("E4-baseline reference [\(name)] step \(ids.count): "
                                 + "nonfinite/wrong-size logits or bad id")
                        }
                        ids.append(am); logits.append(lg); prev = am
                    }
                    guard ids.count == genTokens else {
                        fail("E4-baseline reference [\(name)]: only \(ids.count) steps "
                             + "(EOS) — verification requires exactly \(genTokens)")
                    }
                    refs.append((name, prompt, ids, logits,
                                 runner.q4MatvecGenericDispatchCount - baseEligible0))
                    print("E4-baseline reference [\(name)]: \(ids.count) steps "
                          + "(eligible g4_q4_matvec dispatches "
                          + "\(refs[refs.count - 1].baselineEligible))")
                }
                for (i, ref) in refs.enumerated() {
                    var history = ref.prompt
                    history.append(contentsOf: ref.ids.dropLast())
                    let digest = GemmaForge.workloadDigest(history)
                    guard gate.fixtures[i].promptTokenCount == ref.prompt.count,
                          gate.fixtures[i].historyDigest == digest else {
                        fail("fixture \(ref.name): workload digest/prompt length "
                             + "differs from the recorded selection")
                    }
                }

                // PUBLIC composite install + strict candidate gates.
                try runner.installQ4LaunchGenerated(artifact: artifact,
                                                    ggufDigest: ggufDigest,
                                                    region: region,
                                                    graphDigest: graphDigest)
                print("composite production install OK: \(artifact.schedule.label) on E4_R32")
                let layerCount = artifact.baselineMoeBinding.layerCount
                let topK = region.topK
                let moeE = artifact.baselineMoeSchedule.batch
                var worstAll = 0.0
                for ref in refs {
                    // Fresh context per fixture (installs preserved — the
                    // reset only clears KV/position).
                    try LoweringSelect.ssaReset(runner)
                    let eligible0 = runner.q4MatvecGenericDispatchCount
                    let applied0 = runner.q4LaunchAppliedCount
                    let gu0 = runner.moeGuDispatchCount, act0 = runner.moeActDispatchCount
                    let dn0 = runner.moeDnDispatchCount, m0 = runner.moeAxpyMergeCount
                    let fb0 = runner.q4LaunchFallbackCount
                    let mfb0 = runner.moeResidentFallbackCount
                    var logits = [[Float]]()
                    let (_, lg0) = try ssaMany(runner, ref.prompt)
                    logits.append(lg0)
                    for s in 0..<(ref.ids.count - 1) {
                        let (_, lg) = try ssaOne(runner, ref.ids[s])
                        logits.append(lg)
                    }
                    guard logits.count == genTokens else {
                        fail("candidate replay [\(ref.name)]: step count mismatch")
                    }
                    for s in 0..<ref.ids.count {
                        let lg = logits[s], want = ref.logits[s]
                        guard lg.count == want.count,
                              lg.allSatisfy({ $0.isFinite }) else {
                            fail("verify [\(ref.name)] step \(s): nonfinite/shape")
                        }
                        let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                        guard argmax == ref.ids[s] else {
                            fail("verify [\(ref.name)] step \(s): greedy mismatch")
                        }
                        for i in 0..<lg.count {
                            worstAll = max(worstAll, abs(Double(lg[i] - want[i]))
                                           / max(1.0, abs(Double(want[i]))))
                        }
                        if worstAll >= 1e-4 {
                            fail(String(format: "verify [%@] step %d outside 1e-4 (%.3e)",
                                        ref.name, s, worstAll))
                        }
                    }
                    let forwards = ref.prompt.count + ref.ids.count - 1
                    // One stage dispatch per CHUNK (E4 over topK8 = 2 per
                    // layer per forward) — NO extra factor 2.
                    let wantStage = ((topK + moeE - 1) / moeE) * layerCount * forwards
                    let wantMerge = topK * layerCount * forwards
                    let eligible = runner.q4MatvecGenericDispatchCount - eligible0
                    let applied = runner.q4LaunchAppliedCount - applied0
                    let geoFb = runner.q4LaunchFallbackCount - fb0
                    let moeFb = runner.moeResidentFallbackCount - mfb0
                    guard applied == eligible, eligible > 0,
                          eligible == ref.baselineEligible, geoFb == 0, moeFb == 0,
                          runner.moeGuDispatchCount - gu0 == wantStage,
                          runner.moeActDispatchCount - act0 == wantStage,
                          runner.moeDnDispatchCount - dn0 == wantStage,
                          runner.moeAxpyMergeCount - m0 == wantMerge else {
                        fail("verify [\(ref.name)]: counters wrong (applied \(applied)/"
                             + "eligible \(eligible), geoFb \(geoFb), moeFb \(mfb0))")
                    }
                    print(String(format: "candidate replay [%@]: PASS, worst %.3e; geometry applied %d == eligible, MoE stages %d, merges %d, fallbacks 0",
                                 ref.name, worstAll, applied, wantStage, wantMerge))
                }
                print(String(format: "Q4LAUNCH_VERIFY_PRODUCTION_OK worst rel %.3e artifact %@",
                             worstAll, artifact.artifactDigest.prefix(16) + "…"))
            } catch {
                fail("q4-lower launch-verify-production failed: \(error)")
            }
        default:
            fail("q4-lower requires a mode: forge | verify | select | launch-select | launch-verify-production")
        }

    case "moe-lower":
        // Phase 13 routed MoE region lowering: REAL torch.export(strict=True)
        // region capture (tools/moe_lower/export_moe_region.py, validated
        // and re-derived independently by the Swift importer) -> batched
        // E in {1,2,4,8} x row-tile MSL emission (incumbent serial
        // arithmetic, literal shared activation helpers) -> component
        // bit-identity gates vs the ACTUAL incumbent kernels + FP64
        // oracle -> persisted artifact with full identity binding.
        // forge/verify run no model; select is the full bounded
        // orchestration (see docs/MOE_BATCH_LOWERING_DESIGN.md).
        let rest = Array(args.dropFirst())
        let mode = rest.first ?? ""
        switch mode {
        case "forge":
            let regionPath = requireValue("--region", args)
            let outPath = requireValue("--out", args)
            let ggufPath = requireValue("--gguf", args)
            do {
                let regionJSON = try Data(contentsOf: URL(fileURLWithPath: regionPath))
                let (artifact, results, incumbentRel, region) = try MoELoweringForge.forge(
                    regionJSON: regionJSON, ggufURL: URL(fileURLWithPath: ggufPath))
                print(String(format: "region %@ [h %d, e %d, topK %d] incumbent rel-to-FP64: %.3e",
                             region.region, region.hidden, region.expertIntermediate,
                             region.topK, incumbentRel))
                for r in results {
                    print("  \(r.accepted ? "ACCEPT" : "REJECT") \(r.schedule.guKernelName): \(r.note)")
                }
                try artifact.save(to: URL(fileURLWithPath: outPath))
                print("selected: \(artifact.schedule.guKernelName)")
                print("artifact saved: \(outPath) (digest \(artifact.artifactDigest.prefix(16))…)")
            } catch {
                fail("moe-lower forge failed: \(error)")
            }
        case "verify":
            let artifactPath = requireValue("--artifact", args)
            let ggufPath = requireValue("--gguf", args)
            do {
                let artifact = try MoELoweringArtifact.load(
                    from: URL(fileURLWithPath: artifactPath))
                guard let device = MTLCreateSystemDefaultDevice() else {
                    fail("moe-lower verify failed: Metal unavailable")
                }
                let ggufURL = URL(fileURLWithPath: ggufPath)
                // Fresh binding against the ACTUAL file/device (page-
                // rounded resident length) must equal the recorded one.
                let fresh = try MoELoweringForge.bindRegion(
                    region: artifact.region, file: try GGUFReader.read(ggufURL),
                    device: device)
                guard fresh == artifact.binding else {
                    fail("moe-lower verify failed: recorded binding does not match this file/device")
                }
                try artifact.validate(region: artifact.region,
                                      hardware: MoECurrentHardware.profile(),
                                      ggufDigest: try cachedModelDigest(ggufURL),
                                      executableDigest: try Gemma4Runner.executableDigest())
                let status = artifact.modelGate == nil
                    ? "component-only (diagnostic; not installable)"
                    : (artifact.modelGate!.diagnostic
                           ? "diagnostic smoke (not publicly installable)"
                           : (artifact.modelGate!.performanceSelected
                                  ? "production-selected"
                                  : "gate-passed, incumbent retained (no policy win)"))
                print("artifact VALID: \(artifact.schedule.guKernelName) "
                    + "[h \(artifact.region.hidden), e \(artifact.region.expertIntermediate), "
                    + "topK \(artifact.region.topK)] layers \(artifact.binding.layerCount) — \(status)")
                print("codegen \(artifact.codegenVersion), graph sha \(artifact.region.graphSha256.prefix(16))…, "
                    + "source sha \(artifact.sourceSha256.prefix(16))…")
            } catch {
                fail("moe-lower verify failed: \(error)")
            }
        case "select":
            // Full bounded orchestration in ONE process (component gates +
            // screen, per-finalist strict teacher-forced gates with
            // dispatch/fallback proof, matched AB/BA paired timing, fixed
            // 2% policy, persistence, reload proof). --smoke-test is the
            // bounded PLUMBING path (raw-hello, gen 2 / timed 1): its
            // record is tagged diagnostic and can never install publicly.
            // The normal path is exactly 3 fixtures x 32 steps / 31 timed.
            let regionPath = requireValue("--region", args)
            let ggufPath = requireValue("--gguf", args)
            let outPath = requireValue("--out", args)
            let smokeTest = args.contains("--smoke-test")
            let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
                Int(requireValue("--gen-tokens", args)) ?? 0 } ?? (smokeTest ? 2 : 32)
            let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
                Int(requireValue("--timed-tokens", args)) ?? 0 } ?? (smokeTest ? 1 : 31)
            if smokeTest {
                print("SMOKE MODE: raw-hello only, gen \(genTokens)/timed \(timedTokens) — "
                    + "plumbing check, NOT a quality/performance proof")
            } else {
                guard genTokens == 32, timedTokens == 31 else {
                    fail("production select requires --gen-tokens 32 and --timed-tokens 31")
                }
            }
            let enText = args.firstIndex(of: "--text-en").map { _ in
                requireValue("--text-en", args) } ?? "Hello, can you help me today?"
            let krText = args.firstIndex(of: "--text-kr").map { _ in
                requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
            do {
                let fileURL = URL(fileURLWithPath: ggufPath)
                let graphPath = requireValue("--graph", args)
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                defer { try? graphHandle.close() }
                guard let graphData = try graphHandle.read(upToCount: 8 << 20),
                      !graphData.isEmpty else {
                    fail("graph file unreadable: \(graphPath)")
                }
                let graph = try GemmaGraph.parse(graphData)
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                let fixtures = smokeTest
                    ? [Q4LoweringForge.Q4Fixture(name: "raw-hello",
                                                 tokens: try tokenizer.encode("Hello"))]
                    : [
                    Q4LoweringForge.Q4Fixture(name: "en-chat", tokens: try chatIds(enText)),
                    Q4LoweringForge.Q4Fixture(name: "kr-chat", tokens: try chatIds(krText)),
                    Q4LoweringForge.Q4Fixture(name: "raw-hello", tokens: try tokenizer.encode("Hello")),
                ]
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (_, log) = try MoELoweringForge.selectAndPersist(
                    regionJSON: try Data(contentsOf: URL(fileURLWithPath: regionPath)),
                    ggufURL: fileURL, graph: graph, runner: runner, fixtures: fixtures,
                    genTokens: genTokens, timedTokens: timedTokens,
                    isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                    diagnostic: smokeTest,
                    saveTo: URL(fileURLWithPath: outPath))
                for line in log { print(line) }
                print("MOE_SELECT_COMPLETED -> \(outPath)")
            } catch {
                fail("moe-lower select failed: \(error)")
            }
        case "profile":
            // Phase 14 DIAGNOSTIC INSTRUMENTATION ONLY (no optimization,
            // no public install, no artifact approval/rebinding): the
            // static E4_R32 schedule is freshly compiled from the SAME
            // deterministic emitter and installed via the INTERNAL trial
            // path, then the canonical raw-hello 32-step history is
            // measured on the ACTUAL Forge SSA path two ways — normal
            // production settings (per-token wall/GPU + breakdown) and an
            // intrusive per-kernel replay (segmentBatching OFF ONLY for
            // GPU attribution) gated on all 262144 logits per step. An
            // optional --expect-source-sha (the accepted phase-13
            // artifact's sourceSha256) fails closed if emission drifted.
            let regionPath = requireValue("--region", args)
            let ggufPath = requireValue("--gguf", args)
            let graphPath = requireValue("--graph", args)
            let outPath = requireValue("--out", args)
            let expectSha = args.firstIndex(of: "--expect-source-sha").map { _ in
                requireValue("--expect-source-sha", args) }
            do {
                let fileURL = URL(fileURLWithPath: ggufPath)
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                defer { try? graphHandle.close() }
                guard let graphData = try graphHandle.read(upToCount: 8 << 20),
                      !graphData.isEmpty else {
                    fail("graph file unreadable: \(graphPath)")
                }
                let graph = try GemmaGraph.parse(graphData)
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (report, log) = try MoELoweringProfile.run(
                    regionJSON: try Data(contentsOf: URL(fileURLWithPath: regionPath)),
                    ggufURL: fileURL, graph: graph, runner: runner,
                    promptTokens: try tokenizer.encode("Hello"),
                    isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                    expectedSourceSha256: expectSha)
                for line in log { print(line) }
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                try enc.encode(report).write(to: URL(fileURLWithPath: outPath))
                print("MOE_PROFILE_COMPLETED -> \(outPath) (diagnostic trial only; "
                    + "no artifact approved, nothing publicly installed)")
            } catch {
                fail("moe-lower profile failed: \(error)")
            }
        case "verify-production":
            // FRESH-PROCESS production verification (required before any
            // optimized default is accepted/published): identity/policy
            // validation FIRST, then an UNOPTIMIZED incumbent reference on
            // the actual Forge SSA path (the same engine primitives — this
            // is not an independent semantic-quality proof; the FP64
            // component oracle is bounded separately), then the artifact
            // installed through the PUBLIC production path and the same
            // histories replayed with ALL 262144 logits gated per step and
            // per-stage dispatch DELTAS asserted. A two-token smoke record
            // can NEVER install here; early EOS on any fixture is a FAIL.
            let artifactPath = requireValue("--artifact", args)
            let ggufPath = requireValue("--gguf", args)
            let graphPath = requireValue("--graph", args)
            do {
                let artifact = try MoELoweringArtifact.load(
                    from: URL(fileURLWithPath: artifactPath))
                let fileURL = URL(fileURLWithPath: ggufPath)
                // Identity + full production gate policy BEFORE any heavy
                // reference work (public install enforces productionValid). The
                // model SHA-256 comes from the shared digest cache.
                let ggufDigest = try cachedModelDigest(fileURL)
                let graphHandle = try FileHandle(
                    forReadingFrom: URL(fileURLWithPath: graphPath))
                defer { try? graphHandle.close() }
                guard let graphData = try graphHandle.read(upToCount: 8 << 20),
                      !graphData.isEmpty else {
                    fail("graph file unreadable: \(graphPath)")
                }
                let graph = try GemmaGraph.parse(graphData)
                try artifact.validate(region: artifact.region,
                                      hardware: MoECurrentHardware.profile(),
                                      ggufDigest: ggufDigest,
                                      executableDigest: try Gemma4Runner.executableDigest())
                guard let gate = artifact.modelGate,
                      MoEGatePolicy.productionValid(gate,
                                                    schedule: artifact.schedule) else {
                    fail("artifact is not production-valid (diagnostic/component-only "
                         + "records never verify publicly)")
                }
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (ssaMany, ssaOne) = LoweringSelect.ssaForwardAdapters(graph: graph)
                // Recorded selection workloads must match EXACTLY (names,
                // order, prompt lengths, recomputed history digests).
                let enText = args.firstIndex(of: "--text-en").map { _ in
                    requireValue("--text-en", args) } ?? "Hello, can you help me today?"
                let krText = args.firstIndex(of: "--text-kr").map { _ in
                    requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                let prompts: [(String, [Int])] = [
                    ("en-chat", try chatIds(enText)),
                    ("kr-chat", try chatIds(krText)),
                    ("raw-hello", try tokenizer.encode("Hello")),
                ]
                guard prompts.map(\.0) == gate.fixtures.map(\.name) else {
                    fail("fixture names/order differ from the recorded selection")
                }
                let genTokens = MoEGatePolicy.productionGenTokens

                // Unoptimized incumbent reference (SSA path, nothing installed).
                runner.uninstallMoeGeneratedLowering()
                var refs: [(name: String, prompt: [Int], ids: [Int], logits: [[Float]])] = []
                for (name, prompt) in prompts {
                    try LoweringSelect.ssaReset(runner)
                    let (am0, lg0) = try ssaMany(runner, prompt)
                    guard lg0.count == runner.config.vocabSize,
                          lg0.allSatisfy({ $0.isFinite }) else {
                        fail("incumbent reference [\(name)]: nonfinite/wrong-size logits")
                    }
                    var ids = [am0]
                    var logits = [lg0]
                    var prev = am0
                    while ids.count < genTokens {
                        if tokenizer.isEndOfGeneration(prev) { break }
                        let (am, lg) = try ssaOne(runner, prev)
                        ids.append(am); logits.append(lg); prev = am
                    }
                    // Early EOS is a verification FAIL: the production
                    // record required exactly 32 gated steps.
                    guard ids.count == genTokens else {
                        fail("incumbent reference [\(name)]: only \(ids.count) steps "
                             + "(EOS) — production requires exactly \(genTokens)")
                    }
                    refs.append((name, prompt, ids, logits))
                    print("unoptimized incumbent reference [\(name)]: \(ids.count) steps")
                }
                // Recorded workload digests must match the recomputed ones.
                for (i, ref) in refs.enumerated() {
                    var history = ref.prompt
                    history.append(contentsOf: ref.ids.dropLast())
                    let digest = GemmaForge.workloadDigest(history)
                    guard gate.fixtures[i].promptTokenCount == ref.prompt.count,
                          gate.fixtures[i].historyDigest == digest else {
                        fail("fixture \(ref.name): workload digest/prompt length "
                             + "differs from the recorded selection")
                    }
                }

                // PUBLIC production install + teacher-forced SSA replay.
                try runner.installMoeGeneratedLowering(artifact: artifact,
                                                       ggufDigest: ggufDigest)
                print("production install OK: \(artifact.schedule.guKernelName)")
                let layerCount = artifact.binding.layerCount
                let topK = artifact.region.topK
                let eSel = artifact.schedule.batch
                var worstAll = 0.0
                for ref in refs {
                    let gu0 = runner.moeGuDispatchCount, act0 = runner.moeActDispatchCount
                    let dn0 = runner.moeDnDispatchCount, m0 = runner.moeAxpyMergeCount
                    let fb0 = runner.moeResidentFallbackCount
                    var armIds = [Int]()
                    var logits = [[Float]]()
                    let (am0, lg0) = try ssaMany(runner, ref.prompt)
                    armIds.append(am0); logits.append(lg0)
                    for s in 0..<(ref.ids.count - 1) {
                        let (am, lg) = try ssaOne(runner, ref.ids[s])
                        armIds.append(am); logits.append(lg)
                    }
                    guard armIds.count == genTokens else {
                        fail("replay [\(ref.name)]: \(armIds.count) steps (EOS) — FAIL")
                    }
                    for s in 0..<ref.ids.count {
                        let lg = logits[s]
                        let want = ref.logits[s]
                        guard lg.count == want.count,
                              lg.allSatisfy({ $0.isFinite }) else {
                            fail("verify-production [\(ref.name)] step \(s): nonfinite/shape")
                        }
                        let argmax = lg.enumerated().max { $0.element < $1.element }!.offset
                        guard argmax == ref.ids[s] else {
                            fail("verify-production [\(ref.name)] step \(s): greedy mismatch")
                        }
                        for i in 0..<lg.count {
                            worstAll = max(worstAll, abs(Double(lg[i] - want[i]))
                                           / max(1.0, abs(Double(want[i]))))
                        }
                        if worstAll >= 1e-4 {
                            fail(String(format: "verify-production [%@] step %d outside 1e-4 (%.3e)",
                                        ref.name, s, worstAll))
                        }
                    }
                    let forwards = ref.prompt.count + ref.ids.count - 1
                    let wantBatched = 3 * ((topK + eSel - 1) / eSel) * layerCount * forwards
                    let wantMerges = topK * layerCount * forwards
                    let dGu = runner.moeGuDispatchCount - gu0
                    let dAct = runner.moeActDispatchCount - act0
                    let dDn = runner.moeDnDispatchCount - dn0
                    let dM = runner.moeAxpyMergeCount - m0
                    let dFb = runner.moeResidentFallbackCount - fb0
                    guard dGu == wantBatched / 3, dAct == wantBatched / 3,
                          dDn == wantBatched / 3, dM == wantMerges, dFb == 0 else {
                        fail("verify-production [\(ref.name)]: dispatch deltas "
                             + "GU \(dGu)/\(wantBatched / 3) ACT \(dAct) DN \(dDn) "
                             + "merges \(dM)/\(wantMerges) fallbacks \(dFb)")
                    }
                    print(String(format: "SSA replay [%@]: PASS, worst rel %.3e; batched deltas %d/%d/%d, merges %d, fallbacks 0",
                                 ref.name, worstAll, dGu, dAct, dDn, dM))
                }
                runner.uninstallMoeGeneratedLowering()
                print(String(format: "MOE_VERIFY_PRODUCTION_OK worst rel %.3e artifact %@",
                             worstAll, artifact.artifactDigest.prefix(16) + "…"))
            } catch {
                fail("moe-lower verify-production failed: \(error)")
            }
        default:
            fail("moe-lower requires a mode: forge | verify | select | verify-production | profile")
        }

    case "rms-lower":
        // Phase 10 parameterized serial-order-preserving cooperative
        // RMSNorm lowering (one family). Python op input (captured by
        // tools/rms_lower/export_rms_op.py) -> op IR validation ->
        // shape+hardware-derived schedule enumeration -> Metal source
        // EMISSION -> independent CPU FP64 oracle gate under the existing
        // 1e-4 component contract PLUS hardware bit-identity to the
        // incumbent production kernel -> persisted artifact with full
        // identity binding. forge/verify run no model; select is the full
        // bounded orchestration (see docs/RMS_LOWERING_DESIGN.md).
        let rest = Array(args.dropFirst())
        let mode = rest.first ?? ""
        switch mode {
        case "forge":
            let opPath = requireValue("--op", args)
            let outPath = requireValue("--out", args)
            let ggufPath = args.firstIndex(of: "--gguf").map { _ in
                requireValue("--gguf", args) }
            do {
                let opJSON = try Data(contentsOf: URL(fileURLWithPath: opPath))
                let (artifact, results, incumbentRel) = try RMSLoweringForge.forge(
                    opJSON: opJSON,
                    ggufURL: ggufPath.map { URL(fileURLWithPath: $0) })
                print(String(format: "incumbent rmsnorm rel-to-FP64: %.3e", incumbentRel))
                for r in results {
                    print("  \(r.accepted ? "ACCEPT" : "REJECT") \(r.schedule.kernelName): \(r.note)")
                }
                try artifact.save(to: URL(fileURLWithPath: outPath))
                print("selected: \(artifact.schedule.kernelName)")
                print("artifact saved: \(outPath) (digest \(artifact.artifactDigest.prefix(16))…)")
            } catch {
                fail("rms-lower forge failed: \(error)")
            }
        case "verify":
            let artifactPath = requireValue("--artifact", args)
            let ggufPath = requireValue("--gguf", args)
            do {
                let artifact = try RMSLoweringArtifact.load(
                    from: URL(fileURLWithPath: artifactPath))
                let ggufURL = URL(fileURLWithPath: ggufPath)
                // The op is ALSO re-validated against the actual GGUF
                // tensor directory here — not just the artifact's own copy.
                try artifact.op.validateAgainst(file: try GGUFReader.read(ggufURL))
                // Model SHA-256 from the shared digest cache (hashed once;
                // reused without re-reading unless --rehash-model).
                let digest = try cachedModelDigest(ggufURL)
                try artifact.validate(op: artifact.op,
                                      hardware: RMSCurrentHardware.profile(),
                                      ggufDigest: digest,
                                      executableDigest: try Gemma4Runner.executableDigest())
                print("artifact VALID: \(artifact.schedule.kernelName) "
                      + "bound to \(artifact.op.tensor ?? "<synthetic>") "
                      + "[n=\(artifact.op.n), scaled=\(artifact.op.scaled)]")
                print("codegen \(artifact.codegenVersion), vectors \(artifact.representativeVectors), "
                      + "source sha \(artifact.sourceSha256.prefix(16))…")
            } catch {
                fail("rms-lower verify failed: \(error)")
            }
        case "select":
            // Full bounded orchestration in ONE process (the phase-9
            // protocol): component gates + bit-identity for every derived
            // candidate, warmed component GPU screen (pruning only), at
            // most 2 fastest non-baseline candidates through teacher-forced
            // model gates on the EN/KR chat + raw Hello fixtures, 3 paired
            // matched-timing repetitions, 2% policy, persistence, and a
            // persisted-execution proof with dispatch counters.
            let opPath = requireValue("--op", args)
            let ggufPath = requireValue("--gguf", args)
            let outPath = requireValue("--out", args)
            // --smoke-test: bounded END-TO-END plumbing check of the complete
            // select path (component -> gates -> timing -> persist -> reload
            // -> install -> FINAL formatted proof line). Raw-Hello fixture
            // ONLY, gen 2 / timed 1 defaults. It exercises the exact code the
            // production run executes — including the last String(format:)
            // log — but it is NEVER evidence of 32-step quality or
            // performance; those need the full 3-fixture run.
            let smokeTest = args.contains("--smoke-test")
            let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
                Int(requireValue("--gen-tokens", args)) ?? 0 } ?? (smokeTest ? 2 : 32)
            let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
                Int(requireValue("--timed-tokens", args)) ?? 0 } ?? (smokeTest ? 1 : 31)
            guard genTokens >= 2, genTokens <= 64, timedTokens >= 1, timedTokens < genTokens else {
                fail("--gen-tokens must be 2...64 and --timed-tokens 1...(gen-1)")
            }
            if smokeTest {
                print("SMOKE MODE: raw-hello only, gen \(genTokens)/timed \(timedTokens) — "
                      + "plumbing validation, NOT a quality or performance proof")
            }
            let enText = args.firstIndex(of: "--text-en").map { _ in
                requireValue("--text-en", args) } ?? "Hello, can you help me today?"
            let krText = args.firstIndex(of: "--text-kr").map { _ in
                requireValue("--text-kr", args) } ?? "안녕하세요, 오늘 날씨는 어떤가요?"
            do {
                let fileURL = URL(fileURLWithPath: ggufPath)
                let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
                // Canonical chat fixtures — the SAME user→model template
                // the q4-lower select path uses.
                func chatIds(_ text: String) throws -> [Int] {
                    try tokenizer.encode(
                        "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
                        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                }
                let fixtures = smokeTest
                    ? [RMSLoweringForge.RMSFixture(name: "raw-hello",
                                                   tokens: try tokenizer.encode("Hello"))]
                    : [
                        RMSLoweringForge.RMSFixture(name: "en-chat", tokens: try chatIds(enText)),
                        RMSLoweringForge.RMSFixture(name: "kr-chat", tokens: try chatIds(krText)),
                        RMSLoweringForge.RMSFixture(name: "raw-hello", tokens: try tokenizer.encode("Hello")),
                    ]
                let runner = try Gemma4Runner(fileAt: fileURL)
                let (_, log) = try RMSLoweringForge.selectAndPersist(
                    opJSON: try Data(contentsOf: URL(fileURLWithPath: opPath)),
                    ggufURL: fileURL, runner: runner, fixtures: fixtures,
                    genTokens: genTokens, timedTokens: timedTokens,
                    isEndOfGeneration: { tokenizer.isEndOfGeneration($0) },
                    saveTo: URL(fileURLWithPath: outPath))
                for line in log { print(line) }
                print("RMS_SELECT_COMPLETED -> \(outPath)")
            } catch {
                fail("rms-lower select failed: \(error)")
            }
        default:
            fail("rms-lower requires a mode: forge | verify | select")
        }

    case "gemma-q4gen-gate":
        // Phase 9 model-level acceptance of the generated Q4 lowering.
        // Teacher-forced same-history replay (the gemma-diagnose pattern):
        // the incumbent reference (pass-7 selected configuration:
        // segment batching + vectorized-load RMSNorm) generates a greedy
        // history; the SAME history is replayed with the artifact's
        // emitted kernel installed for its ONE bound tensor, gating full
        // logits at the UNCHANGED strict 1e-4 relative formula + exact
        // greedy token equality + finiteness, per step. Then MATCHED
        // timing: both arms, same binary/model/prompt/context, sequential
        // GPU residency, N timed tokens, wall and GPU medians, plus
        // per-kernel counters proving the generated kernel actually
        // dispatched for the bound tensor. No gate is loosened; a failed
        // gate is reported and the incumbent stays the fallback — no
        // claim of a win from microbenchmarks.
        let path = requireValue("--file", args)
        let artifactPath = requireValue("--artifact", args)
        let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
            Int(requireValue("--gen-tokens", args)) ?? 0 } ?? 32
        guard genTokens >= 2, genTokens <= 64 else {
            fail("--gen-tokens must be 2...64")
        }
        let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
            Int(requireValue("--timed-tokens", args)) ?? 0 } ?? 5
        guard timedTokens >= 1, timedTokens <= 32, timedTokens < genTokens else {
            fail("--timed-tokens must be 1...31 below --gen-tokens")
        }
        do {
            let fileURL = URL(fileURLWithPath: path)
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
            var history: [Int]
            if let tokensArg = args.firstIndex(of: "--tokens").map({ _ in
                requireValue("--tokens", args) }) {
                history = tokensArg.split(separator: ",").map {
                    guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                          id >= 0, id < tokenizer.vocabSize else {
                        fail("invalid token id '\($0)'")
                    }
                    return id
                }
            } else {
                history = try tokenizer.encode(requireValue("--text", args))
            }
            guard !history.isEmpty, history.count + genTokens <= 4096 else {
                fail("prompt + gen-tokens must fit the 4096 context")
            }
            print("prompt ids: \(history.map(String.init).joined(separator: " "))")
            let artifact = try Q4LoweringArtifact.load(
                from: URL(fileURLWithPath: artifactPath))
            let runner = try Gemma4Runner(fileAt: fileURL)

            // ---- reference arm: incumbent kernels, selected pass-7
            // configuration (segment batching + VL RMSNorm). The axis is
            // re-applied AFTER every reset: reset() ->
            // discardOpenSegment() clears autoSegment, so a single
            // up-front apply would silently disable batching. ----
            func resetIncumbentConfig() throws {
                runner.reset()
                runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
                // Fail closed: batching must actually re-engage after the
                // reset (reset() clears autoSegment — the reset-order bug).
                guard runner.segmentBatchingActive else {
                    throw GemmaForgeError.executionFailed(
                        "segment batching failed to re-engage after reset")
                }
            }
            try resetIncumbentConfig()
            var refIds = [Int]()
            var refLogits = [[Float]]()
            do {
                let prefill = try runner.forward(tokenIds: history)
                refLogits.append(try runner.currentLogits())
                refIds.append(prefill.argmaxToken)
                var prev = prefill.argmaxToken
                while refIds.count < genTokens {
                    if tokenizer.isEndOfGeneration(prev) { break }
                    let ev = try runner.forward(tokenIds: [prev])
                    refLogits.append(try runner.currentLogits())
                    refIds.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
            } catch { fail("reference pass failed: \(error)") }
            let steps = refLogits.count
            print("reference greedy ids (\(steps)): \(refIds.map(String.init).joined(separator: " "))")

            // Install the artifact (full identity check; fail closed —
            // an invalid artifact aborts before any generated execution). The
            // model SHA-256 comes from the shared digest cache.
            let ggufDigest = try cachedModelDigest(fileURL)
            do {
                try runner.installQ4GeneratedKernel(artifact: artifact,
                                                    ggufDigest: ggufDigest)
            } catch {
                fail("artifact rejected at install (incumbent stays active): \(error)")
            }
            print("generated kernel installed: \(artifact.schedule.kernelName) "
                  + "for \(artifact.op.tensor ?? "<synthetic>")")

            // Teacher-forced replay of the SAME history under the
            // generated kernel, gated per step.
            var gateWorst = 0.0
            var gateFail: String? = nil
            do {
                try resetIncumbentConfig()
                for t in history {
                    _ = try runner.forward(tokenIds: [t])
                }
                for s in 0..<steps {
                    let lg = try runner.currentLogits()
                    let want = refLogits[s]
                    guard lg.count == want.count,
                          lg.allSatisfy({ $0.isFinite }),
                          want.allSatisfy({ $0.isFinite }) else {
                        gateFail = "step \(s): shape/nonfinite violation"
                        break
                    }
                    // Exact greedy token equality per step (the reference
                    // history IS the argmax of want at every step).
                    let lgArgmax = lg.enumerated().max { $0.element < $1.element }!.offset
                    if lgArgmax != refIds[s] {
                        gateFail = "step \(s): argmax \(lgArgmax) != reference \(refIds[s])"
                        break
                    }
                    var worst = 0.0
                    for i in 0..<lg.count {
                        worst = max(worst, abs(Double(lg[i] - want[i]))
                                    / max(1.0, abs(Double(want[i]))))
                    }
                    gateWorst = max(gateWorst, worst)
                    if worst >= 1e-4 {
                        gateFail = String(format: "step %d outside 1e-4 (worst %.3e)",
                                          s, worst)
                        break
                    }
                    if s + 1 < steps {
                        _ = try runner.forward(tokenIds: [refIds[s]])
                    }
                }
            } catch { fail("generated replay failed: \(error)") }

            // Execution proof via the DEDICATED dispatch counters (they are
            // incremented in the dispatch path itself, so they are correct
            // in segment-batched mode too, where per-kernel GPU-time
            // attribution is deferred to the flush). A zero generated count
            // means the artifact never actually executed.
            let genCalls = runner.q4GeneratedDispatchCount
            let boundIncumbentCalls = runner.q4BoundTensorIncumbentDispatchCount
            if genCalls == 0 {
                runner.uninstallQ4GeneratedKernel()
                fail("generated kernel never dispatched for the bound tensor — not an execution of the artifact")
            }
            print("generated-kernel dispatches (bound tensor): \(genCalls); "
                  + "bound-tensor incumbent fallback dispatches: \(boundIncumbentCalls)")

            if let failMsg = gateFail {
                print("GATE_FAIL: \(failMsg) — incumbent remains the fallback")
                runner.uninstallQ4GeneratedKernel()
                fail("model-level gate rejected the generated lowering")
            }
            print(String(format: "GATE_PASS: %d steps, worst full-logit rel %.3e (strict 1e-4, unchanged)",
                         steps, gateWorst))

            // ---- matched timing: same prompt, same history replay,
            // sequential residency, N timed tokens per arm (generated
            // first while installed, then the incumbent). Dispatch-counter
            // deltas per arm prove which kernel served the bound tensor. ----
            guard steps > timedTokens + 1 else {
                runner.uninstallQ4GeneratedKernel()
                fail("insufficient gate steps (\(steps)) for \(timedTokens) timed tokens")
            }
            var armGeneratedDelta = 0
            func timedArm() throws -> (wallMed: Double, gpuMed: Double) {
                try resetIncumbentConfig()
                for t in history {
                    _ = try runner.forward(tokenIds: [t])
                }
                var idx = 0
                while idx < steps - timedTokens - 1 {
                    _ = try runner.forward(tokenIds: [refIds[idx]])
                    idx += 1
                }
                var walls = [Double]()
                var gpus = [Double]()
                for k in 0..<timedTokens {
                    let tok = refIds[min(idx + k, steps - 2)]
                    let gpuBefore = runner.totalGpuMicros
                    let genBefore = runner.q4GeneratedDispatchCount
                    let t0 = DispatchTime.now()
                    _ = try runner.forward(tokenIds: [tok])
                    walls.append(Double(DispatchTime.now().uptimeNanoseconds
                                        - t0.uptimeNanoseconds) / 1000)
                    gpus.append(runner.totalGpuMicros - gpuBefore)
                    armGeneratedDelta += runner.q4GeneratedDispatchCount - genBefore
                }
                let sortedW = walls.sorted()
                let sortedG = gpus.sorted()
                let n = sortedW.count
                let wm = n.isMultiple(of: 2)
                    ? (sortedW[n/2 - 1] + sortedW[n/2]) / 2 : sortedW[n/2]
                let gm = n.isMultiple(of: 2)
                    ? (sortedG[n/2 - 1] + sortedG[n/2]) / 2 : sortedG[n/2]
                return (wm, gm)
            }
            let genTiming = try timedArm()
            let genArmDispatches = armGeneratedDelta
            runner.uninstallQ4GeneratedKernel()
            armGeneratedDelta = 0
            let incTiming = try timedArm()
            let incArmDispatches = armGeneratedDelta
            print("timing-arm dispatch proof: generated arm bound-tensor "
                  + "generated-kernel dispatches \(genArmDispatches) "
                  + "(incumbent arm: \(incArmDispatches))")
            if genArmDispatches == 0 {
                fail("timing arm executed no generated dispatches — not a matched measurement")
            }
            print(String(format: "MATCHED_TIMING incumbent: wall %.0f us/token, gpu %.0f us/token",
                         incTiming.wallMed, incTiming.gpuMed))
            print(String(format: "MATCHED_TIMING generated: wall %.0f us/token, gpu %.0f us/token",
                         genTiming.wallMed, genTiming.gpuMed))
            print(String(format: "delta: generated vs incumbent wall %+.1f%%",
                         (genTiming.wallMed / incTiming.wallMed - 1) * 100))
            print("Q4GEN_GATE_COMPLETED")
        } catch {
            fail("gemma-q4gen-gate failed: \(error)")
        }

    case "gemma-rmsgen-gate":
        // Phase 10 model-level acceptance of the generated RMSNorm
        // lowering: teacher-forced same-history replay under the artifact's
        // emitted kernel for its bound (n, scaled) family, gating full
        // logits at the UNCHANGED strict 1e-4 relative formula + exact
        // greedy token equality + finiteness per step, then matched timing
        // with dispatch-counter proof. Fail closed — a rejected artifact
        // never executes and the incumbent stays the fallback.
        let path = requireValue("--file", args)
        let artifactPath = requireValue("--artifact", args)
        let genTokens = args.firstIndex(of: "--gen-tokens").map { _ in
            Int(requireValue("--gen-tokens", args)) ?? 0 } ?? 32
        guard genTokens >= 2, genTokens <= 64 else {
            fail("--gen-tokens must be 2...64")
        }
        let timedTokens = args.firstIndex(of: "--timed-tokens").map { _ in
            Int(requireValue("--timed-tokens", args)) ?? 0 } ?? 5
        guard timedTokens >= 1, timedTokens <= 32, timedTokens < genTokens else {
            fail("--timed-tokens must be 1...31 below --gen-tokens")
        }
        do {
            let fileURL = URL(fileURLWithPath: path)
            let tokenizer = try GemmaTokenizer(gguf: GGUFReader.read(fileURL))
            var history: [Int]
            if let tokensArg = args.firstIndex(of: "--tokens").map({ _ in
                requireValue("--tokens", args) }) {
                history = tokensArg.split(separator: ",").map {
                    guard let id = Int($0.trimmingCharacters(in: .whitespaces)),
                          id >= 0, id < tokenizer.vocabSize else {
                        fail("invalid token id '\($0)'")
                    }
                    return id
                }
            } else {
                history = try tokenizer.encode(requireValue("--text", args))
            }
            guard !history.isEmpty, history.count + genTokens <= 4096 else {
                fail("prompt + gen-tokens must fit the 4096 context")
            }
            print("prompt ids: \(history.map(String.init).joined(separator: " "))")
            let artifact = try RMSLoweringArtifact.load(
                from: URL(fileURLWithPath: artifactPath))
            let runner = try Gemma4Runner(fileAt: fileURL)

            // Reference arm: incumbent kernels, pass-7 selected
            // configuration, re-applied after every reset (reset-order
            // guard, fail closed).
            func resetIncumbentConfig() throws {
                runner.reset()
                runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
                guard runner.segmentBatchingActive else {
                    throw GemmaForgeError.executionFailed(
                        "segment batching failed to re-engage after reset")
                }
            }
            try resetIncumbentConfig()
            var refIds = [Int]()
            var refLogits = [[Float]]()
            do {
                let prefill = try runner.forward(tokenIds: history)
                refLogits.append(try runner.currentLogits())
                refIds.append(prefill.argmaxToken)
                var prev = prefill.argmaxToken
                while refIds.count < genTokens {
                    if tokenizer.isEndOfGeneration(prev) { break }
                    let ev = try runner.forward(tokenIds: [prev])
                    refLogits.append(try runner.currentLogits())
                    refIds.append(ev.argmaxToken)
                    prev = ev.argmaxToken
                }
            } catch { fail("reference pass failed: \(error)") }
            let steps = refLogits.count
            print("reference greedy ids (\(steps)): \(refIds.map(String.init).joined(separator: " "))")

            let ggufDigest = try cachedModelDigest(fileURL)
            do {
                try runner.installRmsGeneratedKernel(artifact: artifact,
                                                     ggufDigest: ggufDigest)
            } catch {
                fail("artifact rejected at install (incumbent stays active): \(error)")
            }
            print("generated kernel installed: \(artifact.schedule.kernelName) "
                  + "for n=\(artifact.op.n), scaled=\(artifact.op.scaled) "
                  + "(weight \(artifact.op.tensor ?? "<unweighted>"))")

            var gateWorst = 0.0
            var gateFail: String? = nil
            do {
                try resetIncumbentConfig()
                for t in history {
                    _ = try runner.forward(tokenIds: [t])
                }
                for s in 0..<steps {
                    let lg = try runner.currentLogits()
                    let want = refLogits[s]
                    guard lg.count == want.count,
                          lg.allSatisfy({ $0.isFinite }),
                          want.allSatisfy({ $0.isFinite }) else {
                        gateFail = "step \(s): shape/nonfinite violation"
                        break
                    }
                    let lgArgmax = lg.enumerated().max { $0.element < $1.element }!.offset
                    if lgArgmax != refIds[s] {
                        gateFail = "step \(s): argmax \(lgArgmax) != reference \(refIds[s])"
                        break
                    }
                    var worst = 0.0
                    for i in 0..<lg.count {
                        worst = max(worst, abs(Double(lg[i] - want[i]))
                                    / max(1.0, abs(Double(want[i]))))
                    }
                    gateWorst = max(gateWorst, worst)
                    if worst >= 1e-4 {
                        gateFail = String(format: "step %d outside 1e-4 (worst %.3e)",
                                          s, worst)
                        break
                    }
                    if s + 1 < steps {
                        _ = try runner.forward(tokenIds: [refIds[s]])
                    }
                }
            } catch { fail("generated replay failed: \(error)") }

            let genCalls = runner.rmsGeneratedDispatchCount
            if genCalls == 0 {
                runner.uninstallRmsGeneratedKernel()
                fail("generated kernel never dispatched for the bound shape family — not an execution of the artifact")
            }
            print("generated-kernel dispatches (bound shape family): \(genCalls)")

            if let failMsg = gateFail {
                print("GATE_FAIL: \(failMsg) — incumbent remains the fallback")
                runner.uninstallRmsGeneratedKernel()
                fail("model-level gate rejected the generated lowering")
            }
            print(String(format: "GATE_PASS: %d steps, worst full-logit rel %.3e (strict 1e-4, unchanged)",
                         steps, gateWorst))

            guard steps > timedTokens + 1 else {
                runner.uninstallRmsGeneratedKernel()
                fail("insufficient gate steps (\(steps)) for \(timedTokens) timed tokens")
            }
            var armGeneratedDelta = 0
            func timedArm() throws -> (wallMed: Double, gpuMed: Double) {
                try resetIncumbentConfig()
                for t in history {
                    _ = try runner.forward(tokenIds: [t])
                }
                var idx = 0
                while idx < steps - timedTokens - 1 {
                    _ = try runner.forward(tokenIds: [refIds[idx]])
                    idx += 1
                }
                var walls = [Double]()
                var gpus = [Double]()
                for k in 0..<timedTokens {
                    let tok = refIds[min(idx + k, steps - 2)]
                    let gpuBefore = runner.totalGpuMicros
                    let genBefore = runner.rmsGeneratedDispatchCount
                    let t0 = DispatchTime.now()
                    _ = try runner.forward(tokenIds: [tok])
                    walls.append(Double(DispatchTime.now().uptimeNanoseconds
                                        - t0.uptimeNanoseconds) / 1000)
                    gpus.append(runner.totalGpuMicros - gpuBefore)
                    armGeneratedDelta += runner.rmsGeneratedDispatchCount - genBefore
                }
                let sortedW = walls.sorted()
                let sortedG = gpus.sorted()
                let n = sortedW.count
                let wm = n.isMultiple(of: 2)
                    ? (sortedW[n/2 - 1] + sortedW[n/2]) / 2 : sortedW[n/2]
                let gm = n.isMultiple(of: 2)
                    ? (sortedG[n/2 - 1] + sortedG[n/2]) / 2 : sortedG[n/2]
                return (wm, gm)
            }
            let genTiming = try timedArm()
            let genArmDispatches = armGeneratedDelta
            runner.uninstallRmsGeneratedKernel()
            armGeneratedDelta = 0
            let incTiming = try timedArm()
            let incArmDispatches = armGeneratedDelta
            print("timing-arm dispatch proof: generated arm dispatches \(genArmDispatches) "
                  + "(incumbent arm: \(incArmDispatches))")
            if genArmDispatches == 0 {
                fail("timing arm executed no generated dispatches — not a matched measurement")
            }
            print(String(format: "MATCHED_TIMING incumbent: wall %.0f us/token, gpu %.0f us/token",
                         incTiming.wallMed, incTiming.gpuMed))
            print(String(format: "MATCHED_TIMING generated: wall %.0f us/token, gpu %.0f us/token",
                         genTiming.wallMed, genTiming.gpuMed))
            print(String(format: "delta: generated vs incumbent wall %+.1f%%",
                         (genTiming.wallMed / incTiming.wallMed - 1) * 100))
            print("RMSGEN_GATE_COMPLETED")
        } catch {
            fail("gemma-rmsgen-gate failed: \(error)")
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
        // Paired composite flags (both or neither), parsed BEFORE any GPU
        // allocation; incompatible combinations are rejected here, never
        // after the runner exists.
        let compositeOpts = parseCompositeOptions(args)
        if compositeOpts != nil {
            guard !args.contains("--expert-cache-budget") else {
                fail("--composite-plan/--moe-region require resident weights and cannot "
                      + "be combined with --expert-cache-budget (streaming expert cache)")
            }
            // The composite artifact's 3x32 gate record and this forge's
            // full-workload gate must both run at the full 32-token length.
            guard genTokens == 32 else {
                fail("--composite-plan requires --gen-tokens 32 (full gated workload)")
            }
        }
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
            // Bounded, IR-validated region read BEFORE the runner (GPU) is
            // allocated — an oversized/empty/invalid region fails closed here.
            var compositeRegion: Data? = nil
            if let comp = compositeOpts {
                let regionJSON = try readMoERegionBounded(comp.moeRegionPath)
                _ = try MoERegionIR.parse(regionJSON)
                compositeRegion = regionJSON
            }

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
            // Composite Q4 launch + MoE lowering via the shared production
            // helper: existing Q4LaunchSelect workflow (hardware-derived
            // width + E4 composite), PUBLIC atomic install, invalid artifact
            // fails closed, no-win leaves the incumbent. Invoked AFTER the
            // runner exists and the cached model digest is resolved, but BEFORE the
            // engine digest / hardware profile / Forge plan-cache identity
            // and selection — the installed composite state is part of what
            // the forge then measures and gates.
            let ggufDigest = try cachedModelDigest(fileURL)
            if let comp = compositeOpts, let regionJSON = compositeRegion {
                let compTokenizer = try GemmaTokenizer(gguf: preFile)
                let result = try GemmaCompositeForge.prepare(
                    regionJSON: regionJSON, ggufURL: fileURL, graph: graph,
                    runner: runner, ggufDigest: ggufDigest,
                    artifactURL: URL(fileURLWithPath: comp.compositePlanPath),
                    fixtures: try canonicalCompositeFixtures(tokenizer: compTokenizer),
                    isEndOfGeneration: { compTokenizer.isEndOfGeneration($0) },
                    allowSelection: true)
                for line in result.log { print(line) }
                print("composite: enabled, installed \(result.installed), "
                      + "cache hit \(result.cacheHit), artifact digest "
                      + result.artifactDigest.prefix(16) + "…")
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
                print(String(format: "  candidate batching=%d tiledQ4=%d tiledQ6=%d rmsTiled=%d f32Tiled=%d vecLoads=%d: correct %d, wall %@/token, gpu %@/token, %@ kernels/token, %@ buffers/token — %@",
                             r.candidate.segmentBatching ? 1 : 0,
                             r.candidate.matvecTiledQ4 ? 1 : 0,
                             r.candidate.matvecTiledQ6 ? 1 : 0,
                             r.candidate.rmsnormTiled ? 1 : 0,
                             r.candidate.matvecTiledF32 ? 1 : 0,
                             r.candidate.vectorizedLoads ? 1 : 0,
                             r.correct ? 1 : 0,
                             r.medianWallMicrosPerToken.map { String(format: "%.0f us", $0) } ?? "-",
                             r.medianGpuMicrosPerToken.map { String(format: "%.0f us", $0) } ?? "-",
                             r.kernelsPerToken.map { String(format: "%.0f", $0) } ?? "-",
                             r.buffersPerToken.map { String(format: "%.0f", $0) } ?? "-",
                             r.note))
            }
            print(String(format: "selected: batching=%d tiledQ4=%d tiledQ6=%d rmsTiled=%d f32Tiled=%d vecLoads=%d (tile widths q4=%d q6=%d rms=%d f32=%d), wall %.0f us/token, gpu %.0f us/token",
                         plan.selected.segmentBatching ? 1 : 0,
                         plan.selected.matvecTiledQ4 ? 1 : 0,
                         plan.selected.matvecTiledQ6 ? 1 : 0,
                         plan.selected.rmsnormTiled ? 1 : 0,
                         plan.selected.matvecTiledF32 ? 1 : 0,
                         plan.selected.vectorizedLoads ? 1 : 0,
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
            // FULL-WORKLOAD LEGAL-CANDIDATE FALLBACK: candidates are
            // tried in measured wall-median order (fastest first). The
            // first candidate that passes prefill + EVERY step of the
            // requested --gen-tokens workload at the strict gate becomes
            // the plan's selection (the plan is reselected and re-saved);
            // a faster candidate that fails is reported with its measured
            // worst relative difference and rejected. If NO legal
            // candidate passes, the forge fails — nothing is relaxed.
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
            func worstRel(_ lg: [Float], _ want: [Float]) -> Double {
                guard lg.count == want.count else { return .infinity }
                var worst = 0.0
                for (a, b) in zip(lg, want) {
                    let rel = abs(Double(a - b)) / max(1.0, abs(Double(b)))
                    if rel > worst { worst = rel }
                }
                return worst
            }
            // Ordered try list: correct results by wall median, the
            // current selection first (deduplicated).
            var medianByCandidate: [GemmaForgeCandidate: Double] = [:]
            for r in plan.results where r.correct {
                if let w = r.medianWallMicrosPerToken {
                    medianByCandidate[r.candidate] = w
                }
            }
            // The selection is always tried first; the rest follow by
            // measured wall median with the LEGAL ENUMERATION INDEX as a
            // deterministic tie-break (dictionary iteration order is not
            // stable). No comparator ever sees the selected candidate, so
            // strict weak ordering holds.
            let legalOrder = GemmaForge.candidates()
            func legalIndex(_ c: GemmaForgeCandidate) -> Int {
                legalOrder.firstIndex(of: c) ?? Int.max
            }
            let others = medianByCandidate.keys.filter { $0 != plan.selected }
                .sorted {
                    let a = medianByCandidate[$0] ?? .infinity
                    let b = medianByCandidate[$1] ?? .infinity
                    if a != b { return a < b }
                    return legalIndex($0) < legalIndex($1)
                }
            let tryOrder = [plan.selected] + others
            var selected = plan.selected
            var selectedMedianWall = 0.0
            var selectedGpuSum = 0.0
            var selectedSteps = 0
            var selectedPassed = false
            var fullGateRejections: [String] = []
            for candidate in tryOrder {
                runner.reset()
                let (graphArgmax, graphPrefillLogits, pgpu, _, _) =
                    try GemmaForge.executeGraph(
                        graph, runner: runner, tokenIds: prompt,
                        candidate: candidate, schedule: plan.schedule)
                var rejected: String? = nil
                if graphArgmax != prefill.argmaxToken {
                    rejected = "prefill argmax mismatch"
                } else if graphPrefillLogits.count != prefillLogits.count {
                    rejected = "prefill logit count mismatch"
                } else if !(graphPrefillLogits.allSatisfy { $0.isFinite }
                            && prefillLogits.allSatisfy { $0.isFinite }) {
                    rejected = "prefill non-finite logits"
                } else if worstRel(graphPrefillLogits, prefillLogits) >= 1e-4 {
                    rejected = String(format: "prefill worst rel %.3e >= 1e-4",
                                      worstRel(graphPrefillLogits, prefillLogits))
                }
                var gen = [graphArgmax]
                var wallSamples: [Double] = []
                var gpuSum = pgpu
                if rejected == nil {
                    for step in 1..<genTokens {
                        if tokenizer.isEndOfGeneration(gen.last!) {
                            print("eos at step \(step): token \(gen.last!) — stopping early")
                            refSeq = Array(refSeq.prefix(step))
                            refStepLogits = Array(refStepLogits.prefix(step - 1))
                            break
                        }
                        let tok = refSeq[step - 1]
                        let t0 = DispatchTime.now()
                        let (next, lg, gpu, kernels, buffers) =
                            try GemmaForge.executeGraphContinue(
                                graph, runner: runner, token: tok,
                                candidate: candidate, schedule: plan.schedule)
                        let wall = Double(DispatchTime.now().uptimeNanoseconds
                                          - t0.uptimeNanoseconds) / 1000
                        let want = refStepLogits[step - 1]
                        if next != refSeq[step] {
                            rejected = String(format: "step %d token mismatch (%d != %d)",
                                              step, next, refSeq[step])
                        } else if lg.count != want.count {
                            rejected = String(format: "step %d logit count mismatch", step)
                        } else if !(lg.allSatisfy { $0.isFinite }
                                    && want.allSatisfy { $0.isFinite }) {
                            rejected = String(format: "step %d non-finite logits", step)
                        } else if worstRel(lg, want) >= 1e-4 {
                            rejected = String(format: "step %d worst rel %.3e >= 1e-4",
                                              step, worstRel(lg, want))
                        }
                        if rejected != nil { break }
                        gen.append(next)
                        wallSamples.append(wall)
                        gpuSum += gpu
                        print(String(format: "  step %d: token %d, wall %.0f us, gpu %.0f us, %d kernels, %d buffers",
                                     step, next, wall, gpu, kernels, buffers))
                    }
                }
                if let why = rejected {
                    let note = "candidate \(candidate) rejected: \(why)"
                    fullGateRejections.append(note)
                    print("candidate \(candidate) REJECTED by full \(genTokens)-token gate: \(why)")
                    continue
                }
                selected = candidate
                selectedGpuSum = gpuSum
                selectedSteps = wallSamples.count
                if wallSamples.isEmpty {
                    // EOS on the first generated token: no continuation
                    // was timed; the median is undefined and reported as
                    // 0 with the step count disclosing it.
                    selectedMedianWall = 0
                } else {
                    let sortedWall = wallSamples.sorted()
                    let mid = sortedWall.count / 2
                    selectedMedianWall = sortedWall.count % 2 == 1
                        ? sortedWall[mid]
                        : (sortedWall[mid - 1] + sortedWall[mid]) / 2
                }
                selectedPassed = true
                break
            }
            guard selectedPassed else {
                fail("no legal candidate passed the full \(genTokens)-token gate")
            }
            if selected != plan.selected {
                print(String(format: "full-workload fallback: reselecting from %@ (failed the %d-token gate) to %@",
                             String(describing: plan.selected), genTokens,
                             String(describing: selected)))
            }
            // Persist the full-gate record (and any reselection) so plan
            // reuse cannot silently forget that faster candidates were
            // measured and rejected at this gate length. Same-length
            // re-gating merges with the prior history inside reselect
            // (a cached plan's already-passing selection can produce an
            // empty list this run); a different length resets it.
            plan = try GemmaForge.reselect(plan, to: selected,
                                           fullGateTokens: genTokens,
                                           fullGateRejections: fullGateRejections)
            try GemmaForge.savePlan(plan, to: planURL)
            print("plan saved with full-gate record: \(planPath), digest \(plan.planDigest)")
            print("prompt ids: \(prompt.map(String.init).joined(separator: " "))")
            print("greedy ids (reference == graph): \(refSeq.map(String.init).joined(separator: " "))")
            print("text: \(try tokenizer.decode(refSeq))")
            print(String(format: "verified generation: %d steps, median wall %.0f us/token, gpu total %.1f ms",
                         selectedSteps, selectedMedianWall, selectedGpuSum / 1000))
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
            let ggufDigest = try cachedModelDigest(fileURL)
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

    case "model-digest":
        // Independent inspection of the shared model digest cache; no GPU
        // and no runner. Without --cache-only a cache miss hashes the file
        // once (initial computation); --cache-only never reads model bytes
        // and reports a missing record as missing. With --rehash-model the
        // refresh above already ran and printed its result.
        let digestURL = URL(fileURLWithPath: requireValue("--file", args))
        do {
            if args.contains("--cache-only") {
                guard let record = try ModelDigestCache.shared.cachedRecord(for: digestURL) else {
                    fail("no cached digest for " + ModelDigestCache.canonicalPath(for: digestURL)
                         + "; run model-digest without --cache-only or with --rehash-model")
                }
                print(ModelDigestLookup(record: record, origin: .cacheHit).disclosure)
            } else if !args.contains("--rehash-model") {
                print(try ModelDigestCache.shared.digest(for: digestURL).disclosure)
            }
            print("cache record: " + ModelDigestCache.shared.recordURL(for: digestURL).path)
        } catch {
            fail("model-digest failed: " + String(describing: error))
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
            if args.contains("--vectorized-loads") { runner.vectorizedLoads = true }
            let flagText = "q4Tiled=\(runner.q4TiledKernel) "
                + "q6Tiled=\(runner.q6TiledKernel) "
                + "rmsTiled=\(runner.rmsTiledKernel) "
                + "f32Tiled=\(runner.f32TiledKernel) "
                + "rmsVL=\(runner.vectorizedLoads)"
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
