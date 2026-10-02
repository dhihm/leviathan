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

default:
    usage()
}
