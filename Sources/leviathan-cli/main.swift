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
      run --artifact <file.lvi> [--batch N] [--seed N]
          Load and integrity-check an artifact, run it on the Accelerate
          backend with a deterministic seeded input, print the output.
      check
          Compile the built-in synthetic demo graph in memory, run both the
          Accelerate and scalar reference backends, and compare them.
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
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
        return defaultValue
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
    guard batch >= 1, batch <= Limits.maxBatch else {
        fail("batch must be between 1 and \(Limits.maxBatch), got \(batch)")
    }
    do {
        let artifact = try CompiledArtifact.load(from: URL(fileURLWithPath: artifactPath))
        print("artifact OK, digest \(artifact.digest.prefix(16))…")
        let input = demoInput(dim: artifact.header.inputDim, batch: batch, seed: seed)
        let start = ContinuousClock.now
        let output = try AccelerateBackend.run(artifact.program, batch: batch, input: input)
        let elapsed = ContinuousClock.now - start
        print(String(format: "output[%d]: %@", output.count,
                     output.map { String(format: "%.6f", $0) }.joined(separator: " ")))
        print(String(format: "ran %d step(s), batch %d, in %.3f ms",
                     artifact.header.steps.count, batch,
                     Double(elapsed.components.attoseconds) / 1e15 + Double(elapsed.components.seconds) * 1e3))
    } catch {
        fail("run failed: \(error)")
    }

case "check":
    do {
        _ = try LeviathanCore.validate(demoGraph)
        let artifact = try CompiledArtifact(graph: demoGraph, seed: 42)
        let batch = 4
        let input = demoInput(dim: demoGraph.inputDim, batch: batch, seed: 7)

        let fastStart = ContinuousClock.now
        let fast = try AccelerateBackend.run(artifact.program, batch: batch, input: input)
        let fastElapsed = ContinuousClock.now - fastStart
        let refStart = ContinuousClock.now
        let reference = try ScalarBackend.run(artifact.program, batch: batch, input: input)
        let refElapsed = ContinuousClock.now - refStart

        guard let diff = maxAbsDiff(fast, reference), diff < 1e-4 else {
            fail("backends disagree: maxAbsDiff out of tolerance")
        }
        print("digest: \(artifact.digest)")
        print(String(format: "max |accelerate - scalar| = %.3e (tolerance 1e-4)", diff))
        print(String(format: "accelerate: %.3f ms, scalar: %.3f ms, batch %d, %d ops",
                     Double(fastElapsed.components.attoseconds) / 1e15 + Double(fastElapsed.components.seconds) * 1e3,
                     Double(refElapsed.components.attoseconds) / 1e15 + Double(refElapsed.components.seconds) * 1e3,
                     batch, demoGraph.ops.count))
        print("PASS")
    } catch {
        fail("check failed: \(error)")
    }

default:
    usage()
}
