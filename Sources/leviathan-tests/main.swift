import Foundation
import LeviathanCore
import Metal

// Minimal original test harness. The XCTest and swift-testing modules are
// not available with the active CommandLineTools toolchain, so this
// executable implements the checks directly with the standard library and
// Foundation, and exits nonzero when any check fails.
// Run with: swift run leviathan-tests

var passed = 0
var failed = 0

@MainActor
func check(_ name: String, _ condition: @autoclosure () -> Bool) {
    if condition() { passed += 1 } else { failed += 1; print("FAIL: \(name)") }
}

@MainActor
func checkEqual<T: Equatable>(_ name: String, _ actual: T, _ expected: T) {
    if actual == expected {
        passed += 1
    } else {
        failed += 1
        print("FAIL: \(name) — expected \(expected), got \(actual)")
    }
}

@MainActor
func checkThrows<E: Error & Equatable>(_ name: String, _ expected: E,
                                       _ body: () throws -> Void) {
    do {
        try body()
        failed += 1; print("FAIL: \(name) — expected \(expected), got no error")
    } catch let error as E {
        if error == expected { passed += 1 }
        else { failed += 1; print("FAIL: \(name) — expected \(expected), got \(error)") }
    } catch {
        failed += 1; print("FAIL: \(name) — expected \(expected), got unexpected \(error)")
    }
}

let demoGraph = GraphIR(inputDim: 16, ops: [
    .dense(outDim: 32), .relu, .dense(outDim: 32), .relu, .dense(outDim: 8),
])

func seededInput(dim: Int, batch: Int, seed: UInt64) -> [Float] {
    var rng = SplitMix64(seed: seed)
    return (0..<(batch * dim)).map { _ in rng.nextUniform() }
}

// MARK: - Graph validation

do {
    let dims = try validate(demoGraph)
    checkEqual("validation accepts demo graph", dims, [16, 32, 32, 32, 32, 8])
} catch {
    failed += 1; print("FAIL: validation accepts demo graph — threw \(error)")
}
checkThrows("empty graph rejected", ValidationError.emptyGraph) {
    _ = try validate(GraphIR(inputDim: 4, ops: []))
}
checkThrows("zero inputDim rejected",
            ValidationError.nonPositiveDimension("inputDim")) {
    _ = try validate(GraphIR(inputDim: 0, ops: [.relu]))
}
checkThrows("oversized dim rejected",
            ValidationError.dimensionTooLarge("inputDim", 8192, Limits.maxDim)) {
    _ = try validate(GraphIR(inputDim: 8192, ops: [.relu]))
}
// Dims are individually within maxDim but the 4096x4096 weight matrix
// exceeds the tensor element budget.
checkThrows("weight tensor budget enforced",
            ValidationError.tensorTooLarge(4096 * 4096, Limits.maxTensorElements)) {
    _ = try validate(GraphIR(inputDim: 4096, ops: [.dense(outDim: 4096)]))
}

// MARK: - Determinism and round-trip

do {
    let a = try CompiledArtifact(graph: demoGraph, seed: 42)
    let b = try CompiledArtifact(graph: demoGraph, seed: 42)
    checkEqual("artifact bytes deterministic", a.serialized(), b.serialized())
    checkEqual("artifact digest deterministic", a.digest, b.digest)
    let c = try CompiledArtifact(graph: demoGraph, seed: 43)
    check("different seed changes digest", a.digest != c.digest)
    check("different seed changes weights", a.weights != c.weights)

    let reloaded = try CompiledArtifact(bytes: a.serialized())
    checkEqual("header round-trips", reloaded.header, a.header)
    checkEqual("weights round-trip", reloaded.weights, a.weights)
    checkEqual("round-trip is byte-exact fixed point", reloaded.serialized(), a.serialized())
} catch {
    failed += 1; print("FAIL: determinism/round-trip — threw \(error)")
}

checkThrows("per-tensor budget enforced at compile",
            ValidationError.tensorTooLarge(4096 * 4096, Limits.maxTensorElements)) {
    _ = try CompiledArtifact(
        graph: GraphIR(inputDim: 4096, ops: [.dense(outDim: 4096)]), seed: 1)
}
// Each 1000x1000 weight matrix passes the per-tensor budget, but eight of
// them exceed the cumulative 16 MiB weight byte cap at synthesis time.
checkThrows("cumulative weight byte cap enforced",
            ArtifactError.exceedsWeightByteLimit(8_008_000 * 4)) {
    _ = try CompiledArtifact(
        graph: GraphIR(inputDim: 1000,
                       ops: (0..<8).map { _ in GraphOp.dense(outDim: 1000) }),
        seed: 1)
}

// MARK: - Integrity

do {
    var bytes = try CompiledArtifact(graph: demoGraph, seed: 42).serialized()
    bytes[bytes.count - 40] ^= 0x01  // flip one weight bit
    checkThrows("tampered weights rejected", ArtifactError.digestMismatch) {
        _ = try CompiledArtifact(bytes: bytes)
    }
    checkThrows("truncated artifact rejected", ArtifactError.truncated) {
        _ = try CompiledArtifact(bytes: Array(bytes[0..<10]))
    }
} catch {
    failed += 1; print("FAIL: integrity — threw \(error)")
}

// MARK: - Malformed but correctly checksummed artifacts

/// Rebuilds artifact bytes with a mutated header, recomputing the digest so
/// the checksum passes and only semantic validation can reject.
func reheadered(_ original: CompiledArtifact,
                mutate: (inout ArtifactHeader) -> Void) throws -> [UInt8] {
    var header = original.header
    mutate(&header)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let headerData = try encoder.encode(header)
    var out = Array(CompiledArtifact.magic.utf8)
    out.append(contentsOf: UInt32(headerData.count).littleEndianBytes)
    out.append(contentsOf: Array(headerData))
    out.append(contentsOf: original.weights.flatMap { $0.bitPattern.littleEndianBytes })
    out.append(contentsOf: SHA256.hash(out))
    return out
}

do {
    let good = try CompiledArtifact(graph: demoGraph, seed: 42)

    checkThrows("wrong outputDim rejected",
                ArtifactError.badPlan("declared outputDim does not match plan")) {
        _ = try CompiledArtifact(bytes: reheadered(good) { $0.outputDim = 99 })
    }
    checkThrows("bad buffer count rejected",
                ArtifactError.badPlan("step 0: buffer index out of range")) {
        _ = try CompiledArtifact(bytes: reheadered(good) { $0.bufferCount = 1 })
    }
    checkThrows("negative weightCount rejected cleanly",
                ArtifactError.exceedsWeightByteLimit(0)) {
        _ = try CompiledArtifact(bytes: reheadered(good) { $0.weightCount = -5 })
    }
    checkThrows("Int.max weightCount rejected cleanly",
                ArtifactError.exceedsWeightByteLimit(Int.max)) {
        _ = try CompiledArtifact(bytes: reheadered(good) { $0.weightCount = Int.max })
    }
    checkThrows("empty step list rejected",
                ArtifactError.badPlan("step count 0 out of range")) {
        _ = try CompiledArtifact(bytes: reheadered(good) { $0.steps = [] })
    }
    checkThrows("out-of-range weight offset rejected",
                ArtifactError.badPlan("step 0: weight offsets out of range")) {
        _ = try CompiledArtifact(bytes: reheadered(good) { header in
            header.steps = header.steps.enumerated().map { index, step in
                guard index == 0,
                      case let .dense(i, inDim, outDim, inBuf, outBuf, _, wCount, bOff) = step
                else { return step }
                return .dense(index: i, inputDim: inDim, outputDim: outDim,
                              inputBuffer: inBuf, outputBuffer: outBuf,
                              weightOffset: good.weights.count - 1,
                              weightCount: wCount, biasOffset: bOff)
            }
        })
    }
    do {
        _ = try CompiledArtifact(bytes: reheadered(good) { _ in () })
        passed += 1
    } catch {
        failed += 1; print("FAIL: unmodified re-header loads — threw \(error)")
    }
} catch {
    failed += 1; print("FAIL: malformed headers — threw \(error)")
}

// MARK: - Non-finite weights with matching digest

do {
    let good = try CompiledArtifact(graph: demoGraph, seed: 42)
    var bytes = good.serialized()
    let weightStart = bytes.count - 32 - good.weights.count * 4
    for (i, b) in [0x00, 0x00, 0xc0, 0x7f].enumerated() {  // NaN, little-endian
        bytes[weightStart + i] = UInt8(b)
    }
    bytes = Array(bytes[..<(bytes.count - 32)])
        + SHA256.hash(Array(bytes[..<(bytes.count - 32)]))
    checkThrows("NaN weight with valid checksum rejected",
                ArtifactError.nonFiniteWeight(0)) {
        _ = try CompiledArtifact(bytes: bytes)
    }
} catch {
    failed += 1; print("FAIL: NaN weight — threw \(error)")
}

// MARK: - SHA-256

checkEqual("sha256('abc')", SHA256.hex(SHA256.hash(Array("abc".utf8))),
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
checkEqual("sha256('')", SHA256.hex(SHA256.hash([])),
           "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
checkEqual("sha256(1000 x 'a')",
           SHA256.hex(SHA256.hash(Array(repeating: UInt8(ascii: "a"), count: 1_000))),
           "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3")

// MARK: - Execution backends

do {
    _ = try validate(demoGraph)
    let artifact = try CompiledArtifact(graph: demoGraph, seed: 42)
    let input = seededInput(dim: 16, batch: 8, seed: 7)
    let fast = try AccelerateBackend.run(artifact.program, batch: 8, input: input)
    let reference = try ScalarBackend.run(artifact.program, batch: 8, input: input)
    if let diff = maxAbsDiff(fast, reference) {
        check("accelerate matches scalar within 1e-4", diff < 1e-4)
    } else {
        failed += 1; print("FAIL: backend comparison — maxAbsDiff rejected input")
    }
} catch {
    failed += 1; print("FAIL: backend comparison — threw \(error)")
}

// Hand-computed dense + ReLU: W = [[1,2],[3,4]], b = [-10, 0.5], x = [1, 1]
// dense: [1*1+1*3-10, 1*2+1*4+0.5] = [-6, 6.5]; relu: [0, 6.5]
do {
    let graph = GraphIR(inputDim: 2, ops: [.dense(outDim: 2), .relu])
    _ = try validate(graph)
    let lowered = LoweredProgram(graph: graph, seed: 1)
    let program = LoweredProgram(
        inputDim: 2, outputDim: 2, bufferCount: 2,
        steps: lowered.steps, weights: [1, 2, 3, 4, -10, 0.5])
    checkEqual("scalar hand-computed dense+relu",
               try ScalarBackend.run(program, batch: 1, input: [1, 1]), [0, 6.5])
    checkEqual("accelerate hand-computed dense+relu",
               try AccelerateBackend.run(program, batch: 1, input: [1, 1]), [0, 6.5])
} catch {
    failed += 1; print("FAIL: hand-computed dense+relu — threw \(error)")
}

do {
    let artifact = try CompiledArtifact(graph: demoGraph, seed: 42)
    checkThrows("oversized batch rejected",
                ExecutionError.batchTooLarge(512, Limits.maxBatch)) {
        _ = try ScalarBackend.run(artifact.program, batch: 512, input: [])
    }
    checkThrows("wrong input count rejected",
                ExecutionError.wrongInputCount(3, 16)) {
        _ = try ScalarBackend.run(artifact.program, batch: 1, input: [0, 0, 0])
    }
} catch {
    failed += 1; print("FAIL: run-time limits — threw \(error)")
}

check("maxAbsDiff rejects NaN", maxAbsDiff([Float.nan], [1]) == nil)
check("maxAbsDiff rejects infinity", maxAbsDiff([1], [.infinity]) == nil)
check("maxAbsDiff rejects length mismatch", maxAbsDiff([1, 2], [1]) == nil)
checkEqual("maxAbsDiff computes difference", maxAbsDiff([1, -2], [0.5, -2]) ?? -1, 0.5)

// MARK: - Metal GPU backend

// GPU tests require a real Metal device; any unavailability, shader-compile
// failure, dispatch failure, or command-buffer error must surface as an
// explicit thrown error — never a silent CPU fallback.
do {
    check("metal device available", MTLCreateSystemDefaultDevice() != nil)

    // Hand-computed dense + ReLU on the GPU: W = [[1,2],[3,4]], b = [-10, 0.5],
    // x = [1, 1] → dense [-6, 6.5] → relu [0, 6.5].
    let handGraph = GraphIR(inputDim: 2, ops: [.dense(outDim: 2), .relu])
    _ = try validate(handGraph)
    let handLowered = LoweredProgram(graph: handGraph, seed: 1)
    let handProgram = LoweredProgram(
        inputDim: 2, outputDim: 2, bufferCount: 2,
        steps: handLowered.steps, weights: [1, 2, 3, 4, -10, 0.5])
    let (handOut, handEvidence) = try MetalBackend.runReporting(
        handProgram, batch: 1, input: [1, 1])
    checkEqual("metal hand-computed dense+relu", handOut, [0, 6.5])
    check("metal evidence: device named", !handEvidence.deviceName.isEmpty)
    print("metal device: \(handEvidence.deviceName)")
    check("metal evidence: shader compiled at runtime",
          handEvidence.shaderLibraryCompiled)
    checkEqual("metal evidence: command buffer completed",
               handEvidence.commandStatus, "completed")
    checkEqual("metal evidence: steps dispatched", handEvidence.dispatchedSteps, 2)
    checkEqual("metal evidence: values read back", handEvidence.readbackElements, 2)

    // Hand-computed batch-2 case with a negative and a positive row.
    // x = [[1, 1], [0, 2]] → dense rows [-6, 6.5] and [-4, 8.5] → relu [0, 6.5], [0, 8.5]
    checkEqual("metal hand-computed dense+relu batch 2",
               try MetalBackend.run(handProgram, batch: 2, input: [1, 1, 0, 2]),
               [0, 6.5, 0, 8.5])

    // Dense-only, no ReLU: negative values must pass through untouched.
    // W = [[1,2],[3,4]], b = [-10, 0.5], x = [1, 1] → [-6, 6.5]
    let denseOnly = GraphIR(inputDim: 2, ops: [.dense(outDim: 2)])
    _ = try validate(denseOnly)
    let denseLowered = LoweredProgram(graph: denseOnly, seed: 1)
    let denseProgram = LoweredProgram(
        inputDim: 2, outputDim: 2, bufferCount: 2,
        steps: denseLowered.steps, weights: [1, 2, 3, 4, -10, 0.5])
    checkEqual("metal dense-only passes negatives through",
               try MetalBackend.run(denseProgram, batch: 1, input: [1, 1]), [-6, 6.5])

    // ReLU-only graph: no dense op, empty weights blob (zero-length buffer),
    // single buffer, odd input dim and odd batch (non-power-of-two shape).
    let reluOnly = LoweredProgram(
        inputDim: 5, outputDim: 5, bufferCount: 1,
        steps: [.relu(index: 0, buffer: 0)], weights: [])
    let reluIn: [Float] = [-3, -1.5, 0, 2.25, 7, -8, 0.5, -0.25, 9, -2]
    let reluWant: [Float] = [0, 0, 0, 2.25, 7, 0, 0.5, 0, 9, 0]
    checkEqual("metal relu-only, zero weights, odd dims, batch 2",
               try MetalBackend.run(reluOnly, batch: 2, input: reluIn), reluWant)

    // Odd width graph end to end: 3 → dense(7) → relu, batch 3.
    let oddGraph = GraphIR(inputDim: 3, ops: [.dense(outDim: 7), .relu])
    _ = try validate(oddGraph)
    let oddArtifact = try CompiledArtifact(graph: oddGraph, seed: 5)
    let oddInput = seededInput(dim: 3, batch: 3, seed: 11)
    let oddGPU = try MetalBackend.run(oddArtifact.program, batch: 3, input: oddInput)
    let oddRef = try ScalarBackend.run(oddArtifact.program, batch: 3, input: oddInput)
    if let diff = maxAbsDiff(oddGPU, oddRef) {
        check("metal odd-width graph matches scalar within 1e-4", diff < 1e-4)
    } else {
        failed += 1
        print("FAIL: metal odd-width graph — maxAbsDiff rejected input")
    }

    // Malformed raw plans must be rejected before any GPU access.
    checkThrows("metal rejects out-of-range buffer index",
                ArtifactError.badPlan("step 0: buffer index out of range")) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 1,
            steps: [.dense(index: 0, inputDim: 2, outputDim: 2,
                           inputBuffer: 0, outputBuffer: 5,
                           weightOffset: 0, weightCount: 4, biasOffset: 4)],
            weights: [0, 0, 0, 0, 0, 0]), batch: 1, input: [1, 1])
    }
    checkThrows("metal rejects Int.max weight offset",
                ArtifactError.badPlan("step 0: weight offsets out of range")) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 2,
            steps: [.dense(index: 0, inputDim: 2, outputDim: 2,
                           inputBuffer: 0, outputBuffer: 1,
                           weightOffset: Int.max, weightCount: 4, biasOffset: 4)],
            weights: [0, 0, 0, 0, 0, 0]), batch: 1, input: [1, 1])
    }
    checkThrows("metal rejects zero buffer count",
                ArtifactError.badPlan("buffer count 0 out of range")) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 0,
            steps: [.relu(index: 0, buffer: 0)], weights: []), batch: 1, input: [1, 1])
    }
    checkThrows("metal rejects non-finite weights",
                ArtifactError.nonFiniteWeight(0)) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 2,
            steps: handLowered.steps,
            weights: [.nan, 2, 3, 4, -10, 0.5]), batch: 1, input: [1, 1])
    }
    // Non-canonical buffer allocation must be rejected before GPU access.
    checkThrows("metal rejects dense writing to buffer 0",
                ArtifactError.badPlan("step 0: dense output buffer is not the next fresh buffer")) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 2,
            steps: [.dense(index: 0, inputDim: 2, outputDim: 2,
                           inputBuffer: 0, outputBuffer: 0,
                           weightOffset: 0, weightCount: 4, biasOffset: 4)],
            weights: [1, 2, 3, 4, -10, 0.5]), batch: 1, input: [1, 1])
    }
    checkThrows("metal rejects dense reusing an earlier buffer",
                ArtifactError.badPlan("step 2: dense output buffer is not the next fresh buffer")) {
        // Three buffers declared; second dense reuses buffer 1 instead of
        // allocating the next fresh buffer.
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 3,
            steps: [.dense(index: 0, inputDim: 2, outputDim: 2,
                           inputBuffer: 0, outputBuffer: 1,
                           weightOffset: 0, weightCount: 4, biasOffset: 4),
                    .relu(index: 1, buffer: 1),
                    .dense(index: 2, inputDim: 2, outputDim: 2,
                           inputBuffer: 1, outputBuffer: 1,
                           weightOffset: 6, weightCount: 4, biasOffset: 10)],
            weights: [1, 2, 3, 4, -10, 0.5, 1, 2, 3, 4, -10, 0.5]),
            batch: 1, input: [1, 1])
    }
    checkThrows("metal rejects unused declared buffer",
                ArtifactError.badPlan("buffer count does not match allocated buffers")) {
        _ = try MetalBackend.run(LoweredProgram(
            inputDim: 2, outputDim: 2, bufferCount: 3,
            steps: [.dense(index: 0, inputDim: 2, outputDim: 2,
                           inputBuffer: 0, outputBuffer: 1,
                           weightOffset: 0, weightCount: 4, biasOffset: 4)],
            weights: [1, 2, 3, 4, -10, 0.5]), batch: 1, input: [1, 1])
    }

    // GPU vs independent CPU scalar oracle across batches on the full
    // demo graph (three dense layers, two ReLUs).
    let artifact = try CompiledArtifact(graph: demoGraph, seed: 42)
    for batch in [1, 4, 8, 16, 64] {
        let input = seededInput(dim: 16, batch: batch, seed: 7)
        let gpu = try MetalBackend.run(artifact.program, batch: batch, input: input)
        let reference = try ScalarBackend.run(artifact.program, batch: batch, input: input)
        if let diff = maxAbsDiff(gpu, reference) {
            check("metal matches scalar within 1e-4 at batch \(batch)", diff < 1e-4)
        } else {
            failed += 1
            print("FAIL: metal vs scalar at batch \(batch) — maxAbsDiff rejected input")
        }
        let accelerate = try AccelerateBackend.run(artifact.program, batch: batch, input: input)
        if let diff = maxAbsDiff(gpu, accelerate) {
            check("metal matches accelerate within 1e-4 at batch \(batch)", diff < 1e-4)
        } else {
            failed += 1
            print("FAIL: metal vs accelerate at batch \(batch) — maxAbsDiff rejected input")
        }
    }

    // Run-time limits are enforced before any GPU dispatch.
    checkThrows("metal rejects oversized batch",
                ExecutionError.batchTooLarge(512, Limits.maxBatch)) {
        _ = try MetalBackend.run(artifact.program, batch: 512, input: [])
    }
    checkThrows("metal rejects wrong input count",
                ExecutionError.wrongInputCount(3, 16)) {
        _ = try MetalBackend.run(artifact.program, batch: 1, input: [0, 0, 0])
    }
} catch {
    failed += 1
    print("FAIL: metal backend — threw \(error)")
}

// MARK: - CLI invalid inputs

do {
    let binary = URL(fileURLWithPath: ".build/debug/leviathan-cli")
    if !FileManager.default.fileExists(atPath: binary.path) {
        failed += 1
        print("FAIL: CLI binary not found at \(binary.path); run swift build first")
    } else {
        let cases: [[String]] = [
            ["run", "--artifact", "/dev/null", "--batch", "-1"],
            ["run", "--artifact", "/dev/null", "--batch", "abc"],
            ["run", "--artifact", "/dev/null", "--seed", "-3"],
            ["run", "--artifact", "/dev/null", "--seed", "xyz"],
            ["run", "--artifact", "/dev/null", "--batch", String(Limits.maxBatch + 1)],
            ["run", "--artifact", "/dev/null", "--backend"],
            ["run", "--artifact", "/dev/null", "--backend", "cuda"],
        ]
        for argv in cases {
            let process = Process()
            process.executableURL = binary
            process.arguments = argv
            let stderr = Pipe()
            process.standardError = stderr
            try process.run()
            process.waitUntilExit()
            let message = String(data: stderr.fileHandleForReading.readDataToEndOfFile(),
                                 encoding: .utf8) ?? ""
            check("CLI rejects \(argv[argv.count - 2]) \(argv[argv.count - 1]) cleanly",
                  process.terminationStatus == 1 && message.contains("error:"))
        }
    }
} catch {
    failed += 1; print("FAIL: CLI invalid inputs — threw \(error)")
}

// MARK: - Summary

print("\(passed) passed, \(failed) failed")
if failed > 0 { exit(1) }
print("ALL TESTS PASSED")
