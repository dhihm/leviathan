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

/// Error-category check: passes when the thrown error is the given case,
/// regardless of its associated message text.
@MainActor
func checkThrowsCase(_ name: String, _ expectedCase: String,
                     _ body: () throws -> Void) {
    do {
        try body()
        failed += 1; print("FAIL: \(name) — expected \(expectedCase), got no error")
    } catch {
        let actual = String(describing: error)
        if actual.hasPrefix(expectedCase + "(\"") || actual == expectedCase {
            passed += 1
        } else {
            failed += 1; print("FAIL: \(name) — expected case \(expectedCase), got \(actual)")
        }
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

// MARK: - GGUF reader (synthetic container, known fields)

func u32(_ v: UInt32) -> [UInt8] { [
    UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)] }
func u64(_ v: UInt64) -> [UInt8] { (0..<8).map { UInt8((v >> (8 * UInt64($0))) & 0xFF) } }
func ggufString(_ s: String) -> [UInt8] { u64(UInt64(s.utf8.count)) + Array(s.utf8) }

do {
    // Hand-built v3 container: two metadata keys, one Q4_0 tensor
    // [inDim 32, outDim 4] = 4 blocks of 18 bytes, alignment 32.
    var bytes: [UInt8] = Array("GGUF".utf8)
    bytes += u32(3) + u64(1) + u64(2)
    bytes += ggufString("general.alignment") + u32(GGUFValueType.uint32.rawValue) + u32(32)
    bytes += ggufString("gemma4.context_length") + u32(GGUFValueType.uint64.rawValue) + u64(262144)
    bytes += ggufString("test_tensor") + u32(2) + u64(32) + u64(4) + u32(2) + u64(0)
    while bytes.count % 32 != 0 { bytes.append(0) }
    bytes += [UInt8](repeating: 0x7F, count: 4 * Q4_0.bytesPerBlock)
    let file = try GGUFReader.parse(Data(bytes))
    checkEqual("gguf version parsed", file.version, UInt32(3))
    checkEqual("gguf alignment parsed", try file.metadataUInt("general.alignment"), 32)
    checkEqual("gguf uint64 metadata parsed", try file.metadataUInt("gemma4.context_length"), 262144)
    checkEqual("gguf tensor count", file.tensors.count, 1)
    let info = try file.tensor("test_tensor")
    checkEqual("gguf tensor dims", info.dims, [32, 4])
    checkEqual("gguf tensor ggml type", info.ggmlType, UInt32(2))
    checkEqual("gguf tensor range", try file.tensorRange(info),
               bytes.count - 4 * Q4_0.bytesPerBlock..<bytes.count)
    checkEqual("gguf tensor data length",
               try file.tensorData(info).count, 4 * Q4_0.bytesPerBlock)
    checkThrows("gguf missing tensor rejected", GGUFError.missingTensor("absent")) {
        _ = try file.tensor("absent")
    }
    checkThrows("gguf bad magic rejected", GGUFError.badMagic) {
        _ = try GGUFReader.parse(Data([0x47, 0x47, 0x55, 0x58, 0x03, 0, 0, 0, 0, 0, 0, 0]))
    }
    checkThrows("gguf truncated rejected", GGUFError.truncated("read 8 at 8")) {
        _ = try GGUFReader.parse(Data(Array("GGUF".utf8) + u32(2) + [0, 0, 0, 0]))
    }
    // Malicious counts: UInt64.max tensor/kv counts are rejected, not
    // looped over or converted to a negative Int.
    checkThrows("gguf huge kv count rejected", GGUFError.truncated("implausible tensor/kv counts")) {
        _ = try GGUFReader.parse(Data(
            Array("GGUF".utf8) + u32(3) + u64(0) + u64(UInt64.max)))
    }
    // A string length of UInt64.max is capped before any allocation.
    checkThrows("gguf huge string rejected", GGUFError.truncated("string length beyond cap")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(0) + u64(1)
        b += ggufString("k") + u32(GGUFValueType.string.rawValue) + u64(UInt64.max)
        _ = try GGUFReader.parse(Data(b))
    }
    // Arrays of arrays and oversized arrays are rejected.
    checkThrows("gguf nested array rejected", GGUFError.unknownValueType(9)) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(0) + u64(1)
        b += ggufString("k") + u32(GGUFValueType.array.rawValue) + u32(9) + u64(0)
        _ = try GGUFReader.parse(Data(b))
    }
    checkThrows("gguf huge array rejected", GGUFError.truncated("array too long")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(0) + u64(1)
        b += ggufString("k") + u32(GGUFValueType.array.rawValue)
        b += u32(GGUFValueType.uint8.rawValue) + u64(UInt64.max)
        _ = try GGUFReader.parse(Data(b))
    }
    // Zero dimension rejected.
    checkThrows("gguf zero dim rejected",
                GGUFError.badTensor("tensor t: invalid dimension")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(0) + u64(4) + u32(2) + u64(0)
        _ = try GGUFReader.parse(Data(b))
    }
    // A dimension that would overflow any practical element count.
    checkThrows("gguf huge dim rejected", GGUFError.badTensor("tensor t: invalid dimension")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(UInt64.max) + u64(1) + u32(2) + u64(0)
        _ = try GGUFReader.parse(Data(b))
    }
    // Ragged quantized block count (31 F32 elements is fine for F32 but a
    // Q4_0 tensor of 31 elements is ragged).
    checkThrows("gguf ragged q4 block count rejected",
                GGUFError.badTensor("tensor t: element count 31 is not a multiple of the q4_0 block")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(31) + u64(1) + u32(2) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 18)
        _ = try GGUFReader.parse(Data(b))
    }
    // Unsupported GGML type (3 = Q4_1) is explicit, never guessed.
    checkThrows("gguf unsupported tensor type rejected", GGUFError.unsupportedTensorType(3)) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(3) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 20)
        _ = try GGUFReader.parse(Data(b))
    }
    // Tensor data beyond end of file.
    do {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(2) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        let off = b.count
        checkThrows("gguf data beyond file rejected",
                    GGUFError.dataBeyondEndOffile("tensor t: needs bytes [\(off), \(off + 18)) but file is \(off) bytes")) {
            _ = try GGUFReader.parse(Data(b))
        }
    }
    // Misaligned offset and overlapping tensors are rejected.
    checkThrows("gguf misaligned offset rejected",
                GGUFError.badTensor("tensor t: offset 3 misaligned")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(2) + u64(3)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 36)
        _ = try GGUFReader.parse(Data(b))
    }
    checkThrows("gguf overlapping tensors rejected",
                GGUFError.badTensor("tensor t2 overlaps t1")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(2) + u64(0)
        // t1 at offset 0, t2 at offset 0 — overlapping storage.
        b += ggufString("t1") + u32(2) + u64(32) + u64(1) + u32(2) + u64(0)
        b += ggufString("t2") + u32(2) + u64(32) + u64(1) + u32(2) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 18)
        _ = try GGUFReader.parse(Data(b))
    }
    checkThrows("gguf duplicate tensor names rejected",
                GGUFError.badTensor("duplicate tensor name t")) {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(2) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(2) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(2) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 36)
        _ = try GGUFReader.parse(Data(b))
    }
    // Array fixtures with real GGUF element layout (one element type,
    // then bare payloads), plus trailing metadata proving the cursor
    // position after an array is exact.
    do {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(0) + u64(3)
        b += ggufString("names") + u32(GGUFValueType.array.rawValue)
        b += u32(GGUFValueType.string.rawValue) + u64(2)
        b += ggufString("alpha") + ggufString("beta")
        b += ggufString("ids") + u32(GGUFValueType.array.rawValue)
        b += u32(GGUFValueType.uint32.rawValue) + u64(3)
        b += u32(10) + u32(20) + u32(30)
        b += ggufString("after") + u32(GGUFValueType.uint32.rawValue) + u32(7)
        let f = try GGUFReader.parse(Data(b))
        if case let .array(_, count, _, _, _, strings, _)? = f.metadata["names"],
           count == 2, strings == ["alpha", "beta"] {
            passed += 1
        } else {
            failed += 1; print("FAIL: gguf string array fixture")
        }
        if case let .array(_, count2, uints, _, _, _, _)? = f.metadata["ids"],
           count2 == 3, uints == [10, 20, 30] {
            passed += 1
        } else {
            failed += 1; print("FAIL: gguf uint32 array fixture")
        }
        checkEqual("gguf metadata after array parsed", try f.metadataUInt("after"), 7)
    } catch {
        failed += 1; print("FAIL: gguf array fixtures — threw \(error)")
    }
    // Q6_K geometry: a 256x2 tensor is exactly 2 * 210 bytes.
    do {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(256) + u64(2) + u32(14) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        b += [UInt8](repeating: 0, count: 2 * 210)
        let f = try GGUFReader.parse(Data(b))
        checkEqual("gguf q6_k tensor size",
                   try f.tensorRange(try f.tensor("t")).count, 2 * 210)
    } catch {
        failed += 1; print("FAIL: gguf q6_k geometry — threw \(error)")
    }
    // Regression: a zero-tensor file parses without trapping.
    do {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(0) + u64(0)
        while b.count % 32 != 0 { b.append(0) }
        let empty = try GGUFReader.parse(Data(b))
        checkEqual("gguf zero-tensor file parses", empty.tensors.count, 0)
    } catch {
        failed += 1; print("FAIL: gguf zero-tensor file parses — threw \(error)")
    }
    // Regression: a near-Int.max offset fails through the checked error
    // path, not an overflow trap.
    do {
        var b: [UInt8] = Array("GGUF".utf8) + u32(3) + u64(1) + u64(0)
        b += ggufString("t") + u32(2) + u64(32) + u64(1) + u32(2)
        // Aligned near Int.max so alignment passes and the checked
        // start/end arithmetic overflow path is what fires.
        let huge = Int.max - Int.max % 32
        b += u64(UInt64(huge))
        while b.count % 32 != 0 { b.append(0) }
        let off = b.count
        checkThrows("gguf huge offset rejected safely",
                    GGUFError.dataBeyondEndOffile("tensor t: needs bytes [-1, -1) but file is \(off) bytes")) {
            _ = try GGUFReader.parse(Data(b))
        }
    }
} catch {
    failed += 1; print("FAIL: gguf reader — threw \(error)")
}

// MARK: - Q4_0 scalar oracle (known blocks)

do {
    // Block with scale 1.0 (FP16 0x3C00): low nibble of byte l is value l,
    // high nibble 0 is value -8.
    var block: [UInt8] = [0x00, 0x3C]
    var expected: [Float] = []
    for l in 0..<16 { block.append(UInt8(l)) }
    for l in 0..<16 { expected.append(Float(l) - 8) }   // low nibbles first
    for _ in 0..<16 { expected.append(Float(0) - 8) }   // then high nibbles
    let dequant = try Q4_0.decodeBlock(block[0..<Q4_0.bytesPerBlock])
    checkEqual("q4_0 decode block count", dequant.count, 32)
    var maxDiff: Float = 0
    for i in 0..<32 { maxDiff = max(maxDiff, abs(dequant[i] - expected[i])) }
    check("q4_0 decode known block matches oracle", maxDiff == 0)
    // Scale path: scale 0.5 (FP16 0x3800).
    var halfBlock = block; halfBlock[1] = 0x38
    let scaled = try Q4_0.decodeBlock(halfBlock[0..<Q4_0.bytesPerBlock])
    check("q4_0 fp16 scale decoded", abs(scaled[0] - 0.5 * (-8)) < 1e-6)
    // Non-finite scale (FP16 +Inf 0x7C00) is rejected.
    var infBlock = block; infBlock[1] = 0x7C
    checkThrows("q4_0 non-finite scale rejected", Q4Error.nonFiniteScale(block: 0)) {
        _ = try Q4_0.decodeBlock(infBlock[0..<Q4_0.bytesPerBlock])
    }
    // Streaming matvec oracle on a deterministic tiny matrix: 2 rows x 32.
    // Row 0 all-zero nibbles (value -8, scale 1); row 1 scale -1 -> +8x.
    var w: [UInt8] = []
    w += [0x00, 0x3C] + [UInt8](repeating: 0x00, count: 16)
    w += [0x00, 0xBC] + [UInt8](repeating: 0x00, count: 16)
    let x = seededInput(dim: 32, batch: 1, seed: 7)
    let sumX = x.reduce(0, +)
    let y = try Q4_0.matvecReference(blocks: w, outDim: 2, inDim: 32, x: x)
    checkEqual("q4_0 scalar matvec rows", y.count, 2)
    check("q4_0 scalar matvec row 0", abs(y[0] - (-8 * sumX)) < 1e-3)
    check("q4_0 scalar matvec row 1", abs(y[1] - (8 * sumX)) < 1e-3)
    // Non-finite activation is rejected by the oracle.
    var badX = x; badX[0] = .nan
    checkThrows("q4_0 oracle rejects non-finite input",
                Q4Error.nonFiniteInput("x[0]")) {
        _ = try Q4_0.matvecReference(blocks: w, outDim: 2, inDim: 32, x: badX)
    }
} catch {
    failed += 1; print("FAIL: q4_0 oracle — threw \(error)")
}

// MARK: - Q4_0 fused Metal matvec vs scalar oracle

do {
    let inDim = 256  // 8 blocks per row
    let outDim = 128
    let blocksPerRow = inDim / Q4_0.valuesPerBlock
    // Deterministic pseudo-random Q4_0 blocks with varied finite scales.
    var rng = SplitMix64(seed: 99)
    var blocks: [UInt8] = []
    blocks.reserveCapacity(outDim * blocksPerRow * Q4_0.bytesPerBlock)
    for _ in 0..<(outDim * blocksPerRow) {
        let scale = Float16(rng.nextUniform() * 2 - 1)
        blocks += [UInt8(scale.bitPattern & 0xFF), UInt8(scale.bitPattern >> 8)]
        for _ in 0..<16 { blocks.append(UInt8(rng.next() & 0xFF)) }
    }
    let x = seededInput(dim: inDim, batch: 1, seed: 101)
    let oracle = try Q4_0.matvecReference(blocks: blocks, outDim: outDim, inDim: inDim, x: x)
    let (gpu, evidence) = try Q4MetalBackend.matvecReporting(
        blocks: blocks, outDim: outDim, inDim: inDim, x: x)
    checkEqual("q4_0 metal matvec output count", gpu.count, outDim)
    var gpuMaxDiff: Float = 0
    for i in 0..<outDim {
        gpuMaxDiff = max(gpuMaxDiff, abs(gpu[i] - oracle[i]) / max(1, abs(oracle[i])))
    }
    check("q4_0 metal matvec matches scalar oracle (rel tol 1e-4)", gpuMaxDiff < 1e-4)
    check("q4_0 metal evidence completed", evidence.commandStatus == "completed")
    check("q4_0 metal evidence device reported", !evidence.deviceName.isEmpty)
    check("q4_0 metal evidence working set reported",
          evidence.recommendedMaxWorkingSetSize > 0)
    // Guards fire before dispatch on bad geometry / counts.
    checkThrows("q4_0 metal rejects ragged inDim",
                Q4Error.badDimensions("inDim 33, outDim 8")) {
        _ = try Q4MetalBackend.matvecReporting(blocks: [], outDim: 8, inDim: 33, x: [])
    }
    checkThrows("q4_0 metal rejects wrong weight bytes",
                Q4Error.badDimensions("weights are 0 bytes, geometry needs \(8 * 1 * 18)")) {
        _ = try Q4MetalBackend.matvecReporting(blocks: [], outDim: 8, inDim: 32, x: [])
    }
    checkThrows("q4_0 metal rejects wrong input count",
                Q4Error.wrongInputCount(3, expected: 32)) {
        _ = try Q4MetalBackend.matvecReporting(
            blocks: [UInt8](repeating: 0, count: 8 * 18), outDim: 8, inDim: 32, x: [0, 0, 0])
    }
    var nanX = [Float](repeating: 0, count: 32); nanX[1] = .infinity
    checkThrows("q4_0 metal rejects non-finite input",
                Q4Error.nonFiniteInput("activation vector")) {
        _ = try Q4MetalBackend.matvecReporting(
            blocks: [UInt8](repeating: 0, count: 8 * 18), outDim: 8, inDim: 32, x: nanX)
    }
} catch {
    failed += 1; print("FAIL: q4_0 metal matvec — threw \(error)")
}

// MARK: - Q4 lowering: simdLanePlain legality, attention-output family,
// emission determinism, and actual-Metal parity

do {
    let hw32 = ForgeHardwareProfile(deviceName: "t32",
                                    maxTotalThreadsPerThreadgroup: 1024,
                                    threadExecutionWidth: 32,
                                    recommendedMaxWorkingSetSize: 0)
    let hw16 = ForgeHardwareProfile(deviceName: "t16",
                                    maxTotalThreadsPerThreadgroup: 1024,
                                    threadExecutionWidth: 16,
                                    recommendedMaxWorkingSetSize: 0)
    let cap128 = ForgeHardwareProfile(deviceName: "c128",
                                      maxTotalThreadsPerThreadgroup: 128,
                                      threadExecutionWidth: 32,
                                      recommendedMaxWorkingSetSize: 0)
    func simd(_ r: Int) -> Q4Schedule {
        Q4Schedule(threadsPerRow: 32, rowsPerGroup: r,
                   accumulation: .f32, xLoads: .simdLanePlain)
    }
    // inDim 128 = 4 blocks/row: the lane mapping admits it (every lane
    // traverses ALL blocks) while the old block-chunk split rejects every
    // T>1 (blocksPerRow/T < minWorkBlocks).
    check("simdLanePlain R1 legal on width-32 hardware",
          simd(1).isLegal(blocksPerRow: 4, hardware: hw32))
    check("simdLanePlain R8 legal on width-32 hardware",
          simd(8).isLegal(blocksPerRow: 4, hardware: hw32))
    check("simdLanePlain illegal on width-16 hardware",
          !simd(1).isLegal(blocksPerRow: 4, hardware: hw16))
    check("simdLanePlain R8 illegal when 32*R exceeds the threadgroup cap",
          !simd(8).isLegal(blocksPerRow: 4, hardware: cap128))
    check("simdLanePlain R4 fits the 128-thread cap",
          simd(4).isLegal(blocksPerRow: 4, hardware: cap128))
    check("block-chunk split still rejects T=8 at 4 blocks per row",
          !Q4Schedule(threadsPerRow: 8, rowsPerGroup: 1,
                      accumulation: .f32, xLoads: .scalar)
              .isLegal(blocksPerRow: 4, hardware: hw32))

    let attnJSON = """
    {"schema":1,"op":"q4_0.matvec","semantics":"dot(W_row, x)",
     "inDim":4096,"outDim":2816,"layout":"row-major-q4_0-blocks","quant":"q4_0",
     "accumulate":"f32","inputPrecision":"f32",
     "tensor":"blk.0.attn_output.weight","representativeRows":256,
     "view":"attention-output-family"}
    """
    func parseOp(_ json: String) throws -> Q4OpIR {
        try Q4OpIR.parse(Data(json.utf8))
    }
    let attnOp = try parseOp(attnJSON)

    // Strict family recognizer.
    check("strict name accepts blk.17.attn_output.weight",
          Q4OpIR.isAttentionOutputTensorName("blk.17.attn_output.weight"))
    check("strict name rejects empty layer",
          !Q4OpIR.isAttentionOutputTensorName("blk..attn_output.weight"))
    check("strict name rejects wrong family",
          !Q4OpIR.isAttentionOutputTensorName("blk.0.attn_output.bias"))
    check("strict name rejects non-digit layer",
          !Q4OpIR.isAttentionOutputTensorName("blk.0x.attn_output.weight"))

    // Structural negatives (fail closed before any lowering).
    checkThrows("family view rejects wrong tensor name",
                Q4LoweringError.invalidOp(
                    "attention-output-family binds a strict "
                    + "blk.<digits>.attn_output.weight tensor, got blk.0.mlp.weight")) {
        _ = try parseOp(attnJSON.replacingOccurrences(of: "attn_output", with: "mlp"))
    }
    checkThrows("family view rejects non-ASCII digit layer",
                Q4LoweringError.invalidOp(
                    "attention-output-family binds a strict "
                    + "blk.<digits>.attn_output.weight tensor, got blk.٠.attn_output.weight")) {
        _ = try parseOp(attnJSON.replacingOccurrences(of: "blk.0.", with: "blk.٠."))
    }
    checkThrows("family view rejects missing binding",
                Q4LoweringError.invalidOp(
                    "attention-output-family binds a strict "
                    + "blk.<digits>.attn_output.weight tensor, got nil")) {
        _ = try parseOp(attnJSON.replacingOccurrences(
            of: "\"tensor\":\"blk.0.attn_output.weight\"", with: "\"tensor\":null"))
    }
    checkThrows("unsupported view string rejected",
                Q4LoweringError.invalidOp("unsupported view nope")) {
        _ = try parseOp(attnJSON.replacingOccurrences(of: "attention-output-family", with: "nope"))
    }
    // Arbitrary positive outDim values are structurally legal (see the
    // ragged 2817 test); the serving predicate, not the parser, decides
    // whether a bound out dim can serve a given output row count.
    do {
        let familyOp = try parseOp(attnJSON)
        check("bound outDim 2816 serves exact outRows 2816 (strict family, inDim 4096)",
              Gemma4Runner.q4GeneratedServesTensor(
                boundTensor: familyOp.tensor,
                boundInDim: familyOp.inDim,
                boundOutDim: familyOp.outDim,
                boundView: familyOp.view,
                tensor: "blk.17.attn_output.weight",
                inDim: 4096,
                outRows: 2816))
        check("bound outDim 2816 does not serve outRows 2817",
              !Gemma4Runner.q4GeneratedServesTensor(
                boundTensor: familyOp.tensor,
                boundInDim: familyOp.inDim,
                boundOutDim: familyOp.outDim,
                boundView: familyOp.view,
                tensor: "blk.17.attn_output.weight",
                inDim: 4096,
                outRows: 2817))
    }
    // view nil keeps existing semantics: an attn_output tensor without
    // the family view is just a plain 2-D op.
    _ = try parseOp(attnJSON.replacingOccurrences(
        of: "\"view\":\"attention-output-family\"", with: "\"view\":null"))

    // Deriver: baseline first, existing legal set preserved, the three
    // new candidates appended, bounded at 16, width-32 only.
    let derived = Q4ScheduleDeriver.derive(op: attnOp, hardware: hw32)
    checkEqual("q4 deriver baseline first", derived.first,
               Q4Schedule(threadsPerRow: 1, rowsPerGroup: 1,
                          accumulation: .f32, xLoads: .scalar))
    check("q4 deriver keeps all three simdLanePlain candidates",
          [1, 4, 8].allSatisfy { r in derived.contains(simd(r)) })
    check("q4 deriver preserved the pre-existing legal set", derived.count > 13)
    check("q4 deriver bounded at 16 candidates", derived.count <= 16)
    check("q4 deriver adds no simdLanePlain on width-16 hardware",
          !Q4ScheduleDeriver.derive(op: attnOp, hardware: hw16)
              .contains { $0.xLoads == .simdLanePlain })

    // Runtime serving predicate: ONLY the explicit family view serves
    // other strict attn_output layers, and only at the exact bound dims.
    check("family view serves other strict attn_output layers at exact dims",
          Gemma4Runner.q4GeneratedServesTensor(
              boundTensor: "blk.0.attn_output.weight", boundInDim: 4096,
              boundOutDim: 2816, boundView: "attention-output-family",
              tensor: "blk.17.attn_output.weight", inDim: 4096, outRows: 2816))
    check("family view does not serve mismatched dims",
          !Gemma4Runner.q4GeneratedServesTensor(
              boundTensor: "blk.0.attn_output.weight", boundInDim: 4096,
              boundOutDim: 2816, boundView: "attention-output-family",
              tensor: "blk.17.attn_output.weight", inDim: 8192, outRows: 2816))
    check("view-nil binding does not serve sibling layers",
          !Gemma4Runner.q4GeneratedServesTensor(
              boundTensor: "blk.0.attn_output.weight", boundInDim: 4096,
              boundOutDim: 2816, boundView: nil,
              tensor: "blk.17.attn_output.weight", inDim: 4096, outRows: 2816))
    check("family view does not serve non-strict layer names",
          !Gemma4Runner.q4GeneratedServesTensor(
              boundTensor: "blk.0.attn_output.weight", boundInDim: 4096,
              boundOutDim: 2816, boundView: "attention-output-family",
              tensor: "blk.x.attn_output.weight", inDim: 4096, outRows: 2816))

    // Emission: deterministic, original lane mapping, simd_sum, no
    // barrier, no compensation, codegen v3.
    let simd8 = simd(8)
    let src = try Q4KernelEmitter.emit(op: attnOp, schedule: simd8)
    checkEqual("simdLanePlain emission deterministic",
               try Q4KernelEmitter.emit(op: attnOp, schedule: simd8), src)
    check("simdLanePlain emission carries codegen v3", src.contains("q4gen-v3"))
    check("simdLanePlain emission uses the original lane mapping",
          src.contains("uint row = wid * R + sg;") && src.contains("uint li = lane & 15u;"))
    check("simdLanePlain emission reduces with simd_sum", src.contains("simd_sum(s)"))
    check("simdLanePlain emission has no barrier", !src.contains("threadgroup_barrier"))
    check("simdLanePlain emission lane 0 writes", src.contains("lane == 0u"))

    // Actual Metal: ragged 35 rows x inDim 128, finite Q4 weights, FP64
    // oracle at the ORIGINAL 1e-4 component tolerance (unchanged).
    if let device = MTLCreateSystemDefaultDevice(),
       try Q4CurrentHardware.profile().threadExecutionWidth == 32 {
        let small = try parseOp("""
        {"schema":1,"op":"q4_0.matvec","semantics":"dot(W_row, x)",
         "inDim":128,"outDim":35,"layout":"row-major-q4_0-blocks","quant":"q4_0",
         "accumulate":"f32","inputPrecision":"f32",
         "tensor":null,"representativeRows":35,"view":null}
        """)
        let blocks = try Q4ComponentInputs.weightBytes(op: small, file: nil)
        let x = Q4ComponentInputs.activation(op: small)
        let kernel = try Q4EmittedKernel.compile(op: small, schedule: simd(8), device: device)
        let gpu = try kernel.matvec(blocks: blocks, outDim: 35, inDim: 128, x: x)
        let ref64 = try Q4Oracle.matvecFP64(blocks: blocks[0...], outDim: 35,
                                            inDim: 128, x: x)
        var worst = 0.0
        for i in 0..<35 {
            worst = max(worst, abs(Double(gpu[i]) - ref64[i]) / max(1, abs(ref64[i])))
        }
        check("simdLanePlain GPU ragged 35x128 matches FP64 oracle (rel tol 1e-4)",
              worst < 1e-4)
    } else {
        print("  simdLanePlain GPU test skipped (no width-32 Metal device)")
    }
} catch {
    failed += 1; print("FAIL: q4 simdLanePlain lowering — threw \(error)")
}

// MARK: - Q6_K scalar oracle and fused Metal matvec

do {
    // Known block: d = 1.0 (FP16 0x3C00), all scales = 1, ql/qh crafted
    // so a few positions have known 6-bit levels.
    var block = [UInt8](repeating: 0, count: Q6_K.bytesPerBlock)
    block[208] = 0x00; block[209] = 0x3C                      // d = 1.0
    for i in 0..<16 { block[192 + i] = 1 }                    // scales = 1
    block[0] = 0x21   // ql[0]: low nibble 1 -> v[0]; high nibble 2 -> v[64]
    block[128] = 0x03 // qh[0]: adds 2-bit extensions to v[0],v[32],v[64],v[96]
    let dequant = try Q6_K.decodeBlock(block[0..<Q6_K.bytesPerBlock])
    checkEqual("q6_k decode block count", dequant.count, 256)
    // v[0]:  (ql[0] & 0xF) | ((qh[0] & 3) << 4) = 1 | 48 = 49 -> 49 - 32 = 17
    checkEqual("q6_k v[0] level", dequant[0], Float(17))
    // v[32]: ql[32] low nibble 0 | (((qh[0]>>2) & 3) << 4) = 0 -> -32
    checkEqual("q6_k v[32] level", dequant[32], Float(-32))
    // v[64]: (ql[0] >> 4) | ((qh[0]>>4 & 3) << 4) = 2 | 0 = 2 -> -30
    checkEqual("q6_k v[64] level", dequant[64], Float(-30))
    // v[96]: (ql[32] >> 4) | ((qh[0]>>6 & 3) << 4) = 0 -> -32
    checkEqual("q6_k v[96] level", dequant[96], Float(-32))
    // v[1] untouched: level 0 -> -32
    checkEqual("q6_k untouched value", dequant[1], Float(-32))
    // Subgroup scales apply: set scale of subgroup 1 (values 16..31) to 2.
    block[193] = 2
    let scaled = try Q6_K.decodeBlock(block[0..<Q6_K.bytesPerBlock])
    checkEqual("q6_k subgroup scale applied", scaled[20], Float(2 * -32))
    // Non-finite block scale rejected.
    block[209] = 0x7C
    checkThrows("q6_k non-finite scale rejected", Q4Error.nonFiniteScale(block: -1)) {
        _ = try Q6_K.decodeBlock(block[0..<Q6_K.bytesPerBlock])
    }
} catch {
    failed += 1; print("FAIL: q6_k oracle — threw \(error)")
}

do {
    // GPU vs scalar oracle on pseudo-random blocks, 512 x 64.
    let inDim = 512, outDim = 64
    let blocksPerRow = inDim / Q6_K.valuesPerBlock
    var rng = SplitMix64(seed: 555)
    var blocks: [UInt8] = []
    blocks.reserveCapacity(outDim * blocksPerRow * Q6_K.bytesPerBlock)
    for _ in 0..<(outDim * blocksPerRow) {
        // 208 bytes of ql/qh/scales, then the FP16 d at bytes 208-209 —
        // drawn from a finite Float16 so the scale check passes.
        for _ in 0..<208 { blocks.append(UInt8(rng.next() & 0xFF)) }
        let scale = Float16(rng.nextUniform() * 2 - 1)
        blocks += [UInt8(scale.bitPattern & 0xFF), UInt8(scale.bitPattern >> 8)]
    }
    let x = seededInput(dim: inDim, batch: 1, seed: 202)
    let oracle = try Q6_K.matvecReference(blocks: blocks, outDim: outDim, inDim: inDim, x: x)
    let (gpu, evidence) = try Q6MetalBackend.matvecReporting(
        blocks: blocks, outDim: outDim, inDim: inDim, x: x)
    checkEqual("q6_k metal matvec output count", gpu.count, outDim)
    // Independent FP64 reference gates both the GPU kernel and the F32
    // CPU oracle at the same 1e-4 relative tolerance; Float rounding
    // order differences between them are not permitted to hide behind a
    // looser gate.
    let ref64 = try Q6_K.matvecReferenceF64(blocks: blocks, outDim: outDim,
                                            inDim: inDim, x: x)
    var gpuDiff: Float = 0
    var oracleDiff: Float = 0
    var worstRow = 0
    var worstSumAbs: Double = 0
    // sum of |products| for the worst row — evidence that a small result
    // with large canceling terms (cancellation, not bit error) explains a
    // relative gap.
    var sumAbs = [Double](repeating: 0, count: outDim)
    for row in 0..<outDim {
        for b in 0..<(inDim / Q6_K.valuesPerBlock) {
            let base = (row * (inDim / Q6_K.valuesPerBlock) + b) * Q6_K.bytesPerBlock
            let values = try Q6_K.decodeBlock(blocks[base..<(base + Q6_K.bytesPerBlock)])
            for j in 0..<Q6_K.valuesPerBlock {
                sumAbs[row] += abs(Double(values[j]) * Double(x[b * Q6_K.valuesPerBlock + j]))
            }
        }
    }
    for i in 0..<outDim {
        let denominator = { max(1.0, abs(ref64[i])) }
        let g = Float(abs(Double(gpu[i]) - ref64[i]) / denominator())
        if g > gpuDiff { gpuDiff = g; worstRow = i; worstSumAbs = sumAbs[i] }
        oracleDiff = max(oracleDiff, Float(abs(Double(oracle[i]) - ref64[i]) / denominator()))
    }
    print(String(format: "  q6_k worst rel vs FP64 reference: gpu %.3e, scalar %.3e",
                 gpuDiff, oracleDiff))
    print(String(format: "  q6_k worst row %d: ref64 %.6f gpu %.6f sumAbsProducts %.1f (|result| %.1f)",
                 worstRow, ref64[worstRow], gpu[worstRow], worstSumAbs, abs(ref64[worstRow])))
    check("q6_k metal matvec matches FP64 reference (rel tol 1e-4)", gpuDiff < 1e-4)
    check("q6_k scalar oracle matches FP64 reference (rel tol 1e-4)", oracleDiff < 1e-4)
    check("q6_k metal evidence completed", evidence.commandStatus == "completed")
} catch {
    failed += 1; print("FAIL: q6_k metal matvec — threw \(error)")
}

// Independent hand-computed GPU fixture: one row, one block, x zero
// everywhere except positions 0, 16, 64, 128, 192 — so the result is an
// exact integer with no baseline and no rounding ambiguity. Signed scales
// and a crafted chunk-1 (values 128..255) nibble map are exercised.
do {
    var block = [UInt8](repeating: 0, count: Q6_K.bytesPerBlock)
    block[208] = 0x00; block[209] = 0x3C                    // d = 1.0
    for i in 0..<16 { block[192 + i] = 0 }
    block[192] = 3        // scales[0] = +3
    block[193] = 0xFD     // scales[1] = -3 (signed int8)
    block[196] = 2        // scales[4] = +2
    block[200] = 3        // scales[8] = +3 (chunk-1 subgroup 0)
    block[204] = 2        // scales[12] = +2 (chunk-1 subgroup 2)
    block[0] = 0x21       // ql[0]: v[0] low nibble 1, v[64] high nibble 2
    block[128] = 0x03     // qh[0]: v[0] ext 3, v[32]/v[64]/v[96] ext 0
    block[64] = 0x21      // ql[64]: v[128] low nibble 1, v[192] high nibble 2
    block[160] = 0x01     // qh[32]: v[128] ext 1, v[160]/v[192]/v[224] ext 0
    var x = [Float](repeating: 0, count: 256)
    x[0] = 1; x[16] = -1; x[64] = 2; x[128] = -2; x[192] = 4
    // v[0]   = 1 * scales[0] * ((1 | 3<<4) - 32) = 3 * 17 = 51;  term 51*1  =  51
    // v[16]  = 1 * scales[1] * (0 - 32)           = -3 * -32 = 96; term 96*-1 = -96
    // v[64]  = 1 * scales[4] * (2 - 32)           = 2 * -30 = -60; term -60*2 = -120
    // v[128] = 1 * scales[8] * ((1 | 1<<4) - 32)  = 3 * -15 = -45; term -45*-2 =  90
    // v[192] = 1 * scales[12] * (2 - 32)          = 2 * -30 = -60; term -60*4 = -240
    let expected: Float = 51 - 96 - 120 + 90 - 240   // exactly -315
    let (gpu, _) = try Q6MetalBackend.matvecReporting(
        blocks: block, outDim: 1, inDim: 256, x: x)
    let oracle = try Q6_K.matvecReference(blocks: block, outDim: 1, inDim: 256, x: x)
    print(String(format: "  q6_k known block: ref %.6f gpu %.6f hand %.6f",
                 oracle[0], gpu[0], expected))
    check("q6_k known block gpu matches hand computation (-315)",
          gpu[0] == expected)
    check("q6_k known block scalar matches hand computation (-315)",
          oracle[0] == expected)
} catch {
    failed += 1; print("FAIL: q6_k known block — threw \(error)")
}

// MARK: - Forge: bounded automatic pass generation + plan reuse

do {
    let forgeGraph = GraphIR(inputDim: 64, ops: [
        .dense(outDim: 128), .relu, .dense(outDim: 64), .relu, .dense(outDim: 8),
    ])
    let (plan, output, hardware, results) = try Forge.forge(
        graph: forgeGraph, batch: 4, seed: 42)
    check("forge produced results", results.count > 0)
    check("forge all measured candidates correct",
          results.filter { $0.correct }.allSatisfy { r in
              guard let m = r.medianGpuMicros else { return false }
              return m.isFinite && m > 0 })
    check("forge selected candidate measured",
          plan.selectedMedianGpuMicros > 0 && plan.selectedMedianGpuMicros.isFinite)
    check("forge output finite", output.allSatisfy(\.isFinite))
    check("forge hardware profile measured",
          hardware.maxTotalThreadsPerThreadgroup > 0
          && hardware.threadExecutionWidth > 0
          && hardware.recommendedMaxWorkingSetSize > 0)
    check("forge fusion candidate present (dense+relu graph)",
          results.contains { $0.candidate.fuseDenseRelu })

    // Cache hit: the just-forged plan validates against the same inputs.
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("leviathan-test.forge.json")
    try plan.save(to: tmp)
    let loaded = try ForgePlan.loadValidated(
        from: tmp, graph: forgeGraph, seed: 42, batch: 4, hardware: hardware)
    checkEqual("forge plan cache hit round-trips", loaded, plan)
    // Stale plan: different seed is another model — rejected on workload
    // identity before the digest is even compared.
    checkThrows("forge rejects stale plan (wrong seed)",
                ForgeError.invalidPlan("workload mismatch (batch/seed)")) {
        _ = try ForgePlan.loadValidated(from: tmp, graph: forgeGraph, seed: 43,
                                        batch: 4, hardware: hardware)
    }
    // Corrupt plan: illegal selected width.
    var tampered = plan
    tampered = ForgePlan(
        schemaVersion: plan.schemaVersion, graphDigest: plan.graphDigest,
        hardware: plan.hardware, batch: plan.batch, seed: plan.seed,
        toolchain: plan.toolchain,
        selected: ForgeCandidate(threadgroupWidth: 99_999, fuseDenseRelu: false),
        selectedMedianGpuMicros: plan.selectedMedianGpuMicros,
        results: plan.results, planDigest: plan.planDigest)
    try tampered.save(to: tmp)
    checkThrows("forge rejects illegal selected width",
                ForgeError.invalidPlan("plan fields outside validated bounds")) {
        _ = try ForgePlan.loadValidated(from: tmp, graph: forgeGraph, seed: 42,
                                        batch: 4, hardware: hardware)
    }
    try? FileManager.default.removeItem(at: tmp)

    // Negative batch is rejected before any allocation.
    checkThrows("forge rejects zero batch", ForgeError.invalidBatch("batch 0 outside 1...\(Limits.maxBatch)")) {
        _ = try Forge.forge(graph: forgeGraph, batch: 0, seed: 42)
    }

    // Candidate generation deduplicates to ACTUAL dispatch widths: a
    // graph whose smallest dense output is 16 collapses 32/64/128/256 to
    // one width of 16.
    let dedup = Forge.candidates(fusable: true, maxThreadsPerThreadgroup: 1024,
                                 minDenseOutDim: 16)
    checkEqual("forge dedups widths to actual dispatch",
               Set(dedup.map { $0.threadgroupWidth }), [16])
    checkEqual("forge dedup keeps fusion variants", dedup.count, 2)

    // Fusion legality: fused output equals unfused output and both match
    // the CPU scalar reference.
    let artifact = try CompiledArtifact(graph: forgeGraph, seed: 42)
    var rng = SplitMix64(seed: 42)
    let input = (0..<(4 * artifact.program.inputDim)).map { _ in rng.nextUniform() }
    let prepared = try MetalBackend.Prepared(
        program: artifact.program, batch: 4, input: input)
    check("graph is dense-relu fusable", MetalBackend.isDenseReluFusable(artifact.program))
    let (unfused, _) = try prepared.run(threadgroupWidth: 32, fuseDenseRelu: false)
    let (fused, _) = try prepared.run(threadgroupWidth: 32, fuseDenseRelu: true)
    let scalar = try ScalarBackend.run(artifact.program, batch: 4, input: input)
    checkEqual("fused/unfused output counts match", fused.count, unfused.count)
    var fusedDiff: Float = 0
    for i in 0..<fused.count {
        fusedDiff = max(fusedDiff, abs(fused[i] - unfused[i]))
    }
    check("fused dense+relu equals unfused within 1e-6", fusedDiff < 1e-6)
    check("fused output matches scalar reference within 1e-4",
          zip(fused, scalar).allSatisfy { abs($0.0 - $0.1) < 1e-4 })
} catch {
    failed += 1; print("FAIL: forge pipeline — threw \(error)")
}

// MARK: - Python model frontend (dict contract) -> Forge end-to-end

do {
    let tmp = FileManager.default.temporaryDirectory
    let graphOut = tmp.appendingPathComponent("leviathan-frontend-graph.json")
    defer { try? FileManager.default.removeItem(at: graphOut) }
    let frontend = Process()
    frontend.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    frontend.currentDirectoryURL = URL(fileURLWithPath: "examples")
    frontend.arguments = ["forge_demo.py", "forge_demo_model.py", graphOut.path]
    let pipe = Pipe()
    frontend.standardOutput = pipe
    frontend.standardError = pipe
    try frontend.run()
    frontend.waitUntilExit()
    let outputText = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                            encoding: .utf8) ?? ""
    if frontend.terminationStatus != 0 { print(outputText) }
    check("python dict frontend runs", frontend.terminationStatus == 0)
    if frontend.terminationStatus == 0 {
        let graph = try JSONDecoder().decode(
            GraphIR.self, from: Data(contentsOf: graphOut))
        // The example model's own architecture (not the synthetic demo).
        checkEqual("python dict frontend produced example graph", graph,
                   GraphIR(inputDim: 64, ops: [
                       .dense(outDim: 128), .relu, .dense(outDim: 64), .relu,
                       .dense(outDim: 8)]))
        let (plan, _, _, _) = try Forge.forge(graph: graph, batch: 4, seed: 42)
        check("python dict frontend graph forges on GPU",
              plan.selectedMedianGpuMicros > 0)
    }
} catch {
    failed += 1; print("FAIL: python dict frontend — threw \(error)")
}

// MARK: - PyTorch exporter -> Forge end-to-end with real weights

func runExporter(_ seed: String, _ prefix: String) -> (ok: Bool, log: String) {
    // Prefer an interpreter that actually has torch installed (the user's
    // Anaconda Python); fall back to python3 on PATH.
    var interpreter = "/opt/anaconda3/bin/python3"
    if !FileManager.default.fileExists(atPath: interpreter) {
        interpreter = "/usr/bin/python3"
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: interpreter)
    process.currentDirectoryURL = URL(fileURLWithPath: "examples")
    process.arguments = ["export_torch.py", "torch_model.py", prefix, seed]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try? process.run()
    process.waitUntilExit()
    let log = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                     encoding: .utf8) ?? ""
    return (process.terminationStatus == 0, log)
}

do {
    let tmp = FileManager.default.temporaryDirectory.path
    let prefixA = tmp + "/leviathan-torchA"
    let prefixB = tmp + "/leviathan-torchB"
    defer {
        for p in [prefixA, prefixB] {
            for ext in ["graph.json", "weights.f32", "manifest.json"] {
                try? FileManager.default.removeItem(atPath: p + "." + ext)
            }
        }
    }
    let exportA = runExporter("0", prefixA)
    if !exportA.ok { print(exportA.log) }
    check("torch exporter seed 0 runs", exportA.ok)
    let exportB = runExporter("1", prefixB)
    if !exportB.ok { print(exportB.log) }
    check("torch exporter seed 1 runs", exportB.ok)
    guard exportA.ok, exportB.ok else { throw ExportFailed() }
    struct ExportFailed: Error {}

    let export = try WeightExport.load(prefix: prefixA)
    checkEqual("torch export input dim", export.graph.inputDim, 64)
    check("torch export weights finite", export.weights.allSatisfy { $0.isFinite })

    // Two seeds must produce different parameter weights and different
    // reference outputs — proof these are real per-model parameters, not
    // a seeded template.
    let other = try WeightExport.load(prefix: prefixB)
    check("different seeds give different weight hashes",
          export.weightsSha256 != other.weightsSha256)
    check("different seeds give different reference outputs",
          export.refOutput != other.refOutput)

    // Parity: run the SELECTED plan candidate with the REAL weights on
    // the manifest's deterministic reference input, and compare that GPU
    // vector against the PyTorch-computed reference output. (forge() itself
    // measures with its own seeded probe input; the plan selection and this
    // parity check use the same program and candidate, different valid
    // input — documented behavior.)
    let (plan, _, hardware, _) = try Forge.forge(
        graph: export.graph, batch: 1, seed: 42, weights: export.weights)
    check("torch-weight forge selected a measured candidate",
          plan.selectedMedianGpuMicros > 0)
    let artifactProgram = LoweredProgram(graph: export.graph, seed: 0)
    let realProgram = LoweredProgram(
        inputDim: artifactProgram.inputDim, outputDim: artifactProgram.outputDim,
        bufferCount: artifactProgram.bufferCount, steps: artifactProgram.steps,
        weights: export.weights)
    let cpuWithRealWeights = try ScalarBackend.run(
        realProgram, batch: 1, input: export.refInput)
    let prepared = try MetalBackend.Prepared(
        program: realProgram, batch: 1, input: export.refInput)
    let (gpuParity, _) = try prepared.run(
        threadgroupWidth: plan.selected.threadgroupWidth,
        fuseDenseRelu: plan.selected.fuseDenseRelu)
    guard gpuParity.count == export.refOutput.count,
          cpuWithRealWeights.count == export.refOutput.count,
          gpuParity.allSatisfy({ $0.isFinite }) else {
        failed += 1
        print("FAIL: torch parity vector counts/finiteness (gpu \(gpuParity.count), "
              + "cpu \(cpuWithRealWeights.count), ref \(export.refOutput.count))")
        throw ExitParity()
    }
    struct ExitParity: Error {}
    var worstTorchDiff: Float = 0
    for i in 0..<gpuParity.count {
        worstTorchDiff = max(worstTorchDiff,
                             abs(gpuParity[i] - export.refOutput[i]) / max(1, abs(export.refOutput[i])))
    }
    print(String(format: "  torch parity: worst rel diff GPU vs PyTorch %.3e (scalar %.3e)",
                 worstTorchDiff,
                 zip(cpuWithRealWeights, export.refOutput)
                     .map { abs($0.0 - $0.1) / max(1, abs($0.1)) }.max() ?? -1))
    check("GPU output with real torch weights matches PyTorch reference (rel 1e-4)",
          worstTorchDiff < 1e-4)

    // Plan reuse keyed by weight content: same weights validate, other
    // weights are a different model and must be rejected.
    let planURL = URL(fileURLWithPath: tmp + "/leviathan-torch.forge.json")
    try plan.save(to: planURL)
    let reused = try ForgePlan.loadValidated(
        from: planURL, graph: export.graph, seed: 42, batch: 1,
        hardware: hardware, weights: export.weights)
    checkEqual("torch-weight plan cache hit", reused, plan)
    checkThrows("torch-weight plan rejected for different weights",
                ForgeError.invalidPlan("graph digest mismatch — plan is for another model")) {
        _ = try ForgePlan.loadValidated(
            from: planURL, graph: export.graph, seed: 42, batch: 1,
            hardware: hardware, weights: other.weights)
    }
    try? FileManager.default.removeItem(at: planURL)

    // A tampered weight byte is caught by the manifest content hash.
    let weightsURL = URL(fileURLWithPath: prefixA + ".weights.f32")
    var blob = try Data(contentsOf: weightsURL)
    blob[blob.count - 1] ^= 0xFF
    try blob.write(to: weightsURL)
    checkThrows("tampered weight blob rejected by content hash",
                WeightExportError.hashMismatch(
                    expected: export.weightsSha256,
                    actual: SHA256.hex(SHA256.hash([UInt8](blob))))) {
        _ = try WeightExport.load(prefix: prefixA)
    }
} catch {
    failed += 1; print("FAIL: torch exporter e2e — threw \(error)")
}

// MARK: - torch.export(strict=True) frontend -> Forge end-to-end
// (custom-forward nn.Module, captured ATen graph, fail-closed subset)

func runStrictExporter(_ modelPath: String, _ prefix: String, _ seed: String)
        -> (ok: Bool, log: String) {
    // Same interpreter policy as the Sequential exporter above. The pipe is
    // drained BEFORE waitUntilExit so an exporter that floods more than the
    // pipe capacity (e.g. a long torch traceback) cannot deadlock, and a
    // launch failure is reported instead of crashing the suite.
    var interpreter = "/opt/anaconda3/bin/python3"
    if !FileManager.default.fileExists(atPath: interpreter) {
        interpreter = "/usr/bin/python3"
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: interpreter)
    process.currentDirectoryURL = URL(fileURLWithPath: ".")
    process.arguments = ["tools/torch_frontend/export_torch_graph.py",
                         modelPath, prefix, seed]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do {
        try process.run()
    } catch {
        return (false, "failed to launch \(interpreter): \(error)")
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let log = String(data: data, encoding: .utf8) ?? ""
    return (process.terminationStatus == 0, log)
}

do {
    // Own unique scratch directory so this test never collides with a
    // concurrent run or another suite's leftovers; cleaned up as a whole.
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("leviathan-strict-torch-\(UUID().uuidString)",
                                isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let tmp = dir.path
    let prefix = tmp + "/model"
    let negPrefix = tmp + "/neg"
    let exported = runStrictExporter(
        "examples/torch_export/custom_forward_mlp.py", prefix, "0")
    if !exported.ok { print(exported.log) }
    check("strict torch.export frontend runs", exported.ok)

    // Fail-closed: an unsupported ATen op (torch.sin) is rejected and
    // leaves no output files behind.
    let rejected = runStrictExporter(
        "examples/torch_export/negatives/neg_sin.py", negPrefix, "0")
    let leftBehind = ["graph.json", "weights.f32", "manifest.json"]
        .filter { FileManager.default.fileExists(atPath: negPrefix + "." + $0) }
    check("strict frontend rejects unsupported op (sin)", !rejected.ok)
    check("rejected export wrote no files", leftBehind.isEmpty)

    guard exported.ok else { throw StrictFailed() }
    struct StrictFailed: Error {}

    let strict = try WeightExport.load(prefix: prefix)
    checkEqual("strict export input dim", strict.graph.inputDim, 64)
    checkEqual("strict export dense op count",
               strict.graph.ops.filter {
                   if case .dense = $0 { return true }; return false
               }.count, 4)
    check("strict export weights finite",
          strict.weights.allSatisfy { $0.isFinite })

    // The graph is captured from forward semantics: the module owns FIVE
    // child Linear modules (one never called in forward), the exported
    // graph has exactly four dense ops.
    let (plan, _, hardware, _) = try Forge.forge(
        graph: strict.graph, batch: 1, seed: 42, weights: strict.weights)
    check("strict torch-weight forge selected a measured candidate",
          plan.selectedMedianGpuMicros > 0)
    let artifactProgram = LoweredProgram(graph: strict.graph, seed: 0)
    let realProgram = LoweredProgram(
        inputDim: artifactProgram.inputDim, outputDim: artifactProgram.outputDim,
        bufferCount: artifactProgram.bufferCount, steps: artifactProgram.steps,
        weights: strict.weights)
    let prepared = try MetalBackend.Prepared(
        program: realProgram, batch: 1, input: strict.refInput)
    let (gpuOut, _) = try prepared.run(
        threadgroupWidth: plan.selected.threadgroupWidth,
        fuseDenseRelu: plan.selected.fuseDenseRelu)
    guard gpuOut.count == strict.refOutput.count,
          gpuOut.allSatisfy({ $0.isFinite }) else {
        failed += 1
        print("FAIL: strict torch parity vector counts/finiteness")
        throw StrictParity()
    }
    struct StrictParity: Error {}
    var worstStrict: Float = 0
    for i in 0..<gpuOut.count {
        worstStrict = max(worstStrict,
                          abs(gpuOut[i] - strict.refOutput[i]) / max(1, abs(strict.refOutput[i])))
    }
    print(String(format: "  strict torch.export parity: worst rel diff GPU vs PyTorch %.3e",
                 worstStrict))
    check("strict torch.export GPU output matches PyTorch module (rel 1e-4)",
          worstStrict < 1e-4)

    // Persisted plan reuse keyed on the same validated weights.
    let planURL = URL(fileURLWithPath: tmp + "/leviathan-strict-torch.forge.json")
    try plan.save(to: planURL)
    let reused = try ForgePlan.loadValidated(
        from: planURL, graph: strict.graph, seed: 42, batch: 1,
        hardware: hardware, weights: strict.weights)
    checkEqual("strict torch-weight plan cache hit", reused, plan)
    try? FileManager.default.removeItem(at: planURL)
} catch {
    failed += 1
    print("FAIL: strict torch.export frontend e2e — threw \(error)")
}

// MARK: - Gemma 4 math primitives (stable tanh on GPU vs scalar)

do {
    // Cover the exact region where Metal's built-in tanh returns NaN on
    // the tested driver (|x| > ~49), plus zero and negative arguments.
    var args: [Float] = [0, 1e-8, -1e-8, 0.5, -0.5, 1, -1, 5, -5, 10, -10,
                         30, -30, 49, -49, 49.7, -49.7, 100, -100, 500, -500,
                         1e4, -1e4, 1e6, -1e6, 30.5, 30.25]
    args.append(contentsOf: (0..<32).map { Float($0) * 3.7 - 60 })
    let gpu = try Gemma4Math.stableTanhGPU(args)
    checkEqual("stable tanh GPU result count", gpu.count, args.count)
    var worst = 0.0
    var allFinite = true
    for i in args.indices {
        allFinite = allFinite && gpu[i].isFinite
        let want = tanh(Double(args[i]))
        worst = max(worst, abs(Double(gpu[i]) - want))
    }
    check("stable tanh GPU all finite (built-in tanh NaNs >~49)", allFinite)
    check(String(format: "stable tanh GPU vs scalar fp64 (worst %.2e)", worst), worst < 1e-6)
}

// MARK: - Gemma 4 tensor contract (adversarial synthetic GGUFs)

/// Minimal original GGUF byte builder for contract tests: only the
/// metadata Gemma4Config reads, plus an explicit tensor directory.
struct GGUFTestBuilder {
    var bytes = [UInt8]()

    init() {
        bytes.append(contentsOf: Array("GGUF".utf8))
        appendU32(3)          // version
        appendU64(0)          // tensor count (patched by finish)
        appendU64(0)          // kv count (patched by finish)
    }
    var kvCount = 0
    var tensorCount = 0

    mutating func appendU32(_ v: UInt32) {
        bytes.append(UInt8(v & 0xFF)); bytes.append(UInt8((v >> 8) & 0xFF))
        bytes.append(UInt8((v >> 16) & 0xFF)); bytes.append(UInt8((v >> 24) & 0xFF))
    }
    mutating func appendU64(_ v: UInt64) {
        for i in 0..<8 { bytes.append(UInt8((v >> (8 * UInt64(i))) & 0xFF)) }
    }
    mutating func appendString(_ s: String) {
        appendU64(UInt64(s.utf8.count))
        bytes.append(contentsOf: Array(s.utf8))
    }
    mutating func kvString(_ key: String, _ value: String) {
        appendString(key); appendU32(8); appendString(value); kvCount += 1
    }
    mutating func kvU32(_ key: String, _ value: UInt32) {
        appendString(key); appendU32(4); appendU32(value); kvCount += 1
    }
    mutating func kvF32(_ key: String, _ value: Float) {
        appendString(key); appendU32(6); appendU32(value.bitPattern); kvCount += 1
    }
    mutating func kvInt32Array(_ key: String, _ values: [Int32]) {
        appendString(key); appendU32(9); appendU32(5); appendU64(UInt64(values.count))
        for v in values { appendU32(UInt32(bitPattern: v)) }
        kvCount += 1
    }
    mutating func kvStringArray(_ key: String, _ values: [String]) {
        appendString(key); appendU32(9); appendU32(8); appendU64(UInt64(values.count))
        for v in values { appendString(v) }
        kvCount += 1
    }
    mutating func kvF32Array(_ key: String, _ values: [Float]) {
        appendString(key); appendU32(9); appendU32(6); appendU64(UInt64(values.count))
        for v in values { appendU32(v.bitPattern) }
        kvCount += 1
    }
    mutating func kvBool(_ key: String, _ value: Bool) {
        appendString(key); appendU32(7); bytes.append(value ? 1 : 0); kvCount += 1
    }
    mutating func tensor(_ name: String, dims: [Int], type: UInt32, offset: Int) {
        appendString(name)
        appendU32(UInt32(dims.count))
        for d in dims { appendU64(UInt64(d)) }
        appendU32(type)
        appendU64(UInt64(offset))
        tensorCount += 1
    }
    /// Pads the header to 32-byte alignment and appends a tiny data region.
    mutating func finish(totalBytes: Int = 64) -> Data {
        while bytes.count % 32 != 0 { bytes.append(0) }
        // tensor count / kv count were written at offsets 8 and 16.
        func patch64(_ index: Int, _ v: UInt64, into b: inout [UInt8]) {
            for i in 0..<8 { b[index + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
        }
        patch64(8, UInt64(tensorCount), into: &bytes)
        patch64(16, UInt64(kvCount), into: &bytes)
        var out = bytes
        out.append(contentsOf: [UInt8](repeating: 0xAA, count: totalBytes))
        return Data(out)
    }

    /// The full valid metadata set for the 26B-A4B layout.
    mutating func appendGemma4Metadata(blockCount: UInt32 = 30,
                                       kvHeadsOverride: [Int32]? = nil) {
        kvString("general.architecture", "gemma4")
        kvU32("gemma4.block_count", blockCount)
        kvU32("gemma4.embedding_length", 2816)
        kvU32("gemma4.expert_count", 128)
        kvU32("gemma4.expert_used_count", 8)
        kvU32("gemma4.expert_feed_forward_length", 704)
        kvU32("gemma4.feed_forward_length", 2112)
        kvF32("gemma4.final_logit_softcapping", 30)
        kvU32("gemma4.context_length", 262144)
        var kvHeads: [Int32] = []
        for layer in 0..<30 { kvHeads.append(layer % 6 == 5 ? 2 : 8) }
        kvInt32Array("gemma4.attention.head_count_kv", kvHeadsOverride ?? kvHeads)
    }
}

do {
    // Valid metadata, but an empty tensor directory: the config parses and
    // the full-contract validator must reject the missing tensor set.
    var b = GGUFTestBuilder()
    b.appendGemma4Metadata()
    let data = b.finish()
    let file = try GGUFReader.parse(data)
    let config = try Gemma4Config(file: file)
    checkEqual("synthetic config global layers", config.globalLayers, [5, 11, 17, 23, 29])
    checkThrowsCase("empty tensor directory rejected by full contract",
                    "incompatibleModel") {
        try Gemma4TensorContract.validate(file: file, config: config)
    }
    // The expectations table itself carries the exact geometry contract.
    let table = Gemma4TensorContract.expectations(config: config)
    checkEqual("contract table size", table.count, 658)
    func entry(_ name: String) -> Gemma4TensorContract.ExpectedTensor {
        table.first { $0.name == name }!
    }
    checkEqual("contract blk.0.attn_q (local)",
               entry("blk.0.attn_q.weight").dims, [2816, 4096])
    checkEqual("contract blk.0.attn_q type", entry("blk.0.attn_q.weight").type, .q4_0)
    checkEqual("contract blk.0.attn_v present (local k/v separate)",
               entry("blk.0.attn_v.weight").dims, [2816, 2048])
    checkEqual("contract blk.5.attn_q (global)",
               entry("blk.5.attn_q.weight").dims, [2816, 8192])
    checkEqual("contract blk.5.attn_k (global 2x512)",
               entry("blk.5.attn_k.weight").dims, [2816, 1024])
    check("contract has no blk.5.attn_v (k_eq_v)",
          !table.contains { $0.name == "blk.5.attn_v.weight" })
    checkEqual("contract token_embd", entry("token_embd.weight").dims, [2816, 262144])
    checkEqual("contract token_embd type", entry("token_embd.weight").type, .q6_K)
    checkEqual("contract experts",
               entry("blk.0.ffn_gate_up_exps.weight").dims, [2816, 1408, 128])
    checkEqual("contract expert down",
               entry("blk.0.ffn_down_exps.weight").dims, [704, 2816, 128])
} catch {
    failed += 1; print("FAIL: synthetic contract test — threw \(error)")
}

do {
    // Wrong metadata (block_count 29): rejected before any tensor work.
    var b = GGUFTestBuilder()
    b.appendGemma4Metadata(blockCount: 29)
    let file = try GGUFReader.parse(b.finish())
    checkThrowsCase("wrong block_count rejected", "badMetadata") {
        _ = try Gemma4Config(file: file)
    }
} catch {
    failed += 1; print("FAIL: wrong block_count test — threw \(error)")
}

do {
    // A crafted GGUF with an out-of-contract kv-head pattern (3 heads on a
    // local layer) is rejected at config parse.
    var b = GGUFTestBuilder()
    b.appendGemma4Metadata(kvHeadsOverride: (0..<30).map { _ in Int32(8) })
    let file = try GGUFReader.parse(b.finish())
    checkThrowsCase("kv-head pattern without global layers rejected",
                    "incompatibleModel") {
        _ = try Gemma4Config(file: file)
    }
} catch {
    failed += 1; print("FAIL: kv-head pattern test — threw \(error)")
}

// MARK: - Gemma tokenizer malformed-metadata rejection (crafted fixtures)

do {
    // Minimal but CONTRACT-VALID gemma4 tokenizer metadata: the full
    // 256-token byte-fallback set plus specials and pieces. Fields are
    // parameterized once (no duplicate GGUF keys).
    // `extraTokens`/`extraTypes` append AFTER the 256 byte tokens +
    // <bos>/<eos>/<unk>; `merges` are real ranked merge rules.
    func tokenizerFixture(
        extraTokens: [String] = ["hello", "world"],
        extraTypes: [Int32] = [1, 1],
        merges: [String] = [],
        addBOS: Bool = true, addSpacePrefix: Bool = false,
        bosID: UInt32 = 256, removeByte41: Bool = false
    ) -> Data {
        var b = GGUFTestBuilder()
        var tokens = byteTokens()
        var types = [Int32](repeating: 6, count: 256)   // .byte
        if removeByte41 {
            tokens.remove(at: 0x41)
            types.remove(at: 0x41)
        }
        tokens += ["<bos>", "<eos>", "<unk>"] + extraTokens
        types += [3, 3, 2] + extraTypes
        b.kvString("tokenizer.ggml.model", "gemma4")
        b.kvStringArray("tokenizer.ggml.tokens", tokens)
        b.kvInt32Array("tokenizer.ggml.token_type", types)
        b.kvF32Array("tokenizer.ggml.scores", [Float](repeating: 0, count: tokens.count))
        b.kvStringArray("tokenizer.ggml.merges", merges)
        b.kvU32("tokenizer.ggml.bos_token_id", bosID)
        b.kvU32("tokenizer.ggml.eos_token_id", 257)
        b.kvU32("tokenizer.ggml.unknown_token_id", 258)
        b.kvBool("tokenizer.ggml.add_bos_token", addBOS)
        b.kvBool("tokenizer.ggml.add_space_prefix", addSpacePrefix)
        return b.finish()
    }
    func byteTokens() -> [String] { (0..<256).map { String(format: "<0x%02X>", $0) } }

    // Valid: "hello" has NO merge chain, so BPE falls back to bytes:
    // [BOS, 'h','e','l','l','o'] byte tokens — the exact llama.cpp
    // byte-fallback behavior this contract mandates.
    let tok = try GemmaTokenizer(gguf: GGUFReader.parse(tokenizerFixture()))
    checkEqual("crafted tokenizer encode hello (byte fallback)",
               try tok.encode("hello"), [256, 104, 101, 108, 108, 111])
    checkEqual("crafted tokenizer decode round-trip (BOS control dropped)",
               try tok.decode([256, 104, 101, 108, 108, 111]), "hello")
    checkEqual("crafted tokenizer byte decode", try tok.decode([0x41]), "A")
    check("crafted tokenizer EOG at eos", tok.isEndOfGeneration(257))
    check("crafted tokenizer not EOG at normal", !tok.isEndOfGeneration(259))
    // Merge ranks are honored: with the "l l" rule and the piece "ll" in
    // vocab, the merged symbol resolves to its vocab id.
    let tok2 = try GemmaTokenizer(gguf: GGUFReader.parse(
        tokenizerFixture(extraTokens: ["hello", "world", "ll"],
                         extraTypes: [1, 1, 1], merges: ["l l"])))
    checkEqual("crafted tokenizer merge rule applied",
               try tok2.encode("hello"), [256, 104, 101, 261, 111])
    // Malformed metadata must be REJECTED at init, never silently fixed.
    do {
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(
            tokenizerFixture(extraTokens: ["hello", "world", "<0xZZ>"],
                             extraTypes: [1, 1, 6])))
        failed += 1; print("FAIL: malformed byte syntax accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer malformed byte syntax rejected",
              "\(e)".contains("inconsistent"))
    }
    do {
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(tokenizerFixture(removeByte41: true)))
        failed += 1; print("FAIL: missing byte token accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer missing byte token rejected",
              "\(e)".contains("byte-fallback") || "\(e)".contains("<0x41>"))
    }
    do {
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(tokenizerFixture(addBOS: false)))
        failed += 1; print("FAIL: add_bos=false accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer add_bos=false rejected", "\(e)".contains("add_bos"))
    }
    do {
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(
            tokenizerFixture(addSpacePrefix: true)))
        failed += 1; print("FAIL: add_space_prefix=true accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer add_space_prefix=true rejected",
              "\(e)".contains("add_space_prefix"))
    }
    do {
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(
            tokenizerFixture(bosID: 999)))
        failed += 1; print("FAIL: out-of-range bos accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer out-of-range bos rejected", "\(e)".contains("bos/eos"))
    }
    // Unknown token_type code (9) across the array.
    do {
        var types9 = [Int32](repeating: 9, count: 256 + 3 + 2)
        _ = types9
        // Reuse the fixture with an all-9 type array via a tiny builder:
        var b = GGUFTestBuilder()
        let tokens = byteTokens() + ["<bos>", "<eos>", "<unk>", "hello", "world"]
        b.kvString("tokenizer.ggml.model", "gemma4")
        b.kvStringArray("tokenizer.ggml.tokens", tokens)
        b.kvInt32Array("tokenizer.ggml.token_type",
                       [Int32](repeating: 9, count: tokens.count))
        b.kvF32Array("tokenizer.ggml.scores", [Float](repeating: 0, count: tokens.count))
        b.kvStringArray("tokenizer.ggml.merges", [])
        b.kvU32("tokenizer.ggml.bos_token_id", 256)
        b.kvU32("tokenizer.ggml.eos_token_id", 257)
        b.kvU32("tokenizer.ggml.unknown_token_id", 258)
        b.kvBool("tokenizer.ggml.add_bos_token", true)
        _ = try GemmaTokenizer(gguf: GGUFReader.parse(b.finish()))
        failed += 1; print("FAIL: unknown token_type accepted")
    } catch let e as GemmaTokenizerError {
        check("tokenizer unknown token_type rejected", "\(e)".contains("token_type"))
    }
} catch {
    failed += 1; print("FAIL: crafted tokenizer fixture — threw \(error)")
}

// MARK: - Gemma graph mutation rejection (captured graph governs execution)

do {
    // The captured graph is the ONLY thing the graph executor will run:
    // public executeGraph/executeGraphContinue call GemmaGraph.validate
    // against the real config + tensor directory before any kernel, so
    // any mutation of the captured IR is rejected before GPU access.
    // These tests prove rejection of every mutation class.
    var b = GGUFTestBuilder()
    b.appendGemma4Metadata()
    let file = try GGUFReader.parse(b.finish())
    let cfg = try Gemma4Config(file: file)
    let gc = GemmaGraphIR.GraphConfig(
        layerCount: cfg.layerCount, hidden: cfg.hiddenSize, qHeads: cfg.qHeads,
        localKVHeads: cfg.localKVHeads, globalKVHeads: cfg.globalKVHeads,
        localHeadDim: cfg.localHeadDim, globalHeadDim: cfg.globalHeadDim,
        slidingWindow: cfg.slidingWindow, vocabSize: cfg.vocabSize,
        expertCount: cfg.expertCount, expertUsedCount: cfg.topKExperts,
        expertIntermediate: cfg.expertIntermediate,
        sharedIntermediate: cfg.sharedIntermediate,
        finalSoftcap: cfg.finalSoftcap, globalLayers: cfg.globalLayers)
    let expected = GemmaGraph.expectedSequence(gc)
    checkEqual("expected op count 934", expected.count, 934)
    func graph(_ ops: [GemmaGraphIR.Op], schema: Int = 1,
               model: String = "gemma4-26b-a4b",
               config: GemmaGraphIR.GraphConfig = gc) -> GemmaGraphIR {
        GemmaGraphIR(schema: schema, model: model, config: config, ops: ops)
    }
    // The synthetic file has NO tensors: the exact expected sequence passes
    // the full structural pass and is rejected at tensor-binding existence.
    do {
        try GemmaGraph.validate(graph(expected), config: cfg, file: file)
        failed += 1; print("FAIL: tensorless file accepted")
    } catch let e as GemmaGraphError {
        if case .missingTensor = e { passed += 1 }
        else { failed += 1; print("FAIL: valid sequence on tensorless file — wrong case \(e)") }
    } catch {
        failed += 1; print("FAIL: valid sequence on tensorless file — \(error)")
    }
    func expectGraphError(_ name: String, _ isCase: (GemmaGraphError) -> Bool,
                          _ body: () throws -> Void) {
        do {
            try body()
            failed += 1; print("FAIL: \(name) — no error thrown")
        } catch let e as GemmaGraphError {
            if isCase(e) {
                passed += 1
            } else {
                failed += 1; print("FAIL: \(name) — wrong case: \(e)")
            }
        } catch {
            failed += 1; print("FAIL: \(name) — wrong error type: \(error)")
        }
    }
    // Dropped op (structural pass complete before tensor checks).
    expectGraphError("dropped op rejected", { if case .badSequence = $0 { true } else { false } }) {
        try GemmaGraph.validate(graph(Array(expected.dropLast())), config: cfg, file: file)
    }
    // Reordered ops.
    expectGraphError("reordered ops rejected", { if case .badSequence = $0 { true } else { false } }) {
        var ops = expected; ops.swapAt(0, ops.count - 1)
        try GemmaGraph.validate(graph(ops), config: cfg, file: file)
    }
    // Unknown op string.
    expectGraphError("unknown op rejected", { if case .unsupportedOp = $0 { true } else { false } }) {
        var ops = expected; ops[0] = GemmaGraphIR.Op(op: "conv2d")
        try GemmaGraph.validate(graph(ops), config: cfg, file: file)
    }
    // Wrong layer index on an inner op.
    expectGraphError("wrong layer index rejected", { if case .badSequence = $0 { true } else { false } }) {
        var ops = expected
        ops[1] = GemmaGraphIR.Op(op: "attn_norm", layer: 7,
                                 tensor: "blk.7.attn_norm.weight")
        try GemmaGraph.validate(graph(ops), config: cfg, file: file)
    }
    // Config mismatch (checked before sequence/tensor validation).
    expectGraphError("config mismatch rejected", { if case .configMismatch = $0 { true } else { false } }) {
        let wrong = GemmaGraphIR.GraphConfig(
            layerCount: 29, hidden: gc.hidden, qHeads: gc.qHeads,
            localKVHeads: gc.localKVHeads, globalKVHeads: gc.globalKVHeads,
            localHeadDim: gc.localHeadDim, globalHeadDim: gc.globalHeadDim,
            slidingWindow: gc.slidingWindow, vocabSize: gc.vocabSize,
            expertCount: gc.expertCount, expertUsedCount: gc.expertUsedCount,
            expertIntermediate: gc.expertIntermediate,
            sharedIntermediate: gc.sharedIntermediate,
            finalSoftcap: gc.finalSoftcap, globalLayers: gc.globalLayers)
        try GemmaGraph.validate(graph(expected, config: wrong), config: cfg, file: file)
    }
    // Schema and model tag.
    expectGraphError("wrong schema rejected", { if case .badJSON = $0 { true } else { false } }) {
        try GemmaGraph.validate(graph(expected, schema: 2), config: cfg, file: file)
    }
    expectGraphError("wrong model tag rejected", { if case .badJSON = $0 { true } else { false } }) {
        try GemmaGraph.validate(graph(expected, model: "other"), config: cfg, file: file)
    }
    // Digest: deterministic and mutation-sensitive.
    let d1 = try GemmaGraph.digest(graph(expected))
    checkEqual("graph digest stable", try GemmaGraph.digest(graph(expected)), d1)
    var ops = expected
    ops[1] = GemmaGraphIR.Op(op: "attn_norm", layer: 1,
                             tensor: "blk.1.attn_norm.weight")
    if let d2 = try? GemmaGraph.digest(graph(ops)) {
        check("graph digest changes under attribute mutation", d2 != d1)
    } else {
        failed += 1; print("FAIL: digest mutation threw")
    }

    // Typed SSA: the lowered schedule for the expected graph is valid;
    // adversarially mutated schedules (rewired/stale/undefined operands,
    // shape edits, unknown ops, non-logits terminals) are rejected by
    // dataflow validation — the dataflow, not stage names, governs what
    // the executor may run. Mutations go through the JSON representation
    // (also exercising the parse path).
    let ssa: GemmaSSAIR
    do {
        ssa = try GemmaSSA.lower(graph(expected))
    } catch {
        failed += 1; print("FAIL: SSA lowering of expected graph — \(error)")
        ssa = GemmaSSAIR(schema: 1, values: [], instructions: [])
    }
    checkEqual("ssa instruction count == stage ops", ssa.instructions.count, expected.count)
    do {
        let lowered2 = try GemmaSSA.lower(graph(expected))
        checkEqual("ssa lowering deterministic", lowered2, ssa)
        checkEqual("ssa digest stable", try GemmaSSA.digest(lowered2),
                   try GemmaSSA.digest(ssa))
        // Prepared value-table gate (pure): one entry per distinct value id,
        // and the table maps every id to its exact SSA value.
        let table = GemmaPreparedExecution.valueTable(ssa)
        checkEqual("prepared value table size", table.count,
                   Set(ssa.values.map(\.id)).count)
        check("prepared value table entries exact",
              ssa.values.allSatisfy { table[$0.id] == $0 })
    } catch {
        failed += 1; print("FAIL: SSA determinism/digest — \(error)")
    }
    func mutatedSSA(_ mutate: (inout [String: Any]) -> Void) -> GemmaSSAIR? {
        guard let data = try? JSONEncoder().encode(ssa),
              var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        mutate(&obj)
        guard let out = try? JSONSerialization.data(withJSONObject: obj),
              let back = try? JSONDecoder().decode(GemmaSSAIR.self, from: out)
        else { return nil }
        return back
    }
    func expectSSAReject(_ name: String, _ contains: String,
                         _ mutate: @escaping (inout [String: Any]) -> Void) {
        guard let bad = mutatedSSA(mutate) else {
            failed += 1; print("FAIL: \(name) — mutation unrepresentable")
            return
        }
        do {
            try GemmaSSA.validate(bad, config: gc)
            failed += 1; print("FAIL: \(name) — mutated SSA accepted")
        } catch let e as GemmaSSAError {
            check(name, "\(e)".contains(contains))
        } catch {
            failed += 1; print("FAIL: \(name) — wrong error \(error)")
        }
    }
    expectSSAReject("ssa undefined operand rejected", "undefined") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if (i["op"] as? String) == "matvec", var ins = i["inputs"] as? [String] {
                ins[0] = "x@999"; i["inputs"] = ins
            }
            return i
        }
    }
    expectSSAReject("ssa stale version read rejected", "non-live") { obj in
        // Rewire the FINAL layer's rmsnorm input to an ancient x version.
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if (i["op"] as? String) == "rmsnorm", var ins = i["inputs"] as? [String],
               ins[0].hasPrefix("x@") { ins[0] = "x@0"; i["inputs"] = ins }
            return i
        }
    }
    expectSSAReject("ssa shape mutation rejected", "length") { obj in
        obj["values"] = (obj["values"] as! [[String: Any]]).map { v in
            var w = v
            if (w["name"] as? String) == "h" { w["length"] = 12345 }
            return w
        }
    }
    expectSSAReject("ssa unknown op rejected", "unknown ssa op") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if (i["op"] as? String) == "copy" { i["op"] = "conv2d" }
            return i
        }
    }
    expectSSAReject("ssa non-logits terminal rejected", "logits") { obj in
        var insts = obj["instructions"] as! [[String: Any]]
        insts.removeLast()
        obj["instructions"] = insts
    }
    expectSSAReject("ssa duplicate value id rejected", "duplicate") { obj in
        var vals = obj["values"] as! [[String: Any]]
        vals.append(vals[0])
        obj["values"] = vals
    }
    expectSSAReject("ssa missing cache attrs rejected", "cache_append") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if i["op"] as? String == "cache_append", var a = i["attrs"] as? [String: Int] {
                a.removeValue(forKey: "kvHeads"); i["attrs"] = a
            }
            return i
        }
    }
    expectSSAReject("ssa absurd dimension rejected", "bad value") { obj in
        obj["values"] = (obj["values"] as! [[String: Any]]).map { v in
            var w = v
            if (w["name"] as? String) == "h" { w["length"] = Int.max }
            return w
        }
    }
    expectSSAReject("ssa bad kind rejected", "name/kind") { obj in
        obj["values"] = (obj["values"] as! [[String: Any]]).map { v in
            var w = v
            if (w["name"] as? String) == "h" { w["kind"] = "routing" }
            return w
        }
    }
    // Bad dtype makes the artifact JSON UNDECODABLE — also a rejection
    // (at parse, before validation even runs).
    do {
        if let data = try? JSONEncoder().encode(ssa),
           var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            obj["values"] = (obj["values"] as! [[String: Any]]).map { v in
                var w = v
                if (w["name"] as? String) == "x" { w["dtype"] = "f16" }
                return w
            }
            let out = try JSONSerialization.data(withJSONObject: obj)
            if (try? JSONDecoder().decode(GemmaSSAIR.self, from: out)) != nil {
                failed += 1; print("FAIL: ssa bad dtype decoded")
            } else {
                passed += 1
            }
        } else {
            failed += 1; print("FAIL: ssa dtype mutation unrepresentable")
        }
    } catch {
        failed += 1; print("FAIL: ssa dtype check — \(error)")
    }
    expectSSAReject("ssa layer out of range rejected", "layer") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if i["op"] as? String == "attention" { i["layer"] = 999 }
            return i
        }
    }
    expectSSAReject("ssa Int.max rmsnorm vectors attr rejected",
                   "vectors/width") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if i["op"] as? String == "rmsnorm", var a = i["attrs"] as? [String: Int],
               a["vectors"] != nil {
                a["vectors"] = Int.max; i["attrs"] = a
            }
            return i
        }
    }
    expectSSAReject("ssa Int.max rope heads attr rejected", "rope") { obj in
        obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
            var i = inst
            if i["op"] as? String == "rope", var a = i["attrs"] as? [String: Int] {
                a["heads"] = Int.max; i["attrs"] = a
            }
            return i
        }
    }
    expectSSAReject("ssa gate/up slice interference rejected", "gelu_up") { obj in
        obj["values"] = (obj["values"] as! [[String: Any]]).map { v in
            var w = v
            if (w["name"] as? String) == "up" { w["sliceOffset"] = 0 }
            return w
        }
    }
    do {
        let d1ssa = try GemmaSSA.digest(ssa)
        if let bad = mutatedSSA({ obj in
            obj["instructions"] = (obj["instructions"] as! [[String: Any]]).map { inst in
                var i = inst
                if (i["op"] as? String) == "rope", var a = i["attrs"] as? [String: Int] {
                    a["heads"] = 3; i["attrs"] = a
                }
                return i
            }
        }), let dBad = try? GemmaSSA.digest(bad) {
            check("ssa digest changes under mutation", dBad != d1ssa)
        } else {
            failed += 1; print("FAIL: ssa digest mutation unrepresentable")
        }
    } catch {
        failed += 1; print("FAIL: ssa digest check — \(error)")
    }
} catch {
    failed += 1; print("FAIL: graph mutation test — threw \(error)")
}

// MARK: - Plan artifact identity validation (validatePlanIdentity)
do {
    var b = GGUFTestBuilder()
    b.appendGemma4Metadata()
    let file = try GGUFReader.parse(b.finish())
    let cfg = try Gemma4Config(file: file)
    let gc = GemmaGraphIR.GraphConfig(
        layerCount: cfg.layerCount, hidden: cfg.hiddenSize, qHeads: cfg.qHeads,
        localKVHeads: cfg.localKVHeads, globalKVHeads: cfg.globalKVHeads,
        localHeadDim: cfg.localHeadDim, globalHeadDim: cfg.globalHeadDim,
        slidingWindow: cfg.slidingWindow, vocabSize: cfg.vocabSize,
        expertCount: cfg.expertCount, expertUsedCount: cfg.topKExperts,
        expertIntermediate: cfg.expertIntermediate,
        sharedIntermediate: cfg.sharedIntermediate,
        finalSoftcap: cfg.finalSoftcap, globalLayers: cfg.globalLayers)
    let graph = GemmaGraphIR(schema: 1, model: "gemma4-26b-a4b", config: gc,
                             ops: GemmaGraph.expectedSequence(gc))
    let graphDigest = try GemmaGraph.digest(graph)
    let gguf = String(repeating: "ab", count: 32)   // valid 64-hex
    let engine = String(repeating: "cd", count: 32)
    let hw = ForgeHardwareProfile(deviceName: "t",
                                  maxTotalThreadsPerThreadgroup: 1024,
                                  threadExecutionWidth: 32,
                                  recommendedMaxWorkingSetSize: 1 << 20)
    let results = GemmaForge.candidates().map {
        GemmaForgeCandidateResult(candidate: $0, correct: true,
                                  medianWallMicrosPerToken: 1,
                                  medianGpuMicrosPerToken: 1, note: "test")
    }
    let selected = results[0].candidate
    let plan = try GemmaForge.makePlan(
        graph: graph, graphDigest: graphDigest, ggufDigest: gguf,
        engineDigest: engine, hardware: hw, prompt: [2, 9259],
        decodeTokens: 4, selected: selected, medianWall: 1, medianGpu: 1,
        results: results)
    // A mutation helper that FAILS the test when the JSON round-trip or
    // the requested edit itself breaks (a failed decode must never be
    // mistaken for a validator rejection).
    func expectMutated(_ name: String,
                       _ body: (inout [String: Any]) -> Void) -> GemmaForgePlan? {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(plan),
              var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            failed += 1; print("FAIL: \(name) — plan re-encode failed")
            return nil
        }
        body(&obj)
        guard JSONSerialization.isValidJSONObject(obj),
              let d2 = try? JSONSerialization.data(withJSONObject: obj),
              let decoded = try? JSONDecoder().decode(GemmaForgePlan.self, from: d2)
        else {
            failed += 1; print("FAIL: \(name) — mutated plan did not decode")
            return nil
        }
        guard decoded != plan else {
            failed += 1; print("FAIL: \(name) — mutation did not change the plan")
            return nil
        }
        // A planDigest-only mutation must keep its STALE digest — that
        // is the property under test; re-signing would restore validity.
        if name == "planDigest" { return decoded }
        // Re-sign everything else so planDigest staleness cannot mask
        // the structural gate under test.
        guard let resigned = try? GemmaForge.resignPlan(decoded) else {
            failed += 1; print("FAIL: \(name) — resign failed")
            return nil
        }
        return resigned
    }
    func rejected(_ name: String, _ p: GemmaForgePlan?) {
        guard let p else { return }   // already failed inside expectMutated
        if (try? GemmaForge.validatePlanIdentity(
                p, graph: graph, ggufDigest: gguf, engineDigest: engine,
                hardware: hw)) != nil {
            failed += 1; print("FAIL: \(name) — mutated plan accepted")
        } else {
            passed += 1
        }
    }
    check("plan identity validates", (try? GemmaForge.validatePlanIdentity(
        plan, graph: graph, ggufDigest: gguf, engineDigest: engine,
        hardware: hw)) != nil)
    // Pure config-fingerprint table check: stable for the same config and
    // sensitive to a layer-count change (structural, no hashing/GPU).
    checkEqual("config fingerprint stable for same config",
               GemmaConfigFingerprint(cfg), GemmaConfigFingerprint(cfg))
    check("config fingerprint layer count captured",
          GemmaConfigFingerprint(cfg).layerCount == cfg.layerCount
              && GemmaConfigFingerprint(cfg).hiddenSize == cfg.hiddenSize
              && GemmaConfigFingerprint(cfg).globalLayers == Set(cfg.globalLayers))
    let otherHex = String(repeating: "ff", count: 32)
    rejected("tampered gguf digest rejected",
             expectMutated("gguf") { $0["ggufDigest"] = otherHex })
    rejected("tampered engine digest rejected",
             expectMutated("engine") { $0["engineDigest"] = otherHex })
    rejected("tampered planDigest rejected",
             expectMutated("planDigest") { $0["planDigest"] = otherHex })
    rejected("non-hex workload digest rejected",
             expectMutated("workload") { $0["workloadDigest"] = "zz" })
    rejected("zero promptTokenCount rejected",
             expectMutated("promptCount") { $0["promptTokenCount"] = 0 })
    rejected("negative decodeTokens rejected",
             expectMutated("decodeTokens") { $0["decodeTokens"] = -1 })
    rejected("overflowing workload counts rejected",
             expectMutated("overflow") { $0["promptTokenCount"] = Int.max })
    rejected("workload counts beyond bounded context rejected",
             expectMutated("ctxbound") { $0["promptTokenCount"] = 4093 })
    rejected("non-positive selected median rejected",
             expectMutated("median") { $0["selectedMedianWallMicrosPerToken"] = -1.0 })
    rejected("tampered schedule rejected", expectMutated("schedule") { obj in
        guard var sch = obj["schedule"] as? [String: Any],
              var insts = sch["instructions"] as? [[String: Any]],
              !insts.isEmpty
        else {
            failed += 1; print("FAIL: schedule mutation — fields missing")
            return
        }
        insts.removeLast()
        sch["instructions"] = insts
        obj["schedule"] = sch
    })
    rejected("illegal selected candidate rejected",
             expectMutated("selected") {
                 $0["selected"] = ["segmentBatching": true, "top5Diagnostic": false,
                                   "matvecTiledQ4": true, "matvecTiledQ6": true,
                                   "rmsnormTiled": true, "matvecTiledF32": true, "vectorizedLoads": true,
                                   "rmsStatisticsReuse": false,
                                   "rmsnormCooperative": false, "gpuCacheAppend": false, "parallelAttention": false,
                                   "matvecCooperativeQ6": false,
                                   "routerGPUResidency": false, "routerLaunchDivisor": 0]
             })
    // Wrong CURRENT inputs (not plan mutations) are rejected too.
    check("foreign current gguf digest rejected", (try? GemmaForge.validatePlanIdentity(
        plan, graph: graph, ggufDigest: otherHex, engineDigest: engine,
        hardware: hw)) == nil)
    check("foreign current engine digest rejected", (try? GemmaForge.validatePlanIdentity(
        plan, graph: graph, ggufDigest: gguf, engineDigest: otherHex,
        hardware: hw)) == nil)
    let otherHw = ForgeHardwareProfile(
        deviceName: "t", maxTotalThreadsPerThreadgroup: 512,
        threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1 << 20)
    check("foreign hardware rejected", (try? GemmaForge.validatePlanIdentity(
        plan, graph: graph, ggufDigest: gguf, engineDigest: engine,
        hardware: otherHw)) == nil)
    check("budget mismatch rejected", (try? GemmaForge.validatePlanIdentity(
        plan, graph: graph, ggufDigest: gguf, engineDigest: engine,
        hardware: hw, expertCacheBudget: 123)) == nil)
    // Budget-bound plan round-trips with its budget.
    let planB = try GemmaForge.makePlan(
        graph: graph, graphDigest: graphDigest, ggufDigest: gguf,
        engineDigest: engine, hardware: hw, prompt: [2, 9259],
        decodeTokens: 4, selected: selected, medianWall: 1, medianGpu: 1,
        results: results, expertCacheBudgetBytes: 123)
    check("budget plan validates with matching budget",
          (try? GemmaForge.validatePlanIdentity(
              planB, graph: graph, ggufDigest: gguf, engineDigest: engine,
              hardware: hw, expertCacheBudget: 123)) != nil)
    check("budget plan rejected in resident mode",
          (try? GemmaForge.validatePlanIdentity(
              planB, graph: graph, ggufDigest: gguf, engineDigest: engine,
              hardware: hw)) == nil)
    // Full-gate record (the 32-token legal-candidate fallback): a
    // reselected plan persists the gate length plus every measured
    // rejection; pairing, bounds, and digest coverage are validated.
    let reselected = try GemmaForge.reselect(
        plan, to: selected, fullGateTokens: 32,
        fullGateRejections: ["candidate X rejected: step 25 worst rel 1.171e-04 >= 1e-4"])
    check("reselected plan keeps identity-valid form",
          (try? GemmaForge.validatePlanIdentity(
              reselected, graph: graph, ggufDigest: gguf, engineDigest: engine,
              hardware: hw)) != nil)
    check("reselected plan records gate length and rejections",
          reselected.fullGateTokens == 32 && reselected.fullGateRejections?.count == 1)
    check("reselected plan digest changed", reselected.planDigest != plan.planDigest)
    check("reselect to a non-result candidate throws",
          (try? GemmaForge.reselect(
              plan, to: GemmaForgeCandidate(segmentBatching: false,
                                            top5Diagnostic: true),
              fullGateTokens: 32, fullGateRejections: [])) == nil)
    // Cached-plan re-gate history: the CLI keeps prior rejection
    // reasons when the SAME gate length is re-run (this run's list may
    // be empty because the reselected candidate now passes first) and
    // resets the record when the gate length changes.
    let regate = try GemmaForge.reselect(
        reselected, to: selected, fullGateTokens: 32,
        fullGateRejections: [])
    check("same-length regate keeps prior rejection history",
          regate.fullGateRejections?.count == 1
          && regate.fullGateRejections == reselected.fullGateRejections)
    let regateNew = try GemmaForge.reselect(
        reselected, to: selected, fullGateTokens: 32,
        fullGateRejections: ["new unique reason"])
    check("same-length regate appends only new unique reasons",
          regateNew.fullGateRejections?.count == 2
          && regateNew.fullGateRejections?.first
              == reselected.fullGateRejections?.first)
    let regateOther = try GemmaForge.reselect(
        reselected, to: selected, fullGateTokens: 8,
        fullGateRejections: [])
    check("different-length regate resets the record",
          regateOther.fullGateTokens == 8
          && regateOther.fullGateRejections?.isEmpty == true)
    func gateMutated(_ name: String,
                     _ body: (inout [String: Any]) -> Void) -> GemmaForgePlan? {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(reselected),
              var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            failed += 1; print("FAIL: \(name) — plan re-encode failed")
            return nil
        }
        body(&obj)
        guard JSONSerialization.isValidJSONObject(obj),
              let d2 = try? JSONSerialization.data(withJSONObject: obj),
              let decoded = try? JSONDecoder().decode(GemmaForgePlan.self, from: d2)
        else {
            failed += 1; print("FAIL: \(name) — mutated plan did not decode")
            return nil
        }
        guard decoded != reselected else {
            failed += 1; print("FAIL: \(name) — mutation did not change the plan")
            return nil
        }
        guard let resigned = try? GemmaForge.resignPlan(decoded) else {
            failed += 1; print("FAIL: \(name) — resign failed")
            return nil
        }
        return resigned
    }
    func gateRejected(_ name: String, _ p: GemmaForgePlan?) {
        guard let p else { return }
        if (try? GemmaForge.validatePlanIdentity(
                p, graph: graph, ggufDigest: gguf, engineDigest: engine,
                hardware: hw)) != nil {
            failed += 1; print("FAIL: \(name) — mutated plan accepted")
        } else {
            passed += 1
        }
    }
    gateRejected("full-gate tokens without rejections rejected",
                 gateMutated("unpaired") { $0["fullGateRejections"] = nil })
    gateRejected("full-gate rejections without tokens rejected",
                 gateMutated("unpaired2") { $0["fullGateTokens"] = nil })
    gateRejected("full-gate tokens below 2 rejected",
                 gateMutated("toolo") { $0["fullGateTokens"] = 1 })
    gateRejected("full-gate tokens above 64 rejected",
                 gateMutated("toohi") { $0["fullGateTokens"] = 65 })
    gateRejected("full-gate tokens beyond bounded context rejected",
                 gateMutated("ctx") { $0["fullGateTokens"] = 4095 })
    // Digest coverage: editing the rejection record WITHOUT re-signing
    // must fail the planDigest gate (the record cannot be silently
    // rewritten on a persisted artifact).
    do {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(reselected)
        var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        var rej = obj["fullGateRejections"] as? [String] ?? []
        if rej.count != 1 {
            failed += 1; print("FAIL: stale-digest gate — fields missing")
            rej = []
        }
        if !rej.isEmpty {
            rej[0] = "rewritten history"
            obj["fullGateRejections"] = rej
            if let d2 = try? JSONSerialization.data(withJSONObject: obj),
               let decoded = try? JSONDecoder().decode(GemmaForgePlan.self, from: d2) {
                gateRejected("stale-digest full-gate record rejected", decoded)
            }
        }
    }
} catch {
    failed += 1; print("FAIL: plan identity gates — \(error)")
}

// MARK: - Tiled cooperative Q4_0 matvec vs FP64 oracle (ACTUAL runtime shader)

struct Q4TiledTestError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
    init(_ m: String) { message = m }
}

do {
    // Compiles Gemma4Runner.shaderSource itself (the same MSL the runner
    // uses — never a duplicated shader) and dispatches BOTH
    // g4_q4_matvec (one thread per row) and g4_q4_matvec_tiled (one
    // threadgroup per row, tree reduction) over synthetic Q4_0 weights
    // with MIXED-SIGN FP16 scales and cancellation-prone inputs, at real
    // model dimensions. The tiled reduction reassociates the addition
    // order, so agreement with the FP64 oracle is a tolerance gate
    // (rel < 1e-5 here), not bit equality; row widths 32/704/2816 cover
    // single-block, expert-intermediate, and hidden geometries.
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw Q4TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name),
              let p = try? device.makeComputePipelineState(function: fn) else {
            throw Q4TiledTestError("pipeline \(name) missing")
        }
        return p
    }
    let plain = try pipeline("g4_q4_matvec")
    let tiled = try pipeline("g4_q4_matvec_tiled")

    func fp64Oracle(blocks: [UInt8], outDim: Int, inDim: Int, x: [Float]) -> [Double] {
        let blocksPerRow = inDim / 32
        var y = [Double](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum = 0.0
            for b in 0..<blocksPerRow {
                let base = (row * blocksPerRow + b) * 18
                let dbits = UInt16(blocks[base]) | (UInt16(blocks[base + 1]) << 8)
                let d = Double(Float(Float16(bitPattern: dbits)))
                for l in 0..<16 {
                    let q = blocks[base + 2 + l]
                    let lo = Double(Float(q & 0x0F) - 8)
                    let hi = Double(Float(q >> 4) - 8)
                    sum += d * lo * Double(x[b * 32 + l])
                    sum += d * hi * Double(x[b * 32 + 16 + l])
                }
            }
            y[row] = sum
        }
        return y
    }
    func runKernel(_ p: MTLComputePipelineState, blocks: [UInt8], outDim: Int,
                   inDim: Int, x: [Float], tileWidth: Int?) throws -> [Float] {
        let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
        var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        var off = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        if let w = tileWidth {
            // The PRODUCTION validator decides — no mirrored test guard.
            guard Gemma4Runner.legalCooperativeWidth(w, pipelineMax: p.maxTotalThreadsPerThreadgroup) else {
                throw Q4TiledTestError("illegal tile width \(w)")
            }
            enc.dispatchThreadgroups(MTLSize(width: outDim, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        } else {
            let width = p.maxTotalThreadsPerThreadgroup
            let groups = (outDim + width - 1) / width
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(start: yb.contents().assumingMemoryBound(to: Float.self),
                                           count: outDim))
    }
    var rng = SplitMix64(seed: 4242)
    let widths = [32, 704, 2112, 2816]
    for inDim in widths {
        let outDim = 16
        let blocksPerRow = inDim / 32
        var blocks = [UInt8]()
        blocks.reserveCapacity(outDim * blocksPerRow * 18)
        for _ in 0..<(outDim * blocksPerRow) {
            // Mixed-sign scales (nextUniform() is [-1,1), so
            // nextUniform()*2-1 spans [-3,1)) incl. exact negatives;
            // nibbles uniform for cancellation-heavy dot products.
            let scale = Float16(rng.nextUniform() * 2 - 1)
            blocks += [UInt8(scale.bitPattern & 0xFF), UInt8(scale.bitPattern >> 8)]
            for _ in 0..<16 { blocks.append(UInt8(rng.next() & 0xFF)) }
        }
        // Cancellation-prone input: alternating-sign magnitudes.
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            x[i] = (i % 2 == 0 ? 1 : -1) * Float(rng.nextUniform() * 10 - 5)
        }
        let oracle = fp64Oracle(blocks: blocks, outDim: outDim, inDim: inDim, x: x)
        let plainOut = try runKernel(plain, blocks: blocks, outDim: outDim,
                                     inDim: inDim, x: x, tileWidth: nil)
        var worstPlain = 0.0
        for i in 0..<outDim {
            worstPlain = max(worstPlain,
                             abs(Double(plainOut[i]) - oracle[i]) / max(1, abs(oracle[i])))
        }
        check(String(format: "q4 plain kernel vs fp64 oracle (inDim %d, worst %.2e)",
                     inDim, worstPlain), worstPlain < 1e-5)
        for tile in [32, 64, 128] {
            guard tile <= tiled.maxTotalThreadsPerThreadgroup else { continue }
            let out = try runKernel(tiled, blocks: blocks, outDim: outDim,
                                    inDim: inDim, x: x, tileWidth: tile)
            var worst = 0.0
            for i in 0..<outDim {
                worst = max(worst,
                            abs(Double(out[i]) - oracle[i]) / max(1, abs(oracle[i])))
            }
            check(String(format: "q4 tiled kernel width %d vs fp64 oracle (inDim %d, worst %.2e)",
                         tile, inDim, worst), worst < 1e-5)
        }
    }
    // Illegal widths rejected by the PRODUCTION validator (0, 33, 256;
    // and the same decision the runtime dispatchExplicit makes).
    for bad in [0, 33, 256] {
        check("q4 tiled illegal width \(bad) rejected by production validator",
              !Gemma4Runner.legalCooperativeWidth(bad,
                  pipelineMax: tiled.maxTotalThreadsPerThreadgroup))
    }
    check("q4 tiled legal width 64 accepted by production validator",
          Gemma4Runner.legalCooperativeWidth(64,
              pipelineMax: tiled.maxTotalThreadsPerThreadgroup))
    print("q4 tiled numerical gates completed")
} catch {
    failed += 1; print("FAIL: q4 tiled gate — threw \(error)")
}

// MARK: - Tiled cooperative F32 matvec vs FP64 oracle (ACTUAL runtime shader)

do {
    // Same discipline as the Q4 gate: compiles Gemma4Runner.shaderSource
    // (never a duplicated shader) and dispatches BOTH g4_f32_matvec (one
    // thread per row) and g4_f32_matvec_tiled (one threadgroup per row,
    // tree reduction) over synthetic cancellation-prone F32 weights at
    // real router geometries (inDim = hidden sizes, outDim = expert
    // count). The tiled reduction reassociates the addition order, so
    // agreement with the FP64 oracle is a tolerance gate (rel < 1e-5),
    // not bit equality.
    struct F32TiledTestError: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
        init(_ m: String) { message = m }
    }
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw F32TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name),
              let p = try? device.makeComputePipelineState(function: fn) else {
            throw F32TiledTestError("pipeline \(name) missing")
        }
        return p
    }
    let plain = try pipeline("g4_f32_matvec")
    let tiled = try pipeline("g4_f32_matvec_tiled")

    func runKernel(_ p: MTLComputePipelineState, w: [Float], outDim: Int,
                   inDim: Int, x: [Float], tileWidth: Int?) throws -> [Float] {
        let wb = device.makeBuffer(bytes: w, length: w.count * 4,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
        var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        var off = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        if let tw = tileWidth {
            // The PRODUCTION validator decides — no mirrored test guard.
            guard Gemma4Runner.legalCooperativeWidth(tw, pipelineMax: p.maxTotalThreadsPerThreadgroup) else {
                throw F32TiledTestError("illegal tile width \(tw)")
            }
            enc.dispatchThreadgroups(MTLSize(width: outDim, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tw, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: outDim, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw F32TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(start: yb.contents().assumingMemoryBound(to: Float.self),
                                           count: outDim))
    }
    var rng = SplitMix64(seed: 7777)
    // Real geometries: inDim covers block/expert/router-hidden scales;
    // outDim 128 = expert count. The kernel is stride-based, so any
    // inDim is legal.
    for inDim in [256, 2112, 2816, 3840] {
        let outDim = 128
        var w = [Float](repeating: 0, count: outDim * inDim)
        for i in 0..<w.count {
            w[i] = Float(rng.nextUniform() * 2 - 1)
        }
        // Cancellation-prone input: alternating-sign magnitudes.
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            x[i] = (i % 2 == 0 ? 1 : -1) * Float(rng.nextUniform() * 10 - 5)
        }
        // FP64 oracle over the same float32 weight values, plus per-row
        // magnitude S = sum |w_i x_i|. The gate is an empirically chosen
        // magnitude-scaled bound abs(diff) <= 1e-4*S + 1e-6: on this
        // cancellation-prone data a flat 1e-5 relative gate is not met
        // even by the plain kernel's own summation order, while the
        // scaled bound retains a large margin over the observed
        // differences of both kernels.
        var oracle = [Double](repeating: 0, count: outDim)
        var magnitude = [Double](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum = 0.0
            var mag = 0.0
            for i in 0..<inDim {
                let term = Double(w[row * inDim + i]) * Double(x[i])
                sum += term
                mag += abs(term)
            }
            oracle[row] = sum
            magnitude[row] = mag
        }
        func worstRatio(_ out: [Float]) -> Double {
            var worst = 0.0
            for i in 0..<outDim {
                let diff = abs(Double(out[i]) - oracle[i])
                let tol = 1e-4 * magnitude[i] + 1e-6
                worst = max(worst, diff / tol)
            }
            return worst
        }
        let plainOut = try runKernel(plain, w: w, outDim: outDim,
                                     inDim: inDim, x: x, tileWidth: nil)
        check(String(format: "f32 plain kernel vs fp64 oracle (inDim %d, worst %.2f of tol)",
                     inDim, worstRatio(plainOut)), worstRatio(plainOut) < 1)
        for tile in [32, 64, 128] {
            guard tile <= tiled.maxTotalThreadsPerThreadgroup else { continue }
            let out = try runKernel(tiled, w: w, outDim: outDim,
                                    inDim: inDim, x: x, tileWidth: tile)
            let ratio = worstRatio(out)
            check(String(format: "f32 tiled kernel width %d vs fp64 oracle (inDim %d, worst %.2f of tol)",
                         tile, inDim, ratio), ratio < 1)
        }
    }
    for bad in [0, 33, 256] {
        check("f32 tiled illegal width \(bad) rejected by production validator",
              !Gemma4Runner.legalCooperativeWidth(bad,
                  pipelineMax: tiled.maxTotalThreadsPerThreadgroup))
    }
    check("f32 tiled legal width 64 accepted by production validator",
          Gemma4Runner.legalCooperativeWidth(64,
              pipelineMax: tiled.maxTotalThreadsPerThreadgroup))
    print("f32 tiled numerical gates completed")
} catch {
    failed += 1; print("FAIL: f32 tiled gate — threw \(error)")
}

// MARK: - RMS statistics plan + kernels (rmsStats)

private func rmsStatsValue(_ name: String, _ version: Int, length: Int,
                           kind: String = "vector", sliceOffset: Int? = nil) -> String {
    let slice = sliceOffset == nil ? "null" : String(sliceOffset!)
    return "{\"name\":\"\(name)\",\"version\":\(version),\"dtype\":\"f32\",\"length\":\(length),\"kind\":\"\(kind)\",\"sliceOffset\":\(slice)}"
}

private func rmsStatsInst(_ op: String, layer: Int = 0, inputs: [String], output: String,
                          tensors: [String] = [], attrs: [String: Int] = [:]) -> String {
    let i = inputs.map { "\"\($0)\"" }.joined(separator: ",")
    let t = tensors.map { "\"\($0)\"" }.joined(separator: ",")
    let a = attrs.map { "\"\($0.key)\":\($0.value)" }.joined(separator: ",")
    return "{\"op\":\"\(op)\",\"layer\":\(layer),\"inputs\":[\(i)],\"output\":\"\(output)\",\"tensors\":[\(t)],\"attrs\":{\(a)}}"
}

private func rmsStatsIR(_ values: [String], _ instructions: [String]) -> String {
    "{\"schema\":1,\"values\":[\(values.joined(separator: ","))],\"instructions\":[\(instructions.joined(separator: ","))]}"
}

private func rmsStatsDecode(_ json: String) throws -> GemmaSSAIR {
    try JSONDecoder().decode(GemmaSSAIR.self, from: Data(json.utf8))
}

@MainActor private func rmsStatsAnalyze(_ json: String, epsilon: Float) throws -> RMSStatisticsPlan {
    try RMSStatisticsPlan.analyze(ssa: rmsStatsDecode(json), epsilon: epsilon)
}

/// Group-of-3 shape mirroring the measured shared-input evidence: three
/// rmsnorms on one immutable x version, separated by known-effect ops
/// (shared-FFW matvec/gelu_up, scale) and one router_topk.
private func rmsStatsTripleIR(inputLength: Int = 256, outLength: Int = 256,
                              attrs: [String: Int] = [:], inputSlice: Int? = nil,
                              outSlice: Int? = nil) -> String {
    let values = [
        rmsStatsValue("x", 1, length: inputLength, sliceOffset: inputSlice),
        rmsStatsValue("h", 2, length: outLength, sliceOffset: outSlice),
        rmsStatsValue("gate", 1, length: outLength),
        rmsStatsValue("up", 1, length: outLength),
        rmsStatsValue("act", 1, length: outLength),
        rmsStatsValue("h", 3, length: outLength, sliceOffset: outSlice),
        rmsStatsValue("h", 5, length: outLength, sliceOffset: outSlice),
    ]
    let instructions = [
        rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2",
                     tensors: ["blk.0.ffn_norm.weight"], attrs: attrs),
        rmsStatsInst("matvec", inputs: ["h@2"], output: "gate@1",
                     tensors: ["blk.0.ffn_gate.weight"]),
        rmsStatsInst("matvec", inputs: ["h@2"], output: "up@1",
                     tensors: ["blk.0.ffn_up.weight"]),
        // Canonical activation op: gelu_up consumes the aliased fused
        // gate|up pair and writes only its declared y output.
        rmsStatsInst("gelu_up", inputs: ["gate@1", "up@1"], output: "act@1",
                     attrs: ["width": outLength]),
        rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3", attrs: attrs),
        rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@5",
                     tensors: ["blk.0.pre_ffw_norm_2.weight"], attrs: attrs),
    ]
    return rmsStatsIR(values, instructions)
}

@MainActor private func rmsStatsPlanCompileTests() {
    let epsA: Float = 1e-6

    // Triple on one immutable input: one compute + two reuses, dense
    // actions indexed by instruction index.
    do {
        let plan = try rmsStatsAnalyze(rmsStatsTripleIR(), epsilon: epsA)
        checkEqual("rmsStats groupCount", plan.groupCount, 1)
        checkEqual("rmsStats reuseCount", plan.reuseCount, 2)
        check("rmsStats analysisVersion", plan.analysis == RMSStatisticsPlan.analysisVersion)
        let g = plan.groups[0]
        checkEqual("rmsStats width", g.width, 256)
        checkEqual("rmsStats vectors", g.vectors, 1)
        checkEqual("rmsStats epsilonBits", g.epsilonBits, epsA.bitPattern)
        checkEqual("rmsStats inputID", g.inputID, "x@1")
        checkEqual("rmsStats firstInstructionIndex", g.firstInstructionIndex, 0)
        checkEqual("rmsStats useIndices", g.useIndices, [4, 5])
        check("rmsStats dense action count", plan.actions.count == 6)
        check("rmsStats compute action",
              plan.actions[0]?.kind == .compute && plan.actions[0]?.slot == 0
              && plan.actions[0]?.width == 256 && plan.actions[0]?.inputID == "x@1")
        check("rmsStats reuse actions",
              plan.actions[4]?.kind == .reuse && plan.actions[5]?.kind == .reuse
              && plan.actions[4]?.slot == plan.actions[5]?.slot)
        check("rmsStats non-rms actions nil",
              plan.actions[1] == nil && plan.actions[2] == nil
              && plan.actions[3] == nil)
    } catch {
        failed += 1
        print("FAIL: rmsStats triple analyze threw: \(error)")
    }

    // Epsilon is compared by exact bits: plans must distinguish.
    do {
        let p1 = try rmsStatsAnalyze(rmsStatsTripleIR(), epsilon: 1e-6)
        let p2 = try rmsStatsAnalyze(rmsStatsTripleIR(), epsilon: 1e-3)
        check("rmsStats epsilon distinguishes bits",
              p1 != p2 && p1.groups[0].epsilonBits != p2.groups[0].epsilonBits)
    } catch {
        failed += 1
        print("FAIL: rmsStats epsilon compare threw: \(error)")
    }

    // Invalid epsilon fails closed.
    for e in [Float.nan, .infinity, -.infinity, 0, -1e-6] {
        var threw = false
        do { _ = try rmsStatsAnalyze(rmsStatsTripleIR(), epsilon: e) } catch { threw = true }
        check("rmsStats invalid epsilon fails closed", threw)
    }

    // Differing output widths are different keys: no group of >= 2.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("h", 3, length: 512)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
             rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats differing width prevents sharing", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats differing width threw: \(error)")
    }

    // Vectorized multi-norm (input length != output length) is skipped.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 512),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
             rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats multivector skipped", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats multivector threw: \(error)")
    }

    // width attr: mismatch skips, exact match still groups.
    do {
        let bad = try rmsStatsAnalyze(rmsStatsTripleIR(attrs: ["width": 512]), epsilon: epsA)
        check("rmsStats width attr mismatch skips", bad.groupCount == 0)
        let good = try rmsStatsAnalyze(rmsStatsTripleIR(attrs: ["width": 256]), epsilon: epsA)
        check("rmsStats width attr match groups", good.groupCount == 1 && good.reuseCount == 2)
    } catch {
        failed += 1
        print("FAIL: rmsStats width attr threw: \(error)")
    }

    // Layout: non-zero input/output slice offsets are skipped; 0 is fine.
    do {
        let inSliced = try rmsStatsAnalyze(rmsStatsTripleIR(inputSlice: 4), epsilon: epsA)
        check("rmsStats input slice skip", inSliced.groupCount == 0)
        let outSliced = try rmsStatsAnalyze(rmsStatsTripleIR(outSlice: 4), epsilon: epsA)
        check("rmsStats output slice skip", outSliced.groupCount == 0)
        let zero = try rmsStatsAnalyze(rmsStatsTripleIR(inputSlice: 0, outSlice: 0), epsilon: epsA)
        check("rmsStats zero slices group", zero.groupCount == 1 && zero.reuseCount == 2)
    } catch {
        failed += 1
        print("FAIL: rmsStats slices threw: \(error)")
    }

    // vectors != 1 and unknown attrs are skipped.
    do {
        let multi = try rmsStatsAnalyze(rmsStatsTripleIR(attrs: ["vectors": 2]), epsilon: epsA)
        check("rmsStats vectors=2 skips", multi.groupCount == 0)
        let unknown = try rmsStatsAnalyze(rmsStatsTripleIR(attrs: ["foo": 1]), epsilon: epsA)
        check("rmsStats unknown attr skips", unknown.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats attrs threw: \(error)")
    }

    // Same physical buffer rewritten between the two norms: different
    // SSA versions never share.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("t1", 1, length: 256),
             rmsStatsValue("x", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
             rmsStatsInst("matvec", inputs: ["h@2"], output: "t1@1", tensors: ["m"]),
             rmsStatsInst("matvec", inputs: ["t1@1"], output: "x@2", tensors: ["m"]),
             rmsStatsInst("rmsnorm", inputs: ["x@2"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats mutated physical no sharing", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats mutated physical threw: \(error)")
    }

    // Stale reference to the overwritten version fails closed.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("t1", 1, length: 256),
             rmsStatsValue("x", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
             rmsStatsInst("matvec", inputs: ["h@2"], output: "t1@1", tensors: ["m"]),
             rmsStatsInst("matvec", inputs: ["t1@1"], output: "x@2", tensors: ["m"]),
             rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3")])
        var threw = false
        do { _ = try rmsStatsAnalyze(json, epsilon: epsA) } catch { threw = true }
        check("rmsStats stale reference throws", threw)
    }

    // gate/up alias the shared GU backing: a write through `up` is a
    // physical barrier for a group keyed on `gate`.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("gate", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("up", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["gate@1"], output: "h@2"),
             rmsStatsInst("matvec", inputs: ["h@2"], output: "up@2", tensors: ["m"]),
             rmsStatsInst("rmsnorm", inputs: ["gate@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats gate/up alias write barrier", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats gate/up alias threw: \(error)")
    }

    // Aliased norm OUTPUT on the shared GU backing: the first stats
    // are computed from gate@1, another norm writes its output through
    // the alias (up), and the later gate norm must NOT reuse the first
    // statistics across that physically aliased write.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("gate", 1, length: 256),
             rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("up", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["gate@1"], output: "h@2"),
             rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "up@2"),
             rmsStatsInst("rmsnorm", inputs: ["gate@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats aliased norm output prevents reuse", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats aliased norm output threw: \(error)")
    }

    // In-place first norm writes its own input backing: no sharing.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("x", 2, length: 256),
             rmsStatsValue("h", 2, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "x@2"),
             rmsStatsInst("rmsnorm", inputs: ["x@2"], output: "h@2")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats in-place first norm cannot share", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats in-place threw: \(error)")
    }

    // Opaque unknown ops seal every open group.
    for op in ["attention", "expert_matvec_chain", "cache_append"] {
        do {
            let json = rmsStatsIR(
                [rmsStatsValue("x", 1, length: 256),
                 rmsStatsValue("h", 2, length: 256),
                 rmsStatsValue("t1", 1, length: 256),
                 rmsStatsValue("h", 3, length: 256)],
                [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
                 rmsStatsInst(op, inputs: ["h@2"], output: "t1@1"),
                 rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3")])
            let plan = try rmsStatsAnalyze(json, epsilon: epsA)
            check("rmsStats opaque op \(op) breaks group", plan.groupCount == 0)
        } catch {
            failed += 1
            print("FAIL: rmsStats opaque op \(op) threw: \(error)")
        }
    }

    // router_topk reads router scores / host routing state only: the
    // activation group survives across it.
    do {
        let json = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("router", 1, length: 8),
             rmsStatsValue("rout", 1, length: 2, kind: "routing"),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@2"),
             rmsStatsInst("router_topk", inputs: ["router@1"], output: "rout@1",
                         attrs: ["topK": 2]),
             rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(json, epsilon: epsA)
        check("rmsStats router_topk keeps group",
              plan.groupCount == 1 && plan.reuseCount == 1)
    } catch {
        failed += 1
        print("FAIL: rmsStats router_topk threw: \(error)")
    }

    // Malformed SSA handled without traps.
    do {
        let dup = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256), rmsStatsValue("x", 1, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@1"], output: "x@1")])
        var threw = false
        do { _ = try rmsStatsAnalyze(dup, epsilon: epsA) } catch { threw = true }
        check("rmsStats duplicate IDs throw", threw)
    }
    do {
        let missing = rmsStatsIR(
            [rmsStatsValue("x", 1, length: 256), rmsStatsValue("h", 2, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["x@9"], output: "h@2")])
        var threw = false
        do { _ = try rmsStatsAnalyze(missing, epsilon: epsA) } catch { threw = true }
        check("rmsStats missing input throws", threw)
    }
    do {
        let nonvector = rmsStatsIR(
            [rmsStatsValue("kv", 1, length: 256, kind: "kvstate"),
             rmsStatsValue("h", 2, length: 256),
             rmsStatsValue("h", 3, length: 256)],
            [rmsStatsInst("rmsnorm", inputs: ["kv@1"], output: "h@2"),
             rmsStatsInst("rmsnorm", inputs: ["kv@1"], output: "h@3")])
        let plan = try rmsStatsAnalyze(nonvector, epsilon: epsA)
        check("rmsStats nonvector input skipped, no trap", plan.groupCount == 0)
    } catch {
        failed += 1
        print("FAIL: rmsStats nonvector threw: \(error)")
    }
    do {
        let badDtype = "{\"schema\":1,\"values\":[{\"name\":\"x\",\"version\":1,\"dtype\":\"f16\",\"length\":256,\"kind\":\"vector\",\"sliceOffset\":null}],\"instructions\":[]}"
        var threw = false
        do { _ = try rmsStatsDecode(badDtype) } catch { threw = true }
        check("rmsStats unsupported dtype decode fails closed", threw)
    }
    do {
        let badSchema = "{\"schema\":2,\"values\":[],\"instructions\":[]}"
        var threw = false
        do { _ = try rmsStatsAnalyze(badSchema, epsilon: epsA) } catch { threw = true }
        check("rmsStats schema rejected", threw)
    }
}

@MainActor private func rmsStatsShaderTests() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        failed += 1
        print("FAIL: rmsStats shader: no Metal device")
        return
    }
    guard let queue = device.makeCommandQueue() else {
        failed += 1
        print("FAIL: rmsStats shader: no command queue")
        return
    }
    let combinedSource = RMSCooperative.source + "\n" + RMSStatisticsKernels.source
    guard let library = try? device.makeLibrary(source: combinedSource, options: nil) else {
        failed += 1
        print("FAIL: rmsStats shader: library compile failed")
        return
    }
    guard let coopFn = library.makeFunction(name: RMSCooperative.kernelName),
          let statsFn = library.makeFunction(name: RMSStatisticsKernels.computeKernelName),
          let cachedFn = library.makeFunction(name: RMSStatisticsKernels.applyKernelName) else {
        failed += 1
        print("FAIL: rmsStats shader: function lookup failed")
        return
    }
    guard let coopPipe = try? device.makeComputePipelineState(function: coopFn),
          let statsPipe = try? device.makeComputePipelineState(function: statsFn),
          let cachedPipe = try? device.makeComputePipelineState(function: cachedFn) else {
        failed += 1
        print("FAIL: rmsStats shader: pipeline creation failed")
        return
    }
    check("rmsStats source sha is hex sha256",
          RMSStatisticsKernels.sourceSha256().count == 64)
    check("rmsStats weight offset 16-byte guard",
          RMSStatisticsKernels.weightOffsetIsAligned(0)
          && RMSStatisticsKernels.weightOffsetIsAligned(16)
          && !RMSStatisticsKernels.weightOffsetIsAligned(2))
    check("rmsStats elementCount overflow checked",
          RMSStatisticsKernels.elementCount(width: 256, vectors: 1) == 256
          && RMSStatisticsKernels.elementCount(width: 0, vectors: 1) == nil
          && RMSStatisticsKernels.elementCount(width: Int.max, vectors: 2) == nil)

    func rmsStatsReadFloats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    let strideU32 = MemoryLayout<UInt32>.stride
    let widths = [256, 704, 2816]
    let epsilons: [Float] = [1e-6, 1e-3]

    for width in widths {
        for eps in epsilons {
            for weighted in [false, true] {
                let tag = "w=\(width) eps=\(eps) weighted=\(weighted)"
                var rng = SplitMix64(seed: UInt64(width &* 613) &+ UInt64(eps.bitPattern) &+ (weighted ? 7 : 13))
                var x = [Float](repeating: 0, count: width)
                for i in 0..<width {
                    x[i] = Float(Double(rng.next() % 4096) / 2048.0 - 1.0) * 2.0
                }
                // Weighted passes use different signed scales per lane.
                var w = [Float](repeating: 0, count: width)
                for i in 0..<width {
                    w[i] = (i % 2 == 0 ? 1.5 : -0.75) * Float(i % 7 + 1)
                }
                let originalX = x
                guard let pBuf = device.makeBuffer(bytes: [UInt32(width), UInt32(1)],
                                                   length: strideU32 * 2),
                      let epsBuf = device.makeBuffer(bytes: [eps],
                                                     length: MemoryLayout<Float>.stride),
                      let scaleBuf = device.makeBuffer(bytes: [Int32(weighted ? 1 : 0)],
                                                       length: MemoryLayout<Int32>.stride),
                      let wOffBuf = device.makeBuffer(bytes: [UInt64(0)],
                                                      length: MemoryLayout<UInt64>.stride),
                      let xBuf = device.makeBuffer(bytes: x,
                                                   length: width * MemoryLayout<Float>.stride),
                      let wBuf = device.makeBuffer(bytes: w,
                                                   length: width * MemoryLayout<Float>.stride),
                      let yCoopBuf = device.makeBuffer(length: width * MemoryLayout<Float>.stride),
                      let yStatsBuf = device.makeBuffer(length: width * MemoryLayout<Float>.stride),
                      let yCachedBuf = device.makeBuffer(length: width * MemoryLayout<Float>.stride),
                      let invBuf = device.makeBuffer(length: MemoryLayout<Float>.stride) else {
                    failed += 1
                    print("FAIL: rmsStats shader buffers \(tag)")
                    continue
                }
                let coopThreads = RMSCooperative.threadCount(
                    n: width, executionWidth: coopPipe.threadExecutionWidth,
                    maxThreads: coopPipe.maxTotalThreadsPerThreadgroup)
                let statsThreads = RMSStatisticsKernels.threadCount(
                    n: width, executionWidth: statsPipe.threadExecutionWidth,
                    maxThreads: statsPipe.maxTotalThreadsPerThreadgroup)
                let cachedThreads = RMSStatisticsKernels.threadCount(
                    n: width, executionWidth: cachedPipe.threadExecutionWidth,
                    maxThreads: cachedPipe.maxTotalThreadsPerThreadgroup)
                let tgLen = RMSStatisticsKernels.threadgroupMemoryLength(n: width)
                guard coopThreads > 0, statsThreads > 0, cachedThreads > 0, tgLen > 0,
                      tgLen <= device.maxThreadgroupMemoryLength else {
                    failed += 1
                    print("FAIL: rmsStats shader schedule \(tag)")
                    continue
                }
                guard let cmd = queue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder() else {
                    failed += 1
                    print("FAIL: rmsStats encoder \(tag)")
                    continue
                }
                func rmsStatsBind(_ enc: MTLComputeCommandEncoder) {
                    enc.setBuffer(pBuf, offset: 0, index: 0)
                    enc.setBuffer(epsBuf, offset: 0, index: 1)
                    enc.setBuffer(scaleBuf, offset: 0, index: 2)
                    enc.setBuffer(wOffBuf, offset: 0, index: 3)
                    enc.setBuffer(wBuf, offset: 0, index: 4)
                }
                // All three kernels share one ordered encoder so the
                // cached dispatch depends on the stats dispatch on-GPU.
                rmsStatsBind(enc)
                enc.setBuffer(xBuf, offset: 0, index: 5)
                enc.setBuffer(yCoopBuf, offset: 0, index: 6)
                enc.setBuffer(invBuf, offset: 0, index: 7)
                enc.setComputePipelineState(coopPipe)
                enc.setThreadgroupMemoryLength(tgLen, index: 0)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: coopThreads, height: 1, depth: 1))
                rmsStatsBind(enc)
                enc.setBuffer(xBuf, offset: 0, index: 5)
                enc.setBuffer(yStatsBuf, offset: 0, index: 6)
                enc.setBuffer(invBuf, offset: 0, index: 7)
                enc.setComputePipelineState(statsPipe)
                enc.setThreadgroupMemoryLength(tgLen, index: 0)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: statsThreads, height: 1, depth: 1))
                rmsStatsBind(enc)
                enc.setBuffer(xBuf, offset: 0, index: 5)
                enc.setBuffer(yCachedBuf, offset: 0, index: 6)
                enc.setBuffer(invBuf, offset: 0, index: 7)
                enc.setComputePipelineState(cachedPipe)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: cachedThreads, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                guard cmd.status == .completed else {
                    failed += 1
                    print("FAIL: rmsStats gpu completion \(tag) status=\(cmd.status.rawValue)")
                    continue
                }
                let coopOut = rmsStatsReadFloats(yCoopBuf, width)
                let statsOut = rmsStatsReadFloats(yStatsBuf, width)
                let cachedOut = rmsStatsReadFloats(yCachedBuf, width)
                var statsExact = true
                var cachedExact = true
                var xHeld = true
                for i in 0..<width {
                    if statsOut[i].bitPattern != coopOut[i].bitPattern { statsExact = false }
                    if cachedOut[i].bitPattern != coopOut[i].bitPattern { cachedExact = false }
                    if x[i] != originalX[i] { xHeld = false }
                }
                check("rmsStats first output bit-exact \(tag)", statsExact)
                check("rmsStats cached output bit-exact \(tag)", cachedExact)
                check("rmsStats input held immutable \(tag)", xHeld)
                let inv = rmsStatsReadFloats(invBuf, 1)[0]
                var ms64 = 0.0
                for i in 0..<width { ms64 += Double(originalX[i]) * Double(originalX[i]) }
                let expected64 = 1.0 / (ms64 / Double(width) + Double(eps)).squareRoot()
                check("rmsStats inverse vs FP64 \(tag)",
                      inv.isFinite && abs(Double(inv) - expected64) <= expected64 * 1e-5)
                check("rmsStats all outputs finite \(tag)",
                      statsOut.allSatisfy { $0.isFinite }
                      && cachedOut.allSatisfy { $0.isFinite })

                // In-place variants (x == y) for the weighted case only,
                // tested separately for the first compute and the cached
                // apply; the compiler-level rejection of reuse across a
                // mutated input is covered by the plan tests above.
                if weighted {
                    let byteLen = width * MemoryLayout<Float>.stride
                    guard let xInStats = device.makeBuffer(bytes: originalX, length: byteLen),
                          let xInCached = device.makeBuffer(bytes: originalX, length: byteLen),
                          let cmd2 = queue.makeCommandBuffer(),
                          let enc2 = cmd2.makeComputeCommandEncoder() else {
                        failed += 1
                        print("FAIL: rmsStats in-place setup w=\(width)")
                        continue
                    }
                    rmsStatsBind(enc2)
                    enc2.setBuffer(xInStats, offset: 0, index: 5)
                    enc2.setBuffer(xInStats, offset: 0, index: 6)
                    enc2.setBuffer(invBuf, offset: 0, index: 7)
                    enc2.setComputePipelineState(statsPipe)
                    enc2.setThreadgroupMemoryLength(tgLen, index: 0)
                    enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: statsThreads, height: 1, depth: 1))
                    rmsStatsBind(enc2)
                    enc2.setBuffer(xInCached, offset: 0, index: 5)
                    enc2.setBuffer(xInCached, offset: 0, index: 6)
                    enc2.setBuffer(invBuf, offset: 0, index: 7)
                    enc2.setComputePipelineState(cachedPipe)
                    enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                              threadsPerThreadgroup: MTLSize(width: cachedThreads, height: 1, depth: 1))
                    enc2.endEncoding()
                    cmd2.commit()
                    cmd2.waitUntilCompleted()
                    guard cmd2.status == .completed else {
                        failed += 1
                        print("FAIL: rmsStats in-place completion w=\(width)")
                        continue
                    }
                    let inStatsOut = rmsStatsReadFloats(xInStats, width)
                    let inCachedOut = rmsStatsReadFloats(xInCached, width)
                    var inStatsExact = true
                    var inCachedExact = true
                    for i in 0..<width {
                        if inStatsOut[i].bitPattern != coopOut[i].bitPattern { inStatsExact = false }
                        if inCachedOut[i].bitPattern != coopOut[i].bitPattern { inCachedExact = false }
                    }
                    check("rmsStats in-place first compute bit-exact w=\(width)", inStatsExact)
                    check("rmsStats in-place cached apply bit-exact w=\(width)", inCachedExact)
                }
            }
        }
    }
}

@MainActor func rmsStatsRunAllTests() {
    rmsStatsPlanCompileTests()
    rmsStatsShaderTests()
}
rmsStatsRunAllTests()

// MARK: - Router residency kernel + scalar router geometry (additive)

@MainActor
func routerResidencyPureChecks() {
    // Provenance metadata
    checkEqual("router residency: kernel name",
               RouterResidencyKernel.kernelName, "g4_router_scale")
    checkEqual("router residency: zero kernel name",
               RouterResidencyKernel.zeroKernelName, "g4_router_resident_zero")
    checkEqual("router residency: version is 3 after direct subnormal rounding correction",
               RouterResidencyKernel.version, 3)
    check("router residency: version positive", RouterResidencyKernel.version > 0)
    check("router residency: source non-empty", !RouterResidencyKernel.source.isEmpty)
    let sha = RouterResidencyKernel.sourceSha256
    check("router residency: sha256 is 64 lowercase hex chars",
          sha.count == 64
              && sha.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) })
    checkEqual("router residency: sha256 deterministic",
               RouterResidencyKernel.sourceSha256, sha)
    checkEqual("router residency: allowed divisors",
               RouterResidencyKernel.allowedLaunchDivisors, [0, 1, 2, 4, 8])

    // Geometry derivation from actual pipeline caps (no GPU-name heuristic)
    do {
        let original = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 0, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 128)
        check("router residency: divisor 0 returns nil (original geometry)",
              original == nil)

        let d4 = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 4, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 128)
        check("router residency: divisor 4 of width 32 -> 8 threads", d4 == 8)

        let rowBounded = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 1, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 8)
        check("router residency: bounded by output rows", rowBounded == 8)

        let capBounded = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 1, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 16, outputRows: 128)
        check("router residency: bounded by maxTotalThreadsPerThreadgroup", capBounded == 16)
    } catch {
        check("router residency: scalarRouterThreads derivations", false)
    }

    checkThrows("router residency: divisor 3 rejected",
                RouterResidencyError.divisorNotAllowed(3)) {
        _ = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 3, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 128)
    }
    checkThrows("router residency: zero threadExecutionWidth rejected",
                RouterResidencyError.nonPositiveThreadExecutionWidth(0)) {
        _ = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 1, threadExecutionWidth: 0,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 128)
    }
    checkThrows("router residency: zero maxTotalThreads rejected",
                RouterResidencyError.nonPositiveMaxThreads(0)) {
        _ = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 1, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 0, outputRows: 128)
    }
    checkThrows("router residency: zero rows rejected",
                RouterResidencyError.nonPositiveRows(0)) {
        _ = try RouterResidencyGeometry.scalarRouterThreads(
            divisor: 1, threadExecutionWidth: 32,
            maxTotalThreadsPerThreadgroup: 512, outputRows: 0)
    }

    // Overflow-safe group counts and spans
    do {
        checkEqual("router residency: ceil 128/16 == 8",
                   try RouterResidencyGeometry.groupCount(rows: 128, threadsPerGroup: 16), 8)
        checkEqual("router residency: ceil 129/16 == 9",
                   try RouterResidencyGeometry.groupCount(rows: 129, threadsPerGroup: 16), 9)
        checkEqual("router residency: ceil 16/16 == 1",
                   try RouterResidencyGeometry.groupCount(rows: 16, threadsPerGroup: 16), 1)
        checkEqual("router residency: ceil Int.max/1 == Int.max (no overflow)",
                   try RouterResidencyGeometry.groupCount(rows: Int.max, threadsPerGroup: 1),
                   Int.max)
        checkEqual("router residency: padded grid 704/128 == 768 threads",
                   try RouterResidencyGeometry.gridThreads(rows: 704, threadsPerGroup: 128), 768)
        checkThrows("router residency: grid threads overflow rejected",
                    RouterResidencyError.gridThreadsOverflow) {
            _ = try RouterResidencyGeometry.gridThreads(rows: Int.max, threadsPerGroup: 2)
        }
        checkEqual("router residency: byte span of 2816 floats",
                   try RouterResidencyGeometry.byteSpan(elementCount: 2816), 11264)
        checkEqual("router residency: aligned scale offset passes",
                   try RouterResidencyGeometry.checkedScaleOffset(256), 256)
    } catch {
        check("router residency: groupCount/span/offset pure checks", false)
    }

    checkThrows("router residency: zero threadsPerGroup rejected",
                RouterResidencyError.nonPositiveThreadCount(0)) {
        _ = try RouterResidencyGeometry.groupCount(rows: 128, threadsPerGroup: 0)
    }
    checkThrows("router residency: byte span overflow rejected",
                RouterResidencyError.byteSpanOverflow) {
        _ = try RouterResidencyGeometry.byteSpan(elementCount: Int.max)
    }
    checkThrows("router residency: negative element count rejected",
                RouterResidencyError.negativeElementCount(-1)) {
        _ = try RouterResidencyGeometry.byteSpan(elementCount: -1)
    }
    checkThrows("router residency: negative scale offset rejected",
                RouterResidencyError.negativeScaleOffset(-4)) {
        _ = try RouterResidencyGeometry.checkedScaleOffset(-4)
    }
    checkThrows("router residency: misaligned scale offset rejected",
                RouterResidencyError.misalignedScaleOffset(2)) {
        _ = try RouterResidencyGeometry.checkedScaleOffset(2)
    }
}

@MainActor
func routerResidencyHardwareChecks(device: MTLDevice, queue: MTLCommandQueue) {
    let routerLibrary: MTLLibrary
    let gemmaLibrary: MTLLibrary
    do {
        routerLibrary = try device.makeLibrary(
            source: RouterResidencyKernel.source,
            options: RouterResidencyKernel.makeCompileOptions())
        // The original production pipelines must be compiled exactly as the
        // runtime compiles them: options nil. Only the new second-scale
        // kernel uses precise compile options.
        gemmaLibrary = try device.makeLibrary(source: Gemma4Runner.shaderSource,
                                              options: nil)
    } catch {
        check("router residency: compile pipelines from original sources", false)
        return
    }
    guard let routerFunction = routerLibrary.makeFunction(
              name: RouterResidencyKernel.kernelName),
          let matvecFunction = gemmaLibrary.makeFunction(name: "g4_f32_matvec"),
          let g4ScaleFunction = gemmaLibrary.makeFunction(name: "g4_scale") else {
        check("router residency: kernel functions found in libraries", false)
        return
    }
    let routerScale: MTLComputePipelineState
    let matvec: MTLComputePipelineState
    let g4Scale: MTLComputePipelineState
    do {
        routerScale = try device.makeComputePipelineState(function: routerFunction)
        matvec = try device.makeComputePipelineState(function: matvecFunction)
        g4Scale = try device.makeComputePipelineState(function: g4ScaleFunction)
    } catch {
        check("router residency: build compute pipeline states", false)
        return
    }

    // Hardware capability checks (actual caps, no name heuristics)
    check("router residency: device.maxThreadgroupMemoryLength positive",
          device.maxThreadgroupMemoryLength > 0)
    check("router residency: router pipeline caps positive",
          routerScale.threadExecutionWidth > 0
              && routerScale.maxTotalThreadsPerThreadgroup > 0)

    // Scalar router F32 matvec: original geometry vs derived group geometry
    // with the SAME original pipeline. Because the per-row loop arithmetic
    // is identical, results must be bit-exact on ALL rows, including row
    // tails and the exact-cancellation row.
    do {
        let width = 2816
        let rows = 128
        var rng = SplitMix64(seed: 0x524F_5554_4552)
        var weights = (0..<(rows * width)).map { _ in rng.nextUniform() * 2 - 1 }
        var input = (0..<width).map { _ in rng.nextUniform() * 2 - 1 }
        // Cancellation row: sequential sum returns to exactly +0.0 every pair.
        input[0] = 1.0
        input[1] = 1.0
        weights[0] = 0.5
        weights[1] = -0.5
        for i in 2..<width { weights[i] = 0 }

        let wBytes = try RouterResidencyGeometry.byteSpan(elementCount: rows * width)
        let xBytes = try RouterResidencyGeometry.byteSpan(elementCount: width)
        let yBytes = try RouterResidencyGeometry.byteSpan(elementCount: rows)
        guard let wbuf = device.makeBuffer(bytes: weights, length: wBytes, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: input, length: xBytes, options: .storageModeShared),
              let yOriginal = device.makeBuffer(length: yBytes, options: .storageModeShared),
              let yDerived = device.makeBuffer(length: yBytes, options: .storageModeShared) else {
            check("router residency: matvec buffers allocated", false)
            return
        }

        var p = SIMD2<UInt32>(UInt32(width), UInt32(rows))
        var wOff: UInt64 = 0
        let tew = matvec.threadExecutionWidth
        let maxThreads = matvec.maxTotalThreadsPerThreadgroup
        check("router residency: matvec threadExecutionWidth positive", tew > 0)
        check("router residency: matvec maxTotalThreadsPerThreadgroup positive", maxThreads > 0)

        func runMatvec(_ y: MTLBuffer, threadsPerGroup: Int) {
            memset(y.contents(), 0xFF, yBytes)
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(matvec)
            enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wbuf, offset: 0, index: 2)
            enc.setBuffer(xbuf, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            // dispatchThreadgroups takes the number of GROUPS, not padded
            // total threads; passing gridThreads would over-launch by a
            // factor of threadsPerGroup.
            let grid = MTLSize(
                width: try! RouterResidencyGeometry.groupCount(rows: rows,
                                                               threadsPerGroup: threadsPerGroup),
                height: 1, depth: 1)
            enc.dispatchThreadgroups(
                grid,
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            check("router residency: matvec command completed without error",
                  cb.status == .completed && cb.error == nil)
        }

        let originalThreads = min(rows, maxThreads)
        runMatvec(yOriginal, threadsPerGroup: originalThreads)
        let referencePtr = yOriginal.contents().assumingMemoryBound(to: Float.self)
        let referenceBits = (0..<rows).map { referencePtr[$0].bitPattern }

        for divisor in [1, 2, 4, 8] {
            guard let threads = try RouterResidencyGeometry.scalarRouterThreads(
                divisor: divisor,
                threadExecutionWidth: tew,
                maxTotalThreadsPerThreadgroup: maxThreads,
                outputRows: rows) else {
                check("router residency: derived threads for divisor \(divisor)", false)
                continue
            }
            runMatvec(yDerived, threadsPerGroup: threads)
            let derivedPtr = yDerived.contents().assumingMemoryBound(to: Float.self)
            var allExact = true
            var firstMismatch = -1
            for r in 0..<rows where derivedPtr[r].bitPattern != referenceBits[r] {
                allExact = false
                if firstMismatch < 0 { firstMismatch = r }
            }
            check("router residency: matvec divisor \(divisor) "
                + "(tew/\(divisor) = \(threads) threads) bit-exact vs original geometry "
                + "across all 128 rows incl. tails and cancellation",
                  allExact)
            if !allExact { print("  first differing row: \(firstMismatch)") }
        }

        // Independent FP64 oracle: bounds gross numerical error of the
        // single-geometry accumulation itself (bit-exactness above is
        // strictly between geometries of the same pipeline).
        let oracle = (0..<rows).map { r -> Float in
            var s = 0.0
            for i in 0..<width { s += Double(weights[r * width + i]) * Double(input[i]) }
            return Float(s)
        }
        let reference = (0..<rows).map { referencePtr[$0] }
        check("router residency: matvec matches independent FP64 oracle",
              zip(reference, oracle).allSatisfy { y, o in
                  y.isNaN && o.isNaN
                      || abs(Double(y) - Double(o)) <= abs(Double(o)) * 1e-5 + 1e-37
              })
        checkEqual("router residency: cancellation row is exactly +0.0",
                   referenceBits[0], UInt32(0))
    } catch {
        check("router residency: matvec geometry bit-exactness", false)
    }

    // Second router scale kernel vs original g4_scale followed by the host
    // Float multiply. The GPU performs two ordered, unfused rounds
    // (g4_scale, then g4_router_scale, precise compile options); the CPU
    // reference performs the identical two Float multiplies per element.
    // Bit-exactness is required for all supported cases, including the
    // subnormal boundary. Apple's docs state mathMode .safe prevents
    // transformations affecting results but do NOT promise GPU denormal
    // behavior: if the subnormal slot mismatches on hardware, that failure
    // is preserved here (the bit criteria are never weakened) and a
    // conservative explicit guard must be added before runtime integration.
    let interestingScales: [Float] = [0.5, -2.0, -1.0, 3.0, 1e-30, 6.55e4, 2.9e38, 1.0]
    for width in [256, 704, 2816] {
        var rng = SplitMix64(seed: UInt64(width) &* 0x9E37_79B9_7F4A_7C15)
        var x = (0..<width).map { _ in rng.nextUniform() * 2 - 1 }
        x[0] = 0.0                            // signed zero, positive
        x[1] = -0.0                           // signed zero, negative
        x[2] = Float.leastNonzeroMagnitude    // subnormal boundary
        x[3] = 3.0e38                         // large: large finite scale -> inf bits
        x[4] = 1e-7                           // first round stays normal; the tiny
                                               // scale at slot 4 (1e-30) drives the
                                               // SECOND round subnormal, exercising
                                               // the second-multiply underflow path
        x[5] = -1.0                           // cancellation via scale -1
        let scales = (0..<width).map { interestingScales[$0 % interestingScales.count] }
        let g4Factor = Float(1.0 / (2816.0).squareRoot())  // original first factor

        // Host reference is built below from the ACTUAL GPU output of the
        // original first-scale dispatch: one exact Float multiply per
        // element, never fused.

        // Guarded tail: sentinel elements beyond n must stay untouched
        // (kernel bounds check gid >= n).
        let paddedCount = width + 8
        var paddedX = x
        paddedX.append(contentsOf: [Float](repeating: .nan, count: 8))

        let scaleOffset = 256  // float-aligned offset inside the wbase buffer
        var wbaseBytes = [UInt8](repeating: 0, count: scaleOffset + width * 4)
        scales.withUnsafeBytes { src in
            wbaseBytes.replaceSubrange(scaleOffset..<(scaleOffset + width * 4), with: src)
        }

        guard let xbuf = device.makeBuffer(bytes: paddedX,
                                            length: paddedCount * 4,
                                            options: .storageModeShared),
              let wbase = device.makeBuffer(bytes: wbaseBytes,
                                            length: wbaseBytes.count,
                                            options: .storageModeShared) else {
            check("router residency: scale buffers for width \(width)", false)
            return
        }

        var n = UInt32(width)
        var c = g4Factor
        var off = UInt64(try! RouterResidencyGeometry.checkedScaleOffset(scaleOffset))
        let scaleThreads = min(128, routerScale.maxTotalThreadsPerThreadgroup)
        // dispatchThreadgroups takes GROUP counts, not padded total threads.
        // 704 rows with 128-thread groups yields 6 groups (768 padded
        // threads), exercising the gid >= n tail guard directly.
        let groups = try! RouterResidencyGeometry.groupCount(rows: width,
                                                             threadsPerGroup: scaleThreads)
        let grid = MTLSize(width: groups, height: 1, depth: 1)
        let group = MTLSize(width: scaleThreads, height: 1, depth: 1)
        let gpu = xbuf.contents().assumingMemoryBound(to: Float.self)

        // Round 1 alone on the ORIGINAL pipeline (compiled with options
        // nil, exactly as production): snapshot its actual GPU output.
        let firstCB = queue.makeCommandBuffer()!
        let firstEnc = firstCB.makeComputeCommandEncoder()!
        firstEnc.setComputePipelineState(g4Scale)
        firstEnc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 0)
        firstEnc.setBuffer(xbuf, offset: 0, index: 1)
        firstEnc.setBytes(&c, length: MemoryLayout<Float>.size, index: 2)
        firstEnc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        firstEnc.endEncoding()
        firstCB.commit()
        firstCB.waitUntilCompleted()
        check("router residency: width \(width) first-scale command completed without error",
              firstCB.status == .completed && firstCB.error == nil)
        let snapshot = (0..<width).map { gpu[$0].bitPattern }

        // CPU reference: one exact Float multiply per element on the actual
        // GPU first-scale output — the identical second round, never fused.
        var hostBits = [UInt32](repeating: 0, count: width)
        for i in 0..<width {
            hostBits[i] = (Float(bitPattern: snapshot[i]) * scales[i]).bitPattern
        }

        // Restore the ORIGINAL initial input (including NaN tail
        // sentinels), then run round 1 + round 2 in one ordered command.
        paddedX.withUnsafeBytes { src in
            memcpy(xbuf.contents(), src.baseAddress, paddedCount * 4)
        }

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        // Round 1: ORIGINAL g4_scale kernel, its own ordered dispatch.
        enc.setComputePipelineState(g4Scale)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 0)
        enc.setBuffer(xbuf, offset: 0, index: 1)
        enc.setBytes(&c, length: MemoryLayout<Float>.size, index: 2)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        // Round 2: router residency kernel — second factor only, same
        // command buffer, strictly after round 1 (never fused).
        enc.setComputePipelineState(routerScale)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 0)
        enc.setBuffer(xbuf, offset: 0, index: 1)
        enc.setBuffer(wbase, offset: 0, index: 2)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.size, index: 3)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        check("router residency: width \(width) combined command completed without error",
              cb.status == .completed && cb.error == nil)
        var mismatchCount = 0
        var first: (Int, UInt32, UInt32)?
        for i in 0..<width {
            let got = gpu[i].bitPattern
            if got != hostBits[i] {
                mismatchCount += 1
                if first == nil { first = (i, got, hostBits[i]) }
            }
        }
        check("router residency: second-scale width \(width) bit-exact vs "
            + "original g4_scale + host multiply (weighted +/- finite, signed zero, "
            + "small/large/cancellation, subnormal boundary)",
              mismatchCount == 0)
        if let (i, got, want) = first {
            print("  width \(width): first mismatch at \(i): "
                + "gpu 0x\(String(got, radix: 16)) host 0x\(String(want, radix: 16)) "
                + "— if this is the subnormal slot, GPU denormal flush behavior "
                + "explains it; preserve the failure and add a conservative guard "
                + "before runtime integration")
        }
        var tailOK = true
        for i in width..<paddedCount where gpu[i].bitPattern != Float.nan.bitPattern {
            tailOK = false
        }
        check("router residency: width \(width) guarded tail untouched beyond n", tailOK)
    }
}

@MainActor
func routerResidentZeroHardwareChecks(device: MTLDevice, queue: MTLCommandQueue) {
    // The zero kernel lives in the SAME source library and builds through
    // the parameterized makePipeline(name:) helper (default still resolves
    // the second-scale kernel).
    guard let library = try? device.makeLibrary(
              source: RouterResidencyKernel.source,
              options: RouterResidencyKernel.makeCompileOptions()),
          let zeroFunction = library.makeFunction(
              name: RouterResidencyKernel.zeroKernelName) else {
        check("router residency: zero kernel found in source library", false)
        return
    }
    guard let zeroPipeline = try? device.makeComputePipelineState(function: zeroFunction),
          let namedPipeline = try? RouterResidencyKernel.makePipeline(
              name: RouterResidencyKernel.zeroKernelName, device: device),
          let defaultPipeline = try? RouterResidencyKernel.makePipeline(device: device) else {
        check("router residency: zero/scale pipelines via makePipeline(name:)", false)
        return
    }
    check("router residency: makePipeline default name resolves the scale kernel",
          defaultPipeline.threadExecutionWidth > 0
              && defaultPipeline.maxTotalThreadsPerThreadgroup > 0)
    check("router residency: zero pipeline caps positive",
          zeroPipeline.threadExecutionWidth > 0
              && namedPipeline.maxTotalThreadsPerThreadgroup > 0)

    // Ordered dependency + exact bit pattern: ONE command buffer runs a
    // PRIOR GPU write (the second-scale kernel with finite inputs and a
    // 1.0 scale, so values stay finite) and THEN the zero kernel; the
    // zero must observe the earlier write and overwrite every element
    // with exact +0.0 bits, while the guarded tail beyond n keeps its
    // sentinel NaN bits.
    let n = 300
    let padded = n + 13
    guard let buf = device.makeBuffer(length: padded * 4, options: .storageModeShared),
          let scaleBytes = device.makeBuffer(length: n * 4, options: .storageModeShared) else {
        check("router residency: zero buffers allocated", false)
        return
    }
    var initial = (0..<padded).map { i -> Float in i % 2 == 0 ? 1.5 : -2.5 }
    initial.replaceSubrange(n..<padded, with: [Float](repeating: .nan, count: 13))
    let scales = [Float](repeating: 1.0, count: n)
    memcpy(buf.contents(), initial, padded * 4)
    scales.withUnsafeBytes { raw in
        memcpy(scaleBytes.contents(), raw.baseAddress!, n * 4)
    }

    var count = UInt32(n)
    var off = UInt64(0)
    let threads = min(128, zeroPipeline.maxTotalThreadsPerThreadgroup)
    let groups = try! RouterResidencyGeometry.groupCount(rows: n, threadsPerGroup: threads)
    let grid = MTLSize(width: groups, height: 1, depth: 1)
    let group = MTLSize(width: threads, height: 1, depth: 1)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    // Prior GPU write in the SAME command buffer (ordered before zero).
    enc.setComputePipelineState(defaultPipeline)
    enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 0)
    enc.setBuffer(buf, offset: 0, index: 1)
    enc.setBuffer(scaleBytes, offset: 0, index: 2)
    enc.setBytes(&off, length: MemoryLayout<UInt64>.size, index: 3)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
    // Consumer-visible reset: zero kernel strictly after the write above.
    enc.setComputePipelineState(zeroPipeline)
    enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 0)
    enc.setBuffer(buf, offset: 0, index: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    check("router residency: zero command completed without error",
          cb.status == .completed && cb.error == nil)
    let ptr = buf.contents().assumingMemoryBound(to: Float.self)
    check("router residency: zero kernel writes exact +0.0 bits for all n",
          (0..<n).allSatisfy { ptr[$0].bitPattern == 0 })
    check("router residency: zero kernel leaves guarded tail untouched",
          (n..<padded).allSatisfy { ptr[$0].bitPattern == Float.nan.bitPattern })
}

@MainActor
func routerSecondScaleSubnormalHardwareChecks(device: MTLDevice, queue: MTLCommandQueue) {
    // Direct hardware bit-exactness for the second-scale multiply on the
    // cases the raw GPU multiply gets wrong on M1Max (subnormal flush):
    // normal->subnormal products, signed subnormal inputs/outputs, and
    // round-to-nearest ties-even at the zero and min-normal boundaries.
    // Expected bits come from the host Scalar Float multiply, which keeps
    // gradual underflow and ties-to-even. One case exercises the retained
    // hardware path at the min-normal boundary to pin its exactness too.
    struct Case { let a: Float; let s: Float; let label: String }
    let cases: [Case] = [
        Case(a: 1.0, s: 1e-40, label: "normal * subnormal scale -> subnormal"),
        Case(a: 1e-20, s: 1e-20, label: "normal * normal -> subnormal"),
        Case(a: Float.leastNonzeroMagnitude, s: 1.0, label: "subnormal passthrough"),
        Case(a: -Float.leastNonzeroMagnitude, s: 1.0, label: "negative subnormal passthrough"),
        Case(a: Float.leastNonzeroMagnitude, s: -1.0, label: "subnormal in, signed subnormal out"),
        Case(a: 0.0, s: -1e-30, label: "+0.0 * negative scale -> -0.0"),
        Case(a: -0.0, s: 1e-30, label: "-0.0 * positive scale -> -0.0"),
        Case(a: 0.5, s: Float(bitPattern: 0x007F_FFFF),
             label: "max subnormal halved: tie rounds up to even"),
        Case(a: 0.5, s: Float(bitPattern: 0x0040_0001),
             label: "tie below min normal stays even"),
        Case(a: 0.5, s: Float(bitPattern: 0x0040_0003),
             label: "just above the tie rounds up"),
        Case(a: 0.5, s: Float(bitPattern: 0x0000_0001),
             label: "tie at zero rounds to +0.0"),
        Case(a: -0.5, s: Float(bitPattern: 0x0000_0001),
             label: "tie at zero rounds to -0.0"),
        Case(a: 0.5, s: Float(bitPattern: 0x0000_0003),
             label: "tie one above zero rounds up to 2 ulp"),
        Case(a: 1.0, s: Float(bitPattern: 0x0080_0001),
             label: "normal pair tie at min-normal boundary (hardware path)"),
        // Hand-derived exact-bit double-rounding exposures. Operand bits
        // 0x4955555D * 0x00000003 give the exact 26/27-bit product
        // 67108887 = 4194305*16 + 7, i.e. 4194305.4375 units of 2^-149.
        // One direct RNE yields 0x400001; rounding to 24 bits first
        // (8388610.875 -> 8388611) then to subnormal bits ties 4194305 up
        // to 0x400002, so any double rounding fails these bit comparisons.
        Case(a: Float(bitPattern: 0x4955_5555), s: Float(bitPattern: 0x0000_0003),
             label: "exact product 4194303.9375 ulp rounds once up to 0x00400000"),
        Case(a: Float(bitPattern: 0x4955_555D), s: Float(bitPattern: 0x0000_0003),
             label: "exact product 4194305.4375 ulp rounds once down to 0x00400001"),
        Case(a: Float(bitPattern: 0xC955_555D), s: Float(bitPattern: 0x0000_0003),
             label: "negative twin keeps sign through single subnormal round")
    ]
    // Deterministic boundary sweep: normal*normal pairs whose exact
    // products land in every subnormal result binade from 2^-126 down to
    // 2^-149, jittered across each binade boundary. Expected bits are the
    // host Scalar Float multiply — a single IEEE RNE with gradual
    // underflow — so any double rounding in the kernel shows up here.
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
    var xs = cases.map { $0.a }
    var ss = cases.map { $0.s }
    var expected = cases.map { ($0.a * $0.s).bitPattern }
    for t in -149...(-126) {
        for _ in 0..<16 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let ma = UInt32((rng >> 40) & 0x7F_FFFF)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let ms = UInt32((rng >> 40) & 0x7F_FFFF)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let jitter = Int((rng >> 60) % 3) - 1
            // ea = 40 fixed; es aimed so the 47-bit product's unbiased
            // exponent lands on t; jitter straddles the binade edge.
            let es = t + 214 + jitter
            let a = Float(bitPattern: (UInt32(40) << 23) | ma)
            let s = Float(bitPattern: (UInt32(es) << 23) | ms)
            xs.append(a)
            ss.append(s)
            expected.append((a * s).bitPattern)
        }
    }
    let n = xs.count
    guard let pipeline = try? RouterResidencyKernel.makePipeline(device: device),
          let xbuf = device.makeBuffer(bytes: xs, length: n * 4, options: .storageModeShared),
          let sbuf = device.makeBuffer(bytes: ss, length: n * 4, options: .storageModeShared) else {
        check("router residency: subnormal-case pipeline + buffers allocated", false)
        return
    }
    var count = UInt32(n)
    var off = UInt64(0)
    let threads = min(128, pipeline.maxTotalThreadsPerThreadgroup)
    let groups = try! RouterResidencyGeometry.groupCount(rows: n, threadsPerGroup: threads)
    let grid = MTLSize(width: groups, height: 1, depth: 1)
    let group = MTLSize(width: threads, height: 1, depth: 1)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 0)
    enc.setBuffer(xbuf, offset: 0, index: 1)
    enc.setBuffer(sbuf, offset: 0, index: 2)
    enc.setBytes(&off, length: MemoryLayout<UInt64>.size, index: 3)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    check("router residency: subnormal-case command completed without error",
          cb.status == .completed && cb.error == nil)
    let gpu = xbuf.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<n {
        let label = i < cases.count ? cases[i].label : "sweep \(i - cases.count)"
        check("router residency: subnormal case \(i) \(label) bit-exact",
              gpu[i].bitPattern == expected[i])
    }
}

routerResidencyPureChecks()
if let routerDevice = MTLCreateSystemDefaultDevice(),
   let routerQueue = routerDevice.makeCommandQueue() {
    routerResidencyHardwareChecks(device: routerDevice, queue: routerQueue)
    routerSecondScaleSubnormalHardwareChecks(device: routerDevice, queue: routerQueue)
    routerResidentZeroHardwareChecks(device: routerDevice, queue: routerQueue)
} else {
    check("router residency: Metal device + queue available", false)
}

// MARK: - Intra-block SIMD-reduction Q6_K matvec vs FP64 oracle (ACTUAL runtime shader)

do {
    // Compiles the standalone Q6Cooperative.source (the actual runtime
    // shader, not a copied approximation) and dispatches
    // g4_q6_matvec_coop over synthetic Q6_K blocks with mixed-sign FP16
    // scales, saturated quant bytes and cancellation-prone alternating
    // inputs, gated against the independent Q6_K.matvecReferenceF64
    // oracle. One- and two-row threadgroups plus 1/2/3-row tails are
    // exercised. Bit-equality and order preservation are NOT claimed:
    // the simd_sum reduction reassociates the floating-point addition
    // order relative to the plain kernel, so exact-match counts are
    // reported honestly and only finite + normalized < 1e-4 is enforced,
    // except the exact hand-computed -315 known block which must stay
    // exact.
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw Q4TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Q6Cooperative.source, options: nil)
    guard let fn = library.makeFunction(name: Q6Cooperative.kernelName),
          let coop = try? device.makeComputePipelineState(function: fn) else {
        throw Q4TiledTestError("g4_q6_matvec_coop pipeline missing")
    }
    check("q6 coop version bound to source hash",
          Q6Cooperative.hash == SHA256.hex(SHA256.hash(Array(Q6Cooperative.source.utf8)))
              && Q6Cooperative.hash.count == 64)

    // Pure checked-geometry eligibility and rejection cases.
    let ew = coop.threadExecutionWidth
    let pmax = coop.maxTotalThreadsPerThreadgroup
    let mem = device.maxThreadgroupMemoryLength
    func elig(_ inDim: Int, _ outDim: Int, _ rpg: Int,
              ewOverride: Int? = nil, memOverride: Int? = nil,
              pmaxOverride: Int? = nil) -> Bool {
        Q6Cooperative.eligible(inDim: inDim, outDim: outDim, rowsPerGroup: rpg,
                               executionWidth: ewOverride ?? ew,
                               pipelineMaxThreads: pmaxOverride ?? pmax,
                               maxThreadgroupMemory: memOverride ?? mem)
    }
    check("q6 coop accepts production geometry",
          elig(2816, 262144, 2))
    check("q6 coop rejects inDim below 256", !elig(128, 16, 1))
    check("q6 coop rejects non-multiple-of-256 inDim", !elig(384, 16, 1))
    check("q6 coop rejects inDim above 2816", !elig(3072, 16, 1))
    check("q6 coop rejects rowsPerGroup 3", !elig(256, 16, 3))
    check("q6 coop rejects outDim 0", !elig(256, 0, 1))
    check("q6 coop rejects non-32 execution width (must fall back)",
          !elig(256, 16, 1, ewOverride: 16))
    check("q6 coop accepts zero threadgroup memory (no staging array)",
          elig(256, 16, 1, memOverride: 0))
    check("q6 coop rejects negative threadgroup memory",
          !elig(256, 16, 1, memOverride: -1))
    check("q6 coop rejects threads above pipeline cap",
          !elig(256, 16, 2, pmaxOverride: 32))
    check("q6 coop rejects negative pipeline cap",
          !elig(256, 16, 1, pmaxOverride: -1))
    check("q6 coop rejects threadgroup count above cap",
          !elig(256, Q6Cooperative.maxThreadgroups * 2 + 1, 1))

    func runCoop(blocks: [UInt8], outDim: Int, inDim: Int, x: [Float],
                 rowsPerGroup: Int) throws -> [Float] {
        guard let plan = Q6Cooperative.plan(
            inDim: inDim, outDim: outDim, rowsPerGroup: rowsPerGroup,
            executionWidth: coop.threadExecutionWidth,
            pipelineMaxThreads: coop.maxTotalThreadsPerThreadgroup,
            maxThreadgroupMemory: device.maxThreadgroupMemoryLength) else {
            throw Q4TiledTestError(
                "illegal coop geometry inDim \(inDim) outDim \(outDim) rpg \(rowsPerGroup)")
        }
        let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
        var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        var off = UInt64(0)
        var rpg = UInt32(rowsPerGroup)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(coop)
        enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        enc.setBytes(&rpg, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: plan.threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup:
                MTLSize(width: plan.threadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("coop kernel failed: \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: outDim))
    }

    var rng = SplitMix64(seed: 777)
    let geometryCases: [(inDim: Int, outDim: Int, rpg: Int)] = [
        (256, 16, 1), (256, 16, 2),
        (512, 16, 1), (512, 16, 2),
        (2816, 16, 1), (2816, 16, 2),
        (256, 1, 2), (256, 2, 1), (256, 3, 2), (2816, 3, 2),
    ]
    var totalExact = 0
    var totalRows = 0
    for (inDim, outDim, rpg) in geometryCases {
        let blocksPerRow = inDim / Q6_K.valuesPerBlock
        var blocks = [UInt8]()
        blocks.reserveCapacity(outDim * blocksPerRow * Q6_K.bytesPerBlock)
        for r in 0..<(outDim * blocksPerRow) {
            if r % 4 == 3 {
                // Saturated quant bytes exercise the extreme nibble range.
                for _ in 0..<208 {
                    blocks.append(UInt8(rng.next() & 1 == 0 ? 0x00 : 0xFF))
                }
            } else {
                for _ in 0..<208 { blocks.append(UInt8(rng.next() & 0xFF)) }
            }
            let scale = Float16(rng.nextUniform() * 2 - 1)
            blocks += [UInt8(scale.bitPattern & 0xFF), UInt8(scale.bitPattern >> 8)]
        }
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            x[i] = (i % 2 == 0 ? 1 : -1) * Float(rng.nextUniform() * 10 - 5)
        }
        let ref64 = try Q6_K.matvecReferenceF64(blocks: blocks, outDim: outDim,
                                                inDim: inDim, x: x)
        let out = try runCoop(blocks: blocks, outDim: outDim, inDim: inDim,
                              x: x, rowsPerGroup: rpg)
        var worst = 0.0
        var allFinite = true
        var exact = 0
        for i in 0..<outDim {
            if !out[i].isFinite { allFinite = false }
            worst = max(worst, abs(Double(out[i]) - ref64[i])
                        / max(1.0, abs(ref64[i])))
            if Float(out[i]) == Float(ref64[i]) { exact += 1 }
        }
        totalExact += exact
        totalRows += outDim
        check(String(format: "q6 coop rpg %d vs fp64 oracle (inDim %d, outDim %d, worst %.2e)",
                     rpg, inDim, outDim, worst), allFinite && worst < 1e-4)
    }
    // Honest bit-equality report: exactness and order preservation are
    // not claimed for this candidate; the simd_sum reduction reassociates
    // the floating-point addition order relative to the plain kernel.
    print("q6 coop bit-exact rows vs fp64 oracle: \(totalExact)/\(totalRows)")

    // The exact hand-computed -315 known block must remain exact, in both
    // a single-row group and a two-row group with a row tail.
    do {
        var block = [UInt8](repeating: 0, count: Q6_K.bytesPerBlock)
        block[208] = 0x00; block[209] = 0x3C
        block[192] = 3; block[193] = 0xFD; block[196] = 2
        block[200] = 3; block[204] = 2
        block[0] = 0x21; block[128] = 0x03
        block[64] = 0x21; block[160] = 0x01
        var kx = [Float](repeating: 0, count: 256)
        kx[0] = 1; kx[16] = -1; kx[64] = 2; kx[128] = -2; kx[192] = 4
        let a = try runCoop(blocks: block, outDim: 1, inDim: 256,
                            x: kx, rowsPerGroup: 1)
        check("q6 coop known block matches hand computation (-315, rpg 1)",
              a[0] == -315)
        let b = try runCoop(blocks: block + block, outDim: 2, inDim: 256,
                            x: kx, rowsPerGroup: 2)
        check("q6 coop known block matches hand computation (-315, rpg 2)",
              b[0] == -315 && b[1] == -315)
    }
    print("q6 cooperative numerical gates completed")
} catch {
    failed += 1; print("FAIL: q6 coop gate — threw \(error)")
}
// MARK: - Tiled cooperative Q6_K matvec vs FP64 oracle (ACTUAL runtime shader)

do {
    // Compiles Gemma4Runner.shaderSource itself and dispatches
    // g4_q6_matvec_tiled over synthetic Q6_K blocks (mixed-sign FP16
    // scales, saturated nibbles, cancellation-prone inputs) at real
    // dimensions 256/512/2816, gated against Q6_K.matvecReferenceF64
    // (the independent FP64 oracle already gating the plain kernel),
    // plus the exact hand-computed -315 known block at inDim 256.
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw Q4TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    guard let fn = library.makeFunction(name: "g4_q6_matvec_tiled"),
          let tiled = try? device.makeComputePipelineState(function: fn) else {
        throw Q4TiledTestError("g4_q6_matvec_tiled pipeline missing")
    }
    func runTiled(blocks: [UInt8], outDim: Int, inDim: Int, x: [Float],
                  width: Int) throws -> [Float] {
        guard Gemma4Runner.legalCooperativeWidth(
            width, pipelineMax: tiled.maxTotalThreadsPerThreadgroup) else {
            throw Q4TiledTestError("illegal tile width \(width)")
        }
        let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
        var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        var off = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(tiled)
        enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: outDim, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(
            start: yb.contents().assumingMemoryBound(to: Float.self), count: outDim))
    }
    var rng = SplitMix64(seed: 777)
    for inDim in [256, 512, 2816] {
        let outDim = 16
        let blocksPerRow = inDim / Q6_K.valuesPerBlock
        var blocks = [UInt8]()
        blocks.reserveCapacity(outDim * blocksPerRow * Q6_K.bytesPerBlock)
        for _ in 0..<(outDim * blocksPerRow) {
            for _ in 0..<208 { blocks.append(UInt8(rng.next() & 0xFF)) }
            let scale = Float16(rng.nextUniform() * 2 - 1)
            blocks += [UInt8(scale.bitPattern & 0xFF), UInt8(scale.bitPattern >> 8)]
        }
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            x[i] = (i % 2 == 0 ? 1 : -1) * Float(rng.nextUniform() * 10 - 5)
        }
        let ref64 = try Q6_K.matvecReferenceF64(blocks: blocks, outDim: outDim,
                                                inDim: inDim, x: x)
        for width in [32, 64, 128] {
            guard width <= tiled.maxTotalThreadsPerThreadgroup else { continue }
            let out = try runTiled(blocks: blocks, outDim: outDim, inDim: inDim,
                                   x: x, width: width)
            var worst = 0.0
            for i in 0..<outDim {
                worst = max(worst, abs(Double(out[i]) - ref64[i])
                            / max(1.0, abs(ref64[i])))
            }
            check(String(format: "q6 tiled width %d vs fp64 oracle (inDim %d, worst %.2e)",
                         width, inDim, worst), worst < 1e-4)
        }
        // The exact hand-computed -315 known block (inDim 256 only).
        if inDim == 256 {
            var block = [UInt8](repeating: 0, count: Q6_K.bytesPerBlock)
            block[208] = 0x00; block[209] = 0x3C
            block[192] = 3; block[193] = 0xFD; block[196] = 2
            block[200] = 3; block[204] = 2
            block[0] = 0x21; block[128] = 0x03
            block[64] = 0x21; block[160] = 0x01
            var kx = [Float](repeating: 0, count: 256)
            kx[0] = 1; kx[16] = -1; kx[64] = 2; kx[128] = -2; kx[192] = 4
            let out = try runTiled(blocks: block, outDim: 1, inDim: 256,
                                   x: kx, width: 64)
            check("q6 tiled known block matches hand computation (-315)",
                  out[0] == -315)
        }
    }
    print("q6 tiled numerical gates completed")
} catch {
    failed += 1; print("FAIL: q6 tiled gate — threw \(error)")
}

// MARK: - Tiled cooperative RMSNorm vs Double oracle (ACTUAL runtime shader)

do {
    // Compiles Gemma4Runner.shaderSource and dispatches g4_rmsnorm_tiled
    // vs a Double CPU oracle: weighted and scale-free, heads 1/2/8/16,
    // n 256/512/2816, in-place and out-of-place; deterministic corners
    // (all-zero, eps-dominated tiny values with an ABSOLUTE tolerance —
    // a /max(1,|ref|) denominator would be vacuous there — and a large
    // finite 193400-magnitude input). All GPU outputs and Double
    // references are checked finite BEFORE error accumulation (a NaN
    // could otherwise hide in max()). Dynamic-range sweep spans ~1e-5..
    // 1e3 (nextUniform() is [-1,1), exponent from [-5,3)).
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw Q4TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    guard let fn = library.makeFunction(name: "g4_rmsnorm_tiled"),
          let p = try? device.makeComputePipelineState(function: fn) else {
        throw Q4TiledTestError("g4_rmsnorm_tiled pipeline missing")
    }
    func rmsOracle(x: [Float], w: [Float]?, eps: Float) -> [Double] {
        let n = x.count
        let wd = w.map { $0.map(Double.init) }
        var ms = 0.0
        for v in x { ms += Double(v) * Double(v) }
        let inv = 1.0 / (ms / Double(n) + Double(eps)).squareRoot()
        return (0..<n).map { Double(x[$0]) * inv * (wd?[$0] ?? 1.0) }
    }
    func runRmsTiled(x: [Float], w: [Float]?, vectors: Int, width: Int,
                     inPlace: Bool) throws -> [[Float]] {
        let n = x.count / vectors
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = inPlace ? xb : device.makeBuffer(length: x.count * 4,
                                                  options: .storageModeShared)!
        let wb = w.map { device.makeBuffer(bytes: $0, length: $0.count * 4,
                                           options: .storageModeShared)! }
            ?? device.makeBuffer(length: 16, options: .storageModeShared)!
        var dims = SIMD2<UInt32>(UInt32(n), UInt32(vectors))
        var eps: Float = 1e-6
        var withScale: Int32 = w == nil ? 0 : 1
        var wOff = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 1)
        enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
        enc.setBuffer(wb, offset: 0, index: 4)
        enc.setBuffer(xb, offset: 0, index: 5)
        enc.setBuffer(yb, offset: 0, index: 6)
        guard Gemma4Runner.legalCooperativeWidth(
            width, pipelineMax: p.maxTotalThreadsPerThreadgroup) else {
            throw Q4TiledTestError("illegal tile width \(width)")
        }
        enc.dispatchThreadgroups(MTLSize(width: vectors, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        let ptr = yb.contents().assumingMemoryBound(to: Float.self)
        return (0..<vectors).map { v in
            [Float](UnsafeBufferPointer(start: ptr + v * n, count: n))
        }
    }
    // Deterministic corners, weighted, both in-place and out-of-place.
    do {
        let n = 256
        let corners: [(String, [Float])] = [
            ("all-zero", [Float](repeating: 0, count: n)),
            ("eps-dominated", (0..<n).map { Float($0 % 2 == 0 ? 1e-8 : -1e-8) }),
            ("large-193400", (0..<n).map { ($0 % 2 == 0 ? 1.0 : -1.0) * 193400 }),
        ]
        for (label, x0) in corners {
            var w = [Float](repeating: 1, count: n)
            for i in 0..<n { w[i] = i % 2 == 0 ? 0.5 : -0.5 }
            for inPlace in [false, true] {
                let got = try runRmsTiled(x: x0, w: w, vectors: 1, width: 64,
                                          inPlace: inPlace)[0]
                let ref = rmsOracle(x: x0, w: w, eps: 1e-6)
                check("rms tiled corner \(label) outputs finite (inPlace=\(inPlace))",
                      got.allSatisfy(\.isFinite) && ref.allSatisfy(\.isFinite))
                if label == "all-zero" {
                    check("rms tiled corner all-zero is exact zero (inPlace=\(inPlace))",
                          got.allSatisfy { $0 == 0 })
                    continue
                }
                if label == "eps-dominated" {
                    var worstAbs = 0.0
                    var nonzero = false
                    for i in 0..<n {
                        worstAbs = max(worstAbs, abs(Double(got[i]) - ref[i]))
                        nonzero = nonzero || abs(ref[i]) > 0
                    }
                    check(String(format: "rms tiled corner eps-dominated abs err %.2e, nonzero expected (inPlace=%d)",
                                 worstAbs, inPlace ? 1 : 0),
                          worstAbs < 1e-7 && nonzero)
                    continue
                }
                var worst = 0.0
                for i in 0..<n {
                    worst = max(worst, abs(Double(got[i]) - ref[i]) / max(1.0, abs(ref[i])))
                }
                check(String(format: "rms tiled corner %@ vs Double oracle (worst %.2e, inPlace=%d)",
                             label, worst, inPlace ? 1 : 0), worst < 1e-5)
            }
        }
    }
    var rng = SplitMix64(seed: 31337)
    for n in [256, 512, 2816] {
        for vectors in [1, 2, 8, 16] {
            var w = [Float](repeating: 0, count: n)
            for i in 0..<n { w[i] = Float(rng.nextUniform() * 4 - 2) }
            var x = [Float](repeating: 0, count: n * vectors)
            for i in 0..<x.count {
                x[i] = (i % 2 == 0 ? 1 : -1)
                    * Float(pow(10.0, rng.nextUniform() * 4 - 1))
            }
            for (width, inPlace) in [(32, false), (64, true), (128, false)] {
                let got = try runRmsTiled(x: x, w: w, vectors: vectors,
                                          width: width, inPlace: inPlace)
                check("rms tiled outputs finite (n=\(n) vec=\(vectors) w=\(width))",
                      got.allSatisfy { row in row.allSatisfy(\.isFinite) })
                var worst = 0.0
                var allFinite = true
                for v in 0..<vectors {
                    let ref = rmsOracle(x: Array(x[(v * n)..<((v + 1) * n)]), w: w, eps: 1e-6)
                    allFinite = allFinite && ref.allSatisfy(\.isFinite)
                }
                check("rms tiled Double reference finite (n=\(n) vec=\(vectors) w=\(width))",
                      allFinite)
                if allFinite {
                    for v in 0..<vectors {
                        let ref = rmsOracle(x: Array(x[(v * n)..<((v + 1) * n)]), w: w, eps: 1e-6)
                        for i in 0..<n {
                            worst = max(worst, abs(Double(got[v][i]) - ref[i])
                                        / max(1.0, abs(ref[i])))
                        }
                    }
                }
                check(String(format: "rms tiled n=%d vec=%d w=%d inplace=%d vs Double oracle (worst %.2e)",
                             n, vectors, width, inPlace ? 1 : 0, worst), worst < 1e-5)
            }
            let got = try runRmsTiled(x: x, w: nil, vectors: vectors, width: 64,
                                      inPlace: false)
            check("rms tiled scale-free outputs finite (n=\(n) vec=\(vectors))",
                  got.allSatisfy { row in row.allSatisfy(\.isFinite) })
            var worstFree = 0.0
            var refsFinite = true
            for v in 0..<vectors {
                refsFinite = refsFinite && rmsOracle(
                    x: Array(x[(v * n)..<((v + 1) * n)]), w: nil, eps: 1e-6
                ).allSatisfy(\.isFinite)
            }
            check("rms tiled scale-free Double references finite (n=\(n) vec=\(vectors))",
                  refsFinite)
            if refsFinite {
                for v in 0..<vectors {
                    let ref = rmsOracle(x: Array(x[(v * n)..<((v + 1) * n)]), w: nil, eps: 1e-6)
                    for i in 0..<n {
                        worstFree = max(worstFree, abs(Double(got[v][i]) - ref[i])
                                        / max(1.0, abs(ref[i])))
                    }
                }
            }
            check(String(format: "rms tiled scale-free n=%d vec=%d (worst %.2e)",
                         n, vectors, worstFree), worstFree < 1e-5)
        }
    }
    print("rms tiled numerical gates completed")
} catch {
    failed += 1; print("FAIL: rms tiled gate — threw \(error)")
}

// MARK: - Bounded expert cache components (synthetic file + Metal)
do {
    // Slot table: free-first fill, LRU eviction order, MRU touch on
    // hit, pinned-victim exclusion, pre-IO evict / post-IO install
    // ordering (a slot evicted for a failed IO stays FREE).
    var t = ExpertSlotTable(slotCount: 2)
    checkEqual("slot table starts empty", t.residentKeys, 0)
    checkEqual("free slot found first", t.freeSlot(), 0)
    check("victim nil while unoccupied", t.victim(pinned: []) == nil)
    t.install(key: 100, slot: 0)
    t.install(key: 200, slot: 1)
    checkEqual("table full after two installs", t.residentKeys, 2)
    check("free slot nil when full", t.freeSlot() == nil)
    checkEqual("hit returns slot", t.lookup(100), 0)
    checkEqual("miss returns nil", t.lookup(300) ?? -1, -1)
    // 100 was just touched -> 200 is the LRU victim.
    checkEqual("lru victim is the stale slot", t.victim(pinned: []) ?? -1, 1)
    checkEqual("pinned victim excluded", t.victim(pinned: [1]) ?? -1, 0)
    check("all pinned -> no victim", t.victim(pinned: [0, 1]) == nil)
    t.evict(slot: 1)
    check("evicted slot not occupied", !t.isOccupied(1))
    checkEqual("evicted key missing", t.lookup(200) ?? -1, -1)
    t.install(key: 300, slot: 1)
    checkEqual("retry install works", t.lookup(300), 1)
    checkEqual("other key untouched", t.lookup(100), 0)

    // Geometry: budget/stride arithmetic incl. extreme values rejected
    // cleanly (nil), never trapped.
    check("geometry rejects too-small budget",
          ExpertCacheGeometry.slotCounts(budgetBytes: 10, guStrideBytes: 8,
                                         dnStrideBytes: 8) == nil)
    if let g = ExpertCacheGeometry.slotCounts(budgetBytes: 16, guStrideBytes: 8,
                                              dnStrideBytes: 8) {
        checkEqual("geometry exact pair fits gu", g.gu, 1)
        checkEqual("geometry exact pair fits dn", g.dn, 1)
    } else {
        failed += 1; print("FAIL: geometry exact pair fits")
    }
    check("geometry rejects sub-pair real-stride budget",
          ExpertCacheGeometry.slotCounts(budgetBytes: 1 << 20, guStrideBytes: 2_230_272,
                                         dnStrideBytes: 1_115_136) == nil)
    if let g = ExpertCacheGeometry.slotCounts(budgetBytes: 8_000_000, guStrideBytes: 2_230_272,
                                              dnStrideBytes: 1_115_136) {
        checkEqual("geometry 8 MiB gu slots", g.gu, 2)
        checkEqual("geometry 8 MiB dn slots", g.dn, 3)
    } else {
        failed += 1; print("FAIL: geometry 8 MiB split")
    }
    check("geometry rejects Int.max budget",
          ExpertCacheGeometry.slotCounts(budgetBytes: Int.max, guStrideBytes: 8,
                                         dnStrideBytes: 8) == nil)
    check("geometry rejects Int.max stride",
          ExpertCacheGeometry.slotCounts(budgetBytes: 1 << 20, guStrideBytes: Int.max,
                                         dnStrideBytes: 8) == nil)
    check("geometry rejects negative budget",
          ExpertCacheGeometry.slotCounts(budgetBytes: -5, guStrideBytes: 8,
                                         dnStrideBytes: 8) == nil)
    check("geometry rejects zero stride",
          ExpertCacheGeometry.slotCounts(budgetBytes: 1 << 20, guStrideBytes: 0,
                                         dnStrideBytes: 8) == nil)

    // Raw preadExact: full read returns the exact bytes; beyond-EOF
    // reads abort. (The raw pread may deliver PARTIAL bytes into the
    // destination before aborting — only the CACHE's key visibility is
    // transactional; that is the loader test below.)
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("leviathan-pread-\(UUID().uuidString).bin")
    let payload = Array(0..<100).map { UInt8($0 & 0xff) }
    try Data(payload).write(to: tmp)
    defer { try? FileManager.default.removeItem(at: tmp) }
    let fd = open(tmp.path, O_RDONLY)
    check("pread test file opened", fd >= 0)
    if fd >= 0 {
        defer { close(fd) }
        var buf = [UInt8](repeating: 0, count: 101)
        var threw = false
        do {
            try buf.withUnsafeMutableBytes { raw in
                try Gemma4ExpertCache.preadExact(fd: fd, into: raw.baseAddress!,
                                                 count: 100, from: 0)
            }
        } catch { threw = true }
        check("full pread succeeds", !threw)
        checkEqual("pread bytes match", Array(buf[0..<100]), payload)
        var threwShort = false
        do {
            try buf.withUnsafeMutableBytes { raw in
                try Gemma4ExpertCache.preadExact(fd: fd, into: raw.baseAddress!,
                                                 count: 101, from: 0)
            }
        } catch { threwShort = true }
        check("read beyond EOF aborts", threwShort)
        var threwLate = false
        do {
            try buf.withUnsafeMutableBytes { raw in
                try Gemma4ExpertCache.preadExact(fd: fd, into: raw.baseAddress!,
                                                 count: 10, from: 95)
            }
        } catch { threwLate = true }
        check("read crossing EOF aborts", threwLate)
    }

    // Transactional loader on a REAL cache (tiny synthetic file + Metal
    // device, no GGUF): fill both gate-up slots with valid loads
    // (releasing active pins between loads via the iteration API), then
    // truncate the file so a forced victim reload at a DECLARED-valid
    // offset partial-reads (8 bytes delivered, then EOF). The failed
    // key must NOT become resident (retry after restoring the file
    // MISSES again, not hits) and the retried slice must match the
    // file bytes exactly.
    if let device = MTLCreateSystemDefaultDevice() {
        let f2 = FileManager.default.temporaryDirectory
            .appendingPathComponent("leviathan-cache-\(UUID().uuidString).bin")
        let fileBytes = Array(0..<128).map { UInt8($0 & 0xff) }
        try Data(fileBytes).write(to: f2)
        defer { try? FileManager.default.removeItem(at: f2) }
        // 2 gate-up slots (32 B) + 2 down slots (16 B).
        let cache = try Gemma4ExpertCache(device: device, fileURL: f2,
                                          ggufSize: 128, guStrideBytes: 32,
                                          dnStrideBytes: 16, budgetBytes: 96)
        _ = try cache.slice(key: 7, family: .gateUp, fileOffset: 0, flush: {})
        cache.endExpertIteration(segmentPending: false)
        _ = try cache.slice(key: 8, family: .gateUp, fileOffset: 32, flush: {})
        cache.endExpertIteration(segmentPending: false)
        checkEqual("both gu slots occupied after two loads",
                   cache.stats().residentBytes, 96)
        // Truncate to 40: offset 32 stays within the DECLARED ggufSize
        // (32 + 32 = 64 <= 128) but the physical read delivers only the
        // 8 bytes [32, 40) before EOF.
        let trunc = try FileHandle(forWritingTo: f2)
        try trunc.truncate(atOffset: 40)
        try trunc.close()
        var threwLoad = false
        do {
            _ = try cache.slice(key: 9, family: .gateUp, fileOffset: 32, flush: {})
        } catch { threwLoad = true }
        check("forced partial victim read aborts", threwLoad)
        let stFail = cache.stats()
        checkEqual("failed load read no full slices", stFail.sliceBytesRead, 64)
        checkEqual("failed load counted one miss", stFail.misses, 3)
        // Restore the file; the SAME key must MISS again (it was never
        // cached) and read the exact bytes.
        try Data(fileBytes).write(to: f2)
        let s9 = try cache.slice(key: 9, family: .gateUp, fileOffset: 32, flush: {})
        cache.endExpertIteration(segmentPending: false)
        let stRetry = cache.stats()
        checkEqual("retry after failed load is a miss, not a hit", stRetry.misses, 4)
        checkEqual("failed key was not cached (no bogus hit)", stRetry.hits, 0)
        checkEqual("retry read accounted", stRetry.sliceBytesRead, 96)
        let got = Array(UnsafeBufferPointer(start: s9.buffer.contents()
                .assumingMemoryBound(to: UInt8.self).advanced(by: s9.offset),
                count: 32))
        checkEqual("retried slice bytes match file", got, Array(fileBytes[32..<64]))

        // Prefetch attribution (fresh cache, same tiny file): free-slot
        // read-ahead serves the first hit; per-family slot markers never
        // cross GU/DN; no-free-slot prefetch declines without evicting;
        // a partial-read prefetch failure installs nothing.
        let cache2 = try Gemma4ExpertCache(device: device, fileURL: f2,
                                           ggufSize: 128, guStrideBytes: 32,
                                           dnStrideBytes: 16, budgetBytes: 96)
        _ = try cache2.slice(key: 1, family: .gateUp, fileOffset: 0, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        let pfInstalled = try cache2.prefetch(key: 2, family: .gateUp,
                                              fileOffset: 32)
        check("prefetch installs into free slot", pfInstalled)
        let pfResident = try cache2.prefetch(key: 1, family: .gateUp,
                                             fileOffset: 0)
        check("prefetch of resident key is a no-op", !pfResident)
        // GU pool is full (2/2): key 3 must decline, NOT evict.
        let pfDeclined = try cache2.prefetch(key: 3, family: .gateUp,
                                             fileOffset: 0)
        check("prefetch without free slot declines", !pfDeclined)
        var threwPF = false
        do {
            _ = try cache2.prefetch(key: 4, family: .gateUp, fileOffset: 97)
        } catch { threwPF = true }
        check("prefetch beyond declared size aborts", threwPF)
        // Consume the prefetched slice: a HIT attributed to prefetch
        // exactly once.
        _ = try cache2.slice(key: 2, family: .gateUp, fileOffset: 32, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        var st2 = cache2.stats()
        checkEqual("prefetched key hits", st2.hits, 1)
        checkEqual("prefetch hit attributed once", st2.prefetchHits, 1)
        checkEqual("prefetch slices counted", st2.prefetchSlices, 1)
        // DN family: a demand MISS then HIT must not be prefetch-
        // attributed (the GU-only prefetch marker stays in GU).
        _ = try cache2.slice(key: 2, family: .down, fileOffset: 16, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        _ = try cache2.slice(key: 2, family: .down, fileOffset: 16, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        st2 = cache2.stats()
        checkEqual("dn demand miss counted", st2.misses, 2)
        checkEqual("dn hit counted", st2.hits, 2)
        checkEqual("gu-only prefetch marker not attributed to dn",
                   st2.prefetchHits, 1)
        // A second hit of the consumed GU key is not re-attributed.
        _ = try cache2.slice(key: 2, family: .gateUp, fileOffset: 32, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        st2 = cache2.stats()
        checkEqual("prefetch attribution is first-consumption only",
                   st2.prefetchHits, 1)
        // Failed prefetch via PARTIAL read: DN still has free slots;
        // offset 16 is within the declared ggufSize (16 + 16 = 32 <=
        // 128) but the truncated file delivers 4 bytes then EOF. The
        // failed prefetch must install nothing (byte counters
        // unchanged), and after restoring the file a retry succeeds and
        // serves a prefetch-attributed hit.
        let bytesBeforeFailedPF = st2.sliceBytesRead
        let trunc2 = try FileHandle(forWritingTo: f2)
        try trunc2.truncate(atOffset: 20)
        try trunc2.close()
        var threwPF2 = false
        do {
            _ = try cache2.prefetch(key: 5, family: .down, fileOffset: 16)
        } catch { threwPF2 = true }
        check("partial-read prefetch aborts", threwPF2)
        st2 = cache2.stats()
        checkEqual("failed prefetch read no bytes",
                   st2.sliceBytesRead, bytesBeforeFailedPF)
        try Data(fileBytes).write(to: f2)
        let pf5 = try cache2.prefetch(key: 5, family: .down, fileOffset: 16)
        check("prefetch retry after restore installs", pf5)
        let s5 = try cache2.slice(key: 5, family: .down, fileOffset: 16, flush: {})
        cache2.endExpertIteration(segmentPending: false)
        st2 = cache2.stats()
        checkEqual("retried prefetch hit attributed", st2.prefetchHits, 2)
        let dn5 = Array(UnsafeBufferPointer(start: s5.buffer.contents()
                .assumingMemoryBound(to: UInt8.self).advanced(by: s5.offset),
                count: 16))
        checkEqual("prefetched dn bytes match file", dn5,
                   Array(fileBytes[16..<32]))
    } else {
        print("SKIP: cache loader gates (no Metal device)")
    }
} catch {
    failed += 1; print("FAIL: expert cache component gates — \(error)")
}

// MARK: - Vectorized-load RMSNorm bit gate + Q4 VL rejection diagnostics

do {
    // The vectorizedLoads Forge candidate dispatches g4_rmsnorm_vl
    // (float4 loads, IDENTICAL source arithmetic sequence) when the
    // production guards hold. GATE: EXACT bit equality against the
    // retained original kernel on real hardware across model
    // geometries with cancellation-prone inputs — anything other than
    // bit equality fails (no tolerance). The production
    // alignment/fallback predicate is tested directly. The
    // g4_q4_matvec_vl variant is a MEASURED-REJECTED DIAGNOSTIC ONLY
    // (excluded from the production pipeline list and legal dispatch):
    // its compiled outputs differed from the original kernel (worst
    // rel printed; cause not established), so bit-equality could not
    // be proven despite a faster synthetic per-call time; the prints
    // below preserve the rejection evidence (including an FP64
    // decode/dot oracle bound for the Q4 variant) outside the
    // pass/fail counts.
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw Q4TiledTestError("Metal unavailable")
    }
    let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name),
              let p = try? device.makeComputePipelineState(function: fn) else {
            throw Q4TiledTestError("pipeline \(name) missing")
        }
        return p
    }
    let q4Plain = try pipeline("g4_q4_matvec")
    let q4VL = try pipeline("g4_q4_matvec_vl")
    let rmsPlain = try pipeline("g4_rmsnorm")
    let rmsVL = try pipeline("g4_rmsnorm_vl")

    // Production alignment/fallback predicate: one decision point,
    // tested directly (zero/negative width, non-multiple-of-4 width,
    // unaligned scaled offsets, UInt64.max offset, aligned cases).
    check("rms VL guard rejects n=0", !Gemma4Runner.rmsVectorLoadGuards(n: 0, scaled: false, wOff: 0))
    check("rms VL guard rejects negative n", !Gemma4Runner.rmsVectorLoadGuards(n: -4, scaled: false, wOff: 0))
    check("rms VL guard rejects n%4!=0", !Gemma4Runner.rmsVectorLoadGuards(n: 254, scaled: false, wOff: 0))
    check("rms VL guard rejects misaligned scaled wOff",
          !Gemma4Runner.rmsVectorLoadGuards(n: 256, scaled: true, wOff: 2))
    check("rms VL guard rejects UInt64.max scaled wOff",
          !Gemma4Runner.rmsVectorLoadGuards(n: 256, scaled: true, wOff: UInt64.max))
    check("rms VL guard accepts aligned scaled wOff",
          Gemma4Runner.rmsVectorLoadGuards(n: 2112, scaled: true, wOff: 1 << 20))
    check("rms VL guard accepts unscaled multiple of 4",
          Gemma4Runner.rmsVectorLoadGuards(n: 2816, scaled: false, wOff: 0))

    // Finite Float16 block scales (explicit signed set — random half
    // bits would include NaN/Inf scales), random quant bytes.
    func q4Blocks(rows: Int, blocksPerRow: Int, rng: inout SplitMix64) -> [UInt8] {
        let scales: [Float16] = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25]
        var blocks = [UInt8]()
        blocks.reserveCapacity(rows * blocksPerRow * 18)
        for block in 0..<(rows * blocksPerRow) {
            let d = scales[block % scales.count]
            let bits = d.bitPattern
            blocks.append(UInt8(truncatingIfNeeded: bits))
            blocks.append(UInt8(truncatingIfNeeded: bits >> 8))
            for _ in 0..<16 {
                blocks.append(UInt8(truncatingIfNeeded: rng.next()))
            }
        }
        return blocks
    }

    func runQ4(_ p: MTLComputePipelineState, blocks: [UInt8], outDim: Int,
               inDim: Int, x: [Float]) throws -> [Float] {
        let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
        var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
        var off = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(xb, offset: 0, index: 3)
        enc.setBuffer(yb, offset: 0, index: 4)
        let width = p.maxTotalThreadsPerThreadgroup
        let groups = (outDim + width - 1) / width
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        return [Float](UnsafeBufferPointer(start: yb.contents().assumingMemoryBound(to: Float.self),
                                           count: outDim))
    }

    // Independent FP64 decode/dot oracle: per-row magnitude S = sum |w_i
    // x_i|; the empirical gate is abs(diff) <= 1e-4*S + 1e-6 (identical
    // discipline to the existing tiled-kernel gates).
    func q4Oracle(blocks: [UInt8], outDim: Int, inDim: Int, x: [Float])
        -> (y: [Double], s: [Double]) {
        let blocksPerRow = inDim / 32
        var y = [Double](repeating: 0, count: outDim)
        var s = [Double](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var sum = 0.0
            var mag = 0.0
            for b in 0..<blocksPerRow {
                let base = (row * blocksPerRow + b) * 18
                let dbits = UInt16(blocks[base]) | (UInt16(blocks[base + 1]) << 8)
                let d = Double(Float(Float16(bitPattern: dbits)))
                for l in 0..<16 {
                    let q = blocks[base + 2 + l]
                    let lo = Double(Float(q & 0x0F) - 8)
                    let hi = Double(Float(q >> 4) - 8)
                    let t1 = d * lo * Double(x[b * 32 + l])
                    let t2 = d * hi * Double(x[b * 32 + 16 + l])
                    sum += t1
                    sum += t2
                    mag += abs(t1) + abs(t2)
                }
            }
            y[row] = sum
            s[row] = mag
        }
        return (y, s)
    }

    var rng = SplitMix64(seed: 9090)
    // Q4 matvec: single-block, expert-intermediate, and hidden widths.
    for inDim in [32, 704, 2816] {
        let outDim = 64
        let blocksPerRow = inDim / 32
        let blocks = q4Blocks(rows: outDim, blocksPerRow: blocksPerRow, rng: &rng)
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim {
            x[i] = (i % 2 == 0 ? 1.0 : -1.0) * Float(rng.next() % 1000) / 1000.0
        }
        let a = try runQ4(q4Plain, blocks: blocks, outDim: outDim,
                          inDim: inDim, x: x)
        let b = try runQ4(q4VL, blocks: blocks, outDim: outDim,
                          inDim: inDim, x: x)
        // DIAGNOSTIC ONLY (measured-rejected variant, never dispatched
        // in production): prints the observed deviation of the compiled
        // Q4 VL kernel from the original kernel and from the FP64
        // oracle. Not pass/fail — evidence for the rejection record.
        var worstQ4 = 0.0
        var notBitEqual = 0
        for row in 0..<outDim where a[row].bitPattern != b[row].bitPattern {
            notBitEqual += 1
        }
        for row in 0..<outDim {
            let rel = abs(Double(a[row] - b[row])) / max(1.0, abs(Double(a[row])))
            worstQ4 = max(worstQ4, rel)
        }
        let (ref, mag) = q4Oracle(blocks: blocks, outDim: outDim, inDim: inDim, x: x)
        var worstOracle = 0.0
        for row in 0..<outDim {
            worstOracle = max(worstOracle,
                              abs(Double(b[row]) - ref[row]) / (1e-4 * mag[row] + 1e-6))
        }
        print(String(format: "DIAGNOSTIC q4 matvec vl (inDim %d): %d/%d rows not bit-equal, worst rel vs plain %.3e, worst of FP64 magnitude bound %.3f",
                     inDim, notBitEqual, outDim, worstQ4, worstOracle))
    }

    // cooperative: Bool = false is an OPTIONAL parameter — the default
    // path (all pre-existing call sites) is byte-for-byte the OLD
    // helper behavior; cooperative=true dispatches ONE threadgroup per
    // vector with the hardware-derived thread count and dynamic
    // threadgroup memory from the PUBLIC RMSCooperative enum.
    func runRms(_ p: MTLComputePipelineState, x: [Float], w: [Float]?,
                vectors: Int, inPlace: Bool, cooperative: Bool = false) throws -> [[Float]] {
        let n = x.count / vectors
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = inPlace ? xb : device.makeBuffer(length: x.count * 4,
                                                  options: .storageModeShared)!
        let wb = w.map { device.makeBuffer(bytes: $0, length: $0.count * 4,
                                           options: .storageModeShared)! }
            ?? device.makeBuffer(length: 16, options: .storageModeShared)!
        var dims = SIMD2<UInt32>(UInt32(n), UInt32(vectors))
        var eps: Float = 1e-6
        var withScale: Int32 = w == nil ? 0 : 1
        var wOff = UInt64(0)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        enc.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 1)
        enc.setBytes(&withScale, length: MemoryLayout<Int32>.stride, index: 2)
        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
        enc.setBuffer(wb, offset: 0, index: 4)
        enc.setBuffer(xb, offset: 0, index: 5)
        enc.setBuffer(yb, offset: 0, index: 6)
        if cooperative {
            // Hardware-derived schedule, never a GPU-name guess: threads
            // from the ACTUAL compiled pipeline's threadExecutionWidth,
            // bounded by its max threads per threadgroup and n/4.
            let threads = RMSCooperative.threadCount(
                n: n, executionWidth: p.threadExecutionWidth,
                maxThreads: p.maxTotalThreadsPerThreadgroup)
            let tgLen = RMSCooperative.threadgroupMemoryLength(n: n)
            guard threads > 0, tgLen > 0,
                  p.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
                  tgLen <= device.maxThreadgroupMemoryLength
                      - p.staticThreadgroupMemoryLength else {
                throw Q4TiledTestError("cooperative schedule invalid for n \(n)")
            }
            enc.setThreadgroupMemoryLength(tgLen, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: vectors, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        } else {
            let width = p.maxTotalThreadsPerThreadgroup
            let groups = (vectors + width - 1) / width
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed, cmd.error == nil else {
            throw Q4TiledTestError("kernel failed: \(String(describing: cmd.error))")
        }
        let ptr = yb.contents().assumingMemoryBound(to: Float.self)
        return (0..<vectors).map { v in
            [Float](UnsafeBufferPointer(start: ptr + v * n, count: n))
        }
    }

    // RMSNorm: model widths 2112/256/2816, weighted and scale-free,
    // out-of-place and in-place.
    for (n, vectors, weighted, inPlace) in [(256, 4, true, false),
                                            (2112, 2, true, true),
                                            (2112, 2, false, false),
                                            (2816, 1, true, false)] {
        var x = [Float](repeating: 0, count: n * vectors)
        for i in 0..<x.count {
            x[i] = (i % 2 == 0 ? 1.0 : -1.0) * Float(rng.next() % 1000) / 1000.0
        }
        var w: [Float]? = nil
        if weighted {
            w = [Float](repeating: 0, count: n)
            for i in 0..<n { w![i] = i % 2 == 0 ? 0.5 : -0.5 }
        }
        let a = try runRms(rmsPlain, x: x, w: w, vectors: vectors, inPlace: inPlace)
        let b = try runRms(rmsVL, x: x, w: w, vectors: vectors, inPlace: inPlace)
        check("rmsnorm vl finite (n \(n), weighted=\(weighted), inPlace=\(inPlace))",
              a.flatMap { $0 }.allSatisfy(\.isFinite)
              && b.flatMap { $0 }.allSatisfy(\.isFinite))
        check("rmsnorm vl BIT-IDENTICAL to original (n \(n), weighted=\(weighted), inPlace=\(inPlace))",
              a.count == b.count && zip(a, b).allSatisfy {
                  $0.count == $1.count && zip($0, $1).allSatisfy {
                      $0.bitPattern == $1.bitPattern }
              })
    }

    // Cooperative RMSNorm: build the ACTUAL production pipeline from
    // the PUBLIC RMSCooperative enum source on the test device, then
    // assert EXACT bit equality against the incumbent rmsVL kernel for
    // every tested geometry (256/704/2816 x weighted/unweighted x
    // out-of-place/in-place, multi-vector). The bit-equality gate is
    // not weakened: exact bitPattern comparison, no tolerance.
    let rmsCoopLibrary = try device.makeLibrary(source: RMSCooperative.source,
                                                options: nil)
    guard let rmsCoopFn = rmsCoopLibrary.makeFunction(name: RMSCooperative.kernelName) else {
        throw Q4TiledTestError("RMS cooperative kernel \(RMSCooperative.kernelName) missing")
    }
    let rmsCoop = try device.makeComputePipelineState(function: rmsCoopFn)
    check("rms cooperative source pipeline compiled", rmsCoop.threadExecutionWidth > 0)
    for (n, vectors, weighted, inPlace) in [(256, 4, true, false),
                                            (256, 2, false, true),
                                            (256, 2, true, true),
                                            (256, 3, false, false),
                                            (704, 3, true, true),
                                            (704, 2, false, false),
                                            (704, 2, true, false),
                                            (704, 4, false, true),
                                            (2816, 2, true, false),
                                            (2816, 2, false, true),
                                            (2816, 3, true, true),
                                            (2816, 2, false, false)] {
        var x = [Float](repeating: 0, count: n * vectors)
        for i in 0..<x.count {
            x[i] = (i % 2 == 0 ? 1.0 : -1.0) * Float(rng.next() % 1000) / 1000.0
        }
        var w: [Float]? = nil
        if weighted {
            w = [Float](repeating: 0, count: n)
            for i in 0..<n { w![i] = i % 2 == 0 ? 0.5 : -0.5 }
        }
        let ref = try runRms(rmsVL, x: x, w: w, vectors: vectors, inPlace: inPlace)
        let coop = try runRms(rmsCoop, x: x, w: w, vectors: vectors,
                              inPlace: inPlace, cooperative: true)
        check("rmsnorm coop finite (n \(n), vectors \(vectors), weighted=\(weighted), inPlace=\(inPlace))",
              coop.flatMap { $0 }.allSatisfy(\.isFinite))
        check("rmsnorm coop BIT-IDENTICAL to vl kernel (n \(n), vectors \(vectors), weighted=\(weighted), inPlace=\(inPlace))",
              ref.count == coop.count && zip(ref, coop).allSatisfy {
                  $0.count == $1.count && zip($0, $1).allSatisfy {
                      $0.bitPattern == $1.bitPattern }
              })
    }

    // Hardened derivation guard cases: invalid n / executionWidth /
    // maxThreads return 0 (never trap, never a silent nonzero), and
    // the documented M1 Max schedule is preserved exactly.
    checkEqual("threadCount invalid n returns 0",
               RMSCooperative.threadCount(n: 0, executionWidth: 32, maxThreads: 1024), 0)
    checkEqual("threadCount negative n returns 0",
               RMSCooperative.threadCount(n: -256, executionWidth: 32, maxThreads: 1024), 0)
    checkEqual("threadCount invalid width returns 0",
               RMSCooperative.threadCount(n: 256, executionWidth: 0, maxThreads: 1024), 0)
    checkEqual("threadCount invalid maxThreads returns 0",
               RMSCooperative.threadCount(n: 256, executionWidth: 32, maxThreads: 0), 0)
    checkEqual("threadCount gigantic width clamps to maxThreads",
               RMSCooperative.threadCount(n: 8192, executionWidth: Int.max / 2,
                                          maxThreads: 1024), 1024)
    checkEqual("threadCount M1 Max derivation preserved (n >= 256 -> 64)",
               RMSCooperative.threadCount(n: 256, executionWidth: 32, maxThreads: 1024), 64)
    checkEqual("threadCount M1 Max derivation preserved (n = 2816 -> 64)",
               RMSCooperative.threadCount(n: 2816, executionWidth: 32, maxThreads: 1024), 64)
    checkEqual("threadgroupMemoryLength invalid n returns 0",
               RMSCooperative.threadgroupMemoryLength(n: 0), 0)
    checkEqual("threadgroupMemoryLength non-multiple-of-4 returns 0",
               RMSCooperative.threadgroupMemoryLength(n: 258), 0)
    checkEqual("threadgroupMemoryLength gigantic n returns 0",
               RMSCooperative.threadgroupMemoryLength(n: Int.max), 0)
    checkEqual("threadgroupMemoryLength exact for n = 256",
               RMSCooperative.threadgroupMemoryLength(n: 256), 1040)

    // Coarse per-kernel timing (record only — one dedicated command
    // buffer per call, so this understates batched throughput; the
    // Forge wall/GPU breakdown measures the end-to-end impact).
    do {
        let inDim = 2816, outDim = 512
        let blocks = q4Blocks(rows: outDim, blocksPerRow: inDim / 32, rng: &rng)
        var x = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { x[i] = Float(rng.next() % 1000) / 1000.0 }
        func timeQ4(_ p: MTLComputePipelineState) throws -> Double {
            _ = try runQ4(p, blocks: blocks, outDim: outDim, inDim: inDim, x: x)
            let t0 = DispatchTime.now()
            for _ in 0..<20 { _ = try runQ4(p, blocks: blocks, outDim: outDim, inDim: inDim, x: x) }
            return Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 20000.0
        }
        let plainUs = try timeQ4(q4Plain)
        let vlUs = try timeQ4(q4VL)
        print(String(format: "q4 matvec per-call, dedicated-buffer submit (record only): plain %.0f us, vectorized %.0f us",
                     plainUs, vlUs))
    }
} catch {
    failed += 1; print("FAIL: vectorized-load bit-equality gates — \(error)")
}


// MARK: - Q4 parameterized operator lowering (phase 9)

/// Typed Q4-lowering error-case check: passes only when the body throws
/// EXACTLY the expected Q4LoweringError case (Equatable), so a different
/// failure path cannot satisfy the check.
@MainActor
func checkQ4Throws(_ name: String, _ expected: Q4LoweringError,
                   _ body: () throws -> Void) {
    do {
        try body()
        failed += 1; print("FAIL: \(name) — expected \(expected), got no error")
    } catch let error as Q4LoweringError {
        if error == expected { passed += 1 }
        else { failed += 1; print("FAIL: \(name) — expected \(expected), got \(error)") }
    } catch {
        failed += 1; print("FAIL: \(name) — expected \(expected), got unexpected \(error)")
    }
}

/// Builds op JSON matching the Python capture schema.
func q4OpJSON(op: String = "q4_0.matvec", inDim: Int = 2816, outDim: Int = 4096,
              tensor: Any? = nil, rows: Any? = 256, view: Any? = nil) throws -> Data {
    let obj: [String: Any] = [
        "schema": 1, "op": op, "semantics": "dot(W_row, x)",
        "inDim": inDim, "outDim": outDim,
        "layout": "row-major-q4_0-blocks", "quant": "q4_0",
        "accumulate": "f32", "inputPrecision": "f32",
        "tensor": tensor, "representativeRows": rows,
        "view": view,
    ]
    return try JSONSerialization.data(withJSONObject: obj)
}

do {
    // ---- op IR validation ----
    let good = try Q4OpIR.parse(q4OpJSON())
    checkEqual("q4 op IR dims", good.inDim, 2816)
    checkEqual("q4 op IR digest stable", try good.digest(), try good.digest())
    checkQ4Throws("q4 op IR rejects unknown op",
                  .invalidOp("unsupported op/semantics q8.matvec/dot(W_row, x)")) {
        _ = try Q4OpIR.parse(q4OpJSON(op: "q8.matvec"))
    }
    checkQ4Throws("q4 op IR rejects non-block inDim",
                  .invalidOp("bad dims inDim 2817 outDim 4096")) {
        _ = try Q4OpIR.parse(q4OpJSON(inDim: 2817))
    }
    checkQ4Throws("q4 op IR rejects representativeRows > outDim",
                  .invalidOp("representativeRows 65 out of bounds")) {
        _ = try Q4OpIR.parse(q4OpJSON(outDim: 64, rows: 65))
    }

    // ---- deriver: shape + hardware dependence, bounded, baseline first ----
    let hw = ForgeHardwareProfile(deviceName: "test", maxTotalThreadsPerThreadgroup: 1024,
                                  threadExecutionWidth: 8, recommendedMaxWorkingSetSize: 1 << 30)
    let derived = Q4ScheduleDeriver.derive(op: good, hardware: hw)
    check("q4 deriver bounded to 13", derived.count <= 13)
    checkEqual("q4 deriver baseline first",
               derived.first, Q4Schedule(threadsPerRow: 1, rowsPerGroup: 1,
                                         accumulation: .f32, xLoads: .scalar))
    check("q4 deriver threads depend on shape (blocksPerRow 88 admits T<=22)",
          derived.contains { $0.threadsPerRow == 8 }
              && !derived.contains { $0.threadsPerRow == 32 })
    // Phase-11 row-tile family: executionWidth-derived T=1 tiles plus the
    // 1024-cap control are derived candidates for the full-shape op.
    for r in [8, 16, 32, 64, 1024] {
        check("q4 deriver emits row tile R=\(r) (w=8 cap=1024)",
              derived.contains { $0.threadsPerRow == 1 && $0.rowsPerGroup == r })
    }
    let smallOp = try Q4OpIR.parse(q4OpJSON(inDim: 128, outDim: 64, rows: 32))
    checkEqual("q4 deriver: 4-block rows admit the baseline first",
               Q4ScheduleDeriver.derive(op: smallOp, hardware: hw).first,
               Q4Schedule(threadsPerRow: 1, rowsPerGroup: 1,
                          accumulation: .f32, xLoads: .scalar))
    check("q4 deriver: 4-block rows admit no cooperative T (only T=1 row tiles)",
          Q4ScheduleDeriver.derive(op: smallOp, hardware: hw)
              .allSatisfy { $0.threadsPerRow == 1 })
    let tinyHw = ForgeHardwareProfile(deviceName: "test", maxTotalThreadsPerThreadgroup: 8,
                                      threadExecutionWidth: 4, recommendedMaxWorkingSetSize: 1 << 30)
    check("q4 deriver respects threadgroup maximum",
          Q4ScheduleDeriver.derive(op: good, hardware: tinyHw)
              .allSatisfy { $0.threadsPerRow * $0.rowsPerGroup <= 8 })
    check("q4 deriver row tiles respect the threadgroup maximum (w=4 cap=8 -> R<=8)",
          Q4ScheduleDeriver.derive(op: good, hardware: tinyHw)
              .filter { $0.threadsPerRow == 1 }
              .allSatisfy { $0.rowsPerGroup <= 8 })

    // ---- row-tile schedule legality ----
    check("q4 row tile R=256 legal at cap 1024",
          Q4Schedule(threadsPerRow: 1, rowsPerGroup: 256,
                     accumulation: .f32, xLoads: .scalar)
              .isLegal(blocksPerRow: 88, hardware: hw))
    check("q4 row tile R=2048 illegal (above 1024 bound)",
          !Q4Schedule(threadsPerRow: 1, rowsPerGroup: 2048,
                      accumulation: .f32, xLoads: .scalar)
              .isLegal(blocksPerRow: 88, hardware: hw))
    check("q4 row tile R=96 illegal (not a power of two)",
          !Q4Schedule(threadsPerRow: 1, rowsPerGroup: 96,
                      accumulation: .f32, xLoads: .scalar)
              .isLegal(blocksPerRow: 88, hardware: hw))
    check("q4 row tile R=256 illegal at cap 8",
          !Q4Schedule(threadsPerRow: 1, rowsPerGroup: 256,
                      accumulation: .f32, xLoads: .scalar)
              .isLegal(blocksPerRow: 88, hardware: tinyHw))

    // ---- emitter: deterministic, parameterized, distinct ----
    let s1 = Q4Schedule(threadsPerRow: 8, rowsPerGroup: 1,
                        accumulation: .f32compensated, xLoads: .float4)
    let e1 = try Q4KernelEmitter.emit(op: good, schedule: s1)
    checkEqual("q4 emitter deterministic", e1, try Q4KernelEmitter.emit(op: good, schedule: s1))
    check("q4 emitter bakes thread mapping",
          e1.contains("constexpr uint T = 8u") && e1.contains("constexpr uint R = 1u"))
    check("q4 emitter names kernel from schedule", e1.contains(s1.kernelName))
    let e2 = try Q4KernelEmitter.emit(op: good, schedule: Q4Schedule(
        threadsPerRow: 4, rowsPerGroup: 2, accumulation: .f32, xLoads: .scalar))
    check("q4 emitter distinguishes schedules", e1 != e2)
    checkQ4Throws("q4 emitter rejects illegal schedule for shape",
                  .emissionFailed("schedule Q4Schedule(threadsPerRow: 8, rowsPerGroup: 1, "
                                  + "accumulation: LeviathanCore.Q4Schedule.Accumulation.f32compensated, "
                                  + "xLoads: LeviathanCore.Q4Schedule.XLoads.float4) "
                                  + "illegal for inDim 128")) {
        _ = try Q4KernelEmitter.emit(op: smallOp, schedule: s1)
    }

    // ---- emitter v2: T=1 ROW-TILE direct mapping ----
    let tile = Q4Schedule(threadsPerRow: 1, rowsPerGroup: 128,
                          accumulation: .f32, xLoads: .scalar)
    let et = try Q4KernelEmitter.emit(op: good, schedule: tile)
    check("q4 emitter row tile bakes R",
          et.contains("constexpr uint R = 128u"))
    check("q4 emitter row tile emits the row mapping",
          et.contains("uint row = wid * R + lid;"))
    check("q4 emitter row tile writes rows directly (no barrier/partials)",
          !et.contains("threadgroup_barrier")
              && !et.contains("threadgroup float partials")
              && et.contains("y[row] = sum;"))
    check("q4 emitter row tile codegen is v2",
          et.contains("q4gen-v3"))
    check("q4 emitter T>1 keeps the cooperative barrier structure",
          e1.contains("threadgroup_barrier")
              && e1.contains("threadgroup float partials"))
    checkEqual("q4 emitter row tile deterministic", et,
               try Q4KernelEmitter.emit(op: good, schedule: tile))

    // ---- op IR expert view validation ----
    let expertOp = try Q4OpIR.parse(q4OpJSON(inDim: 2816, outDim: 1408,
                                             tensor: "blk.0.ffn_gate_up_exps.weight",
                                             view: "expert-row-slice"))
    checkEqual("q4 expert view op dims", expertOp.inDim, 2816)
    checkQ4Throws("q4 op IR rejects unknown view",
                  .invalidOp("unsupported view bogus")) {
        _ = try Q4OpIR.parse(q4OpJSON(view: "bogus"))
    }
    checkQ4Throws("q4 op IR rejects expert view without a tensor",
                  .invalidOp("expert-row-slice binds a blk.<L>.ffn_{gate_up,down}_exps.weight tensor")) {
        _ = try Q4OpIR.parse(q4OpJSON(view: "expert-row-slice"))
    }
    checkQ4Throws("q4 op IR rejects expert view on a non-family tensor",
                  .invalidOp("expert-row-slice binds a blk.<L>.ffn_{gate_up,down}_exps.weight tensor")) {
        _ = try Q4OpIR.parse(q4OpJSON(tensor: "blk.0.attn_q.weight",
                                       view: "expert-row-slice"))
    }

    // ---- family routing predicate (pure production decision) ----
    func serves(_ bound: String, _ bi: Int, _ bo: Int, _ bv: String?,
                _ tensor: String, _ inDim: Int, _ outRows: Int) -> Bool {
        Gemma4Runner.q4GeneratedServesTensor(boundTensor: bound, boundInDim: bi,
                                             boundOutDim: bo, boundView: bv,
                                             tensor: tensor, inDim: inDim,
                                             outRows: outRows)
    }
    check("q4 family routing: bound tensor exact match",
          serves("blk.0.ffn_gate_up_exps.weight", 2816, 1408, "expert-row-slice",
                 "blk.0.ffn_gate_up_exps.weight", 2816, 1408))
    check("q4 family routing: every layer's same-shaped GU slice served",
          serves("blk.0.ffn_gate_up_exps.weight", 2816, 1408, "expert-row-slice",
                 "blk.29.ffn_gate_up_exps.weight", 2816, 1408))
    check("q4 family routing: DN family NOT served by a GU binding",
          !serves("blk.0.ffn_gate_up_exps.weight", 2816, 1408, "expert-row-slice",
                  "blk.0.ffn_down_exps.weight", 2816, 1408))
    check("q4 family routing: shape deviation falls through",
          !serves("blk.0.ffn_gate_up_exps.weight", 2816, 1408, "expert-row-slice",
                  "blk.1.ffn_gate_up_exps.weight", 2816, 1410))
    check("q4 family routing: 2-D binding never serves other tensors",
          !serves("blk.0.attn_q.weight", 2816, 4096, nil,
                  "blk.1.attn_q.weight", 2816, 4096))
    check("q4 family routing: expert view required for family spread",
          !serves("blk.0.ffn_gate_up_exps.weight", 2816, 1408, nil,
                  "blk.29.ffn_gate_up_exps.weight", 2816, 1408))

    // ---- FP64 oracle: hand-computed known block ----
    do {
        var block = [UInt8](repeating: 0, count: 18)
        block[0] = 0x00; block[1] = 0x3C   // FP16 1.0
        for l in 0..<16 { block[2 + l] = UInt8(l) | (UInt8(16 + l) << 4) }
        // Decoded values are nibble-8 in [-8, 7]: the packed high nibble
        // is ((16+l) mod 16) = l, so byte l decodes to (l-8) + (l-8) and
        // the row sum over x=1 is 2*sum(l=0...15) - 16*16 = -16.
        let ref = try Q4Oracle.matvecFP64(blocks: block[...], outDim: 1, inDim: 32,
                                          x: [Float](repeating: 1, count: 32))
        checkEqual("q4 FP64 oracle hand-computed block", ref[0], -16.0)
    }
    do {
        var block = [UInt8](repeating: 0, count: 18)
        block[0] = 0xFF; block[1] = 0x7F   // FP16 NaN
        _ = try Q4Oracle.matvecFP64(blocks: block[...], outDim: 1, inDim: 32,
                                    x: [Float](repeating: 1, count: 32))
        failed += 1; print("FAIL: q4 FP64 oracle rejects nonfinite scale — no error")
    } catch let error as Q4LoweringError {
        if case .oracleFailed = error { passed += 1 }
        else { failed += 1; print("FAIL: q4 FP64 oracle rejects nonfinite scale — got \(error)") }
    }
} catch {
    failed += 1; print("FAIL: q4 lowering IR/deriver/emitter/oracle — threw \(error)")
}

// ---- GPU component gates + artifact identity (synthetic op, no GGUF) ----
if let device = MTLCreateSystemDefaultDevice() {
    do {
        // Unbound synthetic op: fast, no 14 GB model needed.
        let op = try Q4OpIR.parse(q4OpJSON(inDim: 2048, outDim: 64, tensor: nil, rows: 32))
        let hwProfile = try Q4CurrentHardware.profile()
        let blocks = try Q4ComponentInputs.weightBytes(op: op, file: nil)
        let x = Q4ComponentInputs.activation(op: op)
        let ref64 = try Q4Oracle.matvecFP64(blocks: blocks[...], outDim: 32,
                                            inDim: 2048, x: x)

        // Emitted T=1 baseline vs the CPU FP64 oracle (existing contract).
        let baseline = Q4Schedule(threadsPerRow: 1, rowsPerGroup: 1,
                                  accumulation: .f32, xLoads: .scalar)
        let k1 = try Q4EmittedKernel.compile(op: op, schedule: baseline, device: device)
        let y1 = try k1.matvec(blocks: blocks, outDim: 32, inDim: 2048, x: x)
        var w1 = 0.0
        for i in 0..<y1.count {
            w1 = max(w1, abs(Double(y1[i]) - ref64[i]) / max(1.0, abs(ref64[i])))
        }
        check(String(format: "q4 emitted T=1 vs FP64 oracle (worst rel %.3e)", w1), w1 < 1e-4)

        // Phase-11 ROW-TILE component bit gates against the ACTUAL
        // incumbent g4_q4_matvec (compiled from the runner's embedded
        // MSL, incumbent dispatch geometry): every T=1 row tile must be
        // MEASURED bit-identical (bitPattern) on the same inputs,
        // including a ragged tail group (rows 48, R=32 -> 2 groups, the
        // second covers only 16 rows).
        let incLib = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
        let incFn = incLib.makeFunction(name: "g4_q4_matvec")!
        let incPipe = try device.makeComputePipelineState(function: incFn)
        func incumbentMatvec(blocks: [UInt8], outDim: Int, inDim: Int,
                             x: [Float]) throws -> [Float] {
            let queue = device.makeCommandQueue()!
            let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                       options: .storageModeShared)!
            let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                       options: .storageModeShared)!
            let yb = device.makeBuffer(length: outDim * 4, options: .storageModeShared)!
            var param = SIMD2<UInt32>(UInt32(inDim), UInt32(outDim))
            var off = UInt64(0)
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(incPipe)
            enc.setBytes(&param, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
            enc.setBytes(&off, length: MemoryLayout<UInt64>.stride, index: 1)
            enc.setBuffer(wb, offset: 0, index: 2)
            enc.setBuffer(xb, offset: 0, index: 3)
            enc.setBuffer(yb, offset: 0, index: 4)
            let width = incPipe.maxTotalThreadsPerThreadgroup
            let groups = (outDim + width - 1) / width
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw Q4LoweringError.componentGateFailed("incumbent command buffer")
            }
            return [Float](UnsafeBufferPointer(
                start: yb.contents().assumingMemoryBound(to: Float.self), count: outDim))
        }
        do {
            let tileOp = try Q4OpIR.parse(q4OpJSON(inDim: 2048, outDim: 64, tensor: nil,
                                                   rows: 48))
            let tileBlocks = try Q4ComponentInputs.weightBytes(op: tileOp, file: nil)
            let tileX = Q4ComponentInputs.activation(op: tileOp)
            let tileRef = try Q4Oracle.matvecFP64(blocks: tileBlocks[...], outDim: 48,
                                                  inDim: 2048, x: tileX)
            let incOut = try incumbentMatvec(blocks: tileBlocks, outDim: 48,
                                             inDim: 2048, x: tileX)
            for r in [8, 32] {   // 6 groups, and ragged tail (48 = 32 + 16)
                let sched = Q4Schedule(threadsPerRow: 1, rowsPerGroup: r,
                                       accumulation: .f32, xLoads: .scalar)
                guard sched.isLegal(blocksPerRow: 2048 / 32, hardware: hwProfile) else {
                    print("SKIP: q4 row tile R=\(r) not legal on this hardware")
                    continue
                }
                let k = try Q4EmittedKernel.compile(op: tileOp, schedule: sched,
                                                    device: device)
                let out = try k.matvec(blocks: tileBlocks, outDim: 48, inDim: 2048,
                                       x: tileX)
                check("q4 row tile R=\(r) MEASURED bit-identical to incumbent g4_q4_matvec",
                      out.count == incOut.count
                          && zip(out, incOut).allSatisfy { $0.bitPattern == $1.bitPattern })
                var wrel = 0.0
                for i in 0..<out.count {
                    wrel = max(wrel, abs(Double(out[i]) - tileRef[i]) / max(1.0, abs(tileRef[i])))
                }
                check(String(format: "q4 row tile R=%d vs FP64 oracle (worst rel %.3e)", r, wrel),
                      wrel < 1e-4)
            }
        } catch {
            failed += 1; print("FAIL: q4 row-tile incumbent bit gates — threw \(error)")
        }

        // Compensated cooperative kernel vs the same oracle.
        let coop = Q4Schedule(threadsPerRow: 8, rowsPerGroup: 1,
                              accumulation: .f32compensated, xLoads: .float4)
        if coop.isLegal(blocksPerRow: 2048 / 32, hardware: hwProfile) {
            let k8 = try Q4EmittedKernel.compile(op: op, schedule: coop, device: device)
            let y8 = try k8.matvec(blocks: blocks, outDim: 32, inDim: 2048, x: x)
            var w8 = 0.0
            for i in 0..<y8.count {
                w8 = max(w8, abs(Double(y8[i]) - ref64[i]) / max(1.0, abs(ref64[i])))
            }
            check(String(format: "q4 emitted T=8 compensated vs FP64 oracle (worst rel %.3e)", w8),
                  w8 < 1e-4)
        } else {
            print("SKIP: q4 T=8 not legal on this hardware profile")
        }

        // Full component forge on the synthetic op -> artifact identity.
        let (artifact, results, incRel) = try Q4LoweringForge.forge(
            opJSON: q4OpJSON(inDim: 2048, outDim: 64, tensor: nil, rows: 32), ggufURL: nil)
        check("q4 forge incumbent within 1e-4 of FP64 oracle", incRel < 1e-4)
        check("q4 forge produced accepted candidates",
              !results.isEmpty && results.contains { $0.accepted })
        check("q4 forge selected an accepted schedule",
              results.first { $0.schedule == artifact.schedule }!.accepted)
        // Valid identity round-trip.
        try artifact.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                              executableDigest: try Gemma4Runner.executableDigest())
        check("q4 artifact validates round-trip", true)
        // Policy contract on the real synthesized artifact: the explicit
        // unverified policy is encoded, required on decode, and the only
        // representable value; a schema-1 record is rejected even when it
        // carries the policy.
        do {
            let encoded = try JSONEncoder().encode(artifact)
            let decoded = try JSONDecoder().decode(Q4LoweringArtifact.self, from: encoded)
            check("q4 artifact policy round-trips through encode/decode",
                  decoded == artifact && decoded.modelFileVerification == .cachedExplicitRefresh)
            var obj = (try JSONSerialization.jsonObject(with: encoded) as? [String: Any]) ?? [:]
            checkEqual("q4 artifact encoding carries the cached explicit-refresh policy",
                       obj["modelFileVerification"] as? String ?? "", "cached-explicit-refresh-v1")
            var missing = obj
            missing.removeValue(forKey: "modelFileVerification")
            let missingDecoded = try? JSONDecoder().decode(
                Q4LoweringArtifact.self, from: JSONSerialization.data(withJSONObject: missing))
            check("q4 artifact without the policy fails to decode", missingDecoded == nil)
            var claimed = obj
            claimed["modelFileVerification"] = "sha256-full-file"
            let claimedDecoded = try? JSONDecoder().decode(
                Q4LoweringArtifact.self, from: JSONSerialization.data(withJSONObject: claimed))
            check("q4 artifact claiming verified model content fails to decode",
                  claimedDecoded == nil)
            obj["artifactSchema"] = 1
            let legacy = try JSONDecoder().decode(
                Q4LoweringArtifact.self, from: JSONSerialization.data(withJSONObject: obj))
            checkQ4Throws("q4 schema-1 artifact rejected even with the policy",
                          .invalidArtifact("schema 1")) {
                try legacy.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                                    executableDigest: try Gemma4Runner.executableDigest())
            }
        }
        // A real-looking 64-hex model hash never satisfies the identity check.
        checkQ4Throws("q4 artifact rejects a stale 64-hex model hash",
                      .invalidArtifact("GGUF digest mismatch")) {
            try artifact.validate(op: artifact.op, hardware: hwProfile,
                                  ggufDigest: String(repeating: "e", count: 64),
                                  executableDigest: try Gemma4Runner.executableDigest())
        }
        // Tampered source is rejected (JSON-level mutation).
        do {
            let encoded = try JSONEncoder().encode(artifact)
            var obj = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
            obj["emittedSource"] = artifact.emittedSource + "\n//x"
            let tampered = try JSONDecoder().decode(
                Q4LoweringArtifact.self, from: JSONSerialization.data(withJSONObject: obj))
            checkQ4Throws("q4 artifact rejects tampered source",
                          .invalidArtifact("emitted source tampered/rule drift")) {
                try tampered.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                                      executableDigest: try Gemma4Runner.executableDigest())
            }
        }
        // Stale hardware is rejected.
        checkQ4Throws("q4 artifact rejects hardware mismatch",
                      .invalidArtifact("hardware mismatch")) {
            try artifact.validate(op: artifact.op,
                                  hardware: ForgeHardwareProfile(
                                    deviceName: "other", maxTotalThreadsPerThreadgroup: 7,
                                    threadExecutionWidth: 7, recommendedMaxWorkingSetSize: 7),
                                  ggufDigest: "",
                                  executableDigest: try Gemma4Runner.executableDigest())
        }
    } catch {
        failed += 1; print("FAIL: q4 lowering GPU/artifact gates — threw \(error)")
    }
} else {
    print("SKIP: q4 lowering GPU gates (no Metal device)")
}

// MARK: - Phase 10: RMS lowering (op IR, deriver, emitter, oracle, GPU
// bit-identity, artifact identity)

/// Typed RMS-lowering error-case check: passes only when the body throws
/// EXACTLY the expected RMSLoweringError case (Equatable).
@MainActor
func checkRMSThrows(_ name: String, _ expected: RMSLoweringError,
                    _ body: () throws -> Void) {
    do {
        try body()
        failed += 1; print("FAIL: \(name) — expected \(expected), got no error")
    } catch let error as RMSLoweringError {
        if error == expected { passed += 1 }
        else { failed += 1; print("FAIL: \(name) — expected \(expected), got \(error)") }
    } catch {
        failed += 1; print("FAIL: \(name) — expected \(expected), got unexpected \(error)")
    }
}

/// Builds op JSON matching the Python capture schema (unscaled by default,
/// so the default suite needs no GGUF).
func rmsOpJSON(op: String = "rmsnorm", n: Int = 2048, vectors: Int = 4,
               scaled: Bool = false, eps: Any = 1e-6, tensor: Any? = nil,
               reps: Any? = 4) throws -> Data {
    let obj: [String: Any] = [
        "schema": 1, "op": op, "semantics": "y[i] = x[i] * inv_rms(x) * w[i]",
        "n": n, "vectors": vectors, "scaled": scaled, "eps": eps,
        "layout": "row-major-f32-vectors", "quant": "f32",
        "reduction": "serial-order-i0..n", "inputPrecision": "f32",
        "tensor": tensor, "representativeVectors": reps,
    ]
    return try JSONSerialization.data(withJSONObject: obj)
}

do {
    // ---- op IR validation ----
    let good = try RMSOpIR.parse(rmsOpJSON())
    checkEqual("rms op IR dims", good.n, 2048)
    checkEqual("rms op IR digest stable", try good.digest(), try good.digest())
    checkRMSThrows("rms op IR rejects unknown op",
                   .invalidOp("unsupported op/semantics layernorm/y[i] = x[i] * inv_rms(x) * w[i]")) {
        _ = try RMSOpIR.parse(rmsOpJSON(op: "layernorm"))
    }
    checkRMSThrows("rms op IR rejects non-multiple-of-4 n",
                   .invalidOp("bad shape n 2049 vectors 4")) {
        _ = try RMSOpIR.parse(rmsOpJSON(n: 2049))
    }
    checkRMSThrows("rms op IR rejects bad eps",
                   .invalidOp("bad eps 0.5")) {
        _ = try RMSOpIR.parse(rmsOpJSON(eps: 0.5))
    }

    // ---- deriver: shape + hardware dependence, bounded, baseline first ----
    let hw = RMSHardwareProfile(deviceName: "test", maxTotalThreadsPerThreadgroup: 1024,
                                threadExecutionWidth: 8, maxThreadgroupMemoryLength: 1 << 16,
                                recommendedMaxWorkingSetSize: 1 << 30)
    let derived = RMSScheduleDeriver.derive(op: good, hardware: hw)
    check("rms deriver bounded to 6", derived.count <= 6)
    checkEqual("rms deriver baseline first", derived.first,
               RMSSchedule(threadsPerVector: 1, staging: .scalar))
    // Width-8 hardware: the cooperative entries must be the two log2-nearest
    // legal widths (8 and 4 here — derived, not a fixed set).
    check("rms deriver T values follow the measured width",
          derived.dropFirst().allSatisfy { [4, 8].contains($0.threadsPerVector) })
    // Threadgroup MEMORY bound is a hard legality input: the staged array
    // (n*4 bytes + the invShared slot) exceeds a 4 KB cap for n = 2048, so
    // NOTHING is legal — not even the baseline (it stages too).
    let tinyMem = RMSHardwareProfile(deviceName: "test", maxTotalThreadsPerThreadgroup: 1024,
                                     threadExecutionWidth: 8, maxThreadgroupMemoryLength: 4096,
                                     recommendedMaxWorkingSetSize: 1 << 30)
    checkEqual("rms deriver respects threadgroup memory cap (empty, not illegal baseline)",
               RMSScheduleDeriver.derive(op: good, hardware: tinyMem), [])
    let tinyThreads = RMSHardwareProfile(deviceName: "test", maxTotalThreadsPerThreadgroup: 4,
                                         threadExecutionWidth: 2, maxThreadgroupMemoryLength: 1 << 16,
                                         recommendedMaxWorkingSetSize: 1 << 30)
    check("rms deriver respects threadgroup thread cap",
          RMSScheduleDeriver.derive(op: good, hardware: tinyThreads)
              .allSatisfy { $0.threadsPerVector <= 4 })

    // ---- emitter: deterministic, parameterized, distinct ----
    let s1 = RMSSchedule(threadsPerVector: 8, staging: .float4)
    let e1 = try RMSKernelEmitter.emit(op: good, schedule: s1)
    checkEqual("rms emitter deterministic", e1, try RMSKernelEmitter.emit(op: good, schedule: s1))
    check("rms emitter bakes shape and mapping",
          e1.contains("constexpr uint N = 2048u") && e1.contains("constexpr uint T = 8u"))
    check("rms emitter names kernel from schedule", e1.contains(s1.kernelName))
    let e2 = try RMSKernelEmitter.emit(op: good, schedule: RMSSchedule(threadsPerVector: 4,
                                                                        staging: .scalar))
    check("rms emitter distinguishes schedules", e1 != e2)
    checkRMSThrows("rms emitter rejects T exceeding n",
                   .emissionFailed("schedule RMSSchedule(threadsPerVector: 4096, staging: "
                                   + "LeviathanCore.RMSSchedule.Staging.scalar) illegal for n 2048")) {
        _ = try RMSKernelEmitter.emit(op: good, schedule: RMSSchedule(threadsPerVector: 4096,
                                                                       staging: .scalar))
    }

    // ---- FP64 oracle: hand-computed case ----
    do {
        // x = [3, 4, 0, 0], w = [2, 1, 1, 1], eps = 0:
        // ms = 25, mean 6.25, inv = 0.4, y = [2.4, 1.6, 0, 0]. (0.4 is not
        // an exact binary double, so compare at 1e-12 relative, not ==.)
        let ref = try RMSOracle.rmsnormFP64(x: [3, 4, 0, 0], w: [2, 1, 1, 1],
                                            eps: 0, vectors: 1)
        let want = [2.4, 1.6, 0, 0]
        check("rms FP64 oracle hand-computed",
              zip(ref, want).allSatisfy { abs($0 - $1) <= 1e-12 * max(1, abs($1)) })
    }
    checkRMSThrows("rms FP64 oracle rejects nonfinite input",
                   .oracleFailed("oracle geometry: x 4, vectors 1")) {
        _ = try RMSOracle.rmsnormFP64(x: [3, .nan, 0, 0], w: nil, eps: 0, vectors: 1)
    }

    // ---- persisted-execution log format regression ----
    // The real 32-step select run once crashed (SIGSEGV via
    // objc_opt_respondsToSelector) because a Swift Bool was passed to
    // String(format:) %@ as a C vararg. This calls the ACTUAL production
    // formatter helper used by selectAndPersist's final proof line (not a
    // duplicated copy), so any future Bool/non-object %@ regression in the
    // production path fails this test rather than crashing a 15-minute
    // model run.
    for scaled in [false, true] {
        for tensor in [nil, "blk.0.attn_norm.weight"] {
            let line = rmsPersistedExecutionLogLine(fixtureName: "raw-hello",
                                                    worstRel: 0.0, dispatches: 96,
                                                    n: 2816, scaled: scaled,
                                                    tensor: tensor)
            check("rms persisted-execution log renders scaled=\(scaled ? "true" : "false") tensor=\(tensor ?? "none")",
                  line.contains("scaled=\(scaled ? "true" : "false")")
                  && line.contains("n=2816")
                  && (tensor == nil ? !line.contains("weight tensor")
                                    : line.contains(", weight tensor \(tensor!)")))
        }
    }
} catch {
    failed += 1; print("FAIL: rms lowering IR/deriver/emitter/oracle — threw \(error)")
}

// ---- GPU component gates + artifact identity (synthetic op, no GGUF) ----
if let device = MTLCreateSystemDefaultDevice() {
    do {
        // Unbound synthetic op (unscaled): fast, no GGUF needed.
        let op = try RMSOpIR.parse(rmsOpJSON(n: 2048, vectors: 4, reps: 4))
        let hwProfile = try RMSCurrentHardware.profile()
        let incumbent = try RMSIncumbentKernel.compile(device: device)
        let wBytes = try RMSComponentInputs.weightBytes(op: op, file: nil)
        let x = RMSComponentInputs.activation(op: op)
        let production = incumbent.productionPipeline(n: op.n, scaled: op.scaled)
        let incOut = try incumbent.run(pipeline: production, n: op.n, vectors: 4,
                                       x: x, w: nil, eps: op.eps)
        let incInPlace = try incumbent.run(pipeline: production, n: op.n, vectors: 4,
                                           x: x, w: nil, eps: op.eps, inPlace: true)
        check("rms incumbent kernel in-place stable (bit patterns)",
              rmsBitEqual(incOut, incInPlace))

        // THE load-bearing test: every emitted schedule (baseline AND
        // cooperative, both staging widths) must be BIT-IDENTICAL to the
        // incumbent production kernel — out-of-place AND in-place.
        for schedule in RMSScheduleDeriver.derive(op: op, hardware: hwProfile)
        where schedule.threadsPerVector > 1 {
            let kernel = try RMSEmittedKernel.compile(op: op, schedule: schedule,
                                                      device: device)
            let out = try kernel.run(op: op, x: x, w: nil, vectors: 4, eps: op.eps)
            let inPlace = try kernel.run(op: op, x: x, w: nil, vectors: 4,
                                         eps: op.eps, inPlace: true)
            check("rms emitted \(schedule.kernelName) bit-identical to incumbent (out-of-place)",
                  rmsBitEqual(out, incOut))
            check("rms emitted \(schedule.kernelName) bit-identical to incumbent (in-place)",
                  rmsBitEqual(inPlace, incOut))
        }

        // Baseline vs the CPU FP64 oracle (existing 1e-4 contract).
        let ref64 = try RMSOracle.rmsnormFP64(x: x, w: nil, eps: op.eps, vectors: 4)
        var w1 = 0.0
        for i in 0..<incOut.count {
            w1 = max(w1, abs(Double(incOut[i]) - ref64[i]) / max(1.0, abs(ref64[i])))
        }
        check(String(format: "rms incumbent vs FP64 oracle (worst rel %.3e)", w1), w1 < 1e-4)

        // Full component forge on the synthetic op -> artifact identity.
        let (artifact, results, incRel) = try RMSLoweringForge.forge(
            opJSON: rmsOpJSON(n: 2048, vectors: 4, reps: 4), ggufURL: nil)
        check("rms forge incumbent within 1e-4 of FP64 oracle", incRel < 1e-4)
        check("rms forge produced accepted candidates",
              !results.isEmpty && results.contains { $0.accepted })
        check("rms forge selected an accepted bit-identical schedule",
              results.first { $0.schedule == artifact.schedule }!.accepted
                  && results.first { $0.schedule == artifact.schedule }!.bitIdenticalToIncumbent == true)
        try artifact.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                              executableDigest: try Gemma4Runner.executableDigest())
        check("rms artifact validates round-trip", true)
        // Policy contract on the real synthesized artifact: the explicit
        // unverified policy is encoded, required on decode, and the only
        // representable value; a schema-1 record is rejected even when it
        // carries the policy.
        do {
            let encoded = try JSONEncoder().encode(artifact)
            let decoded = try JSONDecoder().decode(RMSLoweringArtifact.self, from: encoded)
            check("rms artifact policy round-trips through encode/decode",
                  decoded == artifact && decoded.modelFileVerification == .cachedExplicitRefresh)
            var obj = (try JSONSerialization.jsonObject(with: encoded) as? [String: Any]) ?? [:]
            checkEqual("rms artifact encoding carries the cached explicit-refresh policy",
                       obj["modelFileVerification"] as? String ?? "", "cached-explicit-refresh-v1")
            var missing = obj
            missing.removeValue(forKey: "modelFileVerification")
            let missingDecoded = try? JSONDecoder().decode(
                RMSLoweringArtifact.self, from: JSONSerialization.data(withJSONObject: missing))
            check("rms artifact without the policy fails to decode", missingDecoded == nil)
            var claimed = obj
            claimed["modelFileVerification"] = "sha256-full-file"
            let claimedDecoded = try? JSONDecoder().decode(
                RMSLoweringArtifact.self, from: JSONSerialization.data(withJSONObject: claimed))
            check("rms artifact claiming verified model content fails to decode",
                  claimedDecoded == nil)
            obj["artifactSchema"] = 1
            let legacy = try JSONDecoder().decode(
                RMSLoweringArtifact.self, from: JSONSerialization.data(withJSONObject: obj))
            checkRMSThrows("rms schema-1 artifact rejected even with the policy",
                           .invalidArtifact("schema 1")) {
                try legacy.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                                    executableDigest: try Gemma4Runner.executableDigest())
            }
        }
        // A real-looking 64-hex model hash never satisfies the identity check.
        checkRMSThrows("rms artifact rejects a stale 64-hex model hash",
                       .invalidArtifact("GGUF digest mismatch")) {
            try artifact.validate(op: artifact.op, hardware: hwProfile,
                                  ggufDigest: String(repeating: "e", count: 64),
                                  executableDigest: try Gemma4Runner.executableDigest())
        }
        // Tampered source is rejected (JSON-level mutation).
        do {
            let encoded = try JSONEncoder().encode(artifact)
            var obj = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
            obj["emittedSource"] = (artifact.emittedSource as NSString)
                .replacingOccurrences(of: "T = ", with: "T =  /*x*/ ")
            let tampered = try JSONDecoder().decode(
                RMSLoweringArtifact.self, from: JSONSerialization.data(withJSONObject: obj))
            checkRMSThrows("rms artifact rejects tampered source",
                          .invalidArtifact("emitted source tampered/rule drift")) {
                try tampered.validate(op: artifact.op, hardware: hwProfile, ggufDigest: "",
                                      executableDigest: try Gemma4Runner.executableDigest())
            }
        }
        // Stale hardware is rejected.
        checkRMSThrows("rms artifact rejects hardware mismatch",
                      .invalidArtifact("hardware mismatch")) {
            try artifact.validate(op: artifact.op,
                                  hardware: RMSHardwareProfile(
                                    deviceName: "other", maxTotalThreadsPerThreadgroup: 7,
                                    threadExecutionWidth: 7, maxThreadgroupMemoryLength: 7,
                                    recommendedMaxWorkingSetSize: 7),
                                  ggufDigest: "",
                                  executableDigest: try Gemma4Runner.executableDigest())
        }
    } catch {
        failed += 1; print("FAIL: rms lowering GPU/artifact gates — threw \(error)")
    }
} else {
    print("SKIP: rms lowering GPU gates (no Metal device)")
}

// MARK: - Gemma 4 real-model gates (opt-in via LEVIATHAN_GEMMA_GGUF)

// These load the full 14.44 GB checkpoint onto the GPU (~seconds, ~2 GB
// resident). Opt-in keeps the default suite small and machine-independent.
let gemmaPath = ProcessInfo.processInfo.environment["LEVIATHAN_GEMMA_GGUF"]
if let gemmaPath, !gemmaPath.isEmpty, FileManager.default.fileExists(atPath: gemmaPath) {
    do {
        let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: gemmaPath))
        // GPU embedding-row decoder vs the independent scalar Q6_K oracle.
        let worst = try runner.verifyEmbeddingRow(token: 2)
        check(String(format: "real GGUF embedding parity (worst rel %.2e)", worst),
              worst < 1e-4)
        // Greedy agreement with the same-GGUF Ollama 0.23.0 baseline
        // (raw "Hello", temp 0, num_predict 2): the baseline generated the
        // pieces " upon" then "-". The prompt IDs [BOS=2, 9259] are INFERRED
        // from vocab lookups + prompt_eval_count=2 (Ollama exposes no token
        // IDs); this is greedy token agreement, not full logit parity.
        let e1 = try runner.forward(tokenIds: [2, 9259])
        checkEqual("real GGUF greedy token 1 vs Ollama baseline", e1.argmaxToken, 3324)
        let e2 = try runner.forward(tokenIds: [3324])
        checkEqual("real GGUF greedy token 2 vs Ollama baseline", e2.argmaxToken, 236772)

        // Teacher-forced layerwise diagnostic surface (opt-in tooling):
        // the probe fires at the three checkpoints per layer and does not
        // change the greedy result; applyDiagnosticAxis toggles exactly
        // the executor's candidate flags and restores them; the bounded
        // diagnosticMatvec is deterministic and, with tiled=false, runs
        // the ORIGINAL kernel (bit-stable across two calls).
        runner.reset()
        var probeCalls = 0
        var stagesSeen = Set<String>()
        runner.layerProbe = { stage, layer, position, x in
            probeCalls += 1
            stagesSeen.insert(stage)
            check("probe checkpoint shape (stage \(stage) layer \(layer))",
                  layer >= 0 && layer < 30 && position >= 0 && x.count == 2816)
        }
        let e3 = try runner.forward(tokenIds: [2])
        runner.layerProbe = nil
        checkEqual("probe run advances context by one", e3.contextLength, 1)
        checkEqual("probe fired 3x per layer", probeCalls, 90)
        checkEqual("probe stages", stagesSeen.sorted(),
                   ["attn_norm_h", "post_attn", "post_layer"])
        runner.applyDiagnosticAxis(q4Tiled: true, segmentBatching: true)
        check("applyDiagnosticAxis sets flags",
              runner.q4TiledKernel && runner.segmentBatchingActive && !runner.rmsTiledKernel)
        runner.applyDiagnosticAxis()
        check("applyDiagnosticAxis() restores reference flags",
              !runner.q4TiledKernel && !runner.q6TiledKernel && !runner.rmsTiledKernel
                  && !runner.f32TiledKernel && !runner.vectorizedLoads && !runner.segmentBatchingActive)
        // Deferred-segment batching regression (pass-9 fix): reset() ->
        // discardOpenSegment() clears autoSegment, so the axis must be
        // RE-APPLIED after every reset. Measured regression: one batched
        // forward must submit FEWER command buffers than dispatched
        // kernels (per-kernel mode submits one buffer per kernel).
        do {
            runner.reset()
            runner.applyDiagnosticAxis(segmentBatching: true)
            guard runner.segmentBatchingActive else {
                throw Gemma4Error.badTensorGeometry("axis failed to engage")
            }
            let bufsBefore = runner.submittedBufferTotal
            let dispBefore = runner.dispatchTotal
            _ = try runner.forward(tokenIds: [2])
            let bufsDelta = runner.submittedBufferTotal - bufsBefore
            let dispDelta = runner.dispatchTotal - dispBefore
            check("batched forward engages after reset (buffers < dispatches)",
                  bufsDelta < dispDelta && dispDelta > 1)
            runner.applyDiagnosticAxis()
        }
        let normedH = [Float](repeating: 0.01, count: 2816)
        let d1 = try runner.diagnosticMatvec(tensor: "blk.0.attn_q.weight",
                                              x: normedH, rows: 64, tiled: false)
        let d2 = try runner.diagnosticMatvec(tensor: "blk.0.attn_q.weight",
                                              x: normedH, rows: 64, tiled: false)
        check("diagnosticMatvec plain kernel bit-stable",
              zip(d1, d2).allSatisfy { $0.bitPattern == $1.bitPattern })
        check("diagnosticMatvec rejects bad width",
              (try? runner.diagnosticMatvec(tensor: "blk.0.attn_q.weight",
                                            x: [Float](repeating: 0, count: 16),
                                            rows: 64, tiled: false)) == nil)

        // Phase 10 real-op check: the n=2816 scaled RMSNorm family bound to
        // the REAL weight tensor — component forge (bit-identity vs the
        // production kernel on real weights) plus a trial-install forward
        // proving the generated kernel actually dispatches.
        do {
            let realJSON = try rmsOpJSON(n: 2816, vectors: 8, scaled: true,
                                         eps: 1e-6, tensor: "blk.0.attn_norm.weight",
                                         reps: 8)
            let realOp = try RMSOpIR.parse(realJSON)
            try realOp.validateAgainst(file: runner.ggufFile)
            let (art, res, incRel) = try RMSLoweringForge.forge(
                opJSON: realJSON, ggufURL: URL(fileURLWithPath: gemmaPath))
            check("real rms forge incumbent within 1e-4 of FP64 oracle", incRel < 1e-4)
            check("real rms forge has bit-identical accepted candidates",
                  res.contains { $0.accepted && $0.bitIdenticalToIncumbent == true })
            guard let best = res.first(where: { $0.accepted }) else {
                throw RMSLoweringError.componentGateFailed("no accepted real-op schedule")
            }
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw RMSLoweringError.hardwareUnavailable("no Metal device")
            }
            let kernel = try RMSEmittedKernel.compile(op: realOp,
                                                      schedule: best.schedule,
                                                      device: device)
            try runner.diagnosticRmsTrial(op: realOp, kernel: kernel)
            runner.reset()
            runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
            let before = runner.rmsGeneratedDispatchCount
            _ = try runner.forward(tokenIds: [2])
            check("real rms trial kernel dispatches on a forward",
                  runner.rmsGeneratedDispatchCount > before)
            runner.uninstallRmsGeneratedKernel()
            _ = art

            // One-ULP-off eps is rejected: a neighboring Float eps value
            // must NOT install (exact runtime semantics, not a tolerance).
            let configEps = runner.config.rmsNormEps
            let wrongEps = Double(configEps.nextUp)
            let wrongJSON = try rmsOpJSON(n: 2816, vectors: 8, scaled: true,
                                          eps: wrongEps,
                                          tensor: "blk.0.attn_norm.weight", reps: 8)
            let wrongOp = try RMSOpIR.parse(wrongJSON)
            checkRMSThrows("real rms trial rejects one-ULP-off eps",
                          .invalidArtifact("op eps \(wrongEps) != config rmsNormEps "
                                           + "\(configEps)")) {
                try runner.diagnosticRmsTrial(op: wrongOp, kernel: kernel)
            }
            // Behavioral proof nothing installed: a forward must not route
            // any dispatch through a generated RMS kernel.
            runner.reset()
            runner.applyDiagnosticAxis(vectorizedLoads: true, segmentBatching: true)
            let beforeWrong = runner.rmsGeneratedDispatchCount
            _ = try runner.forward(tokenIds: [2])
            checkEqual("rejected eps installs nothing (no generated dispatches)",
                       runner.rmsGeneratedDispatchCount, beforeWrong)
        } catch {
            failed += 1; print("FAIL: real rms op forge/trial — threw \(error)")
        }
    } catch {
        failed += 1; print("FAIL: real-model Gemma gates — threw \(error)")
    }
} else {
    print("SKIP: real-model Gemma gates (set LEVIATHAN_GEMMA_GGUF to opt in)")
}

// MARK: - MoE routed-region lowering (phase 13)

do {
    // Fixtures captured by the REAL torch.export(strict=True) frontend
    // (tools/moe_lower/export_moe_region.py): the actual Gemma-4 26B
    // region and a tail region (topK 3) for GPU chunk-tail gates.
    let fixtureDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("tools/moe_lower/fixtures")
    let realRegionJSON = try Data(contentsOf:
        fixtureDir.appendingPathComponent("gemma4-26b-region.json"))
    let realRegion = try MoERegionIR.parse(realRegionJSON)
    checkEqual("moe region: real capture dims", realRegion.hidden, 2816)
    checkEqual("moe region: inter", realRegion.expertIntermediate, 704)
    checkEqual("moe region: topK", realRegion.topK, 8)
    checkEqual("moe region: gateUpOut", realRegion.gateUpOut, 1408)
    check("moe region: digest validates", moeIsLowercaseHex64(realRegion.graphSha256))
    let tailRegion = try MoERegionIR.parse(try Data(contentsOf:
        fixtureDir.appendingPathComponent("tail-region.json")))
    checkEqual("moe region: tail topK", tailRegion.topK, 3)

    // Tamper helpers: mutate the JSON, RECOMPUTE the canonical graph
    // digest with the Swift importer itself, and re-parse — only the
    // independent structural re-derivation can still catch these.
    func tampered(_ mutate: (inout [String: Any]) -> Void) throws -> Data {
        var obj = try JSONSerialization.jsonObject(with: realRegionJSON) as! [String: Any]
        mutate(&obj)
        let graphData = try JSONSerialization.data(withJSONObject: obj["graph"]!)
        let graph = try JSONDecoder().decode(MoENormalizedGraph.self, from: graphData)
        obj["graphSha256"] = graph.digest()
        return try JSONSerialization.data(withJSONObject: obj)
    }
    func checkRegionReject(_ name: String, _ needle: String, _ data: Data) {
        do {
            _ = try MoERegionIR.parse(data)
            failed += 1; print("FAIL: moe tamper rejected: \(name) (accepted)")
        } catch {
            let msg = "\(error)"
            if msg.contains(needle) {
                passed += 1; print("PASS: moe tamper rejected: \(name)")
            } else {
                failed += 1; print("FAIL: moe tamper rejected: \(name) — wrong error: \(msg)")
            }
        }
    }
    do { let d = try tampered { obj in obj["topK"] = 7 }
         checkRegionReject("declared topK lies", "disagree with the graph-derived", d) }
    do { let d = try tampered { obj in obj["hidden"] = 2815 }
         checkRegionReject("declared hidden lies", "disagree with the graph-derived", d) }
    do { // swap gate/up getitem indices on every slot (graph + digest updated)
        let d = try tampered { obj in
            var g = obj["graph"] as! [String: Any]
            var nodes = g["nodes"] as! [[String: Any]]
            for i in nodes.indices {
                if nodes[i]["target"] as? String == "operator.getitem" {
                    let args = nodes[i]["args"] as! [Any]
                    if let idx = args[1] as? Int {
                        var n = nodes[i]
                        n["args"] = [args[0], idx == 0 ? 1 : 0]
                        nodes[i] = n
                    }
                }
            }
            g["nodes"] = nodes; obj["graph"] = g
        }
        checkRegionReject("gate/up getitems swapped", "getitem", d) }
    do { let d = try tampered { obj in
        var g = obj["graph"] as! [String: Any]
        var nodes = g["nodes"] as! [[String: Any]]
        for i in nodes.indices where nodes[i]["target"] as? String == "aten.gelu.default" {
            var n = nodes[i]
            n["kwargs"] = ["approximate": "none"]
            nodes[i] = n
        }
        g["nodes"] = nodes; obj["graph"] = g
    }
    checkRegionReject("gelu approximate=none", "tanh", d) }
    do { let d = try tampered { obj in
        var g = obj["graph"] as! [String: Any]
        var nodes = g["nodes"] as! [[String: Any]]
        nodes.append(["name": "extra_sin", "op": "call_function",
                      "target": "aten.sin.default", "args": [[ "__node__": nodes[3]["name"] as! String]],
                      "kwargs": [String: Any](), "meta": NSNull()])
        g["nodes"] = nodes; obj["graph"] = g
    }
    checkRegionReject("extra aten.sin node", "unconsumed/extra graph node", d) }
    do { let d = try tampered { obj in obj["region"] = "some.other.region" }
         checkRegionReject("wrong region name", "unsupported region", d) }

    // Emitter geometry: every (E, R) changes the emitted grid geometry.
    let src1 = try MoEKernelEmitter.emit(region: tailRegion,
                                         schedule: MoESchedule(batch: 2, rowsPerGroup: 32))
    let src2 = try MoEKernelEmitter.emit(region: tailRegion,
                                         schedule: MoESchedule(batch: 8, rowsPerGroup: 256))
    check("moe emitter: schedules emit distinct geometry", src1 != src2)
    check("moe emitter: kernel names parameterized",
          src1.contains("moegen_gu_E2_R32") && src2.contains("moegen_dn_E8_R256"))
    check("moe emitter: embeds the literal incumbent helpers",
          src1.contains("static float g4_tanh")
              && src1.contains("static float g4_gelu_up_value"))
    let literalHelpers = try MoEKernelEmitter.literalActivationHelpers()
    check("moe emitter: literal helpers are the incumbent source text",
          src1.contains(literalHelpers))
    do { _ = try MoEKernelEmitter.emit(region: tailRegion,
                                       schedule: MoESchedule(batch: 3, rowsPerGroup: 32))
         failed += 1; print("FAIL: moe emitter rejects E=3") } catch {
        check("moe emitter: non-power-of-two E rejected", true) }

    // Mapping-length helper (shared runner/forge formula).
    checkEqual("moe maplen: aligned size unchanged",
               MoEModelBinding.residentMappingLength(fileBytes: 4096, pageSize: 4096), 4096)
    checkEqual("moe maplen: rounds up",
               MoEModelBinding.residentMappingLength(fileBytes: 4097, pageSize: 4096), 8192)
    checkEqual("moe maplen: 16K pages",
               MoEModelBinding.residentMappingLength(fileBytes: 16384, pageSize: 16384), 16384)
    check("moe maplen: overflow rejected",
          MoEModelBinding.residentMappingLength(fileBytes: Int.max, pageSize: 4096) == nil)
    check("moe maplen: invalid rejected",
          MoEModelBinding.residentMappingLength(fileBytes: 0, pageSize: 4096) == nil)

    // GPU scratch budget: the REAL region's payload is exactly the three
    // device buffers (8 x (1408+704+2816) x 4 = 157696 B); the host-side
    // setBytes id snapshots are NOT GPU scratch. Overflowing dims decode
    // (validation deliberately bypassed to reach the arithmetic) must
    // THROW, never wrap into a false fits.
    let realScratch = try MoEScratch.budgetBytes(region: realRegion)
    checkEqual("moe scratch: real region GPU bytes", realScratch, 157_696)
    do {
        var obj = try JSONSerialization.jsonObject(with: realRegionJSON) as! [String: Any]
        obj["hidden"] = 1 << 60
        obj["gateUpOut"] = 1 << 60
        obj["topK"] = 1 << 60
        let huge = try JSONDecoder().decode(
            MoERegionIR.self, from: JSONSerialization.data(withJSONObject: obj))
        do { _ = try MoEScratch.budgetBytes(region: huge)
             failed += 1; print("FAIL: moe scratch overflow rejected") } catch {
            check("moe scratch: overflowing dims throw (no wrap)", true) }
        for (name, mutate) in [
            ("zero topK", { (o: inout [String: Any]) in o["topK"] = 0 }),
            ("zero hidden", { (o: inout [String: Any]) in o["hidden"] = 0 }),
            ("negative inter", { (o: inout [String: Any]) in
                o["expertIntermediate"] = -4 }),
        ] as [(String, (inout [String: Any]) -> Void)] {
            var z = try JSONSerialization.jsonObject(with: realRegionJSON) as! [String: Any]
            mutate(&z)
            let dec = try JSONDecoder().decode(
                MoERegionIR.self, from: JSONSerialization.data(withJSONObject: z))
            do { _ = try MoEScratch.budgetBytes(region: dec)
                 failed += 1; print("FAIL: moe scratch rejects \(name)") } catch {
                check("moe scratch: non-positive dims rejected (\(name))", true) }
        }
    }

    // Schedule derivation fails CLOSED on invalid hardware caps (no
    // silent width-1 fallback) and never overflows the w*2/4/8 ladder.
    let realCaps = ForgeHardwareProfile(deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                                        threadExecutionWidth: 32,
                                        recommendedMaxWorkingSetSize: 1 << 30)
    let derived = try MoEScheduleDeriver.derive(region: realRegion, hardware: realCaps)
    checkEqual("moe derive: real caps yield candidates", derived.count, 4 * 5)
    for (name, hw) in [
        ("zero width", ForgeHardwareProfile(deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                                            threadExecutionWidth: 0,
                                            recommendedMaxWorkingSetSize: 1 << 30)),
        ("negative max threads", ForgeHardwareProfile(deviceName: "t",
                                                      maxTotalThreadsPerThreadgroup: -8,
                                                      threadExecutionWidth: 32,
                                                      recommendedMaxWorkingSetSize: 1 << 30)),
        ("width above threadgroup cap", ForgeHardwareProfile(
            deviceName: "t", maxTotalThreadsPerThreadgroup: 32,
            threadExecutionWidth: 64, recommendedMaxWorkingSetSize: 1 << 30)),
        ("working set below scratch", ForgeHardwareProfile(
            deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
            threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1024)),
    ] as [(String, ForgeHardwareProfile)] {
        do { _ = try MoEScheduleDeriver.derive(region: realRegion, hardware: hw)
             failed += 1; print("FAIL: moe derive rejects \(name)") } catch {
            check("moe derive: invalid caps rejected (\(name))", true) }
    }
    do { // width near Int.max: w*8 overflows — products are skipped, no crash
        let huge = ForgeHardwareProfile(deviceName: "t", maxTotalThreadsPerThreadgroup: .max,
                                        threadExecutionWidth: Int.max / 2,
                                        recommendedMaxWorkingSetSize: 1 << 30)
        let tiles = try MoEScheduleDeriver.derive(region: realRegion, hardware: huge)
        check("moe derive: overflowing width ladder skips without crash",
              !tiles.isEmpty && tiles.allSatisfy { $0.rowsPerGroup <= 1024 })
    }

    // Gate-policy SHAPE negatives: a full production-shaped record built
    // via JSON (decode only — these test the policy, not capture), then
    // rejections for diagnostic smoke, short refs, and a nonzero timing
    // start position.
    func gateRecordFixture(diagnostic: Bool, genTokens: Int, gatedSteps: Int,
                           startIdx: Int, overflowRep: Int? = nil) throws
                           -> MoEModelGateRecord {
        func arms(_ wall: Double) -> [[String: Any]] {
            MoEGatePolicy.productionFixtures.map { _ in
                ["walls": [Double](repeating: wall, count: 31),
                 "gpus": [Double](repeating: 10, count: 31),
                 "startIdx": startIdx, "timedCount": 31] as [String: Any]
            }
        }
        let reps = (0..<3).map { i in
            ["candidateFirst": i % 2 == 1,
             "candidate": arms(overflowRep == i ? 1e308 : 1.0),
             "incumbent": arms(2.0)] as [String: Any]
        }
        let obj: [String: Any] = [
            "gatePolicy": MoEGatePolicy.version,
            "executionPath": MoEGatePolicy.executionPathSSA,
            "diagnostic": diagnostic,
            "genTokens": genTokens,
            "timedTokens": MoEGatePolicy.productionTimedTokens,
            "fixtures": MoEGatePolicy.productionFixtures.map {
                ["name": $0, "promptTokenCount": 8, "gatedSteps": gatedSteps,
                 "historyDigest": String(repeating: "a", count: 64)] as [String: Any]
            },
            "results": [[
                "schedule": ["batch": 8, "rowsPerGroup": 128],
                "fixturesPassed": 3, "fixturesFailed": 0, "worstRel": 1e-7,
                "gateFailReason": NSNull(),
                "wallMedPerFixture": [1.0, 1.0, 1.0],
                "gpuMedPerFixture": [10.0, 10.0, 10.0],
                "timingReps": reps] as [String: Any]],
            "incumbentWallPerFixture": [2.0, 2.0, 2.0],
            "incumbentGpuPerFixture": [20.0, 20.0, 20.0],
            "performanceSelected": true,
            "policyNote": "test",
        ]
        return try JSONDecoder().decode(MoEModelGateRecord.self,
                                        from: JSONSerialization.data(withJSONObject: obj))
    }
    let sched8 = MoESchedule(batch: 8, rowsPerGroup: 128)
    let good = try gateRecordFixture(diagnostic: false, genTokens: 32,
                                     gatedSteps: 32, startIdx: 0)
    check("moe policy: production shape accepted",
          MoEGatePolicy.shapeValid(good, schedule: sched8))
    check("moe policy: production winner recomputes",
          MoEGatePolicy.recomputedSelection(good, schedule: sched8) == true)
    check("moe policy: productionValid for honest winner",
          MoEGatePolicy.productionValid(good, schedule: sched8))
    let diagRec = try gateRecordFixture(diagnostic: true, genTokens: 32,
                                        gatedSteps: 32, startIdx: 0)
    check("moe policy: diagnostic smoke record rejected",
          !MoEGatePolicy.shapeValid(diagRec, schedule: sched8))
    let shortRec = try gateRecordFixture(diagnostic: false, genTokens: 2,
                                         gatedSteps: 2, startIdx: 0)
    check("moe policy: short (2-step) reference rejected",
          !MoEGatePolicy.shapeValid(shortRec, schedule: sched8))
    let truncRec = try gateRecordFixture(diagnostic: false, genTokens: 32,
                                         gatedSteps: 17, startIdx: 0)
    check("moe policy: EOS-truncated gated steps rejected",
          !MoEGatePolicy.shapeValid(truncRec, schedule: sched8))
    let shiftedRec = try gateRecordFixture(diagnostic: false, genTokens: 32,
                                           gatedSteps: 32, startIdx: 5)
    check("moe policy: nonzero timing start rejected",
          !MoEGatePolicy.shapeValid(shiftedRec, schedule: sched8))
    check("moe policy: wrong schedule rejected",
          !MoEGatePolicy.shapeValid(good, schedule: MoESchedule(batch: 4, rowsPerGroup: 64)))
    // One repetition whose individually finite walls overflow when summed
    // (3 x 1e308 = inf) must invalidate the recompute — the median of
    // [inf, 3, 3] would otherwise hide it.
    let poisoned = try gateRecordFixture(diagnostic: false, genTokens: 32,
                                         gatedSteps: 32, startIdx: 0, overflowRep: 1)
    check("moe policy: single overflowing repetition invalidates recompute",
          MoEGatePolicy.recomputedSelection(poisoned, schedule: sched8) == nil
              && !MoEGatePolicy.productionValid(poisoned, schedule: sched8))

    // Phase-14 diagnostic profile helpers: family classification,
    // decode-only snapshot diffs, bounded decode-view checks, and
    // serialization roundtrip. Meaningful bounds only — not mirrors of
    // the run orchestration.
    checkEqual("moe profile: moegen family classified",
               MoELoweringProfile.family(ofKernel: "moegen_gu_E4_R32"),
               "moe-generated")
    checkEqual("moe profile: engine kernel family",
               MoELoweringProfile.family(ofKernel: "g4_axpy"), "engine")
    do {
        let before: [String: (gpu: Double, calls: Int)] = [
            "g4_axpy": (100, 4), "g4_rmsnorm": (50, 2), "moegen_act_E4": (10, 1)]
        let after: [String: (gpu: Double, calls: Int)] = [
            "g4_axpy": (250, 10), "g4_rmsnorm": (50, 2),
            "moegen_act_E4": (70, 8), "g4_q4_matvec": (40, 3)]
        let diff = MoELoweringProfile.kernelDiff(before: before, after: after)
        checkEqual("moe profile: kernelDiff counts changed kernels", diff.count, 3)
        check("moe profile: kernelDiff drops unchanged and sorts by GPU",
              !diff.contains { $0.name == "g4_rmsnorm" }
                  && diff[0].name == "g4_axpy"
                  && diff.map(\.name) == ["g4_axpy", "moegen_act_E4", "g4_q4_matvec"])
        check("moe profile: kernelDiff deltas and families",
              diff[0].gpuMicros == 150 && diff[0].calls == 6
                  && diff[1].family == "moe-generated"
                  && diff[2].family == "engine")
    }
    do {
        // A bounded decode view built via JSON (decode-only test of the
        // bounds gate and the Codable roundtrip).
        func decodeView(walls: [Double], fallback: Int) throws
                -> MoEProfileReport.DecodeView {
            let obj: [String: Any] = [
                "decodeTokens": walls.count, "excludedPrefillForwards": 1,
                "sumWallMicros": walls.reduce(0, +),
                "medianWallMicros": LoweringSelect.median(walls),
                "aggregateDecodeTps": Double(walls.count),
                "sumGpuMicros": 10.0, "wallsMicros": walls,
                "gpusMicros": walls.map { _ in 1.0 },
                "encodeMicros": 1.0, "flushMicros": 2.0, "flushHostMicros": 1.0,
                "readCopyMicros": 0.5, "writeCopyMicros": 0.25,
                "residualMicros": 1.0, "submittedBufferDelta": 3,
                "dispatchDelta": 9,
                "stageDeltas": ["gu": 2, "act": 2, "dn": 2],
                "mergeDelta": 8, "fallbackDelta": fallback]
            let data = try JSONSerialization.data(withJSONObject: obj)
            let v = try JSONDecoder().decode(MoEProfileReport.DecodeView.self,
                                             from: data)
            let reencoded = try JSONEncoder().encode(v)
            let rt = try JSONDecoder().decode(
                MoEProfileReport.DecodeView.self, from: reencoded)
            guard rt == v else { throw NSError(
                domain: "rt", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "roundtrip differs"]) }
            return v
        }
        let good = try decodeView(walls: [1.0, 2.0, 3.0], fallback: 0)
        let negWall = try decodeView(walls: [1.0, -0.5, 3.0], fallback: 0)
        let withFallback = try decodeView(walls: [1.0, 2.0, 3.0], fallback: 1)
        check("moe profile: bounded decode view accepted",
              MoELoweringProfile.decodeViewIsBounded(good, expectedTokens: 3))
        check("moe profile: wrong window length rejected",
              !MoELoweringProfile.decodeViewIsBounded(good, expectedTokens: 2))
        check("moe profile: nonpositive wall rejected",
              !MoELoweringProfile.decodeViewIsBounded(negWall, expectedTokens: 3))
        check("moe profile: nonzero fallback rejected",
              !MoELoweringProfile.decodeViewIsBounded(withFallback, expectedTokens: 3))

        // Profile report contract: the explicit policy is required on
        // decode and no verified value is representable. Decode only: no
        // profile run, no model read, no performance claim.
        let profileView = try JSONSerialization.jsonObject(with: JSONEncoder().encode(good))
        let profileHex = String(repeating: "a", count: 64)
        let profileObj: [String: Any] = [
            "profileSchema": MoEProfileReport.schema, "codegenVersion": "test",
            "executableSha256": profileHex,
            "modelFileSha256": profileHex,
            "modelFileVerification": ModelFileIdentity.policy.rawValue,
            "deviceName": "t", "maxTotalThreadsPerThreadgroup": 1024,
            "threadExecutionWidth": 32, "recommendedWorkingSetSize": 1 << 30,
            "regionGraphSha256": profileHex,
            "schedule": ["batch": 4, "rowsPerGroup": 32],
            "sourceSha256": profileHex, "expectedSourceSha256": NSNull(),
            "sourceMatchesExpected": NSNull(), "diagnosticTrialOnly": true,
            "binding": ["layerCount": 30, "expertCount": 128, "topK": 8,
                        "guStrideBytes": 2_230_272, "dnStrideBytes": 1_115_136,
                        "gpuScratchBytes": 157_696],
            "reference": ["fixtureName": "raw-hello", "promptTokenCount": 2,
                          "steps": 32, "historyDigest": profileHex,
                          "vocabSize": 262144],
            "decode": profileView, "intrusive": NSNull(), "overhead": NSNull()]
        func decodeProfile(_ o: [String: Any]) -> MoEProfileReport? {
            guard let data = try? JSONSerialization.data(withJSONObject: o) else { return nil }
            return try? JSONDecoder().decode(MoEProfileReport.self, from: data)
        }
        let profile = decodeProfile(profileObj)
        check("mfv moe profile decodes with the cached model digest and policy",
              profile?.modelFileVerification == ModelFileIdentity.policy
                  && profile?.modelFileSha256 == profileHex)
        if let profile {
            let back = try JSONDecoder().decode(MoEProfileReport.self,
                                                from: JSONEncoder().encode(profile))
            check("mfv moe profile round-trips with the policy", back == profile)
        }
        var profileMissing = profileObj
        profileMissing.removeValue(forKey: "modelFileVerification")
        check("mfv moe profile without the policy fails to decode",
              decodeProfile(profileMissing) == nil)
        var profileClaimed = profileObj
        profileClaimed["modelFileVerification"] = "sha256-full-file"
        check("mfv moe profile claiming verified model content fails to decode",
              decodeProfile(profileClaimed) == nil)
    }

    // Phase-15 Q4 launch geometry: bounded semantic tests (no runner
    // needed) plus a GPU tail-shape bit-identity gate across widths.
    do {
        let hw = ForgeHardwareProfile(deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                                      threadExecutionWidth: 32,
                                      recommendedMaxWorkingSetSize: 1 << 30)
        let ladder = try Q4LaunchDeriver.derive(hardware: hw, pipelineMaxThreads: 1024)
        checkEqual("q4 launch: ladder widths", ladder.map(\.width), [32, 64, 128, 256, 1024])
        check("q4 launch: control is the pipeline max",
              Q4LaunchDeriver.control(pipelineMaxThreads: 1024).width == 1024)
        for bad in [0, -32, 3, 2048] {
            check("q4 launch: width \(bad) illegal",
                  !Q4LaunchSchedule(width: bad).isLegal(pipelineMaxThreads: 1024))
        }
        for (name, hwp, pmax) in [
            ("zero width", ForgeHardwareProfile(deviceName: "t",
                                                maxTotalThreadsPerThreadgroup: 1024,
                                                threadExecutionWidth: 0,
                                                recommendedMaxWorkingSetSize: 1 << 30), 1024),
            ("bad pipeline max", hw, 0),
            ("width above pipeline max", ForgeHardwareProfile(
                deviceName: "t", maxTotalThreadsPerThreadgroup: 64,
                threadExecutionWidth: 128, recommendedMaxWorkingSetSize: 1 << 30), 64),
        ] as [(String, ForgeHardwareProfile, Int)] {
            do { _ = try Q4LaunchDeriver.derive(hardware: hwp, pipelineMaxThreads: pmax)
                 failed += 1; print("FAIL: q4 launch derive rejects \(name)") } catch {
                check("q4 launch: invalid caps rejected (\(name))", true) }
        }
        checkEqual("q4 launch: groups exact ceil", Q4LaunchForge.groups(grid: 33, width: 32), 2)
        checkEqual("q4 launch: groups divides evenly", Q4LaunchForge.groups(grid: 64, width: 32), 2)
        checkEqual("q4 launch: groups width one", Q4LaunchForge.groups(grid: 7, width: 1), 7)
        check("q4 launch: groups zero grid rejected", Q4LaunchForge.groups(grid: 0, width: 32) == nil)
        check("q4 launch: groups negative grid rejected",
              Q4LaunchForge.groups(grid: -4, width: 32) == nil)
        check("q4 launch: groups overflowing grid rejected",
              Q4LaunchForge.groups(grid: Int.max, width: 32) == nil)
        check("q4 launch: groups zero width rejected",
              Q4LaunchForge.groups(grid: 33, width: 0) == nil)
        check("q4 launch: huge grid no overflow",
              Q4LaunchForge.groups(grid: Int(UInt32.max), width: 1024) == 4_194_304)

        // Gate-policy shape negatives (mirror of the phase-13 contract,
        // keyed on width). Counters are checked against the binding-
        // derived arithmetic: prompt 8 + 32 - 1 = 39 forwards x 3
        // fixtures = 117; chunks ceil(8/4)=2; layers 30 → stages 7020,
        // merges 28080.
        // A structurally VALID synthetic binding (the decoder enforces
        // every invariant): real geometry, 128 experts, 30 layers with
        // distinct non-overlapping tensor spans.
        let guBytes = 2_230_272 * 128, dnBytes = 1_115_136 * 128
        func syntheticBindingJSON(layerCount: Int) -> [String: Any] {
            let guStarts = (0..<layerCount).map { $0 * guBytes }
            let dnStarts = (0..<layerCount).map { layerCount * guBytes + $0 * dnBytes }
            // Metadata-only caps sized to the computed resident extent
            // (no allocation happens — these are decoded integers).
            let extent = layerCount * (guBytes + dnBytes)
            return [
                "graphSha256": String(repeating: "b", count: 64),
                "fileByteCount": extent,
                "residentBufferLength": extent,
                "maxBufferLength": extent * 2, "recommendedWorkingSetSize": extent * 4,
                "layerCount": layerCount, "expertCount": 128, "topK": 8,
                "hidden": 2816, "inter": 704, "gateUpOut": 1408,
                "guStride": 2_230_272, "dnStride": 1_115_136,
                "guTensorStarts": guStarts, "dnTensorStarts": dnStarts,
                "guScratchBytes": 45_056, "actScratchBytes": 22_528,
                "dnScratchBytes": 90_112, "scratchTotalBytes": 157_696]
        }
        let launchBindingJSON = syntheticBindingJSON(layerCount: 30)
        let launchBinding = try JSONDecoder().decode(
            MoEModelBinding.self,
            from: try JSONSerialization.data(withJSONObject: launchBindingJSON))
        func launchGate(diagnostic: Bool, gatedSteps: Int, startIdx: Int,
                        eligible: Any = 1000, applied: Any = 1000,
                        stages: Any = 7020, merges: Any = 28080) throws
                -> Q4LaunchGateRecord {
            func arms(_ wall: Double) -> [[String: Any]] {
                Q4LaunchGatePolicy.productionFixtures.map { _ in
                    ["walls": [Double](repeating: wall, count: 31),
                     "gpus": [Double](repeating: 10, count: 31),
                     "startIdx": startIdx, "timedCount": 31] as [String: Any]
                }
            }
            let reps = (0..<3).map { i in
                ["candidateFirst": i % 2 == 1,
                 "candidate": arms(1.0), "incumbent": arms(2.0)] as [String: Any]
            }
            let obj: [String: Any] = [
                "gatePolicy": Q4LaunchGateRecord.policyVersion,
                "executionPath": Q4LaunchGateRecord.executionPathSSA,
                "diagnostic": diagnostic, "genTokens": 32, "timedTokens": 31,
                "fixtures": Q4LaunchGatePolicy.productionFixtures.map {
                    ["name": $0, "promptTokenCount": 8, "gatedSteps": gatedSteps,
                     "historyDigest": String(repeating: "a", count: 64)] as [String: Any]
                },
                "results": [[
                    "schedule": ["width": 32],
                    "fixturesPassed": 3, "fixturesFailed": 0, "worstRel": 0.0,
                    "gateFailReason": NSNull(),
                    "eligibleDispatches": eligible,
                    "appliedDispatches": applied,
                    "moeStageDispatches": stages,
                    "mergeDispatches": merges,
                    "timingReps": reps] as [String: Any]],
                "baselineWallPerFixture": [2.0, 2.0, 2.0],
                "performanceSelected": true, "policyNote": "test"]
            return try JSONDecoder().decode(
                Q4LaunchGateRecord.self,
                from: try JSONSerialization.data(withJSONObject: obj))
        }
        let w32 = Q4LaunchSchedule(width: 32)
        let goodGate = try launchGate(diagnostic: false, gatedSteps: 32, startIdx: 0)
        check("q4 launch policy: production shape accepted",
              Q4LaunchGatePolicy.shapeValid(goodGate, schedule: w32,
                                            binding: launchBinding))
        check("q4 launch policy: winner recomputes",
              Q4LaunchGatePolicy.recomputedSelection(goodGate, schedule: w32,
                                                     binding: launchBinding) == true)
        let diagGate = try launchGate(diagnostic: true, gatedSteps: 32, startIdx: 0)
        check("q4 launch policy: diagnostic rejected",
              !Q4LaunchGatePolicy.shapeValid(diagGate, schedule: w32,
                                             binding: launchBinding))
        let shortGate = try launchGate(diagnostic: false, gatedSteps: 2, startIdx: 0)
        check("q4 launch policy: short reference rejected",
              !Q4LaunchGatePolicy.shapeValid(shortGate, schedule: w32,
                                             binding: launchBinding))
        let shiftedGate = try launchGate(diagnostic: false, gatedSteps: 32, startIdx: 3)
        check("q4 launch policy: nonzero start rejected",
              !Q4LaunchGatePolicy.shapeValid(shiftedGate, schedule: w32,
                                             binding: launchBinding))
        let zeroEligible = try launchGate(diagnostic: false, gatedSteps: 32,
                                          startIdx: 0, eligible: 0, applied: 0)
        check("q4 launch policy: vacuous zero eligible rejected",
              !Q4LaunchGatePolicy.shapeValid(zeroEligible, schedule: w32,
                                             binding: launchBinding))
        let missingEligible = try launchGate(diagnostic: false, gatedSteps: 32,
                                             startIdx: 0, eligible: NSNull(),
                                             applied: NSNull())
        check("q4 launch policy: missing counter records rejected",
              !Q4LaunchGatePolicy.shapeValid(missingEligible, schedule: w32,
                                             binding: launchBinding))
        let appliedMismatch = try launchGate(diagnostic: false, gatedSteps: 32,
                                             startIdx: 0, applied: 999)
        check("q4 launch policy: applied != eligible rejected",
              !Q4LaunchGatePolicy.shapeValid(appliedMismatch, schedule: w32,
                                             binding: launchBinding))
        let corruptStage = try launchGate(diagnostic: false, gatedSteps: 32,
                                          startIdx: 0, stages: 6240)
        check("q4 launch policy: corrupt stage count rejected",
              !Q4LaunchGatePolicy.shapeValid(corruptStage, schedule: w32,
                                             binding: launchBinding))
        let corruptMerge = try launchGate(diagnostic: false, gatedSteps: 32,
                                          startIdx: 0, merges: 27999)
        check("q4 launch policy: corrupt merge count rejected",
              !Q4LaunchGatePolicy.shapeValid(corruptMerge, schedule: w32,
                                             binding: launchBinding))
        // Hostile overflow inputs: out-of-bound layer/topK values must
        // be rejected by the CHECKED expected-count arithmetic, never
        // trap. The binding stays structurally valid apart from the one
        // mutated field the arithmetic must bound-check.
        var overflowObj = launchBindingJSON
        overflowObj = syntheticBindingJSON(
            layerCount: Q4LaunchGatePolicy.Q4LaunchBoundCheck.maxLayerCount)
        let overflowBinding = try JSONDecoder().decode(
            MoEModelBinding.self,
            from: try JSONSerialization.data(withJSONObject: overflowObj))
        checkEqual("q4 launch policy: checked stage arithmetic (hostile layers)",
                   Q4LaunchGatePolicy.expectedStageCount(gate: goodGate,
                                                         binding: overflowBinding),
                   2 * 128 * 117)
        overflowObj["layerCount"] = 30
        overflowObj["topK"] = 8
        overflowObj["guTensorStarts"] = (0..<30).map { $0 * guBytes }
        overflowObj["dnTensorStarts"] = (0..<30).map { 30 * guBytes + $0 * (1_115_136 * 128) }
        let restoredBinding = try JSONDecoder().decode(
            MoEModelBinding.self,
            from: try JSONSerialization.data(withJSONObject: overflowObj))
        checkEqual("q4 launch policy: restored binding counts unchanged",
                   Q4LaunchGatePolicy.expectedStageCount(gate: goodGate,
                                                         binding: restoredBinding),
                   7020)
        // Hostile fixture prompt count (beyond the bound) yields nil
        // expected counts — checked arithmetic, never a trap.
        var hostileGate = goodGate
        let hostileObj: [String: Any] = [
            "gatePolicy": Q4LaunchGateRecord.policyVersion,
            "executionPath": Q4LaunchGateRecord.executionPathSSA,
            "diagnostic": false, "genTokens": 32, "timedTokens": 31,
            "fixtures": [["name": "en-chat",
                          "promptTokenCount":
                              Q4LaunchGatePolicy.Q4LaunchBoundCheck.maxPromptTokens + 1,
                          "gatedSteps": 32,
                          "historyDigest": String(repeating: "a", count: 64)]],
            "results": [[String: Any]](),
            "baselineWallPerFixture": [Double](),
            "performanceSelected": false, "policyNote": "hostile"]
        hostileGate = try JSONDecoder().decode(
            Q4LaunchGateRecord.self,
            from: try JSONSerialization.data(withJSONObject: hostileObj))
        check("q4 launch policy: hostile prompt count yields nil expected counts",
              Q4LaunchGatePolicy.expectedStageCount(gate: hostileGate,
                                                    binding: launchBinding) == nil
                  && Q4LaunchGatePolicy.expectedMergeCount(gate: hostileGate,
                                                           binding: launchBinding) == nil)
        check("q4 launch: verification baseline hard pin accepts E4_R32",
              Gemma4Runner.verificationBaselineIsHardPinned(
                  schedule: MoESchedule(batch: 4, rowsPerGroup: 32),
                  sourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256))
        check("q4 launch: hard pin rejects E2 schedule",
              !Gemma4Runner.verificationBaselineIsHardPinned(
                  schedule: MoESchedule(batch: 2, rowsPerGroup: 32),
                  sourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256))
        check("q4 launch: hard pin rejects R64 schedule",
              !Gemma4Runner.verificationBaselineIsHardPinned(
                  schedule: MoESchedule(batch: 4, rowsPerGroup: 64),
                  sourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256))
        check("q4 launch: hard pin rejects caller-chosen hash",
              !Gemma4Runner.verificationBaselineIsHardPinned(
                  schedule: MoESchedule(batch: 4, rowsPerGroup: 32),
                  sourceSha256: String(repeating: "9", count: 64)))
    }

    if let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() {
        // GPU tail gate: 33 rows (a NON-multiple of every width) at the
        // real hot inDim 2816 — every width must be BIT-identical to the
        // incumbent default width and inside the FP64 oracle contract.
        let library = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
        let pipe = try device.makeComputePipelineState(
            function: library.makeFunction(name: Gemma4Runner.q4LaunchKernel)!)
        let inDim = 2816, rows = 33
        let blocksPerRow = inDim / 32
        var rng = SplitMix64(seed: 0x74_61_69_6C_33)
        var blocks = [UInt8]()
        blocks.reserveCapacity(rows * blocksPerRow * 18)
        for _ in 0..<(rows * blocksPerRow) {
            blocks += [UInt8(rng.next() & 0xFF), UInt8(rng.next() & 0xFF)]
            let n = blocks.count
            let expBits = (UInt16(blocks[n - 2]) | UInt16(blocks[n - 1]) << 8) & 0x7C00
            if expBits == 0x7C00 { blocks[n - 2] = 0x00; blocks[n - 1] = 0x3C }
            for _ in 0..<16 { blocks.append(UInt8(rng.next() & 0xFF)) }
        }
        let x = Q4LaunchForge.fixedInput(inDim: inDim, seed: 99)
        let oracle = try Q4Oracle.matvecFP64(blocks: blocks[...], outDim: rows,
                                             inDim: inDim, x: x)
        let wb = device.makeBuffer(bytes: blocks, length: blocks.count,
                                   options: .storageModeShared)!
        let xb = device.makeBuffer(bytes: x, length: x.count * 4,
                                   options: .storageModeShared)!
        let yb = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
        func run(_ width: Int) throws -> [Float] {
            try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb, yb: yb,
                                        inDim: inDim, rows: rows, width: width).out
        }
        let defWidth = max(1, min(rows, pipe.maxTotalThreadsPerThreadgroup))
        // The incumbent default (33 here) is NOT a power of two — it is
        // the UNCHANGED default geometry, allowed explicitly.
        let incumbent = try Q4LaunchForge.runMatvec(
            pipe: pipe, queue: queue, wb: wb, xb: xb, yb: yb,
            inDim: inDim, rows: rows, width: defWidth,
            requirePowerOfTwo: false).out
        check("q4 launch gpu: incumbent finite", incumbent.allSatisfy { $0.isFinite })
        var incRel = 0.0
        for i in 0..<rows {
            incRel = max(incRel, abs(Double(incumbent[i]) - oracle[i])
                         / max(1.0, abs(oracle[i])))
        }
        check("q4 launch gpu: incumbent inside oracle", incRel < 1e-4)
        for w in [32, 64, 128, 256, 1024] where w <= pipe.maxTotalThreadsPerThreadgroup {
            let out = try run(w)
            let bits = zip(out, incumbent).allSatisfy { $0.bitPattern == $1.bitPattern }
            check("q4 launch gpu: w\(w) tail-33 bit-identical to incumbent", bits)
            var rel = 0.0
            for i in 0..<rows {
                rel = max(rel, abs(Double(out[i]) - oracle[i]) / max(1.0, abs(oracle[i])))
            }
            check("q4 launch gpu: w\(w) inside oracle", rel < 1e-4)
        }
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: yb, inDim: inDim, rows: rows, width: 0)
             failed += 1; print("FAIL: q4 launch zero width rejected") } catch {
            check("q4 launch gpu: zero width rejected", true) }
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: yb, inDim: inDim, rows: 0, width: 32)
             failed += 1; print("FAIL: q4 launch zero rows rejected") } catch {
            check("q4 launch gpu: zero rows rejected", true) }
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: yb, inDim: inDim, rows: rows, width: 48)
             failed += 1; print("FAIL: q4 launch non-power-of-two width rejected") } catch {
            check("q4 launch gpu: non-power-of-two width rejected", true) }
        // Undersized/aliasing buffers are rejected BEFORE any submission.
        let shortY = device.makeBuffer(length: 4, options: .storageModeShared)!
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: shortY, inDim: inDim, rows: rows, width: 32)
             failed += 1; print("FAIL: q4 launch undersized y rejected") } catch {
            check("q4 launch gpu: undersized y buffer rejected", true) }
        let shortW = device.makeBuffer(length: 18, options: .storageModeShared)!
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: shortW, xb: xb,
                                             yb: yb, inDim: inDim, rows: rows, width: 32)
             failed += 1; print("FAIL: q4 launch undersized w rejected") } catch {
            check("q4 launch gpu: undersized weight buffer rejected", true) }
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: xb, inDim: inDim, rows: rows, width: 32)
             failed += 1; print("FAIL: q4 launch aliasing x/y rejected") } catch {
            check("q4 launch gpu: aliasing x/y buffers rejected", true) }
        do { _ = try Q4LaunchForge.runMatvec(pipe: pipe, queue: queue, wb: wb, xb: xb,
                                             yb: yb, inDim: inDim, rows: rows, width: 32,
                                             weightOffset: blocks.count - 1)
             failed += 1; print("FAIL: q4 launch weight extent rejected") } catch {
            check("q4 launch gpu: weight-offset extent rejected", true) }
    } else {
        print("SKIP: q4 launch GPU gates (Metal unavailable)")
    }

    // Artifact self-digest: a tampered body must never validate as a
    // whole (the digest is recomputed over every field except itself).
    do {
        let artGuBytes = 2_230_272 * 128, artDnBytes = 1_115_136 * 128
        let artExtent = 30 * (artGuBytes + artDnBytes)
        let binding: [String: Any] = [
            "graphSha256": String(repeating: "b", count: 64),
            "fileByteCount": artExtent,
            "residentBufferLength": artExtent,
            "maxBufferLength": artExtent * 2, "recommendedWorkingSetSize": artExtent * 4,
            "layerCount": 30, "expertCount": 128, "topK": 8,
            "hidden": 2816, "inter": 704, "gateUpOut": 1408,
            "guStride": 2_230_272, "dnStride": 1_115_136,
            "guTensorStarts": (0..<30).map { $0 * artGuBytes },
            "dnTensorStarts": (0..<30).map { 30 * artGuBytes + $0 * artDnBytes },
            "guScratchBytes": 45_056, "actScratchBytes": 22_528,
            "dnScratchBytes": 90_112, "scratchTotalBytes": 157_696]
        // Current-schema fixture: the current schema, a 64-hex cached model
        // SHA-256 in the legacy-named ggufDigest field, and the required
        // cached explicit-refresh policy.
        func artifactDict(width: Int, device: String) -> [String: Any] {
            ["artifactSchema": Q4LaunchArtifact.schema, "codegenVersion": "q4launch-v1",
             "schedule": ["width": width], "kernelName": "g4_q4_matvec",
             "primitiveSourceSha256": String(repeating: "c", count: 64),
             "baselineMoeSourceSha256": Q4LaunchArtifact.baselineMoeSourceSha256,
             "baselineMoeSchedule": ["batch": 4, "rowsPerGroup": 32],
             "baselineMoeBinding": binding,
             "regionGraphSha256": String(repeating: "d", count: 64),
             "hardware": ["deviceName": device, "maxTotalThreadsPerThreadgroup": 1024,
                          "threadExecutionWidth": 32,
                          "recommendedMaxWorkingSetSize": 1 << 30],
             "toolchain": Forge.toolchainIdentifier,
             "ggufDigest": String(repeating: "e", count: 64),
             "modelFileVerification": ModelFileIdentity.policy.rawValue,
             "executableDigest": String(repeating: "f", count: 64),
             "graphDigest": String(repeating: "0", count: 64),
             "hotShapes": [[String: Any]](),
             "componentResults": [[
                "schedule": ["width": width], "accepted": true,
                "bitIdenticalToIncumbent": true, "worstRelToOracle": 1e-7,
                "screenMicrosPerShape": [1.0], "note": "t"] as [String: Any]],
             "incumbentWorstRelToOracle": 1e-7,
             "modelGate": NSNull(), "artifactDigest": ""]
        }
        let dec = JSONDecoder()
        let honest = try dec.decode(
            Q4LaunchArtifact.self,
            from: try JSONSerialization.data(withJSONObject: artifactDict(width: 32, device: "t")))
        let d = try Q4LaunchArtifact.bodyDigest(honest)
        var withDigest = artifactDict(width: 32, device: "t")
        withDigest["artifactDigest"] = d
        let art = try dec.decode(
            Q4LaunchArtifact.self,
            from: try JSONSerialization.data(withJSONObject: withDigest))
        let artDigest = try Q4LaunchArtifact.bodyDigest(art)
        check("q4 launch artifact: self-digest matches honest body", artDigest == d)
        var tampered = withDigest
        tampered["schedule"] = ["width": 64]  // body changed, digest kept
        let bad = try dec.decode(
            Q4LaunchArtifact.self,
            from: try JSONSerialization.data(withJSONObject: tampered))
        let badDigest = try Q4LaunchArtifact.bodyDigest(bad)
        check("q4 launch artifact: tampered width fails self-digest", badDigest != d)
        var tamperedHw = withDigest
        tamperedHw["hardware"] = ["deviceName": "other",
                                  "maxTotalThreadsPerThreadgroup": 1024,
                                  "threadExecutionWidth": 32,
                                  "recommendedMaxWorkingSetSize": 1 << 30]
        let badHw = try dec.decode(
            Q4LaunchArtifact.self,
            from: try JSONSerialization.data(withJSONObject: tamperedHw))
        let badHwDigest = try Q4LaunchArtifact.bodyDigest(badHw)
        check("q4 launch artifact: tampered hardware fails self-digest",
              badHwDigest != d)

        // Current-schema policy contract on the same CPU fixture: the
        // explicit policy is part of the self-digested body, required on
        // decode, and the only representable value. The legacy schema-1
        // fixture below is an explicit copied mutation, not the default.
        checkEqual("q4 launch artifact: fixture is the current schema",
                   honest.artifactSchema, Q4LaunchArtifact.schema)
        checkEqual("q4 launch artifact: legacy-named ggufDigest holds the cached model SHA-256",
                   honest.ggufDigest, String(repeating: "e", count: 64))
        checkEqual("q4 launch artifact: decoded policy is cached explicit refresh",
                   honest.modelFileVerification, .cachedExplicitRefresh)
        let sortedEncoder = JSONEncoder()
        sortedEncoder.outputFormatting = [.sortedKeys]
        let honestBody = try sortedEncoder.encode(honest)
        let honestObj = (try JSONSerialization.jsonObject(with: honestBody)
            as? [String: Any]) ?? [:]
        checkEqual("q4 launch artifact: body encoding carries the explicit policy",
                   honestObj["modelFileVerification"] as? String ?? "", "cached-explicit-refresh-v1")
        checkEqual("q4 launch artifact: self-digest is over the policy-bearing body",
                   d, Array(SHA256.hash(Array(honestBody)))
                       .map { String(format: "%02x", $0) }.joined())
        var missingDict = withDigest
        missingDict.removeValue(forKey: "modelFileVerification")
        let missingDecoded = try? dec.decode(
            Q4LaunchArtifact.self, from: JSONSerialization.data(withJSONObject: missingDict))
        check("q4 launch artifact: missing policy fails to decode", missingDecoded == nil)
        var claimedDict = withDigest
        claimedDict["modelFileVerification"] = "sha256-full-file"
        let claimedDecoded = try? dec.decode(
            Q4LaunchArtifact.self, from: JSONSerialization.data(withJSONObject: claimedDict))
        check("q4 launch artifact: verified-claiming policy fails to decode",
              claimedDecoded == nil)
        var staleDict = withDigest
        staleDict["ggufDigest"] = String(repeating: "9", count: 64)
        let stale = try dec.decode(
            Q4LaunchArtifact.self,
            from: try JSONSerialization.data(withJSONObject: staleDict))
        let staleDigest = try Q4LaunchArtifact.bodyDigest(stale)
        check("q4 launch artifact: stale 64-hex model hash fails self-digest",
              staleDigest != d)
        var legacyDict = artifactDict(width: 32, device: "t")
        legacyDict["artifactSchema"] = 1
        legacyDict["ggufDigest"] = String(repeating: "e", count: 64)
        legacyDict.removeValue(forKey: "modelFileVerification")
        let legacyDecoded = try? dec.decode(
            Q4LaunchArtifact.self, from: JSONSerialization.data(withJSONObject: legacyDict))
        check("q4 launch artifact: legacy schema-1 record without the policy fails to decode",
              legacyDecoded == nil)
    }

    if let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() {
        // GPU component gates at the TAIL region (topK 3, hidden 96, inter
        // 32) with synthetic Q4_0 bytes: batched E in {1,2,4} (tails) must
        // be BIT-IDENTICAL to the ACTUAL incumbent kernels on all slots,
        // and both must sit inside the FP64 oracle contract.
        let guStride = try MoEModelBinding.sliceStride(
            inDim: tailRegion.hidden, outDim: tailRegion.gateUpOut,
            MoELoweringError.invalidRegion)
        let dnStride = try MoEModelBinding.sliceStride(
            inDim: tailRegion.expertIntermediate, outDim: tailRegion.hidden,
            MoELoweringError.invalidRegion)
        var rng = SplitMix64(seed: 0x5A4C_4F57)
        func syntheticBlocks(stride: Int, experts: Int) -> [UInt8] {
            var blocks = [UInt8]()
            blocks.reserveCapacity(stride * experts)
            for _ in 0..<(stride * experts / 18) {
                blocks += [UInt8(rng.next() & 0xFF), UInt8(rng.next() & 0xFF)]
                let n = blocks.count
                let expBits = (UInt16(blocks[n - 2]) | UInt16(blocks[n - 1]) << 8) & 0x7C00
                if expBits == 0x7C00 { blocks[n - 2] = 0x00; blocks[n - 1] = 0x3C }
                for _ in 0..<16 { blocks.append(UInt8(rng.next() & 0xFF)) }
            }
            return blocks
        }
        let guBlocks = syntheticBlocks(stride: guStride, experts: tailRegion.topK)
        let dnBlocks = syntheticBlocks(stride: dnStride, experts: tailRegion.topK)
        var xrng = SplitMix64(seed: 70_004)
        let x = (0..<tailRegion.hidden).map { _ in xrng.nextUniform() }
        let inc = try MoELoweringForge.runIncumbent(
            device: device, queue: queue, region: tailRegion,
            guBlocks: guBlocks, dnBlocks: dnBlocks, x: x)
        let incRel = try MoELoweringForge.oracleRels(
            region: tailRegion, guBlocks: guBlocks, dnBlocks: dnBlocks,
            guGpu: inc.gu, actGpu: inc.act, dnGpu: inc.dn, x: x)
        check("moe gpu: incumbent inside FP64 contract", incRel < 1e-4)
        let caps = ForgeHardwareProfile(
            deviceName: device.name, maxTotalThreadsPerThreadgroup: 1024,
            threadExecutionWidth: 32,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize))
        for e in [1, 2, 4] {
            // R ladder including padded-row tiles (last group partially
            // active): the row-tile mapping must stay bit-identical.
            for r in [32, 128, 256] {
                let sched = MoESchedule(batch: e, rowsPerGroup: r)
                guard sched.isLegal(region: tailRegion, hardware: caps) else { continue }
                let kernels = try MoEEmittedKernels.compile(region: tailRegion,
                                                            schedule: sched, device: device)
                let out = try kernels.runComponent(region: tailRegion,
                                                   guBlocks: guBlocks,
                                                   dnBlocks: dnBlocks, x: x)
                func bits(_ a: [Float], _ b: [Float]) -> Bool {
                    a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
                }
                check("moe gpu: E\(e) R\(r) tail GU bit-identical to incumbent",
                      bits(out.gu, inc.gu))
                check("moe gpu: E\(e) R\(r) tail ACT bit-identical to incumbent",
                      bits(out.act, inc.act))
                check("moe gpu: E\(e) R\(r) tail DN bit-identical to incumbent",
                      bits(out.dn, inc.dn))
                let rel = try MoELoweringForge.oracleRels(
                    region: tailRegion, guBlocks: guBlocks, dnBlocks: dnBlocks,
                    guGpu: out.gu, actGpu: out.act, dnGpu: out.dn, x: x)
                check("moe gpu: E\(e) R\(r) batched inside FP64 contract", rel < 1e-4)
            }
        }
        // Fail-closed component guards: short weights and wrong-length x.
        let kernels1 = try MoEEmittedKernels.compile(
            region: tailRegion, schedule: MoESchedule(batch: 1, rowsPerGroup: 32),
            device: device)
        do { _ = try kernels1.runComponent(region: tailRegion,
                                           guBlocks: Array(guBlocks.dropLast(18)),
                                           dnBlocks: dnBlocks, x: x)
             failed += 1; print("FAIL: moe short GU rejected") } catch {
            check("moe component: short GU bytes rejected", true) }
        do { _ = try kernels1.runComponent(region: tailRegion, guBlocks: guBlocks,
                                           dnBlocks: dnBlocks, x: Array(x.dropLast()))
             failed += 1; print("FAIL: moe short x rejected") } catch {
            check("moe component: wrong-length x rejected", true) }

        // PRE-REFACTOR vs factored g4_gelu_up bit gate: compile the OLD
        // literal kernel (our own committed source, pre-helper form) and
        // the CURRENT shader's kernel, compare bitPatterns on bounded,
        // extreme, and tail inputs at the real n=704 geometry.
        let oldGeluSource = """
        #include <metal_stdlib>
        using namespace metal;
        static float g4_tanh(float x) {
            float e = exp(-2.0f * fabs(x));
            return copysign((1.0f - e) / (1.0f + e), x);
        }
        kernel void old_gelu_up(constant const uint& n [[buffer(0)]],
                                device const float* gu [[buffer(1)]],
                                device float* y [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
            uint i = gid;
            if (i >= n) { return; }
            float g = gu[i];
            float u = gu[n + i];
            float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
            float gelu = 0.5f * g * (1.0f + g4_tanh(inner));
            y[i] = gelu * u;
        }
        """
        let newLib = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
        let oldLib = try device.makeLibrary(source: oldGeluSource, options: nil)
        let newFn = newLib.makeFunction(name: "g4_gelu_up")!
        let oldFn = oldLib.makeFunction(name: "old_gelu_up")!
        let newPipe = try device.makeComputePipelineState(function: newFn)
        let oldPipe = try device.makeComputePipelineState(function: oldFn)
        func runGelu(_ pipe: MTLComputePipelineState, _ input: [Float], n: Int) throws -> [Float] {
            let guB = device.makeBuffer(bytes: input, length: input.count * 4,
                                        options: .storageModeShared)!
            let yB = device.makeBuffer(length: n * 4, options: .storageModeShared)!
            let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
            var count = UInt32(n)
            enc.setComputePipelineState(pipe)
            enc.setBytes(&count, length: 4, index: 0)
            enc.setBuffer(guB, offset: 0, index: 1)
            enc.setBuffer(yB, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw Gemma4Error.commandBufferFailed(
                    "gelu bit gate kernel: \(String(describing: cmd.error))")
            }
            return [Float](UnsafeBufferPointer(
                start: yB.contents().assumingMemoryBound(to: Float.self), count: n))
        }
        for (name, n, values) in [
            ("bounded n=704", 704, (0..<2 * 704).map { i in
                Float(i % 97) / 13.0 - 3.0 }),
            ("extreme", 704, (0..<2 * 704).map { i in
                i % 2 == 0 ? Float(1e30) : (i % 4 == 1 ? -1e30 : 1e-30) }),
            ("tail n=3", 3, [0.5, -0.25, 8.0, -9.5, 1e20, 0.0]),
        ] as [(String, Int, [Float])] {
            let oldOut = try runGelu(oldPipe, values, n: n)
            let newOut = try runGelu(newPipe, values, n: n)
            let equal = zip(oldOut, newOut).allSatisfy { $0.bitPattern == $1.bitPattern }
            check("moe gelu refactor bit gate: \(name)", equal)
        }
    } else {
        print("SKIP: moe GPU gates (Metal unavailable)")
    }
} catch {
    failed += 1; print("FAIL: moe region tests — threw \(error)")
}

// MARK: - GemmaForge.fileDigest

do {
    let fileManager = FileManager.default
    let temporaryRoot = NSTemporaryDirectory()
        .appending("gemmaforge-filedigest-\(UUID().uuidString)")
    try fileManager.createDirectory(atPath: temporaryRoot, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(atPath: temporaryRoot) }

    func fixturePath(_ name: String) -> String {
        temporaryRoot + "/" + name
    }

    func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: fixturePath(name))
    }

    func writeFixture(_ data: Data, _ name: String) -> URL? {
        do {
            let url = fixtureURL(name)
            try data.write(to: url)
            return url
        } catch {
            failed += 1
            print("FAIL: unexpected error while writing fixture \(name): \(error)")
            return nil
        }
    }

    func digestForFile(at url: URL) -> String? {
        do {
            return try GemmaForge.fileDigest(url)
        } catch {
            failed += 1
            print("FAIL: unexpected error while digesting \(url.path): \(error)")
            return nil
        }
    }

    // Known digest: empty input.
    if let url = writeFixture(Data("".utf8), "empty.bin"),
       let digest = digestForFile(at: url) {
        checkEqual(
            "fileDigest of empty file matches known SHA256",
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
    }

    // Known digest: \"abc\" input.
    if let url = writeFixture(Data("abc".utf8), "abc.bin"),
       let digest = digestForFile(at: url) {
        checkEqual(
            "fileDigest of abc file matches known SHA256",
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    }

    // Repeated calls on the same file must produce the same digest.
    if let url = writeFixture(Data("determinism".utf8), "repeat.bin"),
       let first = digestForFile(at: url),
       let second = digestForFile(at: url) {
        checkEqual(
            "fileDigest repeated calls are deterministic",
            second,
            first
        )
    }

    // A missing file must throw instead of returning a digest.
    do {
        _ = try GemmaForge.fileDigest(fixtureURL("missing.bin"))
        check("fileDigest throws for missing file", false)
    } catch {
        check("fileDigest throws for missing file", true)
    }

    // Patterned fixture larger than 9 MiB with a non-multiple-of-chunk length.
    let patternLength = 9 * 1024 * 1024 + 12345
    let patternFixture = Data((0..<patternLength).map { UInt8($0 % 256) })
    let expectedPatternDigest = SHA256.hex(SHA256.hash(Array(patternFixture)))
    if let url = writeFixture(patternFixture, "pattern.bin"),
       let digest = digestForFile(at: url) {
        checkEqual(
            "fileDigest of 9 MiB plus 12345 patterned file matches LeviathanCore SHA256",
            digest,
            expectedPatternDigest
        )
    }
} catch {
    failed += 1
    print("FAIL: unexpected setup error for GemmaForge.fileDigest tests: \(error)")
}

// MARK: - GPU cache-append (g4_cache_append)

do {
    let enc = JSONEncoder()
    let dec = JSONDecoder()
    let t = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false, gpuCacheAppend: true)
    let f = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false)
    let tj = String(data: try enc.encode(t), encoding: .utf8) ?? ""
    let fj = String(data: try enc.encode(f), encoding: .utf8) ?? ""
    check("gpuCacheAppend JSON reflects true", tj.contains("\"gpuCacheAppend\":true"))
    check("gpuCacheAppend JSON reflects false", fj.contains("\"gpuCacheAppend\":false"))
    check("gpuCacheAppend JSON roundtrip", (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(t))) == t)
    // Legacy artifact carrying ALL old candidate fields but missing the
    // new gpuCacheAppend key must be rejected, not silently defaulted.
    let legacy = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false}"
    check("legacy artifact without gpuCacheAppend rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacy.utf8))) == nil)
    // parallelAttention axis: explicit Codable key, default false,
    // missing key rejected, bounded candidate counts.
    let pa = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                 vectorizedLoads: true, parallelAttention: true)
    let paj = String(data: try enc.encode(pa), encoding: .utf8) ?? ""
    check("parallelAttention JSON reflects true", paj.contains("\"parallelAttention\":true"))
    check("parallelAttention JSON roundtrip",
          (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(pa))) == pa)
    check("parallelAttention default false",
          !GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).parallelAttention)
    let legacyNoPA = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false}"
    check("legacy artifact without parallelAttention rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoPA.utf8))) == nil)
    // rmsnormCooperative axis: explicit Codable key, roundtrip, default
    // false, and missing-key rejection for legacy artifacts (the new
    // key is NEVER silently defaulted — the intended legacy rejections
    // above stay missing-key rejections).
    let rc = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                 vectorizedLoads: true, rmsnormCooperative: true)
    let rcj = String(data: try enc.encode(rc), encoding: .utf8) ?? ""
    check("rmsnormCooperative JSON reflects true", rcj.contains("\"rmsnormCooperative\":true"))
    check("rmsnormCooperative JSON roundtrip",
          (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(rc))) == rc)
    check("rmsnormCooperative default false",
          !GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).rmsnormCooperative)
    let legacyNoRC = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false,\"parallelAttention\":false}"
    check("legacy artifact without rmsnormCooperative rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoRC.utf8))) == nil)
    // matvecCooperativeQ6 axis: explicit Codable key, roundtrip, default
    // false, missing-key rejection for legacy artifacts, and no legal
    // candidate combining the new cooperative Q6 axis with the old
    // tiled Q6 axis.
    let q6c = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                  vectorizedLoads: true, matvecCooperativeQ6: true)
    let q6cj = String(data: try enc.encode(q6c), encoding: .utf8) ?? ""
    check("matvecCooperativeQ6 JSON reflects true", q6cj.contains("\"matvecCooperativeQ6\":true"))
    check("matvecCooperativeQ6 JSON roundtrip",
          (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(q6c))) == q6c)
    check("matvecCooperativeQ6 default false",
          !GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).matvecCooperativeQ6)
    let legacyNoQ6C = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false,\"parallelAttention\":false,\"rmsnormCooperative\":false}"
    check("legacy artifact without matvecCooperativeQ6 rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoQ6C.utf8))) == nil)
    // rmsStatisticsReuse axis: explicit Codable key, roundtrip, default
    // false, and missing-key rejection for legacy artifacts. The legacy
    // fixture carries ALL older fields INCLUDING matvecCooperativeQ6:false
    // so the rejection is attributable to the MISSING rmsStatisticsReuse
    // key alone (no false-positive missingkey from an older field).
    let sr = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                 vectorizedLoads: true, rmsStatisticsReuse: true)
    let srj = String(data: try enc.encode(sr), encoding: .utf8) ?? ""
    check("rmsStatisticsReuse JSON reflects true", srj.contains("\"rmsStatisticsReuse\":true"))
    check("rmsStatisticsReuse JSON reflects false", fj.contains("\"rmsStatisticsReuse\":false"))
    check("rmsStatisticsReuse JSON roundtrip",
          (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(sr))) == sr)
    check("rmsStatisticsReuse default false",
          !GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).rmsStatisticsReuse)
    let legacyNoSR = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false,\"parallelAttention\":false,\"rmsnormCooperative\":false,\"matvecCooperativeQ6\":false}"
    check("legacy artifact without rmsStatisticsReuse rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoSR.utf8))) == nil)
    // Router residency axes: explicit Codable keys, roundtrip, defaults,
    // and missing-key rejection for legacy artifacts (fixtures carry ALL
    // older fields INCLUDING rmsStatisticsReuse:false so the rejection is
    // attributable to the missing router key alone).
    let rr = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false,
                                 vectorizedLoads: true, routerGPUResidency: true,
                                 routerLaunchDivisor: 4)
    let rrj = String(data: try enc.encode(rr), encoding: .utf8) ?? ""
    check("routerGPUResidency JSON reflects true", rrj.contains("\"routerGPUResidency\":true"))
    check("routerLaunchDivisor JSON reflects 4", rrj.contains("\"routerLaunchDivisor\":4"))
    check("router residency candidate JSON roundtrip",
          (try? dec.decode(GemmaForgeCandidate.self, from: enc.encode(rr))) == rr)
    check("routerGPUResidency default false",
          !GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).routerGPUResidency)
    checkEqual("routerLaunchDivisor default 0",
               GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false).routerLaunchDivisor, 0)
    let legacyNoRR = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false,\"parallelAttention\":false,\"rmsnormCooperative\":false,\"matvecCooperativeQ6\":false,\"rmsStatisticsReuse\":false}"
    check("legacy artifact without routerGPUResidency rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoRR.utf8))) == nil)
    let legacyNoDiv = "{\"segmentBatching\":true,\"top5Diagnostic\":false,\"matvecTiledQ4\":false,\"matvecTiledQ6\":false,\"rmsnormTiled\":false,\"matvecTiledF32\":false,\"vectorizedLoads\":false,\"gpuCacheAppend\":false,\"parallelAttention\":false,\"rmsnormCooperative\":false,\"matvecCooperativeQ6\":false,\"rmsStatisticsReuse\":false,\"routerGPUResidency\":false}"
    check("legacy artifact without routerLaunchDivisor rejected",
          (try? dec.decode(GemmaForgeCandidate.self, from: Data(legacyNoDiv.utf8))) == nil)
    // Router candidate legality (PURE, device-free): divisors restricted
    // to {0,1,2,4,8}; nonzero requires routerGPUResidency and no tiled F32.
    check("router divisor 0 candidate validates",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
              rmsnormCooperative: true, matvecCooperativeQ6: true,
              rmsStatisticsReuse: true, routerGPUResidency: true,
              routerLaunchDivisor: 0))) != nil)
    check("router divisor 3 rejected",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
              rmsnormCooperative: true, matvecCooperativeQ6: true,
              rmsStatisticsReuse: true, routerGPUResidency: true,
              routerLaunchDivisor: 3))) == nil)
    check("nonzero router divisor without residency rejected",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true,
              rmsnormCooperative: true, matvecCooperativeQ6: true,
              rmsStatisticsReuse: true, routerLaunchDivisor: 1))) == nil)
    check("nonzero router divisor with tiled F32 rejected",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false, matvecTiledF32: true,
              vectorizedLoads: true, gpuCacheAppend: true, parallelAttention: true,
              rmsnormCooperative: true, matvecCooperativeQ6: true,
              rmsStatisticsReuse: true, routerGPUResidency: true,
              routerLaunchDivisor: 2))) == nil)
    // Statistics candidate legality (PURE, device-free): legal only
    // with cooperative RMSNorm + vectorized loads and tiled RMSNorm
    // off; every incompatible combination is rejected.
    check("legal statistics candidate validates",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false,
              vectorizedLoads: true, rmsnormCooperative: true,
              rmsStatisticsReuse: true))) != nil)
    check("statistics candidate rejected with rmsnormCooperative=false",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false,
              vectorizedLoads: true,
              rmsStatisticsReuse: true))) == nil)
    check("statistics candidate rejected with vectorizedLoads=false",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false,
              rmsnormCooperative: true,
              rmsStatisticsReuse: true))) == nil)
    check("statistics candidate rejected with rmsnormTiled=true",
          (try? GemmaForge.validateCandidate(GemmaForgeCandidate(
              segmentBatching: true, top5Diagnostic: false,
              rmsnormTiled: true, vectorizedLoads: true,
              rmsnormCooperative: true,
              rmsStatisticsReuse: true))) == nil)
    // Statistics kernel/plan guard surface (device-free subset).
    check("rms statistics elementCount overflow fails closed",
          RMSStatisticsKernels.elementCount(width: Int.max, vectors: 2) == nil
          && RMSStatisticsKernels.elementCount(width: 0, vectors: 1) == nil
          && RMSStatisticsKernels.elementCount(width: 4, vectors: 1) == 4)
    check("rms statistics weight offset alignment rule",
          RMSStatisticsKernels.weightOffsetIsAligned(0)
          && RMSStatisticsKernels.weightOffsetIsAligned(16)
          && !RMSStatisticsKernels.weightOffsetIsAligned(8)
          && !RMSStatisticsKernels.weightOffsetIsAligned(-16))
    check("rms statistics eligibility mirrors cooperative gate",
          RMSStatisticsKernels.eligible(n: 256) && RMSStatisticsKernels.eligible(n: 8192)
          && !RMSStatisticsKernels.eligible(n: 254) && !RMSStatisticsKernels.eligible(n: 8196))
    let shaStats = RMSStatisticsKernels.sourceSha256()
    check("rms statistics source sha is 64 lowercase hex chars",
          shaStats.count == 64
          && shaStats.allSatisfy { ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f") })
    let cands = GemmaForge.candidates()
    checkEqual("bounded candidate count", cands.count, 32)
    checkEqual("gpuCacheAppend candidates", cands.filter { $0.gpuCacheAppend }.count, 18)
    checkEqual("parallelAttention candidates", cands.filter { $0.parallelAttention }.count, 17)
    checkEqual("rmsnormCooperative candidates", cands.filter { $0.rmsnormCooperative }.count, 11)
    checkEqual("matvecCooperativeQ6 candidates", cands.filter { $0.matvecCooperativeQ6 }.count, 8)
    checkEqual("rmsStatisticsReuse candidates", cands.filter { $0.rmsStatisticsReuse }.count, 6)
    checkEqual("routerGPUResidency candidates", cands.filter { $0.routerGPUResidency }.count, 5)
    checkEqual("nonzero routerLaunchDivisor candidates",
               cands.filter { $0.routerLaunchDivisor != 0 }.count, 4)
    check("no legal candidate combines nonzero routerLaunchDivisor with matvecTiledF32",
          !cands.contains { $0.routerLaunchDivisor != 0 && $0.matvecTiledF32 })
    check("no legal candidate enables rmsStatisticsReuse and rmsnormTiled together",
          !cands.contains { $0.rmsStatisticsReuse && $0.rmsnormTiled })
    check("every bounded candidate passes validateCandidate",
          cands.allSatisfy { (try? GemmaForge.validateCandidate($0)) != nil })
    check("no legal candidate enables rmsnormCooperative and rmsnormTiled together",
          !cands.contains { $0.rmsnormCooperative && $0.rmsnormTiled })
    check("no legal candidate enables matvecCooperativeQ6 and matvecTiledQ6 together",
          !cands.contains { $0.matvecCooperativeQ6 && $0.matvecTiledQ6 })
    check("old baseline default false", !cands[0].gpuCacheAppend && !cands[1].gpuCacheAppend
          && !cands[0].parallelAttention && !cands[1].parallelAttention
          && !cands[0].rmsnormCooperative && !cands[1].rmsnormCooperative
          && !cands[0].matvecCooperativeQ6 && !cands[1].matvecCooperativeQ6
          && !cands[0].rmsStatisticsReuse && !cands[1].rmsStatisticsReuse
          && !cands[0].routerGPUResidency && !cands[1].routerGPUResidency
          && cands[0].routerLaunchDivisor == 0 && cands[1].routerLaunchDivisor == 0)
} catch {
    failed += 1
    print("FAIL cache-append candidate JSON: \(error)")
}

// Geometry boundary: PURE static helper on real MTL buffers — no GPU
// segment, no runner state, no trap on gigantic width/position.
if let device = MTLCreateSystemDefaultDevice() {
    do {
        let srcK = device.makeBuffer(length: 64, options: [])!
        let srcV = device.makeBuffer(length: 64, options: [])!
        let dstK = device.makeBuffer(length: 4096, options: [])!
        let dstV = device.makeBuffer(length: 4096, options: [])!
        let geo = Gemma4Runner.cacheAppendGeometry(srcK:srcK, srcV:srcV, dstK:dstK, dstV:dstV,
                                                   kvWidth: 8, position: 5)
        check("geometry valid with larger-than-required sources", geo)
        check("geometry insufficient input rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: 32, position: 0))
        check("geometry dest capacity rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: 16, position: 100))
        check("geometry negative position rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: 8, position: -1))
        check("geometry zero width rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: 0, position: 0))
        check("geometry negative width rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: -8, position: 1))
        check("geometry width Int.max rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: Int.max, position: 1))
        check("geometry position Int.max rejected",
              !Gemma4Runner.cacheAppendGeometry(srcK: srcK, srcV: srcV, dstK: dstK, dstV: dstV,
                                                kvWidth: 8, position: Int.max))
    } catch {
        failed += 1
        print("FAIL cache-append geometry: \(error)")
    }
} else {
    print("SKIP cache-append geometry: Metal unavailable")
}

// Component test: the raw uint kernel copies every bit exactly (quiet/
// signaling NaN, +/-inf, +/-0, subnormals, negatives) at two distinct
// destination offsets with an odd ragged width of 9 against 64-thread
// groups; ALL destination words compared, sources verified unchanged.
if let device = MTLCreateSystemDefaultDevice() {
    do {
        let lib = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
        guard let fn = lib.makeFunction(name: "g4_cache_append") else {
            throw NSError(domain: "test", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "makeFunction(g4_cache_append) returned nil"])
        }
        // makeComputePipelineState throws and returns a non-optional
        // MTLComputePipelineState.
        let pipe = try device.makeComputePipelineState(function: fn)
        guard let q = device.makeCommandQueue() else {
            throw NSError(domain: "test", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "makeCommandQueue returned nil"])
        }

        let srcK: [UInt32] = [0x7fc00000, 0xffc00000, 0x3f800000, 0x80000000,
                              0x00000001, 0x7f800000, 0xff800000, 0x3f7fffff, 0x41200000]
        let srcV: [UInt32] = [0x40490fdb, 0xc0490fdb, 0x00000000, 0xffffffff,
                              0x7f7fffff, 0x00000001, 0xbf800000, 0x3eaaaaab, 0x4b189680]
        let width = srcK.count
        let offA = 3
        let offB = offA + width + 6
        let dstLen = offB + width + 5
        let fill: [UInt32] = (0..<dstLen).map { 0xdeadbeef &+ UInt32(truncatingIfNeeded: $0 &* 2654435761) }
        func mkBuffer(_ words: [UInt32]) -> MTLBuffer {
            words.withUnsafeBytes { raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: words.count * 4,
                                  options: .storageModeShared)!
            }
        }
        let kb = mkBuffer(srcK)
        let vb = mkBuffer(srcV)
        let kdb = mkBuffer(fill)
        let vdb = mkBuffer(fill)

        let cmd = q.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        for off in [offA, offB] {
            var n = UInt32(width)
            var wordOffset = UInt32(off)
            enc.setBuffer(kb, offset: 0, index: 0)
            enc.setBuffer(vb, offset: 0, index: 1)
            enc.setBuffer(kdb, offset: 0, index: 2)
            enc.setBuffer(vdb, offset: 0, index: 3)
            enc.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&wordOffset, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (width + 63) / 64, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        check("cache-append command completed", cmd.status == .completed)
        check("cache-append no command error", cmd.error == nil)
        check("cache-append positive GPU time", cmd.gpuStartTime < cmd.gpuEndTime)

        // Read word count from the buffer's own length so the source
        // buffers (9 words) are never read out of bounds; destination
        // buffers yield all dstLen words including untouched sentinels.
        func words(_ b: MTLBuffer) -> [UInt32] {
            Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt32.self),
                                      count: b.length / 4))
        }
        var expK = fill
        var expV = fill
        for off in [offA, offB] {
            expK.replaceSubrange(off..<(off + width), with: srcK)
            expV.replaceSubrange(off..<(off + width), with: srcV)
        }
        checkEqual("cache-append K dest words all match", words(kdb), expK)
        checkEqual("cache-append V dest words all match", words(vdb), expV)
        checkEqual("cache-append K source unchanged", words(kb), srcK)
        checkEqual("cache-append V source unchanged", words(vb), srcV)
    } catch {
        failed += 1
        print("FAIL cache-append GPU copy: \(error)")
    }
} else {
    print("SKIP cache-append GPU copy: Metal unavailable")
}

// MARK: - GemmaCompositeForge.loadArtifact bounded-loader gates

/// Bounded-temporary-fixture IO gates for the composite artifact loader
/// (the ONLY public surface of GemmaCompositeForge that is testable
/// without a real GGUF): missing file -> nil, undecodable JSON -> decode
/// rejection, and a file exactly ONE BYTE over the public read limit ->
/// size-guard rejection BEFORE any decode is attempted (the oversized
/// fixture is deliberately '{'-bytes, i.e. also undecodable, so a
/// decode-flavored error would prove the guard ran too late). The
/// oversized fixture is written in bounded 1 MiB chunks; every fixture
/// is removed by defer.
@MainActor
func runGemmaCompositeForgeLoaderGates() {
    let tmp = FileManager.default.temporaryDirectory
    func freshURL(_ tag: String) -> URL {
        tmp.appendingPathComponent(
            "leviathan-composite-\(tag)-\(UUID().uuidString).json")
    }

    // Missing file: nil, never an error — callers distinguish reuse-only.
    let missing = freshURL("missing")
    do {
        let loaded = try GemmaCompositeForge.loadArtifact(from: missing)
        check("composite loader: missing file returns nil", loaded == nil)
    } catch {
        failed += 1
        print("FAIL: composite loader: missing file returns nil — threw \(error)")
    }

    // Undecodable JSON at a normal size: decode rejection.
    let bad = freshURL("badjson")
    do {
        try Data("{ not json".utf8).write(to: bad)
        defer { try? FileManager.default.removeItem(at: bad) }
        _ = try GemmaCompositeForge.loadArtifact(from: bad)
        failed += 1
        print("FAIL: composite loader: invalid JSON rejected — no error")
    } catch {
        check("composite loader: invalid JSON rejected (decode error)",
              "\(error)".contains("decode"))
    }

    // Exactly limit+1 bytes, written in bounded chunks: the size guard
    // must fire, and must fire BEFORE decode (the bytes are also
    // undecodable, so a decode error here would mean the guard ran late).
    let big = freshURL("oversize")
    do {
        let total = GemmaCompositeForge.artifactReadLimitBytes + 1
        FileManager.default.createFile(atPath: big.path, contents: nil)
        let handle = try FileHandle(forWritingTo: big)
        let chunk = Data(repeating: 0x7B, count: 1 << 20)
        var written = 0
        while written < total {
            let n = min(chunk.count, total - written)
            try handle.write(contentsOf: chunk[0..<n])
            written += n
        }
        try handle.close()
        defer { try? FileManager.default.removeItem(at: big) }
        _ = try GemmaCompositeForge.loadArtifact(from: big)
        failed += 1
        print("FAIL: composite loader: oversized artifact rejected — no error")
    } catch {
        let message = "\(error)"
        check("composite loader: oversized artifact rejected by size guard",
              message.contains("exceeds"))
        check("composite loader: size guard fires before decode",
              !message.contains("decode"))
    }
}

runGemmaCompositeForgeLoaderGates()

// MARK: - Structured compile diagnostics (Gemma4CompileDiagnostic)

/// Synthetic NSError payloads per stage, Codable roundtrip, Gemma4Error
/// payload equality, and one REAL Metal invalid-source compile. Messages
/// are compared only against the originating error's own
/// localizedDescription, never against literal localized text.
@MainActor
func runCompileDiagnosticGates() {
    let synthetic = NSError(domain: "leviathan.test.compile", code: 42,
                            userInfo: [NSLocalizedDescriptionKey: "synthetic compile failure"])
    let stages: [Gemma4CompileDiagnostic.Stage] = [.library, .function, .pipeline]
    checkEqual("compile diagnostic stages distinct", Set(stages.map(\.rawValue)).count, 3)
    for stage in stages {
        let kernel = "k_\(stage.rawValue)"
        let d = Gemma4CompileDiagnostic.make(stage: stage, kernel: kernel, error: synthetic)
        checkEqual("compile diagnostic \(stage.rawValue): stage retained", d.stage, stage)
        checkEqual("compile diagnostic \(stage.rawValue): kernel retained", d.kernel, kernel)
        checkEqual("compile diagnostic \(stage.rawValue): domain retained", d.domain, synthetic.domain)
        checkEqual("compile diagnostic \(stage.rawValue): code retained", d.code, synthetic.code)
        checkEqual("compile diagnostic \(stage.rawValue): message is localizedDescription",
                   d.message, synthetic.localizedDescription)
        do {
            let data = try JSONEncoder().encode(d)
            let back = try JSONDecoder().decode(Gemma4CompileDiagnostic.self, from: data)
            checkEqual("compile diagnostic \(stage.rawValue): Codable roundtrip", back, d)
        } catch {
            failed += 1; print("FAIL: compile diagnostic \(stage.rawValue) Codable — threw \(error)")
        }
    }

    let a = Gemma4CompileDiagnostic.make(stage: .pipeline, kernel: "k", error: synthetic)
    let sameA = Gemma4CompileDiagnostic.make(stage: .pipeline, kernel: "k", error: synthetic)
    let otherStage = Gemma4CompileDiagnostic.make(stage: .library, kernel: "k", error: synthetic)
    let otherCode = Gemma4CompileDiagnostic.make(
        stage: .pipeline, kernel: "k",
        error: NSError(domain: synthetic.domain, code: 43, userInfo: synthetic.userInfo))
    check("Gemma4Error.compileFailed equal for equal payloads",
          Gemma4Error.compileFailed(a) == .compileFailed(sameA))
    check("Gemma4Error.compileFailed differs by stage",
          Gemma4Error.compileFailed(a) != .compileFailed(otherStage))
    check("Gemma4Error.compileFailed differs by code",
          Gemma4Error.compileFailed(a) != .compileFailed(otherCode))
    check("Gemma4Error.compileFailed distinct from shaderCompileFailed",
          Gemma4Error.compileFailed(a) != .shaderCompileFailed("k"))

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("SKIP: compile diagnostic Metal invalid-source gate (no Metal device)")
        return
    }
    // Same contract as the runner's lazy accessors: a failed compile is
    // thrown as Gemma4Error.compileFailed, never a fallback library.
    let invalidSource = "kernel void g4_diag_invalid( { this is not valid MSL"
    var originalError: NSError?
    func compileInvalid() throws -> MTLLibrary {
        do { return try device.makeLibrary(source: invalidSource, options: nil) }
        catch {
            originalError = error as NSError
            throw Gemma4Error.compileFailed(
                .make(stage: .library, kernel: "g4_diag_invalid", error: error))
        }
    }
    do {
        _ = try compileInvalid()
        failed += 1; print("FAIL: invalid MSL source compiled without throwing")
    } catch let Gemma4Error.compileFailed(diagnostic) {
        guard let original = originalError else {
            failed += 1; print("FAIL: invalid MSL compile — original NSError not captured")
            return
        }
        passed += 1
        checkEqual("metal invalid source: stage is library", diagnostic.stage, .library)
        checkEqual("metal invalid source: kernel retained", diagnostic.kernel, "g4_diag_invalid")
        checkEqual("metal invalid source: domain retained", diagnostic.domain, original.domain)
        checkEqual("metal invalid source: code retained", diagnostic.code, original.code)
        checkEqual("metal invalid source: message is original localizedDescription",
                   diagnostic.message, original.localizedDescription)
        check("metal invalid source: domain non-empty", !diagnostic.domain.isEmpty)
    } catch {
        failed += 1; print("FAIL: invalid MSL compile — unexpected error \(error)")
    }
}

runCompileDiagnosticGates()
// MARK: - IR-derived opportunity discovery (pure shadow, gpuCacheAppend)

/// One synthetic cache_append block: fresh t2/kv versions are defined by
/// matvecs reading the live x, then appended to kvstate_<layer>.
private struct ODAppendSpec {
    let layer: Int
    let kvHeads: Int
    let headDim: Int
    let width: Int
    var keySlice: Int? = nil
}

private func odHardware(_ name: String = "shadow-test") -> ForgeHardwareProfile {
    ForgeHardwareProfile(deviceName: name, maxTotalThreadsPerThreadgroup: 1024,
                         threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1 << 30)
}

/// Small synthetic config: local layers 2x4, global layers 1x8 (same
/// 8-word width, different geometry).
private func odSyntheticConfig(layerCount: Int = 2, hidden: Int = 8, vocabSize: Int = 16,
                               localKVHeads: Int = 2, globalKVHeads: Int = 1,
                               localHeadDim: Int = 4, globalHeadDim: Int = 8,
                               finalSoftcap: Float = 30,
                               globalLayers: [Int] = [1]) -> GemmaGraphIR.GraphConfig {
    GemmaGraphIR.GraphConfig(
        layerCount: layerCount, hidden: hidden, qHeads: 2,
        localKVHeads: localKVHeads, globalKVHeads: globalKVHeads,
        localHeadDim: localHeadDim, globalHeadDim: globalHeadDim,
        slidingWindow: 4, vocabSize: vocabSize, expertCount: 4, expertUsedCount: 2,
        expertIntermediate: 8, sharedIntermediate: 8,
        finalSoftcap: finalSoftcap, globalLayers: globalLayers)
}

/// Builds a small but fully VALID schedule from scratch through Codable
/// JSON: fresh versions on every write, live-version uses, and a logits
/// softcap terminal (hidden 8, vocab 16).
private func odSyntheticSSA(_ specs: [ODAppendSpec]) -> GemmaSSAIR? {
    var versions = [String: Int]()
    var values = [[String: Any]]()
    var instructions = [[String: Any]]()
    func def(_ name: String, _ length: Int, kind: String = "vector",
             slice: Int? = nil) -> String {
        let v = (versions[name] ?? -1) + 1
        versions[name] = v
        var value: [String: Any] = ["name": name, "version": v, "dtype": "f32",
                                    "length": length, "kind": kind]
        if let slice { value["sliceOffset"] = slice }
        values.append(value)
        return "\(name)@\(v)"
    }
    func live(_ name: String) -> String { "\(name)@\(versions[name] ?? 0)" }
    func emit(_ op: String, _ layer: Int?, _ inputs: [String], _ output: String,
              tensors: [String] = [], attrs: [String: Int] = [:]) {
        var inst: [String: Any] = ["op": op, "inputs": inputs, "output": output,
                                   "tensors": tensors, "attrs": attrs]
        if let layer { inst["layer"] = layer }
        instructions.append(inst)
    }
    let x = def("x", 8)
    emit("embed_row", nil, [], x, tensors: ["token_embd.weight"])
    for s in specs {
        let src = live("x")
        let k = def("t2", s.width, slice: s.keySlice)
        emit("matvec", s.layer, [src], k, tensors: ["k"], attrs: ["rows": s.width])
        let v = def("kv", s.width)
        emit("matvec", s.layer, [src], v, tensors: ["v"], attrs: ["rows": s.width])
        let state = def("kvstate_\(s.layer)", s.width, kind: "kvstate")
        emit("cache_append", s.layer, [k, v], state,
             attrs: ["kvHeads": s.kvHeads, "headDim": s.headDim])
    }
    let logitsIn = live("x")
    let l0 = def("logits", 16)
    emit("matvec", nil, [logitsIn], l0, tensors: ["token_embd.weight"], attrs: ["rows": 16])
    let l1 = def("logits", 16)
    emit("softcap", nil, [l0], l1, attrs: ["cap": 30])
    let root: [String: Any] = ["schema": GemmaSSA.schemaVersion, "values": values,
                               "instructions": instructions]
    guard let data = try? JSONSerialization.data(withJSONObject: root) else { return nil }
    return try? JSONDecoder().decode(GemmaSSAIR.self, from: data)
}

private func odMutated(_ ssa: GemmaSSAIR,
                       _ mutate: (inout [String: Any]) -> Void) -> GemmaSSAIR? {
    guard let data = try? JSONEncoder().encode(ssa),
          var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else { return nil }
    mutate(&obj)
    guard let out = try? JSONSerialization.data(withJSONObject: obj) else { return nil }
    return try? JSONDecoder().decode(GemmaSSAIR.self, from: out)
}

private func odIsConfigError(_ e: GemmaOpportunityDiscoveryError) -> Bool {
    if case .invalidConfig = e { return true }
    return false
}

private func odIsHardwareError(_ e: GemmaOpportunityDiscoveryError) -> Bool {
    if case .invalidHardware = e { return true }
    return false
}

private func odIsSSAError(_ e: GemmaOpportunityDiscoveryError) -> Bool {
    if case .malformedSSA = e { return true }
    return false
}

/// A fixture counts toward the valid-but-unsupported boundary ONLY when
/// the existing validator accepts it.
@MainActor
private func odValidates(_ name: String, _ ssa: GemmaSSAIR,
                         _ config: GemmaGraphIR.GraphConfig) -> Bool {
    do {
        try GemmaSSA.validate(ssa, config: config)
        passed += 1
        return true
    } catch {
        failed += 1
        print("FAIL: \(name): fixture rejected by GemmaSSA.validate (not a valid-but-unsupported proof) - \(error)")
        return false
    }
}

@MainActor
private func odExpectThrows(_ name: String,
                            _ matches: (GemmaOpportunityDiscoveryError) -> Bool,
                            _ body: () throws -> Void) {
    do {
        try body()
        failed += 1; print("FAIL: \(name) - no error")
    } catch let e as GemmaOpportunityDiscoveryError {
        if matches(e) { passed += 1 }
        else { failed += 1; print("FAIL: \(name) - wrong case \(e)") }
    } catch {
        failed += 1; print("FAIL: \(name) - unexpected \(error)")
    }
}

@MainActor
private func odCheckShadow(_ name: String, _ ssa: GemmaSSAIR,
                           _ config: GemmaGraphIR.GraphConfig, supported: Bool) {
    let manual = GemmaForge.candidates()
    do {
        let shadow = try GemmaOpportunityDiscovery.shadowCandidates(
            ssa: ssa, config: config, hardware: odHardware())
        checkEqual("\(name): baseline retained first", shadow.first, manual.first)
        checkEqual("\(name): shadow list", shadow,
                   supported ? manual
                       : manual.enumerated()
                           .filter { $0.offset == 0 || !$0.element.gpuCacheAppend }
                           .map(\.element))
        check("\(name): non-cache-append tuples untouched",
              manual.filter { !$0.gpuCacheAppend }.allSatisfy { shadow.contains($0) })
    } catch {
        failed += 1; print("FAIL: \(name) shadow - threw \(error)")
    }
}

@MainActor
private func odExpectSyntheticUnsupported(_ name: String, _ specs: [ODAppendSpec],
                                          reasons: [GemmaOpportunityReport.Reason],
                                          provenLayers: [Int]) {
    let config = odSyntheticConfig()
    guard let ssa = odSyntheticSSA(specs) else {
        failed += 1; print("FAIL: \(name) - synthetic fixture unrepresentable")
        return
    }
    guard odValidates(name, ssa, config) else { return }
    do {
        let report = try GemmaOpportunityDiscovery.discover(
            ssa: ssa, config: config, hardware: odHardware())
        check("\(name): unsupported", !report.body.supported)
        checkEqual("\(name): exact reasons", report.body.reasons, reasons)
        checkEqual("\(name): proven layers", report.body.proofs.map(\.layer), provenLayers)
        let again = try GemmaOpportunityDiscovery.discover(
            ssa: ssa, config: config, hardware: odHardware())
        checkEqual("\(name): deterministic report", again, report)
        let verified = try report.verifyDigest()
        check("\(name): digest verifies", verified)
    } catch {
        failed += 1; print("FAIL: \(name) - threw \(error)")
    }
    odCheckShadow(name, ssa, config, supported: false)
}

@MainActor
func runOpportunityDiscoveryGates() {
    let hw = odHardware()
    let manual = GemmaForge.candidates()

    // Canonical real config lowered from the expected sequence (no Metal,
    // no GGUF tensors needed).
    let canonicalConfig: GemmaGraphIR.GraphConfig
    let canonicalSSA: GemmaSSAIR
    do {
        var b = GGUFTestBuilder()
        b.appendGemma4Metadata()
        let cfg = try Gemma4Config(file: GGUFReader.parse(b.finish()))
        canonicalConfig = GemmaGraphIR.GraphConfig(
            layerCount: cfg.layerCount, hidden: cfg.hiddenSize, qHeads: cfg.qHeads,
            localKVHeads: cfg.localKVHeads, globalKVHeads: cfg.globalKVHeads,
            localHeadDim: cfg.localHeadDim, globalHeadDim: cfg.globalHeadDim,
            slidingWindow: cfg.slidingWindow, vocabSize: cfg.vocabSize,
            expertCount: cfg.expertCount, expertUsedCount: cfg.topKExperts,
            expertIntermediate: cfg.expertIntermediate,
            sharedIntermediate: cfg.sharedIntermediate,
            finalSoftcap: cfg.finalSoftcap, globalLayers: cfg.globalLayers)
        canonicalSSA = try GemmaSSA.lower(GemmaGraphIR(
            schema: 1, model: "gemma4-26b-a4b", config: canonicalConfig,
            ops: GemmaGraph.expectedSequence(canonicalConfig)))
    } catch {
        failed += 1; print("FAIL: od canonical setup - threw \(error)")
        return
    }

    // Canonical: supported, each layer evidenced once, whole-tuple
    // equivalence, repeated/Codable determinism, digest tamper.
    do {
        let report = try GemmaOpportunityDiscovery.discover(
            ssa: canonicalSSA, config: canonicalConfig, hardware: hw)
        let body = report.body
        check("od canonical: gpuCacheAppend axis supported", body.supported)
        checkEqual("od canonical: no withheld reasons", body.reasons, [])
        checkEqual("od canonical: axis name", body.axis, "gpuCacheAppend")
        checkEqual("od canonical: analysis version", body.analysisVersion,
                   GemmaOpportunityDiscovery.analysisVersion)
        checkEqual("od canonical: each layer evidenced once, in order",
                   body.proofs.map(\.layer), Array(0..<canonicalConfig.layerCount))
        checkEqual("od canonical: global evidence matches config",
                   body.proofs.filter(\.global).map(\.layer),
                   canonicalConfig.globalLayers.sorted())
        check("od canonical: every proof binds t2 then kv into its kvstate",
              body.proofs.allSatisfy { p in
                  p.keyInputID.hasPrefix("t2@") && p.valueInputID.hasPrefix("kv@")
                      && p.outputID == "kvstate_\(p.layer)@0" })
        check("od canonical: proof geometry is the layer's config geometry",
              body.proofs.allSatisfy { p in
                  let heads = p.global ? canonicalConfig.globalKVHeads : canonicalConfig.localKVHeads
                  let dim = p.global ? canonicalConfig.globalHeadDim : canonicalConfig.localHeadDim
                  return p.kvHeads == heads && p.headDim == dim && p.wordWidth == heads * dim
              })
        check("od canonical: proof indices are cache_append instructions",
              body.proofs.allSatisfy {
                  canonicalSSA.instructions[$0.instructionIndex].op == "cache_append" })
        checkEqual("od canonical: SSA digest recorded", body.ssaDigest,
                   try GemmaSSA.digest(canonicalSSA))
        checkEqual("od canonical: config recorded", body.config, canonicalConfig)
        checkEqual("od canonical: hardware recorded", body.hardware, hw)
        let verified = try report.verifyDigest()
        check("od canonical: digest verifies", verified)
        // Recorded regression constants for the current canonical fixture
        // and report codec under odHardware() (shadow-test/1024/32/1GiB),
        // observed independently from the validated lowered IR. Not a
        // portable proof: Foundation's JSONEncoder output may differ across
        // platforms or releases. The report digest already covers ssaDigest;
        // the separate SSA pin tells lowering drift apart from codec drift.
        checkEqual("od canonical: recorded SSA digest", body.ssaDigest,
                   "28a04586de4fe9b62d56c56723e55ba9f8668548d3cb29db1a565a42097f8583")
        checkEqual("od canonical: recorded report digest", report.reportDigest,
                   "9900d8cd07e974722f6a9bc3ff712476746f504c1ea0b4cefefd0df3c5272199")

        let shadow = try GemmaOpportunityDiscovery.shadowCandidates(
            ssa: canonicalSSA, config: canonicalConfig, hardware: hw)
        checkEqual("od canonical: shadow list equals manual list (ordered)", shadow, manual)
        checkEqual("od canonical: baseline retained first", shadow.first, manual.first)

        let again = try GemmaOpportunityDiscovery.discover(
            ssa: canonicalSSA, config: canonicalConfig, hardware: hw)
        checkEqual("od repeated: identical report", again, report)
        checkEqual("od repeated: identical digest", again.reportDigest, report.reportDigest)

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(GemmaOpportunityReport.self, from: encoded)
        checkEqual("od codable: body roundtrip", decoded.body, report.body)
        checkEqual("od codable: report roundtrip", decoded, report)
        checkEqual("od codable: digest recomputed from body alone",
                   try GemmaOpportunityReport.digest(of: decoded.body), report.reportDigest)
        let decodedVerified = try decoded.verifyDigest()
        check("od codable: decoded digest verifies", decodedVerified)

        func tampered(_ edit: (inout [String: Any]) -> Void) throws -> GemmaOpportunityReport {
            var obj = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] ?? [:]
            edit(&obj)
            return try JSONDecoder().decode(
                GemmaOpportunityReport.self,
                from: JSONSerialization.data(withJSONObject: obj))
        }
        let metaTamper = try tampered { obj in
            var b = obj["body"] as? [String: Any] ?? [:]
            var h = b["hardware"] as? [String: Any] ?? [:]
            h["deviceName"] = "tampered-device"
            b["hardware"] = h
            obj["body"] = b
        }
        check("od tamper: metadata edit changes body", metaTamper.body != report.body)
        let metaVerified = try metaTamper.verifyDigest()
        check("od tamper: metadata edit fails digest", !metaVerified)
        check("od tamper: metadata edit changes recomputed digest",
              (try? GemmaOpportunityReport.digest(of: metaTamper.body)) != report.reportDigest)
        let configTamper = try tampered { obj in
            var b = obj["body"] as? [String: Any] ?? [:]
            var c = b["config"] as? [String: Any] ?? [:]
            c["layerCount"] = 29
            b["config"] = c
            obj["body"] = b
        }
        check("od tamper: config edit changes body", configTamper.body != report.body)
        let configVerified = try configTamper.verifyDigest()
        check("od tamper: config edit fails digest", !configVerified)
        let proofTamper = try tampered { obj in
            var b = obj["body"] as? [String: Any] ?? [:]
            var proofs = b["proofs"] as? [[String: Any]] ?? []
            if !proofs.isEmpty { proofs[0]["headDim"] = 7 }
            b["proofs"] = proofs
            obj["body"] = b
        }
        check("od tamper: proof edit changes body", proofTamper.body != report.body)
        let proofVerified = try proofTamper.verifyDigest()
        check("od tamper: proof edit fails digest", !proofVerified)

        let otherHW = try GemmaOpportunityDiscovery.discover(
            ssa: canonicalSSA, config: canonicalConfig, hardware: odHardware("other-device"))
        check("od digest covers hardware profile", otherHW.reportDigest != report.reportDigest)
        check("od other hardware keeps the static verdict", otherHW.body.supported)
    } catch {
        failed += 1; print("FAIL: od canonical gates - threw \(error)")
    }

    // Swapped cache operands that still pass SSA validation.
    if let appendIndex = canonicalSSA.instructions.firstIndex(where: { $0.op == "cache_append" }) {
        let originalInputs = canonicalSSA.instructions[appendIndex].inputs
        let swappedInputs = Array(originalInputs.reversed())
        let swapped = odMutated(canonicalSSA) { obj in
            var insts = obj["instructions"] as? [[String: Any]] ?? []
            insts[appendIndex]["inputs"] = swappedInputs
            obj["instructions"] = insts
        }
        if let swapped, swapped != canonicalSSA {
            if odValidates("od swapped operands", swapped, canonicalConfig) {
                do {
                    let r = try GemmaOpportunityDiscovery.discover(
                        ssa: swapped, config: canonicalConfig, hardware: hw)
                    check("od swapped operands: unsupported", !r.body.supported)
                    checkEqual("od swapped operands: exact instruction/operand evidence",
                               r.body.reasons,
                               [GemmaOpportunityReport.Reason(
                                   code: .operandBinding,
                                   layer: canonicalSSA.instructions[appendIndex].layer,
                                   instructionIndices: [appendIndex],
                                   valueIDs: swappedInputs)])
                    checkEqual("od swapped operands: remaining layers proven",
                               r.body.proofs.map(\.layer),
                               Array(1..<canonicalConfig.layerCount))
                } catch {
                    failed += 1; print("FAIL: od swapped operands - threw \(error)")
                }
                odCheckShadow("od swapped operands", swapped, canonicalConfig, supported: false)
            }
        } else {
            failed += 1; print("FAIL: od swapped operands - fixture unrepresentable or unchanged")
        }
    } else {
        failed += 1; print("FAIL: od canonical schedule has no cache_append")
    }

    // Stale references: rejected by the existing validator, so discovery
    // throws before producing any report.
    if let stale = odMutated(canonicalSSA, { obj in
        obj["instructions"] = (obj["instructions"] as? [[String: Any]] ?? []).map { inst in
            var i = inst
            if (i["op"] as? String) == "rmsnorm", var ins = i["inputs"] as? [String],
               ins.first?.hasPrefix("x@") == true {
                ins[0] = "x@0"; i["inputs"] = ins
            }
            return i
        }
    }) {
        check("od stale: fixture rejected by GemmaSSA.validate",
              (try? GemmaSSA.validate(stale, config: canonicalConfig)) == nil)
        odExpectThrows("od stale reference throws before report", odIsSSAError) {
            _ = try GemmaOpportunityDiscovery.discover(
                ssa: stale, config: canonicalConfig, hardware: hw)
        }
        odExpectThrows("od stale reference throws from shadow helper", odIsSSAError) {
            _ = try GemmaOpportunityDiscovery.shadowCandidates(
                ssa: stale, config: canonicalConfig, hardware: hw)
        }
    } else {
        failed += 1; print("FAIL: od stale fixture unrepresentable")
    }

    // Synthetic schedules rebuilt with fresh versions and live uses.
    let synthConfig = odSyntheticConfig()
    let local0 = ODAppendSpec(layer: 0, kvHeads: 2, headDim: 4, width: 8)
    let global1 = ODAppendSpec(layer: 1, kvHeads: 1, headDim: 8, width: 8)
    for (label, specs) in [("od synthetic covered", [local0, global1]),
                           ("od synthetic zero slice offset",
                            [ODAppendSpec(layer: 0, kvHeads: 2, headDim: 4, width: 8,
                                          keySlice: 0), global1])] {
        guard let ssa = odSyntheticSSA(specs) else {
            failed += 1; print("FAIL: \(label) - fixture unrepresentable")
            continue
        }
        guard odValidates(label, ssa, synthConfig) else { continue }
        do {
            let r = try GemmaOpportunityDiscovery.discover(
                ssa: ssa, config: synthConfig, hardware: hw)
            check("\(label): supported", r.body.supported)
            checkEqual("\(label): no reasons", r.body.reasons, [])
            checkEqual("\(label): proof layers", r.body.proofs.map(\.layer), [0, 1])
            checkEqual("\(label): proof globalness", r.body.proofs.map(\.global), [false, true])
            checkEqual("\(label): proof indices", r.body.proofs.map(\.instructionIndex), [3, 6])
            checkEqual("\(label): proof heads", r.body.proofs.map(\.kvHeads), [2, 1])
            checkEqual("\(label): proof word widths", r.body.proofs.map(\.wordWidth), [8, 8])
        } catch {
            failed += 1; print("FAIL: \(label) - threw \(error)")
        }
        odCheckShadow(label, ssa, synthConfig, supported: true)
    }

    odExpectSyntheticUnsupported(
        "od duplicate coverage", [local0, local0, global1],
        reasons: [GemmaOpportunityReport.Reason(
            code: .duplicateLayer, layer: 0, instructionIndices: [3, 6],
            valueIDs: ["kvstate_0@0", "kvstate_0@1"])],
        provenLayers: [0, 0, 1])
    odExpectSyntheticUnsupported(
        "od missing coverage", [local0],
        reasons: [GemmaOpportunityReport.Reason(
            code: .missingLayer, layer: 1, instructionIndices: [], valueIDs: [])],
        provenLayers: [0])
    odExpectSyntheticUnsupported(
        "od local geometry on global layer",
        [local0, ODAppendSpec(layer: 1, kvHeads: 2, headDim: 4, width: 8)],
        reasons: [GemmaOpportunityReport.Reason(
            code: .geometryMismatch, layer: 1, instructionIndices: [6],
            valueIDs: ["t2@1", "kv@1"])],
        provenLayers: [0])
    odExpectSyntheticUnsupported(
        "od nonzero input slice offset",
        [ODAppendSpec(layer: 0, kvHeads: 2, headDim: 4, width: 8, keySlice: 4), global1],
        reasons: [GemmaOpportunityReport.Reason(
            code: .operandSlice, layer: 0, instructionIndices: [3],
            valueIDs: ["t2@0", "kv@0"])],
        provenLayers: [1])
    // Validator-consistent (attrs 2*8 == length 16 <= 8*hidden) but not the
    // config's local 2x4 geometry: three ordered reasons on one instruction.
    odExpectSyntheticUnsupported(
        "od multi-reason local geometry",
        [ODAppendSpec(layer: 0, kvHeads: 2, headDim: 8, width: 16), global1],
        reasons: [
            GemmaOpportunityReport.Reason(
                code: .geometryMismatch, layer: 0, instructionIndices: [3],
                valueIDs: ["t2@0", "kv@0"]),
            GemmaOpportunityReport.Reason(
                code: .operandLength, layer: 0, instructionIndices: [3],
                valueIDs: ["t2@0", "kv@0"]),
            GemmaOpportunityReport.Reason(
                code: .outputMismatch, layer: 0, instructionIndices: [3],
                valueIDs: ["kvstate_0@0"]),
        ],
        provenLayers: [1])

    // A cache_append without a layer that the existing validator still
    // accepts (output kvstate_-1): withheld as layerAbsent, leaving layer 0
    // uncovered.
    if let absentBase = odSyntheticSSA([local0, global1]),
       let absent = odMutated(absentBase, { obj in
           var insts = obj["instructions"] as? [[String: Any]] ?? []
           insts[3].removeValue(forKey: "layer")
           insts[3]["output"] = "kvstate_-1@0"
           obj["instructions"] = insts
           obj["values"] = (obj["values"] as? [[String: Any]] ?? []).map { value in
               var v = value
               if (v["name"] as? String) == "kvstate_0" {
                   v["name"] = "kvstate_-1"
                   if v["id"] != nil { v["id"] = "kvstate_-1@0" }
               }
               return v
           }
       }) {
        if odValidates("od absent layer", absent, synthConfig) {
            do {
                let r = try GemmaOpportunityDiscovery.discover(
                    ssa: absent, config: synthConfig, hardware: hw)
                check("od absent layer: unsupported", !r.body.supported)
                checkEqual("od absent layer: exact reasons", r.body.reasons, [
                    GemmaOpportunityReport.Reason(
                        code: .layerAbsent, layer: nil, instructionIndices: [3],
                        valueIDs: ["kvstate_-1@0"]),
                    GemmaOpportunityReport.Reason(
                        code: .missingLayer, layer: 0, instructionIndices: [],
                        valueIDs: []),
                ])
                checkEqual("od absent layer: proven layers", r.body.proofs.map(\.layer), [1])
            } catch {
                failed += 1; print("FAIL: od absent layer - threw \(error)")
            }
            odCheckShadow("od absent layer", absent, synthConfig, supported: false)
        }
    } else {
        failed += 1; print("FAIL: od absent-layer fixture unrepresentable")
    }

    // Unknown IR, invalid config, and invalid profile throw.
    if let base = odSyntheticSSA([local0, global1]) {
        if let unknown = odMutated(base, { obj in
            var insts = obj["instructions"] as? [[String: Any]] ?? []
            insts[1]["op"] = "conv2d"
            obj["instructions"] = insts
        }) {
            odExpectThrows("od unknown op fails the existing validator", odIsSSAError) {
                _ = try GemmaOpportunityDiscovery.discover(
                    ssa: unknown, config: synthConfig, hardware: hw)
            }
        } else {
            failed += 1; print("FAIL: od unknown-op fixture unrepresentable")
        }
        let badConfigs: [(String, GemmaGraphIR.GraphConfig)] = [
            ("8*hidden overflow", odSyntheticConfig(hidden: Int.max / 4)),
            ("zero layers", odSyntheticConfig(layerCount: 0, globalLayers: [])),
            ("layer count beyond diagnostic bound",
             odSyntheticConfig(layerCount: GemmaOpportunityDiscovery.maxLayerCount + 1)),
            ("zero vocab", odSyntheticConfig(vocabSize: 0)),
            ("zero local kv heads", odSyntheticConfig(localKVHeads: 0)),
            ("negative global head dim", odSyntheticConfig(globalHeadDim: -8)),
            ("local geometry product overflow",
             odSyntheticConfig(localKVHeads: Int.max, localHeadDim: 2)),
            ("NaN softcap", odSyntheticConfig(finalSoftcap: .nan)),
            ("infinite softcap", odSyntheticConfig(finalSoftcap: .infinity)),
            ("global layer out of range", odSyntheticConfig(globalLayers: [2])),
            ("negative global layer", odSyntheticConfig(globalLayers: [-1])),
            ("duplicate global layer", odSyntheticConfig(globalLayers: [1, 1])),
        ]
        for (label, config) in badConfigs {
            odExpectThrows("od invalid config throws: \(label)", odIsConfigError) {
                _ = try GemmaOpportunityDiscovery.discover(
                    ssa: base, config: config, hardware: hw)
            }
        }
        let badHardware: [(String, ForgeHardwareProfile)] = [
            ("empty device name", ForgeHardwareProfile(
                deviceName: "", maxTotalThreadsPerThreadgroup: 1024,
                threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1 << 30)),
            ("zero max threads", ForgeHardwareProfile(
                deviceName: "t", maxTotalThreadsPerThreadgroup: 0,
                threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1 << 30)),
            ("zero execution width", ForgeHardwareProfile(
                deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                threadExecutionWidth: 0, recommendedMaxWorkingSetSize: 1 << 30)),
            ("execution width above max threads", ForgeHardwareProfile(
                deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                threadExecutionWidth: 2048, recommendedMaxWorkingSetSize: 1 << 30)),
            ("zero working set", ForgeHardwareProfile(
                deviceName: "t", maxTotalThreadsPerThreadgroup: 1024,
                threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 0)),
        ]
        for (label, profile) in badHardware {
            odExpectThrows("od invalid profile throws: \(label)", odIsHardwareError) {
                _ = try GemmaOpportunityDiscovery.discover(
                    ssa: base, config: synthConfig, hardware: profile)
            }
            odExpectThrows("od invalid profile throws from shadow helper: \(label)",
                           odIsHardwareError) {
                _ = try GemmaOpportunityDiscovery.shadowCandidates(
                    ssa: base, config: synthConfig, hardware: profile)
            }
        }
    } else {
        failed += 1; print("FAIL: od synthetic base fixture unrepresentable")
    }
}

runOpportunityDiscoveryGates()

// MARK: - Two-axis shadow composition (gpuCacheAppend + rmsStatisticsReuse)

private typealias SCReport = GemmaComposedOpportunityReport

/// Independent whole-tuple oracle over the manual list: index 0 is pinned,
/// every later tuple is kept, in order, only when each composed axis it
/// enables is supported. All other flags ride along untouched.
private func scExpectedShadow(cache: Bool, rms: Bool) -> [GemmaForgeCandidate] {
    GemmaForge.candidates().enumerated()
        .filter { $0.offset == 0
            || ((!$0.element.gpuCacheAppend || cache)
                && (!$0.element.rmsStatisticsReuse || rms)) }
        .map(\.element)
}

/// Re-encodes a report through Foundation JSON, editing only `body`, so a
/// tampered fixture is a genuinely decoded report (never a public init).
private func scTampered(_ report: SCReport,
                        _ edit: (inout [String: Any]) -> Void) throws -> SCReport {
    let data = try JSONEncoder().encode(report)
    var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    var body = obj["body"] as? [String: Any] ?? [:]
    edit(&body)
    obj["body"] = body
    return try JSONDecoder().decode(
        SCReport.self, from: JSONSerialization.data(withJSONObject: obj))
}

@MainActor
private func scExpectTamperFails(_ name: String, _ original: SCReport,
                                 _ tampered: SCReport) {
    check("\(name): edit changes body", tampered.body != original.body)
    do {
        let verified = try tampered.verifyDigest()
        check("\(name): edit fails digest", !verified)
        let recomputed = try SCReport.digest(of: tampered.body)
        check("\(name): recomputed digest differs", recomputed != original.reportDigest)
    } catch {
        failed += 1; print("FAIL: \(name) - threw \(error)")
    }
}

/// One support combination end to end: both evidence pieces equal the
/// independent direct analyses, verdicts follow the stated predicates,
/// and the fresh shadow list is the exact whole-tuple filter.
@MainActor
private func scCheckCombination(_ name: String, _ ssa: GemmaSSAIR,
                                _ config: GemmaGraphIR.GraphConfig, epsilon: Float,
                                cache: Bool, rms: Bool,
                                filteredCount: Int) -> SCReport? {
    let hw = odHardware()
    do {
        let report = try GemmaShadowComposition.discover(
            ssa: ssa, config: config, hardware: hw, epsilon: epsilon)
        let directCache = try GemmaOpportunityDiscovery.discover(
            ssa: ssa, config: config, hardware: hw)
        let directPlan = try RMSStatisticsPlan.analyze(ssa: ssa, epsilon: epsilon)
        checkEqual("\(name): cache evidence is the existing report",
                   report.body.cacheReport, directCache)
        checkEqual("\(name): RMS evidence is the direct existing plan",
                   report.body.rmsPlan, directPlan)
        checkEqual("\(name): ordered axes", report.body.verdicts.map(\.axis),
                   SCReport.Axis.allCases)
        checkEqual("\(name): cache verdict reflects existing report",
                   report.verdict(for: .gpuCacheAppend)?.supported,
                   directCache.body.supported)
        checkEqual("\(name): cache verdict",
                   report.verdict(for: .gpuCacheAppend)?.supported, cache)
        checkEqual("\(name): cache reason",
                   report.verdict(for: .gpuCacheAppend)?.reason,
                   cache ? nil : SCReport.ReasonCode.cacheAppendUnsupported)
        checkEqual("\(name): RMS verdict follows groupCount>0 && reuseCount>0",
                   report.verdict(for: .rmsStatisticsReuse)?.supported,
                   directPlan.groupCount > 0 && directPlan.reuseCount > 0)
        checkEqual("\(name): RMS verdict",
                   report.verdict(for: .rmsStatisticsReuse)?.supported, rms)
        checkEqual("\(name): RMS reason",
                   report.verdict(for: .rmsStatisticsReuse)?.reason,
                   rms ? nil : SCReport.ReasonCode.noReusableGroups)
        checkEqual("\(name): epsilon bits recorded exactly",
                   report.body.epsilonBits, epsilon.bitPattern)
        checkEqual("\(name): composition version", report.body.compositionVersion,
                   GemmaShadowComposition.compositionVersion)
        checkEqual("\(name): RMS analysis version", report.body.rmsAnalysisVersion,
                   RMSStatisticsPlan.analysisVersion)
        checkEqual("\(name): recorded cache report version unchanged",
                   report.body.cacheReport.body.analysisVersion, 1)
        let verified = try report.verifyDigest()
        check("\(name): digest verifies", verified)
        let shadow = try GemmaShadowComposition.shadowCandidates(
            ssa: ssa, config: config, hardware: hw, epsilon: epsilon)
        checkEqual("\(name): shadow whole tuples", shadow,
                   scExpectedShadow(cache: cache, rms: rms))
        checkEqual("\(name): shadow count pinned", shadow.count, filteredCount)
        checkEqual("\(name): manual baseline retained first", shadow.first,
                   GemmaForge.candidates().first)
        return report
    } catch {
        failed += 1; print("FAIL: \(name) - threw \(error)")
        return nil
    }
}

@MainActor
func runShadowCompositionGates() {
    let hw = odHardware()
    let manual = GemmaForge.candidates()
    // Explicit test epsilons; the composed API has no default epsilon.
    let epsA: Float = 1e-6
    let epsB: Float = 1e-3

    checkEqual("sc manual baseline count pinned", manual.count, 32)
    checkEqual("sc manual gpuCacheAppend tuples pinned",
               manual.filter(\.gpuCacheAppend).count, 18)
    checkEqual("sc manual rmsStatisticsReuse tuples pinned",
               manual.filter(\.rmsStatisticsReuse).count, 6)
    check("sc manual rmsStatisticsReuse tuples all carry gpuCacheAppend",
          manual.filter(\.rmsStatisticsReuse).allSatisfy(\.gpuCacheAppend))
    check("sc manual index 0 enables neither axis",
          !manual[0].gpuCacheAppend && !manual[0].rmsStatisticsReuse)
    checkEqual("sc existing cache analysis version unchanged",
               GemmaOpportunityDiscovery.analysisVersion, 1)
    checkEqual("sc existing RMS analysis version", RMSStatisticsPlan.analysisVersion, 2)
    checkEqual("sc composition version", GemmaShadowComposition.compositionVersion, 1)

    // Canonical real config lowered from the expected sequence.
    let canonicalConfig: GemmaGraphIR.GraphConfig
    let canonicalSSA: GemmaSSAIR
    do {
        var b = GGUFTestBuilder()
        b.appendGemma4Metadata()
        let cfg = try Gemma4Config(file: GGUFReader.parse(b.finish()))
        canonicalConfig = GemmaGraphIR.GraphConfig(
            layerCount: cfg.layerCount, hidden: cfg.hiddenSize, qHeads: cfg.qHeads,
            localKVHeads: cfg.localKVHeads, globalKVHeads: cfg.globalKVHeads,
            localHeadDim: cfg.localHeadDim, globalHeadDim: cfg.globalHeadDim,
            slidingWindow: cfg.slidingWindow, vocabSize: cfg.vocabSize,
            expertCount: cfg.expertCount, expertUsedCount: cfg.topKExperts,
            expertIntermediate: cfg.expertIntermediate,
            sharedIntermediate: cfg.sharedIntermediate,
            finalSoftcap: cfg.finalSoftcap, globalLayers: cfg.globalLayers)
        canonicalSSA = try GemmaSSA.lower(GemmaGraphIR(
            schema: 1, model: "gemma4-26b-a4b", config: canonicalConfig,
            ops: GemmaGraph.expectedSequence(canonicalConfig)))
    } catch {
        failed += 1; print("FAIL: sc canonical setup - threw \(error)")
        return
    }

    // Combination 1: canonical, both axes supported.
    let both = scCheckCombination("sc canonical (both supported)", canonicalSSA,
                                  canonicalConfig, epsilon: epsA, cache: true,
                                  rms: true, filteredCount: 32)
    if let both {
        let plan = both.body.rmsPlan
        checkEqual("sc canonical: 30 cache proofs",
                   both.body.cacheReport.body.proofs.count, 30)
        checkEqual("sc canonical: 30 RMS groups", plan.groupCount, 30)
        checkEqual("sc canonical: 60 RMS reuses", plan.reuseCount, 60)
        let actions = plan.actions.compactMap { $0 }
        checkEqual("sc canonical: 30 compute actions",
                   actions.filter { $0.kind == .compute }.count, 30)
        checkEqual("sc canonical: 60 reuse actions",
                   actions.filter { $0.kind == .reuse }.count, 60)
        checkEqual("sc canonical: dense actions cover every instruction",
                   plan.actions.count, canonicalSSA.instructions.count)
        check("sc canonical: every group is one compute plus two reuses",
              plan.groups.allSatisfy { $0.useIndices.count == 2 })
        checkEqual("sc canonical: one group per layer, in order",
                   plan.groups.map { canonicalSSA.instructions[$0.firstInstructionIndex].layer },
                   (0..<30).map { Optional($0) })
        check("sc canonical: groups keyed on x at hidden width with the given epsilon",
              plan.groups.allSatisfy {
                  $0.inputID.hasPrefix("x@") && $0.width == canonicalConfig.hidden
                      && $0.vectors == 1 && $0.epsilonBits == epsA.bitPattern })
        checkEqual("sc canonical: plan epsilon bits", plan.epsilonBits, epsA.bitPattern)
        check("sc canonical: embedded cache report verifies",
              (try? both.body.cacheReport.verifyDigest()) == true)
        checkEqual("sc canonical: existing single-axis shadow helper unchanged",
                   try? GemmaOpportunityDiscovery.shadowCandidates(
                       ssa: canonicalSSA, config: canonicalConfig, hardware: hw),
                   manual)
    }

    // Combination 2: swapped first cache inputs (cache no, RMS yes).
    if let idx = canonicalSSA.instructions.firstIndex(where: { $0.op == "cache_append" }),
       let swapped = odMutated(canonicalSSA, { obj in
           var insts = obj["instructions"] as? [[String: Any]] ?? []
           insts[idx]["inputs"] = Array(canonicalSSA.instructions[idx].inputs.reversed())
           obj["instructions"] = insts
       }), swapped != canonicalSSA {
        if odValidates("sc swapped first cache inputs", swapped, canonicalConfig),
           let r = scCheckCombination("sc swapped (cache unsupported, RMS supported)",
                                      swapped, canonicalConfig, epsilon: epsA,
                                      cache: false, rms: true, filteredCount: 14) {
            checkEqual("sc swapped: existing cache reason codes",
                       r.body.cacheReport.body.reasons.map(\.code), [.operandBinding])
            checkEqual("sc swapped: remaining layers proven",
                       r.body.cacheReport.body.proofs.map(\.layer),
                       Array(1..<canonicalConfig.layerCount))
            checkEqual("sc swapped: RMS groups unaffected", r.body.rmsPlan.groupCount, 30)
            checkEqual("sc swapped: RMS reuses unaffected", r.body.rmsPlan.reuseCount, 60)
        }
    } else {
        failed += 1; print("FAIL: sc swapped fixture unrepresentable or unchanged")
    }

    // Combinations 3 and 4: synthetic schedules with no rmsnorm at all.
    let synthConfig = odSyntheticConfig()
    let local0 = ODAppendSpec(layer: 0, kvHeads: 2, headDim: 4, width: 8)
    let global1 = ODAppendSpec(layer: 1, kvHeads: 1, headDim: 8, width: 8)
    let base = odSyntheticSSA([local0, global1])
    if let base {
        if odValidates("sc cache-only fixture", base, synthConfig),
           let r = scCheckCombination("sc cache-only (RMS unsupported)", base, synthConfig,
                                      epsilon: epsA, cache: true, rms: false,
                                      filteredCount: 26) {
            checkEqual("sc cache-only: cache proofs present",
                       r.body.cacheReport.body.proofs.map(\.layer), [0, 1])
            checkEqual("sc cache-only: no RMS groups", r.body.rmsPlan.groupCount, 0)
            checkEqual("sc cache-only: no RMS reuses", r.body.rmsPlan.reuseCount, 0)
            checkEqual("sc cache-only: dense actions cover every instruction",
                       r.body.rmsPlan.actions.count, base.instructions.count)
            check("sc cache-only: every action nil",
                  r.body.rmsPlan.actions.allSatisfy { $0 == nil })
        }
    } else {
        failed += 1; print("FAIL: sc cache-only fixture unrepresentable")
    }
    if let zero = odSyntheticSSA([]) {
        if odValidates("sc zero-append fixture", zero, synthConfig),
           let r = scCheckCombination("sc zero-append (both unsupported)", zero, synthConfig,
                                      epsilon: epsA, cache: false, rms: false,
                                      filteredCount: 14) {
            checkEqual("sc zero-append: existing missing-coverage reasons",
                       r.body.cacheReport.body.reasons.map(\.code),
                       [.missingLayer, .missingLayer])
            checkEqual("sc zero-append: no cache proofs",
                       r.body.cacheReport.body.proofs.count, 0)
            checkEqual("sc zero-append: no RMS groups", r.body.rmsPlan.groupCount, 0)
        }
    } else {
        failed += 1; print("FAIL: sc zero-append fixture unrepresentable")
    }

    // Determinism, Codable, and digest tamper on the canonical report.
    if let both {
        do {
            let again = try GemmaShadowComposition.discover(
                ssa: canonicalSSA, config: canonicalConfig, hardware: hw, epsilon: epsA)
            checkEqual("sc repeated: identical report", again, both)
            checkEqual("sc repeated: identical digest", again.reportDigest, both.reportDigest)
            let encoded = try JSONEncoder().encode(both)
            let decoded = try JSONDecoder().decode(SCReport.self, from: encoded)
            checkEqual("sc codable: report roundtrip", decoded, both)
            checkEqual("sc codable: digest recomputed from body alone",
                       try SCReport.digest(of: decoded.body), both.reportDigest)
            let decodedVerified = try decoded.verifyDigest()
            check("sc codable: decoded digest verifies", decodedVerified)
            let control = try scTampered(both) { _ in }
            checkEqual("sc tamper control: untouched JSON roundtrip equal", control, both)
            let controlVerified = try control.verifyDigest()
            check("sc tamper control: untouched JSON roundtrip verifies", controlVerified)

            let firstCompute = both.body.rmsPlan.groups.first?.firstInstructionIndex ?? 0
            let groupTamper = try scTampered(both) { b in
                var plan = b["rmsPlan"] as? [String: Any] ?? [:]
                var groups = plan["groups"] as? [[String: Any]] ?? []
                if !groups.isEmpty { groups[0]["useIndices"] = [Int]() }
                plan["groups"] = groups
                b["rmsPlan"] = plan
            }
            scExpectTamperFails("sc tamper RMS group proof", both, groupTamper)
            let actionTamper = try scTampered(both) { b in
                var plan = b["rmsPlan"] as? [String: Any] ?? [:]
                var actions = plan["actions"] as? [Any] ?? []
                if firstCompute < actions.count { actions[firstCompute] = NSNull() }
                plan["actions"] = actions
                b["rmsPlan"] = plan
            }
            scExpectTamperFails("sc tamper RMS action", both, actionTamper)
            let epsilonTamper = try scTampered(both) { b in
                b["epsilonBits"] = NSNumber(value: epsB.bitPattern)
            }
            scExpectTamperFails("sc tamper epsilon bits", both, epsilonTamper)
            let verdictTamper = try scTampered(both) { b in
                var verdicts = b["verdicts"] as? [[String: Any]] ?? []
                if verdicts.count == 2 {
                    verdicts[1]["supported"] = false
                    verdicts[1]["reason"] = "noReusableGroups"
                }
                b["verdicts"] = verdicts
            }
            scExpectTamperFails("sc tamper verdict", both, verdictTamper)
            let cacheTamper = try scTampered(both) { b in
                var cache = b["cacheReport"] as? [String: Any] ?? [:]
                var cacheBody = cache["body"] as? [String: Any] ?? [:]
                var proofs = cacheBody["proofs"] as? [[String: Any]] ?? []
                if !proofs.isEmpty { proofs[0]["headDim"] = 7 }
                cacheBody["proofs"] = proofs
                cache["body"] = cacheBody
                b["cacheReport"] = cache
            }
            scExpectTamperFails("sc tamper cache proof", both, cacheTamper)
        } catch {
            failed += 1; print("FAIL: sc determinism/codable/tamper - threw \(error)")
        }

        // A different positive epsilon changes bits and digest only.
        do {
            let other = try GemmaShadowComposition.discover(
                ssa: canonicalSSA, config: canonicalConfig, hardware: hw, epsilon: epsB)
            check("sc epsilon variant: test epsilons differ in bits",
                  epsA.bitPattern != epsB.bitPattern)
            checkEqual("sc epsilon variant: bits recorded exactly",
                       other.body.epsilonBits, epsB.bitPattern)
            checkEqual("sc epsilon variant: plan bits recorded",
                       other.body.rmsPlan.epsilonBits, epsB.bitPattern)
            check("sc epsilon variant: digest changes", other.reportDigest != both.reportDigest)
            checkEqual("sc epsilon variant: verdicts stable",
                       other.body.verdicts, both.body.verdicts)
            checkEqual("sc epsilon variant: cache evidence stable",
                       other.body.cacheReport, both.body.cacheReport)
            checkEqual("sc epsilon variant: group count stable", other.body.rmsPlan.groupCount, 30)
            checkEqual("sc epsilon variant: reuse count stable", other.body.rmsPlan.reuseCount, 60)
            checkEqual("sc epsilon variant: group indices stable",
                       other.body.rmsPlan.groups.map(\.useIndices),
                       both.body.rmsPlan.groups.map(\.useIndices))
            checkEqual("sc epsilon variant: plan equals direct analysis", other.body.rmsPlan,
                       try RMSStatisticsPlan.analyze(ssa: canonicalSSA, epsilon: epsB))
            let otherShadow = try GemmaShadowComposition.shadowCandidates(
                ssa: canonicalSSA, config: canonicalConfig, hardware: hw, epsilon: epsB)
            checkEqual("sc epsilon variant: shadow list stable", otherShadow, manual)
        } catch {
            failed += 1; print("FAIL: sc epsilon variant - threw \(error)")
        }
    }

    // Invalid epsilon: typed error carrying the exact bits, from both entry points.
    let invalidEpsilons: [(String, Float)] = [
        ("NaN", Float.nan), ("+infinity", Float.infinity),
        ("-infinity", -Float.infinity), ("zero", Float(0)),
        ("negative zero", Float(-0.0)), ("negative", Float(-1e-6)),
    ]
    for (label, e) in invalidEpsilons {
        checkThrows("sc epsilon \(label) rejected (discover)",
                    GemmaShadowCompositionError.invalidEpsilon(e.bitPattern)) {
            _ = try GemmaShadowComposition.discover(
                ssa: canonicalSSA, config: canonicalConfig, hardware: hw, epsilon: e)
        }
        checkThrows("sc epsilon \(label) rejected (shadowCandidates)",
                    GemmaShadowCompositionError.invalidEpsilon(e.bitPattern)) {
            _ = try GemmaShadowComposition.shadowCandidates(
                ssa: canonicalSSA, config: canonicalConfig, hardware: hw, epsilon: e)
        }
    }

    // Errors from the wrapped analyses propagate unchanged.
    if let stale = odMutated(canonicalSSA, { obj in
        obj["instructions"] = (obj["instructions"] as? [[String: Any]] ?? []).map { inst in
            var i = inst
            if (i["op"] as? String) == "rmsnorm", var ins = i["inputs"] as? [String],
               ins.first?.hasPrefix("x@") == true {
                ins[0] = "x@0"; i["inputs"] = ins
            }
            return i
        }
    }) {
        check("sc stale: fixture rejected by GemmaSSA.validate",
              (try? GemmaSSA.validate(stale, config: canonicalConfig)) == nil)
        odExpectThrows("sc stale reference propagates malformedSSA (discover)",
                       odIsSSAError) {
            _ = try GemmaShadowComposition.discover(
                ssa: stale, config: canonicalConfig, hardware: hw, epsilon: epsA)
        }
        odExpectThrows("sc stale reference propagates malformedSSA (shadowCandidates)",
                       odIsSSAError) {
            _ = try GemmaShadowComposition.shadowCandidates(
                ssa: stale, config: canonicalConfig, hardware: hw, epsilon: epsA)
        }
        checkThrows("sc epsilon validated before any analysis",
                    GemmaShadowCompositionError.invalidEpsilon(Float.infinity.bitPattern)) {
            _ = try GemmaShadowComposition.discover(
                ssa: stale, config: canonicalConfig, hardware: hw, epsilon: .infinity)
        }
    } else {
        failed += 1; print("FAIL: sc stale fixture unrepresentable")
    }
    if let base {
        if let unknown = odMutated(base, { obj in
            var insts = obj["instructions"] as? [[String: Any]] ?? []
            insts[1]["op"] = "conv2d"
            obj["instructions"] = insts
        }) {
            odExpectThrows("sc unknown op propagates malformedSSA", odIsSSAError) {
                _ = try GemmaShadowComposition.discover(
                    ssa: unknown, config: synthConfig, hardware: hw, epsilon: epsA)
            }
        } else {
            failed += 1; print("FAIL: sc unknown-op fixture unrepresentable")
        }
        odExpectThrows("sc invalid config propagates invalidConfig", odIsConfigError) {
            _ = try GemmaShadowComposition.discover(
                ssa: base, config: odSyntheticConfig(hidden: Int.max / 4),
                hardware: hw, epsilon: epsA)
        }
        odExpectThrows("sc invalid profile propagates invalidHardware", odIsHardwareError) {
            _ = try GemmaShadowComposition.shadowCandidates(
                ssa: base, config: synthConfig,
                hardware: ForgeHardwareProfile(
                    deviceName: "", maxTotalThreadsPerThreadgroup: 1024,
                    threadExecutionWidth: 32, recommendedMaxWorkingSetSize: 1 << 30),
                epsilon: epsA)
        }
    }
}

runShadowCompositionGates()

// MARK: - Model-file digest policy (cached, explicit refresh)

@MainActor
func runModelFileVerificationPolicyGates() {
    func expectInvalidPlan(_ name: String, _ body: () throws -> Void) {
        do {
            try body()
            failed += 1; print("FAIL: " + name + " — expected invalidPlan, got no error")
        } catch let error as GemmaForgeError {
            if case .invalidPlan = error {
                passed += 1
            } else {
                failed += 1
                print("FAIL: " + name + " — expected invalidPlan, got " + String(describing: error))
            }
        } catch {
            failed += 1
            print("FAIL: " + name + " — unexpected " + String(describing: error))
        }
    }
    func lowerHex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    // Policy serialization: the only representable value is the cached
    // explicit-refresh policy; the retired intermediate unverified value
    // and a verified-sounding value do not decode.
    do {
        let encoded = try JSONEncoder().encode([ModelFileIdentity.policy])
        let raw = try JSONSerialization.jsonObject(with: encoded) as? [String]
        checkEqual("mfv policy serializes as the cached explicit-refresh value",
                   raw ?? [], ["cached-explicit-refresh-v1"])
        let decoded = try JSONDecoder().decode([ModelFileVerificationPolicy].self,
                                               from: encoded)
        checkEqual("mfv policy round-trips", decoded, [ModelFileIdentity.policy])
        for retired in ["sha256-full-file", "none-unverified-v1"] {
            let retiredData = try JSONSerialization.data(withJSONObject: [retired])
            let retiredDecoded = try? JSONDecoder().decode([ModelFileVerificationPolicy].self,
                                                           from: retiredData)
            check("mfv retired policy value does not decode: " + retired, retiredDecoded == nil)
        }
    } catch {
        failed += 1; print("FAIL: mfv policy serialization — threw " + String(describing: error))
    }
    checkEqual("mfv current policy is cached explicit refresh",
               ModelFileIdentity.policy, .cachedExplicitRefresh)

    // The general SHA-256 utility is unchanged (small temp fixtures only).
    do {
        let dir = FileManager.default.temporaryDirectory
        let abcURL = dir.appendingPathComponent("leviathan-mfv-" + UUID().uuidString + ".bin")
        let emptyURL = dir.appendingPathComponent("leviathan-mfv-" + UUID().uuidString + ".bin")
        defer {
            try? FileManager.default.removeItem(at: abcURL)
            try? FileManager.default.removeItem(at: emptyURL)
        }
        try Data("abc".utf8).write(to: abcURL)
        try Data().write(to: emptyURL)
        let abcDigest = try GemmaForge.fileDigest(abcURL)
        let emptyDigest = try GemmaForge.fileDigest(emptyURL)
        checkEqual("mfv generic fileDigest unchanged (SHA-256 of abc)", abcDigest,
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        checkEqual("mfv generic fileDigest unchanged (SHA-256 of empty)", emptyDigest,
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    } catch {
        failed += 1; print("FAIL: mfv generic fileDigest — threw " + String(describing: error))
    }

    // Schema/identity versions bumped for every model-bound contract.
    checkEqual("mfv gemma plan schema bumped", GemmaForge.schemaVersion, 3)
    checkEqual("mfv q4 launch artifact schema bumped", Q4LaunchArtifact.schema, 3)
    checkEqual("mfv q4 lowering artifact schema bumped", Q4LoweringArtifact.schema, 3)
    checkEqual("mfv rms lowering artifact schema bumped", RMSLoweringArtifact.schema, 3)
    checkEqual("mfv moe lowering artifact schema bumped", MoELoweringArtifact.schema, 3)
    checkEqual("mfv moe profile schema bumped", MoEProfileReport.schema, 3)

    // Plan contract: pre-policy and verified-claiming plans are rejected;
    // the policy is inside the planDigest-covered encoding.
    do {
        let h64 = String(repeating: "a", count: 64)
        let hw = ForgeHardwareProfile(deviceName: "mfv-synthetic",
                                      maxTotalThreadsPerThreadgroup: 1024,
                                      threadExecutionWidth: 32,
                                      recommendedMaxWorkingSetSize: 1 << 30)
        let hwObj = try JSONSerialization.jsonObject(with: JSONEncoder().encode(hw))
        let candObj = try JSONSerialization.jsonObject(with: JSONEncoder().encode(
            GemmaForgeCandidate(segmentBatching: false, top5Diagnostic: false)))
        let prePolicy: [String: Any] = [
            "schemaVersion": 1, "graphDigest": h64, "ggufDigest": h64,
            "engineDigest": h64, "hardware": hwObj, "toolchain": "mfv",
            "workloadDigest": h64, "promptTokenCount": 2, "decodeTokens": 4,
            "selected": candObj, "selectedMedianWallMicrosPerToken": 1.5,
            "selectedMedianGpuMicrosPerToken": 1.5, "results": [Any](),
            "scheduleDigest": h64, "planDigest": h64,
        ]
        func parse(_ obj: [String: Any]) throws -> GemmaForgePlan {
            try GemmaForge.parsePlan(JSONSerialization.data(withJSONObject: obj))
        }
        expectInvalidPlan("mfv pre-policy plan (old schema, no policy field) is rejected") {
            _ = try parse(prePolicy)
        }
        var claimed = prePolicy
        claimed["modelFileVerification"] = "sha256-full-file"
        expectInvalidPlan("mfv plan claiming verified model content is rejected") {
            _ = try parse(claimed)
        }
        var intermediate = prePolicy
        intermediate["schemaVersion"] = 2
        intermediate["modelFileVerification"] = "none-unverified-v1"
        expectInvalidPlan("mfv intermediate unverified plan is rejected, not re-tagged") {
            _ = try parse(intermediate)
        }
        var explicit = prePolicy
        explicit["modelFileVerification"] = ModelFileIdentity.policy.rawValue
        let plan = try parse(explicit)
        checkEqual("mfv decoded plan carries the cached explicit-refresh policy",
                   plan.modelFileVerification, ModelFileIdentity.policy)
        check("mfv legacy schema 1 is not current (identity gate rejects it)",
              plan.schemaVersion != GemmaForge.schemaVersion)
        let resigned = try GemmaForge.resignPlan(plan)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let signedBytes = try encoder.encode(resigned)
        var signedObj = (try JSONSerialization.jsonObject(with: signedBytes)
            as? [String: Any]) ?? [:]
        checkEqual("mfv policy is inside the planDigest-covered encoding",
                   signedObj["modelFileVerification"] as? String ?? "",
                   "cached-explicit-refresh-v1")
        let reResigned = try GemmaForge.resignPlan(resigned)
        checkEqual("mfv resign is deterministic over the policy-bearing body",
                   reResigned.planDigest, resigned.planDigest)
        signedObj.removeValue(forKey: "modelFileVerification")
        let strippedObj = signedObj
        expectInvalidPlan("mfv stripping the policy from a signed plan is rejected") {
            _ = try parse(strippedObj)
        }
    } catch {
        failed += 1; print("FAIL: mfv plan policy contract — threw " + String(describing: error))
    }

    // No numerical-gate or policy relaxation came with this change.
    checkEqual("mfv 2% selection policy unchanged", GemmaForge.noiseRelWall, 0.02)
    checkEqual("mfv q4 launch production gate length unchanged",
               Q4LaunchGatePolicy.productionGenTokens, 32)
    checkEqual("mfv q4 launch production timed tokens unchanged",
               Q4LaunchGatePolicy.productionTimedTokens, 31)
    checkEqual("mfv q4 launch production fixtures unchanged",
               Q4LaunchGatePolicy.productionFixtures, ["en-chat", "kr-chat", "raw-hello"])
    checkEqual("mfv moe production gate length unchanged",
               MoEGatePolicy.productionGenTokens, 32)
    checkEqual("mfv moe production timed tokens unchanged",
               MoEGatePolicy.productionTimedTokens, 31)
}

runModelFileVerificationPolicyGates()

// MARK: - Shared model digest cache (hash once, reuse, explicit refresh)

@MainActor
func runModelDigestCacheGates() {
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    let abcd = "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"
    func lowerHex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
    func expectMalformed(_ name: String, _ body: () throws -> Void) {
        do {
            try body()
            failed += 1; print("FAIL: " + name + " — expected malformedRecord, got no error")
        } catch let error as ModelDigestCacheError {
            if case .malformedRecord(let message) = error, message.contains("--rehash-model") {
                passed += 1
            } else {
                failed += 1
                print("FAIL: " + name + " — expected malformedRecord with a refresh instruction, got "
                      + String(describing: error))
            }
        } catch {
            failed += 1
            print("FAIL: " + name + " — unexpected " + String(describing: error))
        }
    }
    let fm = FileManager.default
    let root = fm.temporaryDirectory.appendingPathComponent(
        "leviathan-mdc-" + UUID().uuidString, isDirectory: true)
    defer { try? fm.removeItem(at: root) }
    do {
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let model = root.appendingPathComponent("model.bin")
        let link = root.appendingPathComponent("model-link.bin")
        let dirA = root.appendingPathComponent("cache-a", isDirectory: true)
        let dirB = root.appendingPathComponent("cache-b", isDirectory: true)
        try Data("abc".utf8).write(to: model)
        try fm.createSymbolicLink(at: link, withDestinationURL: model)

        // Initial miss: the actual bytes are hashed once and persisted.
        let first = try ModelDigestCache(directory: dirA).digest(for: model)
        checkEqual("mdc initial miss hashes the actual bytes", first.sha256, abc)
        check("mdc initial miss is reported as an initial computation",
              first.origin == .initialComputation)
        checkEqual("mdc record stores the bytes hashed", first.record.bytes, 3)
        checkEqual("mdc record source is initial", first.record.source, "initial")
        checkEqual("mdc record carries the cached policy", first.record.policy, ModelFileIdentity.policy)

        // Shared format and key algorithm (tools/baseline/model_digest_cache.py).
        let canonical = ModelDigestCache.canonicalPath(for: model)
        let key = ModelDigestCache.cacheKey(for: model)
        checkEqual("mdc key is SHA-256 of the canonical path", key,
                   lowerHex(Array(SHA256.hash(Array(canonical.utf8)))))
        let recordURL = ModelDigestCache(directory: dirA).recordURL(for: model)
        checkEqual("mdc record file is <key>.json", recordURL.lastPathComponent, key + ".json")
        let stored = (try JSONSerialization.jsonObject(with: Data(contentsOf: recordURL))
            as? [String: Any]) ?? [:]
        checkEqual("mdc record keys", Set(stored.keys),
                   ["bytes", "format", "hashedAtUTC", "path", "policy", "sha256", "source"])
        checkEqual("mdc record format tag", stored["format"] as? String ?? "",
                   "leviathan.model-digest-cache.v1")
        checkEqual("mdc record path is canonical", stored["path"] as? String ?? "", canonical)
        checkEqual("mdc symlink resolves to the same key",
                   ModelDigestCache.cacheKey(for: link), key)

        // Mutation is NOT detected automatically: no stat/mtime/size check
        // and no rehash; new readers get the historical digest.
        try Data("abcd".utf8).write(to: model)
        let hit = try ModelDigestCache(directory: dirA).digest(for: model)
        check("mdc mutated file still hits the cache (no automatic rehash)",
              hit.origin == .cacheHit && hit.sha256 == abc)
        checkEqual("mdc hit keeps the historical timestamp",
                   hit.record.hashedAtUTC, first.record.hashedAtUTC)
        check("mdc hit disclosure is not a fresh verification",
              hit.disclosure.contains("NOT re-read"))
        let viaLink = try ModelDigestCache(directory: dirA).digest(for: link)
        check("mdc symlink lookup hits the same record",
              viaLink.origin == .cacheHit && viaLink.sha256 == abc)
        var nestedAllHits = true
        for _ in 0..<3 {
            let nested = try ModelDigestCache(directory: dirA).digest(for: model)
            nestedAllHits = nestedAllHits && nested.origin == .cacheHit
        }
        check("mdc nested lookups never recompute", nestedAllHits)

        // Explicit refresh re-hashes and persists for new readers.
        let refreshed = try ModelDigestCache(directory: dirA).digest(for: model, refresh: true)
        check("mdc explicit refresh hashes the new bytes",
              refreshed.origin == .explicitRefresh && refreshed.sha256 == abcd)
        checkEqual("mdc refreshed record source", refreshed.record.source, "explicit-refresh")
        let reloaded = try ModelDigestCache(directory: dirA).cachedRecord(for: model)
        checkEqual("mdc refreshed record persists for new readers", reloaded?.sha256 ?? "", abcd)

        // Custom directories are isolated.
        let cacheB = ModelDigestCache(directory: dirB)
        let isolated = try cacheB.cachedRecord(for: model)
        check("mdc custom directory is isolated", isolated == nil)

        // Malformed, retired-policy and mismatched records fail with an
        // explicit refresh instruction; they are never silently rehashed.
        try fm.createDirectory(at: dirB, withIntermediateDirectories: true)
        try Data("{not json".utf8).write(to: cacheB.recordURL(for: model))
        expectMalformed("mdc malformed record requires explicit refresh") {
            _ = try cacheB.digest(for: model)
        }
        expectMalformed("mdc malformed record is not served cache-only") {
            _ = try cacheB.cachedRecord(for: model)
        }
        var retired = stored
        retired["policy"] = "none-unverified-v1"
        try JSONSerialization.data(withJSONObject: retired).write(to: cacheB.recordURL(for: model))
        expectMalformed("mdc retired unverified policy record is rejected") {
            _ = try cacheB.digest(for: model)
        }
        var moved = stored
        moved["path"] = canonical + ".other"
        try JSONSerialization.data(withJSONObject: moved).write(to: cacheB.recordURL(for: model))
        expectMalformed("mdc record for another path is rejected") {
            _ = try cacheB.digest(for: model)
        }
        let repaired = try cacheB.digest(for: model, refresh: true)
        check("mdc explicit refresh repairs a malformed record",
              repaired.origin == .explicitRefresh && repaired.sha256 == abcd)

        // A deleted model still resolves from the cache: nothing is read.
        try fm.removeItem(at: link)
        try fm.removeItem(at: model)
        let afterDelete = try ModelDigestCache(directory: dirA).digest(for: model)
        check("mdc deleted model resolves from the cache without reading",
              afterDelete.origin == .cacheHit && afterDelete.sha256 == abcd)
        let uncached = try ModelDigestCache(
            directory: root.appendingPathComponent("cache-c", isDirectory: true))
            .cachedRecord(for: model)
        check("mdc cache-only lookup of an uncached model reads nothing", uncached == nil)

        // Unicode spellings share one NFC record: realpath(3) reports the
        // on-disk spelling while the file exists, and the fallback keeps the
        // caller's spelling after deletion; both must key identically and
        // match the Python tools.
        let accented = "mod\u{E8}le.bin"
        let nfcModel = root.appendingPathComponent(accented.precomposedStringWithCanonicalMapping)
        let nfdModel = root.appendingPathComponent(accented.decomposedStringWithCanonicalMapping)
        let dirU = root.appendingPathComponent("cache-u", isDirectory: true)
        try Data("abc".utf8).write(to: nfcModel)
        let unicodeCanonical = ModelDigestCache.canonicalPath(for: nfdModel)
        check("mdc existing Unicode canonical path is NFC",
              Array(unicodeCanonical.utf8)
                == Array(unicodeCanonical.precomposedStringWithCanonicalMapping.utf8))
        let unicodeKey = lowerHex(Array(SHA256.hash(Array(unicodeCanonical.utf8))))
        checkEqual("mdc existing NFC spelling key", ModelDigestCache.cacheKey(for: nfcModel), unicodeKey)
        checkEqual("mdc existing NFD spelling key", ModelDigestCache.cacheKey(for: nfdModel), unicodeKey)
        let unicodeFirst = try ModelDigestCache(directory: dirU).digest(for: nfcModel)
        check("mdc Unicode initial miss hashes the actual bytes",
              unicodeFirst.origin == .initialComputation && unicodeFirst.sha256 == abc)
        check("mdc Unicode record path is stored as NFC bytes",
              Array(unicodeFirst.record.path.utf8) == Array(unicodeCanonical.utf8))
        try Data("abcd".utf8).write(to: nfcModel)
        let unicodeHit = try ModelDigestCache(directory: dirU).digest(for: nfdModel)
        check("mdc mutated Unicode model hits the historical record",
              unicodeHit.origin == .cacheHit && unicodeHit.sha256 == abc)
        try fm.removeItem(at: nfcModel)
        for (label, spelling) in [("NFC", nfcModel), ("NFD", nfdModel)] {
            checkEqual("mdc deleted Unicode model " + label + " spelling keeps the key",
                       ModelDigestCache.cacheKey(for: spelling), unicodeKey)
            let gone = try ModelDigestCache(directory: dirU).digest(for: spelling)
            check("mdc deleted Unicode model " + label + " spelling hits without reading",
                  gone.origin == .cacheHit && gone.sha256 == abc)
        }

        // A record written by tools/baseline/model_digest_cache.py
        // (json.dump: ASCII escapes, indent 2) is found after deletion.
        let escapedPath = unicodeCanonical.unicodeScalars.map {
            $0.isASCII ? String($0) : String(format: "\\u%04x", $0.value)
        }.joined()
        let pythonRecord = "{\n  \"bytes\": 4,\n  \"format\": \"leviathan.model-digest-cache.v1\",\n"
            + "  \"hashedAtUTC\": \"2026-10-04T00:00:00Z\",\n  \"path\": \"" + escapedPath + "\",\n"
            + "  \"policy\": \"cached-explicit-refresh-v1\",\n  \"sha256\": \"" + abcd + "\",\n"
            + "  \"source\": \"initial\"\n}"
        let dirP = root.appendingPathComponent("cache-p", isDirectory: true)
        try fm.createDirectory(at: dirP, withIntermediateDirectories: true)
        try Data(pythonRecord.utf8).write(to: dirP.appendingPathComponent(unicodeKey + ".json"))
        let fromPython = try ModelDigestCache(directory: dirP).digest(for: nfdModel)
        check("mdc Python-written Unicode record hits after deletion",
              fromPython.origin == .cacheHit && fromPython.sha256 == abcd)
    } catch {
        failed += 1; print("FAIL: mdc digest cache — threw " + String(describing: error))
    }
}

runModelDigestCacheGates()

// MARK: - command-buffer timeline diagnostics

@MainActor
func runCommandTimelineGates() {
    typealias CT = Gemma4CommandTimeline
    func ev(_ seq: Int, encode: Double? = nil, bc: Double, ac: Double, aw: Double,
            gs: Double, ge: Double, completed: Bool = true) -> CT.Event {
        CT.Event(sequence: seq, site: .directDispatch, kernelCount: 1, encodeSeconds: encode,
                 hostBeforeCommit: bc, hostAfterCommit: ac, hostAfterWait: aw,
                 gpuStart: gs, gpuEnd: ge, completed: completed, statusRaw: completed ? 4 : 5)
    }
    func analyze(_ events: [CT.Event], dropped: Int = 0) -> CT.Summary {
        CT.analyze(CT.Trace(capacity: CT.maxCapacity, droppedCount: dropped, events: events))
    }
    func reasons(_ events: [CT.Event]) -> [CT.RejectionReason] {
        analyze(events).rejections.map(\.reason)
    }

    // Known synthetic timeline (binary-exact values, so totals compare exactly).
    let e1 = ev(1, encode: 0.5, bc: 10.0, ac: 10.25, aw: 13.0, gs: 11.0, ge: 12.5)
    let e2 = ev(2, bc: 14.0, ac: 14.5, aw: 17.0, gs: 15.0, ge: 16.0)
    let e3 = ev(3, bc: 18.0, ac: 18.5, aw: 21.0, gs: 19.0, ge: 20.0)
    let good = analyze([e1, e2])
    checkEqual("ct valid events accepted", good.validEventCount, 2)
    check("ct valid events have no rejections", good.rejections.isEmpty)
    check("ct encode total sums known encodes only",
          good.encodeSeconds == 0.5 && good.encodeUnknownCount == 1)
    check("ct commit-call total", good.commitCallSeconds == 0.75)
    check("ct wait-call total", good.waitCallSeconds == 5.25)
    check("ct GPU interval total", good.gpuSeconds == 2.5)
    check("ct commit-to-GPU-start total", good.commitToGpuStartSeconds == 2.0)
    check("ct GPU-end-to-wait-return total", good.gpuEndToWaitReturnSeconds == 1.5)
    check("ct host span", good.hostSpanSeconds == 7.0)
    check("ct previous GPU end to next commit gap",
          good.gaps == [CT.Gap(fromSequence: 1, toSequence: 2,
                               gpuEndToNextCommitSeconds: 1.5,
                               gpuEndToWaitReturnSeconds: 0.5,
                               waitReturnToNextCommitSeconds: 1.0)])
    check("ct gap totals", good.gapTotalSeconds == 1.5 && good.gapMaxSeconds == 1.5
          && good.gapPairsSkipped == 0)
    check("ct complete trace is not truncated",
          !good.truncated && good.droppedCount == 0 && good.missingSequenceCount == 0)

    // Invalid timestamps are rejected, never clamped.
    check("ct zero GPU start rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 0, ge: 12.5)]) == [.nonPositiveTimestamp])
    check("ct zero GPU end rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: 0)]) == [.nonPositiveTimestamp])
    check("ct non-finite GPU end rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: .nan)]) == [.nonFinite])
    check("ct infinite host timestamp rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: .infinity, gs: 11, ge: 12.5)]) == [.nonFinite])
    check("ct non-finite encode rejected",
          reasons([ev(1, encode: .nan, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: 12.5)]) == [.nonFinite])
    check("ct GPU start before host commit rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 9.5, ge: 12.5)]) == [.gpuOutsideHostBounds])
    check("ct GPU end after host wait rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: 13.5)]) == [.gpuOutsideHostBounds])
    check("ct reversed GPU interval rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 12.5, ge: 11)]) == [.gpuReversed])
    check("ct reversed host order rejected",
          reasons([ev(1, bc: 10, ac: 9.75, aw: 13, gs: 11, ge: 12.5)]) == [.hostOrderReversed])
    check("ct negative encode rejected",
          reasons([ev(1, encode: -0.25, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: 12.5)]) == [.negativeEncode])
    check("ct incomplete command buffer rejected",
          reasons([ev(1, bc: 10, ac: 10.25, aw: 13, gs: 11, ge: 12.5, completed: false)]) == [.notCompleted])
    check("ct overlapping history rejected",
          reasons([e1, ev(2, bc: 12.75, ac: 13.0, aw: 15.0, gs: 13.5, ge: 14.5)]) == [.overlapsPrevious])
    check("ct repeated sequence rejected", reasons([e1, e1]) == [.sequenceNotIncreasing])

    // Missing / rejected neighbors are never bridged by a derived gap.
    let acrossMissing = analyze([e1, ev(3, bc: 14.0, ac: 14.5, aw: 17.0, gs: 15.0, ge: 16.0)])
    check("ct missing sequence is counted and not bridged by a gap",
          acrossMissing.missingSequenceCount == 1 && acrossMissing.gaps.isEmpty
          && acrossMissing.gapPairsSkipped == 1 && acrossMissing.validEventCount == 2)
    let acrossRejected = analyze([e1, ev(2, bc: 14.0, ac: 14.5, aw: 17.0, gs: 0, ge: 16.0), e3])
    check("ct rejected neighbor is not bridged by a gap",
          acrossRejected.validEventCount == 2 && acrossRejected.gaps.isEmpty
          && acrossRejected.gapPairsSkipped == 1 && acrossRejected.missingSequenceCount == 0)
    checkEqual("ct leading missing sequence is counted", analyze([e2]).missingSequenceCount, 1)
    check("ct dropped events mark truncation", analyze([e1, e2], dropped: 3).truncated)

    // Bounded recorder: keeps the first `capacity` events, counts the rest.
    do {
        var recorder = try CT.Recorder(capacity: 2)
        for i in 0..<5 {
            let t = Double(i) * 4 + 1
            recorder.record(site: .deferredFlush, kernelCount: i, encodeSeconds: nil,
                            hostBeforeCommit: t, hostAfterCommit: t + 0.25, hostAfterWait: t + 2,
                            gpuStart: t + 0.5, gpuEnd: t + 1.5, completed: true, statusRaw: 4)
        }
        let trace = recorder.trace()
        check("ct recorder keeps at most capacity events",
              trace.events.map(\.sequence) == [1, 2])
        check("ct recorder counts dropped events", trace.droppedCount == 3 && trace.truncated)
        let summary = CT.analyze(trace)
        check("ct summary reports truncation",
              summary.truncated && summary.droppedCount == 3
              && summary.validEventCount == 2 && summary.gaps.count == 1)
    } catch {
        failed += 1; print("FAIL: ct recorder — threw " + String(describing: error))
    }
    for bad in [0, -1, CT.maxCapacity + 1] {
        do {
            _ = try CT.Recorder(capacity: bad)
            failed += 1; print("FAIL: ct recorder capacity \(bad) accepted")
        } catch let error as Gemma4CommandTimelineError {
            check("ct recorder capacity \(bad) rejected", error == .invalidCapacity(bad))
        } catch {
            failed += 1; print("FAIL: ct recorder capacity — unexpected " + String(describing: error))
        }
    }
    checkEqual("ct default recorder capacity", CT.Recorder().capacity, CT.maxCapacity)

    // Codable roundtrips.
    do {
        let trace = CT.Trace(capacity: 8, droppedCount: 1, events: [e1, e2])
        let decodedTrace = try JSONDecoder().decode(CT.Trace.self,
                                                    from: JSONEncoder().encode(trace))
        check("ct trace Codable roundtrip", decodedTrace == trace)
        let summary = analyze([e1, ev(2, bc: 14.0, ac: 14.5, aw: 17.0, gs: 0, ge: 16.0), e3])
        let decodedSummary = try JSONDecoder().decode(CT.Summary.self,
                                                      from: JSONEncoder().encode(summary))
        check("ct summary Codable roundtrip", decodedSummary == summary)
    } catch {
        failed += 1; print("FAIL: ct Codable roundtrip — threw " + String(describing: error))
    }

    // Real device: GPU timestamps must validate inside their host bounds.
    do {
        let trace = try CT.deviceProbe(commandBuffers: 3)
        let summary = CT.analyze(trace)
        check("ct device probe records every command buffer",
              trace.events.count == 3 && !trace.truncated)
        let valid = summary.validEventCount == 3 && summary.rejections.isEmpty
        check("ct device probe GPU timestamps validate inside host bounds", valid)
        if !valid {
            print("  ct device probe rejections: " + String(describing: summary.rejections))
        }
        check("ct device probe derives non-negative adjacent gaps",
              summary.gaps.count == 2
              && summary.gaps.allSatisfy { $0.gpuEndToNextCommitSeconds >= 0 })
    } catch Gemma4Error.metalUnavailable {
        print("SKIP: ct device probe — Metal unavailable")
    } catch {
        failed += 1; print("FAIL: ct device probe — threw " + String(describing: error))
    }
}

runCommandTimelineGates()

// MARK: - MoE SIMD lowering (correctness prototype)

/// Deterministic test-local generator (independent of the core inputs).
struct MoESIMDTestRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
    mutating func unit() -> Double { Double(next() >> 11) / 9007199254740992.0 }
}

/// Synthetic Q4_0 rows with finite FP16 scales in [0.002, 0.03].
func moesimdSyntheticBlocks(rows: Int, inDim: Int, rng: inout MoESIMDTestRNG) -> [UInt8] {
    var out = [UInt8]()
    out.reserveCapacity(rows * (inDim / 32) * 18)
    for _ in 0..<(rows * (inDim / 32)) {
        let bits = Float16(Float(0.002 + 0.028 * rng.unit())).bitPattern
        out.append(UInt8(bits & 0xFF))
        out.append(UInt8(bits >> 8))
        for _ in 0..<16 { out.append(UInt8(truncatingIfNeeded: rng.next() >> 32)) }
    }
    return out
}

func moesimdProfile(width: Int = 32, maxThreads: Int = 1024, tgMemory: Int = 32768,
                    staticMemory: Int = 0) -> MoESIMDHardwareProfile {
    MoESIMDHardwareProfile(deviceName: "synthetic", threadExecutionWidth: width,
                           maxTotalThreadsPerThreadgroup: maxThreads,
                           maxThreadgroupMemoryLength: tgMemory,
                           staticThreadgroupMemoryLength: staticMemory,
                           maxBufferLength: 1 << 34, recommendedMaxWorkingSetSize: 1 << 34)
}

/// Records real-GGUF probe calls (class box: mutated from stored closures).
final class MoESIMDProbeLog {
    var calls = [String]()
    var allFiniteHidden = true
    var stages = Set<String>()
}

@MainActor
func runMoESIMDPureGates() {
    print("— MoE SIMD lowering correctness-prototype gates (pure)")
    checkEqual("moesimd: v1 MoE codegen version untouched", MoEKernelEmitter.codegenVersion, "moegen-v1")
    checkEqual("moesimd: v1 MoE artifact schema untouched", MoELoweringArtifact.schema, 3)
    checkEqual("moesimd: prototype codegen version", MoESIMDKernelEmitter.codegenVersion, "moesimd-v1")

    let fixtureDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("tools/moe_lower/fixtures")
    do {
        let region = try MoERegionIR.parse(try Data(contentsOf:
            fixtureDir.appendingPathComponent("gemma4-26b-region.json")))
        let facts = try MoESIMDShapeFacts(region: region)
        checkEqual("moesimd: facts from the validated region",
                   [facts.hidden, facts.inter, facts.topK], [2816, 704, 8])
        let wide = moesimdProfile()
        let d = try MoESIMDScheduleDeriver.derive(region: region, hardware: wide)
        checkEqual("moesimd: 12 legal fused GU subschedules", d.guLegal.count, 12)
        checkEqual("moesimd: 12 legal down subschedules", d.dnLegal.count, 12)
        checkEqual("moesimd: bounded same-L catalog is 48", d.schedules.count, 48)
        check("moesimd: catalog within its bound", d.schedules.count <= MoESIMDScheduleDeriver.maxCatalog)
        check("moesimd: catalog entries unique", Set(d.schedules).count == d.schedules.count)
        check("moesimd: RT derived as 128 or 256 over L", d.schedules.allSatisfy { s in
            [128, 256].contains(s.gu.lanes * s.gu.rowsPerGroup)
                && [128, 256].contains(s.dn.lanes * s.dn.rowsPerGroup) })
        check("moesimd: RT divides 704 GU row pairs and 2816 down rows", d.schedules.allSatisfy {
            704 % $0.gu.rowsPerGroup == 0 && 2816 % $0.dn.rowsPerGroup == 0 })
        checkEqual("moesimd: L32 admits RT4 and RT8",
                   Set(d.guLegal.filter { $0.lanes == 32 }.map { $0.rowsPerGroup }), Set([4, 8]))
        check("moesimd: composition pairs equal L and accumulation", d.schedules.allSatisfy {
            $0.gu.lanes == $0.dn.lanes && $0.gu.accumulation == $0.dn.accumulation
                && $0.gu.family == .fusedGateUpAct && $0.dn.family == .down })
        checkEqual("moesimd: E4 and E8 chunk schedules", Set(d.schedules.map { $0.batch }), Set([4, 8]))

        let narrow = try MoESIMDScheduleDeriver.derive(region: region, hardware: moesimdProfile(width: 16))
        check("moesimd: execution width 16 rejects every subschedule",
              narrow.schedules.isEmpty && narrow.guLegal.isEmpty && narrow.dnLegal.isEmpty
              && narrow.rejections.count == 24
              && narrow.rejections.allSatisfy { $0.reason.contains("execution width") })
        let tiny = try MoESIMDScheduleDeriver.derive(region: region, hardware: moesimdProfile(tgMemory: 4096))
        check("moesimd: 4 KiB threadgroup memory rejects GU staging of hidden*4",
              tiny.guLegal.isEmpty && tiny.rejections.contains {
                  $0.candidate.hasPrefix("moesimd_gu") && $0.reason.contains("threadgroup memory") })
        checkEqual("moesimd: 4 KiB keeps inter*4 down staging legal", tiny.dnLegal.count, 12)
        check("moesimd: no composite without a legal GU", tiny.schedules.isEmpty)
        let withStatic = try MoESIMDScheduleDeriver.derive(
            region: region, hardware: moesimdProfile(tgMemory: 12000, staticMemory: 1000))
        check("moesimd: static threadgroup memory counts against the cap",
              withStatic.guLegal.isEmpty && withStatic.dnLegal.count == 12)
        let capped = try MoESIMDScheduleDeriver.derive(region: region, hardware: moesimdProfile(maxThreads: 128))
        check("moesimd: pipeline thread cap 128 keeps only T=128",
              capped.guLegal.count == 6 && capped.guLegal.allSatisfy { $0.lanes * $0.rowsPerGroup == 128 })

        let report = try MoESIMDComponentReport.make(
            region: region, hardware: wide, derivation: d, results: [], incumbent: [],
            inputs: [MoESIMDInputRecord(name: "synthetic", elementCount: 2816, provenance: "test")],
            expertIds: Array(0..<8), modelSha256: String(repeating: "a", count: 64),
            executableDigest: String(repeating: "b", count: 64))
        check("moesimd report: explicitly non-installable",
              !report.installable && !report.productionArtifact && !report.winnerSelected
              && report.diagnosticTimingOnly)
        checkEqual("moesimd report: numerical class", report.numericalClass, "simd-reassociated-oracle")
        let data = try report.encoded()
        let decoded = try MoESIMDComponentReport.decodeValidated(data)
        check("moesimd report: Codable roundtrip", decoded == report)
        let tampers: [(String, Any)] = [("installable", true), ("productionArtifact", true),
                                        ("winnerSelected", true), ("numericalClass", "bit-identical")]
        for (key, value) in tampers {
            do {
                var obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                obj[key] = value
                _ = try MoESIMDComponentReport.decodeValidated(try JSONSerialization.data(withJSONObject: obj))
                failed += 1; print("FAIL: moesimd report: tampered \(key) accepted")
            } catch MoESIMDError.invalidReport(_) {
                passed += 1
            } catch {
                failed += 1; print("FAIL: moesimd report: tampered \(key) wrong error \(error)")
            }
        }
    } catch {
        failed += 1; print("FAIL: moesimd region derivation/report — threw \(error)")
    }

    func expectShapeRejection(_ name: String, _ body: () throws -> Void) {
        do { try body(); failed += 1; print("FAIL: \(name) — accepted") }
        catch MoESIMDError.invalidShape(_) { passed += 1 }
        catch { failed += 1; print("FAIL: \(name) — wrong error \(error)") }
    }
    expectShapeRejection("moesimd: hidden not a Q4 block multiple") {
        _ = try MoESIMDShapeFacts(hidden: 100, inter: 64, topK: 2) }
    expectShapeRejection("moesimd: zero topK") { _ = try MoESIMDShapeFacts(hidden: 64, inter: 64, topK: 0) }
    expectShapeRejection("moesimd: topK above the region bound") {
        _ = try MoESIMDShapeFacts(hidden: 64, inter: 64, topK: 65) }
    expectShapeRejection("moesimd: inter above the region bound") {
        _ = try MoESIMDShapeFacts(hidden: 64, inter: 8224, topK: 2) }
    do {
        let oneBlock = try MoESIMDShapeFacts(hidden: 2816, inter: 32, topK: 8)
        let why = MoESIMDLegality.reason(
            MoESIMDSubschedule(family: .down, lanes: 8, rowsPerGroup: 16, accumulation: .plain),
            facts: oneBlock, hardware: moesimdProfile()) ?? ""
        check("moesimd: one down block per row cannot feed half of 8 lanes", why.contains("fewer than half"))
        let tail = try MoESIMDShapeFacts(hidden: 1056, inter: 512, topK: 5)
        check("moesimd: 33-block GU tail legal at L32 RT4", MoESIMDLegality.reason(
            MoESIMDSubschedule(family: .fusedGateUpAct, lanes: 32, rowsPerGroup: 4, accumulation: .plain),
            facts: tail, hardware: moesimdProfile()) == nil)
        check("moesimd: lane count 4 rejected", MoESIMDLegality.reason(
            MoESIMDSubschedule(family: .down, lanes: 4, rowsPerGroup: 32, accumulation: .plain),
            facts: tail, hardware: moesimdProfile()) != nil)
        check("moesimd: L*RT = 64 rejected", MoESIMDLegality.reason(
            MoESIMDSubschedule(family: .down, lanes: 16, rowsPerGroup: 4, accumulation: .plain),
            facts: tail, hardware: moesimdProfile()) != nil)
        check("moesimd: batch E3 rejected", MoESIMDLegality.batchReason(3, facts: tail) != nil)
        check("moesimd: E4 with a 5-slot tail legal", MoESIMDLegality.batchReason(4, facts: tail) == nil)
    } catch {
        failed += 1; print("FAIL: moesimd legality tails — threw \(error)")
    }

    do {
        let facts = try MoESIMDShapeFacts(hidden: 32, inter: 32, topK: 1)
        func block(_ lo: UInt8, _ hi: UInt8, _ q: UInt8) -> [UInt8] {
            [lo, hi] + [UInt8](repeating: q, count: 16)
        }
        var gu = [UInt8]()
        var dn = [UInt8]()
        // Gate rows: d = 1.0 (FP16 0x3C00); byte 0x98 gives low nibble 8
        // (weight 0) on x[0..15] and high nibble 9 (weight +1) on x[16..31].
        for _ in 0..<32 { gu += block(0x00, 0x3C, 0x98) }
        // Up rows: d = 0.125 (FP16 0x3000): dot = 2.
        for _ in 0..<32 { gu += block(0x00, 0x30, 0x98) }
        // Down rows: d = 0.5 (FP16 0x3800); byte 0x99 makes every weight +0.5.
        for _ in 0..<32 { dn += block(0x00, 0x38, 0x99) }
        let ones = [Float](repeating: 1, count: 32)
        let ref = try MoESIMDOracle.reference(facts: facts, guBlocks: gu, dnBlocks: dn, x: ones)
        check("moesimd oracle: hand-computed gate rows", ref.gate == [Double](repeating: 16, count: 32))
        check("moesimd oracle: hand-computed up rows", ref.up == [Double](repeating: 2, count: 32))
        // gelu_tanh(16) saturates to 16 in Double; act = 16 * 2.
        check("moesimd oracle: hand-computed activations", ref.act == [Double](repeating: 32, count: 32))
        // down = 32 activations * 32 * 0.5.
        check("moesimd oracle: hand-computed down rows", ref.dn == [Double](repeating: 512, count: 32))
        check("moesimd oracle: gelu_tanh(1) reference value",
              abs(MoESIMDOracle.geluTanhUp(gate: 1, up: 1) - 0.841192) < 1e-6)
        check("moesimd oracle: gelu_tanh saturates to zero for a large negative gate",
              MoESIMDOracle.geluTanhUp(gate: -20, up: 3) == 0)
        check("moesimd oracle: normalized rel uses max(1, |ref|)",
              MoESIMDOracle.worstNormalizedRel([1001], [1000]) == 0.001
              && MoESIMDOracle.worstNormalizedRel([0.5], [0]) == 0.5)
        check("moesimd oracle: non-finite GPU value is infinitely wrong",
              MoESIMDOracle.worstNormalizedRel([Float.nan], [0]) == .infinity)
        var badScale = gu
        badScale[1] = 0x7C   // first block scale bits 0x7C00 = +inf
        do {
            _ = try MoESIMDOracle.reference(facts: facts, guBlocks: badScale, dnBlocks: dn, x: ones)
            failed += 1; print("FAIL: moesimd oracle: non-finite scale accepted")
        } catch MoESIMDError.oracleFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd oracle: non-finite scale wrong error \(error)") }
        var nanX = ones
        nanX[3] = .nan
        do {
            _ = try MoESIMDOracle.reference(facts: facts, guBlocks: gu, dnBlocks: dn, x: nanX)
            failed += 1; print("FAIL: moesimd oracle: NaN input accepted")
        } catch MoESIMDError.oracleFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd oracle: NaN input wrong error \(error)") }

        var rng = MoESIMDTestRNG(state: 0x5EED_0049)
        let blocks = moesimdSyntheticBlocks(rows: 7, inDim: 64, rng: &rng)
        let xf = (0..<64).map { _ in Float(rng.unit() * 2 - 1) }
        let a = try MoESIMDOracle.q4MatvecFP64(blocks: blocks, byteOffset: 0, outDim: 7,
                                              inDim: 64, x: xf.map { Double($0) })
        let b = try Q4Oracle.matvecFP64(blocks: blocks[...], outDim: 7, inDim: 64, x: xf)
        check("moesimd oracle: agrees with the existing Q4 FP64 oracle",
              a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= 1e-12 * max(1, abs($1)) })
        do {
            _ = try MoESIMDOracle.q4MatvecFP64(blocks: blocks, byteOffset: 18, outDim: 7,
                                               inDim: 64, x: xf.map { Double($0) })
            failed += 1; print("FAIL: moesimd oracle: out-of-extent read accepted")
        } catch MoESIMDError.oracleFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd oracle: extent wrong error \(error)") }
    } catch {
        failed += 1; print("FAIL: moesimd oracle gates — threw \(error)")
    }

    do {
        let realFacts = try MoESIMDShapeFacts(hidden: 2816, inter: 704, topK: 8)
        let gu = MoESIMDSubschedule(family: .fusedGateUpAct, lanes: 16, rowsPerGroup: 8, accumulation: .plain)
        let src = try MoESIMDKernelEmitter.emit(gu, facts: realFacts)
        check("moesimd emit: literal incumbent activation helpers embedded",
              src.contains("[g4-activation-helpers-begin]") && src.contains("g4_gelu_up_value(gs + gc, us + uc)"))
        check("moesimd emit: fixed xor butterfly and staging barrier",
              src.contains("simd_shuffle_xor") && src.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))
        check("moesimd emit: kernel name", src.contains("kernel void moesimd_gu_L16_RT8_plain("))
        check("moesimd emit: distinct from v1 and the rejected VL variant",
              !src.contains("moegen_") && !src.contains("g4_q4_matvec_vl"))
        let again = try MoESIMDKernelEmitter.emit(gu, facts: realFacts)
        checkEqual("moesimd emit: deterministic", again, src)
        let dn = MoESIMDSubschedule(family: .down, lanes: 32, rowsPerGroup: 8, accumulation: .neumaierPrecise)
        let dsrc = try MoESIMDKernelEmitter.emit(dn, facts: realFacts)
        check("moesimd emit: Neumaier compensation in the precise down kernel",
              dsrc.contains("fabs(s) >= fabs(t)") && dsrc.contains("kernel void moesimd_dn_L32_RT8_neum("))
        check("moesimd emit: down kernel carries no activation helper", !dsrc.contains("g4_gelu_up_value"))
        do {
            _ = try MoESIMDKernelEmitter.emit(
                MoESIMDSubschedule(family: .down, lanes: 64, rowsPerGroup: 4, accumulation: .plain),
                facts: realFacts)
            failed += 1; print("FAIL: moesimd emit: illegal lanes accepted")
        } catch MoESIMDError.emissionFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd emit: illegal lanes wrong error \(error)") }
    } catch {
        failed += 1; print("FAIL: moesimd emission gates — threw \(error)")
    }
}

@MainActor
func runMoESIMDMetalGates() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("SKIP: moesimd Metal component gates — Metal unavailable")
        return
    }
    do {
        let hw = try MoESIMDHardwareProfile.probe(device: device)
        guard hw.threadExecutionWidth == 32 else {
            print("SKIP: moesimd Metal component gates — execution width \(hw.threadExecutionWidth) != 32")
            return
        }
        // 33 GU blocks (lane tails at L8/L16/L32), 16 down blocks (idle
        // lanes at L32), topK 5 (E4 slot tail).
        let facts = try MoESIMDShapeFacts(hidden: 1056, inter: 512, topK: 5)
        var rng = MoESIMDTestRNG(state: 0x4D6F_4553_494D_4431)
        let gu = moesimdSyntheticBlocks(rows: facts.topK * facts.gateUpRows, inDim: facts.hidden, rng: &rng)
        let dn = moesimdSyntheticBlocks(rows: facts.topK * facts.hidden, inDim: facts.inter, rng: &rng)
        let x = (0..<facts.hidden).map { _ in Float(rng.unit() * 2 - 1) }
        let derivation = MoESIMDScheduleDeriver.derive(facts: facts, hardware: hw)
        checkEqual("moesimd metal: synthetic tail catalog covers L8/L16/L32 x E4/E8",
                   derivation.schedules.count, 48)
        let reference = try MoESIMDOracle.reference(facts: facts, guBlocks: gu, dnBlocks: dn, x: x)
        let bench = try MoESIMDComponentBench(device: device, facts: facts, guBlocks: gu, dnBlocks: dn)
        let results = try MoESIMDQualification.screen(
            derivation: derivation, bench: bench,
            inputs: [MoESIMDNamedInput(name: "synthetic-uniform", values: x)],
            references: [reference], log: { _ in })
        checkEqual("moesimd metal: every schedule passes FP64, repeat bits and batch invariance",
                   results.filter { $0.componentAccepted }.count, results.count)
        for r in results where !r.componentAccepted {
            let detail = r.inputs.compactMap { $0.failure }.joined(separator: "; ")
            print("  moesimd metal rejection \(r.schedule.label): \(r.rejection ?? "") \(detail)")
        }
        check("moesimd metal: batch invariance proven for every schedule",
              results.allSatisfy { $0.batchInvariantBits == true })
        check("moesimd metal: repeat runs bit-identical",
              results.allSatisfy { $0.inputs.allSatisfy { $0.repeatBitIdentical } })
        check("moesimd metal: positive GPU intervals recorded",
              results.allSatisfy { $0.inputs.allSatisfy { ($0.guGpuMicros ?? 0) > 0 && ($0.dnGpuMicros ?? 0) > 0 } })
        checkEqual("moesimd metal: L8/L16/L32 exercised", Set(results.map { $0.schedule.gu.lanes }), Set([8, 16, 32]))

        let k = try MoESIMDCompiledKernel.compile(
            MoESIMDSubschedule(family: .down, lanes: 8, rowsPerGroup: 16, accumulation: .neumaierPrecise),
            facts: facts, device: device)
        check("moesimd metal: compiled width 32 and recorded precise options",
              k.pipeline.threadExecutionWidth == 32 && k.compileOptions == "fastMathEnabled=false")
        let guK = try MoESIMDCompiledKernel.compile(
            MoESIMDSubschedule(family: .fusedGateUpAct, lanes: 8, rowsPerGroup: 16, accumulation: .plain),
            facts: facts, device: device)
        do {
            _ = try bench.runComponent(gu: k, dn: k, batch: 4, x: x)
            failed += 1; print("FAIL: moesimd metal: wrong-family kernel accepted")
        } catch MoESIMDError.componentFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd metal: wrong family wrong error \(error)") }
        var bad = x
        bad[0] = .infinity
        do {
            _ = try bench.runComponent(gu: guK, dn: k, batch: 4, x: bad)
            failed += 1; print("FAIL: moesimd metal: non-finite input accepted")
        } catch MoESIMDError.componentFailed(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd metal: non-finite input wrong error \(error)") }
        do {
            _ = try bench.runComponent(gu: guK, dn: k, batch: 3, x: x)
            failed += 1; print("FAIL: moesimd metal: batch E3 accepted")
        } catch MoESIMDError.illegalSchedule(_) { passed += 1 }
        catch { failed += 1; print("FAIL: moesimd metal: batch E3 wrong error \(error)") }
        let iso = try bench.runDown(k, batch: 8, act: reference.act.map { Float($0) })
        check("moesimd metal: isolated down within the FP64 contract",
              MoESIMDOracle.worstNormalizedRel(iso.dn, reference.dn) < 1e-4 && iso.gpuSeconds > 0)
    } catch {
        failed += 1; print("FAIL: moesimd Metal component gates — threw \(error)")
    }
}

@MainActor
func runMoESIMDRealProbeGate() {
    let env = ProcessInfo.processInfo.environment
    let names = ["LEVIATHAN_GEMMA4_GGUF", "LEVIATHAN_GGUF", "GEMMA4_GGUF"]
    guard let path = names.compactMap({ env[$0] }).first,
          FileManager.default.fileExists(atPath: path) else {
        print("SKIP: moesimd real-GGUF MoE input probe — set LEVIATHAN_GEMMA4_GGUF")
        return
    }
    do {
        // mmap-backed resident runner; no model hashing.
        let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: path))
        check("moesimd probe: default nil", runner.moeInputProbe == nil)
        let base = try runner.forward(tokenIds: [2, 9259])
        let baseLogits = try runner.currentLogits()
        runner.reset()
        let log = MoESIMDProbeLog()
        runner.moeInputProbe = { layer, position, x in
            log.calls.append("\(layer):\(position)")
            if x.count != 2816 || !x.allSatisfy({ $0.isFinite }) { log.allFiniteHidden = false }
        }
        runner.layerProbe = { stage, _, _, _ in log.stages.insert(stage) }
        let probed = try runner.forward(tokenIds: [2, 9259])
        let probedLogits = try runner.currentLogits()
        runner.moeInputProbe = nil
        runner.layerProbe = nil
        let expected = (0..<2).flatMap { p in (0..<30).map { l in "\(l):\(p)" } }
        checkEqual("moesimd probe: one call per layer per position, in order", log.calls, expected)
        check("moesimd probe: finite 2816-wide vectors", log.allFiniteHidden)
        checkEqual("moesimd probe: layerProbe three-stage contract unchanged", log.stages,
                   Set(["attn_norm_h", "post_attn", "post_layer"]))
        checkEqual("moesimd probe: argmax unchanged", probed.argmaxToken, base.argmaxToken)
        check("moesimd probe: logits bit-identical with and without the probe",
              MoESIMDBits.identical(probedLogits, baseLogits))
    } catch {
        failed += 1; print("FAIL: moesimd real-GGUF MoE input probe — threw \(error)")
    }
}

runMoESIMDPureGates()
runMoESIMDMetalGates()
runMoESIMDRealProbeGate()

// MARK: - Summary

print("\(passed) passed, \(failed) failed")
if failed > 0 { exit(1) }
print("ALL TESTS PASSED")
