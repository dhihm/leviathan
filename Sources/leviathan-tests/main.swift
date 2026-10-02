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
    } catch {
        failed += 1; print("FAIL: real-model Gemma gates — threw \(error)")
    }
} else {
    print("SKIP: real-model Gemma gates (set LEVIATHAN_GEMMA_GGUF to opt in)")
}

// MARK: - Summary

print("\(passed) passed, \(failed) failed")
if failed > 0 { exit(1) }
print("ALL TESTS PASSED")
