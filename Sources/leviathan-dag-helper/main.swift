import CryptoKit
import Foundation
import LeviathanCore

// leviathan-dag-helper: Swift side of the root Python DAG forge/run stage.
// It never executes Python. model_source.py is only read and hashed. Nothing runs an
// agent, an optimizer or a retune at inference time.
//   identity
//   forge --bundle DIR --python-binding FILE [--diagnostic-fail-optimizations]
//   run --bundle DIR --input-f32 FILE --output-json FILE
// bundle.sha256 is an integrity check over the exact bundle.json bytes. It is not
// authentication. The root Python stage publishes atomically. This helper only refuses
// to overwrite files.

struct HelperFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct PythonBinding: Codable, Equatable {
    let modelSourceSha256: String
    let modelSourceName: String
    let modelSeed: Int
    let backendSourceSha256: String
    let backendSourceFiles: Int
    let torchVersion: String
    let pythonVersion: String
    let atenOps: [String]
    let exampleInputSha256: String
}

struct FileBinding: Codable, Equatable {
    let graphSha256: String
    let weightsSha256: String
    let manifestSha256: String
    let modelSourceCopySha256: String
}

struct HelperIdentity: Codable, Equatable {
    let helperProtocol: Int
    let binarySha256: String
    let shaderFingerprint: String
    let toolchain: String
}

struct ShapeBinding: Codable, Equatable {
    let dtype: String
    let input: [Int]
    let output: [Int]
}

struct ManifestLayout: Codable, Equatable {
    let nodeID: Int
    let inDim: Int
    let outDim: Int
    let weightOffset: Int
    let weightCount: Int
    let biasOffset: Int
}

/// model.manifest.json (leviathan-dag-weights-1).
struct DAGManifest: Codable {
    let format: String
    let inputDim: Int
    let floatCount: Int
    let weightsSha256: String
    let layouts: [ManifestLayout]
    let refInput: [Float]
    let refOutput: [Float]
    let modelSeed: Int
}

/// bundle.json. bundle.sha256 covers its exact bytes as an integrity check, not a signature.
struct ForgeBundle: Codable {
    let format: String
    let python: PythonBinding
    let files: FileBinding
    let helper: HelperIdentity
    let hardware: ForgeHardwareProfile
    let shape: ShapeBinding
    let selection: ForgeBaselineSelection
}

struct ForgeSummary: Codable {
    let selected: ForgeCandidate
    let selectedRole: String
    let decision: String
    let baselineMedianGpuMicros: Double
    let selectedMedianGpuMicros: Double
    let optimizationsEvaluated: Int
    let optimizationsQualified: Int
    let diagnosticFailOptimizations: Bool
    let deviceName: String
}

struct GPUEvidence: Codable {
    let deviceName: String
    let commandStatus: String
    let computeDispatches: Int
    let kernelSequence: [String]
    let threadgroupWidth: Int
    let fusedDenseRelu: Bool
    let gpuCommandBufferMicros: Double
}

struct RunTiming: Codable {
    let gpuCommandBufferMicros: Double
    let helperValidationMicros: Double
    let helperPrepareMicros: Double
    let helperTotalMicros: Double
}

struct RunReport: Codable {
    let format: String
    let output: [Double]
    let outputShape: [Int]
    let selected: ForgeCandidate
    let selectedRole: String
    let decision: String
    let gpu: GPUEvidence
    let timing: RunTiming
}

struct LoadedModel {
    let program: DAGProgram
    let manifest: DAGManifest
    let files: FileBinding
}

struct Options {
    private(set) var values: [String: String] = [:]
    private(set) var flags: Set<String> = []

    init(_ arguments: [String], valued: Set<String>, allowedFlags: Set<String>) throws {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if valued.contains(argument) {
                guard index + 1 < arguments.count, values[argument] == nil else {
                    throw HelperFailure("\(argument) requires exactly one value")
                }
                values[argument] = arguments[index + 1]
                index += 2
            } else if allowedFlags.contains(argument), !flags.contains(argument) {
                flags.insert(argument)
                index += 1
            } else {
                throw HelperFailure("unknown or repeated argument '\(argument)'")
            }
        }
    }

    func required(_ name: String) throws -> String {
        guard let value = values[name], !value.isEmpty else { throw HelperFailure("missing \(name)") }
        return value
    }
}

enum Helper {
    static let protocolVersion = 1
    static let bundleFormat = "leviathan-dag-forge-bundle-1"
    static let manifestFormat = "leviathan-dag-weights-1"
    static let runFormat = "leviathan-dag-run-1"
    static let maxJSONBytes = 1 << 20
    static let maxModelSourceBytes = 1 << 20
    static let maxWeightBytes = 16 << 20
    static let maxExecutableBytes = 512 << 20
    static let qualificationSeed: UInt64 = 0x4C45_5649_4154_4841
    static let failedMarker = "FORGE_FAILED.json"
    static let legacyFailedMarker = "FAILED"
    static let bindingKeys: Set<String> = ["modelSourceSha256", "modelSourceName", "modelSeed", "backendSourceSha256",
                                           "backendSourceFiles", "torchVersion", "pythonVersion", "atenOps",
                                           "exampleInputSha256"]
    static let manifestKeys: Set<String> = ["format", "inputDim", "floatCount", "weightsSha256", "layouts",
                                            "refInput", "refOutput", "modelSeed"]
    static let graphKeys: Set<String> = ["format", "inputDim", "nodes", "outputValue"]
    static let bundleKeys: Set<String> = ["format", "python", "files", "helper", "hardware", "shape", "selection"]
    static let usage = "usage: leviathan-dag-helper identity | forge --bundle DIR --python-binding FILE [--diagnostic-fail-optimizations] | run --bundle DIR --input-f32 FILE --output-json FILE | serve --bundle DIR"

    static func main() -> Int32 {
        var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
        do {
            guard !arguments.isEmpty else { throw HelperFailure(usage) }
            let command = arguments.removeFirst()
            switch command {
            case "identity": try identityCommand(arguments)
            case "forge": try forgeCommand(arguments)
            case "run": try runCommand(arguments)
            case "serve": try serveCommand(arguments)
            default: throw HelperFailure("unknown command '\(command)'; \(usage)")
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("leviathan-dag-helper: error: \(error)\n".utf8))
            return 2
        }
    }

    // MARK: utilities

    static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(_ data: Data) -> String { hex(CryptoKit.SHA256.hash(data: data)) }

    static func isHex64(_ s: String) -> Bool {
        s.utf8.count == 64 && s.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    static func boundedText(_ s: String, max: Int) -> Bool {
        !s.isEmpty && s.utf8.count <= max && !s.unicodeScalars.contains { $0.value < 0x20 || $0.value == 0x7f }
    }

    static func boundedRead(_ url: URL, cap: Int, what: String) throws -> Data {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch { throw HelperFailure("cannot open \(what) (\(url.path))") }
        defer { try? handle.close() }
        let data = try handle.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else { throw HelperFailure("\(what) exceeds \(cap) bytes") }
        return data
    }

    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    static func decode<T: Decodable>(_ type: T.Type, from data: Data, what: String) throws -> T {
        do { return try JSONDecoder().decode(type, from: data) } catch { throw HelperFailure("\(what) decode failed: \(error)") }
    }

    /// Wraps a non-helper error with context so the failure message stays clear.
    static func step<T>(_ what: String, _ body: () throws -> T) throws -> T {
        do { return try body() } catch let failure as HelperFailure { throw failure } catch {
            throw HelperFailure("\(what): \(error)")
        }
    }

    /// Rejects unknown or missing top-level fields. Codable alone would ignore extra keys.
    static func requireExactKeys(_ data: Data, _ keys: Set<String>, what: String) throws {
        guard let object = try? JSONSerialization.jsonObject(with: data), let dict = object as? [String: Any],
              Set(dict.keys) == keys else {
            throw HelperFailure("\(what) must be a JSON object with exactly the fields \(keys.sorted())")
        }
    }

    static func littleEndianFloats(_ data: Data) -> [Float] {
        let raw = [UInt8](data)
        var out = [Float](repeating: 0, count: raw.count / 4)
        for i in out.indices {
            let b0 = UInt32(raw[i * 4]), b1 = UInt32(raw[i * 4 + 1]) << 8
            let b2 = UInt32(raw[i * 4 + 2]) << 16, b3 = UInt32(raw[i * 4 + 3]) << 24
            out[i] = Float(bitPattern: b0 | b1 | b2 | b3)
        }
        return out
    }

    static func float32Digest(_ values: [Float]) -> String {
        var bytes = Data(capacity: values.count * 4)
        for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
        return sha256Hex(bytes)
    }

    // MARK: identity

    /// Streams this executable through SHA-256, with a size bound.
    static func executableSha256() throws -> String {
        let path = ProcessInfo.processInfo.arguments[0]
        let url = (Bundle.main.executableURL ?? URL(fileURLWithPath: path)).resolvingSymlinksInPath()
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch { throw HelperFailure("cannot read own executable for identity") }
        defer { try? handle.close() }
        var hasher = CryptoKit.SHA256()
        var total = 0
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            total += chunk.count
            guard total <= maxExecutableBytes else { throw HelperFailure("own executable too large") }
            hasher.update(data: chunk)
        }
        return hex(hasher.finalize())
    }

    /// Computed once per command. Forge.toolchainIdentifier is itself cached for the process.
    static func currentIdentity() throws -> HelperIdentity {
        HelperIdentity(helperProtocol: protocolVersion, binarySha256: try executableSha256(),
                       shaderFingerprint: DAGMetalBackend.shaderFingerprint, toolchain: Forge.toolchainIdentifier)
    }

    // MARK: model validation (all before any GPU work)

    static func aten(_ op: DAGOp) -> String {
        switch op {
        case .dense: return "aten.linear.default"
        case .relu: return "aten.relu.default"
        case .add: return "aten.add.Tensor"
        }
    }

    static func validate(_ p: PythonBinding) throws {
        guard isHex64(p.modelSourceSha256), isHex64(p.backendSourceSha256), isHex64(p.exampleInputSha256),
              (0...Int(Int32.max)).contains(p.modelSeed), (1...4096).contains(p.backendSourceFiles),
              boundedText(p.torchVersion, max: 64), boundedText(p.pythonVersion, max: 64),
              boundedText(p.modelSourceName, max: 1024),
              !p.atenOps.isEmpty, p.atenOps.count <= DAGGraph.maxNodes else {
            throw HelperFailure("python binding fields outside validated bounds")
        }
    }

    static func rejectFailedMarker(_ dir: URL) throws {
        for marker in [failedMarker, legacyFailedMarker] where FileManager.default.fileExists(atPath: dir.appendingPathComponent(marker).path) {
            throw HelperFailure("bundle directory carries a \(marker) marker; refusing to use it")
        }
    }

    static func shape(_ program: DAGProgram) -> ShapeBinding {
        ShapeBinding(dtype: "float32", input: [1, program.graph.inputDim], output: [1, program.outputDim])
    }

    /// Reads the bound files with bounded reads, fully validates the DAGProgram built from the
    /// actual weights, and checks the manifest against values recomputed from that program.
    static func loadModel(_ dir: URL) throws -> LoadedModel {
        func file(_ name: String) -> URL { dir.appendingPathComponent(name) }
        let graphData = try boundedRead(file("model.graph.json"), cap: maxJSONBytes, what: "model.graph.json")
        let weightData = try boundedRead(file("model.weights.f32"), cap: maxWeightBytes, what: "model.weights.f32")
        let manifestData = try boundedRead(file("model.manifest.json"), cap: maxJSONBytes, what: "model.manifest.json")
        let source = try boundedRead(file("model_source.py"), cap: maxModelSourceBytes, what: "model_source.py")
        let files = FileBinding(graphSha256: sha256Hex(graphData), weightsSha256: sha256Hex(weightData),
                                manifestSha256: sha256Hex(manifestData), modelSourceCopySha256: sha256Hex(source))
        try requireExactKeys(graphData, graphKeys, what: "model.graph.json")
        try requireExactKeys(manifestData, manifestKeys, what: "model.manifest.json")
        let graph = try decode(DAGGraph.self, from: graphData, what: "model.graph.json")
        let manifest = try decode(DAGManifest.self, from: manifestData, what: "model.manifest.json")
        guard weightData.count % 4 == 0 else { throw HelperFailure("model.weights.f32 is not whole float32 values") }
        let weights = littleEndianFloats(weightData)
        let program = try step("graph/weights rejected") { try DAGProgram(graph: graph, weights: weights) }
        let layouts = program.layouts.values.sorted { $0.node < $1.node }.map {
            ManifestLayout(nodeID: $0.node, inDim: $0.inDim, outDim: $0.outDim, weightOffset: $0.weightOffset,
                           weightCount: $0.weightCount, biasOffset: $0.biasOffset)
        }
        guard manifest.format == manifestFormat else { throw HelperFailure("unsupported manifest format") }
        guard manifest.inputDim == graph.inputDim, manifest.floatCount == weights.count else {
            throw HelperFailure("manifest inputDim/floatCount do not match the validated graph and weights")
        }
        guard manifest.weightsSha256 == files.weightsSha256 else {
            throw HelperFailure("manifest weightsSha256 does not match model.weights.f32")
        }
        guard manifest.layouts == layouts else {
            throw HelperFailure("manifest layouts do not exactly match layouts recomputed from the graph")
        }
        guard manifest.refInput.count == graph.inputDim, manifest.refOutput.count == program.outputDim,
              manifest.refInput.allSatisfy({ $0.isFinite }), manifest.refOutput.allSatisfy({ $0.isFinite }) else {
            throw HelperFailure("manifest refInput/refOutput must be finite with widths \(graph.inputDim)/\(program.outputDim)")
        }
        return LoadedModel(program: program, manifest: manifest, files: files)
    }

    static func crossCheck(_ python: PythonBinding, _ model: LoadedModel) throws {
        try validate(python)
        guard model.files.modelSourceCopySha256 == python.modelSourceSha256 else {
            throw HelperFailure("model_source.py does not match the bound model source digest")
        }
        guard model.manifest.modelSeed == python.modelSeed else {
            throw HelperFailure("manifest modelSeed does not match the bound model seed")
        }
        guard float32Digest(model.manifest.refInput) == python.exampleInputSha256 else {
            throw HelperFailure("manifest refInput does not match the bound example input digest")
        }
        let expected = model.program.graph.nodes.map { aten($0.op) }
        guard python.atenOps == expected else {
            throw HelperFailure("bound ATen ops (\(python.atenOps.count)) do not match the \(expected.count) graph ops")
        }
    }

    // MARK: commands

    static func identityCommand(_ arguments: [String]) throws {
        _ = try Options(arguments, valued: [], allowedFlags: [])
        print(String(decoding: try encode(try currentIdentity()), as: UTF8.self))
    }

    static func forgeCommand(_ arguments: [String]) throws {
        let options = try Options(arguments, valued: ["--bundle", "--python-binding"],
                                  allowedFlags: ["--diagnostic-fail-optimizations"])
        let dir = URL(fileURLWithPath: try options.required("--bundle"), isDirectory: true)
        let bindingURL = URL(fileURLWithPath: try options.required("--python-binding"))
        let diagnostic = options.flags.contains("--diagnostic-fail-optimizations")
        try rejectFailedMarker(dir)
        let bundleURL = dir.appendingPathComponent("bundle.json")
        let digestURL = dir.appendingPathComponent("bundle.sha256")
        for url in [bundleURL, digestURL] where FileManager.default.fileExists(atPath: url.path) {
            throw HelperFailure("\(url.lastPathComponent) already exists; refusing to overwrite")
        }
        let bindingData = try boundedRead(bindingURL, cap: maxJSONBytes, what: "python binding")
        try requireExactKeys(bindingData, bindingKeys, what: "python binding")
        let python = try decode(PythonBinding.self, from: bindingData, what: "python binding")
        let model = try loadModel(dir)
        try crossCheck(python, model)
        let identity = try currentIdentity()
        // Qualification: the actual source example plus 3 deterministic inputs mapped to [-1, 1].
        let dim = model.program.graph.inputDim
        var inputs = [model.manifest.refInput]
        var rng = SplitMix64(seed: qualificationSeed)
        for _ in 0..<3 {
            var x = [Float](repeating: 0, count: dim)
            for i in 0..<dim {
                x[i] = rng.nextUniform()
            }
            inputs.append(x)
        }
        let (selection, hardware) = try step("forge selection failed") {
            try DAGForge.select(program: model.program, qualificationInputs: inputs,
                                sourceReference: model.manifest.refOutput, diagnosticFailOptimizations: diagnostic)
        }
        try step("forged selection failed validation") {
            try DAGForge.validateSelection(selection, program: model.program, hardware: hardware)
        }
        let bundle = ForgeBundle(format: bundleFormat, python: python, files: model.files, helper: identity,
                                 hardware: hardware, shape: shape(model.program), selection: selection)
        let data = try encode(bundle)
        guard data.count <= maxJSONBytes else { throw HelperFailure("bundle.json too large") }
        try step("cannot write bundle.json") { try data.write(to: bundleURL, options: .withoutOverwriting) }
        try step("cannot write bundle.sha256") {
            try Data((sha256Hex(data) + "\n").utf8).write(to: digestURL, options: .withoutOverwriting)
        }
        let summary = ForgeSummary(
            selected: selection.selected, selectedRole: selection.selectedRole, decision: selection.decision,
            baselineMedianGpuMicros: selection.baseline.medianGpuMicros ?? 0,
            selectedMedianGpuMicros: selection.selectedMedianGpuMicros,
            optimizationsEvaluated: selection.optimizations.count,
            optimizationsQualified: selection.optimizations.filter { $0.qualified }.count,
            diagnosticFailOptimizations: diagnostic, deviceName: hardware.deviceName)
        print(String(decoding: try encode(summary), as: UTF8.self))
    }

    static func runCommand(_ arguments: [String]) throws {
        let start = ProcessInfo.processInfo.systemUptime
        let options = try Options(arguments, valued: ["--bundle", "--input-f32", "--output-json"], allowedFlags: [])
        let dir = URL(fileURLWithPath: try options.required("--bundle"), isDirectory: true)
        let inputURL = URL(fileURLWithPath: try options.required("--input-f32"))
        let outputURL = URL(fileURLWithPath: try options.required("--output-json"))
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw HelperFailure("output already exists; refusing to overwrite")
        }
        // Full bundle validation (exact-byte digest, file bindings, cross-checks, helper identity)
        // is shared with `serve` and lives in loadValidatedBundle. The checks are identical, not
        // duplicated, and all of them run before any GPU dispatch below.
        let loaded = try loadValidatedBundle(dir)
        let bundle = loaded.bundle
        let program = loaded.program
        // Input: raw little-endian float32 [1, D], exact size, finite.
        let inputBytes = program.graph.inputDim * 4
        let inputData = try boundedRead(inputURL, cap: inputBytes, what: "input")
        guard inputData.count == inputBytes else {
            throw HelperFailure("input must be exactly [1, \(program.graph.inputDim)] float32")
        }
        let input = littleEndianFloats(inputData)
        guard input.allSatisfy({ $0.isFinite }) else { throw HelperFailure("input contains non-finite values") }
        let validated = ProcessInfo.processInfo.systemUptime
        // Device + persisted selection: validated against the actual hardware, never re-selected.
        let prepared = try step("Metal preparation failed") { try DAGMetalBackend.Prepared(program: program, input: input) }
        let hardware = ForgeHardwareProfile(
            deviceName: prepared.deviceName, maxTotalThreadsPerThreadgroup: prepared.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: prepared.threadExecutionWidth,
            recommendedMaxWorkingSetSize: prepared.recommendedMaxWorkingSetSize)
        guard hardware == bundle.hardware else {
            throw HelperFailure("hardware mismatch: bundle forged for '\(bundle.hardware.deviceName)', current device '\(hardware.deviceName)'; re-forge explicitly (no silent retune)")
        }
        try step("saved selection rejected") {
            try DAGForge.validateSelection(bundle.selection, program: program, hardware: hardware)
        }
        let selected = bundle.selection.selected
        let preparedAt = ProcessInfo.processInfo.systemUptime
        let result = try step("GPU run failed") {
            try prepared.run(threadgroupWidth: selected.threadgroupWidth, fuseDenseRelu: selected.fuseDenseRelu)
        }
        let finished = ProcessInfo.processInfo.systemUptime
        guard result.gpuMicros.isFinite, result.gpuMicros > 0 else { throw HelperFailure("GPU timestamps unavailable") }
        guard result.output.count == program.outputDim, result.output.allSatisfy({ $0.isFinite }) else {
            throw HelperFailure("GPU output has wrong size or non-finite values")
        }
        guard result.dispatchedSteps > 0, result.kernels.count == result.dispatchedSteps,
              result.deviceName == hardware.deviceName else {
            throw HelperFailure("GPU dispatch evidence inconsistent")
        }
        let report = RunReport(
            format: runFormat, output: result.output.map { Double($0) }, outputShape: [1, program.outputDim],
            selected: selected, selectedRole: bundle.selection.selectedRole, decision: bundle.selection.decision,
            gpu: GPUEvidence(deviceName: result.deviceName, commandStatus: "completed",
                             computeDispatches: result.dispatchedSteps, kernelSequence: result.kernels,
                             threadgroupWidth: selected.threadgroupWidth, fusedDenseRelu: selected.fuseDenseRelu,
                             gpuCommandBufferMicros: result.gpuMicros),
            timing: RunTiming(gpuCommandBufferMicros: result.gpuMicros,
                              helperValidationMicros: (validated - start) * 1e6,
                              helperPrepareMicros: (preparedAt - validated) * 1e6,
                              helperTotalMicros: (finished - start) * 1e6))
        let data = try encode(report)
        try step("cannot write output JSON") { try data.write(to: outputURL, options: .withoutOverwriting) }
    }
}

exit(Helper.main())
