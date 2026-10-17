import CryptoKit
import Foundation
import LeviathanCore

// leviathan-forge-helper: Swift side of tools/forge_run/leviathan_forge.py.
// Never executes Python; no agent/optimizer at inference time.
//   identity
//   forge --bundle DIR --python-binding FILE [--diagnostic-fail-optimizations]
//   run --bundle DIR --input-f32 FILE --output-json FILE
// Identity = SHA-256 of this executable (computed at run time and
// validated), compiled-in Metal shader fingerprint, Swift toolchain/OS.
// Any mismatch rejects the bundle; nothing is silently re-tuned.

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

/// bundle.json; its exact bytes are covered by bundle.sha256 (integrity
/// check, not a signature).
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
    let export: WeightExport
    let program: LoweredProgram
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
            } else if allowedFlags.contains(argument) {
                flags.insert(argument)
                index += 1
            } else {
                throw HelperFailure("unknown argument '\(argument)'")
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
    static let bundleFormat = "leviathan-forge-bundle-1"
    static let maxBundleJSONBytes = 1 << 20
    static let maxModelSourceBytes = 1 << 20
    static let maxExecutableBytes = 512 << 20
    static let qualificationSeed: UInt64 = 0x4C45_5649_4154_4841
    static let usage = "usage: leviathan-forge-helper identity | forge --bundle DIR --python-binding FILE [--diagnostic-fail-optimizations] | run --bundle DIR --input-f32 FILE --output-json FILE"

    static func main() -> Int32 {
        var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
        guard !arguments.isEmpty else {
            FileHandle.standardError.write(Data((usage + "\n").utf8))
            return 64
        }
        let command = arguments.removeFirst()
        do {
            switch command {
            case "identity": try identityCommand(arguments)
            case "forge": try forgeCommand(arguments)
            case "run": try runCommand(arguments)
            default: throw HelperFailure("unknown command '\(command)'; \(usage)")
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("leviathan-forge-helper: error: \(error)\n".utf8))
            return 2
        }
    }

    static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(_ data: Data) -> String { hex(CryptoKit.SHA256.hash(data: data)) }

    static func isHex64(_ s: String) -> Bool {
        s.utf8.count == 64 && s.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
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

    /// Streams this executable through SHA-256 (bounded).
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

    static func currentIdentity() throws -> HelperIdentity {
        try HelperIdentity(helperProtocol: protocolVersion, binarySha256: executableSha256(),
                           shaderFingerprint: MetalBackend.shaderFingerprint, toolchain: Forge.toolchainIdentifier)
    }

    static func float32Digest(_ values: [Float]) -> String {
        var bytes = Data(capacity: values.count * 4)
        for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
        return sha256Hex(bytes)
    }

    static func validate(_ p: PythonBinding) throws {
        guard isHex64(p.modelSourceSha256), isHex64(p.backendSourceSha256), isHex64(p.exampleInputSha256),
              p.modelSeed >= 0, p.modelSeed <= Int(Int32.max), p.backendSourceFiles > 0, p.backendSourceFiles <= 4096,
              !p.torchVersion.isEmpty, p.torchVersion.utf8.count <= 64,
              !p.pythonVersion.isEmpty, p.pythonVersion.utf8.count <= 64, p.modelSourceName.utf8.count <= 1024,
              !p.atenOps.isEmpty, p.atenOps.count <= Limits.maxOps else {
            throw HelperFailure("python binding fields outside validated bounds")
        }
    }

    /// Loads the bound model files with bounded reads and the existing
    /// validated weights-1 loader; builds the lowered program.
    static func loadModel(_ dir: URL) throws -> LoadedModel {
        func file(_ name: String) -> URL { dir.appendingPathComponent(name) }
        let graph = try boundedRead(file("model.graph.json"), cap: Limits.maxGraphJSONBytes, what: "model.graph.json")
        let weights = try boundedRead(file("model.weights.f32"), cap: Limits.maxWeightBytes, what: "model.weights.f32")
        let manifest = try boundedRead(file("model.manifest.json"), cap: WeightExport.maxManifestBytes, what: "model.manifest.json")
        let source = try boundedRead(file("model_source.py"), cap: maxModelSourceBytes, what: "model_source.py")
        let files = FileBinding(graphSha256: sha256Hex(graph), weightsSha256: sha256Hex(weights),
                                manifestSha256: sha256Hex(manifest), modelSourceCopySha256: sha256Hex(source))
        let export: WeightExport
        do { export = try WeightExport.load(prefix: file("model").path) } catch { throw HelperFailure("model export rejected: \(error)") }
        guard export.weightsSha256 == files.weightsSha256 else { throw HelperFailure("model.weights.f32 changed while loading") }
        let base: LoweredProgram
        do { base = try CompiledArtifact(graph: export.graph, seed: 0).program } catch { throw HelperFailure("graph rejected: \(error)") }
        guard export.weights.count == base.weights.count else { throw HelperFailure("weight count does not match graph") }
        let program = LoweredProgram(inputDim: base.inputDim, outputDim: base.outputDim, bufferCount: base.bufferCount,
                                     steps: base.steps, weights: export.weights)
        return LoadedModel(export: export, program: program, files: files)
    }

    static func crossCheck(_ python: PythonBinding, _ model: LoadedModel) throws {
        try validate(python)
        guard model.files.modelSourceCopySha256 == python.modelSourceSha256 else {
            throw HelperFailure("model_source.py does not match the bound model source digest")
        }
        let ops = model.export.graph.ops
        guard python.atenOps.count == ops.count else { throw HelperFailure("bound ATen op list does not match the graph") }
        for (aten, op) in zip(python.atenOps, ops) {
            let expected: String
            switch op {
            case .dense: expected = "aten.linear.default"
            case .relu: expected = "aten.relu.default"
            }
            guard aten == expected else { throw HelperFailure("bound ATen op '\(aten)' does not match graph op '\(expected)'") }
        }
    }

    static func identityCommand(_ arguments: [String]) throws {
        _ = try Options(arguments, valued: [], allowedFlags: [])
        let data = try encode(try currentIdentity())
        print(String(decoding: data, as: UTF8.self))
    }

    static func forgeCommand(_ arguments: [String]) throws {
        let options = try Options(arguments, valued: ["--bundle", "--python-binding"],
                                  allowedFlags: ["--diagnostic-fail-optimizations"])
        let dir = URL(fileURLWithPath: try options.required("--bundle"), isDirectory: true)
        let bindingURL = URL(fileURLWithPath: try options.required("--python-binding"))
        let diagnostic = options.flags.contains("--diagnostic-fail-optimizations")
        let bundleURL = dir.appendingPathComponent("bundle.json")
        let digestURL = dir.appendingPathComponent("bundle.sha256")
        for url in [bundleURL, digestURL] where FileManager.default.fileExists(atPath: url.path) {
            throw HelperFailure("\(url.lastPathComponent) already exists; refusing to overwrite")
        }
        let bindingData = try boundedRead(bindingURL, cap: 64 << 10, what: "python binding")
        let python = try decode(PythonBinding.self, from: bindingData, what: "python binding")
        let model = try loadModel(dir)
        try crossCheck(python, model)
        let example = model.export.refInput
        guard float32Digest(example) == python.exampleInputSha256 else {
            throw HelperFailure("manifest example input does not match the bound example input digest")
        }
        var inputs = [example]
        var rng = SplitMix64(seed: qualificationSeed)
        for _ in 0..<3 { inputs.append((0..<model.program.inputDim).map { _ in rng.nextUniform() }) }
        let (selection, hardware) = try ForgeBaseline.select(
            program: model.program, qualificationInputs: inputs,
            sourceReference: model.export.refOutput, diagnosticFailOptimizations: diagnostic)
        let identity = try currentIdentity()
        let bundle = ForgeBundle(
            format: bundleFormat, python: python, files: model.files, helper: identity, hardware: hardware,
            shape: ShapeBinding(dtype: "float32", input: [1, model.program.inputDim], output: [1, model.program.outputDim]),
            selection: selection)
        let data = try encode(bundle)
        guard data.count <= maxBundleJSONBytes else { throw HelperFailure("bundle.json too large") }
        try data.write(to: bundleURL, options: .withoutOverwriting)
        try Data((sha256Hex(data) + "\n").utf8).write(to: digestURL, options: .withoutOverwriting)
        let summary = ForgeSummary(
            selected: selection.selected, selectedRole: selection.selectedRole, decision: selection.decision,
            baselineMedianGpuMicros: selection.baseline.medianGpuMicros ?? 0,
            selectedMedianGpuMicros: selection.selectedMedianGpuMicros,
            optimizationsEvaluated: selection.optimizations.count,
            optimizationsQualified: selection.optimizations.filter { $0.qualified }.count,
            diagnosticFailOptimizations: diagnostic, deviceName: hardware.deviceName)
        let summaryData = try encode(summary)
        print(String(decoding: summaryData, as: UTF8.self))
    }

    static func runCommand(_ arguments: [String]) throws {
        let options = try Options(arguments, valued: ["--bundle", "--input-f32", "--output-json"], allowedFlags: [])
        let start = ProcessInfo.processInfo.systemUptime
        let dir = URL(fileURLWithPath: try options.required("--bundle"), isDirectory: true)
        let inputURL = URL(fileURLWithPath: try options.required("--input-f32"))
        let outputURL = URL(fileURLWithPath: try options.required("--output-json"))
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw HelperFailure("output already exists; refusing to overwrite")
        }
        // Bundle integrity.
        let bundleData = try boundedRead(dir.appendingPathComponent("bundle.json"), cap: maxBundleJSONBytes, what: "bundle.json")
        let digestData = try boundedRead(dir.appendingPathComponent("bundle.sha256"), cap: 128, what: "bundle.sha256")
        let recorded = String(decoding: digestData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard isHex64(recorded), recorded == sha256Hex(bundleData) else {
            throw HelperFailure("bundle digest mismatch: bundle.json is corrupt or was modified")
        }
        let bundle = try decode(ForgeBundle.self, from: bundleData, what: "bundle.json")
        guard bundle.format == bundleFormat else { throw HelperFailure("unsupported bundle format") }
        // Graph, actual weights, manifest, model source.
        let model = try loadModel(dir)
        guard model.files == bundle.files else { throw HelperFailure("bundle file mismatch: corrupt or modified model files") }
        try crossCheck(bundle.python, model)
        let program = model.program
        guard bundle.shape == ShapeBinding(dtype: "float32", input: [1, program.inputDim], output: [1, program.outputDim]) else {
            throw HelperFailure("shape binding mismatch")
        }
        // Backend identity.
        let identity = try currentIdentity()
        guard identity.helperProtocol == bundle.helper.helperProtocol else { throw HelperFailure("helper protocol mismatch") }
        guard identity.binarySha256 == bundle.helper.binarySha256 else {
            throw HelperFailure("helper binary identity mismatch: this helper \(identity.binarySha256.prefix(16)) did not forge this bundle (\(bundle.helper.binarySha256.prefix(16))); re-forge explicitly (no silent retune)")
        }
        guard identity.shaderFingerprint == bundle.helper.shaderFingerprint else {
            throw HelperFailure("Metal shader source mismatch; re-forge explicitly")
        }
        guard identity.toolchain == bundle.helper.toolchain else {
            throw HelperFailure("toolchain mismatch: bundle '\(bundle.helper.toolchain)', current '\(identity.toolchain)'")
        }
        // Input: raw little-endian float32, exact static size, finite.
        let inputData = try boundedRead(inputURL, cap: Limits.maxDim * 4, what: "input")
        guard inputData.count == program.inputDim * 4 else {
            throw HelperFailure("input must be exactly [1, \(program.inputDim)] float32")
        }
        let raw = [UInt8](inputData)
        var input = [Float](repeating: 0, count: program.inputDim)
        for i in 0..<input.count {
            let b0 = UInt32(raw[i * 4]), b1 = UInt32(raw[i * 4 + 1]) << 8
            let b2 = UInt32(raw[i * 4 + 2]) << 16, b3 = UInt32(raw[i * 4 + 3]) << 24
            input[i] = Float(bitPattern: b0 | b1 | b2 | b3)
        }
        guard input.allSatisfy({ $0.isFinite }) else { throw HelperFailure("input contains non-finite values") }
        let validated = ProcessInfo.processInfo.systemUptime
        // Device + saved selection (validated, never re-tuned).
        let prepared = try MetalBackend.Prepared(program: program, batch: 1, input: input)
        let hardware = ForgeHardwareProfile(
            deviceName: prepared.deviceName, maxTotalThreadsPerThreadgroup: prepared.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: prepared.threadExecutionWidth,
            recommendedMaxWorkingSetSize: prepared.recommendedMaxWorkingSetSize)
        guard hardware == bundle.hardware else {
            throw HelperFailure("hardware mismatch: bundle forged for '\(bundle.hardware.deviceName)', current device '\(hardware.deviceName)'; re-forge explicitly (no silent retune)")
        }
        do { try ForgeBaseline.validateSelection(bundle.selection, program: program, hardware: hardware) }
        catch { throw HelperFailure("saved selection rejected: \(error)") }
        let selected = bundle.selection.selected
        let preparedAt = ProcessInfo.processInfo.systemUptime
        let (output, gpuMicros) = try prepared.run(threadgroupWidth: selected.threadgroupWidth,
                                                   fuseDenseRelu: selected.fuseDenseRelu)
        let finished = ProcessInfo.processInfo.systemUptime
        guard gpuMicros.isFinite, gpuMicros > 0 else { throw HelperFailure("GPU timestamps unavailable") }
        guard output.count == program.outputDim, output.allSatisfy({ $0.isFinite }) else {
            throw HelperFailure("GPU output has wrong size or non-finite values")
        }
        let kernels = ForgeBaseline.kernelSequence(program, candidate: selected)
        let report = RunReport(
            format: "leviathan-forge-run-1", output: output.map { Double($0) }, outputShape: [1, program.outputDim],
            selected: selected, selectedRole: bundle.selection.selectedRole, decision: bundle.selection.decision,
            gpu: GPUEvidence(deviceName: hardware.deviceName, commandStatus: "completed",
                             computeDispatches: kernels.count, kernelSequence: kernels,
                             threadgroupWidth: selected.threadgroupWidth, fusedDenseRelu: selected.fuseDenseRelu,
                             gpuCommandBufferMicros: gpuMicros),
            timing: RunTiming(gpuCommandBufferMicros: gpuMicros,
                              helperValidationMicros: (validated - start) * 1e6,
                              helperPrepareMicros: (preparedAt - validated) * 1e6,
                              helperTotalMicros: (finished - start) * 1e6))
        try encode(report).write(to: outputURL, options: .withoutOverwriting)
    }
}

exit(Helper.main())
