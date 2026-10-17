import Foundation
import LeviathanCore

// Loaded runtime for `serve`. The validated bundle is a per-process immutable snapshot: graph,
// weights, manifest, model source copy, Python binding, helper identity and selection are read,
// hashed and pinned once at startup. Later changes to those files on disk do not alter the loaded
// snapshot, and a later process load rejects any digest mismatch. There is no per-request disk
// revalidation by design. Nothing here executes Python, re-selects, retunes or writes output files.

/// The immutable per-process snapshot produced by loadValidatedBundle. Externally immutable; no
/// Sendable conformance is promised.
struct ValidatedBundle {
    let bundle: ForgeBundle
    let program: DAGProgram
    let bundleSha256: String
}

/// Single-line ready message, exact schema leviathan-dag-session-1.
struct SessionReady: Codable {
    let format: String
    let bundleSha256: String
    let helper: HelperIdentity
    let hardware: ForgeHardwareProfile
    let shape: ShapeBinding
    let selected: ForgeCandidate
    let selectedRole: String
    let decision: String
    let startupTiming: StartupTiming
}

struct StartupTiming: Codable {
    let validationMillis: Double
    let prepareMillis: Double
    let totalMillis: Double
}

/// One NDJSON response line: exactly {id, report}.
struct SessionResponse: Codable {
    let id: Int
    let report: RunReport
}

extension Helper {
    /// Worst case is 4096 float32 features as JSON text (~15 chars each ~ 70 KiB, up to ~95 KiB with
    /// separators), so request lines are bounded at 128 KiB. The reader never grows beyond this.
    static let maxRequestBytes = 128 << 10
    static let sessionFormat = "leviathan-dag-session-1"
    static let requestKeys: Set<String> = ["id", "input"]

    /// Compact (single-line, no prettyPrinted) JSON, sorted keys, forward slashes unescaped.
    static func compactEncode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    /// Writes bytes directly to stdout and appends the newline terminator; one flush per line.
    static func writeLine(_ data: Data) throws {
        var out = data
        out.append(0x0A)
        try FileHandle.standardOutput.write(contentsOf: out)
    }

    // MARK: shared bundle validation (single source of truth for run and serve)

    /// Every strict check the old runCommand performed, in the same order, factored out so `run` and
    /// `serve` validate identically with no duplication: failed-marker rejection, exact-byte
    /// bundle.json digest, exact root keys, format, real file loads and hashes, manifest cross-checks,
    /// shape binding, and the full helper identity (binary/shader/toolchain). No GPU work happens here.
    static func loadValidatedBundle(_ dir: URL) throws -> ValidatedBundle {
        try rejectFailedMarker(dir)
        // Bundle integrity over exact bytes.
        let bundleData = try boundedRead(dir.appendingPathComponent("bundle.json"), cap: maxJSONBytes, what: "bundle.json")
        let digestData = try boundedRead(dir.appendingPathComponent("bundle.sha256"), cap: 128, what: "bundle.sha256")
        let recorded = String(decoding: digestData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard isHex64(recorded), recorded == sha256Hex(bundleData) else {
            throw HelperFailure("bundle digest mismatch: bundle.json is corrupt or was modified")
        }
        try requireExactKeys(bundleData, bundleKeys, what: "bundle.json")
        let bundle = try decode(ForgeBundle.self, from: bundleData, what: "bundle.json")
        guard bundle.format == bundleFormat else { throw HelperFailure("unsupported bundle format") }
        // Graph, actual weights, manifest, model source copy, Python binding.
        let model = try loadModel(dir)
        guard model.files == bundle.files else { throw HelperFailure("bundle file mismatch: corrupt or modified model files") }
        try crossCheck(bundle.python, model)
        let program = model.program
        guard bundle.shape == shape(program) else { throw HelperFailure("shape binding mismatch") }
        // Helper identity.
        let identity = try currentIdentity()
        let bound = bundle.helper
        guard identity.helperProtocol == bound.helperProtocol else { throw HelperFailure("helper protocol mismatch") }
        guard identity.binarySha256 == bound.binarySha256 else {
            throw HelperFailure("helper binary identity mismatch: this helper \(identity.binarySha256.prefix(16)) did not forge this bundle (\(bound.binarySha256.prefix(16))); re-forge explicitly (no silent retune)")
        }
        guard identity.shaderFingerprint == bound.shaderFingerprint else {
            throw HelperFailure("DAG Metal shader fingerprint mismatch; re-forge explicitly")
        }
        guard identity.toolchain == bound.toolchain else {
            throw HelperFailure("toolchain/OS mismatch: bundle '\(bound.toolchain)', current '\(identity.toolchain)'")
        }
        return ValidatedBundle(bundle: bundle, program: program, bundleSha256: recorded)
    }

    // MARK: bounded stdin line reader

    /// Per-serve stdin line reader. The pending buffer is instance state created once before the
    /// request loop, so trailing lines that arrive in the same 4096-byte read chunk as an earlier
    /// request are retained for the next readLine() call instead of being discarded. Reads are
    /// chunked Darwin.read with EINTR retry; no unbounded readLine and no one-byte syscalls.
    /// Each newline-terminated line is capped at maxRequestBytes before any JSON decode, the
    /// pending accumulator never exceeds maxRequestBytes + 4096, and UTF-8 is validated strictly
    /// (malformed bytes are fatal, never silently replaced). Oversize and EOF-truncated lines, and
    /// empty lines, are fatal. A nil return means clean EOF.
    private final class RequestLineReader {
        private var pending = Data()
        private var buffer = [UInt8](repeating: 0, count: 4096)

        func readLine() throws -> String? {
            while true {
                if let newline = pending.firstIndex(of: 0x0A) {
                    let lineData = pending.subdata(in: pending.startIndex..<newline)
                    pending.removeSubrange(pending.startIndex...newline)
                    guard lineData.count <= maxRequestBytes else {
                        throw HelperFailure("request line exceeds \(maxRequestBytes) bytes")
                    }
                    guard !lineData.isEmpty else { throw HelperFailure("empty request line") }
                    guard let line = String(data: lineData, encoding: .utf8) else {
                        throw HelperFailure("request line is not valid UTF-8")
                    }
                    return line
                }
                var n = 0
                repeat {
                    n = Darwin.read(0, &buffer, buffer.count)
                } while n < 0 && errno == EINTR
                if n < 0 { throw HelperFailure("stdin read failed (errno \(errno))") }
                if n == 0 {
                    if pending.isEmpty { return nil }
                    throw HelperFailure("truncated request line at EOF")
                }
                pending.append(contentsOf: buffer[0..<n])
                guard pending.count <= maxRequestBytes + buffer.count else {
                    throw HelperFailure("request line exceeds \(maxRequestBytes) bytes")
                }
            }
        }
    }

    // MARK: request parsing

    static func isBooleanNumber(_ value: NSNumber) -> Bool {
        CFGetTypeID(value) == CFBooleanGetTypeID()
    }

    /// Strict request parse: a JSON object with exactly the keys {id, input}; id is a positive
    /// integer <= 2^31-1 (booleans and non-integer values rejected), strictly sequential starting
    /// at 1; input is an array of exactly expectedInputDim finite numeric values, each accepted by
    /// rounding to the nearest float32 (ordinary finite JSON numbers such as 0.1 are accepted;
    /// non-finite values and values that overflow float32 are rejected) before any buffer upload or
    /// GPU dispatch.
    static func parseRequest(_ line: String, expectedID: Int, expectedInputDim: Int) throws -> [Float] {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any],
              Set(dict.keys) == requestKeys else {
            throw HelperFailure("request must be a JSON object with exactly the fields ['id', 'input']")
        }
        guard let idNumber = dict["id"] as? NSNumber, !isBooleanNumber(idNumber),
              !CFNumberIsFloatType(idNumber) else {
            throw HelperFailure("request id must be an integer (booleans rejected)")
        }
        let idValue = idNumber.doubleValue
        guard idValue.rounded() == idValue, idValue >= 1, idValue <= Double(Int32.max) else {
            throw HelperFailure("request id must be a positive integer <= 2^31-1")
        }
        let id = Int(idValue)
        guard id == expectedID else {
            throw HelperFailure("request id must be strictly sequential: expected \(expectedID), got \(id)")
        }
        guard let rawInput = dict["input"] as? [Any] else {
            throw HelperFailure("request input must be an array")
        }
        guard rawInput.count == expectedInputDim else {
            throw HelperFailure("request input must contain exactly \(expectedInputDim) values")
        }
        var input = [Float](repeating: 0, count: rawInput.count)
        for (i, element) in rawInput.enumerated() {
            guard let number = element as? NSNumber, !isBooleanNumber(number) else {
                throw HelperFailure("request input[\(i)] is not a number")
            }
            let double = number.doubleValue
            let float32Limit = Double(Float.greatestFiniteMagnitude)
            guard double.isFinite, double.magnitude <= float32Limit else {
                throw HelperFailure("request input[\(i)] is not a finite float32 value")
            }
            input[i] = Float(double)
        }
        return input
    }

    // MARK: serve command

    /// Persistent stdio session. Loads and validates the bundle once, prepares GPU state once, prints
    /// one single-line ready message, then answers bounded NDJSON requests one compact line each.
    /// No ports, no network, no per-request disk revalidation, no output files, no Python execution.
    static func serveCommand(_ arguments: [String]) throws {
        let start = ProcessInfo.processInfo.systemUptime
        let options = try Options(arguments, valued: ["--bundle"], allowedFlags: [])
        let dir = URL(fileURLWithPath: try options.required("--bundle"), isDirectory: true)
        // Load-time validation only; the returned snapshot is immutable for this process.
        let loaded = try loadValidatedBundle(dir)
        let validatedAt = ProcessInfo.processInfo.systemUptime
        // GPU state prepared once with a zero input; per-request inputs go through prepared.run(input:).
        let zeroInput = [Float](repeating: 0, count: loaded.program.graph.inputDim)
        let prepared = try step("Metal preparation failed") { try DAGMetalBackend.Prepared(program: loaded.program, input: zeroInput) }
        let hardware = ForgeHardwareProfile(
            deviceName: prepared.deviceName, maxTotalThreadsPerThreadgroup: prepared.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: prepared.threadExecutionWidth,
            recommendedMaxWorkingSetSize: prepared.recommendedMaxWorkingSetSize)
        guard hardware == loaded.bundle.hardware else {
            throw HelperFailure("hardware mismatch: bundle forged for '\(loaded.bundle.hardware.deviceName)', current device '\(hardware.deviceName)'; re-forge explicitly (no silent retune)")
        }
        try step("saved selection rejected") {
            try DAGForge.validateSelection(loaded.bundle.selection, program: loaded.program, hardware: hardware)
        }
        let readyAt = ProcessInfo.processInfo.systemUptime
        let ready = SessionReady(
            format: sessionFormat, bundleSha256: loaded.bundleSha256, helper: loaded.bundle.helper,
            hardware: hardware, shape: loaded.bundle.shape, selected: loaded.bundle.selection.selected,
            selectedRole: loaded.bundle.selection.selectedRole, decision: loaded.bundle.selection.decision,
            startupTiming: StartupTiming(validationMillis: (validatedAt - start) * 1e3,
                                         prepareMillis: (readyAt - validatedAt) * 1e3,
                                         totalMillis: (readyAt - start) * 1e3))
        try writeLine(try compactEncode(ready))
        // Request loop. Every failure below (bad line, bad JSON, GPU error, inconsistent evidence) is
        // fatal: stderr message and exit 2, with no half-state reuse.
        var expectedID = 1
        let reader = RequestLineReader()
        while let line = try reader.readLine() {
            let requestStart = ProcessInfo.processInfo.systemUptime
            let input = try parseRequest(line, expectedID: expectedID,
                                         expectedInputDim: loaded.program.graph.inputDim)
            let selected = loaded.bundle.selection.selected
            // Per request: run the persisted selected plan only. Input upload happens under the
            // Prepared lock after validation; weights, pipelines and value buffers are reused.
            let result = try step("GPU run failed") {
                try prepared.run(input: input, threadgroupWidth: selected.threadgroupWidth,
                                 fuseDenseRelu: selected.fuseDenseRelu)
            }
            guard result.gpuMicros.isFinite, result.gpuMicros > 0 else { throw HelperFailure("GPU timestamps unavailable") }
            guard result.output.count == loaded.program.outputDim, result.output.allSatisfy({ $0.isFinite }) else {
                throw HelperFailure("GPU output has wrong size or non-finite values")
            }
            guard result.dispatchedSteps > 0, result.kernels.count == result.dispatchedSteps,
                  result.deviceName == hardware.deviceName else {
                throw HelperFailure("GPU dispatch evidence inconsistent")
            }
            let readBackAt = ProcessInfo.processInfo.systemUptime
            let report = RunReport(
                format: runFormat, output: result.output.map { Double($0) },
                outputShape: [1, loaded.program.outputDim], selected: selected,
                selectedRole: loaded.bundle.selection.selectedRole, decision: loaded.bundle.selection.decision,
                gpu: GPUEvidence(deviceName: result.deviceName, commandStatus: "completed",
                                 computeDispatches: result.dispatchedSteps, kernelSequence: result.kernels,
                                 threadgroupWidth: selected.threadgroupWidth,
                                 fusedDenseRelu: selected.fuseDenseRelu,
                                 gpuCommandBufferMicros: result.gpuMicros),
                timing: RunTiming(gpuCommandBufferMicros: result.gpuMicros,
                                  helperValidationMicros: 0, helperPrepareMicros: 0,
                                  helperTotalMicros: (readBackAt - requestStart) * 1e6))
            try writeLine(try compactEncode(SessionResponse(id: expectedID, report: report)))
            expectedID += 1
        }
        // Clean EOF: fall through, main returns 0.
    }
}
