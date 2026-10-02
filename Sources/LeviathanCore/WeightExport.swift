import Foundation

/// Errors from loading an exporter-produced weights/manifest pair.
public enum WeightExportError: Error, Equatable, Sendable {
    case fileTooLarge(String)
    case malformed(String)
    case hashMismatch(expected: String, actual: String)
    case inconsistentOpTable(String)
}

/// A loaded and fully validated weight export from the restricted PyTorch
/// exporter (`examples/export_torch.py`): architecture, float32
/// little-endian weight blob, content hash, and the PyTorch-computed
/// deterministic reference input/output for parity checking.
public struct WeightExport: Sendable {
    public let graph: GraphIR
    public let weights: [Float]
    public let weightsSha256: String
    public let refInput: [Float]
    public let refOutput: [Float]

    public static let maxManifestBytes = 1 << 20
    public static let maxGraphBytes = 1 << 20
    public static let maxWeightBytes = 16 << 20

    struct ManifestFile: Codable {
        let format: String
        let inputDim: Int
        let floatCount: Int
        let weightsSha256: String
        let ops: [Op]
        let refInputSeed: Int?
        let refInput: [Double]
        let refOutput: [Double]
        let modelSeed: Int?

        struct Op: Codable {
            let inDim: Int
            let outDim: Int
            let weightFloatOffset: Int
            let weightFloatCount: Int
            let biasFloatOffset: Int
        }
    }

    /// Reads at most `cap` bytes plus one, so an oversized untrusted file
    /// is rejected on its actual byte count without ever being fully read.
    static func boundedRead(_ url: URL, cap: Int, what: String) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else {
            throw WeightExportError.fileTooLarge("\(what) exceeds \(cap) bytes")
        }
        return data
    }

    /// The exact per-dense-op float layout the engine expects (W[in,out]
    /// row-major then bias, concatenated in op order), computed from the
    /// validated graph WITHOUT synthesizing any weight values.
    static func expectedLayout(_ graph: GraphIR) -> [(inDim: Int, outDim: Int,
                                                      wOff: Int, wCount: Int, bOff: Int)] {
        var layout: [(Int, Int, Int, Int, Int)] = []
        var currentDim = graph.inputDim
        var offset = 0
        for op in graph.ops {
            if case let .dense(outDim) = op {
                let wCount = currentDim * outDim
                layout.append((currentDim, outDim, offset, wCount, offset + wCount))
                offset += wCount + outDim
                currentDim = outDim
            }
        }
        return layout
    }

    /// Loads `<prefix>.graph.json`, `<prefix>.weights.f32`, and
    /// `<prefix>.manifest.json`, validating every untrusted field before
    /// anything is used: bounded reads, format tag, exact float count
    /// divisible by 4, little-endian SHA-256 over the raw blob, an op
    /// table consistent with the graph's layout, manifest/graph inputDim
    /// agreement, and reference vectors of the right length and finite in
    /// Float (a finite Double like 1e300 converts to Float infinity and
    /// is rejected).
    public static func load(prefix: String) throws -> WeightExport {
        let manifestData = try boundedRead(
            URL(fileURLWithPath: prefix + ".manifest.json"),
            cap: maxManifestBytes, what: "manifest")
        let manifest: ManifestFile
        do { manifest = try JSONDecoder().decode(ManifestFile.self, from: manifestData) }
        catch { throw WeightExportError.malformed("manifest decode: \(error)") }
        guard manifest.format == "leviathan-forge-weights-1" else {
            throw WeightExportError.malformed("unknown format \(manifest.format)")
        }
        guard manifest.floatCount > 0, manifest.floatCount <= maxWeightBytes / 4,
              manifest.ops.count > 0, manifest.ops.count <= 64 else {
            throw WeightExportError.malformed("manifest counts out of bounds")
        }
        let weightsData = try boundedRead(
            URL(fileURLWithPath: prefix + ".weights.f32"),
            cap: maxWeightBytes, what: "weights")
        guard weightsData.count % 4 == 0,
              weightsData.count / 4 == manifest.floatCount else {
            throw WeightExportError.malformed(
                "weights \(weightsData.count) bytes vs manifest floatCount \(manifest.floatCount)")
        }
        let actualHash = SHA256.hex(SHA256.hash([UInt8](weightsData)))
        guard actualHash == manifest.weightsSha256 else {
            throw WeightExportError.hashMismatch(
                expected: manifest.weightsSha256, actual: actualHash)
        }
        // Architecture from the graph JSON; validated through the normal
        // GraphIR validation before use.
        let graphData = try boundedRead(
            URL(fileURLWithPath: prefix + ".graph.json"),
            cap: maxGraphBytes, what: "graph json")
        let graph: GraphIR
        do { graph = try JSONDecoder().decode(GraphIR.self, from: graphData) }
        catch { throw WeightExportError.malformed("graph decode: \(error)") }
        _ = try LeviathanCore.validate(graph)
        guard manifest.inputDim == graph.inputDim else {
            throw WeightExportError.inconsistentOpTable(
                "manifest inputDim \(manifest.inputDim) != graph \(graph.inputDim)")
        }

        // The op table must exactly match the graph's dense layout.
        let layout = expectedLayout(graph)
        guard manifest.ops.count == layout.count else {
            throw WeightExportError.inconsistentOpTable(
                "manifest ops \(manifest.ops.count) != dense ops \(layout.count)")
        }
        for (op, step) in zip(manifest.ops, layout) {
            guard op.inDim == step.inDim, op.outDim == step.outDim,
                  op.weightFloatOffset == step.wOff, op.weightFloatCount == step.wCount,
                  op.biasFloatOffset == step.bOff else {
                throw WeightExportError.inconsistentOpTable("op table disagrees with graph layout")
            }
        }
        var outputDim = graph.inputDim
        for op in graph.ops {
            if case let .dense(outDim) = op { outputDim = outDim }
        }
        guard manifest.refInput.count == graph.inputDim,
              manifest.refOutput.count == outputDim else {
            throw WeightExportError.malformed("reference vectors wrong size")
        }
        let refInput = manifest.refInput.map { Float($0) }
        let refOutput = manifest.refOutput.map { Float($0) }
        guard refInput.allSatisfy({ $0.isFinite }),
              refOutput.allSatisfy({ $0.isFinite }) else {
            throw WeightExportError.malformed("reference values non-finite in Float")
        }

        // Hash validated; decode explicit little-endian float32.
        let raw = [UInt8](weightsData)
        var weights = [Float](repeating: 0, count: raw.count / 4)
        for i in 0..<weights.count {
            let bits = UInt32(raw[i * 4]) | UInt32(raw[i * 4 + 1]) << 8
                     | UInt32(raw[i * 4 + 2]) << 16 | UInt32(raw[i * 4 + 3]) << 24
            weights[i] = Float(bitPattern: bits)
        }
        guard weights.allSatisfy({ $0.isFinite }) else {
            throw WeightExportError.malformed("weight blob contains non-finite floats")
        }
        return WeightExport(
            graph: graph, weights: weights, weightsSha256: actualHash,
            refInput: refInput, refOutput: refOutput)
    }
}
