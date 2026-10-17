import Foundation

/// Serialized artifact header. JSON with sorted keys and stable field order,
/// produced only by `JSONEncoder(outputFormatting: [.sortedKeys])`, so the
/// same program always serializes to identical bytes and the digest is a
/// function of the program alone.
public struct ArtifactHeader: Codable, Equatable, Sendable {
    public var magic: String            // "LVI1"
    public var version: Int             // artifact format version, currently 1
    public var inputDim: Int
    public var outputDim: Int
    public var bufferCount: Int
    public var weightCount: Int
    public var steps: [LoweredOp]
    public var seed: UInt64
}

public enum ArtifactError: Error, Equatable, Sendable {
    case badMagic
    case unsupportedVersion(Int)
    case truncated
    case digestMismatch
    case weightCountMismatch(Int, Int)
    case exceedsWeightByteLimit(Int)
    case badPlan(String)
    case nonFiniteWeight(Int)
}

/// On-disk compiled artifact. Layout, all little-endian:
///
/// | offset | contents                                    |
/// |--------|---------------------------------------------|
/// | 0      | 4-byte magic `LVI1`                          |
/// | 4      | 4-byte UInt32 header byte length             |
/// | 8      | canonical header JSON bytes                  |
/// | 8+L    | weights blob, `weightCount` float32 values   |
/// | end-32 | SHA-256 digest of every preceding byte       |
///
/// The digest is an integrity check (detects accidental corruption and
/// byte-level tampering); it is not a digital signature and provides no
/// authentication of the author. Loading recomputes the digest and then
/// re-validates the plan semantically, so even a correctly checksummed
/// malformed header is rejected.
public struct CompiledArtifact: Sendable {
    public static let magic = "LVI1"
    public static let version = 1

    public let header: ArtifactHeader
    public let weights: [Float]
    public let digest: String

    /// Build an artifact from a source graph by lowering it with a
    /// deterministic seed. Validates the graph and computes the cumulative
    /// dense weight+bias element count from the validated dimensions,
    /// rejecting totals over the weight budget BEFORE any weights are
    /// synthesized or allocated.
    public init(graph: GraphIR, seed: UInt64) throws {
        let dims = try validate(graph)
        var totalElements = 0
        for (index, op) in graph.ops.enumerated() {
            if case let .dense(outDim) = op {
                totalElements += dims[index] * outDim + outDim  // weights + bias
            }
        }
        guard totalElements <= Limits.maxWeightBytes / 4 else {
            throw ArtifactError.exceedsWeightByteLimit(totalElements * 4)
        }
        let lowered = LoweredProgram(graph: graph, seed: seed)
        self.header = ArtifactHeader(
            magic: Self.magic,
            version: Self.version,
            inputDim: lowered.inputDim,
            outputDim: lowered.outputDim,
            bufferCount: lowered.bufferCount,
            weightCount: lowered.weights.count,
            steps: lowered.steps,
            seed: seed)
        self.weights = lowered.weights
        self.digest = SHA256.hex(SHA256.hash(Self.bodyBytes(header: header, weights: weights)))
    }

    /// Parse, integrity-check, and semantically validate artifact bytes.
    public init(bytes: [UInt8]) throws {
        guard bytes.count >= 8 + 32 else { throw ArtifactError.truncated }
        guard String(bytes: bytes[0..<4], encoding: .ascii) == Self.magic else {
            throw ArtifactError.badMagic
        }
        let headerLength = Int(bytes[4]) | Int(bytes[5]) << 8
            | Int(bytes[6]) << 16 | Int(bytes[7]) << 24
        // headerLength is attacker-controlled; bound it before any use.
        guard headerLength >= 2, 8 + headerLength + 32 <= bytes.count,
              headerLength <= Limits.maxWeightBytes else {
            throw ArtifactError.truncated
        }
        let weightsStart = 8 + headerLength

        let storedDigest = Array(bytes[(bytes.count - 32)...])
        let recomputed = SHA256.hash(Array(bytes[..<(bytes.count - 32)]))
        guard storedDigest == recomputed else { throw ArtifactError.digestMismatch }

        let decoder = JSONDecoder()
        let header = try decoder.decode(
            ArtifactHeader.self, from: Data(bytes[8..<weightsStart]))
        guard header.magic == Self.magic else { throw ArtifactError.badMagic }
        guard header.version == Self.version else {
            throw ArtifactError.unsupportedVersion(header.version)
        }

        // Bound weightCount before multiplying to avoid overflow and the
        // UInt64 conversion trap on negative values; the error payload
        // saturates instead of wrapping for huge decoded counts.
        guard header.weightCount >= 0,
              header.weightCount <= Limits.maxWeightBytes / 4 else {
            let requested: UInt64 = header.weightCount >= 0
                ? UInt64(header.weightCount) &* 4 : 0
            let safe = requested > UInt64(Int.max) ? Int.max : Int(requested)
            throw ArtifactError.exceedsWeightByteLimit(safe)
        }
        let weightBytes = bytes[weightsStart..<(bytes.count - 32)]
        guard weightBytes.count == header.weightCount * 4 else {
            throw ArtifactError.weightCountMismatch(weightBytes.count / 4, header.weightCount)
        }
        var weights = [Float](repeating: 0, count: header.weightCount)
        _ = weights.withUnsafeMutableBytes { weightBytes.copyBytes(to: $0) }

        try Self.validatePlan(header: header, weights: weights)

        self.header = header
        self.weights = weights
        self.digest = SHA256.hex(storedDigest)
    }

    /// Semantic validation of a decoded plan, independent of the digest:
    /// bounds every dimension, step count, buffer index and weight range,
    /// checks the buffer chain is topologically consistent, and rejects
    /// non-finite weights.
    static func validatePlan(header: ArtifactHeader, weights: [Float]) throws {
        func bad(_ message: String) -> ArtifactError { .badPlan(message) }

        guard header.inputDim > 0, header.inputDim <= Limits.maxDim else {
            throw bad("inputDim \(header.inputDim) out of range")
        }
        guard header.outputDim > 0, header.outputDim <= Limits.maxDim else {
            throw bad("outputDim \(header.outputDim) out of range")
        }
        guard header.steps.count > 0, header.steps.count <= Limits.maxOps else {
            throw bad("step count \(header.steps.count) out of range")
        }
        guard header.bufferCount > 0, header.bufferCount <= Limits.maxOps + 1 else {
            throw bad("buffer count \(header.bufferCount) out of range")
        }
        guard weights.count == header.weightCount else {
            throw bad("weight blob does not match weightCount")
        }

        var currentBuffer = 0
        var currentDim = header.inputDim
        var lastDenseOutput: Int? = nil
        // Canonical buffer allocation: buffer 0 is the graph input; every
        // dense op must allocate the NEXT fresh buffer (1, 2, ...), so no
        // plan can alias its input, resurrect a stale buffer, or declare
        // buffers that are never initialized.
        var allocatedBuffers = 1

        for (position, step) in header.steps.enumerated() {
            switch step {
            case let .dense(_, inDim, outDim, inBuf, outBuf, wOff, wCount, bOff):
                guard inDim > 0, inDim <= Limits.maxDim,
                      outDim > 0, outDim <= Limits.maxDim else {
                    throw bad("step \(position): dims out of range")
                }
                guard wCount == inDim * outDim, wCount <= Limits.maxTensorElements else {
                    throw bad("step \(position): weight count \(wCount) invalid")
                }
                // Overflow-safe: bound each offset by the blob size before
                // any addition, so Int.max offsets cannot wrap around.
                guard wOff >= 0, bOff >= 0, wOff <= weights.count, bOff <= weights.count,
                      wCount >= 0, wCount <= weights.count, outDim <= weights.count,
                      wOff <= weights.count - wCount,
                      bOff <= weights.count - outDim else {
                    throw bad("step \(position): weight offsets out of range")
                }
                guard (0..<header.bufferCount).contains(inBuf),
                      (0..<header.bufferCount).contains(outBuf) else {
                    throw bad("step \(position): buffer index out of range")
                }
                // Topological consistency: reads the chain head at the
                // current feature width and allocates the next fresh
                // buffer — never buffer 0, a reused earlier buffer, or a
                // buffer out of canonical order.
                guard inBuf == currentBuffer, inDim == currentDim else {
                    throw bad("step \(position): dense does not read chain head")
                }
                guard outBuf == allocatedBuffers else {
                    throw bad("step \(position): dense output buffer is not the next fresh buffer")
                }
                currentBuffer = outBuf
                currentDim = outDim
                lastDenseOutput = outBuf
                allocatedBuffers += 1
            case let .relu(_, buf):
                guard (0..<header.bufferCount).contains(buf), buf == currentBuffer else {
                    throw bad("step \(position): relu buffer is not chain head")
                }
            }
        }

        guard header.bufferCount == allocatedBuffers else {
            throw bad("buffer count does not match allocated buffers")
        }

        guard currentDim == header.outputDim else {
            throw bad("declared outputDim does not match plan")
        }
        let expectedOutput = lastDenseOutput ?? 0
        guard currentBuffer == expectedOutput else {
            throw bad("output buffer is not the last dense output")
        }

        for (index, value) in weights.enumerated() where !value.isFinite {
            throw ArtifactError.nonFiniteWeight(index)
        }
    }

    public var program: LoweredProgram {
        LoweredProgram(
            inputDim: header.inputDim, outputDim: header.outputDim,
            bufferCount: header.bufferCount, steps: header.steps, weights: weights)
    }

    public func serialized() -> [UInt8] {
        var out = Self.bodyBytes(header: header, weights: weights)
        out.append(contentsOf: SHA256.hash(out))
        return out
    }

    public func save(to url: URL) throws {
        try Data(serialized()).write(to: url, options: .atomic)
    }

    /// Loads an artifact from disk, bounding the file size before any
    /// allocation so a huge file cannot exhaust memory.
    public static func load(from url: URL) throws -> CompiledArtifact {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let fileSize = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard fileSize >= 0, fileSize <= Int64(Limits.maxArtifactBytes) else {
            throw ArtifactError.exceedsWeightByteLimit(Int(fileSize))
        }
        return try CompiledArtifact(bytes: Array(try Data(contentsOf: url)))
    }

    private static func bodyBytes(header: ArtifactHeader, weights: [Float]) -> [UInt8] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try! encoder.encode(header)
        var out = Array(Self.magic.utf8)
        out.append(contentsOf: UInt32(headerData.count).littleEndianBytes)
        out.append(contentsOf: Array(headerData))
        return out + weights.flatMap { $0.bitPattern.littleEndianBytes }
    }
}

extension UInt32 {
    public var littleEndianBytes: [UInt8] {
        let v = self.littleEndian
        return [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)]
    }
}
