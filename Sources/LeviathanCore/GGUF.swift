import Foundation

/// Errors from the original GGUF container reader. Every untrusted integer
/// from the file is range-checked; nothing traps or overflows.
public enum GGUFError: Error, Equatable, Sendable {
    case badMagic
    case unsupportedVersion(UInt32)
    /// A read went past the end of the file, the bounded header region,
    /// or a declared length was impossible.
    case truncated(String)
    /// A metadata value type code is unknown (nested arrays included).
    case unknownValueType(UInt32)
    /// An untrusted integer did not fit or produced an overflowing product.
    case integerOverflow(String)
    /// A tensor's geometry is inconsistent (zero dims, ragged blocks,
    /// misaligned offset, overlapping storage, duplicate name).
    case badTensor(String)
    /// The tensor data region does not fit in the file.
    case dataBeyondEndOffile(String)
    case missingMetadata(String)
    case missingTensor(String)
    /// A GGML tensor type this reader does not support was encountered.
    case unsupportedTensorType(UInt32)
}

/// GGUF metadata value types (ggml public format spec).
public enum GGUFValueType: UInt32, Sendable {
    case uint8 = 0, int8 = 1, uint16 = 2, int16 = 3
    case uint32 = 4, int32 = 5, float32 = 6, bool = 7
    case string = 8, array = 9, uint64 = 10, int64 = 11, float64 = 12
}

/// A parsed metadata value. Arrays keep decoded elements per kind.
public enum GGUFValue: Equatable, Sendable {
    case uint8(UInt8), int8(Int8), uint16(UInt16), int16(Int16)
    case uint32(UInt32), int32(Int32), float32(Float), bool(Bool)
    case string(String)
    case uint64(UInt64), int64(Int64), float64(Double)
    case array(elementType: GGUFValueType, count: Int,
               uints: [UInt64], ints: [Int64], floats: [Double],
               strings: [String], bools: [Bool])
}

/// One tensor directory entry. `offset` is relative to the tensor data
/// region and aligned per the file's alignment setting.
public struct GGUFTensorInfo: Equatable, Sendable {
    public let name: String
    /// GGML dimension order: dims[0] is the innermost (fastest-varying)
    /// dimension, as in ggml (`ne[0]`).
    public let dims: [Int]
    /// Raw GGML type code, e.g. 0=F32, 1=F16, 2=Q4_0.
    public let ggmlType: UInt32
    public let offset: Int
}

/// GGML tensor type codes this reader understands. Everything else is
/// rejected explicitly rather than guessed.
public enum GGMLTensorType: UInt32, Sendable {
    case float32 = 0
    case float16 = 1
    case q4_0 = 2
    case q6_K = 14

    public var elementsPerBlock: Int {
        switch self {
        case .float32, .float16: return 1
        case .q4_0: return 32
        case .q6_K: return 256
        }
    }

    public var bytesPerBlock: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .q4_0: return 18   // 2-byte FP16 delta + 16 bytes of nibbles
        case .q6_K: return 210  // FP16 d + 16 int8 scales + 128 ql + 64 qh
        }
    }

    public var isQuantized: Bool { elementsPerBlock > 1 }

    /// Checked byte count for `elements` values; nil on overflow or a
    /// ragged quantized block.
    public static func byteCount(type: GGMLTensorType, elements: Int) -> Int? {
        guard elements > 0, elements % type.elementsPerBlock == 0 else { return nil }
        let blocks = elements / type.elementsPerBlock
        let (product, overflow) = blocks.multipliedReportingOverflow(by: type.bytesPerBlock)
        return overflow ? nil : product
    }
}

/// A parsed GGUF file: header, metadata, tensor directory, and a
/// memory-mapped, zero-copy view of the bytes. Tensor payloads are handed
/// out as `Data` slices of the mapping — the file is never copied.
public struct GGUFFile: Sendable {
    public let version: UInt32
    public let metadata: [String: GGUFValue]
    public let tensors: [GGUFTensorInfo]
    /// Byte offset of the tensor data region in the file.
    public let dataOffset: Int
    /// General alignment (metadata `general.alignment`, default 32).
    public let alignment: Int
    public let byteCount: Int
    /// Memory-mapped (or caller-provided) file bytes; slices only.
    let data: Data

    public func metadataString(_ key: String) throws -> String {
        guard case let .string(s)? = metadata[key] else {
            throw GGUFError.missingMetadata(key)
        }
        return s
    }

    public func metadataUInt(_ key: String) throws -> UInt64 {
        guard let value = metadata[key] else { throw GGUFError.missingMetadata(key) }
        switch value {
        case let .uint8(v): return UInt64(v)
        case let .uint16(v): return UInt64(v)
        case let .uint32(v): return UInt64(v)
        case let .uint64(v): return v
        case let .int8(v): return v >= 0 ? UInt64(v) : 0
        case let .int16(v): return v >= 0 ? UInt64(v) : 0
        case let .int32(v): return v >= 0 ? UInt64(v) : 0
        case let .int64(v): return v >= 0 ? UInt64(v) : 0
        default: throw GGUFError.missingMetadata("\(key) (not an integer)")
        }
    }

    public func tensor(_ name: String) throws -> GGUFTensorInfo {
        guard let info = tensors.first(where: { $0.name == name }) else {
            throw GGUFError.missingTensor(name)
        }
        return info
    }

    /// File byte range of a tensor's storage, after validating type,
    /// element count, alignment, and file bounds.
    public func tensorRange(_ info: GGUFTensorInfo) throws -> Range<Int> {
        guard let type = GGMLTensorType(rawValue: info.ggmlType) else {
            throw GGUFError.unsupportedTensorType(info.ggmlType)
        }
        var elements = 1
        for d in info.dims {
            guard d > 0 else { throw GGUFError.badTensor("tensor \(info.name): zero dimension") }
            let (p, overflow) = elements.multipliedReportingOverflow(by: d)
            if overflow || p > GGUFReader.maxTensorElements {
                throw GGUFError.integerOverflow("tensor \(info.name): element count")
            }
            elements = p
        }
        guard let bytes = GGMLTensorType.byteCount(type: type, elements: elements) else {
            throw GGUFError.badTensor("tensor \(info.name): element count \(elements) is not a multiple of the \(type) block")
        }
        guard info.offset % alignment == 0 else {
            throw GGUFError.badTensor("tensor \(info.name): offset \(info.offset) misaligned")
        }
        let (start, overflow1) = dataOffset.addingReportingOverflow(info.offset)
        let (end, overflow2) = overflow1
            ? (0, true)
            : start.addingReportingOverflow(bytes)
        guard !overflow1, !overflow2, end <= byteCount else {
            // Only the checked values are reported; recomputing the sum
            // for the message could itself overflow.
            let safeStart = overflow1 ? -1 : start
            let safeEnd = overflow1 || overflow2 ? -1 : end
            throw GGUFError.dataBeyondEndOffile(
                "tensor \(info.name): needs bytes [\(safeStart), \(safeEnd)) but file is \(byteCount) bytes")
        }
        return start..<end
    }

    /// Zero-copy slice of a tensor's storage.
    public func tensorData(_ info: GGUFTensorInfo) throws -> Data {
        data[try tensorRange(info)]
    }
}

/// Original little-endian GGUF reader over memory-mapped bytes. Written
/// from the public ggml GGUF specification; no ggml code or weights are
/// included. All untrusted integers are checked before use.
public enum GGUFReader {

    public static let magic: [UInt8] = Array("GGUF".utf8)
    /// Hard caps on untrusted counts, lengths, and the header region.
    public static let maxEntries = 100_000
    public static let maxStringLength = 1 << 20
    public static let maxArrayElements = 10_000_000
    /// The header + tensor directory must fit well inside this; real
    /// model headers are a few hundred KiB.
    public static let maxHeaderBytes = 256 << 20
    public static let maxTensorElements = 1 << 40
    public static let maxAlignment = 1 << 20

    /// Maps the file read-only; no full-file copy is ever made.
    public static func read(_ url: URL) throws -> GGUFFile {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe, .alwaysMapped])
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> GGUFFile {
        var cursor = Cursor(data: data)
        guard cursor.remainingBytes >= 12 else {
            throw GGUFError.truncated("header")
        }
        guard try cursor.readBytes(4).elementsEqual(magic) else { throw GGUFError.badMagic }
        let version = try cursor.readUInt32()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }
        let tensorCount = try cursor.readUInt64()
        let kvCount = try cursor.readUInt64()
        guard tensorCount <= maxEntries, kvCount <= maxEntries else {
            throw GGUFError.truncated("implausible tensor/kv counts")
        }

        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<tensorCountOr(kvCount) {
            let key = try cursor.readString()
            guard metadata[key] == nil else {
                throw GGUFError.truncated("duplicate metadata key \(key)")
            }
            let value = try cursor.readValue()
            metadata[key] = value
        }

        var tensors: [GGUFTensorInfo] = []
        var seenNames = Set<String>()
        for _ in 0..<tensorCountOr(tensorCount) {
            let name = try cursor.readString()
            guard !seenNames.contains(name) else {
                throw GGUFError.badTensor("duplicate tensor name \(name)")
            }
            seenNames.insert(name)
            let nDims = try cursor.readUInt32()
            guard nDims >= 1, nDims <= 4 else {
                throw GGUFError.badTensor("tensor \(name): bad dim count \(nDims)")
            }
            var dims: [Int] = []
            for _ in 0..<nDims {
                guard let d = Int(exactly: try cursor.readUInt64()), d > 0,
                      d <= maxTensorElements else {
                    throw GGUFError.badTensor("tensor \(name): invalid dimension")
                }
                dims.append(d)
            }
            let type = try cursor.readUInt32()
            guard let offset = Int(exactly: try cursor.readUInt64()) else {
                throw GGUFError.integerOverflow("tensor \(name): offset")
            }
            tensors.append(GGUFTensorInfo(name: name, dims: dims, ggmlType: type, offset: offset))
        }

        var alignment = 32
        if case let .uint32(v)? = metadata["general.alignment"] {
            guard v > 0, v <= UInt32(maxAlignment) else {
                throw GGUFError.truncated("invalid general.alignment \(v)")
            }
            alignment = Int(v)
        }
        // Checked alignment rounding of the data start.
        let pos = cursor.position
        let dataOffset: Int
        if pos > Int.max - alignment {
            throw GGUFError.integerOverflow("data offset")
        }
        let padded = (pos + alignment - 1) / alignment * alignment
        guard padded >= pos else { throw GGUFError.integerOverflow("data offset") }
        dataOffset = padded

        let file = GGUFFile(version: version, metadata: metadata, tensors: tensors,
                            dataOffset: dataOffset, alignment: alignment,
                            byteCount: data.count, data: data)
        // Validate every tensor immediately — geometry, alignment, file
        // bounds — and reject overlapping storage, so a malformed
        // directory can never produce out-of-range or aliased reads.
        var ranges: [(String, Range<Int>)] = []
        for info in tensors {
            let r = try file.tensorRange(info)
            ranges.append((info.name, r))
        }
        if ranges.count > 1 {
            ranges.sort { $0.1.lowerBound < $1.1.lowerBound }
            for i in 1..<ranges.count {
                guard ranges[i - 1].1.upperBound <= ranges[i].1.lowerBound else {
                    throw GGUFError.badTensor(
                        "tensor \(ranges[i].0) overlaps \(ranges[i - 1].0)")
                }
            }
        }
        return file
    }

    private static func tensorCountOr(_ v: UInt64) -> Int {
        Int(clamping: v)
    }

    /// Bounds-checked cursor over the mapped bytes; never copies the file.
    struct Cursor {
        let data: Data
        var position = 0
        init(data: Data) { self.data = data }

        var remainingBytes: Int { data.count - position }

        mutating func readBytes(_ count: Int) throws -> Data {
            guard count >= 0, remainingBytes >= count, position <= maxHeaderBytes else {
                throw GGUFError.truncated("read \(count) at \(position)")
            }
            defer { position += count }
            return data[position..<(position + count)]
        }

        mutating func readUInt32() throws -> UInt32 {
            let b = try readBytes(4)
            let i = b.startIndex
            return UInt32(b[i]) | UInt32(b[i + 1]) << 8
                 | UInt32(b[i + 2]) << 16 | UInt32(b[i + 3]) << 24
        }

        mutating func readUInt64() throws -> UInt64 {
            var v: UInt64 = 0
            let b = try readBytes(8)
            let i = b.startIndex
            for j in 0..<8 { v |= UInt64(b[i + j]) << (8 * UInt64(j)) }
            return v
        }

        mutating func readString() throws -> String {
            let length = Int(clamping: try readUInt64())
            guard length >= 0, length <= maxStringLength else {
                throw GGUFError.truncated("string length beyond cap")
            }
            let raw = try readBytes(length)
            guard let s = String(data: raw, encoding: .utf8) else {
                throw GGUFError.truncated("invalid UTF-8 string")
            }
            return s
        }

        mutating func readValue() throws -> GGUFValue {
            let typeCode = try readUInt32()
            guard let type = GGUFValueType(rawValue: typeCode) else {
                throw GGUFError.unknownValueType(typeCode)
            }
            return try readPayload(type)
        }

        /// Reads a value of a known type — used both for top-level
        /// entries (after their type code) and for array elements, which
        /// in GGUF are stored as a bare payload with no per-element type.
        mutating func readPayload(_ type: GGUFValueType) throws -> GGUFValue {
            switch type {
            case .uint8: return .uint8(try readBytes(1).first!)
            case .int8: return .int8(Int8(bitPattern: try readBytes(1).first!))
            case .uint16:
                let b = try readBytes(2); let i = b.startIndex
                return .uint16(UInt16(b[i]) | UInt16(b[i + 1]) << 8)
            case .int16:
                let b = try readBytes(2); let i = b.startIndex
                return .int16(Int16(bitPattern: UInt16(b[i]) | UInt16(b[i + 1]) << 8))
            case .uint32: return .uint32(try readUInt32())
            case .int32: return .int32(Int32(bitPattern: try readUInt32()))
            case .float32: return .float32(Float(bitPattern: try readUInt32()))
            case .bool: return .bool(try readBytes(1).first! != 0)
            case .string: return .string(try readString())
            case .uint64: return .uint64(try readUInt64())
            case .int64: return .int64(Int64(bitPattern: try readUInt64()))
            case .float64: return .float64(Double(bitPattern: try readUInt64()))
            case .array:
                let elementTypeCode = try readUInt32()
                guard let elementType = GGUFValueType(rawValue: elementTypeCode),
                      elementType != .array else {
                    throw GGUFError.unknownValueType(elementTypeCode)
                }
                let countRaw = try readUInt64()
                guard countRaw <= maxArrayElements, let count = Int(exactly: countRaw) else {
                    throw GGUFError.truncated("array too long")
                }
                var uints: [UInt64] = [], ints: [Int64] = [], floats: [Double] = []
                var strings: [String] = [], bools: [Bool] = []
                for _ in 0..<count {
                    switch try readPayload(elementType) {
                    case let .uint8(v): uints.append(UInt64(v)); ints.append(Int64(v))
                    case let .int8(v): ints.append(Int64(v))
                    case let .uint16(v): uints.append(UInt64(v)); ints.append(Int64(v))
                    case let .int16(v): ints.append(Int64(v))
                    case let .uint32(v): uints.append(UInt64(v)); ints.append(Int64(v))
                    case let .int32(v): ints.append(Int64(v))
                    case let .float32(v): floats.append(Double(v))
                    case let .bool(v): bools.append(v)
                    case let .string(v): strings.append(v)
                    case let .uint64(v): uints.append(v); ints.append(Int64(clamping: v))
                    case let .int64(v): ints.append(v)
                    case let .float64(v): floats.append(v)
                    case .array: throw GGUFError.unknownValueType(type.rawValue)
                    }
                }
                return .array(elementType: elementType, count: count, uints: uints,
                              ints: ints, floats: floats, strings: strings, bools: bools)
            }
        }
    }
}
