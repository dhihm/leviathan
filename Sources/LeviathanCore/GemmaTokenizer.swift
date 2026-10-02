// Gemma 4 tokenizer (GGUF `tokenizer.ggml.model = "gemma4"`).
//
// Original Swift implementation of the tokenization semantics used by the
// Gemma 4 GGUF runtime, pinned against public licensed reference material
// (llama.cpp MIT llama-vocab.cpp, semantics only, no source copied):
// SPM-style BPE where spaces are escaped to U+2581 (▁), the only
// pre-split is on newline runs, merges follow the GGUF merge-list rank
// with a lowest-(rank, left-index) priority queue, unknown final symbols
// fall back to `<0xXX>` byte tokens (uppercase hex), BOS is always
// prepended, EOS is never appended, and CONTROL/USER_DEFINED/UNKNOWN
// token texts are partitioned out of raw text before BPE. All piece and
// merge comparisons are BYTE-EXACT ([UInt8]) — Swift String equality is
// Unicode-canonicalizing and would wrongly merge NFC/NFD-distinct
// pieces. Duplicate pieces resolve LAST-WINS, matching the reference's
// ascending-id map insertion.

import Foundation

public enum GemmaTokenizerError: Error, CustomStringConvertible {
    case unsupportedModel(String)
    case missingMetadata(String)
    case inconsistentMetadata(String)
    case invalidMerge(String)
    case missingByteToken(String)
    case unmappedSymbol(String)
    case malformedTokenID(Int)
    case inputTooLarge(Int)

    public var description: String {
        switch self {
        case .unsupportedModel(let m): return "unsupported tokenizer model: \(m)"
        case .missingMetadata(let k): return "tokenizer metadata missing: \(k)"
        case .inconsistentMetadata(let m): return "tokenizer metadata inconsistent: \(m)"
        case .invalidMerge(let m): return "invalid merge entry: \(m)"
        case .missingByteToken(let m): return "byte-fallback token missing from vocab: \(m)"
        case .unmappedSymbol(let m): return "tokenization produced unmapped symbol: \(m)"
        case .malformedTokenID(let id): return "malformed token id \(id)"
        case .inputTooLarge(let n): return "input exceeds bound (\(n) bytes)"
        }
    }
}

public struct GemmaTokenizer: Sendable {
    public enum TokenType: Int, Sendable {
        case unknown = 2
        case normal = 1
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    /// Hard bound on encode input size (defense in depth for callers).
    public static let maxInputBytes = 1 << 20

    public let bosID: Int
    public let eosID: Int
    public let unkID: Int
    public let vocabSize: Int

    private let pieces: [String]
    private let types: [TokenType?]
    private let idByBytes: [[UInt8]: Int]
    private let mergeRank: [Bigram: Int]
    /// Special (CONTROL/USER_DEFINED/UNKNOWN) pieces for text partitioning,
    /// longest byte length first — matching the reference's special-token
    /// cache ordering.
    private let specialsByLength: [(piece: [UInt8], id: Int)]

    private struct Bigram: Hashable {
        var left: [UInt8]
        var right: [UInt8]
    }

    public init(gguf: GGUFFile) throws {
        guard case let .string(model)? = gguf.metadata["tokenizer.ggml.model"],
              model == "gemma4" else {
            throw GemmaTokenizerError.unsupportedModel(
                (try? gguf.metadataString("tokenizer.ggml.model")) ?? "<absent>")
        }
        guard case let .array(_, tokenCount, _, _, _, tokens, _)? =
                gguf.metadata["tokenizer.ggml.tokens"] else {
            throw GemmaTokenizerError.missingMetadata("tokenizer.ggml.tokens")
        }
        guard case let .array(_, typeCount, _, typeInts, _, _, _)? =
                gguf.metadata["tokenizer.ggml.token_type"],
              typeCount == tokenCount else {
            throw GemmaTokenizerError.missingMetadata("tokenizer.ggml.token_type")
        }
        guard case let .array(_, mergeCount, _, _, _, merges, _)? =
                gguf.metadata["tokenizer.ggml.merges"] else {
            throw GemmaTokenizerError.missingMetadata("tokenizer.ggml.merges")
        }
        func specialID(_ key: String) -> Int? {
            if case let .uint32(v)? = gguf.metadata[key] { return Int(v) }
            return nil
        }
        // Parsed array lengths must match their declared element counts.
        guard tokens.count == tokenCount, typeInts.count == tokenCount,
              merges.count == mergeCount else {
            throw GemmaTokenizerError.inconsistentMetadata("array length mismatch")
        }
        // The gemma4 runtime forces add_bos; a file that says otherwise is
        // not the contract this tokenizer implements. Spaces are handled
        // by the U+2581 escape, never a prefix flag.
        guard case let .bool(addBOS)? = gguf.metadata["tokenizer.ggml.add_bos_token"],
              addBOS else {
            throw GemmaTokenizerError.inconsistentMetadata(
                "tokenizer.ggml.add_bos_token must be true")
        }
        if case let .bool(addSpacePrefix)? = gguf.metadata["tokenizer.ggml.add_space_prefix"],
           addSpacePrefix {
            throw GemmaTokenizerError.inconsistentMetadata(
                "tokenizer.ggml.add_space_prefix must be false")
        }
        guard let bos = specialID("tokenizer.ggml.bos_token_id"),
              let eos = specialID("tokenizer.ggml.eos_token_id"),
              let unk = specialID("tokenizer.ggml.unknown_token_id"),
              bos < tokenCount, eos < tokenCount, unk < tokenCount else {
            throw GemmaTokenizerError.missingMetadata("bos/eos/unknown token id")
        }

        var idByBytes: [[UInt8]: Int] = [:]
        idByBytes.reserveCapacity(tokenCount)
        var pieces = [String]()
        pieces.reserveCapacity(tokenCount)
        for (i, piece) in tokens.enumerated() {
            let bytes = Array(piece.utf8)
            // Last-wins on duplicate pieces (reference inserts ascending).
            idByBytes[bytes] = i
            pieces.append(piece)
        }
        var types = [TokenType?]()
        types.reserveCapacity(tokenCount)
        for t in typeInts {
            guard let type = TokenType(rawValue: Int(t)) else {
                throw GemmaTokenizerError.inconsistentMetadata(
                    "unknown token_type code \(t)")
            }
            types.append(type)
        }
        // Byte-fallback contract: the complete 256-token set must exist,
        // and every BYTE-typed piece must have exact <0xXX> uppercase-hex
        // syntax so decode can never hit a malformed byte token.
        for v in 0..<256 where idByBytes[[UInt8(ascii: "<"), UInt8(ascii: "0"),
                                          UInt8(ascii: "x"),
                                          UInt8(hexDigit: UInt8(v >> 4)),
                                          UInt8(hexDigit: UInt8(v & 15)),
                                          UInt8(ascii: ">")]] == nil {
            throw GemmaTokenizerError.missingByteToken(String(format: "<0x%02X>", v))
        }
        for (i, t) in types.enumerated() where t == .byte {
            let bytes = Array(pieces[i].utf8)
            let wellFormed = bytes.count == 6 && bytes[0] == UInt8(ascii: "<")
                && bytes[1] == UInt8(ascii: "0") && bytes[2] == UInt8(ascii: "x")
                && isHexDigit(bytes[3]) && isHexDigit(bytes[4])
                && bytes[5] == UInt8(ascii: ">")
            guard wellFormed, let _ = UInt8(String(decoding: bytes[3..<5], as: UTF8.self),
                                            radix: 16) else {
                throw GemmaTokenizerError.inconsistentMetadata(
                    "malformed byte token id \(i): \(pieces[i])")
            }
        }

        var mergeRank: [Bigram: Int] = [:]
        mergeRank.reserveCapacity(mergeCount)
        for (rank, merge) in merges.enumerated() {
            let bytes = Array(merge.utf8)
            // The separator is the first 0x20 at byte offset >= 1, so
            // merge halves may themselves contain leading spaces. Entries
            // with no such separator (e.g. " //") become unmatchable empty
            // pairs in the reference and are skipped here identically.
            guard bytes.count >= 2, let separator = bytes.dropFirst().firstIndex(of: 0x20) else {
                continue
            }
            mergeRank[Bigram(left: Array(bytes[0..<separator]),
                             right: Array(bytes[(separator + 1)...]))] = rank
        }

        var specials: [([UInt8], Int)] = []
        for (i, t) in types.enumerated() {
            switch t {
            case .control, .userDefined, .unknown:
                let bytes = Array(pieces[i].utf8)
                if !bytes.isEmpty { specials.append((bytes, i)) }
            default: break
            }
        }
        specials.sort { $0.0.count > $1.0.count }

        self.pieces = pieces
        self.types = types
        self.idByBytes = idByBytes
        self.mergeRank = mergeRank
        self.specialsByLength = specials
        self.bosID = bos
        self.eosID = eos
        self.unkID = unk
        self.vocabSize = tokenCount
    }

    // MARK: encoding

    /// Encode text into token IDs with the model's forced BOS. Special
    /// token texts (CONTROL/USER_DEFINED/UNKNOWN pieces) occurring in the
    /// text are partitioned out and emitted as their exact IDs. Any input
    /// that cannot be represented exactly (missing byte-fallback token)
    /// is an error, never a silent drop.
    public func encode(_ text: String) throws -> [Int] {
        let bytes = Array(text.utf8)
        guard bytes.count <= Self.maxInputBytes else {
            throw GemmaTokenizerError.inputTooLarge(bytes.count)
        }
        var ids = [Int]()
        ids.reserveCapacity(bytes.count / 4 + 2)
        ids.append(bosID)
        for fragment in try partitionSpecials(bytes) {
            switch fragment {
            case .token(let id): ids.append(id)
            case .raw(let raw): try encodeFragment(raw, into: &ids)
            }
        }
        return ids
    }

    private enum Fragment {
        case token(Int)
        case raw([UInt8])
    }

    private func partitionSpecials(_ bytes: [UInt8]) throws -> [Fragment] {
        var fragments = [Fragment.raw(bytes)]
        for (piece, id) in specialsByLength {
            var result = [Fragment]()
            for fragment in fragments {
                guard case .raw(let s) = fragment else {
                    result.append(fragment)
                    continue
                }
                var i = 0
                var start = 0
                while i <= s.count - piece.count {
                    if s[i] == piece[0], Array(s[i..<i + piece.count]) == piece {
                        if i > start { result.append(.raw(Array(s[start..<i]))) }
                        result.append(.token(id))
                        i += piece.count
                        start = i
                    } else {
                        i += 1
                    }
                }
                if start < s.count { result.append(.raw(Array(s[start...]))) }
            }
            fragments = result
        }
        return fragments
    }

    /// One special-free fragment: escape 0x20 → ▁ (E2 96 81), split into
    /// newline/non-newline runs, BPE-merge each run, byte-fallback unknowns.
    private func encodeFragment(_ fragment: [UInt8], into ids: inout [Int]) throws {
        var escaped = [UInt8]()
        escaped.reserveCapacity(fragment.count * 3 / 2)
        for b in fragment {
            if b == 0x20 { escaped.append(contentsOf: [0xE2, 0x96, 0x81]) }
            else { escaped.append(b) }
        }
        var runStart = 0
        var i = 1
        while i <= escaped.count {
            let boundary = i == escaped.count
                || (escaped[i] == 0x0A) != (escaped[i - 1] == 0x0A)
            if boundary {
                try encodeRun(Array(escaped[runStart..<i]), into: &ids)
                runStart = i
            }
            i += 1
        }
    }

    private func encodeRun(_ run: [UInt8], into ids: inout [Int]) throws {
        // A pure-newline run that is exactly a vocab piece is emitted
        // directly (the reference's newline-run fix, llama.cpp PR #21343).
        if !run.isEmpty, run.allSatisfy({ $0 == 0x0A }), let id = idByBytes[run] {
            ids.append(id)
            return
        }

        // Symbols are one Unicode scalar's UTF-8 bytes each.
        struct Symbol {
            var bytes: [UInt8]
            var prev: Int
            var next: Int
            var isEmpty: Bool { bytes.isEmpty }
        }
        var symbols = [Symbol]()
        var i = 0
        while i < run.count {
            // Leading byte of a UTF-8 sequence encodes its own length.
            let first = run[i]
            let len = first < 0x80 ? 1 : first < 0xE0 ? 2 : first < 0xF0 ? 3 : 4
            symbols.append(Symbol(bytes: Array(run[i..<(i + len)]),
                                  prev: symbols.count - 1, next: -1))
            if symbols.count >= 2 {
                symbols[symbols.count - 2].next = symbols.count - 1
            }
            i += len
        }
        if symbols.isEmpty { return }

        // Min-heap keyed on (rank, left index): lowest rank merges first,
        // ties broken by the earlier symbol position.
        struct Entry: Comparable {
            var rank: Int
            var left: Int
            var right: Int
            static func < (a: Entry, b: Entry) -> Bool {
                a.rank < b.rank || (a.rank == b.rank && a.left < b.left)
            }
        }
        var heap = Heap<Entry>()
        func pushBigram(_ left: Int, _ right: Int) {
            guard left >= 0, right >= 0, !symbols[left].isEmpty, !symbols[right].isEmpty else {
                return
            }
            if let rank = mergeRank[Bigram(left: symbols[left].bytes,
                                           right: symbols[right].bytes)] {
                heap.push(Entry(rank: rank, left: left, right: right))
            }
        }
        for j in 1..<symbols.count { pushBigram(j - 1, j) }

        while let entry = heap.pop() {
            let left = entry.left, right = entry.right
            if symbols[left].isEmpty || symbols[right].isEmpty { continue }
            // Stale entries are skipped as in the reference (text match)
            // plus an adjacency check: the two symbols must still be
            // neighbors in the linked list.
            guard symbols[left].next == right,
                  mergeRank[Bigram(left: symbols[left].bytes,
                                   right: symbols[right].bytes)] == entry.rank else {
                continue
            }
            symbols[left].bytes += symbols[right].bytes
            symbols[right].bytes = []
            symbols[left].next = symbols[right].next
            if symbols[right].next >= 0 { symbols[symbols[right].next].prev = left }
            pushBigram(symbols[left].prev, left)
            pushBigram(left, symbols[left].next)
        }

        for j in 0..<symbols.count where !symbols[j].isEmpty {
            let piece = symbols[j].bytes
            if let id = idByBytes[piece] {
                ids.append(id)
            } else {
                for byte in piece {
                    let hex = [UInt8(ascii: "<"), UInt8(ascii: "0"), UInt8(ascii: "x"),
                               UInt8(hexDigit: byte >> 4), UInt8(hexDigit: byte & 15),
                               UInt8(ascii: ">")]
                    guard let id = idByBytes[hex] else {
                        throw GemmaTokenizerError.missingByteToken(
                            String(format: "<0x%02X>", byte))
                    }
                    ids.append(id)
                }
            }
        }
    }

    // MARK: decoding

    /// Decode token IDs to text following the runtime's detokenizer:
    /// special/user-defined pieces verbatim, control tokens suppressed,
    /// normal pieces with ▁ unescaped to a space, byte tokens as raw
    /// bytes. Out-of-range IDs are an error, never a silent skip.
    public func decode(_ ids: [Int]) throws -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard id >= 0, id < pieces.count else {
                throw GemmaTokenizerError.malformedTokenID(id)
            }
            switch types[id] {
            case .userDefined, .unknown, .unused:
                bytes.append(contentsOf: pieces[id].utf8)
            case .control:
                continue
            case .byte:
                let piece = pieces[id]
                guard let v = UInt8(piece.dropFirst(3).prefix(2), radix: 16) else {
                    throw GemmaTokenizerError.inconsistentMetadata(
                        "malformed byte token id \(id): \(piece)")
                }
                bytes.append(v)
            case .normal, nil:
                let piece = Array(pieces[id].utf8)
                var i = 0
                while i < piece.count {
                    // ▁ = E2 96 81 decodes to a single space.
                    if piece[i] == 0xE2, i + 2 < piece.count,
                       piece[i + 1] == 0x96, piece[i + 2] == 0x81 {
                        bytes.append(0x20)
                        i += 3
                    } else {
                        bytes.append(piece[i])
                        i += 1
                    }
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Tokens that end generation in the reference runtime for gemma4.
    public func isEndOfGeneration(_ id: Int) -> Bool {
        guard id >= 0, id < pieces.count else { return false }
        return id == eosID || pieces[id] == "<turn|>" || pieces[id] == "<|tool_response>"
    }
}

extension UInt8 {
    /// Uppercase hex digit for 0...15.
    init(hexDigit v: UInt8) {
        self = v < 10 ? UInt8(ascii: "0") + v : UInt8(ascii: "A") + v - 10
    }
}

private func isHexDigit(_ b: UInt8) -> Bool {
    (b >= UInt8(ascii: "0") && b <= UInt8(ascii: "9"))
        || (b >= UInt8(ascii: "A") && b <= UInt8(ascii: "F"))
}

/// Minimal binary min-heap (the tokenizer's merge priority queue).
struct Heap<Element: Comparable> {
    private var items = [Element]()

    var isEmpty: Bool { items.isEmpty }

    mutating func push(_ e: Element) {
        items.append(e)
        var i = items.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            if items[i] < items[p] { items.swapAt(i, p); i = p } else { break }
        }
    }

    mutating func pop() -> Element? {
        guard !items.isEmpty else { return nil }
        items.swapAt(0, items.count - 1)
        let top = items.removeLast()
        var i = 0
        while true {
            let l = 2 * i + 1, r = 2 * i + 2
            var m = i
            if l < items.count, items[l] < items[m] { m = l }
            if r < items.count, items[r] < items[m] { m = r }
            if m == i { break }
            items.swapAt(i, m)
            i = m
        }
        return top
    }
}
