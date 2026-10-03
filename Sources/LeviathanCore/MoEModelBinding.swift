// Phase 13 — binds a validated MoERegionIR (strict torch.export capture,
// independently re-validated in Swift) to the ACTUAL resident GGUF across
// ALL layers, and validates per-route expert slice addresses before any
// Metal enqueue. Model-only: no Metal, no payload access, no weight copies,
// no CPU dequantization. Route offsets are returned as immutable values for
// encoder.setBytes snapshots — never stored in shared mutable GPU buffers.
// Bounded to the one routed-region family; no performance claim is made.

import Foundation

/// Immutable, checked binding of the routed MoE region to the resident GGUF.
/// Only `bind` (against the real file) or a fully re-checked `Decodable`
/// path can produce a value; there is no public memberwise initializer.
/// File-level facts (tensor names, Q4_0 types, uniqueness) can only be
/// proven by `bind`; decoding re-checks every self-contained invariant.
public struct MoEModelBinding: Codable, Equatable, Sendable {
    public static let scratchBudgetBytes = 160 * 1024
    public static let architecture = "gemma4"
    public static let guAnchor = "blk.0.ffn_gate_up_exps.weight"
    public static let dnAnchor = "blk.0.ffn_down_exps.weight"
    static let maxLayers = 128, maxExperts = 4096, maxTopK = 64, maxDim = 8192

    public let graphSha256: String
    public let fileByteCount: Int
    /// Resident buffer maps the GGUF file at offset 0 (buffer offset == file offset).
    public let residentBufferLength: Int
    public let maxBufferLength: Int
    public let recommendedWorkingSetSize: Int
    public let layerCount: Int
    public let expertCount: Int
    public let topK: Int
    public let hidden: Int
    public let inter: Int
    public let gateUpOut: Int
    /// Bytes per expert slice: (inDim/32)*18*outDim (whole Q4_0 rows).
    public let guStride: Int
    public let dnStride: Int
    /// Absolute byte offset of blk.L's expert tensor in the resident buffer.
    public let guTensorStarts: [Int]
    public let dnTensorStarts: [Int]
    /// Once-per-engine scratch sizes (GU/ACT/DN), all topK slots retained.
    public let guScratchBytes: Int
    public let actScratchBytes: Int
    public let dnScratchBytes: Int
    public let scratchTotalBytes: Int

    enum CodingKeys: String, CodingKey {
        case graphSha256, fileByteCount, residentBufferLength, maxBufferLength
        case recommendedWorkingSetSize, layerCount, expertCount, topK, hidden, inter
        case gateUpOut, guStride, dnStride, guTensorStarts, dnTensorStarts
        case guScratchBytes, actScratchBytes, dnScratchBytes, scratchTotalBytes
    }

    private init(graphSha256: String, fileByteCount: Int, residentBufferLength: Int,
                 maxBufferLength: Int, recommendedWorkingSetSize: Int, layerCount: Int,
                 expertCount: Int, topK: Int, hidden: Int, inter: Int, gateUpOut: Int,
                 guStride: Int, dnStride: Int, guTensorStarts: [Int], dnTensorStarts: [Int],
                 guScratchBytes: Int, actScratchBytes: Int, dnScratchBytes: Int,
                 scratchTotalBytes: Int) {
        self.graphSha256 = graphSha256; self.fileByteCount = fileByteCount
        self.residentBufferLength = residentBufferLength; self.maxBufferLength = maxBufferLength
        self.recommendedWorkingSetSize = recommendedWorkingSetSize; self.layerCount = layerCount
        self.expertCount = expertCount; self.topK = topK; self.hidden = hidden
        self.inter = inter; self.gateUpOut = gateUpOut; self.guStride = guStride
        self.dnStride = dnStride; self.guTensorStarts = guTensorStarts
        self.dnTensorStarts = dnTensorStarts; self.guScratchBytes = guScratchBytes
        self.actScratchBytes = actScratchBytes; self.dnScratchBytes = dnScratchBytes
        self.scratchTotalBytes = scratchTotalBytes
    }

    /// Decoding never yields an unchecked binding: every invariant that does
    /// not need the GGUF itself is re-derived and must match exactly.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func i(_ k: CodingKeys) throws -> Int { try c.decode(Int.self, forKey: k) }
        func a(_ k: CodingKeys) throws -> [Int] { try c.decode([Int].self, forKey: k) }
        try self.init(
            graphSha256: c.decode(String.self, forKey: .graphSha256),
            fileByteCount: i(.fileByteCount), residentBufferLength: i(.residentBufferLength),
            maxBufferLength: i(.maxBufferLength),
            recommendedWorkingSetSize: i(.recommendedWorkingSetSize),
            layerCount: i(.layerCount), expertCount: i(.expertCount), topK: i(.topK),
            hidden: i(.hidden), inter: i(.inter), gateUpOut: i(.gateUpOut),
            guStride: i(.guStride), dnStride: i(.dnStride),
            guTensorStarts: a(.guTensorStarts), dnTensorStarts: a(.dnTensorStarts),
            guScratchBytes: i(.guScratchBytes), actScratchBytes: i(.actScratchBytes),
            dnScratchBytes: i(.dnScratchBytes), scratchTotalBytes: i(.scratchTotalBytes))
        try checkInvariants()
    }

    // MARK: - Binding against the actual GGUF

    public static func bind(region: MoERegionIR, file: GGUFFile, residentBufferLength: Int,
                            maxBufferLength: Int,
                            recommendedWorkingSetSize: Int) throws -> MoEModelBinding {
        try region.validate()   // all graph-semantic trust checks live there
        let reg = MoELoweringError.invalidRegion
        let arch = try gguf("general.architecture") {
            try file.metadataString("general.architecture")
        }
        guard arch == architecture else { throw reg("general.architecture \(arch) != \(architecture)") }
        let layers = try metaInt(file, "gemma4.block_count", maxLayers)
        let experts = try metaInt(file, "gemma4.expert_count", maxExperts)
        let topK = try metaInt(file, "gemma4.expert_used_count", maxTopK)
        let hidden = try metaInt(file, "gemma4.embedding_length", maxDim)
        let inter = try metaInt(file, "gemma4.expert_feed_forward_length", maxDim)
        let gateUpOut = try mul(2, inter, "2*inter", reg)
        guard region.hidden == hidden, region.expertIntermediate == inter,
              region.topK == topK, region.gateUpOut == gateUpOut else {
            throw reg("region (hidden \(region.hidden), inter \(region.expertIntermediate), "
                + "topK \(region.topK), gateUpOut \(region.gateUpOut)) != GGUF (\(hidden), "
                + "\(inter), \(topK), \(gateUpOut))")
        }
        guard topK <= experts else { throw reg("topK \(topK) > expert_count \(experts)") }
        guard region.guTensor == guAnchor, region.dnTensor == dnAnchor else {
            throw reg("anchors \(region.guTensor)/\(region.dnTensor) != \(guAnchor)/\(dnAnchor)")
        }
        guard maxBufferLength > 0, residentBufferLength > 0,
              residentBufferLength >= file.byteCount, residentBufferLength <= maxBufferLength else {
            throw MoELoweringError.invalidSchedule("resident buffer \(residentBufferLength) "
                + "must be in [file \(file.byteCount), maxBufferLength \(maxBufferLength)]")
        }
        var nameCounts: [String: Int] = [:]
        var guFamily = 0, dnFamily = 0
        for t in file.tensors {
            nameCounts[t.name, default: 0] += 1
            if t.name.hasSuffix(".ffn_gate_up_exps.weight") { guFamily += 1 }
            if t.name.hasSuffix(".ffn_down_exps.weight") { dnFamily += 1 }
        }
        guard guFamily == layers, dnFamily == layers else {
            throw reg("expert families \(guFamily)/\(dnFamily) tensors != block_count \(layers)")
        }
        let guStride = try sliceStride(inDim: hidden, outDim: gateUpOut, reg)
        let dnStride = try sliceStride(inDim: inter, outDim: hidden, reg)
        var guStarts: [Int] = [], dnStarts: [Int] = []
        for layer in 0..<layers {
            for (isGU, dims, stride) in [(true, [hidden, gateUpOut, experts], guStride),
                                         (false, [inter, hidden, experts], dnStride)] {
                let name = "blk.\(layer).\(isGU ? "ffn_gate_up_exps" : "ffn_down_exps").weight"
                guard nameCounts[name] == 1 else {
                    throw reg("tensor \(name) missing or duplicated (count \(nameCounts[name] ?? 0))")
                }
                let info = try gguf(name) { try file.tensor(name) }
                guard info.ggmlType == GGMLTensorType.q4_0.rawValue, info.dims == dims else {
                    throw reg("\(name): type \(info.ggmlType) dims \(info.dims) != Q4_0 \(dims)")
                }
                let range = try gguf(name) { try file.tensorRange(info) }
                let bytes = try mul(stride, experts, "\(name) bytes", reg)
                guard range.count == bytes, range.lowerBound >= 0,
                      range.upperBound <= residentBufferLength else {
                    throw reg("\(name): range \(range) != \(bytes) bytes inside resident buffer")
                }
                if isGU { guStarts.append(range.lowerBound) } else { dnStarts.append(range.lowerBound) }
            }
        }
        let s = try scratchBytes(topK: topK, gateUpOut: gateUpOut, inter: inter, hidden: hidden)
        let binding = MoEModelBinding(
            graphSha256: region.graphSha256, fileByteCount: file.byteCount,
            residentBufferLength: residentBufferLength, maxBufferLength: maxBufferLength,
            recommendedWorkingSetSize: recommendedWorkingSetSize, layerCount: layers,
            expertCount: experts, topK: topK, hidden: hidden, inter: inter,
            gateUpOut: gateUpOut, guStride: guStride, dnStride: dnStride,
            guTensorStarts: guStarts, dnTensorStarts: dnStarts, guScratchBytes: s.gu,
            actScratchBytes: s.act, dnScratchBytes: s.dn, scratchTotalBytes: s.total)
        try binding.checkInvariants()
        return binding
    }

    // MARK: - Per-route slice offsets (setBytes snapshots)

    /// Absolute resident-buffer byte offsets of the routed expert slices, in
    /// the EXACT given slot order (never sorted). Requires exactly topK
    /// distinct ids in 0..<expertCount; every span is re-checked here.
    public func offsets(forLayer layer: Int, expertIds: [Int]) throws -> (gu: [UInt64], dn: [UInt64]) {
        let bad = MoELoweringError.invalidSchedule
        guard layer >= 0, layer < layerCount, layer < guTensorStarts.count,
              layer < dnTensorStarts.count else {
            throw bad("layer \(layer) outside 0..<\(layerCount)")
        }
        guard topK > 0, expertIds.count == topK else {
            throw bad("route has \(expertIds.count) ids, expected exactly topK=\(topK)")
        }
        var seen = Set<Int>()
        var gu: [UInt64] = [], dn: [UInt64] = []
        gu.reserveCapacity(topK); dn.reserveCapacity(topK)
        for (slot, id) in expertIds.enumerated() {
            guard id >= 0, id < expertCount else {
                throw bad("slot \(slot): expert id \(id) outside 0..<\(expertCount)")
            }
            guard seen.insert(id).inserted else { throw bad("slot \(slot): duplicate expert id \(id)") }
            gu.append(try slice(guTensorStarts[layer], guStride, id))
            dn.append(try slice(dnTensorStarts[layer], dnStride, id))
        }
        return (gu, dn)
    }

    private func slice(_ start: Int, _ stride: Int, _ id: Int) throws -> UInt64 {
        let bad = MoELoweringError.invalidSchedule
        guard start >= 0, stride > 0, stride % GGMLTensorType.q4_0.bytesPerBlock == 0 else {
            throw bad("malformed slice geometry start \(start) stride \(stride)")
        }
        let abs = try Self.add(start, Self.mul(id, stride, "slice offset", bad), "slice offset", bad)
        let end = try Self.add(abs, stride, "slice end", bad)
        // Own-family tensor end (start + stride*expertCount), the file
        // bytes, and the resident buffer are ALL upper bounds.
        let familyEnd = try Self.add(start, try Self.mul(stride, expertCount, "family end", bad),
                                     "family end", bad)
        guard end <= familyEnd, end <= residentBufferLength, end <= fileByteCount,
              let v = UInt64(exactly: abs) else {
            throw bad("slice [\(abs), \(end)) outside family [\(start), \(familyEnd)), "
                + "buffer \(residentBufferLength) or file \(fileByteCount)")
        }
        return v
    }

    // MARK: - Invariants and checked arithmetic

    private func checkInvariants() throws {
        let bad = MoELoweringError.invalidSchedule
        guard moeIsLowercaseHex64(graphSha256) else { throw bad("graphSha256 form invalid") }
        guard (1...Self.maxLayers).contains(layerCount), (1...Self.maxExperts).contains(expertCount),
              (1...Self.maxTopK).contains(topK), topK <= expertCount,
              (1...Self.maxDim).contains(hidden), (1...Self.maxDim).contains(inter),
              hidden % 32 == 0, inter % 32 == 0 else { throw bad("geometry out of bounds") }
        guard try Self.mul(2, inter, "2*inter", bad) == gateUpOut,
              try Self.sliceStride(inDim: hidden, outDim: gateUpOut, bad) == guStride,
              try Self.sliceStride(inDim: inter, outDim: hidden, bad) == dnStride else {
            throw bad("derived gateUpOut/strides disagree with stored values")
        }
        guard maxBufferLength > 0, recommendedWorkingSetSize > 0, fileByteCount > 0,
              residentBufferLength >= fileByteCount, residentBufferLength <= maxBufferLength,
              guTensorStarts.count == layerCount, dnTensorStarts.count == layerCount else {
            throw bad("resident buffer/caps/layer table invalid")
        }
        let guBytes = try Self.mul(guStride, expertCount, "GU tensor bytes", bad)
        let dnBytes = try Self.mul(dnStride, expertCount, "DN tensor bytes", bad)
        var spans: [(Int, Int)] = []
        for (starts, bytes) in [(guTensorStarts, guBytes), (dnTensorStarts, dnBytes)] {
            for s in starts {
                let end = try Self.add(s, bytes, "tensor end", bad)
                guard s >= 0, end <= residentBufferLength, end <= fileByteCount else {
                    throw bad("tensor span [\(s), \(end)) outside resident buffer "
                        + "\(residentBufferLength) / file \(fileByteCount)")
                }
                spans.append((s, end))
            }
        }
        spans.sort { $0.0 < $1.0 }
        for k in spans.indices.dropFirst() where spans[k - 1].1 > spans[k].0 {
            throw bad("expert tensor spans overlap at \(spans[k].0)")
        }
        let s = try Self.scratchBytes(topK: topK, gateUpOut: gateUpOut, inter: inter, hidden: hidden)
        guard s.gu == guScratchBytes, s.act == actScratchBytes, s.dn == dnScratchBytes,
              s.total == scratchTotalBytes else { throw bad("scratch sizes disagree with geometry") }
        guard s.gu > 0, s.act > 0, s.dn > 0, s.total <= Self.scratchBudgetBytes,
              s.total <= recommendedWorkingSetSize, s.gu <= maxBufferLength,
              s.act <= maxBufferLength, s.dn <= maxBufferLength else {
            throw bad("scratch \(s.total) B exceeds \(Self.scratchBudgetBytes) B budget or device caps")
        }
    }

    static func scratchBytes(topK: Int, gateUpOut: Int, inter: Int,
                             hidden: Int) throws -> (gu: Int, act: Int, dn: Int, total: Int) {
        let bad = MoELoweringError.invalidSchedule, f = MemoryLayout<Float>.stride
        let gu = try mul(mul(topK, gateUpOut, "GU scratch", bad), f, "GU scratch", bad)
        let act = try mul(mul(topK, inter, "ACT scratch", bad), f, "ACT scratch", bad)
        let dn = try mul(mul(topK, hidden, "DN scratch", bad), f, "DN scratch", bad)
        return (gu, act, dn, try add(add(gu, act, "scratch total", bad), dn, "scratch total", bad))
    }

    /// The runner's resident mapping length for a file of `fileBytes`:
    /// the SAME page-rounding the runner's mmap uses (checked, shared
    /// formula — a divergent copy would make exact binding equality
    /// reject real artifacts). Returns nil on overflow/invalid size.
    public static func residentMappingLength(fileBytes: Int, pageSize: Int) -> Int? {
        guard fileBytes > 0, pageSize > 0 else { return nil }
        let (sum, o1) = fileBytes.addingReportingOverflow(pageSize - 1)
        guard !o1 else { return nil }
        let (len, o2) = (sum / pageSize).multipliedReportingOverflow(by: pageSize)
        guard !o2, len >= fileBytes else { return nil }
        return len
    }

    public static func sliceStride(inDim: Int, outDim: Int,
                            _ err: (String) -> MoELoweringError) throws -> Int {
        let q4 = GGMLTensorType.q4_0
        guard inDim > 0, outDim > 0, inDim % q4.elementsPerBlock == 0 else {
            throw err("Q4_0 slice dims \(inDim)x\(outDim) invalid")
        }
        return try mul(mul(inDim / q4.elementsPerBlock, q4.bytesPerBlock, "stride", err),
                       outDim, "stride", err)
    }

    static func mul(_ a: Int, _ b: Int, _ what: String,
                    _ err: (String) -> MoELoweringError) throws -> Int {
        let (p, o) = a.multipliedReportingOverflow(by: b)
        guard !o, a >= 0, b >= 0 else { throw err("\(what): \(a)*\(b) overflows or is negative") }
        return p
    }

    static func add(_ a: Int, _ b: Int, _ what: String,
                    _ err: (String) -> MoELoweringError) throws -> Int {
        let (s, o) = a.addingReportingOverflow(b)
        guard !o, a >= 0, b >= 0 else { throw err("\(what): \(a)+\(b) overflows or is negative") }
        return s
    }

    private static func gguf<T>(_ what: String, _ body: () throws -> T) throws -> T {
        do { return try body() } catch { throw MoELoweringError.invalidRegion("\(what): \(error)") }
    }

    private static func metaInt(_ file: GGUFFile, _ key: String, _ cap: Int) throws -> Int {
        let raw = try gguf(key) { try file.metadataUInt(key) }
        guard raw > 0, raw <= UInt64(cap), let v = Int(exactly: raw) else {
            throw MoELoweringError.invalidRegion("\(key)=\(raw) not in 1...\(cap)")
        }
        return v
    }
}
