import Foundation
import LeviathanCore

// Phase 56 stage 2 pure tests: capture slot scoping, frame validation, capacity and reset, sequence
// ordering, expected frame counts, chunking, digests, report contract and status. No Metal device,
// no GGUF, no model weights.

struct Harness {
    var passed = 0
    var failed = 0

    mutating func check(_ ok: Bool, _ name: String) {
        if ok {
            passed += 1
        } else {
            failed += 1
            print("FAIL " + name)
        }
    }
}

struct Boom: Error {}

func sampleScores(_ seed: Int) -> [Float] {
    (0..<128).map { Float(($0 * 37 + seed * 11) % 257) / 64 - 2 }
}

func sampleScales() -> [Float] {
    (0..<128).map { Float(1 + ($0 % 5)) / 4 }
}

/// Independent copy of the incumbent router_topk quotient and coefficient expressions.
func incumbent(_ scores: [Float], _ scales: [Float]) -> (ids: [Int], quotientBits: [UInt32], coefficientBits: [UInt32]) {
    let maxScore = scores.max() ?? 0
    let exps = scores.map { exp(Double($0 - maxScore)) }
    let sumExp = exps.reduce(0, +)
    let probs = exps.map { $0 / sumExp }
    var idx = Array(0..<scores.count)
    idx.sort { probs[$0] > probs[$1] || (probs[$0] == probs[$1] && $0 < $1) }
    let topIdx = Array(idx.prefix(8))
    let probSum = topIdx.reduce(0.0) { $0 + probs[$1] }
    let weights = topIdx.map { Float(probs[$0] / probSum) * scales[$0] }
    return (topIdx, topIdx.map { Float(probs[$0] / probSum).bitPattern }, weights.map { $0.bitPattern })
}

func sampleFrame(layer: Int, position: Int, seed: Int) -> RouterRouteFrame {
    let scores = sampleScores(seed)
    let d = incumbent(scores, sampleScales())
    return RouterRouteFrame(layer: layer, position: position, scoreBits: scores.map { $0.bitPattern }, ids: d.ids,
                            quotientBits: d.quotientBits, coefficientBits: d.coefficientBits,
                            guOffsets: d.ids.map { UInt64(1000 + $0 * 64) },
                            dnOffsets: d.ids.map { UInt64(90000 + $0 * 32) })
}

func edit(_ f: RouterRouteFrame, layer: Int? = nil, position: Int? = nil, scoreBits: [UInt32]? = nil,
          ids: [Int]? = nil, quotientBits: [UInt32]? = nil, coefficientBits: [UInt32]? = nil,
          guOffsets: [UInt64]? = nil, dnOffsets: [UInt64]? = nil) -> RouterRouteFrame {
    RouterRouteFrame(layer: layer ?? f.layer, position: position ?? f.position, scoreBits: scoreBits ?? f.scoreBits,
                     ids: ids ?? f.ids, quotientBits: quotientBits ?? f.quotientBits,
                     coefficientBits: coefficientBits ?? f.coefficientBits, guOffsets: guOffsets ?? f.guOffsets,
                     dnOffsets: dnOffsets ?? f.dnOffsets)
}

func checkSlot(_ h: inout Harness) {
    let slot = RouterRouteCaptureSlot()
    h.check(!slot.isActive && slot.active == nil, "capture slot is inactive by default")
    guard let cap = try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 30, maxFrames: 4) else {
        h.check(false, "capture construction")
        return
    }
    let inside = try? slot.scoped(cap) { () -> Bool in slot.isActive && slot.active === cap }
    h.check(inside == true && !slot.isActive, "scoped capture is active only inside the scope")
    var threw = false
    do { _ = try slot.scoped(cap) { () -> Int in throw Boom() } } catch { threw = error is Boom }
    h.check(threw && !slot.isActive, "a throwing scope rethrows the original error and restores the inactive slot")
    var nestedRefused = false
    _ = try? slot.scoped(cap) { () -> Int in
        do { _ = try slot.scoped(cap) { () -> Int in 0 } } catch { nestedRefused = true }
        return 0
    }
    h.check(nestedRefused && !slot.isActive, "nested capture scope is refused and the outer scope restores")
    let reentered = try? slot.scoped(cap) { () -> Bool in slot.isActive }
    h.check(reentered == true && !slot.isActive, "a scope can be re-entered after it ended")
}

func checkShapeAndFrames(_ h: inout Harness) {
    h.check((try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 30, maxFrames: 0)) == nil, "maxFrames 0 rejected")
    h.check((try? RouterRouteCapture(expertCount: 128, topK: 0, layerCount: 30, maxFrames: 4)) == nil, "topK 0 rejected")
    h.check((try? RouterRouteCapture(expertCount: 128, topK: 129, layerCount: 30, maxFrames: 4)) == nil, "topK > experts rejected")
    h.check((try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 0, maxFrames: 4)) == nil, "layerCount 0 rejected")
    h.check((try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 30,
                                     maxFrames: RouterRouteCapture.maxFramesLimit + 1)) == nil, "over-limit capacity rejected")
    guard let cap = try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 30, maxFrames: 64) else {
        h.check(false, "capture construction")
        return
    }
    let base = sampleFrame(layer: 0, position: 0, seed: 1)
    h.check(RouterRouteCapture.frameFailures(base, expertCount: 128, topK: 8, layerCount: 30).isEmpty, "a well-formed frame validates")
    h.check((try? cap.append(base)) != nil && cap.frames.count == 1 && cap.rejected == 0, "a well-formed frame is stored")
    var nanScores = base.scoreBits
    nanScores[5] = Float.nan.bitPattern
    var infScores = base.scoreBits
    infScores[7] = Float.infinity.bitPattern
    var dupIds = base.ids
    dupIds[1] = dupIds[0]
    var badId = base.ids
    badId[2] = 128
    var negQ = base.quotientBits
    negQ[0] = Float(-0.5).bitPattern
    var nanQ = base.quotientBits
    nanQ[3] = Float.nan.bitPattern
    var nanCoef = base.coefficientBits
    nanCoef[4] = Float.nan.bitPattern
    let malformed: [(String, RouterRouteFrame)] = [
        ("short score vector", edit(base, scoreBits: Array(base.scoreBits.prefix(127)))),
        ("NaN score", edit(base, scoreBits: nanScores)),
        ("infinite score", edit(base, scoreBits: infScores)),
        ("seven ids", edit(base, ids: Array(base.ids.prefix(7)))),
        ("id 128", edit(base, ids: badId)),
        ("duplicate id", edit(base, ids: dupIds)),
        ("negative quotient", edit(base, quotientBits: negQ)),
        ("NaN quotient", edit(base, quotientBits: nanQ)),
        ("seven quotients", edit(base, quotientBits: Array(base.quotientBits.prefix(7)))),
        ("NaN coefficient", edit(base, coefficientBits: nanCoef)),
        ("nine coefficients", edit(base, coefficientBits: base.coefficientBits + [0])),
        ("missing GU offsets (no installed lowering)", edit(base, guOffsets: [])),
        ("missing DN offsets", edit(base, dnOffsets: [])),
        ("layer 30", edit(base, layer: 30)),
        ("layer -1", edit(base, layer: -1)),
        ("negative position", edit(base, position: -1)),
    ]
    for (name, frame) in malformed {
        let count = cap.frames.count
        let rejected = cap.rejected
        let accepted = (try? cap.append(frame)) != nil
        h.check(!accepted && cap.frames.count == count && cap.rejected == rejected + 1, "malformed frame rejected: " + name)
    }
    guard let small = try? RouterRouteCapture(expertCount: 128, topK: 8, layerCount: 30, maxFrames: 2) else {
        h.check(false, "small capture construction")
        return
    }
    let a = (try? small.append(base)) != nil
    let b = (try? small.append(sampleFrame(layer: 1, position: 0, seed: 2))) != nil
    let c = (try? small.append(sampleFrame(layer: 2, position: 0, seed: 3))) != nil
    h.check(a && b && !c && small.frames.count == 2 && small.rejected == 1, "capacity is bounded and overflow throws")
    small.reset()
    h.check(small.frames.isEmpty && small.rejected == 0, "reset clears frames and the rejection count")
    h.check((try? small.append(base)) != nil && small.frames.count == 1, "capture is reusable after reset")
}

func checkIncumbentExpression(_ h: inout Harness) {
    var same = true
    for seed in 0..<16 {
        let scores = sampleScores(seed)
        guard let cpu = RouterSelectionKernel.cpuReference(scores: scores, scales: sampleScales(), topK: 8) else {
            same = false
            continue
        }
        let d = incumbent(scores, sampleScales())
        if cpu.ids != d.ids || cpu.rawBits != d.quotientBits || cpu.weightBits != d.coefficientBits { same = false }
    }
    h.check(same, "captured quotient/coefficient expressions equal the public CPU reference bits")
}

func checkSequence(_ h: inout Harness) {
    var seq = [RouterRouteFrame]()
    for pos in 0..<3 {
        for layer in 0..<4 { seq.append(sampleFrame(layer: layer, position: pos, seed: pos * 4 + layer)) }
    }
    h.check(RouterRouteCapture.sequenceFailures(seq, forwards: 3, layerCount: 4).isEmpty, "ordered complete sequence passes")
    var swapped = seq
    swapped.swapAt(1, 2)
    h.check(!RouterRouteCapture.sequenceFailures(swapped, forwards: 3, layerCount: 4).isEmpty, "layer order violation detected")
    h.check(!RouterRouteCapture.sequenceFailures(Array(seq.dropLast()), forwards: 3, layerCount: 4).isEmpty, "missing frame detected")
    var wrongPosition = seq
    wrongPosition[5] = edit(seq[5], position: 2)
    h.check(!RouterRouteCapture.sequenceFailures(wrongPosition, forwards: 3, layerCount: 4).isEmpty, "position violation detected")
    h.check(!RouterRouteCapture.sequenceFailures(seq, forwards: 4, layerCount: 4).isEmpty, "short capture detected")
    h.check(!RouterRouteCapture.sequenceFailures(seq, forwards: 3, layerCount: 0).isEmpty, "zero layer count refused")
    let pinned = RouterOverlapGateFixtures.pinned
    let forwards = pinned.map { RouterRouteCapture.expectedForwards(promptCount: $0.prompt.count, outputs: 32) }
    h.check(pinned.map({ $0.prompt.count }) == [21, 23, 2] && forwards == [52, 54, 33], "per-fixture forwards 52, 54, 33")
    h.check(RouterRouteReplay.expectedFrames(fixtures: pinned, outputs: 32, layers: 30) == 4170, "139 forwards x 30 layers = 4170 frames")
}

func checkChunksAndDigest(_ h: inout Harness) {
    let chunks = RouterRouteReplay.chunks(total: 4170, maxPerChunk: RouterRouteReplay.chunkSize)
    h.check(chunks.map({ $0.count }) == [1024, 1024, 1024, 1024, 74], "4170 frames give 4 x 1024 + 74")
    var contiguous = true
    var next = 0
    for c in chunks {
        if c.lowerBound != next { contiguous = false }
        next = c.upperBound
    }
    h.check(contiguous && next == 4170, "chunks are contiguous and cover every frame")
    h.check(RouterRouteReplay.chunkSize <= RouterSelectionKernel.maxCases, "chunk size within the component maximum")
    h.check(RouterRouteReplay.chunks(total: 0, maxPerChunk: 1024).isEmpty
            && RouterRouteReplay.chunks(total: 10, maxPerChunk: 0).isEmpty, "degenerate chunk requests are empty")
    let seq = (0..<6).map { sampleFrame(layer: $0 % 3, position: $0 / 3, seed: $0) }
    let d1 = RouterRouteReplay.frameDigest(seq)
    h.check(d1.count == 64 && d1 == RouterRouteReplay.frameDigest(seq), "frame digest is a stable SHA-256")
    var coef = seq
    var bits = coef[2].coefficientBits
    bits[0] ^= 1
    coef[2] = edit(coef[2], coefficientBits: bits)
    h.check(RouterRouteReplay.frameDigest(coef) == d1, "coefficient bits are excluded from the frame digest")
    var score = seq
    var sb = score[1].scoreBits
    sb[9] ^= 1
    score[1] = edit(score[1], scoreBits: sb)
    h.check(RouterRouteReplay.frameDigest(score) != d1, "score bits are covered by the frame digest")
    var offs = seq
    var gu = offs[4].guOffsets
    gu[7] += 1
    offs[4] = edit(offs[4], guOffsets: gu)
    h.check(RouterRouteReplay.frameDigest(offs) != d1, "GU offsets are covered by the frame digest")
    let realLike = RouterOffsetTable(
        expertCount: 128, alignment: 1, gu: [RouterOffsetFamily(base: 3, stride: 18, expertBytes: 18)],
        dn: [RouterOffsetFamily(base: 7, stride: 36, expertBytes: 36)], guBufferLength: 3 + 128 * 18,
        dnBufferLength: 7 + 128 * 36)
    h.check((try? realLike.validate()) == 256, "byte-aligned real-style families are proven exactly")
}

func checkReport(_ h: inout Harness) {
    var report = RouterRouteReplayReport()
    report.frames = [RouterRouteReplayFrameRecord(fixture: "en-chat", position: 0, layer: 0, ids: [1, 2, 3, 4, 5, 6, 7, 8],
                                                  quotientBits: [UInt32](repeating: 0x3E00_0000, count: 8),
                                                  selectorFlags: [], consumerFlags: [], verdict: "EXACT", problems: [])]
    guard let data = try? RouterRouteReplayDiagnostics.encodeReport(report),
          let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        h.check(false, "report encodes")
        return
    }
    let falseKeys = ["gainQualified", "installable", "productionPerformanceClaim", "realE4GateComplete", "scalesSerialized"]
    h.check(falseKeys.allSatisfy { (object[$0] as? Bool) == false }, "diagnostic contract fields encode false")
    let frameKeys = ((object["frames"] as? [[String: Any]])?.first?.keys).map { Set($0) } ?? []
    h.check(!frameKeys.isEmpty && !frameKeys.contains("coefficientBits") && !frameKeys.contains("scales"),
            "frame records carry no coefficient bits or scales")
    var tampered = object
    for key in falseKeys { tampered[key] = true }
    if let raw = try? JSONSerialization.data(withJSONObject: tampered),
       let decoded = try? JSONDecoder().decode(RouterRouteReplayReport.self, from: raw),
       let again = try? RouterRouteReplayDiagnostics.encodeReport(decoded),
       let reobject = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
        h.check(falseKeys.allSatisfy { (reobject[$0] as? Bool) == false }, "a decoded report never re-emits a true contract field")
    } else {
        h.check(false, "tampered report round trip")
    }
    h.check(RouterRouteReplayDiagnostics.readyToken == RouterOverlapModelGate.readyToken
            && RouterRouteReplayDiagnostics.goToken == RouterOverlapModelGate.goToken
            && RouterRouteReplayDiagnostics.endToken == RouterOverlapModelGate.endToken
            && RouterOverlapModelGate.readyToken == "BENCHMARK_READY" && RouterOverlapModelGate.goToken == "GO"
            && RouterOverlapModelGate.endToken == "TIMED_WINDOW_END", "protocol tokens are the gate constants")
    h.check(RouterRouteReplayDiagnostics.goAccepted("GO") && !RouterRouteReplayDiagnostics.goAccepted(nil)
            && !RouterRouteReplayDiagnostics.goAccepted("GO ") && !RouterRouteReplayDiagnostics.goAccepted("go"),
            "only exact GO is accepted; EOF refused")
}

func checkStatus(_ h: inout Harness) {
    typealias D = RouterRouteReplayDiagnostics
    var r = RouterRouteReplayReport()
    h.check(D.finalStatus(r) == .runnerNotRestored, "unrestored runner status")
    r.replayTerminal = false
    h.check(D.finalStatus(r) == .notEnded, "unobserved GPU completion is NOT_ENDED first")
    r.replayTerminal = true
    r.restoredPristine = true
    r.abortReason = "x"
    h.check(D.finalStatus(r) == .aborted, "abort status")
    r.abortReason = nil
    h.check(D.finalStatus(r) == .captureFailed, "capture failure status")
    r.capturePassed = true
    h.check(D.finalStatus(r) == .gpuError, "missing replay completion is a GPU error")
    r.replayCompletion = "COMPLETED"
    r.refusedFrames = 1
    h.check(D.finalStatus(r) == .replayRefused, "any refused real frame fails as refused")
    r.refusedFrames = 0
    r.expectedTotalFrames = 2
    h.check(D.finalStatus(r) == .replayMismatch, "missing exact frames is a mismatch")
    let record = RouterRouteReplayFrameRecord(fixture: "raw-hello", position: 0, layer: 0, ids: [0, 1, 2, 3, 4, 5, 6, 7],
                                              quotientBits: [UInt32](repeating: 0, count: 8), selectorFlags: [],
                                              consumerFlags: [], verdict: "EXACT", problems: [])
    r.frames = [record, record]
    r.exactFrames = 2
    r.chunks = [RouterRouteReplayChunk(start: 0, count: 2, selectorCompletion: "COMPLETED", selectorGuardsIntact: true,
                                       selectorGpuMicros: nil, consumerCompletion: "COMPLETED",
                                       consumerGuardsIntact: true, consumerGpuMicros: nil)]
    h.check(D.finalStatus(r) == .replayExact, "all frames exact with intact guards")
    r.cpuReferenceMismatches = 1
    h.check(D.finalStatus(r) == .replayMismatch, "a CPU reference mismatch fails")
    r.cpuReferenceMismatches = 0
    r.chunks = [RouterRouteReplayChunk(start: 0, count: 2, selectorCompletion: "COMPLETED", selectorGuardsIntact: false,
                                       selectorGpuMicros: nil, consumerCompletion: "COMPLETED",
                                       consumerGuardsIntact: true, consumerGpuMicros: nil)]
    h.check(D.finalStatus(r) == .replayMismatch, "a violated canary guard fails")
}

func runAll() -> Int32 {
    var h = Harness()
    checkSlot(&h)
    checkShapeAndFrames(&h)
    checkIncumbentExpression(&h)
    checkSequence(&h)
    checkChunksAndDigest(&h)
    checkReport(&h)
    checkStatus(&h)
    print("router-route-replay-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
