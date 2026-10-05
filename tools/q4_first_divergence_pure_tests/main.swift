import Foundation
import LeviathanCore

// Phase 64 pure tests: trace layout, exact half decode, CPU accumulation models, instrumented emitter structure,
// first-difference and locator helpers, frozen-case identity, the attribution rule, report contract, options and
// status mapping. No Metal device, no GGUF.

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

func occurrences(_ s: String, _ needle: String) -> Int { s.components(separatedBy: needle).count - 1 }

func checkLayout(_ h: inout Harness) {
    typealias L = Q4FirstDivergenceLayout
    h.check(L.unpackWords(blocksPerRow: 2, inDim: 64) == 98 && L.quantIndex(blocksPerRow: 2, block: 1, byte: 3) == 21
            && L.xIndex(blocksPerRow: 2, element: 5) == 39 && L.termIndex(block: 1, byte: 3, high: 1) == 39
            && L.traceWords(inDim: 64, blocksPerRow: 2, variant: .split) == 256
            && L.traceWords(inDim: 64, blocksPerRow: 2, variant: .sumOnly) == 64
            && L.traceWords(inDim: 64, blocksPerRow: 2, variant: .unpack) == 98, "layout indices and word counts")
    let split = (0..<256).map { UInt32($0) }
    let st = L.stages(split, variant: .split, blocksPerRow: 2, inDim: 64)
    h.check(st.count == 4 && st[2].name == "p" && st[2].values.count == 64 && st[2].values[1] == 6
            && st[3].values[63] == 255, "split stages are interleaved views w, dw, p, sum")
    let un = L.stages((0..<98).map { UInt32($0) }, variant: .unpack, blocksPerRow: 2, inDim: 64)
    h.check(un.count == 3 && un[0].values == [0, 1] && un[1].values.count == 32 && un[2].values.first == 34
            && L.stages([1, 2], variant: .unpack, blocksPerRow: 2, inDim: 64).isEmpty, "unpack stages and size guard")
    h.check(L.firstDifference([1, 2, 3], [1, 2, 3]) == nil && L.firstDifference([1, 2, 3], [1, 9, 3]) == 1
            && L.firstDifference([1, 2], [1, 2, 3]) == 2, "first difference: identical nil, index, count mismatch")
    h.check(L.locator(stage: "sum", index: 39) == "term 39 = block 1 byte 3 high nibble"
            && L.locator(stage: "quantBytes", index: 21) == "block 1 byte 5" && L.locator(stage: "xBits", index: 7) == "x element 7"
            && L.hex8(0x1F) == "0000001f", "locator strings and 8-digit hex")
    let ub: [UInt8] = [0x34, 0x12] + (0..<16).map { UInt8(0xA0 + $0) }
    let ux = (0..<32).map { Float($0) }
    let e = L.expectedUnpack(blocks: ub, inDim: 32, x: ux)
    h.check(e?.count == 49 && e?[0] == 0x1234 && e?[1] == 0xA0 && e?[16] == 0xAF && e?[17] == 0
            && e?[48] == Float(31).bitPattern && L.expectedUnpack(blocks: [0x34], inDim: 32, x: ux) == nil,
            "host unpack expectation: little-endian scale, quant byte order, x bit order, short input refused")
}

func checkHalf(_ h: inout Harness) {
    typealias C = Q4FirstDivergenceCPU
    h.check(C.halfToFloat(0x3C00) == 1 && C.halfToFloat(0xC000) == -2 && C.halfToFloat(0x5BFF) == 255.875
            && C.halfToFloat(0x0001) == Float(bitPattern: 0x3380_0000) && C.halfToFloat(0x8001) == -Float(bitPattern: 0x3380_0000)
            && C.halfToFloat(0x8000).bitPattern == 0x8000_0000 && C.halfToFloat(0x7C00) == Float.infinity
            && C.halfToFloat(0xFC00) == -Float.infinity && C.halfToFloat(0x7E00).isNaN, "exact half decode incl. edges")
}

func checkModels(_ h: inout Harness) {
    var blocks: [UInt8] = [0x00, 0x3C]
    blocks += [UInt8](repeating: 0x00, count: 16)
    let x = [Float](repeating: 0.5, count: 32)
    var all = true
    for m in Q4FirstDivergenceCPU.Model.allCases {
        guard let s = Q4FirstDivergenceCPU.runningSums(model: m, blocks: blocks, inDim: 32, x: x), s.count == 32 else {
            all = false
            continue
        }
        for k in 0..<32 where Float(bitPattern: s[k]) != Float(-4 * (k + 1)) { all = false }
    }
    h.check(all, "all four models give the exact serial sums -4, -8, ..., -128 on an exact input")
    h.check(Q4FirstDivergenceCPU.runningSums(model: .strictLeftProductThenAdd, blocks: blocks, inDim: 64, x: x) == nil
            && Q4FirstDivergenceCPU.runningSums(model: .strictLeftProductThenAdd, blocks: blocks, inDim: 33, x: x) == nil,
            "short blocks or a non-multiple of 32 are refused")
}

func checkEmitter(_ h: inout Harness) {
    typealias E = Q4FirstDivergenceEmitter
    var names = Set<String>()
    var ok = true
    for v in Q4FirstDivergenceVariant.allCases {
        for s in Q4FirstDivergenceShape.allCases {
            let name = E.kernelName(shape: s, variant: v)
            names.insert(name)
            let src = E.emit(shape: s, variant: v)
            if occurrences(src, "kernel void " + name + "(") != 1 { ok = false }
            if occurrences(src, "device uint* tr [[buffer(5)]]") != 1 { ok = false }
            if occurrences(src, "for (uint b = 0; b < blocksPerRow; b++) {") != 1
                || occurrences(src, "for (uint l = 0; l < 16u; l++) {") != 1 { ok = false }
            if occurrences(src, "fma(") != 0 || occurrences(src, "threadgroup") != 0 || occurrences(src, "simd_") != 0
                || occurrences(src, "float4") != 0 { ok = false }
            if s == .production, occurrences(src, "uchar q = blk[2u + l];") != 1 { ok = false }
            if s == .register, occurrences(src, "b0 = (t4 >> 16u) | (t5 << 16u);") != 1
                || occurrences(src, "uchar q = uchar((word >> ((bi & 3u) * 8u)) & 0xFFu);") != 1 { ok = false }
            switch v {
            case .unpack:
                if occurrences(src, "tr[b] = uint(dbits);") != 1 || occurrences(src, "sum +=") != 0 { ok = false }
            case .sumOnly:
                if occurrences(src, "sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];") != 1
                    || occurrences(src, "sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];") != 1 { ok = false }
            case .split:
                if occurrences(src, "float dw0 = d * w0;") != 1 || occurrences(src, "sum += p1;") != 1 { ok = false }
            }
        }
    }
    h.check(ok && names.count == 6, "six distinct instrumented kernels with the expected structure and no forbidden constructs")
    h.check(Gemma4Runner.shaderSource.contains("uchar q = blk[2u + l];")
            && Gemma4Runner.shaderSource.contains("sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];"),
            "the production kernel carries the mirrored byte read and term line")
}

func checkMirror(_ h: inout Harness) {
    typealias M = Q4FirstDivergenceMirror
    let descriptor = try? Q4RegisterLoopDescriptor.derive(executionWidth: 32)
    let registerSource = descriptor.flatMap { try? Q4RegisterLoopEmitter.emit($0) }
    let errors = M.check(controlSource: Gemma4Runner.shaderSource, registerSource: registerSource ?? "",
                         registerKernelName: descriptor?.kernelName ?? "")
    for e in errors { print("mirror: " + e) }
    h.check(registerSource != nil && errors.isEmpty,
            "every instrumented copy mirrors its uniquely extracted original (skeleton; sum-only verbatim)")
    h.check(M.kernelLines("kernel void k(int a) { }\nkernel void k(int a) { }\n", name: "k") == nil
            && M.kernelLines("kernel void k(int a) {\n    x;\n", name: "k") == nil
            && M.kernelLines("// head\nkernel void k(int a) {\n    // c\n    x;\n}\nkernel void j() {}\n", name: "k")
                == ["kernel void K(int a) {", "x;", "}"],
            "extraction refuses duplicate and unbalanced declarations, drops comments, normalizes the name")
}

func checkFrozenCase(_ h: inout Harness) {
    let c = Q4FirstDivergenceDiagnostics.frozenCase()
    let first = Q4WeightStageCorpus.synthetic().first
    h.check(c?.name == "min-1x64" && c?.inDim == 64 && c?.outDim == 1 && c?.blocksPerRow == 2 && c?.blocks.count == 36
            && c?.x.count == 64 && c?.blocks == first?.blocks && c?.x == first?.x && c?.expectedDecline == false,
            "frozen case is the first unchanged corpus case")
}

func checkAttribution(_ h: inout Harness) {
    typealias D = Q4FirstDivergenceDiagnostics
    let same = Q4FirstDivergenceStageDiff.make(stage: "sum", production: [1, 2], register: [1, 2])
    let diff = Q4FirstDivergenceStageDiff.make(stage: "sum", production: [1, 2, 3], register: [1, 2, 4])
    h.check(same.identical && diff.firstIndex == 2 && diff.productionHex == "00000003" && diff.registerHex == "00000004"
            && diff.locator == "term 2 = block 0 byte 1 low nibble", "stage diff construction")
    func variant(_ name: String, finals: Bool = true, last: Bool = true, intact: Bool = true,
                 _ stages: [Q4FirstDivergenceStageDiff]) -> Q4FirstDivergenceVariantDiff {
        Q4FirstDivergenceVariantDiff(variant: name, productionTraceFinalEqualsControl: true,
                                     registerTraceFinalEqualsCandidate: finals, lastTracedSumsEqualOwnFinals: last,
                                     runsIntact: intact, stages: stages)
    }
    func attr(_ v: [Q4FirstDivergenceVariantDiff], reproduced: Bool? = true, intact: Bool = true, host: Bool = true,
              unpackIdentical: Bool? = true,
              unpack: [Q4FirstDivergenceStageDiff] = []) -> (established: Bool, statement: String) {
        D.attribute(reproduced: reproduced, uninstrumentedIntact: intact, unpackHostVerified: host,
                    unpackIdentical: unpackIdentical, unpack: unpack, variants: v)
    }
    let good = variant("sumOnly", [diff])
    let refused = [
        attr([good], reproduced: nil), attr([good], intact: false), attr([good], host: false),
        attr([good], unpackIdentical: false,
             unpack: [Q4FirstDivergenceStageDiff.make(stage: "quantBytes", production: [1], register: [2])]),
        attr([variant("sumOnly", finals: false, [diff])]), attr([variant("sumOnly", last: false, [diff])]),
        attr([variant("sumOnly", intact: false, [diff])]), attr([variant("split", [diff])]),
        attr([variant("sumOnly", [same])]),
    ]
    h.check(refused.allSatisfy { !$0.established }
            && refused.dropFirst().allSatisfy { $0.statement.contains("NOT ESTABLISHED") }
            && good.attributableToOriginals && !variant("split", [diff]).attributableToOriginals,
            "not reproduced, uninstrumented integrity, host unpack, unpack difference, final, last-sum, run integrity, "
                + "split and empty-diff contradiction never establish")
    let ok = attr([good])
    h.check(ok.established && ok.statement.contains("ESTABLISHED by the sumOnly trace") && ok.statement.contains("term 2")
            && ok.statement.contains("not the compiled instruction"),
            "a sum-only trace passing every gate with a differing stage establishes the first differing running sum")
}

func checkContract(_ h: inout Harness) {
    typealias D = Q4FirstDivergenceDiagnostics
    h.check((try? D.parseOptions(["--report", "o.json"]))?.report == "o.json"
            && (try? D.parseOptions(["--report", "o.json", "--gguf", "m"])) == nil && (try? D.parseOptions([])) == nil,
            "options: report only; gguf refused")
    let flags = ["fullModelQualified", "gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    var contract = false
    if let data = try? D.encodeReport(Q4FirstDivergenceReport()),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let base = flags.allSatisfy { (obj[$0] as? Bool) == false } && (obj["kind"] as? String) == D.kind
            && (obj["originalCauseEstablished"] as? Bool) == false && (obj["closedRecords"] as? [String])?.count == 3
            && (obj["kind"] as? String) == "q4-first-divergence-diagnostic-v2"
        for f in flags { obj[f] = true }
        if let raw = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(Q4FirstDivergenceReport.self, from: raw),
           let again = try? D.encodeReport(decoded),
           let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
            contract = base && flags.allSatisfy { (back[$0] as? Bool) == false }
        }
    }
    h.check(contract, "claim flags encode false and re-emit false after tampering; v2 kind; three closed records present")
    var r = Q4FirstDivergenceReport()
    h.check(D.finalStatus(r) == .unsupported, "empty report is unsupported")
    r.terminal = false
    h.check(D.finalStatus(r) == .notEnded && D.exitCode(.notEnded) == 5 && D.exitCode(.notReproduced) == 7
            && D.exitCode(.completed) == 0 && D.exitCode(.commandBufferError) == 4
            && D.exitCode(.integrityFailed) == 3, "status and exit mapping")
}

func runAll() -> Int32 {
    var h = Harness()
    checkLayout(&h)
    checkHalf(&h)
    checkModels(&h)
    checkEmitter(&h)
    checkMirror(&h)
    checkFrozenCase(&h)
    checkAttribution(&h)
    checkContract(&h)
    print("q4-first-divergence-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
