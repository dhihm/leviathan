import Foundation
import LeviathanCore

// Phase 65 pure tests: production serial term order, prefix masks, the FP64 prefix oracle, summary rule, report
// contract, options, status and exit mapping, frozen-case identity. No Metal device, no GGUF.

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

func checkOrder(_ h: inout Harness) {
    typealias O = Q4PrefixProbeOrder
    h.check(O.xIndex(term: 0) == 0 && O.xIndex(term: 1) == 16 && O.xIndex(term: 2) == 1 && O.xIndex(term: 3) == 17
            && O.xIndex(term: 30) == 15 && O.xIndex(term: 31) == 31 && O.xIndex(term: 32) == 32
            && O.xIndex(term: 33) == 48 && O.xIndex(term: 63) == 63, "serial term order 0,16,1,17,...,32,48,...,63")
    let order = O.termOrder(inDim: 64)
    h.check(order.count == 64 && Set(order).count == 64 && order.allSatisfy { $0 >= 0 && $0 < 64 },
            "term order is a permutation of 0..<64")
    h.check(Gemma4Runner.shaderSource.contains("x[b * 32u + l]") && Gemma4Runner.shaderSource.contains("x[b * 32u + 16u + l]"),
            "the production kernel reads x at b*32+l then b*32+16+l, the modelled order")
}

func checkMasks(_ h: inout Harness) {
    typealias O = Q4PrefixProbeOrder
    let x = (0..<64).map { Float($0 + 1) }
    let m0 = O.maskedInput(x, prefix: 0)
    let m3 = O.maskedInput(x, prefix: 3)
    let m64 = O.maskedInput(x, prefix: 64)
    h.check(m0 == [Float](repeating: 0, count: 64) && m64 == x, "prefix 0 is all zeros and prefix 64 is the original")
    h.check(m3 != nil && m3![0] == 1 && m3![16] == 17 && m3![1] == 2 && m3![17] == 0 && m3![2] == 0
            && m3!.filter({ $0 != 0 }).count == 3, "prefix 3 keeps x[0], x[16], x[1] only")
    h.check(O.maskedInput(x, prefix: 65) == nil && O.maskedInput(x, prefix: -1) == nil
            && O.maskedInput([1, 2, 3], prefix: 1) == nil, "out-of-range prefix or non-multiple-of-32 input refused")
    var allPrefixesExact = true
    for k in 0...64 {
        guard let m = O.maskedInput(x, prefix: k) else { allPrefixesExact = false; continue }
        for t in 0..<64 {
            let i = O.xIndex(term: t)
            if (t < k && m[i] != x[i]) || (t >= k && m[i] != 0) { allPrefixesExact = false }
        }
    }
    h.check(allPrefixesExact, "every prefix mask keeps exactly the first k terms")
}

func checkOracle(_ h: inout Harness) {
    var blocks: [UInt8] = [0x00, 0x3C]
    blocks += [UInt8](repeating: 0x00, count: 16)
    blocks += [0x00, 0xC0]
    blocks += [UInt8](repeating: 0xFF, count: 16)
    let x = [Float](repeating: 0.5, count: 64)
    var ok = true
    for k in 0...64 {
        guard let v = Q4PrefixProbeOrder.prefixOracle(blocks: blocks, inDim: 64, x: x, prefix: k) else { ok = false; continue }
        // block 0: d = 1, nibbles 0 -> w = -8, term = -4 each; block 1: d = -2, nibbles 15 -> w = 7, term = -7 each.
        let expected = Double(-4 * min(k, 32)) + Double(-7 * max(0, k - 32))
        if v != expected { ok = false }
    }
    h.check(ok, "FP64 prefix oracle equals the exact partial sums on a hand case across both blocks")
    h.check(Q4PrefixProbeOrder.prefixOracle(blocks: blocks, inDim: 64, x: x, prefix: 65) == nil
            && Q4PrefixProbeOrder.prefixOracle(blocks: [1], inDim: 64, x: x, prefix: 1) == nil, "oracle refuses bad bounds")
    if let masked = Q4PrefixProbeOrder.maskedInput(x, prefix: 40),
       let ref = try? Q4Oracle.matvecFP64(blocks: blocks[...], outDim: 1, inDim: 64, x: masked), ref.count == 1,
       let p = Q4PrefixProbeOrder.prefixOracle(blocks: blocks, inDim: 64, x: x, prefix: 40) {
        h.check(abs(ref[0] - p) <= 1e-12 * max(1.0, abs(p)), "masked-input oracle agrees with the explicit prefix oracle")
    } else {
        h.check(false, "masked-input oracle evaluates")
    }
}

func probe(_ k: Int, _ c: String?, _ d: String?, completed: Bool = true) -> Q4PrefixProbeResult {
    Q4PrefixProbeResult(prefix: k, termIndex: k > 0 ? k - 1 : nil, xIndexAdded: nil, locator: nil,
                        controlCompletion: completed ? "COMPLETED" : "ERROR", candidateCompletion: "COMPLETED",
                        controlHex: c, candidateHex: d, bitIdentical: c == d && c != nil, finite: true, guardsIntact: true,
                        prefixOracle: 0, oracleAgreesWithMaskedOracle: true, controlFP64Rel: 0, candidateFP64Rel: 0,
                        fp64Pass: true, passed: completed)
}

func checkSummary(_ h: inout Harness) {
    typealias D = Q4PrefixProbeDiagnostics
    let probes = [probe(0, "00000000", "00000000"), probe(1, "3f800000", "3f800000"), probe(2, "40000000", "40000001"),
                  probe(3, "40400000", "40400001"), probe(4, "40800000", "40800000", completed: false)]
    let s = D.summarize(probes)
    h.check(s.first?.prefix == 2 && s.first?.termIndex == 1 && s.count == 2, "summary: first differing prefix 2 (term 1), 2 differing")
    h.check(D.summarize([probe(0, "0", "0"), probe(1, "1", "1")]).first == nil, "no difference gives no first prefix")
    h.check(D.phase64ControlHex == "bd6b6117" && D.phase64CandidateHex == "bd6b6116", "phase-64 recorded finals")
}

func checkContract(_ h: inout Harness) {
    typealias D = Q4PrefixProbeDiagnostics
    h.check((try? D.parseOptions(["--report", "o.json"]))?.report == "o.json"
            && (try? D.parseOptions(["--report", "o.json", "--gguf", "m"])) == nil && (try? D.parseOptions([])) == nil,
            "options: report only; gguf refused")
    let flags = ["fullModelQualified", "gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    var contract = false
    if let data = try? D.encodeReport(Q4PrefixProbeReport()),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let base = flags.allSatisfy { (obj[$0] as? Bool) == false } && (obj["kind"] as? String) == D.kind
            && (obj["closedRecords"] as? [String])?.count == 3
            && ((obj["causalityStatement"] as? String) ?? "").contains("does not prove")
        for f in flags { obj[f] = true }
        if let raw = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(Q4PrefixProbeReport.self, from: raw),
           let again = try? D.encodeReport(decoded),
           let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
            contract = base && flags.allSatisfy { (back[$0] as? Bool) == false }
        }
    }
    h.check(contract, "claim flags encode false and re-emit false after tampering; closed records and causality statement present")
    var r = Q4PrefixProbeReport()
    h.check(D.finalStatus(r) == .unsupported, "empty report is unsupported")
    r.terminal = false
    h.check(D.finalStatus(r) == .notEnded && D.exitCode(.notEnded) == 5 && D.exitCode(.notReproduced) == 7
            && D.exitCode(.probeContractFailed) == 3 && D.exitCode(.completed) == 0 && D.exitCode(.commandBufferError) == 4,
            "status and exit mapping")
    let c = Q4FirstDivergenceDiagnostics.frozenCase()
    h.check(c?.name == "min-1x64" && c?.inDim == 64 && c?.outDim == 1 && c?.blocks == Q4WeightStageCorpus.synthetic().first?.blocks,
            "frozen case unchanged")
}

func runAll() -> Int32 {
    var h = Harness()
    checkOrder(&h)
    checkMasks(&h)
    checkOracle(&h)
    checkSummary(&h)
    checkContract(&h)
    print("q4-prefix-probe-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
