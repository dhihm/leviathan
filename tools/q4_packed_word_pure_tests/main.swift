import Foundation
import LeviathanCore

// Phase 61 pure tests: descriptor identity, fail-closed plan, pair word layout mirror, emitted source structure and
// term order, report contract, options and status. No Metal device, no GGUF, no model weights.

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

func checkDescriptor(_ h: inout Harness) {
    let d = try? Q4PackedWordDescriptor.derive(executionWidth: 32)
    h.check(d?.blocksPerPair == 2 && d?.bytesPerPair == 36 && d?.wordsPerPair == 9
            && d?.kernelName == "q4pw_matvec_W32_pair9", "canonical descriptor: 2 blocks, 36 bytes, 9 words, width 32")
    h.check((try? Q4PackedWordDescriptor.derive(executionWidth: 64)) == nil, "a width other than 32 is refused")
    if let d, let data = try? JSONEncoder().encode(d),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        obj["wordsPerPair"] = 10
        let tampered = (try? JSONSerialization.data(withJSONObject: obj)).flatMap {
            try? JSONDecoder().decode(Q4PackedWordDescriptor.self, from: $0)
        }
        if let t = tampered {
            h.check((try? t.validate()) == nil, "a decoded descriptor with a non-canonical word count fails validate")
        } else {
            h.check(false, "tampered descriptor decodes")
        }
    } else {
        h.check(false, "descriptor encodes for the tamper test")
    }
}

func checkPlan(_ h: inout Harness) {
    guard let d = try? Q4PackedWordDescriptor.derive(executionWidth: 32) else {
        h.check(false, "descriptor for plan tests")
        return
    }
    func reason(inDim: Int = 2816, outDim: Int = 64, offset: Int = 256, length: Int = 1 << 30, width: Int = 32,
                maxThreads: Int = 1024, staticBytes: Int = 0) -> String? {
        Q4PackedWordPlan.declineReason(inDim: inDim, outDim: outDim, weightOffset: offset, weightLength: length,
                                       descriptor: d, executionWidth: width, pipelineMaxThreads: maxThreads,
                                       staticThreadgroupBytes: staticBytes)
    }
    h.check(reason() == nil, "2816 x 64 at offset 256 is supported")
    h.check(reason(inDim: 96, outDim: 8) != nil, "odd block count (96 -> 3 blocks) is declined")
    h.check(reason(offset: 2) != nil, "misaligned weight offset is declined")
    h.check(reason(length: 256 + 63 * 1584) != nil, "a span past the bound buffer is declined")
    h.check(reason(width: 64) != nil, "a compiled width other than the descriptor width is declined")
    h.check(reason(maxThreads: 16) != nil, "a pipeline maximum below the launch width is declined")
    h.check(reason(staticBytes: 4096) != nil, "a register candidate reporting static threadgroup memory is declined")
    h.check(reason(inDim: 48) != nil, "inDim not a multiple of 32 is declined")
    let l = Q4PackedWordPlan.launch(outDim: 33, descriptor: d)
    let c = Q4WeightStagePlan.controlLaunch(outDim: 33, pipelineMaxThreads: 1024)
    h.check(l.threadgroups == 2 && l.threadsPerThreadgroup == 32 && l == c,
            "candidate launch equals the width-32 control geometry")
}

func checkLayout(_ h: inout Harness) {
    var bytes = [UInt8]()
    for i in 0..<36 { bytes.append(UInt8((i * 37 + 11) & 0xFF)) }
    guard let words = Q4PackedWordPlan.pairWords(bytes[...]), let decoded = Q4PackedWordPlan.decodePair(words) else {
        h.check(false, "pair words and decode")
        return
    }
    h.check(words.count == 9 && words[0] == (UInt32(bytes[0]) | UInt32(bytes[1]) << 8 | UInt32(bytes[2]) << 16
                                              | UInt32(bytes[3]) << 24), "nine little-endian words per pair")
    h.check(decoded.d0Bits == (UInt16(bytes[0]) | UInt16(bytes[1]) << 8)
            && decoded.d1Bits == (UInt16(bytes[18]) | UInt16(bytes[19]) << 8),
            "block scales come from the low half of word 0 and the high half of word 4")
    h.check(decoded.q0 == Array(bytes[2..<18]) && decoded.q1 == Array(bytes[20..<36]),
            "quant bytes of both blocks are recovered in order from the word halves")
    let s2 = Q4PackedWordPlan.byteSource(2)
    let s17 = Q4PackedWordPlan.byteSource(17)
    let s20 = Q4PackedWordPlan.byteSource(20)
    let s35 = Q4PackedWordPlan.byteSource(35)
    h.check(s2 == (0, 16) && s17 == (4, 8) && s20 == (5, 0) && s35 == (8, 24), "byte sources at the pair edges")
    h.check(Q4PackedWordPlan.pairWords(bytes[0..<35]) == nil && Q4PackedWordPlan.decodePair([1, 2, 3]) == nil,
            "wrong pair lengths are refused")
}

func checkEmitter(_ h: inout Harness) {
    guard let d = try? Q4PackedWordDescriptor.derive(executionWidth: 32), let a = try? Q4PackedWordEmitter.emit(d),
          let b = try? Q4PackedWordEmitter.emit(d) else {
        h.check(false, "emitter produces the kernel")
        return
    }
    h.check(a == b && occurrences(a, "kernel void q4pw_matvec_W32_pair9(") == 1, "deterministic single kernel")
    h.check(occurrences(a, "threadgroup") == 0 && occurrences(a, "barrier") == 0 && occurrences(a, "simd_") == 0
            && occurrences(a, "float4") == 0 && occurrences(a, "fma(") == 0,
            "no threadgroup memory, barrier, SIMD, vector-load or explicit fma constructs")
    h.check((0..<9).allSatisfy { occurrences(a, "uint w" + String($0) + " = pw[" + String($0) + "];") == 1 },
            "nine word loads per pair into registers")
    h.check(occurrences(a, "sum += d * (float(q & 0x0Fu) - 8.0f) * x[") == 32
            && occurrences(a, "sum += d * (float(q >> 4) - 8.0f) * x[") == 32
            && occurrences(a, "d = float(as_type<half>(dbits));") == 2
            && occurrences(a, "ushort dbits = ushort(w0 & 0xFFFFu);") == 1
            && occurrences(a, "dbits = ushort(w4 >> 16u);") == 1,
            "32 low and 32 high production term lines, two scale decodes from the word halves")
    let firstLo = a.range(of: "q = uchar((w0 >> 16u) & 0xFFu);")
    let lastBlock0 = a.range(of: "x[xb + 16u + 15u];")
    let firstBlock1 = a.range(of: "q = uchar((w5 >> 0u) & 0xFFu);")
    let lastBlock1 = a.range(of: "x[xb + 48u + 15u];")
    h.check(firstLo != nil && lastBlock0 != nil && firstBlock1 != nil && lastBlock1 != nil
            && firstLo!.lowerBound < lastBlock0!.lowerBound && lastBlock0!.lowerBound < firstBlock1!.lowerBound
            && firstBlock1!.lowerBound < lastBlock1!.lowerBound,
            "block 0's sixteen terms precede block 1's in the emitted order")
    h.check(Gemma4Runner.shaderSource.contains("kernel void g4_q4_matvec(")
            && Gemma4Runner.shaderSource.contains("sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];"),
            "the control is the production function with the same term expression")
}

func checkContract(_ h: inout Harness) {
    typealias D = Q4PackedWordDiagnostics
    let plain = try? D.parseOptions(["--report", "o.json"])
    let withModel = try? D.parseOptions(["--report", "o.json", "--gguf", "m.gguf"])
    h.check(plain?.report == "o.json" && plain?.gguf == nil && withModel?.gguf == "m.gguf"
            && (try? D.parseOptions(["--gguf", "m.gguf"])) == nil
            && (try? D.parseOptions(["--report", "o.json", "--other", "x"])) == nil
            && (try? D.parseOptions(["--report", "a", "--report", "b"])) == nil,
            "options: report required, gguf optional, unknown and duplicate refused")
    let flags = ["fullModelQualified", "gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    var contract = false
    if let data = try? D.encodeReport(Q4PackedWordReport()),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let base = flags.allSatisfy { (obj[$0] as? Bool) == false } && (obj["kind"] as? String) == D.kind
        for f in flags { obj[f] = true }
        obj["status"] = "correctnessPassed"
        if let raw = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(Q4PackedWordReport.self, from: raw),
           let again = try? D.encodeReport(decoded),
           let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
            contract = base && flags.allSatisfy { (back[$0] as? Bool) == false }
        }
    }
    h.check(contract, "every claim flag encodes false and tampered flags re-emit false")
    var r = Q4PackedWordReport()
    h.check(D.finalStatus(r) == .unsupported, "an empty report is unsupported")
    r.terminal = false
    h.check(D.finalStatus(r) == .notEnded, "a non-terminal run is not ended")
    r.terminal = true
    r.error = "x"
    h.check(D.finalStatus(r) == .aborted, "an error aborts")
    h.check(D.kind == "q4-packed-word-register-decode-component-v1"
            && Q4WeightStageDiagnostics.warmupPairs == 2 && Q4WeightStageDiagnostics.timedPairs == 6
            && Q4WeightStageDiagnostics.correctnessContract == 1e-4 && Q4WeightStageCorpus.synthetic().count == 10,
            "reused fixed protocol constants and the unchanged corpus size")
}

func runAll() -> Int32 {
    var h = Harness()
    checkDescriptor(&h)
    checkPlan(&h)
    checkLayout(&h)
    checkEmitter(&h)
    checkContract(&h)
    print("q4-packed-word-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
