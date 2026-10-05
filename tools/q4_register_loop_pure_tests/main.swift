import Foundation
import LeviathanCore

// Phase 62 pure tests: descriptor identity, plan delegation to the phase-61 pair alignment contract, the host
// mirror of the per-block five-word fetch (even and odd blocks), emitted source structure (production loop shape,
// no unrolling, no forbidden constructs), report contract, options and status. No Metal device, no GGUF.

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
    let d = try? Q4RegisterLoopDescriptor.derive(executionWidth: 32)
    h.check(d?.wordsPerBlockFetch == 5 && d?.wordsPerPair == 9 && d?.oddBlockShiftBits == 16
            && d?.kernelName == "q4rl_matvec_W32_fetch5" && d?.pairLayout?.wordsPerPair == 9,
            "canonical descriptor: five-word block fetch over the width-32 nine-word pair layout")
    h.check((try? Q4RegisterLoopDescriptor.derive(executionWidth: 64)) == nil, "a width other than 32 is refused")
    if let d, let data = try? JSONEncoder().encode(d),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        obj["oddBlockShiftBits"] = 8
        let tampered = (try? JSONSerialization.data(withJSONObject: obj)).flatMap {
            try? JSONDecoder().decode(Q4RegisterLoopDescriptor.self, from: $0)
        }
        if let t = tampered {
            h.check((try? t.validate()) == nil, "a decoded descriptor with a non-canonical shift fails validate")
        } else {
            h.check(false, "tampered descriptor decodes")
        }
    } else {
        h.check(false, "descriptor encodes for the tamper test")
    }
}

func checkPlan(_ h: inout Harness) {
    guard let d = try? Q4RegisterLoopDescriptor.derive(executionWidth: 32) else {
        h.check(false, "descriptor for plan tests")
        return
    }
    func reason(inDim: Int = 2816, outDim: Int = 64, offset: Int = 256, length: Int = 1 << 30, width: Int = 32,
                maxThreads: Int = 1024, staticBytes: Int = 0) -> String? {
        Q4RegisterLoopPlan.declineReason(inDim: inDim, outDim: outDim, weightOffset: offset, weightLength: length,
                                         descriptor: d, executionWidth: width, pipelineMaxThreads: maxThreads,
                                         staticThreadgroupBytes: staticBytes)
    }
    h.check(reason() == nil, "2816 x 64 at offset 256 is supported")
    h.check(reason(inDim: 96, outDim: 8) != nil && reason(offset: 2) != nil && reason(length: 256 + 63 * 1584) != nil,
            "odd block count, misaligned offset and short buffer are declined through the pair contract")
    h.check(reason(width: 64) != nil && reason(maxThreads: 16) != nil && reason(staticBytes: 4096) != nil,
            "width mismatch, thread maximum and non-zero static threadgroup memory are declined")
    let l = Q4RegisterLoopPlan.launch(outDim: 33, descriptor: d)
    h.check(l == Q4WeightStagePlan.controlLaunch(outDim: 33, pipelineMaxThreads: 1024) && l.threadgroups == 2,
            "candidate launch equals the width-32 control geometry")
}

func checkFetchMirror(_ h: inout Harness) {
    var bytes = [UInt8]()
    for i in 0..<36 { bytes.append(UInt8((i * 53 + 7) & 0xFF)) }
    guard let pair = Q4PackedWordPlan.pairWords(bytes[...]),
          let even = Q4RegisterLoopPlan.blockWords(pair: pair, odd: false),
          let odd = Q4RegisterLoopPlan.blockWords(pair: pair, odd: true),
          let evenBytes = Q4RegisterLoopPlan.blockBytes(even), let oddBytes = Q4RegisterLoopPlan.blockBytes(odd) else {
        h.check(false, "block words and bytes")
        return
    }
    h.check(evenBytes == Array(bytes[0..<18]), "even block: five pair words reproduce the first 18 bytes")
    h.check(oddBytes == Array(bytes[18..<36]), "odd block: funnel-shifted words reproduce bytes 18..35")
    h.check(even.count == 5 && odd.count == 5 && odd[4] == pair[8] >> 16 && (odd[0] & 0xFFFF) == (pair[4] >> 16),
            "odd block scale bits are the high half of pair word 4")
    h.check(Q4RegisterLoopPlan.blockWords(pair: [1, 2, 3], odd: false) == nil
            && Q4RegisterLoopPlan.blockBytes([1, 2]) == nil, "wrong word counts are refused")
}

func checkEmitter(_ h: inout Harness) {
    guard let d = try? Q4RegisterLoopDescriptor.derive(executionWidth: 32), let a = try? Q4RegisterLoopEmitter.emit(d),
          let b = try? Q4RegisterLoopEmitter.emit(d) else {
        h.check(false, "emitter produces the kernel")
        return
    }
    h.check(a == b && occurrences(a, "kernel void q4rl_matvec_W32_fetch5(") == 1, "deterministic single kernel")
    h.check(occurrences(a, "for (uint b = 0; b < blocksPerRow; b++) {") == 1
            && occurrences(a, "for (uint l = 0; l < 16u; l++) {") == 1
            && occurrences(a, "sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];") == 1
            && occurrences(a, "sum += d * (float(q >> 4) - 8.0f) * x[b * 32u + 16u + l];") == 1
            && occurrences(a, "float d = float(as_type<half>(dbits));") == 1,
            "production block loop, 16-term inner loop and the production term lines, each exactly once")
    h.check(occurrences(a, "threadgroup") == 0 && occurrences(a, "barrier") == 0 && occurrences(a, "simd_") == 0
            && occurrences(a, "float4") == 0 && occurrences(a, "fma(") == 0 && occurrences(a, "x[xb") == 0,
            "no threadgroup memory, barrier, SIMD, vector load, explicit fma or unrolled constant indices")
    h.check(occurrences(a, "b0 = pw[0]; b1 = pw[1]; b2 = pw[2]; b3 = pw[3]; b4 = pw[4];") == 1
            && occurrences(a, "b0 = (t4 >> 16u) | (t5 << 16u);") == 1 && occurrences(a, "b4 = t8 >> 16u;") == 1
            && occurrences(a, "uchar q = uchar((word >> ((bi & 3u) * 8u)) & 0xFFu);") == 1,
            "five-word even fetch, 16-bit odd funnel and in-loop byte extraction")
    h.check(Gemma4Runner.shaderSource.contains("for (uint b = 0; b < blocksPerRow; b++) {")
            && Gemma4Runner.shaderSource.contains("for (uint l = 0; l < 16u; l++) {")
            && Gemma4Runner.shaderSource.contains("sum += d * (float(q & 0x0Fu) - 8.0f) * x[b * 32u + l];"),
            "the production kernel carries the same loop shape and term line")
}

func checkContract(_ h: inout Harness) {
    typealias D = Q4RegisterLoopDiagnostics
    let plain = try? D.parseOptions(["--report", "o.json"])
    h.check(plain?.report == "o.json" && plain?.gguf == nil
            && (try? D.parseOptions(["--report", "o.json", "--gguf", "m.gguf"]))?.gguf == "m.gguf"
            && (try? D.parseOptions(["--gguf", "m.gguf"])) == nil
            && (try? D.parseOptions(["--report", "o.json", "--other", "x"])) == nil,
            "options: report required, gguf optional, unknown refused")
    let flags = ["fullModelQualified", "gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    var contract = false
    if let data = try? D.encodeReport(Q4RegisterLoopReport()),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let base = flags.allSatisfy { (obj[$0] as? Bool) == false } && (obj["kind"] as? String) == D.kind
            && (obj["closedPredecessors"] as? [String])?.count == 2
        for f in flags { obj[f] = true }
        obj["status"] = "correctnessPassed"
        if let raw = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(Q4RegisterLoopReport.self, from: raw),
           let again = try? D.encodeReport(decoded),
           let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
            contract = base && flags.allSatisfy { (back[$0] as? Bool) == false }
        }
    }
    h.check(contract, "every claim flag encodes false, tampered flags re-emit false, both closed predecessors are recorded")
    var r = Q4RegisterLoopReport()
    h.check(D.finalStatus(r) == .unsupported, "an empty report is unsupported")
    r.terminal = false
    h.check(D.finalStatus(r) == .notEnded, "a non-terminal run is not ended")
    r.terminal = true
    r.error = "x"
    h.check(D.finalStatus(r) == .aborted, "an error aborts")
    h.check(D.kind == "q4-register-loop-decode-component-v1" && Q4WeightStageDiagnostics.warmupPairs == 2
            && Q4WeightStageDiagnostics.timedPairs == 6 && Q4WeightStageDiagnostics.correctnessContract == 1e-4
            && Q4WeightStageCorpus.synthetic().count == 10,
            "reused fixed protocol constants and the unchanged corpus size")
}

func runAll() -> Int32 {
    var h = Harness()
    checkDescriptor(&h)
    checkPlan(&h)
    checkFetchMirror(&h)
    checkEmitter(&h)
    checkContract(&h)
    print("q4-register-loop-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
