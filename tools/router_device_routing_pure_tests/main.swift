import Foundation
import LeviathanCore

// Phase 57 part 1 pure tests: record geometry, expectations, plan validation, shared-body source structure,
// record encoder/decoder, per-token CPU verifier, report contract, options and fixed component cases.
// No Metal device, no GGUF, no model weights.

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
    let l = RouterSelectionIR.gemma4.layout
    h.check(RouterDeviceRoutingLayout.bindingFailures(layers: 30, layout: l).isEmpty, "30-layer binding offsets aligned and in bounds")
    h.check(RouterDeviceRoutingLayout.recordBufferLength(layers: 30, layout: l) == 84480, "record buffer is 30 x 2816 bytes")
    var ok = true
    for layer in 0..<30 {
        let base = layer * 2816
        if RouterDeviceRoutingLayout.recordBase(layer: layer, layout: l) != base
            || RouterDeviceRoutingLayout.guOffsetsBinding(layer: layer, layout: l) != base + 256
            || RouterDeviceRoutingLayout.dnOffsetsBinding(layer: layer, layout: l) != base + 512 {
            ok = false
        }
        for slot in 0..<8 where RouterDeviceRoutingLayout.coefficientBinding(layer: layer, slot: slot, layout: l)
            != base + 768 + 256 * slot {
            ok = false
        }
    }
    h.check(ok, "record, GU/DN offset and coefficient slot bindings are 256-aligned per layer")
    h.check(RouterDeviceRoutingLayout.scoreLogLength(layers: 30, experts: 128) == 15360
            && RouterDeviceRoutingLayout.scoreLogBase(layer: 7, experts: 128) == 3584, "score log geometry")
    h.check(!RouterDeviceRoutingLayout.bindingFailures(layers: 0, layout: l).isEmpty, "zero layers refused")
}

func checkExpectation(_ h: inout Harness) {
    let e = RouterDeviceRoutingExpectation.derive(layers: 30)
    h.check(e.deviceRoutedKernels(controlKernels: 1349) == 1379, "kernels per forward: control + 30")
    h.check(e.deviceRoutedCommandBuffers(controlCommandBuffers: 37) == 7, "command buffers per forward: 37 -> 7")
    h.check(e.deviceRoutedCommandBuffers(controlCommandBuffers: 20) == nil, "impossible buffer count refused")
    h.check(e.gpuSelectionsPerForward == 30 && e.cpuSelectionsPerForwardControl == 30
            && e.cpuSelectionsPerForwardDeviceRouted == 0 && e.recordChainsPerForward == 30, "selection expectations")
}

func plan(_ layers: Int, mutate: (inout [RouterDeviceRoutingSSAOp]) -> Void) -> [RouterDeviceRoutingSSAOp] {
    var ops = [RouterDeviceRoutingSSAOp]()
    for layer in 0..<layers {
        ops.append(RouterDeviceRoutingSSAOp(op: "rmsnorm", layer: layer, topK: nil))
        ops.append(RouterDeviceRoutingSSAOp(op: "router_topk", layer: layer, topK: 8))
        ops.append(RouterDeviceRoutingSSAOp(op: "matvec", layer: layer, topK: nil))
        ops.append(RouterDeviceRoutingSSAOp(op: "expert_matvec_chain", layer: layer, topK: nil))
    }
    mutate(&ops)
    return ops
}

func checkPlan(_ h: inout Harness) {
    func valid(_ ops: [RouterDeviceRoutingSSAOp]) -> Bool {
        (try? RouterDeviceRoutingPlan.validate(ops: ops, layers: 30, topK: 8)) != nil
    }
    let good = try? RouterDeviceRoutingPlan.validate(ops: plan(30) { _ in }, layers: 30, topK: 8)
    h.check(good?.router.count == 30 && good?.chain.count == 30 && good?.router.first == 1 && good?.chain.first == 3,
            "canonical plan validates with router and chain indices")
    h.check(!valid(plan(30) { $0.remove(at: 15) }), "missing expert chain refused")
    h.check(!valid(plan(30) { $0.append(RouterDeviceRoutingSSAOp(op: "router_topk", layer: 2, topK: 8)) }), "duplicate router refused")
    h.check(!valid(plan(30) { $0[1] = RouterDeviceRoutingSSAOp(op: "router_topk", layer: 0, topK: 4) }), "wrong topK refused")
    h.check(!valid(plan(30) { $0.swapAt(1, 3) }), "chain before router refused")
    h.check(!valid(plan(30) { $0[1] = RouterDeviceRoutingSSAOp(op: "router_topk", layer: nil, topK: 8) }), "router without layer refused")
    h.check(!valid(plan(30) { $0[5] = RouterDeviceRoutingSSAOp(op: "router_topk", layer: 30, topK: 8) }), "out-of-range layer refused")
}

func checkSource(_ h: inout Harness) {
    guard let s = try? RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4),
          let s2 = try? RouterSelectionKernel.emitSource(RouterSelectionIR.gemma4) else {
        h.check(false, "selector source emits")
        return
    }
    h.check(s == s2, "emission deterministic")
    h.check(occurrences(s, "kernel void " + RouterSelectionKernel.selectFunction + "(") == 1
            && occurrences(s, "kernel void " + RouterSelectionKernel.selectRuntimeFunction + "(") == 1
            && occurrences(s, "kernel void " + RouterSelectionKernel.consumeFunction + "(") == 1, "three entry points, one each")
    h.check(occurrences(s, "rs_select_body(") == 3, "one shared body: definition plus two calls")
    h.check(occurrences(s, "rs_add(rs_mul(p, r)") == 1 && occurrences(s, "RS_BUDGET * m.hi") == 1,
            "selection arithmetic emitted exactly once")
    h.check(occurrences(s, "scoreLog[lid] = scoreRow[lid];") == 1, "runtime entry copies its score row to the log")
}

func layerScores(_ layer: Int) -> [Float] { (0..<128).map { Float((($0 * 73 + layer * 29) % 128) - 64) / 16 } }
func layerScales(_ layer: Int) -> [Float] { (0..<128).map { Float(1 + (($0 + layer) % 7)) / 4 } }

func checkVerifier(_ h: inout Harness) {
    let l = RouterSelectionIR.gemma4.layout
    let table = RouterOffsetTable.synthetic(layers: 30, expertCount: 128, expertBytes: 16)
    var records = [RouterSelectionGPURecord]()
    var scores = [[UInt32]]()
    var scales = [[Float]]()
    var words = [[UInt32]]()
    for layer in 0..<30 {
        let sc = layerScales(layer)
        let s = layerScores(layer)
        guard let cpu = RouterSelectionKernel.cpuReference(scores: s, scales: sc, topK: 8) else {
            h.check(false, "cpu reference for verifier fixture")
            return
        }
        let w = RouterDeviceRoutingRecordEncoder.words(
            ids: cpu.ids.map { UInt32($0) }, flags: 0, layer: UInt32(layer), rawBits: cpu.rawBits,
            weightBits: cpu.weightBits, gu: cpu.ids.map { table.offset(layer: layer, dn: false, expert: $0) },
            dn: cpu.ids.map { table.offset(layer: layer, dn: true, expert: $0) }, layout: l)
        words.append(w)
        records.append(w.withUnsafeBytes { RouterSelectionKernel.decodeRecord($0, layout: l) })
        scores.append(s.map { $0.bitPattern })
        scales.append(sc)
    }
    func check(_ recs: [RouterSelectionGPURecord], _ sb: [[UInt32]]) -> RouterDeviceRoutingTokenCheck {
        RouterDeviceRoutingVerifier.check(position: 4, records: recs, scoreBits: sb, scales: scales, table: table, topK: 8)
    }
    func mutated(_ layer: Int, _ change: (inout [UInt32]) -> Void) -> [RouterSelectionGPURecord] {
        var w = words[layer]
        change(&w)
        var r = records
        r[layer] = w.withUnsafeBytes { RouterSelectionKernel.decodeRecord($0, layout: l) }
        return r
    }
    h.check(records.allSatisfy { $0.paddingClean } && check(records, scores).exact, "encoded CPU decisions verify exactly")
    h.check(check(mutated(3) { $0[(l.coefficientOffset + 2 * l.coefficientStride) / 4] ^= 1 }, scores).coefficientMismatchLayers == [3],
            "one coefficient bit flip detected")
    h.check(check(mutated(4) { $0[l.rawCoefficientOffset / 4] ^= 1 }, scores).quotientMismatchLayers == [4], "quotient bit flip detected")
    h.check(check(mutated(5) { $0[l.flagsOffset / 4] = 16 }, scores).flaggedLayers == [5], "flagged record detected")
    h.check(check(mutated(7) { $0[l.guOffsetsOffset / 4] ^= 16 }, scores).offsetMismatchLayers == [7], "wrong GU offset detected")
    h.check(check(mutated(9) { $0[l.magicOffset / 4] = 0 }, scores).recordErrorLayers == [9], "unwritten record detected")
    h.check(check(mutated(11) { $0.swapAt(0, 1) }, scores).idMismatchLayers == [11], "swapped ids detected")
    h.check(check(mutated(12) { $0[700] = 1 }, scores).recordErrorLayers == [12], "dirty padding detected")
    var badScores = scores
    badScores[13][0] = Float.nan.bitPattern
    h.check(check(records, badScores).scoreErrorLayers == [13], "non-finite logged score detected")
    h.check(check(Array(records.prefix(29)), scores).recordErrorLayers.count == 30, "missing layer record refused")
}

func checkReport(_ h: inout Harness) {
    let flags = ["gainQualified", "installable", "productionPerformanceClaim", "realE4GateComplete", "scalesSerialized"]
    guard let data = try? RouterDeviceRoutingComponent.encodeReport(RouterDeviceRoutingComponentReport()),
          let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        h.check(false, "component report encodes")
        return
    }
    h.check(flags.allSatisfy { (obj[$0] as? Bool) == false }, "contract fields encode false")
    var tampered = obj
    for key in flags { tampered[key] = true }
    if let raw = try? JSONSerialization.data(withJSONObject: tampered),
       let decoded = try? JSONDecoder().decode(RouterDeviceRoutingComponentReport.self, from: raw),
       let again = try? RouterDeviceRoutingComponent.encodeReport(decoded),
       let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
        h.check(flags.allSatisfy { (back[$0] as? Bool) == false }, "decoded report never re-emits a true contract field")
    } else {
        h.check(false, "tampered report round trip")
    }
    h.check(RouterDeviceRoutingComponent.status(RouterDeviceRoutingComponentReport()) == .aborted, "empty report is aborted")
    h.check(RouterDeviceRoutingComponent.readyToken == RouterOverlapModelGate.readyToken
            && RouterDeviceRoutingComponent.goToken == RouterOverlapModelGate.goToken
            && RouterDeviceRoutingComponent.endToken == RouterOverlapModelGate.endToken, "protocol tokens are the gate constants")
    h.check(RouterDeviceRoutingComponent.goAccepted("GO") && !RouterDeviceRoutingComponent.goAccepted(nil)
            && !RouterDeviceRoutingComponent.goAccepted("GO "), "only exact GO accepted")
    let parsed = try? RouterDeviceRoutingComponent.parseOptions(["--gguf", "m.gguf", "--region", "r.json", "--report", "x.json"])
    h.check(parsed?.gguf == "m.gguf" && parsed?.report == "x.json", "options parse")
    h.check((try? RouterDeviceRoutingComponent.parseOptions(["--gguf", "m.gguf"])) == nil
            && (try? RouterDeviceRoutingComponent.parseOptions(["--gguf", "a", "--gguf", "b", "--region", "r", "--report", "x"])) == nil,
            "missing or duplicate options refused")
}

func checkCases(_ h: inout Harness) {
    let a = RouterDeviceRoutingComponent.entryCases()
    let b = RouterDeviceRoutingComponent.entryCases()
    h.check(a.count == 102 && a.filter { $0.declared.isEmpty }.count == 94, "fixed 94 ordinary + 8 declared refusal cases")
    h.check(zip(a, b).allSatisfy { $0.0.name == $0.1.name && $0.0.scores.map { $0.bitPattern } == $0.1.scores.map { $0.bitPattern }
                && $0.0.scales.map { $0.bitPattern } == $0.1.scales.map { $0.bitPattern } }, "cases deterministic")
    h.check(a.filter { $0.declared.isEmpty }.allSatisfy {
        RouterSelectionKernel.cpuReference(scores: $0.scores, scales: $0.scales, topK: 8) != nil }, "ordinary cases have CPU references")
    h.check(Set(a.map { $0.name }).count == a.count, "case names unique")
}

func runAll() -> Int32 {
    var h = Harness()
    checkLayout(&h)
    checkExpectation(&h)
    checkPlan(&h)
    checkSource(&h)
    checkVerifier(&h)
    checkReport(&h)
    checkCases(&h)
    print("router-device-routing-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
