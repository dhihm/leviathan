import Foundation
#if canImport(LeviathanCore)
import LeviathanCore
#endif

// A standalone real-Metal test. Nothing is skipped: any Metal failure is a test failure.
// There is no global mutable state, so the file compiles under both Swift 5 and Swift 6.

final class Checker {
    var passes = 0
    var failures = 0
    func check(_ ok: Bool, _ what: String) {
        if ok { passes += 1 } else { failures += 1; print("FAIL: " + what) }
    }
    func throwsError(_ what: String, _ match: (Error) -> Bool, _ body: () throws -> Void) {
        do { try body(); check(false, what + " (no error)") }
        catch { check(match(error), what + " (wrong error: \(error))") }
    }
}

func lcgWeights(_ n: Int, seed: UInt32) -> [Float] {
    var s = seed
    return (0..<n).map { _ in
        s = s &* 1664525 &+ 1013904223
        return Float(Int(s >> 8) % 2001 - 1000) / 2000
    }
}

func makeProgram(_ inputDim: Int, _ nodes: [DAGNode], out: Int, seed: UInt32) throws -> DAGProgram {
    let g = DAGGraph(inputDim: inputDim, nodes: nodes, outputValue: out)
    let wc = try g.validate().weightCount
    return try DAGProgram(graph: g, weights: lcgWeights(wc, seed: seed))
}

func close(_ a: [Float], _ b: [Float]) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= 1e-4 * (1 + abs($1)) }
}

func sampleInput(_ n: Int) -> [Float] { (0..<n).map { Float(($0 * 7) % 11) / 5 - 1 } }

func runAll() throws -> Checker {
    let c = Checker()
    let fp = DAGMetalBackend.shaderFingerprint
    c.check(!fp.isEmpty && fp == DAGMetalBackend.shaderFingerprint, "stable fingerprint")

    // 1. Negative alias canary: y = h + relu(h). h has two consumers, so no fusion is safe.
    // Where h < 0, y must equal h. An in-place relu would make y 0 there.
    let canaryNodes = [DAGNode(id: 1, op: .dense, inputs: [0], outputDim: 12),
                       DAGNode(id: 2, op: .relu, inputs: [1], outputDim: 12),
                       DAGNode(id: 3, op: .add, inputs: [1, 2], outputDim: 12)]
    let canary = try makeProgram(6, canaryNodes, out: 3, seed: 7)
    let hOnly = try makeProgram(6, [canaryNodes[0]], out: 1, seed: 7)
    let x6 = sampleInput(6)
    let h = try DAGScalarBackend.run(hOnly, input: x6)
    let canaryRef = try DAGScalarBackend.run(canary, input: x6)
    c.check(canary.safeFusionPairs.isEmpty, "canary has no safe fusion pairs")
    let neg = h.indices.filter { h[$0] < -1e-3 }
    c.check(!neg.isEmpty, "canary dense value has negative entries")
    let cp = try DAGMetalBackend.Prepared(program: canary, input: x6)
    c.check(cp.maxTotalThreadsPerThreadgroup >= 1 && cp.threadExecutionWidth >= 1, "pipeline caps")
    for fuse in [false, true] {
        let r = try cp.run(threadgroupWidth: cp.baselineThreadgroupWidth, fuseDenseRelu: fuse)
        c.check(close(r.output, canaryRef), "canary matches CPU fuse=\(fuse)")
        c.check(r.dispatchedSteps == 3
                && r.kernels == ["leviathan_dense", "leviathan_dag_relu", "leviathan_dag_add"],
                "canary dispatch fuse=\(fuse): \(r.kernels)")
        c.check(neg.allSatisfy { abs(r.output[$0] - h[$0]) <= 1e-4 * (1 + abs(h[$0])) },
                "canary keeps h where h<0 fuse=\(fuse)")
        c.check(r.gpuMicros > 0 && r.gpuMicros.isFinite, "canary gpuMicros")
        c.check(r.deviceName == cp.deviceName, "device name")
    }

    // 2. Safe dense->relu followed by a residual add of the graph input.
    let resNodes = [DAGNode(id: 1, op: .dense, inputs: [0], outputDim: 16),
                    DAGNode(id: 2, op: .relu, inputs: [1], outputDim: 16),
                    DAGNode(id: 3, op: .dense, inputs: [2], outputDim: 8),
                    DAGNode(id: 4, op: .add, inputs: [3, 0], outputDim: 8)]
    let res = try makeProgram(8, resNodes, out: 4, seed: 11)
    let x8 = sampleInput(8)
    let resRef = try DAGScalarBackend.run(res, input: x8)
    c.check(res.safeFusionPairs == [DAGFusionPair(dense: 1, relu: 2)], "residual fusion pair")
    let rp = try DAGMetalBackend.Prepared(program: res, input: x8)
    let rcap = rp.maxTotalThreadsPerThreadgroup
    let widths = Array(Set([1, 3, 7, 16, 32, rp.baselineThreadgroupWidth] + rp.candidateThreadgroupWidths))
        .filter { $0 <= min(rcap, cp.maxTotalThreadsPerThreadgroup) }.sorted()
    for w in widths {
        let plain = try rp.run(threadgroupWidth: w, fuseDenseRelu: false)
        let fused = try rp.run(threadgroupWidth: w, fuseDenseRelu: true)
        c.check(close(plain.output, resRef) && close(fused.output, resRef), "residual matches CPU w=\(w)")
        c.check(plain.dispatchedSteps == 4 && plain.kernels ==
                ["leviathan_dense", "leviathan_dag_relu", "leviathan_dense", "leviathan_dag_add"],
                "residual unfused dispatch w=\(w): \(plain.kernels)")
        c.check(fused.dispatchedSteps == 3 && fused.kernels ==
                ["leviathan_dense_relu", "leviathan_dense", "leviathan_dag_add"],
                "residual fused dispatch w=\(w): \(fused.kernels)")
        c.check(plain.gpuMicros > 0 && fused.gpuMicros > 0, "residual gpuMicros w=\(w)")
        let cw = try cp.run(threadgroupWidth: w, fuseDenseRelu: true)
        c.check(close(cw.output, canaryRef) && cw.dispatchedSteps == 3, "canary w=\(w)")
    }
    let a1 = try rp.run(threadgroupWidth: rp.baselineThreadgroupWidth, fuseDenseRelu: true)
    let a2 = try rp.run(threadgroupWidth: rp.baselineThreadgroupWidth, fuseDenseRelu: true)
    c.check(a1.output == a2.output, "repeated run deterministic")
    c.check(rp.baselineThreadgroupWidth == max(1, min(32, rcap, 8)), "baseline width")
    c.check(!rp.candidateThreadgroupWidths.isEmpty
            && rp.candidateThreadgroupWidths.allSatisfy { $0 >= 1 && $0 <= min(rcap, 8) }
            && Set(rp.candidateThreadgroupWidths).count == rp.candidateThreadgroupWidths.count,
            "candidate widths legal and deduplicated")

    // 3. Rejections.
    for bad in [0, -1, rcap + 1] {
        c.throwsError("reject width \(bad)", { ($0 as? DAGMetalError) == .invalidThreadgroupWidth(bad, cap: rcap) }) {
            _ = try rp.run(threadgroupWidth: bad, fuseDenseRelu: false)
        }
    }
    c.throwsError("reject short input", { ($0 as? DAGExecutionError) == .inputCountMismatch(expected: 8, got: 7) }) {
        _ = try DAGMetalBackend.Prepared(program: res, input: Array(x8.prefix(7)))
    }
    var nanX = x8
    nanX[2] = .nan
    c.throwsError("reject NaN input", { ($0 as? DAGExecutionError) == .nonFiniteInput(2) }) {
        _ = try DAGMetalBackend.Prepared(program: res, input: nanX)
    }
    var infX = x8
    infX[5] = .infinity
    c.throwsError("reject Inf input", { ($0 as? DAGExecutionError) == .nonFiniteInput(5) }) {
        _ = try DAGMetalBackend.Prepared(program: res, input: infX)
    }
    return c
}

do {
    let c = try runAll()
    print("dag_metal_tests: \(c.passes) passed, \(c.failures) failed")
    exit(c.failures == 0 ? 0 : 1)
} catch {
    print("dag_metal_tests: FAIL unexpected error \(error)")
    exit(1)
}
