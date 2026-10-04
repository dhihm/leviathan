import Foundation
import LeviathanCore

// Standalone stage-1 checks for the typed value DAG: representation, validation and CPU references.
// The weights here are small hand-written test values. Stage 3 end-to-end needs real Torch captures.
var failures = 0
func check(_ ok: Bool, _ name: String) {
    print((ok ? "PASS " : "FAIL ") + name)
    if !ok { failures += 1 }
}
func expectError<E: Error & Equatable>(_ expected: E, _ name: String, _ body: () throws -> Void) {
    do { try body(); check(false, name + " (accepted)") }
    catch let e as E { check(e == expected, name + " -> \(e)") }
    catch { check(false, name + " wrong error \(error)") }
}
func node(_ id: Int, _ op: DAGOp, _ inputs: [Int], _ dim: Int) -> DAGNode {
    DAGNode(id: id, op: op, inputs: inputs, outputDim: dim)
}
func runBoth(_ p: DAGProgram, _ x: [Float]) throws -> ([Float], [Float]) {
    (try DAGScalarBackend.run(p, input: x), try DAGAccelerateBackend.run(p, input: x))
}
let eye3: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
let x3: [Float] = [1, -2, 3]

// relu(dense(x)) + x
let residual = DAGGraph(inputDim: 3, nodes: [node(1, .dense, [0], 3), node(2, .relu, [1], 3), node(3, .add, [2, 0], 3)], outputValue: 3)
do {
    let p = try DAGProgram(graph: residual, weights: eye3 + [-1, 0, 1])
    let (s, a) = try runBoth(p, x3)
    check(s == [1, -2, 7] && a == [1, -2, 7], "residual known output \(s) \(a)")
    check(p.safeFusionPairs == [DAGFusionPair(dense: 1, relu: 2)], "residual dense+relu fusion safe")
    check(p.consumerCounts == [2, 1, 1, 1] && p.lastUse == [3, 2, 3, 4], "residual consumers/lifetimes")
    check(p.bufferDims == [3, 3, 3, 3] && p.outputDim == 3, "residual dims")
    let decoded = try JSONDecoder().decode(DAGGraph.self, from: JSONEncoder().encode(residual))
    check(decoded == residual, "codable round trip")
} catch { check(false, "residual threw \(error)") }

// h + relu(h): relu must not mutate h, so the negative value survives. h has two consumers, so fusion is unsafe.
do {
    let g = DAGGraph(inputDim: 3, nodes: [node(1, .dense, [0], 3), node(2, .relu, [1], 3), node(3, .add, [1, 2], 3)], outputValue: 3)
    let p = try DAGProgram(graph: g, weights: eye3 + [0, 0, 0])
    let (s, a) = try runBoth(p, x3)
    check(s == [2, -2, 6] && a == [2, -2, 6], "fanout keeps negative \(s) \(a)")
    check(p.safeFusionPairs.isEmpty && p.consumerCounts[1] == 2, "fanout fusion unsafe")
} catch { check(false, "fanout threw \(error)") }

do {
    let g = DAGGraph(inputDim: 3, nodes: [node(1, .dense, [0], 3), node(2, .dense, [0], 3), node(3, .relu, [1], 3), node(4, .add, [3, 2], 3)], outputValue: 4)
    let p = try DAGProgram(graph: g, weights: (0..<24).map { Float($0) / 10 })
    check(p.safeFusionPairs.isEmpty, "non-adjacent relu fusion unsafe")
} catch { check(false, "nonadjacent threw \(error)") }

do {
    let g = DAGGraph(inputDim: 4, nodes: [node(1, .dense, [0], 5), node(2, .relu, [1], 5), node(3, .dense, [2], 4), node(4, .add, [3, 0], 4), node(5, .relu, [4], 4)], outputValue: 5)
    let p = try DAGProgram(graph: g, weights: (0..<49).map { Float(($0 * 7) % 11 - 5) / 8 })
    let (s, a) = try runBoth(p, [0.5, -1, 2, -0.25])
    let d = maxAbsDiff(s, a)
    check(d != nil && d! <= 1e-5, "CPU agreement diff \(String(describing: d))")
    check(p.safeFusionPairs == [DAGFusionPair(dense: 1, relu: 2)] && p.layouts[3]?.weightOffset == 25, "agreement fusion/layout")
} catch { check(false, "agreement threw \(error)") }

func rejects(_ e: DAGValidationError, _ name: String, _ mutate: (inout DAGGraph) -> Void) {
    var g = residual
    mutate(&g)
    expectError(e, name) { _ = try g.validate() }
}
rejects(.badFormat("x"), "format") { $0.format = "x" }
rejects(.badNodeCount(0), "empty") { $0.nodes = [] }
rejects(.badNodeCount(65), "too many nodes") { g in g.nodes = (1...65).map { node($0, .relu, [$0 - 1], 3) } }
rejects(.badDimension("inputDim", 0), "zero inputDim") { $0.inputDim = 0 }
rejects(.badDimension("nodes[0].outputDim", 4097), "dim bound") { $0.nodes[0].outputDim = 4097 }
rejects(.badNodeID(index: 1, id: 3), "id gap") { $0.nodes[1].id = 3 }
rejects(.badNodeID(index: 1, id: 1), "duplicate id") { $0.nodes[1].id = 1 }
rejects(.undefinedInput(node: 1, input: 2), "forward ref") { $0.nodes[0].inputs = [2] }
rejects(.undefinedInput(node: 2, input: 2), "self cycle") { $0.nodes[1].inputs = [2] }
rejects(.undefinedInput(node: 3, input: -1), "negative ref") { $0.nodes[2].inputs = [2, -1] }
rejects(.badArity(node: 3, expected: 2, got: 1), "add arity") { $0.nodes[2].inputs = [2] }
rejects(.badArity(node: 2, expected: 1, got: 2), "relu arity") { $0.nodes[1].inputs = [1, 1] }
rejects(.shapeMismatch(node: 2, expected: 3, got: 4), "relu width") { $0.nodes[1].outputDim = 4 }
rejects(.shapeMismatch(node: 3, expected: 4, got: 3), "add no broadcast") { $0.nodes[0].outputDim = 4; $0.nodes[1].outputDim = 4 }
rejects(.noDense, "no dense") { $0.nodes[0].op = .relu }
rejects(.outputUndefined(0), "output is input") { $0.outputValue = 0 }
rejects(.outputUndefined(4), "output undefined") { $0.outputValue = 4 }
rejects(.deadNode(3), "dead node") { $0.outputValue = 2 }
rejects(.weightsTooLarge(4096 * 4096 + 4096, DAGGraph.maxWeightFloats), "weight bound") {
    $0.inputDim = 4096; $0.nodes = [node(1, .dense, [0], 4096)]; $0.outputValue = 1
}
expectError(DAGValidationError.weightCountMismatch(expected: 12, got: 11), "weight count") {
    _ = try DAGProgram(graph: residual, weights: eye3 + [0, 0])
}
expectError(DAGValidationError.nonFiniteWeight(4), "NaN weight") {
    var w = eye3 + [0, 0, 0]; w[4] = .nan
    _ = try DAGProgram(graph: residual, weights: w)
}
if let p = try? DAGProgram(graph: residual, weights: eye3 + [0, 0, 0]) {
    expectError(DAGExecutionError.inputCountMismatch(expected: 3, got: 2), "short input") { _ = try DAGScalarBackend.run(p, input: [1, 2]) }
    expectError(DAGExecutionError.nonFiniteInput(1), "non-finite input") { _ = try DAGAccelerateBackend.run(p, input: [1, .infinity, 0]) }
} else { check(false, "program build for input checks") }

print(failures == 0 ? "ALL PASS" : "\(failures) FAILURES")
exit(failures == 0 ? 0 : 1)
