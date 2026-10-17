import Accelerate
import Foundation

/// Typed value-DAG ops, encoded in JSON as strings. Backends choose this path from the validated
/// graph format and ops only. They never choose it from a model name.
public enum DAGOp: String, Codable, Equatable, Sendable {
    case dense, relu, add
    /// The exact number of input values the op uses.
    public var arity: Int { self == .add ? 2 : 1 }
}

/// A node that produces a new, immutable value. The value's id equals `id` (1...N).
/// Value 0 is the graph input. No node mutates any value in place.
public struct DAGNode: Codable, Equatable, Sendable {
    public var id: Int
    public var op: DAGOp
    public var inputs: [Int]
    public var outputDim: Int
    public init(id: Int, op: DAGOp, inputs: [Int], outputDim: Int) {
        self.id = id; self.op = op; self.inputs = inputs; self.outputDim = outputDim
    }
}

/// A static value DAG: FP32, batch 1, one input and one output. It has no broadcast and no control flow.
public struct DAGGraph: Codable, Equatable, Sendable {
    public static let formatID = "leviathan-value-dag-1"
    public static let maxNodes = Limits.maxOps
    public static let maxWeightFloats = Limits.maxWeightBytes / MemoryLayout<Float>.size
    public var format: String
    public var inputDim: Int
    public var nodes: [DAGNode]
    public var outputValue: Int
    public init(format: String = DAGGraph.formatID, inputDim: Int, nodes: [DAGNode], outputValue: Int) {
        self.format = format; self.inputDim = inputDim; self.nodes = nodes; self.outputValue = outputValue
    }
}

public enum DAGValidationError: Error, Equatable, Sendable {
    case badFormat(String)
    case badNodeCount(Int)
    case badDimension(String, Int)
    case badNodeID(index: Int, id: Int)
    case badArity(node: Int, expected: Int, got: Int)
    case undefinedInput(node: Int, input: Int)
    case shapeMismatch(node: Int, expected: Int, got: Int)
    case noDense
    case outputUndefined(Int)
    case deadNode(Int)
    case weightsTooLarge(Int, Int)
    case weightCountMismatch(expected: Int, got: Int)
    case nonFiniteWeight(Int)
}

public enum DAGExecutionError: Error, Equatable, Sendable {
    case inputCountMismatch(expected: Int, got: Int)
    case nonFiniteInput(Int)
}

private typealias VE = DAGValidationError

extension DAGGraph {
    /// Runs strict validation. Node ids must run 1...N in order, and each input must refer to an
    /// earlier value. Because of this, forward references, self-references and cycles all fail as
    /// undefinedInput. Returns the width of each value (index = value id) and the exact weight float
    /// count. Overflow-checked bounds run before anything is allocated. Having no dead nodes
    /// implies outputValue == N.
    public func validate() throws -> (valueDims: [Int], weightCount: Int) {
        guard format == DAGGraph.formatID else { throw VE.badFormat(format) }
        guard !nodes.isEmpty, nodes.count <= DAGGraph.maxNodes else { throw VE.badNodeCount(nodes.count) }
        try dagCheckDim(inputDim, "inputDim")
        var dims = [inputDim]
        var weightCount = 0
        var sawDense = false
        for (i, n) in nodes.enumerated() {
            guard n.id == i + 1 else { throw VE.badNodeID(index: i, id: n.id) }
            guard n.inputs.count == n.op.arity else {
                throw VE.badArity(node: n.id, expected: n.op.arity, got: n.inputs.count)
            }
            for v in n.inputs where v < 0 || v >= n.id { throw VE.undefinedInput(node: n.id, input: v) }
            try dagCheckDim(n.outputDim, "nodes[\(i)].outputDim")
            let a = dims[n.inputs[0]]
            switch n.op {
            case .dense:
                sawDense = true
                let (w, o1) = a.multipliedReportingOverflow(by: n.outputDim)
                let (t, o2) = w.addingReportingOverflow(n.outputDim)
                let (s, o3) = weightCount.addingReportingOverflow(t)
                guard !o1, !o2, !o3, s <= DAGGraph.maxWeightFloats else {
                    throw VE.weightsTooLarge(o1 || o2 || o3 ? Int.max : s, DAGGraph.maxWeightFloats)
                }
                weightCount = s
            case .relu:
                guard n.outputDim == a else { throw VE.shapeMismatch(node: n.id, expected: a, got: n.outputDim) }
            case .add:
                let b = dims[n.inputs[1]]
                guard b == a else { throw VE.shapeMismatch(node: n.id, expected: a, got: b) }
                guard n.outputDim == a else { throw VE.shapeMismatch(node: n.id, expected: a, got: n.outputDim) }
            }
            dims.append(n.outputDim)
        }
        guard sawDense else { throw VE.noDense }
        guard outputValue >= 1, outputValue <= nodes.count else { throw VE.outputUndefined(outputValue) }
        var live = [Bool](repeating: false, count: nodes.count + 1)
        live[outputValue] = true
        for id in stride(from: nodes.count, through: 1, by: -1) where live[id] {
            for v in nodes[id - 1].inputs { live[v] = true }
        }
        if let dead = (1...nodes.count).first(where: { !live[$0] }) { throw VE.deadNode(dead) }
        return (dims, weightCount)
    }
}

private func dagCheckDim(_ v: Int, _ name: String) throws {
    guard v >= 1, v <= Limits.maxDim else { throw VE.badDimension(name, v) }
}

/// Contiguous weights for one dense node, in the same layout as the legacy weights:
/// W[inDim, outDim] row-major, followed by bias[outDim].
public struct DAGDenseLayout: Codable, Equatable, Sendable {
    public let node: Int, inDim: Int, outDim: Int, weightOffset: Int, weightCount: Int, biasOffset: Int
    public init(node: Int, inDim: Int, outDim: Int, weightOffset: Int, weightCount: Int, biasOffset: Int) {
        self.node = node; self.inDim = inDim; self.outDim = outDim
        self.weightOffset = weightOffset; self.weightCount = weightCount; self.biasOffset = biasOffset
    }
}

/// A dense node directly followed (id + 1) by a relu that is its only consumer. The graph output
/// counts as a consumer. A fusing backend writes only the relu value, so the dense value is dead.
public struct DAGFusionPair: Codable, Equatable, Sendable {
    public let dense: Int, relu: Int
    public init(dense: Int, relu: Int) { self.dense = dense; self.relu = relu }
}

/// A validated graph plus its imported weights. Every value has its own buffer, with no reuse yet.
/// Lifetimes are computed now for checks and for later buffer planning.
public struct DAGProgram: Sendable {
    public let graph: DAGGraph
    public let weights: [Float]
    /// Dense layouts keyed by node id, packed in node order.
    public let layouts: [Int: DAGDenseLayout]
    /// Width of each value (index = value id, 0 = input).
    public let bufferDims: [Int]
    /// Number of consumer edges per value, plus 1 for the graph output.
    public let consumerCounts: [Int]
    /// Id of the last node that uses each value. nodes.count + 1 means the value is the graph output.
    public let lastUse: [Int]
    public let safeFusionPairs: [DAGFusionPair]
    public var outputDim: Int { bufferDims[graph.outputValue] }

    public init(graph: DAGGraph, weights: [Float]) throws {
        let (dims, expected) = try graph.validate()
        guard weights.count == expected else { throw VE.weightCountMismatch(expected: expected, got: weights.count) }
        if let bad = weights.firstIndex(where: { !$0.isFinite }) { throw VE.nonFiniteWeight(bad) }
        let n = graph.nodes.count
        var layouts: [Int: DAGDenseLayout] = [:]
        var offset = 0
        var counts = [Int](repeating: 0, count: n + 1)
        var last = [Int](repeating: 0, count: n + 1)
        for node in graph.nodes {
            for v in node.inputs { counts[v] += 1; last[v] = node.id }
            if node.op == .dense {
                let inDim = dims[node.inputs[0]]
                let wc = inDim * node.outputDim
                layouts[node.id] = DAGDenseLayout(node: node.id, inDim: inDim, outDim: node.outputDim,
                                                  weightOffset: offset, weightCount: wc, biasOffset: offset + wc)
                offset += wc + node.outputDim
            }
        }
        counts[graph.outputValue] += 1
        last[graph.outputValue] = n + 1
        var pairs: [DAGFusionPair] = []
        for node in graph.nodes where node.op == .relu {
            let d = node.inputs[0]
            if d >= 1, d == node.id - 1, graph.nodes[d - 1].op == .dense, counts[d] == 1 {
                pairs.append(DAGFusionPair(dense: d, relu: node.id))
            }
        }
        self.graph = graph; self.weights = weights; self.layouts = layouts
        self.bufferDims = dims; self.consumerCounts = counts; self.lastUse = last; self.safeFusionPairs = pairs
    }
}

private func dagCheckInput(_ p: DAGProgram, _ x: [Float]) throws {
    guard x.count == p.graph.inputDim else {
        throw DAGExecutionError.inputCountMismatch(expected: p.graph.inputDim, got: x.count)
    }
    if let i = x.firstIndex(where: { !$0.isFinite }) { throw DAGExecutionError.nonFiniteInput(i) }
}

/// Independent scalar reference that uses plain Swift loops. Each node writes a new value array.
public enum DAGScalarBackend {
    public static func run(_ program: DAGProgram, input: [Float]) throws -> [Float] {
        try dagCheckInput(program, input)
        let w = program.weights
        var values: [[Float]] = [input]
        for node in program.graph.nodes {
            let x = values[node.inputs[0]]
            var out = [Float](repeating: 0, count: node.outputDim)
            switch node.op {
            case .dense:
                let l = program.layouts[node.id]!
                for c in 0..<l.outDim {
                    var s: Float = 0
                    for k in 0..<l.inDim { s += x[k] * w[l.weightOffset + k * l.outDim + c] }
                    out[c] = s + w[l.biasOffset + c]
                }
            case .relu:
                for i in out.indices { out[i] = x[i] > 0 ? x[i] : 0 }
            case .add:
                let y = values[node.inputs[1]]
                for i in out.indices { out[i] = x[i] + y[i] }
            }
            values.append(out)
        }
        return values[program.graph.outputValue]
    }
}

/// Accelerate (vDSP) reference. Each node writes a new value array.
public enum DAGAccelerateBackend {
    public static func run(_ program: DAGProgram, input: [Float]) throws -> [Float] {
        try dagCheckInput(program, input)
        var values: [[Float]] = [input]
        for node in program.graph.nodes {
            let x = values[node.inputs[0]]
            let n = vDSP_Length(node.outputDim)
            var out = [Float](repeating: 0, count: node.outputDim)
            switch node.op {
            case .dense:
                let l = program.layouts[node.id]!
                var prod = [Float](repeating: 0, count: l.outDim)
                program.weights.withUnsafeBufferPointer { wp in
                    let base = wp.baseAddress!
                    vDSP_mmul(x, 1, base + l.weightOffset, 1, &prod, 1, 1, n, vDSP_Length(l.inDim))
                    vDSP_vadd(prod, 1, base + l.biasOffset, 1, &out, 1, n)
                }
            case .relu:
                let zeros = [Float](repeating: 0, count: node.outputDim)
                vDSP_vmax(x, 1, zeros, 1, &out, 1, n)
            case .add:
                vDSP_vadd(x, 1, values[node.inputs[1]], 1, &out, 1, n)
            }
            values.append(out)
        }
        return values[program.graph.outputValue]
    }
}
