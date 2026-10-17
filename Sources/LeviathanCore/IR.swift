import Foundation

/// A single operation in the graph IR. The milestone supports exactly two op
/// types, both operating on row-major `[batch, features]` float32 tensors:
/// - `dense`: affine transform `y = x * W + b` with `W` of shape `[inDim, outDim]`
/// - `relu`: elementwise `max(x, 0)`
///
/// JSON encoding: dense is `{"dense": {"outDim": 32}}`, relu is `"relu"`.
public enum GraphOp: Equatable, Sendable {
    case dense(outDim: Int)
    case relu
}

extension GraphOp: Codable {
    private enum CodingError: Error { case malformedOp(String) }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let name = try? container.decode(String.self) {
            guard name == "relu" else { throw CodingError.malformedOp(name) }
            self = .relu
            return
        }
        guard let dict = try? container.decode([String: [String: Int]].self),
              let outDim = dict["dense"]?["outDim"], outDim > 0 else {
            throw CodingError.malformedOp(
                #"expected "relu" or {"dense": {"outDim": N}}"#)
        }
        self = .dense(outDim: outDim)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .relu:
            try container.encode("relu")
        case let .dense(outDim):
            try container.encode(["dense": ["outDim": outDim]])
        }
    }
}

/// Source-level graph description, deserialized from JSON:
/// `{"inputDim": 16, "ops": [{"dense": {"outDim": 32}}, {"relu": null}]}`
public struct GraphIR: Codable, Equatable, Sendable {
    public var inputDim: Int
    public var ops: [GraphOp]

    public init(inputDim: Int, ops: [GraphOp]) {
        self.inputDim = inputDim
        self.ops = ops
    }

    /// Output feature dimension after all ops have been applied.
    public var outputDim: Int {
        var dim = inputDim
        for op in ops {
            if case let .dense(outDim) = op { dim = outDim }
        }
        return dim
    }
}

public enum ValidationError: Error, Equatable, Sendable {
    case nonPositiveDimension(String)
    case dimensionTooLarge(String, Int, Int)
    case tooManyOps(Int, Int)
    case emptyGraph
    case tensorTooLarge(Int, Int)
}

/// Validates structural constraints and resource limits of a graph.
/// Returns the ordered per-op feature dimensions on success.
public func validate(_ graph: GraphIR) throws -> [Int] {
    guard !graph.ops.isEmpty else { throw ValidationError.emptyGraph }
    guard graph.ops.count <= Limits.maxOps else {
        throw ValidationError.tooManyOps(graph.ops.count, Limits.maxOps)
    }
    var dims: [Int] = []
    var current = try checkDim(graph.inputDim, "inputDim")
    dims.append(current)
    for (index, op) in graph.ops.enumerated() {
        switch op {
        case let .dense(outDim):
            let d = try checkDim(outDim, "ops[\(index)].dense.outDim")
            // Intermediate tensor elements are bounded by batch * in * out
            // only for the weights; tensor buffers are batch * dim, but the
            // weight matrix itself is a tensor we allocate.
            let weightElements = current * d
            guard weightElements <= Limits.maxTensorElements else {
                throw ValidationError.tensorTooLarge(weightElements, Limits.maxTensorElements)
            }
            current = d
        case .relu:
            break
        }
        dims.append(current)
    }
    return dims
}

private func checkDim(_ value: Int, _ name: String) throws -> Int {
    guard value > 0 else { throw ValidationError.nonPositiveDimension(name) }
    guard value <= Limits.maxDim else {
        throw ValidationError.dimensionTooLarge(name, value, Limits.maxDim)
    }
    return value
}
