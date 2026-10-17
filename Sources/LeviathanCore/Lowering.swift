import Foundation

/// One executable step in the lowered program. Dense steps read from
/// `inputBuffer` and write to `outputBuffer`; ReLU steps rewrite their buffer
/// in place. Buffer assignment happens at lower time, not run time, so the
/// execution plan is fully determined before any tensor is allocated.
public enum LoweredOp: Codable, Equatable, Sendable {
    case dense(index: Int, inputDim: Int, outputDim: Int,
               inputBuffer: Int, outputBuffer: Int,
               weightOffset: Int, weightCount: Int, biasOffset: Int)
    case relu(index: Int, buffer: Int)
}

/// The deterministic result of lowering a validated graph: an ordered step
/// list plus the synthesized weight blob (row-major float32, little-endian
/// when serialized). Weights are generated from a seeded SplitMix64 stream
/// so the same source graph and seed always produce identical bytes.
public struct LoweredProgram: Sendable {
    public let inputDim: Int
    public let outputDim: Int
    public let bufferCount: Int
    public let steps: [LoweredOp]
    public let weights: [Float]

    public init(inputDim: Int, outputDim: Int, bufferCount: Int,
                steps: [LoweredOp], weights: [Float]) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.bufferCount = bufferCount
        self.steps = steps
        self.weights = weights
    }

    public init(graph: GraphIR, seed: UInt64) {
        precondition(!graph.ops.isEmpty)
        var rng = SplitMix64(seed: seed)
        var steps: [LoweredOp] = []
        var weights: [Float] = []

        // Buffer 0 holds the graph input; each dense op allocates a new buffer.
        var currentBuffer = 0
        var currentDim = graph.inputDim
        var bufferCount = 1

        for (index, op) in graph.ops.enumerated() {
            switch op {
            case let .dense(outDim):
                let weightCount = currentDim * outDim
                let weightOffset = weights.count
                for _ in 0..<weightCount { weights.append(rng.nextUniform()) }
                let biasOffset = weights.count
                for _ in 0..<outDim { weights.append(rng.nextUniform() * 0.1) }
                let outBuffer = bufferCount
                bufferCount += 1
                steps.append(.dense(
                    index: index, inputDim: currentDim, outputDim: outDim,
                    inputBuffer: currentBuffer, outputBuffer: outBuffer,
                    weightOffset: weightOffset, weightCount: weightCount,
                    biasOffset: biasOffset))
                currentBuffer = outBuffer
                currentDim = outDim
            case .relu:
                steps.append(.relu(index: index, buffer: currentBuffer))
            }
        }

        self.inputDim = graph.inputDim
        self.outputDim = currentDim
        self.bufferCount = bufferCount
        self.steps = steps
        self.weights = weights
    }
}
