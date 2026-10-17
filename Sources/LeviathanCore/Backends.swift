import Accelerate
import Foundation

public enum ExecutionError: Error, Equatable, Sendable {
    case batchTooLarge(Int, Int)
    case tensorTooLarge(Int, Int)
    case wrongInputCount(Int, Int)
}

/// Shared run-time checks: bound the batch and every allocated buffer.
/// Returns total elements of the largest buffer plan.
func preflight(_ program: LoweredProgram, batch: Int) throws {
    guard batch > 0, batch <= Limits.maxBatch else {
        throw ExecutionError.batchTooLarge(batch, Limits.maxBatch)
    }
    let maxFeatures = program.steps.reduce(program.inputDim) { acc, step in
        switch step {
        case let .dense(_, _, outputDim, _, _, _, _, _): return max(acc, outputDim)
        default: return acc
        }
    }
    let elements = batch * maxFeatures
    guard elements <= Limits.maxTensorElements else {
        throw ExecutionError.tensorTooLarge(elements, Limits.maxTensorElements)
    }
}

/// CPU backend using Apple Accelerate (vDSP). Dense runs as a row-major
/// matrix product via `vDSP_mmul` followed by a bias add; ReLU is a clip
/// against a zero lower bound.
public enum AccelerateBackend {
    public static func run(_ program: LoweredProgram, batch: Int, input: [Float]) throws -> [Float] {
        try preflight(program, batch: batch)
        guard input.count == batch * program.inputDim else {
            throw ExecutionError.wrongInputCount(input.count, batch * program.inputDim)
        }
        var buffers = [[Float]](repeating: [], count: program.bufferCount)
        buffers[0] = input

        for step in program.steps {
            switch step {
            case let .dense(_, inDim, outDim, inBuf, outBuf, wOff, wCount, bOff):
                var out = [Float](repeating: 0, count: batch * outDim)
                let w = Array(program.weights[wOff..<(wOff + wCount)])
                let b = Array(program.weights[bOff..<(bOff + outDim)])
                // y[batch,out] = x[batch,in] * w[in,out]
                vDSP_mmul(buffers[inBuf], 1, w, 1,
                          &out, 1,
                          vDSP_Length(batch), vDSP_Length(outDim), vDSP_Length(inDim))
                // out += tile(b)
                if outDim > 0 {
                    var row = b
                    for _ in 1..<batch { row.append(contentsOf: b) }
                    var added = [Float](repeating: 0, count: batch * outDim)
                    vDSP_vadd(out, 1, row, 1, &added, 1, vDSP_Length(batch * outDim))
                    out = added
                }
                buffers[outBuf] = out
            case let .relu(_, buf):
                let zeros = [Float](repeating: 0, count: buffers[buf].count)
                var out = [Float](repeating: 0, count: buffers[buf].count)
                vDSP_vmax(buffers[buf], 1, zeros, 1,
                          &out, 1, vDSP_Length(buffers[buf].count))
                buffers[buf] = out
            }
        }
        return buffers[program.outputBufferIndex]
    }
}

/// Independent scalar reference backend: plain Swift loops, no Accelerate.
/// Used as the ground truth the Accelerate path is checked against.
public enum ScalarBackend {
    public static func run(_ program: LoweredProgram, batch: Int, input: [Float]) throws -> [Float] {
        try preflight(program, batch: batch)
        guard input.count == batch * program.inputDim else {
            throw ExecutionError.wrongInputCount(input.count, batch * program.inputDim)
        }
        var buffers = [[Float]](repeating: [], count: program.bufferCount)
        buffers[0] = input

        for step in program.steps {
            switch step {
            case let .dense(_, inDim, outDim, inBuf, outBuf, wOff, _, bOff):
                var out = [Float](repeating: 0, count: batch * outDim)
                for r in 0..<batch {
                    for c in 0..<outDim {
                        var sum: Float = 0
                        for k in 0..<inDim {
                            sum += buffers[inBuf][r * inDim + k] * program.weights[wOff + k * outDim + c]
                        }
                        out[r * outDim + c] = sum + program.weights[bOff + c]
                    }
                }
                buffers[outBuf] = out
            case let .relu(_, buf):
                buffers[buf] = buffers[buf].map { Swift.max($0, 0) }
            }
        }
        return buffers[program.outputBufferIndex]
    }
}

extension LoweredProgram {
    /// Index of the buffer holding the final output: the output buffer of
    /// the last dense op, or 0 if the graph ends without one.
    var outputBufferIndex: Int {
        var index = 0
        for step in steps {
            if case let .dense(_, _, _, _, outBuf, _, _, _) = step { index = outBuf }
        }
        return index
    }
}

/// Maximum absolute elementwise difference between two equal-length vectors.
/// Returns nil (rejects) for mismatched lengths, empty input, or any
/// non-finite element, so NaNs cannot slip through a numeric comparison.
public func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float? {
    guard a.count == b.count, !a.isEmpty else { return nil }
    var worst: Float = 0
    for i in a.indices {
        guard a[i].isFinite, b[i].isFinite else { return nil }
        worst = max(worst, abs(a[i] - b[i]))
    }
    return worst
}
