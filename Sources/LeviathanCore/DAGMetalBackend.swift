import Foundation
import Metal

/// Errors that belong only to the value-DAG Metal path. Device, compile, pipeline and command-buffer
/// failures use the existing `MetalError` cases and carry the real Metal error text.
public enum DAGMetalError: Error, Equatable, Sendable {
    case invalidThreadgroupWidth(Int, cap: Int)
    case inconsistentProgram(String)
    case invalidGPUTime(Double)
    case nonFiniteOutput(Int)
}

/// The result of one GPU run. The host builds `dispatchedSteps` and `kernels` from what it actually
/// encoded, with fusions applied. They are not a profiler trace.
public struct DAGMetalRunResult: Sendable {
    public let output: [Float]
    public let gpuMicros: Double
    public let dispatchedSteps: Int
    public let kernels: [String]
    public let deviceName: String
}

/// Metal backend for validated value DAGs. Each immutable value gets its own shared buffer. Each node
/// writes a new buffer, so relu and add never run in place. Nodes are encoded in topological (id)
/// order, one compute encoder per dispatch, so every producer finishes before its consumers run.
/// There is no CPU fallback.
public enum DAGMetalBackend {
    static let denseKernel = "leviathan_dense"
    static let denseReluKernel = "leviathan_dense_relu"
    static let reluKernel = "leviathan_dag_relu"
    static let addKernel = "leviathan_dag_add"

    static let elementwiseSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Out-of-place: y[i] = max(x[i], 0). x and y are distinct value buffers.
    kernel void leviathan_dag_relu(constant const uint& n [[buffer(0)]],
                                   device const float* x [[buffer(1)]],
                                   device float* y [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        y[gid] = max(x[gid], 0.0f);
    }

    // Out-of-place: y[i] = a[i] + b[i]. a may alias b; y is always a distinct buffer.
    kernel void leviathan_dag_add(constant const uint& n [[buffer(0)]],
                                  device const float* a [[buffer(1)]],
                                  device const float* b [[buffer(2)]],
                                  device float* y [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
        if (gid >= n) { return; }
        y[gid] = a[gid] + b[gid];
    }
    """

    /// The full compiled source: the shared personal dense kernels plus the DAG elementwise kernels.
    static var fullSource: String { MetalBackend.denseSource + "\n" + elementwiseSource }

    /// A fingerprint of the full compiled source, including `MetalBackend.denseSource`.
    public static var shaderFingerprint: String {
        SHA256.hex(SHA256.hash(Array(("leviathan-dag-metal-1\n" + fullSource).utf8)))
    }

    struct Step {
        let node: DAGNode
        let layout: DAGDenseLayout?
        let fusableRelu: Int?
    }

    /// Re-checks the program's graph, dims, weights, packed layouts and fusion pairs, so the run never
    /// trusts derived data it has not checked. Throws before any Metal work.
    static func revalidate(_ p: DAGProgram) throws -> [Step] {
        let (dims, count) = try p.graph.validate()
        guard dims == p.bufferDims else { throw DAGMetalError.inconsistentProgram("bufferDims") }
        guard p.weights.count == count else {
            throw DAGValidationError.weightCountMismatch(expected: count, got: p.weights.count)
        }
        if let i = p.weights.firstIndex(where: { !$0.isFinite }) { throw DAGValidationError.nonFiniteWeight(i) }
        let nodes = p.graph.nodes
        var consumers = [Int](repeating: 0, count: dims.count)
        for n in nodes { for v in n.inputs { consumers[v] += 1 } }
        consumers[p.graph.outputValue] += 1
        var fusable: [Int: Int] = [:]
        for pair in p.safeFusionPairs {
            guard pair.dense >= 1, pair.relu == pair.dense + 1, pair.relu <= nodes.count,
                  nodes[pair.dense - 1].op == .dense, nodes[pair.relu - 1].op == .relu,
                  nodes[pair.relu - 1].inputs == [pair.dense], consumers[pair.dense] == 1,
                  pair.dense != p.graph.outputValue, fusable[pair.dense] == nil
            else { throw DAGMetalError.inconsistentProgram("fusion pair \(pair.dense)->\(pair.relu)") }
            fusable[pair.dense] = pair.relu
        }
        var offset = 0
        var steps: [Step] = []
        for n in nodes {
            var layout: DAGDenseLayout? = nil
            if n.op == .dense {
                guard let l = p.layouts[n.id], l.node == n.id, l.inDim == dims[n.inputs[0]],
                      l.outDim == n.outputDim, l.weightOffset == offset,
                      l.weightCount == l.inDim * l.outDim, l.biasOffset == offset + l.weightCount
                else { throw DAGMetalError.inconsistentProgram("layout \(n.id)") }
                offset = l.biasOffset + l.outDim
                layout = l
            }
            steps.append(Step(node: n, layout: layout, fusableRelu: fusable[n.id]))
        }
        guard offset == p.weights.count, p.layouts.count == steps.filter({ $0.layout != nil }).count else {
            throw DAGMetalError.inconsistentProgram("weight layout")
        }
        return steps
    }

    /// A validated, immutable program with its device, queue, compiled library, four pipelines and
    /// buffers prepared once. Repeated runs only encode and dispatch. They never upload or compile again.
    public struct Prepared {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let densePipeline: MTLComputePipelineState
        let denseReluPipeline: MTLComputePipelineState
        let reluPipeline: MTLComputePipelineState
        let addPipeline: MTLComputePipelineState
        public let program: DAGProgram
        let steps: [Step]
        let weightBuffer: MTLBuffer
        let values: [MTLBuffer]
        let outCount: Int
        // Shared lock for ALL runs on these buffers. NSLock is a class, so struct copies of a
        // Prepared value share the same lock object and serialize against each other. This type
        // is still not Sendable and parallel requests are not supported; the lock only prevents
        // races between the old and new run entry points sharing the same MTLBuffers.
        let lock: NSLock

        public var deviceName: String { device.name }
        /// The true legal cap: the strictest limit across all four pipelines a run may dispatch.
        public var maxTotalThreadsPerThreadgroup: Int {
            min(min(densePipeline.maxTotalThreadsPerThreadgroup, denseReluPipeline.maxTotalThreadsPerThreadgroup),
                min(reluPipeline.maxTotalThreadsPerThreadgroup, addPipeline.maxTotalThreadsPerThreadgroup))
        }
        public var threadExecutionWidth: Int { densePipeline.threadExecutionWidth }
        public var recommendedMaxWorkingSetSize: Int { Int(clamping: device.recommendedMaxWorkingSetSize) }
        public var minDenseOutputDim: Int { program.layouts.values.map { $0.outDim }.min() ?? 1 }
        /// Baseline: min(32, cap, smallest dense outDim), run without fusion.
        public var baselineThreadgroupWidth: Int {
            max(1, min(32, maxTotalThreadsPerThreadgroup, minDenseOutputDim))
        }
        /// Candidates 32/64/128/256, clamped to the cap and the smallest dense outDim, then deduplicated.
        public var candidateThreadgroupWidths: [Int] {
            var out: [Int] = []
            for w in [32, 64, 128, 256] {
                let c = max(1, min(w, maxTotalThreadsPerThreadgroup, minDenseOutputDim))
                if !out.contains(c) { out.append(c) }
            }
            return out
        }

        public init(program: DAGProgram, input: [Float]) throws {
            guard input.count == program.graph.inputDim else {
                throw DAGExecutionError.inputCountMismatch(expected: program.graph.inputDim, got: input.count)
            }
            if let i = input.firstIndex(where: { !$0.isFinite }) { throw DAGExecutionError.nonFiniteInput(i) }
            let steps = try DAGMetalBackend.revalidate(program)

            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw MetalError.metalUnavailable
            }
            let library: MTLLibrary
            do { library = try device.makeLibrary(source: DAGMetalBackend.fullSource, options: nil) }
            catch { throw MetalError.shaderCompileFailed("\(error)") }
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let fn = library.makeFunction(name: name) else { throw MetalError.functionMissing(name) }
                do { return try device.makeComputePipelineState(function: fn) }
                catch { throw MetalError.shaderCompileFailed("pipeline \(name): \(error)") }
            }
            let dense = try pipeline(DAGMetalBackend.denseKernel)
            let denseRelu = try pipeline(DAGMetalBackend.denseReluKernel)
            let relu = try pipeline(DAGMetalBackend.reluKernel)
            let add = try pipeline(DAGMetalBackend.addKernel)

            // Shared storage, with one distinct buffer per immutable value. Weights are the real blob.
            let stride = MemoryLayout<Float>.stride
            guard let wb = device.makeBuffer(bytes: program.weights, length: program.weights.count * stride,
                                             options: .storageModeShared),
                  let ib = device.makeBuffer(bytes: input, length: input.count * stride,
                                             options: .storageModeShared) else {
                throw MetalError.metalUnavailable
            }
            var values: [MTLBuffer] = [ib]
            for v in 1..<program.bufferDims.count {
                let length = program.bufferDims[v] * stride
                guard let b = device.makeBuffer(length: length, options: .storageModeShared) else {
                    throw MetalError.metalUnavailable
                }
                memset(b.contents(), 0, length)
                values.append(b)
            }
            self.device = device
            self.queue = queue
            self.densePipeline = dense
            self.denseReluPipeline = denseRelu
            self.reluPipeline = relu
            self.addPipeline = add
            self.program = program
            self.steps = steps
            self.weightBuffer = wb
            self.values = values
            self.outCount = program.bufferDims[program.graph.outputValue]
            self.lock = NSLock()
        }

        /// One GPU run on the last uploaded input (the constructor's initial input if `run(input:)` was
        /// never called; otherwise the most recent `run(input:)` upload). Throws if `threadgroupWidth`
        /// is <= 0 or > the cap; the width is never silently changed. For each dispatch it is clamped
        /// only to that node's output width. With `fuseDenseRelu`, only `program.safeFusionPairs` are
        /// fused. The fused kernel writes the relu value buffer and the relu node is skipped.
        public func run(threadgroupWidth: Int, fuseDenseRelu: Bool) throws -> DAGMetalRunResult {
            let cap = maxTotalThreadsPerThreadgroup
            guard threadgroupWidth >= 1, threadgroupWidth <= cap else {
                throw DAGMetalError.invalidThreadgroupWidth(threadgroupWidth, cap: cap)
            }
            // The lock spans encoding, commit, GPU wait and readback so concurrent callers sharing
            // these MTLBuffers cannot interleave uploads or reads. unlockedRun never re-locks.
            lock.lock()
            defer { lock.unlock() }
            return try unlockedRun(threadgroupWidth: threadgroupWidth, fuseDenseRelu: fuseDenseRelu)
        }

        /// One GPU run with a distinct real input. The input is validated (exact count and finiteness)
        /// and the threadgroup width checked BEFORE any buffer upload, so an invalid input never
        /// dispatches GPU work. Only value buffer 0 (the input) is copied into; the weights, pipelines
        /// and all other value buffers are reused as-is. Every written node value is recomputed each
        /// run; when fusion skips a relu node its value buffer is dead by the consumer-safe proof in
        /// `revalidate` (the fused dense already wrote that buffer and no other consumer reads it).
        public func run(input: [Float], threadgroupWidth: Int, fuseDenseRelu: Bool) throws -> DAGMetalRunResult {
            guard input.count == program.graph.inputDim else {
                throw DAGExecutionError.inputCountMismatch(expected: program.graph.inputDim, got: input.count)
            }
            if let i = input.firstIndex(where: { !$0.isFinite }) { throw DAGExecutionError.nonFiniteInput(i) }
            let cap = maxTotalThreadsPerThreadgroup
            guard threadgroupWidth >= 1, threadgroupWidth <= cap else {
                throw DAGMetalError.invalidThreadgroupWidth(threadgroupWidth, cap: cap)
            }
            lock.lock()
            defer { lock.unlock() }
            // Bounded copy of exactly input.count * Float.stride bytes into the input value buffer.
            let byteCount = input.count * MemoryLayout<Float>.stride
            input.withUnsafeBytes { raw in
                memcpy(values[0].contents(), raw.baseAddress!, byteCount)
            }
            return try unlockedRun(threadgroupWidth: threadgroupWidth, fuseDenseRelu: fuseDenseRelu)
        }

        /// Encode, commit, wait and read back. Callers must hold `lock`; this function must not lock
        /// again (a recursive lock here would deadlock). All writes, encodings, commit, wait and the
        /// output readback happen under the caller's lock.
        private func unlockedRun(threadgroupWidth: Int, fuseDenseRelu: Bool) throws -> DAGMetalRunResult {
            let skipped: Set<Int> = fuseDenseRelu ? Set(steps.compactMap { $0.fusableRelu }) : []
            guard let cmd = queue.makeCommandBuffer() else { throw MetalError.metalUnavailable }
            var kernels: [String] = []
            for step in steps where !skipped.contains(step.node.id) {
                guard let enc = cmd.makeComputeCommandEncoder() else { throw MetalError.metalUnavailable }
                let n = step.node
                let count: Int
                if let l = step.layout {
                    let fused = fuseDenseRelu ? step.fusableRelu : nil
                    var p = SIMD3<UInt32>(UInt32(l.inDim), UInt32(l.outDim), 1)
                    enc.setComputePipelineState(fused == nil ? densePipeline : denseReluPipeline)
                    enc.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
                    enc.setBuffer(values[n.inputs[0]], offset: 0, index: 1)
                    enc.setBuffer(weightBuffer, offset: l.weightOffset * 4, index: 2)
                    enc.setBuffer(weightBuffer, offset: l.biasOffset * 4, index: 3)
                    enc.setBuffer(values[fused ?? n.id], offset: 0, index: 4)
                    kernels.append(fused == nil ? DAGMetalBackend.denseKernel : DAGMetalBackend.denseReluKernel)
                    count = l.outDim
                } else {
                    var u = UInt32(n.outputDim)
                    let isAdd = n.op == .add
                    enc.setComputePipelineState(isAdd ? addPipeline : reluPipeline)
                    enc.setBytes(&u, length: MemoryLayout<UInt32>.stride, index: 0)
                    for (i, v) in n.inputs.enumerated() { enc.setBuffer(values[v], offset: 0, index: 1 + i) }
                    enc.setBuffer(values[n.id], offset: 0, index: 1 + n.inputs.count)
                    kernels.append(isAdd ? DAGMetalBackend.addKernel : DAGMetalBackend.reluKernel)
                    count = n.outputDim
                }
                let width = max(1, min(count, threadgroupWidth))
                enc.dispatchThreadgroups(MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw MetalError.commandBufferFailed(
                    "status \(cmd.status.rawValue), error \(String(describing: cmd.error))")
            }
            let micros = (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
            guard micros.isFinite, micros > 0 else { throw DAGMetalError.invalidGPUTime(micros) }
            let ptr = values[program.graph.outputValue].contents().bindMemory(to: Float.self, capacity: outCount)
            let out = Array(UnsafeBufferPointer(start: ptr, count: outCount))
            if let i = out.firstIndex(where: { !$0.isFinite }) { throw DAGMetalError.nonFiniteOutput(i) }
            return DAGMetalRunResult(output: out, gpuMicros: micros, dispatchedSteps: kernels.count,
                                     kernels: kernels, deviceName: device.name)
        }
    }
}
