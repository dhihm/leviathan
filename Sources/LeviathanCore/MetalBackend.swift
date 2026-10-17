import Foundation
import Metal

public enum MetalError: Error, Sendable {
    /// No Metal device is available on this machine.
    case metalUnavailable
    /// Runtime MSL shader compilation failed (source is bundled, so this is
    /// a toolchain/driver-level failure, reported verbatim).
    case shaderCompileFailed(String)
    /// A kernel entry point was missing from the compiled library.
    case functionMissing(String)
    /// The command buffer did not complete successfully.
    case commandBufferFailed(String)
}

/// Evidence from an actual GPU run, captured from the very device, library,
/// and command buffer used — not from a separate probe.
public struct MetalRunEvidence: Sendable {
    public let deviceName: String
    public let shaderLibraryCompiled: Bool
    public let dispatchedSteps: Int
    public let commandStatus: String
    public let readbackElements: Int
    /// Command-buffer GPU execution time in microseconds; 0 when the run
    /// did not come from the prepared timed path.
    public var gpuMicros: Double = 0
}

/// GPU backend. Fully validates the raw `LoweredProgram` (via the same
/// semantic plan validator artifacts use) BEFORE any GPU work, compiles the
/// MSL kernels from source at run time with `device.makeLibrary(source:)`,
/// uploads the weights blob and input to shared-storage MTLBuffers, encodes
/// one compute encoder per step (encoders within a command buffer are
/// serialized, giving a dependency barrier between steps), commits and waits
/// for completion, checks `commandStatus == .completed` and `cmd.error`,
/// then reads the output buffer back. Any failure (no device, shader
/// compile, dispatch, command-buffer error) throws; there is no silent CPU
/// fallback.
public enum MetalBackend {

    static let denseSource = """
    #include <metal_stdlib>
    using namespace metal;

    // y[row, col] = b[col] + sum_k x[row, k] * w[k, col]
    // p = (inDim, outDim, batch); gid.x = col, gid.y = row.
    kernel void leviathan_dense(constant const uint3& p [[buffer(0)]],
                                device const float* x [[buffer(1)]],
                                device const float* w [[buffer(2)]],
                                device const float* b [[buffer(3)]],
                                device float* y [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
        uint col = gid.x;
        uint row = gid.y;
        if (col >= p.y || row >= p.z) { return; }
        float sum = 0.0;
        for (uint k = 0; k < p.x; k++) {
            sum += x[row * p.x + k] * w[k * p.y + col];
        }
        y[row * p.y + col] = sum + b[col];
    }

    // Fused pass: identical math followed by ReLU in the same kernel, so a
    // dense step + relu step becomes one dispatch and one buffer write.
    kernel void leviathan_dense_relu(constant const uint3& p [[buffer(0)]],
                                     device const float* x [[buffer(1)]],
                                     device const float* w [[buffer(2)]],
                                     device const float* b [[buffer(3)]],
                                     device float* y [[buffer(4)]],
                                     uint2 gid [[thread_position_in_grid]]) {
        uint col = gid.x;
        uint row = gid.y;
        if (col >= p.y || row >= p.z) { return; }
        float sum = 0.0;
        for (uint k = 0; k < p.x; k++) {
            sum += x[row * p.x + k] * w[k * p.y + col];
        }
        y[row * p.y + col] = max(sum + b[col], 0.0f);
    }
    """

    static let reluSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void leviathan_relu(constant const uint& count [[buffer(0)]],
                               device float* x [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        x[gid] = max(x[gid], 0.0f);
    }
    """

    /// Full validation of a raw, untrusted `LoweredProgram` before any GPU
    /// resource is touched. Reuses the artifact plan validator via a
    /// temporary header, so buffer indices, the buffer chain, dims, step
    /// count, weight offsets, the weight byte cap, and weight finiteness are
    /// all enforced — an unvalidated plan can never become an out-of-bounds
    /// GPU memory access.
    static func validateProgram(_ program: LoweredProgram) throws {
        let header = ArtifactHeader(
            magic: CompiledArtifact.magic, version: CompiledArtifact.version,
            inputDim: program.inputDim, outputDim: program.outputDim,
            bufferCount: program.bufferCount, weightCount: program.weights.count,
            steps: program.steps, seed: 0)
        try CompiledArtifact.validatePlan(header: header, weights: program.weights)
        guard program.weights.count * 4 <= Limits.maxWeightBytes else {
            throw ArtifactError.exceedsWeightByteLimit(program.weights.count * 4)
        }
    }

    /// Per-buffer feature widths implied by the validated plan: buffer 0 is
    /// the graph input; each dense step's output buffer gets that step's
    /// outputDim. Only called after `validateProgram`.
    static func bufferDims(_ program: LoweredProgram) -> [Int] {
        var dims = [Int](repeating: 0, count: program.bufferCount)
        dims[0] = program.inputDim
        for step in program.steps {
            switch step {
            case let .dense(_, _, outDim, _, outBuf, _, _, _):
                dims[outBuf] = outDim
            case .relu:
                break
            }
        }
        return dims
    }

    /// True if the plan contains a dense step immediately followed by a
    /// relu on that step's output buffer — the pattern the fused pass
    /// optimizes. Only called after `validateProgram`.
    public static func isDenseReluFusable(_ program: LoweredProgram) -> Bool {
        for (i, step) in program.steps.enumerated() {
            if case let .dense(_, _, _, _, outBuf, _, _, _) = step,
               i + 1 < program.steps.count,
               case let .relu(_, inBuf) = program.steps[i + 1],
               inBuf == outBuf {
                return true
            }
        }
        return false
    }

    public static func run(_ program: LoweredProgram, batch: Int, input: [Float]) throws -> [Float] {
        try runReporting(program, batch: batch, input: input).output
    }

    /// `threadgroupWidth` optionally pins the dense kernel's threadgroup
    /// width; `fuseDenseRelu` runs the fused dense+ReLU pass where the
    /// plan allows it. Both are legal for any validated plan: the kernel's
    /// bounds guard makes any positive width legal, and the fused kernel
    /// is numerically dense-then-relu by definition. Widths are clamped to
    /// the pipeline's `maxTotalThreadsPerThreadgroup`.
    public static func runReporting(_ program: LoweredProgram, batch: Int, input: [Float],
                                    threadgroupWidth: Int? = nil,
                                    fuseDenseRelu: Bool = false)
            throws -> (output: [Float], evidence: MetalRunEvidence) {
        let prepared = try Prepared(program: program, batch: batch, input: input)
        let (output, micros) = try prepared.run(threadgroupWidth: threadgroupWidth,
                                                fuseDenseRelu: fuseDenseRelu)
        return (output, MetalRunEvidence(
            deviceName: prepared.deviceName,
            shaderLibraryCompiled: true,
            dispatchedSteps: program.steps.count,
            commandStatus: "completed",
            readbackElements: output.count,
            gpuMicros: micros))
    }

    /// Fingerprint of the MSL backend kernels; part of plan cache
    /// identity so a plan is invalidated when backend shaders change even
    /// if the Swift toolchain did not.
    public static var shaderFingerprint: String {
        SHA256.hex(SHA256.hash(Array((denseSource + reluSource).utf8)))
    }

    /// A program with its Metal resources prepared once — device, command
    /// queue, compiled library, pipelines, and buffers — so repeated runs
    /// (correctness gates, measured selection) dispatch kernels only and
    /// never re-time shader compilation or uploads.
    public struct Prepared {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let densePipeline: MTLComputePipelineState
        let denseReluPipeline: MTLComputePipelineState
        let reluPipeline: MTLComputePipelineState
        let program: LoweredProgram
        let batch: Int
        let dims: [Int]
        let weightBuffer: MTLBuffer?
        let gpuBuffers: [MTLBuffer?]
        let outCount: Int

        public var deviceName: String { device.name }
        /// The legal width cap is the strictest of the pipelines that any
        /// candidate may dispatch (dense and the fused dense+ReLU pass).
        public var maxTotalThreadsPerThreadgroup: Int {
            min(densePipeline.maxTotalThreadsPerThreadgroup,
                denseReluPipeline.maxTotalThreadsPerThreadgroup)
        }
        public var threadExecutionWidth: Int { densePipeline.threadExecutionWidth }
        public var recommendedMaxWorkingSetSize: Int {
            Int(clamping: device.recommendedMaxWorkingSetSize)
        }

        public init(program: LoweredProgram, batch: Int, input: [Float]) throws {
            try validateProgram(program)
            try preflight(program, batch: batch)
            guard input.count == batch * program.inputDim else {
                throw ExecutionError.wrongInputCount(input.count, batch * program.inputDim)
            }
            let dims = bufferDims(program)

            guard let device = MTLCreateSystemDefaultDevice(),
                  let queue = device.makeCommandQueue() else {
                throw MetalError.metalUnavailable
            }
            let library: MTLLibrary
            do {
                library = try device.makeLibrary(source: denseSource + "\n" + reluSource,
                                                 options: nil)
            } catch {
                throw MetalError.shaderCompileFailed("\(error)")
            }
            guard let denseFn = library.makeFunction(name: "leviathan_dense"),
                  let denseReluFn = library.makeFunction(name: "leviathan_dense_relu"),
                  let reluFn = library.makeFunction(name: "leviathan_relu"),
                  let densePipeline = try? device.makeComputePipelineState(function: denseFn),
                  let denseReluPipeline = try? device.makeComputePipelineState(function: denseReluFn),
                  let reluPipeline = try? device.makeComputePipelineState(function: reluFn) else {
                throw MetalError.functionMissing(
                    "leviathan_dense/leviathan_dense_relu/leviathan_relu")
            }

            // Explicit shared storage: CPU-writable upload and CPU-readable
            // readback without blits, and no managed-buffer sync hazards.
            let stride = MemoryLayout<Float>.stride
            // Metal returns nil for zero-length buffers; weights-empty plans
            // (ReLU-only graphs) simply have no dense kernel to bind them to.
            let weightBuffer: MTLBuffer?
            if program.weights.isEmpty {
                weightBuffer = nil
            } else {
                guard let buffer = device.makeBuffer(
                          bytes: program.weights, length: program.weights.count * stride,
                          options: .storageModeShared) else {
                    throw MetalError.metalUnavailable
                }
                weightBuffer = buffer
            }
            var gpuBuffers = [MTLBuffer?](repeating: nil, count: program.bufferCount)
            guard let inputBuffer = device.makeBuffer(
                      bytes: input, length: input.count * stride, options: .storageModeShared) else {
                throw MetalError.metalUnavailable
            }
            gpuBuffers[0] = inputBuffer
            for i in 1..<program.bufferCount {
                guard let buffer = device.makeBuffer(length: batch * dims[i] * stride,
                                                     options: .storageModeShared) else {
                    throw MetalError.metalUnavailable
                }
                gpuBuffers[i] = buffer
            }
            self.device = device
            self.queue = queue
            self.densePipeline = densePipeline
            self.denseReluPipeline = denseReluPipeline
            self.reluPipeline = reluPipeline
            self.program = program
            self.batch = batch
            self.dims = dims
            self.weightBuffer = weightBuffer
            self.gpuBuffers = gpuBuffers
            self.outCount = batch * dims[program.outputBufferIndex]
        }

        /// One dispatch of the prepared program. Returns the output and the
        /// GPU execution time in microseconds measured from the command
        /// buffer's own `GPUStartTime`/`GPUEndTime` — device time only, no
        /// shader compilation or upload cost.
        public func run(threadgroupWidth: Int? = nil,
                        fuseDenseRelu: Bool = false) throws -> (output: [Float], gpuMicros: Double) {
            guard let cmd = queue.makeCommandBuffer() else {
                throw MetalError.metalUnavailable
            }
            var dispatched = 0
            var skipNext = false
            for step in program.steps {
                if skipNext { skipNext = false; dispatched += 1; continue }
                guard let encoder = cmd.makeComputeCommandEncoder() else {
                    throw MetalError.metalUnavailable
                }
                switch step {
                case let .dense(_, inDim, outDim, inBuf, outBuf, wOff, _, bOff):
                    var p = SIMD3<UInt32>(UInt32(inDim), UInt32(outDim), UInt32(batch))
                    let fuseNow = fuseDenseRelu
                        && program.steps.indices.contains(dispatched + 1)
                        && isRelu(on: outBuf, at: dispatched + 1)
                    encoder.setComputePipelineState(
                        fuseNow ? denseReluPipeline : densePipeline)
                    encoder.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)
                    encoder.setBuffer(gpuBuffers[inBuf], offset: 0, index: 1)
                    // Weight and bias slices into the single shared weights blob.
                    encoder.setBuffer(weightBuffer, offset: wOff * 4, index: 2)
                    encoder.setBuffer(weightBuffer, offset: bOff * 4, index: 3)
                    precondition(weightBuffer != nil, "validated dense step requires weights")
                    encoder.setBuffer(gpuBuffers[outBuf], offset: 0, index: 4)
                    // Legal width: clamped to the pipeline's true maximum,
                    // never to threadExecutionWidth (which would collapse
                    // distinct candidates into identical dispatches).
                    let requested = threadgroupWidth ?? densePipeline.threadExecutionWidth
                    let width = max(1, min(outDim,
                                           min(requested, maxTotalThreadsPerThreadgroup)))
                    let groups = (outDim + width - 1) / width
                    encoder.dispatchThreadgroups(
                        MTLSize(width: groups, height: batch, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                    if fuseNow { skipNext = true }
                case let .relu(_, buf):
                    var count = UInt32(batch * dims[buf])
                    encoder.setComputePipelineState(reluPipeline)
                    encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
                    encoder.setBuffer(gpuBuffers[buf], offset: 0, index: 1)
                    let total = batch * dims[buf]
                    let width = min(64, reluPipeline.threadExecutionWidth)
                    let groups = (total + width - 1) / width
                    encoder.dispatchThreadgroups(
                        MTLSize(width: groups, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                }
                encoder.endEncoding()
                dispatched += 1
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            guard cmd.status == .completed, cmd.error == nil else {
                throw MetalError.commandBufferFailed(
                    "status \(cmd.status.rawValue), error \(String(describing: cmd.error))")
            }
            let gpuMicros = (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6
            let outputBuffer = gpuBuffers[program.outputBufferIndex]!
            let out = [Float](UnsafeBufferPointer(
                start: outputBuffer.contents().assumingMemoryBound(to: Float.self),
                count: outCount))
            return (out, gpuMicros)
        }

        func isRelu(on buf: Int, at index: Int) -> Bool {
            guard program.steps.indices.contains(index) else { return false }
            if case let .relu(_, inBuf) = program.steps[index] { return inBuf == buf }
            return false
        }
    }
}
