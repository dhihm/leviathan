import Foundation
import Metal

enum Invalid: Error { case arguments, tensor, metal }

func tensor(_ path: String, count: Int) throws -> Data {
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? handle.close() }
    let data = try handle.read(upToCount: count * 4 + 1) ?? Data()
    guard data.count == count * 4 else { throw Invalid.tensor }
    return data
}

let convolutionSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void conv2d_fp32(device const float *x [[buffer(0)]],
                                device const float *weights [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant uint *p [[buffer(3)]],
                                device const float *bias [[buffer(4)]],
                                uint index [[thread_position_in_grid]]) {
            uint ci=p[0], h=p[1], w=p[2], kh=p[4], kw=p[5];
            uint stride=p[6], pad=p[7], oh=p[8], ow=p[9];
            uint oc=index/(oh*ow), oy=(index/ow)%oh, ox=index%ow;
            float sum=0.0f;
            for (uint ic=0; ic<ci; ++ic) {
                for (uint ky=0; ky<kh; ++ky) {
                    int iy=int(oy*stride+ky)-int(pad);
                    if (iy<0 || iy>=int(h)) continue;
                    for (uint kx=0; kx<kw; ++kx) {
                        int ix=int(ox*stride+kx)-int(pad);
                        if (ix<0 || ix>=int(w)) continue;
                        sum=fma(x[(ic*h+uint(iy))*w+uint(ix)],
                                weights[((oc*ci+ic)*kh+ky)*kw+kx], sum);
                    }
                }
            }
            float value = p[10] ? sum + bias[oc] : sum;
            output[index] = p[11] ? max(value, 0.0f) : value;
        }
        kernel void residual_add_relu(device const float *main [[buffer(0)]],
                                      device const float *skip [[buffer(1)]],
                                      device float *output [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
            output[i] = max(main[i] + skip[i], 0.0f);
        }
        kernel void maxpool_3x3_s2_p1(device const float *x [[buffer(0)]],
                                       device float *y [[buffer(1)]],
                                       constant uint *p [[buffer(2)]],
                                       uint index [[thread_position_in_grid]]) {
            uint h = p[0], w = p[1], oh = p[2], ow = p[3];
            uint channel = index / (oh * ow);
            int iy = int((index / ow) % oh) * 2 - 1;
            int ix = int(index % ow) * 2 - 1;
            // Every window has a valid sample for this fixed geometry.
            float peak = x[(channel * h + uint(max(iy, 0))) * w + uint(max(ix, 0))];
            for (int y = max(iy, 0); y < min(iy + 3, int(h)); ++y)
                for (int x0 = max(ix, 0); x0 < min(ix + 3, int(w)); ++x0)
                    peak = max(peak, x[(channel * h + uint(y)) * w + uint(x0)]);
            y[index] = peak;
        }
        """

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 14 else { throw Invalid.arguments }
    let dims = args[4...11].compactMap { UInt32($0) }
    guard dims.count == 8 else { throw Invalid.arguments }
    let (ci, h, w, co, kh, kw, stride, padding) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7])
    guard (1...256).contains(ci), (1...256).contains(co),
          (1...512).contains(h), (1...512).contains(w),
          (1...11).contains(kh), (1...11).contains(kw),
          (1...4).contains(stride), padding <= 5,
          h + 2 * padding >= kh, w + 2 * padding >= kw else { throw Invalid.arguments }
    let oh = (h + 2 * padding - kh) / stride + 1
    let ow = (w + 2 * padding - kw) / stride + 1
    let count = Int(co) * Int(oh) * Int(ow)
    let weightCount = Int(co * ci * kh * kw)
    guard count <= 1 << 20, Int(ci * h * w) <= 1 << 20, weightCount <= 1 << 20,
          count * Int(ci * kh * kw) <= 100_000_000 else { throw Invalid.arguments }

    let x = try tensor(args[1], count: Int(ci * h * w))
    let weights = try tensor(args[2], count: weightCount)
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Invalid.metal
    }
    let library = try device.makeLibrary(source: convolutionSource, options: nil)
    guard let function = library.makeFunction(name: "conv2d_fp32") else { throw Invalid.metal }
    let pipeline = try device.makeComputePipelineState(function: function)
    let width = pipeline.threadExecutionWidth
    let deviceID = String(device.registryID)
    let cachedWidth = Int(args[13])
    let reused = args[12] == deviceID && cachedWidth == width
    let dispatchWidth = (reused ? cachedWidth : nil) ?? width
    func buffer(_ data: Data) -> MTLBuffer? {
        data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count) }
    }
    guard let input = buffer(x), let filters = buffer(weights),
          let output = device.makeBuffer(length: count * 4, options: .storageModeShared),
          let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
        throw Invalid.metal
    }
    let constants = [ci, h, w, co, kh, kw, stride, padding, oh, ow, 0, 0]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(filters, offset: 0, index: 1)
    constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBuffer(filters, offset: 0, index: 4)
    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: dispatchWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    let micros = (command.gpuEndTime - command.gpuStartTime) * 1e6
    guard command.status == .completed, micros.isFinite, micros > 0 else { throw Invalid.metal }
    try Data(bytes: output.contents(), count: count * 4).write(to: URL(fileURLWithPath: args[3]))
    let evidence: [String: Any] = ["kernel": "conv2d_fp32", "device": device.name,
                                  "registryID": deviceID, "width": width,
                                  "reused": reused, "gpuMicros": micros]
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: evidence))
}

do {
    let mode = CommandLine.arguments.dropFirst().first
    if mode == "--residual" || mode == "--stage1" {
        try runResidual(stage1: mode == "--stage1")
    } else {
        try run()
    }
} catch {
    FileHandle.standardError.write(Data("Metal Conv2D: \(error)\n".utf8))
    exit(2)
}

private struct BlockCommand: Decodable {
    let channels: Int
    let height: Int
    let width: Int
    let outputChannels: Int
    let stride: Int
    let projection: Bool
    let cachedDevice: String
    let cachedWidth: Int
}

func runResidual(stage1: Bool) throws {
    let args = CommandLine.arguments
    guard args.count == 6 else { throw Invalid.arguments }
    let configHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: args[2]))
    defer { try? configHandle.close() }
    let data = try configHandle.read(upToCount: 4097) ?? Data()
    guard data.count <= 4096 else { throw Invalid.arguments }
    let cfg = try JSONDecoder().decode(BlockCommand.self, from: data)
    let ci = cfg.channels, co = cfg.outputChannels, h = cfg.height, w = cfg.width
    guard (1...256).contains(ci), (1...256).contains(co),
          (1...512).contains(h), (1...512).contains(w),
          stage1 || (cfg.projection ? cfg.stride == 2 : (cfg.stride == 1 && ci == co)) else {
        throw Invalid.arguments
    }
    if stage1 {
        guard ci == 3, co == 64, h == 224, w == 224, cfg.stride == 2, !cfg.projection else {
            throw Invalid.arguments
        }
    }
    let oh = stage1 ? 56 : (h - 1) / cfg.stride + 1
    let ow = stage1 ? 56 : (w - 1) / cfg.stride + 1
    let inputCount = ci * h * w, outputCount = co * oh * ow
    let filterCount = stage1 ? (64 * 3 * 49 + 64) + 4 * (64 * 64 * 9 + 64)
        : co * ci * 9 + co + co * co * 9 + co + (cfg.projection ? co * ci + co : 0)
    guard inputCount <= 1 << 20, outputCount <= 1 << 20,
          stage1 || outputCount * (ci * 9 + co * 9 + (cfg.projection ? ci : 0)) <= 200_000_000 else {
        throw Invalid.arguments
    }
    let x = try tensor(args[3], count: inputCount)
    let packed = try tensor(args[4], count: filterCount)
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Invalid.metal
    }
    let library = try device.makeLibrary(source: convolutionSource, options: nil)
    guard let conv = library.makeFunction(name: "conv2d_fp32"),
          let add = library.makeFunction(name: "residual_add_relu") else { throw Invalid.metal }
    let convPipe = try device.makeComputePipelineState(function: conv)
    let addPipe = try device.makeComputePipelineState(function: add)
    let width = convPipe.threadExecutionWidth
    let deviceID = String(device.registryID)
    let reused = cfg.cachedDevice == deviceID && cfg.cachedWidth == width
    let dispatchWidth = reused ? cfg.cachedWidth : width
    func buffer(_ data: Data) -> MTLBuffer? {
        data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count) }
    }
    guard let input = buffer(x), let filters = buffer(packed),
          let first = device.makeBuffer(length: outputCount * 4),
          let second = device.makeBuffer(length: outputCount * 4),
          let output = device.makeBuffer(length: outputCount * 4, options: .storageModeShared),
          let command = queue.makeCommandBuffer() else { throw Invalid.metal }
    var offset = 0
    func dispatch(_ source: MTLBuffer, _ dest: MTLBuffer, _ channels: Int,
                  _ height: Int, _ columns: Int, _ kernel: Int, _ stride: Int, _ relu: Bool,
                  outHeight: Int, outWidth: Int) throws {
        guard let enc = command.makeComputeCommandEncoder() else { throw Invalid.metal }
        let count = co * channels * kernel * kernel
        let params = [channels, height, columns, co, kernel, kernel, stride, kernel / 2,
                      outHeight, outWidth, 1, relu ? 1 : 0].map(UInt32.init)
        enc.setComputePipelineState(convPipe)
        enc.setBuffer(source, offset: 0, index: 0)
        enc.setBuffer(filters, offset: offset * 4, index: 1)
        enc.setBuffer(dest, offset: 0, index: 2)
        params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        enc.setBuffer(filters, offset: (offset + count) * 4, index: 4)
        enc.dispatchThreads(MTLSize(width: co * outHeight * outWidth, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: dispatchWidth, height: 1, depth: 1))
        enc.endEncoding()
        offset += count + co
    }
    func block(_ source: MTLBuffer, _ dest: MTLBuffer, channels: Int, height: Int,
               columns: Int, stride: Int, projection: Bool) throws {
        try dispatch(source, first, channels, height, columns, 3, stride, true, outHeight: oh, outWidth: ow)
        try dispatch(first, second, co, oh, ow, 3, 1, false, outHeight: oh, outWidth: ow)
        let skip: MTLBuffer
        if projection {
            guard let projected = device.makeBuffer(length: outputCount * 4) else { throw Invalid.metal }
            try dispatch(source, projected, channels, height, columns, 1, stride, false,
                         outHeight: oh, outWidth: ow)
            skip = projected
        } else {
            skip = source
        }
        guard let enc = command.makeComputeCommandEncoder() else { throw Invalid.metal }
        enc.setComputePipelineState(addPipe)
        enc.setBuffer(second, offset: 0, index: 0)
        enc.setBuffer(skip, offset: 0, index: 1)
        enc.setBuffer(dest, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: outputCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: addPipe.threadExecutionWidth, height: 1, depth: 1))
        enc.endEncoding()
    }
    var snapshots: [(String, MTLBuffer, Int)] = []
    if stage1 {
        let stemCount = 64 * 112 * 112
        guard let stem = device.makeBuffer(length: stemCount * 4, options: .storageModeShared),
              let pooled = device.makeBuffer(length: outputCount * 4, options: .storageModeShared),
              let block1 = device.makeBuffer(length: outputCount * 4, options: .storageModeShared),
              let pool = library.makeFunction(name: "maxpool_3x3_s2_p1") else { throw Invalid.metal }
        let poolPipe = try device.makeComputePipelineState(function: pool)
        try dispatch(input, stem, ci, h, w, 7, 2, true, outHeight: 112, outWidth: 112)
        try encodePool(command, poolPipe, source: stem, output: pooled, channels: 64, height: 112, width: 112)
        try block(pooled, block1, channels: co, height: oh, columns: ow, stride: 1, projection: false)
        try block(block1, output, channels: co, height: oh, columns: ow, stride: 1, projection: false)
        snapshots = [("stem.f32", stem, stemCount), ("pool.f32", pooled, outputCount),
                     ("block1.f32", block1, outputCount)]
    } else {
        try block(input, output, channels: ci, height: h, columns: w, stride: cfg.stride, projection: cfg.projection)
    }
    command.commit()
    command.waitUntilCompleted()
    let micros = (command.gpuEndTime - command.gpuStartTime) * 1e6
    guard command.status == .completed, micros.isFinite, micros > 0 else { throw Invalid.metal }
    try Data(bytes: output.contents(), count: outputCount * 4).write(to: URL(fileURLWithPath: args[5]))
    for (name, buffer, count) in snapshots {
        let dir = URL(fileURLWithPath: args[5]).deletingLastPathComponent()
        try Data(bytes: buffer.contents(), count: count * 4).write(to: dir.appendingPathComponent(name))
    }
    let evidence: [String: Any] = ["kernel": stage1 ? "resnet_stage1_fp32" : "resnet_basic_block_fp32", "device": device.name,
                                  "registryID": deviceID, "width": width,
                                  "reused": reused, "gpuMicros": micros]
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: evidence))
}

func encodePool(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                source: MTLBuffer, output: MTLBuffer, channels: Int, height: Int, width: Int) throws {
    guard let enc = command.makeComputeCommandEncoder() else { throw Invalid.metal }
    let oh = (height + 1) / 2, ow = (width + 1) / 2
    let params = [height, width, oh, ow].map(UInt32.init)
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(source, offset: 0, index: 0)
    enc.setBuffer(output, offset: 0, index: 1)
    params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
    enc.dispatchThreads(MTLSize(width: channels * oh * ow, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
    enc.endEncoding()
}
