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
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void conv2d_fp32(device const float *x [[buffer(0)]],
                                device const float *weights [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant uint *p [[buffer(3)]],
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
            output[index]=sum;
        }
        """, options: nil)
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
    let constants = [ci, h, w, co, kh, kw, stride, padding, oh, ow]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(filters, offset: 0, index: 1)
    constants.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
    encoder.setBuffer(output, offset: 0, index: 2)
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
    try run()
} catch {
    FileHandle.standardError.write(Data("Metal Conv2D: \(error)\n".utf8))
    exit(2)
}
