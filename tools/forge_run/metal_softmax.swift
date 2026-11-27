import Foundation
import Metal

enum Invalid: Error { case input, metal }

func run() throws {
    guard CommandLine.arguments.count == 2 else { throw Invalid.input }
    let wall = ProcessInfo.processInfo.systemUptime
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Invalid.metal
    }
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void final_softmax(device const float *x [[buffer(0)]],
                                  device float *y [[buffer(1)]],
                                  constant uint &width [[buffer(2)]],
                                  uint row [[thread_position_in_grid]]) {
            uint first = row * width;
            float peak = x[first];
            for (uint i = 1; i < width; ++i) peak = max(peak, x[first + i]);
            float sum = 0.0f;
            for (uint i = 0; i < width; ++i) {
                float value = exp(x[first + i] - peak);
                y[first + i] = value;
                sum += value;
            }
            for (uint i = 0; i < width; ++i) y[first + i] /= sum;
        }
        """, options: nil)
    guard let function = library.makeFunction(name: "final_softmax") else { throw Invalid.metal }
    let pipeline = try device.makeComputePipelineState(function: function)
    let preparedPipeline = ProcessInfo.processInfo.systemUptime
    let readHandle = try CommandLine.arguments[1] == "--stdin"
        ? FileHandle.standardInput
        : FileHandle(forReadingFrom: URL(fileURLWithPath: CommandLine.arguments[1]))
    var data = Data()
    while let chunk = try readHandle.read(upToCount: (1 << 20) + 1 - data.count), !chunk.isEmpty {
        data.append(chunk)
        guard data.count <= 1 << 20 else { throw Invalid.input }
    }
    let rows = try JSONDecoder().decode([[Float]].self, from: data)
    guard let width = rows.first?.count, (1...4096).contains(width), rows.count <= 64,
          rows.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }) else { throw Invalid.input }
    let inputReady = ProcessInfo.processInfo.systemUptime
    let flat = rows.flatMap { $0 }
    let length = flat.count * MemoryLayout<Float>.stride
    let input = flat.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: length) }
    guard let input, let output = device.makeBuffer(length: length, options: .storageModeShared),
          let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
        throw Invalid.metal
    }
    var gpuWidth = UInt32(width)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBytes(&gpuWidth, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.dispatchThreads(MTLSize(width: rows.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(rows.count, pipeline.threadExecutionWidth), height: 1, depth: 1))
    encoder.endEncoding()
    let prepared = ProcessInfo.processInfo.systemUptime
    command.commit()
    command.waitUntilCompleted()
    let micros = (command.gpuEndTime - command.gpuStartTime) * 1e6
    guard command.status == .completed, micros.isFinite, micros > 0 else { throw Invalid.metal }
    let values = output.contents().bindMemory(to: Float.self, capacity: flat.count)
    let result = rows.indices.map { row in Array(UnsafeBufferPointer(start: values + row * width, count: width)) }
    let json = try JSONSerialization.data(withJSONObject: [
        "output": result, "device": device.name, "kernel": "final_softmax",
        "gpuMicros": micros, "preparationMillis": (preparedPipeline - wall + prepared - inputReady) * 1000,
    ])
    FileHandle.standardOutput.write(json)
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("Metal Softmax failed: \(error)\n".utf8))
    exit(2)
}
