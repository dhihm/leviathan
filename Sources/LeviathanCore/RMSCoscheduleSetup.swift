import Foundation
import Metal

/// Phase 63 bounded setup. GGUFReader.read maps the WHOLE file read-only (Data(contentsOf:) with mappedIfSafe and
/// alwaysMapped, a virtual mapping; no resident-memory claim is made) and tensorData is a zero-copy slice of that
/// mapping. Only the frozen cases' small F32 norm tensors are copied into one small shared Metal buffer (16-byte
/// aligned per tensor). The cooperative RMS pipeline is compiled with the same source, nil options and
/// makeComputePipelineState(function:) calls as Gemma4Runner.rmsCooperativePipeline(). No runner, no forward, no
/// whole-weight Metal buffer, no KV or activation buffers. Weight bytes stay in process and never enter the report.
/// Setup boundaries are emitted as CPU-side lines so a hang can be located; nothing is encoded or committed here.
public struct RMSCoscheduleContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let pipeline: Result<MTLComputePipelineState, Error>
    public let weights: MTLBuffer
    public let weightTable: [String: (offset: Int, bytes: Int)]
    public let eps: Float
}

public enum RMSCoscheduleSetup {
    /// Setup boundary line prefixes (CPU side; no weights, outputs or credentials are ever part of a line).
    public static let beginPrefix = "setup stage begin: "
    public static let donePrefix = "setup stage done: "
    public static let failedPrefix = "setup stage failed: "
    /// Ordered boundary names, in the order make() reaches them.
    public static let stages = ["gguf-map-parse", "config", "tensor-contract", "norm-packing",
                                "metal-device-queue", "weight-buffer", "library-compile", "pipeline-create",
                                "context-ready"]

    /// True only when the reached boundary list is exactly the complete ordered stage list: no missing, extra,
    /// duplicated or reordered boundary. Pure; used by the setup-only diagnostic.
    public static func stagesComplete(_ reached: [String]) -> Bool { reached == stages }

    /// Stage name of a done line (the norm-packing done line carries a parenthesized suffix); nil for any other line.
    public static func stageName(fromDoneLine line: String) -> String? {
        guard line.hasPrefix(donePrefix) else { return nil }
        let rest = line.dropFirst(donePrefix.count)
        guard let first = rest.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true).first,
              !first.isEmpty else { return nil }
        return String(first)
    }

    public static func make(gguf url: URL, emit: (String) -> Void = { _ in }) throws -> RMSCoscheduleContext {
        emit(beginPrefix + "gguf-map-parse")
        let file = try GGUFReader.read(url)
        emit(donePrefix + "gguf-map-parse")
        emit(beginPrefix + "config")
        let config = try Gemma4Config(file: file)
        emit(donePrefix + "config")
        emit(beginPrefix + "tensor-contract")
        try Gemma4TensorContract.validate(file: file, config: config)
        emit(donePrefix + "tensor-contract")
        emit(beginPrefix + "norm-packing")
        let names = RMSCoschedule.cases.flatMap { $0.instances.compactMap { $0.weightTensor } }
        var packed = [UInt8]()
        var table = [String: (offset: Int, bytes: Int)]()
        for name in names where table[name] == nil {
            let info = try file.tensor(name)
            guard GGMLTensorType(rawValue: info.ggmlType) == .float32, info.dims.count == 1 else {
                throw RMSCoscheduleError("weight tensor " + name + " is not a 1-D F32 tensor")
            }
            let data = try file.tensorData(info)
            guard data.count == info.dims[0] * 4 else {
                throw RMSCoscheduleError("weight tensor " + name + " byte count does not match its dims")
            }
            while packed.count % 16 != 0 { packed.append(0) }
            table[name] = (packed.count, data.count)
            packed.append(contentsOf: data)
        }
        emit(donePrefix + "norm-packing" + " (" + String(table.count) + " small norm tensors, "
             + String(packed.count) + " bytes)")
        emit(beginPrefix + "metal-device-queue")
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            emit(failedPrefix + "metal-device-queue")
            throw RMSCoscheduleError("Metal unavailable")
        }
        emit(donePrefix + "metal-device-queue")
        emit(beginPrefix + "weight-buffer")
        guard !packed.isEmpty,
              let buffer = device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared) else {
            emit(failedPrefix + "weight-buffer")
            throw RMSCoscheduleError("weight buffer allocation failed")
        }
        emit(donePrefix + "weight-buffer")
        let pipeline: Result<MTLComputePipelineState, Error>
        do {
            emit(beginPrefix + "library-compile")
            let library = try device.makeLibrary(source: RMSCooperative.source, options: nil)
            emit(donePrefix + "library-compile")
            emit(beginPrefix + "pipeline-create")
            guard let fn = library.makeFunction(name: RMSCooperative.kernelName) else {
                throw RMSCoscheduleError("function " + RMSCooperative.kernelName + " missing")
            }
            pipeline = .success(try device.makeComputePipelineState(function: fn))
            emit(donePrefix + "pipeline-create")
        } catch {
            emit(failedPrefix + "library-compile-or-pipeline-create: " + String(describing: error))
            pipeline = .failure(error)
        }
        emit(donePrefix + "context-ready")
        return RMSCoscheduleContext(device: device, queue: queue, pipeline: pipeline, weights: buffer,
                                    weightTable: table, eps: config.rmsNormEps)
    }
}