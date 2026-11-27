import Foundation
import LeviathanCore
#if os(Linux)
import Glibc
#else
import Darwin
#endif

enum CUDAGeneration {
    private typealias Generate = @convention(c) (
        UnsafePointer<CChar>, UnsafePointer<Int32>, Int32, Int32,
        UnsafeMutablePointer<Int32>, UnsafeMutablePointer<CChar>, Int32
    ) -> Int32

    private typealias GenerateTimed = @convention(c) (
        UnsafePointer<CChar>, UnsafePointer<Int32>, Int32, Int32,
        UnsafeMutablePointer<Int32>, UnsafeMutablePointer<CChar>, Int32,
        UnsafeMutablePointer<Double>, Int32
    ) -> Int32

    static func run(library: String, model: String, prompt: [Int], count: Int, timed: Bool = false) throws
        -> (tokens: [Int32], evidence: String, timing: [Double]) {
        let prompt32 = prompt.compactMap(Int32.init(exactly:))
        guard prompt32.count == prompt.count, prompt32.allSatisfy({ $0 >= 0 }) else {
            throw LlamaCPUError.invalid("invalid CUDA prompt token ID")
        }
        let libraryStarted = timed ? ProcessInfo.processInfo.systemUptime : 0
        let libraryPath = URL(fileURLWithPath: library).path
        guard let handle = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL) else {
            let detail = dlerror().map { String(cString: $0) } ?? libraryPath
            throw LlamaCPUError.unsupported("cannot load llama.cpp CUDA library: " + detail)
        }
        defer { dlclose(handle) }
        let abi = timed ? "leviathan_llama_cuda_generate_v2" : "leviathan_llama_cuda_generate_v1"
        guard let symbol = dlsym(handle, abi) else {
            throw LlamaCPUError.unsupported(timed ? "CUDA timing requires the v2 library ABI"
                                                  : "invalid Leviathan CUDA library ABI")
        }
        let openMillis = timed ? (ProcessInfo.processInfo.systemUptime - libraryStarted) * 1000 : 0
        var tokens = [Int32](repeating: 0, count: count)
        var message = [CChar](repeating: 0, count: 1024)
        var timing = [Double](repeating: 0, count: timed ? count + 3 : 0)
        let timingCapacity = Int32(timing.count)
        let length = model.withCString {
            if timed {
                let generate = unsafeBitCast(symbol, to: GenerateTimed.self)
                return generate($0, prompt32, Int32(prompt32.count), Int32(count), &tokens, &message, 1024,
                                &timing, timingCapacity)
            }
            let generate = unsafeBitCast(symbol, to: Generate.self)
            return generate($0, prompt32, Int32(prompt32.count), Int32(count), &tokens, &message, 1024)
        }
        let evidence = String(decoding: message.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        guard (1...count).contains(Int(length)) else {
            throw LlamaCPUError.unsupported(evidence.isEmpty ? "CUDA generation failed" : evidence)
        }
        if timed {
            timing = Array(timing.prefix(Int(length) + 3))
            timing[1] += openMillis
            guard timing.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw LlamaCPUError.invalid("invalid CUDA timing")
            }
        }
        return (Array(tokens.prefix(Int(length))), evidence, timing)
    }
}

extension CUDAGeneration {
    final class Session {
        private typealias Open = @convention(c) (
            UnsafePointer<CChar>, UnsafeMutablePointer<CChar>, Int32, UnsafeMutablePointer<Double>
        ) -> OpaquePointer?
        private typealias Close = @convention(c) (OpaquePointer) -> Void
        private typealias Request = @convention(c) (
            OpaquePointer, UnsafePointer<Int32>, Int32, Int32,
            UnsafeMutablePointer<Int32>, UnsafeMutablePointer<CChar>, Int32,
            UnsafeMutablePointer<Double>, Int32, Int32
        ) -> Int32

        private let library: UnsafeMutableRawPointer
        private let model: OpaquePointer
        private let request: Request
        private let close: Close
        let loadMillis: Double
        let preparationMillis: Double

        init(library path: String, model pathToModel: String) throws {
            let started = ProcessInfo.processInfo.systemUptime
            guard let handle = dlopen(URL(fileURLWithPath: path).path, RTLD_NOW | RTLD_LOCAL) else {
                throw LlamaCPUError.unsupported("cannot load llama.cpp CUDA library: " + path)
            }
            var retained = false
            defer { if !retained { dlclose(handle) } }
            guard let openSymbol = dlsym(handle, "leviathan_llama_cuda_open_v1"),
                  let requestSymbol = dlsym(handle, "leviathan_llama_cuda_request_v1"),
                  let closeSymbol = dlsym(handle, "leviathan_llama_cuda_close_v1") else {
                throw LlamaCPUError.unsupported("CUDA serving requires the resident library ABI")
            }
            let libraryMillis = (ProcessInfo.processInfo.systemUptime - started) * 1000
            var message = [CChar](repeating: 0, count: 1024)
            var timing = [Double](repeating: 0, count: 2)
            let open = unsafeBitCast(openSymbol, to: Open.self)
            guard let opened = pathToModel.withCString({ open($0, &message, 1024, &timing) }) else {
                throw LlamaCPUError.unsupported(Self.text(message))
            }
            library = handle
            model = opened
            request = unsafeBitCast(requestSymbol, to: Request.self)
            close = unsafeBitCast(closeSymbol, to: Close.self)
            loadMillis = timing[0]
            preparationMillis = timing[1] + libraryMillis
            retained = true
        }

        deinit {
            close(model)
            dlclose(library)
        }

        private static func text(_ bytes: [CChar]) -> String {
            String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }

        func generate(prompt: [Int], count: Int, cancelAfterTokens: Int?) throws
            -> (tokens: [Int], timing: CPUGenerationCommand.TimingReport, evidence: String) {
            let prompt32 = prompt.compactMap(Int32.init(exactly:))
            guard prompt32.count == prompt.count, prompt32.allSatisfy({ $0 >= 0 }) else {
                throw LlamaCPUError.invalid("invalid CUDA prompt token ID")
            }
            var tokens = [Int32](repeating: 0, count: count)
            var message = [CChar](repeating: 0, count: 1024)
            var t = [Double](repeating: 0, count: count + 3)
            let timingCapacity = Int32(t.count)
            let length = request(model, prompt32, Int32(prompt32.count), Int32(count), &tokens, &message,
                                 1024, &t, timingCapacity, Int32(cancelAfterTokens ?? 0))
            if length == -2 { throw LlamaCPUError.cancelled }
            guard (1...count).contains(Int(length)) else {
                throw LlamaCPUError.unsupported(Self.text(message))
            }
            let used = t.prefix(Int(length) + 3)
            guard used.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw LlamaCPUError.invalid("invalid CUDA request timing")
            }
            let samples = CPUGenerationCommand.TimingReport(
                backend: "cuda", modelLoadMillis: used[0], backendPreparationMillis: used[1],
                prefillMillis: used[2], firstTokenMillis: used[3], decodeMillis: Array(used.dropFirst(4)))
            return (tokens.prefix(Int(length)).map(Int.init), samples, Self.text(message))
        }
    }
}
