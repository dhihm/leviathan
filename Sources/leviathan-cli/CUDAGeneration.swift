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

    static func run(library: String, model: String, prompt: [Int], count: Int) throws
        -> (tokens: [Int32], evidence: String) {
        let prompt32 = prompt.compactMap(Int32.init(exactly:))
        guard prompt32.count == prompt.count, prompt32.allSatisfy({ $0 >= 0 }) else {
            throw LlamaCPUError.invalid("invalid CUDA prompt token ID")
        }
        let libraryPath = URL(fileURLWithPath: library).path
        guard let handle = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL) else {
            let detail = dlerror().map { String(cString: $0) } ?? libraryPath
            throw LlamaCPUError.unsupported("cannot load llama.cpp CUDA library: " + detail)
        }
        defer { dlclose(handle) }
        guard let symbol = dlsym(handle, "leviathan_llama_cuda_generate_v1") else {
            throw LlamaCPUError.unsupported("invalid Leviathan CUDA library ABI")
        }
        let generate = unsafeBitCast(symbol, to: Generate.self)
        var tokens = [Int32](repeating: 0, count: count)
        var message = [CChar](repeating: 0, count: 1024)
        let length = model.withCString {
            generate($0, prompt32, Int32(prompt32.count), Int32(count), &tokens, &message, 1024)
        }
        let evidence = String(decoding: message.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        guard (1...count).contains(Int(length)) else {
            throw LlamaCPUError.unsupported(evidence.isEmpty ? "CUDA generation failed" : evidence)
        }
        return (Array(tokens.prefix(Int(length))), evidence)
    }
}
