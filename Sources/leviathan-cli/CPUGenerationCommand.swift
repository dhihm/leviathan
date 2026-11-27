import Foundation
import LeviathanCore

enum CPUGenerationCommand {
    static func run(_ arguments: [String]) -> Int32 {
        let usage = "Usage: leviathan-cli llama-generate --file MODEL.gguf --prompt-ids 1,2 [--gen-tokens N] [--backend cpu-scalar]"
        if arguments.isEmpty || arguments == ["--help"] || arguments == ["llama-generate", "--help"] {
            print(usage)
            return 0
        }
        do {
            let validFlags: Set<String> = ["--file", "--prompt-ids", "--gen-tokens", "--backend"]
            guard arguments.first == "llama-generate", arguments.count % 2 == 1 else {
                throw LlamaCPUError.invalid(usage)
            }
            var options: [String: String] = [:]
            for i in stride(from: 1, to: arguments.count, by: 2) {
                guard validFlags.contains(arguments[i]), options[arguments[i]] == nil else {
                    throw LlamaCPUError.invalid("unknown or duplicate option: " + arguments[i])
                }
                options[arguments[i]] = arguments[i + 1]
            }
            guard options["--backend", default: "cpu-scalar"] == "cpu-scalar" else {
                throw LlamaCPUError.unsupported("llama-generate supports --backend cpu-scalar")
            }
            guard let path = options["--file"], let ids = options["--prompt-ids"],
                  let count = Int(options["--gen-tokens", default: "8"]) else {
                throw LlamaCPUError.invalid(usage)
            }
            let fields = ids.split(separator: ",", omittingEmptySubsequences: false)
            let prompt = fields.compactMap { Int($0) }
            guard prompt.count == fields.count, (1...512).contains(prompt.count),
                  (1...128).contains(count) else {
                throw LlamaCPUError.invalid("invalid prompt IDs or generation count")
            }

            let file = try GGUFReader.read(URL(fileURLWithPath: path))
            let model = try LlamaCPU(file: file)
            let eos = try file.metadata["tokenizer.ggml.eos_token_id"].map { _ in
                let value = try file.metadataUInt("tokenizer.ggml.eos_token_id")
                guard let id = Int(exactly: value) else { throw LlamaCPUError.invalid("EOS ID") }
                return id
            }
            let result = try model.generate(prompt: prompt, count: count, eos: eos)
            print("backend: cpu-scalar")
            print("generated token IDs: " + result.map(String.init).joined(separator: ","))
            return 0
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }
}
