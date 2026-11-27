import Foundation
import LeviathanCore

enum CPUGenerationCommand {
    struct TimingReport: Encodable {
        let format = "leviathan-llama-timing-1"
        let backend: String
        let modelLoadMillis: Double
        let backendPreparationMillis: Double
        let prefillMillis: Double
        let firstTokenMillis: Double
        let decodeMillis: [Double]

        func write() throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            var line = try encoder.encode(self)
            line.append(0x0a)
            FileHandle.standardError.write(line)
        }
    }

    static func openCPU(_ path: String, timed: Bool) throws
        -> (model: LlamaCPU, eos: Int?, loadMillis: Double, preparationMillis: Double) {
        let started = timed ? ProcessInfo.processInfo.systemUptime : 0
        let file = try GGUFReader.read(URL(fileURLWithPath: path))
        let loaded = timed ? ProcessInfo.processInfo.systemUptime : 0
        let model = try LlamaCPU(file: file)
        let eos = try file.metadata["tokenizer.ggml.eos_token_id"].map { _ in
            let value = try file.metadataUInt("tokenizer.ggml.eos_token_id")
            guard let id = Int(exactly: value) else { throw LlamaCPUError.invalid("EOS ID") }
            return id
        }
        let ended = timed ? ProcessInfo.processInfo.systemUptime : 0
        return (model, eos, (loaded - started) * 1000, (ended - loaded) * 1000)
    }

    static func run(_ arguments: [String]) -> Int32 {
        if arguments.first == "llama-serve" { return LlamaServeCommand.run(arguments) }
        let usage = "Usage: leviathan-cli llama-generate --file MODEL.gguf --prompt-ids 1,2 [--gen-tokens N] [--backend cpu-scalar|cuda] [--cuda-library PATH] [--timing json]"
        if arguments.isEmpty || arguments == ["--help"] || arguments == ["llama-generate", "--help"] {
            print(usage)
            return 0
        }
        do {
            let validFlags: Set<String> = ["--file", "--prompt-ids", "--gen-tokens", "--backend", "--cuda-library", "--timing"]
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
            if let timing = options["--timing"], timing != "json" {
                throw LlamaCPUError.invalid("--timing must be json")
            }
            let timed = options["--timing"] != nil
            let backend = options["--backend", default: "cpu-scalar"]
            guard backend == "cpu-scalar" || backend == "cuda" else {
                throw LlamaCPUError.unsupported("llama-generate supports cpu-scalar or cuda")
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

            if backend == "cuda" {
                guard let library = options["--cuda-library"] else {
                    throw LlamaCPUError.invalid("--backend cuda requires --cuda-library")
                }
                let result = try CUDAGeneration.run(library: library, model: path, prompt: prompt, count: count, timed: timed)
                if timed {
                    let t = result.timing
                    try TimingReport(backend: backend, modelLoadMillis: t[0], backendPreparationMillis: t[1],
                                     prefillMillis: t[2], firstTokenMillis: t[3],
                                     decodeMillis: Array(t.dropFirst(4))).write()
                }
                print("backend: cuda (llama.cpp)")
                print("device: " + result.evidence)
                print("generated token IDs: " + result.tokens.map(String.init).joined(separator: ","))
                return 0
            }
            guard options["--cuda-library"] == nil else {
                throw LlamaCPUError.invalid("--cuda-library requires --backend cuda")
            }
            let opened = try openCPU(path, timed: timed)
            var samples: LlamaCPU.GenerationTiming?
            let capture: ((LlamaCPU.GenerationTiming) -> Void)? = timed ? { samples = $0 } : nil
            let result = try opened.model.generate(prompt: prompt, count: count, eos: opened.eos, timing: capture)
            if let t = samples {
                try TimingReport(backend: backend, modelLoadMillis: opened.loadMillis,
                                 backendPreparationMillis: opened.preparationMillis + t.setupMillis,
                                 prefillMillis: t.prefillMillis, firstTokenMillis: t.firstTokenMillis,
                                 decodeMillis: t.decodeMillis).write()
            }
            print("backend: cpu-scalar")
            print("generated token IDs: " + result.map(String.init).joined(separator: ","))
            return 0
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }
}
