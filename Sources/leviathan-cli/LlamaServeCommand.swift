import Foundation
import LeviathanCore
#if os(Linux)
import Glibc
#else
import Darwin
#endif

enum LlamaServeCommand {
    private static let usage = "Usage: leviathan-cli llama-serve --file MODEL.gguf [--backend cpu-scalar|cuda] [--cuda-library PATH] [--stream jsonl]"

    private struct Request: Decodable {
        let id: Int
        let promptIds: [Int]
        let count: Int
        let cancelAfterTokens: Int?

        func validate() throws {
            guard id > 0, (1...512).contains(promptIds.count), (1...128).contains(count),
                  cancelAfterTokens.map({ $0 > 0 && $0 < count }) ?? true else {
                throw LlamaCPUError.invalid("request id, prompt length, token count or cancellation budget")
            }
        }
    }

    private struct Response: Encodable {
        let kind: String
        let id: Int?
        var tokens: [Int]?
        var timing: CPUGenerationCommand.TimingReport?
        var device: String?
        var error: String?
    }

    private struct TokenEvent: Encodable {
        let kind = "token"
        let id: Int
        let index: Int
        let tokenId: Int
    }

    private struct Ready: Encodable {
        let kind = "ready"
        let format: String
        let backend: String
        let modelLoadMillis: Double
        let modelPreparationMillis: Double
    }

    private final class LineReader {
        private var pending = Data()
        private var buffer = [UInt8](repeating: 0, count: 4096)
        private let limit = 16 << 10

        func next() throws -> Data? {
            while true {
                if let newline = pending.firstIndex(of: 0x0a) {
                    let line = pending.subdata(in: pending.startIndex..<newline)
                    pending.removeSubrange(pending.startIndex...newline)
                    guard line.count <= limit else {
                        throw LlamaCPUError.invalid("request line exceeds 16 KiB")
                    }
                    return line
                }
                guard pending.count <= limit else {
                    throw LlamaCPUError.invalid("request line exceeds 16 KiB")
                }
                let capacity = buffer.count
                var length: Int
                repeat {
                    length = read(STDIN_FILENO, &buffer, capacity)
                } while length < 0 && errno == EINTR
                if length < 0 { throw LlamaCPUError.invalid("stdin read failed") }
                if length == 0 {
                    if pending.isEmpty { return nil }
                    throw LlamaCPUError.invalid("unterminated request")
                }
                pending.append(contentsOf: buffer[..<length])
            }
        }
    }

    static func run(_ arguments: [String]) -> Int32 {
        if arguments == ["llama-serve", "--help"] {
            print(usage)
            return 0
        }
        do {
            guard arguments.count % 2 == 1 else { throw LlamaCPUError.invalid(usage) }
            let flags: Set<String> = ["--file", "--backend", "--cuda-library", "--stream"]
            var options: [String: String] = [:]
            for i in stride(from: 1, to: arguments.count, by: 2) {
                guard flags.contains(arguments[i]), options[arguments[i]] == nil else {
                    throw LlamaCPUError.invalid("unknown or duplicate option: " + arguments[i])
                }
                options[arguments[i]] = arguments[i + 1]
            }
            guard let file = options["--file"] else { throw LlamaCPUError.invalid(usage) }
            if let mode = options["--stream"], mode != "jsonl" {
                throw LlamaCPUError.invalid("--stream must be jsonl")
            }
            let streaming = options["--stream"] != nil
            let format = streaming ? "leviathan-llama-serve-stream-1" : "leviathan-llama-serve-1"
            let backend = options["--backend", default: "cpu-scalar"]
            switch backend {
            case "cpu-scalar":
                guard options["--cuda-library"] == nil else {
                    throw LlamaCPUError.invalid("--cuda-library requires --backend cuda")
                }
                let opened = try CPUGenerationCommand.openCPU(file, timed: true)
                let ready = Ready(format: format, backend: backend, modelLoadMillis: opened.loadMillis,
                                  modelPreparationMillis: opened.preparationMillis)
                try serve(ready: ready, streaming: streaming) { request, onToken in
                    var measurements: LlamaCPU.GenerationTiming?
                    let tokens = try opened.model.generate(prompt: request.promptIds, count: request.count,
                        eos: opened.eos, timing: { measurements = $0 }, cancelAfterTokens: request.cancelAfterTokens,
                        onToken: onToken)
                    guard let timing = measurements else {
                        throw LlamaCPUError.invalid("missing CPU timing")
                    }
                    return Response(kind: "result", id: request.id, tokens: tokens,
                        timing: .init(backend: backend, modelLoadMillis: 0,
                                      backendPreparationMillis: timing.setupMillis,
                                      prefillMillis: timing.prefillMillis, firstTokenMillis: timing.firstTokenMillis,
                                      decodeMillis: timing.decodeMillis))
                }
            case "cuda":
                guard let library = options["--cuda-library"] else {
                    throw LlamaCPUError.invalid("--backend cuda requires --cuda-library")
                }
                let session = try CUDAGeneration.Session(library: library, model: file, streaming: streaming)
                let ready = Ready(format: format, backend: backend, modelLoadMillis: session.loadMillis,
                                  modelPreparationMillis: session.preparationMillis)
                try serve(ready: ready, streaming: streaming) { request, onToken in
                    let result = try session.generate(prompt: request.promptIds, count: request.count,
                                                      cancelAfterTokens: request.cancelAfterTokens, onToken: onToken)
                    return Response(kind: "result", id: request.id, tokens: result.tokens,
                                    timing: result.timing, device: result.evidence)
                }
            default:
                throw LlamaCPUError.unsupported("llama-serve supports cpu-scalar or cuda")
            }
            return 0
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            return 1
        }
    }

    private static func serve(ready: Ready, streaming: Bool,
                              execute: (Request, ((Int, Int) throws -> Void)?) throws -> Response) throws {
        let reader = LineReader()
        let decoder = JSONDecoder()
        let requestFields: Set<String> = ["id", "promptIds", "count", "cancelAfterTokens"]
        let encoder = JSONEncoder()
        func write<T: Encodable>(_ response: T) throws {
            var line = try encoder.encode(response)
            line.append(0x0a)
            try FileHandle.standardOutput.write(contentsOf: line)
        }

        try write(ready)
        while let line = try reader.next() {
            var id: Int?
            let response: Response
            do {
                guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any],
                      Set(object.keys).isSubset(of: requestFields) else {
                    throw LlamaCPUError.invalid("unexpected request fields")
                }
                let request = try decoder.decode(Request.self, from: line)
                id = request.id
                try request.validate()
                let onToken: ((Int, Int) throws -> Void)? = streaming ? { token, index in
                    try write(TokenEvent(id: request.id, index: index, tokenId: token))
                } : nil
                response = try execute(request, onToken)
            } catch LlamaCPUError.cancelled {
                response = Response(kind: "cancelled", id: id)
            } catch {
                response = Response(kind: "error", id: id, error: String(describing: error))
            }
            try write(response)
        }
    }
}
