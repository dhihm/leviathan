import Foundation
import LeviathanCore
let model = URL(fileURLWithPath: CommandLine.arguments[1])
func boundary(_ name: String, _ fields: [String: Any] = [:]) throws {
    var row = fields; row["phase"] = name
    print(String(data: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), encoding: .utf8)!)
    fflush(stdout)
    guard readLine() == "continue" else { throw NSError(domain: "LocalMemoryProbe", code: 1) }
}
try boundary("process_started")
let tokenizer = try autoreleasepool { try GemmaTokenizer(gguf: GGUFReader.read(model)) }
let prompt = try tokenizer.encode("Hello")
try boundary("tokenizer_only", ["promptIDs": prompt])
let digest = try GemmaForge.fileDigest(model)
try boundary("stream_hash_only", ["ggufDigest": digest])
var runner: Gemma4Runner? = try Gemma4Runner(fileAt: model)
try boundary("one_resident_runner_idle", ["metalAllocatedBytes": runner!.metalAllocatedSample(), "maxContext": runner!.config.maxContext, "hidden": runner!.config.hiddenSize, "scope": "No inference, no artifacts installed, no kernel timing; one original resident runner"])
runner = nil
try boundary("runner_released")
print("LOCAL_MEMORY_PROBE_COMPLETED")
