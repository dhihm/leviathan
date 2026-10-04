import Foundation
import LeviathanCore
let args = CommandLine.arguments
let file = try GGUFReader.read(URL(fileURLWithPath: args[1]))
let config = try Gemma4Config(file: file)
try Gemma4TensorContract.validate(file: file, config: config)
let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: args[2])))
try GemmaGraph.validate(graph, config: config, file: file)
let ssa = try GemmaSSA.lower(graph)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
try encoder.encode(ssa).write(to: URL(fileURLWithPath: args[3]), options: .withoutOverwriting)
let counts = Dictionary(grouping: ssa.instructions, by: { $0.op }).mapValues { $0.count }
let report: [String: Any] = ["scope": "Independent IR census from existing actual GGUF with full tensor contract and graph validation; no Metal dispatch or performance measurement", "modelBytes": file.byteCount, "tensorCount": file.tensors.count, "graphOps": graph.ops.count, "ssaInstructions": ssa.instructions.count, "ssaValues": ssa.values.count, "ssaDigest": try GemmaSSA.digest(ssa), "counts": counts]
try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: args[4]), options: .withoutOverwriting)
print("ACTUAL_MODEL_IR_EXPORT_PASS")
