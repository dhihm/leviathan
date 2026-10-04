import Foundation
import LeviathanCore
let args = CommandLine.arguments
let ssa = try JSONDecoder().decode(GemmaSSAIR.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: args[2])))
let hardware = try JSONDecoder().decode(ForgeHardwareProfile.self, from: Data(contentsOf: URL(fileURLWithPath: args[3])))
let report = try GemmaOpportunityDiscovery.discover(ssa: ssa, config: graph.config, hardware: hardware)
let candidates = try GemmaOpportunityDiscovery.shadowCandidates(ssa: ssa, config: graph.config, hardware: hardware)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
try encoder.encode(report).write(to: URL(fileURLWithPath: args[4]), options: .withoutOverwriting)
let measurement: [String: Any] = ["scope": "Independent measurement of new pure shadow discovery over actual GGUF-validated IR; captured measured hardware profile from the immediately prior validated main lifecycle; no Metal dispatch or speed claim", "actualSSAInstructions": ssa.instructions.count, "reportDigest": report.reportDigest, "supported": report.body.supported, "proofCount": report.body.proofs.count, "reasonCount": report.body.reasons.count, "digestVerified": try report.verifyDigest(), "candidateCount": candidates.count, "candidateListEqualsManual": candidates == GemmaForge.candidates(), "baselineRetained": candidates.first == GemmaForge.candidates().first, "proofLayers": report.body.proofs.map { $0.layer }, "wordWidths": Dictionary(grouping: report.body.proofs, by: { String($0.wordWidth) }).mapValues { $0.count }]
try JSONSerialization.data(withJSONObject: measurement, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[5]), options: .withoutOverwriting)
print("ACTUAL_IR_SHADOW_DISCOVERY_MEASURED")
