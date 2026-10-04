import Foundation
import LeviathanCore
let a = CommandLine.arguments
let file = try GGUFReader.read(URL(fileURLWithPath: a[1]))
let config = try Gemma4Config(file: file)
try Gemma4TensorContract.validate(file: file, config: config)
let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: a[2])))
try GemmaGraph.validate(graph, config: config, file: file)
let ssa = try GemmaSSA.lower(graph)
let hardware = try JSONDecoder().decode(ForgeHardwareProfile.self, from: Data(contentsOf: URL(fileURLWithPath: a[3])))
let report = try GemmaShadowComposition.discover(ssa: ssa, config: graph.config, hardware: hardware, epsilon: config.rmsNormEps)
let candidates = try GemmaShadowComposition.shadowCandidates(ssa: ssa, config: graph.config, hardware: hardware, epsilon: config.rmsNormEps)
let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]
try e.encode(report).write(to: URL(fileURLWithPath: a[4]), options: .withoutOverwriting)
let measurement: [String: Any] = ["scope": "Independent read-only measurement on GGUF tensor-contract/graph validated actual IR. Captured actual hardware profile from prior validated lifecycle. No Metal dispatch, runtime activation or throughput claim.", "ssaInstructions": ssa.instructions.count, "ssaDigest": try GemmaSSA.digest(ssa), "epsilon": config.rmsNormEps, "epsilonBits": config.rmsNormEps.bitPattern, "epsilonSource": "Validated Gemma4Config fixed model-family epsilon; not a tunable default in composition", "cacheProofCount": report.body.cacheReport.body.proofs.count, "cacheSupported": report.verdict(for: .gpuCacheAppend)!.supported, "rmsSupported": report.verdict(for: .rmsStatisticsReuse)!.supported, "rmsGroupCount": report.body.rmsPlan.groupCount, "rmsReuseCount": report.body.rmsPlan.reuseCount, "rmsActionCount": report.body.rmsPlan.actions.count, "candidateCount": candidates.count, "candidateListEqualsManual": candidates == GemmaForge.candidates(), "digestVerified": try report.verifyDigest(), "reportDigest": report.reportDigest]
try JSONSerialization.data(withJSONObject: measurement, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: a[5]), options: .withoutOverwriting)
print("ACTUAL_TWO_AXIS_SHADOW_MEASUREMENT_PASS")
