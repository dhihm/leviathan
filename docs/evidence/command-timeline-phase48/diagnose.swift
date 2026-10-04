import Foundation
import LeviathanCore

let args = CommandLine.arguments
guard args.count == 6 else { fatalError("usage: diagnose MODEL GRAPH REGION FROZEN_IDS OUTPUT") }
let model = URL(fileURLWithPath: args[1])
let cached = try ModelDigestCache.shared.cachedRecord(for: model)
guard cached != nil else { fatalError("historical model cache required; no fresh model hash in this diagnosis") }
let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: args[2])))
let region = try MoERegionIR.parse(Data(contentsOf: URL(fileURLWithPath: args[3])))
let frozenJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[4]))) as! [String: Any]
let frozenRows = frozenJSON["fixtures"] as! [[String: Any]]
var frozenIDs: [String: [Int]] = [:]
for row in frozenRows { frozenIDs[row["fixture"] as! String] = (row["generatedIDs"] as! String).split(separator: " ").map { Int($0)! } }
let file = try GGUFReader.read(model)
let tokenizer = try GemmaTokenizer(gguf: file)
let config = try Gemma4Config(file: file)
try GemmaGraph.validate(graph, config: config, file: file)
let runner = try Gemma4Runner(fileAt: model)
try runner.installMoeBaselineForVerification(region: region, schedule: MoESchedule(batch: 4, rowsPerGroup: 32), expectedSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256)
let candidate = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true, gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true, matvecCooperativeQ6: true, rmsStatisticsReuse: true, routerGPUResidency: true, routerLaunchDivisor: 1)
struct Output { let id: Int; let logits: [Float] }
struct StepRecord: Codable { let step: Int; let wallSeconds: Double; let gpuSeconds: Double }
struct FixtureReport: Codable {
    let fixture: String; let promptIDs: [Int]; let generatedIDs: [Int]
    let frozenBaselineIDsMatch: Bool; let instrumentationIDsMatch: Bool
    let fullVocab: Int; let fullOutputCount: Int; let worstRelToSerial: Double; let worstRelOnToOff: Double
    let trace: Gemma4CommandTimeline.Trace; let summary: Gemma4CommandTimeline.Summary
    let steps: [StepRecord]
}
struct Report: Codable {
    let scope: String; let cachedModelDigest: ModelDigestRecord
    let fixtures: [FixtureReport]
}
func worst(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == 262144, b.count == 262144, a.allSatisfy({ $0.isFinite }), b.allSatisfy({ $0.isFinite }) else { fatalError("full finite vocabulary gate failed") }
    return zip(a,b).reduce(0.0) { max($0, abs(Double($1.0) - Double($1.1)) / max(1.0, abs(Double($1.1)))) }
}
func seconds(_ d: ContinuousClock.Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
var reports: [FixtureReport] = []
try runner.withQ4DiagnosticReferenceWidth(32) {
    let prepared = try GemmaPreparedExecution.prepare(graph: graph, runner: runner)
    for (name,text,chat) in [("raw-hello","Hello",false),("en-chat","Hello, can you help me today?",true),("kr-chat","안녕하세요, 오늘 날씨는 어떤가요?",true)] {
        let input = chat ? "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines) + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>" : text
        let prompt = try tokenizer.encode(input)
        runner.commandTimelineEnabled = false
        runner.reset()
        var serial: [Output] = []
        for step in 0..<32 {
            let e = try runner.forward(tokenIds: step == 0 ? prompt : [serial.last!.id])
            serial.append(Output(id:e.argmaxToken, logits: try runner.currentLogits()))
        }
        guard serial.map({ $0.id }) == frozenIDs[name] else { fatalError("serial sequence differs from frozen baseline: " + name) }
        var arms: [[Output]] = []
        var tracedSteps: [StepRecord] = []
        for enabled in [false,true] {
            runner.commandTimelineEnabled = false
            runner.reset()
            let prefill = try GemmaForge.executeGraph(graph, runner: runner, tokenIds: prompt, candidate: candidate, prepared: prepared)
            var outputs = [Output(id:prefill.argmax, logits:prefill.logits)]
            try runner.commandTimelineReset(capacity:8192)
            runner.commandTimelineEnabled = enabled
            for step in 1..<32 {
                let t = ContinuousClock.now
                let x = try GemmaForge.executeGraphContinue(graph, runner:runner, token:serial[step-1].id, candidate:candidate, prepared:prepared)
                let wall = seconds(ContinuousClock.now - t)
                outputs.append(Output(id:x.argmax, logits:x.logits))
                if enabled { tracedSteps.append(StepRecord(step:step,wallSeconds:wall,gpuSeconds:x.gpuMicros/1e6)) }
            }
            runner.commandTimelineEnabled = false
            arms.append(outputs)
        }
        let trace = runner.commandTimelineSnapshot()
        let summary = Gemma4CommandTimeline.analyze(trace)
        guard summary.eventCount > 0, summary.validEventCount == summary.eventCount, summary.rejections.isEmpty, !summary.truncated, summary.missingSequenceCount == 0, summary.gapPairsSkipped == 0 else { fatalError("invalid or truncated timeline " + name) }
        guard arms.allSatisfy({ $0.map({$0.id}) == serial.map({$0.id}) }) else { fatalError("instrumentation IDs differ " + name) }
        var serialWorst = 0.0, toggleWorst = 0.0
        for step in 0..<32 {
            serialWorst = max(serialWorst,worst(arms[0][step].logits,serial[step].logits),worst(arms[1][step].logits,serial[step].logits))
            toggleWorst = max(toggleWorst,worst(arms[1][step].logits,arms[0][step].logits))
        }
        guard serialWorst < 1e-4, toggleWorst < 1e-4 else { fatalError("strict numerical gate failed " + name) }
        reports.append(FixtureReport(fixture:name,promptIDs:prompt,generatedIDs:serial.map({$0.id}),frozenBaselineIDsMatch:true,instrumentationIDsMatch:true,fullVocab:262144,fullOutputCount:32,worstRelToSerial:serialWorst,worstRelOnToOff:toggleWorst,trace:trace,summary:summary,steps:tracedSteps))
        let encoder = JSONEncoder();encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        let report = Report(scope:"Diagnostic verification-only MoE E4/R32 and Q4 width32 installation, same current modern prepared candidate. No persisted production artifacts rewritten or admitted. Serial reference and OFF/ON strict 32-output full-vocabulary gates; trace includes only31 continuation steps. No speed/gain qualification under failed prior A/A controls. Captured gaps include this diagnostic driver's output storage/loop overhead, host scheduling and possible paging; not CPU busy time or another-process attribution.",cachedModelDigest:cached!,fixtures:reports)
        try encoder.encode(report).write(to:URL(fileURLWithPath:args[5]),options:[.atomic])
        print("FIXTURE_GATE_AND_TRACE_PASS \(name): \(summary.eventCount) valid commands, worst serial \(serialWorst), toggle \(toggleWorst)")
        fflush(stdout)
    }
}
print("THREE_FIXTURE_COMMAND_TIMELINE_DIAGNOSIS_COMPLETE")
