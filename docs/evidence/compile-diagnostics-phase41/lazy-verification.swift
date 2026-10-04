import Foundation
import LeviathanCore

let a = CommandLine.arguments
guard a.count == 5 else {
    FileHandle.standardError.write("usage: \(a.first ?? "driver") <model> <graph> <region> <outputJSON>\n".data(using: .utf8)!)
    exit(2)
}

let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: a[2])))
let region = try MoERegionIR.parse(Data(contentsOf: URL(fileURLWithPath: a[3])))
let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: a[1]))
try runner.installMoeBaselineForVerification(region: region, schedule: MoESchedule(batch: 4, rowsPerGroup: 32), expectedSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256)

let prompt = [2, 9259]
let vocab = 262144
let steps = 2

struct LazyCounts {
    var rmsCooperativeApplied = 0, rmsCooperativeFallback = 0
    var q6CooperativeApplied = 0, q6CooperativeFallback = 0
    var parallelAttentionApplied = 0, parallelAttentionFallback = 0
    var rmsStatsComputed = 0, rmsStatsReused = 0, rmsStatsFallback = 0

    init(_ r: Gemma4Runner) {
        rmsCooperativeApplied = r.rmsCooperativeAppliedCount
        rmsCooperativeFallback = r.rmsCooperativeFallbackCount
        q6CooperativeApplied = r.q6CooperativeAppliedCount
        q6CooperativeFallback = r.q6CooperativeFallbackCount
        parallelAttentionApplied = r.parallelAttentionAppliedCount
        parallelAttentionFallback = r.parallelAttentionFallbackCount
        rmsStatsComputed = r.rmsStatsComputedCount
        rmsStatsReused = r.rmsStatsReusedCount
        rmsStatsFallback = r.rmsStatsFallbackCount
    }

    var allPathsEngaged: Bool {
        rmsCooperativeApplied > 0 && q6CooperativeApplied > 0 && parallelAttentionApplied > 0 && rmsStatsComputed > 0 && rmsStatsReused > 0
    }

    var noFallbacks: Bool {
        rmsCooperativeFallback == 0 && q6CooperativeFallback == 0 && parallelAttentionFallback == 0 && rmsStatsFallback == 0
    }

    var json: [String: Any] {
        ["rmsCooperativeApplied": rmsCooperativeApplied, "rmsCooperativeFallback": rmsCooperativeFallback,
         "q6CooperativeApplied": q6CooperativeApplied, "q6CooperativeFallback": q6CooperativeFallback,
         "parallelAttentionApplied": parallelAttentionApplied, "parallelAttentionFallback": parallelAttentionFallback,
         "rmsStatsComputed": rmsStatsComputed, "rmsStatsReused": rmsStatsReused, "rmsStatsFallback": rmsStatsFallback]
    }
}

func resetLazyCounters(_ r: Gemma4Runner) {
    r.rmsCooperativeCountersReset()
    r.q6CooperativeCountersReset()
    r.parallelAttentionCountersReset()
    r.rmsStatisticsCountersReset()
}

try runner.withQ4DiagnosticReferenceWidth(64) {
    let prep = try GemmaPreparedExecution.prepare(graph: graph, runner: runner)
    let candidate = GemmaForgeCandidate(segmentBatching: true, top5Diagnostic: false, vectorizedLoads: true, gpuCacheAppend: true, parallelAttention: true, rmsnormCooperative: true, matvecCooperativeQ6: true, rmsStatisticsReuse: true, routerGPUResidency: true, routerLaunchDivisor: 2)

    let identityBefore = try runner.engineIdentity()

    // First arm: lazy pipelines compile here; teacher tokens come from this arm's argmax.
    resetLazyCounters(runner)
    var reference = [(argmax: Int, logits: [Float])]()
    var last = 0
    for step in 0..<steps {
        let x = try (step == 0
            ? GemmaForge.executeGraph(graph, runner: runner, tokenIds: prompt, candidate: candidate, prepared: prep)
            : GemmaForge.executeGraphContinue(graph, runner: runner, token: last, candidate: candidate, prepared: prep))
        guard x.logits.count == vocab && x.logits.allSatisfy({ $0.isFinite }) else { fatalError("first arm step \(step): full logits shape/finite") }
        reference.append((x.argmax, x.logits))
        last = x.argmax
    }
    let firstCounts = LazyCounts(runner)
    guard firstCounts.allPathsEngaged else { fatalError("first arm: not all five lazy paths engaged \(firstCounts.json)") }
    guard firstCounts.noFallbacks else { fatalError("first arm: fallback observed \(firstCounts.json)") }

    let identityAfterLazy = try runner.engineIdentity()
    guard identityAfterLazy == identityBefore else { fatalError("engine identity changed after lazy compilation") }

    // Replay arm: same prepared executor, same teacher-forced tokens.
    resetLazyCounters(runner)
    var bitExact = true
    var exactIDs = true
    var replayIDs = [Int]()
    for step in 0..<steps {
        let x = try (step == 0
            ? GemmaForge.executeGraph(graph, runner: runner, tokenIds: prompt, candidate: candidate, prepared: prep)
            : GemmaForge.executeGraphContinue(graph, runner: runner, token: reference[step - 1].argmax, candidate: candidate, prepared: prep))
        guard x.logits.count == vocab && x.logits.allSatisfy({ $0.isFinite }) else { fatalError("replay step \(step): full logits shape/finite") }
        exactIDs = exactIDs && x.argmax == reference[step].argmax
        for (u, v) in zip(x.logits, reference[step].logits) where u.bitPattern != v.bitPattern {
            bitExact = false
            break
        }
        replayIDs.append(x.argmax)
    }
    let replayCounts = LazyCounts(runner)
    guard replayCounts.allPathsEngaged else { fatalError("replay: not all five lazy paths engaged \(replayCounts.json)") }
    guard replayCounts.noFallbacks else { fatalError("replay: fallback observed \(replayCounts.json)") }
    guard bitExact && exactIDs else { fatalError("replay mismatch bitExact=\(bitExact) exactIDs=\(exactIDs)") }

    let identityAfterReplay = try runner.engineIdentity()
    let identityStable = identityAfterReplay == identityBefore && identityAfterLazy == identityBefore
    guard identityStable else { fatalError("engine identity changed after replay") }

    let report: [String: Any] = [
        "scope": "Actual resident model lazy-path verification on current main modern candidate (all existing flags true, router divisor 2). Prompt [2,9259] plus one teacher-forced continuation, replayed with the same prepared executor. Diagnostic E4R32/Q4 width64 install; not production selection, qualification, or throughput approval. No artifacts rewritten.",
        "prompt": prompt,
        "steps": steps,
        "processedTokens": prompt.count + steps - 1,
        "fullVocab": vocab,
        "groupCount": prep.statisticsPlan.groupCount,
        "reuseCount": prep.statisticsPlan.reuseCount,
        "referenceIDs": reference.map { $0.argmax },
        "replayIDs": replayIDs,
        "firstArmCounts": firstCounts.json,
        "replayCounts": replayCounts.json,
        "exactIDs": exactIDs,
        "bitExact": bitExact,
        "identityStable": identityStable,
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: a[4]), options: .withoutOverwriting)
    print("LAZY_PATH_REPLAY_PASS")
}