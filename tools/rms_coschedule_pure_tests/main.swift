import Foundation
import LeviathanCore

struct Tally {
    var passed = 0
    var failed = 0
    mutating func check(_ name: String, _ ok: Bool) {
        if ok {
            passed += 1
        } else {
            failed += 1
            print("FAIL " + name)
        }
    }
}

func shapes(_ c: RMSCoscheduleCase) -> [String] {
    var out: [String] = []
    for i in c.instances {
        let n: String = String(i.n)
        let vectors: String = String(i.vectors)
        let weight: String = i.weightTensor ?? "-"
        var s: String = i.label
        s += ":"
        s += n
        s += "x"
        s += vectors
        s += ":"
        s += weight
        out.append(s)
    }
    return out
}

func planAt(_ i: RMSCoscheduleInstance, offset: Int = 0, bytes: Int? = nil,
            staticTG: Int = 0) -> (plan: RMSCoschedulePlan?, decline: String?) {
    let scaled = i.weightTensor != nil
    return RMSCoschedule.plan(i, executionWidth: 32, pipelineMaxThreads: 1024, staticThreadgroupMemory: staticTG,
                              deviceMaxThreadgroupMemory: 32768, weightOffset: scaled ? offset : nil,
                              weightBytes: scaled ? (bytes ?? i.n * 4) : nil)
}

func near(_ a: [Double]?, _ b: [Double]) -> Bool {
    guard let a = a, a.count == b.count else { return false }
    for i in a.indices where abs(a[i] - b[i]) > 1e-12 { return false }
    return true
}

func ev(_ a: Double?, _ b: Double?, complete: Bool = true) -> RMSCoscheduleEvidenceInput {
    RMSCoscheduleEvidenceInput(modelIndependent: true, timingComplete: complete, medianSerialFirst: a,
                               medianConcurrentFirst: b)
}

var t = Tally()
let cases = RMSCoschedule.cases

// Frozen case table (forward data flow of Gemma4Runner.forwardLayer).
t.check("frozen case names", cases.map { $0.name }
        == ["local-qkv-norms-blk0", "global-qkv-norms-blk5", "post-ffw-norm-pair-blk0"])
t.check("every case is model-independent", cases.allSatisfy { $0.modelIndependent })
t.check("local trio shapes", shapes(cases[0]) == ["q_norm:256x16:blk.0.attn_q_norm.weight",
                                                 "k_norm:256x8:blk.0.attn_k_norm.weight", "v_norm:256x8:-"])
t.check("global trio shapes", shapes(cases[1]) == ["q_norm:512x16:blk.5.attn_q_norm.weight",
                                                  "k_norm:512x2:blk.5.attn_k_norm.weight", "v_norm:512x2:-"])
t.check("post-ffw pair shapes", shapes(cases[2]) == ["post_ffw_norm_1:2816x1:blk.0.post_ffw_norm_1.weight",
                                                    "post_ffw_norm_2:2816x1:blk.0.post_ffw_norm_2.weight"])
t.check("labels unique per case", cases.allSatisfy { Set($0.instances.map { $0.label }).count == $0.instances.count })

// Fail-closed plan.
let q = cases[0].instances[0]
let v = cases[0].instances[2]
t.check("every frozen instance planned on M1 Max limits", cases.allSatisfy { $0.instances.allSatisfy { planAt($0).plan != nil } })
t.check("misaligned weight offset declines", planAt(q, offset: 4).plan == nil)
t.check("weight byte mismatch declines", planAt(q, bytes: 1020).plan == nil)
t.check("zero vectors declines", planAt(RMSCoscheduleInstance(label: "z", n: 256, vectors: 0, weightTensor: nil)).plan == nil)
t.check("static threadgroup memory over device declines", planAt(q, staticTG: 32769).plan == nil)
t.check("non-multiple-of-4 width declines", planAt(RMSCoscheduleInstance(label: "odd", n: 6, vectors: 1, weightTensor: nil)).plan == nil)
t.check("scale-free instance given a weight declines",
        RMSCoschedule.plan(v, executionWidth: 32, pipelineMaxThreads: 1024, staticThreadgroupMemory: 0,
                           deviceMaxThreadgroupMemory: 32768, weightOffset: 0, weightBytes: 1024).plan == nil)
if let p = planAt(q).plan {
    t.check("plan spans", p.floats == 4096 && p.spanBytes == 16384 && p.scaled
            && p.bufferBytes == p.spanBytes + 2 * RMSCoschedule.guardBytes)
} else {
    t.check("plan spans", false)
}

// Frozen inputs.
let a1 = RMSCoschedule.inputs(seed: 7, count: 1000)
t.check("inputs deterministic and seed-dependent", a1 == RMSCoschedule.inputs(seed: 7, count: 1000)
        && a1 != RMSCoschedule.inputs(seed: 8, count: 1000))
t.check("inputs bounded and exact", a1.count == 1000
        && a1.allSatisfy { $0 >= -2 && $0 < 2 && ($0 * 2097152).rounded() == $0 * 2097152 })
t.check("seeds distinct per case and instance",
        RMSCoschedule.seed(caseIndex: 0, instance: 1) != RMSCoschedule.seed(caseIndex: 1, instance: 0))

// FP64 oracle and contract.
t.check("oracle unscaled", near(RMSCoschedule.oracle(x: [3, 4, 0, 0], weight: nil, n: 4, vectors: 1, eps: 0),
                                [1.2, 1.6, 0, 0]))
t.check("oracle scaled", near(RMSCoschedule.oracle(x: [3, 4, 0, 0], weight: [1, 2, 1, 1], n: 4, vectors: 1, eps: 0),
                              [1.2, 3.2, 0, 0]))
t.check("oracle rejects weight length", RMSCoschedule.oracle(x: [1, 2, 3, 4], weight: [1, 1], n: 4, vectors: 1, eps: 0) == nil)
let want: [Double] = [1.2, 1.6, 0, 0]
t.check("contract accepts Float rounding", (RMSCoschedule.worstVectorRel(got: want.map { Float($0) }, want: want, n: 4) ?? 1) < 1e-7)
t.check("contract flags perturbation", (RMSCoschedule.worstVectorRel(got: [1.2, 1.601, 0, 0], want: want, n: 4) ?? 0) > 1e-4)
t.check("contract nil on count mismatch", RMSCoschedule.worstVectorRel(got: [1], want: want, n: 4) == nil)
t.check("contract infinite on NaN", RMSCoschedule.worstVectorRel(got: [.nan, 1.6, 0, 0], want: want, n: 4) == .infinity)

// Bit identity.
t.check("bits identical", RMSCoschedule.bitMismatches([1, 2, .nan], [1, 2, .nan]) == 0)
t.check("bits one ulp", RMSCoschedule.bitMismatches([1, 2], [1, Float(2).nextUp]) == 1)
t.check("bits signed zero", RMSCoschedule.bitMismatches([0], [-0.0]) == 1)
t.check("bits nil on empty or unequal", RMSCoschedule.bitMismatches([1], [1, 2]) == nil
        && RMSCoschedule.bitMismatches([], []) == nil)

// Pair sample checks.
let okS = RMSCoschedule.analyzePair(raw: [100, 200, 250, 300], pair: 0, serialFirst: true)
t.check("pair serial-first", okS.failures.isEmpty && okS.serialTicks == 100 && okS.concurrentTicks == 50
        && okS.concurrentOverSerial == 0.5)
let okC = RMSCoschedule.analyzePair(raw: [400, 500, 100, 180], pair: 1, serialFirst: false)
t.check("pair concurrent-first", okC.failures.isEmpty && okC.concurrentOverSerial == 0.8)
t.check("pair rejects counter error value",
        !RMSCoschedule.analyzePair(raw: [UInt64.max, 200, 250, 300], pair: 0, serialFirst: true).failures.isEmpty)
t.check("pair rejects zero", !RMSCoschedule.analyzePair(raw: [0, 200, 250, 300], pair: 0, serialFirst: true).failures.isEmpty)
t.check("pair rejects end not after start",
        !RMSCoschedule.analyzePair(raw: [100, 100, 250, 300], pair: 0, serialFirst: true).failures.isEmpty)
t.check("pair rejects overlapping command buffers",
        !RMSCoschedule.analyzePair(raw: [100, 300, 250, 400], pair: 0, serialFirst: true).failures.isEmpty)
t.check("pair rejects wrong sample count", !RMSCoschedule.analyzePair(raw: [1, 2, 3], pair: 0, serialFirst: true).failures.isEmpty)

// Pre-declared evidence rule.
t.check("evidence met at threshold", RMSCoschedule.evidenceThresholdMet([ev(0.5, 0.85), ev(0.7, 0.6)]))
t.check("evidence not met above threshold", !RMSCoschedule.evidenceThresholdMet([ev(0.5, 0.86), ev(0.7, 0.6)]))
t.check("evidence not met when incomplete", !RMSCoschedule.evidenceThresholdMet([ev(0.5, 0.5, complete: false)]))
t.check("evidence not met when empty", !RMSCoschedule.evidenceThresholdMet([]))

// Status precedence and exit codes.
var r = RMSCoscheduleReport()
t.check("empty report aborted", RMSCoschedule.finalStatus(r) == .aborted)
r.commandBufferFailure = "x"
t.check("command-buffer failure first", RMSCoschedule.finalStatus(r) == .commandBufferFailed)
r.commandBufferFailure = nil
r.compileError = "c"
t.check("compile error unsupported", RMSCoschedule.finalStatus(r) == .unsupported)
r.compileError = nil
r.cases = RMSCoschedule.cases.map { RMSCoscheduleCaseResult($0) }
t.check("default case results correctness failed", RMSCoschedule.finalStatus(r) == .correctnessFailed)
for i in r.cases.indices { r.cases[i].correctnessPassed = true }
t.check("no support record unsupported", RMSCoschedule.finalStatus(r) == .unsupported)
r.support = KernelStageSupport(device: "d", supportsAtStageBoundary: true, supportsAtDispatchBoundary: false,
                               timestampCounterSetFound: true, sampleBufferCreated: true, sampleCapacity: 4, error: nil)
t.check("incomplete timing failed", RMSCoschedule.finalStatus(r) == .timingFailed)
for i in r.cases.indices { r.cases[i].timingComplete = true }
t.check("all gates passed", RMSCoschedule.finalStatus(r) == .passed)
t.check("timed arm integrity fields default to unchecked",
        RMSCoscheduleCaseResult(RMSCoschedule.cases[0]).timedArmsChecked == 0
            && RMSCoscheduleCaseResult(RMSCoschedule.cases[0]).timedArmIntegrityFailure == nil)
r.cases[0].timingComplete = false
r.cases[0].timedArmIntegrityFailure = "pair 3 concurrent arm: 1 element(s) differ"
t.check("withheld timing after arm integrity failure is timingFailed", RMSCoschedule.finalStatus(r) == .timingFailed)
t.check("withheld timing never meets the evidence rule",
        !RMSCoschedule.evidenceThresholdMet([ev(0.5, 0.5, complete: false), ev(0.5, 0.5)]))
t.check("exit codes", [RMSCoscheduleStatus.passed, .timingFailed, .correctnessFailed, .unsupported,
                       .commandBufferFailed, .aborted].map { RMSCoschedule.exitCode(status: $0) } == [0, 6, 3, 2, 4, 1])

// Options, report contract and GO.
let sha = String(repeating: "a", count: 64)
let good = try? RMSCoschedule.parseOptions(["--gguf", "m.gguf", "--report", "r.json", "--model-sha256", sha])
t.check("options parse", good == RMSCoscheduleOptions(gguf: "m.gguf", report: "r.json", modelSha256: sha))
let bad1 = (try? RMSCoschedule.parseOptions(["--gguf", "m", "--gguf", "n", "--report", "r", "--model-sha256", sha])) == nil
let bad2 = (try? RMSCoschedule.parseOptions(["--gguf", "m", "--report", "r"])) == nil
let bad3 = (try? RMSCoschedule.parseOptions(["--gguf", "m", "--report", "r", "--model-sha256", sha.uppercased()])) == nil
let bad4 = (try? RMSCoschedule.parseOptions(["--gguf", "m", "--report", "r", "--model-sha256", sha, "--x", "y"])) == nil
t.check("options reject duplicate, missing, non-hex and unknown", bad1 && bad2 && bad3 && bad4)
t.check("setup-only defaults false", good?.setupOnly == false)
let setupOpt = try? RMSCoschedule.parseOptions(["--setup-only", "--gguf", "m.gguf", "--report", "r.json",
                                                 "--model-sha256", sha])
t.check("setup-only flag parsed without a value", setupOpt?.setupOnly == true && setupOpt?.gguf == "m.gguf")
t.check("duplicate setup-only refused",
        (try? RMSCoschedule.parseOptions(["--setup-only", "--setup-only", "--gguf", "m", "--report", "r",
                                          "--model-sha256", sha])) == nil)
let setupReport = RMSCoschedule.setupDiagnostic(context: nil, setupError: "gguf missing",
                                                stagesReached: ["gguf-map-parse"], modelSha256: sha,
                                                executableDigest: sha)
t.check("setup report without context is not completed",
        !setupReport.setupCompleted && setupReport.stagesReached == ["gguf-map-parse"]
            && setupReport.setupError == "gguf missing" && setupReport.hardware == nil)
if let d = try? RMSCoschedule.encodeSetupReport(setupReport), let s = String(data: d, encoding: .utf8) {
    t.check("setup report claims and evaluations false",
            s.contains("\"fullModelQualified\":false") && s.contains("\"gainQualified\":false")
                && s.contains("\"installable\":false") && s.contains("\"productionPerformanceClaim\":false")
                && s.contains("\"busyTimeClaim\":false") && s.contains("\"correctnessEvaluated\":false")
                && s.contains("\"timingEvaluated\":false") && s.contains("\"commandBuffersCommitted\":0")
                && s.contains(RMSCoschedule.setupKind))
} else {
    t.check("setup report claims and evaluations false", false)
}
t.check("setup stage list ends at context-ready", RMSCoscheduleSetup.stages.last == "context-ready"
        && RMSCoscheduleSetup.stages.count == 9)
let allStages = RMSCoscheduleSetup.stages
t.check("stagesComplete accepts the exact ordered list", RMSCoscheduleSetup.stagesComplete(allStages))
t.check("stagesComplete rejects last-only", !RMSCoscheduleSetup.stagesComplete(["context-ready"]))
t.check("stagesComplete rejects missing compile stage",
        !RMSCoscheduleSetup.stagesComplete(allStages.filter { $0 != "library-compile" }))
t.check("stagesComplete rejects missing pipeline stage",
        !RMSCoscheduleSetup.stagesComplete(allStages.filter { $0 != "pipeline-create" }))
t.check("stagesComplete rejects reorder", !RMSCoscheduleSetup.stagesComplete(Array(allStages.reversed())))
t.check("stagesComplete rejects duplicate", !RMSCoscheduleSetup.stagesComplete(allStages + ["context-ready"]))
t.check("stagesComplete rejects extra", !RMSCoscheduleSetup.stagesComplete(allStages + ["run"]))
t.check("stagesComplete rejects empty", !RMSCoscheduleSetup.stagesComplete([]))
let doneP = RMSCoscheduleSetup.donePrefix
t.check("stage name parsed from done lines only",
        RMSCoscheduleSetup.stageName(fromDoneLine: doneP + "norm-packing (3 small norm tensors, 12345 bytes)")
            == "norm-packing"
            && RMSCoscheduleSetup.stageName(fromDoneLine: doneP + "context-ready") == "context-ready"
            && RMSCoscheduleSetup.stageName(fromDoneLine: RMSCoscheduleSetup.beginPrefix + "config") == nil
            && RMSCoscheduleSetup.stageName(fromDoneLine: RMSCoscheduleSetup.failedPrefix + "weight-buffer") == nil
            && RMSCoscheduleSetup.stageName(fromDoneLine: doneP) == nil
            && RMSCoscheduleSetup.stageName(fromDoneLine: doneP + " ") == nil)
let fullNoContext = RMSCoschedule.setupDiagnostic(context: nil, setupError: nil, stagesReached: allStages,
                                                  modelSha256: sha, executableDigest: sha)
t.check("complete stages without context is not completed",
        !fullNoContext.setupCompleted && fullNoContext.stagesReached == allStages && fullNoContext.compileError == nil)
let lastOnlyWithError = RMSCoschedule.setupDiagnostic(context: nil, setupError: nil,
                                                      stagesReached: ["context-ready"], modelSha256: sha,
                                                      executableDigest: sha)
t.check("last-only stages is not completed", !lastOnlyWithError.setupCompleted)
if let data = try? RMSCoschedule.encodeReport(RMSCoscheduleReport()), let s = String(data: data, encoding: .utf8) {
    t.check("report claims false", s.contains("\"fullModelQualified\":false") && s.contains("\"gainQualified\":false")
            && s.contains("\"installable\":false") && s.contains("\"productionPerformanceClaim\":false")
            && s.contains("\"busyTimeClaim\":false") && s.contains(RMSCoschedule.kind))
} else {
    t.check("report claims false", false)
}
t.check("GO exact", RMSCoschedule.goAccepted("GO") && !RMSCoschedule.goAccepted("GO ")
        && !RMSCoschedule.goAccepted("go") && !RMSCoschedule.goAccepted(nil))

print("rms-coschedule-pure-tests passed " + String(t.passed) + " failed " + String(t.failed))
exit(t.failed == 0 ? 0 : 1)
