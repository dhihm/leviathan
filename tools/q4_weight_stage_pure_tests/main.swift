import Foundation
import LeviathanCore

// Phase 60 pure tests: schedule derivation, fail-closed plan, emitted source structure, the host mirror of the
// cooperative copy, the fixed corpus, tick checks, options, status and the report contract. No Metal device, no
// GGUF, no model weights.

struct Harness {
    var passed = 0
    var failed = 0

    mutating func check(_ ok: Bool, _ name: String) {
        if ok {
            passed += 1
        } else {
            failed += 1
            print("FAIL " + name)
        }
    }
}

func occurrences(_ s: String, _ needle: String) -> Int { s.components(separatedBy: needle).count - 1 }

func checkSchedules(_ h: inout Harness) {
    let s22 = try? Q4WeightStageSchedule.derive(blocksPerTile: 22, rowsPerTile: 32)
    h.check(s22?.segmentBytes == 396 && s22?.segmentWords == 99 && s22?.rowPitchWords == 99 && s22?.paddedPitch == false
            && s22?.threadgroupBytes == 12672 && s22?.kernelName == "q4ws_matvec_R32_B22_P99",
            "B22: 396-byte segments, odd pitch 99 unpadded, 12672 threadgroup bytes")
    let s8 = try? Q4WeightStageSchedule.derive(blocksPerTile: 8, rowsPerTile: 32)
    h.check(s8?.segmentBytes == 144 && s8?.segmentWords == 36 && s8?.rowPitchWords == 37 && s8?.paddedPitch == true
            && s8?.threadgroupBytes == 4736 && s8?.kernelName == "q4ws_matvec_R32_B8_P37",
            "B8: 144-byte segments padded to pitch 37, 4736 threadgroup bytes")
    h.check((try? Q4WeightStageSchedule.derive(blocksPerTile: 7, rowsPerTile: 32)) == nil, "odd block tile refused")
    h.check((try? Q4WeightStageSchedule.derive(blocksPerTile: 22, rowsPerTile: 48)) == nil, "non power-of-two rows refused")
    h.check((try? Q4WeightStageSchedule.derive(blocksPerTile: 64, rowsPerTile: 64)) == nil,
            "64 rows x 64 blocks exceeds 32 KiB and is refused")
    if let s = s22, let data = try? JSONEncoder().encode(s),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        // Decode-based tamper: the memberwise initializer stays internal; a decoded descriptor must still fail
        // validate() when it is not the canonical derivation.
        obj["rowPitchWords"] = 100
        obj["paddedPitch"] = false
        let tampered = (try? JSONSerialization.data(withJSONObject: obj)).flatMap {
            try? JSONDecoder().decode(Q4WeightStageSchedule.self, from: $0)
        }
        if let t = tampered {
            h.check((try? t.validate()) == nil, "an even pitch is not the canonical derivation")
        } else {
            h.check(false, "tampered schedule decodes")
        }
    } else {
        h.check(false, "schedule encodes for the tamper test")
    }
    h.check(Q4WeightStageSchedule.fixedBlockTiles == [22, 8], "exactly two fixed variants")
}

func checkPlan(_ h: inout Harness) {
    guard let s = try? Q4WeightStageSchedule.derive(blocksPerTile: 22, rowsPerTile: 32) else {
        h.check(false, "schedule for plan tests")
        return
    }
    func reason(inDim: Int = 2816, outDim: Int = 64, offset: Int = 256, length: Int = 1 << 30, width: Int = 32,
                maxThreads: Int = 1024, staticBytes: Int = 12672, cap: Int = 32768) -> String? {
        Q4WeightStagePlan.declineReason(inDim: inDim, outDim: outDim, weightOffset: offset, weightLength: length,
                                        schedule: s, executionWidth: width, pipelineMaxThreads: maxThreads,
                                        staticThreadgroupBytes: staticBytes, deviceThreadgroupCap: cap)
    }
    h.check(reason() == nil, "2816 x 64 at offset 256 is supported")
    h.check(reason(inDim: 96, outDim: 8) != nil, "odd block count (96 -> 3 blocks) is declined")
    h.check(reason(offset: 2) != nil, "misaligned weight offset is declined")
    h.check(reason(length: 256 + 63 * 1584) != nil, "a span past the bound buffer is declined")
    h.check(reason(width: 64) != nil, "execution width other than the rows per tile is declined")
    h.check(reason(cap: 8192) != nil, "threadgroup memory above the device cap is declined")
    h.check(reason(staticBytes: 4096) != nil, "a pipeline reporting less static threadgroup memory than the tile is declined")
    h.check(reason(inDim: 48) != nil, "inDim not a multiple of 32 is declined")
    let l = Q4WeightStagePlan.launch(outDim: 33, schedule: s)
    h.check(l.threadgroups == 2 && l.threadsPerThreadgroup == 32
            && Q4WeightStagePlan.launch(outDim: 32, schedule: s).threadgroups == 1, "row tails need one extra threadgroup")
    let c = Q4WeightStagePlan.controlLaunch(outDim: 1, pipelineMaxThreads: 1024)
    h.check(c.threadsPerThreadgroup == 32 && c.threadgroups == 1, "control launch uses the width-32 reference geometry")
}

func checkMirror(_ h: inout Harness) {
    let inDim = 704
    let outDim = 33
    let c = Q4WeightStageCorpus.synthetic().first { $0.inDim == inDim && $0.outDim == 100 }
    guard let base = c else {
        h.check(false, "corpus case for the mirror")
        return
    }
    let rowBytes = (inDim / 32) * 18
    let blocks = Array(base.blocks.prefix(outDim * rowBytes))
    for blocksPerTile in Q4WeightStageSchedule.fixedBlockTiles {
        guard let s = try? Q4WeightStageSchedule.derive(blocksPerTile: blocksPerTile, rowsPerTile: 32) else {
            h.check(false, "schedule " + String(blocksPerTile))
            continue
        }
        var exact = true
        var inactiveNil = true
        for group in 0..<2 {
            var reassembled = [[UInt8]](repeating: [], count: 32)
            var t0 = 0
            while t0 < inDim / 32 {
                let staged = Q4WeightStagePlan.stagedRows(blocks: blocks, inDim: inDim, outDim: outDim, group: group,
                                                          tileStartBlock: t0, schedule: s)
                for lane in 0..<32 {
                    if let seg = staged[lane] { reassembled[lane] += seg } else if group * 32 + lane < outDim { inactiveNil = false }
                }
                t0 += blocksPerTile
            }
            for lane in 0..<32 {
                let row = group * 32 + lane
                if row < outDim {
                    let want = Array(blocks[(row * rowBytes)..<((row + 1) * rowBytes)])
                    if reassembled[lane] != want { exact = false }
                } else if !reassembled[lane].isEmpty {
                    inactiveNil = false
                }
            }
        }
        h.check(exact, "B" + String(blocksPerTile) + ": staged tiles reassemble every active row's bytes exactly, tails included")
        h.check(inactiveNil, "B" + String(blocksPerTile) + ": inactive tail rows stage nothing")
    }
    let (r0, w0) = Q4WeightStagePlan.tileWord(index: 0, segmentWords: 99)
    let (r1, w1) = Q4WeightStagePlan.tileWord(index: 98, segmentWords: 99)
    let (r2, w2) = Q4WeightStagePlan.tileWord(index: 99, segmentWords: 99)
    h.check(r0 == 0 && w0 == 0 && r1 == 0 && w1 == 98 && r2 == 1 && w2 == 0,
            "consecutive copy indices walk consecutive words of one row before the next row")
}

func checkEmitter(_ h: inout Harness) {
    guard let s22 = try? Q4WeightStageSchedule.derive(blocksPerTile: 22, rowsPerTile: 32),
          let s8 = try? Q4WeightStageSchedule.derive(blocksPerTile: 8, rowsPerTile: 32),
          let a = try? Q4WeightStageEmitter.emit(s22), let b = try? Q4WeightStageEmitter.emit(s22),
          let c = try? Q4WeightStageEmitter.emit(s8) else {
        h.check(false, "emitter produces both variants")
        return
    }
    h.check(a == b && a != c, "emission is deterministic and variant-specific")
    h.check(occurrences(a, "kernel void q4ws_matvec_R32_B22_P99(") == 1
            && occurrences(c, "kernel void q4ws_matvec_R32_B8_P37(") == 1
            && occurrences(a, "threadgroup uint tile[3168];") == 1
            && occurrences(c, "threadgroup uint tile[1184];") == 1,
            "one kernel per variant with the derived threadgroup tile size")
    h.check(occurrences(a, "threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
            && occurrences(a, "simd_sum") == 0 && occurrences(a, "partials") == 0
            && occurrences(a, "float4") == 0,
            "two barriers per tile step and no reduction, SIMD or vector-load constructs")
    h.check(occurrences(a, "sum += d * (float(q & 0x0Fu) - 8.0f) * x[") == 1
            && occurrences(a, "sum += d * (float(q >> 4) - 8.0f) * x[") == 1
            && occurrences(a, "float d = float(as_type<half>(dbits));") == 1
            && Q4WeightStageEmitter.controlFunction == "g4_q4_matvec"
            && Gemma4Runner.shaderSource.contains("kernel void g4_q4_matvec("),
            "the production scalar term lines and decode are emitted once; the control is the production function")
}

func checkCorpus(_ h: inout Harness) {
    let a = Q4WeightStageCorpus.synthetic()
    let b = Q4WeightStageCorpus.synthetic()
    h.check(a.count == 10 && Set(a.map { $0.name }).count == 10 && a.filter { $0.expectedDecline }.count == 1,
            "fixed corpus: nine ordinary cases and one predeclared unsupported shape")
    h.check(zip(a, b).allSatisfy { $0.blocks == $1.blocks && $0.x.map { $0.bitPattern } == $1.x.map { $0.bitPattern } },
            "corpus bytes and activations are deterministic")
    h.check(a.allSatisfy { $0.inDim % 32 == 0 && $0.blocks.count == $0.outDim * $0.blocksPerRow * 18
                && $0.x.count == $0.inDim && ($0.expectedDecline ? $0.blocksPerRow % 2 == 1 : $0.blocksPerRow % 2 == 0) },
            "geometry consistent; only the declared unsupported shape has an odd block count")
    var finiteScales = true
    var negativeZero = false
    var subnormal = false
    for c in a {
        for blockIndex in 0..<(c.outDim * c.blocksPerRow) {
            let o = blockIndex * 18
            let bits = UInt16(c.blocks[o]) | UInt16(c.blocks[o + 1]) << 8
            if bits & 0x7C00 == 0x7C00 { finiteScales = false }
            if c.name.hasPrefix("edge-scales") {
                if bits == 0x8000 { negativeZero = true }
                if bits == 0x0001 { subnormal = true }
            }
        }
    }
    h.check(finiteScales && negativeZero && subnormal, "every scale is finite; the edge case carries -0 and subnormal scales")
    let edges = a.filter { $0.name.hasPrefix("edge-") }
    h.check(edges.count == 3 && edges.allSatisfy {
        (try? Q4Oracle.matvecFP64(blocks: $0.blocks[...], outDim: $0.outDim, inDim: $0.inDim, x: $0.x)) != nil },
            "the FP64 oracle accepts every edge case")
    let t = Q4WeightStageCorpus.syntheticTiming()
    h.check(t.count == 2 && t.map { $0.name } == ["timing-2816x1024", "timing-2112x1024"]
            && t.allSatisfy { $0.outDim == 1024 && !$0.expectedDecline },
            "two fixed synthetic timing shapes")
}

func ticks(_ v: [UInt64]) -> Data {
    var d = Data()
    for x in v { withUnsafeBytes(of: x.littleEndian) { d.append(contentsOf: $0) } }
    return d
}

func checkTicks(_ h: inout Harness) {
    let good = ticks([10, 20, 25, 30])
    let ok = Q4WeightStageTicks.analyze(first: good, second: good, previousMax: 5)
    h.check(ok.checks.all && ok.samples == [10, 20, 25, 30], "valid identical resolves pass every check")
    let errorValue = Q4WeightStageTicks.analyze(first: ticks([10, UInt64.max, 25, 30]),
                                                second: ticks([10, UInt64.max, 25, 30]), previousMax: nil)
    let zero = Q4WeightStageTicks.analyze(first: ticks([0, 20, 25, 30]), second: ticks([0, 20, 25, 30]), previousMax: nil)
    h.check(!errorValue.checks.noErrorValues && !zero.checks.noZeroSamples, "counter error values and zero samples are refused")
    let reversed = Q4WeightStageTicks.analyze(first: ticks([20, 10, 25, 30]), second: ticks([20, 10, 25, 30]), previousMax: nil)
    let unordered = Q4WeightStageTicks.analyze(first: ticks([10, 20, 15, 30]), second: ticks([10, 20, 15, 30]), previousMax: nil)
    let stale = Q4WeightStageTicks.analyze(first: good, second: good, previousMax: 10)
    h.check(!reversed.checks.pairsMonotonic && !unordered.checks.encodersOrdered && !stale.checks.laterThanPrevious,
            "end before start, overlapping encoders and stale samples are refused")
    let different = Q4WeightStageTicks.analyze(first: good, second: ticks([10, 20, 25, 31]), previousMax: nil)
    let short = Q4WeightStageTicks.analyze(first: ticks([10, 20]), second: ticks([10, 20]), previousMax: nil)
    h.check(!different.checks.resolveIdentical && different.samples == nil && !short.checks.resolvedSizeExact
            && Q4WeightStageTicks.analyze(first: nil, second: nil, previousMax: nil).samples == nil,
            "differing resolves, short data and missing data never yield samples")
}

func checkContract(_ h: inout Harness) {
    typealias D = Q4WeightStageDiagnostics
    let plain = try? D.parseOptions(["--report", "o.json"])
    let withModel = try? D.parseOptions(["--report", "o.json", "--gguf", "m.gguf"])
    h.check(plain?.report == "o.json" && plain?.gguf == nil && withModel?.gguf == "m.gguf"
            && (try? D.parseOptions(["--gguf", "m.gguf"])) == nil
            && (try? D.parseOptions(["--report", "o.json", "--other", "x"])) == nil
            && (try? D.parseOptions(["--report", "a", "--report", "b"])) == nil,
            "options: report required, gguf optional, unknown and duplicate refused")
    h.check(D.exitCode(.correctnessPassed) == 0 && D.exitCode(.timingChecksFailed) == 6
            && D.exitCode(.correctnessFailed) == 3 && D.exitCode(.unsupported) == 2
            && D.exitCode(.commandBufferError) == 4 && D.exitCode(.notEnded) == 5 && D.exitCode(.aborted) == 1,
            "exit codes")
    let flags = ["fullModelQualified", "gainQualified", "installable", "productionPerformanceClaim", "busyTimeClaim"]
    var contract = false
    if let data = try? D.encodeReport(Q4WeightStageReport()),
       var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
        let base = flags.allSatisfy { (obj[$0] as? Bool) == false }
        for f in flags { obj[f] = true }
        obj["status"] = "correctnessPassed"
        if let raw = try? JSONSerialization.data(withJSONObject: obj),
           let decoded = try? JSONDecoder().decode(Q4WeightStageReport.self, from: raw),
           let again = try? D.encodeReport(decoded),
           let back = (try? JSONSerialization.jsonObject(with: again)) as? [String: Any] {
            contract = base && flags.allSatisfy { (back[$0] as? Bool) == false }
        }
    }
    h.check(contract && D.finalStatus(Q4WeightStageReport()) == .unsupported,
            "every claim flag encodes false, tampered flags re-emit false, an empty report is unsupported")
    h.check(D.warmupPairs == 2 && D.timedPairs == 6 && D.maxReportBytes == 16 << 20
            && Q4WeightStageTicks.sampleCount == 4 && Q4WeightStagePlan.controlLaunchWidth == 32
            && D.correctnessContract == 1e-4 && D.readyToken == RouterOverlapModelGate.readyToken
            && D.goToken == RouterOverlapModelGate.goToken && D.endToken == RouterOverlapModelGate.endToken
            && D.goAccepted("GO") && !D.goAccepted(nil) && !D.goAccepted("GO "),
            "fixed protocol constants and gate tokens")
}

func runAll() -> Int32 {
    var h = Harness()
    checkSchedules(&h)
    checkPlan(&h)
    checkMirror(&h)
    checkEmitter(&h)
    checkCorpus(&h)
    checkTicks(&h)
    checkContract(&h)
    print("q4-weight-stage-pure-tests passed " + String(h.passed) + " failed " + String(h.failed))
    return h.failed == 0 ? 0 : 1
}

exit(runAll())
