// Bounded standalone Q4_0 SIMD component qualification experiment.
// Compares the incumbent one-thread-per-row g4_q4_matvec (compiled from
// the actual Gemma4Runner.shaderSource) against original SIMD-group
// kernels q4simd_R{1,4,8} on bounded rows of real GGUF weight tensors.
// No production code is modified; component-only conclusions.
import Foundation
import Metal
import LeviathanCore

struct Err: Error {
    let m: String
    init(_ m: String) { self.m = m }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("error: " + msg + "\n").data(using: .utf8)!)
    exit(1)
}

// ---------- CLI ----------
var ggufPath: String? = nil
var outPath: String? = nil
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let f = it.next() {
        if f == "--gguf", let v = it.next() { ggufPath = v }
        else if f == "--out", let v = it.next() { outPath = v }
        else { fail("unexpected argument \(f); usage: --gguf PATH --out PATH") }
    }
}
guard let ggufArg = ggufPath, let outArg = outPath else {
    fail("usage: --gguf PATH --out PATH")
}

// ---------- Original candidate kernels ----------
// One SIMD-group (32 lanes) per output row; each lane owns quant component
// (lane) in every block, accumulates with Neumaier compensation, and the
// SIMD group reduces with simd_sum. Row = threadgroup * R + simdgroup index.
// Bindings match the incumbent exactly (p uint2 b0, wOff ulong b1, wbase b2,
// x b3, y b4). No threadgroup barriers; the row bound is uniform per group.
let simdKernelSource = """
#include <metal_stdlib>
using namespace metal;
template <uint R, bool compensation>
inline void q4simd_body(constant const uint2& p,
                        constant const ulong& wOff,
                        device const uchar* wbase,
                        device const float* x,
                        device float* y,
                        uint tg, uint sg, uint lane) {
    uint row = tg * R + sg;
    if (row >= p.y) { return; }
    uint blocksPerRow = p.x / 32u;
    device const uchar* rowBytes = wbase + wOff + (ulong)row * blocksPerRow * 18u;
    float s = 0.0f;
    float c = 0.0f;
    for (uint b = 0; b < blocksPerRow; b++) {
        device const uchar* blk = rowBytes + (ulong)b * 18u;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        float d = float(as_type<half>(dbits));
        uint li = lane & 15u;
        uchar q = blk[2u + li];
        float w = (lane < 16u) ? d * (float(q & 0x0Fu) - 8.0f)
                               : d * (float(q >> 4) - 8.0f);
        float prod = w * x[b * 32u + lane];
        if (compensation) {
            float t = s + prod;
            c += (fabs(s) >= fabs(prod)) ? ((s - t) + prod) : ((prod - t) + s);
            s = t;
        } else {
            s += prod;
        }
    }
    float r = 0.0f;
    if (compensation) {
        r = simd_sum(s + c);
    } else {
        r = simd_sum(s);
    }
    if (lane == 0u) { y[row] = r; }
}
kernel void q4simd_R1(constant const uint2& p [[buffer(0)]],
                      constant const ulong& wOff [[buffer(1)]],
                      device const uchar* wbase [[buffer(2)]],
                      device const float* x [[buffer(3)]],
                      device float* y [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<1u, true>(p, wOff, wbase, x, y, tg, sg, lane);
}
kernel void q4simd_R4(constant const uint2& p [[buffer(0)]],
                      constant const ulong& wOff [[buffer(1)]],
                      device const uchar* wbase [[buffer(2)]],
                      device const float* x [[buffer(3)]],
                      device float* y [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<4u, true>(p, wOff, wbase, x, y, tg, sg, lane);
}
kernel void q4simd_R8(constant const uint2& p [[buffer(0)]],
                      constant const ulong& wOff [[buffer(1)]],
                      device const uchar* wbase [[buffer(2)]],
                      device const float* x [[buffer(3)]],
                      device float* y [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<8u, true>(p, wOff, wbase, x, y, tg, sg, lane);
}
kernel void q4simd_plain_R1(constant const uint2& p [[buffer(0)]],
                            constant const ulong& wOff [[buffer(1)]],
                            device const uchar* wbase [[buffer(2)]],
                            device const float* x [[buffer(3)]],
                            device float* y [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<1u, false>(p, wOff, wbase, x, y, tg, sg, lane);
}
kernel void q4simd_plain_R4(constant const uint2& p [[buffer(0)]],
                            constant const ulong& wOff [[buffer(1)]],
                            device const uchar* wbase [[buffer(2)]],
                            device const float* x [[buffer(3)]],
                            device float* y [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<4u, false>(p, wOff, wbase, x, y, tg, sg, lane);
}
kernel void q4simd_plain_R8(constant const uint2& p [[buffer(0)]],
                            constant const ulong& wOff [[buffer(1)]],
                            device const uchar* wbase [[buffer(2)]],
                            device const float* x [[buffer(3)]],
                            device float* y [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    q4simd_body<8u, false>(p, wOff, wbase, x, y, tg, sg, lane);
}
"""

// ---------- Deterministic bounded FP32 inputs (three cases) ----------
func makeInputs(inDim: Int) -> [(label: String, x: [Float])] {
    var sinus = [Float](repeating: 0, count: inDim)
    var alt = [Float](repeating: 0, count: inDim)
    var onehot = [Float](repeating: 0, count: inDim)
    for i in 0..<inDim {
        let d = Double(i)
        sinus[i] = Float(sin(d * 0.137 + 0.31) * (0.5 + 0.5 * cos(d * 0.011)))
        alt[i] = (i % 2 == 0 ? 1.0 : -1.0) * 1e-3
    }
    onehot[inDim - 1] = 0.75
    return [("signedSinusoidal", sinus), ("alternatingSmall", alt), ("onehotNearLast", onehot)]
}

struct Variant {
    let name: String
    let R: Int
    let baseline: Bool
    let pipeline: MTLComputePipelineState
}

struct RunStats { let gpuUS: Double; let wallMS: Double; let groups: Int; let threadsPerThreadgroup: Int }

// Actual dispatch geometry per variant. dispatchThreadgroups takes the number
// of threadGROUPS, never total threads. Baseline: one thread per row, threadgroup
// width capped at min(64, maxTotalThreadsPerThreadgroup). Candidate: one simd-
// group of 32*R per group, R rows per group; groups = ceil(rows/R), no multiplication.
func dispatchGeometry(v: Variant, rows: Int) -> (groups: Int, tgWidth: Int) {
    if v.baseline {
        let w = min(64, v.pipeline.maxTotalThreadsPerThreadgroup)
        return ((rows + w - 1) / w, w)
    }
    return ((rows + v.R - 1) / v.R, 32 * v.R)
}

// One bounded dispatch: NaN-fill first, separate command buffer, completed
// status + positive GPU timestamps enforced, every output row checked finite.
func dispatchOnce(queue: MTLCommandQueue, v: Variant, inDim: Int, rows: Int,
                  wbase: MTLBuffer, xBuf: MTLBuffer, yBuf: MTLBuffer) throws -> RunStats {
    let yCount = rows
    let nan = Float.nan
    let yp = yBuf.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<yCount { yp[i] = nan }
    var p = SIMD2<UInt32>(UInt32(inDim), UInt32(rows))
    var off = UInt64(0)
    guard let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { throw Err("command buffer/encoder") }
    enc.setComputePipelineState(v.pipeline)
    enc.setBytes(&p, length: 8, index: 0)
    enc.setBytes(&off, length: 8, index: 1)
    enc.setBuffer(wbase, offset: 0, index: 2)
    enc.setBuffer(xBuf, offset: 0, index: 3)
    enc.setBuffer(yBuf, offset: 0, index: 4)
    let geo = dispatchGeometry(v: v, rows: rows)
    enc.dispatchThreadgroups(MTLSize(width: geo.groups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: geo.tgWidth, height: 1, depth: 1))
    enc.endEncoding()
    let t0 = DispatchTime.now().uptimeNanoseconds
    cb.commit()
    cb.waitUntilCompleted()
    guard cb.status == .completed, cb.error == nil else {
        throw Err("status \(cb.status.rawValue) error \(String(describing: cb.error))")
    }
    let gs = cb.gpuStartTime, ge = cb.gpuEndTime
    guard gs > 0, ge > gs else { throw Err("nonpositive GPU timestamps") }
    for i in 0..<yCount { guard yp[i].isFinite else { throw Err("nonfinite output row \(i)") } }
    let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    let gpuUS = (ge - gs) * 1e6
    guard gpuUS.isFinite, wall.isFinite else { throw Err("nonfinite timing sample") }
    return RunStats(gpuUS: gpuUS, wallMS: wall, groups: geo.groups, threadsPerThreadgroup: geo.tgWidth)
}

func median(_ a: [Double]) -> Double {
    guard !a.isEmpty else { return .nan }
    let s = a.sorted()
    let m = s.count / 2
    return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
}

// ---------- Setup ----------
let experimentStart = Date()
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
// No MTLDevice.threadExecutionWidth exists; the width-32 requirement is
// enforced on the actual pipeline states in makeVariant and reported from
// variants[0].pipeline.threadExecutionWidth.
guard let queue = device.makeCommandQueue() else { fail("no command queue") }

let incumbentLib: MTLLibrary
let simdLib: MTLLibrary
let simdPlainLib: MTLLibrary
do {
    // Baseline compiled exactly as production (options nil / toolchain default).
    // Candidate Neumaier-compensation kernels compiled with fast math disabled
    // so compensation cannot be optimized away.
    // The plain lane-accumulation family is compiled from the SAME source with
    // options nil (toolchain default, fast math allowed) — the fast-math
    // experiment is the point of that family. Both options are reported.
    incumbentLib = try device.makeLibrary(source: Gemma4Runner.shaderSource, options: nil)
    let simdOpts = MTLCompileOptions()
    simdOpts.fastMathEnabled = false
    simdLib = try device.makeLibrary(source: simdKernelSource, options: simdOpts)
    simdPlainLib = try device.makeLibrary(source: simdKernelSource, options: nil)
} catch { fail("library compile: \(error)") }

func makeVariant(name: String, R: Int, baseline: Bool, lib: MTLLibrary,
                 fn: String) throws -> Variant {
    guard let f = lib.makeFunction(name: fn) else { throw Err("missing function \(fn)") }
    let pipe = try device.makeComputePipelineState(function: f)
    if !baseline {
        let threads = 32 * R
        guard pipe.maxTotalThreadsPerThreadgroup >= threads else {
            throw Err("maxThreads \(pipe.maxTotalThreadsPerThreadgroup) < \(threads) for \(name)")
        }
        guard pipe.threadExecutionWidth == 32 else {
            throw Err("pipeline threadExecutionWidth \(pipe.threadExecutionWidth) for \(name)")
        }
    }
    return Variant(name: name, R: R, baseline: baseline, pipeline: pipe)
}

var unsupportedVariants: [[String: String]] = []
let baselineVariant: Variant
do {
    baselineVariant = try makeVariant(name: "g4_q4_matvec", R: 1, baseline: true,
                                      lib: incumbentLib, fn: "g4_q4_matvec")
} catch { fail("baseline variant setup (fatal): \(error)") }
var variants: [Variant] = [baselineVariant]
for r in [1, 4, 8] {
    do {
        variants.append(try makeVariant(name: "q4simd_R\(r)", R: r, baseline: false,
                                        lib: simdLib, fn: "q4simd_R\(r)"))
    } catch {
        unsupportedVariants.append(["variant": "q4simd_R\(r)", "reason": "\(error)"])
    }
    do {
        variants.append(try makeVariant(name: "q4simd_plain_R\(r)", R: r, baseline: false,
                                        lib: simdPlainLib, fn: "q4simd_plain_R\(r)"))
    } catch {
        unsupportedVariants.append(["variant": "q4simd_plain_R\(r)", "reason": "\(error)"])
    }
}

// ---------- Bounded GGUF case loading ----------
let tensorNames = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight"]
let maxCaseBytes = 32 * 1024 * 1024
let maxRows = 4096

struct CaseDef { let name: String; let inDim: Int; let rows: Int; let bytes: [UInt8] }

let gguf: GGUFFile
do { gguf = try GGUFReader.read(URL(fileURLWithPath: ggufArg)) }
catch { fail("gguf read: \(error)") }

guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: ggufArg)) else {
    fail("cannot open gguf file")
}

var cases: [CaseDef] = []
var unsupportedCases: [[String: String]] = []
for name in tensorNames {
    do {
        let info = try gguf.tensor(name)
        guard info.ggmlType == GGMLTensorType.q4_0.rawValue else {
            unsupportedCases.append(["tensor": name, "reason": "ggml type \(info.ggmlType) is not Q4_0"])
            continue
        }
        guard info.dims.count == 2 else {
            unsupportedCases.append(["tensor": name, "reason": "rank \(info.dims.count) != 2"])
            continue
        }
        let inDim = info.dims[0]
        let outDim = info.dims[1]
        let maxQ4InputDim = 1 << 20
        guard inDim > 0, inDim <= maxQ4InputDim else {
            unsupportedCases.append(["tensor": name, "reason": "inDim \(inDim) outside (0, \(maxQ4InputDim)]"])
            continue
        }
        guard outDim > 0 else {
            unsupportedCases.append(["tensor": name, "reason": "outDim \(outDim) <= 0"])
            continue
        }
        guard inDim % 32 == 0 else {
            unsupportedCases.append(["tensor": name, "reason": "inDim \(inDim) not multiple of 32"])
            continue
        }
        let rowBytesR = (inDim / 32).multipliedReportingOverflow(by: 18)
        guard !rowBytesR.overflow, rowBytesR.partialValue > 0 else {
            unsupportedCases.append(["tensor": name, "reason": "rowBytes overflow for inDim \(inDim)"])
            continue
        }
        let rowBytes = rowBytesR.partialValue
        var rows = min(outDim, maxRows)
        let capRows = maxCaseBytes / rowBytes
        if rows > capRows { rows = capRows }
        guard rows > 0 else {
            unsupportedCases.append(["tensor": name, "reason": "case exceeds 32MiB with rows<1"])
            continue
        }
        let countR = rows.multipliedReportingOverflow(by: rowBytes)
        guard !countR.overflow else { throw Err("rows*rowBytes overflow for \(name)") }
        let count = countR.partialValue
        let range = try gguf.tensorRange(info)
        guard count <= range.count else {
            throw Err("bounded read past tensor end for \(name)")
        }
        try handle.seek(toOffset: UInt64(range.lowerBound))
        guard let data = try handle.read(upToCount: count), data.count == count else {
            throw Err("short bounded read for \(name)")
        }
        cases.append(CaseDef(name: name, inDim: inDim, rows: rows, bytes: [UInt8](data)))
    } catch let e as Err {
        fail("case \(name): \(e.m)")
    } catch {
        unsupportedCases.append(["tensor": name, "reason": "\(error)"])
    }
}

guard !cases.isEmpty else {
    fail("no supported Q4_0 cases in \(ggufArg); refusing to select a candidate with no evidence")
}

// ---------- Per-case gating then timing ----------
let warmups = 5
let measured = 15
let samples = warmups + measured
var gateComparisons = 0
var caseReports: [[String: Any]] = []
var caseSelections: [String: String] = [:]
var candidateEligible: [String: Bool] = [:]
var candidateReductionOK: [String: Bool] = [:]
var candidateRatios: [String: [Double]] = [:]
for v in variants where !v.baseline {
    candidateEligible[v.name] = true
    candidateReductionOK[v.name] = true
}

for c in cases {
    let rowBytes = c.inDim / 32 * 18
    guard let wbuf = device.makeBuffer(bytes: c.bytes, length: c.bytes.count, options: .storageModeShared),
          let xbuf = device.makeBuffer(length: c.inDim * 4, options: .storageModeShared),
          let ybuf = device.makeBuffer(length: c.rows * 4, options: .storageModeShared) else {
        fail("buffer allocation")
    }
    let inputs = makeInputs(inDim: c.inDim)
    let oracleRows = min(c.rows, 256)
    var report: [String: Any] = ["tensor": c.name, "inDim": c.inDim, "rows": c.rows,
                                 "weightBytes": c.bytes.count, "oracleRows": oracleRows,
                                 "dispatch": variants.map { v in
                                     let g = dispatchGeometry(v: v, rows: c.rows)
                                     return ["variant": v.name, "groups": g.groups,
                                             "threadsPerThreadgroup": g.tgWidth,
                                             "totalThreads": g.groups * g.tgWidth] as [String: Any]
                                 }]
    var rejection: [String: String] = [:]
    var timingReport: [[String: Any]] = []
    var caseRatios: [String: [Double]] = [:]

    // Gate before timing: every variant (baseline included) against the
    // independent FP64 oracle on the same actual weight bytes.
    var gatesAllPass: [String: Bool] = [:]
    var gateDetail: [[String: Any]] = []
    for v in variants {
        gatesAllPass[v.name] = true
        for inp in inputs {
            memcpy(xbuf.contents(), inp.x, c.inDim * 4)
            do {
                _ = try dispatchOnce(queue: queue, v: v, inDim: c.inDim, rows: c.rows,
                                     wbase: wbuf, xBuf: xbuf, yBuf: ybuf)
                let y = Array(UnsafeBufferPointer(start: ybuf.contents().assumingMemoryBound(to: Float.self),
                                                  count: oracleRows))
                let oracle = try Q4Oracle.matvecFP64(blocks: c.bytes[0..<oracleRows * rowBytes],
                                                     outDim: oracleRows, inDim: c.inDim, x: inp.x)
                var pass = true
                var worst = 0.0
                var worstRatio = 0.0
                for i in 0..<oracleRows {
                    gateComparisons += 1
                    let d = abs(Double(y[i]) - oracle[i])
                    let tol = 1e-4 + 1e-4 * abs(oracle[i])
                    if !(d <= tol) { pass = false }
                    worst = max(worst, d)
                    worstRatio = max(worstRatio, d / tol)
                }
                gatesAllPass[v.name] = gatesAllPass[v.name]! && pass
                gateDetail.append(["variant": v.name, "input": inp.label, "pass": pass,
                                   "worstAbsDiff": worst, "worstRatioToTolerance": worstRatio])
            } catch {
                gatesAllPass[v.name] = false
                rejection[v.name] = "\(error)"
                gateDetail.append(["variant": v.name, "input": inp.label, "pass": false,
                                   "error": "\(error)"])
            }
        }
        if v.baseline {
            guard gatesAllPass[v.name] == true else { fail("baseline gate failed for \(c.name)") }
        } else if gatesAllPass[v.name] != true {
            candidateEligible[v.name] = false
            if rejection[v.name] == nil { rejection[v.name] = "numerical gate failed" }
        }
    }
    report["gates"] = gateDetail

    // Timing per input, interleaved variant order across samples, separate
    // command buffer per dispatch, weights resident for the whole case.
    for inp in inputs {
        memcpy(xbuf.contents(), inp.x, c.inDim * 4)
        var gpu: [String: [Double]] = [:]
        var wall: [String: [Double]] = [:]
        for s in 0..<samples {
            let rot = s % variants.count
            let order = Array(variants[rot...] + variants[..<rot])
            for v in order {
                if rejection[v.name] != nil { continue }
                do {
                    let st = try dispatchOnce(queue: queue, v: v, inDim: c.inDim, rows: c.rows,
                                              wbase: wbuf, xBuf: xbuf, yBuf: ybuf)
                    gpu[v.name, default: []].append(st.gpuUS)
                    wall[v.name, default: []].append(st.wallMS)
                } catch {
                    rejection[v.name] = "\(error)"
                    candidateEligible[v.name] = false
                }
            }
        }
        var entry: [String: Any] = ["input": inp.label]
        var med: [String: Double] = [:]
        for v in variants {
            guard rejection[v.name] == nil,
                  let g = gpu[v.name], g.count == samples,
                  let w = wall[v.name], w.count == samples else { continue }
            let measuredG = Array(g[warmups...])
            let measuredW = Array(w[warmups...])
            med[v.name] = median(measuredG)
            entry[v.name] = ["gpuUS_raw": g, "wallMS_raw": w,
                             "gpuUS_median": med[v.name]!,
                             "wallMS_median": median(measuredW)]
        }
        if let baseMed = med["g4_q4_matvec"], baseMed > 0 {
            for v in variants where !v.baseline {
                if let m = med[v.name], m > 0, m.isFinite, baseMed.isFinite {
                    let ratio = m / baseMed
                    let reduction = 1.0 - ratio
                    entry["ratio_\(v.name)"] = ratio
                    entry["reduction_\(v.name)"] = reduction
                    candidateRatios[v.name, default: []].append(ratio)
                    caseRatios[v.name, default: []].append(ratio)
                    if !(reduction > 0.05) { candidateReductionOK[v.name] = false }
                } else {
                    candidateReductionOK[v.name] = false
                }
            }
        }
        timingReport.append(entry)
    }
    report["timing"] = timingReport
    if !rejection.isEmpty { report["rejections"] = rejection }

    // Per-tensor shape-specific selection: choose the fastest geometric-mean
    // GPU ratio across THIS tensor's three inputs, only if all gates pass and
    // the win exceeds 5% on EACH of the three inputs; baseline otherwise.
    // First measured results motivate this: q4simd_R8 was slower on
    // attn_q (2816->4096, ratio ~1.11) but faster on attn_output
    // (4096->2816, ratio ~0.84-0.87), so no single universal mapping fits.
    var caseSelected = "g4_q4_matvec (baseline fallback)"
    var caseBestGM: Double? = nil
    var caseSelectionDetail: [[String: Any]] = []
    for v in variants where !v.baseline {
        let ratios = caseRatios[v.name] ?? []
        let gatesOK = gatesAllPass[v.name] == true && rejection[v.name] == nil
        var entry: [String: Any] = ["variant": v.name, "allGatesPass": gatesOK,
                                    "gpuRatioSamples": ratios.count,
                                    "gpuRatios": ratios]
        if ratios.count == inputs.count, ratios.allSatisfy({ $0.isFinite }),
           let product = Optional(ratios.reduce(1.0, *)) {
            let gm = pow(product, 1.0 / Double(ratios.count))
            entry["geometricMeanGpuRatio"] = gm
            let eachInputWin = ratios.allSatisfy({ (1.0 - $0) > 0.05 })
            if gatesOK && eachInputWin && (caseBestGM == nil || gm < caseBestGM!) {
                caseSelected = v.name
                caseBestGM = gm
                entry["reason"] = "per-tensor winner: fastest geometric mean with >5% GPU win on each of the three inputs"
            } else if gatesOK && eachInputWin {
                entry["reason"] = "eligible per-tensor but slower geometric mean than selected"
            } else {
                entry["reason"] = !gatesOK ? "gate failed on this tensor" : "no >5% GPU win on every input of this tensor"
            }
        } else {
            entry["reason"] = !gatesOK ? "gate failed on this tensor" : "insufficient finite timing ratios on this tensor"
        }
        caseSelectionDetail.append(entry)
    }
    report["perTensorSelection"] = ["selected": caseSelected,
                                    "geometricMeanGpuRatio": caseBestGM ?? NSNull(),
                                    "detail": caseSelectionDetail] as [String: Any]
    caseSelections[c.name] = caseSelected
    caseReports.append(report)
}

// ---------- Selection (component-only claim) ----------
var selected: String? = nil
var selectionAgg: Double? = nil
var selectionDetail: [[String: Any]] = []
for v in variants where !v.baseline {
    let ratios = candidateRatios[v.name] ?? []
    let allGates = candidateEligible[v.name] == true
    let eachInputWin = candidateReductionOK[v.name] == true
    var entry: [String: Any] = ["variant": v.name,
                                "allGatesPass": allGates,
                                "winsEveryCaseInput": eachInputWin,
                                "gpuRatioSamples": ratios.count]
    if !ratios.isEmpty, ratios.allSatisfy({ $0.isFinite }), let product = Optional(ratios.reduce(1.0, *)) {
        let gm = pow(product, 1.0 / Double(ratios.count))
        entry["aggregateGeometricMeanGpuRatio"] = gm
        if allGates && eachInputWin && (selectionAgg == nil || gm < selectionAgg!) {
            selected = v.name
            selectionAgg = gm
            entry["reason"] = "fastest eligible by aggregate geometric mean GPU ratio"
        } else if allGates && eachInputWin {
            entry["reason"] = "eligible but slower aggregate geometric mean GPU ratio"
        } else {
            entry["reason"] = !allGates ? "gate failed; excluded from timing" : "no >5% win on every case/input"
        }
    } else {
        entry["reason"] = !allGates ? "gate failed; excluded from timing" : "no finite timing ratios"
    }
    selectionDetail.append(entry)
}

let report: [String: Any] = [
    "device": device.name,
    "threadExecutionWidth": variants[0].pipeline.threadExecutionWidth,
    "compileOptions": [
        "g4_q4_matvec": "nil (toolchain default, matches production)",
        "q4simd_R*": "MTLCompileOptions fastMathEnabled=false (compensated Neumaier)",
        "q4simd_plain_R*": "nil (toolchain default, fast math enabled; plain lane accumulation)",
    ] as [String: String],
    "gguf": ggufArg,
    "elapsedSeconds": Date().timeIntervalSince(experimentStart),
    "warmups": warmups,
    "measured": measured,
    "pipelineCaps": variants.map { [
        "variant": $0.name,
        "maxTotalThreadsPerThreadgroup": $0.pipeline.maxTotalThreadsPerThreadgroup,
        "threadExecutionWidth": $0.pipeline.threadExecutionWidth,
        "staticThreadgroupMemoryLength": $0.pipeline.staticThreadgroupMemoryLength,
    ] },
    "unsupportedCases": unsupportedCases,
    "unsupportedVariants": unsupportedVariants,
    "gateComparisons": gateComparisons,
    "cases": caseReports,
    "selectionDetail": selectionDetail,
    "selected": selected ?? "none",
    "selectionScope": "component-only (no full-model win claim); per-tensor shape-specific selections are recorded in cases[].perTensorSelection (fastest geometric-mean GPU ratio with all gates passing and >5% GPU win on each of the tensor's three inputs, baseline otherwise); global 'selected' remains the conservative universal-winner-or-none",
]

do {
    let data = try JSONSerialization.data(withJSONObject: report,
                                          options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: outArg), options: .withoutOverwriting)
} catch { fail("report write: \(error)") }

print("cases: \(cases.count) supported, \(unsupportedCases.count) unsupported; "
    + "gate comparisons: \(gateComparisons); per-tensor selection: "
    + caseSelections.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ", ")
    + "; universal selected: \(selected ?? "none (no universal winner; use per-tensor mappings)")")
exit(0)
