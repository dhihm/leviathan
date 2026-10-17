// moe-simd-qualify — real-GGUF component qualification for the cooperative
// SIMD routed-MoE lowering (correctness prototype; NON-INSTALLABLE report).
//
// Build (Package.swift unchanged):
//   swift build -c release
//   swiftc -O -I .build/release/Modules tools/moe_simd_qualification/main.swift .build/release/LeviathanCore.build/*.swift.o -o .build/release/moe-simd-qualify
// Run:
//   .build/release/moe-simd-qualify --gguf MODEL.gguf --region tools/moe_lower/fixtures/gemma4-26b-region.json --out NEW_REPORT.json
//
// The tool refuses to overwrite and never downloads or hashes the model:
// it reads the already-seeded ModelDigestCache record and fails when that
// record is missing (refreshing it is a separate explicit user command).
// It reads only the routed expert slices of the layer-0 families (bounded
// by MoEComponentInputs) plus one captured layer-0 pre_ffw_norm_2
// activation, and writes results and identities only — no weight bytes and
// no activation values. GPU intervals are diagnostic: the A/A
// reproducibility gate failed, so no paired comparison, winner or
// activation is produced.

import Foundation
import Metal
import LeviathanCore

struct ToolError: Error, CustomStringConvertible {
    static let usage =
        "usage: moe-simd-qualify --gguf MODEL.gguf --region REGION.json --out NEW_REPORT.json"
    let description: String
    init(_ description: String) { self.description = description }
}

func parseArguments(_ argv: [String]) throws -> (gguf: URL, region: URL, out: URL) {
    var values = [String: String]()
    var i = 0
    while i < argv.count {
        let key = argv[i]
        guard ["--gguf", "--region", "--out"].contains(key), i + 1 < argv.count,
              values[key] == nil else {
            throw ToolError(ToolError.usage)
        }
        values[key] = argv[i + 1]
        i += 2
    }
    guard let gguf = values["--gguf"], let region = values["--region"],
          let out = values["--out"] else {
        throw ToolError(ToolError.usage)
    }
    return (URL(fileURLWithPath: gguf), URL(fileURLWithPath: region),
            URL(fileURLWithPath: out))
}

final class ActivationCapture {
    var vector: [Float]?
    var calls = 0
}

/// Captures the actual layer-0 routed-MoE input (the pre_ffw_norm_2 output)
/// at position 1 of the raw-Hello history [2, 9259] through the default-off
/// moeInputProbe on the direct reference forward. The callback detaches
/// itself after the capture; the runner and its resident mapping are
/// released before any component allocation. Diagnostic only, never timed.
func captureLayer0MoEInput(gguf: URL, hidden: Int) throws -> [Float] {
    let capture = ActivationCapture()
    do {
        let runner = try Gemma4Runner(fileAt: gguf)
        runner.moeInputProbe = { [weak runner] layer, position, x in
            capture.calls += 1
            guard layer == 0, position == 1 else { return }
            capture.vector = x
            runner?.moeInputProbe = nil
        }
        defer { runner.moeInputProbe = nil }
        _ = try runner.forward(tokenIds: [2, 9259])
    }
    guard let vector = capture.vector else {
        throw ToolError("layer-0 MoE input was not captured (probe calls \(capture.calls))")
    }
    guard vector.count == hidden, vector.allSatisfy({ $0.isFinite }) else {
        throw ToolError("captured MoE input must be \(hidden) finite floats (got \(vector.count))")
    }
    return vector
}

func runQualification() throws {
    let args = try parseArguments(Array(CommandLine.arguments.dropFirst()))
    guard !FileManager.default.fileExists(atPath: args.out.path) else {
        throw ToolError("refusing to overwrite existing report \(args.out.path)")
    }
    let region = try MoERegionIR.parse(try Data(contentsOf: args.region))
    guard region.guTensor == "blk.0.ffn_gate_up_exps.weight",
          region.dnTensor == "blk.0.ffn_down_exps.weight" else {
        throw ToolError("region must bind the layer-0 expert families (the captured activation is layer 0); got \(region.guTensor) / \(region.dnTensor)")
    }
    let facts = try MoESIMDShapeFacts(region: region)
    let file = try GGUFReader.read(args.gguf)
    guard let cached = try ModelDigestCache.shared.cachedRecord(for: args.gguf) else {
        throw ToolError("no cached model SHA-256 for \(args.gguf.path); refresh it with the separate explicit user command (this tool never hashes the model)")
    }
    let modelSha256 = cached.sha256
    let guInfo = try file.tensor(region.guTensor)
    let dnInfo = try file.tensor(region.dnTensor)
    let q4 = GGMLTensorType.q4_0.rawValue
    guard guInfo.ggmlType == q4, dnInfo.ggmlType == q4,
          guInfo.dims.count == 3, dnInfo.dims.count == 3,
          guInfo.dims[0] == facts.hidden, guInfo.dims[1] == facts.gateUpRows,
          dnInfo.dims[0] == facts.inter, dnInfo.dims[1] == facts.hidden,
          guInfo.dims[2] == dnInfo.dims[2], guInfo.dims[2] >= facts.topK else {
        throw ToolError("GGUF expert families do not match the region geometry: \(guInfo.dims) / \(dnInfo.dims)")
    }
    let experts = guInfo.dims[2]

    print("capturing the layer-0 routed-MoE input (direct reference forward, diagnostic only)")
    let captured = try captureLayer0MoEInput(gguf: args.gguf, hidden: facts.hidden)

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ToolError("Metal unavailable")
    }
    let hardware = try MoESIMDHardwareProfile.probe(device: device)
    let derivation = try MoESIMDScheduleDeriver.derive(region: region, hardware: hardware)
    print("derived \(derivation.schedules.count) schedules; \(derivation.rejections.count) rejections recorded")

    let ids = try MoEComponentInputs.expertIds(count: region.topK, experts: experts)
    let guBlocks = try MoEComponentInputs.expertBlocks(
        file: file, tensor: region.guTensor, ids: ids,
        inDim: facts.hidden, outDim: facts.gateUpRows)
    let dnBlocks = try MoEComponentInputs.expertBlocks(
        file: file, tensor: region.dnTensor, ids: ids,
        inDim: facts.inter, outDim: facts.hidden)

    let uniform = try MoEComponentInputs.activation(region: region)
    let alternating: [Float] = (0..<facts.hidden).map { (i: Int) -> Float in
        let magnitude = 0.25 + 0.5 * Float(i % 7) / 7
        return i % 2 == 0 ? magnitude : -magnitude
    }
    var oneHot = [Float](repeating: 0, count: facts.hidden)
    let hotIndex = 1337 % facts.hidden
    oneHot[hotIndex] = 1
    let inputs = [
        MoESIMDNamedInput(name: "seeded-uniform", values: uniform),
        MoESIMDNamedInput(name: "signed-alternating", values: alternating),
        MoESIMDNamedInput(name: "one-hot", values: oneHot),
        MoESIMDNamedInput(name: "captured-layer0-pre_ffw_norm_2", values: captured),
    ]
    let records = [
        MoESIMDInputRecord(name: "seeded-uniform", elementCount: facts.hidden,
                           provenance: "MoEComponentInputs.activation default seed"),
        MoESIMDInputRecord(name: "signed-alternating", elementCount: facts.hidden,
                           provenance: "(+/-)(0.25 + 0.5 * (i mod 7) / 7), sign alternating by index"),
        MoESIMDInputRecord(name: "one-hot", elementCount: facts.hidden,
                           provenance: "1 at index \(hotIndex), else 0"),
        MoESIMDInputRecord(name: "captured-layer0-pre_ffw_norm_2", elementCount: facts.hidden,
                           provenance: "moeInputProbe layer 0 position 1 of raw-Hello [2, 9259], direct reference forward; values not exported"),
    ]

    print("computing one independent FP64 reference per input")
    var references = [MoESIMDReference]()
    for input in inputs {
        references.append(try MoESIMDOracle.reference(
            facts: facts, guBlocks: guBlocks, dnBlocks: dnBlocks, x: input.values))
    }
    let bench = try MoESIMDComponentBench(device: device, facts: facts,
                                          guBlocks: guBlocks, dnBlocks: dnBlocks)
    let results = try MoESIMDQualification.screen(
        derivation: derivation, bench: bench, inputs: inputs,
        references: references, log: { print($0) })
    let incumbent = MoESIMDQualification.incumbentV1(
        region: region, device: device, guBlocks: guBlocks, dnBlocks: dnBlocks,
        inputs: inputs, references: references)
    for r in incumbent {
        let verdict = r.withinContract ? "within" : "OUTSIDE"
        let detail = r.failure ?? ""
        print("incumbent \(r.schedule) [\(r.input)]: \(verdict) the full FP64 contract \(detail)")
    }
    let report = try MoESIMDComponentReport.make(
        region: region, hardware: hardware, derivation: derivation,
        results: results, incumbent: incumbent, inputs: records, expertIds: ids,
        modelSha256: modelSha256, executableDigest: try Gemma4Runner.executableDigest())
    let data = try report.encoded()
    try data.write(to: args.out, options: [.withoutOverwriting])
    print("wrote NON-INSTALLABLE component report \(args.out.path): \(report.acceptedCount)/\(report.catalogSize) schedules component-accepted; diagnostic GPU intervals only, no winner selected")
}

do {
    try runQualification()
} catch {
    FileHandle.standardError.write(Data("moe-simd-qualify: \(error)\n".utf8))
    exit(1)
}
