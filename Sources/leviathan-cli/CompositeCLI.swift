import Foundation
import LeviathanCore

// Shared CLI helpers for the composite Q4 launch-geometry + MoE lowering
// integration used by gemma-forge and gemma-generate. Pure option
// parsing, bounded file reading, and canonical fixture construction —
// no GPU allocation, no numerics, no selection logic (all heavyweight
// work lives in GemmaCompositeForge.prepare).

/// Paired optional composite flags: --composite-plan PATH and
/// --moe-region PATH. Both or neither — a lone flag is rejected.
/// Returns nil when neither is present (existing no-composite behavior).
struct CompositeCLIOptions {
    let compositePlanPath: String
    let moeRegionPath: String
}

func parseCompositeOptions(_ args: [String]) -> CompositeCLIOptions? {
    let hasPlan = args.contains("--composite-plan")
    let hasRegion = args.contains("--moe-region")
    guard hasPlan == hasRegion else {
        fail("--composite-plan and --moe-region must be passed together (both or neither)")
    }
    guard hasPlan else { return nil }
    return CompositeCLIOptions(
        compositePlanPath: requireValue("--composite-plan", args),
        moeRegionPath: requireValue("--moe-region", args))
}

/// Bounded MoE region read: streamed in 64 KiB chunks, at most 8 MiB,
/// non-empty. Returns the raw JSON bytes for MoERegionIR.parse and
/// GemmaCompositeForge.prepare.
func readMoERegionBounded(_ path: String) throws -> Data {
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? handle.close() }
    var data = Data()
    while data.count <= 8 << 20 {
        let chunk = try handle.read(upToCount: 1 << 16) ?? Data()
        if chunk.isEmpty { break }
        data.append(chunk)
    }
    guard !data.isEmpty else { fail("region file is empty: \(path)") }
    guard data.count <= 8 << 20 else { fail("region file > 8 MiB: \(path)") }
    return data
}

/// Canonical composite fixtures — EXACTLY the existing q4-lower
/// launch-select production defaults: en-chat and kr-chat through the
/// same user→model chat template (derived from the GGUF's embedded
/// template), plus the raw Hello fixture. Fixture tokens always come
/// from the ACTUAL GGUF tokenizer; no text is modified.
func canonicalCompositeFixtures(tokenizer: GemmaTokenizer) throws
        -> [Q4LoweringForge.Q4Fixture] {
    func chatIds(_ text: String) throws -> [Int] {
        try tokenizer.encode(
            "<|turn>user\n" + text.trimmingCharacters(in: .whitespacesAndNewlines)
            + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    }
    return [
        Q4LoweringForge.Q4Fixture(name: "en-chat",
                                  tokens: try chatIds("Hello, can you help me today?")),
        Q4LoweringForge.Q4Fixture(name: "kr-chat",
                                  tokens: try chatIds("안녕하세요, 오늘 날씨는 어떤가요?")),
        Q4LoweringForge.Q4Fixture(name: "raw-hello",
                                  tokens: try tokenizer.encode("Hello")),
    ]
}
