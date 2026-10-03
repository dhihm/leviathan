// ATTENTION candidate qualification and diagnostic bounded attention Forge
// selection (DIAGNOSTIC ONLY; never production).
//
// One candidate: AttentionDiagnostic.kernelName (g4_attn_par64_diag), a
// bounded-parallel single-query attention kernel with one 64-lane
// threadgroup per query head.
//
// Gate order. Any failure rejects the candidate, and later stages never run.
//   1. Component (--component, and at the start of every gate-running mode):
//      - Capacity guard: T = 4097 must be host-ineligible, and the kernel
//        must write NaN.
//      - Synthetic local and global cases, compared against the runner's
//        ACTUAL cached incumbent g4_attn and an independent FP64 oracle.
//      - Pass needs: finite outputs, deterministic candidate, and every
//        relative error < 1e-4. Bit identity is reported, not required.
//   2. Full model (--full, --select, --verify-artifact):
//      - Both arms: hard-pinned E4 plus the Q4 reference width 64.
//      - Comparison over 3 fixtures x 32 steps x all vocab logits: finite,
//        first-argmax exact, identical IDs, relative error < 1e-4.
//      - Counter evidence: applied > 0, fallback 0, attempts == applied, and
//        no counting outside the wrapper.
//      - A zero canary shows the diagnostic slot is reached (evidence only),
//        and a re-run proves the restore.
//   3. Timing (only after 1 and 2):
//      - An untimed warm-up, then 3 paired AB/BA/AB warm-Hello runs of
//        31 decode steps.
//      - Generated IDs must match strictly; whole-decode wall seconds are
//        recorded.
//      - Policy: candidate <= 0.98 x reference in every pair.
//
// Diagnostic bounded attention Forge workflow. The scope is fixed: the
// existing Gemma4 GGUF, the existing captured SSA graph and MoE region IR,
// and this single built-in attention candidate. It is NOT a general model
// compiler, does not capture new PyTorch modules, and never changes a
// serving default.
//
//   --select
//     Refuses an existing --artifact path, then runs gates 1-3 and collects
//     typed runtime evidence. Only if every gate and policy passes does it
//     write ONE bounded JSON artifact, via a temporary file and an exclusive
//     hard link, so the artifact is never overwritten and never partially
//     written. A failed selection writes nothing.
//
//   Artifact layout
//     The envelope holds payload, digest, digestAlgorithm and digestScope.
//     digest is the SHA-256 of the canonical sorted-keys JSON encoding of
//     the payload. The whole file must equal its canonical re-encoding byte
//     for byte, so unknown keys, reordering and whitespace edits are
//     rejected.
//     The payload records:
//       - the policy version, tolerances, fixtures, steps and pairs;
//       - the component matrix, fixture quality, counters and dispatch
//         evidence;
//       - raw paired walls and UTC timestamps;
//       - SHA-256 identities of the executable bytes, the GGUF (streamed in
//         4 MiB chunks, never loaded whole), the region JSON, the graph
//         JSON, the candidate source, the incumbent shader source, the E4
//         baseline MoE source, the engine identity string and the canonical
//         hardware profile JSON;
//       - OS and architecture facts;
//       - productionQualified=false.
//     Files use fixed ids, never paths. An artifact that contains the local
//     user name, the home path or an input path is refused.
//
//   --validate-artifact
//     Bounds, canonical form, digest, schema, policy, evidence and full
//     identity checks only. The runner is initialized for the engine,
//     hardware and geometry identity, but the candidate is never compiled
//     and no gates run.
//
//   --verify-artifact
//     Runs the same validation first and prints identity_validation=passed
//     before any candidate compile or GPU candidate use. It then recompiles
//     the candidate on this process's device and reruns gates 1-3 under the
//     same policy. The fresh result must pass, and the component geometry,
//     prompt token counts and raw_hello greedy IDs must equal the artifact.
//
//   The digest is content integrity only, NOT authenticity. The JSON is
//   user-editable and is never an independent authorization. Nothing is
//   installed, and no production default or production artifact changes.
//
// Usage:
//   attnqual --component --gguf PATH
//   attnqual --full --gguf PATH --region PATH --graph PATH
//   attnqual --select --gguf PATH --region PATH --graph PATH --artifact NEW_PATH
//   attnqual --validate-artifact --gguf PATH --region PATH --graph PATH --artifact PATH
//   attnqual --verify-artifact --gguf PATH --region PATH --graph PATH --artifact PATH
// Exit codes:
//   0 = all requested gates or validations passed
//   1 = a gate failed (candidate rejected; --select writes no artifact)
//   2 = usage or environment error (including a refused existing artifact path)
//   3 = artifact rejected (bounds, canonical form, digest, schema, policy,
//       evidence, privacy or identity)

import Foundation
import Metal
import CryptoKit
import LeviathanCore

enum ToolError: Error {
    case env(String)
    case gate(String)
    case artifact(String)
}

enum Tool {
    static let steps = 32
    static let pairs = 3
    static let policyFactor = 0.98
    static let requiredVocab = 262144
    static let relBound = 1e-4
    static let q4ReferenceWidth = 64
    static let referenceArmName = "E4 + Q4 width 64 + incumbent g4_attn"
    static let candidateArmName = "E4 + Q4 width 64 + ATTENTION g4_attn_par64_diag"
    static let label = "diagnostic bounded attention Forge"
    static let schema = "leviathan.diagnostic-attention-forge-selection.v1"
    static let policyVersion = "attention-forge-policy-1"
    static let componentCases = 18
    static let fixtureNames = ["en_chat", "kr_chat", "raw_hello"]
    static let pairOrders = ["AB", "BA", "AB"]
    static let compileOptions = "nil (defaults)"
    static let moeSchedule = "E4_R32"
    static let maxArtifactBytes = 1 << 20
    static let maxProfileJSONBytes = 1 << 16
    static let hashChunkBytes = 4 << 20
    static let digestAlgorithm = "sha256"
    static let digestScope = "SHA-256 of the canonical sorted-keys JSON encoding of payload; content integrity only, not authenticity"
    static let limitations = [
        "Fixed workload only: the existing Gemma4 Q4_0 GGUF, the existing captured SSA graph and MoE region IR, and the single built-in attention candidate g4_attn_par64_diag. This is not a general PyTorch module compiler and no new model capture is implemented.",
        "Quality evidence covers 3 fixed fixtures x 32 greedy steps with short histories; long histories, other prompts, other models and other hardware are not qualified.",
        "Timing evidence is 3 paired AB/BA/AB warm raw_hello 31-step decode walls in one process; it is not a throughput benchmark, not a comparison with any other engine, and makes no occupancy claim.",
        "The SHA-256 digest is content integrity only, not authenticity; this user-editable JSON is never an independent authorization.",
        "productionQualified is always false; the artifact is never installed and changes no production default or production artifact.",
        "Compile options are the fixed nil (defaults) used by AttentionDiagnostic.compile, recorded as a constant rather than introspected from Metal.",
    ]
    static var kernel: String { return String(describing: AttentionDiagnostic.kernelName) }
}

typealias Prefill = (Gemma4Runner, [Int]) throws -> (Int, [Float])
typealias Step = (Gemma4Runner, Int, Int) throws -> (Int, [Float])

func num(_ v: Double) -> Any {
    return v.isFinite ? v : "nonfinite"
}

func emit(_ obj: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
       let s = String(data: d, encoding: .utf8) {
        print(s)
    }
}

func median(_ v: [Double]) -> Double {
    guard !v.isEmpty else { return Double.nan }
    let s = v.sorted()
    return s[s.count / 2]
}

func sameBits(_ a: [Float], _ b: [Float]) -> Bool {
    guard a.count == b.count else { return false }
    for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern { return false }
    return true
}

func argmaxFirst(_ v: [Float]) -> Int {
    guard !v.isEmpty else { return -1 }
    var best = 0
    var i = 1
    while i < v.count {
        if v[i] > v[best] { best = i }
        i += 1
    }
    return best
}

// MARK: - Artifact schema (strict; every field is required unless optional)

struct FileIdentityRecord: Codable, Equatable {
    let id: String
    let sha256: String
    let bytes: Int
}

struct FilesRecord: Codable, Equatable {
    let executable: FileIdentityRecord
    let gguf: FileIdentityRecord
    let region: FileIdentityRecord
    let graph: FileIdentityRecord
}

struct SourcesRecord: Codable, Equatable {
    let candidateKernel: String
    let candidateSourceSha256: String
    let candidateSourceBytes: Int
    let incumbentShaderSourceSha256: String
    let incumbentShaderSourceBytes: Int
    let compileOptions: String
    let moeBaselineSourceSha256: String
    let moeSchedule: String
    let q4ReferenceWidth: Int
}

struct OSRecord: Codable, Equatable {
    let operatingSystemVersion: String
    let operatingSystemVersionString: String
    let architecture: String
    let physicalMemoryBytes: UInt64
    let processorCount: Int
}

struct GeometryRecord: Codable, Equatable {
    let qHeads: Int
    let slidingWindow: Int
    let vocab: Int
    let maxContext: Int
    let threadsPerHead: Int
}

struct RuntimeIdentityRecord: Codable, Equatable {
    let engineIdentitySha256: String
    let hardwareProfileSha256: String
    let hardwareProfileCanonicalJSON: String
    let geometry: GeometryRecord
}

struct IdentityRecord: Codable, Equatable {
    let files: FilesRecord
    let sources: SourcesRecord
    let os: OSRecord
    let runtime: RuntimeIdentityRecord
}

struct PolicyRecord: Codable, Equatable {
    let version: String
    let relativeBound: Double
    let fixtureRelativeDefinition: String
    let componentRelativeDefinition: String
    let policyFactor: Double
    let steps: Int
    let decodeSteps: Int
    let pairs: Int
    let pairOrders: [String]
    let requiredVocab: Int
    let fixtures: [String]
    let componentCases: Int
    let q4ReferenceWidth: Int
    let moeSchedule: String
    let timingWorkload: String

    static var current: PolicyRecord {
        return PolicyRecord(
            version: Tool.policyVersion, relativeBound: Tool.relBound,
            fixtureRelativeDefinition: "max over all logits of |candidate - reference| / max(1, |reference|)",
            componentRelativeDefinition: "max |candidate - baseline| / max(1, max |baseline|)",
            policyFactor: Tool.policyFactor, steps: Tool.steps, decodeSteps: Tool.steps - 1,
            pairs: Tool.pairs, pairOrders: Tool.pairOrders, requiredVocab: Tool.requiredVocab,
            fixtures: Tool.fixtureNames, componentCases: Tool.componentCases,
            q4ReferenceWidth: Tool.q4ReferenceWidth, moeSchedule: Tool.moeSchedule,
            timingWorkload: "untimed warm-up of both arms, then 3 paired warm raw_hello whole-decode walls of 31 steps; each pair candidate <= 0.98 x reference")
    }
}

struct SelectionRecord: Codable, Equatable {
    let candidateKernel: String
    let candidateKind: String
    let decision: String
    let referenceArm: String
    let candidateArm: String
    let scope: String

    static var current: SelectionRecord {
        return SelectionRecord(
            candidateKernel: Tool.kernel, candidateKind: "ATTENTION",
            decision: "selected_for_diagnostic_experiment_only",
            referenceArm: Tool.referenceArmName, candidateArm: Tool.candidateArmName,
            scope: "fixed Gemma4 GGUF, existing captured SSA graph and MoE region IR, single built-in attention candidate")
    }
}

struct ComponentCaseEvidence: Codable, Equatable {
    let label: String
    let tokens: Int
    let queryPosition: Int
    let attended: Int
    let headDim: Int
    let qHeads: Int
    let kvHeads: Int
    let window: Int
    let bitIdenticalToActualIncumbent: Bool
    let maxAbsCandidateVsIncumbent: Double
    let relCandidateVsIncumbent: Double
    let maxAbsCandidateVsFP64: Double
    let relCandidateVsFP64: Double
    let maxAbsIncumbentVsFP64: Double
    let relIncumbentVsFP64: Double
    let allFinite: Bool
    let candidateDeterministic: Bool
    let incumbentGpuMicros: Double?
    let candidateGpuMicros: Double?
    let pass: Bool
}

struct ComponentEvidence: Codable, Equatable {
    let candidateKernel: String
    let candidateSourceSha256: String
    let threadgroupMemoryBytes: Int
    let candidateMaxThreads: Int
    let threadsPerHead: Int
    let maxContext: Int
    let capacityGuardTokens: Int
    let capacityGuardHostIneligible: Bool
    let capacityGuardKernelWroteNaN: Bool
    let expectedCases: Int
    let cases: [ComponentCaseEvidence]
    let pass: Bool
}

struct FixtureEvidence: Codable, Equatable {
    let fixture: String
    let promptTokens: Int
    let steps: Int
    let vocab: Int
    let shapeOK: Bool
    let nonFiniteLogits: Int
    let greedyMismatches: Int
    let generatedIdMismatches: Int
    let logitBitMismatches: Int
    let worstAbsolute: Double
    let worstRelative: Double
    let attempts: Int
    let applied: Int
    let fallback: Int
    let countersPass: Bool
    let pass: Bool
}

struct DispatchEvidence: Codable, Equatable {
    let noLeakCountersUnchanged: Bool
    let noLeakSlotEmpty: Bool
    let canaryChangedLogits: Bool
    let canaryAttempts: Int
    let canaryApplied: Int
    let canaryFallback: Int
    let referenceRestoredBitIdentical: Bool
    let slotEmptyAfterRestore: Bool
    let pass: Bool
}

struct PairEvidence: Codable, Equatable {
    let pair: Int
    let order: String
    let referenceDecodeWallSeconds: Double
    let candidateDecodeWallSeconds: Double
    let reductionPercent: Double
    let candidateAttempts: Int
    let candidateApplied: Int
    let candidateFallback: Int
    let referenceIdsMatchGated: Bool
    let candidateIdsMatchGated: Bool
    let policyMet: Bool
}

struct TimingEvidence: Codable, Equatable {
    let decodeSteps: Int
    let warmupUntimed: Bool
    let pairs: [PairEvidence]
    let medianReductionPercent: Double
    let policyMet: Bool
}

struct FullEvidence: Codable, Equatable {
    let fixtures: [FixtureEvidence]
    let dispatch: DispatchEvidence
    let timing: TimingEvidence
    let rawHelloReferenceIds: [Int]
}

struct TimestampRecord: Codable, Equatable {
    let selectionStartedUTC: String
    let selectionFinishedUTC: String
}

struct ArtifactPayload: Codable, Equatable {
    let schema: String
    let label: String
    let policy: PolicyRecord
    let selection: SelectionRecord
    let identity: IdentityRecord
    let component: ComponentEvidence
    let full: FullEvidence
    let timestamps: TimestampRecord
    let productionQualified: Bool
    let limitations: [String]
}

struct ArtifactEnvelope: Codable, Equatable {
    let payload: ArtifactPayload
    let digest: String
    let digestAlgorithm: String
    let digestScope: String
}

// MARK: - Hashing and identity

func hexString<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
    return bytes.map { String(format: "%02x", $0) }.joined()
}

func sha256Hex(_ data: Data) -> String {
    return hexString(SHA256.hash(data: data))
}

func isSha256Hex(_ s: String) -> Bool {
    let u = Array(s.utf8)
    return u.count == 64 && u.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
}

func finiteNonNegative(_ v: Double) -> Bool {
    return v.isFinite && v >= 0
}

func finiteOrNil(_ v: Double) -> Double? {
    return v.isFinite ? v : nil
}

func anyInt(_ v: Any?) -> Int {
    guard let v = v else { return -1 }
    if let i = v as? Int { return i }
    if let i = v as? Int32 { return Int(i) }
    if let u = v as? UInt32 { return Int(u) }
    if let u = v as? UInt, u <= UInt(Int.max) { return Int(u) }
    if let u = v as? UInt64, u <= UInt64(Int.max) { return Int(u) }
    return -1
}

func isoNow() -> String {
    return ISO8601DateFormatter().string(from: Date())
}

func canonicalEncoder() -> JSONEncoder {
    let e = JSONEncoder()
    e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return e
}

/// Streams a file through SHA-256 in bounded chunks of at most 4 MiB, so the
/// 14 GB GGUF is never read into memory as a whole.
func streamFileIdentity(id: String, url: URL) throws -> FileIdentityRecord {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    var total = 0
    while true {
        let chunk: Data? = try autoreleasepool { try handle.read(upToCount: Tool.hashChunkBytes) }
        guard let c = chunk, !c.isEmpty else { break }
        hasher.update(data: c)
        total += c.count
    }
    guard total > 0 else { throw ToolError.env("input file \(id) is empty") }
    return FileIdentityRecord(id: id, sha256: hexString(hasher.finalize()), bytes: total)
}

struct Inputs {
    let gguf: String
    let region: String
    let graph: String
    let artifact: String
}

func currentFiles(_ inp: Inputs) throws -> FilesRecord {
    let exe = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
    return try FilesRecord(
        executable: streamFileIdentity(id: "executable", url: exe),
        gguf: streamFileIdentity(id: "gguf", url: URL(fileURLWithPath: inp.gguf)),
        region: streamFileIdentity(id: "region", url: URL(fileURLWithPath: inp.region)),
        graph: streamFileIdentity(id: "graph", url: URL(fileURLWithPath: inp.graph)))
}

func currentSources() -> SourcesRecord {
    let candidate = Data(AttentionDiagnostic.candidateSource.utf8)
    let incumbent = Data(Gemma4Runner.shaderSource.utf8)
    return SourcesRecord(
        candidateKernel: Tool.kernel, candidateSourceSha256: sha256Hex(candidate),
        candidateSourceBytes: candidate.count, incumbentShaderSourceSha256: sha256Hex(incumbent),
        incumbentShaderSourceBytes: incumbent.count, compileOptions: Tool.compileOptions,
        moeBaselineSourceSha256: String(describing: Q4LaunchArtifact.baselineMoeSourceSha256),
        moeSchedule: Tool.moeSchedule, q4ReferenceWidth: Tool.q4ReferenceWidth)
}

func currentOSRecord() -> OSRecord {
    #if arch(arm64)
    let arch = "arm64"
    #elseif arch(x86_64)
    let arch = "x86_64"
    #else
    let arch = "unknown"
    #endif
    let info = ProcessInfo.processInfo
    let v = info.operatingSystemVersion
    return OSRecord(
        operatingSystemVersion: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
        operatingSystemVersionString: info.operatingSystemVersionString,
        architecture: arch, physicalMemoryBytes: info.physicalMemory,
        processorCount: info.processorCount)
}

/// Engine, hardware and geometry identity. Needs the runner (which may
/// compile the baseline) but never compiles the attention candidate.
func currentRuntimeIdentity(_ runner: Gemma4Runner) throws -> RuntimeIdentityRecord {
    let engine = try runner.engineIdentity()
    let profile = try runner.hardwareProfile()
    let profileData = try canonicalEncoder().encode(profile)
    guard profileData.count > 0, profileData.count <= Tool.maxProfileJSONBytes,
          let profileText = String(data: profileData, encoding: .utf8) else {
        throw ToolError.env("hardware profile encoding is empty, too large or not UTF-8")
    }
    let config = AttentionDiagnostic.configSummary(runner: runner)
    let geometry = GeometryRecord(
        qHeads: anyInt(config["qHeads"]), slidingWindow: anyInt(config["slidingWindow"]),
        vocab: runner.config.vocabSize, maxContext: Int(AttentionDiagnostic.maxContext),
        threadsPerHead: Int(AttentionDiagnostic.threads))
    return RuntimeIdentityRecord(
        engineIdentitySha256: sha256Hex(Data(engine.utf8)),
        hardwareProfileSha256: sha256Hex(profileData),
        hardwareProfileCanonicalJSON: profileText, geometry: geometry)
}

func identityMismatches(_ a: IdentityRecord, files: FilesRecord, sources: SourcesRecord, os: OSRecord) -> [String] {
    var m: [String] = []
    if a.files.executable != files.executable { m.append("executable") }
    if a.files.gguf != files.gguf { m.append("gguf") }
    if a.files.region != files.region { m.append("region") }
    if a.files.graph != files.graph { m.append("graph") }
    if a.sources.candidateSourceSha256 != sources.candidateSourceSha256
        || a.sources.candidateSourceBytes != sources.candidateSourceBytes { m.append("candidateSource") }
    if a.sources.incumbentShaderSourceSha256 != sources.incumbentShaderSourceSha256
        || a.sources.incumbentShaderSourceBytes != sources.incumbentShaderSourceBytes { m.append("incumbentShaderSource") }
    if a.sources != sources { m.append("sourcesRecord") }
    if a.os != os { m.append("os") }
    return m
}

func runtimeMismatches(_ a: RuntimeIdentityRecord, _ b: RuntimeIdentityRecord) -> [String] {
    var m: [String] = []
    if a.engineIdentitySha256 != b.engineIdentitySha256 { m.append("engineIdentity") }
    if a.hardwareProfileSha256 != b.hardwareProfileSha256
        || a.hardwareProfileCanonicalJSON != b.hardwareProfileCanonicalJSON { m.append("hardwareProfile") }
    if a.geometry != b.geometry { m.append("geometry") }
    return m
}

// MARK: - Evidence validation (shared by --select before writing and by
// --validate-artifact / --verify-artifact before any candidate use)

func checkComponent(_ c: ComponentEvidence, sourceSha: String) -> [String] {
    var bad: [String] = []
    func req(_ ok: Bool, _ msg: String) { if !ok { bad.append("component: " + msg) } }
    let maxContext = Int(AttentionDiagnostic.maxContext)
    req(c.candidateKernel == Tool.kernel, "unknown candidate kernel")
    req(isSha256Hex(c.candidateSourceSha256) && c.candidateSourceSha256 == sourceSha, "candidate source SHA mismatch")
    req(c.threadsPerHead == Int(AttentionDiagnostic.threads), "threads per head differ")
    req(c.maxContext == maxContext, "context capacity differs")
    req(c.capacityGuardTokens == maxContext + 1 && c.capacityGuardHostIneligible && c.capacityGuardKernelWroteNaN,
        "capacity guard not proven")
    req(c.threadgroupMemoryBytes > 0 && c.threadgroupMemoryBytes <= 32768, "threadgroup memory out of bounds")
    req(c.candidateMaxThreads >= c.threadsPerHead, "pipeline cannot run the threads per head")
    req(c.expectedCases == Tool.componentCases && c.cases.count == Tool.componentCases,
        "component matrix is not exactly \(Tool.componentCases) cases")
    req(Set(c.cases.map { $0.label }).count == c.cases.count, "duplicate component labels")
    for k in c.cases {
        let contextOK = !k.label.isEmpty && k.tokens > 0 && k.tokens <= maxContext
            && k.attended > 0 && k.attended <= k.tokens && k.queryPosition >= 0 && k.window >= 0
        let headsOK = k.headDim > 0 && k.qHeads > 0 && k.kvHeads > 0 && k.qHeads % k.kvHeads == 0
        req(contextOK && headsOK, "case \(k.label) has unknown head or context geometry")
        let metrics = [k.maxAbsCandidateVsIncumbent, k.relCandidateVsIncumbent, k.maxAbsCandidateVsFP64,
                       k.relCandidateVsFP64, k.maxAbsIncumbentVsFP64, k.relIncumbentVsFP64]
        req(metrics.allSatisfy(finiteNonNegative), "case \(k.label) has nonfinite or negative error metrics")
        req(k.relCandidateVsIncumbent < Tool.relBound && k.relCandidateVsFP64 < Tool.relBound,
            "case \(k.label) exceeds the relative bound")
        req(k.allFinite && k.candidateDeterministic && k.pass, "case \(k.label) did not pass")
        if let g = k.incumbentGpuMicros { req(finiteNonNegative(g), "case \(k.label) incumbent GPU time invalid") }
        if let g = k.candidateGpuMicros { req(finiteNonNegative(g), "case \(k.label) candidate GPU time invalid") }
    }
    req(c.pass, "component stage did not pass")
    return bad
}

func checkFull(_ f: FullEvidence) -> [String] {
    var bad: [String] = []
    func req(_ ok: Bool, _ msg: String) { if !ok { bad.append("full: " + msg) } }
    req(f.fixtures.map { $0.fixture } == Tool.fixtureNames, "fixture coverage is not exactly en_chat, kr_chat, raw_hello")
    for x in f.fixtures {
        let n = x.fixture
        req(x.steps == Tool.steps && x.vocab == Tool.requiredVocab && x.shapeOK, "\(n) shape is not 32 steps x 262144")
        req(x.promptTokens > 0, "\(n) has no prompt tokens")
        req(x.nonFiniteLogits == 0 && x.greedyMismatches == 0 && x.generatedIdMismatches == 0,
            "\(n) has nonfinite logits, greedy or ID mismatches")
        req(x.logitBitMismatches >= 0 && x.logitBitMismatches <= Tool.steps * Tool.requiredVocab,
            "\(n) bit mismatch count out of range")
        req(finiteNonNegative(x.worstAbsolute) && finiteNonNegative(x.worstRelative) && x.worstRelative < Tool.relBound,
            "\(n) error is not finite or not below the bound")
        req(x.applied > 0 && x.fallback == 0 && x.attempts == x.applied && x.countersPass,
            "\(n) counters do not show full application")
        req(x.pass, "\(n) did not pass")
    }
    let d = f.dispatch
    req(d.noLeakCountersUnchanged && d.noLeakSlotEmpty, "no-leak evidence missing")
    req(d.canaryChangedLogits && d.canaryApplied > 0 && d.canaryFallback == 0 && d.canaryAttempts == d.canaryApplied,
        "canary reachability evidence missing")
    req(d.referenceRestoredBitIdentical && d.slotEmptyAfterRestore && d.pass, "restore evidence missing")
    let t = f.timing
    req(t.decodeSteps == Tool.steps - 1 && t.warmupUntimed, "timing workload differs")
    req(t.pairs.count == Tool.pairs && t.pairs.map { $0.order } == Tool.pairOrders, "pairs are not exactly AB, BA, AB")
    var reductions: [Double] = []
    for (i, p) in t.pairs.enumerated() {
        let refOK = p.referenceDecodeWallSeconds.isFinite && p.referenceDecodeWallSeconds > 0
        let candOK = p.candidateDecodeWallSeconds.isFinite && p.candidateDecodeWallSeconds > 0
        req(p.pair == i && refOK && candOK, "pair \(i) walls are not finite and positive")
        let r = (p.referenceDecodeWallSeconds - p.candidateDecodeWallSeconds) / p.referenceDecodeWallSeconds * 100
        reductions.append(r)
        req(p.reductionPercent == r, "pair \(i) reduction does not match its walls")
        req(p.candidateDecodeWallSeconds <= Tool.policyFactor * p.referenceDecodeWallSeconds && p.policyMet,
            "pair \(i) misses the 0.98 policy")
        req(p.candidateApplied > 0 && p.candidateFallback == 0 && p.candidateAttempts == p.candidateApplied,
            "pair \(i) counters do not show full application")
        req(p.referenceIdsMatchGated && p.candidateIdsMatchGated, "pair \(i) IDs differ from the gated reference")
    }
    req(t.medianReductionPercent == median(reductions) && t.policyMet, "timing policy or median mismatch")
    req(f.rawHelloReferenceIds.count == Tool.steps
        && f.rawHelloReferenceIds.allSatisfy { $0 >= 0 && $0 < Tool.requiredVocab }, "raw_hello reference IDs invalid")
    return bad
}

func validatePayload(_ p: ArtifactPayload) -> [String] {
    var bad: [String] = []
    func req(_ ok: Bool, _ msg: String) { if !ok { bad.append(msg) } }
    req(p.schema == Tool.schema, "unknown schema")
    req(p.label == Tool.label, "unknown label")
    req(p.policy == PolicyRecord.current, "unknown or modified policy")
    req(p.selection == SelectionRecord.current, "unknown selection candidate")
    req(p.productionQualified == false && AttentionDiagnostic.productionQualified == false,
        "productionQualified must be false")
    req(p.limitations == Tool.limitations, "limitations text modified")
    let id = p.identity
    let fileList = [id.files.executable, id.files.gguf, id.files.region, id.files.graph]
    req(fileList.map { $0.id } == ["executable", "gguf", "region", "graph"], "unknown file identity ids")
    req(fileList.allSatisfy { isSha256Hex($0.sha256) && $0.bytes > 0 }, "malformed file identity")
    let s = id.sources
    req(s.candidateKernel == Tool.kernel, "unknown candidate kernel")
    req(isSha256Hex(s.candidateSourceSha256) && s.candidateSourceBytes > 0, "malformed candidate source identity")
    req(isSha256Hex(s.incumbentShaderSourceSha256) && s.incumbentShaderSourceBytes > 0,
        "malformed incumbent shader identity")
    req(s.compileOptions == Tool.compileOptions, "unknown compile options")
    req(s.moeSchedule == Tool.moeSchedule && s.q4ReferenceWidth == Tool.q4ReferenceWidth, "unknown E4 or Q4 configuration")
    req(s.moeBaselineSourceSha256 == String(describing: Q4LaunchArtifact.baselineMoeSourceSha256),
        "E4 baseline source SHA differs from this binary")
    let r = id.runtime
    let g = r.geometry
    let geometryConstantsOK = g.vocab == Tool.requiredVocab && g.maxContext == Int(AttentionDiagnostic.maxContext)
        && g.threadsPerHead == Int(AttentionDiagnostic.threads)
    req(geometryConstantsOK && g.qHeads > 0 && g.slidingWindow > 0, "unknown head or context geometry")
    req(isSha256Hex(r.engineIdentitySha256), "malformed engine identity")
    let profileBytes = Data(r.hardwareProfileCanonicalJSON.utf8)
    req(profileBytes.count > 0 && profileBytes.count <= Tool.maxProfileJSONBytes
        && sha256Hex(profileBytes) == r.hardwareProfileSha256, "hardware profile JSON does not match its SHA-256")
    req(!id.os.architecture.isEmpty && id.os.physicalMemoryBytes > 0 && id.os.processorCount > 0, "malformed OS facts")
    bad += checkComponent(p.component, sourceSha: s.candidateSourceSha256)
    req(p.component.threadsPerHead == g.threadsPerHead && p.component.maxContext == g.maxContext,
        "component geometry disagrees with identity geometry")
    bad += checkFull(p.full)
    let formatter = ISO8601DateFormatter()
    if let a = formatter.date(from: p.timestamps.selectionStartedUTC),
       let b = formatter.date(from: p.timestamps.selectionFinishedUTC) {
        req(b >= a, "timestamps out of order")
    } else {
        bad.append("timestamps are not ISO 8601 UTC")
    }
    return bad
}

// MARK: - Artifact encoding and bounded IO

func encodeEnvelope(_ payload: ArtifactPayload) throws -> (Data, String) {
    let enc = canonicalEncoder()
    let digest = sha256Hex(try enc.encode(payload))
    let envelope = ArtifactEnvelope(payload: payload, digest: digest,
                                    digestAlgorithm: Tool.digestAlgorithm, digestScope: Tool.digestScope)
    return (try enc.encode(envelope), digest)
}

/// Strict parse: the bytes must decode to the schema AND be byte-identical to
/// their canonical re-encoding (this rejects unknown keys, reordering and
/// whitespace edits). Then the payload digest must match.
func parseArtifact(_ data: Data) throws -> ArtifactEnvelope {
    guard data.count > 0, data.count <= Tool.maxArtifactBytes else {
        throw ToolError.artifact("artifact size out of bounds")
    }
    let envelope: ArtifactEnvelope
    do {
        envelope = try JSONDecoder().decode(ArtifactEnvelope.self, from: data)
    } catch {
        throw ToolError.artifact("artifact does not match the strict schema")
    }
    let enc = canonicalEncoder()
    let canonical: Data
    let payloadBytes: Data
    do {
        canonical = try enc.encode(envelope)
        payloadBytes = try enc.encode(envelope.payload)
    } catch {
        throw ToolError.artifact("artifact cannot be re-encoded canonically")
    }
    guard canonical == data else {
        throw ToolError.artifact("artifact is not canonical: unknown keys, reordering or edits")
    }
    guard envelope.digestAlgorithm == Tool.digestAlgorithm, envelope.digestScope == Tool.digestScope else {
        throw ToolError.artifact("unknown digest algorithm or scope")
    }
    guard isSha256Hex(envelope.digest), envelope.digest == sha256Hex(payloadBytes) else {
        throw ToolError.artifact("payload digest mismatch")
    }
    return envelope
}

func privacyViolations(_ data: Data, paths: [String]) -> [String] {
    let text = String(decoding: data, as: UTF8.self)
    var hits: [String] = []
    let user = NSUserName()
    if user.count >= 3 && text.contains(user) { hits.append("contains the local user name") }
    let home = NSHomeDirectory()
    if home.count > 1 && text.contains(home) { hits.append("contains the home directory path") }
    if text.contains("/Users/") { hits.append("contains a /Users/ path") }
    for p in paths where p.contains("/") && text.contains(p) { hits.append("contains an input path") }
    return hits
}

func pathExists(_ path: String) -> Bool {
    var st = stat()
    return lstat(path, &st) == 0
}

func readBoundedArtifact(_ path: String) throws -> Data {
    var st = stat()
    guard lstat(path, &st) == 0 else { throw ToolError.artifact("artifact file not found") }
    guard (Int(st.st_mode) & 0o170000) == 0o100000 else {
        throw ToolError.artifact("artifact is not a regular file")
    }
    let size = Int(st.st_size)
    guard size > 0, size <= Tool.maxArtifactBytes else { throw ToolError.artifact("artifact size out of bounds") }
    let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? handle.close() }
    let data = try handle.read(upToCount: Tool.maxArtifactBytes + 1) ?? Data()
    guard data.count == size else { throw ToolError.artifact("artifact size changed while reading") }
    return data
}

/// Writes a temporary file in the destination directory, syncs it, then
/// publishes it with link(2), which fails instead of replacing an existing
/// path. The temporary name is always removed.
func writeArtifactExclusive(_ data: Data, to path: String) throws {
    let dest = URL(fileURLWithPath: path)
    let dir = dest.deletingLastPathComponent()
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
        throw ToolError.env("artifact parent directory does not exist")
    }
    let tmp = dir.appendingPathComponent(".attnqual-" + UUID().uuidString + ".tmp")
    try data.write(to: tmp, options: [.withoutOverwriting])
    defer { _ = unlink(tmp.path) }
    let fh = try FileHandle(forWritingTo: tmp)
    try fh.synchronize()
    try fh.close()
    guard link(tmp.path, dest.path) == 0 else {
        throw ToolError.env("artifact path appeared or cannot be linked (errno \(errno)); nothing was overwritten")
    }
}

// MARK: - Component records

func recordDict(_ r: AttentionComponentRecord) -> [String: Any] {
    return [
        "stage": "attention_component",
        "candidate": AttentionDiagnostic.kernelName,
        "label": r.label,
        "tokens": r.tokens,
        "queryPosition": r.queryPosition,
        "attended": r.attended,
        "headDim": r.headDim,
        "qHeads": r.qHeads,
        "kvHeads": r.kvHeads,
        "window": r.window,
        "bitIdenticalToActualIncumbent": r.bitIdenticalToReference,
        "maxAbsCandidateVsIncumbent": num(r.maxAbsCandidateVsReference),
        "relCandidateVsIncumbent": num(r.relCandidateVsReference),
        "maxAbsCandidateVsFP64": num(r.maxAbsCandidateVsFP64),
        "relCandidateVsFP64": num(r.relCandidateVsFP64),
        "maxAbsIncumbentVsFP64": num(r.maxAbsReferenceVsFP64),
        "relIncumbentVsFP64": num(r.relReferenceVsFP64),
        "maxAbsIncumbentOutput": num(r.maxAbsReferenceOutput),
        "allFinite": r.allFinite,
        "candidateDeterministic": r.deterministic,
        "incumbentGpuMicros": num(r.referenceGpuMicros),
        "candidateGpuMicros": num(r.candidateGpuMicros),
        "relativeDenominator": "max(1, max |baseline|)",
        "pass": r.pass,
    ]
}

func caseEvidence(_ r: AttentionComponentRecord) -> ComponentCaseEvidence {
    return ComponentCaseEvidence(
        label: String(describing: r.label), tokens: anyInt(r.tokens), queryPosition: anyInt(r.queryPosition),
        attended: anyInt(r.attended), headDim: anyInt(r.headDim), qHeads: anyInt(r.qHeads),
        kvHeads: anyInt(r.kvHeads), window: anyInt(r.window),
        bitIdenticalToActualIncumbent: r.bitIdenticalToReference,
        maxAbsCandidateVsIncumbent: r.maxAbsCandidateVsReference,
        relCandidateVsIncumbent: r.relCandidateVsReference,
        maxAbsCandidateVsFP64: r.maxAbsCandidateVsFP64, relCandidateVsFP64: r.relCandidateVsFP64,
        maxAbsIncumbentVsFP64: r.maxAbsReferenceVsFP64, relIncumbentVsFP64: r.relReferenceVsFP64,
        allFinite: r.allFinite, candidateDeterministic: r.deterministic,
        incumbentGpuMicros: finiteOrNil(r.referenceGpuMicros),
        candidateGpuMicros: finiteOrNil(r.candidateGpuMicros), pass: r.pass)
}

// MARK: - SSA adapter shims
//
// The supported adapter closure shapes are selected at compile time by
// overload resolution. Results are read by reflection as
// (token Int, logits [Float]), so tuple labels do not matter.

func stepResult(_ value: Any) throws -> (Int, [Float]) {
    var token: Int? = nil
    var logits: [Float]? = nil
    for child in Mirror(reflecting: value).children {
        if token == nil, let t = child.value as? Int {
            token = t
        } else if logits == nil, let l = child.value as? [Float] {
            logits = l
        }
    }
    guard let t = token, let l = logits else {
        throw ToolError.env("SSA adapter result is not (token Int, logits [Float])")
    }
    return (t, l)
}

func callMany<R>(_ f: (Gemma4Runner, [Int]) throws -> R, _ r: Gemma4Runner, _ t: [Int]) throws -> (Int, [Float]) {
    let v = try f(r, t)
    return try stepResult(v)
}

func callMany<R>(_ f: ([Int]) throws -> R, _ r: Gemma4Runner, _ t: [Int]) throws -> (Int, [Float]) {
    let v = try f(t)
    return try stepResult(v)
}

func callOne<R>(_ f: (Gemma4Runner, Int) throws -> R, _ r: Gemma4Runner, _ t: Int, _ p: Int) throws -> (Int, [Float]) {
    let v = try f(r, t)
    return try stepResult(v)
}

func callOne<R>(_ f: (Int) throws -> R, _ r: Gemma4Runner, _ t: Int, _ p: Int) throws -> (Int, [Float]) {
    let v = try f(t)
    return try stepResult(v)
}

func callOne<R>(_ f: (Gemma4Runner, Int, Int) throws -> R, _ r: Gemma4Runner, _ t: Int, _ p: Int) throws -> (Int, [Float]) {
    let v = try f(r, t, p)
    return try stepResult(v)
}

func callOne<R>(_ f: (Int, Int) throws -> R, _ r: Gemma4Runner, _ t: Int, _ p: Int) throws -> (Int, [Float]) {
    let v = try f(t, p)
    return try stepResult(v)
}

// MARK: - Arms

/// Reference arm: actual incumbent attention under the qualified Q4 width 64.
func referenceArm<T>(_ runner: Gemma4Runner, _ body: () throws -> T) throws -> T {
    return try runner.withQ4DiagnosticReferenceWidth(Tool.q4ReferenceWidth) {
        try body()
    }
}

/// Candidate arm: the same Q4 width 64 plus the attention diagnostic slot.
func candidateArm<T>(_ runner: Gemma4Runner, _ candidate: MTLComputePipelineState,
                     _ body: () throws -> T) throws -> T {
    return try runner.withQ4DiagnosticReferenceWidth(Tool.q4ReferenceWidth) {
        try runner.withAttentionDiagnostic(candidate: candidate) {
            try body()
        }
    }
}

// MARK: - Component gates

struct ComponentOutcome {
    let candidate: MTLComputePipelineState?
    let evidence: ComponentEvidence
}

/// Compiles the single ATTENTION candidate and runs the capacity guard and
/// the synthetic component matrix. The candidate is nil when it is rejected.
/// The evidence is typed and comes from the same runtime records that are
/// printed.
func runComponentGates(runner: Gemma4Runner) throws -> ComponentOutcome {
    let candidate = try AttentionDiagnostic.compile(runner: runner)
    let config = AttentionDiagnostic.configSummary(runner: runner)
    emit(["stage": "attention_candidate", "kernel": AttentionDiagnostic.kernelName,
          "sourceSha256": AttentionDiagnostic.candidateSourceSha256(),
          "threadgroupMemoryBytes": candidate.staticThreadgroupMemoryLength,
          "candidateMaxThreads": candidate.maxTotalThreadsPerThreadgroup,
          "threadsPerHead": AttentionDiagnostic.threads,
          "maxContext": AttentionDiagnostic.maxContext,
          "configQHeads": config["qHeads"] ?? -1,
          "configSlidingWindow": config["slidingWindow"] ?? -1,
          "compileOptions": "nil (defaults)"])
    let guardResult = try AttentionDiagnostic.capacityGuardCase(runner: runner, candidate: candidate)
    let guardOK = guardResult.ineligible && guardResult.wroteNaN
    emit(["stage": "attention_capacity_guard", "tokens": AttentionDiagnostic.maxContext + 1,
          "hostIneligible": guardResult.ineligible, "kernelWroteNaN": guardResult.wroteNaN,
          "pass": guardOK])
    let expected = AttentionDiagnostic.tokenCases.count * 2
    var cases: [ComponentCaseEvidence] = []
    func outcome(_ pass: Bool) -> ComponentOutcome {
        let evidence = ComponentEvidence(
            candidateKernel: Tool.kernel,
            candidateSourceSha256: sha256Hex(Data(AttentionDiagnostic.candidateSource.utf8)),
            threadgroupMemoryBytes: candidate.staticThreadgroupMemoryLength,
            candidateMaxThreads: candidate.maxTotalThreadsPerThreadgroup,
            threadsPerHead: Int(AttentionDiagnostic.threads),
            maxContext: Int(AttentionDiagnostic.maxContext),
            capacityGuardTokens: Int(AttentionDiagnostic.maxContext) + 1,
            capacityGuardHostIneligible: guardResult.ineligible,
            capacityGuardKernelWroteNaN: guardResult.wroteNaN,
            expectedCases: expected, cases: cases, pass: pass)
        return ComponentOutcome(candidate: pass ? candidate : nil, evidence: evidence)
    }
    if !guardOK { return outcome(false) }
    let records = try AttentionDiagnostic.syntheticComponent(runner: runner, candidate: candidate)
    for rec in records {
        emit(recordDict(rec))
        cases.append(caseEvidence(rec))
        if !rec.pass { return outcome(false) }
    }
    guard records.count == expected else {
        throw ToolError.gate("only \(records.count) of \(expected) attention component cases ran")
    }
    return outcome(true)
}

// MARK: - Whole-model gates and paired timing

struct Trace {
    var ids: [Int] = []
    var logits: [[Float]] = []
    var decodeWallSeconds = 0.0
}

/// Runs one fixture from a fresh SSA state: a prefill step, then
/// (steps - 1) single-token decode steps. Only the decode steps are timed.
func decode(_ runner: Gemma4Runner, _ prefill: Prefill, _ step: Step,
            tokens: [Int], steps: Int, keepLogits: Bool) throws -> Trace {
    try LoweringSelect.ssaReset(runner)
    var trace = Trace()
    var (token, logits) = try prefill(runner, tokens)
    trace.ids.append(token)
    if keepLogits { trace.logits.append(logits) }
    let t0 = DispatchTime.now().uptimeNanoseconds
    var s = 1
    while s < steps {
        (token, logits) = try step(runner, token, tokens.count + s - 1)
        trace.ids.append(token)
        if keepLogits { trace.logits.append(logits) }
        s += 1
    }
    trace.decodeWallSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    return trace
}

struct FixtureComparison {
    let shapeOK: Bool
    let nonFinite: Int
    let greedyFail: Int
    let idMismatch: Int
    let bitMismatch: Int
    let worstAbs: Double
    let worstRel: Double
    let pass: Bool
}

func compareFixture(_ name: String, ref: Trace, cand: Trace, vocab: Int, steps: Int) -> FixtureComparison {
    var shapeOK = ref.ids.count == steps && cand.ids.count == steps
        && ref.logits.count == steps && cand.logits.count == steps
    var worstRel = 0.0
    var worstAbs = 0.0
    var bitMismatch = 0
    var nonFinite = 0
    var greedyFail = 0
    var idMismatch = 0
    if shapeOK {
        for s in 0..<steps {
            let r = ref.logits[s]
            let c = cand.logits[s]
            guard r.count == vocab, c.count == vocab else { shapeOK = false; continue }
            for i in 0..<vocab {
                let rv = r[i]
                let cv = c[i]
                if !rv.isFinite || !cv.isFinite { nonFinite += 1; continue }
                if rv.bitPattern != cv.bitPattern { bitMismatch += 1 }
                let diff = abs(Double(cv) - Double(rv))
                let rel = diff / max(1.0, abs(Double(rv)))
                if diff > worstAbs { worstAbs = diff }
                if rel > worstRel { worstRel = rel }
            }
            if argmaxFirst(r) != ref.ids[s] || argmaxFirst(c) != cand.ids[s] { greedyFail += 1 }
            if ref.ids[s] != cand.ids[s] { idMismatch += 1 }
        }
    }
    let ok = shapeOK && nonFinite == 0 && greedyFail == 0 && idMismatch == 0 && worstRel < Tool.relBound
    emit(["stage": "attention_full_fixture", "fixture": name, "steps": steps, "vocab": vocab,
          "shapeOK": shapeOK, "nonFiniteLogits": nonFinite, "greedyMismatches": greedyFail,
          "generatedIdMismatches": idMismatch, "logitBitMismatches": bitMismatch,
          "worstAbsolute": num(worstAbs), "worstRelative": num(worstRel),
          "referenceArm": Tool.referenceArmName, "candidateArm": Tool.candidateArmName, "pass": ok])
    return FixtureComparison(shapeOK: shapeOK, nonFinite: nonFinite, greedyFail: greedyFail,
                             idMismatch: idMismatch, bitMismatch: bitMismatch,
                             worstAbs: worstAbs, worstRel: worstRel, pass: ok)
}

func countersOK(_ c: (attempts: Int, applied: Int, fallback: Int)) -> Bool {
    return c.applied > 0 && c.fallback == 0 && c.attempts == c.applied
}

func chatPrompt(_ text: String) -> String {
    return "<|turn>user\n" + text + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
}

/// Typed result of the whole-model, dispatch and timing gates. pass is true
/// only when every gate and the timing policy passed. stage names the stage
/// that rejected the candidate, or full_and_timing_passed.
struct FullOutcome {
    let pass: Bool
    let stage: String
    let fixtures: [FixtureEvidence]
    let dispatch: DispatchEvidence?
    let timing: TimingEvidence?
    let rawHelloReferenceIds: [Int]
}

func fullEvidence(_ o: FullOutcome) -> FullEvidence? {
    guard o.pass, let d = o.dispatch, let t = o.timing else { return nil }
    return FullEvidence(fixtures: o.fixtures, dispatch: d, timing: t, rawHelloReferenceIds: o.rawHelloReferenceIds)
}

func runFull(runner: Gemma4Runner, candidate: MTLComputePipelineState,
             ggufPath: String, regionPath: String, graphPath: String) throws -> FullOutcome {
    let region = try MoERegionIR.parse(Data(contentsOf: URL(fileURLWithPath: regionPath)))
    let graph = try GemmaGraph.parse(Data(contentsOf: URL(fileURLWithPath: graphPath)))
    let gguf = try GGUFReader.read(URL(fileURLWithPath: ggufPath))
    let tokenizer = try GemmaTokenizer(gguf: gguf)
    let vocab = runner.config.vocabSize
    guard vocab == Tool.requiredVocab else {
        throw ToolError.gate("vocab \(vocab) is not the required \(Tool.requiredVocab)")
    }
    _ = try runner.installMoeBaselineForVerification(
        region: region, schedule: MoESchedule(batch: 4, rowsPerGroup: 32),
        expectedSourceSha256: Q4LaunchArtifact.baselineMoeSourceSha256)
    let adapters = try LoweringSelect.ssaForwardAdapters(graph: graph)
    let prefill: Prefill = { r, t in try callMany(adapters.0, r, t) }
    let step: Step = { r, t, p in try callOne(adapters.1, r, t, p) }
    let steps = Tool.steps
    let krText = "\u{C548}\u{B155}\u{D558}\u{C138}\u{C694}, \u{C624}\u{B298} \u{B0A0}\u{C528}\u{B294} \u{C5B4}\u{B5A4}\u{AC00}\u{C694}?"
    let fixtures: [(String, String)] = [
        ("en_chat", chatPrompt("Hello, can you help me today?")),
        ("kr_chat", chatPrompt(krText)),
        ("raw_hello", "Hello"),
    ]
    var fixtureEvidence: [FixtureEvidence] = []
    func outcome(_ pass: Bool, _ stage: String, dispatch: DispatchEvidence? = nil,
                 timing: TimingEvidence? = nil, ids: [Int] = []) -> FullOutcome {
        return FullOutcome(pass: pass, stage: stage, fixtures: fixtureEvidence, dispatch: dispatch,
                           timing: timing, rawHelloReferenceIds: ids)
    }
    var helloRef: Trace? = nil
    var helloTokens: [Int] = []
    for (name, text) in fixtures {
        let tokens = try tokenizer.encode(text).map { Int($0) }
        guard !tokens.isEmpty else { throw ToolError.env("empty tokenization for \(name)") }
        let ref = try referenceArm(runner) {
            try decode(runner, prefill, step, tokens: tokens, steps: steps, keepLogits: true)
        }
        let cand = try candidateArm(runner, candidate) {
            try decode(runner, prefill, step, tokens: tokens, steps: steps, keepLogits: true)
        }
        let counts = AttentionDiagnostic.counters.snapshot()
        let cOK = countersOK(counts)
        emit(["stage": "attention_counters", "fixture": name, "attempts": counts.attempts,
              "applied": counts.applied, "fallback": counts.fallback, "pass": cOK])
        if !cOK { return outcome(false, "fixture_counters_rejected") }
        let cmp = compareFixture(name, ref: ref, cand: cand, vocab: vocab, steps: steps)
        fixtureEvidence.append(FixtureEvidence(
            fixture: name, promptTokens: tokens.count, steps: steps, vocab: vocab, shapeOK: cmp.shapeOK,
            nonFiniteLogits: cmp.nonFinite, greedyMismatches: cmp.greedyFail,
            generatedIdMismatches: cmp.idMismatch, logitBitMismatches: cmp.bitMismatch,
            worstAbsolute: cmp.worstAbs, worstRelative: cmp.worstRel, attempts: counts.attempts,
            applied: counts.applied, fallback: counts.fallback, countersPass: cOK, pass: cmp.pass))
        if !cmp.pass { return outcome(false, "fixture_quality_rejected") }
        if name == "raw_hello" {
            helloRef = ref
            helloTokens = tokens
        }
    }
    guard let hello = helloRef, hello.logits.count == steps else {
        throw ToolError.env("raw_hello reference missing")
    }

    // No-leak check: a reference run after the candidate arm must not touch
    // the diagnostic counters, and the slot must be empty.
    let beforeLeak = AttentionDiagnostic.counters.snapshot()
    _ = try referenceArm(runner) {
        try decode(runner, prefill, step, tokens: helloTokens, steps: 2, keepLogits: false)
    }
    let afterLeak = AttentionDiagnostic.counters.snapshot()
    let leakUnchanged = beforeLeak == afterLeak
    let leakSlotEmpty = !runner.attentionDiagnosticActive
    let noLeak = leakUnchanged && leakSlotEmpty
    emit(["stage": "attention_no_leak", "countersUnchangedOutsideWrapper": leakUnchanged,
          "slotEmpty": leakSlotEmpty, "pass": noLeak])
    if !noLeak { return outcome(false, "no_leak_rejected") }

    // Reachability: the zero canary in the diagnostic slot must change the
    // logits and be counted as applied. A reference re-run must reproduce
    // the original bits. This is evidence only, never a quality result.
    let canary = try AttentionDiagnostic.compile(runner: runner, canary: true)
    let canaryTrace = try candidateArm(runner, canary) {
        try decode(runner, prefill, step, tokens: helloTokens, steps: 1, keepLogits: true)
    }
    let canaryCounts = AttentionDiagnostic.counters.snapshot()
    let canaryChanged = canaryTrace.logits.count == 1 && !sameBits(canaryTrace.logits[0], hello.logits[0])
    let reachable = canaryChanged && countersOK(canaryCounts)
    let restored = try referenceArm(runner) {
        try decode(runner, prefill, step, tokens: helloTokens, steps: 1, keepLogits: true)
    }
    let restoreOK = restored.logits.count == 1 && sameBits(restored.logits[0], hello.logits[0])
        && restored.ids.first == hello.ids.first
    let inactive = !runner.attentionDiagnosticActive
    let dispatchEvidence = DispatchEvidence(
        noLeakCountersUnchanged: leakUnchanged, noLeakSlotEmpty: leakSlotEmpty,
        canaryChangedLogits: canaryChanged, canaryAttempts: canaryCounts.attempts,
        canaryApplied: canaryCounts.applied, canaryFallback: canaryCounts.fallback,
        referenceRestoredBitIdentical: restoreOK, slotEmptyAfterRestore: inactive,
        pass: reachable && restoreOK && inactive)
    emit(["stage": "attention_dispatch_evidence", "canaryChangedLogits": canaryChanged,
          "canaryReached": reachable,
          "canaryApplied": canaryCounts.applied, "canaryFallback": canaryCounts.fallback,
          "referenceRestoredBitIdentical": restoreOK, "slotEmptyAfterRestore": inactive,
          "note": "canary output is reachability evidence only, never a quality pass"])
    guard dispatchEvidence.pass else {
        return outcome(false, "dispatch_evidence_rejected", dispatch: dispatchEvidence)
    }

    // Untimed warm-up of both arms.
    _ = try referenceArm(runner) {
        try decode(runner, prefill, step, tokens: helloTokens, steps: steps, keepLogits: false)
    }
    _ = try candidateArm(runner, candidate) {
        try decode(runner, prefill, step, tokens: helloTokens, steps: steps, keepLogits: false)
    }
    var pairRecords: [[String: Any]] = []
    var pairEvidence: [PairEvidence] = []
    var reductions: [Double] = []
    var policyMet = true
    for pair in 0..<Tool.pairs {
        let candidateFirst = pair % 2 == 1
        var refSeconds = Double.nan
        var candSeconds = Double.nan
        var candCounts: (attempts: Int, applied: Int, fallback: Int) = (0, 0, 0)
        for slot in 0..<2 {
            let runCandidate = (slot == 0) == candidateFirst
            let t: Trace
            if runCandidate {
                t = try candidateArm(runner, candidate) {
                    try decode(runner, prefill, step, tokens: helloTokens, steps: steps, keepLogits: false)
                }
                let counts = AttentionDiagnostic.counters.snapshot()
                candCounts = counts
                guard countersOK(counts) else {
                    emit(["stage": "attention_paired_timing", "pair": pair,
                          "error": "candidate counters show fallback or no application",
                          "applied": counts.applied, "fallback": counts.fallback])
                    return outcome(false, "timing_counters_rejected", dispatch: dispatchEvidence)
                }
            } else {
                t = try referenceArm(runner) {
                    try decode(runner, prefill, step, tokens: helloTokens, steps: steps, keepLogits: false)
                }
            }
            guard t.ids == hello.ids else {
                emit(["stage": "attention_paired_timing", "pair": pair,
                      "arm": runCandidate ? "candidate" : "reference",
                      "error": "generated IDs differ from the gated raw_hello reference"])
                return outcome(false, "timing_ids_rejected", dispatch: dispatchEvidence)
            }
            if runCandidate { candSeconds = t.decodeWallSeconds } else { refSeconds = t.decodeWallSeconds }
        }
        let reduction = (refSeconds - candSeconds) / refSeconds * 100
        reductions.append(reduction)
        let pairOK = candSeconds <= Tool.policyFactor * refSeconds
        if !pairOK { policyMet = false }
        pairEvidence.append(PairEvidence(
            pair: pair, order: candidateFirst ? "BA" : "AB",
            referenceDecodeWallSeconds: refSeconds, candidateDecodeWallSeconds: candSeconds,
            reductionPercent: reduction, candidateAttempts: candCounts.attempts,
            candidateApplied: candCounts.applied, candidateFallback: candCounts.fallback,
            referenceIdsMatchGated: true, candidateIdsMatchGated: true, policyMet: pairOK))
        pairRecords.append(["pair": pair, "order": candidateFirst ? "BA" : "AB",
                            "referenceDecodeWallSeconds": num(refSeconds),
                            "candidateDecodeWallSeconds": num(candSeconds),
                            "reductionPercent": num(reduction)])
    }
    emit(["stage": "attention_paired_timing", "pairs": pairRecords, "decodeSteps": steps - 1,
          "medianReductionPercent": num(median(reductions)),
          "policy": "each of 3 paired AB/BA warm-Hello 31-step decode walls: candidate <= 0.98 x reference",
          "policyMet": policyMet,
          "note": "A = " + Tool.referenceArmName + ", B = " + Tool.candidateArmName
              + "; whole-decode wall seconds of this diagnostic pair only; no occupancy claim"])
    let timing = TimingEvidence(decodeSteps: steps - 1, warmupUntimed: true, pairs: pairEvidence,
                                medianReductionPercent: median(reductions), policyMet: policyMet)
    return outcome(policyMet, policyMet ? "full_and_timing_passed" : "timing_policy_rejected",
                   dispatch: dispatchEvidence, timing: timing, ids: hello.ids)
}

// MARK: - Forge modes

func emitIdentity(_ mode: String, _ id: IdentityRecord) {
    emit(["stage": "attention_forge_identity", "label": Tool.label, "mode": mode,
          "executableSha256": id.files.executable.sha256, "ggufSha256": id.files.gguf.sha256,
          "ggufBytes": id.files.gguf.bytes, "regionSha256": id.files.region.sha256,
          "graphSha256": id.files.graph.sha256,
          "candidateSourceSha256": id.sources.candidateSourceSha256,
          "incumbentShaderSourceSha256": id.sources.incumbentShaderSourceSha256,
          "engineIdentitySha256": id.runtime.engineIdentitySha256,
          "hardwareProfileSha256": id.runtime.hardwareProfileSha256])
}

func componentShape(_ c: ComponentEvidence) -> [String] {
    var s = ["tgmem=\(c.threadgroupMemoryBytes)", "maxThreads=\(c.candidateMaxThreads)",
             "threads=\(c.threadsPerHead)", "maxContext=\(c.maxContext)"]
    for k in c.cases {
        s.append("\(k.label)|\(k.tokens)|\(k.queryPosition)|\(k.attended)|\(k.headDim)|\(k.qHeads)|\(k.kvHeads)|\(k.window)")
    }
    return s
}

/// --select: gates 1-3, then one exclusive, canonical, digested artifact.
func runSelect(_ inp: Inputs) throws -> Int32 {
    let mode = "--select"
    if pathExists(inp.artifact) {
        print("REFUSED: the --artifact path already exists; artifacts are never overwritten")
        emitSummary(mode, "artifact_path_exists", false, ["artifactWritten": false])
        return 2
    }
    let started = isoNow()
    let files = try currentFiles(inp)
    let sources = currentSources()
    let os = currentOSRecord()
    let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: inp.gguf))
    let runtime = try currentRuntimeIdentity(runner)
    let identity = IdentityRecord(files: files, sources: sources, os: os, runtime: runtime)
    emitIdentity(mode, identity)
    let comp = try runComponentGates(runner: runner)
    guard let candidate = comp.candidate else {
        emitSummary(mode, "component_rejected", false, ["artifactWritten": false])
        return 1
    }
    let compProblems = checkComponent(comp.evidence, sourceSha: sources.candidateSourceSha256)
    guard compProblems.isEmpty else {
        emit(["stage": "attention_forge_evidence", "problems": compProblems, "pass": false])
        emitSummary(mode, "component_evidence_rejected", false, ["artifactWritten": false])
        return 1
    }
    let full = try runFull(runner: runner, candidate: candidate, ggufPath: inp.gguf,
                           regionPath: inp.region, graphPath: inp.graph)
    guard let fullEv = fullEvidence(full) else {
        emitSummary(mode, full.stage, false, ["artifactWritten": false])
        return 1
    }
    let payload = ArtifactPayload(
        schema: Tool.schema, label: Tool.label, policy: PolicyRecord.current,
        selection: SelectionRecord.current, identity: identity, component: comp.evidence, full: fullEv,
        timestamps: TimestampRecord(selectionStartedUTC: started, selectionFinishedUTC: isoNow()),
        productionQualified: false, limitations: Tool.limitations)
    let problems = validatePayload(payload)
    guard problems.isEmpty else {
        emit(["stage": "attention_forge_evidence", "problems": problems, "pass": false])
        emitSummary(mode, "selection_evidence_rejected", false, ["artifactWritten": false])
        return 1
    }
    let (bytes, digest) = try encodeEnvelope(payload)
    let privacy = privacyViolations(bytes, paths: [inp.gguf, inp.region, inp.graph, inp.artifact])
    guard privacy.isEmpty else {
        emit(["stage": "attention_forge_privacy", "problems": privacy, "pass": false])
        emitSummary(mode, "artifact_privacy_rejected", false, ["artifactWritten": false])
        return 1
    }
    _ = try parseArtifact(bytes)
    try writeArtifactExclusive(bytes, to: inp.artifact)
    guard try readBoundedArtifact(inp.artifact) == bytes else {
        throw ToolError.env("artifact read-back differs from the bytes written")
    }
    emit(["stage": "attention_forge_artifact", "label": Tool.label, "written": true,
          "artifactDigest": digest, "artifactBytes": bytes.count,
          "medianReductionPercent": num(fullEv.timing.medianReductionPercent),
          "note": "experiment record only; digest is integrity, not authenticity; nothing installed"])
    emitSummary(mode, "selected_and_artifact_written", true,
                ["artifactWritten": true, "artifactDigest": digest, "artifactBytes": bytes.count])
    return 0
}

/// --validate-artifact (runGates false) and --verify-artifact (runGates true).
/// All artifact, evidence and identity checks finish before the candidate is
/// compiled or used on the GPU.
func runVerify(_ inp: Inputs, runGates: Bool) throws -> Int32 {
    let mode = runGates ? "--verify-artifact" : "--validate-artifact"
    let data = try readBoundedArtifact(inp.artifact)
    let privacy = privacyViolations(data, paths: [inp.gguf, inp.region, inp.graph, inp.artifact])
    guard privacy.isEmpty else { throw ToolError.artifact("privacy: " + privacy.joined(separator: "; ")) }
    let envelope = try parseArtifact(data)
    let a = envelope.payload
    let problems = validatePayload(a)
    guard problems.isEmpty else {
        throw ToolError.artifact("schema, policy or evidence: " + problems.joined(separator: "; "))
    }
    emit(["stage": "attention_forge_artifact", "label": Tool.label, "mode": mode,
          "artifactDigest": envelope.digest, "schemaDigestEvidenceValidation": "passed",
          "candidateCompiled": false])
    let files = try currentFiles(inp)
    let sources = currentSources()
    let os = currentOSRecord()
    let pre = identityMismatches(a.identity, files: files, sources: sources, os: os)
    guard pre.isEmpty else {
        throw ToolError.artifact("identity mismatch before runner init: " + pre.joined(separator: ", "))
    }
    let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: inp.gguf))
    let runtime = try currentRuntimeIdentity(runner)
    let post = runtimeMismatches(a.identity.runtime, runtime)
    guard post.isEmpty else {
        throw ToolError.artifact("runtime identity mismatch: " + post.joined(separator: ", "))
    }
    emit(["stage": "attention_forge_identity_validation", "label": Tool.label, "mode": mode,
          "identity_validation": "passed", "artifactDigest": envelope.digest, "candidateCompiled": false,
          "note": "executable, gguf, region, graph, sources, OS, engine, hardware and geometry identity validated before any candidate compile or GPU candidate use; digest is integrity only, not authenticity"])
    if !runGates {
        emitSummary(mode, "identity_validation_passed", true, ["artifactDigest": envelope.digest])
        return 0
    }
    let comp = try runComponentGates(runner: runner)
    guard let candidate = comp.candidate else {
        emitSummary(mode, "component_rejected", false, ["artifactDigest": envelope.digest])
        return 1
    }
    var compProblems = checkComponent(comp.evidence, sourceSha: sources.candidateSourceSha256)
    if componentShape(comp.evidence) != componentShape(a.component) {
        compProblems.append("fresh component matrix geometry differs from the artifact")
    }
    guard compProblems.isEmpty else {
        emit(["stage": "attention_forge_evidence", "problems": compProblems, "pass": false])
        emitSummary(mode, "component_evidence_rejected", false, ["artifactDigest": envelope.digest])
        return 1
    }
    let full = try runFull(runner: runner, candidate: candidate, ggufPath: inp.gguf,
                           regionPath: inp.region, graphPath: inp.graph)
    guard let fullEv = fullEvidence(full) else {
        emitSummary(mode, full.stage, false, ["artifactDigest": envelope.digest])
        return 1
    }
    var fullProblems = checkFull(fullEv)
    if fullEv.fixtures.map({ $0.promptTokens }) != a.full.fixtures.map({ $0.promptTokens }) {
        fullProblems.append("fresh prompt token counts differ from the artifact")
    }
    if fullEv.rawHelloReferenceIds != a.full.rawHelloReferenceIds {
        fullProblems.append("fresh raw_hello greedy IDs differ from the artifact")
    }
    guard fullProblems.isEmpty else {
        emit(["stage": "attention_forge_evidence", "problems": fullProblems, "pass": false])
        emitSummary(mode, "fresh_evidence_rejected", false, ["artifactDigest": envelope.digest])
        return 1
    }
    emit(["stage": "attention_forge_verify", "label": Tool.label,
          "freshMedianReductionPercent": num(fullEv.timing.medianReductionPercent),
          "artifactMedianReductionPercent": num(a.full.timing.medianReductionPercent),
          "freshPolicyMet": fullEv.timing.policyMet, "rawHelloIdsMatchArtifact": true,
          "note": "fresh-process rerun of the same gates and policy; timings are re-measured and not expected to equal the artifact"])
    emitSummary(mode, "artifact_verified_fresh_gates_passed", true, ["artifactDigest": envelope.digest])
    return 0
}

// MARK: - Entry

func usageText() -> String {
    return [
        "usage: attnqual --component --gguf PATH",
        "       attnqual --full --gguf PATH --region PATH --graph PATH",
        "       attnqual --select --gguf PATH --region PATH --graph PATH --artifact NEW_PATH",
        "       attnqual --validate-artifact --gguf PATH --region PATH --graph PATH --artifact PATH",
        "       attnqual --verify-artifact --gguf PATH --region PATH --graph PATH --artifact PATH",
    ].joined(separator: "\n")
}

func emitSummary(_ mode: String, _ stage: String, _ pass: Bool, _ extra: [String: Any] = [:]) {
    var obj: [String: Any] = [
        "tool": "attention_qualification", "label": Tool.label, "mode": mode,
        "candidate": AttentionDiagnostic.kernelName, "candidateKind": "ATTENTION",
        "stageReached": stage, "pass": pass,
        "productionQualified": AttentionDiagnostic.productionQualified,
        "note": "diagnostic only; production default and production artifacts unchanged; a --select artifact is an experiment record that is never installed; no occupancy claims"]
    for (k, v) in extra { obj[k] = v }
    emit(obj)
}

func mainEntry() -> Int32 {
    let args = Array(CommandLine.arguments.dropFirst())
    let modes = ["--component", "--full", "--select", "--validate-artifact", "--verify-artifact"]
    let optionNames = ["--gguf", "--region", "--graph", "--artifact"]
    var mode: String? = nil
    var opts: [String: String] = [:]
    var i = 0
    while i < args.count {
        let a = args[i]
        if modes.contains(a) {
            guard mode == nil else { print(usageText()); return 2 }
            mode = a
        } else if optionNames.contains(a) {
            i += 1
            guard i < args.count, opts[a] == nil else { print(usageText()); return 2 }
            opts[a] = args[i]
        } else {
            print(usageText())
            return 2
        }
        i += 1
    }
    guard let m = mode, let ggufPath = opts["--gguf"] else { print(usageText()); return 2 }
    let region = opts["--region"]
    let graph = opts["--graph"]
    let artifact = opts["--artifact"]
    let forgeMode = m == "--select" || m == "--validate-artifact" || m == "--verify-artifact"
    if m == "--full" && (region == nil || graph == nil) { print(usageText()); return 2 }
    if !forgeMode && artifact != nil { print(usageText()); return 2 }
    do {
        if forgeMode {
            guard let r = region, let g = graph, let p = artifact else { print(usageText()); return 2 }
            let inputs = Inputs(gguf: ggufPath, region: r, graph: g, artifact: p)
            if m == "--select" { return try runSelect(inputs) }
            return try runVerify(inputs, runGates: m == "--verify-artifact")
        }
        let runner = try Gemma4Runner(fileAt: URL(fileURLWithPath: ggufPath))
        let comp = try runComponentGates(runner: runner)
        guard let candidate = comp.candidate else {
            emitSummary(m, "component_rejected", false)
            return 1
        }
        if m == "--component" {
            emitSummary(m, "component_passed", true)
            return 0
        }
        let outcome = try runFull(runner: runner, candidate: candidate, ggufPath: ggufPath,
                                  regionPath: region ?? "", graphPath: graph ?? "")
        emitSummary(m, outcome.pass ? "full_and_timing_passed" : "full_or_timing_rejected", outcome.pass,
                    ["fullStage": outcome.stage])
        return outcome.pass ? 0 : 1
    } catch ToolError.artifact(let msg) {
        print("ARTIFACT REJECTED: \(msg)")
        emitSummary(m, "artifact_rejected", false)
        return 3
    } catch ToolError.gate(let msg) {
        print("GATE FAIL: \(msg)")
        emitSummary(m, "gate_error", false)
        return 1
    } catch ToolError.env(let msg) {
        print("ENVIRONMENT ERROR: \(msg)")
        emitSummary(m, "environment_error", false)
        return 2
    } catch let e as AttentionDiagnosticError {
        print("ATTENTION ERROR: \(e)")
        emitSummary(m, "attention_error", false)
        if case .gate = e { return 1 }
        return 2
    } catch {
        print("ERROR: \(error)")
        emitSummary(m, "error", false)
        return 2
    }
}

exit(mainEntry())
