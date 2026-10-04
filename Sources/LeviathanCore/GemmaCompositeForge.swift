// Phase 32 — shared activation/composition helper for the EXISTING MoE
// E4 lowering and the hardware-derived Q4 launch-geometry pass.
//
// This file adds NO new selector, NO new policy, and NO new GemmaForge
// candidate axes. It is a thin orchestration shell around the existing
// production machinery:
//   - the existing artifact identity validation
//     (Q4LaunchArtifact.validate: self-digest, actual hardware, actual
//     graph digest, region graph SHA, fresh E4 emission hash, primitive
//     MSL source, toolchain, cached model-file SHA-256 (ModelDigestCache;
//     hashed once, refreshed only explicitly), executable digest, pipeline width
//     legality),
//   - the existing selection workflow (Q4LaunchSelect.selectAndPersist
//     with diagnostic:false, 32 gen / 31 timed tokens, the caller-supplied
//     canonical fixtures, the unchanged fixed 2% policy and all its
//     counter proofs),
//   - the existing PUBLIC atomic verified composite installer
//     (Gemma4Runner.installQ4LaunchGenerated).
//
// The helper never performs a trial install, never re-signs or re-binds
// a stale artifact, and never silently falls back: an existing artifact
// that fails identity validation is a HARD failure (no overwrite, no
// rebinding); a missing artifact with allowSelection == false is
// rejected; an honest no-win record (performanceSelected == false AND
// the existing policy's recomputed selection is also false, i.e. the
// claim is internally consistent) leaves the incumbent active and
// returns installed == false with an explicit log.
//
// The runner must be PRISTINE on entry (no moeGenerated, no
// q4LaunchWidth, no q4GeneratedArtifactDigest, no
// rmsGeneratedArtifactDigest): a runner that is already configured is
// rejected clearly instead of having its prior valid install state
// silently erased, so any measured baseline the caller holds is valid.
//
// No-win limitation, stated exactly: Q4LaunchArtifact.validate checks
// the model-gate SHAPE (Q4LaunchGatePolicy.shapeValid), while the
// production install gate additionally requires
// Q4LaunchGatePolicy.productionValid (performanceSelected == recomputed
// selection == true). A no-win artifact therefore CANNOT be installed —
// by design. Without mutating the policy there is no way to make a
// no-win record installable, and none is invented here: structural
// validity AND consistency of the retained no-win record are confirmed
// with the EXISTING policy validators (shapeValid, plus recomputed
// selection == false matching the recorded performanceSelected ==
// false), malformed/diagnostic records are hard-rejected, and the
// incumbent stays active.

import Foundation

public enum GemmaCompositeForge {

    /// Result of one activation attempt.
    public struct PreparationResult: Sendable, Equatable {
        public let installed: Bool
        public let cacheHit: Bool
        public let artifactDigest: String
        public let log: [String]
    }

    /// Bound on reading a persisted composite artifact: the bounded
    /// loader refuses anything larger BEFORE any full-file allocation.
    public static let artifactReadLimitBytes = 8 << 20

    /// Activate the composite (accepted E4 MoE lowering + hardware-derived
    /// Q4 launch width) for one runner/model/graph.
    ///
    /// The runner must be PRISTINE: any already-installed lowering or
    /// launch width, or any recorded generated-artifact digest, is a
    /// clear rejection — the helper never erases prior valid state and
    /// thus never invalidates a baseline the caller has measured.
    /// Region, graph, model, and resident mode are validated BEFORE any
    /// load/selection/install is attempted. If a persisted artifact
    /// exists at `artifactURL` it is loaded with a bounded read
    /// (<= 8 MiB) and validated against the actual graph digest, region,
    /// hardware, model binding, executable, and pipeline; any mismatch
    /// is a hard failure — the file is never overwritten or re-bound. A
    /// production-valid artifact installs through the PUBLIC installer
    /// (atomic, verified); a structurally valid, internally consistent
    /// no-win record reports installed == false honestly. If no
    /// artifact exists and `allowSelection` is false the call is
    /// rejected; with `allowSelection` true the EXISTING selection
    /// workflow runs (production settings only) and the freshly
    /// produced artifact is validated exactly like a cached one before
    /// any install.
    public static func prepare(
        regionJSON: Data,
        ggufURL: URL,
        graph: GemmaGraphIR,
        runner: Gemma4Runner,
        ggufDigest: String,
        artifactURL: URL,
        fixtures: [Q4LoweringForge.Q4Fixture],
        isEndOfGeneration: (Int) -> Bool,
        allowSelection: Bool) throws -> PreparationResult {

        var log = [String]()

        // ---- pristine-runner gate FIRST: never erase prior state ----
        if runner.moeGenerated != nil || runner.q4LaunchWidth != nil
            || runner.q4GeneratedArtifactDigest != nil
            || runner.rmsGeneratedArtifactDigest != nil {
            throw Q4LoweringError.invalidArtifact(
                "composite prepare: runner is already configured "
                + "(installed lowering / launch width / generated digests "
                + "present) — reject instead of erasing prior valid state")
        }

        // ---- validate region / graph / model / resident mode ----
        let region = try MoERegionIR.parse(regionJSON)
        try region.validate()
        try GemmaGraph.validate(graph, config: runner.config, file: runner.ggufFile)
        let graphDigest = try GemmaGraph.digest(graph)
        // The supplied URL must BE the file the runner actually mapped —
        // an unrelated file is rejected rather than silently trusted.
        guard let mappedURL = runner.ggufFile.sourceURL, mappedURL == ggufURL else {
            throw Q4LoweringError.invalidArtifact(
                "composite prepare: ggufURL is not the runner's mapped GGUF file")
        }
        // Fresh model binding: this also rejects expert-cache (streaming)
        // runners — the composite binds the resident mapping.
        let freshBinding = try Gemma4Runner.moeBindingForProfile(
            region: region, runner: runner)
        let hardware = runner.hardwareProfile()
        let pipelineMaxThreads = try runner.q4MatvecPipelineMaxThreads()
        let executableDigest = try Gemma4Runner.executableDigest()
        log.append("validated region/graph/model (resident, "
                   + "\(freshBinding.layerCount) layers, pipeline max \(pipelineMaxThreads))")

        // Shared fail-closed validation against the ACTUAL current facts,
        // including the fresh binding equality (no rebinding of a stale
        // artifact to a different file/device).
        func validate(_ artifact: Q4LaunchArtifact) throws {
            try artifact.validate(region: region,
                                  pipelineMaxThreads: pipelineMaxThreads,
                                  hardware: hardware,
                                  graphDigest: graphDigest,
                                  ggufDigest: ggufDigest,
                                  executableDigest: executableDigest)
            guard artifact.baselineMoeBinding == freshBinding else {
                throw Q4LoweringError.invalidArtifact(
                    "recorded E4 binding does not match this file/device — "
                    + "no rebinding of a stale artifact")
            }
        }

        // No-win classification uses the EXISTING policy machinery only:
        // structural validity (shapeValid) AND internal consistency —
        // the policy's recomputed selection must ALSO be false, so an
        // inconsistent claimed no-win (shape valid but the policy would
        // actually select) can never be accepted.
        func isConsistentNoWin(_ artifact: Q4LaunchArtifact,
                               _ gate: Q4LaunchGateRecord) throws -> Bool {
            guard Q4LaunchGatePolicy.shapeValid(gate, schedule: artifact.schedule,
                                                binding: artifact.baselineMoeBinding),
                  gate.performanceSelected == false,
                  Q4LaunchGatePolicy.recomputedSelection(
                      gate, schedule: artifact.schedule,
                      binding: artifact.baselineMoeBinding) == false else {
                return false
            }
            return true
        }

        // ---- existing artifact path (bounded read, hard-fail) ----
        if let existing = try loadArtifact(from: artifactURL) {
            try validate(existing)
            guard let gate = existing.modelGate, !gate.diagnostic else {
                throw Q4LoweringError.invalidArtifact(
                    "existing artifact carries no non-diagnostic gate record — "
                    + "rejecting as not production-selected (no silent fallback)")
            }
            if Q4LaunchGatePolicy.productionValid(gate, schedule: existing.schedule,
                                                  binding: existing.baselineMoeBinding) {
                try runner.installQ4LaunchGenerated(
                    artifact: existing, ggufDigest: ggufDigest,
                    region: region, graphDigest: graphDigest)
                log.append("cache hit: existing composite \(existing.schedule.label) "
                           + "validated and installed through the public installer")
                return PreparationResult(installed: true, cacheHit: true,
                                         artifactDigest: existing.artifactDigest, log: log)
            }
            // Structural validity AND consistency of a retained no-win
            // record are decided by the EXISTING policy validators —
            // never a weaker invented check, never a policy mutation,
            // never a pretend install.
            guard try isConsistentNoWin(existing, gate) else {
                throw Q4LoweringError.invalidArtifact(
                    "existing artifact is not production-selected and not a "
                    + "structurally valid, consistent no-win record — rejecting "
                    + "(no silent fallback)")
            }
            log.append("cache hit: honest no-win record (existing policy "
                       + "validators confirm shape validity and recomputed "
                       + "selection == false matching performanceSelected == "
                       + "false) — incumbent stays active, nothing installed")
            return PreparationResult(installed: false, cacheHit: true,
                                     artifactDigest: existing.artifactDigest, log: log)
        }

        // ---- missing artifact: reuse-only callers are rejected ----
        guard allowSelection else {
            throw Q4LoweringError.invalidArtifact(
                "no composite artifact at \(artifactURL.path) and selection is "
                + "not allowed (reuse-only)")
        }

        // EXISTING selection workflow, production settings only: every
        // hardware-derivation, FP64/bit component gate, fixture/32/31
        // strict replay, counter-proof, and the fixed 2% policy remain
        // exactly as implemented in Q4LaunchSelect.
        let (produced, selectLog) = try Q4LaunchSelect.selectAndPersist(
            regionJSON: regionJSON, ggufURL: ggufURL, graph: graph,
            runner: runner, fixtures: fixtures,
            genTokens: Q4LaunchGatePolicy.productionGenTokens,
            timedTokens: Q4LaunchGatePolicy.productionTimedTokens,
            isEndOfGeneration: isEndOfGeneration,
            diagnostic: false,
            saveTo: artifactURL)
        log.append(contentsOf: selectLog)

        // Validate the freshly produced artifact exactly like a cached one.
        try validate(produced)
        guard let gate = produced.modelGate, !gate.diagnostic else {
            throw Q4LoweringError.invalidArtifact(
                "freshly selected artifact carries no non-diagnostic gate record")
        }
        if Q4LaunchGatePolicy.productionValid(gate, schedule: produced.schedule,
                                              binding: produced.baselineMoeBinding) {
            // Atomic verified composite install through the PUBLIC
            // installer only — never a direct trial install from here.
            try runner.installQ4LaunchGenerated(
                artifact: produced, ggufDigest: ggufDigest,
                region: region, graphDigest: graphDigest)
            log.append("fresh selection \(produced.schedule.label) installed "
                       + "through the public installer")
            return PreparationResult(installed: true, cacheHit: false,
                                     artifactDigest: produced.artifactDigest, log: log)
        }
        guard try isConsistentNoWin(produced, gate) else {
            throw Q4LoweringError.invalidArtifact(
                "freshly produced artifact is not production-selected and not a "
                + "structurally valid, consistent no-win record — rejecting")
        }
        log.append("honest no-win: record persisted, incumbent stays active, "
                   + "nothing installed")
        return PreparationResult(installed: false, cacheHit: false,
                                 artifactDigest: produced.artifactDigest, log: log)
    }

    /// Bounded artifact loader: reads in chunks and refuses anything over
    /// `artifactReadLimitBytes` before any full-file allocation. Missing
    /// file returns nil; unreadable/oversized/undecodable content throws.
    public static func loadArtifact(from url: URL) throws -> Q4LaunchArtifact? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var data = Data()
        while data.count <= artifactReadLimitBytes {
            let chunk = try handle.read(upToCount: 1 << 16) ?? Data()
            if chunk.isEmpty { break }
            data.append(chunk)
        }
        guard data.count <= artifactReadLimitBytes else {
            throw Q4LoweringError.invalidArtifact(
                "composite artifact exceeds 8 MiB at \(url.path)")
        }
        do { return try JSONDecoder().decode(Q4LaunchArtifact.self, from: data) }
        catch { throw Q4LoweringError.invalidArtifact("composite artifact decode: \(error)") }
    }
}
