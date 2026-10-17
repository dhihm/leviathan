// Model-file digest policy for every model-bound artifact, plan, profile,
// and qualification contract (policy cached-explicit-refresh-v1).
//
// The legacy-named `ggufDigest` fields (and the profile/qualification
// model SHA-256 fields) carry the ACTUAL SHA-256 of the model file's
// bytes, resolved through the shared persisted ModelDigestCache: the file
// is hashed once (initial cache miss) or on an explicit refresh
// (`--rehash-model`), and every other lookup reuses that historical
// record without re-reading the model bytes and without stat/mtime/size
// checks. A cached digest is therefore NOT a fresh per-run verification:
// editing the weights in place at the same path is intentionally not
// detected until an explicit refresh.
//
// Contracts record `modelFileVerification == .cachedExplicitRefresh`,
// encoded into their body/plan digests and required on decode. Schema
// versions were bumped again (3), so full-hash contracts (schema 1) and
// the intermediate unverified contracts (schema 2, "none-unverified-v1")
// are rejected and must be re-forged; neither is ever re-tagged.
//
// Execution-essential GGUF header / tensor-directory geometry checks,
// mapped-URL and byte-count binding checks, and the graph, executable,
// source, hardware, and toolchain identity are unchanged. They are
// execution binding checks, not cache freshness checks.
//
// GemmaForge.fileDigest remains the general streaming SHA-256 utility
// for executable/source/small fixture hashing and is unchanged.

import Foundation

/// Model-file digest policy recorded in serialized model-bound
/// contracts. Exactly one value exists, so decoding never has to choose a
/// default, and records carrying a retired value ("none-unverified-v1")
/// or no value fail to decode.
public enum ModelFileVerificationPolicy: String, Codable, Equatable, Hashable, Sendable {
    /// The model digest is an actual SHA-256 of the model file, computed
    /// once from its bytes (or on an explicit refresh) and reused from the
    /// shared ModelDigestCache. It is a historical computation, not a
    /// fresh verification.
    case cachedExplicitRefresh = "cached-explicit-refresh-v1"
}

public enum ModelFileIdentity {
    /// The policy every model-bound contract produced by this build records.
    public static let policy = ModelFileVerificationPolicy.cachedExplicitRefresh
}
