// Shared persisted cache of actual model-file SHA-256 digests
// (policy cached-explicit-refresh-v1).
//
// A model file is hashed from its actual bytes ONCE, in bounded streaming
// chunks: on the first lookup that finds no cache record (an initial
// computation), or when the caller explicitly asks for a refresh
// (`--rehash-model`, LEVIATHAN_REHASH_MODEL=1). Every other lookup, in this
// process, any later process, or the Python tools, reuses the persisted
// record WITHOUT reading the model bytes, without stat/mtime/size checks,
// and without any automatic rehash. A cached digest is therefore a
// historical computation, never a fresh verification: editing the weights
// at the same path is intentionally NOT detected until an explicit refresh.
// A missing record triggers the one initial computation; a malformed,
// mismatched, or retired-policy record is an error carrying an explicit
// refresh instruction, never a silent full rehash and never a digest.
//
// Shared on-disk format (also read and written by
// tools/baseline/model_digest_cache.py):
//   directory: $LEVIATHAN_MODEL_DIGEST_CACHE_DIR if set, otherwise
//              ~/Library/Caches/Leviathan/model-digests
//   canonical path: realpath(3) of the absolute path (symlinks resolved;
//              for a file that no longer exists, realpath of its parent
//              directory plus the file name), normalized to Unicode NFC
//              in both cases so existing and deleted files share a key
//   key: lowercase hex SHA-256 of the UTF-8 canonical path
//   record: <key>.json, one JSON object with exactly these fields:
//     bytes        bytes hashed at hashedAtUTC (history, not a freshness check)
//     format       "leviathan.model-digest-cache.v1"
//     hashedAtUTC  "YYYY-MM-DDTHH:MM:SSZ"
//     path         the canonical path
//     policy       "cached-explicit-refresh-v1"
//     sha256       64 lowercase hex characters
//     source       "initial" | "explicit-refresh" | "historical-receipt"
//   lock: <key>.lock, held with flock(LOCK_EX) around every read and write
//   writes: atomic replace (temporary file + rename); reads bounded to 64 KiB

import Foundation
import CryptoKit
import Darwin

/// How a lookup obtained its digest.
public enum ModelDigestOrigin: String, Equatable, Sendable {
    case cacheHit = "cache-hit"
    case initialComputation = "initial-computation"
    case explicitRefresh = "explicit-refresh"
}

public enum ModelDigestCacheError: Error, CustomStringConvertible, Equatable {
    case malformedRecord(String)
    case ioFailure(String)

    public var description: String {
        switch self {
        case .malformedRecord(let m): return "model digest cache record invalid: " + m
        case .ioFailure(let m): return "model digest cache I/O failure: " + m
        }
    }
}

/// One persisted cache record (shared JSON format; see the file header).
public struct ModelDigestRecord: Codable, Equatable, Sendable {
    public static let currentFormat = "leviathan.model-digest-cache.v1"
    public static let sources: Set<String> = ["initial", "explicit-refresh", "historical-receipt"]

    public let format: String
    public let path: String
    public let sha256: String
    public let bytes: Int
    public let hashedAtUTC: String
    public let policy: ModelFileVerificationPolicy
    public let source: String

    public init(format: String, path: String, sha256: String, bytes: Int,
                hashedAtUTC: String, policy: ModelFileVerificationPolicy,
                source: String) {
        self.format = format
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.hashedAtUTC = hashedAtUTC
        self.policy = policy
        self.source = source
    }

    /// Structural validity of a decoded record (the policy value itself is
    /// enforced by decoding).
    public var isWellFormed: Bool {
        format == Self.currentFormat
            && sha256.utf8.count == 64
            && sha256.utf8.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
            && bytes >= 0
            && !hashedAtUTC.isEmpty
            && !path.isEmpty
            && Self.sources.contains(source)
    }
}

/// The result of one lookup: the record plus how it was obtained.
public struct ModelDigestLookup: Equatable, Sendable {
    public let record: ModelDigestRecord
    public let origin: ModelDigestOrigin

    public init(record: ModelDigestRecord, origin: ModelDigestOrigin) {
        self.record = record
        self.origin = origin
    }

    public var sha256: String { record.sha256 }

    /// Truthful one-line disclosure for logs: a cache hit never claims a
    /// fresh verification.
    public var disclosure: String {
        let head = "model file SHA-256 " + record.sha256 + " for " + record.path
        switch origin {
        case .cacheHit:
            return head + ": cache hit (historically hashed at " + record.hashedAtUTC
                + ", source " + record.source + ", " + String(record.bytes)
                + " bytes); model bytes NOT re-read now and NOT freshly verified;"
                + " same-path weight edits require --rehash-model"
        case .initialComputation:
            return head + ": computed now from " + String(record.bytes)
                + " bytes (initial cache miss); cached for reuse"
        case .explicitRefresh:
            return head + ": recomputed now from " + String(record.bytes)
                + " bytes (explicit refresh); cached for reuse"
        }
    }
}

/// Thread-safe (in-process lock) and process-safe (per-key flock) cache.
public final class ModelDigestCache: @unchecked Sendable {
    public static let directoryEnvironmentKey = "LEVIATHAN_MODEL_DIGEST_CACHE_DIR"
    public static let maxRecordBytes = 64 << 10
    static let hashChunkBytes = 1 << 23
    static let refreshInstruction =
        "refresh explicitly with --rehash-model (leviathan CLI) or "
        + "LEVIATHAN_REHASH_MODEL=1 (Python tools); the record is not used as a digest"

    /// $LEVIATHAN_MODEL_DIGEST_CACHE_DIR, else the user-owned
    /// ~/Library/Caches/Leviathan/model-digests.
    public static func defaultDirectory() -> URL {
        if let custom = ProcessInfo.processInfo.environment[directoryEnvironmentKey],
           !custom.isEmpty {
            return URL(fileURLWithPath: custom, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Caches", isDirectory: true)
            .appendingPathComponent("Leviathan", isDirectory: true)
            .appendingPathComponent("model-digests", isDirectory: true)
    }

    /// The process-wide cache in the default directory.
    public static let shared = ModelDigestCache(directory: defaultDirectory())

    public let directory: URL
    private let lock = NSLock()

    /// A cache rooted at `directory` (tests inject temporary directories).
    public init(directory: URL) {
        self.directory = directory.absoluteURL.standardizedFileURL
    }

    /// Canonical path used as the cache identity (see the file header).
    /// Normalized to Unicode NFC: realpath(3) reports the on-disk (often
    /// NFD) spelling of an existing file, while the deleted-file fallback
    /// keeps the caller's spelling, so every branch must normalize for the
    /// key to survive deletion and to match the Python tools.
    public static func canonicalPath(for url: URL) -> String {
        let absolute = url.absoluteURL.standardizedFileURL.path
        if let resolved = resolvedPath(absolute) {
            return resolved.precomposedStringWithCanonicalMapping
        }
        let parent = (absolute as NSString).deletingLastPathComponent
        let name = (absolute as NSString).lastPathComponent
        if let resolvedParent = resolvedPath(parent) {
            return (resolvedParent as NSString).appendingPathComponent(name)
                .precomposedStringWithCanonicalMapping
        }
        return absolute.precomposedStringWithCanonicalMapping
    }

    /// Lowercase hex SHA-256 of the UTF-8 canonical path.
    public static func cacheKey(for url: URL) -> String {
        key(canonical: canonicalPath(for: url))
    }

    /// The record file this cache uses for `url`.
    public func recordURL(for url: URL) -> URL {
        directory.appendingPathComponent(Self.cacheKey(for: url) + ".json")
    }

    /// Cache-only read: never hashes. nil when no record exists; throws for
    /// a malformed, mismatched, or retired-policy record.
    public func cachedRecord(for url: URL) throws -> ModelDigestRecord? {
        let canonical = Self.canonicalPath(for: url)
        let key = Self.key(canonical: canonical)
        return try withKeyLock(key) { () throws -> ModelDigestRecord? in
            try readRecord(key: key, canonical: canonical)
        }
    }

    /// The model's SHA-256. A valid record is reused without reading the
    /// model; a missing record is filled by one streaming hash; `refresh`
    /// re-hashes explicitly and replaces the record (also repairing a
    /// malformed one). A malformed record without `refresh` throws.
    public func digest(for url: URL, refresh: Bool = false) throws -> ModelDigestLookup {
        let canonical = Self.canonicalPath(for: url)
        let key = Self.key(canonical: canonical)
        return try withKeyLock(key) { () throws -> ModelDigestLookup in
            if !refresh, let existing = try readRecord(key: key, canonical: canonical) {
                return ModelDigestLookup(record: existing, origin: .cacheHit)
            }
            // Open the caller's spelling: the NFC canonical path may not be
            // the on-disk name on a normalization-sensitive volume.
            let (sha256, bytes) = try Self.streamSHA256(url)
            let record = ModelDigestRecord(
                format: ModelDigestRecord.currentFormat, path: canonical,
                sha256: sha256, bytes: bytes,
                hashedAtUTC: ISO8601DateFormatter().string(from: Date()),
                policy: ModelFileIdentity.policy,
                source: refresh ? "explicit-refresh" : "initial")
            try writeRecord(record, key: key)
            return ModelDigestLookup(record: record,
                                     origin: refresh ? .explicitRefresh : .initialComputation)
        }
    }

    // MARK: - internals

    static func key(canonical: String) -> String {
        hex(CryptoKit.SHA256.hash(data: Data(canonical.utf8)))
    }

    static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    private static func resolvedPath(_ path: String) -> String? {
        guard let resolved = realpath(path, nil) else { return nil }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    /// Bounded streaming SHA-256 of the model file (the only model read).
    static func streamSHA256(_ url: URL) throws -> (String, Int) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch {
            throw ModelDigestCacheError.ioFailure(
                "cannot open model " + url.path + ": " + String(describing: error))
        }
        defer { try? handle.close() }
        var hasher = CryptoKit.SHA256()
        var total = 0
        var done = false
        while !done {
            done = try autoreleasepool {
                guard let data = try handle.read(upToCount: hashChunkBytes),
                      !data.isEmpty else { return true }
                hasher.update(data: data)
                total += data.count
                return false
            }
        }
        return (hex(hasher.finalize()), total)
    }

    private func withKeyLock<T>(_ key: String, _ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        do {
            try FileManager.default.createDirectory(at: directory,
                                                    withIntermediateDirectories: true)
        } catch {
            throw ModelDigestCacheError.ioFailure(
                "cannot create " + directory.path + ": " + String(describing: error))
        }
        let lockPath = directory.appendingPathComponent(key + ".lock").path
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else {
            throw ModelDigestCacheError.ioFailure(
                "cannot open lock " + lockPath + " (errno " + String(errno) + ")")
        }
        defer { _ = close(fd) }
        while flock(fd, LOCK_EX) != 0 {
            guard errno == EINTR else {
                throw ModelDigestCacheError.ioFailure(
                    "cannot lock " + lockPath + " (errno " + String(errno) + ")")
            }
        }
        defer { _ = flock(fd, LOCK_UN) }
        return try body()
    }

    private func readRecord(key: String, canonical: String) throws -> ModelDigestRecord? {
        let url = directory.appendingPathComponent(key + ".json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data: Data
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            data = try handle.read(upToCount: Self.maxRecordBytes + 1) ?? Data()
        } catch {
            throw ModelDigestCacheError.ioFailure(
                "cannot read " + url.path + ": " + String(describing: error))
        }
        func malformed(_ reason: String) -> ModelDigestCacheError {
            .malformedRecord(url.path + " " + reason + "; " + Self.refreshInstruction)
        }
        guard data.count <= Self.maxRecordBytes else { throw malformed("exceeds 64 KiB") }
        let record: ModelDigestRecord
        do { record = try JSONDecoder().decode(ModelDigestRecord.self, from: data) }
        catch {
            throw malformed("is undecodable or carries a retired policy ("
                            + String(describing: error) + ")")
        }
        guard record.isWellFormed else { throw malformed("is not well formed") }
        guard record.path == canonical else {
            throw malformed("belongs to " + record.path + ", not " + canonical)
        }
        return record
    }

    private func writeRecord(_ record: ModelDigestRecord, key: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        let url = directory.appendingPathComponent(key + ".json")
        do { try encoder.encode(record).write(to: url, options: [.atomic]) }
        catch {
            throw ModelDigestCacheError.ioFailure(
                "cannot write " + url.path + ": " + String(describing: error))
        }
    }
}
