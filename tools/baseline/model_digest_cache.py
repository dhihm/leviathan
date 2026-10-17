"""Shared persisted model SHA-256 cache (stdlib only).

Same on-disk format, key algorithm, and locking as the Swift
ModelDigestCache (Sources/LeviathanCore/ModelDigestCache.swift), policy
cached-explicit-refresh-v1:
  directory: $LEVIATHAN_MODEL_DIGEST_CACHE_DIR, else
             ~/Library/Caches/Leviathan/model-digests
  canonical path: Unicode NFC of os.path.realpath(os.path.abspath(path))
  key: lowercase hex SHA-256 of the UTF-8 canonical path
  record: <key>.json with exactly bytes, format, hashedAtUTC, path, policy,
          sha256, source
  lock: <key>.lock held with fcntl.flock(LOCK_EX) around every read/write
  writes: atomic (temporary file + os.replace); reads bounded to 64 KiB
The model is hashed in bounded chunks only on a cache miss (initial) or an
explicit refresh (refresh=True or LEVIATHAN_REHASH_MODEL=1). A hit reads no
model bytes and does no stat/mtime/size check: it is a historical digest,
not a fresh verification. A malformed, mismatched, or retired-policy
record raises MalformedRecord with an explicit refresh instruction; it is
never silently rehashed and never returned as a digest.
"""
import datetime
import fcntl
import hashlib
import json
import os
import tempfile
import unicodedata

FORMAT = "leviathan.model-digest-cache.v1"
POLICY = "cached-explicit-refresh-v1"
SOURCES = ("initial", "explicit-refresh", "historical-receipt")
DIR_ENV = "LEVIATHAN_MODEL_DIGEST_CACHE_DIR"
REHASH_ENV = "LEVIATHAN_REHASH_MODEL"
MAX_RECORD_BYTES = 64 << 10
CHUNK_BYTES = 1 << 23
REFRESH_INSTRUCTION = ("refresh explicitly with --rehash-model (leviathan CLI) "
                       "or LEVIATHAN_REHASH_MODEL=1 (Python tools); the record "
                       "is not used as a digest")
HEX = set("0123456789abcdef")


class MalformedRecord(Exception):
    """A cache record that must not be used as a digest."""


def default_directory():
    custom = os.environ.get(DIR_ENV)
    if custom:
        return custom
    return os.path.join(os.path.expanduser("~"), "Library", "Caches",
                        "Leviathan", "model-digests")


def canonical_path(path):
    """realpath of the absolute path, normalized to Unicode NFC so NFC and
    NFD spellings, existing or deleted, share one key with the Swift cache."""
    return unicodedata.normalize(
        "NFC", os.path.realpath(os.path.abspath(os.fspath(path))))


def cache_key(path):
    return hashlib.sha256(canonical_path(path).encode("utf-8")).hexdigest()


def record_path(path, directory=None):
    return os.path.join(directory or default_directory(), cache_key(path) + ".json")


def _well_formed(record, canonical):
    return (isinstance(record, dict)
            and record.get("format") == FORMAT
            and record.get("path") == canonical
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            and set(record["sha256"]) <= HEX
            and type(record.get("bytes")) is int
            and record["bytes"] >= 0
            and isinstance(record.get("hashedAtUTC"), str)
            and bool(record["hashedAtUTC"])
            and record.get("policy") == POLICY
            and record.get("source") in SOURCES)


def _read(rec_path, canonical):
    try:
        f = open(rec_path, "rb")
    except FileNotFoundError:
        return None
    with f:
        data = f.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise MalformedRecord(f"{rec_path} exceeds 64 KiB; {REFRESH_INSTRUCTION}")
    try:
        record = json.loads(data)
    except ValueError as e:
        raise MalformedRecord(f"{rec_path} is undecodable ({e}); {REFRESH_INSTRUCTION}")
    if not _well_formed(record, canonical):
        raise MalformedRecord(
            f"{rec_path} is malformed, mismatched or carries a retired policy; "
            f"{REFRESH_INSTRUCTION}")
    return record


def _sha256_file(path):
    """Bounded streaming SHA-256; the only place model bytes are read."""
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(rec_path, directory, record):
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".model-digest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, rec_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def lookup(path, directory=None, refresh=None):
    """Returns (record, origin); origin is "cache-hit",
    "initial-computation", or "explicit-refresh". refresh=None means
    LEVIATHAN_REHASH_MODEL=1 requests an explicit refresh."""
    if refresh is None:
        refresh = os.environ.get(REHASH_ENV) == "1"
    directory = directory or default_directory()
    canonical = canonical_path(path)
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    os.makedirs(directory, exist_ok=True)
    rec_path = os.path.join(directory, key + ".json")
    with open(os.path.join(directory, key + ".lock"), "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not refresh:
                existing = _read(rec_path, canonical)
                if existing is not None:
                    return existing, "cache-hit"
            # Open the caller's spelling: the NFC canonical path may not be
            # the on-disk name on a normalization-sensitive volume.
            sha256, size = _sha256_file(os.fspath(path))
            record = {"bytes": size, "format": FORMAT, "hashedAtUTC": _utc_now(),
                      "path": canonical, "policy": POLICY, "sha256": sha256,
                      "source": "explicit-refresh" if refresh else "initial"}
            _write(rec_path, directory, record)
            return record, ("explicit-refresh" if refresh else "initial-computation")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
