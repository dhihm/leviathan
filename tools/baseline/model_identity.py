"""Model identity record for the Ollama baseline harnesses (stdlib only).

Policy cached-explicit-refresh-v1: the GGUF SHA-256 comes from the shared
persisted model digest cache (model_digest_cache.py, the same format the
Swift ModelDigestCache writes). The GGUF is hashed only on an initial
cache miss or an explicit LEVIATHAN_REHASH_MODEL=1 refresh; a cache hit
reads no model bytes and is a historical digest, not a fresh
verification. The Ollama parent_model blob is never opened or hashed: the
digest in its file name is declared metadata only, and its equality with
the cached GGUF digest is reported, never certified as the current bytes.
"""
import os

from model_digest_cache import POLICY, lookup

MODEL_FILE_VERIFICATION = POLICY
BLOB_PREFIX = "sha256-"


def model_identity_record(gguf_path, blob_path, gguf_record, origin):
    """Builds the identity record from a cache record. Pure: only string
    operations on the given values; no file access."""
    blob_name = os.path.basename(blob_path)
    declared = (blob_name[len(BLOB_PREFIX):]
                if blob_name.startswith(BLOB_PREFIX) else None)
    return {
        "model_file_verification": MODEL_FILE_VERIFICATION,
        "gguf_sha256": gguf_record["sha256"],
        "gguf_sha256_origin": origin,
        "gguf_sha256_hashed_at_utc": gguf_record["hashedAtUTC"],
        "gguf_sha256_source": gguf_record["source"],
        "gguf_sha256_bytes_at_hash_time": gguf_record["bytes"],
        "compared_weights_identity": ("gguf sha256 from the shared digest cache "
                                      "(historical computation, not re-verified "
                                      "this run); ollama blob not hashed"),
        "declared_metadata": {
            "gguf_path": gguf_path,
            "ollama_parent_model_path": blob_path,
            "ollama_blob_name_digest": declared,
            "blob_name_digest_equals_cached_gguf_sha256": (
                None if declared is None else declared == gguf_record["sha256"]),
        },
        "note": ("the GGUF digest is a cached historical SHA-256 (refresh with "
                 "LEVIATHAN_REHASH_MODEL=1); the blob-name digest is declared "
                 "metadata and does not certify the blob's current bytes"),
    }


def resolve_model_identity(gguf_path, blob_path, directory=None, refresh=None):
    """Looks the GGUF up in the shared cache (hashing only on a miss or an
    explicit refresh) and builds the identity record."""
    record, origin = lookup(gguf_path, directory=directory, refresh=refresh)
    return model_identity_record(gguf_path, blob_path, record, origin)
