"""Pure tests for the shared model digest cache (policy
cached-explicit-refresh-v1). Small temporary files only: no model file,
network request, benchmark, process enumeration, or model unload.
Run with: python3 -m unittest discover -s tools/baseline -p 'test_*.py'
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_digest_cache as mdc  # noqa: E402
import model_identity  # noqa: E402

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
ABCD = "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"


class ModelDigestCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.cache = os.path.join(self.root, "cache")
        self.model = os.path.join(self.root, "model.bin")
        self.write_model(b"abc")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(mdc.REHASH_ENV, None)
        os.environ.pop(mdc.DIR_ENV, None)

    def write_model(self, data):
        with open(self.model, "wb") as f:
            f.write(data)

    def lookup(self, **kwargs):
        return mdc.lookup(self.model, directory=self.cache, **kwargs)

    def valid_record(self, **changes):
        record = {"bytes": 3, "format": mdc.FORMAT,
                  "hashedAtUTC": "2026-10-04T00:00:00Z",
                  "path": mdc.canonical_path(self.model),
                  "policy": mdc.POLICY, "sha256": ABC, "source": "initial"}
        record.update(changes)
        return record

    def plant(self, text):
        os.makedirs(self.cache, exist_ok=True)
        with open(mdc.record_path(self.model, self.cache), "w") as f:
            f.write(text)

    def test_initial_miss_hashes_the_actual_bytes(self):
        record, origin = self.lookup()
        self.assertEqual(origin, "initial-computation")
        self.assertEqual(record["sha256"], ABC)
        self.assertEqual(record["bytes"], 3)
        self.assertEqual(record["source"], "initial")
        self.assertEqual(record["policy"], "cached-explicit-refresh-v1")

    def test_hit_after_mutation_and_deletion_reads_nothing(self):
        first, _ = self.lookup()
        self.write_model(b"abcd")
        with mock.patch.object(mdc, "_sha256_file",
                               side_effect=AssertionError("rehashed")):
            record, origin = self.lookup()
            self.assertEqual((origin, record["sha256"]), ("cache-hit", ABC))
            self.assertEqual(record["hashedAtUTC"], first["hashedAtUTC"])
            os.remove(self.model)
            record, origin = self.lookup()
            self.assertEqual((origin, record["sha256"]), ("cache-hit", ABC))

    def test_explicit_refresh_persists_across_processes(self):
        self.lookup()
        self.write_model(b"abcd")
        record, origin = self.lookup(refresh=True)
        self.assertEqual((origin, record["sha256"], record["source"]),
                         ("explicit-refresh", ABCD, "explicit-refresh"))
        code = ("import sys; sys.path.insert(0, " + repr(HERE) + "); "
                "import model_digest_cache as m; "
                "r, o = m.lookup(" + repr(self.model) + ", directory="
                + repr(self.cache) + ", refresh=False); print(o, r['sha256'])")
        child = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=60, check=True)
        self.assertEqual(child.stdout.split(), ["cache-hit", ABCD])

    def test_environment_requests_explicit_refresh(self):
        self.lookup()
        self.write_model(b"abcd")
        os.environ[mdc.REHASH_ENV] = "1"
        record, origin = self.lookup()
        self.assertEqual((origin, record["sha256"]), ("explicit-refresh", ABCD))

    def test_nested_lookups_hash_once(self):
        real = mdc._sha256_file
        with mock.patch.object(mdc, "_sha256_file", side_effect=real) as spy:
            for _ in range(4):
                self.lookup()
        self.assertEqual(spy.call_count, 1)

    def test_historical_receipt_is_reused_without_reading(self):
        self.plant(json.dumps(self.valid_record(sha256="5e" * 32, bytes=14439363584,
                                                source="historical-receipt")))
        with mock.patch.object(mdc, "_sha256_file",
                               side_effect=AssertionError("rehashed")):
            record, origin = self.lookup()
        self.assertEqual((origin, record["sha256"], record["source"]),
                         ("cache-hit", "5e" * 32, "historical-receipt"))

    def test_record_format_and_key(self):
        self.lookup()
        canonical = os.path.realpath(self.model)
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(mdc.cache_key(self.model), key)
        path = os.path.join(self.cache, key + ".json")
        self.assertEqual(mdc.record_path(self.model, self.cache), path)
        with open(path) as f:
            record = json.load(f)
        self.assertEqual(sorted(record), ["bytes", "format", "hashedAtUTC", "path",
                                          "policy", "sha256", "source"])
        self.assertEqual(record["format"], "leviathan.model-digest-cache.v1")
        self.assertEqual(record["path"], canonical)

    def test_symlink_shares_the_record(self):
        link = os.path.join(self.root, "link.bin")
        os.symlink(self.model, link)
        self.lookup()
        record, origin = mdc.lookup(link, directory=self.cache)
        self.assertEqual((origin, record["sha256"]), ("cache-hit", ABC))

    def unicode_spellings(self):
        name = "mod\u00e8le.bin"
        return (os.path.join(self.root, unicodedata.normalize("NFC", name)),
                os.path.join(self.root, unicodedata.normalize("NFD", name)))

    def test_unicode_spellings_share_one_nfc_record_across_deletion(self):
        nfc, nfd = self.unicode_spellings()
        with open(nfc, "wb") as f:
            f.write(b"abc")
        canonical = mdc.canonical_path(nfd)
        self.assertEqual(canonical, unicodedata.normalize("NFC", canonical))
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual((mdc.cache_key(nfc), mdc.cache_key(nfd)), (key, key))
        record, origin = mdc.lookup(nfc, directory=self.cache)
        self.assertEqual((origin, record["sha256"], record["path"]),
                         ("initial-computation", ABC, canonical))
        with open(nfc, "wb") as f:
            f.write(b"abcd")
        with mock.patch.object(mdc, "_sha256_file",
                               side_effect=AssertionError("rehashed")):
            record, origin = mdc.lookup(nfd, directory=self.cache)
            self.assertEqual((origin, record["sha256"]), ("cache-hit", ABC))
            os.remove(nfc)
            for spelling in (nfc, nfd):
                self.assertEqual(mdc.cache_key(spelling), key)
                record, origin = mdc.lookup(spelling, directory=self.cache)
                self.assertEqual((origin, record["sha256"]), ("cache-hit", ABC))

    def test_swift_written_unicode_record_hits_after_deletion(self):
        nfc, nfd = self.unicode_spellings()
        with open(nfc, "wb") as f:
            f.write(b"abcd")
        canonical = mdc.canonical_path(nfc)
        os.remove(nfc)
        self.assertEqual(mdc.canonical_path(nfd), canonical)
        record = self.valid_record(path=canonical, sha256=ABCD, bytes=4)
        # Mirrors Swift JSONEncoder output: sorted keys, " : " separators,
        # raw UTF-8, unescaped slashes.
        text = "{\n" + ",\n".join(
            "  " + json.dumps(k) + " : " + json.dumps(v, ensure_ascii=False)
            for k, v in sorted(record.items())) + "\n}"
        os.makedirs(self.cache, exist_ok=True)
        with open(mdc.record_path(nfd, self.cache), "w", encoding="utf-8") as f:
            f.write(text)
        with mock.patch.object(mdc, "_sha256_file",
                               side_effect=AssertionError("rehashed")):
            record, origin = mdc.lookup(nfd, directory=self.cache)
        self.assertEqual((origin, record["sha256"], record["path"]),
                         ("cache-hit", ABCD, canonical))

    def test_malformed_records_require_explicit_refresh(self):
        variants = [
            "{not json",
            json.dumps(self.valid_record(policy="none-unverified-v1")),
            json.dumps(self.valid_record(path=mdc.canonical_path(self.model) + ".other")),
            json.dumps(self.valid_record(sha256="xyz")),
            json.dumps(self.valid_record(source="guessed")),
        ]
        for text in variants:
            self.plant(text)
            with mock.patch.object(mdc, "_sha256_file",
                                   side_effect=AssertionError("silent rehash")):
                with self.assertRaises(mdc.MalformedRecord) as err:
                    self.lookup()
            self.assertIn("--rehash-model", str(err.exception))
        record, origin = self.lookup(refresh=True)
        self.assertEqual((origin, record["sha256"]), ("explicit-refresh", ABC))

    def test_custom_directories_are_isolated(self):
        self.lookup()
        other = os.path.join(self.root, "other-cache")
        self.assertFalse(os.path.exists(mdc.record_path(self.model, other)))
        _, origin = mdc.lookup(self.model, directory=other)
        self.assertEqual(origin, "initial-computation")

    def test_identity_record_from_cache(self):
        identity = model_identity.resolve_model_identity(
            self.model, "/blobs/sha256-" + ABC, directory=self.cache)
        self.assertEqual(identity["model_file_verification"], "cached-explicit-refresh-v1")
        self.assertEqual(identity["gguf_sha256"], ABC)
        self.assertEqual(identity["gguf_sha256_origin"], "initial-computation")
        self.assertTrue(
            identity["declared_metadata"]["blob_name_digest_equals_cached_gguf_sha256"])


if __name__ == "__main__":
    unittest.main()
