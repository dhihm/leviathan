"""Pure regression tests: the Ollama baseline harnesses take the GGUF
SHA-256 from the shared digest cache and read no model bytes on a cache hit
(policy cached-explicit-refresh-v1).

HTTP, subprocess, and open() are mocked. No network request, process
enumeration, model unload, or benchmark runs: each main() call stops at the
preflight "model already loaded" exit, after the identity record is built
and before any measured request.
Run with: python3 -m unittest discover -s tools/baseline -p 'test_*.py'
"""
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_digest_cache  # noqa: E402
import model_identity  # noqa: E402


def load_script(name):
    """Imports a harness while network and subprocess access raise, so a
    missing __main__ guard would fail here instead of contacting anything."""
    spec = importlib.util.spec_from_file_location(
        name + "_under_test", os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("network access at import")))
        stack.enter_context(mock.patch(
            "subprocess.run",
            side_effect=AssertionError("subprocess at import")))
        spec.loader.exec_module(module)
    return module


class GuardedOpen:
    """open() replacement that records and refuses any open of the guarded
    model paths and delegates every other path to the real open()."""

    def __init__(self, guarded):
        self.real = builtins.open
        self.guarded = {os.path.realpath(p) for p in guarded}
        self.attempts = []

    def __call__(self, file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(file)) in self.guarded:
                self.attempts.append(os.fsdecode(file))
                raise AssertionError(f"model file opened: {file!r}")
        return self.real(file, *args, **kwargs)


def fake_urlopen(blob_path, urls):
    def urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        urls.append(url)
        if url.endswith("/api/version"):
            body = {"version": "test"}
        elif url.endswith("/api/show"):
            body = {"details": {"parent_model": blob_path}}
        elif url.endswith("/api/ps"):
            # Reports the model as loaded, so main() stops at its preflight
            # exit before any measured or unload request.
            body = {"models": [{"name": "gemma4-task5"}]}
        else:
            raise AssertionError(f"unexpected request {url}")
        return io.BytesIO(json.dumps(body).encode())
    return urlopen


def fake_run(args, **kwargs):
    """Answers the resident leviathan-cli check without enumerating
    processes; any other command fails the test."""
    if list(args[:2]) != ["pgrep", "-fl"]:
        raise AssertionError(f"unexpected subprocess {args!r}")
    return subprocess.CompletedProcess(args, 1, stdout="", stderr="")


PLANTED_SHA = "5e" * 32


def cache_record(path, sha, source="historical-receipt"):
    return {"bytes": 14439363584, "format": model_digest_cache.FORMAT,
            "hashedAtUTC": "2026-10-04T00:00:00Z",
            "path": model_digest_cache.canonical_path(path),
            "policy": model_digest_cache.POLICY, "sha256": sha, "source": source}


def plant_cache_record(path, directory, sha):
    os.makedirs(directory, exist_ok=True)
    with open(model_digest_cache.record_path(path, directory), "w") as f:
        json.dump(cache_record(path, sha), f, sort_keys=True)


class ModelIdentityRecordTests(unittest.TestCase):
    def test_record_never_touches_the_model_files(self):
        blob = "/nonexistent/blobs/sha256-" + "cd" * 32
        cached = cache_record("/nonexistent/model.gguf", "cd" * 32)
        with contextlib.ExitStack() as stack:
            for target in ("builtins.open", "os.open", "os.stat"):
                stack.enter_context(mock.patch(
                    target, side_effect=AssertionError(target)))
            record = model_identity.model_identity_record(
                "/nonexistent/model.gguf", blob, cached, "cache-hit")
        self.assertEqual(record["model_file_verification"], "cached-explicit-refresh-v1")
        self.assertEqual(record["gguf_sha256"], "cd" * 32)
        self.assertEqual(record["gguf_sha256_origin"], "cache-hit")
        self.assertEqual(record["declared_metadata"]["ollama_blob_name_digest"],
                         "cd" * 32)
        self.assertTrue(
            record["declared_metadata"]["blob_name_digest_equals_cached_gguf_sha256"])

    def test_unprefixed_blob_name_declares_no_digest(self):
        record = model_identity.model_identity_record(
            "model.gguf", "/blobs/model", cache_record("model.gguf", PLANTED_SHA),
            "cache-hit")
        self.assertIsNone(record["declared_metadata"]["ollama_blob_name_digest"])
        self.assertIsNone(
            record["declared_metadata"]["blob_name_digest_equals_cached_gguf_sha256"])


class HarnessPreflightTests(unittest.TestCase):
    def run_preflight(self, name):
        module = load_script(name)
        self.assertFalse(hasattr(module, "sha256_file"))
        content = b"placeholder model bytes; never read"
        urls, out = [], io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            gguf = os.path.join(tmp, "model.gguf")
            blob = os.path.join(tmp, "sha256-" + "ab" * 32)
            for path in (gguf, blob):
                with open(path, "wb") as f:
                    f.write(content)
            args = [os.path.join(tmp, "out.json")]
            if name == "ollama_fairdecode":
                args.append(os.path.join(tmp, "daemon.log"))
            cache_dir = os.path.join(tmp, "digest-cache")
            plant_cache_record(gguf, cache_dir, PLANTED_SHA)
            guard = GuardedOpen([gguf, blob])
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {
                    "GGUF": gguf, model_digest_cache.DIR_ENV: cache_dir}))
                os.environ.pop(model_digest_cache.REHASH_ENV, None)
                stack.enter_context(mock.patch(
                    "urllib.request.urlopen", side_effect=fake_urlopen(blob, urls)))
                stack.enter_context(mock.patch("subprocess.run", side_effect=fake_run))
                stack.enter_context(mock.patch("builtins.open", guard))
                stack.enter_context(contextlib.redirect_stdout(out))
                with self.assertRaises(SystemExit) as stop:
                    module.main(*args)
            self.assertFalse(os.path.exists(args[0]))
        content_sha = hashlib.sha256(content).hexdigest()
        return stop.exception, guard, urls, out.getvalue(), content_sha

    def check_harness(self, name):
        stop, guard, urls, stdout, content_sha = self.run_preflight(name)
        self.assertIn("model already loaded", str(stop.code))
        self.assertEqual(guard.attempts, [])
        self.assertTrue(urls and urls[-1].endswith("/api/ps"))
        self.assertFalse(any(u.endswith("/api/generate") for u in urls))
        records = [json.loads(line)["model_file_identity"]
                   for line in stdout.splitlines()
                   if line.startswith("{") and "model_file_identity" in line]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["model_file_verification"], "cached-explicit-refresh-v1")
        self.assertEqual(record["gguf_sha256"], PLANTED_SHA)
        self.assertEqual(record["gguf_sha256_origin"], "cache-hit")
        self.assertEqual(record["gguf_sha256_source"], "historical-receipt")
        self.assertEqual(record["declared_metadata"]["ollama_blob_name_digest"],
                         "ab" * 32)
        self.assertFalse(
            record["declared_metadata"]["blob_name_digest_equals_cached_gguf_sha256"])
        self.assertNotIn(content_sha, stdout)

    def test_fairdecode_preflight_reads_no_model_bytes(self):
        self.check_harness("ollama_fairdecode")

    def test_warmcold_preflight_reads_no_model_bytes(self):
        self.check_harness("ollama_warmcold")


if __name__ == "__main__":
    unittest.main()
