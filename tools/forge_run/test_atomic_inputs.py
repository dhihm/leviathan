#!/usr/bin/env python3
"""Focused tests: build publishes only after the real supplied-input run and
parity pass (atomic, exclusive), plus shared early input validation.
Install before the orchestrator fix to see RED; apply the fix to see GREEN.
    swift build -c release --product leviathan-forge-helper
    python3.11 tools/forge_run/test_atomic_inputs.py -v
LEVIATHAN_FORGE_KEEP_TEST_OUTPUT=1 keeps every output. Mocks are used only
for targeted fault injection; the GPU and parity proofs use the real helper.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

TOOL_DIR = Path(__file__).resolve().parent
ORCHESTRATOR = TOOL_DIR / "leviathan_forge.py"
sys.path.insert(0, str(TOOL_DIR))
import leviathan_forge as forge  # noqa: E402

HELPER = None
MODEL = '''import os
import torch
from torch import nn
INPUT_DIM = 8
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 4)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))
def build(seed):
    log = os.environ.get("LEVIATHAN_TEST_BUILD_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write("build")
    torch.manual_seed(seed)
    return Tiny()
'''


def setUpModule():
    global HELPER
    HELPER = str(forge.resolve_helper())


def snapshot(root):
    return sorted((str(p.relative_to(root)), p.read_bytes() if p.is_file() else None) for p in Path(root).rglob("*"))


class AtomicPublicationAndInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leviathan-forge-atomic-"))
        cls.model = cls.tmp / "tiny_model.py"
        cls.model.write_text(MODEL)
        cls.example, cls.input, cls.input2 = (cls.write_json(n, s) for n, s in (("example", 1), ("input", 2), ("input2", 3)))
        cls.real_run_helper = staticmethod(forge.run_helper)  # plain callable: no self binding
        cls.good = cls.tmp / "good_bundle"
        cls.good_report = forge.build_bundle(str(cls.model), str(cls.example), str(cls.input), str(cls.good), helper=HELPER)

    @classmethod
    def write_json(cls, name, seed):
        path = cls.tmp / f"{name}.json"
        values = (torch.rand((1, 8), generator=torch.Generator().manual_seed(seed)) * 2 - 1).tolist()
        path.write_text(json.dumps(values))
        return path

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("LEVIATHAN_FORGE_KEEP_TEST_OUTPUT"):
            print(f"kept test output in {cls.tmp}")
        else:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def workdir(self, tag):
        return Path(tempfile.mkdtemp(prefix=tag + "-", dir=self.tmp))

    def failed_evidence(self, parent, out):
        return [p for p in parent.iterdir() if p.is_dir() and p != out and "fail" in p.name.lower()
                and (p / "bundle.json").is_file()]

    def build_with(self, wrapper, parent, **kwargs):
        out = parent / "bundle"
        calls = []

        def run_helper(helper, args):
            calls.append(args[0])
            return wrapper(helper, args, out)
        with mock.patch.object(forge, "run_helper", side_effect=run_helper):
            with self.assertRaises(forge.ForgeRunError) as ctx:
                forge.build_bundle(str(self.model), str(self.example), str(self.input), str(out), helper=HELPER, **kwargs)
        return out, calls, str(ctx.exception)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ORCHESTRATOR), *map(str, args)],
                              capture_output=True, text=True, timeout=900)

    def test_parity_failure_after_real_helper_run_publishes_nothing(self):
        parent = self.workdir("parity")

        def wrapper(helper, args, out):
            result = self.real_run_helper(helper, args)
            if args[0] == "run":
                path = Path(args[args.index("--output-json") + 1])
                data = json.loads(path.read_text())
                data["output"] = [v + 0.5 for v in data["output"]]  # finite, wrong
                path.write_text(json.dumps(data))
            return result
        out, calls, message = self.build_with(wrapper, parent)
        self.assertEqual(calls, ["forge", "run"])  # both real helper invocations happened
        self.assertIn("GPU output disagrees with the source PyTorch module", message)
        self.assertNotIn("TypeError", message)
        self.assertFalse(out.exists())
        self.assertTrue(self.failed_evidence(parent, out))

    def test_late_helper_failure_publishes_nothing_and_keeps_evidence(self):
        parent = self.workdir("late")

        def wrapper(helper, args, out):
            if args[0] == "run":
                raise forge.ForgeRunError("injected late helper failure")
            return self.real_run_helper(helper, args)
        out, _, message = self.build_with(wrapper, parent)
        self.assertIn("injected late helper failure", message)
        self.assertFalse(out.exists())
        self.assertTrue(self.failed_evidence(parent, out))

    def test_output_created_during_run_is_not_clobbered(self):
        parent = self.workdir("race-run")

        def wrapper(helper, args, out):
            result = self.real_run_helper(helper, args)
            if args[0] == "run":
                out.mkdir(exist_ok=True)  # another process creates an empty dir
            return result
        out, _, _ = self.build_with(wrapper, parent)
        self.assertTrue(out.is_dir())
        self.assertEqual(list(out.iterdir()), [])
        self.assertTrue(self.failed_evidence(parent, out))

    def test_foreign_output_created_during_forge_is_untouched(self):
        parent = self.workdir("race-forge")

        def wrapper(helper, args, out):
            result = self.real_run_helper(helper, args)
            if args[0] == "forge":
                out.mkdir()
                (out / "theirs.txt").write_text("other process")
            return result
        out, _, _ = self.build_with(wrapper, parent)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["theirs.txt"])
        self.assertEqual((out / "theirs.txt").read_text(), "other process")
        self.assertTrue(self.failed_evidence(parent, out))

    def test_existing_valid_bundle_bytes_unchanged(self):
        before = snapshot(self.good)
        with self.assertRaises(forge.ForgeRunError):
            forge.build_bundle(str(self.model), str(self.example), str(self.input2), str(self.good), helper=HELPER)
        self.assertEqual(snapshot(self.good), before)

    def test_happy_build_then_fresh_process_reuse(self):
        report = self.good_report
        self.assertEqual(report["bundle"], str(self.good))
        self.assertEqual(report["run"]["bundle"], str(self.good))
        self.assertTrue(report["run"]["parity"]["passed"])
        self.assertGreater(report["run"]["gpuEvidence"]["gpuCommandBufferMicros"], 0)
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.startswith(".")], [])
        proc = self.cli("run", self.good, "--model", self.model, "--input", self.input2, "--helper", HELPER)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        run = json.loads(proc.stdout)
        self.assertTrue(run["parity"]["passed"])
        self.assertNotEqual(run["output"], report["run"]["output"])
        self.assertEqual(run["selected"], report["run"]["selected"])

    def test_invalid_tensor_inputs_rejected_before_expensive_work(self):
        parent = self.workdir("invalid")
        log = self.tmp / "invalid-build.log"
        cases = {
            "float64": torch.zeros(1, 8, dtype=torch.float64),
            "int32": torch.zeros(1, 8, dtype=torch.int32),
            "bool": torch.zeros(1, 8, dtype=torch.bool),
            "nan": torch.tensor([[float("nan")] + [0.0] * 7]),
            "inf": torch.tensor([[float("inf")] + [0.0] * 7]),
            "meta": torch.empty(1, 8, device="meta"),
            "sparse": torch.zeros(1, 8).to_sparse(),
            "rank1": torch.zeros(8),
            "rank3": torch.zeros(1, 1, 8),
            "two_rows": torch.zeros(2, 8),
            "empty_width": torch.zeros(1, 0),
            "oversize": torch.zeros(1, forge.MAX_DIM + 1),
            "wrong_width": torch.zeros(1, 5),
            "not_tensor": [[0.0] * 8],
        }
        good_before = snapshot(self.good)
        for name, bad in cases.items():
            for phase in ("build-example", "build-input", "run"):
                with self.subTest(case=name, phase=phase):
                    log.unlink(missing_ok=True)
                    before = snapshot(parent)
                    export = mock.Mock(side_effect=AssertionError("capture reached"))
                    helper = mock.Mock(side_effect=AssertionError("helper reached"))
                    with mock.patch.object(torch.export, "export", export), \
                            mock.patch.object(forge, "run_helper", helper), \
                            mock.patch.dict(os.environ, {"LEVIATHAN_TEST_BUILD_LOG": str(log)}):
                        with self.assertRaises(forge.ForgeRunError):
                            if phase == "run":
                                forge.run_bundle(str(self.good), str(self.model), bad, helper=HELPER)
                            else:
                                ok = torch.zeros(1, 8)
                                ex, inp = (bad, ok) if phase == "build-example" else (ok, bad)
                                forge.build_bundle(str(self.model), ex, inp, str(parent / f"out-{name}-{phase}"),
                                                   helper=HELPER)
                    self.assertFalse(export.called)
                    self.assertFalse(helper.called)
                    self.assertEqual(snapshot(parent), before)
                    if name != "wrong_width":
                        self.assertFalse(log.exists(), "model build(seed) ran before an independent input check")
        self.assertEqual(snapshot(self.good), good_before)

    def test_cli_report_success(self):
        parent = self.workdir("report-ok")
        out, report = parent / "bundle", parent / "report.json"
        proc = self.cli("build", self.model, "--example-input", self.example, "--input", self.input,
                        "--out", out, "--helper", HELPER, "--report", report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        written = json.loads(report.read_text())
        self.assertEqual(written, json.loads(proc.stdout))
        self.assertEqual(written["run"]["bundle"], str(out))
        self.assertTrue((out / "bundle.json").is_file())
        self.assertEqual(sorted(p.name for p in parent.iterdir()), ["bundle", "report.json"])

    def test_cli_report_failure_leaves_no_misleading_partial_build(self):
        parent = self.workdir("report-bad")
        out = parent / "bundle"
        proc = self.cli("build", self.model, "--example-input", self.example, "--input", self.input,
                        "--out", out, "--helper", HELPER, "--report", parent / "missing" / "report.json")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("report", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(out.exists())

    def test_report_created_during_build_is_not_overwritten(self):
        parent = self.workdir("report-race")
        out, report = parent / "bundle", parent / "report.json"

        def run_helper(helper, args):
            result = self.real_run_helper(helper, args)
            if args[0] == "run" and not report.exists():
                report.write_text("other process")
            return result
        sink = io.StringIO()
        with mock.patch.object(forge, "run_helper", side_effect=run_helper), \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            code = forge.main(["build", str(self.model), "--example-input", str(self.example), "--input",
                               str(self.input), "--out", str(out), "--helper", HELPER, "--report", str(report)])
        self.assertEqual(code, 2, sink.getvalue())
        self.assertIn("report", sink.getvalue())
        self.assertIn("NOT published", sink.getvalue())
        self.assertEqual(report.read_text(), "other process")
        self.assertFalse(out.exists())
        self.assertTrue(self.failed_evidence(parent, out))
        self.assertEqual([p.name for p in parent.iterdir() if "partial" in p.name], [])

    def test_injected_report_link_failure_publishes_nothing(self):
        parent = self.workdir("report-link")
        out, report = parent / "bundle", parent / "report.json"
        with mock.patch.object(forge.os, "link", side_effect=OSError("injected report link failure")):
            with self.assertRaises(forge.ForgeRunError) as ctx:
                forge.build_bundle(str(self.model), str(self.example), str(self.input), str(out),
                                   helper=HELPER, report_path=str(report))
        self.assertIn("injected report link failure", str(ctx.exception))
        self.assertFalse(out.exists())
        self.assertFalse(report.exists())
        self.assertTrue(self.failed_evidence(parent, out))
        self.assertEqual([p.name for p in parent.iterdir() if "partial" in p.name], [])

    def test_output_race_with_report_removes_own_report_keeps_foreign_output(self):
        parent = self.workdir("report-out-race")
        report = parent / "report.json"

        def wrapper(helper, args, out):
            result = self.real_run_helper(helper, args)
            if args[0] == "run":
                out.mkdir()
                (out / "theirs.txt").write_text("other process")
            return result
        out, calls, message = self.build_with(wrapper, parent, report_path=str(report))
        self.assertEqual(calls, ["forge", "run"])
        self.assertIn(f"own report {report} removed", message)
        self.assertFalse(report.exists())
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["theirs.txt"])
        self.assertEqual((out / "theirs.txt").read_text(), "other process")
        self.assertTrue(self.failed_evidence(parent, out))
        self.assertEqual([p.name for p in parent.iterdir() if "partial" in p.name], [])


if __name__ == "__main__":
    unittest.main()
