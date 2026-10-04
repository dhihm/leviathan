#!/usr/bin/env python3
"""Acceptance tests (real Mac + Metal). First:
    swift build -c release --product leviathan-forge-helper
    python3.11 tools/forge_run/test_leviathan_forge.py -v
"""

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
ORCHESTRATOR = TOOL_DIR / "leviathan_forge.py"
MODEL_A = TOOL_DIR / "examples" / "mlp_tower.py"
MODEL_B = TOOL_DIR / "examples" / "bottleneck_stack.py"
sys.path.insert(0, str(TOOL_DIR))
import leviathan_forge as forge  # noqa: E402

HELPER = None
SIGMOID_MODEL = '''import torch
from torch import nn
INPUT_DIM = 8
class SigmoidHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
def build(seed):
    torch.manual_seed(seed)
    return SigmoidHead()
'''


def setUpModule():
    global HELPER
    HELPER = str(forge.resolve_helper())


def cli(*args, script=ORCHESTRATOR):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, timeout=1800)


def cli_ok(*args, **kwargs):
    proc = cli(*args, **kwargs)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return json.loads(proc.stdout)


def write_input(path, dim, seed, scale=1.0):
    values = ((torch.rand((1, dim), generator=torch.Generator().manual_seed(seed)) * 2 - 1) * scale).tolist()
    Path(path).write_text(json.dumps(values))
    return Path(path)


def tree_digest(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(p.name.encode() + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def load_meta(bundle):
    return json.loads((Path(bundle) / "bundle.json").read_text())


def rewrite_bundle(bundle, mutate):
    meta = load_meta(bundle)
    mutate(meta)
    data = json.dumps(meta, indent=2, sort_keys=True).encode()
    (bundle / "bundle.json").write_bytes(data)
    (bundle / "bundle.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")


def source_reference(model_path, input_path):
    spec = importlib.util.spec_from_file_location("ref_" + Path(model_path).stem, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        return module.build(0).eval()(forge.load_input_json(input_path))


class ForgeRuntimeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leviathan-forge-test-"))
        cls.inputs = {}
        for name, dim in (("a", 128), ("b", 24)):
            for kind, seed, scale in (("example", 101, 1.0), ("input", 202, 1.0), ("input2", 303, 2.0)):
                cls.inputs[(name, kind)] = write_input(cls.tmp / f"{name}.{kind}.json", dim, seed + dim, scale)
        def build(name, model, out, *extra):
            return cli_ok("build", model, "--example-input", cls.inputs[(name, "example")], "--input",
                          cls.inputs[(name, "input")], "--out", out, "--helper", HELPER, *extra)
        cls.bundle_a, cls.bundle_b, cls.bundle_f = cls.tmp / "bundle_a", cls.tmp / "bundle_b", cls.tmp / "bundle_f"
        cls.report_a = build("a", MODEL_A, cls.bundle_a)
        cls.report_b = build("b", MODEL_B, cls.bundle_b)
        cls.report_f = build("a", MODEL_A, cls.bundle_f, "--diagnostic-fail-optimizations")

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("LEVIATHAN_FORGE_KEEP_TEST_OUTPUT"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def cases(self):
        return ((self.report_a, self.bundle_a, MODEL_A, "a"), (self.report_b, self.bundle_b, MODEL_B, "b"))

    def copy_bundle(self, source, tag):
        dest = Path(tempfile.mkdtemp(prefix=tag + "-", dir=self.tmp)) / "bundle"
        shutil.copytree(source, dest)
        return dest

    def run_cli(self, bundle, model=MODEL_A, name="a", **kwargs):
        return cli("run", bundle, "--model", model, "--input", self.inputs[(name, "input2")], "--helper",
                   kwargs.pop("helper", HELPER), **kwargs)

    def helper_direct(self, bundle, dim):
        work = Path(tempfile.mkdtemp(prefix="direct-", dir=self.tmp))
        (work / "in.f32").write_bytes(struct.pack(f"<{dim}f", *([0.25] * dim)))
        proc = subprocess.run([HELPER, "run", "--bundle", str(bundle), "--input-f32", str(work / "in.f32"),
                               "--output-json", str(work / "out.json")], capture_output=True, text=True, timeout=600)
        return proc, work / "out.json"

    def assert_rejected(self, proc, text):
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn(text, proc.stderr)

    def assert_gpu_evidence(self, run, bundle):
        meta = load_meta(bundle)
        gpu, timing, selected = run["gpuEvidence"], run["timing"], meta["selection"]["selected"]
        self.assertEqual(gpu["deviceName"], meta["hardware"]["deviceName"])
        self.assertTrue(gpu["deviceName"])
        self.assertEqual(gpu["commandStatus"], "completed")
        self.assertGreater(gpu["gpuCommandBufferMicros"], 0)
        self.assertEqual(gpu["computeDispatches"], len(gpu["kernelSequence"]))
        self.assertEqual(run["selected"], selected)
        self.assertEqual("leviathan_dense_relu" in gpu["kernelSequence"], selected["fuseDenseRelu"])
        self.assertGreater(timing["endToEndMillis"] * 1000.0, timing["gpuCommandBufferMicros"])
        self.assertIn("GPU only", timing["labels"]["gpuCommandBufferMicros"])

    def test_build_runs_real_input_on_gpu_with_source_parity(self):
        for report, bundle, model, name in self.cases():
            with self.subTest(model=model.name):
                run = report["run"]
                self.assertTrue(run["parity"]["passed"])
                self.assert_gpu_evidence(run, bundle)
                ref = source_reference(model, self.inputs[(name, "input")])
                self.assertTrue(torch.allclose(torch.tensor(run["output"]), ref, atol=1e-4, rtol=1e-4))

    def test_selection_is_baseline_first_and_consistent(self):
        for _, bundle, model, _ in self.cases():
            with self.subTest(model=model.name):
                sel = load_meta(bundle)["selection"]
                base = sel["baseline"]
                self.assertTrue(base["qualified"])
                self.assertFalse(base["candidate"]["fuseDenseRelu"])
                self.assertGreaterEqual(len(sel["optimizations"]), 1)
                if sel["selectedRole"] == "optimization":
                    chosen = next(o for o in sel["optimizations"] if o["candidate"] == sel["selected"])
                    self.assertTrue(chosen["qualified"])
                    self.assertLess(chosen["medianGpuMicros"], base["medianGpuMicros"] * (1 - sel["minRelativeGain"]))
                else:
                    self.assertEqual(sel["selected"], base["candidate"])

    def test_two_examples_are_genuinely_different(self):
        ga = json.loads((self.bundle_a / "model.graph.json").read_text())
        gb = json.loads((self.bundle_b / "model.graph.json").read_text())
        self.assertNotEqual(ga, gb)
        self.assertNotEqual(ga["ops"][-1], "relu")
        self.assertEqual(gb["ops"][-1], "relu")
        self.assertTrue(any(isinstance(x, dict) and isinstance(y, dict) for x, y in zip(gb["ops"], gb["ops"][1:])))

    def test_fresh_process_reload_uses_saved_plan_on_different_input(self):
        for report, bundle, model, name in self.cases():
            with self.subTest(model=model.name):
                before = tree_digest(bundle)
                run = cli_ok("run", bundle, "--model", model, "--input", self.inputs[(name, "input2")], "--helper", HELPER)
                self.assertEqual(tree_digest(bundle), before)
                self.assertNotEqual(run["output"], report["run"]["output"])
                self.assert_gpu_evidence(run, bundle)
                ref = source_reference(model, self.inputs[(name, "input2")])
                self.assertTrue(torch.allclose(torch.tensor(run["output"]), ref, atol=1e-4, rtol=1e-4))

    def test_forced_optimization_failure_falls_back_to_verified_baseline(self):
        sel = load_meta(self.bundle_f)["selection"]
        self.assertTrue(sel["diagnosticFailOptimizations"])
        self.assertGreaterEqual(len(sel["optimizations"]), 1)
        for opt in sel["optimizations"]:
            self.assertFalse(opt["qualified"])
            self.assertTrue(opt["diagnosticForcedFailure"])
        self.assertFalse(sel["baseline"]["diagnosticForcedFailure"])
        self.assertEqual(sel["selectedRole"], "baseline")
        self.assertEqual(sel["selected"], sel["baseline"]["candidate"])
        self.assertIn("fallback", sel["decision"])
        run = self.report_f["run"]
        self.assertTrue(run["parity"]["passed"])
        self.assertNotIn("leviathan_dense_relu", run["gpuEvidence"]["kernelSequence"])

    def test_unsupported_aten_node_is_identified_and_nothing_written(self):
        model = self.tmp / "sigmoid_model.py"
        model.write_text(SIGMOID_MODEL)
        example = write_input(self.tmp / "sigmoid.json", 8, 5)
        out = self.tmp / "bundle_sigmoid"
        proc = cli("build", model, "--example-input", example, "--input", example, "--out", out, "--helper", HELPER)
        self.assert_rejected(proc, "unsupported ATen op 'aten.sigmoid.default'")
        self.assertRegex(proc.stderr, r"at node '\w+'")
        self.assertFalse(out.exists())
        self.assertEqual(list(self.tmp.glob(".leviathan-forge-staging-*")), [])

    def test_corrupt_weights_rejected_by_orchestrator_and_helper(self):
        control, control_out = self.helper_direct(self.bundle_a, 128)
        self.assertEqual(control.returncode, 0, control.stderr)
        self.assertTrue(control_out.exists())
        bad = self.copy_bundle(self.bundle_a, "weights")
        data = bytearray((bad / "model.weights.f32").read_bytes())
        data[len(data) // 2] ^= 0x40
        (bad / "model.weights.f32").write_bytes(bytes(data))
        self.assert_rejected(self.run_cli(bad), "model.weights.f32 sha256 mismatch")
        direct, out = self.helper_direct(bad, 128)
        self.assertNotEqual(direct.returncode, 0)
        self.assertIn("mismatch", direct.stderr.lower())
        self.assertFalse(out.exists())

    def test_corrupt_bundle_json_rejected(self):
        bad = self.copy_bundle(self.bundle_a, "json")
        meta = load_meta(bad)
        meta["selection"]["selected"]["threadgroupWidth"] += 1
        (bad / "bundle.json").write_text(json.dumps(meta))
        self.assert_rejected(self.run_cli(bad), "bundle digest mismatch")
        direct, out = self.helper_direct(bad, 128)
        self.assertIn("bundle digest mismatch", direct.stderr)
        self.assertFalse(out.exists())

    def test_illegal_saved_selection_rejected_even_with_valid_digest(self):
        bad = self.copy_bundle(self.bundle_a, "selection")
        def mutate(meta):
            meta["selection"]["selected"] = {"threadgroupWidth": 7, "fuseDenseRelu": True}
            meta["selection"]["selectedRole"] = "optimization"
        rewrite_bundle(bad, mutate)
        self.assert_rejected(self.run_cli(bad), "saved selection rejected")

    def test_model_source_mismatch_rejected(self):
        # Wrong model (MODEL_B) with a CORRECT tower-128 input (name "a"). The early bound-shape
        # gate passes, so only the source-hash gate can reject. This happens before MODEL_B runs.
        self.assert_rejected(self.run_cli(self.bundle_a, MODEL_B, "a"), "model source mismatch")

    def test_actual_weights_are_bound(self):
        bad = self.copy_bundle(self.bundle_a, "seed")
        rewrite_bundle(bad, lambda meta: meta["python"].update(modelSeed=5))
        self.assert_rejected(self.run_cli(bad), "model weights mismatch")

    def test_device_mismatch_rejected(self):
        bad = self.copy_bundle(self.bundle_a, "device")
        rewrite_bundle(bad, lambda meta: meta["hardware"].update(deviceName="Leviathan Imaginary GPU"))
        self.assert_rejected(self.run_cli(bad), "hardware mismatch")

    def test_altered_backend_source_rejects_prior_bundle(self):
        root = self.tmp / "installed_copy"
        for rel in ("Package.swift", "tools/torch_frontend/export_torch_graph.py", "tools/forge_run/leviathan_forge.py"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel, root / rel)
        for rel in ("Sources/LeviathanCore", "Sources/leviathan-forge-helper"):
            shutil.copytree(REPO_ROOT / rel, root / rel)
        script = root / "tools" / "forge_run" / "leviathan_forge.py"
        unaltered = self.run_cli(self.bundle_a, script=script)
        self.assertEqual(unaltered.returncode, 0, unaltered.stderr)
        with open(root / "Sources" / "LeviathanCore" / "MetalBackend.swift", "a") as handle:
            handle.write("\n// locally altered backend\n")
        self.assert_rejected(self.run_cli(self.bundle_a, script=script), "backend source mismatch")

    def test_altered_helper_binary_rejects_prior_bundle(self):
        altered = self.tmp / "altered-helper"
        shutil.copy2(HELPER, altered)
        with open(altered, "ab") as handle:
            handle.write(b"\0leviathan-altered\0")
        os.chmod(altered, 0o755)
        proc = self.run_cli(self.bundle_a, helper=str(altered))
        if "identity mismatch" not in proc.stderr and "(exit -" in proc.stderr:
            self.skipTest("altered copy could not launch (code signing); identity check unreachable")
        self.assert_rejected(proc, "helper binary identity mismatch")

    def test_existing_outputs_are_never_overwritten(self):
        before = tree_digest(self.bundle_a)
        proc = cli("build", MODEL_A, "--example-input", self.inputs[("a", "example")], "--input",
                   self.inputs[("a", "input")], "--out", self.bundle_a, "--helper", HELPER)
        self.assert_rejected(proc, "already exists")
        self.assertEqual(tree_digest(self.bundle_a), before)
        report = self.tmp / "existing-report.json"
        report.write_text("keep")
        self.assert_rejected(self.run_cli(self.bundle_a, "--report" and MODEL_A, "a") if False else
                             cli("run", self.bundle_a, "--model", MODEL_A, "--input", self.inputs[("a", "input2")],
                                 "--helper", HELPER, "--report", report), "already exists")
        self.assertEqual(report.read_text(), "keep")

    def test_input_json_is_bounded_finite_rank2_fp32(self):
        bad = ["[[NaN, 1.0]]", "[[Infinity]]", "[[1e400]]", "[[3.5e38]]", "[1.0, 2.0]", "[[[1.0]]]",
               "[[1.0], [2.0]]", "[[]]", "[[true]]", '[["1"]]', "[[1.0,"]
        for index, text in enumerate(bad):
            with self.subTest(text=text):
                path = self.tmp / f"bad-{index}.json"
                path.write_text(text)
                with self.assertRaises(forge.ForgeRunError):
                    forge.load_input_json(path)
        big = self.tmp / "bad-big.json"
        big.write_text("[[" + ",".join(["0.0"] * 400000) + "]]")
        with self.assertRaises(forge.ForgeRunError):
            forge.load_input_json(big)
        good = self.tmp / "good.json"
        good.write_text("[[1, -2.5, 0.0]]")
        self.assertEqual(tuple(forge.load_input_json(good).shape), (1, 3))
        proc = cli("run", self.bundle_a, "--model", MODEL_A, "--input", self.inputs[("b", "input")], "--helper", HELPER)
        self.assert_rejected(proc, "does not match the bound static shape")

    def test_existing_torch_frontend_still_works(self):
        prefix = str(self.tmp / "frontend_a")
        frontend = REPO_ROOT / "tools" / "torch_frontend" / "export_torch_graph.py"
        proc = subprocess.run([sys.executable, str(frontend), str(MODEL_A), prefix, "0"],
                              capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        files = load_meta(self.bundle_a)["files"]
        self.assertEqual(hashlib.sha256(Path(prefix + ".graph.json").read_bytes()).hexdigest(), files["graphSha256"])
        self.assertEqual(hashlib.sha256(Path(prefix + ".weights.f32").read_bytes()).hexdigest(), files["weightsSha256"])


if __name__ == "__main__":
    unittest.main()
