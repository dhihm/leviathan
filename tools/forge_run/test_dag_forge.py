#!/usr/bin/env python3
"""Acceptance tests for dag_forge: real torch 2.9.1 CPU eager + M1 Max Metal GPU.

    python3 tools/forge_run/test_dag_forge.py -v
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
HELPER = REPO_ROOT / ".build" / "debug" / "leviathan-dag-helper"
sys.path.insert(0, str(TOOL_DIR))
import dag_forge as forge  # noqa: E402
from dag_forge import ForgeRunError  # noqa: E402

RESIDUAL = '''import torch
from torch import nn
INPUT_DIM = 32
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(32, 96)
        self.contract = nn.Linear(96, 32)
    def forward(self, x):
        return self.contract(torch.relu(self.expand(x))) + x
def build(seed):
    torch.manual_seed(seed)
    return ResidualBlock()
'''
PROJECTED = '''import torch
from torch import nn
INPUT_DIM = 24
class ProjectedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(24, 48)
        self.left2 = nn.Linear(48, 24)
        self.right = nn.Linear(24, 24)
        self.project = nn.Linear(24, 24)
    def forward(self, x):
        deep = self.left2(torch.relu(self.left(x)))
        shallow = self.right(self.project(x))
        return deep + shallow
def build(seed):
    torch.manual_seed(seed)
    return ProjectedBranches()
'''
CANARY = '''import torch
from torch import nn
INPUT_DIM = 3
class Canary(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3))
            self.fc.bias.copy_(torch.tensor([1.0, -2.0, 0.0]))
    def forward(self, x):
        h = self.fc(x)
        return h + torch.relu(h)
def build(seed):
    return Canary()
'''


def reject_model(body, dim):
    return ("import torch\nfrom torch import nn\nINPUT_DIM = %d\n" % dim
            + "class M(nn.Module):\n"
              "    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(%d, %d)\n" % (dim, dim)
            + "    def forward(self, x):\n        h = self.fc(x)\n        return " + body + "\n"
              "def build(seed):\n    torch.manual_seed(seed)\n    return M()\n")


SIGMOID_SRC = reject_model("torch.sigmoid(h)", 8)
BROADCAST_SRC = '''import torch
from torch import nn
INPUT_DIM = 8
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 1)
    def forward(self, x):
        h = self.fc(x)
        return h + torch.relu(self.proj(h))
def build(seed):
    torch.manual_seed(seed)
    return M()
'''
ALPHA_SRC = reject_model("torch.add(h, x, alpha=2.0)", 8)


def setUpModule():
    assert HELPER.exists(), "helper missing: %s (build it first)" % HELPER


def cli(*args):
    return subprocess.run([sys.executable, str(TOOL_DIR / "dag_forge.py"), *map(str, args)],
                          capture_output=True, text=True, timeout=1800)


def write_input(path, dim, lo, hi):
    values = torch.linspace(lo, hi, dim).reshape(1, dim).tolist()
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


def source_module(model_path):
    spec = importlib.util.spec_from_file_location("ref_" + Path(model_path).stem, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_reference(model_path, input_path):
    with torch.no_grad():
        return source_module(model_path).build(0).eval()(torch.tensor(json.loads(Path(input_path).read_text())))


class DagForgeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="dag-forge-test-"))
        cls.model_a = cls.tmp / "residual_block.py"
        cls.model_b = cls.tmp / "projected_branches.py"
        cls.model_can = cls.tmp / "canary.py"
        cls.model_a.write_text(RESIDUAL)
        cls.model_b.write_text(PROJECTED)
        cls.model_can.write_text(CANARY)
        cls.inputs = {}
        for name, dim, ranges in (("a", 32, (("example", -1.0, 1.0), ("input", 1.0, -1.0), ("input2", 0.5, -1.5))),
                                  ("b", 24, (("example", -1.0, 1.0), ("input", 1.0, -1.0), ("input2", -1.5, 0.5)))):
            for kind, lo, hi in ranges:
                cls.inputs[(name, kind)] = write_input(cls.tmp / f"{name}.{kind}.json", dim, lo, hi)
        cls.reports = {}
        cls.bundle_a = cls.tmp / "bundle_a"
        cls.bundle_b = cls.tmp / "bundle_b"
        cls.bundle_f = cls.tmp / "bundle_f"
        cls.reports["a"] = forge.build_bundle(cls.model_a, cls.inputs[("a", "example")],
                                               cls.inputs[("a", "input")], cls.bundle_a, helper=str(HELPER))
        cls.reports["b"] = forge.build_bundle(cls.model_b, cls.inputs[("b", "example")],
                                               cls.inputs[("b", "input")], cls.bundle_b, helper=str(HELPER))
        cls.reports["f"] = forge.build_bundle(cls.model_a, cls.inputs[("a", "example")],
                                               cls.inputs[("a", "input")], cls.bundle_f, helper=str(HELPER),
                                               diagnostic_fail_optimizations=True)

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("DAG_FORGE_KEEP_TEST_OUTPUT"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def cases(self):
        return ((self.reports["a"], self.bundle_a, self.model_a, "a"),
                (self.reports["b"], self.bundle_b, self.model_b, "b"))

    def fresh_bundle(self, tag):
        dest = Path(tempfile.mkdtemp(prefix=tag + "-", dir=self.tmp)) / "bundle"
        shutil.copytree(self.bundle_a, dest)
        return dest

    def run_rejects(self, bundle, pattern, model=None, inp=None):
        with self.assertRaisesRegex(ForgeRunError, pattern):
            forge.run_bundle(bundle, model or self.model_a,
                             inp or self.inputs[("a", "input2")], helper=str(HELPER))

    def assert_gpu(self, run, bundle):
        meta = load_meta(bundle)
        gpu, timing, sel = run["gpuEvidence"], run["timing"], meta["selection"]["selected"]
        self.assertTrue(gpu["deviceName"])
        self.assertEqual(gpu["deviceName"], meta["hardware"]["deviceName"])
        self.assertEqual(gpu["commandStatus"], "completed")
        micros = gpu["gpuCommandBufferMicros"]
        self.assertGreater(micros, 0)
        self.assertTrue(math.isfinite(micros) and math.isfinite(timing["endToEndMillis"]))
        dispatches = gpu["computeDispatches"]
        self.assertGreater(dispatches, 0)
        self.assertEqual(dispatches, len(gpu["kernelSequence"]))
        self.assertEqual(gpu.get("threadgroupWidth"), sel["threadgroupWidth"])
        self.assertEqual("leviathan_dense_relu" in gpu["kernelSequence"], bool(sel["fuseDenseRelu"]))
        self.assertGreater(timing["endToEndMillis"] * 1000.0, micros)
        self.assertIn("GPU", timing["labels"]["gpuCommandBufferMicros"])
        self.assertEqual(run["selected"], sel)

    def test_build_runs_real_gpu_with_cpu_eager_parity(self):
        for report, bundle, model, name in self.cases():
            with self.subTest(model=model.name):
                run = report["run"]
                self.assertTrue(run["parity"]["passed"])
                self.assert_gpu(run, bundle)
                ref = source_reference(model, self.inputs[(name, "input")])
                self.assertTrue(torch.allclose(torch.tensor(run["output"]), ref, atol=1e-4, rtol=1e-4))

    def test_two_examples_are_genuinely_different(self):
        ga = json.loads((self.bundle_a / "model.graph.json").read_text())
        gb = json.loads((self.bundle_b / "model.graph.json").read_text())
        self.assertNotEqual(ga, gb)

    def test_selection_baselines_and_strict_gain(self):
        for _, bundle, _, _ in self.cases():
            with self.subTest(bundle=bundle.name):
                sel = load_meta(bundle)["selection"]
                self.assertEqual(sel["qualificationInputs"], 4)
                self.assertLessEqual(abs(sel["minRelativeGain"] - 0.02), 1e-9)
                base = sel["baseline"]
                self.assertTrue(base["qualified"])
                self.assertGreaterEqual(len(sel["optimizations"]), 1)
                if sel["selectedRole"] == "optimization":
                    chosen = next(o for o in sel["optimizations"] if o["candidate"] == sel["selected"])
                    self.assertTrue(chosen["qualified"])
                    self.assertLess(chosen["medianGpuMicros"], base["medianGpuMicros"] * (1 - sel["minRelativeGain"]))
                else:
                    self.assertEqual(sel["selected"], base["candidate"])

    def test_fresh_process_cli_run_third_input_no_retuning(self):
        for report, bundle, model, name in self.cases():
            with self.subTest(model=model.name):
                before = tree_digest(bundle)
                proc = cli("run", "--bundle", bundle, "--model", model,
                           "--input", self.inputs[(name, "input2")], "--helper", HELPER)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                run = json.loads(proc.stdout)
                self.assertEqual(tree_digest(bundle), before)
                self.assertNotEqual(run["output"], report["run"]["output"])
                self.assert_gpu(run, bundle)
                ref = source_reference(model, self.inputs[(name, "input2")])
                self.assertTrue(torch.allclose(torch.tensor(run["output"]), ref, atol=1e-4, rtol=1e-4))

    def test_canary_identity_weights_exact_and_unfused(self):
        ex = write_input(self.tmp / "can.ex.json", 3, -1.0, 1.0)
        inp = write_input(self.tmp / "can.in.json", 3, 1.0, -1.0)
        out = self.tmp / "bundle_can"
        rep = forge.build_bundle(self.model_can, ex, inp, out, helper=str(HELPER))
        run = rep["run"]
        self.assertTrue(run["parity"]["passed"])
        got = torch.tensor(run["output"])
        module = source_module(self.model_can).build(0).eval()
        with torch.no_grad():
            h = module.fc(torch.tensor(json.loads(inp.read_text())))
        self.assertTrue(torch.allclose(got, h + torch.relu(h), atol=1e-6, rtol=0.0))
        self.assertTrue(torch.equal(got[h < 0], h[h < 0]))
        graph = json.loads((out / "model.graph.json").read_text())
        consumers = {}
        for n in graph["nodes"]:
            for i in n["inputs"]:
                consumers[i] = consumers.get(i, 0) + 1
        consumers[graph["outputValue"]] = consumers.get(graph["outputValue"], 0) + 1
        dense = next(n for n in graph["nodes"] if n["op"] == "dense")
        relu = next(n for n in graph["nodes"] if n["op"] == "relu")
        self.assertEqual(relu["inputs"], [dense["id"]])
        self.assertGreater(consumers[dense["id"]], 1)
        self.assertFalse(load_meta(out)["selection"]["selected"]["fuseDenseRelu"])
        self.assertEqual(len(run["gpuEvidence"]["kernelSequence"]), 3)

    def test_forced_optimization_failure_falls_back_to_baseline(self):
        sel = load_meta(self.bundle_f)["selection"]
        self.assertTrue(sel["diagnosticFailOptimizations"])
        for opt in sel["optimizations"]:
            self.assertFalse(opt["qualified"])
            self.assertIsNone(opt.get("medianGpuMicros"))
        self.assertFalse(sel["baseline"]["diagnosticForcedFailure"])
        self.assertEqual(sel["selectedRole"], "baseline")
        self.assertEqual(sel["selected"], sel["baseline"]["candidate"])
        run = self.reports["f"]["run"]
        self.assertTrue(run["parity"]["passed"])
        self.assertGreater(run["gpuEvidence"]["gpuCommandBufferMicros"], 0)
        self.assert_gpu(run, self.bundle_f)

    def assert_build_rejected(self, source, dim, pattern, tag):
        model = self.tmp / f"model_{tag}.py"
        model.write_text(source)
        ex = write_input(self.tmp / f"ex_{tag}.json", dim, -1.0, 1.0)
        out = self.tmp / f"out_{tag}"
        with self.assertRaisesRegex(ForgeRunError, pattern):
            forge.build_bundle(model, ex, ex, out, helper=str(HELPER))
        self.assertFalse(out.exists())

    def test_unsupported_sigmoid_rejected_no_final(self):
        self.assert_build_rejected(SIGMOID_SRC, 8, r"(?i)sigmoid", "sigmoid")

    def test_broadcast_add_and_alpha_two_rejected_no_final(self):
        self.assert_build_rejected(BROADCAST_SRC, 8, r"(?i)broadcast", "bcast")
        self.assert_build_rejected(ALPHA_SRC, 8, r"(?i)alpha", "alpha")

    def test_corrupt_weights_and_graph_rejected(self):
        bad = self.fresh_bundle("w")
        p = bad / "model.weights.f32"
        data = bytearray(p.read_bytes())
        data[len(data) // 2] ^= 0x40
        p.write_bytes(bytes(data))
        self.run_rejects(bad, r"(?i)weights.*mismatch|mismatch.*weights")
        bad = self.fresh_bundle("g")
        g = bad / "model.graph.json"
        raw = bytearray(g.read_bytes())
        raw[len(raw) // 2] ^= 0x20
        g.write_bytes(bytes(raw))
        self.run_rejects(bad, r"(?i)graph|mismatch|digest")

    def test_model_source_mismatch_rejected_before_source_exec(self):
        marker = self.tmp / "marker.txt"
        src = RESIDUAL.replace("def build(seed):",
                               "def build(seed):\n    open(%r, 'w').write('exec')" % str(marker))
        other = self.tmp / "other_residual.py"
        other.write_text(src)
        self.assertFalse(marker.exists())
        self.run_rejects(self.bundle_a, r"(?i)model source mismatch", model=other)
        self.assertFalse(marker.exists())

    def test_backend_source_tamper_with_valid_digest_rejected(self):
        bad = self.fresh_bundle("be")
        rewrite_bundle(bad, lambda m: m["python"].update(backendSourceSha256="0" * 64))
        self.run_rejects(bad, r"(?i)backend")

    def test_semantic_tampering_rejected_beyond_checksums(self):
        bad = self.fresh_bundle("hw")
        rewrite_bundle(bad, lambda m: m["hardware"].update(deviceName="Leviathan Imaginary GPU"))
        self.run_rejects(bad, r"(?i)hardware")
        bad = self.fresh_bundle("sel")
        rewrite_bundle(bad, lambda m: m["selection"]["selected"].update(threadgroupWidth=4096))
        self.run_rejects(bad, r"(?i)selection|threadgroup")
        bad = self.fresh_bundle("edge")
        gpath = bad / "model.graph.json"
        graph = json.loads(gpath.read_text())
        graph["nodes"][1]["inputs"] = [graph["nodes"][1]["id"]]
        edges = graph.setdefault("edges", [])
        edges.append({"from": "n0", "to": "n0"})
        edges.append({"from": "missing_node", "to": "n1"})
        data = json.dumps(graph, indent=2).encode()
        gpath.write_bytes(data)
        rewrite_bundle(bad, lambda m: m["files"].update(graphSha256=hashlib.sha256(data).hexdigest()))
        self.run_rejects(bad, r"(?i)edge|reference|graph")
        self.run_rejects(self.bundle_a, r"(?i)shape",
                         inp=write_input(self.tmp / "badwidth.json", 24, -1.0, 1.0))

    def test_existing_outputs_never_overwritten_and_inputs_gated_before_model_load(self):
        before = tree_digest(self.bundle_a)
        with self.assertRaisesRegex(ForgeRunError, r"(?i)already exists"):
            forge.build_bundle(self.model_a, self.inputs[("a", "example")],
                               self.inputs[("a", "input")], self.bundle_a, helper=str(HELPER))
        self.assertEqual(tree_digest(self.bundle_a), before)
        report = self.tmp / "existing-report.json"
        report.write_text("keep")
        with self.assertRaisesRegex(ForgeRunError, r"(?i)already exists"):
            forge.build_bundle(self.model_a, self.inputs[("a", "example")],
                               self.inputs[("a", "input")], self.tmp / "bundle_new",
                               report_path=report, helper=str(HELPER))
        self.assertEqual(report.read_text(), "keep")
        marker = self.tmp / "gate_marker.txt"
        gated = RESIDUAL.replace("def build(seed):",
                                 "def build(seed):\n    open(%r, 'w').write('exec')" % str(marker))
        model = self.tmp / "gated_model.py"
        model.write_text(gated)
        bad_inputs = ["[[NaN]]", "[[Infinity]]", "[[3.5e38]]", "[[1e400]]", "[[1.0]]"]
        for i, text in enumerate(bad_inputs):
            with self.subTest(text=text):
                p = self.tmp / f"gate-{i}.json"
                p.write_text(text)
                out = self.tmp / f"gate-out-{i}"
                with self.assertRaises(ForgeRunError):
                    forge.build_bundle(model, p, p, out, helper=str(HELPER))
                self.assertFalse(out.exists())
        self.assertFalse(marker.exists())

    def test_atomic_post_gpu_parity_fault_leaves_failed_marker_not_final(self):
        out = self.tmp / "bundle_fault"
        original = forge.run_bundle

        def exploding(*a, **k):
            original(*a, **k)  # real GPU execution + source parity completed first
            raise ForgeRunError("test after real GPU")

        forge.run_bundle = exploding
        try:
            with self.assertRaisesRegex(ForgeRunError, "test after real GPU"):
                forge.build_bundle(self.model_a, self.inputs[("a", "example")],
                                   self.inputs[("a", "input")], out, helper=str(HELPER))
        finally:
            forge.run_bundle = original
        self.assertFalse(out.exists())
        siblings = sorted(out.parent.glob(f".{out.name}.FAILED-*"))
        self.assertTrue(siblings, "expected a preserved FAILED sibling stage")
        failed_stage = siblings[0]
        marker = failed_stage / "FORGE_FAILED.json"
        self.assertTrue(marker.exists())
        failed = json.loads(marker.read_text())
        self.assertTrue(any(failed.get(k) for k in ("stage", "error", "reason")))
        self.assertTrue(any(p.name != "FORGE_FAILED.json" for p in failed_stage.rglob("*") if p.is_file()))
        with self.assertRaisesRegex(ForgeRunError, r"(?i)fail|FORGE_FAILED|refus"):
            forge.run_bundle(failed_stage, self.model_a, self.inputs[("a", "input")], helper=str(HELPER))


if __name__ == "__main__":
    unittest.main()
