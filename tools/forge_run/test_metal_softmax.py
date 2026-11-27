"""Real Metal Softmax and declarative cache coverage."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tools.forge_run import dag_build_declared as declared
from tools.forge_run.test_dag_runtime import HELPER, TOOL_DIR, residual, work


@pytest.fixture
def softmax_helper(tmp_path):
    helper = tmp_path / "metal-softmax"
    subprocess.run(["swiftc", "-O", str(TOOL_DIR / "metal_softmax.swift"), "-o", str(helper)],
                   check=True, timeout=90)
    return helper


def test_metal_rows_and_extreme_logits(softmax_helper, tmp_path):
    rows = [[3e38, 3e38, -3e38], [10000, 9999, -10000], [-3e38, -3e38, -3e38]]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    out = json.loads(subprocess.check_output([str(softmax_helper), str(path)], timeout=30))
    expected = []
    for row in rows:
        scores = [math.exp(x - max(row)) for x in row]
        expected.append([x / math.fsum(scores) for x in scores])
    np.testing.assert_allclose(declared.softmax_reference(rows), expected, atol=1e-7)
    np.testing.assert_allclose(out["output"], expected, atol=1e-7)
    assert out["gpuMicros"] > 0
    for invalid in ("[]", "[[0],[0,1]]", "[[1e40,0]]", "[[NaN,0]]"):
        path.write_text(invalid)
        rejected = subprocess.run([str(softmax_helper), str(path)], capture_output=True, timeout=30)
        assert rejected.returncode == 2 and not rejected.stdout
    with pytest.raises(declared.BuildError):
        declared.softmax_reference([[float("inf")]])


def test_softmax_declaration_cache_binding(softmax_helper, residual, tmp_path):
    _, _, baseline, _, _ = residual
    graph, weights = (baseline / name for name in ("model.graph.json", "model.weights.f32"))
    data = dict(format=declared.FORMAT, inputShape=[1, 32], outputShape=[1, 32],
                graphSha256=declared.digest(graph.read_bytes()),
                weightsSha256=declared.digest(weights.read_bytes()))
    contract = tmp_path / "contract.json"
    x = tmp_path / "input.json"
    x.write_text(json.dumps([[0.25] * 32]))
    command = [sys.executable, str(TOOL_DIR / "dag_build_declared.py"), "--helper", str(HELPER),
               "--contract", str(contract), "--graph", str(graph), "--weights", str(weights),
               "--input", str(x), "--example-input", str(x), "--cache", str(tmp_path / "plans")]

    def build(name, helper=None, extra=()):
        args = ["--softmax-helper", str(helper)] if helper else []
        out = subprocess.check_output(command + ["--out", str(tmp_path / name), *args, *extra], timeout=75)
        return json.loads(out)

    contract.write_text(json.dumps(data))
    logits = build("v1")
    contract.write_text(json.dumps(dict(data, format=declared.SOFTMAX_FORMAT, outputActivation="softmax")))
    cold = build("v2", softmax_helper)
    warm = build("v2-warm", softmax_helper)
    assert not cold["cache"]["hit"] and warm["cache"]["hit"]
    assert cold["cache"]["key"] != logits["cache"]["key"]
    assert cold["run"]["output"] == warm["run"]["output"]
    concurrent = build("v2-concurrent", softmax_helper, ("--concurrent-softmax",))
    assert concurrent["cache"]["hit"] and concurrent["run"]["output"] == cold["run"]["output"]
    record = tmp_path / "plans" / (cold["cache"]["key"] + ".json")
    record.write_text("{}")
    recovered = build("v2-corrupt", softmax_helper, ("--concurrent-softmax",))
    assert not recovered["cache"]["hit"] and recovered["run"]["output"] == cold["run"]["output"]
    np.testing.assert_allclose(cold["run"]["output"], declared.softmax_reference(logits["run"]["output"]),
                               atol=2e-6)
    alternate = tmp_path / "softmax-unoptimized"
    subprocess.run(["swiftc", "-Onone", str(TOOL_DIR / "metal_softmax.swift"), "-o", str(alternate)],
                   check=True, timeout=90)
    changed = build("changed-executor", alternate)
    assert not changed["cache"]["hit"] and changed["cache"]["key"] != cold["cache"]["key"]
