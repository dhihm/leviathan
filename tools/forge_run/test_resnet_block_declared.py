"""Run checkpoint-backed Metal basic blocks against unfused PyTorch."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tools.forge_run import resnet_block_declared as block
from tools.forge_run.test_conv2d_declared import RESNET_SHA256, TOOLS

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")


@pytest.fixture(scope="module")
def checkpoint_and_helper(tmp_path_factory):
    path = os.environ.get("LEVIATHAN_RESNET18_CHECKPOINT")
    if path is None:
        pytest.skip("set LEVIATHAN_RESNET18_CHECKPOINT to the official checkpoint")
    path = Path(path)
    assert block.conv.digest(path.read_bytes()) == RESNET_SHA256
    weights = torch.load(path, map_location="cpu", weights_only=True)
    helper = tmp_path_factory.mktemp("resnet-metal") / "helper"
    subprocess.run(["swiftc", "-O", str(TOOLS / "metal_conv2d.swift"), "-o", str(helper)],
                   check=True, timeout=90)
    return weights, helper


@pytest.mark.parametrize("prefix,projection", [("layer1.0", False), ("layer2.0", True)])
def test_public_block(checkpoint_and_helper, tmp_path, prefix, projection):
    state, helper = checkpoint_and_helper
    names = [("conv1", "bn1"), ("conv2", "bn2")]
    if projection:
        names.append(("downsample.0", "downsample.1"))
    tensors = [
        tuple(state[f"{prefix}.{name}"].detach() for name in (
            f"{conv}.weight", f"{bn}.weight", f"{bn}.bias", f"{bn}.running_mean", f"{bn}.running_var"))
        for conv, bn in names
    ]
    stride = 2 if projection else 1
    ci = tensors[0][0].shape[1]
    x = torch.linspace(-1, 1, ci * 17 * 23).reshape(1, ci, 17, 23)
    eps = [1e-5] * len(tensors)

    def oracle(groups, epsilon):
        def layer(value, params, step, e):
            weights, gamma, beta, mean, var = params
            output = F.conv2d(value, weights, stride=step, padding=weights.shape[-1] // 2)
            return F.batch_norm(output, mean, var, gamma, beta, training=False, eps=e)

        first = layer(x, groups[0], stride, epsilon[0]).relu()
        main = layer(first, groups[1], 1, epsilon[1])
        skip = layer(x, groups[2], stride, epsilon[2]) if projection else x
        return (main + skip).relu().numpy()

    expected = oracle(tensors, eps)
    input_path = tmp_path / "input"
    input_path.write_bytes(x.numpy().astype("<f4").tobytes())
    weights_path = tmp_path / "weights"
    declaration_path = tmp_path / "contract"

    def run(name, cache, groups=tensors, epsilon=eps, reject=False, overrides=None):
        packed = b"".join(t.numpy().astype("<f4").tobytes() for layer in groups for t in layer)
        weights_path.write_bytes(packed)
        declaration = dict(format=block.FORMAT, folding=block.FOLDING, inputShape=list(x.shape),
                           outputShape=list(expected.shape), skip="projection" if projection else "identity",
                           stride=stride, originalSha256=block.conv.digest(packed), epsilon=epsilon)
        declaration.update(overrides or {})
        declaration_path.write_text(json.dumps(declaration))
        out = tmp_path / name
        command = [sys.executable, str(TOOLS / "resnet_block_declared.py"), "--helper", str(helper),
                   "--input", str(input_path), "--weights", str(weights_path), "--contract", str(declaration_path),
                   "--cache", str(cache), "--out", str(out)]
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, timeout=90)
        elapsed = (time.perf_counter() - start) * 1000
        if reject:
            assert proc.returncode == 2 and not out.exists()
            return None
        assert proc.returncode == 0, proc.stderr.decode()
        r = json.loads(proc.stdout)
        assert r["parity"] and r["gpu"]["gpuMicros"] > 0
        actual = np.fromfile(out / "output.f32", dtype="<f4").reshape(expected.shape)
        wanted = oracle(groups, epsilon)
        np.testing.assert_allclose(actual, wanted, rtol=2e-4, atol=2e-4)
        return r, (elapsed, r["gpu"]["gpuMicros"] / 1000, float(np.max(np.abs(actual - wanted))))

    samples = [[], []]
    for i in range(5):
        cache = tmp_path / f"cache-{i}"
        cold, c = run(f"cold-{i}", cache)
        warm, w = run(f"warm-{i}", cache)
        assert not cold["cache"]["hit"] and warm["cache"]["hit"]
        assert cold["outputSha256"] == warm["outputSha256"]
        samples[0].append(c)
        samples[1].append(w)
    print(prefix, expected.shape, warm["gpu"]["device"], [
        [(statistics.median(column), min(column), max(column)) for column in zip(*rows)] for rows in samples])

    cache = tmp_path / "cache-0"
    altered_eps = [2e-5, *eps[1:]]
    changed, _ = run("new-eps", cache, epsilon=altered_eps)
    shifted = list(tensors)
    weight, gamma, beta, mean, var = shifted[0]
    shifted[0] = (weight, gamma, beta, mean + 0.01, var)
    changed_stats, _ = run("new-stats", cache, groups=shifted)
    shifted[0] = (weight + 0.001, gamma, beta, mean, var)
    changed_weights, _ = run("new-weights", cache, groups=shifted)
    for record in (changed, changed_stats, changed_weights):
        assert not record["cache"]["hit"] and record["cache"]["key"] != warm["cache"]["key"]
    (cache / (warm["cache"]["key"] + ".json")).write_text("{}")
    repaired, _ = run("repair", cache)
    assert not repaired["cache"]["hit"] and repaired["outputSha256"] == warm["outputSha256"]
    run("bad-stride", cache, reject=True, overrides={"stride": 1 if projection else 2})
    run("bad-eps", cache, reject=True, overrides={"epsilon": [0, *eps[1:]]})
    run("bad-digest", cache, reject=True, overrides={"originalSha256": "0" * 64})
