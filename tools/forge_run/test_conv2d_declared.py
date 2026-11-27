"""Exercise the declared Metal Conv2D CLI against PyTorch."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.forge_run import conv2d_build_declared as conv

TOOLS = Path(__file__).resolve().parent
RESNET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")


@pytest.fixture
def invocation(tmp_path):
    helper = tmp_path / "helper"
    subprocess.run(["swiftc", "-O", str(TOOLS / "metal_conv2d.swift"), "-o", str(helper)],
                   check=True, timeout=90)

    def run(name, x, weights, stride=2, padding=3, *, overrides=None, reject=False):
        out = tmp_path / name
        payload = weights.detach().numpy().astype("<f4").tobytes()
        args = {"input": x.numpy().astype("<f4").tobytes(), "weights": payload}
        oh, ow = [(n + 2 * padding - k) // stride + 1 for n, k in zip(x.shape[-2:], weights.shape[-2:])]
        contract = dict(format=conv.FORMAT, inputShape=list(x.shape), weightShape=list(weights.shape),
                        outputShape=[1, weights.shape[0], oh, ow], weightsSha256=conv.digest(payload),
                        stride=stride, padding=padding, groups=1, dilation=1)
        contract.update(overrides or {})
        args["contract"] = json.dumps(contract).encode()
        cmd = [sys.executable, str(TOOLS / "conv2d_build_declared.py"), "--helper", str(helper),
               "--out", str(out), "--cache", str(tmp_path / "cache")]
        for key, data in args.items():
            path = tmp_path / key
            path.write_bytes(data)
            cmd.extend([f"--{key}", str(path)])
        proc = subprocess.run(cmd, capture_output=True, timeout=90)
        if reject:
            assert proc.returncode == 2 and not out.exists()
            return proc.stderr.decode()
        assert proc.returncode == 0, proc.stderr.decode()
        report = json.loads(proc.stdout)
        assert report["parity"] and report["gpu"]["gpuMicros"] > 0
        assert report["gpu"]["kernel"] == "conv2d_fp32"
        output = np.fromfile(out / "output.f32", dtype="<f4").reshape(contract["outputShape"])
        oracle = torch.nn.functional.conv2d(x, weights, stride=stride, padding=padding)
        error = np.max(np.abs(output - oracle.detach().numpy()))
        np.testing.assert_allclose(output, oracle.detach().numpy(), atol=1e-4, rtol=1e-4)
        return report, error

    return run


def test_cache_and_validation(invocation, tmp_path):
    x = torch.linspace(-1, 1, 2 * 9 * 13).reshape(1, 2, 9, 13)
    weights = torch.linspace(-0.2, 0.3, 3 * 2 * 5 * 7).reshape(3, 2, 5, 7)
    cold, _ = invocation("cold", x, weights)
    warm, _ = invocation("warm", x, weights)
    changed, _ = invocation("stride", x, weights, stride=1)
    changed_weight, _ = invocation("changed-weights", x, weights + 0.1)
    assert not cold["cache"]["hit"] and warm["cache"]["hit"]
    assert cold["outputSha256"] == warm["outputSha256"]
    assert all(not record["cache"]["hit"] and record["cache"]["key"] != cold["cache"]["key"]
               for record in (changed, changed_weight))
    cache = tmp_path / "cache" / (cold["cache"]["key"] + ".json")
    record = json.loads(cache.read_text())
    plan = json.loads(record["bundle"])
    plan["registryID"] = "0"
    record["bundle"] = conv.encode(plan).decode()
    record["sha256"] = conv.digest(record["bundle"].encode())
    cache.write_text(json.dumps(record))
    other_device, _ = invocation("device", x, weights)
    assert not other_device["cache"]["hit"] and other_device["outputSha256"] == cold["outputSha256"]
    cache.write_text("{}")
    recovered, _ = invocation("recovered", x, weights)
    assert not recovered["cache"]["hit"] and recovered["outputSha256"] == cold["outputSha256"]
    for name, overrides in (("shape", {"outputShape": [1, 3, 9, 9]}),
                            ("groups", {"groups": 2}), ("dilation", {"dilation": 2}),
                            ("digest", {"weightsSha256": "0" * 64})):
        assert invocation(name, x, weights, overrides=overrides, reject=True)
    x[0, 0, 0, 0] = float("nan")
    assert "non-finite" in invocation("nan", x, weights, reject=True)


def test_public_resnet18_stem(invocation):
    checkpoint = os.environ.get("LEVIATHAN_RESNET18_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LEVIATHAN_RESNET18_CHECKPOINT to the official ResNet-18 checkpoint")
    path = Path(checkpoint)
    assert conv.digest(path.read_bytes()) == RESNET_SHA256
    weights = torch.load(path, map_location="cpu", weights_only=True)["conv1.weight"]
    x = torch.linspace(-1, 1, 3 * 113 * 151).reshape(1, 3, 113, 151)
    report, error = invocation("resnet18", x, weights)
    print(f"ResNet-18 {report['contract']['outputShape']}, max PyTorch error {error:.8g}, {report['gpu']}")
