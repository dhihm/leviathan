"""Verify portable declared outputs against the public TorchVision checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.forge_run import resnet_block_declared as spec

TOOLS = Path(__file__).resolve().parent
CHECKPOINT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
IMAGE_SHA = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"


def command(tmp_path, backend):
    return [sys.executable, str(TOOLS / "resnet_run_declared.py"), "--backend", backend,
            "--contract", str(tmp_path / "contract"), "--weights", str(tmp_path / "weights"),
            "--input", str(tmp_path / "input"), "--out", str(tmp_path / "outputs.npz")]


@pytest.mark.parametrize("full", [False, True], ids=["stage1", "full"])
def test_public_portable(tmp_path, full):
    path = os.environ.get("LEVIATHAN_RESNET18_CHECKPOINT")
    if not path:
        pytest.skip("set LEVIATHAN_RESNET18_CHECKPOINT to the pinned checkpoint")
    from torchvision.models import ResNet18_Weights, resnet18

    path = Path(path)
    assert spec.conv.digest(path.read_bytes()) == CHECKPOINT_SHA
    model = resnet18().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    photo = os.environ.get("LEVIATHAN_RESNET18_IMAGE")
    if photo:
        from PIL import Image
        assert spec.conv.digest(Path(photo).read_bytes()) == IMAGE_SHA
        with Image.open(photo) as image:
            x = ResNet18_Weights.IMAGENET1K_V1.transforms()(image.convert("RGB"))[None]
    else:
        x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    groups = [(model.conv1, model.bn1)]
    residual_stages = (model.layer1, model.layer2, model.layer3, model.layer4) if full else (model.layer1,)
    for stage in residual_stages:
        for layer in stage:
            groups.extend(((layer.conv1, layer.bn1), (layer.conv2, layer.bn2)))
            if layer.downsample is not None:
                groups.append(tuple(layer.downsample))

    def pack(value):
        return value.detach().numpy().astype("<f4").tobytes()

    weights = b"".join(pack(t) for layer, bn in groups for t in (
        layer.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var))
    if full:
        weights += pack(model.fc.weight) + pack(model.fc.bias)
    shapes = spec.FULL_SHAPES if full else spec.STAGE_SHAPES
    contract = dict(format=spec.FULL_FORMAT if full else spec.STAGE_FORMAT,
                    folding=spec.FOLDING, ops=spec.FULL_OPS if full else spec.STAGE_OPS,
                    inputShape=[1, 3, 224, 224], outputShape=shapes["output"],
                    originalSha256=spec.conv.digest(weights), epsilon=[bn.eps for _, bn in groups])
    (tmp_path / "contract").write_text(json.dumps(contract))
    (tmp_path / "weights").write_bytes(weights)
    (tmp_path / "input").write_bytes(pack(x))

    expected = {}

    def capture(name):
        def hook(_, __, value):
            expected[name] = value.detach().clone().numpy().reshape(shapes[name])
        return hook

    modules = {"stem": model.relu, "pool": model.maxpool, "block1": model.layer1[0]}
    if full:
        modules.update(layer1=model.layer1, layer2=model.layer2, layer3=model.layer3,
                       layer4=model.layer4, features=model.avgpool)
    handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
    try:
        with torch.inference_mode():
            value = model(x) if full else model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            expected["output"] = value.numpy()
    finally:
        for handle in handles:
            handle.remove()

    backend = os.environ.get("LEVIATHAN_RESNET18_BACKEND", "cpu")
    proc = subprocess.run(command(tmp_path, backend), capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode()
    report = json.loads(proc.stdout)
    assert report["backend"] == backend and report["parity"]
    if backend == "cuda":
        assert report["device"] != "cpu" and report["cudaEventMillis"] > 0
    else:
        assert report["device"] == "cpu" and "cudaEventMillis" not in report
    with np.load(tmp_path / "outputs.npz") as results:
        assert results.files == list(expected)
        errors = {}
        for name, want in expected.items():
            assert list(results[name].shape) == shapes[name]
            np.testing.assert_allclose(results[name], want, atol=2e-4, rtol=2e-4)
            errors[name] = float(np.max(np.abs(results[name] - want)))
        if full:
            np.testing.assert_array_equal(np.argsort(-results["output"][0])[:5],
                                          np.argsort(-expected["output"][0])[:5])
    print(json.dumps({"report": report, "torchvisionMaxAbsError": errors}, sort_keys=True))

    (tmp_path / "outputs.npz").unlink()
    contract["originalSha256"] = "0" * 64
    (tmp_path / "contract").write_text(json.dumps(contract))
    rejected = subprocess.run(command(tmp_path, backend), capture_output=True, timeout=120)
    assert rejected.returncode == 2 and b"original tensor digest mismatch" in rejected.stderr
    assert not (tmp_path / "outputs.npz").exists()


def test_cuda_never_falls_back(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this check requires a host without CUDA")
    proc = subprocess.run(command(tmp_path, "cuda"), capture_output=True, timeout=30)
    assert proc.returncode == 2 and b"requires an NVIDIA GPU" in proc.stderr
    assert not (tmp_path / "outputs.npz").exists()
