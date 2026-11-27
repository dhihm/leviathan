"""Full public checkpoint parity and fresh-process Metal BUILD timings."""
import copy
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

from tools.forge_run import resnet_block_declared as spec
from tools.forge_run.test_resnet_block_declared import TOOLS, checkpoint_and_helper

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"


@pytest.mark.parametrize("image", [False, True], ids=["synthetic", "astronaut"])
def test_full_resnet18(checkpoint_and_helper, tmp_path, image):
    from torchvision.models import ResNet18_Weights, resnet18

    state, helper = checkpoint_and_helper
    model = resnet18().eval()
    model.load_state_dict(state)
    if image:
        from PIL import Image

        path = os.environ.get("LEVIATHAN_RESNET18_IMAGE")
        if not path:
            pytest.skip("set LEVIATHAN_RESNET18_IMAGE to the pinned astronaut PNG")
        assert spec.conv.digest(Path(path).read_bytes()) == IMAGE_SHA256
        with Image.open(path) as photo:
            x = ResNet18_Weights.IMAGENET1K_V1.transforms()(photo.convert("RGB"))[None]
    else:
        x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    boundaries = {}

    def capture(name):
        def save(_, __, value):
            boundaries[name] = value.detach().clone().numpy()
        return save

    handles = [module.register_forward_hook(capture(name)) for name, module in (
        ("stem", model.relu), ("pool", model.maxpool), ("block1", model.layer1[0]),
        ("layer1", model.layer1), ("layer2", model.layer2), ("layer3", model.layer3),
        ("layer4", model.layer4), ("features", model.avgpool),
    )]
    try:
        with torch.inference_mode():
            boundaries["output"] = model(x).numpy()
    finally:
        for handle in handles:
            handle.remove()
    boundaries["features"] = boundaries["features"].reshape(1, 512)

    groups = [(model.conv1, model.bn1)]
    for layers in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layers:
            groups.extend(((block.conv1, block.bn1), (block.conv2, block.bn2)))
            if block.downsample is not None:
                groups.append(tuple(block.downsample))
    def pack(tensor):
        return tensor.detach().numpy().astype("<f4").tobytes()

    original = b"".join(pack(t) for layer, bn in groups for t in (
        layer.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var))
    original += pack(model.fc.weight) + pack(model.fc.bias)
    eps = [bn.eps for _, bn in groups]
    declaration = dict(format=spec.FULL_FORMAT, folding=spec.FOLDING, ops=spec.FULL_OPS,
                       inputShape=[1, 3, 224, 224], outputShape=[1, 1000],
                       originalSha256=spec.conv.digest(original), epsilon=eps)
    ts = tmp_path / "weights"
    ts.write_bytes(original)
    xp = tmp_path / "input"
    xp.write_bytes(pack(x))
    cp = tmp_path / "contract"
    top = np.argsort(-boundaries["output"][0])[:5]

    if not image:
        sources, head, _ = spec.full_weights(original, eps)
        references = []
        for convolution in (spec.conv.reference, spec.vectorized_convolution):
            start = time.perf_counter()
            reference = spec.full_reference(x.numpy(), sources, head, convolution)
            references.append(reference)
            print(convolution.__name__, "full independent reference ms", (time.perf_counter() - start) * 1000)
        for name in boundaries:
            np.testing.assert_allclose(references[0][name], references[1][name], atol=1e-10, rtol=1e-10)

    def run(name, cache, modified=None, reject=False):
        cp.write_text(json.dumps(modified or declaration))
        out = tmp_path / name
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, str(TOOLS / "resnet_block_declared.py"),
                               "--helper", str(helper), "--input", str(xp), "--weights", str(ts),
                               "--contract", str(cp), "--cache", str(cache), "--out", str(out)],
                              capture_output=True, timeout=90)
        total = (time.perf_counter() - start) * 1000
        if reject:
            assert proc.returncode == 2 and not out.exists()
            assert "unsupported stage" in proc.stderr.decode()
            return
        assert proc.returncode == 0, proc.stderr.decode()
        report = json.loads(proc.stdout)
        assert report["parity"] and report["gpu"]["kernel"] == "resnet18_fp32"
        assert report["gpu"]["gpuMicros"] > 0
        errors = {}
        for key, wanted in boundaries.items():
            assert report["boundaries"][key]["shape"] == list(wanted.shape)
            actual = np.fromfile(out / f"{key}.f32", dtype="<f4").reshape(wanted.shape)
            np.testing.assert_allclose(actual, wanted, atol=2e-4, rtol=2e-4)
            errors[key] = float(np.max(np.abs(actual - wanted)))
            if key == "output":
                np.testing.assert_array_equal(np.argsort(-actual[0])[:5], top)
        return report, [total, report["gpu"]["gpuMicros"] / 1000, report["cpuReferenceMillis"]], errors

    samples = [[], []]
    for i in range(5):
        cache = tmp_path / f"cache-{i}"
        cold, c, errors = run(f"c{i}", cache)
        warm, w, warm_errors = run(f"w{i}", cache)
        assert not cold["cache"]["hit"] and warm["cache"]["hit"]
        assert cold["boundaries"] == warm["boundaries"] and errors == warm_errors
        samples[0].append(c)
        samples[1].append(w)
    print("astronaut" if image else "synthetic", warm["gpu"]["device"], errors, "top5", top.tolist(),
          [[(statistics.median(col), min(col), max(col)) for col in zip(*rows)] for rows in samples])

    if not image:
        cache = tmp_path / "cache-0"
        updated = dict(declaration, epsilon=[2e-5, *eps[1:]])
        # Change by one FP32 ULP to test the original weight hash without
        # materially changing the external TorchVision parity expectation.
        changed, _, _ = run("changed-epsilon", cache, updated)
        assert not changed["cache"]["hit"] and changed["cache"]["key"] != warm["cache"]["key"]
        modified = np.frombuffer(original, dtype="<f4").copy()
        modified[0] = np.nextafter(modified[0], np.float32(np.inf))
        ts.write_bytes(modified.tobytes())
        weighted = dict(declaration, originalSha256=spec.conv.digest(modified.tobytes()))
        changed, _, _ = run("changed-weight", cache, weighted)
        assert not changed["cache"]["hit"] and changed["cache"]["key"] != warm["cache"]["key"]
        ts.write_bytes(original)
        ops = copy.deepcopy(spec.FULL_OPS)
        ops["head"][1] = 999
        run("wrong-ops", cache, dict(declaration, ops=ops), reject=True)
        (cache / f"{warm['cache']['key']}.json").write_text("{}")
        restored, _, _ = run("recovered", cache)
        assert not restored["cache"]["hit"] and restored["boundaries"] == warm["boundaries"]
