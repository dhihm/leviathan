"""Checkpoint-backed 224×224 stem and stage 1 validation on Metal."""
import copy
import json
import statistics
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tools.forge_run import resnet_block_declared as stage
from tools.forge_run.test_resnet_block_declared import TOOLS, checkpoint_and_helper

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Metal")


def test_stage1(checkpoint_and_helper, tmp_path):
    state, helper = checkpoint_and_helper
    prefixes = [("conv1", "bn1")]
    for index in range(2):
        prefixes.extend((f"layer1.{index}.conv{j}", f"layer1.{index}.bn{j}") for j in (1, 2))
    groups = [tuple(state[key].detach() for key in (f"{c}.weight", f"{b}.weight", f"{b}.bias",
                                                    f"{b}.running_mean", f"{b}.running_var"))
              for c, b in prefixes]
    x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    weights = tmp_path / "weights"
    input_path = tmp_path / "input"
    contract = tmp_path / "contract"
    input_path.write_bytes(x.numpy().astype("<f4").tobytes())
    eps = [1e-5] * 5

    def oracle(tensors, e):
        def layer(value, index, stride):
            w, scale, bias, mean, var = tensors[index]
            z = F.conv2d(value, w, stride=stride, padding=w.shape[-1] // 2)
            return F.batch_norm(z, mean, var, scale, bias, training=False, eps=e[index])
        stem = layer(x, 0, 2).relu()
        pool = F.max_pool2d(stem, 3, 2, 1)
        first = (layer(layer(pool, 1, 1).relu(), 2, 1) + pool).relu()
        final = (layer(layer(first, 3, 1).relu(), 4, 1) + first).relu()
        return dict(stem=stem, pool=pool, block1=first, output=final)

    expected = oracle(groups, eps)

    def run(name, cache, tensors=None, ops=None, reject=False, executor=helper):
        payload = b"".join(t.numpy().astype("<f4").tobytes() for layer in (tensors or groups) for t in layer)
        weights.write_bytes(payload)
        declaration = dict(format=stage.STAGE_FORMAT, folding=stage.FOLDING,
                           inputShape=list(x.shape), outputShape=list(expected["output"].shape),
                           originalSha256=stage.conv.digest(payload), epsilon=eps, ops=ops or stage.STAGE_OPS)
        contract.write_text(json.dumps(declaration))
        out = tmp_path / name
        command = [sys.executable, str(TOOLS / "resnet_block_declared.py"), "--helper", str(executor),
                   "--input", str(input_path), "--weights", str(weights), "--contract", str(contract),
                   "--cache", str(cache), "--out", str(out)]
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, timeout=90)
        elapsed = (time.perf_counter() - start) * 1000
        if reject:
            assert proc.returncode == 2 and "unsupported stage" in proc.stderr.decode()
            assert not out.exists()
            return
        assert proc.returncode == 0, proc.stderr.decode()
        report = json.loads(proc.stdout)
        assert report["parity"] and report["gpu"]["gpuMicros"] > 0
        assert report["gpu"]["kernel"] == "resnet_stage1_fp32"
        errors = {}
        for boundary, target in (oracle(tensors, eps) if tensors else expected).items():
            assert report["boundaries"][boundary]["shape"] == list(target.shape)
            actual = np.fromfile(out / f"{boundary}.f32", dtype="<f4").reshape(target.shape)
            np.testing.assert_allclose(actual, target.numpy(), rtol=2e-4, atol=2e-4)
            errors[boundary] = float(np.max(np.abs(actual - target.numpy())))
        return report, (elapsed, report["gpu"]["gpuMicros"] / 1000, report["cpuReferenceMillis"]), errors

    samples = [[], []]
    for i in range(5):
        cache = tmp_path / f"cache-{i}"
        c, cold, errors = run(f"cold-{i}", cache)
        w, warm, warm_errors = run(f"warm-{i}", cache)
        assert not c["cache"]["hit"] and w["cache"]["hit"]
        assert c["boundaries"] == w["boundaries"] and errors == warm_errors
        samples[0].append(cold)
        samples[1].append(warm)
    print(w["gpu"]["device"], w["boundaries"], errors,
          [[(statistics.median(col), min(col), max(col)) for col in zip(*rows)] for rows in samples])

    altered = list(groups)
    weight, scale, bias, mean, var = groups[0]
    altered[0] = (weight, scale, bias, mean + 0.01, var)
    cache = tmp_path / "cache-0"
    changed, _, _ = run("changed-source", cache, tensors=altered)
    assert not changed["cache"]["hit"] and changed["cache"]["key"] != w["cache"]["key"]
    alternate = tmp_path / "alternate-helper"
    subprocess.run(["swiftc", "-Onone", str(TOOLS / "metal_conv2d.swift"), "-o", str(alternate)],
                   check=True, timeout=90)
    changed_executor, _, _ = run("changed-executor", cache, executor=alternate)
    assert not changed_executor["cache"]["hit"] and changed_executor["cache"]["key"] != w["cache"]["key"]
    assert changed_executor["boundaries"] == w["boundaries"]
    before = sorted(cache.iterdir())
    for name, value in (("dilation", 2), ("ceilMode", True), ("returnIndices", True), ("padding", 0)):
        ops = copy.deepcopy(stage.STAGE_OPS)
        ops["maxPool"][name] = value
        run(f"bad-{name}", cache, ops=ops, reject=True)
    assert sorted(cache.iterdir()) == before
    (cache / (w["cache"]["key"] + ".json")).write_text("{}")
    rebuilt, _, _ = run("recovered", cache)
    assert not rebuilt["cache"]["hit"] and rebuilt["boundaries"] == w["boundaries"]


def test_maxpool_negative_padding_on_metal(tmp_path):
    source = (TOOLS / "metal_conv2d.swift").read_text()
    start, end = source.index("\ndo {"), source.index("\nprivate struct BlockCommand")
    harness = """
let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(source: convolutionSource, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "maxpool_3x3_s2_p1")!)
let command = device.makeCommandQueue()!.makeCommandBuffer()!
let values: [Float] = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
let input = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
let output = device.makeBuffer(length: 16, options: .storageModeShared)!
try encodePool(command, pipeline, source: input, output: output, channels: 1, height: 3, width: 3)
command.commit()
command.waitUntilCompleted()
precondition(command.status == .completed)
let actual = output.contents().bindMemory(to: Float.self, capacity: 4)
precondition(Array(UnsafeBufferPointer(start: actual, count: 4)) == [-5, -4, -2, -1])
"""
    main = tmp_path / "main.swift"
    main.write_text(source[:start] + "\n" + harness + source[end:])
    executable = tmp_path / "probe"
    subprocess.run(["swiftc", "-O", str(main), "-o", str(executable)], check=True, timeout=90)
    subprocess.run([str(executable)], check=True, timeout=30)
