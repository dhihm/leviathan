"""Build and verify one inference-mode ResNet basic block on Metal."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

if __package__:
    from . import conv2d_build_declared as conv
else:
    import conv2d_build_declared as conv

FORMAT = "leviathan-declared-resnet-basic-block-1"
STAGE_FORMAT = "leviathan-declared-resnet18-stage1-1"
STAGE_OPS = {
    "stem": {"kernel": 7, "stride": 2, "padding": 3},
    "maxPool": {"kernel": 3, "stride": 2, "padding": 1, "dilation": 1,
                "ceilMode": False, "returnIndices": False},
    "blocks": ["identity", "identity"],
}
STAGE_SHAPES = {"stem": [1, 64, 112, 112], "pool": [1, 64, 56, 56],
                "block1": [1, 64, 56, 56], "output": [1, 64, 56, 56]}
FOLDING = "eval-batchnorm-fp64-fold-to-fp32-1"


def geometry(declaration):
    if (type(declaration) is not dict or set(declaration) != {
            "format", "inputShape", "outputShape", "skip", "stride", "folding", "originalSha256", "epsilon"
    } or declaration["format"] != FORMAT or declaration["folding"] != FOLDING):
        raise conv.BuildError("unsupported residual declaration or BatchNorm folding")
    _, ci, h, w = conv.shape(declaration["inputShape"], [1, 256, 512, 512])
    _, co, oh, ow = conv.shape(declaration["outputShape"], [1, 256, 512, 512])
    stride = declaration["stride"]
    projection = declaration["skip"] == "projection"
    if (type(stride) is not int or (stride != 2 if projection else
                                   stride != 1 or ci != co or declaration["skip"] != "identity")
            or (oh, ow) != ((h - 1) // stride + 1, (w - 1) // stride + 1)
            or max(ci * h * w, co * oh * ow) > conv.MAX_VALUES
            or co * oh * ow * (ci * 9 + co * 9 + (ci if projection else 0)) > 200_000_000):
        raise conv.BuildError("invalid residual geometry")
    layers = [(ci, 3), (co, 3)] + ([(ci, 1)] if projection else [])
    eps = declaration["epsilon"]
    if (type(eps) is not list or len(eps) != len(layers)
            or any(type(e) not in (int, float) or not 0 < e <= 1 for e in eps)):
        raise conv.BuildError("invalid BatchNorm epsilon")
    return ci, h, w, co, stride, projection, layers


def folding(original, channels, layers, eps):
    count = sum(channels * (ci * k * k + 4) for ci, k in layers)
    remaining = conv.tensor(original, [count])
    sources, packed = [], []
    for (ci, k), epsilon in zip(layers, eps):
        n = channels * ci * k * k
        weights = remaining[:n].reshape(channels, ci, k, k)
        gamma, beta, mean, var = remaining[n:n + 4 * channels].reshape(4, channels)
        remaining = remaining[n + 4 * channels:]
        if np.any(var < 0):
            raise conv.BuildError("negative BatchNorm variance")
        scale = gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + epsilon)
        folded = (weights * scale[:, None, None, None], beta - mean * scale)
        for values in folded:
            with np.errstate(over="ignore", invalid="ignore"):
                cast = values.astype("<f4")
            if not np.isfinite(cast).all():
                raise conv.BuildError("non-finite folded tensor")
            packed.append(cast.tobytes())
        sources.append((weights, gamma, beta, mean, var, epsilon))
    return sources, b"".join(packed)


def layer(data, params, step):
    weights, gamma, beta, mean, var, eps = params
    co, ci, k, _ = weights.shape
    _, _, h, w = data.shape
    convolved = conv.reference(data, weights, (ci, h, w, co, k, k, step, k // 2))
    return ((convolved - mean[:, None, None]) / np.sqrt(var.astype(np.float64)[:, None, None] + eps)
            * gamma[:, None, None] + beta[:, None, None])


def reference(x, sources, stride):
    first = np.maximum(layer(x, sources[0], stride), 0)
    main = layer(first, sources[1], 1)
    skip = layer(x, sources[2], stride) if len(sources) == 3 else x
    return np.maximum(main + skip, 0)


def stage_geometry(declaration):
    if (set(declaration) != {"format", "inputShape", "outputShape", "ops", "folding", "originalSha256", "epsilon"}
            or declaration["folding"] != FOLDING
            or conv.shape(declaration["inputShape"], [1, 3, 224, 224]) != [1, 3, 224, 224]
            or conv.shape(declaration["outputShape"], STAGE_SHAPES["output"]) != STAGE_SHAPES["output"]
            or conv.encode(declaration["ops"]) != conv.encode(STAGE_OPS)):
        raise conv.BuildError("unsupported stage shape or operations; max pool requires 3x3/stride 2/pad 1")
    eps = declaration["epsilon"]
    if type(eps) is not list or len(eps) != 5 or any(type(e) not in (int, float) or not 0 < e <= 1 for e in eps):
        raise conv.BuildError("invalid BatchNorm epsilon")
    return 3, 224, 224, 64, 2, False, [(3, 7)] + [(64, 3)] * 4


def stage_reference(x, sources):
    stem = np.maximum(layer(x, sources[0], 2), 0)
    padded = np.pad(stem, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(-2, -1))
    pool = windows[:, :, ::2, ::2].max(axis=(-2, -1))
    first = reference(pool, sources[1:3], 1)
    return {"stem": stem, "pool": pool, "block1": first, "output": reference(first, sources[3:], 1)}


def build(args):
    started = time.perf_counter()
    if sys.platform != "darwin":
        raise conv.BuildError("declared residual BUILD requires macOS")
    if args.out.exists() or args.out.is_symlink():
        raise conv.BuildError("output already exists")
    declaration = json.loads(conv.read(args.contract))
    stage1 = isinstance(declaration, dict) and declaration.get("format") == STAGE_FORMAT
    ci, h, w, co, stride, projection, layers = (stage_geometry if stage1 else geometry)(declaration)
    format_id = STAGE_FORMAT if stage1 else FORMAT
    original = conv.read(args.weights, 8 << 20)
    if conv.digest(original) != declaration["originalSha256"]:
        raise conv.BuildError("original tensor digest mismatch")
    raw = conv.read(args.input, conv.MAX_VALUES * 4)
    x = conv.tensor(raw, declaration["inputShape"])
    sources, packed = folding(original, co, layers, declaration["epsilon"])
    reference_started = time.perf_counter()
    expected = stage_reference(x, sources) if stage1 else {"output": reference(x, sources, stride)}
    if any(not np.isfinite(values).all() for values in expected.values()):
        raise conv.BuildError("non-finite reference output")
    reference_millis = (time.perf_counter() - reference_started) * 1000
    helper_identity = conv.digest(conv.read(args.helper, 512 << 20))
    key = conv.digest(conv.encode([declaration, helper_identity, conv.digest(Path(__file__).read_bytes()),
                                   conv.digest(Path(conv.__file__).read_bytes()), platform.version()]))
    cache = conv.cache_path(args.cache, key) if args.cache else None
    device, width = conv.cached_plan(cache, key, format_id)
    with tempfile.TemporaryDirectory(prefix=".lev-resnet-block-", dir=args.out.parent) as temp:
        folder = Path(temp)
        stage = folder / "result"
        stage.mkdir()
        config, input_file, weights_file = (folder / name for name in ("config", "input", "weights"))
        config.write_bytes(conv.encode(dict(channels=ci, height=h, width=w, outputChannels=co,
                                            stride=stride, projection=projection,
                                            cachedDevice=device, cachedWidth=width)))
        input_file.write_bytes(raw)
        weights_file.write_bytes(packed)
        output = stage / "output.f32"
        mode = "--stage1" if stage1 else "--residual"
        proc = subprocess.run([str(args.helper.resolve()), mode, *map(str, (
            config, input_file, weights_file, output))], capture_output=True, timeout=120)
        if proc.returncode:
            raise conv.BuildError(proc.stderr.decode(errors="replace")[:1000])
        gpu = json.loads(proc.stdout)
        kernel = "resnet_stage1_fp32" if stage1 else "resnet_basic_block_fp32"
        if gpu["kernel"] != kernel or not gpu["device"] or not 0 < gpu["gpuMicros"] < float("inf"):
            raise conv.BuildError("Metal GPU evidence failed")
        boundaries = {}
        for name, wanted in expected.items():
            payload = conv.read(stage / (name + ".f32"), conv.MAX_VALUES * 4)
            dimensions = STAGE_SHAPES[name] if stage1 else declaration["outputShape"]
            actual = conv.tensor(payload, dimensions)
            if not np.allclose(actual, wanted, atol=2e-4, rtol=2e-4):
                raise conv.BuildError(f"independent unfused parity failed at {name}")
            boundaries[name] = dict(shape=dimensions, sha256=conv.digest(payload),
                                    maxAbsError=float(np.max(np.abs(actual - wanted))))
        hit = bool(gpu["reused"] and device == gpu["registryID"] and width == gpu["width"])
        report = dict(contract=declaration, foldedSha256=conv.digest(packed),
                      inputSha256=conv.digest(raw), outputSha256=boundaries["output"]["sha256"],
                      gpu=gpu, parity=True, maxAbsError=boundaries["output"]["maxAbsError"],
                      cache=dict(hit=hit, key=key))
        if stage1:
            report["boundaries"] = boundaries
            report["cpuReferenceMillis"] = reference_millis
        if cache is not None and not hit:
            conv.store_plan(cache, key, folder, gpu, format_id)
        report["buildMillisBeforePublication"] = (time.perf_counter() - started) * 1000
        (stage / "report.json").write_bytes(conv.encode(report))
        conv.publish(stage, args.out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "weights", "input", "helper", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    try:
        print(conv.encode(build(parser.parse_args())).decode())
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"declared residual BUILD: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
