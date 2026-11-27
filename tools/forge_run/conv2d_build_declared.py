"""Verify a single declared FP32 NCHW convolution with native Metal."""

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

if __package__:
    from .dag_build_declared import BuildError, cache_path, digest, encode, publish, read, save_cache
else:
    from dag_build_declared import BuildError, cache_path, digest, encode, publish, read, save_cache

FORMAT = "leviathan-declared-conv2d-1"
MAX_VALUES = 1 << 20


def shape(values, limits):
    if (type(values) is not list or len(values) != len(limits)
            or any(type(v) is not int or not 1 <= v <= cap for v, cap in zip(values, limits))):
        raise BuildError("invalid tensor shape")
    return values


def contract_geometry(contract):
    fields = {"format", "inputShape", "weightShape", "outputShape", "weightsSha256",
              "stride", "padding", "groups", "dilation"}
    if type(contract) is not dict or set(contract) != fields or contract["format"] != FORMAT:
        raise BuildError("unsupported Conv2D declaration")
    if any(type(contract[k]) is not int or contract[k] != 1 for k in ("groups", "dilation")):
        raise BuildError("Conv2D requires groups=1 and dilation=1")
    _, ci, h, w = shape(contract["inputShape"], [1, 256, 512, 512])
    co, weight_channels, kh, kw = shape(contract["weightShape"], [256, 256, 11, 11])
    stride, padding = contract["stride"], contract["padding"]
    if (type(stride) is not int or not 1 <= stride <= 4
            or type(padding) is not int or not 0 <= padding <= 5
            or weight_channels != ci or h + 2 * padding < kh or w + 2 * padding < kw):
        raise BuildError("invalid convolution geometry")
    oh, ow = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    output = shape(contract["outputShape"], [1, 256, 522, 522])
    if (output != [1, co, oh, ow]
            or any(math.prod(contract[k]) > MAX_VALUES for k in ("inputShape", "weightShape", "outputShape"))
            or co * oh * ow * ci * kh * kw > 100_000_000):
        raise BuildError("output shape mismatch or convolution exceeds work limit")
    return ci, h, w, co, kh, kw, stride, padding


def tensor(data, dimensions):
    if len(data) != math.prod(dimensions) * 4:
        raise BuildError("tensor byte count does not match declaration")
    result = np.frombuffer(data, dtype="<f4").reshape(dimensions)
    if not np.isfinite(result).all():
        raise BuildError("non-finite tensor")
    return result


def reference(x, weights, geometry):
    *_, kh, kw, stride, padding = geometry
    padded = np.pad(x[0].astype(np.float64), ((0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))
    out = np.einsum("cyxij,ocij->oyx", windows[:, ::stride, ::stride], weights.astype(np.float64))
    return out[None, ...]


def cached_plan(path, key, format_id=FORMAT):
    if path is not None:
        try:
            record = json.loads(read(path))
            bundle = record["bundle"].encode()
            if record["key"] == key and record["sha256"] == digest(bundle):
                plan = json.loads(bundle)
                if (plan["format"] == format_id and plan["key"] == key
                        and type(plan["width"]) is int and 1 <= plan["width"] <= 1024
                        and type(plan["registryID"]) is str and plan["registryID"].isdigit()):
                    return plan["registryID"], plan["width"]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
    return "-", 0


def store_plan(path, key, directory, evidence, format_id):
    plan = directory / "plan"
    plan.mkdir()
    (plan / "bundle.json").write_bytes(encode({"format": format_id, "key": key,
                                               "registryID": evidence["registryID"],
                                               "width": evidence["width"]}))
    save_cache(path, key, plan)


def build(args):
    if sys.platform != "darwin":
        raise BuildError("declared Conv2D BUILD requires macOS")
    if args.out.exists() or args.out.is_symlink():
        raise BuildError("output already exists")
    declaration = json.loads(read(args.contract))
    geometry = contract_geometry(declaration)
    weights_data, input_data = (read(path, MAX_VALUES * 4) for path in (args.weights, args.input))
    if digest(weights_data) != declaration["weightsSha256"]:
        raise BuildError("weight digest mismatch")
    weights = tensor(weights_data, declaration["weightShape"])
    x = tensor(input_data, declaration["inputShape"])
    wanted = reference(x, weights, geometry)
    key = digest(encode([declaration, digest(read(args.helper, 512 << 20)),
                         digest(Path(__file__).read_bytes()), platform.mac_ver()[0], platform.version()]))
    path = cache_path(args.cache, key) if args.cache else None
    device, width = cached_plan(path, key)
    with tempfile.TemporaryDirectory(prefix=".lev-conv2d-", dir=args.out.parent) as temporary:
        directory = Path(temporary)
        stage = directory / "result"
        stage.mkdir()
        input_path, weights_path = directory / "input.f32", directory / "weights.f32"
        input_path.write_bytes(input_data)
        weights_path.write_bytes(weights_data)
        output = stage / "output.f32"
        proc = subprocess.run([str(args.helper.resolve()), str(input_path), str(weights_path), str(output),
                               *map(str, (*geometry, device, width))],
                              capture_output=True, check=False, timeout=120)
        if proc.returncode:
            raise BuildError(proc.stderr.decode(errors="replace")[:1000])
        evidence = json.loads(proc.stdout)
        output_data = read(output, MAX_VALUES * 4)
        actual = tensor(output_data, declaration["outputShape"])
        if (evidence["kernel"] != "conv2d_fp32" or not evidence["device"]
                or not 0 < evidence["gpuMicros"] < float("inf")
                or not np.allclose(actual, wanted, atol=1e-4, rtol=1e-4)):
            raise BuildError("Metal GPU evidence or independent convolution parity failed")
        hit = bool(evidence["reused"] and device == evidence["registryID"] and width == evidence["width"])
        report = {"contract": declaration, "inputSha256": digest(input_data),
                  "outputSha256": digest(output_data), "gpu": evidence, "parity": True,
                  "maxAbsError": float(np.max(np.abs(actual - wanted))), "cache": {"hit": hit, "key": key}}
        (stage / "report.json").write_bytes(encode(report))
        if path is not None and not hit:
            store_plan(path, key, directory, evidence, FORMAT)
        publish(stage, args.out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "weights", "input", "helper", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    try:
        print(encode(build(parser.parse_args())).decode())
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"declared Conv2D BUILD: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
