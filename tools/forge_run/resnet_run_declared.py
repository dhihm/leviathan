"""Run the same declarative ResNet-18 files with PyTorch CPU or CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from . import resnet_block_declared as spec
else:
    import resnet_block_declared as spec

conv = spec.conv


def execute(x, sources, classifier, stride, staged):
    def layer(data, params, step):
        weights, gamma, beta, mean, var, eps = params
        z = F.conv2d(data, weights, stride=step, padding=weights.shape[-1] // 2)
        return F.batch_norm(z, mean, var, gamma, beta, training=False, eps=eps)

    def block(data, layers, step):
        first = layer(data, layers[0], step).relu()
        main = layer(first, layers[1], 1)
        skip = layer(data, layers[2], step) if len(layers) == 3 else data
        return (main + skip).relu()

    if not staged:
        return {"output": block(x, sources, stride)}
    stem = layer(x, sources[0], 2).relu()
    pool = F.max_pool2d(stem, 3, stride=2, padding=1)
    first = block(pool, sources[1:3], 1)
    value = block(first, sources[3:5], 1)
    outputs = {"stem": stem, "pool": pool, "block1": first}
    if classifier is None:
        return {**outputs, "output": value}
    outputs["layer1"] = value
    for i in range(3):
        offset = 5 + i * 5
        value = block(value, sources[offset:offset + 3], 2)
        value = block(value, sources[offset + 3:offset + 5], 1)
        outputs[f"layer{i + 2}"] = value
    features = F.avg_pool2d(value, 7).flatten(1)
    outputs["features"] = features
    outputs["output"] = F.linear(features, classifier[:512000].reshape(1000, 512), classifier[512000:])
    return outputs


def run(args):
    cuda = args.backend == "cuda"
    if cuda and (torch.version.cuda is None or not torch.cuda.is_available()):
        raise conv.BuildError("CUDA requires an NVIDIA GPU and CUDA-enabled PyTorch")
    device = torch.device("cuda:0" if cuda else "cpu")
    declaration = json.loads(conv.read(args.contract))
    format_id = declaration.get("format") if isinstance(declaration, dict) else None
    full = format_id == spec.FULL_FORMAT
    staged = full or format_id == spec.STAGE_FORMAT
    _, _, _, channels, stride, _, layouts = (
        spec.stage_geometry(declaration, full) if staged else spec.geometry(declaration))
    xbytes = conv.read(args.input, conv.MAX_VALUES * 4)
    x = conv.tensor(xbytes, declaration["inputShape"])
    body = conv.read(args.weights, (48 if full else 8) << 20)
    if conv.digest(body) != declaration["originalSha256"]:
        raise conv.BuildError("original tensor digest mismatch")
    classifier = None
    if full:
        body, _, classifier = spec.split_full_weights(body)
    layers = spec.full_layers() if full else [(channels, ci, k) for ci, k in layouts]
    sources = spec.original_layers(body, layers, declaration["epsilon"])
    started = time.perf_counter()
    if full:
        references = spec.full_reference(x, sources, classifier)
        shapes = spec.FULL_SHAPES
    elif staged:
        references = spec.stage_reference(x, sources, spec.vectorized_convolution)
        shapes = spec.STAGE_SHAPES
    else:
        references = {"output": spec.reference(x, sources, stride)}
        shapes = {"output": declaration["outputShape"]}
    reference_millis = (time.perf_counter() - started) * 1000

    def tensor(value):
        return torch.from_numpy(value.astype(np.float32, copy=True)).to(device)

    tx = tensor(x)
    params = [(*map(tensor, layer[:-1]), layer[-1]) for layer in sources]
    head = tensor(classifier) if classifier is not None else None
    event_millis = None
    precision = torch.get_float32_matmul_precision()
    try:
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=False):
            if cuda:
                torch.set_float32_matmul_precision("highest")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            outputs = execute(tx, params, head, stride, staged)
            if cuda:
                end.record()
                end.synchronize()
                event_millis = start.elapsed_time(end)
    finally:
        if cuda:
            torch.set_float32_matmul_precision(precision)
    if cuda and not 0 < event_millis < float("inf"):
        raise conv.BuildError("CUDA execution returned no device timing evidence")
    if set(outputs) != set(references) or set(outputs) != set(shapes):
        raise conv.BuildError("execution boundaries do not match declaration")

    results, boundaries = {}, {}
    for name, result in outputs.items():
        if result.device != device:
            raise conv.BuildError(f"unexpected execution device at {name}")
        actual = result.cpu().numpy()
        reference = references[name]
        if (list(actual.shape) != shapes[name] or not np.isfinite(reference).all()
                or not np.isfinite(actual).all() or not np.allclose(actual, reference, atol=2e-4, rtol=2e-4)):
            raise conv.BuildError(f"independent unfused parity failed at {name}")
        results[name] = actual
        boundaries[name] = {"shape": shapes[name], "maxAbsError": float(np.max(np.abs(actual - reference)))}
    # Publish the complete archive exclusively, on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".lev-resnet-portable-", dir=args.out.parent) as tmp:
        archive = Path(tmp) / "boundaries.npz"
        np.savez(archive, **results)
        os.link(archive, args.out)
    report = {"backend": args.backend, "runtime": "PyTorch", "parity": True, "format": format_id,
              "device": torch.cuda.get_device_name(device) if cuda else "cpu",
              "originalSha256": declaration["originalSha256"], "inputSha256": conv.digest(xbytes),
              "cpuReferenceMillis": reference_millis, "boundaries": boundaries}
    if cuda:
        report["cudaEventMillis"] = event_millis
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    for name in ("contract", "weights", "input", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    try:
        args = parser.parse_args()
        print(conv.encode(run(args)).decode())
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"declared portable ResNet: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
