"""Build a fixed FP32 value DAG and optionally reuse its measured Metal plan."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np

FORMAT = "leviathan-declared-dag-1"
MAX_JSON = 1 << 20
ATEN = {"dense": "aten.linear.default", "relu": "aten.relu.default", "add": "aten.add.Tensor"}


class BuildError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def read(path, limit=MAX_JSON):
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise BuildError(f"{path.name} exceeds {limit} bytes")
    return data


def load_model(contract_path, graph_path, weights_path):
    contract = json.loads(read(contract_path))
    graph_data = read(graph_path)
    blob = read(weights_path, 16 << 20)
    graph = json.loads(graph_data)
    if (set(contract) != {"format", "graphSha256", "weightsSha256", "inputShape", "outputShape"}
            or contract["format"] != FORMAT or contract["graphSha256"] != digest(graph_data)
            or contract["weightsSha256"] != digest(blob)):
        raise BuildError("declaration format or content hash mismatch")
    if graph["format"] != "leviathan-value-dag-1" or not 1 <= len(graph["nodes"]) <= 64:
        raise BuildError("unsupported graph format or node count")
    width = graph["inputDim"]
    if type(width) is not int or not 1 <= width <= 4096 or contract["inputShape"] != [1, width]:
        raise BuildError("input shape mismatch")
    dims, layouts, offset = [width], [], 0
    for idx, node in enumerate(graph["nodes"], 1):
        op, refs, out = node["op"], node["inputs"], node["outputDim"]
        if (node["id"] != idx or op not in ATEN or len(refs) != (2 if op == "add" else 1)
                or any(type(ref) is not int or not 0 <= ref < idx for ref in refs)
                or type(out) is not int or not 1 <= out <= 4096):
            raise BuildError("invalid node or input geometry")
        if op == "dense":
            size = dims[refs[0]] * out
            layouts.append(dict(nodeID=idx, inDim=dims[refs[0]], outDim=out,
                                weightOffset=offset, weightCount=size, biasOffset=offset + size))
            offset += size + out
        elif any(dims[ref] != out for ref in refs):
            raise BuildError("invalid elementwise shape")
        dims.append(out)
    if (graph["outputValue"] != len(graph["nodes"]) or contract["outputShape"] != [1, dims[-1]]
            or not layouts or offset * 4 != len(blob)):
        raise BuildError("output shape or weight count mismatch")
    weights = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(weights).all():
        raise BuildError("non-finite weights")
    return contract, graph, graph_data, blob, weights, layouts


def read_input(path, width):
    values = json.loads(read(path))
    if (type(values) is not list or len(values) != 1 or type(values[0]) is not list
            or len(values[0]) != width or any(type(x) not in (int, float) for x in values[0])):
        raise BuildError("input must be one numeric row of the declared width")
    tensor = np.asarray(values[0], dtype="<f4")
    if not np.isfinite(tensor).all():
        raise BuildError("non-finite input")
    return tensor


def reference(graph, weights, layouts, x):
    """CPU NumPy oracle; evaluates immutable values independently of Metal."""
    values = [x]
    dense = {row["nodeID"]: row for row in layouts}
    for node in graph["nodes"]:
        op = node["op"]
        refs = [values[idx] for idx in node["inputs"]]
        if op == "dense":
            row = dense[node["id"]]
            coeff = weights[row["weightOffset"]:row["biasOffset"]].reshape(row["inDim"], row["outDim"])
            values.append(refs[0] @ coeff + weights[row["biasOffset"]:row["biasOffset"] + row["outDim"]])
        elif op == "relu":
            values.append(np.maximum(refs[0], 0))
        else:
            values.append(refs[0] + refs[1])
    out = values[graph["outputValue"]]
    if not np.isfinite(out).all():
        raise BuildError("non-finite CPU reference")
    return out


def helper_call(helper, *args):
    result = subprocess.run([str(helper), *map(str, args)], capture_output=True, timeout=120)
    if result.returncode:
        raise BuildError(result.stderr.decode(errors="replace").strip()[:1000])
    return result.stdout


def verify(helper, stage, x, wanted, scratch):
    input_path, report_path = scratch / "input.f32", scratch / "run.json"
    input_path.write_bytes(x.tobytes())
    report_path.unlink(missing_ok=True)
    helper_call(helper, "run", "--bundle", stage, "--input-f32", input_path,
                "--output-json", report_path)
    report = json.loads(read(report_path))
    gpu = report["gpu"]
    actual = np.asarray(report["output"], dtype=np.float32)
    if (report["format"] != "leviathan-dag-run-1"
            or report["outputShape"] != [1, len(wanted)]
            or gpu["commandStatus"] != "completed" or gpu["computeDispatches"] <= 0
            or not 0 < gpu["gpuCommandBufferMicros"] < float("inf")
            or actual.shape != wanted.shape
            or not np.allclose(actual, wanted, atol=1e-4, rtol=1e-4)):
        raise BuildError("Metal evidence or independent CPU parity failed")
    return report


def publish(stage, out):
    """macOS exclusive rename, including when a concurrent output is empty."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renamex_np
    rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    if rename(os.fsencode(stage), os.fsencode(out), 4):
        raise OSError(ctypes.get_errno(), "exclusive bundle publish failed", out)


def cache_path(directory, key):
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise BuildError("cache must be a private directory owned by the caller")
    return directory / (key + ".json")


def save_cache(path, key, stage):
    bundle = read(stage / "bundle.json")
    record = encode({"key": key, "bundle": bundle.decode(), "sha256": digest(bundle)})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(record)
            handle.flush()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def build(args):
    started = time.perf_counter()
    if sys.platform != "darwin":
        raise BuildError("declared BUILD requires the Metal helper on macOS")
    if args.out.exists() or args.out.is_symlink():
        raise BuildError("output bundle already exists")
    contract, graph, graph_data, blob, weights, layouts = load_model(args.contract, args.graph, args.weights)
    example = read_input(args.example_input, graph["inputDim"])
    actual = read_input(args.input, graph["inputDim"])
    ref_example = reference(graph, weights, layouts, example)
    ref_actual = reference(graph, weights, layouts, actual)
    source = b"# Declarative DAG; contract is encoded below.\n# " + encode(contract) + b"\n"
    backend = digest(Path(__file__).read_bytes())
    manifest = encode(dict(format="leviathan-dag-weights-1", inputDim=graph["inputDim"],
                           floatCount=len(weights), weightsSha256=digest(blob), layouts=layouts,
                           refInput=example.tolist(), refOutput=ref_example.tolist(), modelSeed=0))
    binding = encode(dict(modelSourceSha256=digest(source), modelSourceName="declared-dag",
                          modelSeed=0, backendSourceSha256=backend, backendSourceFiles=1,
                          torchVersion="declared-numpy-" + np.__version__,
                          pythonVersion=platform.python_version(),
                          atenOps=[ATEN[n["op"]] for n in graph["nodes"]],
                          exampleInputSha256=digest(example.tobytes())))
    key = digest(encode([FORMAT, "metal", platform.mac_ver()[0], digest(binding), digest(manifest),
                         digest(read(args.helper, 512 << 20)), args.diagnostic_fail_optimizations]))
    path = cache_path(args.cache, key) if args.cache else None
    prepared = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".lev-declared-", dir=args.out.parent) as temporary:
        scratch = Path(temporary)
        stage = scratch / "bundle"
        stage.mkdir()
        for name, data in (("model.graph.json", graph_data), ("model.weights.f32", blob),
                           ("model.manifest.json", manifest), ("model_source.py", source)):
            (stage / name).write_bytes(data)
        bundle_file, sha_file = stage / "bundle.json", stage / "bundle.sha256"
        report, selection, reason = None, None, "disabled" if path is None else "missing"
        forge_millis = 0.0
        if path is not None and path.exists():
            try:
                cached = json.loads(read(path, 2 << 20))
                bundle = cached["bundle"].encode()
                if cached["key"] != key or cached["sha256"] != digest(bundle):
                    raise BuildError("cache digest mismatch")
                meta = json.loads(bundle)
                if (encode(meta["python"]) != binding
                        or meta["selection"]["diagnosticFailOptimizations"] != args.diagnostic_fail_optimizations):
                    raise BuildError("cached plan binding mismatch")
                bundle_file.write_bytes(bundle)
                sha_file.write_text(cached["sha256"] + "\n")
                report = verify(args.helper, stage, actual, ref_actual, scratch)
                reason = "verified"
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                reason = f"rejected: {exc}"[:300]
                bundle_file.unlink(missing_ok=True)
                sha_file.unlink(missing_ok=True)
        if report is None:
            binding_file = scratch / "binding.json"
            binding_file.write_bytes(binding)
            flags = ["--diagnostic-fail-optimizations"] if args.diagnostic_fail_optimizations else []
            forge_start = time.perf_counter()
            selection = json.loads(helper_call(args.helper, "forge", "--bundle", stage,
                                               "--python-binding", binding_file, *flags))
            forge_millis = (time.perf_counter() - forge_start) * 1000
            report = verify(args.helper, stage, actual, ref_actual, scratch)
            if path is not None:
                save_cache(path, key, stage)
        publish(stage, args.out)
    return {"cache": {"hit": reason == "verified", "reason": reason, "key": key},
            "selection": selection, "run": report, "parity": True,
            "timing": {"prepareMillis": (prepared - started) * 1000, "forgeMillis": forge_millis,
                       "totalMillis": (time.perf_counter() - started) * 1000}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "graph", "weights", "example-input", "input", "helper", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--diagnostic-fail-optimizations", action="store_true")
    try:
        print(encode(build(parser.parse_args())).decode())
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.TimeoutExpired) as exc:
        print(f"declared BUILD: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
