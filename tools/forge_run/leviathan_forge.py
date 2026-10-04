#!/usr/bin/env python3
"""Leviathan Forge Python runtime (generic Linear/ReLU subset).

build: TRUSTED model.py + example JSON + real input JSON
  -> torch.export(strict=True) capture -> existing subset validation/lowering
     (reused from tools/torch_frontend/export_torch_graph.py)
  -> leviathan-forge-helper forge: baseline-first legal candidate
     qualification, GPU benchmark, selection, bound bundle
  -> fresh helper process runs the saved plan on the real input
  -> GPU output compared with the source nn.Module here.
run: bundle + same model.py + (different) input JSON -> Python-side binding
  checks BEFORE GPU execution, Swift-side re-validation, saved plan only.

TRUST: the model file is executed as ordinary local Python with your
privileges (no sandbox). Bundles, inputs, and helper output are untrusted
data. Supported: FP32, static [1, INPUT_DIM], a single aten.linear (with
bias) / aten.relu chain, <=64 ops, <=16 MiB weights. Not an LLM runtime.
Model file contract: INPUT_DIM (int 1..4096) and build(seed) -> nn.Module.
Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch
from torch import nn

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
FRONTEND_PATH = REPO_ROOT / "tools" / "torch_frontend" / "export_torch_graph.py"
HELPER_NAME = "leviathan-forge-helper"
BUNDLE_FORMAT = "leviathan-forge-bundle-1"
MAX_DIM = 4096
MAX_SEED = 2**31 - 1
MAX_JSON = 1 << 20
MAX_WEIGHT_BYTES = 16 << 20
ATOL = 1e-4
RTOL = 1e-4
# Backend identity: relative paths + content (portable across checkouts).
BACKEND_SOURCE_PATTERNS = (
    "Package.swift",
    "Sources/LeviathanCore/*.swift",
    "Sources/leviathan-forge-helper/*.swift",
    "tools/torch_frontend/export_torch_graph.py",
    "tools/forge_run/leviathan_forge.py",
)
BUNDLE_FILES = (
    ("model.graph.json", "graphSha256", MAX_JSON),
    ("model.weights.f32", "weightsSha256", MAX_WEIGHT_BYTES),
    ("model.manifest.json", "manifestSha256", MAX_JSON),
    ("model_source.py", "modelSourceCopySha256", MAX_JSON),
)


class ForgeRunError(Exception):
    """Fail-closed rejection or failed check (CLI exit status 2)."""


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def read_bounded(path, cap, label):
    try:
        with open(path, "rb") as handle:
            data = handle.read(cap + 1)
    except OSError as exc:
        raise ForgeRunError(f"cannot read {label} ({path}): {exc}") from None
    if len(data) > cap:
        raise ForgeRunError(f"{label} exceeds {cap} bytes")
    return data


def _reject_constant(token):
    raise ForgeRunError(f"non-finite JSON constant '{token}' is not allowed")


def parse_json(data, label):
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except ForgeRunError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ForgeRunError(f"{label}: invalid JSON ({exc})") from None


def to_float32(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForgeRunError(f"{label} is not a JSON number")
    try:
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ForgeRunError(f"{label} is not finite")
        packed = struct.pack("<f", as_float)
    except (OverflowError, struct.error):
        raise ForgeRunError(f"{label} is not representable as a finite float32") from None
    return struct.unpack("<f", packed)[0]


def load_input_json(path, label="input"):
    """Bounded rank-2 [1, D] JSON array of finite float32 values -> tensor."""
    value = parse_json(read_bounded(path, MAX_JSON, label), label)
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)):
        raise ForgeRunError(f"{label}: must be a rank-2 JSON array of shape [1, INPUT_DIM]")
    row = value[0]
    if not 1 <= len(row) <= MAX_DIM:
        raise ForgeRunError(f"{label}: row length {len(row)} outside 1..{MAX_DIM}")
    return torch.tensor([[to_float32(v, f"{label}[0][{i}]") for i, v in enumerate(row)]], dtype=torch.float32)


FAILED_MARKER = "FORGE_FAILED.json"


def discard_own_report(path, ident, data):
    """Unlinks a report this build linked, only while it is still our inode with
    our exact bytes; a foreign or modified file is never touched."""
    try:
        st = os.lstat(path)
        if (st.st_dev, st.st_ino) == (ident.st_dev, ident.st_ino) and Path(path).read_bytes() == data:
            os.unlink(path)
            return "removed"
    except OSError:
        pass
    return "left in place (no longer ours)"


def validate_input_tensor(value, label, expected_dim=None, shape_name="static shape"):
    """Shared early input gate used for JSON and tensor inputs on the build,
    example, and run paths. It never converts anything. The input must be a
    CPU, dense strided torch.float32 [1, D] tensor with 1 <= D <= 4096 and
    finite values. Device and layout are checked before any value is read."""
    if not isinstance(value, torch.Tensor):
        raise ForgeRunError(f"{label}: expected a torch.Tensor or an input JSON path, got {type(value).__name__}")
    if value.device.type != "cpu":
        raise ForgeRunError(f"{label}: tensor is on device '{value.device}'; only CPU tensors are accepted")
    if value.layout is not torch.strided or getattr(value, "is_nested", False):
        raise ForgeRunError(f"{label}: layout {value.layout} is not supported; pass a dense strided tensor")
    if value.dtype is not torch.float32:
        raise ForgeRunError(f"{label}: dtype {value.dtype} is not torch.float32 (no implicit conversion)")
    if value.dim() != 2 or value.shape[0] != 1:
        raise ForgeRunError(f"{label}: shape {list(value.shape)} is not rank-2 with exactly one row [1, D]")
    if not 1 <= value.shape[1] <= MAX_DIM:
        raise ForgeRunError(f"{label}: width {value.shape[1]} outside 1..{MAX_DIM}")
    if expected_dim is not None and value.shape[1] != expected_dim:
        raise ForgeRunError(f"{label} shape {list(value.shape)} does not match the {shape_name} [1, {expected_dim}]")
    if not bool(torch.isfinite(value).all()):
        raise ForgeRunError(f"{label}: contains NaN or Inf")
    return value.detach().contiguous().clone()


def resolve_input(value, label):
    if isinstance(value, (str, os.PathLike)):
        value = load_input_json(value, label)
    return validate_input_tensor(value, label)


def forward_source(model, x, what):
    """Runs the trusted source module. This is not a sandbox: failures are only reported clearly."""
    try:
        with torch.no_grad():
            y = model(x)
    except Exception as exc:
        raise ForgeRunError(f"source module forward failed on {what}: {type(exc).__name__}: {exc}") from None
    if not (isinstance(y, torch.Tensor) and y.dtype is torch.float32 and y.dim() == 2 and y.shape[0] == 1
            and y.device.type == "cpu" and bool(torch.isfinite(y).all())):
        raise ForgeRunError(f"source module output on {what} is not a finite CPU float32 [1, N] tensor")
    return y


def publish_exclusive(src, dst):
    """Atomic no-replace rename (macOS renamex_np RENAME_EXCL, Linux renameat2
    RENAME_NOREPLACE). It never replaces dst, not even an empty directory that
    another process created concurrently."""
    libc = ctypes.CDLL(None, use_errno=True)
    s, d = os.fsencode(str(src)), os.fsencode(str(dst))
    if sys.platform == "darwin":
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rc = fn(s, d, 0x00000004)
    elif hasattr(libc, "renameat2"):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rc = fn(-100, s, -100, d, 1)
    else:
        raise ForgeRunError("no atomic no-replace rename on this platform; refusing to publish")
    if rc != 0:
        err = ctypes.get_errno()
        note = " (it already exists; the other output is left untouched)" if err in (errno.EEXIST, errno.ENOTEMPTY) else ""
        raise ForgeRunError(f"cannot publish {dst}: {os.strerror(err)}{note}")


def preserve_failed_stage(stage, out, error):
    """Keeps our own failed stage (raw evidence) under a non-final FAILED name."""
    try:
        (stage / FAILED_MARKER).write_text(json.dumps({"finalBundle": False, "intendedOutput": str(out),
                                                       "error": f"{type(error).__name__}: {error}"[:4000]}))
    except OSError:
        pass
    for attempt in range(50):
        target = out.parent / f".{out.name}.FAILED-{os.getpid()}-{int(time.time())}-{attempt}"
        try:
            publish_exclusive(stage, target)
            return target
        except ForgeRunError:
            continue
    return stage


def float32_bytes(tensor):
    values = tensor.detach().to(torch.float32).flatten().tolist()
    return struct.pack(f"<{len(values)}f", *values)


_FRONTEND = None


def frontend():
    global _FRONTEND
    if _FRONTEND is None:
        spec = importlib.util.spec_from_file_location("leviathan_forge_frontend", FRONTEND_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FRONTEND = module
    return _FRONTEND


def call_frontend(fn, *args):
    try:
        return fn(*args)
    except SystemExit as exc:
        raise ForgeRunError(str(exc.code)) from None


def backend_source_identity(root=REPO_ROOT):
    files = []
    for pattern in BACKEND_SOURCE_PATTERNS:
        matches = sorted(p for p in Path(root).glob(pattern) if p.is_file())
        if not matches:
            raise ForgeRunError(f"backend source missing: {pattern}")
        files.extend(matches)
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\0{sha256_hex(read_bounded(path, 4 << 20, rel))}\n".encode())
    return digest.hexdigest(), len(files)


def resolve_helper(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    elif os.environ.get("LEVIATHAN_FORGE_HELPER"):
        candidates = [Path(os.environ["LEVIATHAN_FORGE_HELPER"])]
    else:
        candidates = [REPO_ROOT / ".build" / "release" / HELPER_NAME, REPO_ROOT / ".build" / "debug" / HELPER_NAME]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ForgeRunError(f"helper not found ({', '.join(map(str, candidates))}); build: swift build -c release --product {HELPER_NAME}")


def load_model(model_path, seed, inputs=()):
    """Executes the TRUSTED model file (exactly the hashed bytes). Each
    (label, tensor) in inputs is checked against INPUT_DIM before build(seed)."""
    path = Path(model_path)
    source = read_bounded(path, MAX_JSON, "model definition")
    digest = sha256_hex(source)
    name = f"leviathan_forge_model_{digest[:16]}"
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    sys.modules[name] = module
    try:
        exec(compile(source, str(path.resolve()), "exec"), module.__dict__)
        input_dim = getattr(module, "INPUT_DIM", None)
        if type(input_dim) is not int or not 0 < input_dim <= MAX_DIM:
            raise ForgeRunError(f"{path}: INPUT_DIM must be an int in 1..{MAX_DIM}")
        if not callable(getattr(module, "build", None)):
            raise ForgeRunError(f"{path}: must define build(seed)")
        for label, tensor in inputs:
            validate_input_tensor(tensor, label, input_dim, "static shape (INPUT_DIM)")
        model = module.build(seed)
    except ForgeRunError:
        raise
    except Exception as exc:
        raise ForgeRunError(f"model definition failed: {type(exc).__name__}: {exc}") from None
    if not isinstance(model, nn.Module):
        raise ForgeRunError("build(seed) must return an nn.Module")
    model = model.eval()
    call_frontend(frontend().validate_module_side_conditions, model)
    return model, input_dim, digest, source


def capture(model, example, input_dim):
    """torch.export strict capture, existing subset validation, and a
    bounded exact differential gate (example + 8 seeded probes)."""
    fe = frontend()
    try:
        program = torch.export.export(model, (example,), strict=True)
    except Exception as exc:
        raise ForgeRunError(f"torch.export failed (unsupported model): {type(exc).__name__}: {str(exc)[:2000]}") from None
    ops, aten_ops = call_frontend(fe.validate_and_lower, program, input_dim)
    if not any(op[0] == "dense" for op in ops):
        raise ForgeRunError("the model must contain at least one nn.Linear")
    exported = program.module()
    probes = [example] + [torch.rand((1, input_dim), generator=torch.Generator().manual_seed(fe.REF_SEED + 100 + k))
                          for k in range(fe.SEMANTICS_INPUTS)]
    for k, probe in enumerate(probes):
        try:
            with torch.no_grad():
                want, got = model(probe), exported(probe)
        except Exception as exc:
            raise ForgeRunError(f"source module or exported graph failed on probe {k}: {type(exc).__name__}: {exc}") from None
        if not (isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor) and want.shape == got.shape
                and want.dtype is torch.float32 and torch.equal(want, got) and bool(torch.isfinite(want).all())):
            raise ForgeRunError(f"exported graph disagrees with the source module on probe {k}")
    return ops, aten_ops


def lower(ops, input_dim):
    graph_ops, blob, table, out_dim = [], bytearray(), [], input_dim
    for op in ops:
        if op[0] == "relu":
            graph_ops.append("relu")
            continue
        _, in_dim, o_dim, w, b = op
        graph_ops.append({"dense": {"outDim": o_dim}})
        wt = w.detach().to("cpu", torch.float32).t().contiguous().flatten()
        bt = b.detach().to("cpu", torch.float32).flatten()
        offset = len(blob) // 4
        table.append({"inDim": in_dim, "outDim": o_dim, "weightFloatOffset": offset,
                      "weightFloatCount": wt.numel(), "biasFloatOffset": offset + wt.numel()})
        blob += struct.pack(f"<{wt.numel()}f", *wt.tolist())
        blob += struct.pack(f"<{bt.numel()}f", *bt.tolist())
        out_dim = o_dim
    if len(blob) > MAX_WEIGHT_BYTES:
        raise ForgeRunError("weights exceed 16 MiB")
    graph = json.dumps({"inputDim": input_dim, "ops": graph_ops}, separators=(",", ":")).encode()
    return graph, bytes(blob), table, out_dim


def run_helper(helper, args):
    start = time.perf_counter()
    try:
        proc = subprocess.run([str(helper), *args], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ForgeRunError(f"helper failed to run: {exc}") from None
    if proc.returncode != 0:
        raise ForgeRunError(f"helper rejected or failed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[-4000:]}")
    return proc.stdout, time.perf_counter() - start


def build_bundle(model_path, example_input, user_input, out_dir, seed=0, helper=None,
                 diagnostic_fail_optimizations=False, report_path=None):
    """API: capture and forge into a hidden stage, run the supplied input on the
    GPU from that stage, and check source-PyTorch parity. Only after all of
    that passes is the stage published atomically and exclusively as out_dir.
    On any failure nothing final is created; the stage is kept as a marked
    .<name>.FAILED-* sibling. With report_path, the report is written to a
    private temp and hard-linked exclusively (never replacing anything) after
    run + parity but BEFORE the bundle rename; any report failure is a build
    failure (no bundle). If the bundle rename then fails, the report is removed
    only while it is still our own unchanged file. Not a joint transaction: a
    crash between the two steps can leave a report without its bundle; only the
    bundle directory is the build result.
    diagnostic_fail_optimizations is DIAGNOSTIC ONLY (fails every optimization
    candidate, never the baseline, to exercise fallback; recorded in bundle)."""
    started = time.perf_counter()
    out = Path(out_dir)
    if out.exists() or out.is_symlink():
        raise ForgeRunError(f"output bundle {out} already exists; refusing to overwrite")
    if not out.parent.is_dir():
        raise ForgeRunError(f"parent directory {out.parent} does not exist")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ForgeRunError(f"seed must be an int in 0..{MAX_SEED}")
    example = resolve_input(example_input, "example input")
    real = resolve_input(user_input, "input")
    report = Path(report_path) if report_path else None
    if report is not None and (report.exists() or report.is_symlink() or not report.parent.is_dir()):
        raise ForgeRunError(f"report {report} already exists or its directory does not exist")
    helper_path = resolve_helper(helper)
    source_sha, source_count = backend_source_identity()
    model, input_dim, model_sha, model_source = load_model(
        model_path, seed, (("example input", example), ("input", real)))
    ops, aten_ops = capture(model, example, input_dim)
    graph, blob, table, out_dim = lower(ops, input_dim)
    reference = forward_source(model, example, "the example input")
    manifest = {
        "format": "leviathan-forge-weights-1", "inputDim": input_dim, "floatCount": len(blob) // 4,
        "weightsSha256": sha256_hex(blob), "ops": table, "refInputSeed": None,
        "refInput": example.flatten().tolist(), "refOutput": reference.flatten().tolist(), "modelSeed": seed,
        "forgeRun": {"advisoryOnly": True, "torch": str(torch.__version__), "atenOps": aten_ops},
    }
    binding = {
        "modelSourceSha256": model_sha, "modelSourceName": Path(model_path).name[:256], "modelSeed": seed,
        "backendSourceSha256": source_sha, "backendSourceFiles": source_count,
        "torchVersion": str(torch.__version__), "pythonVersion": platform.python_version(),
        "atenOps": aten_ops, "exampleInputSha256": sha256_hex(float32_bytes(example)),
    }
    staging = Path(tempfile.mkdtemp(prefix=".leviathan-forge-stage-", dir=out.parent))
    report_tmp = report_id = None
    report_data, report_linked = b"", False
    try:
        for name, data in (("model.graph.json", graph), ("model.weights.f32", blob),
                           ("model.manifest.json", json.dumps(manifest).encode()), ("model_source.py", model_source)):
            with open(staging / name, "xb") as handle:
                handle.write(data)
        with tempfile.TemporaryDirectory(prefix="leviathan-forge-build-") as tmp:
            binding_path = Path(tmp) / "python_binding.json"
            binding_path.write_text(json.dumps(binding))
            args = ["forge", "--bundle", str(staging), "--python-binding", str(binding_path)]
            if diagnostic_fail_optimizations:
                args.append("--diagnostic-fail-optimizations")
            stdout, forge_seconds = run_helper(helper_path, args)
        summary = parse_json(stdout.encode(), "helper forge summary")
        forge_millis = (time.perf_counter() - started) * 1000.0
        # Actual supplied-input helper run + source parity, from the hidden stage.
        run = run_bundle(staging, model_path, real, helper=helper_path)
        result = {"bundle": str(out), "helper": str(helper_path), "forge": summary,
                  "forgeTiming": {"forgeHelperProcessMillis": forge_seconds * 1000.0,
                                  "captureAndForgeMillis": forge_millis},
                  "run": run}
        if report is not None:
            report_data = (json.dumps(dict(result, run=dict(run, bundle=str(out))), indent=2) + "\n").encode()
            fd, name = tempfile.mkstemp(prefix=f".{report.name}.partial-", dir=report.parent)
            report_tmp = Path(name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(report_data)
            report_id = os.stat(report_tmp)
            try:
                os.link(report_tmp, report)  # exclusive: never replaces an existing or foreign report
            except OSError as exc:
                raise ForgeRunError(f"report {report} could not be written exclusively ({exc}); "
                                    "nothing was overwritten") from None
            report_linked = True
        publish_exclusive(staging, out)
    except BaseException as exc:
        note = f"; own report {report} {discard_own_report(report, report_id, report_data)}" if report_linked else ""
        kept = preserve_failed_stage(staging, out, exc)
        if not isinstance(exc, Exception):
            raise
        detail = str(exc) if isinstance(exc, ForgeRunError) else f"{type(exc).__name__}: {exc}"
        raise ForgeRunError(f"build NOT published ({detail}){note}; failed stage evidence kept at {kept}") from None
    finally:
        if report_tmp is not None:
            report_tmp.unlink(missing_ok=True)  # our own private temp name only
    run["bundle"] = str(out)  # remapped only after publication
    return result


def run_bundle(bundle_dir, model_path, user_input, helper=None):
    """API: fresh-process reload of a bundle on a (different) real input."""
    started = time.perf_counter()
    bundle = Path(bundle_dir)
    x = resolve_input(user_input, "input")
    if (bundle / FAILED_MARKER).exists():
        raise ForgeRunError(f"{bundle} is a preserved FAILED build stage, not a bundle")
    helper_path = resolve_helper(helper)
    data = read_bounded(bundle / "bundle.json", MAX_JSON, "bundle.json")
    recorded = read_bounded(bundle / "bundle.sha256", 128, "bundle.sha256").decode("ascii", "replace").strip()
    if sha256_hex(data) != recorded:
        raise ForgeRunError("bundle digest mismatch: bundle.json is corrupt or was modified")
    meta = parse_json(data, "bundle.json")
    try:
        if meta["format"] != BUNDLE_FORMAT:
            raise ForgeRunError("unsupported bundle format")
        py, files, shape = dict(meta["python"]), dict(meta["files"]), dict(meta["shape"])
    except (KeyError, TypeError, ValueError):
        raise ForgeRunError("bundle.json is malformed") from None
    bound = shape.get("input")
    if not (isinstance(bound, list) and len(bound) == 2 and bound[0] == 1 and type(bound[1]) is int):
        raise ForgeRunError("bundle.json shape is malformed")
    validate_input_tensor(x, "input", bound[1], "bound static shape")
    for name, key, cap in BUNDLE_FILES:
        if sha256_hex(read_bounded(bundle / name, cap, name)) != files.get(key):
            raise ForgeRunError(f"bundle file {name} sha256 mismatch: corrupt or modified bundle")
    if sha256_hex(read_bounded(model_path, MAX_JSON, "model definition")) != py.get("modelSourceSha256"):
        raise ForgeRunError("model source mismatch: this bundle was forged from a different model definition")
    if backend_source_identity()[0] != py.get("backendSourceSha256"):
        raise ForgeRunError("backend source mismatch: installed backend/orchestrator sources differ from the forging ones; re-forge explicitly (no silent retune)")
    if py.get("torchVersion") != str(torch.__version__) or py.get("pythonVersion") != platform.python_version():
        raise ForgeRunError("torch/python version mismatch")
    seed = py.get("modelSeed")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ForgeRunError("bound seed is malformed")
    model, input_dim, model_sha, _ = load_model(model_path, seed, (("input", x),))
    if model_sha != py["modelSourceSha256"]:
        raise ForgeRunError("model source mismatch: file changed while loading")
    if tuple(x.shape) != (1, input_dim) or shape.get("input") != [1, input_dim]:
        raise ForgeRunError(f"input shape {list(x.shape)} does not match the bound static shape [1, {input_dim}]")
    manifest = parse_json(read_bounded(bundle / "model.manifest.json", MAX_JSON, "manifest"), "manifest")
    ref_input = manifest.get("refInput") if isinstance(manifest, dict) else None
    if not isinstance(ref_input, list) or len(ref_input) != input_dim:
        raise ForgeRunError("manifest refInput is malformed")
    example = torch.tensor([[to_float32(v, "refInput") for v in ref_input]], dtype=torch.float32)
    graph, blob, _, out_dim = lower(capture(model, example, input_dim)[0], input_dim)
    if sha256_hex(graph) != files["graphSha256"]:
        raise ForgeRunError("model graph mismatch: torch.export no longer produces the bound graph")
    if sha256_hex(blob) != files["weightsSha256"]:
        raise ForgeRunError("model weights mismatch: build(seed) produced parameters that differ from the bound weights")
    with tempfile.TemporaryDirectory(prefix="leviathan-forge-run-") as tmp:
        input_path, output_path = Path(tmp) / "input.f32", Path(tmp) / "output.json"
        input_path.write_bytes(float32_bytes(x))
        _, helper_seconds = run_helper(helper_path, ["run", "--bundle", str(bundle), "--input-f32", str(input_path),
                                                     "--output-json", str(output_path)])
        result = parse_json(read_bounded(output_path, 4 << 20, "helper output"), "helper output")
    output_ready = time.perf_counter()
    try:
        gpu, timing = dict(result["gpu"]), dict(result["timing"])
        gpu_out = torch.tensor([[to_float32(v, "helper output") for v in result["output"]]], dtype=torch.float32)
    except (KeyError, TypeError, ValueError):
        raise ForgeRunError("helper output is malformed") from None
    reference = forward_source(model, x, "the input")
    if tuple(gpu_out.shape) != (1, out_dim) or tuple(reference.shape) != (1, out_dim):
        raise ForgeRunError("output shape mismatch")
    max_abs = (gpu_out - reference).abs().max().item()
    if not torch.allclose(gpu_out, reference, atol=ATOL, rtol=RTOL):
        raise ForgeRunError(f"GPU output disagrees with the source PyTorch module (max abs diff {max_abs:.3e})")
    return {
        "bundle": str(bundle), "output": gpu_out.tolist(), "outputShape": [1, out_dim],
        "selected": result.get("selected"), "selectedRole": result.get("selectedRole"),
        "decision": result.get("decision"), "gpuEvidence": gpu,
        "parity": {"reference": "source PyTorch nn.Module (eager CPU float32)", "maxAbsDiff": max_abs,
                   "atol": ATOL, "rtol": RTOL, "passed": True},
        "timing": {
            "gpuCommandBufferMicros": timing.get("gpuCommandBufferMicros"),
            "helperValidationMicros": timing.get("helperValidationMicros"),
            "helperPrepareMicros": timing.get("helperPrepareMicros"),
            "helperTotalMicros": timing.get("helperTotalMicros"),
            "helperProcessMillis": helper_seconds * 1000.0,
            "endToEndMillis": (output_ready - started) * 1000.0,
            "labels": {
                "gpuCommandBufferMicros": "GPU only: GPUEndTime - GPUStartTime of the selected plan's command buffer",
                "helperProcessMillis": "fresh helper process wall time (launch, identity, validation, shader compile, upload, dispatch, readback)",
                "endToEndMillis": "end-to-end orchestrator wall time to GPU output (validation, re-capture, helper process); excludes the reference comparison",
            },
        },
    }


def write_new_file(path, text):
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        raise ForgeRunError(f"{path} already exists; refusing to overwrite") from None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="leviathan_forge.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="capture, forge, bind, run a real input")
    build.add_argument("model", help="TRUSTED local Python model file (executed)")
    build.add_argument("--example-input", required=True)
    build.add_argument("--input", required=True)
    build.add_argument("--out", required=True, help="new bundle directory (must not exist)")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--helper")
    build.add_argument("--diagnostic-fail-optimizations", action="store_true",
                       help="DIAGNOSTIC ONLY: fail every optimization candidate (never the baseline)")
    build.add_argument("--report")
    run = commands.add_parser("run", help="fresh-process reload on a new input")
    run.add_argument("bundle")
    run.add_argument("--model", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--helper")
    run.add_argument("--report")
    sample = commands.add_parser("sample-input", help="write a seeded [1, D] input JSON")
    sample.add_argument("--dim", type=int, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report_path = getattr(args, "report", None)
        if report_path and (Path(report_path).exists() or Path(report_path).is_symlink()):
            raise ForgeRunError(f"report {report_path} already exists; refusing to overwrite")
        if report_path and not Path(report_path).parent.is_dir():
            raise ForgeRunError(f"report directory for {report_path} does not exist")
        if args.command == "build":
            report = build_bundle(args.model, args.example_input, args.input, args.out, seed=args.seed,
                                  helper=args.helper, diagnostic_fail_optimizations=args.diagnostic_fail_optimizations,
                                  report_path=report_path)
            report_path = None  # written exclusively by build_bundle
        elif args.command == "run":
            report = run_bundle(args.bundle, args.model, args.input, helper=args.helper)
        else:
            if not 0 < args.dim <= MAX_DIM or not 0 <= args.seed <= MAX_SEED:
                raise ForgeRunError("dim/seed out of range")
            values = (torch.rand((1, args.dim), generator=torch.Generator().manual_seed(args.seed)) * 2 - 1).tolist()
            write_new_file(args.out, json.dumps(values))
            report = {"wrote": args.out, "shape": [1, args.dim]}
        text = json.dumps(report, indent=2)
        if report_path:
            write_new_file(report_path, text + "\n")
        print(text)
        return 0
    except ForgeRunError as exc:
        print(f"leviathan_forge: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
