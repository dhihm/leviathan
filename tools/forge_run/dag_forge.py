#!/usr/bin/env python3
"""Leviathan Forge DAG runtime (generic static value-DAG, FP32 CPU batch 1).

build: TRUSTED model.py + example JSON + real input JSON -> real
 torch.export(strict=True) static CPU FP32 [1,D] capture, lowered by the
 attached dag_frontend.validate_and_lower_dag to a leviathan-value-dag-1
 graph dict, the ACTUAL exported parameter bytes and per-dense layouts ->
 leviathan-dag-helper forge (same contract as the old helper, new bundle
 format leviathan-dag-forge-bundle-1 and manifest format
 leviathan-dag-weights-1 with a layouts key) -> fresh helper run of the
 actual input from the hidden stage -> eager source parity -> atomic
 exclusive publication. run: fresh-process reload, exact re-capture match
 of graph/weights/layouts/aten ops BEFORE the GPU, then helper run and
 source parity. The model file is executed as ordinary local Python with
 your privileges (no sandbox). Bundles, inputs and helper output are
 untrusted data. Model names never branch runtime behavior; there is no
 seeded/fake CPU fallback. Not an LLM runtime. No agent runtime.
Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
DAG_FRONTEND_PATH = REPO_ROOT / "tools" / "forge_run" / "dag_frontend.py"
HELPER_NAME = "leviathan-dag-helper"
BUNDLE_FORMAT = "leviathan-dag-forge-bundle-1"
MANIFEST_FORMAT = "leviathan-dag-weights-1"
MAX_JSON = 1 << 20
MAX_SEED = 2 ** 31 - 1
ATOL = 1e-4
RTOL = 1e-4
# Binding digest covers ALL backend sources: Swift core + Package + old and
# new helpers, MVP and DAG Python, and both frontends (portable relative paths
# + file bytes).
BACKEND_SOURCE_PATTERNS = (
    "Package.swift",
    "Sources/LeviathanCore/*.swift",
    "Sources/leviathan-forge-helper/*.swift",
    "Sources/leviathan-dag-helper/*.swift",
    "tools/torch_frontend/export_torch_graph.py",
    "tools/forge_run/dag_frontend.py",
    "tools/forge_run/leviathan_forge.py",
    "tools/forge_run/dag_forge.py",
    "tools/forge_run/dag_runtime.py",
)
BUNDLE_FILES = (
    ("model.graph.json", "graphSha256", MAX_JSON),
    ("model.weights.f32", "weightsSha256", 16 << 20),
    ("model.manifest.json", "manifestSha256", MAX_JSON),
    ("model_source.py", "modelSourceCopySha256", MAX_JSON),
)


def _load_common():
    """Normal import of the sibling MVP module for its safe common functions
    (path setup only; nothing is monkeypatched or rerouted)."""
    spec = importlib.util.spec_from_file_location("leviathan_forge_mvp", TOOL_DIR / "leviathan_forge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lf = _load_common()
for _name in ("ForgeRunError", "sha256_hex", "read_bounded", "parse_json", "to_float32",
              "validate_input_tensor", "resolve_input", "forward_source", "float32_bytes",
              "run_helper", "publish_exclusive", "preserve_failed_stage", "discard_own_report",
              "load_model", "frontend", "MAX_SEED", "MAX_JSON", "FAILED_MARKER"):
    globals()[_name] = getattr(lf, _name)


def resolve_helper(helper):
    """Own explicit resolution (the common one would pick the OLD helper by
    default): user path, else .build/release, then .build/debug, else fail."""
    if helper:
        path = Path(helper)
        if not path.is_file():
            raise ForgeRunError(f"helper {helper} not found")
        return path
    for rel in (".build/release/leviathan-dag-helper", ".build/debug/leviathan-dag-helper"):
        path = REPO_ROOT / rel
        if path.is_file():
            return path
    raise ForgeRunError("leviathan-dag-helper not found; pass --helper explicitly")

_DAG = None


def dag_frontend():
    global _DAG
    if _DAG is None:
        spec = importlib.util.spec_from_file_location("leviathan_dag_frontend", DAG_FRONTEND_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DAG = module
    return _DAG


def backend_source_identity(root=REPO_ROOT):
    """Relative paths + content digest over every backend source file."""
    import hashlib
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


def graph_bytes(graph):
    """Exact canonical JSON: identical bytes on the capture and run paths."""
    return json.dumps(graph, separators=(",", ":"), sort_keys=True).encode()


def capture(model, example, input_dim):
    """Real strict torch.export + dag_frontend lowering + exact source-vs-EP
    differential gate (the example plus 8 seeded probes)."""
    try:
        program = torch.export.export(model, (example,), strict=True)
    except Exception as exc:
        raise ForgeRunError(f"torch.export failed (unsupported model): {type(exc).__name__}: {str(exc)[:2000]}") from None
    try:
        graph, blob, layouts, aten_ops = dag_frontend().validate_and_lower_dag(program, input_dim)
    except ValueError as exc:  # DagFrontendError is a ValueError
        raise ForgeRunError(str(exc)) from None
    exported = program.module()
    fe = frontend()
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
    return graph, blob, layouts, aten_ops


def graph_output_dim(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    return int(nodes[graph["outputValue"]]["outputDim"])


def expected_kernel_sequence(graph, candidate):
    """Kernels the helper must have dispatched. Consumer counts cover ALL
    edges PLUS the graph output. A dense node is fused with the immediately
    following relu (matching edge, single consumer, not the graph output)
    ONLY when the selected plan has fuseDenseRelu; the fused dense emits
    leviathan_dense_relu and its relu is skipped. Host-derived dispatch
    expectation, not a profiler trace."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    consumers = {}
    for n in graph["nodes"]:
        for i in n["inputs"]:
            consumers[i] = consumers.get(i, 0) + 1
    consumers[graph["outputValue"]] = consumers.get(graph["outputValue"], 0) + 1
    kernel = {"dense": "leviathan_dense", "relu": "leviathan_dag_relu", "add": "leviathan_dag_add"}
    fused = set()
    if isinstance(candidate, dict) and candidate.get("fuseDenseRelu"):
        for n in graph["nodes"]:
            r = nodes.get(n["id"] + 1)
            if (n["op"] == "dense" and r is not None and r["op"] == "relu" and r["inputs"] == [n["id"]]
                    and consumers.get(n["id"], 0) == 1 and n["id"] != graph["outputValue"]):
                fused.add(n["id"])
    sequence = []
    for n in graph["nodes"]:
        if n["op"] == "relu" and n["id"] - 1 in fused:
            continue
        sequence.append("leviathan_dense_relu" if n["id"] in fused else kernel[n["op"]])
    return sequence


def check_kernel_evidence(graph, gpu, meta):
    """The GPU dict is mandatory host-derived dispatch evidence and must match
    the bundle's persisted selection exactly; missing evidence is never
    accepted and no fallback schema is guessed."""
    if not isinstance(gpu, dict):
        raise ForgeRunError("helper output is missing the mandatory GPU evidence dict")
    try:
        selection = meta["selection"]
        selected, hardware = selection["selected"], meta["hardware"]
    except (KeyError, TypeError):
        raise ForgeRunError("bundle.json selection/hardware is malformed") from None
    expected = expected_kernel_sequence(graph, selected)
    if gpu.get("deviceName") != hardware.get("deviceName"):
        raise ForgeRunError("GPU evidence deviceName does not match the bound hardware")
    if gpu.get("commandStatus") != "completed":
        raise ForgeRunError("GPU command buffer did not complete")
    if gpu.get("kernelSequence") != expected:
        raise ForgeRunError(f"reported kernel sequence {gpu.get('kernelSequence')} does not match the expected DAG kernels {expected}")
    if gpu.get("computeDispatches") != len(expected):
        raise ForgeRunError(f"reported computeDispatches {gpu.get('computeDispatches')} does not match the expected {len(expected)}")
    if gpu.get("threadgroupWidth") != selected.get("threadgroupWidth"):
        raise ForgeRunError("GPU threadgroupWidth does not match the selected plan")
    if gpu.get("fusedDenseRelu") != selected.get("fuseDenseRelu"):
        raise ForgeRunError("GPU fusedDenseRelu does not match the selected plan")
    micros = gpu.get("gpuCommandBufferMicros")
    if not (isinstance(micros, (int, float)) and not isinstance(micros, bool)
            and micros == micros and micros != float("inf") and micros > 0.0):
        raise ForgeRunError("helper did not report a positive finite GPU time (no actual GPU evidence)")
    return micros


def _check_run_evidence(graph, result, timing, meta):
    """Full run-report contract: format, exact persisted selection, bound
    output shape, mandatory GPU dict and finite non-negative timings."""
    try:
        if result["format"] != "leviathan-dag-run-1":
            raise ForgeRunError("helper run report format mismatch")
        selection = meta["selection"]
        if result["selected"] != selection["selected"]:
            raise ForgeRunError("run selected plan does not exactly match the persisted bundle selection")
        if result["selectedRole"] != selection["selectedRole"] or result["decision"] != selection["decision"]:
            raise ForgeRunError("run selectedRole/decision does not match the persisted bundle selection")
        if result["outputShape"] != [1, graph_output_dim(graph)]:
            raise ForgeRunError("run outputShape is not the bound output shape")
    except ForgeRunError:
        raise
    except (KeyError, TypeError, ValueError):
        raise ForgeRunError("helper output is malformed") from None
    micros = check_kernel_evidence(graph, result.get("gpu"), meta)
    if timing.get("gpuCommandBufferMicros") != micros:
        raise ForgeRunError("timing gpuCommandBufferMicros does not match the GPU evidence value")
    for key in ("gpuCommandBufferMicros", "helperValidationMicros", "helperPrepareMicros", "helperTotalMicros"):
        v = timing.get(key)
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v != float("inf") and v >= 0.0):
            raise ForgeRunError(f"timing field {key} is not a finite non-negative number")


def build_bundle(model_path, example_input, user_input, out_dir, seed=0, helper=None,
                 diagnostic_fail_optimizations=False, report_path=None):
    """API: capture + forge into a hidden stage, run the ACTUAL supplied input
    on the GPU from that stage, check eager source parity, then publish the
    stage atomically/exclusively as out_dir. On failure nothing final is
    created; the stage is preserved as a marked FAILED sibling and existing
    foreign outputs are untouched. Optional report uses the exact exclusive
    pre-bundle link + rollback pattern from the MVP; a crash between report
    link and bundle rename can leave a report without a bundle (no joint
    two-artifact transaction claim). No final artifact before parity."""
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
    graph, blob, layouts, aten_ops = capture(model, example, input_dim)
    gbytes = graph_bytes(graph)
    reference = forward_source(model, example, "the example input")
    manifest = {
        "format": MANIFEST_FORMAT, "inputDim": input_dim, "floatCount": len(blob) // 4,
        "weightsSha256": sha256_hex(blob), "layouts": layouts,
        "refInput": example.flatten().tolist(), "refOutput": reference.flatten().tolist(),
        "modelSeed": seed,
    }
    binding = {
        "modelSourceSha256": model_sha, "modelSourceName": Path(model_path).name[:256], "modelSeed": seed,
        "backendSourceSha256": source_sha, "backendSourceFiles": source_count,
        "torchVersion": str(torch.__version__), "pythonVersion": platform.python_version(),
        "atenOps": aten_ops, "exampleInputSha256": sha256_hex(float32_bytes(example)),
    }
    staging = Path(tempfile.mkdtemp(prefix=".leviathan-dag-stage-", dir=out.parent))
    report_tmp = report_id = None
    report_data, report_linked = b"", False
    try:
        for name, data in (("model.graph.json", gbytes), ("model.weights.f32", blob),
                           ("model.manifest.json", json.dumps(manifest).encode()), ("model_source.py", model_source)):
            with open(staging / name, "xb") as handle:
                handle.write(data)
        with tempfile.TemporaryDirectory(prefix="leviathan-dag-build-") as tmp:
            binding_path = Path(tmp) / "python_binding.json"
            binding_path.write_text(json.dumps(binding))
            args = ["forge", "--bundle", str(staging), "--python-binding", str(binding_path)]
            if diagnostic_fail_optimizations:
                args.append("--diagnostic-fail-optimizations")
            stdout, forge_seconds = run_helper(helper_path, args)
        summary = parse_json(stdout.encode(), "helper forge summary")
        forge_millis = (time.perf_counter() - started) * 1000.0
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
                os.link(report_tmp, report)  # exclusive: never replaces anything
            except OSError as exc:
                raise ForgeRunError(f"report {report} could not be written exclusively ({exc}); "
                                    "nothing was overwritten") from None
            report_linked = True
        publish_exclusive(staging, out)  # macOS renamex_np RENAME_EXCL
    except BaseException as exc:
        note = f"; own report {report} {discard_own_report(report, report_id, report_data)}" if report_linked else ""
        kept = preserve_failed_stage(staging, out, exc)
        if not isinstance(exc, Exception):
            raise
        detail = str(exc) if isinstance(exc, ForgeRunError) else f"{type(exc).__name__}: {exc}"
        raise ForgeRunError(f"build NOT published ({detail}){note}; failed stage evidence kept at {kept}") from None
    finally:
        if report_tmp is not None:
            report_tmp.unlink(missing_ok=True)
    run["bundle"] = str(out)
    return result


def load_bound_model(bundle_dir, model_path, helper=None, user_input=None):
    """Explicit runtime initialization API: constructor-style load-once of a
    bound bundle with NO GPU execution. When user_input is provided it is
    resolved and validated early (CPU, dense strided torch.float32 [1, D],
    finite) and checked against the bound static shape BEFORE any model
    construction; the model is then constructed once from the exact hashed
    source with that input in its validation tuple. Every run-path binding
    gate still applies, in the same order and with no weakening: FAILED
    marker, exact bundle digest/format, per-file SHA256, model source SHA,
    backend source SHA (explicit re-forge only, never a silent retune),
    torch/python versions, bound seed, static shape, manifest format and
    refInput snapshot, and the real torch.export re-capture whose graph
    bytes, actual exported weight bytes, per-dense layouts and ATen op
    sequence must match the bound ones exactly. No cached global state and
    no model-name routing. Only the bound helper identity fields are parsed
    from bundle.json here; the actual Metal device/binary are checked by the
    helper later."""
    started = time.perf_counter()
    bundle = Path(bundle_dir)
    x = resolve_input(user_input, "input") if user_input is not None else None
    if (bundle / FAILED_MARKER).exists():
        raise ForgeRunError(f"{bundle} is a preserved FAILED build stage, not a bundle")
    helper_path = resolve_helper(helper)
    data = read_bounded(bundle / "bundle.json", MAX_JSON, "bundle.json")
    recorded = read_bounded(bundle / "bundle.sha256", 128, "bundle.sha256").decode("ascii", "replace").strip()
    bundle_sha = sha256_hex(data)
    if bundle_sha != recorded:
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
    if x is not None:
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
    model_inputs = (("input", x),) if x is not None else ()
    model, input_dim, model_sha, _ = load_model(model_path, seed, model_inputs)
    if model_sha != py["modelSourceSha256"]:
        raise ForgeRunError("model source mismatch: file changed while loading")
    if shape.get("input") != [1, input_dim] or (x is not None and tuple(x.shape) != (1, input_dim)):
        got = list(x.shape) if x is not None else shape.get("input")
        raise ForgeRunError(f"input shape {got} does not match the bound static shape [1, {input_dim}]")
    manifest = parse_json(read_bounded(bundle / "model.manifest.json", MAX_JSON, "manifest"), "manifest")
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ForgeRunError("manifest format mismatch")
    ref_input = manifest.get("refInput")
    layouts = manifest.get("layouts")
    if not isinstance(ref_input, list) or len(ref_input) != input_dim or not isinstance(layouts, list):
        raise ForgeRunError("manifest refInput/layouts is malformed")
    example = torch.tensor([[to_float32(v, "refInput") for v in ref_input]], dtype=torch.float32)
    graph, blob, layouts2, aten_ops = capture(model, example, input_dim)
    if sha256_hex(graph_bytes(graph)) != files["graphSha256"]:
        raise ForgeRunError("model graph mismatch: torch.export no longer produces the bound graph")
    if sha256_hex(blob) != files["weightsSha256"]:
        raise ForgeRunError("model weights mismatch: build(seed) produced parameters that differ from the bound weights")
    if json.dumps(layouts, separators=(",", ":"), sort_keys=True) != json.dumps(layouts2, separators=(",", ":"), sort_keys=True):
        raise ForgeRunError("weight layouts mismatch: exported parameter layout differs from the bound layouts")
    if json.dumps(aten_ops, separators=(",", ":")) != json.dumps(py.get("atenOps"), separators=(",", ":")):
        raise ForgeRunError("aten ops mismatch: the exported ATen sequence differs from the bound one")
    return {
        "bundle": bundle,
        "helper": helper_path,
        "model": model,
        "graph": graph,
        "meta": meta,
        "inputDim": input_dim,
        "outputDim": graph_output_dim(graph),
        "bundleSha256": bundle_sha,
        "validatedLoadMillis": (time.perf_counter() - started) * 1000.0,
        "input": x,
    }


def run_bundle(bundle_dir, model_path, user_input, helper=None):
    """API: fresh-process reload of a bundle on a (different) real input.
    load_bound_model performs every Python-side binding check BEFORE any GPU
    work (FAILED marker, exact bundle digest/format, file SHAs, model source
    SHA, backend SHA, torch/python versions, bound seed and input shape, plus
    the exact re-capture of graph bytes, weights, layouts and aten ops from
    the manifest refInput snapshot); run_bundle then runs the fresh helper on
    the actual input and checks eager source parity before any final use."""
    started = time.perf_counter()
    loaded = load_bound_model(bundle_dir, model_path, helper=helper, user_input=user_input)
    bundle, x = loaded["bundle"], loaded["input"]
    helper_path, model = loaded["helper"], loaded["model"]
    graph, meta = loaded["graph"], loaded["meta"]
    capture_done = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="leviathan-dag-run-") as tmp:
        input_path, output_path = Path(tmp) / "input.f32", Path(tmp) / "output.json"
        input_path.write_bytes(float32_bytes(x))  # serialized float32
        _, helper_seconds = run_helper(helper_path, ["run", "--bundle", str(bundle), "--input-f32", str(input_path),
                                                     "--output-json", str(output_path)])
        result = parse_json(read_bounded(output_path, 4 << 20, "helper output"), "helper output")
    output_ready = time.perf_counter()
    try:
        gpu, timing = dict(result["gpu"]), dict(result["timing"])
        gpu_out = torch.tensor([[to_float32(v, "helper output") for v in result["output"]]], dtype=torch.float32)
    except (KeyError, TypeError, ValueError):
        raise ForgeRunError("helper output is malformed") from None
    _check_run_evidence(graph, result, timing, meta)
    reference = forward_source(model, x, "the input")
    out_dim = reference.shape[1]
    if tuple(gpu_out.shape) != (1, out_dim):
        raise ForgeRunError("output shape mismatch")
    if not bool(torch.isfinite(gpu_out).all()):
        raise ForgeRunError("GPU output is non-finite")
    max_abs = (gpu_out - reference).abs().max().item()
    if not torch.allclose(gpu_out, reference, atol=ATOL, rtol=RTOL):
        raise ForgeRunError(f"GPU output disagrees with the source PyTorch module (max abs diff {max_abs:.3e})")
    end = time.perf_counter()
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
            "pythonCaptureMillis": (capture_done - started) * 1000.0,
            "outputReadyMillis": (output_ready - started) * 1000.0,
            "endToEndMillis": (end - started) * 1000.0,
            "labels": {
                "gpuCommandBufferMicros": "GPU only: command buffer end - start of the selected plan (actual GPU completion)",
                "helperProcessMillis": "fresh helper process wall time (launch, identity, validation, shader compile, upload, dispatch, readback)",
                "pythonCaptureMillis": "CPU-side Python validation + torch.export re-capture overhead",
                "outputReadyMillis": "wall time to the helper GPU output being read back",
                "endToEndMillis": "full runtime including the CPU parity check; no low-latency claim",
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
    parser = argparse.ArgumentParser(prog="dag_forge.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="capture, forge, bind, run a real input")
    build.add_argument("--model", required=True, help="TRUSTED local Python model file (executed)")
    build.add_argument("--example-input", required=True)
    build.add_argument("--input", required=True)
    build.add_argument("--out", required=True, help="new bundle directory (must not exist)")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--helper")
    build.add_argument("--diagnostic-fail-optimizations", action="store_true",
                       help="DIAGNOSTIC ONLY: fail every optimization candidate (never the baseline)")
    build.add_argument("--report")
    run = commands.add_parser("run", help="fresh-process reload on a new input")
    run.add_argument("--bundle", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--helper")
    run.add_argument("--report")
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
            report_path = None
        else:
            report = run_bundle(args.bundle, args.model, args.input, helper=args.helper)
        text = json.dumps(report, indent=2)
        if report_path:
            write_new_file(report_path, text + "\n")
        print(text)
        return 0
    except ForgeRunError as exc:
        print(f"dag_forge: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
