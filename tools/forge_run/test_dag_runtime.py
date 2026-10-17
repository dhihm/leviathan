#!/usr/bin/env python3
"""Acceptance tests for dag_runtime.LoadedRuntime.

Real Torch CPU references plus the real Metal helper worker: no GPU mocks,
no skips, no CPU fallbacks. Only the wire-fault tests monkeypatch a
client-side instance method, and only after a verified real GPU run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest.mock as mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from torch import nn

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import dag_forge as f
import dag_runtime

EX = TOOL_DIR / "examples"
REPO_ROOT = TOOL_DIR.parent.parent
HELPER_NAME = "leviathan-dag-helper"
READY_KEYS = {"format", "bundleSha256", "helper", "hardware", "shape",
              "selected", "selectedRole", "decision", "startupTiming"}
VAL32 = b",".join([b"0.1"] * 32)

ALIAS_SRC = '''#!/usr/bin/env python3
import torch
from torch import nn
INPUT_DIM = 4
class AliasGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
    def forward(self, x):
        h = self.lin(x)
        return h + torch.relu(h)
def build(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = AliasGate()
        with torch.no_grad():
            m.lin.weight.copy_(torch.eye(4))
            m.lin.bias.copy_(torch.tensor([-1.0, 2.0, -3.0, 4.0]))
        return m.eval()
'''


def _import_example(name, path=None):
    path = Path(path) if path else EX / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linspace(dim, lo=-1.0, hi=1.0):
    return torch.linspace(lo, hi, dim, dtype=torch.float32).reshape(1, dim)


# One explicit helper binding for forge, runtime and pipe servers: the debug
# build that matches the forged artifact (auto-resolve would prefer release
# when both exist and bind the runtime to a different binary).
DEBUG_HELPER = REPO_ROOT / ".build" / "debug" / HELPER_NAME
RELEASE_HELPER = REPO_ROOT / ".build" / "release" / HELPER_NAME
HELPER = DEBUG_HELPER if DEBUG_HELPER.is_file() else RELEASE_HELPER
assert HELPER.is_file(), f"{HELPER_NAME} binary not found under {REPO_ROOT}/.build"


def _forge(source, seed, out_dir, example=None, real=None):
    """Exact documented API call: no signature introspection, no fake routing."""
    mod = _import_example(Path(source).stem, source)
    dim = int(mod.INPUT_DIM)
    example = _linspace(dim) if example is None else example
    real = _linspace(dim, -0.5, 0.5) if real is None else real
    return f.build_bundle(model_path=str(source), example_input=example,
                          user_input=real, out_dir=str(out_dir), seed=seed,
                          helper=str(HELPER), diagnostic_fail_optimizations=False,
                          report_path=None)


def _helper_path(bundle, built=None):
    # Prefer the exact helper the forge stage actually used.
    if isinstance(built, dict):
        v = built.get("helper")
        if v and Path(v).is_file():
            return Path(v)
    assert HELPER.is_file()
    return HELPER


def _digest(bundle):
    h = hashlib.sha256()
    for p in sorted(Path(bundle).rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _inputs(dim, n=12, seed=777):
    g = torch.Generator().manual_seed(seed)
    xs = []
    for i in range(n):
        x = torch.randn(1, dim, generator=g)
        if i % 3 == 0:
            x = -x.abs()
        elif i % 3 == 1:
            x = x.abs() + 0.25
        xs.append(x)
    return xs


def _dead(pid):
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _ref_out(ref, x):
    with torch.no_grad():
        return ref(x.clone())


def _runtime(bundle, src):
    return dag_runtime.LoadedRuntime(bundle, src, helper=str(HELPER))


@pytest.fixture(scope="session")
def work(tmp_path_factory):
    if os.environ.get("LEVIATHAN_RUNTIME_KEEP_TEST_OUTPUT"):
        d = Path(tempfile.mkdtemp(prefix="dag-runtime-tests-"))
        print(f"LEVIATHAN test artifacts kept at: {d}", file=sys.stderr)
        return d
    return tmp_path_factory.mktemp("dag-runtime")


@pytest.fixture(scope="session")
def residual(work):
    mod = _import_example("residual_block")
    src = work / "residual_block.py"
    shutil.copy(EX / "residual_block.py", src)
    bundle = work / "bundle_residual"
    built = _forge(src, 1234, bundle)  # seed matches the reference build below
    return mod, src, bundle, built, mod.build(1234).eval()


@pytest.fixture(scope="session")
def projected(work):
    mod = _import_example("projected_branches")
    src = work / "projected_branches.py"
    shutil.copy(EX / "projected_branches.py", src)
    bundle = work / "bundle_projected"
    built = _forge(src, 4321, bundle)
    return mod, src, bundle, built, mod.build(4321).eval()


@pytest.fixture(scope="session")
def alias(work):
    src = work / "alias_gate.py"
    src.write_text(ALIAS_SRC)
    mod = _import_example("alias_gate", src)
    bundle = work / "bundle_alias"
    built = _forge(src, 7, bundle)
    return mod, src, bundle, built, mod.build(7).eval()


def test_two_source_examples_twelve_inputs_each(residual, projected):
    ma, sa, ba, _, refa = residual
    mb, sb, bb, _, refb = projected
    da, db = _digest(ba), _digest(bb)
    for mod_src_bundle_ref, dim, out_dim in ((residual, 32, 32), (projected, 24, 48)):
        _, src, bundle, _, ref = mod_src_bundle_ref
        outs = []
        with _runtime(bundle, src) as rt:
            pid = rt.helperPID
            assert pid and pid != os.getpid()
            assert rt.startupTimings["helperStartupTiming"]["totalMillis"] >= 0
            for x in _inputs(dim):
                r = rt.run(x)
                assert rt.helperPID == pid
                assert r["outputShape"] == [1, out_dim]
                assert r["parity"]["passed"]
                got = torch.tensor(r["output"])
                assert torch.allclose(got, _ref_out(ref, x), atol=1e-4, rtol=1e-4)
                outs.append(float(got[0, 0]))
        assert len(set(outs)) > 1  # different seeded inputs give different outputs
    assert _digest(ba) == da and _digest(bb) == db  # bundle bytes unchanged, no silent retune


def test_load_model_and_capture_counted_once(residual):
    _, src, bundle, _, ref = residual
    with mock.patch.object(f, "load_model", wraps=f.load_model) as lm, \
         mock.patch.object(f, "capture", wraps=f.capture) as cap:
        with _runtime(bundle, src) as rt:
            assert lm.call_count == 1
            assert cap.call_count == 1  # exactly one export required at load
            base_lm, base_cap = lm.call_count, cap.call_count
            for x in _inputs(32, n=4, seed=21):
                assert rt.run(x)["parity"]["passed"]
            assert lm.call_count == base_lm
            assert cap.call_count == base_cap  # zero additional across requests


def test_alias_canary_negative_gate_exact(alias):
    _, src, bundle, built, ref = alias
    meta = json.loads((Path(bundle) / "bundle.json").read_text())
    sel = meta["selection"]["selected"]
    assert sel["fuseDenseRelu"] is False  # lin output has two consumers: never fusable
    with _runtime(bundle, src) as rt:
        for x in _inputs(4, n=12, seed=9):
            r = rt.run(x)
            with torch.no_grad():
                h = ref.lin(x.clone())  # pre-activation; ref(x) is already h + relu(h)
                want = ref(x.clone())
            neg = h[0] < 0
            assert neg.any()
            out = torch.tensor(r["output"])
            assert torch.equal(out[0][neg], h[0][neg])  # exact negatives preserved (relu passthrough)
            assert torch.allclose(out, want, atol=1e-5, rtol=1e-5)  # full reference parity
            gpu = r["gpuEvidence"]
            assert gpu["fusedDenseRelu"] is False
            assert "leviathan_dense_relu" not in gpu["kernelSequence"]
            assert gpu["computeDispatches"] == len(gpu["kernelSequence"])


def test_invalid_client_tensors_rejected_before_ipc(residual):
    _, src, bundle, _, ref = residual
    with _runtime(bundle, src) as rt:
        pid = rt.helperPID
        bad = [
            torch.randn(1, 32, dtype=torch.float64),
            torch.zeros(1, 32, device="meta"),
            torch.sparse_coo_tensor([[0], [0]], [1.0], [1, 32]),
            torch.full((1, 32), float("nan")),
            torch.full((1, 32), float("inf")),
            torch.randn(1, 24),
            torch.randn(32),
        ]
        for t in bad:
            with pytest.raises(f.ForgeRunError):
                rt.run(t)
            assert rt.helperPID == pid  # rejected before IPC; worker untouched
        x = torch.randn(1, 32, generator=torch.Generator().manual_seed(4))
        before = x.clone()
        r = rt.run(x.requires_grad_(True))
        assert torch.equal(x.detach(), before)  # input cloned, never modified
        assert r["parity"]["passed"]


def test_thread_pool_requests_serialized_no_mixups(residual):
    _, src, bundle, _, ref = residual
    xs = _inputs(32, n=4, seed=5)
    wants = [_ref_out(ref, x) for x in xs]
    with _runtime(bundle, src) as rt:
        pid = rt.helperPID
        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(lambda x: rt.run(x)["output"], xs))
        assert rt.helperPID == pid  # same worker, no restart
        for got, want, x in zip(outs, wants, xs):
            assert torch.allclose(torch.tensor(got), want, atol=1e-4, rtol=1e-4)


def test_pinned_snapshot_immune_to_disk_mutation(work):
    mod = _import_example("residual_block")
    src = work / "residual_mut.py"
    shutil.copy(EX / "residual_block.py", src)
    bundle = work / "bundle_mut"
    _forge(src, 1234, bundle)
    ref = mod.build(1234).eval()
    snap = work / "bundle_mut_snap"
    shutil.copytree(bundle, snap)
    x = _inputs(32, n=1, seed=8)[0]
    want = _ref_out(ref, x)
    with _runtime(snap, src) as rt:
        files = [p for p in sorted(snap.rglob("*")) if p.is_file() and p.name != "bundle.json"]
        target = max(files, key=lambda p: p.stat().st_size)
        original = target.read_bytes()
        data = bytearray(original)
        data[len(data) // 2] ^= 0xFF
        target.write_bytes(bytes(data))
        r = rt.run(x)  # live session pinned to loaded snapshot, still valid
        assert torch.allclose(torch.tensor(r["output"]), want, atol=1e-4, rtol=1e-4)
        with pytest.raises(f.ForgeRunError):
            with _runtime(snap, src):
                pass  # fresh session rejects the corrupted checksum
        target.write_bytes(original)  # restore weights so the next failure isolates the source SHA
        src.write_text(src.read_text() + "\n# mutated source\n")
        with pytest.raises(f.ForgeRunError):
            with _runtime(snap, src):
                pass  # fresh session rejects the mutated model source (weights valid again)
        r2 = rt.run(x)  # cached old snapshot still valid
        assert torch.allclose(torch.tensor(r2["output"]), want, atol=1e-4, rtol=1e-4)


def test_close_idempotent_worker_exits(residual):
    _, src, bundle, _, _ = residual
    with _runtime(bundle, src) as rt:
        pid = rt.helperPID
        assert rt.run(_inputs(32, n=1)[0])["parity"]["passed"]
    rt.close()
    rt.close()
    assert rt.helperPID is None or _dead(pid)
    with pytest.raises(f.ForgeRunError):
        rt.run(torch.randn(1, 32))


def test_real_gpu_then_wire_corruption_poisons_session(residual):
    _, src, bundle, _, _ = residual
    x = _inputs(32, n=1, seed=3)[0]
    with _runtime(bundle, src) as rt:
        r = rt.run(x)  # verified REAL GPU run before any fault injection
        assert r["parity"]["passed"]
        assert isinstance(r["gpuEvidence"], dict) and r["gpuEvidence"]
        micros = r["timing"]["gpuCommandBufferMicros"]
        assert isinstance(micros, (int, float)) and not isinstance(micros, bool) and micros > 0
        pid = rt.helperPID
        orig = rt._recv_line

        def corrupt(deadline):
            line = orig(deadline)
            obj = json.loads(line)
            obj["report"]["output"][0] = obj["report"]["output"][0] + 1.0
            return json.dumps(obj, separators=(",", ":")).encode()

        rt._recv_line = corrupt  # client-side instance patch only; GPU path untouched
        with pytest.raises(f.ForgeRunError) as ei:
            rt.run(x)
        assert "source" in str(ei.value).lower()  # parity rejection, not a wire parse error
        with pytest.raises(f.ForgeRunError):
            rt.run(x)  # no silent restart
        assert _dead(pid) or rt.helperPID is None


def test_constructor_rejects_semantic_mismatches(work):
    src_a = work / "mm_a.py"
    src_b = work / "mm_b.py"
    shutil.copy(EX / "residual_block.py", src_a)
    shutil.copy(EX / "projected_branches.py", src_b)
    bundle = work / "bundle_mismatch"
    _forge(src_a, 11, bundle)
    with pytest.raises(f.ForgeRunError):
        with _runtime(bundle, src_b):
            pass  # source/binding mismatch must not yield a usable Session
    failed = work / "bundle_failed"
    shutil.copytree(bundle, failed)
    marker = getattr(f, "FAILED_MARKER", "FORGE_FAILED.json")
    (failed / marker).write_text("forced failure marker")
    with pytest.raises(f.ForgeRunError):
        with _runtime(failed, src_a):
            pass


def _serve(helper, bundle, payload, timeout=60):
    return subprocess.run([str(helper), "serve", "--bundle", str(bundle)],
                          input=payload, capture_output=True, timeout=timeout)


def test_pipe_batched_valid_requests_and_identity(residual):
    _, src, bundle, built, _ = residual
    helper = _helper_path(bundle, built)
    vals_list = [
        [0.1] * 32,
        [(-1.5 if i % 2 else 2.5) for i in range(32)],
        [float(i) + 0.1 for i in range(32)],
        [0.0] * 32,
    ]
    ids = [1, 2, 3, 4]  # JSON integers: the helper rejects CFNumber float ids like 4.0
    payload = ("\n".join(json.dumps({"id": i, "input": v}, separators=(",", ":"))
                          for i, v in zip(ids, vals_list)) + "\n").encode()
    proc = _serve(helper, bundle, payload)
    assert proc.returncode == 0, proc.stderr.decode()
    lines = [l for l in proc.stdout.decode().splitlines() if l.strip()]
    assert len(lines) == 5  # ready + 4 replies, pending requests retained past EOF-ready chunk
    ready = json.loads(lines[0])
    assert set(ready) == READY_KEYS  # exact ready schema, no extra/missing keys
    meta = json.loads((Path(bundle) / "bundle.json").read_text())
    sel = meta.get("selection") or {}
    recorded = (Path(bundle) / "bundle.sha256").read_text().strip()
    assert ready["format"] == "leviathan-dag-session-1"
    assert ready["bundleSha256"] == recorded
    assert ready["helper"] == meta["helper"]
    assert ready["hardware"] == meta["hardware"]
    assert ready["shape"] == meta.get("shape")
    assert ready["selected"] == sel.get("selected")
    assert ready["selectedRole"] == sel.get("selectedRole")
    st = ready["startupTiming"]
    for k in ("validationMillis", "prepareMillis", "totalMillis"):
        assert isinstance(st[k], (int, float)) and st[k] >= 0
    replies = [json.loads(l) for l in lines[1:]]
    assert {r["id"] for r in replies} == {1, 2, 3, 4}
    for rid, vals in zip(ids, vals_list):
        rep = next(r for r in replies if r["id"] == rid)
        assert type(rep["id"]) is int
        report = rep["report"]
        gpu = report["gpu"]
        assert gpu["commandStatus"] == "completed"
        assert gpu["computeDispatches"] == len(gpu["kernelSequence"]) > 0
        micros = gpu["gpuCommandBufferMicros"]
        assert isinstance(micros, (int, float)) and not isinstance(micros, bool) and micros > 0
        for k in ("gpuCommandBufferMicros", "helperValidationMicros", "helperPrepareMicros", "helperTotalMicros"):
            v = report["timing"][k]
            assert isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
        got = torch.tensor([report["output"]])
        want = torch.tensor([vals]).float()  # server-side f32 rounding of 0.1 matched here
        assert got.shape == (1, 32)
        assert torch.allclose(got, _ref_out(residual[4], torch.tensor([vals]).float()), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("line", [
    b'{"id":true,"input":[' + VAL32 + b']}',
    b'{"id":"1","input":[' + VAL32 + b']}',
    b'{"id":1,"input":[NaN,' + VAL32[4:] + b']}',
    b'{"id":1,"input":[Infinity,' + VAL32[4:] + b']}',
    b'{"id":1,"input":[1e999,' + VAL32[4:] + b']}',
    b'{"id":1,"input":[true,' + VAL32[4:] + b']}',
    b'{"id":1,"input":[' + b",".join([b"0.1"] * 33) + b']}',
    b'{"id":1,"input":[' + VAL32 + b'],"extra":1}',
    b'{"id":1}',
    b'{"id":1,"input":[',
    b'{"id":1,"input":[' + b",".join([b"0.1"] * 34000) + b']}',  # > 128 KiB + 1
])
def test_pipe_invalid_wire_exits_two_ready_only(residual, line):
    _, _, bundle, built, _ = residual
    helper = _helper_path(bundle, built)
    proc = _serve(helper, bundle, line + b"\n")
    assert proc.returncode == 2
    out = [l for l in proc.stdout.decode().splitlines() if l.strip()]
    assert len(out) == 1  # ONLY the ready line; no GPU reports on stdout
    json.loads(out[0])


def test_pipe_truncated_eof_without_newline(residual):
    _, _, bundle, built, _ = residual
    helper = _helper_path(bundle, built)
    truncated = b'{"id":1,"input":[0.1,0.2,0.3'  # EOF mid-object, no trailing newline
    proc = _serve(helper, bundle, truncated)
    assert proc.returncode == 2
    out = [l for l in proc.stdout.decode().splitlines() if l.strip()]
    assert len(out) == 1  # only ready; truncated EOF is a protocol error, not a valid request
    json.loads(out[0])
