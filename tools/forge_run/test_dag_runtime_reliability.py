#!/usr/bin/env python3
"""Bounded reliability tests for the LoadedRuntime persistent worker session.

Requires an ACTUAL GPU helper and a real pre-forged bundle:
  LEVIATHAN_RELIABILITY_HELPER  path to the real helper binary
  LEVIATHAN_RELIABILITY_BUNDLE  path to a bundle forged from the unmodified
                                original examples/residual_block.py
Skips (not fails) when either is missing. No fakes, no network, no model
 downloads, no new dependencies; CPU torch is used only for eager source
 parity and fault injection. Every test only stops/kills processes it
 created itself and closes the runtime in a finally block.
"""
from __future__ import annotations

import math
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TOOL_DIR))

import dag_forge as f  # noqa: E402
import dag_runtime as rt  # noqa: E402

HELPER = os.environ.get("LEVIATHAN_RELIABILITY_HELPER")
BUNDLE = os.environ.get("LEVIATHAN_RELIABILITY_BUNDLE")
MODEL = TOOL_DIR / "examples" / "residual_block.py"

pytestmark = pytest.mark.skipif(
    not HELPER or not BUNDLE or not MODEL.is_file(),
    reason="LEVIATHAN_RELIABILITY_HELPER / LEVIATHAN_RELIABILITY_BUNDLE not set "
           "or original examples/residual_block.py not present; a real GPU "
           "helper and a real forged bundle are mandatory (no fakes)",
)


def _make_runtime(timeout=60.0):
    return rt.LoadedRuntime(BUNDLE, MODEL, helper=HELPER, timeout=timeout)


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _input(dim, value):
    return torch.full((1, dim), float(value), dtype=torch.float32)


def _source_forward(x):
    """Eager CPU float32 forward through a freshly loaded bound model."""
    loaded = f.load_bound_model(BUNDLE, MODEL, helper=HELPER)
    x = f.resolve_input(x, "input")
    with torch.no_grad():
        return f.forward_source(loaded["model"], x, "the input")


@pytest.fixture(scope="module")
def input_dim():
    loaded = f.load_bound_model(BUNDLE, MODEL, helper=HELPER)
    return int(loaded["inputDim"])


def test_sigkill_worker_poisons_session(input_dim):
    runtime = _make_runtime()
    try:
        pid = runtime.helperPID
        proc = runtime._proc
        assert pid is not None and _alive(pid)
        assert runtime.run(_input(input_dim, 0.25))["parity"]["passed"] is True
        os.kill(pid, signal.SIGKILL)  # own worker only
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.25))
        # terminal: no silent restart, session stays closed afterwards
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.25))
        assert runtime.helperPID is None
        proc.wait(timeout=2)  # reap our own killed worker; os.kill(pid,0) polls an unreaped zombie
        assert proc.poll() is not None
    finally:
        runtime.close()


def test_clean_eof_poisons_session(input_dim):
    runtime = _make_runtime()
    try:
        pid = runtime.helperPID
        proc = runtime._proc
        assert pid is not None
        assert runtime.run(_input(input_dim, 0.5))["parity"]["passed"] is True
        # Clean graceful EOF: closing the worker's stdin makes the worker exit
        # normally, so the next read sees EOF. This is the only non-API access
        # needed to simulate an orderly worker exit.
        runtime._proc.stdin.close()
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.5))
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.5))
        assert runtime.helperPID is None
        proc.wait(timeout=2)
        assert proc.poll() is not None
    finally:
        runtime.close()


def test_sigstop_timeout_closes_within_bound(input_dim):
    runtime = _make_runtime()
    try:
        pid = runtime.helperPID
        assert pid is not None and _alive(pid)
        os.kill(pid, signal.SIGSTOP)  # own worker only
        stopped = time.perf_counter()
        try:
            # Force the per-request IPC deadline to 50 ms (bounded).
            runtime._timeout = 0.05
            with pytest.raises(f.ForgeRunError):
                runtime.run(_input(input_dim, 0.75))
        finally:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass  # runtime already killed and reaped the stopped worker
            runtime.close()
        assert time.perf_counter() - stopped < 4.0
        assert runtime.helperPID is None
        deadline = time.monotonic() + 1.0
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _alive(pid)
    finally:
        runtime.close()


def test_valid_inputs_independent_and_bad_inputs_do_not_poison(input_dim):
    runtime = _make_runtime()
    try:
        pid = runtime.helperPID
        a = _input(input_dim, 0.3)
        b = _input(input_dim, -0.9)
        out_a1 = runtime.run(a)
        out_b = runtime.run(b)
        out_a2 = runtime.run(a)
        assert runtime.helperPID == pid  # same persistent worker throughout
        # Deterministic replay of the same input.
        self_a1 = torch.tensor(out_a1["output"], dtype=torch.float32)
        self_a2 = torch.tensor(out_a2["output"], dtype=torch.float32)
        self_b = torch.tensor(out_b["output"], dtype=torch.float32)
        assert torch.equal(self_a1, self_a2)
        # A and B are genuinely different inputs for this network.
        assert not torch.allclose(self_a1, self_b)
        # Actual output equals the eager CPU source module.
        ref_a = _source_forward(a)
        ref_b = _source_forward(b)
        assert torch.allclose(self_a1, ref_a, atol=1e-4, rtol=1e-4)
        assert torch.allclose(self_b, ref_b, atol=1e-4, rtol=1e-4)
        # Malformed width is rejected client-side before any IPC.
        with pytest.raises(f.ForgeRunError):
            runtime.run(torch.full((1, input_dim + 1), 0.1, dtype=torch.float32))
        with pytest.raises(f.ForgeRunError):
            runtime.run(torch.full((1, max(1, input_dim - 1)), 0.1, dtype=torch.float32))
        # NaN / Inf are rejected without reaching the worker.
        with pytest.raises(f.ForgeRunError):
            runtime.run(torch.full((1, input_dim), float("nan"), dtype=torch.float32))
        with pytest.raises(f.ForgeRunError):
            runtime.run(torch.full((1, input_dim), float("inf"), dtype=torch.float32))
        # None of the above poisoned the session or restarted the worker.
        assert runtime.helperPID == pid
        out_next = runtime.run(a)
        assert torch.equal(torch.tensor(out_next["output"], dtype=torch.float32), self_a1)
        assert runtime.helperPID == pid
    finally:
        runtime.close()


def test_cached_model_fault_injection_parity_failure_cleans_up(input_dim):
    runtime = _make_runtime()
    try:
        proc = runtime._proc
        assert runtime.run(_input(input_dim, 0.6))["parity"]["passed"] is True
        # Fault injection (the one sanctioned private-field access): perturb
        # the cached source module so CPU parity must fail terminally.
        linears = [m for m in runtime._ctx["model"].modules() if isinstance(m, torch.nn.Linear)]
        assert linears
        with torch.no_grad():
            linears[-1].bias += 1.0
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.6))
        assert runtime.helperPID is None  # terminal cleanup, no restart
        with pytest.raises(f.ForgeRunError):
            runtime.run(_input(input_dim, 0.6))
        proc.wait(timeout=2)
        assert proc.poll() is not None
    finally:
        runtime.close()


def test_timing_fields_include_parity(monkeypatch, input_dim):
    runtime = _make_runtime()
    try:
        real_forward = f.forward_source

        def slow_forward(model, x, label):
            time.sleep(0.02)
            return real_forward(model, x, label)

        # Wrap the source execution the live session actually invokes.
        monkeypatch.setattr(f, "forward_source", slow_forward)
        started = time.perf_counter()
        result = runtime.run(_input(input_dim, 0.4))
        external_wall_ms = (time.perf_counter() - started) * 1000.0
        timing = result["timing"]
        for key in ("requestWallMillis", "sourceParityMillis"):
            assert isinstance(timing[key], float) and math.isfinite(timing[key])
        # The injected 20 ms parity delay must appear in both wall and parity
        # timers (allowing scheduler slack on the lower bound).
        assert timing["sourceParityMillis"] >= 15.0
        assert timing["requestWallMillis"] >= 15.0
        assert timing["requestWallMillis"] >= timing["sourceParityMillis"]
        assert external_wall_ms >= timing["requestWallMillis"]
        assert timing["ipcRoundTripMillis"] >= 0.0
        assert result["parity"]["passed"] is True
    finally:
        runtime.close()


def test_four_concurrent_calls_serialize_on_same_pid(input_dim):
    runtime = _make_runtime()
    try:
        pid = runtime.helperPID
        values = [0.10, 0.45, -0.70, 0.95]
        inputs = [_input(input_dim, v) for v in values]
        references = [_source_forward(x) for x in inputs]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(runtime.run, inputs))
        assert runtime.helperPID == pid  # serialized on one persistent worker
        for out, ref, v in zip(outputs, references, values):
            got = torch.tensor(out["output"], dtype=torch.float32)
            assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), f"parity failed for value {v}"
            assert out["outputShape"] == [1, ref.shape[1]]
        # Pairwise distinct inputs give pairwise distinct outputs.
        tensors = [torch.tensor(o["output"], dtype=torch.float32) for o in outputs]
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                assert not torch.allclose(tensors[i], tensors[j])
        # Session still healthy afterwards.
        assert runtime.run(_input(input_dim, 0.5))["parity"]["passed"] is True
        assert runtime.helperPID == pid
    finally:
        runtime.close()
