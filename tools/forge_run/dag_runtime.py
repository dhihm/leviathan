#!/usr/bin/env python3
"""Loaded Leviathan DAG runtime client (persistent helper worker session).

Loads an existing immutable bundle once via dag_forge.load_bound_model (full
artifact/source/backend/weights/export validation, no GPU execution, no
re-capture per request), verifies the actual helper binary SHA against the
bundle binding, then launches ONE `helper serve --bundle` worker pinned to a
verified on-disk snapshot. Each .run() sends one compact NDJSON request line
and receives one report line that must pass the exact existing run-evidence
contract (_check_run_evidence) before CPU source parity. Any wire error, EOF,
timeout or parity failure poisons and closes the session terminally (no fake
CPU fallback, no silent restart). Not an LLM runtime. No agent runtime.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import dag_forge as f  # normal sibling import; runtime globals untouched

MAX_LINE = 1 << 20          # ready/response cap 1 MiB
MAX_REQUEST = 128 << 10     # request line cap 128 KiB
CHUNK = 4096
HELPER_CAP = 512 << 20      # bounded helper binary read for the SHA gate
ATOL = 1e-4
RTOL = 1e-4


def _finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v != float("inf") and v != float("-inf")


class LoadedRuntime:
    """Context manager over one verified bundle and one persistent worker."""

    def __init__(self, bundle_dir, model_path, helper=None, timeout=60.0):
        if not _finite_number(timeout) or isinstance(timeout, bool) or not 0.0 < float(timeout) <= 600.0:
            raise f.ForgeRunError("timeout must be a finite number in (0, 600]")
        self._timeout = float(timeout)
        self._lock = threading.Lock()
        self._proc = None
        self._err = None
        self._in_buf = b""
        self._next_id = 0
        self._poisoned = False
        t0 = time.perf_counter()
        self._ctx = f.load_bound_model(Path(bundle_dir), Path(model_path), helper=helper)
        python_validation_ms = float(self._ctx["validatedLoadMillis"])
        if not (_finite_number(python_validation_ms) and python_validation_ms >= 0.0):
            raise f.ForgeRunError("load_bound_model returned a malformed validation timer")
        helper_path = Path(self._ctx["helper"])
        meta = self._ctx["meta"]
        helper_meta = meta.get("helper") if isinstance(meta.get("helper"), dict) else None
        expected_sha = helper_meta.get("binarySha256") if isinstance(helper_meta, dict) else None
        if not isinstance(expected_sha, str) or not expected_sha:
            raise f.ForgeRunError("bundle.json helper binding is missing binarySha256")
        actual = f.sha256_hex(f.read_bounded(helper_path, HELPER_CAP, "helper binary"))
        if actual != expected_sha:
            raise f.ForgeRunError("helper binary sha256 does not match the bundle binding; re-forge explicitly (no silent retune)")
        t_launch = time.perf_counter()
        self._err = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                [str(helper_path), "serve", "--bundle", str(Path(bundle_dir))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err)
        except OSError as exc:
            self._close_unlocked()
            raise f.ForgeRunError(f"failed to launch worker: {exc}") from None
        os.set_blocking(self._proc.stdout.fileno(), False)
        os.set_blocking(self._proc.stdin.fileno(), False)
        try:
            deadline = time.monotonic() + self._timeout
            ready = f.parse_json(self._recv_line(deadline), "worker ready line")
            self._check_ready(ready, meta)
            st = ready["startupTiming"]
            if set(st.keys()) != {"validationMillis", "prepareMillis", "totalMillis"}:
                raise f.ForgeRunError("worker startupTiming keys are not the exact expected set")
            for key in ("validationMillis", "prepareMillis", "totalMillis"):
                if not _finite_number(st[key]) or st[key] < 0:
                    raise f.ForgeRunError(f"worker startupTiming.{key} is not a finite non-negative number")
        except BaseException:
            self._poisoned = True
            self._close_unlocked()
            raise
        t_ready = time.perf_counter()
        self.startupTimings = {
            "pythonValidationMillis": python_validation_ms,
            "helperLaunchReadyMillis": (t_ready - t_launch) * 1000.0,
            "clientInitializationMillis": (t_ready - t0) * 1000.0,
            "helperStartupTiming": dict(st),
        }

    @property
    def helperPID(self):
        proc = self._proc
        return proc.pid if proc is not None else None

    def _check_ready(self, ready, meta):
        if not isinstance(ready, dict):
            raise f.ForgeRunError("worker ready line is not an object")
        if set(ready.keys()) != {"format", "bundleSha256", "helper", "hardware", "shape",
                                 "selected", "selectedRole", "decision", "startupTiming"}:
            raise f.ForgeRunError("worker ready line root keys are not the exact expected set")
        if ready["format"] != "leviathan-dag-session-1":
            raise f.ForgeRunError("worker session format mismatch")
        selection = meta.get("selection") or {}
        checks = (
            ("bundleSha256", ready["bundleSha256"], self._ctx["bundleSha256"]),
            ("helper", ready["helper"], meta.get("helper")),
            ("hardware", ready["hardware"], meta.get("hardware")),
            ("shape", ready["shape"], meta.get("shape")),
            ("selected", ready["selected"], selection.get("selected")),
            ("selectedRole", ready["selectedRole"], selection.get("selectedRole")),
            ("decision", ready["decision"], selection.get("decision")),
        )
        for name, got, want in checks:
            if got != want:
                raise f.ForgeRunError(f"worker ready {name} does not match the bound bundle")
        if not isinstance(ready["startupTiming"], dict):
            raise f.ForgeRunError("worker ready line is missing startupTiming")

    def _wait_fd(self, fd, deadline, write=False):
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise f.ForgeRunError("worker IPC deadline exceeded")
            try:
                if write:
                    _, w, _ = select.select([], [fd], [], remain)
                    if w:
                        return
                else:
                    r, _, _ = select.select([fd], [], [], remain)
                    if r:
                        return
            except InterruptedError:
                continue
            except OSError as exc:
                raise f.ForgeRunError(f"worker IPC select failed: {exc}") from None

    def _recv_line(self, deadline):
        fd = self._proc.stdout.fileno()
        while True:
            i = self._in_buf.find(b"\n")
            if i >= 0:
                if i > MAX_LINE:
                    raise f.ForgeRunError("worker line exceeds 1 MiB")
                line, self._in_buf = self._in_buf[:i], self._in_buf[i + 1:]
                return line
            if len(self._in_buf) > MAX_LINE:
                raise f.ForgeRunError("worker line exceeds 1 MiB")
            self._wait_fd(fd, deadline)
            try:
                data = os.read(fd, CHUNK)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise f.ForgeRunError(f"worker IPC read failed: {exc}") from None
            if not data:
                raise f.ForgeRunError("worker closed the session (EOF)")
            self._in_buf += data

    def _send_all(self, data, deadline):
        fd = self._proc.stdin.fileno()
        view = memoryview(data)
        while view:
            self._wait_fd(fd, deadline, write=True)
            try:
                n = os.write(fd, view)
            except BlockingIOError:
                continue
            except OSError as exc:
                raise f.ForgeRunError(f"worker IPC write failed: {exc}") from None
            view = view[n:]

    def _stderr_tail(self):
        try:
            self._err.seek(0, os.SEEK_END)
            size = self._err.tell()
            self._err.seek(max(0, size - 4096))
            return self._err.read(4096).decode("utf-8", "replace")
        except Exception:
            return ""

    def _close_unlocked(self):
        proc, self._proc = self._proc, None
        err, self._err = self._err, None
        try:
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()  # graceful EOF
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            proc.kill()
                            proc.wait(timeout=1.0)
                        except Exception:
                            pass
                finally:
                    for stream in (proc.stdout, proc.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except OSError:
                                pass
        finally:
            if err is not None:
                try:
                    err.close()
                except OSError:
                    pass

    def close(self):
        with self._lock:
            self._close_unlocked()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _next_request_id(self):
        self._next_id += 1
        if self._next_id > 2 ** 31 - 1:
            raise f.ForgeRunError("request id space exhausted")
        return self._next_id

    def run(self, user_input):
        t_entry = time.perf_counter()
        with self._lock:
            if self._poisoned or self._proc is None:
                raise f.ForgeRunError("runtime session is closed/poisoned; no automatic restart")
            # Detached clone snapshot prior to any write; rejects before send.
            # Client-side gate: an invalid input here leaves the worker healthy.
            x = f.resolve_input(user_input, "input")
            f.validate_input_tensor(x, "input", self._ctx["inputDim"], "bound input dimension")
            try:
                rid = self._next_request_id()
                req = json.dumps({"id": rid, "input": x.flatten().tolist()},
                                 separators=(",", ":"), allow_nan=False).encode() + b"\n"
                if len(req) > MAX_REQUEST:
                    raise f.ForgeRunError(f"request line exceeds 128 KiB ({len(req)} bytes)")
                t_ipc = time.perf_counter()
                deadline = time.monotonic() + self._timeout
                self._send_all(req, deadline)
                reply = f.parse_json(self._recv_line(deadline), "worker reply")
                ipc_ms = (time.perf_counter() - t_ipc) * 1000.0
                if not isinstance(reply, dict) or set(reply.keys()) != {"id", "report"}:
                    raise f.ForgeRunError("worker reply is malformed; session closed")
                if not isinstance(reply["id"], int) or isinstance(reply["id"], bool) or reply["id"] != rid:
                    raise f.ForgeRunError(f"worker reply id {reply['id']!r} does not match request id {rid}; session closed")
                report = reply["report"]
                if not isinstance(report, dict):
                    raise f.ForgeRunError("worker report is not an object; session closed")
                timing_in = report.get("timing")
                if not isinstance(timing_in, dict):
                    raise f.ForgeRunError("worker report timing is not an object; session closed")
                f._check_run_evidence(self._ctx["graph"], report, timing_in, self._ctx["meta"])
                out_dim = self._ctx["outputDim"]
                raw = report["output"]
                if not isinstance(raw, list) or len(raw) != out_dim:
                    raise f.ForgeRunError(f"worker output width {len(raw) if isinstance(raw, list) else '?'} != bound {out_dim}")
                if report.get("outputShape") != [1, out_dim]:
                    raise f.ForgeRunError("worker output shape is not the bound [1, N] result")
                gpu_out = torch.tensor([[f.to_float32(v, "worker output") for v in raw]], dtype=torch.float32)
                if not bool(torch.isfinite(gpu_out).all()):
                    raise f.ForgeRunError("worker output values are not finite; session closed")
                t_par = time.perf_counter()
                reference = f.forward_source(self._ctx["model"], x, "the input")
                if tuple(reference.shape) != (1, out_dim) or not bool(torch.isfinite(reference).all()):
                    raise f.ForgeRunError("source reference output is not the bound finite [1, N] result")
                max_abs = (gpu_out - reference).abs().max().item()
                if not torch.allclose(gpu_out, reference, atol=ATOL, rtol=RTOL):
                    raise f.ForgeRunError(f"GPU output disagrees with the source module (max abs diff {max_abs:.3e}); no CPU fallback")
                parity_ms = (time.perf_counter() - t_par) * 1000.0
            except BaseException as exc:
                # Everything after the client tensor gate is terminal: poison and
                # close under the already-held lock, then reraise non-Exceptions.
                self._poisoned = True
                tail = self._stderr_tail()
                self._close_unlocked()
                suffix = f"; worker stderr: {tail}" if tail else ""
                if isinstance(exc, f.ForgeRunError):
                    raise f.ForgeRunError(f"{exc}{suffix}") from None
                if isinstance(exc, Exception):
                    raise f.ForgeRunError(f"unexpected {type(exc).__name__}: {exc}{suffix}") from None
                raise
            return {
                "output": gpu_out.tolist(),
                "outputShape": [1, out_dim],
                "selected": report.get("selected"),
                "selectedRole": report.get("selectedRole"),
                "decision": report.get("decision"),
                "gpuEvidence": report.get("gpu"),
                "parity": {"passed": True, "maxAbsDiff": max_abs, "atol": ATOL, "rtol": RTOL,
                           "reference": "source PyTorch nn.Module (eager CPU float32)"},
                "timing": {
                    "gpuCommandBufferMicros": timing_in.get("gpuCommandBufferMicros"),
                    "helperTotalMicros": timing_in.get("helperTotalMicros"),
                    "ipcRoundTripMillis": ipc_ms,
                    "sourceParityMillis": parity_ms,
                    "requestWallMillis": (time.perf_counter() - t_entry) * 1000.0,
                },
            }
