#!/usr/bin/env python3
"""Supervise one owned benchmark process with durable, recoverable receipts.

Launch protocol
---------------
1. A new private output directory (mode 0700) is created. Existing paths are
   refused and never reused or overwritten.
2. The supervisor spawns its own Python wrapper (``_exec-wrapper``). The
   wrapper blocks until it reads exactly ``START`` on stdin.
3. The supervisor reads the native start identity (PID, PPID, UID, start
   time) of itself and of that Popen-owned child with ``proc_pidinfo``. It
   durably writes that identity, the exact argv/cwd and the launch UUID to
   ``state.json`` before it releases ``START``.
4. The wrapper ``execvpe``s the exact benchmark argv in the same PID, so the
   recorded native start identity also identifies the benchmark.

Benchmark contract
------------------
* print a line ``BENCHMARK_READY`` (flushed), then wait for ``GO`` on stdin;
* print a line ``TIMED_WINDOW_END`` (flushed) when the timed window ends.

Before GO the supervisor can verify a fresh parent quiet acknowledgement, and
it snapshots ``vm_stat``/``pmset``. The acknowledgement is a pair of files:
``--quiet-request-file`` holds ``requestUTC`` and ``scope``, and
``--quiet-ready-file`` holds ``"ready": true``, ``readyUTC`` and the same
``scope``. Both timestamps must be timezone-aware UTC ISO 8601, and
``readyUTC`` must not precede ``requestUTC``. Freshness is derived only from
that explicit request, never from the local clock. After the end marker the
supervisor snapshots again.

Between GO and the end marker the supervisor takes no resource snapshots and
makes no native process reads. It reads the child's stdout through
``selectors``, calls ``Popen.poll``, appends each line to the logs and
atomically rewrites the state file. Those writes are observation overhead
inside the window; the overhead is small but not zero.

Recovery
--------
``status`` reads the persisted state and looks up only the exact stored
PIDs. It never sends signals and never infers an exit outcome from an
absent PID. Exit evidence comes only from the terminal receipt
(``receipt.json``), which records the supervisor's own ``Popen`` wait status.
The lock file is removed only after that receipt has been written, and the
removal is recorded in ``state.json`` (``cleanupResult``). If the receipt
cannot be written, the lock stays and the run exits nonzero.

Signals
-------
SIGINT and SIGTERM unwind into cleanup and a receipt. Cleanup sends at most
one SIGTERM to the owned child through its ``Popen`` handle and never sends
SIGKILL. A signal that arrives while cleanup or error handling is already
under way is recorded in ``signalsReceived`` but does not interrupt it.

SIGCHLD is set to its default disposition for the duration of ``run``. An
inherited ``SIG_IGN`` would let the kernel auto-reap the child, which would
fabricate its exit status. The previous disposition is restored afterwards.

SIGHUP is deliberately not handled. A hangup terminates the supervisor and
may leave partial state and no receipt, while the benchmark keeps running in
its own session. Recover with ``status``, which reads only the exact stored
PIDs.

Echo of benchmark stdout to the supervisor's own stdout is best effort. If
that stream fails, echo is disabled, a note is recorded, and logging and
state updates continue unchanged.

Usage::

    owned_benchmark_supervisor.py run --output-dir NEWDIR --cwd ABS \
        [--timeout-seconds N] [--quiet-request-file ABS --quiet-ready-file ABS] -- ARGV...
    owned_benchmark_supervisor.py status --output-dir DIR
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import datetime as _dt
import errno as _errno
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Protocol

SCHEMA_VERSION = 1
READY_MARKER = "BENCHMARK_READY"
END_MARKER = "TIMED_WINDOW_END"
START_TOKEN = b"START\n"
GO_TOKEN = b"GO\n"
WRAPPER_SUBCOMMAND = "_exec-wrapper"
LAUNCH_UUID_ENV = "OWNED_BENCHMARK_LAUNCH_UUID"
WRAPPER_NOT_RELEASED_EXIT = 125
WRAPPER_EXEC_FAILED_EXIT = 127

DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_QUIET_ACK_WAIT_SECONDS = 120.0
DEFAULT_GRACE_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.25
EXIT_DRAIN_SECONDS = 2.0
MAX_TPOT_LINES = 64
MAX_STATE_LINE_CHARS = 2000
MAX_READY_FILE_BYTES = 64 * 1024
MAX_PARTIAL_LINE_BYTES = 1024 * 1024
RESOURCE_OUTPUT_LIMIT = 64 * 1024

STATE_FILE = "state.json"
RECEIPT_FILE = "receipt.json"
LOCK_FILE = "supervisor.lock"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"
PROGRESS_FILE = "progress.jsonl"

DIR_MODE = 0o700
FILE_MODE = 0o600
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class ConfigError(ValueError):
    """Invalid run configuration; nothing has been created yet."""


class LaunchRefused(Exception):
    """START was withheld because the launch receipt could not be completed."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class SupervisorInterrupted(BaseException):
    """Raised from the SIGINT/SIGTERM handler to unwind into cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


# --- Native macOS start identity (libproc) ----------------------------------

LIBPROC_PATH = "/usr/lib/libproc.dylib"
PROC_PIDTBSDINFO = 3
MAXCOMLEN = 16
PROC_BSDINFO_SIZE = 136  # sizeof(struct proc_bsdinfo) on macOS

LOOKUP_OK = "ok"
LOOKUP_ABSENT = "absent"
LOOKUP_DENIED = "denied"
LOOKUP_ERROR = "error"
LOOKUP_UNAVAILABLE = "unavailable"


class ProcBsdInfo(ctypes.Structure):
    """ctypes mirror of ``struct proc_bsdinfo`` from the macOS SDK <sys/proc_info.h>."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),  # uid_t
        ("pbi_gid", ctypes.c_uint32),  # gid_t
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * MAXCOMLEN),
        ("pbi_name", ctypes.c_char * (2 * MAXCOMLEN)),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


@dataclasses.dataclass(frozen=True)
class NativeIdentity:
    """Start identity of one process as reported by ``proc_pidinfo``.

    ``comm`` is informational only: it changes when the wrapper execs the
    benchmark, so it is never used to decide identity.
    """

    pid: int
    ppid: int
    uid: int
    start_tvsec: int
    start_tvusec: int
    comm: str = ""

    @classmethod
    def from_bsdinfo(cls, info: ProcBsdInfo) -> NativeIdentity:
        return cls(
            pid=int(info.pbi_pid),
            ppid=int(info.pbi_ppid),
            uid=int(info.pbi_uid),
            start_tvsec=int(info.pbi_start_tvsec),
            start_tvusec=int(info.pbi_start_tvusec),
            comm=info.pbi_comm.decode("utf-8", "replace"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeIdentity:
        return cls(
            pid=int(data["pid"]),
            ppid=int(data["ppid"]),
            uid=int(data["uid"]),
            start_tvsec=int(data["start_tvsec"]),
            start_tvusec=int(data["start_tvusec"]),
            comm=str(data.get("comm", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProcLookup:
    """Result of one native lookup of one exact PID."""

    status: str
    pid: int
    identity: NativeIdentity | None = None
    errno: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pid": self.pid,
            "identity": self.identity.to_dict() if self.identity else None,
            "errno": self.errno,
            "errnoName": _errno.errorcode.get(self.errno) if self.errno else None,
            "detail": self.detail,
            "source": "proc_pidinfo(PROC_PIDTBSDINFO)",
        }


class ProcInfoProvider(Protocol):
    def lookup(self, pid: int) -> ProcLookup: ...


def interpret_proc_pidinfo(pid: int, returned: int, err: int, info: ProcBsdInfo) -> ProcLookup:
    """Map a ``proc_pidinfo(PROC_PIDTBSDINFO)`` return value and errno to a lookup."""
    size = ctypes.sizeof(ProcBsdInfo)
    if returned == size:
        identity = NativeIdentity.from_bsdinfo(info)
        if identity.pid != pid:
            return ProcLookup(LOOKUP_ERROR, pid, detail=f"kernel returned pid {identity.pid}")
        return ProcLookup(LOOKUP_OK, pid, identity=identity)
    if returned <= 0:
        if err == _errno.ESRCH:
            return ProcLookup(LOOKUP_ABSENT, pid, errno=err, detail="ESRCH: no process with this PID")
        if err in (_errno.EPERM, _errno.EACCES):
            return ProcLookup(LOOKUP_DENIED, pid, errno=err, detail="read denied; identity unknown")
        return ProcLookup(LOOKUP_ERROR, pid, errno=err or None, detail=f"proc_pidinfo returned {returned}")
    return ProcLookup(LOOKUP_ERROR, pid, errno=err or None, detail=f"short read: {returned} of {size} bytes")


class LibprocInfo:
    """Reads ``struct proc_bsdinfo`` for one exact PID via ``/usr/lib/libproc.dylib``."""

    def __init__(self, library_path: str = LIBPROC_PATH, loader: Callable[..., Any] = ctypes.CDLL) -> None:
        library = loader(library_path, use_errno=True)
        func = library.proc_pidinfo
        func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        func.restype = ctypes.c_int
        self._proc_pidinfo = func

    def lookup(self, pid: int) -> ProcLookup:
        info = ProcBsdInfo()
        ctypes.set_errno(0)
        returned = self._proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, ctypes.addressof(info), ctypes.sizeof(info))
        return interpret_proc_pidinfo(pid, int(returned), ctypes.get_errno(), info)


class UnavailableProcInfo:
    """Provider used where libproc cannot be loaded; every lookup is unknown."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def lookup(self, pid: int) -> ProcLookup:
        return ProcLookup(LOOKUP_UNAVAILABLE, pid, detail=self.reason)


def default_proc_info() -> ProcInfoProvider:
    if sys.platform != "darwin":
        return UnavailableProcInfo(f"native start identity requires macOS libproc; platform is {sys.platform}")
    try:
        return LibprocInfo()
    except (OSError, AttributeError) as exc:
        return UnavailableProcInfo(f"cannot load {LIBPROC_PATH}: {exc}")


def validate_launch_identities(
    supervisor: ProcLookup, child: ProcLookup, supervisor_pid: int, child_pid: int, uid: int
) -> list[str]:
    """Return problems that must withhold START; empty means the receipt is complete."""
    problems: list[str] = []
    for role, lookup, pid in (("supervisor", supervisor, supervisor_pid), ("child", child, child_pid)):
        if lookup.status != LOOKUP_OK or lookup.identity is None:
            problems.append(f"{role} lookup {lookup.status}: {lookup.detail}")
            continue
        if lookup.identity.pid != pid:
            problems.append(f"{role} identity pid {lookup.identity.pid} != {pid}")
        if lookup.identity.uid != uid:
            problems.append(f"{role} uid {lookup.identity.uid} != {uid}")
    if child.identity is not None and child.identity.ppid != supervisor_pid:
        problems.append(f"child ppid {child.identity.ppid} != supervisor pid {supervisor_pid}")
    return problems


IDENTITY_SAME = "same_process"
IDENTITY_DISTINCT = "distinct_process"
IDENTITY_UNCONFIRMED = "unconfirmed"
IDENTITY_ABSENT = "absent"
IDENTITY_UNKNOWN = "unknown"

ABSENCE_NOTE = (
    "An absent PID proves neither success nor failure; "
    "only a terminal receipt carries exit evidence."
)


def classify_identity(stored: dict[str, Any], lookup: ProcLookup) -> dict[str, Any]:
    """Compare a persisted identity with a fresh lookup of the same exact PID."""
    expected = NativeIdentity.from_dict(stored)
    result: dict[str, Any] = {"pid": expected.pid, "evidence": lookup.to_dict()}
    if lookup.status == LOOKUP_ABSENT:
        result["classification"] = IDENTITY_ABSENT
        result["reason"] = "proc_pidinfo reported ESRCH for the stored PID"
        result["note"] = ABSENCE_NOTE
    elif lookup.status != LOOKUP_OK or lookup.identity is None:
        result["classification"] = IDENTITY_UNKNOWN
        result["reason"] = f"lookup {lookup.status}: {lookup.detail}"
    else:
        actual = lookup.identity
        mismatched = [
            name for name in ("uid", "start_tvsec", "start_tvusec") if getattr(actual, name) != getattr(expected, name)
        ]
        if mismatched:
            result["classification"] = IDENTITY_DISTINCT
            result["reason"] = "PID now belongs to a different process (" + ", ".join(mismatched) + " differ)"
        elif actual.ppid != expected.ppid:
            result["classification"] = IDENTITY_UNCONFIRMED
            result["reason"] = f"uid and start time match but ppid is {actual.ppid}, expected {expected.ppid}"
        else:
            result["classification"] = IDENTITY_SAME
            result["reason"] = "uid, start time and ppid match the persisted identity"
    return result


# --- Files -------------------------------------------------------------------


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def create_output_dir(path: str) -> str:
    """Create a brand-new private directory; refuse anything that already exists."""
    path = os.path.abspath(path)
    os.mkdir(path, DIR_MODE)  # FileExistsError if present; never reused
    os.chmod(path, DIR_MODE)
    return path


def open_new_file(path: str) -> Any:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC, FILE_MODE)
    os.fchmod(fd, FILE_MODE)
    return os.fdopen(fd, "wb")


def _fsync_directory(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: str, payload: Any, *, overwrite: bool = True, durable: bool = True) -> None:
    """Write JSON via a private temp file and rename (or link when not overwriting).

    ``durable`` fsyncs the file and its directory. ``overwrite=False`` uses
    ``os.link`` so an existing target is never replaced.
    """
    directory = os.path.dirname(os.path.abspath(path))
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC, FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    if durable:
        _fsync_directory(directory)


def read_json(path: str) -> Any:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _read_small_json_object(path: str) -> tuple[str, Any, str]:
    """Return ``(status, data, detail)``; status is ok/absent/denied/error/invalid."""
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC)
    except FileNotFoundError:
        return "absent", None, ""
    except PermissionError as exc:
        return "denied", None, str(exc)
    except OSError as exc:
        return "error", None, str(exc)
    with os.fdopen(fd, "rb") as handle:
        raw = handle.read(MAX_READY_FILE_BYTES + 1)
    if len(raw) > MAX_READY_FILE_BYTES:
        return "error", None, "file too large"
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "invalid", None, str(exc)
    if not isinstance(data, dict):
        return "invalid", None, "expected a JSON object"
    return "ok", data, ""


def read_quiet_ack(path: str, checked_utc: str) -> dict[str, Any]:
    """Read the parent's quiet ready file; GO requires ``"ready": true``.

    ``readyUTC``, ``scope`` and ``source`` are preserved exactly as the parent wrote them.
    """
    ack: dict[str, Any] = {
        "path": path,
        "checkedUTC": checked_utc,
        "status": "error",
        "readyUTC": None,
        "scope": None,
        "source": None,
        "detail": "",
    }
    status, data, detail = _read_small_json_object(path)
    if status != "ok":
        ack["status"] = status
        ack["detail"] = detail
        return ack
    ack["readyUTC"] = data.get("readyUTC")
    ack["scope"] = data.get("scope")
    ack["source"] = data.get("source")
    ack["status"] = "ready" if data.get("ready") is True else "not_ready"
    return ack


def read_quiet_request(path: str, checked_utc: str) -> dict[str, Any]:
    """Read the parent's quiet request file; fields are preserved exactly as written."""
    request: dict[str, Any] = {
        "path": path,
        "checkedUTC": checked_utc,
        "status": "error",
        "requestUTC": None,
        "scope": None,
        "source": None,
        "detail": "",
    }
    status, data, detail = _read_small_json_object(path)
    if status != "ok":
        request["status"] = status
        request["detail"] = detail
        return request
    request["requestUTC"] = data.get("requestUTC")
    request["scope"] = data.get("scope")
    request["source"] = data.get("source")
    request["status"] = "present"
    return request


def parse_aware_utc(value: Any) -> tuple[_dt.datetime | None, str]:
    """Parse a timezone-aware UTC ISO 8601 timestamp; naive or non-UTC values are rejected."""
    if not isinstance(value, str) or not value:
        return None, "missing or non-string timestamp"
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        return None, f"malformed ISO 8601 timestamp: {exc}"
    offset = parsed.utcoffset()
    if offset is None:
        return None, "naive timestamp without a UTC offset"
    if offset != _dt.timedelta(0):
        return None, f"timestamp offset {offset} is not UTC"
    return parsed, ""


def evaluate_quiet_pair(request: dict[str, Any], ready: dict[str, Any]) -> tuple[str, str]:
    """Decide whether a ready file acknowledges this explicit request.

    Freshness is derived only from ``requestUTC``; the local clock is never consulted.
    """
    if request["status"] != "present":
        return f"request_{request['status']}", request["detail"]
    request_time, problem = parse_aware_utc(request["requestUTC"])
    if request_time is None:
        return "request_malformed_utc", problem
    if not isinstance(request["scope"], str) or not request["scope"]:
        return "request_invalid_scope", "request scope must be a non-empty string"
    if ready["status"] == "not_ready":
        return "not_ready", 'ready file lacks a literal "ready": true'
    if ready["status"] != "ready":
        return f"ready_{ready['status']}", ready["detail"]
    ready_time, problem = parse_aware_utc(ready["readyUTC"])
    if ready_time is None:
        return "ready_malformed_utc", problem
    if ready["scope"] != request["scope"]:
        return "scope_mismatch", f"ready scope {ready['scope']!r} != request scope {request['scope']!r}"
    if ready_time < request_time:
        return "stale", f"readyUTC {ready['readyUTC']} precedes requestUTC {request['requestUTC']}"
    return "ready", "readyUTC is not before requestUTC and scopes match"


# --- Resource snapshots (outside the timed window only) ---------------------


class ResourceProvider(Protocol):
    def snapshot(self, label: str) -> dict[str, Any]: ...


DEFAULT_RESOURCE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vm_stat", ("/usr/bin/vm_stat",)),
    ("pmset_therm", ("/usr/bin/pmset", "-g", "therm")),
)


class CommandResourceProvider:
    """Runs fixed read-only system commands and records their output verbatim."""

    def __init__(
        self,
        commands: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_RESOURCE_COMMANDS,
        runner: Callable[..., Any] = subprocess.run,
        timeout: float = 10.0,
    ) -> None:
        self.commands = commands
        self.runner = runner
        self.timeout = timeout

    def snapshot(self, label: str) -> dict[str, Any]:
        result: dict[str, Any] = {"label": label, "utc": utc_now(), "commands": {}}
        for name, argv in self.commands:
            entry: dict[str, Any] = {"argv": list(argv)}
            try:
                completed = self.runner(
                    list(argv),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
                entry["returncode"] = completed.returncode
                entry["stdout"] = completed.stdout[:RESOURCE_OUTPUT_LIMIT]
                entry["stderr"] = completed.stderr[:RESOURCE_OUTPUT_LIMIT]
            except (OSError, subprocess.SubprocessError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            result["commands"][name] = entry
        return result


# --- Child process and stdout reading -----------------------------------------


def wrapper_argv(benchmark_argv: list[str]) -> list[str]:
    return [sys.executable, "-I", os.path.abspath(__file__), WRAPPER_SUBCOMMAND, "--", *benchmark_argv]


def popen_spawn(argv: list[str], cwd: str, env: dict[str, str], stderr: Any) -> subprocess.Popen:
    # A new session keeps terminal SIGINT away from the child; cleanup is
    # performed only by this supervisor through its own Popen handle.
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        close_fds=True,
        start_new_session=True,
        bufsize=0,
    )


def _decode_line(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").rstrip("\r")


class SelectorLineReader:
    """Line reader over a pipe that never blocks longer than the given timeout."""

    def __init__(self, stream: Any) -> None:
        self._fd = stream.fileno()
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._fd, selectors.EVENT_READ)
        self._buffer = b""
        self._eof = False

    def read_lines(self, timeout: float) -> tuple[list[str], bool]:
        if self._eof:
            return [], True
        if not self._selector.select(max(0.0, timeout)):
            return [], False
        chunk = os.read(self._fd, 65536)
        if not chunk:
            self._eof = True
            self._selector.close()
            tail, self._buffer = self._buffer, b""
            return ([_decode_line(tail)] if tail else []), True
        self._buffer += chunk
        parts = self._buffer.split(b"\n")
        self._buffer = parts.pop()
        if len(self._buffer) > MAX_PARTIAL_LINE_BYTES:
            parts.append(self._buffer)
            self._buffer = b""
        return [_decode_line(part) for part in parts], False


# --- Supervisor ----------------------------------------------------------------


@dataclasses.dataclass
class RunConfig:
    output_dir: str
    cwd: str
    argv: list[str]
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    quiet_ready_file: str | None = None
    quiet_ack_wait_seconds: float = DEFAULT_QUIET_ACK_WAIT_SECONDS
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    echo: bool = True
    quiet_request_file: str | None = None

    def validate(self) -> None:
        if not self.argv:
            raise ConfigError("benchmark argv is empty; pass it after --")
        if not os.path.isabs(self.cwd) or not os.path.isdir(self.cwd):
            raise ConfigError(f"--cwd must be an existing absolute directory: {self.cwd!r}")
        if self.quiet_ready_file is not None and not os.path.isabs(self.quiet_ready_file):
            raise ConfigError(f"--quiet-ready-file must be absolute: {self.quiet_ready_file!r}")
        if (self.quiet_ready_file is None) != (self.quiet_request_file is None):
            raise ConfigError("--quiet-ready-file and --quiet-request-file must be given together")
        if self.quiet_request_file is not None and not os.path.isabs(self.quiet_request_file):
            raise ConfigError(f"--quiet-request-file must be absolute: {self.quiet_request_file!r}")
        if not self.timeout_seconds > 0:
            raise ConfigError("--timeout-seconds must be positive")
        if self.quiet_ack_wait_seconds < 0 or self.grace_seconds < 0:
            raise ConfigError("wait durations must not be negative")


def _span(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(end - start, 6)


class Supervisor:
    """Runs one benchmark through the START-gated wrapper and writes receipts."""

    def __init__(
        self,
        config: RunConfig,
        *,
        proc_info: ProcInfoProvider | None = None,
        resources: ResourceProvider | None = None,
        spawn: Callable[..., Any] | None = None,
        reader_factory: Callable[[Any], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], str] = utc_now,
        install_signal_handlers: bool = True,
        launch_uuid: str | None = None,
        echo_stream: Any = None,
    ) -> None:
        self.config = config
        self.proc_info = proc_info if proc_info is not None else default_proc_info()
        self.resources = resources if resources is not None else CommandResourceProvider()
        self.spawn = spawn or popen_spawn
        self.reader_factory = reader_factory or (lambda child: SelectorLineReader(child.stdout))
        self.monotonic = monotonic
        self.wall = wall
        self.install_signal_handlers = install_signal_handlers
        self.launch_uuid = launch_uuid or str(uuid.uuid4())
        self.echo_stream = echo_stream if echo_stream is not None else sys.stdout
        self.output_dir = os.path.abspath(config.output_dir)

        self.child: Any = None
        self.reader: Any = None
        self.files: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.notes: list[str] = []
        self.signals_received: list[str] = []
        self._signal_raised = False
        self._in_cleanup = False

        self.spawn_mono: float | None = None
        self.start_released = False
        self.ready_observed = False
        self.ready_mono: float | None = None
        self.go_sent = False
        self.go_mono: float | None = None
        self.end_observed = False
        self.end_mono: float | None = None
        self.returncode: int | None = None
        self.exit_mono: float | None = None
        self.timed_out = False
        self.interrupted_by: str | None = None
        self.interrupted_signum: int | None = None
        self.abort_reason: str | None = None
        self.error: dict[str, str] | None = None
        self.cleanup_action: str | None = None
        self.child_status = "never_spawned"
        self.resources_pre: dict[str, Any] | None = None
        self.resources_post: dict[str, Any] | None = None
        self.post_skipped_reason: str | None = None
        self.quiet_ack: dict[str, Any] | None = None
        self.lock_created = False
        self.lock_released = False
        self.lock_release_error: str | None = None
        self.receipt_written = False
        self.echo_enabled = config.echo

    # -- top level ---------------------------------------------------------

    def run(self) -> int:
        """Run to completion; raises only if the output directory cannot be created."""
        self.config.validate()
        create_output_dir(self.output_dir)
        previous = self._install_signals()
        try:
            try:
                self._launch_and_stream()
            except SupervisorInterrupted as exc:
                self._in_cleanup = True  # later signals are recorded, not raised
                self.interrupted_signum = int(exc.signum)
                self.interrupted_by = signal.Signals(exc.signum).name
                self._safe_phase("interrupted")
            except LaunchRefused as exc:
                self._in_cleanup = True
                self.abort_reason = exc.reason
                self.notes.append(str(exc))
            except BaseException as exc:  # recorded in the receipt; cleanup follows
                self._in_cleanup = True
                self.error = {"type": type(exc).__name__, "message": str(exc)}
                self._safe_phase("error")
            finally:
                self._in_cleanup = True
                self._cleanup_child()
                self._post_snapshot_after_exit()
                self._write_receipt()
                self._finish_after_receipt()
        finally:
            self._restore_signals(previous)
            self._close_files()
        return self._exit_code()

    def _launch_and_stream(self) -> None:
        cfg = self.config
        self._create_lock()
        self._open_files()
        self.state = self._initial_state()
        self._set_phase("created")

        supervisor_lookup = self.proc_info.lookup(os.getpid())
        self.state["supervisor"]["nativeLookup"] = supervisor_lookup.to_dict()
        if supervisor_lookup.identity is not None:
            self.state["supervisor"]["nativeIdentity"] = supervisor_lookup.identity.to_dict()
        self._set_phase("spawning_wrapper")

        env = dict(os.environ)
        env[LAUNCH_UUID_ENV] = self.launch_uuid
        self.child = self.spawn(self.state["wrapperArgv"], cfg.cwd, env, self.files["stderr"])
        self.spawn_mono = self.monotonic()
        self.child_status = "running"
        self.state["child"]["pid"] = self.child.pid
        self.state["child"]["spawnedUTC"] = self.wall()
        self._set_phase("wrapper_spawned")

        child_lookup = self.proc_info.lookup(self.child.pid)
        self.state["child"]["nativeLookup"] = child_lookup.to_dict()
        problems = validate_launch_identities(
            supervisor_lookup, child_lookup, os.getpid(), self.child.pid, os.getuid()
        )
        if problems:
            self.state["identityProblems"] = problems
            self._set_phase("identity_rejected")
            raise LaunchRefused(
                "native_identity_unverified",
                "native identity receipt incomplete; START withheld: " + "; ".join(problems),
            )
        self.state["child"]["nativeIdentity"] = child_lookup.identity.to_dict()
        self._set_phase("identity_recorded")  # durable before START

        self.child.stdin.write(START_TOKEN)
        self.child.stdin.flush()
        self.start_released = True
        self._set_phase("start_released")
        self.reader = self.reader_factory(self.child)
        self._stream_loop()

    def _stream_loop(self) -> None:
        deadline = self.spawn_mono + self.config.timeout_seconds
        eof = False
        exit_drain_deadline: float | None = None
        while True:
            now = self.monotonic()
            if now >= deadline:
                self.timed_out = True
                self._set_phase("timeout")
                return
            if eof:
                break
            if self.ready_observed and not self.go_sent:
                self._maybe_send_go()
                if self.abort_reason:
                    return
            returncode = self.child.poll()
            if returncode is not None:
                self._record_exit(returncode)
                if exit_drain_deadline is None:
                    exit_drain_deadline = now + EXIT_DRAIN_SECONDS
                elif now >= exit_drain_deadline:
                    self.notes.append("stdout stayed open after child exit; stopped reading")
                    break
            lines, eof = self.reader.read_lines(min(deadline - now, POLL_INTERVAL_SECONDS))
            for line in lines:
                self._handle_line(line)
        if self.returncode is None:
            self._set_phase("stdout_closed")
            try:
                returncode = self.child.wait(timeout=max(0.0, deadline - self.monotonic()))
            except subprocess.TimeoutExpired:
                self.timed_out = True
                self._set_phase("timeout")
                return
            self._record_exit(returncode)
        self._set_phase("child_exited")

    def _maybe_send_go(self) -> None:
        cfg = self.config
        if cfg.quiet_ready_file is not None:
            checked_utc = self.wall()
            request = read_quiet_request(cfg.quiet_request_file, checked_utc)
            ready = read_quiet_ack(cfg.quiet_ready_file, checked_utc)
            status, detail = evaluate_quiet_pair(request, ready)
            ack = {"status": status, "detail": detail, "checkedUTC": checked_utc, "request": request, "ready": ready}
            self.quiet_ack = ack
            if ack["status"] != "ready":
                if self.monotonic() - self.ready_mono >= cfg.quiet_ack_wait_seconds:
                    self.abort_reason = f"quiet_ack_{ack['status']}"
                    self.state["quietAck"] = ack
                    self._set_phase("quiet_ack_refused")
                return
            self.state["quietAck"] = ack
            self._set_phase("quiet_ack_verified")
        self._set_phase("pre_go_snapshot")
        self.resources_pre = self.resources.snapshot("pre_go")
        self._set_phase("sending_go")  # after a session loss: GO may or may not have been sent
        try:
            self.child.stdin.write(GO_TOKEN)
            self.child.stdin.flush()
        except BrokenPipeError:
            # The child closed stdin, typically by exiting; cleanup collects its
            # exit status through Popen.
            self.abort_reason = "go_pipe_closed"
            self._set_phase("go_pipe_closed")
            return
        self.child.stdin.close()
        self.go_sent = True
        self.go_mono = self.monotonic()
        self.state["markers"]["goSentUTC"] = self.wall()
        self._set_phase("timed_window", durable=False)

    def _handle_line(self, line: str) -> None:
        now = self.monotonic()
        utc = self.wall()
        stdout_log = self.files["stdout"]
        stdout_log.write(line.encode("utf-8", "replace") + b"\n")
        stdout_log.flush()
        if self.echo_enabled:
            # Echo is best effort: a closed or hung-up launcher stream must not
            # abort a healthy benchmark. The log, progress and state writes below
            # do not depend on it.
            try:
                self.echo_stream.write(line + "\n")
                self.echo_stream.flush()
            except (OSError, ValueError) as exc:
                self.echo_enabled = False
                self.notes.append(f"echo disabled after {type(exc).__name__}: {exc}; logging continues")

        stripped = line.strip()
        marker = None
        if stripped == READY_MARKER and not self.ready_observed:
            marker = "ready"
        elif stripped == END_MARKER and not self.end_observed:
            marker = "end" if self.go_sent else "end_before_go_ignored"
        is_tpot = "tpot" in line.lower()

        progress = self.state["progress"]
        progress["lineCount"] += 1
        progress["lastLineUTC"] = utc
        progress["lastLine"] = line[:MAX_STATE_LINE_CHARS]
        if is_tpot:
            progress["tpotLines"].append({"utc": utc, "line": line[:MAX_STATE_LINE_CHARS]})
            del progress["tpotLines"][:-MAX_TPOT_LINES]
        record = {
            "seq": progress["lineCount"],
            "utc": utc,
            "elapsedSeconds": _span(self.spawn_mono, now),
            "phase": self.state["phase"],
            "marker": marker,
            "tpot": is_tpot,
            "line": line,
        }
        progress_log = self.files["progress"]
        progress_log.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        progress_log.flush()

        if marker == "ready":
            self.ready_observed = True
            self.ready_mono = now
            self.state["markers"]["readyObservedUTC"] = utc
            self._set_phase("awaiting_quiet_ack" if self.config.quiet_ready_file else "ready_observed")
        elif marker == "end":
            self.end_observed = True
            self.end_mono = now
            self.state["markers"]["endObservedUTC"] = utc
            self._set_phase("window_ended")
            self.resources_post = self.resources.snapshot("post_end")
            self._set_phase("post_window")
        else:
            if marker == "end_before_go_ignored":
                self.notes.append(f"{END_MARKER} seen before GO at line {progress['lineCount']}; ignored")
            self._write_state(durable=False)

    # -- cleanup and receipt -------------------------------------------------

    def _record_exit(self, returncode: int) -> None:
        if self.returncode is None:
            self.returncode = returncode
            self.exit_mono = self.monotonic()

    def _bounded_wait(self, child: Any) -> int | None:
        try:
            return child.wait(timeout=self.config.grace_seconds)
        except subprocess.TimeoutExpired:
            return None

    def _cleanup_child(self) -> None:
        """Gracefully stop only our own live Popen child; never escalate."""
        child = self.child
        if child is None:
            self.child_status = "never_spawned"
            return
        returncode = self.returncode
        try:
            returncode = child.poll()
            if returncode is None and not self.start_released:
                # The wrapper exits on stdin EOF without ever exec'ing.
                self.cleanup_action = "closed_stdin_before_start"
                self._close_quietly(child.stdin)
                returncode = self._bounded_wait(child)
            if returncode is None:
                self.cleanup_action = "sigterm"
                child.terminate()
                returncode = self._bounded_wait(child)
        except Exception as exc:  # record and fall through; no escalation
            self.notes.append(f"cleanup error: {type(exc).__name__}: {exc}")
        if returncode is None:
            self.child_status = "unknown_still_running"
            self.notes.append("child did not exit within the grace period; status unknown, lock kept")
        else:
            self._record_exit(returncode)
            self.child_status = "exited"
        self._close_quietly(getattr(child, "stdin", None))
        self._close_quietly(getattr(child, "stdout", None))

    def _post_snapshot_after_exit(self) -> None:
        if self.resources_post is not None:
            return
        if not self.go_sent:
            self.post_skipped_reason = "go_not_sent"
        elif self.child_status != "exited":
            self.post_skipped_reason = "child_status_unknown_window_may_be_open"
        else:
            try:
                self.resources_post = self.resources.snapshot("post_exit")
            except Exception as exc:
                self.post_skipped_reason = f"snapshot failed: {type(exc).__name__}: {exc}"

    def _create_lock(self) -> None:
        payload = {
            "launchUUID": self.launch_uuid,
            "supervisorPid": os.getpid(),
            "createdUTC": self.wall(),
            "note": "Removed only after the child exit status is known and the terminal receipt is written.",
        }
        write_json_atomic(os.path.join(self.output_dir, LOCK_FILE), payload, overwrite=False)
        self.lock_created = True

    def _lock_release_eligible(self) -> bool:
        return self.lock_created and self.child_status in ("exited", "never_spawned")

    def _release_lock(self) -> None:
        try:
            os.unlink(os.path.join(self.output_dir, LOCK_FILE))
            self.lock_released = True
        except OSError as exc:
            self.lock_release_error = f"{type(exc).__name__}: {exc}"
            print(f"supervisor: lock release failed: {self.lock_release_error}", file=sys.stderr)

    def _result(self) -> str:
        if self.child_status == "unknown_still_running":
            return "child_status_unknown"
        if self.interrupted_by:
            return "interrupted"
        if self.timed_out:
            return "timeout"
        if self.error:
            return "error"
        if self.abort_reason:
            return "aborted_" + self.abort_reason
        if self.child_status != "exited" or self.returncode is None:
            return "unknown"
        if self.returncode == 0 and self.end_observed:
            return "success"
        if self.returncode == 0:
            return "exited_zero_without_end_marker"
        return "exited_nonzero"

    def _build_receipt(self) -> dict[str, Any]:
        rc = self.returncode
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "owned_benchmark_supervisor_receipt",
            "launchUUID": self.launch_uuid,
            "receiptUTC": self.wall(),
            "benchmark": {"argv": list(self.config.argv), "cwd": self.config.cwd},
            "supervisorPid": os.getpid(),
            "childPid": self.child.pid if self.child is not None else None,
            "childStatus": self.child_status,
            "exitCode": rc if rc is not None and rc >= 0 else None,
            "exitSignal": -rc if rc is not None and rc < 0 else None,
            "exitEvidenceSource": "supervisor_popen_wait_status" if rc is not None else None,
            "result": self._result(),
            "timedOut": self.timed_out,
            "interruptedBy": self.interrupted_by,
            "signalsReceived": list(self.signals_received),
            "abortReason": self.abort_reason,
            "error": self.error,
            "cleanupAction": self.cleanup_action,
            "startReleased": self.start_released,
            "readyObserved": self.ready_observed,
            "goSent": self.go_sent,
            "endMarkerObserved": self.end_observed,
            "markers": dict(self.state.get("markers") or {}),
            "elapsedSeconds": _span(self.spawn_mono, self.exit_mono),
            "windowElapsedSeconds": _span(self.go_mono, self.end_mono),
            "resources": {
                "preGo": self.resources_pre,
                "postWindow": self.resources_post,
                "postWindowSkippedReason": self.post_skipped_reason,
            },
            "quietAck": self.quiet_ack,
            "progress": self.state.get("progress"),
            "lockPath": os.path.join(self.output_dir, LOCK_FILE),
            # Eligibility is decided before this receipt is written. The actual
            # removal happens only afterwards and is recorded in state.json.
            "lockReleaseEligible": self._lock_release_eligible(),
            "notes": list(self.notes),
        }

    def _write_receipt(self) -> None:
        try:
            write_json_atomic(os.path.join(self.output_dir, RECEIPT_FILE), self._build_receipt(), overwrite=False)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"supervisor: failed to write terminal receipt: {message}; lock kept", file=sys.stderr)
            if self.state:
                self.state["receiptError"] = message
                self._safe_phase("receipt_write_failed")
            return
        self.receipt_written = True

    def _finish_after_receipt(self) -> None:
        """Release the lock only once the terminal receipt exists, then record what happened."""
        if not self.receipt_written:
            return
        eligible = self._lock_release_eligible()
        if eligible:
            self._release_lock()
        if self.state:
            self.state["receiptWritten"] = True
            self.state["cleanupResult"] = {
                "lockReleaseEligible": eligible,
                "lockReleased": self.lock_released,
                "lockReleaseError": self.lock_release_error,
            }
            self._safe_phase("finished")

    def _exit_code(self) -> int:
        if self.interrupted_signum is not None:
            return 128 + self.interrupted_signum
        if not self.receipt_written:
            return 1  # no durable exit evidence, so never report success
        return 0 if self._result() == "success" else 1

    # -- state helpers -------------------------------------------------------

    def _initial_state(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "owned_benchmark_supervisor_state",
            "launchUUID": self.launch_uuid,
            "createdUTC": self.wall(),
            "outputDir": self.output_dir,
            "benchmark": {"argv": list(cfg.argv), "cwd": cfg.cwd},
            "wrapperArgv": wrapper_argv(cfg.argv),
            "timeoutSeconds": cfg.timeout_seconds,
            "quietReadyFile": cfg.quiet_ready_file,
            "quietRequestFile": cfg.quiet_request_file,
            "phase": None,
            "phaseUTC": None,
            "phaseHistory": [],
            "supervisor": {"pid": os.getpid(), "uid": os.getuid(), "nativeLookup": None, "nativeIdentity": None},
            "child": {"pid": None, "spawnedUTC": None, "nativeLookup": None, "nativeIdentity": None},
            "markers": {"readyObservedUTC": None, "goSentUTC": None, "endObservedUTC": None},
            "progress": {"lineCount": 0, "lastLineUTC": None, "lastLine": None, "tpotLines": []},
            "quietAck": None,
            "receiptWritten": False,
        }

    def _set_phase(self, phase: str, *, durable: bool = True) -> None:
        utc = self.wall()
        self.state["phase"] = phase
        self.state["phaseUTC"] = utc
        self.state["phaseHistory"].append({"phase": phase, "utc": utc})
        self._write_state(durable=durable)

    def _write_state(self, *, durable: bool) -> None:
        write_json_atomic(os.path.join(self.output_dir, STATE_FILE), self.state, durable=durable)

    def _safe_phase(self, phase: str) -> None:
        if not self.state:
            return
        try:
            self._set_phase(phase)
        except Exception as exc:
            self.notes.append(f"state write failed: {type(exc).__name__}: {exc}")

    def _open_files(self) -> None:
        for key, name in (("stdout", STDOUT_LOG), ("stderr", STDERR_LOG), ("progress", PROGRESS_FILE)):
            self.files[key] = open_new_file(os.path.join(self.output_dir, name))

    def _close_files(self) -> None:
        for handle in self.files.values():
            self._close_quietly(handle)

    @staticmethod
    def _close_quietly(stream: Any) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except OSError:
            pass

    def _install_signals(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}
        # An inherited SIG_IGN for SIGCHLD lets the kernel auto-reap the child.
        # Popen would then report a fabricated exit status of 0, and terminate()
        # could reach a reused PID. Use the default while this run owns its child.
        previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        if self.install_signal_handlers:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, self._on_signal)
        return previous

    @staticmethod
    def _restore_signals(previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            if handler is not None:  # None: the previous handler was not installed from Python
                signal.signal(signum, handler)

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.signals_received.append(signal.Signals(signum).name)
        if self._in_cleanup or self._signal_raised:
            return
        self._signal_raised = True
        raise SupervisorInterrupted(signum)


# --- Status (read-only recovery) ---------------------------------------------


def _outcome_from_receipt(state: dict[str, Any], receipt: Any) -> dict[str, Any]:
    if receipt is None:
        return {"result": "unknown", "evidenceSource": "none", "note": "No terminal receipt. " + ABSENCE_NOTE}
    if not isinstance(receipt, dict) or receipt.get("launchUUID") != state.get("launchUUID"):
        return {
            "result": "unknown",
            "evidenceSource": "none",
            "note": "Receipt launchUUID does not match the state file; receipt ignored.",
        }
    keys = (
        "result",
        "childStatus",
        "exitCode",
        "exitSignal",
        "exitEvidenceSource",
        "endMarkerObserved",
        "timedOut",
        "lockReleaseEligible",
        "receiptUTC",
    )
    return {"evidenceSource": "terminal_receipt", **{key: receipt.get(key) for key in keys}}


def collect_status(output_dir: str, proc_info: ProcInfoProvider) -> dict[str, Any]:
    """Report persisted progress plus a fresh identity check of the exact stored PIDs.

    Read-only: no signals, no process enumeration, and no exit status is
    inferred from an absent PID.
    """
    output_dir = os.path.abspath(output_dir)
    state = read_json(os.path.join(output_dir, STATE_FILE))
    if not isinstance(state, dict):
        raise ValueError("state file is not a JSON object")
    try:
        receipt = read_json(os.path.join(output_dir, RECEIPT_FILE))
    except FileNotFoundError:
        receipt = None
    progress = state.get("progress") or {}
    report: dict[str, Any] = {
        "outputDir": output_dir,
        "checkedUTC": utc_now(),
        "launchUUID": state.get("launchUUID"),
        "phase": state.get("phase"),
        "phaseUTC": state.get("phaseUTC"),
        "benchmark": state.get("benchmark"),
        "lineCount": progress.get("lineCount"),
        "lastLineUTC": progress.get("lastLineUTC"),
        "lockPresent": os.path.lexists(os.path.join(output_dir, LOCK_FILE)),
        "processes": {},
    }
    for role in ("supervisor", "child"):
        entry = state.get(role) or {}
        pid = entry.get("pid")
        stored = entry.get("nativeIdentity")
        if not isinstance(pid, int) or not isinstance(stored, dict) or stored.get("pid") != pid:
            report["processes"][role] = {
                "pid": pid,
                "classification": IDENTITY_UNKNOWN,
                "reason": "no persisted native identity for this PID; not inspected",
                "evidence": None,
            }
            continue
        report["processes"][role] = classify_identity(stored, proc_info.lookup(pid))
    report["outcome"] = _outcome_from_receipt(state, receipt)
    return report


# --- Wrapper and CLI ------------------------------------------------------------


def _read_token(fd: int, limit: int = 64) -> bytes:
    """Read one line byte by byte so nothing after it is consumed from stdin."""
    buffer = bytearray()
    while len(buffer) < limit:
        byte = os.read(fd, 1)
        if not byte:
            break
        buffer += byte
        if byte == b"\n":
            break
    return bytes(buffer)


def wrapper_main(argv: list[str]) -> int:
    """Block until START, then exec the exact argv in this same PID."""
    if not argv:
        os.write(2, b"owned-benchmark wrapper: empty argv\n")
        return WRAPPER_NOT_RELEASED_EXIT
    if _read_token(0) != START_TOKEN:
        os.write(2, b"owned-benchmark wrapper: START not released; benchmark not started\n")
        return WRAPPER_NOT_RELEASED_EXIT
    try:
        os.execvpe(argv[0], argv, os.environ)
    except OSError as exc:
        os.write(2, f"owned-benchmark wrapper: exec failed: {exc}\n".encode("utf-8", "replace"))
    return WRAPPER_EXEC_FAILED_EXIT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owned_benchmark_supervisor.py",
        description="Supervise one owned benchmark run with recoverable receipts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="launch a benchmark; its exact argv follows --")
    run.add_argument("--output-dir", required=True, help="new directory to create; must not exist")
    run.add_argument("--cwd", required=True, help="absolute working directory for the benchmark")
    run.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument(
        "--quiet-request-file",
        default=None,
        help="absolute path to the parent request JSON (requestUTC, scope); required with --quiet-ready-file",
    )
    run.add_argument(
        "--quiet-ready-file",
        default=None,
        help='absolute path; GO requires {"ready": true}, readyUTC >= requestUTC and the same scope',
    )
    run.add_argument("--quiet-ack-wait-seconds", type=float, default=DEFAULT_QUIET_ACK_WAIT_SECONDS)
    run.add_argument("--grace-seconds", type=float, default=DEFAULT_GRACE_SECONDS)
    run.add_argument("--no-echo", action="store_true", help="do not copy benchmark stdout to this stdout")
    status = sub.add_parser("status", help="read-only identity and outcome report")
    status.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list[:1] == [WRAPPER_SUBCOMMAND]:
        rest = args_list[1:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        return wrapper_main(rest)
    if "--" in args_list:
        split = args_list.index("--")
        head, benchmark_argv = args_list[:split], args_list[split + 1 :]
    else:
        head, benchmark_argv = args_list, []
    args = build_parser().parse_args(head)

    if args.command == "status":
        try:
            report = collect_status(args.output_dir, default_proc_info())
        except (OSError, ValueError) as exc:
            print(f"status: cannot read state: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    config = RunConfig(
        output_dir=args.output_dir,
        cwd=args.cwd,
        argv=benchmark_argv,
        timeout_seconds=args.timeout_seconds,
        quiet_ready_file=args.quiet_ready_file,
        quiet_request_file=args.quiet_request_file,
        quiet_ack_wait_seconds=args.quiet_ack_wait_seconds,
        grace_seconds=args.grace_seconds,
        echo=not args.no_echo,
    )
    try:
        return Supervisor(config).run()
    except (ConfigError, OSError) as exc:
        print(f"run: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
