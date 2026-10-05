"""Unit tests for owned_benchmark_supervisor (stdlib unittest only).

Native process information, resource snapshots and benchmark streams are
injected fakes, so the suite runs on Linux and macOS without scanning
processes, signalling foreign PIDs, touching GPUs or downloading anything.
A few lifecycle tests spawn trivial Python children through the real wrapper.
macOS-only tests read the native identity of this test process and of one
child that this test owns.
"""

from __future__ import annotations

import ctypes
import errno
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import owned_benchmark_supervisor as obs  # noqa: E402

SUPERVISOR_PID = os.getpid()
CHILD_PID = 424242
UID = os.getuid()
LAUNCH_UUID = "00000000-0000-4000-8000-000000000001"
GO = obs.GO_TOKEN
START = obs.START_TOKEN

GOOD_SCRIPT = (
    ("line", "warming up fake benchmark"),
    ("line", "BENCHMARK_READY"),
    ("await", GO),
    ("line", "step 1 TPOT 21.5 ms"),
    ("line", "step 2 TPOT 22.0 ms"),
    ("line", "TIMED_WINDOW_END"),
    ("line", "summary steps=2"),
    ("exit", 0),
)


def identity(pid, ppid, *, uid=UID, sec=1_700_000_000, usec=123, comm="python3"):
    return obs.NativeIdentity(pid=pid, ppid=ppid, uid=uid, start_tvsec=sec, start_tvusec=usec, comm=comm)


def ok(ident):
    return obs.ProcLookup(obs.LOOKUP_OK, ident.pid, identity=ident)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def mode_of(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += max(seconds or 0.0, 0.001)

    def wall(self):
        return f"fake+{self.now:.3f}"


class FakeProcInfo:
    """Answers from a fixed table; unknown PIDs are reported absent (ESRCH)."""

    def __init__(self, events, table):
        self.events = events
        self.table = dict(table)
        self.calls = []

    def lookup(self, pid):
        self.events.append(("native_lookup", pid))
        self.calls.append(pid)
        result = self.table.get(pid)
        if result is None:
            return obs.ProcLookup(obs.LOOKUP_ABSENT, pid, errno=errno.ESRCH, detail="fake ESRCH")
        return result


class LiveFakeProcInfo:
    """Fake native info for real PIDs owned by this test; performs no process reads."""

    def lookup(self, pid):
        ppid = os.getppid() if pid == os.getpid() else os.getpid()
        return ok(identity(pid, ppid, sec=1_700_000_000 + pid % 1000))


class FakeResources:
    def __init__(self, events):
        self.events = events

    def snapshot(self, label):
        self.events.append(("snapshot", label))
        return {"label": label, "fake": True}


class FakeStdin:
    def __init__(self, child):
        self.child = child
        self.written = []
        self.closed = False

    def write(self, data):
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.child.events.append(("stdin", data))
        if self.child.on_stdin_write is not None:
            self.child.on_stdin_write(data)
        self.written.append(data)
        return len(data)

    def flush(self):
        pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.child.events.append(("stdin_close",))
        if START not in self.written and self.child.returncode is None:
            self.child.returncode = obs.WRAPPER_NOT_RELEASED_EXIT  # wrapper exits on EOF


class FakeChild:
    """Stands in for the Popen-owned wrapper/benchmark process."""

    def __init__(self, events, clock, *, pid=CHILD_PID, ignore_terminate=False):
        self.events = events
        self.clock = clock
        self.pid = pid
        self.ignore_terminate = ignore_terminate
        self.stdin = FakeStdin(self)
        self.stdout = None
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.on_stdin_write = None

    def sent(self, token):
        return token in self.stdin.written

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.clock.advance(timeout)
            raise subprocess.TimeoutExpired("fake-bench", timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.events.append(("terminate",))
        if not self.ignore_terminate:
            self.returncode = -int(signal.SIGTERM)

    def kill(self):
        self.kill_calls += 1
        raise AssertionError("SIGKILL must never be sent")


class FakeReader:
    """Scripted stdout: ("line", text), ("await", token), ("exit", code), ("hang", None), ("raise", exc)."""

    def __init__(self, events, child, clock, script):
        self.events = events
        self.child = child
        self.clock = clock
        self.script = list(script)

    def read_lines(self, timeout):
        while self.script:
            kind, value = self.script[0]
            if kind == "line":
                self.script.pop(0)
                self.clock.advance(0.01)
                self.events.append(("line", value))
                return [value], False
            if kind == "await":
                if self.child.sent(value):
                    self.script.pop(0)
                    continue
                self.clock.advance(timeout)
                return [], False
            if kind == "exit":
                self.script.pop(0)
                self.child.returncode = value
                return [], True
            if kind == "hang":
                self.clock.advance(timeout)
                return [], False
            if kind == "raise":
                self.script.pop(0)
                raise value
        return [], True


class SupervisorHarness(unittest.TestCase):
    """Builds a Supervisor wired to fakes; no real processes are involved."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = os.path.realpath(self._tmp.name)
        self.events = []
        self.clock = FakeClock()
        self.output_dir = os.path.join(self.tmp, "run-1")

    def build(
        self,
        script,
        *,
        quiet_file=None,
        quiet_request=None,
        quiet_wait=2.0,
        timeout=60.0,
        child_lookup=None,
        ignore_terminate=False,
    ):
        self.child = FakeChild(self.events, self.clock, ignore_terminate=ignore_terminate)
        self.reader = FakeReader(self.events, self.child, self.clock, script)
        table = {
            SUPERVISOR_PID: ok(identity(SUPERVISOR_PID, os.getppid(), sec=1_700_000_000)),
            CHILD_PID: child_lookup or ok(identity(CHILD_PID, SUPERVISOR_PID, sec=1_700_000_100)),
        }
        self.proc = FakeProcInfo(self.events, table)
        config = obs.RunConfig(
            output_dir=self.output_dir,
            cwd=self.tmp,
            argv=["fake-bench", "--steps", "3"],
            timeout_seconds=timeout,
            quiet_ready_file=quiet_file,
            quiet_request_file=quiet_request,
            quiet_ack_wait_seconds=quiet_wait,
            grace_seconds=1.0,
            echo=False,
        )

        def spawn(argv, cwd, env, stderr):
            self.events.append(("spawn", tuple(argv)))
            self.spawn_env = env
            return self.child

        return obs.Supervisor(
            config,
            proc_info=self.proc,
            resources=FakeResources(self.events),
            spawn=spawn,
            reader_factory=lambda child: self.reader,
            monotonic=self.clock.monotonic,
            wall=self.clock.wall,
            install_signal_handlers=False,
            launch_uuid=LAUNCH_UUID,
        )

    def run_supervisor(self, supervisor):
        with mock.patch.object(obs.os, "kill", side_effect=AssertionError("os.kill must not be used")):
            code = supervisor.run()
        return code, self.read(obs.RECEIPT_FILE), self.read(obs.STATE_FILE)

    def read(self, name):
        path = os.path.join(self.output_dir, name)
        return read_json(path) if os.path.exists(path) else None

    def kinds(self, events):
        return [event[0] for event in events]


class TestNativeLayout(unittest.TestCase):
    def test_struct_matches_sdk_header_layout(self):
        expected_offsets = {
            "pbi_flags": 0,
            "pbi_pid": 12,
            "pbi_ppid": 16,
            "pbi_uid": 20,
            "rfu_1": 44,
            "pbi_comm": 48,
            "pbi_name": 64,
            "pbi_nfiles": 96,
            "pbi_nice": 116,
            "pbi_start_tvsec": 120,
            "pbi_start_tvusec": 128,
        }
        for name, offset in expected_offsets.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(obs.ProcBsdInfo, name).offset, offset)
        self.assertEqual(obs.ProcBsdInfo.pbi_comm.size, 16)
        self.assertEqual(obs.ProcBsdInfo.pbi_name.size, 32)
        self.assertEqual(ctypes.sizeof(obs.ProcBsdInfo), 136)
        self.assertEqual(obs.PROC_BSDINFO_SIZE, 136)
        self.assertEqual(obs.PROC_PIDTBSDINFO, 3)

    def test_full_read_decodes_start_identity(self):
        info = obs.ProcBsdInfo()
        info.pbi_pid = 4321
        info.pbi_ppid = 99
        info.pbi_uid = 501
        info.pbi_start_tvsec = 1_700_000_123
        info.pbi_start_tvusec = 456_789
        info.pbi_comm = b"bench"
        lookup = obs.interpret_proc_pidinfo(4321, ctypes.sizeof(info), 0, info)
        self.assertEqual(lookup.status, obs.LOOKUP_OK)
        self.assertEqual(lookup.identity, obs.NativeIdentity(4321, 99, 501, 1_700_000_123, 456_789, "bench"))

    def test_errors_map_to_explicit_statuses_without_identity(self):
        info = obs.ProcBsdInfo()
        cases = [
            (0, errno.ESRCH, obs.LOOKUP_ABSENT),
            (0, errno.EPERM, obs.LOOKUP_DENIED),
            (0, errno.EINVAL, obs.LOOKUP_ERROR),
            (40, 0, obs.LOOKUP_ERROR),
        ]
        for returned, err, expected in cases:
            with self.subTest(returned=returned, err=err):
                lookup = obs.interpret_proc_pidinfo(77, returned, err, info)
                self.assertEqual(lookup.status, expected)
                self.assertIsNone(lookup.identity)
        info.pbi_pid = 78
        mismatch = obs.interpret_proc_pidinfo(77, ctypes.sizeof(info), 0, info)
        self.assertEqual(mismatch.status, obs.LOOKUP_ERROR)
        self.assertIsNone(mismatch.identity)

    def test_libproc_lookup_uses_exact_pid_flavor_and_buffer_size(self):
        source = obs.ProcBsdInfo()
        source.pbi_pid = 321
        source.pbi_ppid = SUPERVISOR_PID
        source.pbi_uid = UID
        source.pbi_start_tvsec = 10
        source.pbi_start_tvusec = 20
        source.pbi_comm = b"wrapper"
        calls = []
        loaded = []

        def proc_pidinfo(pid, flavor, arg, buffer, size):
            calls.append((pid, flavor, arg, size))
            if pid != 321:
                ctypes.set_errno(errno.ESRCH)
                return 0
            ctypes.memmove(buffer, ctypes.addressof(source), size)
            return size

        def loader(path, use_errno):
            loaded.append((path, use_errno))
            return types.SimpleNamespace(proc_pidinfo=proc_pidinfo)

        provider = obs.LibprocInfo(loader=loader)
        found = provider.lookup(321)
        missing = provider.lookup(999)
        self.assertEqual(loaded, [("/usr/lib/libproc.dylib", True)])
        self.assertEqual(calls, [(321, 3, 0, 136), (999, 3, 0, 136)])
        self.assertEqual(found.status, obs.LOOKUP_OK)
        self.assertEqual(found.identity, identity(321, SUPERVISOR_PID, sec=10, usec=20, comm="wrapper"))
        self.assertEqual(missing.status, obs.LOOKUP_ABSENT)
        self.assertEqual(missing.errno, errno.ESRCH)

    @unittest.skipUnless(sys.platform == "darwin", "macOS libproc only")
    def test_native_lookup_of_own_process(self):
        # libproc lives in the dyld shared cache, so its path need not exist on disk.
        try:
            provider = obs.LibprocInfo()
        except (OSError, AttributeError) as exc:
            self.skipTest(f"libproc could not be loaded: {exc}")
        first = provider.lookup(os.getpid())
        second = provider.lookup(os.getpid())
        self.assertEqual(first.status, obs.LOOKUP_OK)
        self.assertEqual(first.identity.pid, os.getpid())
        self.assertEqual(first.identity.ppid, os.getppid())
        self.assertEqual(first.identity.uid, os.getuid())
        self.assertGreater(first.identity.start_tvsec, 0)
        self.assertEqual(
            (first.identity.start_tvsec, first.identity.start_tvusec),
            (second.identity.start_tvsec, second.identity.start_tvusec),
        )

    @unittest.skipUnless(sys.platform == "darwin", "macOS libproc only")
    def test_native_start_identity_survives_wrapper_exec(self):
        try:
            provider = obs.LibprocInfo()
        except (OSError, AttributeError) as exc:
            self.skipTest(f"libproc could not be loaded: {exc}")
        code = "import sys\nprint('BENCHMARK_READY', flush=True)\nsys.stdin.readline()\n"
        seen = []
        with tempfile.TemporaryDirectory() as tmp, open(os.devnull, "wb") as devnull:
            child = obs.popen_spawn(obs.wrapper_argv([sys.executable, "-c", code]), tmp, dict(os.environ), devnull)
            try:
                before = provider.lookup(child.pid)  # wrapper, before START
                child.stdin.write(START)
                child.stdin.flush()
                reader = obs.SelectorLineReader(child.stdout)
                deadline = time.monotonic() + 30
                while "BENCHMARK_READY" not in seen and time.monotonic() < deadline:
                    lines, eof = reader.read_lines(deadline - time.monotonic())
                    seen.extend(lines)
                    if eof:
                        break
                after = provider.lookup(child.pid)  # exec'd benchmark, after READY
                child.stdin.write(GO)
                child.stdin.close()
                self.assertEqual(child.wait(timeout=30), 0)
            finally:
                if child.poll() is None:
                    child.stdin.close()  # the wrapper or benchmark exits on stdin EOF
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                child.stdout.close()
        self.assertIn("BENCHMARK_READY", seen)
        for lookup in (before, after):
            self.assertEqual(lookup.status, obs.LOOKUP_OK, lookup)
            self.assertEqual(lookup.identity.pid, child.pid)
            self.assertEqual(lookup.identity.ppid, os.getpid())
            self.assertEqual(lookup.identity.uid, os.getuid())
        self.assertEqual(
            (before.identity.start_tvsec, before.identity.start_tvusec),
            (after.identity.start_tvsec, after.identity.start_tvusec),
        )


class TestIdentityClassification(unittest.TestCase):
    stored = identity(CHILD_PID, SUPERVISOR_PID, sec=1_700_000_100).to_dict()

    def classify(self, lookup):
        return obs.classify_identity(self.stored, lookup)

    def test_matching_uid_start_and_ppid_is_same_even_after_exec(self):
        result = self.classify(ok(identity(CHILD_PID, SUPERVISOR_PID, sec=1_700_000_100, comm="bench")))
        self.assertEqual(result["classification"], obs.IDENTITY_SAME)

    def test_reused_pid_is_distinct(self):
        candidates = [
            identity(CHILD_PID, SUPERVISOR_PID, sec=1_700_000_101),
            identity(CHILD_PID, SUPERVISOR_PID, sec=1_700_000_100, usec=124),
            identity(CHILD_PID, SUPERVISOR_PID, uid=UID + 1, sec=1_700_000_100),
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(self.classify(ok(candidate))["classification"], obs.IDENTITY_DISTINCT)

    def test_ppid_change_is_unconfirmed_not_same(self):
        result = self.classify(ok(identity(CHILD_PID, 1, sec=1_700_000_100)))
        self.assertEqual(result["classification"], obs.IDENTITY_UNCONFIRMED)

    def test_absent_pid_is_not_exit_evidence(self):
        result = self.classify(obs.ProcLookup(obs.LOOKUP_ABSENT, CHILD_PID, errno=errno.ESRCH))
        self.assertEqual(result["classification"], obs.IDENTITY_ABSENT)
        self.assertIn("terminal receipt", result["note"])
        self.assertEqual(result["evidence"]["errnoName"], "ESRCH")

    def test_denied_and_unavailable_are_unknown(self):
        for lookup in (
            obs.ProcLookup(obs.LOOKUP_DENIED, CHILD_PID, errno=errno.EPERM),
            obs.ProcLookup(obs.LOOKUP_UNAVAILABLE, CHILD_PID, detail="not macOS"),
            obs.ProcLookup(obs.LOOKUP_ERROR, CHILD_PID, errno=errno.EINVAL),
        ):
            with self.subTest(status=lookup.status):
                self.assertEqual(self.classify(lookup)["classification"], obs.IDENTITY_UNKNOWN)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_output_dir_is_new_and_private(self):
        path = obs.create_output_dir(os.path.join(self.tmp, "out"))
        self.assertEqual(mode_of(path), 0o700)
        with self.assertRaises(FileExistsError):
            obs.create_output_dir(path)
        with self.assertRaises(FileNotFoundError):
            obs.create_output_dir(os.path.join(self.tmp, "missing", "out"))

    def test_atomic_json_write_is_private_and_leaves_no_temp_files(self):
        path = os.path.join(self.tmp, "state.json")
        obs.write_json_atomic(path, {"phase": "a"})
        obs.write_json_atomic(path, {"phase": "b"}, durable=False)
        self.assertEqual(read_json(path), {"phase": "b"})
        self.assertEqual(mode_of(path), 0o600)
        self.assertEqual(os.listdir(self.tmp), ["state.json"])

    def test_no_overwrite_mode_preserves_existing_file(self):
        path = os.path.join(self.tmp, "receipt.json")
        obs.write_json_atomic(path, {"result": "first"}, overwrite=False)
        with self.assertRaises(FileExistsError):
            obs.write_json_atomic(path, {"result": "second"}, overwrite=False)
        self.assertEqual(read_json(path), {"result": "first"})
        self.assertEqual(os.listdir(self.tmp), ["receipt.json"])

    def test_quiet_ack_requires_literal_true_and_preserves_parent_fields(self):
        path = os.path.join(self.tmp, "quiet.json")
        self.assertEqual(obs.read_quiet_ack(path, "t")["status"], "absent")
        cases = [
            ('{"ready": true, "readyUTC": "2026-10-05T12:00:00Z", "source": "parent"}', "ready"),
            ('{"ready": "true"}', "not_ready"),
            ('{"ready": false}', "not_ready"),
            ("[true]", "invalid"),
            ('{"ready": tr', "invalid"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                self.assertEqual(obs.read_quiet_ack(path, "t")["status"], expected)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(cases[0][0])
        ack = obs.read_quiet_ack(path, "t")
        self.assertEqual((ack["readyUTC"], ack["source"]), ("2026-10-05T12:00:00Z", "parent"))


class TestLaunchHandshake(SupervisorHarness):
    def test_identity_receipt_is_durable_before_start_is_released(self):
        supervisor = self.build(GOOD_SCRIPT)
        captured = {}

        def on_write(data):
            if data == START:
                captured["state"] = read_json(os.path.join(self.output_dir, obs.STATE_FILE))

        self.child.on_stdin_write = on_write
        with mock.patch.object(obs.os, "fsync", side_effect=lambda fd: self.events.append(("fsync", fd))):
            code, receipt, _ = self.run_supervisor(supervisor)

        self.assertEqual(code, 0)
        state = captured["state"]
        self.assertEqual(state["phase"], "identity_recorded")
        self.assertEqual(state["launchUUID"], LAUNCH_UUID)
        self.assertEqual(state["benchmark"], {"argv": ["fake-bench", "--steps", "3"], "cwd": self.tmp})
        self.assertEqual(state["child"]["pid"], CHILD_PID)
        self.assertEqual(state["child"]["nativeIdentity"]["start_tvsec"], 1_700_000_100)
        self.assertEqual(state["child"]["nativeIdentity"]["ppid"], SUPERVISOR_PID)
        self.assertEqual(state["supervisor"]["nativeIdentity"]["pid"], SUPERVISOR_PID)

        start_index = self.events.index(("stdin", START))
        child_lookup_index = self.events.index(("native_lookup", CHILD_PID))
        fsyncs = [i for i, event in enumerate(self.events) if event[0] == "fsync"]
        self.assertTrue(any(child_lookup_index < i < start_index for i in fsyncs))

        spawn_index = self.kinds(self.events).index("spawn")
        spawn_argv = self.events[spawn_index][1]
        self.assertLess(self.events.index(("native_lookup", SUPERVISOR_PID)), spawn_index)
        self.assertIn(obs.WRAPPER_SUBCOMMAND, spawn_argv)
        self.assertEqual(list(spawn_argv[-4:]), ["--", "fake-bench", "--steps", "3"])
        self.assertEqual(self.spawn_env[obs.LAUNCH_UUID_ENV], LAUNCH_UUID)
        self.assertEqual(self.proc.calls, [SUPERVISOR_PID, CHILD_PID])
        self.assertEqual(receipt["launchUUID"], LAUNCH_UUID)

    def test_unverified_child_identity_withholds_start(self):
        denied = obs.ProcLookup(obs.LOOKUP_DENIED, CHILD_PID, errno=errno.EPERM)
        supervisor = self.build(GOOD_SCRIPT, child_lookup=denied)
        code, receipt, state = self.run_supervisor(supervisor)

        self.assertEqual(code, 1)
        self.assertNotIn(("stdin", START), self.events)
        self.assertIn(("stdin_close",), self.events)
        self.assertNotIn("line", self.kinds(self.events))
        self.assertEqual(self.child.terminate_calls, 0)
        self.assertEqual(receipt["result"], "aborted_native_identity_unverified")
        self.assertEqual(receipt["cleanupAction"], "closed_stdin_before_start")
        self.assertEqual(receipt["exitCode"], obs.WRAPPER_NOT_RELEASED_EXIT)
        self.assertFalse(receipt["startReleased"])
        self.assertTrue(receipt["lockReleaseEligible"])
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, obs.LOCK_FILE)))
        self.assertTrue(state["cleanupResult"]["lockReleased"])
        self.assertIsNone(state["child"]["nativeIdentity"])
        self.assertTrue(state["identityProblems"])

    def test_child_with_foreign_parent_withholds_start(self):
        foreign = ok(identity(CHILD_PID, 1, sec=1_700_000_100))
        supervisor = self.build(GOOD_SCRIPT, child_lookup=foreign)
        _, receipt, _ = self.run_supervisor(supervisor)
        self.assertNotIn(("stdin", START), self.events)
        self.assertEqual(receipt["result"], "aborted_native_identity_unverified")

    def test_existing_output_directory_is_refused_untouched(self):
        supervisor = self.build(GOOD_SCRIPT)
        os.mkdir(self.output_dir)
        with open(os.path.join(self.output_dir, "keep.txt"), "w", encoding="utf-8") as handle:
            handle.write("keep")
        with self.assertRaises(FileExistsError):
            supervisor.run()
        self.assertEqual(os.listdir(self.output_dir), ["keep.txt"])
        self.assertEqual(self.events, [])


class TestWindowOrdering(SupervisorHarness):
    def test_snapshots_bracket_window_and_nothing_samples_inside(self):
        code, receipt, _ = self.run_supervisor(self.build(GOOD_SCRIPT))

        self.assertEqual(code, 0)
        pre = self.events.index(("snapshot", "pre_go"))
        go = self.events.index(("stdin", GO))
        end = self.events.index(("line", "TIMED_WINDOW_END"))
        post = self.events.index(("snapshot", "post_end"))
        self.assertLess(pre, go)
        self.assertLess(go, end)
        self.assertLess(end, post)
        inside = self.kinds(self.events[go + 1 : end])
        self.assertNotIn("native_lookup", inside)
        self.assertNotIn("snapshot", inside)
        self.assertEqual(self.kinds(self.events).count("snapshot"), 2)

        self.assertEqual(receipt["result"], "success")
        self.assertEqual(receipt["exitCode"], 0)
        self.assertEqual(receipt["exitEvidenceSource"], "supervisor_popen_wait_status")
        self.assertTrue(receipt["endMarkerObserved"])
        self.assertGreater(receipt["windowElapsedSeconds"], 0)
        self.assertEqual(receipt["resources"]["preGo"]["label"], "pre_go")
        self.assertEqual(receipt["resources"]["postWindow"]["label"], "post_end")

    def test_every_line_is_logged_and_progress_is_persisted(self):
        self.run_supervisor(self.build(GOOD_SCRIPT))
        lines = [value for kind, value in GOOD_SCRIPT if kind == "line"]

        with open(os.path.join(self.output_dir, obs.STDOUT_LOG), encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(), lines)
        with open(os.path.join(self.output_dir, obs.PROGRESS_FILE), encoding="utf-8") as handle:
            records = [json.loads(row) for row in handle]
        self.assertEqual([record["line"] for record in records], lines)
        self.assertEqual([record["seq"] for record in records], list(range(1, len(lines) + 1)))
        self.assertEqual([record["marker"] for record in records if record["marker"]], ["ready", "end"])
        self.assertEqual(sum(record["tpot"] for record in records), 2)

        state = self.read(obs.STATE_FILE)
        self.assertEqual(state["phase"], "finished")
        self.assertEqual(state["progress"]["lineCount"], len(lines))
        self.assertEqual([entry["line"] for entry in state["progress"]["tpotLines"]], lines[2:4])

        self.assertEqual(mode_of(self.output_dir), 0o700)
        for name in (obs.STATE_FILE, obs.RECEIPT_FILE, obs.STDOUT_LOG, obs.STDERR_LOG, obs.PROGRESS_FILE):
            with self.subTest(name=name):
                self.assertEqual(mode_of(os.path.join(self.output_dir, name)), 0o600)
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, obs.LOCK_FILE)))


QUIET_REQUEST_UTC = "2026-10-05T12:00:00Z"
QUIET_SCOPE = "phase53-gpu-window"


def fresh_request(**overrides):
    payload = {"requestUTC": QUIET_REQUEST_UTC, "scope": QUIET_SCOPE, "source": "root-request"}
    payload.update(overrides)
    return payload


def fresh_ready(**overrides):
    payload = {
        "ready": True,
        "readyUTC": "2026-10-05T12:00:05Z",
        "scope": QUIET_SCOPE,
        "source": "parent-quiet-coordinator",
    }
    payload.update(overrides)
    return payload


def write_payload(path, payload):
    """Write a dict as JSON, a str verbatim, or ensure absence for None."""
    if payload is None:
        if os.path.exists(path):
            os.unlink(path)
        return
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)


class TestQuietPair(unittest.TestCase):
    """Pure checks of the request/ready pairing rules through real file reads."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.request_path = os.path.join(self._tmp.name, "request.json")
        self.ready_path = os.path.join(self._tmp.name, "ready.json")

    def evaluate(self, request, ready):
        write_payload(self.request_path, request)
        write_payload(self.ready_path, ready)
        return obs.evaluate_quiet_pair(
            obs.read_quiet_request(self.request_path, "t"), obs.read_quiet_ack(self.ready_path, "t")
        )[0]

    def test_pairing_rules(self):
        cases = [
            ("fresh", fresh_request(), fresh_ready(), "ready"),
            ("equal_times", fresh_request(), fresh_ready(readyUTC=QUIET_REQUEST_UTC), "ready"),
            ("explicit_zero_offset", fresh_request(), fresh_ready(readyUTC="2026-10-05T12:00:05+00:00"), "ready"),
            ("stale_phase52", fresh_request(), fresh_ready(readyUTC="2026-10-04T08:00:00Z"), "stale"),
            ("scope_mismatch", fresh_request(), fresh_ready(scope="phase52-gpu-window"), "scope_mismatch"),
            ("ready_scope_missing", fresh_request(), fresh_ready(scope=None), "scope_mismatch"),
            ("ready_naive", fresh_request(), fresh_ready(readyUTC="2026-10-05T12:00:05"), "ready_malformed_utc"),
            ("ready_garbage", fresh_request(), fresh_ready(readyUTC="yesterday"), "ready_malformed_utc"),
            ("ready_non_utc", fresh_request(), fresh_ready(readyUTC="2026-10-05T21:00:05+09:00"), "ready_malformed_utc"),
            ("ready_missing_time", fresh_request(), fresh_ready(readyUTC=None), "ready_malformed_utc"),
            ("request_naive", fresh_request(requestUTC="2026-10-05T12:00:00"), fresh_ready(), "request_malformed_utc"),
            ("request_numeric", fresh_request(requestUTC=1_790_000_000), fresh_ready(), "request_malformed_utc"),
            ("request_empty_scope", fresh_request(scope=""), fresh_ready(scope=""), "request_invalid_scope"),
            ("request_invalid_json", "{", fresh_ready(), "request_invalid"),
            ("request_absent", None, fresh_ready(), "request_absent"),
            ("ready_absent", fresh_request(), None, "ready_absent"),
            ("ready_invalid_json", fresh_request(), "[true]", "ready_invalid"),
            ("ready_string_true", fresh_request(), fresh_ready(ready="true"), "not_ready"),
        ]
        for name, request, ready, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(self.evaluate(request, ready), expected)


class TestQuietAck(SupervisorHarness):
    def run_pair(self, *, request, ready):
        request_path = os.path.join(self.tmp, "quiet-request.json")
        ready_path = os.path.join(self.tmp, "quiet-ready.json")
        write_payload(request_path, request)
        write_payload(ready_path, ready)
        return self.run_supervisor(self.build(GOOD_SCRIPT, quiet_file=ready_path, quiet_request=request_path))

    def assert_refused(self, receipt, reason):
        self.assertNotIn(("stdin", GO), self.events)
        self.assertNotIn("snapshot", self.kinds(self.events))
        self.assertEqual(self.child.terminate_calls, 1)
        self.assertEqual(receipt["result"], "aborted_quiet_ack_" + reason)
        self.assertFalse(receipt["goSent"])
        self.assertEqual(receipt["quietAck"]["status"], reason)

    def test_absent_ready_file_never_sends_go(self):
        code, receipt, _ = self.run_pair(request=fresh_request(), ready=None)
        self.assertEqual(code, 1)
        self.assert_refused(receipt, "ready_absent")
        self.assertEqual(receipt["exitSignal"], int(signal.SIGTERM))
        self.assertEqual(receipt["resources"]["postWindowSkippedReason"], "go_not_sent")

    def test_absent_request_file_never_sends_go(self):
        _, receipt, _ = self.run_pair(request=None, ready=fresh_ready())
        self.assert_refused(receipt, "request_absent")

    def test_not_ready_ack_never_sends_go(self):
        _, receipt, _ = self.run_pair(request=fresh_request(), ready=fresh_ready(ready=False))
        self.assert_refused(receipt, "not_ready")

    def test_stale_ready_from_earlier_phase_never_sends_go(self):
        stale = fresh_ready(readyUTC="2026-10-04T08:00:00Z", source="phase52-coordinator")
        _, receipt, state = self.run_pair(request=fresh_request(), ready=stale)
        self.assert_refused(receipt, "stale")
        self.assertEqual(state["quietAck"]["ready"]["readyUTC"], "2026-10-04T08:00:00Z")
        self.assertEqual(state["quietAck"]["ready"]["source"], "phase52-coordinator")
        self.assertEqual(state["quietAck"]["request"]["requestUTC"], QUIET_REQUEST_UTC)

    def test_scope_mismatch_never_sends_go(self):
        _, receipt, _ = self.run_pair(request=fresh_request(), ready=fresh_ready(scope="phase52-gpu-window"))
        self.assert_refused(receipt, "scope_mismatch")

    def test_malformed_ready_time_never_sends_go(self):
        _, receipt, _ = self.run_pair(request=fresh_request(), ready=fresh_ready(readyUTC="2026-10-05T12:00:05"))
        self.assert_refused(receipt, "ready_malformed_utc")

    def test_fresh_paired_ack_is_preserved_and_go_follows(self):
        request = fresh_request()
        ready = fresh_ready()
        code, receipt, state = self.run_pair(request=request, ready=ready)

        self.assertEqual(code, 0)
        self.assertIn(("stdin", GO), self.events)
        self.assertEqual(state["quietRequestFile"], os.path.join(self.tmp, "quiet-request.json"))
        for record in (receipt["quietAck"], state["quietAck"]):
            self.assertEqual(record["status"], "ready")
            self.assertEqual(record["request"]["requestUTC"], request["requestUTC"])
            self.assertEqual(record["request"]["source"], request["source"])
            self.assertEqual(record["request"]["scope"], QUIET_SCOPE)
            self.assertEqual(record["ready"]["readyUTC"], ready["readyUTC"])
            self.assertEqual(record["ready"]["source"], ready["source"])
            self.assertEqual(record["ready"]["scope"], QUIET_SCOPE)


class TestCli(unittest.TestCase):
    def test_quiet_files_must_be_paired_and_absolute(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = os.path.realpath(raw_tmp)
            ready = os.path.join(tmp, "ready.json")
            request = os.path.join(tmp, "request.json")
            flag_sets = [
                ["--quiet-ready-file", ready],
                ["--quiet-request-file", request],
                ["--quiet-request-file", "request.json", "--quiet-ready-file", ready],
            ]
            for flags in flag_sets:
                with self.subTest(flags=flags):
                    output_dir = os.path.join(tmp, "run")
                    stderr = io.StringIO()
                    with mock.patch.object(sys, "stderr", stderr):
                        code = obs.main(["run", "--output-dir", output_dir, "--cwd", tmp, *flags, "--", "true"])
                    self.assertEqual(code, 2)
                    self.assertIn("refused", stderr.getvalue())
                    self.assertFalse(os.path.exists(output_dir))


class TestTerminalReceipts(SupervisorHarness):
    def test_exit_zero_without_end_marker_is_not_success(self):
        script = [("line", "BENCHMARK_READY"), ("await", GO), ("line", "TPOT 30 ms"), ("exit", 0)]
        code, receipt, _ = self.run_supervisor(self.build(script))

        self.assertEqual(code, 1)
        self.assertEqual(receipt["result"], "exited_zero_without_end_marker")
        self.assertEqual(receipt["exitCode"], 0)
        self.assertFalse(receipt["endMarkerObserved"])
        self.assertIsNone(receipt["windowElapsedSeconds"])
        self.assertIn(("snapshot", "post_exit"), self.events)
        self.assertNotIn(("snapshot", "post_end"), self.events)

    def test_nonzero_exit_before_ready(self):
        code, receipt, _ = self.run_supervisor(self.build([("line", "fatal: bad config"), ("exit", 3)]))
        self.assertEqual(code, 1)
        self.assertEqual(receipt["result"], "exited_nonzero")
        self.assertEqual(receipt["exitCode"], 3)
        self.assertFalse(receipt["goSent"])
        self.assertNotIn("snapshot", self.kinds(self.events))
        self.assertEqual(self.child.terminate_calls, 0)

    def test_timeout_with_stubborn_child_reports_unknown_and_keeps_lock(self):
        script = [("line", "BENCHMARK_READY"), ("hang", None)]
        code, receipt, _ = self.run_supervisor(self.build(script, timeout=5.0, ignore_terminate=True))

        self.assertEqual(code, 1)
        self.assertEqual(self.child.terminate_calls, 1)
        self.assertEqual(self.child.kill_calls, 0)
        self.assertEqual(receipt["result"], "child_status_unknown")
        self.assertEqual(receipt["childStatus"], "unknown_still_running")
        self.assertTrue(receipt["timedOut"])
        self.assertIsNone(receipt["exitCode"])
        self.assertIsNone(receipt["exitSignal"])
        self.assertIsNone(receipt["exitEvidenceSource"])
        self.assertFalse(receipt["lockReleaseEligible"])
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, obs.LOCK_FILE)))
        self.assertNotIn(("snapshot", "post_exit"), self.events)
        self.assertEqual(receipt["resources"]["postWindowSkippedReason"], "child_status_unknown_window_may_be_open")

    def test_timeout_with_cooperative_child_records_sigterm_exit(self):
        script = [("line", "BENCHMARK_READY"), ("hang", None)]
        _, receipt, _ = self.run_supervisor(self.build(script, timeout=5.0))
        self.assertEqual(receipt["result"], "timeout")
        self.assertEqual(receipt["exitSignal"], int(signal.SIGTERM))
        self.assertEqual(receipt["cleanupAction"], "sigterm")
        self.assertTrue(receipt["lockReleaseEligible"])
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, obs.LOCK_FILE)))

    def test_supervisor_sigterm_cleans_up_only_own_child(self):
        script = [
            ("line", "BENCHMARK_READY"),
            ("await", GO),
            ("line", "TPOT 25 ms"),
            ("raise", obs.SupervisorInterrupted(signal.SIGTERM)),
        ]
        code, receipt, _ = self.run_supervisor(self.build(script))
        self.assertEqual(code, 128 + int(signal.SIGTERM))
        self.assertEqual(receipt["result"], "interrupted")
        self.assertEqual(receipt["interruptedBy"], "SIGTERM")
        self.assertEqual(self.child.terminate_calls, 1)
        self.assertEqual(self.child.kill_calls, 0)

    def test_unexpected_error_still_writes_receipt(self):
        script = [("line", "BENCHMARK_READY"), ("raise", RuntimeError("reader exploded"))]
        code, receipt, state = self.run_supervisor(self.build(script))
        self.assertEqual(code, 1)
        self.assertEqual(receipt["result"], "error")
        self.assertEqual(receipt["error"], {"type": "RuntimeError", "message": "reader exploded"})
        self.assertEqual(self.child.terminate_calls, 1)
        self.assertTrue(state["receiptWritten"])

    def test_lock_is_released_only_after_receipt_exists(self):
        receipt_path = os.path.join(self.output_dir, obs.RECEIPT_FILE)
        real_unlink = os.unlink
        observed = []

        def unlink(path, *args, **kwargs):
            if os.path.basename(path) == obs.LOCK_FILE:
                observed.append(os.path.exists(receipt_path))
            return real_unlink(path, *args, **kwargs)

        supervisor = self.build(GOOD_SCRIPT)
        with mock.patch.object(obs.os, "unlink", side_effect=unlink):
            code, receipt, state = self.run_supervisor(supervisor)

        self.assertEqual(code, 0)
        self.assertEqual(observed, [True])
        self.assertTrue(receipt["lockReleaseEligible"])
        self.assertNotIn("lockReleased", receipt)
        self.assertEqual(
            state["cleanupResult"],
            {"lockReleaseEligible": True, "lockReleased": True, "lockReleaseError": None},
        )
        self.assertEqual(state["phase"], "finished")

    def test_failed_receipt_write_keeps_lock_and_exits_nonzero(self):
        real_write = obs.write_json_atomic

        def write(path, payload, **kwargs):
            if os.path.basename(path) == obs.RECEIPT_FILE:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write(path, payload, **kwargs)

        supervisor = self.build(GOOD_SCRIPT)
        with mock.patch.object(obs, "write_json_atomic", side_effect=write), mock.patch.object(
            sys, "stderr", io.StringIO()
        ) as stderr:
            code, receipt, state = self.run_supervisor(supervisor)

        self.assertEqual(code, 1)
        self.assertIsNone(receipt)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, obs.LOCK_FILE)))
        self.assertFalse(state["receiptWritten"])
        self.assertEqual(state["phase"], "receipt_write_failed")
        self.assertIn("No space left on device", state["receiptError"])
        self.assertNotIn("cleanupResult", state)
        self.assertEqual(self.child.terminate_calls, 0)
        self.assertIn("lock kept", stderr.getvalue())

    def test_go_to_exited_child_is_an_explicit_abort(self):
        supervisor = self.build(GOOD_SCRIPT)

        def on_write(data):
            if data == GO:
                self.child.returncode = 1
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")

        self.child.on_stdin_write = on_write
        code, receipt, _ = self.run_supervisor(supervisor)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["result"], "aborted_go_pipe_closed")
        self.assertEqual(receipt["exitCode"], 1)
        self.assertFalse(receipt["goSent"])
        self.assertEqual(self.child.terminate_calls, 0)

    def test_signal_during_error_handling_is_recorded_not_raised(self):
        supervisor = self.build([("line", "BENCHMARK_READY"), ("raise", RuntimeError("reader exploded"))])
        real_set_phase = supervisor._set_phase

        def set_phase(phase, **kwargs):
            if phase == "error":
                supervisor._on_signal(signal.SIGINT, None)  # simulated delivery inside the except branch
            return real_set_phase(phase, **kwargs)

        supervisor._set_phase = set_phase
        code, receipt, _ = self.run_supervisor(supervisor)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["result"], "error")
        self.assertEqual(receipt["signalsReceived"], ["SIGINT"])
        self.assertIsNone(receipt["interruptedBy"])
        self.assertEqual(self.child.terminate_calls, 1)


class TestStatus(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = obs.create_output_dir(os.path.join(self._tmp.name, "run"))

    def write_state(self, *, child_identity=True):
        child = identity(CHILD_PID, SUPERVISOR_PID, sec=200).to_dict() if child_identity else None
        state = {
            "schemaVersion": 1,
            "launchUUID": LAUNCH_UUID,
            "phase": "timed_window",
            "phaseUTC": "2026-10-05T12:00:00Z",
            "benchmark": {"argv": ["fake-bench"], "cwd": "/tmp"},
            "progress": {"lineCount": 4, "lastLineUTC": "2026-10-05T12:00:01Z"},
            "supervisor": {"pid": SUPERVISOR_PID, "nativeIdentity": identity(SUPERVISOR_PID, 1, sec=100).to_dict()},
            "child": {"pid": CHILD_PID, "nativeIdentity": child},
        }
        obs.write_json_atomic(os.path.join(self.out, obs.STATE_FILE), state)

    def collect(self, table):
        proc = FakeProcInfo([], table)
        with mock.patch.object(obs.os, "kill", side_effect=AssertionError("no signals")), mock.patch.object(
            obs.subprocess, "Popen", side_effect=AssertionError("no subprocesses")
        ):
            report = obs.collect_status(self.out, proc)
        return report, proc

    def test_reused_and_absent_pids_are_reported_without_outcome(self):
        self.write_state()
        report, proc = self.collect({CHILD_PID: ok(identity(CHILD_PID, SUPERVISOR_PID, sec=999))})
        self.assertEqual(proc.calls, [SUPERVISOR_PID, CHILD_PID])
        self.assertEqual(report["processes"]["child"]["classification"], obs.IDENTITY_DISTINCT)
        self.assertEqual(report["processes"]["supervisor"]["classification"], obs.IDENTITY_ABSENT)
        self.assertEqual(report["outcome"]["result"], "unknown")
        self.assertEqual(report["outcome"]["evidenceSource"], "none")
        self.assertIn("absent PID", report["outcome"]["note"])
        self.assertEqual(report["lineCount"], 4)

    def test_denied_read_is_unknown_and_matching_identity_is_same(self):
        self.write_state()
        report, _ = self.collect(
            {
                SUPERVISOR_PID: ok(identity(SUPERVISOR_PID, 1, sec=100)),
                CHILD_PID: obs.ProcLookup(obs.LOOKUP_DENIED, CHILD_PID, errno=errno.EPERM),
            }
        )
        self.assertEqual(report["processes"]["supervisor"]["classification"], obs.IDENTITY_SAME)
        child = report["processes"]["child"]
        self.assertEqual(child["classification"], obs.IDENTITY_UNKNOWN)
        self.assertEqual(child["evidence"]["errnoName"], "EPERM")

    def test_pid_without_persisted_identity_is_not_inspected(self):
        self.write_state(child_identity=False)
        report, proc = self.collect({})
        self.assertEqual(proc.calls, [SUPERVISOR_PID])
        self.assertEqual(report["processes"]["child"]["classification"], obs.IDENTITY_UNKNOWN)

    def test_outcome_comes_only_from_matching_receipt(self):
        self.write_state()
        receipt_path = os.path.join(self.out, obs.RECEIPT_FILE)
        obs.write_json_atomic(receipt_path, {"launchUUID": "someone-else", "result": "success"})
        report, _ = self.collect({})
        self.assertEqual(report["outcome"]["result"], "unknown")

        obs.write_json_atomic(receipt_path, {"launchUUID": LAUNCH_UUID, "result": "success", "exitCode": 0})
        report, _ = self.collect({})
        self.assertEqual(report["outcome"]["evidenceSource"], "terminal_receipt")
        self.assertEqual(report["outcome"]["result"], "success")
        self.assertEqual(report["outcome"]["exitCode"], 0)


class TestRealWrapperLifecycle(unittest.TestCase):
    """Spawns only trivial Python children owned by this test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = os.path.realpath(self._tmp.name)

    def test_trivial_child_runs_through_start_go_handshake(self):
        code = (
            "import os, sys\n"
            "print('BENCHMARK_READY', flush=True)\n"
            "token = sys.stdin.readline().strip()\n"
            "print('got ' + token, flush=True)\n"
            f"print('uuid ' + os.environ.get({obs.LAUNCH_UUID_ENV!r}, ''), flush=True)\n"
            "print('TPOT 1.25 ms', flush=True)\n"
            "print('TIMED_WINDOW_END', flush=True)\n"
        )
        output_dir = os.path.join(self.tmp, "run")
        events = []
        config = obs.RunConfig(
            output_dir=output_dir,
            cwd=self.tmp,
            argv=[sys.executable, "-c", code],
            timeout_seconds=60.0,
            grace_seconds=5.0,
            echo=False,
        )
        supervisor = obs.Supervisor(
            config,
            proc_info=LiveFakeProcInfo(),
            resources=FakeResources(events),
            install_signal_handlers=False,
            launch_uuid=LAUNCH_UUID,
        )
        exit_code = supervisor.run()

        receipt = read_json(os.path.join(output_dir, obs.RECEIPT_FILE))
        state = read_json(os.path.join(output_dir, obs.STATE_FILE))
        self.assertEqual(exit_code, 0, receipt)
        self.assertEqual(receipt["result"], "success")
        self.assertEqual(receipt["exitCode"], 0)
        self.assertEqual(receipt["childPid"], state["child"]["pid"])
        self.assertEqual(events, [("snapshot", "pre_go"), ("snapshot", "post_end")])
        with open(os.path.join(output_dir, obs.STDOUT_LOG), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(
            lines,
            ["BENCHMARK_READY", "got GO", f"uuid {LAUNCH_UUID}", "TPOT 1.25 ms", "TIMED_WINDOW_END"],
        )

    def test_wrapper_never_execs_without_start(self):
        marker = os.path.join(self.tmp, "executed")
        argv = obs.wrapper_argv([sys.executable, "-c", f"open({marker!r}, 'w').close()"])
        for stdin_bytes in (b"", b"STOP\n", b"START"):
            with self.subTest(stdin=stdin_bytes):
                completed = subprocess.run(argv, input=stdin_bytes, capture_output=True, timeout=30, check=False)
                self.assertEqual(completed.returncode, obs.WRAPPER_NOT_RELEASED_EXIT)
                self.assertFalse(os.path.exists(marker))

    def test_inherited_sigchld_ignore_cannot_fabricate_exit_status(self):
        previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        self.addCleanup(signal.signal, signal.SIGCHLD, previous)
        output_dir = os.path.join(self.tmp, "run")
        config = obs.RunConfig(
            output_dir=output_dir,
            cwd=self.tmp,
            argv=[sys.executable, "-c", "import sys; sys.exit(3)"],
            timeout_seconds=60.0,
            grace_seconds=5.0,
            echo=False,
        )
        supervisor = obs.Supervisor(
            config,
            proc_info=LiveFakeProcInfo(),
            resources=FakeResources([]),
            install_signal_handlers=False,
            launch_uuid=LAUNCH_UUID,
        )
        exit_code = supervisor.run()

        receipt = read_json(os.path.join(output_dir, obs.RECEIPT_FILE))
        self.assertEqual(exit_code, 1)
        self.assertEqual(receipt["result"], "exited_nonzero")
        self.assertEqual(receipt["exitCode"], 3)
        self.assertIs(signal.getsignal(signal.SIGCHLD), signal.SIG_IGN)

    def test_broken_echo_stream_does_not_abort_healthy_benchmark(self):
        class BrokenEcho:
            def __init__(self):
                self.writes = 0

            def write(self, text):
                self.writes += 1
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")

            def flush(self):
                pass

        code = (
            "import sys\n"
            "print('BENCHMARK_READY', flush=True)\n"
            "token = sys.stdin.readline().strip()\n"
            "print('got ' + token, flush=True)\n"
            "print('TPOT 1.25 ms', flush=True)\n"
            "print('TIMED_WINDOW_END', flush=True)\n"
            "print('summary done', flush=True)\n"
        )
        expected = ["BENCHMARK_READY", "got GO", "TPOT 1.25 ms", "TIMED_WINDOW_END", "summary done"]
        output_dir = os.path.join(self.tmp, "run")
        echo = BrokenEcho()
        config = obs.RunConfig(
            output_dir=output_dir,
            cwd=self.tmp,
            argv=[sys.executable, "-c", code],
            timeout_seconds=60.0,
            grace_seconds=5.0,
            echo=True,
        )
        supervisor = obs.Supervisor(
            config,
            proc_info=LiveFakeProcInfo(),
            resources=FakeResources([]),
            install_signal_handlers=False,
            launch_uuid=LAUNCH_UUID,
            echo_stream=echo,
        )
        exit_code = supervisor.run()

        receipt = read_json(os.path.join(output_dir, obs.RECEIPT_FILE))
        state = read_json(os.path.join(output_dir, obs.STATE_FILE))
        self.assertEqual(exit_code, 0, receipt)
        self.assertEqual(receipt["result"], "success")
        self.assertTrue(receipt["endMarkerObserved"])
        self.assertIsNone(receipt["cleanupAction"])
        self.assertEqual(echo.writes, 1)
        self.assertTrue(any(note.startswith("echo disabled") for note in receipt["notes"]))
        with open(os.path.join(output_dir, obs.STDOUT_LOG), encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(), expected)
        with open(os.path.join(output_dir, obs.PROGRESS_FILE), encoding="utf-8") as handle:
            self.assertEqual([json.loads(row)["line"] for row in handle], expected)
        self.assertEqual(state["progress"]["lineCount"], len(expected))
        self.assertEqual(state["phase"], "finished")


if __name__ == "__main__":
    unittest.main()
