import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from vla_eval import import_jobs
from vla_eval.config import LocalSource, RemoteSource
from vla_eval.datasets import DatasetInspection, DatasetKind
from vla_eval.import_jobs import (
    CONNECTING,
    FAILED,
    PREFLIGHT,
    READY,
    TRANSFERRING,
    VERIFYING,
    DatasetValidationError,
    ImportCallbacks,
    ImportSpec,
    TransferError,
    execute_import,
    run_rsync,
)


def import_spec(staging: Path, inbox: Path) -> ImportSpec:
    return ImportSpec(
        job_id="job-1",
        source_name="lab-a",
        remote_root="/data/rollouts",
        remote_relative_path="run-1",
        staging_path=staging,
        target_path=inbox,
    )


def test_import_publishes_only_after_preflight(tmp_path, monkeypatch):
    staging = tmp_path / "staging" / "job-1"
    inbox = tmp_path / "inbox" / "alice" / "run-1"

    def fake_transfer(_argv, on_progress):
        staging.mkdir(parents=True)
        (staging / "received.marker").write_text("ok")
        on_progress(100.0)

    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "a" * 64, 2, 0, ())
    result = execute_import(
        import_spec(staging, inbox),
        transfer=fake_transfer,
        inspector=lambda _path: inspection,
    )
    assert result.dataset_path == inbox
    assert inbox.exists()
    assert not staging.exists()


def test_import_failure_keeps_partial_files(tmp_path):
    spec = import_spec(tmp_path / "staging/job-2", tmp_path / "inbox/alice/run-2")
    with pytest.raises(TransferError):
        execute_import(
            spec,
            transfer=lambda _argv, _progress: (_ for _ in ()).throw(TransferError("network")),
        )
    assert spec.staging_path.exists()


def test_import_reports_ordered_states_and_monotonic_progress(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    events = []

    def fake_transfer(_argv, progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "data").write_text("received", encoding="utf-8")
        for value in (-4, 25, 20, 140):
            progress(value)

    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "f" * 64, 8, 1, ())
    callbacks = ImportCallbacks(
        on_state=lambda state: events.append(("state", state)),
        on_progress=lambda progress: events.append(("progress", progress)),
    )

    execute_import(
        spec,
        transfer=fake_transfer,
        inspector=lambda _path: inspection,
        callbacks=callbacks,
    )

    assert [value for kind, value in events if kind == "state"] == [
        CONNECTING,
        TRANSFERRING,
        VERIFYING,
        PREFLIGHT,
        READY,
    ]
    assert [value for kind, value in events if kind == "progress"] == [0.0, 25.0, 25.0, 100.0]


def test_preflight_failure_keeps_staging_and_reports_failed(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    states = []

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "partial").write_text("data", encoding="utf-8")

    inspection = DatasetInspection(None, False, "0" * 64, 4, None, ("unknown dataset",))

    with pytest.raises(DatasetValidationError, match="unknown dataset"):
        execute_import(
            spec,
            transfer=fake_transfer,
            inspector=lambda _path: inspection,
            callbacks=ImportCallbacks(on_state=states.append),
        )

    assert spec.staging_path.exists()
    assert not spec.target_path.exists()
    assert states[-1] == FAILED


def test_target_collision_rejected_before_transfer(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    spec.target_path.mkdir(parents=True)
    called = False

    def fake_transfer(_argv, _progress):
        nonlocal called
        called = True

    with pytest.raises(FileExistsError):
        execute_import(spec, transfer=fake_transfer)

    assert called is False
    assert spec.target_path.exists()


def test_cross_filesystem_publish_is_rejected_before_transfer(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    called = False

    def fake_transfer(_argv, _progress):
        nonlocal called
        called = True

    real_stat = Path.stat

    def different_device(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        device = result.st_dev + (1 if path == spec.target_path.parent else 0)
        values = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
        values["st_dev"] = device
        return SimpleNamespace(**values)

    monkeypatch.setattr(Path, "stat", different_device)

    with pytest.raises(OSError, match="same filesystem"):
        execute_import(spec, transfer=fake_transfer)

    assert called is False


def test_replace_happens_after_ready_inspection_and_preserves_fingerprint(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "9" * 64, 7, 1, ())
    inspected = False
    real_rename = import_jobs._rename_no_replace

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "data").write_text("received", encoding="utf-8")

    def fake_inspector(path):
        nonlocal inspected
        assert path == spec.staging_path
        inspected = True
        return inspection

    def checked_rename(path, target):
        assert inspected is True
        assert path == spec.staging_path
        assert target == spec.target_path
        return real_rename(path, target)

    monkeypatch.setattr(import_jobs, "_rename_no_replace", checked_rename)

    result = execute_import(spec, transfer=fake_transfer, inspector=fake_inspector)

    assert result.inspection.fingerprint == inspection.fingerprint


class FakeProcess:
    def __init__(self, output: str, returncode: int = 0):
        self.stdout = FakeStream(output)
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    @property
    def stdout(self):
        return self._stdout

    @stdout.setter
    def stdout(self, stream):
        previous = getattr(self, "_stdout", None)
        if previous is not None:
            previous.close()
        self._stdout = stream


class FakeStream:
    def __init__(self, output: str):
        self.read_descriptor, write_descriptor = os.pipe()
        self.closed = False
        self.encoding = "utf-8"
        if output:
            os.write(write_descriptor, output.encode())
        os.close(write_descriptor)

    def read(self, size=-1):
        if size < 0:
            size = 4096
        return os.read(self.read_descriptor, size).decode()

    def fileno(self):
        return self.read_descriptor

    def close(self):
        if not self.closed:
            os.close(self.read_descriptor)
            self.closed = True


class SelectableFakeStream(FakeStream):
    pass


class ChunkingSelectableFakeStream(SelectableFakeStream):
    def __init__(self, chunks, *, delay=0.0):
        super().__init__("")
        self.chunks = iter(chunks)
        self.delay = delay

    def read(self, _size=-1):
        if self.delay:
            time.sleep(self.delay)
        return next(self.chunks, "")


class ErrorSelectableFakeStream(SelectableFakeStream):
    def read(self, _size=-1):
        raise OSError("output read failed")


class NoDescriptorStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeSelector:
    def __init__(self):
        self.calls = 0

    def register(self, _file, _events):
        return None

    def select(self, _timeout):
        self.calls += 1
        if self.calls == 1:
            return []
        return [(object(), 1)]

    def close(self):
        return None


class DelayedExitFakeProcess(FakeProcess):
    def __init__(self, output: str, timeouts_before_exit: int):
        super().__init__(output)
        self.timeouts_before_exit = timeouts_before_exit
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.terminated or self.killed:
            return -15
        if len(self.wait_timeouts) <= self.timeouts_before_exit:
            raise subprocess.TimeoutExpired(["rsync"], timeout)
        return self.returncode


class RegisterFailingSelector:
    instances: ClassVar[list] = []

    def __init__(self):
        self.closed = False
        self.__class__.instances.append(self)

    def register(self, _file, _events):
        raise OSError("selector does not support this pipe")

    def close(self):
        self.closed = True


class AlwaysReadySelector:
    instances: ClassVar[list] = []

    def __init__(self):
        self.closed = False
        self.__class__.instances.append(self)

    def register(self, _file, _events):
        return None

    def select(self, _timeout):
        return [(object(), 1)]

    def close(self):
        self.closed = True


def test_run_rsync_uses_exact_argv_without_shell(monkeypatch):
    process = FakeProcess("")
    captured = {}
    argv = ["rsync", "--", "host:run/", "/staging/"]

    def fake_popen(received, **kwargs):
        captured["argv"] = received
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", fake_popen)

    run_rsync(argv, lambda _progress: None)

    assert captured == {
        "argv": argv,
        "kwargs": {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "shell": False,
        },
    }


def test_run_rsync_prepares_redaction_before_starting_process(monkeypatch):
    launched = False

    def fake_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        return FakeProcess("")

    monkeypatch.setattr(import_jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        import_jobs,
        "_argv_secrets",
        lambda _argv: (_ for _ in ()).throw(RuntimeError("redaction setup failed")),
    )

    with pytest.raises(RuntimeError, match="redaction setup failed"):
        run_rsync(["rsync"], lambda _progress: None)

    assert launched is False


def test_argv_secrets_normalizes_paths_without_touching_filesystem(monkeypatch):
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *_args, **_kwargs: pytest.fail("redaction must not resolve filesystem paths"),
    )

    secrets = import_jobs._argv_secrets(["rsync", "-e", "ssh -i /private/keys/../keys/id_ed25519/"])

    assert "/private/keys/id_ed25519" in secrets
    assert "/private/keys/id_ed25519/" in secrets


def test_run_rsync_parses_cr_progress_monotonically_and_ignores_filenames(monkeypatch):
    process = FakeProcess(
        "  1,000  12% 1MB/s\r  2,000  8% 1MB/s\r>f+++++++++|4|frame-99%.jpg\r  3,000 100% 1MB/s\n"
    )
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    progress = []

    run_rsync(["rsync"], progress.append)

    assert progress == [12.0, 12.0, 100.0]


def test_run_rsync_nonzero_error_has_bounded_sanitized_tail(monkeypatch):
    output = "\n".join([f"line-{index}" for index in range(205)])
    output += "\ntoken=very-secret\n/private/credentials/id_ed25519\x1b[31m"
    process = FakeProcess(output, returncode=23)
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    argv = [
        "rsync",
        "-e",
        "ssh -i /private/credentials/id_ed25519 -o UserKnownHostsFile=/private/known_hosts",
    ]

    with pytest.raises(TransferError) as caught:
        run_rsync(argv, lambda _progress: None)

    error = caught.value
    assert len(error.safe_tail) <= 200
    rendered = "\n".join(error.safe_tail)
    assert "very-secret" not in rendered
    assert "/private/credentials/id_ed25519" not in rendered
    assert "\x1b" not in rendered


def test_run_rsync_redacts_path_variants_and_complete_credentials(monkeypatch):
    key = "/private/credentials/../credentials/id_ed25519/"
    known_hosts = "/private/known_hosts"
    output = "\n".join(
        [
            "/private/credentials/id_ed25519",
            f"{known_hosts}/",
            "Authorization: Bearer bearer-secret",
            "Bearer second-bearer-secret",
            "api-key=api-secret-value",
            "token: token-secret-value",
            "password = password-secret-value",
            "secret=generic-secret-value\x1b[31m",
        ]
    )
    process = FakeProcess(output, returncode=23)
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    argv = [
        "rsync",
        "-e",
        f"ssh -i {key} -o UserKnownHostsFile={known_hosts}",
    ]

    with pytest.raises(TransferError) as caught:
        run_rsync(argv, lambda _progress: None)

    rendered = "\n".join(caught.value.safe_tail)
    for leaked in (
        "/private/credentials/id_ed25519",
        known_hosts,
        "bearer-secret",
        "second-bearer-secret",
        "api-secret-value",
        "token-secret-value",
        "password-secret-value",
        "generic-secret-value",
        "\x1b",
    ):
        assert leaked not in rendered


def test_run_rsync_cleans_up_process_when_progress_callback_fails(monkeypatch):
    process = FakeProcess("  1,000  20% 1MB/s\rremaining")
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)

    with pytest.raises(RuntimeError, match="database unavailable"):
        run_rsync(
            ["rsync"],
            lambda _progress: (_ for _ in ()).throw(RuntimeError("database unavailable")),
        )

    assert process.terminated is True
    assert process.stdout.closed is True


def test_run_rsync_selector_polls_while_quiet_and_streams_partial_cr(monkeypatch):
    process = FakeProcess("")
    process.stdout = SelectableFakeStream("")
    chunks = iter([b"  1,000  10% 1MB/s\r", b""])
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr("vla_eval.import_jobs.selectors.DefaultSelector", FakeSelector)
    monkeypatch.setattr("vla_eval.import_jobs.os.read", lambda _fd, _size: next(chunks))
    progress = []
    polls = []

    run_rsync(["rsync"], progress.append, on_poll=lambda: polls.append("poll"))

    assert polls == ["poll"]
    assert progress == [10.0]


def test_run_rsync_quiet_poll_cancellation_terminates_without_fake_progress(monkeypatch):
    process = FakeProcess("")
    process.stdout = SelectableFakeStream("")
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr("vla_eval.import_jobs.selectors.DefaultSelector", FakeSelector)
    progress = []

    with pytest.raises(TransferError, match="cancelled"):
        run_rsync(
            ["rsync"],
            progress.append,
            on_poll=lambda: (_ for _ in ()).throw(TransferError("cancelled")),
        )

    assert process.terminated is True
    assert process.stdout.closed is True
    assert progress == []


def test_run_rsync_polls_cancellation_while_waiting_after_stdout_eof(monkeypatch):
    process = DelayedExitFakeProcess("", timeouts_before_exit=2)
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    polls = []

    run_rsync(["rsync"], lambda _progress: None, on_poll=lambda: polls.append("poll"))

    assert polls == ["poll", "poll"]
    assert process.wait_timeouts == [0.1, 0.1, 0.1]


def test_run_rsync_eof_wait_cancellation_terminates_process(monkeypatch):
    process = DelayedExitFakeProcess("", timeouts_before_exit=100)
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)

    with pytest.raises(TransferError, match="cancelled"):
        run_rsync(
            ["rsync"],
            lambda _progress: None,
            on_poll=lambda: (_ for _ in ()).throw(TransferError("cancelled")),
        )

    assert process.terminated is True


def test_selector_registration_failure_closes_and_uses_safe_fallback(monkeypatch):
    RegisterFailingSelector.instances.clear()
    process = FakeProcess("")
    process.stdout = SelectableFakeStream("12%\r")
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(
        "vla_eval.import_jobs.selectors.DefaultSelector",
        RegisterFailingSelector,
    )
    progress = []

    run_rsync(["rsync"], progress.append)

    assert progress == [12.0]
    assert RegisterFailingSelector.instances[0].closed is True


def test_selector_registration_fallback_polls_cancellation(monkeypatch):
    RegisterFailingSelector.instances.clear()
    process = FakeProcess("")
    process.stdout = SelectableFakeStream("")
    monkeypatch.setattr(import_jobs.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(
        import_jobs.selectors,
        "DefaultSelector",
        RegisterFailingSelector,
    )
    monkeypatch.setattr(import_jobs.os, "set_blocking", lambda _fd, _blocking: None)
    monkeypatch.setattr(
        import_jobs.os,
        "read",
        lambda _fd, _size: (_ for _ in ()).throw(BlockingIOError()),
    )

    with pytest.raises(TransferError, match="cancelled"):
        run_rsync(
            ["rsync"],
            lambda _progress: None,
            on_poll=lambda: (_ for _ in ()).throw(TransferError("cancelled")),
        )

    assert process.terminated is True
    assert process.stdout.closed is True
    assert RegisterFailingSelector.instances[0].closed is True


def test_fallback_callback_failures_leave_no_reader_threads(monkeypatch):
    active_process = None
    fd_root = "/proc/self/fd" if Path("/proc/self/fd").exists() else "/dev/fd"
    baseline_fd_count = len(os.listdir(fd_root))
    baseline = {
        thread.ident for thread in threading.enumerate() if thread.name.endswith("(read_output)")
    }
    monkeypatch.setattr(import_jobs.selectors, "DefaultSelector", RegisterFailingSelector)
    monkeypatch.setattr(import_jobs.os, "set_blocking", lambda _fd, _blocking: None)

    def fake_popen(*_args, **_kwargs):
        assert active_process is not None
        return active_process

    def fake_read(_fd, size):
        assert active_process is not None
        return active_process.stdout.read(size).encode()

    monkeypatch.setattr(import_jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(import_jobs.os, "read", fake_read)

    for _ in range(100):
        active_process = FakeProcess("")
        active_process.stdout = ChunkingSelectableFakeStream(["12%\r"] * 4)

        with pytest.raises(RuntimeError, match="callback failed"):
            run_rsync(
                ["rsync"],
                lambda _progress: (_ for _ in ()).throw(RuntimeError("callback failed")),
            )

        assert active_process.terminated is True
        assert active_process.stdout.closed is True

    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        remaining = {
            thread.ident
            for thread in threading.enumerate()
            if thread.name.endswith("(read_output)")
        }
        if remaining == baseline:
            break
        time.sleep(0.01)

    assert remaining == baseline
    assert len(os.listdir(fd_root)) == baseline_fd_count


def test_real_fallback_bursts_leave_threads_and_fds_stable(monkeypatch):
    fd_root = "/proc/self/fd" if Path("/proc/self/fd").exists() else "/dev/fd"
    baseline_fd_count = len(os.listdir(fd_root))
    baseline_threads = {thread.ident for thread in threading.enumerate()}
    monkeypatch.setattr(import_jobs.selectors, "DefaultSelector", RegisterFailingSelector)
    argv = [
        sys.executable,
        "-c",
        "import sys;sys.stdout.write('12%\\r'*5000);sys.stdout.flush()",
    ]

    for _ in range(100):
        with pytest.raises(RuntimeError, match="callback failed"):
            run_rsync(
                argv,
                lambda _progress: (_ for _ in ()).throw(RuntimeError("callback failed")),
            )

    assert {thread.ident for thread in threading.enumerate()} == baseline_threads
    assert len(os.listdir(fd_root)) == baseline_fd_count


def test_selector_polls_cancellation_during_continuous_output(monkeypatch):
    AlwaysReadySelector.instances.clear()
    process = FakeProcess("")
    process.stdout = SelectableFakeStream("")
    reads = 0
    polls = []

    def continuous_read(_fd, _size):
        nonlocal reads
        time.sleep(0.01)
        reads += 1
        return b"x" if reads <= 30 else b""

    monkeypatch.setattr(import_jobs.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(import_jobs.selectors, "DefaultSelector", AlwaysReadySelector)
    monkeypatch.setattr(import_jobs.os, "read", continuous_read)
    started = time.monotonic()

    with pytest.raises(TransferError, match="cancelled"):
        run_rsync(
            ["rsync"],
            lambda _progress: None,
            on_poll=lambda: (
                polls.append(time.monotonic()),
                (_ for _ in ()).throw(TransferError("cancelled")),
            )[-1],
        )

    assert polls
    assert polls[0] - started < 0.25
    assert process.terminated is True
    assert process.stdout.closed is True
    assert AlwaysReadySelector.instances[0].closed is True


def test_registration_fallback_polls_cancellation_during_continuous_output(monkeypatch):
    RegisterFailingSelector.instances.clear()
    process = FakeProcess("")
    process.stdout = ChunkingSelectableFakeStream(["x"] * 30, delay=0.01)
    polls = []
    monkeypatch.setattr(import_jobs.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(import_jobs.selectors, "DefaultSelector", RegisterFailingSelector)
    monkeypatch.setattr(import_jobs.os, "set_blocking", lambda _fd, _blocking: None)
    monkeypatch.setattr(
        import_jobs.os,
        "read",
        lambda _fd, size: process.stdout.read(size).encode(),
    )
    started = time.monotonic()

    with pytest.raises(TransferError, match="cancelled"):
        run_rsync(
            ["rsync"],
            lambda _progress: None,
            on_poll=lambda: (
                polls.append(time.monotonic()),
                (_ for _ in ()).throw(TransferError("cancelled")),
            )[-1],
        )

    assert polls
    assert polls[0] - started < 0.25
    assert process.terminated is True
    assert process.stdout.closed is True
    assert RegisterFailingSelector.instances[0].closed is True


def test_registration_fallback_read_error_cleans_up(monkeypatch):
    RegisterFailingSelector.instances.clear()
    process = FakeProcess("")
    process.stdout = ErrorSelectableFakeStream("")
    monkeypatch.setattr(import_jobs.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(import_jobs.selectors, "DefaultSelector", RegisterFailingSelector)
    monkeypatch.setattr(import_jobs.os, "set_blocking", lambda _fd, _blocking: None)
    monkeypatch.setattr(
        import_jobs.os,
        "read",
        lambda _fd, size: process.stdout.read(size).encode(),
    )

    with pytest.raises(OSError, match="output read failed"):
        run_rsync(["rsync"], lambda _progress: None)

    assert process.terminated is True
    assert process.stdout.closed is True
    assert RegisterFailingSelector.instances[0].closed is True


def test_output_without_descriptor_fails_closed_and_cleans_up(monkeypatch):
    process = FakeProcess("")
    process.stdout.close()
    process.stdout = NoDescriptorStream()
    monkeypatch.setattr(import_jobs.subprocess, "Popen", lambda *_a, **_kw: process)

    with pytest.raises(TransferError, match="pollable descriptor"):
        run_rsync(["rsync"], lambda _progress: None)

    assert process.terminated is True
    assert process.stdout.closed is True


def test_run_rsync_bounds_newline_free_pending_output():
    pending_sizes = []
    argv = [
        sys.executable,
        "-c",
        "import sys;sys.stdout.buffer.write(b'x'*(8*1024*1024));sys.stdout.flush();sys.exit(23)",
    ]
    started = time.monotonic()

    with pytest.raises(TransferError):
        run_rsync(
            argv,
            lambda _progress: None,
            on_pending_size=pending_sizes.append,
        )

    assert time.monotonic() - started < 10
    assert pending_sizes
    assert max(pending_sizes) <= 8192


def test_run_rsync_observes_flushed_cr_before_local_child_exits():
    progress_seen = threading.Event()
    finished = threading.Event()
    errors = []
    argv = [
        sys.executable,
        "-u",
        "-c",
        "import sys,time;sys.stdout.write('  1,000  10% 1MB/s\\r');sys.stdout.flush();time.sleep(0.4)",
    ]

    def worker():
        try:
            run_rsync(argv, lambda _progress: progress_seen.set())
        except Exception as error:  # noqa: BLE001 - relay worker failures to the test thread
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert progress_seen.wait(timeout=0.3)
    assert not finished.is_set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []


def test_run_rsync_rejects_unrelated_percent_records(monkeypatch):
    process = FakeProcess("12%\r  1,000  42% 1MB/s\r2026 99% packet loss\r")
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)
    progress = []

    run_rsync(["rsync"], progress.append)

    assert progress == [12.0, 42.0]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        [""],
        ["rsync", "bad\x00argument"],
        ["rsync", "bad\nargument"],
        ["rsync", 1],
        "rsync",
        None,
    ],
)
def test_run_rsync_rejects_invalid_argv_without_launch(argv, monkeypatch):
    monkeypatch.setattr(
        "vla_eval.import_jobs.subprocess.Popen",
        lambda *_a, **_kw: pytest.fail("invalid argv must not launch"),
    )

    with pytest.raises(TransferError):
        run_rsync(argv, lambda _progress: None)


@pytest.mark.parametrize("launch_error", [ValueError("nul"), TypeError("bad"), IndexError("bad")])
def test_run_rsync_normalizes_launch_errors(launch_error, monkeypatch):
    monkeypatch.setattr(
        "vla_eval.import_jobs.subprocess.Popen",
        lambda *_a, **_kw: (_ for _ in ()).throw(launch_error),
    )

    with pytest.raises(TransferError, match="could not be started"):
        run_rsync(["rsync"], lambda _progress: None)


def test_run_rsync_copies_argv_to_a_new_list(monkeypatch):
    process = FakeProcess("")
    original = ("rsync", "--version")
    captured = []

    def fake_popen(argv, **_kwargs):
        captured.append(argv)
        return process

    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", fake_popen)

    run_rsync(original, lambda _progress: None)

    assert captured == [["rsync", "--version"]]
    assert captured[0] is not original


def test_callback_and_cleanup_failures_are_both_observable(monkeypatch):
    process = FakeProcess("  1,000  20% 1MB/s\r")
    real_close = process.stdout.close

    def fail_close():
        real_close()
        raise RuntimeError("close failed")

    process.stdout.close = fail_close
    monkeypatch.setattr("vla_eval.import_jobs.subprocess.Popen", lambda *_a, **_kw: process)

    with pytest.raises(BaseExceptionGroup) as caught:
        run_rsync(
            ["rsync"],
            lambda _progress: (_ for _ in ()).throw(ValueError("callback failed")),
        )

    messages = [str(error) for error in caught.value.exceptions]
    assert any("callback failed" in message for message in messages)
    assert any("close failed" in message for message in messages)


def test_failed_state_callback_does_not_hide_original_failure(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")

    def fake_transfer(_argv, _progress):
        raise TransferError("network failed")

    def on_state(state):
        if state == FAILED:
            raise RuntimeError("failed persistence failed")

    with pytest.raises(BaseExceptionGroup) as caught:
        execute_import(
            spec,
            transfer=fake_transfer,
            callbacks=ImportCallbacks(on_state=on_state),
        )

    messages = [str(error) for error in caught.value.exceptions]
    assert any("network failed" in message for message in messages)
    assert any("failed persistence failed" in message for message in messages)


def test_rejected_symlink_staging_parent_does_not_create_outside_placeholder(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    spec = import_spec(linked / "job-1", tmp_path / "inbox/run-1")
    transfer_called = False

    def fake_transfer(_argv, _progress):
        nonlocal transfer_called
        transfer_called = True

    with pytest.raises(ValueError, match="symlink"):
        execute_import(spec, transfer=fake_transfer)

    assert transfer_called is False
    assert not (outside / "job-1").exists()


def test_injected_staging_rejects_lexical_parent_traversal(tmp_path):
    staging = tmp_path / "safe" / ".." / "outside" / "job-1"
    spec = import_spec(staging, tmp_path / "inbox/run-1")

    with pytest.raises(ValueError, match=r"\.\."):
        execute_import(spec, transfer=lambda _argv, _progress: None)

    assert not (tmp_path / "outside").exists()


def test_production_transfer_refuses_missing_trust_context(tmp_path):
    spec = replace(
        import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1"),
        mode="production",
    )

    with pytest.raises(ValueError, match="trust context"):
        execute_import(spec)


def test_execute_import_public_default_is_run_rsync():
    assert signature(execute_import).parameters["transfer"].default is run_rsync


def production_spec(tmp_path: Path) -> ImportSpec:
    credentials = tmp_path / "credentials"
    staging_root = tmp_path / "staging"
    inbox_root = tmp_path / "inbox"
    for directory in (credentials, staging_root, inbox_root):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
    source = RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=credentials / "id_ed25519",
        known_hosts_path=credentials / "known_hosts",
        roots=("/data/rollouts",),
    )
    return ImportSpec(
        job_id="job-prod",
        source_name="lab-a",
        remote_root="/data/rollouts",
        remote_relative_path="run-1",
        staging_path=staging_root / "job-prod",
        target_path=inbox_root / "alice" / "run-1",
        mode="production",
        source=source,
        trusted_credentials_root=credentials,
        trusted_staging_root=staging_root,
        trusted_inbox_root=inbox_root,
    )


def local_production_spec(tmp_path: Path) -> ImportSpec:
    source_root = tmp_path / "local-source"
    dataset = source_root / "run-1"
    staging_root = tmp_path / "staging-local"
    inbox_root = tmp_path / "inbox-local"
    for directory in (dataset, staging_root, inbox_root):
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
    source = LocalSource(name="this-host", roots=(source_root,))
    return ImportSpec(
        job_id="job-local",
        source_name="this-host",
        remote_root=str(source_root),
        remote_relative_path="run-1",
        staging_path=staging_root / "job-local",
        target_path=inbox_root / "run-1",
        mode="production",
        local_source=source,
        trusted_staging_root=staging_root,
        trusted_inbox_root=inbox_root,
    )


def test_production_local_source_reuses_preflight_and_atomic_publication(tmp_path):
    spec = local_production_spec(tmp_path)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "e" * 64, 4, 1, ())
    calls = []

    def fake_run(argv, progress):
        calls.append(argv)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")
        progress(100)

    result = execute_import(spec, transfer=fake_run, inspector=lambda _path: inspection)

    assert calls == [
        [
            "rsync",
            "-a",
            "--partial",
            "--append-verify",
            "--info=progress2",
            "--out-format=%i|%l|%n",
            "--",
            f"{spec.local_source.roots[0] / 'run-1'}/",
            f"{spec.staging_path}/",
        ]
    ]
    assert result.dataset_path == spec.target_path
    assert result.inspection is inspection
    assert spec.target_path.is_dir()
    assert not spec.staging_path.exists()


def test_production_rejects_both_remote_and_local_sources(tmp_path):
    remote = production_spec(tmp_path)
    local = local_production_spec(tmp_path)
    spec = replace(remote, local_source=local.local_source)

    with pytest.raises(ValueError, match="exactly one"):
        execute_import(spec, transfer=lambda _argv, _progress: None)


def test_production_rejects_missing_transport_source(tmp_path):
    spec = replace(production_spec(tmp_path), source=None)

    with pytest.raises(ValueError, match="exactly one"):
        execute_import(spec, transfer=lambda _argv, _progress: None)


def test_production_rejects_mismatched_local_source_name(tmp_path):
    spec = replace(local_production_spec(tmp_path), source_name="different")

    with pytest.raises(ValueError, match="source name"):
        execute_import(spec, transfer=lambda _argv, _progress: None)


def test_production_mode_revalidates_task8_with_wrapped_runner(tmp_path, monkeypatch):
    spec = production_spec(tmp_path)
    calls = []
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "b" * 64, 4, 1, ())

    monkeypatch.setattr(
        "vla_eval.import_jobs.validate_remote_source_files",
        lambda source, *, trusted_credentials_root: calls.append(
            ("credentials", source, trusted_credentials_root)
        ),
    )
    monkeypatch.setattr(
        "vla_eval.import_jobs.validate_staging_path",
        lambda staging, trusted_root, **_kwargs: (
            calls.append(("staging", staging, trusted_root)) or staging
        ),
    )

    def fake_build(
        source,
        remote_root,
        relative,
        staging,
        *,
        trusted_staging_root,
        minimum_checked_ancestor=None,
    ):
        calls.append(("build", source, remote_root, relative, staging, trusted_staging_root))
        return ["rsync", "--safe-argv"]

    def fake_run(argv, progress):
        calls.append(("run", argv))
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")
        progress(100)

    monkeypatch.setattr("vla_eval.import_jobs.build_rsync_argv", fake_build)
    execute_import(spec, transfer=fake_run, inspector=lambda _path: inspection)

    run_index = next(index for index, call in enumerate(calls) if call[0] == "run")
    assert [call[0] for call in calls[run_index - 3 : run_index + 1]] == [
        "credentials",
        "staging",
        "build",
        "run",
    ]
    assert calls[run_index][1] == ["rsync", "--safe-argv"]


@pytest.mark.parametrize("context_field", ["source", "trusted_inbox_root"])
def test_injected_mode_rejects_production_context(tmp_path, context_field):
    production = production_spec(tmp_path)
    spec = import_spec(tmp_path / "injected/job-1", tmp_path / "target/run-1")
    spec = replace(spec, **{context_field: getattr(production, context_field)})

    with pytest.raises(ValueError, match="production context"):
        execute_import(spec, transfer=lambda _argv, _progress: None)


def test_cancellation_immediately_before_replace_keeps_staging(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "d" * 64, 4, 1, ())
    cancellation_polls = 0
    replace_called = False

    def is_cancelled():
        nonlocal cancellation_polls
        cancellation_polls += 1
        return cancellation_polls == 5

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    def forbidden_replace(_path, _target):
        nonlocal replace_called
        replace_called = True

    monkeypatch.setattr(import_jobs, "_rename_no_replace", forbidden_replace)

    with pytest.raises(TransferError, match="cancelled"):
        execute_import(
            spec,
            transfer=fake_transfer,
            inspector=lambda _path: inspection,
            callbacks=ImportCallbacks(is_cancelled=is_cancelled),
        )

    assert cancellation_polls == 5
    assert replace_called is False
    assert spec.staging_path.exists()
    assert not spec.target_path.exists()


def test_success_does_not_poll_cancellation_after_replace(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "e" * 64, 4, 1, ())
    cancellation_polls = 0

    def is_cancelled():
        nonlocal cancellation_polls
        cancellation_polls += 1
        return cancellation_polls > 5

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    result = execute_import(
        spec,
        transfer=fake_transfer,
        inspector=lambda _path: inspection,
        callbacks=ImportCallbacks(is_cancelled=is_cancelled),
    )

    assert cancellation_polls == 5
    assert result.dataset_path.exists()


def test_publish_target_appearing_at_final_moment_is_never_replaced(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "2" * 64, 4, 1, ())

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("staging", encoding="utf-8")

    def race_rename(_source, target):
        target.mkdir()
        (target / "owner").write_text("other job", encoding="utf-8")
        raise FileExistsError("target appeared")

    monkeypatch.setattr(import_jobs, "_rename_no_replace", race_rename, raising=False)

    with pytest.raises(FileExistsError, match="target appeared"):
        execute_import(spec, transfer=fake_transfer, inspector=lambda _path: inspection)

    assert (spec.target_path / "owner").read_text(encoding="utf-8") == "other job"
    assert (spec.staging_path / "received").read_text(encoding="utf-8") == "staging"


def test_no_replace_publish_fails_closed_on_unsupported_platform(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    monkeypatch.setattr(import_jobs.sys, "platform", "unsupported-os")

    with pytest.raises(OSError, match="not supported"):
        import_jobs._rename_no_replace(source, target)

    assert source.exists()
    assert not target.exists()


def test_post_publish_verification_failure_rolls_back_before_failed_state(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "3" * 64, 4, 1, ())
    states = []

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("staging", encoding="utf-8")

    def fail_verification(_spec, _target, _production):
        raise RuntimeError("post-publish verification failed")

    monkeypatch.setattr(
        import_jobs,
        "_verify_published_target",
        fail_verification,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="post-publish verification failed"):
        execute_import(
            spec,
            transfer=fake_transfer,
            inspector=lambda _path: inspection,
            callbacks=ImportCallbacks(on_state=states.append),
        )

    assert states[-1] == FAILED
    assert spec.staging_path.exists()
    assert not spec.target_path.exists()


def test_ready_callback_failure_rolls_target_back_to_staging(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "f" * 64, 4, 1, ())
    states = []

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    def on_state(state):
        states.append(state)
        if state == READY:
            raise RuntimeError("ready persistence failed")

    with pytest.raises(RuntimeError, match="ready persistence failed"):
        execute_import(
            spec,
            transfer=fake_transfer,
            inspector=lambda _path: inspection,
            callbacks=ImportCallbacks(on_state=on_state),
        )

    assert states[-2:] == [READY, FAILED]
    assert spec.staging_path.exists()
    assert not spec.target_path.exists()


def test_ready_callback_and_rollback_failures_are_both_observable(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "1" * 64, 4, 1, ())
    real_rename = import_jobs._rename_no_replace
    replace_calls = 0

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    def fail_rollback(path, target):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("rollback failed")
        return real_rename(path, target)

    def on_state(state):
        if state == READY:
            raise RuntimeError("ready persistence failed")

    monkeypatch.setattr(import_jobs, "_rename_no_replace", fail_rollback)

    with pytest.raises(ExceptionGroup) as caught:
        execute_import(
            spec,
            transfer=fake_transfer,
            inspector=lambda _path: inspection,
            callbacks=ImportCallbacks(on_state=on_state),
        )

    messages = [str(error) for error in caught.value.exceptions]
    assert any("ready persistence failed" in message for message in messages)
    assert any("rollback failed" in message for message in messages)
    assert spec.target_path.exists()
    assert not spec.staging_path.exists()


def test_default_inspector_is_bound_to_staging_root(tmp_path, monkeypatch):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "c" * 64, 4, 1, ())
    calls = []

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    def fake_inspect(path, allowed_root):
        calls.append((path, allowed_root))
        return inspection

    monkeypatch.setattr("vla_eval.import_jobs.inspect_dataset", fake_inspect)

    execute_import(spec, transfer=fake_transfer)

    assert calls == [(spec.staging_path, spec.staging_path)]


def test_production_target_must_be_within_trusted_inbox(tmp_path, monkeypatch):
    spec = production_spec(tmp_path)
    spec = ImportSpec(
        **{
            **spec.__dict__,
            "target_path": tmp_path / "other-inbox" / "alice" / "run-1",
        }
    )
    monkeypatch.setattr(
        "vla_eval.import_jobs.run_rsync",
        lambda _argv, _progress: pytest.fail("runner must not start"),
    )

    with pytest.raises(ValueError, match="trusted root"):
        execute_import(spec)


def test_production_staging_must_be_strict_descendant_of_trusted_root(tmp_path):
    spec = production_spec(tmp_path)
    spec = replace(spec, staging_path=spec.trusted_staging_root)

    with pytest.raises(ValueError, match="strict descendant"):
        execute_import(
            spec,
            transfer=lambda _argv, _progress: pytest.fail("transfer must not start"),
        )


def test_production_rejects_symlinked_inbox_component(tmp_path, monkeypatch):
    spec = production_spec(tmp_path)
    actual = spec.trusted_inbox_root / "actual"
    actual.mkdir(mode=0o700)
    linked = spec.trusted_inbox_root / "alice"
    linked.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(
        "vla_eval.import_jobs.run_rsync",
        lambda _argv, _progress: pytest.fail("runner must not start"),
    )

    with pytest.raises(ValueError, match="directories"):
        execute_import(spec)


def test_production_rejects_writable_inbox_root(tmp_path, monkeypatch):
    spec = production_spec(tmp_path)
    spec.trusted_inbox_root.chmod(0o770)
    monkeypatch.setattr(
        "vla_eval.import_jobs.run_rsync",
        lambda _argv, _progress: pytest.fail("runner must not start"),
    )

    with pytest.raises(ValueError, match="group or other writable"):
        execute_import(spec)


def test_production_accepts_writable_ancestor_above_data_root(tmp_path):
    shared = tmp_path / "shared"
    data_root = shared / "data"
    staging_root = data_root / "staging"
    inbox_root = data_root / "inbox"
    source_root = tmp_path / "source"
    source_dataset = source_root / "run-1"
    for directory in (staging_root, inbox_root, source_dataset):
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
    data_root.chmod(0o700)
    shared.chmod(0o777)
    spec = ImportSpec(
        job_id="job-shared",
        source_name="this-host",
        remote_root=str(source_root),
        remote_relative_path="run-1",
        staging_path=staging_root / "job-shared",
        target_path=inbox_root / "run-1",
        mode="production",
        local_source=LocalSource(name="this-host", roots=(source_root,)),
        trusted_staging_root=staging_root,
        trusted_inbox_root=inbox_root,
        storage_trust_boundary=data_root,
    )
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "b" * 64, 2, 1, ())

    def fake_transfer(_argv, _progress):
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")

    result = execute_import(spec, transfer=fake_transfer, inspector=lambda _path: inspection)

    assert result.dataset_path == spec.target_path


def test_production_boundary_still_rejects_writable_data_root(tmp_path):
    spec = local_production_spec(tmp_path)
    data_root = tmp_path / "data"
    spec.trusted_staging_root.rename(data_root)
    staging_root = data_root / "staging"
    inbox_root = data_root / "inbox"
    staging_root.mkdir(mode=0o700)
    inbox_root.mkdir(mode=0o700)
    data_root.chmod(0o770)
    spec = replace(
        spec,
        staging_path=staging_root / spec.job_id,
        target_path=inbox_root / "run-1",
        trusted_staging_root=staging_root,
        trusted_inbox_root=inbox_root,
        storage_trust_boundary=data_root,
    )

    with pytest.raises(ValueError, match="group or other writable"):
        execute_import(spec, transfer=lambda _argv, _progress: None)
