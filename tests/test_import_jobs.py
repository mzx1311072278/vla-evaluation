import subprocess
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_eval.config import RemoteSource
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
    real_replace = Path.replace

    def fake_transfer(_argv, _progress):
        spec.staging_path.mkdir(parents=True)
        (spec.staging_path / "data").write_text("received", encoding="utf-8")

    def fake_inspector(path):
        nonlocal inspected
        assert path == spec.staging_path
        inspected = True
        return inspection

    def checked_replace(path, target):
        assert inspected is True
        assert path == spec.staging_path
        assert target == spec.target_path
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", checked_replace)

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


class FakeStream:
    def __init__(self, output: str):
        self.output = output
        self.offset = 0
        self.closed = False

    def read(self, size=-1):
        if self.offset >= len(self.output):
            return ""
        if size < 0:
            size = len(self.output)
        value = self.output[self.offset : self.offset + size]
        self.offset += len(value)
        return value

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


def test_default_transfer_refuses_missing_trust_context(tmp_path):
    spec = import_spec(tmp_path / "staging/job-1", tmp_path / "inbox/alice/run-1")

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
        source=source,
        trusted_credentials_root=credentials,
        trusted_staging_root=staging_root,
        trusted_inbox_root=inbox_root,
    )


def test_default_execution_revalidates_task8_immediately_before_runner(tmp_path, monkeypatch):
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
        lambda staging, trusted_root: calls.append(("staging", staging, trusted_root)) or staging,
    )

    def fake_build(source, remote_root, relative, staging, *, trusted_staging_root):
        calls.append(("build", source, remote_root, relative, staging, trusted_staging_root))
        return ["rsync", "--safe-argv"]

    def fake_run(argv, progress):
        calls.append(("run", argv))
        (spec.staging_path / "received").write_text("ok", encoding="utf-8")
        progress(100)

    monkeypatch.setattr("vla_eval.import_jobs.build_rsync_argv", fake_build)
    monkeypatch.setattr("vla_eval.import_jobs.run_rsync", fake_run)

    execute_import(spec, inspector=lambda _path: inspection)

    run_index = next(index for index, call in enumerate(calls) if call[0] == "run")
    assert [call[0] for call in calls[run_index - 3 : run_index + 1]] == [
        "credentials",
        "staging",
        "build",
        "run",
    ]
    assert calls[run_index][1] == ["rsync", "--safe-argv"]


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
