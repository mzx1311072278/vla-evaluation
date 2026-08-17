"""Behavioral tests for the management CLI (`python -m vla_eval.cli`)."""

import csv
import json
from pathlib import Path

import fakeredis
import pytest
from rq import Worker
from sqlalchemy import Engine, select
from typer.testing import CliRunner

import vla_eval.cli as cli_module
import vla_eval.tasks as tasks_module
from tests.conftest import reload_job
from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.queueing import create_queues
from vla_eval.security import verify_password
from vla_eval.tasks import TaskRuntime, clear_runtime


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _reset_task_runtime():
    clear_runtime()
    yield
    clear_runtime()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "inbox").mkdir(parents=True)
    return root


@pytest.fixture
def app_config(tmp_path: Path, data_dir: Path, monkeypatch) -> Path:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"data_root: {data_dir}\n"
        f"database_url: sqlite:///{tmp_path}/app.db\n"
        "redis_url: redis://127.0.0.1:1/0\n"
        "session_secret: test-session-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLA_EVAL_CONFIG", str(config_path))
    return config_path


@pytest.fixture
def db_engine(app_config, tmp_path: Path) -> Engine:
    engine = create_engine_for_url(f"sqlite:///{tmp_path}/app.db")
    init_db(engine)
    yield engine
    engine.dispose()


def load_user(engine: Engine, username: str) -> User | None:
    with session_scope(engine) as session:
        return session.scalar(select(User).where(User.username == username))


# Module-level so RQ can serialize it as ``tests.test_cli._rq_runtime_sentinel``
# and re-import it in the forked work-horse. It resolves the task runtime via the
# global configured by ``build_runtime`` (inherited across ``fork()``) and writes
# a filesystem marker so the parent process can observe execution (fakeredis
# state written by the child is invisible to the parent after fork).
def _rq_runtime_sentinel() -> tuple[str, str]:
    import os

    from vla_eval.tasks import _require_runtime

    runtime = _require_runtime(None)  # raises RuntimeError if not configured
    marker_path = os.environ["VLA_EVAL_RQ_SENTINEL_PATH"]
    Path(marker_path).write_text(runtime.config.data_root.name, encoding="utf-8")
    return ("ran", runtime.config.data_root.name)


def _write_ready_session_dataset(path: Path) -> None:
    """Build a ready Genie02 native session dataset on disk (mirrors conftest)."""
    import numpy as np

    (path / "trajectories").mkdir(parents=True)
    (path / "session.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "session_id": path.name,
                "created_at": "2026-01-02T03:04:05+08:00",
                "status": "completed",
                "rollout_config_path": "rollout.yaml",
                "rollout_mode": "default",
                "policy_path": "policy",
                "task": "fixture",
                "num_episodes_target": 1,
                "fps": 10,
                "dataset_backend": "native",
                "dataset_root": "unused",
            }
        ),
        encoding="utf-8",
    )
    fields = (
        "session_id",
        "episode_index",
        "episode_path",
        "trajectory_path",
        "t_start",
        "t_end",
        "duration_s",
        "outcome",
        "operator_intervened",
        "notes",
    )
    with (path / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "session_id": path.name,
                "episode_index": "0",
                "episode_path": "",
                "trajectory_path": "trajectories/episode_000.npz",
                "t_start": "0",
                "t_end": "1",
                "duration_s": "1",
                "outcome": "success",
                "operator_intervened": "false",
                "notes": "",
            }
        )
    np.savez(path / "trajectories" / "episode_000.npz", action=np.ones((4, 3)))


def test_init_db_creates_tables(cli_runner: CliRunner, app_config):
    result = cli_runner.invoke(cli_module.app, ["init-db", "--config", str(app_config)])
    assert result.exit_code == 0


def test_create_user_hashes_password(cli_runner: CliRunner, app_config, db_engine: Engine):
    result = cli_runner.invoke(cli_module.app, ["create-user", "alice", "--password", "secret"])
    assert result.exit_code == 0
    user = load_user(db_engine, "alice")
    assert user is not None
    assert user.password_hash != "secret"
    assert verify_password("secret", user.password_hash)


def test_create_user_does_not_print_password(cli_runner: CliRunner, app_config):
    result = cli_runner.invoke(cli_module.app, ["create-user", "alice", "--password", "topsecret"])
    assert result.exit_code == 0
    assert "topsecret" not in result.stdout


def test_create_user_reads_initial_password_env(
    cli_runner: CliRunner, app_config, db_engine: Engine, monkeypatch
):
    monkeypatch.setenv("VLA_EVAL_INITIAL_PASSWORD", "env-pw")
    result = cli_runner.invoke(cli_module.app, ["create-user", "bob"])
    assert result.exit_code == 0
    user = load_user(db_engine, "bob")
    assert user is not None
    assert verify_password("env-pw", user.password_hash)


def test_create_user_errors_without_password(cli_runner: CliRunner, app_config, monkeypatch):
    monkeypatch.delenv("VLA_EVAL_INITIAL_PASSWORD", raising=False)
    result = cli_runner.invoke(cli_module.app, ["create-user", "carol"])
    assert result.exit_code != 0


def test_create_user_errors_on_duplicate(cli_runner: CliRunner, app_config):
    first = cli_runner.invoke(cli_module.app, ["create-user", "alice", "--password", "secret"])
    assert first.exit_code == 0
    second = cli_runner.invoke(cli_module.app, ["create-user", "alice", "--password", "other"])
    assert second.exit_code != 0


def test_create_user_admin_flag(cli_runner: CliRunner, app_config, db_engine: Engine):
    result = cli_runner.invoke(cli_module.app, ["create-user", "admin", "--password", "secret", "--admin"])
    assert result.exit_code == 0
    user = load_user(db_engine, "admin")
    assert user is not None
    assert user.is_admin is True


def test_disable_user(cli_runner: CliRunner, app_config, db_engine: Engine):
    cli_runner.invoke(cli_module.app, ["create-user", "alice", "--password", "secret"])
    user = load_user(db_engine, "alice")
    assert user.active is True

    result = cli_runner.invoke(cli_module.app, ["disable-user", "alice"])
    assert result.exit_code == 0
    refreshed = load_user(db_engine, "alice")
    assert refreshed.active is False


def test_disable_user_errors_when_missing(cli_runner: CliRunner, app_config):
    result = cli_runner.invoke(cli_module.app, ["disable-user", "ghost"])
    assert result.exit_code != 0


def test_scan_datasets_registers_ready_dataset(
    cli_runner: CliRunner, app_config, data_dir: Path, db_engine: Engine
):
    _write_ready_session_dataset(data_dir / "inbox" / "run-1")
    result = cli_runner.invoke(cli_module.app, ["scan-datasets"])
    assert result.exit_code == 0
    with session_scope(db_engine) as session:
        dataset = session.scalar(select(Dataset).where(Dataset.name == "run-1"))
        assert dataset is not None
        assert dataset.status == "READY"
        assert dataset.kind == "genie02_session"
        assert dataset.fingerprint
        assert dataset.episode_count == 1
        assert dataset.size_bytes > 0


def test_scan_datasets_skips_non_ready(cli_runner: CliRunner, app_config, data_dir: Path, db_engine: Engine):
    (data_dir / "inbox" / "junk").mkdir()
    result = cli_runner.invoke(cli_module.app, ["scan-datasets"])
    assert result.exit_code == 0
    with session_scope(db_engine) as session:
        assert session.scalar(select(Dataset).where(Dataset.name == "junk")) is None


def test_scan_datasets_is_idempotent(
    cli_runner: CliRunner, app_config, data_dir: Path, db_engine: Engine
):
    _write_ready_session_dataset(data_dir / "inbox" / "run-1")
    cli_runner.invoke(cli_module.app, ["scan-datasets"])
    cli_runner.invoke(cli_module.app, ["scan-datasets"])
    with session_scope(db_engine) as session:
        count = len(session.scalars(select(Dataset).where(Dataset.name == "run-1")).all())
        assert count == 1


def test_recover_jobs_reports_interrupted_count(
    cli_runner: CliRunner, app_config, db_engine: Engine, data_dir: Path
):
    with session_scope(db_engine) as session:
        dataset = Dataset(
            name="d",
            path=str(data_dir / "inbox" / "d"),
            kind="genie02_session",
            status="READY",
        )
        session.add(dataset)
        session.flush()
        job = EvaluationJob(
            dataset_id=dataset.id,
            profile_name="genie02-full",
            state="RUNNING",
            stage="METRICS",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    result = cli_runner.invoke(cli_module.app, ["recover-jobs"])
    assert result.exit_code == 0
    assert "1" in result.stdout
    assert reload_job(db_engine, job_id).state == "INTERRUPTED"


def test_smoke_success(cli_runner: CliRunner, app_config, monkeypatch):
    fake = fakeredis.FakeRedis()

    def _fake_create_queues(_url, *, connection=None):
        return create_queues("redis://unused", connection=fake)

    monkeypatch.setattr(cli_module, "create_queues", _fake_create_queues)
    result = cli_runner.invoke(cli_module.app, ["smoke"])
    assert result.exit_code == 0
    assert "test-session-secret" not in result.stdout


def test_smoke_failure_exits_nonzero_without_secrets(cli_runner: CliRunner, app_config, monkeypatch):
    class _BrokenConnection:
        def ping(self):
            raise OSError("redis down")

    def _fake_create_queues(_url, *, connection=None):
        return create_queues("redis://unused", connection=_BrokenConnection())

    monkeypatch.setattr(cli_module, "create_queues", _fake_create_queues)
    result = cli_runner.invoke(cli_module.app, ["smoke"])
    assert result.exit_code != 0
    assert "test-session-secret" not in result.stdout


def test_build_runtime_configures_and_recovers(app_config, monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        cli_module, "create_queues", lambda _url, **_kw: create_queues("redis://unused", connection=fake)
    )
    recovered = {}

    def _fake_recover(*, runtime=None):
        recovered["runtime"] = runtime
        return 0

    monkeypatch.setattr(cli_module, "recover_interrupted_jobs", _fake_recover)
    try:
        engine, queues, runtime = cli_module.build_runtime(app_config)
        assert isinstance(runtime, TaskRuntime)
        assert tasks_module._configured_runtime is runtime
        assert recovered["runtime"] is runtime
        assert queues.evaluation.name == "evaluations"
        assert queues.transfer.name == "transfers"
    finally:
        clear_runtime()
        engine.dispose()


def test_run_worker_uses_selected_queue(app_config, monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        cli_module, "create_queues", lambda _url, **_kw: create_queues("redis://unused", connection=fake)
    )
    captured: dict = {}

    class _FakeWorker:
        def __init__(self, queues, connection=None, **_kwargs):
            captured["queues"] = list(queues)
            captured["connection"] = connection

        def work(self, **_kwargs):
            captured["worked"] = True
            return True

    monkeypatch.setattr(cli_module, "Worker", _FakeWorker)
    try:
        cli_module.run_worker("evaluations", app_config)
    finally:
        clear_runtime()

    assert captured.get("worked") is True
    assert [queue.name for queue in captured["queues"]] == ["evaluations"]
    assert captured["connection"] is fake


def test_worker_command_runs(cli_runner: CliRunner, app_config, monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        cli_module, "create_queues", lambda _url, **_kw: create_queues("redis://unused", connection=fake)
    )
    captured: dict = {}

    class _FakeWorker:
        def __init__(self, queues, connection=None, **_kwargs):
            captured["queues"] = list(queues)

        def work(self, **_kwargs):
            captured["worked"] = True
            return True

    monkeypatch.setattr(cli_module, "Worker", _FakeWorker)
    result = cli_runner.invoke(cli_module.app, ["worker", "--queue", "evaluations"])
    assert result.exit_code == 0
    assert captured.get("worked") is True
    assert [queue.name for queue in captured["queues"]] == ["evaluations"]
    clear_runtime()


def test_rq_worker_executes_enqueued_function_against_configured_runtime(
    app_config, tmp_path, monkeypatch
):
    """End-to-end: an enqueued function is deserialized and run in the work-horse,
    and the task runtime configured by ``build_runtime`` (inherited across RQ's
    ``fork()``) resolves inside it. Observed via a filesystem marker because
    fakeredis state written by the child is invisible to the parent after fork.
    """
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(
        cli_module,
        "create_queues",
        lambda _url, **_kw: create_queues("redis://unused", connection=fake),
    )
    marker = tmp_path / "sentinel.out"
    monkeypatch.setenv("VLA_EVAL_RQ_SENTINEL_PATH", str(marker))

    engine, queues, runtime = cli_module.build_runtime(app_config)
    try:
        queues.evaluation.enqueue(_rq_runtime_sentinel)
        Worker([queues.evaluation], connection=queues.evaluation.connection).work(
            burst=True, logging_level="WARNING"
        )
        assert marker.exists()
        assert marker.read_text(encoding="utf-8") == runtime.config.data_root.name
    finally:
        clear_runtime()
        engine.dispose()


def test_rq_sentinel_fails_when_runtime_not_configured(tmp_path, monkeypatch):
    """The round-trip test above is load-bearing: with no configured runtime the
    sentinel raises before writing the marker, so the marker is absent."""
    clear_runtime()
    queues = create_queues("redis://unused", connection=fakeredis.FakeRedis())
    marker = tmp_path / "sentinel-fail.out"
    monkeypatch.setenv("VLA_EVAL_RQ_SENTINEL_PATH", str(marker))

    queues.evaluation.enqueue(_rq_runtime_sentinel)
    Worker([queues.evaluation], connection=queues.evaluation.connection).work(
        burst=True, logging_level="WARNING"
    )
    assert not marker.exists()
