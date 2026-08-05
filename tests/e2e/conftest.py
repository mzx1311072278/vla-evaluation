"""End-to-end test fixtures.

These tests drive the REAL task layer (``run_import_task`` /
``run_evaluation_task``) against a REAL ``TaskRuntime`` and the REAL FastAPI
app, but replace only the network transfer (rsync over SSH) with a local
in-process copy. The web routes, state machines, dataset inspection, atomic
publish, profile loading, and the Genie02 metrics/report pipeline all run
unmodified -- this is a behavioural end-to-end acceptance test, not a mock.

A configured runtime is the one thing ``tests/web/conftest.py`` does not
provide, so a dedicated e2e conftest is required here.
"""

from __future__ import annotations

import csv
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

import vla_eval.import_jobs as import_jobs_module
import vla_eval.tasks as tasks_module
from tests.fakes import FakeQueueBundle
from vla_eval.config import AppConfig, RemoteSource
from vla_eval.db import init_db, session_scope
from vla_eval.models import User
from vla_eval.security import hash_password
from vla_eval.tasks import (
    TaskRuntime,
    clear_runtime,
    configure_runtime,
    run_evaluation_task,
    run_import_task,
)
from vla_eval.web.app import create_app

# The remote root the lab-a source exposes; the import form must select it and
# production-mode execute_import requires the source to expose exactly one root.
REMOTE_ROOT = "/srv/datasets"
SESSION_SECRET = "e2e-session-secret"


class _CsrfInputParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and attributes.get("name") == "csrf_token":
            value = attributes.get("value")
            if value is not None:
                self.values.append(value)


def extract_csrf(html: str) -> str:
    parser = _CsrfInputParser()
    parser.feed(html)
    assert parser.values, "response did not contain a CSRF input"
    return parser.values[0]


def build_native_session(root: Path, *, session_id: str = "e2e-session") -> Path:
    """Build a minimal but valid on-disk Genie02 native session (2 episodes).

    Mirrors the ``ready_dataset`` fixture (inspect_dataset -> READY) and the
    ``minimal_native_session`` regression fixture (real report pipeline runs to
    completion). Two episodes (one success, one failure) yield a non-trivial
    GSR of 0.5 that the assertions can check.
    """
    session_dir = root / session_id
    (session_dir / "trajectories").mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "session_id": session_id,
                "created_at": "2026-01-02T03:04:05+08:00",
                "status": "completed",
                "rollout_config_path": "rollout.yaml",
                "rollout_mode": "default",
                "policy_path": "policy",
                "task": "e2e fixture",
                "num_episodes_target": 2,
                "fps": 10,
                "dataset_backend": "native",
                "dataset_root": "unused",
            }
        ),
        encoding="utf-8",
    )
    fieldnames = (
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
    rows = [
        {
            "session_id": session_id,
            "episode_index": "0",
            "episode_path": "",
            "trajectory_path": "trajectories/episode_000.npz",
            "t_start": "0",
            "t_end": "2",
            "duration_s": "2",
            "outcome": "success",
            "operator_intervened": "false",
            "notes": "",
        },
        {
            "session_id": session_id,
            "episode_index": "1",
            "episode_path": "",
            "trajectory_path": "trajectories/episode_001.npz",
            "t_start": "0",
            "t_end": "3",
            "duration_s": "3",
            "outcome": "failure",
            "operator_intervened": "false",
            "notes": "fixture failure",
        },
    ]
    with (session_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    action = np.arange(5, dtype=float)[:, None]
    np.savez(session_dir / "trajectories" / "episode_000.npz", action=action)
    np.savez(session_dir / "trajectories" / "episode_001.npz", action=action * 2)
    return session_dir


def install_fake_rsync(monkeypatch: pytest.MonkeyPatch, remote_fixture_dir: Path) -> None:
    """Replace the rsync network transfer with an in-process directory copy.

    The REAL ``execute_import`` state machine still runs end to end (credential
    validation, staging creation, verification, real dataset inspection, atomic
    no-replace publish). Only the SSH/rsync bytes-on-the-wire is faked -- there
    is no SSH server on a dev/CI machine -- by copying the prepared fixture into
    the job's staging path and reporting 100%% progress.
    """
    real_execute_import = import_jobs_module.execute_import
    fixture = remote_fixture_dir

    def fake_execute_import(spec, *, inspector, callbacks):
        def fake_transfer(_argv, on_progress):
            shutil.copytree(fixture, spec.staging_path, dirs_exist_ok=True)
            on_progress(100.0)

        return real_execute_import(
            spec, inspector=inspector, callbacks=callbacks, transfer=fake_transfer
        )

    monkeypatch.setattr(tasks_module, "execute_import", fake_execute_import)


def drain_queues(fake_queues: FakeQueueBundle, runtime: TaskRuntime) -> None:
    """Play the worker: pop enqueued transfer/evaluation calls and run them.

    Imports are drained before evaluations so a freshly imported READY dataset
    is available to the evaluation that follows. Each call is asserted to be the
    expected task entry function and is executed in-process against the runtime.
    """
    while fake_queues.transfer.enqueued or fake_queues.evaluation.enqueued:
        if fake_queues.transfer.enqueued:
            call = fake_queues.transfer.enqueued.pop(0)
            assert call.function is run_import_task, "transfer queue held an unexpected function"
            call.function(*call.args, runtime=runtime)
            continue
        call = fake_queues.evaluation.enqueued.pop(0)
        assert call.function is run_evaluation_task, "evaluation queue held an unexpected function"
        call.function(*call.args, runtime=runtime)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Service data root under a symlink-free path.

    macOS surfaces ``/tmp`` and ``/var`` as symlinks to ``/private/...``. The
    production trust-anchor validators (``validate_staging_path`` /
    ``validate_remote_source_files``) reject symlink components anywhere along a
    protected path, so the data root is resolved before any directories are
    created.
    """
    root = (tmp_path / "data").resolve()
    for name in ("inbox", "staging", "runs", "models", "db", "credentials"):
        directory = root / name
        directory.mkdir(parents=True)
        directory.chmod(0o700)
    root.chmod(0o700)
    return root


@pytest.fixture
def credentials(data_root: Path) -> Path:
    """Sealed SSH credential files satisfying ``validate_remote_source_files``.

    The credential directory is made non-owner-writable (mode 0o500) because the
    validator rejects service-owned credential directories that are owner-writable
    (a same-owner TOCTOU hardening). The key/known_hosts contents are never read
    by the fake-rsync path; they only need to exist with safe permissions.
    """
    credentials_root = data_root / "credentials"
    key_path = credentials_root / "lab-a-key"
    known_hosts_path = credentials_root / "known_hosts"
    key_path.write_text("dummy-test-private-key", encoding="utf-8")
    known_hosts_path.write_text("lab-a.example.test ssh-ed25519 AAAAfixture", encoding="utf-8")
    key_path.chmod(0o400)
    known_hosts_path.chmod(0o400)
    credentials_root.chmod(0o500)
    return credentials_root


@pytest.fixture
def db_engine() -> Engine:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def app_config(data_root: Path, credentials: Path) -> AppConfig:
    return AppConfig(
        data_root=data_root,
        database_url="sqlite://",
        redis_url="redis://unused.invalid/0",
        session_secret=SESSION_SECRET,
        remote_sources={
            "lab-a": RemoteSource(
                name="lab-a",
                host="lab-a.example.test",
                port=22,
                username="reader",
                key_path=credentials / "lab-a-key",
                known_hosts_path=credentials / "known_hosts",
                roots=(REMOTE_ROOT,),
            )
        },
    )


@pytest.fixture
def runtime(db_engine: Engine, app_config: AppConfig, data_root: Path) -> TaskRuntime:
    configured = TaskRuntime(
        engine=db_engine,
        config=app_config,
        profiles_root=Path("config/profiles"),
        credentials_root=data_root / "credentials",
    )
    configure_runtime(configured)
    yield configured
    clear_runtime()


@pytest.fixture
def app(app_config: AppConfig, db_engine: Engine, fake_queues: FakeQueueBundle):
    return create_app(app_config, db_engine, fake_queues)


@pytest.fixture
def client(app):
    with TestClient(app, base_url="https://testserver") as value:
        yield value


@pytest.fixture
def user(db_engine: Engine) -> User:
    with session_scope(db_engine) as session:
        value = User(username="alice", password_hash=hash_password("secret"), active=True)
        session.add(value)
        session.flush()
        return value


@pytest.fixture
def auth_client(client: TestClient, user: User) -> TestClient:
    login_page = client.get("/login")
    login_csrf = extract_csrf(login_page.text)
    response = client.post(
        "/login",
        data={"username": user.username, "password": "secret", "csrf_token": login_csrf},
        follow_redirects=False,
    )
    assert response.status_code == 303
    protected_page = client.get("/datasets")
    client.csrf = extract_csrf(protected_page.text)
    return client
