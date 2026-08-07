import stat
from types import SimpleNamespace
from urllib.parse import urlencode

import paramiko
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from vla_eval.db import session_scope
from vla_eval.models import ImportJob
from vla_eval.tasks import run_import_task


def _import_form(csrf: str, **overrides: str) -> dict[str, str]:
    values = {
        "csrf_token": csrf,
        "source_name": "lab-a",
        "root": "/srv/datasets",
        "relative_path": "team/run-01",
        "target_name": "team run-01",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("path", ["/imports", "/imports/new", "/imports/missing"])
def test_import_html_pages_require_login(client: TestClient, path: str):
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_import_rejects_unconfigured_source_without_database_or_queue_side_effects(
    auth_client, db_engine, fake_queues
):
    response = auth_client.post(
        "/imports",
        data=_import_form(auth_client.csrf, source_name="evil-host"),
    )

    assert response.status_code == 422
    assert fake_queues.transfer.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []


def test_import_page_groups_local_and_remote_sources(auth_client, app):
    response = auth_client.get("/imports/new")

    assert response.status_code == 200
    assert '<label for="source-name">数据来源' in response.text
    assert '<optgroup label="本机目录">' in response.text
    assert '<option value="this-host" data-kind="local">this-host</option>' in response.text
    assert '<optgroup label="远程 SSH">' in response.text
    assert '<option value="lab-a" data-kind="remote">lab-a</option>' in response.text
    local_root = str(app.state.config.local_sources["this-host"].roots[0])
    assert f'data-source="this-host" value="{local_root}"' in response.text


def test_valid_local_import_commits_job_then_enqueues_worker(
    auth_client, app, db_engine, fake_queues
):
    local_root = str(app.state.config.local_sources["this-host"].roots[0])
    response = auth_client.post(
        "/imports",
        data=_import_form(
            auth_client.csrf,
            source_name="this-host",
            root=local_root,
            relative_path="run-01",
            target_name="local run-01",
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/imports/")
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, job_id)
        assert job.source_name == "this-host"
        assert job.remote_root == local_root
        assert job.remote_path == "run-01"
        assert job.target_name == "local run-01"
        assert job.state == "QUEUED"
    assert fake_queues.transfer.count == 1


def test_local_import_rejects_root_from_another_source_before_enqueue(
    auth_client, db_engine, fake_queues
):
    response = auth_client.post(
        "/imports",
        data=_import_form(
            auth_client.csrf,
            source_name="this-host",
            root="/srv/datasets",
        ),
    )

    assert response.status_code == 422
    assert fake_queues.transfer.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "/srv/datasets/child"),
        ("relative_path", "../etc"),
        ("relative_path", "team//run"),
        ("target_name", ""),
        ("target_name", "."),
        ("target_name", ".."),
        ("target_name", "nested/run"),
        ("target_name", "nested\\run"),
        ("target_name", "run\u200bname"),
        ("target_name", "e\u0301"),
        ("target_name", "\u6570" * 86),
    ],
)
def test_import_rejects_invalid_trusted_fields(
    auth_client, db_engine, fake_queues, field: str, value: str
):
    response = auth_client.post("/imports", data=_import_form(auth_client.csrf, **{field: value}))

    assert response.status_code == 422
    assert fake_queues.transfer.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []


def test_import_rejects_duplicate_form_values(auth_client, db_engine, fake_queues):
    fields = list(_import_form(auth_client.csrf).items())
    fields.append(("target_name", "second"))

    response = auth_client.post(
        "/imports",
        content=urlencode(fields),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    assert fake_queues.transfer.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []


def test_import_requires_valid_csrf_before_creating_job(auth_client, db_engine, fake_queues):
    response = auth_client.post("/imports", data=_import_form("wrong-token"))

    assert response.status_code == 403
    assert fake_queues.transfer.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []


def test_valid_import_commits_job_then_enqueues_only_worker(auth_client, db_engine, fake_queues):
    response = auth_client.post(
        "/imports", data=_import_form(auth_client.csrf), follow_redirects=False
    )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/imports/")
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, job_id)
        assert job.source_name == "lab-a"
        assert job.remote_root == "/srv/datasets"
        assert job.remote_path == "team/run-01"
        assert job.target_name == "team run-01"
        assert job.state == "QUEUED"
    assert fake_queues.transfer.count == 1
    call = fake_queues.transfer.enqueued[0]
    assert call.function is run_import_task
    assert call.args == (job_id,)


def test_enqueue_failure_removes_unclaimed_job(auth_client, db_engine, fake_queues, monkeypatch):
    def fail_enqueue(*_args):
        raise RuntimeError("redis password=secret")

    monkeypatch.setattr(fake_queues.transfer, "enqueue", fail_enqueue)

    response = auth_client.post(
        "/imports", data=_import_form(auth_client.csrf), follow_redirects=False
    )

    assert response.status_code == 503
    with session_scope(db_engine) as session:
        assert session.scalars(select(ImportJob)).all() == []
    assert "secret" not in response.text


def test_ambiguous_enqueue_failure_does_not_clobber_job_claimed_by_worker(
    auth_client, db_engine, fake_queues, monkeypatch
):
    def claim_then_fail(_function, job_id):
        with session_scope(db_engine) as session:
            job = session.get_one(ImportJob, job_id)
            job.state = "CONNECTING"
            job.execution_token = "worker-token"
        raise RuntimeError("connection dropped after enqueue")

    monkeypatch.setattr(fake_queues.transfer, "enqueue", claim_then_fail)

    response = auth_client.post(
        "/imports", data=_import_form(auth_client.csrf), follow_redirects=False
    )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/imports/")
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, job_id)
        assert job.state == "CONNECTING"
        assert job.execution_token == "worker-token"


def test_import_pages_and_status_api_render_persisted_jobs(auth_client, db_engine):
    with session_scope(db_engine) as session:
        job = ImportJob(
            source_name="lab-a",
            remote_root="/srv/datasets",
            remote_path="team/a/very/long/path/run-01",
            target_name="run-01",
            state="TRANSFERRING",
            progress=42.5,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    listing = auth_client.get("/imports")
    creation = auth_client.get("/imports/new")
    detail = auth_client.get(f"/imports/{job_id}")
    status = auth_client.get(f"/api/imports/{job_id}")

    assert listing.status_code == creation.status_code == detail.status_code == 200
    assert job_id in listing.text
    assert "team/a/very/long/path/run-01" in detail.text
    assert "lab-a" in creation.text
    assert status.status_code == 200
    assert status.json() == {
        "id": job_id,
        "state": "TRANSFERRING",
        "progress": 42.5,
        "error_code": None,
        "error_message": None,
        "dataset_id": None,
        "finished": False,
    }
    assert "HX-Trigger" not in status.headers


def test_import_detail_shows_stage_progress(auth_client, db_engine):
    with session_scope(db_engine) as session:
        job = ImportJob(
            source_name="lab-a",
            remote_root="/srv/datasets",
            remote_path="team/run-01",
            target_name="run-01",
            state="VERIFYING",
            progress=80,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.get(f"/imports/{job_id}")

    assert response.status_code == 200
    for stage in ("\u8fde\u63a5", "\u4f20\u8f93", "\u9a8c\u8bc1", "\u9884检", "\u5b8c成"):
        assert stage in response.text
    assert 'hx-trigger="every 2s"' in response.text


@pytest.mark.parametrize("state", ["READY", "FAILED", "CANCELLED"])
def test_terminal_import_status_triggers_polling_completion(auth_client, db_engine, state: str):
    with session_scope(db_engine) as session:
        job = ImportJob(
            source_name="lab-a",
            remote_root="/srv/datasets",
            remote_path="team/run-01",
            target_name="run-01",
            state=state,
            progress=100,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.get(f"/api/imports/{job_id}")

    assert response.status_code == 200
    assert response.json()["finished"] is True
    assert response.headers["HX-Trigger"] == "job-finished"


@pytest.mark.parametrize("path", ["/imports/not-a-job", "/api/imports/not-a-job"])
def test_missing_import_is_not_found(auth_client, path: str):
    assert auth_client.get(path).status_code == 404


class FakeSftp:
    def __init__(self, entries=(), error: Exception | None = None):
        self.entries = list(entries)
        self.error = error
        self.listdir_calls: list[str] = []
        self.closed = False
        self.channel = SimpleNamespace(timeout=None, settimeout=self._set_channel_timeout)

    def _set_channel_timeout(self, timeout):
        self.channel.timeout = timeout

    def get_channel(self):
        return self.channel

    def listdir_attr(self, path: str):
        self.listdir_calls.append(path)
        if self.error is not None:
            raise self.error
        return self.entries

    def close(self):
        self.closed = True


class FakeSshClient:
    def __init__(self, sftp: FakeSftp):
        self.sftp = sftp
        self.loaded_host_keys: list[str] = []
        self.policy = None
        self.connect_kwargs = None
        self.closed = False

    def load_host_keys(self, filename: str):
        self.loaded_host_keys.append(filename)

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.closed = True


def _remote_directory(path: str):
    return SimpleNamespace(filename=path, st_mode=stat.S_IFDIR | 0o755)


def _remote_file(path: str):
    return SimpleNamespace(filename=path, st_mode=stat.S_IFREG | 0o644)


def test_remote_browser_rejects_traversal_before_connect_or_listdir(auth_client, app):
    fake_sftp = FakeSftp()
    clients = []

    def factory():
        client = FakeSshClient(fake_sftp)
        clients.append(client)
        return client

    app.state.ssh_client_factory = factory

    response = auth_client.get(
        "/api/remote-sources/lab-a/directories",
        params={"root": "/srv/datasets", "path": "../etc"},
    )

    assert response.status_code == 422
    assert clients == []
    assert fake_sftp.listdir_calls == []


def test_remote_browser_rejects_duplicate_security_parameters_before_connect(auth_client, app):
    fake_sftp = FakeSftp()
    clients = []
    app.state.ssh_client_factory = lambda: clients.append(FakeSshClient(fake_sftp)) or clients[-1]

    response = auth_client.get(
        "/api/remote-sources/lab-a/directories"
        "?root=/srv/datasets&root=/srv/archive&path=../etc&path=team"
    )

    assert response.status_code == 422
    assert clients == []
    assert fake_sftp.listdir_calls == []


@pytest.mark.parametrize(
    ("source_name", "root"),
    [("evil-host", "/srv/datasets"), ("lab-a", "/srv/datasets/child")],
)
def test_remote_browser_allows_only_configured_source_and_exact_root(
    auth_client, app, source_name: str, root: str
):
    fake_sftp = FakeSftp()
    clients = []
    app.state.ssh_client_factory = lambda: clients.append(FakeSshClient(fake_sftp)) or clients[-1]

    response = auth_client.get(
        f"/api/remote-sources/{source_name}/directories",
        params={"root": root, "path": ""},
    )

    assert response.status_code == 422
    assert clients == []
    assert fake_sftp.listdir_calls == []


def test_remote_browser_lists_only_current_level_directories_with_pinned_ssh(auth_client, app):
    fake_sftp = FakeSftp(
        [_remote_directory("zeta"), _remote_file("session.json"), _remote_directory("alpha")]
    )
    fake_client = FakeSshClient(fake_sftp)
    app.state.ssh_client_factory = lambda: fake_client

    response = auth_client.get(
        "/api/remote-sources/lab-a/directories",
        params={"root": "/srv/datasets", "path": "team"},
    )

    assert response.status_code == 200
    assert response.json() == {"directories": ["alpha", "zeta"], "path": "team"}
    source = app.state.config.remote_sources["lab-a"]
    assert fake_client.loaded_host_keys == [str(source.known_hosts_path)]
    assert isinstance(fake_client.policy, paramiko.RejectPolicy)
    assert fake_client.connect_kwargs == {
        "hostname": source.host,
        "port": source.port,
        "username": source.username,
        "key_filename": str(source.key_path),
        "look_for_keys": False,
        "allow_agent": False,
        "timeout": 10,
        "banner_timeout": 10,
        "auth_timeout": 10,
    }
    assert fake_sftp.channel.timeout == 10
    assert fake_sftp.listdir_calls == ["/srv/datasets/team"]
    assert fake_sftp.closed is True
    assert fake_client.closed is True


def test_remote_browser_empty_path_lists_root(auth_client, app):
    fake_sftp = FakeSftp()
    fake_client = FakeSshClient(fake_sftp)
    app.state.ssh_client_factory = lambda: fake_client

    response = auth_client.get(
        "/api/remote-sources/lab-a/directories",
        params={"root": "/srv/archive", "path": ""},
    )

    assert response.status_code == 200
    assert fake_sftp.listdir_calls == ["/srv/archive"]


def test_remote_browser_sanitizes_remote_errors_and_closes_client(auth_client, app):
    fake_sftp = FakeSftp(error=OSError("failed with /secret/key token=top-secret"))
    fake_client = FakeSshClient(fake_sftp)
    app.state.ssh_client_factory = lambda: fake_client

    response = auth_client.get(
        "/api/remote-sources/lab-a/directories",
        params={"root": "/srv/datasets", "path": "team"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Remote directory listing is unavailable"}
    assert "secret" not in response.text
    assert fake_sftp.closed is True
    assert fake_client.closed is True
