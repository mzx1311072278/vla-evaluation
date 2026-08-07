import csv
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from Genie02_report.genie02_eval_common import EPISODE_METRIC_FIELDS
from tests.fakes import FakeQueueBundle
from vla_eval.config import AppConfig, RemoteSource
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.security import hash_password
from vla_eval.web.app import create_app


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


@pytest.fixture
def app_config(data_root: Path) -> AppConfig:
    credentials = data_root / "credentials"
    credentials.mkdir()
    return AppConfig(
        data_root=data_root,
        database_url="sqlite://",
        redis_url="redis://unused.invalid/0",
        session_secret="test-session-secret",
        remote_sources={
            "lab-a": RemoteSource(
                name="lab-a",
                host="lab-a.example.test",
                port=22,
                username="reader",
                key_path=credentials / "lab-a-key",
                known_hosts_path=credentials / "known_hosts",
                roots=("/srv/datasets", "/srv/archive"),
            )
        },
        local_sources={},
    )


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


@pytest.fixture
def successful_job(
    db_engine: Engine,
    ready_dataset: Dataset,
    data_root: Path,
    user: User,
) -> EvaluationJob:
    output_dir = data_root / "runs" / "successful-job"
    output_dir.mkdir(parents=True)
    metrics = {
        "schema_version": "1.0",
        "session_id": "ready-dataset",
        "n_episodes": 1,
        "n_success": 1,
        "n_failure": 0,
        "gsr": 1.0,
        "mean_tts_success_s": 1.0,
        "smoothness": {
            "space": "joint",
            "left": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_episodes": 1},
            "right": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_episodes": 0,
            },
            "n_episodes": 1,
        },
    }
    (output_dir / "metrics_core.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EPISODE_METRIC_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "ready-dataset",
                "episode_index": 0,
                "outcome": "success",
                "duration_s": "1.000",
                "smoothness": "0",
                "left_smoothness": "0",
                "right_smoothness": "",
                "smoothness_space": "joint",
                "smoothness_frames": 4,
                "smoothness_skipped_reason": "",
            }
        )

    with session_scope(db_engine) as session:
        value = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="SUCCEEDED",
            stage="REPORT",
            progress=100.0,
            output_dir=str(output_dir),
            created_by=user.id,
        )
        session.add(value)
        session.flush()
        return value
