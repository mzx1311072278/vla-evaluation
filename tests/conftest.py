import csv
import json
from pathlib import Path

import fakeredis
import numpy as np
import pytest
from sqlalchemy import Engine

from tests.fakes import FakeQueueBundle
from vla_eval.datasets import inspect_dataset
from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.models import Dataset, EvaluationJob


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for name in ("inbox", "staging", "runs", "models", "db"):
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture
def fake_redis():
    connection = fakeredis.FakeRedis()
    yield connection
    connection.close()


@pytest.fixture
def db_engine() -> Engine:
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def fake_queues() -> FakeQueueBundle:
    return FakeQueueBundle.create()


@pytest.fixture
def dataset(db_engine: Engine, data_root: Path) -> Dataset:
    with session_scope(db_engine) as session:
        value = Dataset(
            name="dataset-pending",
            path=str(data_root / "inbox" / "dataset-pending"),
            kind="genie02_session",
            status="PENDING",
        )
        session.add(value)
        session.flush()
        return value


@pytest.fixture
def ready_dataset(db_engine: Engine, data_root: Path) -> Dataset:
    path = data_root / "inbox" / "ready-dataset"
    (path / "trajectories").mkdir(parents=True)
    (path / "session.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "session_id": "ready-dataset",
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
    with (path / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
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
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "ready-dataset",
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
    np.savez(path / "trajectories/episode_000.npz", action=np.ones((4, 3)))
    inspection = inspect_dataset(path, allowed_root=path)
    assert inspection.ready is True
    with session_scope(db_engine) as session:
        value = Dataset(
            name="ready-dataset",
            path=str(path),
            kind=inspection.kind.value,
            status="READY",
            fingerprint=inspection.fingerprint,
            size_bytes=inspection.size_bytes,
            episode_count=1,
        )
        session.add(value)
        session.flush()
        return value


@pytest.fixture
def evaluation_job(db_engine: Engine, ready_dataset: Dataset) -> EvaluationJob:
    with session_scope(db_engine) as session:
        value = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
        )
        session.add(value)
        session.flush()
        return value


def reload_job(engine: Engine, job_id: str) -> EvaluationJob:
    with session_scope(engine) as session:
        return session.get_one(EvaluationJob, job_id)
