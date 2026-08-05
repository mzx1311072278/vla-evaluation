from pathlib import Path

import fakeredis
import pytest
from sqlalchemy import Engine

from tests.fakes import FakeQueueBundle
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
    path.mkdir(parents=True)
    with session_scope(db_engine) as session:
        value = Dataset(
            name="ready-dataset",
            path=str(path),
            kind="genie02_session",
            status="READY",
            fingerprint="f" * 64,
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
