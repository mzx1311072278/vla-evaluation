from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import JSON, BigInteger, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

import vla_eval.db as db_module
from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.models import Dataset, EvaluationJob, ImportJob, User


def test_database_persists_dataset_and_job(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with session_scope(engine) as session:
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        session.add(dataset)
        session.flush()
        session.add(EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full"))

    with session_scope(engine) as session:
        job = session.scalar(select(EvaluationJob))
        assert job is not None
        assert job.state == "QUEUED"
        assert job.dataset.name == "run-1"


def test_model_defaults_use_uuid_utc_and_independent_json(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with session_scope(engine) as session:
        user = User(username="admin", password_hash="hash")
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        other_dataset = Dataset(
            name="run-2", path="/data/run-2", kind="lerobot", status="READY"
        )
        import_job = ImportJob(
            source_name="lab-a",
            remote_path="/data/rollouts/run-1",
            target_name="run-1",
        )
        session.add_all([user, dataset, other_dataset, import_job])
        session.flush()
        jobs = [
            EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full"),
            EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full"),
        ]
        session.add_all(jobs)
        session.flush()

        for model in (user, dataset, other_dataset, import_job, *jobs):
            assert str(UUID(model.id)) == model.id
            assert model.created_at.utcoffset() == timedelta(0)
            assert model.updated_at.utcoffset() == timedelta(0)

        assert user.is_admin is False
        assert user.active is True
        assert dataset.size_bytes == 0
        assert dataset.episode_count == 0
        assert import_job.state == "QUEUED"
        assert import_job.progress == 0.0
        assert jobs[0].profile_version == "unknown"
        assert jobs[0].vlm_enabled is False
        assert jobs[0].stage == "PENDING"
        assert jobs[0].progress == 0.0
        assert jobs[0].cancel_requested is False
        assert dataset.inspection_json == {}
        assert dataset.inspection_json is not other_dataset.inspection_json
        assert jobs[0].params_json == {}
        assert jobs[0].provenance_json == {}
        assert jobs[0].params_json is not jobs[1].params_json
        assert jobs[0].provenance_json is not jobs[1].provenance_json


def test_json_fields_use_sqlalchemy_json_columns():
    dataset_columns = inspect(Dataset).columns
    evaluation_columns = inspect(EvaluationJob).columns

    assert isinstance(dataset_columns.inspection_json.type, JSON)
    assert isinstance(evaluation_columns.params_json.type, JSON)
    assert isinstance(evaluation_columns.provenance_json.type, JSON)


def test_dataset_size_uses_big_integer_and_persists_large_values(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    large_size = 2**40

    assert isinstance(inspect(Dataset).columns.size_bytes.type, BigInteger)

    with session_scope(engine) as session:
        session.add(
            Dataset(
                name="large-run",
                path="/data/large-run",
                kind="lerobot",
                status="READY",
                size_bytes=large_size,
            )
        )

    with session_scope(engine) as session:
        assert session.scalar(select(Dataset.size_bytes)) == large_size


def test_json_in_place_updates_are_persisted(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with session_scope(engine) as session:
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        session.add(dataset)
        session.flush()
        job = EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full")
        session.add(job)
        session.flush()
        dataset_id = dataset.id
        job_id = job.id

    with session_scope(engine) as session:
        dataset = session.get_one(Dataset, dataset_id)
        job = session.get_one(EvaluationJob, job_id)
        dataset.inspection_json["checked"] = True
        job.params_json["temperature"] = 0.2
        job.provenance_json["git_sha"] = "abc123"

    with session_scope(engine) as session:
        dataset = session.get_one(Dataset, dataset_id)
        job = session.get_one(EvaluationJob, job_id)
        assert dataset.inspection_json == {"checked": True}
        assert job.params_json == {"temperature": 0.2}
        assert job.provenance_json == {"git_sha": "abc123"}


def test_evaluation_job_dataset_id_is_indexed():
    indexes = inspect(EvaluationJob).local_table.indexes

    assert any(tuple(index.columns.keys()) == ("dataset_id",) for index in indexes)


@pytest.mark.parametrize("url", ["sqlite://", "sqlite+pysqlite:///:memory:"])
def test_in_memory_sqlite_schema_and_data_are_visible_across_threads(url):
    engine = create_engine_for_url(url)
    init_db(engine)

    with session_scope(engine) as session:
        session.add(User(username="shared", password_hash="hash"))

    def read_username() -> str | None:
        with session_scope(engine) as session:
            return session.scalar(select(User.username))

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(read_username).result() == "shared"


def test_session_scope_keeps_new_model_scalars_available_after_commit(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with session_scope(engine) as session:
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        session.add(dataset)
        session.flush()
        job = EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full")
        session.add(job)

    assert str(UUID(dataset.id)) == dataset.id
    assert job.state == "QUEUED"


def test_session_scope_keeps_explicitly_loaded_data_available_after_commit(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with session_scope(engine) as session:
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        session.add(dataset)
        session.flush()
        session.add(EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full"))

    with session_scope(engine) as session:
        job = session.scalar(
            select(EvaluationJob).options(selectinload(EvaluationJob.dataset))
        )
        assert job is not None

    assert job.state == "QUEUED"
    assert job.dataset.name == "run-1"


def test_sqlite_enables_wal_and_foreign_keys(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one().lower() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

    with pytest.raises(IntegrityError), session_scope(engine) as session:
        session.add(EvaluationJob(dataset_id="missing", profile_name="genie02-full"))


def test_session_scope_rolls_back_on_error(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)

    with pytest.raises(RuntimeError, match="abort"), session_scope(engine) as session:
        session.add(User(username="rolled-back", password_hash="hash"))
        session.flush()
        raise RuntimeError("abort")

    with session_scope(engine) as session:
        assert session.scalar(select(User).where(User.username == "rolled-back")) is None


def test_non_sqlite_engine_does_not_receive_sqlite_connect_args(monkeypatch):
    received: dict[str, object] = {}

    def fake_create_engine(url, **kwargs):
        received["url"] = url
        received.update(kwargs)
        return object()

    monkeypatch.setattr(db_module, "create_engine", fake_create_engine)

    engine = create_engine_for_url("postgresql://db.example/vla_eval")

    assert engine is not None
    assert str(received.pop("url")) == "postgresql://db.example/vla_eval"
    assert received == {}
