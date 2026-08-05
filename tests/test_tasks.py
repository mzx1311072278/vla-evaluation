from pathlib import Path

import pytest
from sqlalchemy import select

from tests.conftest import reload_job
from vla_eval.config import AppConfig, RemoteSource
from vla_eval.datasets import DatasetInspection, DatasetKind
from vla_eval.db import session_scope
from vla_eval.exceptions import EvaluationCancelled
from vla_eval.import_jobs import ImportResult, TransferError
from vla_eval.models import Dataset, EvaluationJob, ImportJob
from vla_eval.queueing import create_queues
from vla_eval.tasks import (
    TaskRuntime,
    clear_runtime,
    configure_runtime,
    recover_interrupted_jobs,
    run_evaluation_task,
    run_import_task,
)


def test_queue_names_are_isolated(fake_redis):
    queues = create_queues("redis://unused", connection=fake_redis)

    assert queues.transfer.name == "transfers"
    assert queues.evaluation.name == "evaluations"
    assert queues.transfer.connection is fake_redis
    assert queues.evaluation.connection is fake_redis


def test_task_entry_uses_explicitly_configured_worker_runtime(
    db_engine, data_root, evaluation_job, monkeypatch
):
    clear_runtime()
    monkeypatch.setattr(
        "vla_eval.tasks.run_evaluation",
        lambda **kwargs: kwargs["callbacks"].on_stage("REPORT"),
    )

    with pytest.raises(RuntimeError, match="runtime has not been configured"):
        run_evaluation_task(evaluation_job.id)

    configure_runtime(_runtime(db_engine, data_root))
    try:
        run_evaluation_task(evaluation_job.id)
    finally:
        clear_runtime()

    assert reload_job(db_engine, evaluation_job.id).state == "SUCCEEDED"


def _runtime(db_engine, data_root: Path) -> TaskRuntime:
    credentials_root = data_root / "credentials"
    source = RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=credentials_root / "lab-a-key",
        known_hosts_path=credentials_root / "known_hosts",
        roots=("/data/rollouts",),
    )
    return TaskRuntime(
        engine=db_engine,
        config=AppConfig(
            data_root=data_root,
            database_url="sqlite://",
            redis_url="redis://unused",
            session_secret="test-secret",
            remote_sources={source.name: source},
        ),
        profiles_root=Path("config/profiles"),
        credentials_root=credentials_root,
    )


def test_evaluation_task_records_sanitized_failure_and_reraises(
    db_engine, data_root, evaluation_job, monkeypatch
):
    error = RuntimeError(
        f"boom at {data_root}/private; token=top-secret\nTraceback (most recent call last)"
    )

    def fail(**_kwargs):
        raise error

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", fail)

    with pytest.raises(RuntimeError) as raised:
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is error
    job = reload_job(db_engine, evaluation_job.id)
    assert job.state == "FAILED"
    assert job.error_code == "EVALUATION_FAILED"
    assert job.error_message == "Evaluation failed. Review worker logs for details."
    assert str(data_root) not in job.error_message
    assert "top-secret" not in job.error_message
    assert "Traceback" not in job.error_message


def test_evaluation_task_commits_callbacks_and_records_success(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.error_code = "OLD_ERROR"
        job.error_message = "old failure"
    result = object()
    received = {}

    def evaluate(**kwargs):
        received.update(kwargs)
        callbacks = kwargs["callbacks"]
        callbacks.on_stage("METRICS")
        with session_scope(db_engine) as session:
            persisted = session.get_one(EvaluationJob, evaluation_job.id)
            assert (persisted.state, persisted.stage) == ("RUNNING", "METRICS")
            assert (persisted.error_code, persisted.error_message) == (None, None)
        callbacks.on_progress(30.0)
        with session_scope(db_engine) as session:
            assert session.get_one(EvaluationJob, evaluation_job.id).progress == 30.0
        callbacks.on_stage("REPORT")
        callbacks.on_progress(100.0)
        return result

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)

    actual = run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert actual is result
    assert received["dataset_path"] == str(data_root / "inbox" / "ready-dataset")
    assert received["output_dir"] == str(data_root / "runs" / evaluation_job.id)
    assert received["resume_from"] == "METRICS"
    assert received["initial_progress"] == 0.0
    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.stage, job.progress) == ("SUCCEEDED", "REPORT", 100.0)


@pytest.mark.parametrize("failed_stage", ["VLM", "REPORT"])
def test_evaluation_retry_resumes_from_persisted_failed_stage(
    db_engine, data_root, evaluation_job, monkeypatch, failed_stage
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "INTERRUPTED"
        job.stage = failed_stage
        job.progress = 63.0
        job.vlm_enabled = True

    received = {}

    def evaluate(**kwargs):
        received.update(kwargs)
        kwargs["callbacks"].on_stage("REPORT")

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)

    run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert received["resume_from"] == failed_stage
    assert received["initial_progress"] == 63.0


def test_evaluation_retry_uses_core_artifact_validation_before_skipping(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "INTERRUPTED"
        job.stage = "REPORT"
        job.progress = 80.0

    monkeypatch.setattr(
        "vla_eval.evaluation.generate_episode_metrics",
        lambda *_args: pytest.fail("METRICS must not rerun for a REPORT resume"),
    )

    with pytest.raises(ValueError, match="missing required artifacts"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert reload_job(db_engine, evaluation_job.id).state == "FAILED"


def test_evaluation_task_rejects_changed_profile_version(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        session.get_one(EvaluationJob, evaluation_job.id).profile_version = "9.9.9"
    monkeypatch.setattr(
        "vla_eval.tasks.run_evaluation",
        lambda **_kwargs: pytest.fail("changed profile must not execute"),
    )

    with pytest.raises(ValueError, match="profile version changed"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert reload_job(db_engine, evaluation_job.id).state == "FAILED"


def test_evaluation_task_records_cancelled_and_reraises(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        session.get_one(EvaluationJob, evaluation_job.id).cancel_requested = True

    monkeypatch.setattr(
        "vla_eval.tasks.run_evaluation",
        lambda **_kwargs: pytest.fail("cancelled job must not enter evaluation core"),
    )

    with pytest.raises(EvaluationCancelled, match="cancelled before execution"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    job = reload_job(db_engine, evaluation_job.id)
    assert job.state == "CANCELLED"
    assert job.error_code == "EVALUATION_CANCELLED"
    assert job.error_message == "Evaluation was cancelled."


def test_evaluation_task_preserves_core_and_failure_persistence_errors(
    db_engine, data_root, evaluation_job, monkeypatch
):
    core_error = RuntimeError("core failed")
    persistence_error = OSError("database unavailable")

    def fail_core(**_kwargs):
        raise core_error

    def fail_persistence(*_args):
        raise persistence_error

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", fail_core)
    monkeypatch.setattr("vla_eval.tasks._record_evaluation_failure", fail_persistence)

    with pytest.raises(BaseExceptionGroup) as raised:
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value.exceptions == (core_error, persistence_error)


def test_evaluation_task_maps_final_persistence_failure_to_failed(
    db_engine, data_root, evaluation_job, monkeypatch
):
    persistence_error = RuntimeError("final persistence failed")

    def evaluate(**kwargs):
        kwargs["callbacks"].on_stage("REPORT")

    def fail_success(*_args):
        raise persistence_error

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)
    monkeypatch.setattr("vla_eval.tasks._record_evaluation_success", fail_success)

    with pytest.raises(RuntimeError) as raised:
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is persistence_error
    assert reload_job(db_engine, evaluation_job.id).state == "FAILED"


def _create_import_job(db_engine, *, cancel_requested: bool = False) -> ImportJob:
    with session_scope(db_engine) as session:
        job = ImportJob(
            source_name="lab-a",
            remote_path="run-1",
            target_name="run-1",
            cancel_requested=cancel_requested,
        )
        session.add(job)
        session.flush()
        return job


def test_import_task_commits_callbacks_and_persists_ready_dataset(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine)
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        job.error_code = "OLD_ERROR"
        job.error_message = "old failure"
    inspection = DatasetInspection(
        kind=DatasetKind.GENIE02_SESSION,
        ready=True,
        fingerprint="a" * 64,
        size_bytes=123,
        episode_count=4,
        errors=(),
    )
    received = {}

    def execute(spec, *, callbacks):
        received["spec"] = spec
        callbacks.on_state("CONNECTING")
        with session_scope(db_engine) as session:
            persisted = session.get_one(ImportJob, import_job.id)
            assert (persisted.error_code, persisted.error_message) == (None, None)
        callbacks.on_state("TRANSFERRING")
        callbacks.on_progress(45.0)
        with session_scope(db_engine) as session:
            persisted = session.get_one(ImportJob, import_job.id)
            assert (persisted.state, persisted.progress) == ("TRANSFERRING", 45.0)
        callbacks.on_state("PREFLIGHT")
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr("vla_eval.tasks.execute_import", execute)

    result = run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert result.inspection is inspection
    spec = received["spec"]
    assert spec.job_id == import_job.id
    assert spec.source_name == "lab-a"
    assert spec.remote_root == "/data/rollouts"
    assert spec.remote_relative_path == "run-1"
    assert spec.staging_path == data_root / "staging" / import_job.id
    assert spec.target_path == data_root / "inbox" / "run-1"
    assert spec.mode == "production"
    assert spec.trusted_credentials_root == data_root / "credentials"
    with session_scope(db_engine) as session:
        persisted_job = session.get_one(ImportJob, import_job.id)
        persisted_dataset = session.scalar(select(Dataset))
        assert persisted_dataset is not None
        assert persisted_job.dataset_id == persisted_dataset.id
        assert (persisted_job.state, persisted_job.progress) == ("READY", 100.0)
        assert persisted_dataset.status == "READY"
        assert persisted_dataset.path == str(data_root / "inbox" / "run-1")
        assert persisted_dataset.kind == DatasetKind.GENIE02_SESSION.value
        assert persisted_dataset.fingerprint == "a" * 64
        assert persisted_dataset.size_bytes == 123
        assert persisted_dataset.episode_count == 4


def test_import_task_records_sanitized_failure_and_reraises(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    error = TransferError(f"network failed at {data_root}/credentials/lab-a-key token=top-secret")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("vla_eval.tasks.execute_import", fail)

    with pytest.raises(TransferError) as raised:
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is error
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert job.state == "FAILED"
        assert job.error_code == "IMPORT_TRANSFER_FAILED"
        assert job.error_message == "Dataset transfer failed. Retry the import."
        assert str(data_root) not in job.error_message
        assert "top-secret" not in job.error_message


def test_import_task_records_cancelled_without_entering_transfer_core(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine, cancel_requested=True)
    monkeypatch.setattr(
        "vla_eval.tasks.execute_import",
        lambda *_args, **_kwargs: pytest.fail("cancelled import must not enter transfer core"),
    )

    with pytest.raises(TransferError, match="cancelled before execution"):
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert job.state == "CANCELLED"
        assert job.error_code == "IMPORT_CANCELLED"
        assert job.error_message == "Dataset import was cancelled."


def test_import_task_preserves_core_and_failure_persistence_errors(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine)
    core_error = RuntimeError("import core failed")
    persistence_error = OSError("database unavailable")

    def fail_core(*_args, **_kwargs):
        raise core_error

    def fail_persistence(*_args):
        raise persistence_error

    monkeypatch.setattr("vla_eval.tasks.execute_import", fail_core)
    monkeypatch.setattr("vla_eval.tasks._record_import_failure", fail_persistence)

    with pytest.raises(BaseExceptionGroup) as raised:
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value.exceptions == (core_error, persistence_error)


def test_import_task_records_configuration_failure(db_engine, data_root):
    import_job = _create_import_job(db_engine)
    runtime = TaskRuntime(
        engine=db_engine,
        config=AppConfig(
            data_root=data_root,
            database_url="sqlite://",
            redis_url="redis://unused",
            session_secret="test-secret",
            remote_sources={},
        ),
        profiles_root=Path("config/profiles"),
        credentials_root=data_root / "credentials",
    )

    with pytest.raises(KeyError, match="lab-a"):
        run_import_task(import_job.id, runtime=runtime)

    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert job.state == "FAILED"
        assert job.error_code == "IMPORT_FAILED"


def test_import_task_maps_final_persistence_failure_to_failed(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(
        DatasetKind.LEROBOT,
        True,
        "b" * 64,
        10,
        1,
        (),
    )
    persistence_error = RuntimeError("final persistence failed")

    def execute(spec, *, callbacks):
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    def fail_success(*_args):
        raise persistence_error

    monkeypatch.setattr("vla_eval.tasks.execute_import", execute)
    monkeypatch.setattr("vla_eval.tasks._record_import_success", fail_success)

    with pytest.raises(RuntimeError) as raised:
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is persistence_error
    with session_scope(db_engine) as session:
        assert session.get_one(ImportJob, import_job.id).state == "FAILED"


def test_recover_interrupted_jobs_is_selective_idempotent_and_keeps_staging(
    db_engine, data_root, ready_dataset
):
    evaluation_states = [
        "RUNNING",
        "METRICS",
        "VLM",
        "REPORT",
        "QUEUED",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
    ]
    import_states = [
        "TRANSFERRING",
        "QUEUED",
        "READY",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
    ]
    with session_scope(db_engine) as session:
        evaluations = [
            EvaluationJob(
                dataset_id=ready_dataset.id,
                profile_name="genie02-full",
                state=state,
                stage="VLM" if state == "RUNNING" else "METRICS",
            )
            for state in evaluation_states
        ]
        imports = [
            ImportJob(
                source_name="lab-a",
                remote_path=f"run-{index}",
                target_name=f"run-{index}",
                state=state,
            )
            for index, state in enumerate(import_states)
        ]
        session.add_all([*evaluations, *imports])
        session.flush()
        evaluation_ids = [job.id for job in evaluations]
        import_ids = [job.id for job in imports]

    staging_marker = data_root / "staging" / import_ids[0] / "partial.bin"
    staging_marker.parent.mkdir(parents=True)
    staging_marker.write_bytes(b"partial")
    runtime = _runtime(db_engine, data_root)

    assert recover_interrupted_jobs(runtime=runtime) == 5
    assert recover_interrupted_jobs(runtime=runtime) == 0

    with session_scope(db_engine) as session:
        actual_evaluations = [
            session.get_one(EvaluationJob, job_id).state for job_id in evaluation_ids
        ]
        actual_imports = [session.get_one(ImportJob, job_id).state for job_id in import_ids]
        assert session.get_one(EvaluationJob, evaluation_ids[0]).stage == "VLM"
    assert actual_evaluations == [
        "INTERRUPTED",
        "INTERRUPTED",
        "INTERRUPTED",
        "INTERRUPTED",
        "QUEUED",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
    ]
    assert actual_imports == [
        "INTERRUPTED",
        "QUEUED",
        "READY",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
    ]
    assert staging_marker.read_bytes() == b"partial"
