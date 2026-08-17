import errno
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import select

import vla_eval.import_jobs as import_jobs_module
import vla_eval.tasks as tasks_module
from tests.conftest import reload_job
from vla_eval.config import AppConfig, LocalSource, RemoteSource
from vla_eval.datasets import DatasetInspection, DatasetKind
from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.exceptions import EvaluationCancelled, ModelLoadError
from vla_eval.import_jobs import ImportResult, TransferError
from vla_eval.models import Dataset, EvaluationJob, ImportJob
from vla_eval.queueing import create_queues
from vla_eval.tasks import (
    ImportIntegrityError,
    StaleTaskExecution,
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


def test_configure_runtime_warns_when_shared_storage_boundary_is_active(
    db_engine, data_root, caplog
):
    runtime = _runtime(
        db_engine,
        data_root,
        storage_trust_mode="data_root_boundary",
    )

    with caplog.at_level(logging.WARNING, logger=tasks_module.__name__):
        configure_runtime(runtime)
    try:
        assert "storage_trust_mode=data_root_boundary" in caplog.text
        assert str(data_root) in caplog.text
        assert "delegated to the storage platform" in caplog.text
    finally:
        clear_runtime()


def _runtime(
    db_engine,
    data_root: Path,
    *,
    remote_roots: tuple[str, ...] = ("/data/rollouts",),
    local_root: Path | None = None,
    storage_trust_mode: str = "strict",
) -> TaskRuntime:
    credentials_root = data_root / "credentials"
    source = RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=credentials_root / "lab-a-key",
        known_hosts_path=credentials_root / "known_hosts",
        roots=remote_roots,
    )
    return TaskRuntime(
        engine=db_engine,
        config=AppConfig(
            data_root=data_root,
            database_url="sqlite://",
            redis_url="redis://unused",
            session_secret="test-secret",
            remote_sources={source.name: source},
            local_sources=(
                {
                    "this-host": LocalSource(
                        name="this-host",
                        roots=(local_root,),
                    )
                }
                if local_root is not None
                else {}
            ),
            storage_trust_mode=storage_trust_mode,
        ),
        profiles_root=Path("config/profiles"),
        credentials_root=credentials_root,
    )


@pytest.mark.parametrize("operation", ["stage", "progress", "success", "failure", "cancel"])
def test_stale_evaluation_callbacks_cannot_overwrite_terminal_job(
    db_engine, data_root, evaluation_job, operation
):
    runtime = _runtime(db_engine, data_root)
    token = str(uuid4())
    tasks_module._claim_evaluation_execution(runtime, evaluation_job.id, token)
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "SUCCEEDED"
        job.execution_token = None
        job.progress = 100.0

    if operation == "stage":
        callback = lambda: tasks_module._update_evaluation_stage(
            runtime, evaluation_job.id, token, "REPORT"
        )
    elif operation == "progress":
        callback = lambda: tasks_module._update_evaluation_progress(
            runtime, evaluation_job.id, token, 12.0
        )
    elif operation == "success":
        callback = lambda: tasks_module._record_evaluation_success(
            runtime, evaluation_job.id, token
        )
    elif operation == "failure":
        callback = lambda: tasks_module._record_evaluation_failure(
            runtime, evaluation_job.id, token, RuntimeError("stale")
        )
    else:
        callback = lambda: tasks_module._record_evaluation_cancelled(
            runtime, evaluation_job.id, token
        )

    if operation in {"failure", "cancel"}:
        callback()
    else:
        with pytest.raises(tasks_module.StaleTaskExecution):
            callback()

    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.stage, job.progress, job.execution_token) == (
        "SUCCEEDED",
        "PENDING",
        100.0,
        None,
    )


def test_evaluation_success_cancel_race_commits_cancelled(db_engine, data_root, evaluation_job):
    runtime = _runtime(db_engine, data_root)
    token = str(uuid4())
    tasks_module._claim_evaluation_execution(runtime, evaluation_job.id, token)
    ready_barrier = Barrier(2)
    commit_barrier = Barrier(2)

    def commit_success():
        ready_barrier.wait(timeout=5)
        commit_barrier.wait(timeout=5)
        return tasks_module._record_evaluation_success(runtime, evaluation_job.id, token)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(commit_success)
        ready_barrier.wait(timeout=5)
        with session_scope(db_engine) as session:
            session.get_one(EvaluationJob, evaluation_job.id).cancel_requested = True
        commit_barrier.wait(timeout=5)
        with pytest.raises(EvaluationCancelled, match="before success commit"):
            future.result(timeout=5)

    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.execution_token) == ("CANCELLED", None)


def test_stale_import_callbacks_cannot_overwrite_ready_job(db_engine, data_root):
    runtime = _runtime(db_engine, data_root)
    import_job = _create_import_job(db_engine)
    token = str(uuid4())
    tasks_module._claim_import_execution(runtime, import_job.id, token)
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        job.state = "READY"
        job.execution_token = None
        job.progress = 100.0

    with pytest.raises(tasks_module.StaleTaskExecution):
        tasks_module._update_import_state(runtime, import_job.id, token, "CONNECTING")
    with pytest.raises(tasks_module.StaleTaskExecution):
        tasks_module._update_import_progress(runtime, import_job.id, token, 5.0)
    tasks_module._record_import_failure(
        runtime,
        import_job.id,
        token,
        TransferError("stale failure"),
    )
    tasks_module._record_import_cancelled(runtime, import_job.id, token)

    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.progress, job.execution_token) == ("READY", 100.0, None)


def test_import_claim_crash_is_recoverable(db_engine, data_root):
    job = _create_import_job(db_engine)
    runtime = _runtime(db_engine, data_root)
    tasks_module._claim_import_execution(runtime, job.id, str(uuid4()))

    assert recover_interrupted_jobs(runtime=runtime) == 1
    with session_scope(db_engine) as session:
        recovered = session.get_one(ImportJob, job.id)
        assert (recovered.state, recovered.execution_token) == ("INTERRUPTED", None)

    tasks_module._claim_import_execution(runtime, job.id, str(uuid4()))


@pytest.mark.parametrize("task_kind", ["evaluation", "import"])
def test_running_job_cannot_start_duplicate_core(
    db_engine, data_root, evaluation_job, monkeypatch, task_kind
):
    if task_kind == "evaluation":
        job_id = evaluation_job.id
        with session_scope(db_engine) as session:
            job = session.get_one(EvaluationJob, job_id)
            job.state = "RUNNING"
            job.execution_token = str(uuid4())
        monkeypatch.setattr(
            tasks_module,
            "run_evaluation",
            lambda **_kwargs: pytest.fail("duplicate evaluation must not enter core"),
        )
        entry = run_evaluation_task
    else:
        job = _create_import_job(db_engine)
        job_id = job.id
        with session_scope(db_engine) as session:
            persisted = session.get_one(ImportJob, job_id)
            persisted.state = "CONNECTING"
            persisted.execution_token = str(uuid4())
        monkeypatch.setattr(
            tasks_module,
            "execute_import",
            lambda *_args, **_kwargs: pytest.fail("duplicate import must not enter core"),
        )
        entry = run_import_task

    with pytest.raises(StaleTaskExecution, match="claim"):
        entry(job_id, runtime=_runtime(db_engine, data_root))


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


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_message"),
    [
        (
            OSError(errno.ENOSPC, "secret disk path"),
            "DISK_FULL",
            "Evaluation storage is full. Free space and retry.",
        ),
        (
            type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})(
                "secret tensor shape"
            ),
            "CUDA_OUT_OF_MEMORY",
            "GPU memory was exhausted. Reduce workload or retry.",
        ),
        (
            ModelLoadError("secret model path"),
            "MODEL_LOAD_FAILED",
            "The configured model could not be loaded. Review worker logs.",
        ),
    ],
    ids=["disk_full", "out_of_memory", "model_load"],
)
def test_evaluation_failure_classification_is_safe_and_actionable(
    db_engine,
    data_root,
    evaluation_job,
    monkeypatch,
    error,
    expected_code,
    expected_message,
):
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error)) as raised:
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is error
    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.error_code, job.error_message) == (
        "FAILED",
        expected_code,
        expected_message,
    )
    assert "secret" not in job.error_message


def test_model_words_do_not_trigger_model_load_classification():
    error = RuntimeError("dataset model metadata failed to load")

    assert tasks_module._classify_evaluation_failure(error)[0] == "EVALUATION_FAILED"


def test_evaluation_rejects_dataset_outside_trusted_path(
    db_engine, data_root, evaluation_job, monkeypatch
):
    outside = data_root / "outside-dataset"
    outside.mkdir()
    with session_scope(db_engine) as session:
        dataset = session.get_one(Dataset, evaluation_job.dataset_id)
        dataset.path = str(outside)
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: pytest.fail("untrusted dataset must not reach evaluation core"),
    )

    with pytest.raises(ValueError, match="dataset identity changed"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert list(outside.iterdir()) == []
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        dataset = session.get_one(Dataset, evaluation_job.dataset_id)
        assert (job.state, job.error_code) == ("FAILED", "DATASET_CHANGED")
        assert dataset.status == "PREFLIGHT_FAILED"


@pytest.mark.parametrize("case", ["persisted_outside", "canonical_symlink"])
def test_evaluation_rejects_untrusted_output_path(
    db_engine, data_root, evaluation_job, monkeypatch, case
):
    outside = data_root / "outside-output"
    if case == "persisted_outside":
        with session_scope(db_engine) as session:
            session.get_one(EvaluationJob, evaluation_job.id).output_dir = str(outside)
    else:
        outside.mkdir()
        (data_root / "runs" / evaluation_job.id).symlink_to(outside, target_is_directory=True)

    def write_outside(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "escaped.marker").write_text("unsafe", encoding="utf-8")

    monkeypatch.setattr(tasks_module, "run_evaluation", write_outside)

    with pytest.raises(ValueError, match="output"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert not (outside / "escaped.marker").exists()


@pytest.mark.parametrize("case", ["path_escape", "name_mismatch"])
def test_evaluation_rejects_unsafe_profile_selector(
    db_engine, data_root, evaluation_job, monkeypatch, case
):
    profiles_root = data_root / "profiles"
    profiles_root.mkdir(mode=0o700)
    source = Path("config/profiles/genie02-full.yaml").read_text(encoding="utf-8")
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        if case == "path_escape":
            job.profile_name = "../escaped-profile"
            (data_root / "escaped-profile.yaml").write_text(source, encoding="utf-8")
        else:
            job.profile_name = "selected-profile"
            mismatched = source.replace("name: genie02-full", "name: different-profile", 1)
            (profiles_root / "selected-profile.yaml").write_text(mismatched, encoding="utf-8")
    runtime = replace(_runtime(db_engine, data_root), profiles_root=profiles_root)
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: pytest.fail("unsafe profile must not reach evaluation core"),
    )

    with pytest.raises(ValueError, match="profile"):
        run_evaluation_task(evaluation_job.id, runtime=runtime)


def test_evaluation_accepts_read_only_trusted_profile_root(
    db_engine, data_root, evaluation_job, monkeypatch
):
    profiles_root = data_root / "read-only-profiles"
    profiles_root.mkdir(mode=0o700)
    profile_path = profiles_root / "genie02-full.yaml"
    profile_path.write_text(
        Path("config/profiles/genie02-full.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    profile_path.chmod(0o444)
    profiles_root.chmod(0o555)
    runtime = replace(_runtime(db_engine, data_root), profiles_root=profiles_root)
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **kwargs: kwargs["callbacks"].on_stage("REPORT"),
    )

    run_evaluation_task(evaluation_job.id, runtime=runtime)

    assert reload_job(db_engine, evaluation_job.id).state == "SUCCEEDED"


def test_evaluation_accepts_data_root_profile_under_writable_shared_parent(
    db_engine, data_root, evaluation_job, monkeypatch
):
    profiles_root = data_root / "profiles"
    profiles_root.mkdir(mode=0o700)
    profile_path = profiles_root / "genie02-full.yaml"
    profile_path.write_text(
        Path("config/profiles/genie02-full.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    profile_path.chmod(0o400)
    shared_parent = data_root.parent
    original_mode = shared_parent.stat().st_mode & 0o777
    shared_parent.chmod(0o777)
    runtime = replace(
        _runtime(
            db_engine,
            data_root,
            storage_trust_mode="data_root_boundary",
        ),
        profiles_root=profiles_root,
    )
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **kwargs: kwargs["callbacks"].on_stage("REPORT"),
    )

    try:
        run_evaluation_task(evaluation_job.id, runtime=runtime)
    finally:
        shared_parent.chmod(original_mode)

    assert reload_job(db_engine, evaluation_job.id).state == "SUCCEEDED"


def test_evaluation_keeps_external_profile_root_strict_in_boundary_mode(
    db_engine, data_root, evaluation_job, monkeypatch
):
    shared_parent = data_root.parent
    profiles_root = shared_parent / "external-profiles"
    profiles_root.mkdir(mode=0o700)
    (profiles_root / "genie02-full.yaml").write_text(
        Path("config/profiles/genie02-full.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    original_mode = shared_parent.stat().st_mode & 0o777
    shared_parent.chmod(0o777)
    runtime = replace(
        _runtime(
            db_engine,
            data_root,
            storage_trust_mode="data_root_boundary",
        ),
        profiles_root=profiles_root,
    )
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: pytest.fail("untrusted profile must not reach evaluation core"),
    )

    try:
        with pytest.raises(ValueError, match="profiles root.*group or other writable"):
            run_evaluation_task(evaluation_job.id, runtime=runtime)
    finally:
        shared_parent.chmod(original_mode)


@pytest.mark.parametrize("resume_stage", ["METRICS", "VLM", "REPORT"])
def test_evaluation_dataset_changed_fails_before_any_resume_stage(
    db_engine, data_root, evaluation_job, monkeypatch, resume_stage
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.stage = resume_stage
        if resume_stage != "METRICS":
            job.state = "INTERRUPTED"
        dataset = session.get_one(Dataset, job.dataset_id)
        dataset_id = dataset.id
        dataset_path = Path(dataset.path)
    trajectory = dataset_path / "trajectories/episode_000.npz"
    trajectory.write_bytes(trajectory.read_bytes() + b"changed")
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: pytest.fail("changed dataset must not enter evaluation core"),
    )

    with pytest.raises(ValueError, match="dataset identity changed"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert trajectory.exists()
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        dataset = session.get_one(Dataset, dataset_id)
        assert (job.state, job.error_code, job.execution_token) == (
            "FAILED",
            "DATASET_CHANGED",
            None,
        )
        assert dataset.status == "PREFLIGHT_FAILED"


def test_evaluation_dataset_trust_failure_is_dataset_changed(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        dataset = session.get_one(Dataset, evaluation_job.dataset_id)
        dataset_id = dataset.id
        dataset_path = Path(dataset.path)
    for path in sorted(dataset_path.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    dataset_path.rmdir()
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **_kwargs: pytest.fail("missing dataset must not enter core"),
    )

    with pytest.raises(ValueError):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        dataset = session.get_one(Dataset, dataset_id)
        assert (job.state, job.error_code) == ("FAILED", "DATASET_CHANGED")
        assert dataset.status == "PREFLIGHT_FAILED"


def test_evaluation_accepts_exact_all_zero_fingerprint(
    db_engine, data_root, evaluation_job, monkeypatch
):
    fingerprint = "0" * 64
    with session_scope(db_engine) as session:
        dataset = session.get_one(Dataset, evaluation_job.dataset_id)
        dataset.fingerprint = fingerprint
        kind = DatasetKind(dataset.kind)
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: DatasetInspection(
            kind,
            True,
            fingerprint,
            1,
            1,
            (),
        ),
    )
    monkeypatch.setattr(
        tasks_module,
        "run_evaluation",
        lambda **kwargs: kwargs["callbacks"].on_stage("REPORT"),
    )

    run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert reload_job(db_engine, evaluation_job.id).state == "SUCCEEDED"


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
        with session_scope(db_engine) as session:
            persisted = session.get_one(EvaluationJob, evaluation_job.id)
            assert (persisted.state, persisted.stage) == ("RUNNING", "PREFLIGHT")
            assert persisted.output_dir == str(data_root / "runs" / evaluation_job.id)
            assert (persisted.error_code, persisted.error_message) == (None, None)
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
    assert received["camera_keys"] == ()
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
        job.params_json = {
            "vlm_enabled": True,
            "camera_keys": ["observation.images.front", "observation.images.right_wrist"],
        }

    received = {}

    def evaluate(**kwargs):
        received.update(kwargs)
        kwargs["callbacks"].on_stage("REPORT")

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)

    run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert received["resume_from"] == failed_stage
    assert received["initial_progress"] == 63.0
    assert received["camera_keys"] == (
        "observation.images.front",
        "observation.images.right_wrist",
    )


def test_evaluation_task_legacy_vlm_job_falls_back_to_profile_image_key(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.vlm_enabled = True
        job.params_json = {"vlm_enabled": True}
    received = {}

    def evaluate(**kwargs):
        received.update(kwargs)
        kwargs["callbacks"].on_stage("REPORT")

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)

    run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    assert received["camera_keys"] == ("observation.images.right_wrist",)


def test_evaluation_task_rejects_empty_vlm_camera_snapshot(
    db_engine, data_root, evaluation_job, monkeypatch
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.vlm_enabled = True
        job.params_json = {"vlm_enabled": True, "camera_keys": []}
    monkeypatch.setattr(
        "vla_eval.tasks.run_evaluation",
        lambda **_kwargs: pytest.fail("invalid camera snapshot reached evaluation"),
    )

    with pytest.raises(ValueError, match="camera snapshot"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))


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


@pytest.mark.parametrize("dataset_status", ["PENDING", "FAILED"])
def test_evaluation_task_rejects_dataset_that_is_not_ready(
    db_engine, data_root, dataset, monkeypatch, dataset_status
):
    with session_scope(db_engine) as session:
        persisted_dataset = session.get_one(Dataset, dataset.id)
        persisted_dataset.status = dataset_status
        job = EvaluationJob(
            dataset_id=dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
        )
        session.add(job)
        session.flush()
        job_id = job.id
    monkeypatch.setattr(
        "vla_eval.tasks.run_evaluation",
        lambda **_kwargs: pytest.fail("non-ready dataset must not enter evaluation core"),
    )

    with pytest.raises(ValueError, match="dataset is not READY"):
        run_evaluation_task(job_id, runtime=_runtime(db_engine, data_root))

    assert reload_job(db_engine, job_id).state == "FAILED"


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


def test_evaluation_task_checks_cancellation_in_success_transaction(
    db_engine, data_root, evaluation_job, monkeypatch
):
    def evaluate(**kwargs):
        kwargs["callbacks"].on_stage("REPORT")
        with session_scope(db_engine) as session:
            session.get_one(EvaluationJob, evaluation_job.id).cancel_requested = True

    monkeypatch.setattr("vla_eval.tasks.run_evaluation", evaluate)

    with pytest.raises(EvaluationCancelled, match="before success commit"):
        run_evaluation_task(evaluation_job.id, runtime=_runtime(db_engine, data_root))

    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.stage) == ("CANCELLED", "REPORT")
    assert job.error_code == "EVALUATION_CANCELLED"


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


def _create_import_job(
    db_engine,
    *,
    cancel_requested: bool = False,
    source_name: str = "lab-a",
    remote_root: str = "/data/rollouts",
) -> ImportJob:
    with session_scope(db_engine) as session:
        job = ImportJob(
            source_name=source_name,
            remote_root=remote_root,
            remote_path="run-1",
            target_name="run-1",
            cancel_requested=cancel_requested,
        )
        session.add(job)
        session.flush()
        return job


def test_import_task_dispatches_configured_local_source_without_ssh_credentials(
    db_engine, data_root, monkeypatch
):
    local_root = data_root / "local-source"
    (local_root / "run-1").mkdir(parents=True)
    import_job = _create_import_job(
        db_engine,
        source_name="this-host",
        remote_root=str(local_root),
    )
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "f" * 64, 12, 2, ())
    received = {}
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
        received["spec"] = spec
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr(tasks_module, "execute_import", execute)

    run_import_task(
        import_job.id,
        runtime=_runtime(db_engine, data_root, local_root=local_root),
    )

    spec = received["spec"]
    assert spec.source is None
    assert spec.local_source == LocalSource(name="this-host", roots=(local_root,))
    assert spec.trusted_credentials_root is None
    assert spec.remote_root == str(local_root)
    assert spec.remote_relative_path == "run-1"


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
        camera_keys=("observation.images.front",),
    )
    received = {}
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
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
        inspector(spec.staging_path)
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
        assert persisted_dataset.inspection_json == {
            "errors": [],
            "camera_keys": ["observation.images.front"],
        }


@pytest.mark.parametrize(
    ("mode", "expected_boundary"),
    [("strict", None), ("data_root_boundary", "data_root")],
)
def test_import_task_forwards_configured_storage_boundary(
    db_engine, data_root, monkeypatch, mode, expected_boundary
):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "c" * 64, 1, 1, ())
    received = {}
    monkeypatch.setattr(tasks_module, "inspect_dataset", lambda *_args, **_kwargs: inspection)

    def execute(spec, *, inspector, callbacks):
        received["boundary"] = spec.storage_trust_boundary
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr(tasks_module, "execute_import", execute)

    run_import_task(
        import_job.id,
        runtime=_runtime(db_engine, data_root, storage_trust_mode=mode),
    )

    assert received["boundary"] == (
        data_root if expected_boundary == "data_root" else None
    )


def test_import_task_uses_persisted_second_remote_root(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine, remote_root="/data/archive")
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "d" * 64, 8, 1, ())
    received = {}

    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
        received["spec"] = spec
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr("vla_eval.tasks.execute_import", execute)

    run_import_task(
        import_job.id,
        runtime=_runtime(
            db_engine,
            data_root,
            remote_roots=("/data/rollouts", "/data/archive"),
        ),
    )

    assert received["spec"].remote_root == "/data/archive"
    assert received["spec"].source.roots == ("/data/archive",)


def test_import_ready_inspection_persists_fingerprint_before_publish(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "1" * 64, 8, 1, ())
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
        with session_scope(db_engine) as session:
            assert session.get_one(ImportJob, import_job.id).publish_fingerprint == "1" * 64
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr("vla_eval.tasks.execute_import", execute)

    run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))


def test_import_task_rejects_unregistered_persisted_remote_root(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine, remote_root="/data/unregistered")
    monkeypatch.setattr(
        "vla_eval.tasks.execute_import",
        lambda *_args, **_kwargs: pytest.fail("unregistered root must not reach import core"),
    )

    with pytest.raises(ValueError, match="remote root is not registered"):
        run_import_task(
            import_job.id,
            runtime=_runtime(
                db_engine,
                data_root,
                remote_roots=("/data/rollouts", "/data/archive"),
            ),
        )

    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert job.state == "FAILED"
        assert job.error_code == "IMPORT_FAILED"


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
            local_sources={},
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
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
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


def test_import_ready_persistence_failure_rolls_publication_back(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "c" * 64, 10, 1, ())
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name
    observed_ready_without_dataset = []
    inspect_calls = []

    def inspect(path, *, allowed_root):
        inspect_calls.append((path, allowed_root))
        return inspection

    def injected_execute(spec, **kwargs):
        callbacks = kwargs["callbacks"]

        def transfer(_argv, _progress):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "received.marker").write_text("ok")

        def observe_state(state):
            callbacks.on_state(state)
            if state == "READY":
                with session_scope(db_engine) as session:
                    persisted = session.get_one(ImportJob, import_job.id)
                    observed_ready_without_dataset.append(
                        persisted.state == "READY" and persisted.dataset_id is None
                    )

        injected_spec = replace(
            spec,
            mode="injected",
            source=None,
            trusted_credentials_root=None,
            trusted_staging_root=None,
            trusted_inbox_root=None,
            storage_trust_boundary=None,
        )
        return import_jobs_module.execute_import(
            injected_spec,
            transfer=transfer,
            inspector=kwargs.get("inspector", lambda _path: inspection),
            callbacks=import_jobs_module.ImportCallbacks(
                on_state=observe_state,
                on_progress=callbacks.on_progress,
                is_cancelled=callbacks.is_cancelled,
            ),
        )

    persistence_error = OSError("database write failed")

    def fail_ready(*_args):
        raise persistence_error

    monkeypatch.setattr(tasks_module, "inspect_dataset", inspect, raising=False)
    monkeypatch.setattr("vla_eval.tasks.execute_import", injected_execute)
    monkeypatch.setattr("vla_eval.tasks._record_import_success", fail_ready)

    with pytest.raises(OSError) as raised:
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is persistence_error
    assert inspect_calls == [(staging, staging)]
    assert observed_ready_without_dataset == []
    assert staging.is_dir()
    assert (staging / "received.marker").is_file()
    assert not target.exists()
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.dataset_id) == ("FAILED", None)


def test_import_cancel_before_ready_cas_rolls_publication_back(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "c" * 64, 10, 1, ())
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name

    def inspect(_path, *, allowed_root):
        return inspection

    real_verify = import_jobs_module._verify_published_target

    def verify_then_cancel(spec, published_target, production):
        real_verify(spec, published_target, production)
        with session_scope(db_engine) as session:
            session.get_one(ImportJob, import_job.id).cancel_requested = True

    def injected_execute(spec, **kwargs):
        def transfer(_argv, _progress):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "received.marker").write_text("ok", encoding="utf-8")

        return import_jobs_module.execute_import(
            replace(
                spec,
                mode="injected",
                source=None,
                trusted_credentials_root=None,
                trusted_staging_root=None,
                trusted_inbox_root=None,
                storage_trust_boundary=None,
            ),
            transfer=transfer,
            inspector=kwargs["inspector"],
            callbacks=kwargs["callbacks"],
        )

    monkeypatch.setattr(tasks_module, "inspect_dataset", inspect)
    monkeypatch.setattr(tasks_module, "execute_import", injected_execute)
    monkeypatch.setattr(import_jobs_module, "_verify_published_target", verify_then_cancel)

    with pytest.raises(ValueError, match="state changed before READY"):
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert staging.is_dir()
    assert not target.exists()
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.dataset_id, job.execution_token) == ("CANCELLED", None, None)
        assert session.scalar(select(Dataset)) is None


def test_publish_fingerprint_failure_stops_before_rename(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "6" * 64, 10, 1, ())
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def injected_execute(spec, **kwargs):
        def transfer(_argv, _progress):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "partial.bin").write_bytes(b"partial")

        return import_jobs_module.execute_import(
            replace(
                spec,
                mode="injected",
                source=None,
                trusted_credentials_root=None,
                trusted_staging_root=None,
                trusted_inbox_root=None,
                storage_trust_boundary=None,
            ),
            transfer=transfer,
            inspector=kwargs["inspector"],
            callbacks=kwargs["callbacks"],
        )

    marker_error = OSError("marker database unavailable")

    def fail_marker(*_args):
        raise marker_error

    monkeypatch.setattr("vla_eval.tasks.execute_import", injected_execute)
    monkeypatch.setattr("vla_eval.tasks._record_publish_fingerprint", fail_marker)

    with pytest.raises(OSError) as raised:
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert raised.value is marker_error
    assert staging.is_dir()
    assert (staging / "partial.bin").is_file()
    assert not target.exists()
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.publish_fingerprint) == ("FAILED", None)


def test_import_ready_commits_only_with_dataset_link_and_published_target(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "e" * 64, 11, 2, ())
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name

    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda path, *, allowed_root: inspection,
    )

    def injected_execute(spec, **kwargs):
        def transfer(_argv, _progress):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "received.marker").write_text("ok")

        return import_jobs_module.execute_import(
            replace(
                spec,
                mode="injected",
                source=None,
                trusted_credentials_root=None,
                trusted_staging_root=None,
                trusted_inbox_root=None,
                storage_trust_boundary=None,
            ),
            transfer=transfer,
            inspector=kwargs["inspector"],
            callbacks=kwargs["callbacks"],
        )

    monkeypatch.setattr("vla_eval.tasks.execute_import", injected_execute)

    result = run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert result.dataset_path == target
    assert target.is_dir()
    assert (target / "received.marker").is_file()
    assert not staging.exists()
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        dataset = session.get_one(Dataset, job.dataset_id)
        assert (job.state, job.progress) == ("READY", 100.0)
        assert dataset.path == str(target)
        assert dataset.status == "READY"

    monkeypatch.setattr(
        "vla_eval.tasks.execute_import",
        lambda *_args, **_kwargs: pytest.fail("completed import must not rerun Task 9"),
    )

    repeated = run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert repeated.dataset_path == target
    assert repeated.inspection.fingerprint == inspection.fingerprint
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.dataset_id) == ("READY", dataset.id)


def test_interrupted_import_reconciles_published_target_by_fingerprint(
    db_engine, data_root, monkeypatch
):
    import_job = _create_import_job(db_engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "2" * 64, 12, 3, ())
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda path, *, allowed_root: inspection,
    )

    def injected_execute(spec, **kwargs):
        def transfer(_argv, _progress):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "received.marker").write_text("ok")

        return import_jobs_module.execute_import(
            replace(
                spec,
                mode="injected",
                source=None,
                trusted_credentials_root=None,
                trusted_staging_root=None,
                trusted_inbox_root=None,
                storage_trust_boundary=None,
            ),
            transfer=transfer,
            inspector=kwargs["inspector"],
            callbacks=kwargs["callbacks"],
        )

    monkeypatch.setattr("vla_eval.tasks.execute_import", injected_execute)
    monkeypatch.setattr(
        import_jobs_module,
        "_publish_and_report_ready",
        lambda _spec, source, destination, _production, _callbacks: (
            import_jobs_module._rename_no_replace(source, destination)
        ),
    )

    run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    with session_scope(db_engine) as session:
        crashed = session.get_one(ImportJob, import_job.id)
        assert (crashed.state, crashed.dataset_id, crashed.publish_fingerprint) == (
            "PREFLIGHT",
            None,
            "2" * 64,
        )
    assert target.is_dir()
    assert not staging.exists()
    assert recover_interrupted_jobs(runtime=_runtime(db_engine, data_root)) == 1

    monkeypatch.setattr(
        "vla_eval.tasks.execute_import",
        lambda *_args, **_kwargs: pytest.fail("reconcile must not rerun Task 9"),
    )

    result = run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert result.dataset_path == target
    assert result.inspection.fingerprint == "2" * 64
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        dataset = session.get_one(Dataset, job.dataset_id)
        assert (job.state, job.progress) == ("READY", 100.0)
        assert dataset.fingerprint == "2" * 64


def test_concurrent_interrupted_import_reconciliation_commits_one_dataset(data_root, monkeypatch):
    engine = create_engine_for_url(f"sqlite:///{data_root / 'db/concurrent.sqlite3'}")
    init_db(engine)
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "7" * 64, 12, 3, ())
    with session_scope(engine) as session:
        import_job = ImportJob(
            source_name="lab-a",
            remote_root="/data/rollouts",
            remote_path="run-1",
            target_name="run-1",
            state="INTERRUPTED",
            publish_fingerprint=inspection.fingerprint,
        )
        session.add(import_job)
        session.flush()
        import_id = import_job.id

    target = data_root / "inbox" / "run-1"
    target.mkdir()
    (target / "evidence.bin").write_bytes(b"evidence")
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )
    runtime = _runtime(engine, data_root)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_import_task, import_id, runtime=runtime) for _ in range(2)
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=10))
                except StaleTaskExecution as error:
                    outcomes.append(error)

        assert sum(isinstance(value, ImportResult) for value in outcomes) == 1
        assert sum(isinstance(value, StaleTaskExecution) for value in outcomes) == 1
        with session_scope(engine) as session:
            job = session.get_one(ImportJob, import_id)
            datasets = list(session.scalars(select(Dataset)))
            assert (job.state, job.dataset_id) == ("READY", datasets[0].id)
            assert len(datasets) == 1
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_marker", "no durable publish fingerprint"),
        ("fingerprint_mismatch", "fingerprint does not match"),
        ("target_not_ready", "did not pass preflight"),
        ("target_symlink", "must not be a symbolic link"),
        ("inbox_root_symlink", "symlink components"),
        ("intermediate_symlink", "symlink components"),
        ("staging_exists", "still has a staging path"),
    ],
)
def test_interrupted_import_never_adopts_unproven_target(
    db_engine, data_root, monkeypatch, case, message
):
    import_job = _create_import_job(db_engine)
    staging = data_root / "staging" / import_job.id
    target = data_root / "inbox" / import_job.target_name
    marker = None if case == "missing_marker" else "3" * 64
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        job.state = "INTERRUPTED"
        job.publish_fingerprint = marker
        if case == "intermediate_symlink":
            job.target_name = "team/run-1"
            target = data_root / "inbox" / job.target_name

    if case == "target_symlink":
        outside = data_root / "outside-target"
        outside.mkdir()
        target.symlink_to(outside, target_is_directory=True)
    elif case == "inbox_root_symlink":
        inbox = data_root / "inbox"
        inbox.rmdir()
        actual_inbox = data_root / "actual-inbox"
        target = actual_inbox / import_job.target_name
        target.mkdir(parents=True)
        inbox.symlink_to(actual_inbox, target_is_directory=True)
    elif case == "intermediate_symlink":
        actual_team = data_root / "actual-team"
        target = actual_team / "run-1"
        target.mkdir(parents=True)
        (data_root / "inbox" / "team").symlink_to(actual_team, target_is_directory=True)
    else:
        target.mkdir()
    (target / "evidence.bin").write_bytes(b"evidence")
    if case == "staging_exists":
        staging.mkdir()
        (staging / "partial.bin").write_bytes(b"partial")

    actual_fingerprint = "4" * 64 if case == "fingerprint_mismatch" else "3" * 64
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: DatasetInspection(
            DatasetKind.LEROBOT,
            case != "target_not_ready",
            actual_fingerprint,
            8,
            1,
            (),
        ),
    )
    monkeypatch.setattr(
        "vla_eval.tasks.execute_import",
        lambda *_args, **_kwargs: pytest.fail("unproven target must not enter Task 9"),
    )

    with pytest.raises(ValueError, match=message):
        run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert os.path.lexists(target)
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        assert (job.state, job.dataset_id) == ("FAILED", None)


def test_ready_import_integrity_drift_marks_job_and_dataset_failed(
    db_engine, data_root, monkeypatch
):
    target = data_root / "inbox" / "run-1"
    target.mkdir()
    evidence = target / "evidence.bin"
    evidence.write_bytes(b"changed")
    expected_fingerprint = "8" * 64
    with session_scope(db_engine) as session:
        dataset = Dataset(
            name="run-1",
            path=str(target),
            kind=DatasetKind.LEROBOT.value,
            status="READY",
            fingerprint=expected_fingerprint,
            size_bytes=8,
            episode_count=1,
        )
        session.add(dataset)
        session.flush()
        job = ImportJob(
            source_name="lab-a",
            remote_root="/data/rollouts",
            remote_path="run-1",
            target_name="run-1",
            state="READY",
            progress=100.0,
            publish_fingerprint=expected_fingerprint,
            dataset_id=dataset.id,
        )
        session.add(job)
        session.flush()
        import_id = job.id
        dataset_id = dataset.id

    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: DatasetInspection(
            DatasetKind.LEROBOT,
            True,
            "9" * 64,
            8,
            1,
            (),
        ),
    )

    with pytest.raises(ImportIntegrityError):
        run_import_task(import_id, runtime=_runtime(db_engine, data_root))

    assert evidence.read_bytes() == b"changed"
    with session_scope(db_engine) as session:
        persisted_job = session.get_one(ImportJob, import_id)
        persisted_dataset = session.get_one(Dataset, dataset_id)
        assert persisted_job.state == "FAILED"
        assert persisted_job.error_code == "IMPORT_INTEGRITY_FAILED"
        assert (
            persisted_job.error_message
            == "Published dataset integrity check failed. Review or re-import the dataset."
        )
        assert persisted_dataset.status == "PREFLIGHT_FAILED"


def test_ready_import_uses_data_root_boundary_under_writable_shared_parent(
    db_engine, data_root, monkeypatch
):
    shared_parent = data_root.parent
    original_mode = shared_parent.stat().st_mode & 0o777
    shared_parent.chmod(0o777)
    target = data_root / "inbox" / "run-1"
    target.mkdir()
    fingerprint = "d" * 64
    with session_scope(db_engine) as session:
        dataset = Dataset(
            name="run-1",
            path=str(target),
            kind=DatasetKind.LEROBOT.value,
            status="READY",
            fingerprint=fingerprint,
            size_bytes=8,
            episode_count=1,
        )
        session.add(dataset)
        session.flush()
        job = ImportJob(
            source_name="lab-a",
            remote_root="/data/rollouts",
            remote_path="run-1",
            target_name="run-1",
            state="READY",
            progress=100.0,
            publish_fingerprint=fingerprint,
            dataset_id=dataset.id,
        )
        session.add(job)
        session.flush()
        import_id = job.id
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, fingerprint, 8, 1, ())
    monkeypatch.setattr(tasks_module, "inspect_dataset", lambda *_args, **_kwargs: inspection)

    try:
        result = run_import_task(
            import_id,
            runtime=_runtime(
                db_engine,
                data_root,
                storage_trust_mode="data_root_boundary",
            ),
        )
    finally:
        shared_parent.chmod(original_mode)

    assert result.dataset_path == target


def test_evaluation_output_uses_data_root_boundary_under_writable_shared_parent(
    db_engine, data_root
):
    shared_parent = data_root.parent
    original_mode = shared_parent.stat().st_mode & 0o777
    shared_parent.chmod(0o777)
    runtime = _runtime(
        db_engine,
        data_root,
        storage_trust_mode="data_root_boundary",
    )

    try:
        output = tasks_module._trusted_evaluation_output(runtime, "job-shared", None)
    finally:
        shared_parent.chmod(original_mode)

    assert output == data_root / "runs" / "job-shared"


@pytest.mark.parametrize("recorder_name", ["_record_import_failure", "_record_import_cancelled"])
def test_import_terminal_recorders_do_not_overwrite_ready_dataset(
    db_engine, data_root, recorder_name
):
    target = data_root / "inbox" / "run-1"
    target.mkdir()
    fingerprint = "a" * 64
    with session_scope(db_engine) as session:
        dataset = Dataset(
            name="run-1",
            path=str(target),
            kind=DatasetKind.LEROBOT.value,
            status="READY",
            fingerprint=fingerprint,
        )
        session.add(dataset)
        session.flush()
        job = ImportJob(
            source_name="lab-a",
            remote_root="/data/rollouts",
            remote_path="run-1",
            target_name="run-1",
            state="READY",
            progress=100.0,
            publish_fingerprint=fingerprint,
            dataset_id=dataset.id,
        )
        session.add(job)
        session.flush()
        import_id = job.id

    recorder = getattr(tasks_module, recorder_name)
    stale_token = str(uuid4())
    if recorder_name == "_record_import_failure":
        recorder(
            _runtime(db_engine, data_root),
            import_id,
            stale_token,
            RuntimeError("stale failure"),
        )
    else:
        recorder(_runtime(db_engine, data_root), import_id, stale_token)

    with session_scope(db_engine) as session:
        persisted = session.get_one(ImportJob, import_id)
        assert (persisted.state, persisted.error_code, persisted.error_message) == (
            "READY",
            None,
            None,
        )


def test_stale_integrity_failure_does_not_downgrade_newer_ready_commit(db_engine, data_root):
    target = data_root / "inbox" / "run-1"
    target.mkdir()
    current_fingerprint = "b" * 64
    with session_scope(db_engine) as session:
        dataset = Dataset(
            name="run-1",
            path=str(target),
            kind=DatasetKind.LEROBOT.value,
            status="READY",
            fingerprint=current_fingerprint,
        )
        session.add(dataset)
        session.flush()
        job = ImportJob(
            source_name="lab-a",
            remote_root="/data/rollouts",
            remote_path="run-1",
            target_name="run-1",
            state="READY",
            progress=100.0,
            publish_fingerprint=current_fingerprint,
            dataset_id=dataset.id,
        )
        session.add(job)
        session.flush()
        import_id = job.id
        dataset_id = dataset.id

    tasks_module._record_import_integrity_failure(
        _runtime(db_engine, data_root),
        import_id,
        dataset_id,
        "older-fingerprint",
    )

    with session_scope(db_engine) as session:
        persisted_job = session.get_one(ImportJob, import_id)
        persisted_dataset = session.get_one(Dataset, dataset_id)
        assert (persisted_job.state, persisted_job.error_code) == ("READY", None)
        assert persisted_dataset.status == "READY"


def test_interrupted_import_with_no_target_resumes_through_task9(db_engine, data_root, monkeypatch):
    import_job = _create_import_job(db_engine)
    staging = data_root / "staging" / import_job.id
    staging.mkdir()
    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "5" * 64, 8, 1, ())
    with session_scope(db_engine) as session:
        job = session.get_one(ImportJob, import_job.id)
        job.state = "INTERRUPTED"
        job.publish_fingerprint = "old-marker"
    called = []
    monkeypatch.setattr(
        tasks_module,
        "inspect_dataset",
        lambda _path, *, allowed_root: inspection,
    )

    def execute(spec, *, inspector, callbacks):
        called.append(spec.job_id)
        callbacks.on_state("PREFLIGHT")
        inspector(spec.staging_path)
        callbacks.on_state("READY")
        return ImportResult(spec.target_path, inspection)

    monkeypatch.setattr("vla_eval.tasks.execute_import", execute)

    run_import_task(import_job.id, runtime=_runtime(db_engine, data_root))

    assert called == [import_job.id]


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
        "CONNECTING",
        "TRANSFERRING",
        "VERIFYING",
        "PREFLIGHT",
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
                remote_root="/data/rollouts",
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

    assert recover_interrupted_jobs(runtime=runtime) == 8
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
        "INTERRUPTED",
        "INTERRUPTED",
        "INTERRUPTED",
        "QUEUED",
        "READY",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
    ]
    assert staging_marker.read_bytes() == b"partial"


def test_recovery_does_not_overwrite_success_committed_after_selection(
    db_engine, data_root, evaluation_job, monkeypatch
):
    token = str(uuid4())
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "RUNNING"
        job.execution_token = token
    selected_barrier = Barrier(2)
    update_barrier = Barrier(2)
    real_update = tasks_module.update
    blocked = False

    def pause_recovery_update(model):
        nonlocal blocked
        statement = real_update(model)
        if model is EvaluationJob and not blocked:
            blocked = True
            selected_barrier.wait(timeout=5)
            update_barrier.wait(timeout=5)
        return statement

    monkeypatch.setattr(tasks_module, "update", pause_recovery_update)
    runtime = _runtime(db_engine, data_root)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(recover_interrupted_jobs, runtime=runtime)
        selected_barrier.wait(timeout=5)
        with session_scope(db_engine) as session:
            job = session.get_one(EvaluationJob, evaluation_job.id)
            job.state = "SUCCEEDED"
            job.execution_token = None
        update_barrier.wait(timeout=5)
        assert future.result(timeout=5) == 0

    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.execution_token) == ("SUCCEEDED", None)


def test_recovery_does_not_clear_new_execution_token_after_selection(
    db_engine, data_root, evaluation_job, monkeypatch
):
    runtime = _runtime(db_engine, data_root)
    token_a = str(uuid4())
    token_b = str(uuid4())
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "RUNNING"
        job.execution_token = token_a
    selected = Barrier(2)
    resume = Barrier(2)
    real_update = tasks_module.update
    blocked = False

    def pause_update(model):
        nonlocal blocked
        statement = real_update(model)
        if model is EvaluationJob and not blocked:
            blocked = True
            selected.wait(timeout=5)
            resume.wait(timeout=5)
        return statement

    monkeypatch.setattr(tasks_module, "update", pause_update)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(recover_interrupted_jobs, runtime=runtime)
        selected.wait(timeout=5)
        with session_scope(db_engine) as session:
            job = session.get_one(EvaluationJob, evaluation_job.id)
            job.state = "INTERRUPTED"
            job.execution_token = None
        tasks_module._claim_evaluation_execution(runtime, evaluation_job.id, token_b)
        resume.wait(timeout=5)
        assert future.result(timeout=5) == 0

    job = reload_job(db_engine, evaluation_job.id)
    assert (job.state, job.execution_token) == ("RUNNING", token_b)
