from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import Engine

from vla_eval.config import AppConfig
from vla_eval.db import session_scope
from vla_eval.evaluation import EvaluationCallbacks, run_evaluation
from vla_eval.exceptions import EvaluationCancelled
from vla_eval.import_jobs import (
    DatasetValidationError,
    ImportCallbacks,
    ImportResult,
    ImportSpec,
    TransferError,
    execute_import,
)
from vla_eval.models import Dataset, EvaluationJob, ImportJob
from vla_eval.profiles import load_profile


@dataclass(frozen=True)
class TaskRuntime:
    engine: Engine
    config: AppConfig
    profiles_root: Path
    credentials_root: Path


_configured_runtime: TaskRuntime | None = None


def configure_runtime(runtime: TaskRuntime) -> None:
    if not isinstance(runtime, TaskRuntime):
        raise TypeError("runtime must be a TaskRuntime")
    global _configured_runtime
    _configured_runtime = runtime


def clear_runtime() -> None:
    global _configured_runtime
    _configured_runtime = None


def _require_runtime(runtime: TaskRuntime | None) -> TaskRuntime:
    resolved = runtime or _configured_runtime
    if resolved is None:
        raise RuntimeError("task runtime has not been configured")
    return resolved


def _record_evaluation_failure(runtime: TaskRuntime, job_id: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "FAILED"
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "Evaluation failed. Review worker logs for details."


def _record_evaluation_success(runtime: TaskRuntime, job_id: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "SUCCEEDED"
        job.progress = 100.0
        job.error_code = None
        job.error_message = None


def _record_evaluation_cancelled(runtime: TaskRuntime, job_id: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "CANCELLED"
        job.error_code = "EVALUATION_CANCELLED"
        job.error_message = "Evaluation was cancelled."


def _update_evaluation_stage(runtime: TaskRuntime, job_id: str, stage: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "RUNNING"
        job.stage = stage


def _update_evaluation_progress(runtime: TaskRuntime, job_id: str, progress: float) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.progress = progress


def _evaluation_cancel_requested(runtime: TaskRuntime, job_id: str) -> bool:
    with session_scope(runtime.engine) as session:
        return session.get_one(EvaluationJob, job_id).cancel_requested


def recover_interrupted_jobs(*, runtime: TaskRuntime | None = None) -> int:
    resolved = _require_runtime(runtime)
    evaluation_states = {"RUNNING", "METRICS", "VLM", "REPORT"}
    import_states = {"TRANSFERRING"}
    with session_scope(resolved.engine) as session:
        evaluation_ids = tuple(
            session.scalars(
                select(EvaluationJob.id).where(EvaluationJob.state.in_(evaluation_states))
            )
        )
        import_ids = tuple(
            session.scalars(select(ImportJob.id).where(ImportJob.state.in_(import_states)))
        )

    recovered = 0
    for job_id in evaluation_ids:
        with session_scope(resolved.engine) as session:
            job = session.get_one(EvaluationJob, job_id)
            if job.state in evaluation_states:
                job.state = "INTERRUPTED"
                recovered += 1
    for import_id in import_ids:
        with session_scope(resolved.engine) as session:
            job = session.get_one(ImportJob, import_id)
            if job.state in import_states:
                job.state = "INTERRUPTED"
                recovered += 1
    return recovered


def _update_import_state(runtime: TaskRuntime, import_id: str, state: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(ImportJob, import_id)
        job.state = state
        if state == "CONNECTING":
            job.error_code = None
            job.error_message = None


def _update_import_progress(runtime: TaskRuntime, import_id: str, progress: float) -> None:
    with session_scope(runtime.engine) as session:
        session.get_one(ImportJob, import_id).progress = progress


def _import_cancel_requested(runtime: TaskRuntime, import_id: str) -> bool:
    with session_scope(runtime.engine) as session:
        return session.get_one(ImportJob, import_id).cancel_requested


def _record_import_success(
    runtime: TaskRuntime,
    import_id: str,
    target_name: str,
    result: ImportResult,
) -> None:
    inspection = result.inspection
    with session_scope(runtime.engine) as session:
        dataset = Dataset(
            name=target_name,
            path=str(result.dataset_path),
            kind=inspection.kind.value if inspection.kind is not None else "unknown",
            status="READY",
            fingerprint=inspection.fingerprint,
            size_bytes=inspection.size_bytes,
            episode_count=inspection.episode_count or 0,
            inspection_json={"errors": list(inspection.errors)},
        )
        session.add(dataset)
        session.flush()
        job = session.get_one(ImportJob, import_id)
        job.dataset_id = dataset.id
        job.state = "READY"
        job.progress = 100.0
        job.error_code = None
        job.error_message = None


def _record_import_failure(
    runtime: TaskRuntime,
    import_id: str,
    error: BaseException,
) -> None:
    if isinstance(error, TransferError):
        code = "IMPORT_TRANSFER_FAILED"
        message = "Dataset transfer failed. Retry the import."
    elif isinstance(error, DatasetValidationError):
        code = "IMPORT_DATASET_INVALID"
        message = "Dataset preflight failed. Review the dataset format."
    else:
        code = "IMPORT_FAILED"
        message = "Dataset import failed. Review worker logs for details."
    with session_scope(runtime.engine) as session:
        job = session.get_one(ImportJob, import_id)
        job.state = "FAILED"
        job.error_code = code
        job.error_message = message


def _record_import_cancelled(runtime: TaskRuntime, import_id: str) -> None:
    with session_scope(runtime.engine) as session:
        job = session.get_one(ImportJob, import_id)
        job.state = "CANCELLED"
        job.error_code = "IMPORT_CANCELLED"
        job.error_message = "Dataset import was cancelled."


def run_import_task(import_id: str, *, runtime: TaskRuntime | None = None) -> ImportResult:
    resolved = _require_runtime(runtime)
    with session_scope(resolved.engine) as session:
        job = session.get_one(ImportJob, import_id)
        source_name = job.source_name
        remote_path = job.remote_path
        target_name = job.target_name

    callbacks = ImportCallbacks(
        on_state=lambda state: _update_import_state(resolved, import_id, state),
        on_progress=lambda progress: _update_import_progress(resolved, import_id, progress),
        is_cancelled=lambda: _import_cancel_requested(resolved, import_id),
    )
    try:
        if callbacks.is_cancelled():
            raise TransferError("import cancelled before execution")
        source = resolved.config.remote_sources[source_name]
        spec = ImportSpec(
            job_id=import_id,
            source_name=source.name,
            remote_root=source.roots[0],
            remote_relative_path=remote_path,
            staging_path=resolved.config.data_root / "staging" / import_id,
            target_path=resolved.config.data_root / "inbox" / target_name,
            mode="production",
            source=source,
            trusted_credentials_root=resolved.credentials_root,
            trusted_staging_root=resolved.config.data_root / "staging",
            trusted_inbox_root=resolved.config.data_root / "inbox",
        )
        result = execute_import(spec, callbacks=callbacks)
        _record_import_success(resolved, import_id, target_name, result)
    except BaseException as original:
        try:
            if callbacks.is_cancelled():
                _record_import_cancelled(resolved, import_id)
            else:
                _record_import_failure(resolved, import_id, original)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve worker interrupts
            raise BaseExceptionGroup(
                "import and failure persistence both failed",
                [original, persistence_error],
            ) from original
        raise
    return result


def run_evaluation_task(job_id: str, *, runtime: TaskRuntime | None = None):
    resolved = _require_runtime(runtime)
    with session_scope(resolved.engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        dataset = session.get_one(Dataset, job.dataset_id)
        dataset_path = dataset.path
        profile_name = job.profile_name
        profile_version = job.profile_version
        vlm_enabled = job.vlm_enabled
        output_dir = job.output_dir or str(resolved.config.data_root / "runs" / job.id)
        resume_from = job.stage if job.stage in {"VLM", "REPORT"} else "METRICS"
        initial_progress = job.progress
        job.state = "RUNNING"
        job.output_dir = output_dir
        job.error_code = None
        job.error_message = None

    callbacks = EvaluationCallbacks(
        on_stage=lambda stage: _update_evaluation_stage(resolved, job_id, stage),
        on_progress=lambda progress: _update_evaluation_progress(resolved, job_id, progress),
        should_cancel=lambda: _evaluation_cancel_requested(resolved, job_id),
    )
    try:
        if callbacks.should_cancel():
            raise EvaluationCancelled("evaluation cancelled before execution")
        profile = load_profile(resolved.profiles_root / f"{profile_name}.yaml")
        if profile.version != profile_version:
            raise ValueError("evaluation profile version changed after job submission")
        result = run_evaluation(
            dataset_path=dataset_path,
            output_dir=output_dir,
            profile=profile,
            vlm_enabled=vlm_enabled,
            callbacks=callbacks,
            resume_from=resume_from,
            initial_progress=initial_progress,
        )
        _record_evaluation_success(resolved, job_id)
    except EvaluationCancelled as original:
        try:
            _record_evaluation_cancelled(resolved, job_id)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve worker interrupts
            raise BaseExceptionGroup(
                "evaluation cancellation and persistence both failed",
                [original, persistence_error],
            ) from original
        raise
    except BaseException as original:
        try:
            _record_evaluation_failure(resolved, job_id)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve worker interrupts
            raise BaseExceptionGroup(
                "evaluation and failure persistence both failed",
                [original, persistence_error],
            ) from original
        raise
    return result
