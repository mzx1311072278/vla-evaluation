from __future__ import annotations

import errno
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from vla_eval.config import AppConfig
from vla_eval.datasets import DatasetInspection, inspect_dataset
from vla_eval.db import session_scope
from vla_eval.evaluation import EvaluationCallbacks, run_evaluation
from vla_eval.exceptions import EvaluationCancelled, ModelLoadError
from vla_eval.import_jobs import (
    DatasetValidationError,
    ImportCallbacks,
    ImportResult,
    ImportSpec,
    TransferError,
    execute_import,
    validate_published_target,
    validate_trusted_directory,
    validate_trusted_readable_directory,
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


class ImportIntegrityError(ValueError):
    """A READY import no longer matches its durable dataset identity."""


class StaleTaskExecution(RuntimeError):
    """A callback belongs to an execution generation that no longer owns the job."""


class DatasetChangedError(ValueError):
    """The on-disk dataset no longer matches the submitted database identity."""


_EVALUATION_CLAIM_STATES = ("QUEUED", "FAILED", "INTERRUPTED")
_IMPORT_CLAIM_STATES = ("QUEUED", "FAILED", "INTERRUPTED")
_IMPORT_ACTIVE_STATES = (
    "QUEUED",
    "FAILED",
    "INTERRUPTED",
    "CONNECTING",
    "TRANSFERRING",
    "VERIFYING",
    "PREFLIGHT",
)
_PROFILE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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


def _claim_evaluation_execution(runtime: TaskRuntime, job_id: str, token: str) -> None:
    with session_scope(runtime.engine) as session:
        claimed = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.state.in_(_EVALUATION_CLAIM_STATES),
                EvaluationJob.execution_token.is_(None),
            )
            .values(
                state="RUNNING",
                execution_token=token,
                error_code=None,
                error_message=None,
            )
        )
        if claimed.rowcount != 1:
            raise StaleTaskExecution("evaluation execution could not claim the job")


def _walk_exceptions(error: BaseException):
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)


def _classify_evaluation_failure(error: BaseException) -> tuple[str, str]:
    chain = tuple(_walk_exceptions(error))
    if any(isinstance(item, OSError) and item.errno == errno.ENOSPC for item in chain):
        return "DISK_FULL", "Evaluation storage is full. Free space and retry."
    if any(
        item.__class__.__name__ == "OutOfMemoryError" or "cuda out of memory" in str(item).lower()
        for item in chain
    ):
        return (
            "CUDA_OUT_OF_MEMORY",
            "GPU memory was exhausted. Reduce workload or retry.",
        )
    if any(isinstance(item, ModelLoadError) for item in chain):
        return (
            "MODEL_LOAD_FAILED",
            "The configured model could not be loaded. Review worker logs.",
        )
    return "EVALUATION_FAILED", "Evaluation failed. Review worker logs for details."


def _record_evaluation_failure(
    runtime: TaskRuntime,
    job_id: str,
    token: str,
    error: BaseException,
) -> None:
    code, message = _classify_evaluation_failure(error)
    with session_scope(runtime.engine) as session:
        session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
            .values(
                state="FAILED",
                execution_token=None,
                error_code=code,
                error_message=message,
            )
        )


def _record_evaluation_success(runtime: TaskRuntime, job_id: str, token: str) -> None:
    cancellation_committed = False
    with session_scope(runtime.engine) as session:
        succeeded = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
                EvaluationJob.cancel_requested.is_(False),
            )
            .values(
                state="SUCCEEDED",
                progress=100.0,
                execution_token=None,
                error_code=None,
                error_message=None,
            )
        )
        if succeeded.rowcount == 1:
            return
        cancelled = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
                EvaluationJob.cancel_requested.is_(True),
            )
            .values(
                state="CANCELLED",
                execution_token=None,
                error_code="EVALUATION_CANCELLED",
                error_message="Evaluation was cancelled.",
            )
        )
        if cancelled.rowcount == 1:
            cancellation_committed = True
        else:
            raise StaleTaskExecution("evaluation success callback is stale")
    if cancellation_committed:
        raise EvaluationCancelled("evaluation cancelled before success commit")


def _record_evaluation_cancelled(runtime: TaskRuntime, job_id: str, token: str) -> None:
    with session_scope(runtime.engine) as session:
        session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
            .values(
                state="CANCELLED",
                execution_token=None,
                error_code="EVALUATION_CANCELLED",
                error_message="Evaluation was cancelled.",
            )
        )


def _update_evaluation_stage(runtime: TaskRuntime, job_id: str, token: str, stage: str) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
            .values(stage=stage)
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("evaluation stage callback is stale")


def _update_evaluation_progress(
    runtime: TaskRuntime, job_id: str, token: str, progress: float
) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
            .values(progress=progress)
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("evaluation progress callback is stale")


def _evaluation_cancel_requested(runtime: TaskRuntime, job_id: str, token: str) -> bool:
    with session_scope(runtime.engine) as session:
        value = session.scalar(
            select(EvaluationJob.cancel_requested).where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
        )
        if value is None:
            raise StaleTaskExecution("evaluation cancellation callback is stale")
        return value


def _set_evaluation_output(runtime: TaskRuntime, job_id: str, token: str, output_dir: Path) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
            )
            .values(output_dir=str(output_dir))
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("evaluation output assignment is stale")


def _record_dataset_changed(
    runtime: TaskRuntime,
    job_id: str,
    token: str,
    dataset_id: str,
) -> None:
    with session_scope(runtime.engine) as session:
        failed = session.execute(
            update(EvaluationJob)
            .where(
                EvaluationJob.id == job_id,
                EvaluationJob.execution_token == token,
                EvaluationJob.state == "RUNNING",
                EvaluationJob.dataset_id == dataset_id,
            )
            .values(
                state="FAILED",
                execution_token=None,
                error_code="DATASET_CHANGED",
                error_message=(
                    "Dataset contents changed after submission. Revalidate the dataset."
                ),
            )
        )
        if failed.rowcount == 1:
            session.execute(
                update(Dataset).where(Dataset.id == dataset_id).values(status="PREFLIGHT_FAILED")
            )


def _verify_evaluation_dataset_identity(
    runtime: TaskRuntime,
    job_id: str,
    token: str,
    dataset_id: str,
    dataset_path: Path,
    expected_kind: str,
    expected_fingerprint: str | None,
) -> DatasetInspection:
    try:
        trusted_dataset = validate_published_target(
            dataset_path, runtime.config.data_root / "inbox"
        )
        inspection = inspect_dataset(trusted_dataset, allowed_root=trusted_dataset)
    except (OSError, RuntimeError, ValueError) as error:
        _record_dataset_changed(runtime, job_id, token, dataset_id)
        raise DatasetChangedError("evaluation dataset identity changed") from error
    actual_kind = inspection.kind.value if inspection.kind is not None else None
    if (
        not inspection.ready
        or actual_kind != expected_kind
        or inspection.fingerprint != expected_fingerprint
    ):
        _record_dataset_changed(runtime, job_id, token, dataset_id)
        raise DatasetChangedError("evaluation dataset identity changed")
    return inspection


def _trusted_evaluation_output(
    runtime: TaskRuntime,
    job_id: str,
    persisted_output: str | None,
) -> Path:
    runs_root = validate_trusted_directory(runtime.config.data_root / "runs", "trusted runs root")
    expected = runs_root / job_id
    if persisted_output is not None and persisted_output != str(expected):
        raise ValueError("evaluation output must match the canonical job output path")
    try:
        if not os.path.lexists(expected):
            expected.mkdir(mode=0o700)
        return validate_published_target(expected, runs_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("evaluation output path failed trust validation") from error


def _trusted_profile_path(runtime: TaskRuntime, profile_name: str) -> Path:
    if not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
        raise ValueError("evaluation profile selector must be a safe identifier")
    root = validate_trusted_readable_directory(
        Path(os.path.abspath(runtime.profiles_root)), "trusted profiles root"
    )
    candidate = root / f"{profile_name}.yaml"
    try:
        candidate_stat = os.lstat(candidate)
    except OSError as error:
        raise ValueError("evaluation profile file does not exist") from error
    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
        raise ValueError("evaluation profile file must be a regular non-symlink file")
    if candidate_stat.st_uid not in {0, os.geteuid()}:
        raise ValueError("evaluation profile file has an untrusted owner")
    if candidate_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("evaluation profile file must not be group or other writable")
    if not os.access(candidate, os.R_OK):
        raise ValueError("evaluation profile file is not readable")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("evaluation profile file escaped its trusted root") from error
    return candidate


def recover_interrupted_jobs(*, runtime: TaskRuntime | None = None) -> int:
    resolved = _require_runtime(runtime)
    evaluation_states = {"RUNNING", "METRICS", "VLM", "REPORT"}
    import_states = {"CONNECTING", "TRANSFERRING", "VERIFYING", "PREFLIGHT"}
    with session_scope(resolved.engine) as session:
        evaluation_snapshots = tuple(
            session.execute(
                select(EvaluationJob.id, EvaluationJob.execution_token).where(
                    EvaluationJob.state.in_(evaluation_states)
                )
            )
        )
        import_snapshots = tuple(
            session.execute(
                select(ImportJob.id, ImportJob.execution_token).where(
                    ImportJob.state.in_(import_states)
                )
            )
        )

    recovered = 0
    with session_scope(resolved.engine) as session:
        for job_id, snapshot_token in evaluation_snapshots:
            token_condition = (
                EvaluationJob.execution_token.is_(None)
                if snapshot_token is None
                else EvaluationJob.execution_token == snapshot_token
            )
            changed = session.execute(
                update(EvaluationJob)
                .where(
                    EvaluationJob.id == job_id,
                    EvaluationJob.state.in_(evaluation_states),
                    token_condition,
                )
                .values(state="INTERRUPTED", execution_token=None)
            )
            recovered += int(changed.rowcount == 1)
        for import_id, snapshot_token in import_snapshots:
            token_condition = (
                ImportJob.execution_token.is_(None)
                if snapshot_token is None
                else ImportJob.execution_token == snapshot_token
            )
            changed = session.execute(
                update(ImportJob)
                .where(
                    ImportJob.id == import_id,
                    ImportJob.state.in_(import_states),
                    token_condition,
                )
                .values(state="INTERRUPTED", execution_token=None)
            )
            recovered += int(changed.rowcount == 1)
    return recovered


def _claim_import_execution(runtime: TaskRuntime, import_id: str, token: str) -> None:
    with session_scope(runtime.engine) as session:
        claimed = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.state.in_(_IMPORT_CLAIM_STATES),
                ImportJob.execution_token.is_(None),
            )
            .values(
                state="CONNECTING",
                execution_token=token,
                error_code=None,
                error_message=None,
            )
        )
        if claimed.rowcount != 1:
            raise StaleTaskExecution("import execution could not claim the job")


def _update_import_state(runtime: TaskRuntime, import_id: str, token: str, state: str) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
            )
            .values(state=state)
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("import state callback is stale")


def _update_import_progress(
    runtime: TaskRuntime, import_id: str, token: str, progress: float
) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
            )
            .values(progress=progress)
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("import progress callback is stale")


def _import_cancel_requested(runtime: TaskRuntime, import_id: str, token: str) -> bool:
    with session_scope(runtime.engine) as session:
        value = session.scalar(
            select(ImportJob.cancel_requested).where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
            )
        )
        if value is None:
            raise StaleTaskExecution("import cancellation callback is stale")
        return value


def _record_publish_fingerprint(
    runtime: TaskRuntime,
    import_id: str,
    token: str,
    fingerprint: str,
) -> None:
    with session_scope(runtime.engine) as session:
        changed = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
                ImportJob.dataset_id.is_(None),
            )
            .values(publish_fingerprint=fingerprint)
        )
        if changed.rowcount != 1:
            raise StaleTaskExecution("import fingerprint callback is stale")


def _record_import_success(
    runtime: TaskRuntime,
    import_id: str,
    token: str,
    target_name: str,
    result: ImportResult,
    *,
    reconciliation: bool = False,
) -> bool:
    inspection = result.inspection
    with session_scope(runtime.engine) as session:
        claim = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.dataset_id.is_(None),
                ImportJob.publish_fingerprint == inspection.fingerprint,
                ImportJob.state == ("CONNECTING" if reconciliation else "PREFLIGHT"),
                ImportJob.cancel_requested.is_(False),
            )
            .values(state="FINALIZING")
            .execution_options(synchronize_session=False)
        )
        if claim.rowcount != 1:
            return False
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
        job.execution_token = None
    return True


def _record_import_failure(
    runtime: TaskRuntime,
    import_id: str,
    token: str,
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
        session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
            )
            .values(
                state="FAILED",
                execution_token=None,
                error_code=code,
                error_message=message,
            )
        )


def _record_import_cancelled(runtime: TaskRuntime, import_id: str, token: str) -> None:
    with session_scope(runtime.engine) as session:
        session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.execution_token == token,
                ImportJob.state.in_(_IMPORT_ACTIVE_STATES),
            )
            .values(
                state="CANCELLED",
                execution_token=None,
                error_code="IMPORT_CANCELLED",
                error_message="Dataset import was cancelled.",
            )
        )


def _record_import_integrity_failure(
    runtime: TaskRuntime,
    import_id: str,
    dataset_id: str | None,
    publish_fingerprint: str | None,
) -> None:
    if dataset_id is None or publish_fingerprint is None:
        return
    with session_scope(runtime.engine) as session:
        degraded = session.execute(
            update(ImportJob)
            .where(
                ImportJob.id == import_id,
                ImportJob.state == "READY",
                ImportJob.dataset_id == dataset_id,
                ImportJob.publish_fingerprint == publish_fingerprint,
            )
            .values(
                state="FAILED",
                error_code="IMPORT_INTEGRITY_FAILED",
                error_message=(
                    "Published dataset integrity check failed. Review or re-import the dataset."
                ),
            )
            .execution_options(synchronize_session=False)
        )
        if degraded.rowcount == 1:
            session.execute(
                update(Dataset)
                .where(Dataset.id == dataset_id)
                .values(status="PREFLIGHT_FAILED")
                .execution_options(synchronize_session=False)
            )


def _inspect_trusted_published_target(target: Path, inbox_root: Path) -> DatasetInspection:
    try:
        validated_target = validate_published_target(target, inbox_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ImportIntegrityError(str(error)) from error
    try:
        return inspect_dataset(validated_target, allowed_root=validated_target)
    except (OSError, RuntimeError, ValueError) as error:
        raise ImportIntegrityError("published import target could not be inspected") from error


def _load_completed_import(
    runtime: TaskRuntime,
    import_id: str,
    dataset_id: str | None,
    publish_fingerprint: str | None,
    target: Path,
) -> ImportResult:
    if dataset_id is None or publish_fingerprint is None:
        raise ImportIntegrityError("completed import is missing its dataset identity")
    inspection = _inspect_trusted_published_target(
        target,
        runtime.config.data_root / "inbox",
    )
    with session_scope(runtime.engine) as session:
        dataset = session.get(Dataset, dataset_id)
        job = session.get(ImportJob, import_id)
        if dataset is None or job is None:
            raise ImportIntegrityError("completed import is missing its persisted identity")
        if dataset.status != "READY" or dataset.path != str(target):
            raise ImportIntegrityError("completed import dataset record is inconsistent")
        if not inspection.ready or inspection.fingerprint != publish_fingerprint:
            raise ImportIntegrityError("completed import fingerprint no longer matches its target")
        if dataset.fingerprint != publish_fingerprint:
            raise ImportIntegrityError("completed import dataset fingerprint is inconsistent")
        if (
            job.state != "READY"
            or job.dataset_id != dataset_id
            or job.publish_fingerprint != publish_fingerprint
        ):
            raise ImportIntegrityError("completed import state changed during validation")
    return ImportResult(target, inspection)


def _reconcile_interrupted_import(
    runtime: TaskRuntime,
    import_id: str,
    token: str,
    dataset_id: str | None,
    publish_fingerprint: str | None,
    staging: Path,
    target: Path,
    target_name: str,
) -> ImportResult:
    if dataset_id is not None:
        raise ValueError("interrupted import already has a dataset record")
    if publish_fingerprint is None:
        raise ValueError("interrupted import has no durable publish fingerprint")
    if os.path.lexists(staging):
        raise ValueError("interrupted import still has a staging path")
    inspection = _inspect_trusted_published_target(
        target,
        runtime.config.data_root / "inbox",
    )
    if not inspection.ready:
        raise ValueError("interrupted published target did not pass preflight")
    if inspection.fingerprint != publish_fingerprint:
        raise ValueError("interrupted published target fingerprint does not match")
    result = ImportResult(target, inspection)
    if _record_import_success(
        runtime,
        import_id,
        token,
        target_name,
        result,
        reconciliation=True,
    ):
        return result
    with session_scope(runtime.engine) as session:
        completed = session.get_one(ImportJob, import_id)
        completed_dataset_id = completed.dataset_id
        completed_fingerprint = completed.publish_fingerprint
    return _load_completed_import(
        runtime,
        import_id,
        completed_dataset_id,
        completed_fingerprint,
        target,
    )


def run_import_task(import_id: str, *, runtime: TaskRuntime | None = None) -> ImportResult:
    resolved = _require_runtime(runtime)
    with session_scope(resolved.engine) as session:
        job = session.get_one(ImportJob, import_id)
        source_name = job.source_name
        remote_root = job.remote_root
        remote_path = job.remote_path
        target_name = job.target_name
        state = job.state
        dataset_id = job.dataset_id
        publish_fingerprint = job.publish_fingerprint

    target_path = resolved.config.data_root / "inbox" / target_name
    staging_path = resolved.config.data_root / "staging" / import_id
    if state == "READY":
        try:
            return _load_completed_import(
                resolved,
                import_id,
                dataset_id,
                publish_fingerprint,
                target_path,
            )
        except ImportIntegrityError:
            _record_import_integrity_failure(
                resolved,
                import_id,
                dataset_id,
                publish_fingerprint,
            )
            raise
    token = str(uuid4())
    _claim_import_execution(resolved, import_id, token)
    captured_inspection: DatasetInspection | None = None

    def inspect_and_capture(path: Path) -> DatasetInspection:
        nonlocal captured_inspection
        captured_inspection = inspect_dataset(path, allowed_root=path)
        if captured_inspection.ready:
            _record_publish_fingerprint(
                resolved,
                import_id,
                token,
                captured_inspection.fingerprint,
            )
        return captured_inspection

    def update_state(next_state: str) -> None:
        if next_state == "FAILED":
            return
        if next_state == "READY":
            if captured_inspection is None:
                raise RuntimeError("import reached READY before dataset inspection completed")
            committed = _record_import_success(
                resolved,
                import_id,
                token,
                target_name,
                ImportResult(target_path, captured_inspection),
            )
            if not committed:
                raise ValueError("import state changed before READY commit")
            return
        _update_import_state(resolved, import_id, token, next_state)

    callbacks = ImportCallbacks(
        on_state=update_state,
        on_progress=lambda progress: _update_import_progress(resolved, import_id, token, progress),
        is_cancelled=lambda: _import_cancel_requested(resolved, import_id, token),
    )
    try:
        if callbacks.is_cancelled():
            raise TransferError("import cancelled before execution")
        if state == "INTERRUPTED" and os.path.lexists(target_path):
            return _reconcile_interrupted_import(
                resolved,
                import_id,
                token,
                dataset_id,
                publish_fingerprint,
                staging_path,
                target_path,
                target_name,
            )
        configured_source = resolved.config.remote_sources[source_name]
        if remote_root not in configured_source.roots:
            raise ValueError("persisted remote root is not registered for the selected source")
        source = replace(configured_source, roots=(remote_root,))
        spec = ImportSpec(
            job_id=import_id,
            source_name=source.name,
            remote_root=remote_root,
            remote_relative_path=remote_path,
            staging_path=staging_path,
            target_path=target_path,
            mode="production",
            source=source,
            trusted_credentials_root=resolved.credentials_root,
            trusted_staging_root=resolved.config.data_root / "staging",
            trusted_inbox_root=resolved.config.data_root / "inbox",
        )
        result = execute_import(spec, inspector=inspect_and_capture, callbacks=callbacks)
    except BaseException as original:
        try:
            try:
                cancelled = callbacks.is_cancelled()
            except StaleTaskExecution:
                cancelled = False
            if cancelled:
                _record_import_cancelled(resolved, import_id, token)
            else:
                _record_import_failure(resolved, import_id, token, original)
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
        dataset_status = dataset.status
        dataset_id = dataset.id
        dataset_kind = dataset.kind
        dataset_fingerprint = dataset.fingerprint
        profile_name = job.profile_name
        profile_version = job.profile_version
        vlm_enabled = job.vlm_enabled
        persisted_output = job.output_dir
        resume_from = job.stage if job.stage in {"VLM", "REPORT"} else "METRICS"
        initial_progress = job.progress
    token = str(uuid4())
    _claim_evaluation_execution(resolved, job_id, token)
    _update_evaluation_stage(resolved, job_id, token, "PREFLIGHT")

    callbacks = EvaluationCallbacks(
        on_stage=lambda stage: _update_evaluation_stage(resolved, job_id, token, stage),
        on_progress=lambda progress: _update_evaluation_progress(resolved, job_id, token, progress),
        should_cancel=lambda: _evaluation_cancel_requested(resolved, job_id, token),
    )
    try:
        if callbacks.should_cancel():
            raise EvaluationCancelled("evaluation cancelled before execution")
        if dataset_status != "READY":
            raise ValueError("evaluation dataset is not READY")
        _verify_evaluation_dataset_identity(
            resolved,
            job_id,
            token,
            dataset_id,
            Path(dataset_path),
            dataset_kind,
            dataset_fingerprint,
        )
        trusted_dataset = Path(dataset_path)
        output_dir = _trusted_evaluation_output(resolved, job_id, persisted_output)
        _set_evaluation_output(resolved, job_id, token, output_dir)
        profile = load_profile(_trusted_profile_path(resolved, profile_name))
        if profile.name != profile_name:
            raise ValueError("evaluation profile name changed after job submission")
        if profile.version != profile_version:
            raise ValueError("evaluation profile version changed after job submission")
        result = run_evaluation(
            dataset_path=str(trusted_dataset),
            output_dir=str(output_dir),
            profile=profile,
            vlm_enabled=vlm_enabled,
            callbacks=callbacks,
            resume_from=resume_from,
            initial_progress=initial_progress,
        )
        _record_evaluation_success(resolved, job_id, token)
    except EvaluationCancelled as original:
        try:
            _record_evaluation_cancelled(resolved, job_id, token)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve worker interrupts
            raise BaseExceptionGroup(
                "evaluation cancellation and persistence both failed",
                [original, persistence_error],
            ) from original
        raise
    except BaseException as original:
        try:
            _record_evaluation_failure(resolved, job_id, token, original)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve worker interrupts
            raise BaseExceptionGroup(
                "evaluation and failure persistence both failed",
                [original, persistence_error],
            ) from original
        raise
    return result
