import hashlib
import json
import os
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError

import vla_eval
from vla_eval.datasets import inspect_dataset
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, EvaluationJobArchive, User
from vla_eval.profiles import Profile, load_profile
from vla_eval.security import require_csrf, require_html_user
from vla_eval.tasks import run_evaluation_task
from vla_eval.web.list_management import (
    ListControls,
    literal_contains_pattern,
    parse_list_controls,
    validate_return_to,
)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"})
_ACTIVE_STATES = frozenset({"QUEUED", "PREFLIGHT", "METRICS", "VLM", "REPORT", "RUNNING"})
_JOB_STATES = tuple(sorted(_TERMINAL_STATES | _ACTIVE_STATES))
_ARCHIVABLE_JOB_STATES = _TERMINAL_STATES
_ARCHIVE_FORM_FIELDS = frozenset({"csrf_token", "return_to"})
_ALLOWED_FIELDS = frozenset(
    {"csrf_token", "dataset_id", "profile", "vlm_enabled", "force"}
)
_PROFILE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _template_context(request: Request, current_user: User, **values):
    return {
        "current_user": current_user,
        "csrf_token": request.session["csrf_token"],
        **values,
    }


def _single_form_value(form, field: str) -> str:
    values = form.getlist(field)
    if len(values) != 1 or not isinstance(values[0], str):
        raise HTTPException(status_code=422, detail="Invalid evaluation form")
    return values[0]


def _load_job(request: Request, job_id: str) -> EvaluationJob:
    with session_scope(request.app.state.engine) as session:
        job = session.get(EvaluationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return job


def _load_dataset(request: Request, dataset_id: str) -> Dataset:
    with session_scope(request.app.state.engine) as session:
        dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


async def _archive_form_target(request: Request, job_id: str) -> str:
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    if set(form.keys()) != _ARCHIVE_FORM_FIELDS:
        raise HTTPException(status_code=422, detail="Invalid evaluation archive form")
    return_values = form.getlist("return_to")
    if len(return_values) != 1 or not isinstance(return_values[0], str):
        raise HTTPException(status_code=422, detail="Invalid evaluation archive form")
    return validate_return_to(
        return_values[0],
        allowed_paths=frozenset(
            {
                "/evaluations",
                f"/evaluations/{job_id}",
                f"/reports/{job_id}",
            }
        ),
        fallback="/evaluations",
    )


def _evaluation_order(controls: ListControls):
    if controls.sort == "oldest":
        return (EvaluationJob.created_at.asc(), EvaluationJob.id.asc())
    if controls.sort == "name_asc":
        return (
            func.lower(Dataset.name).asc(),
            Dataset.name.asc(),
            EvaluationJob.created_at.desc(),
            EvaluationJob.id.asc(),
        )
    if controls.sort == "name_desc":
        return (
            func.lower(Dataset.name).desc(),
            Dataset.name.desc(),
            EvaluationJob.created_at.desc(),
            EvaluationJob.id.desc(),
        )
    return (EvaluationJob.created_at.desc(), EvaluationJob.id.desc())


def _current_local_url(request: Request) -> str:
    query = request.url.query
    return request.url.path + (f"?{query}" if query else "")


def _profiles_root() -> Path:
    return Path(os.environ.get("VLA_EVAL_PROFILES_ROOT", "config/profiles"))


def _load_profile_for_submission(name: str) -> Profile:
    """Resolve a submitted profile name to a loaded Profile, rejecting unsafe input."""
    if not isinstance(name, str) or not _PROFILE_NAME_PATTERN.fullmatch(name):
        raise HTTPException(status_code=422, detail="Invalid evaluation profile")
    profiles_root = _profiles_root()
    candidate = profiles_root / f"{name}.yaml"
    try:
        resolved_root = profiles_root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Invalid evaluation profile") from error
    if resolved_root not in resolved_candidate.parents:
        raise HTTPException(status_code=422, detail="Invalid evaluation profile")
    try:
        return load_profile(resolved_candidate)
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail="Invalid evaluation profile") from error


def _discover_profiles() -> list[str]:
    """List available profile stems under the profiles root, safely contained.

    Discovery is best-effort and only feeds the selector; POST-side validation
    (`_load_profile_for_submission`) remains authoritative.
    """
    profiles_root = _profiles_root()
    try:
        resolved_root = profiles_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return []
    names: list[str] = []
    try:
        entries = list(profiles_root.iterdir())
    except (OSError, RuntimeError):
        return []
    for entry in entries:
        if not entry.name.endswith(".yaml") or not entry.is_file():
            continue
        stem = entry.name[: -len(".yaml")]
        if not _PROFILE_NAME_PATTERN.fullmatch(stem):
            continue
        try:
            resolved_entry = entry.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved_root in resolved_entry.parents:
            names.append(stem)
    return sorted(names)


def _evaluation_phases(vlm_enabled: bool) -> tuple[tuple[str, str], ...]:
    phases: list[tuple[str, str]] = [("PREFLIGHT", "预检"), ("METRICS", "指标")]
    if vlm_enabled:
        phases.append(("VLM", "VLM"))
    phases.append(("REPORT", "报告"))
    return tuple(phases)


def _verify_retry_dataset_identity(
    request: Request, job: EvaluationJob, dataset: Dataset
) -> None:
    """Re-verify the on-disk dataset fingerprint matches the submit-time snapshot.

    The worker re-verifies strictly at execution time; this guard gives the user a
    clear 409 before re-enqueueing a retry against a dataset whose contents have
    drifted since submission.
    """
    try:
        inbox = (request.app.state.config.data_root / "inbox").resolve(strict=True)
        raw_path = Path(dataset.path)
        if not raw_path.is_absolute():
            raise ValueError("dataset path must be absolute")
        resolved_path = raw_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail="Dataset path is no longer valid. Re-import or re-validate the dataset before retrying.",
        ) from error
    if resolved_path == inbox or inbox not in resolved_path.parents or not resolved_path.is_dir():
        raise HTTPException(
            status_code=409,
            detail="Dataset path is no longer valid. Re-import or re-validate the dataset before retrying.",
        )
    try:
        inspection = inspect_dataset(resolved_path, allowed_root=resolved_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail="Dataset is no longer valid. Re-import or re-validate the dataset before retrying.",
        ) from error
    if not inspection.ready:
        raise HTTPException(
            status_code=409,
            detail="Dataset is no longer valid. Re-import or re-validate the dataset before retrying.",
        )
    stored_fingerprint = (
        job.provenance_json.get("dataset_fingerprint")
        if job.provenance_json
        else None
    )
    if (
        isinstance(stored_fingerprint, str)
        and stored_fingerprint
        and inspection.fingerprint != stored_fingerprint
    ):
        raise HTTPException(
            status_code=409,
            detail="Dataset contents changed after submission. Create a new evaluation instead of retrying.",
        )


@router.get("/evaluations/new")
def evaluation_new(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    dataset_id_values = request.query_params.getlist("dataset_id")
    if len(dataset_id_values) != 1 or not dataset_id_values[0]:
        raise HTTPException(status_code=422, detail="Invalid dataset parameter")
    dataset = _load_dataset(request, dataset_id_values[0])
    if dataset.status != "READY":
        raise HTTPException(status_code=422, detail="Dataset is not ready for evaluation")
    profile_names = _discover_profiles()
    default_profile = (
        "genie02-full"
        if "genie02-full" in profile_names
        else (profile_names[0] if profile_names else "")
    )
    return templates.TemplateResponse(
        request=request,
        name="evaluations/new.html",
        context=_template_context(
            request,
            current_user,
            dataset=dataset,
            profiles=profile_names,
            profile_name=default_profile,
        ),
    )


@router.get("/evaluations")
def evaluation_list(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    controls = parse_list_controls(
        request,
        allowed_extra=frozenset({"state", "dataset_id"}),
    )
    state_values = request.query_params.getlist("state")
    dataset_id_values = request.query_params.getlist("dataset_id")
    if len(state_values) > 1 or len(dataset_id_values) > 1:
        raise HTTPException(status_code=422, detail="Invalid evaluation filters")
    selected_state = state_values[0] if state_values else ""
    dataset_id = dataset_id_values[0] if dataset_id_values else ""
    if selected_state and selected_state not in _JOB_STATES:
        raise HTTPException(status_code=422, detail="Invalid evaluation state")
    if dataset_id:
        _load_dataset(request, dataset_id)

    query = select(EvaluationJob, Dataset).join(
        Dataset, EvaluationJob.dataset_id == Dataset.id
    ).outerjoin(
        EvaluationJobArchive,
        EvaluationJobArchive.evaluation_job_id == EvaluationJob.id,
    )
    if not controls.include_archived:
        query = query.where(EvaluationJobArchive.evaluation_job_id.is_(None))
    if controls.q:
        pattern = literal_contains_pattern(controls.q)
        query = query.where(
            or_(
                Dataset.name.ilike(pattern, escape="\\"),
                EvaluationJob.profile_name.ilike(pattern, escape="\\"),
                EvaluationJob.id.ilike(pattern, escape="\\"),
            )
        )
    if selected_state:
        query = query.where(EvaluationJob.state == selected_state)
    if dataset_id:
        query = query.where(EvaluationJob.dataset_id == dataset_id)
    query = query.order_by(*_evaluation_order(controls))
    with session_scope(request.app.state.engine) as session:
        jobs = session.execute(query).all()
    return templates.TemplateResponse(
        request=request,
        name="evaluations/index.html",
        context=_template_context(
            request,
            current_user,
            jobs=jobs,
            job_states=_JOB_STATES,
            selected_state=selected_state,
            dataset_id=dataset_id,
            controls=controls,
            current_url=_current_local_url(request),
        ),
    )
@router.post("/evaluations")
async def create_evaluation(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))

    unknown = set(form.keys()) - _ALLOWED_FIELDS
    if unknown:
        raise HTTPException(status_code=422, detail="Invalid evaluation form")

    dataset_id = _single_form_value(form, "dataset_id")
    profile_selector = _single_form_value(form, "profile")
    vlm_enabled_raw = _single_form_value(form, "vlm_enabled")
    force_values = form.getlist("force")
    if len(force_values) > 1:
        raise HTTPException(status_code=422, detail="Invalid evaluation form")
    force_raw = force_values[0] if force_values else "false"
    if vlm_enabled_raw not in ("true", "false") or force_raw not in ("true", "false"):
        raise HTTPException(status_code=422, detail="Invalid evaluation form")
    vlm_enabled = vlm_enabled_raw == "true"
    force = force_raw == "true"

    dataset = _load_dataset(request, dataset_id)
    if dataset.status != "READY":
        raise HTTPException(status_code=422, detail="Dataset is not ready for evaluation")
    profile = _load_profile_for_submission(profile_selector)

    params = {"vlm_enabled": vlm_enabled}
    canonical_params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    run_key = hashlib.sha256(
        (
            (dataset.fingerprint or "")
            + profile.name
            + profile.version
            + canonical_params_json
        ).encode("utf-8")
    ).hexdigest()

    if not force:
        with session_scope(request.app.state.engine) as session:
            existing = session.scalar(
                select(EvaluationJob).where(
                    EvaluationJob.state == "SUCCEEDED",
                    or_(
                        EvaluationJob.run_key == run_key,
                        and_(
                            EvaluationJob.run_key.is_(None),
                            EvaluationJob.dataset_id == dataset.id,
                            EvaluationJob.profile_name == profile.name,
                            EvaluationJob.profile_version == profile.version,
                            EvaluationJob.vlm_enabled == vlm_enabled,
                        ),
                    ),
                )
            )
            existing_id = existing.id if existing is not None else None
        if existing_id is not None:
            return RedirectResponse(
                f"/evaluations/{existing_id}",
                status_code=303,
            )

    # API-backend connection details are recorded only for backend=api. The
    # api_key_env is the env-var NAME; the secret VALUE is never persisted.
    api_provenance = {}
    if profile.vlm.backend == "api" and profile.vlm.api is not None:
        api = profile.vlm.api
        api_provenance = {
            "vlm_api_base_url": api.base_url,
            "vlm_api_model": api.model,
            "vlm_api_key_env": api.api_key_env,
            "vlm_api_timeout": api.timeout,
            "vlm_api_max_retries": api.max_retries,
        }
    provenance = {
        "dataset_fingerprint": dataset.fingerprint,
        "profile_name": profile.name,
        "profile_version": profile.version,
        "app_version": vla_eval.__version__,
        "git_sha": os.environ.get("VLA_EVAL_GIT_SHA", ""),
        "image_key": profile.image_key,
        "adapter": profile.adapter,
        "plugin": profile.plugin,
        "vlm_backend": profile.vlm.backend,
        "vlm_model_path": profile.vlm.model_path,
        "prompt_version": profile.vlm.prompt_version,
        "max_image_size": profile.vlm.max_image_size,
        "max_new_tokens": profile.vlm.max_new_tokens,
        "sampling": {
            "max_global_frames": profile.vlm.sampling.max_global_frames,
            "global_sample_interval": profile.vlm.sampling.global_sample_interval,
            "max_dense_frames": profile.vlm.sampling.max_dense_frames,
            "dense_sample_interval": profile.vlm.sampling.dense_sample_interval,
            "dense_region": profile.vlm.sampling.dense_region,
        },
        "review": {
            "mode": profile.review.mode,
            "confidence_threshold": profile.review.confidence_threshold,
            "min_episode_duration": profile.review.min_episode_duration,
            "min_sampled_frames": profile.review.min_sampled_frames,
        },
        "outputs": {
            "required": list(profile.outputs.required),
            "optional": list(profile.outputs.optional),
        },
        "params": params,
        **api_provenance,
    }

    with session_scope(request.app.state.engine) as session:
        job = EvaluationJob(
            dataset_id=dataset.id,
            profile_name=profile.name,
            profile_version=profile.version,
            vlm_enabled=vlm_enabled,
            state="QUEUED",
            stage="PENDING",
            run_key=run_key,
            params_json=params,
            provenance_json=provenance,
            created_by=current_user.id,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    try:
        request.app.state.queues.evaluation.enqueue(run_evaluation_task, job_id)
    except Exception as error:
        with session_scope(request.app.state.engine) as session:
            removed = session.execute(
                delete(EvaluationJob).where(
                    EvaluationJob.id == job_id,
                    EvaluationJob.state == "QUEUED",
                    EvaluationJob.execution_token.is_(None),
                )
            )
        if removed.rowcount == 1:
            raise HTTPException(
                status_code=503, detail="Evaluation queue unavailable"
            ) from error
        return RedirectResponse(f"/evaluations/{job_id}", status_code=303)

    return RedirectResponse(f"/evaluations/{job_id}", status_code=303)


@router.get("/evaluations/{job_id}")
def evaluation_detail(
    request: Request,
    job_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request, job_id)
    dataset = _load_dataset(request, job.dataset_id)
    phases = _evaluation_phases(job.vlm_enabled)
    phase_states = [state for state, _label in phases]
    phase_index = phase_states.index(job.stage) if job.stage in phase_states else -1
    return templates.TemplateResponse(
        request=request,
        name="evaluations/detail.html",
        context=_template_context(
            request,
            current_user,
            job=job,
            dataset=dataset,
            evaluation_phases=phases,
            phase_index=phase_index,
            active_states=sorted(_ACTIVE_STATES),
            retry_states=["FAILED", "INTERRUPTED"],
        ),
    )


@router.get("/api/evaluations/{job_id}")
def evaluation_status(
    request: Request,
    job_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request, job_id)
    finished = job.state in _TERMINAL_STATES
    response = JSONResponse(
        {
            "id": job.id,
            "state": job.state,
            "stage": job.stage,
            "progress": job.progress,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "finished": finished,
        }
    )
    if finished:
        response.headers["HX-Trigger"] = "job-finished"
    return response


@router.post("/evaluations/{job_id}/archive")
async def archive_evaluation(
    request: Request,
    job_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    return_to = await _archive_form_target(request, job_id)
    try:
        with session_scope(request.app.state.engine) as session:
            job = session.get(EvaluationJob, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Evaluation job not found")
            if job.state not in _ARCHIVABLE_JOB_STATES:
                raise HTTPException(
                    status_code=409,
                    detail="Evaluation cannot be archived",
                )
            session.add(
                EvaluationJobArchive(
                    evaluation_job_id=job_id,
                    archived_by=current_user.id,
                )
            )
            session.flush()
    except IntegrityError as error:
        raise HTTPException(
            status_code=409,
            detail="Evaluation is already archived",
        ) from error
    return RedirectResponse(return_to, status_code=303)


@router.post("/evaluations/{job_id}/restore")
async def restore_evaluation(
    request: Request,
    job_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    return_to = await _archive_form_target(request, job_id)
    with session_scope(request.app.state.engine) as session:
        job = session.get(EvaluationJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        removed = session.execute(
            delete(EvaluationJobArchive).where(
                EvaluationJobArchive.evaluation_job_id == job_id
            )
        )
        if removed.rowcount != 1:
            raise HTTPException(
                status_code=409,
                detail="Evaluation is not archived",
            )
    return RedirectResponse(return_to, status_code=303)


@router.post("/evaluations/{job_id}/retry")
async def retry_evaluation(
    request: Request,
    job_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    job = _load_job(request, job_id)
    if job.state not in ("FAILED", "INTERRUPTED"):
        raise HTTPException(status_code=409, detail="Evaluation cannot be retried")
    dataset = _load_dataset(request, job.dataset_id)
    if dataset.status != "READY":
        raise HTTPException(
            status_code=422, detail="Dataset is not ready for evaluation"
        )
    _verify_retry_dataset_identity(request, job, dataset)
    previous_snapshot = {
        "state": job.state,
        "stage": job.stage,
        "progress": job.progress,
        "execution_token": job.execution_token,
        "cancel_requested": job.cancel_requested,
        "error_code": job.error_code,
        "error_message": job.error_message,
    }
    with session_scope(request.app.state.engine) as session:
        session.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == job_id)
            .values(
                state="QUEUED",
                stage="PENDING",
                progress=0.0,
                execution_token=None,
                error_code=None,
                error_message=None,
                cancel_requested=False,
            )
        )
    try:
        request.app.state.queues.evaluation.enqueue(run_evaluation_task, job_id)
    except Exception as error:
        with session_scope(request.app.state.engine) as session:
            restored = session.execute(
                update(EvaluationJob)
                .where(
                    EvaluationJob.id == job_id,
                    EvaluationJob.state == "QUEUED",
                    EvaluationJob.execution_token.is_(None),
                )
                .values(**previous_snapshot)
            )
        if restored.rowcount == 1:
            raise HTTPException(
                status_code=503, detail="Evaluation queue unavailable"
            ) from error
        return RedirectResponse(f"/evaluations/{job_id}", status_code=303)

    return RedirectResponse(f"/evaluations/{job_id}", status_code=303)


@router.post("/evaluations/{job_id}/cancel")
async def cancel_evaluation(
    request: Request,
    job_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    job = _load_job(request, job_id)
    if job.state not in _ACTIVE_STATES:
        raise HTTPException(status_code=409, detail="Evaluation cannot be cancelled")
    with session_scope(request.app.state.engine) as session:
        session.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == job_id)
            .values(cancel_requested=True)
        )
    return RedirectResponse(f"/evaluations/{job_id}", status_code=303)
