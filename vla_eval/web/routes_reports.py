"""Read-only report rendering and secure artifact download.

The report page reads persisted metrics artifacts directly (``metrics_core.json``,
``episode_metrics.csv`` and the optional ``attempt_eval/attempt_summary.json``);
it never recomputes metrics. Downloads are constrained to a fixed whitelist of
filenames and must resolve strictly inside the job's ``output_dir``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import Engine

from Genie02_report.genie02_eval_common import EvaluationError
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.security import require_html_user
from vla_eval.web.report_view import build_report_view
from vla_eval.web.templating import templates

router = APIRouter()

_REPORT_GLOB = "report_*.md"
_EXACT_ARTIFACTS = frozenset(
    {
        "metrics_core.json",
        "episode_metrics.csv",
        "smoothness_curve.svg",
        "attempt_eval/attempt_summary.json",
        "attempt_eval/attempt_summary.csv",
    }
)
_ARTIFACT_PRESENTATION = {
    "episode_metrics.csv": {"description": "Episode 逐项指标", "format": "CSV"},
    "metrics_core.json": {"description": "评测汇总指标", "format": "JSON"},
    "smoothness_curve.svg": {"description": "平滑度矢量图", "format": "SVG"},
    "attempt_eval/attempt_summary.json": {
        "description": "VLM 尝试汇总",
        "format": "JSON",
    },
    "attempt_eval/attempt_summary.csv": {
        "description": "VLM 尝试明细",
        "format": "CSV",
    },
}
_REPORT_PRESENTATION = {"description": "完整文本报告", "format": "MD"}
_OUTCOME_FILTERS = frozenset({"success", "failure", "all"})
_REVIEW_FILTERS = frozenset({"needs_review", "reviewed", "ok", "all"})


def _template_context(request: Request, current_user: User, **values):
    return {
        "current_user": current_user,
        "csrf_token": request.session["csrf_token"],
        **values,
    }


def _load_job(engine: Engine, job_id: str) -> EvaluationJob:
    with session_scope(engine) as session:
        job = session.get(EvaluationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return job


def _load_dataset(engine: Engine, dataset_id: str) -> Dataset:
    with session_scope(engine) as session:
        dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def _parse_report_filters(request: Request) -> tuple[str, str]:
    """Parse and strictly validate the optional outcome/review query filters."""
    outcome_values = request.query_params.getlist("outcome")
    review_values = request.query_params.getlist("review")
    if len(outcome_values) > 1 or len(review_values) > 1:
        raise HTTPException(status_code=422, detail="Invalid report filter")
    outcome = outcome_values[0].strip() if outcome_values else "all"
    review = review_values[0].strip() if review_values else "all"
    if outcome not in _OUTCOME_FILTERS or review not in _REVIEW_FILTERS:
        raise HTTPException(status_code=422, detail="Invalid report filter")
    return outcome, review


def _apply_review_filter(
    episodes: list[dict[str, Any]], review: str
) -> list[dict[str, Any]]:
    """Filter by VLM review state. Buckets partition ``needs_manual_review``:

    - ``needs_review``: ``True`` (auto mode, flagged with warnings)
    - ``reviewed``: ``None`` (manual_review mode, awaiting human review)
    - ``ok``: ``False`` (auto mode, clean)

    True / None / False are mutually exclusive and exhaustive across both
    review modes (see ``review_policy.apply_review_policy``).
    """
    if review == "needs_review":
        return [
            episode
            for episode in episodes
            if episode["vlm"] is not None
            and episode["vlm"].get("needs_manual_review") is True
        ]
    if review == "reviewed":
        return [
            episode
            for episode in episodes
            if episode["vlm"] is not None
            and episode["vlm"].get("needs_manual_review") is None
        ]
    if review == "ok":
        return [
            episode
            for episode in episodes
            if episode["vlm"] is not None
            and episode["vlm"].get("needs_manual_review") is False
        ]
    return episodes


def _available_downloads(output_dir: Path, job_id: str) -> list[dict[str, str]]:
    downloads: list[dict[str, str]] = []
    for name in sorted(_EXACT_ARTIFACTS):
        path = output_dir.joinpath(*PurePosixPath(name).parts)
        if path.is_file():
            downloads.append(
                {
                    "name": name,
                    "url": f"/reports/{job_id}/files/{quote(name, safe='/')}",
                    **_ARTIFACT_PRESENTATION[name],
                }
            )
    try:
        children = sorted(output_dir.glob(_REPORT_GLOB))
    except OSError:
        children = []
    for path in children:
        if path.is_file():
            downloads.append(
                {
                    "name": path.name,
                    "url": f"/reports/{job_id}/files/{path.name}",
                    **_REPORT_PRESENTATION,
                }
            )
    return downloads


def _safe_artifact_path(output_dir: Path, filename: str) -> Path:
    """Resolve an allowed relative artifact path strictly inside ``output_dir``."""
    if (
        not isinstance(filename, str)
        or not filename
        or "\\" in filename
        or any(ord(character) < 32 for character in filename)
    ):
        raise HTTPException(status_code=404, detail="File not found")
    relative_path = PurePosixPath(filename)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or str(relative_path) != filename
        or relative_path in {PurePosixPath("."), PurePosixPath("")}
    ):
        raise HTTPException(status_code=404, detail="File not found")
    relative_name = relative_path.as_posix()
    allowed_report = len(relative_path.parts) == 1 and fnmatchcase(
        relative_name, _REPORT_GLOB
    )
    if relative_name not in _EXACT_ARTIFACTS and not allowed_report:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        base = output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    candidate = base.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    try:
        relative = resolved.relative_to(base)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    if relative.as_posix() != relative_name or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return resolved


@router.get("/reports/{job_id}")
def report_detail(
    request: Request,
    job_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request.app.state.engine, job_id)
    if not job.output_dir:
        raise HTTPException(status_code=404, detail="Report is not available")
    output_dir = Path(job.output_dir)
    if not (output_dir / "metrics_core.json").is_file():
        raise HTTPException(status_code=404, detail="Report is not available")
    dataset = _load_dataset(request.app.state.engine, job.dataset_id)
    try:
        view = build_report_view(job=job, dataset=dataset, output_dir=output_dir)
    except (EvaluationError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=404, detail="Report is not available"
        ) from error
    has_vlm = view["has_vlm"]
    outcome_filter, review_filter = _parse_report_filters(request)
    episodes = list(view["episodes"])
    total_episodes = len(episodes)
    if outcome_filter != "all":
        episodes = [
            episode for episode in episodes if episode["outcome"] == outcome_filter
        ]
    # The review filter is a no-op when no VLM data is present.
    if has_vlm and review_filter != "all":
        episodes = _apply_review_filter(episodes, review_filter)
    shown_episodes = len(episodes)
    pending_review = view["pending_review"]
    provenance = job.provenance_json or {}
    provenance_view = {
        "profile_name": job.profile_name,
        "profile_version": job.profile_version,
        "vlm_model": provenance.get("vlm_model_path") or "—",
        "prompt_version": provenance.get("prompt_version") or "—",
        "app_version": provenance.get("app_version") or "—",
        "git_sha": provenance.get("git_sha") or "",
    }
    downloads = _available_downloads(output_dir, job.id)
    smoothness_preview_url = next(
        (
            item["url"]
            for item in downloads
            if item["name"] == "smoothness_curve.svg"
        ),
        None,
    )
    context = {
        **view,
        "job": job,
        "dataset": dataset,
        "episodes": episodes,
        "has_vlm": has_vlm,
        "pending_review": pending_review,
        "provenance": provenance_view,
        "downloads": downloads,
        "smoothness_preview_url": smoothness_preview_url,
        "total_episodes": total_episodes,
        "shown_episodes": shown_episodes,
        "filter_outcome": outcome_filter,
        "filter_review": review_filter,
    }
    return templates.TemplateResponse(
        request=request,
        name="reports/detail.html",
        context=_template_context(
            request,
            current_user,
            **context,
        ),
    )


@router.get("/reports/{job_id}/files/{filename:path}")
def report_download(
    request: Request,
    job_id: str,
    filename: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request.app.state.engine, job_id)
    if not job.output_dir:
        raise HTTPException(status_code=404, detail="File not found")
    path = _safe_artifact_path(Path(job.output_dir), filename)
    return FileResponse(path, filename=path.name)
