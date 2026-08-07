"""Read-only report rendering and secure artifact download.

The report page reads persisted metrics artifacts directly (``metrics_core.json``,
``episode_metrics.csv`` and the optional ``attempt_eval/attempt_summary.json``);
it never recomputes metrics. Downloads are constrained to a fixed whitelist of
filenames and must resolve strictly inside the job's ``output_dir``.
"""

from __future__ import annotations

import csv
import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine

from Genie02_report.genie02_eval_common import EvaluationError
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.security import require_html_user
from vla_eval.web.report_view import build_report_view

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

_REPORT_GLOB = "report_*.md"
_EXACT_WHITELIST = frozenset(
    {
        "metrics_core.json",
        "episode_metrics.csv",
        "attempt_summary.json",
        "attempt_summary.csv",
        "smoothness_curve.svg",
    }
)
_ARTIFACT_PRESENTATION = {
    "episode_metrics.csv": {"description": "Episode 逐项指标", "format": "CSV"},
    "metrics_core.json": {"description": "评测汇总指标", "format": "JSON"},
    "smoothness_curve.svg": {"description": "平滑度矢量图", "format": "SVG"},
    "attempt_summary.json": {"description": "VLM 尝试汇总", "format": "JSON"},
    "attempt_summary.csv": {"description": "VLM 尝试明细", "format": "CSV"},
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


def _format_percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _format_float(value: Any, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    try:
        return f"{float(value):.3f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _smoothness_summary(smoothness: Any) -> str:
    if not isinstance(smoothness, dict):
        return "—"
    space = smoothness.get("space") or "—"
    left = smoothness.get("left")
    left_mean = left.get("mean") if isinstance(left, dict) else None
    if isinstance(left_mean, (int, float)) and not isinstance(left_mean, bool):
        return f"{space} · 平均 {float(left_mean):.3f}"
    return f"{space}"


def _load_core_metrics(output_dir: Path) -> dict[str, Any]:
    try:
        return json.loads(
            (output_dir / "metrics_core.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=404, detail="Report is not available"
        ) from error


def _load_episode_rows(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / "episode_metrics.csv"
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _load_attempt_summary(output_dir: Path) -> dict[int, dict[str, Any]]:
    path = output_dir / "attempt_eval" / "attempt_summary.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(loaded, list):
        return {}
    attempts: dict[int, dict[str, Any]] = {}
    for row in loaded:
        if not isinstance(row, dict):
            continue
        index = row.get("episode_index")
        if isinstance(index, int) and not isinstance(index, bool):
            attempts[index] = row
    return attempts


def _format_timestamp(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _episode_evidence(attempt: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return (video_path, timestamp_range) evidence locator for a failed episode.

    The attempt-eval writer stores ``video_file``/``from_timestamp``/``to_timestamp``
    on each summary row; ``load_attempt_summary`` preserves these extra fields.
    """
    if not attempt:
        return None, None
    video_file = attempt.get("video_file")
    evidence_path = video_file if isinstance(video_file, str) and video_file else None
    from_ts = _format_timestamp(attempt.get("from_timestamp"))
    to_ts = _format_timestamp(attempt.get("to_timestamp"))
    evidence_range: str | None = None
    if from_ts is not None or to_ts is not None:
        evidence_range = f"{from_ts or '—'} → {to_ts or '—'}"
    return evidence_path, evidence_range


def _build_episodes(
    rows: list[dict[str, str]], attempts: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for row in rows:
        try:
            index = int(row.get("episode_index", ""))
        except (TypeError, ValueError):
            index = None
        outcome = row.get("outcome", "") or ""
        attempt = attempts.get(index) if index is not None else None
        evidence_path: str | None = None
        evidence_range: str | None = None
        if outcome == "failure":
            evidence_path, evidence_range = _episode_evidence(attempt)
        episodes.append(
            {
                "index": index,
                "outcome": outcome,
                "duration": row.get("duration_s", "") or "—",
                "smoothness": row.get("smoothness", "") or "—",
                "vlm": attempt,
                "evidence_path": evidence_path,
                "evidence_range": evidence_range,
            }
        )
    return episodes


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
    for name in sorted(_EXACT_WHITELIST):
        path = output_dir / name
        if path.is_file():
            downloads.append(
                {
                    "name": name,
                    "url": f"/reports/{job_id}/files/{name}",
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
    """Resolve ``filename`` under ``output_dir`` enforcing whitelist + containment.

    The candidate is resolved strictly (following symlinks, requiring existence),
    must live directly inside ``output_dir`` (no subdirectories, no traversal),
    and its relative name must match the whitelist. Anything else -> 404.
    """
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        base = output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    candidate = base / filename
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    try:
        relative = resolved.relative_to(base)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="File not found") from error
    rel_posix = relative.as_posix()
    if "/" in rel_posix:
        raise HTTPException(status_code=404, detail="File not found")
    if rel_posix not in _EXACT_WHITELIST and not fnmatchcase(rel_posix, _REPORT_GLOB):
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
