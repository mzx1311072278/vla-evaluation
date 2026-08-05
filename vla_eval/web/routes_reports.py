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

from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.security import require_html_user

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
    except OSError:
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
        episodes.append(
            {
                "index": index,
                "outcome": outcome,
                "duration": row.get("duration_s", "") or "—",
                "smoothness": row.get("smoothness", "") or "—",
                "vlm": attempt,
            }
        )
    return episodes


def _available_downloads(output_dir: Path, job_id: str) -> list[dict[str, str]]:
    downloads: list[dict[str, str]] = []
    for name in sorted(_EXACT_WHITELIST):
        path = output_dir / name
        if path.is_file():
            downloads.append(
                {"name": name, "url": f"/reports/{job_id}/files/{name}"}
            )
    try:
        children = sorted(output_dir.glob(_REPORT_GLOB))
    except OSError:
        children = []
    for path in children:
        if path.is_file():
            downloads.append(
                {"name": path.name, "url": f"/reports/{job_id}/files/{path.name}"}
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
    metrics = _load_core_metrics(output_dir)
    rows = _load_episode_rows(output_dir)
    attempts = _load_attempt_summary(output_dir)
    episodes = _build_episodes(rows, attempts)
    pending_review = sum(
        1
        for attempt in attempts.values()
        if attempt.get("needs_manual_review") is True
    )
    provenance = job.provenance_json or {}
    headline = {
        "gsr": _format_percent(metrics.get("gsr")),
        "n_success": metrics.get("n_success", "—"),
        "n_failure": metrics.get("n_failure", "—"),
        "tts": _format_float(metrics.get("mean_tts_success_s"), suffix=" s"),
        "smoothness": _smoothness_summary(metrics.get("smoothness")),
        "pending_review": pending_review,
    }
    provenance_view = {
        "profile_name": job.profile_name,
        "profile_version": job.profile_version,
        "vlm_model": provenance.get("vlm_model_path") or "—",
        "prompt_version": provenance.get("prompt_version") or "—",
        "app_version": provenance.get("app_version") or "—",
        "git_sha": provenance.get("git_sha") or "",
    }
    downloads = _available_downloads(output_dir, job.id)
    return templates.TemplateResponse(
        request=request,
        name="reports/detail.html",
        context=_template_context(
            request,
            current_user,
            job=job,
            dataset=dataset,
            headline=headline,
            episodes=episodes,
            has_vlm=bool(attempts),
            pending_review=pending_review,
            provenance=provenance_view,
            downloads=downloads,
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
