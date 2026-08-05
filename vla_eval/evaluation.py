"""Synchronous orchestration for deterministic and optional VLM evaluation stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Genie02_report.genie02_episode_metrics import generate_episode_metrics
from Genie02_report.genie02_eval_common import load_metrics_core, load_session
from Genie02_report.genie02_markdown_report import generate_markdown_report
from Genie02_report.genie02_metrics_core import generate_metrics_core

from .profiles import Profile


class EvaluationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationCallbacks:
    on_stage: Callable[[str], None]
    on_progress: Callable[[float], None]
    should_cancel: Callable[[], bool]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    report_path: Path
    vlm_summary_path: Path | None


def _check_cancelled(callbacks: EvaluationCallbacks) -> None:
    if callbacks.should_cancel():
        raise EvaluationCancelled("evaluation was cancelled")


def _prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create output directory {output_dir}: {exc}") from exc
    if not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    return output_dir


def _load_existing_metrics(dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    missing = [
        path.name
        for path in (output_dir / "episode_metrics.csv", output_dir / "metrics_core.json")
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            "cannot resume after METRICS; missing required artifacts in "
            f"{output_dir}: {', '.join(missing)}"
        )
    try:
        session = load_session(dataset_path)
        return load_metrics_core(output_dir, session)
    except Exception as exc:
        raise ValueError(f"cannot resume from persisted metrics in {output_dir}: {exc}") from exc


def run_profile_vlm(
    dataset_path: Path,
    output_dir: Path,
    profile: Profile,
    callbacks: EvaluationCallbacks,
) -> Path:
    """Run the optional Task 7 service without importing GPU packages at module import time."""
    from Genie02_report.attempt_eval.run_episode_attempt_eval import (
        AttemptEvalConfig,
        run_attempt_evaluation,
    )

    sampling = profile.vlm.sampling
    config = AttemptEvalConfig(
        dataset_root=dataset_path,
        model_path=Path(profile.vlm.model_path),
        image_key=profile.image_key,
        output_dir=output_dir,
        max_image_size=profile.vlm.max_image_size,
        max_global_frames=sampling.max_global_frames,
        global_sample_interval=sampling.global_sample_interval,
        max_dense_frames=sampling.max_dense_frames,
        dense_sample_interval=sampling.dense_sample_interval,
        dense_region=sampling.dense_region,
        review_mode=profile.review.mode,
        confidence_threshold=profile.review.confidence_threshold,
        min_episode_duration=profile.review.min_episode_duration,
        min_sampled_frames=profile.review.min_sampled_frames,
        max_new_tokens=profile.vlm.max_new_tokens,
    )

    last_progress = 30.0

    def progress(done: int, total: int, _stage: str) -> None:
        nonlocal last_progress
        if isinstance(done, bool) or isinstance(total, bool) or total <= 0:
            return
        fraction = min(max(done / total, 0.0), 1.0)
        value = max(last_progress, 30.0 + 60.0 * fraction)
        last_progress = value
        callbacks.on_progress(value)

    run_attempt_evaluation(
        config,
        progress=progress,
        should_cancel=callbacks.should_cancel,
    )
    summary_path = output_dir / "attempt_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"VLM evaluation did not create required artifact: {summary_path}")
    return summary_path


def run_evaluation(
    dataset_path: str | Path,
    output_dir: str | Path,
    profile: Profile,
    vlm_enabled: bool,
    callbacks: EvaluationCallbacks,
    resume_from: str = "METRICS",
) -> EvaluationResult:
    """Run METRICS, optional VLM, then REPORT with resumable stage boundaries."""
    if resume_from not in {"METRICS", "VLM", "REPORT"}:
        raise ValueError("resume_from must be one of METRICS, VLM, or REPORT")
    if not isinstance(vlm_enabled, bool):
        raise TypeError("vlm_enabled must be a boolean")

    dataset = Path(dataset_path)
    output = _prepare_output_dir(Path(output_dir))
    _check_cancelled(callbacks)

    metrics_end = 30.0 if vlm_enabled else 80.0
    if resume_from == "METRICS":
        callbacks.on_stage("METRICS")
        callbacks.on_progress(0.0)
        generate_episode_metrics(dataset, output)
        metrics = generate_metrics_core(dataset, output)
        callbacks.on_progress(metrics_end)
    else:
        metrics = _load_existing_metrics(dataset, output)
        callbacks.on_progress(metrics_end)

    _check_cancelled(callbacks)
    vlm_path: Path | None = None
    if vlm_enabled and resume_from in {"METRICS", "VLM"}:
        callbacks.on_stage("VLM")
        callbacks.on_progress(30.0)
        vlm_path = run_profile_vlm(dataset, output / "attempt_eval", profile, callbacks)
        callbacks.on_progress(90.0)
    elif vlm_enabled:
        vlm_path = output / "attempt_eval" / "attempt_summary.json"
        if not vlm_path.is_file():
            raise ValueError(
                f"cannot resume REPORT with VLM enabled; missing required artifact: {vlm_path}"
            )

    _check_cancelled(callbacks)
    callbacks.on_stage("REPORT")
    callbacks.on_progress(90.0 if vlm_enabled else 80.0)
    report_path = generate_markdown_report(dataset, output)
    callbacks.on_progress(100.0)
    return EvaluationResult(metrics, report_path, vlm_path)
