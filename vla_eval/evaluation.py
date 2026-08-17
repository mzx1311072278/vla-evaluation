"""Synchronous orchestration for deterministic and optional VLM evaluation stages."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from Genie02_report.genie02_episode_metrics import generate_episode_metrics
from Genie02_report.genie02_eval_common import (
    load_episode_metrics,
    load_episodes,
    load_metrics_core,
    load_session,
)
from Genie02_report.genie02_markdown_report import generate_markdown_report
from Genie02_report.genie02_metrics_core import build_core_metrics, generate_metrics_core

from .exceptions import EvaluationCancelled
from .profiles import Profile, VLMApiProfile

_ATTEMPT_REQUIRED_FIELDS = frozenset(
    {
        "episode_index",
        "metadata_episode_success",
        "episode_success",
        "pre_success_failed_attempt_count",
        "failed_attempts_before_success",
        "attempt_count",
        "success_count",
        "failed_count",
        "confidence",
        "vlm_valid",
        "parse_error",
        "needs_manual_review",
        "review_note",
        "auto_warning",
        "review_mode",
        "reason",
    }
)


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


@dataclass(frozen=True)
class _PersistedMetricsState:
    metrics: dict[str, Any]
    episode_indices: frozenset[int]


class _ProgressEmitter:
    def __init__(self, callback: Callable[[float], None], initial_progress: float) -> None:
        self._callback = callback
        self._last = initial_progress

    def emit(self, value: float) -> None:
        self._last = max(self._last, value)
        self._callback(self._last)


def _validate_initial_progress(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("initial_progress must be a number from 0 to 100")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError("initial_progress must be finite and between 0 and 100")
    return result


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


def _check_output_components(output_dir: Path, relative: str) -> Path:
    current = output_dir
    parts = Path(relative).parts
    for index, part in enumerate(parts):
        current /= part
        if current.is_symlink():
            raise ValueError(f"output artifact path must not contain a symbolic link: {current}")
        if current.exists() and index < len(parts) - 1 and not current.is_dir():
            raise ValueError(f"output artifact parent is not a directory: {current}")
    return current


def _preflight_output_contract(output_dir: Path, profile: Profile) -> None:
    for pattern in (*profile.outputs.required, *profile.outputs.optional):
        if "*" in pattern:
            for path in output_dir.glob(pattern):
                _check_output_components(output_dir, path.relative_to(output_dir).as_posix())
        else:
            _check_output_components(output_dir, pattern)


def _validate_regular_artifact(output_dir: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(output_dir.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{label} must remain within output directory {output_dir}: {path}"
        ) from exc
    return resolved


def _artifact_matches(output_dir: Path, pattern: str) -> list[Path]:
    if "*" in pattern:
        return list(output_dir.glob(pattern))
    path = output_dir / pattern
    return [path] if path.exists() or path.is_symlink() else []


def _validate_output_contract(output_dir: Path, profile: Profile, report_path: Path) -> None:
    resolved_report = _validate_regular_artifact(output_dir, report_path, "returned report")
    relative_report = resolved_report.relative_to(output_dir.resolve(strict=True)).as_posix()
    report_patterns = tuple(
        pattern
        for pattern in (*profile.outputs.required, *profile.outputs.optional)
        if Path(pattern).name.startswith("report_") and pattern.endswith(".md")
    )
    if not any(fnmatchcase(relative_report, pattern) for pattern in report_patterns):
        raise ValueError(
            f"returned report does not match a profile report output pattern: {relative_report}"
        )

    for pattern in profile.outputs.required:
        matches = _artifact_matches(output_dir, pattern)
        if not matches:
            raise ValueError(f"required output is missing: {pattern}")
        for path in matches:
            _validate_regular_artifact(output_dir, path, f"required output {pattern}")
    for pattern in profile.outputs.optional:
        for path in _artifact_matches(output_dir, pattern):
            _validate_regular_artifact(output_dir, path, f"optional output {pattern}")


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _compare_core_metrics(rebuilt: Any, persisted: Any, field: str = "metrics_core") -> None:
    if isinstance(rebuilt, Mapping):
        if not isinstance(persisted, Mapping):
            raise TypeError(f"{field} must be an object")
        if set(rebuilt) != set(persisted):
            missing = sorted(set(rebuilt) - set(persisted))
            unexpected = sorted(set(persisted) - set(rebuilt))
            raise ValueError(f"{field} fields differ: missing={missing}, unexpected={unexpected}")
        for key, value in rebuilt.items():
            _compare_core_metrics(value, persisted[key], f"{field}.{key}")
        return

    if isinstance(rebuilt, bool) or isinstance(persisted, bool):
        if type(rebuilt) is not type(persisted) or rebuilt != persisted:
            raise ValueError(f"{field} does not match rebuilt metrics")
        return
    if isinstance(rebuilt, int):
        if not isinstance(persisted, int) or persisted != rebuilt:
            raise ValueError(f"{field} does not match rebuilt metrics")
        return
    if isinstance(rebuilt, float):
        if (
            not isinstance(persisted, (int, float))
            or not math.isfinite(rebuilt)
            or not math.isfinite(persisted)
            or not math.isclose(rebuilt, persisted, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"{field} does not match rebuilt metrics")
        return
    if type(rebuilt) is not type(persisted) or rebuilt != persisted:
        raise ValueError(f"{field} does not match rebuilt metrics")


def _load_persisted_metrics_state(dataset_path: Path, output_dir: Path) -> _PersistedMetricsState:
    missing = [
        path.name
        for path in (output_dir / "episode_metrics.csv", output_dir / "metrics_core.json")
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            "cannot load persisted metrics; missing required artifacts in "
            f"{output_dir}: {', '.join(missing)}"
        )
    try:
        session = load_session(dataset_path)
        episodes = load_episodes(dataset_path, session)
        episode_rows = load_episode_metrics(output_dir, session)
        metrics = load_metrics_core(output_dir, session)
        rebuilt = build_core_metrics(session, episodes, episode_rows)
        _compare_core_metrics(rebuilt, metrics)
        return _PersistedMetricsState(
            metrics=metrics,
            episode_indices=frozenset(int(row["episode_index"]) for row in episodes),
        )
    except Exception as exc:
        raise ValueError(f"cannot load persisted metrics in {output_dir}: {exc}") from exc


def load_persisted_metrics(dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    """Load and cross-check persisted METRICS artifacts before a resumed stage."""
    return _load_persisted_metrics_state(dataset_path, output_dir).metrics


def _optional_bool(value: Any, field: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")


def _optional_count(value: Any, field: str) -> None:
    if value is not None:
        _nonnegative_int(value, field)


def load_attempt_summary(path: Path) -> list[dict[str, Any]]:
    """Load the current attempt_eval writer's JSON list and validate core result fields."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid attempt_summary.json at {path}: {exc}") from exc
    if not isinstance(loaded, list):
        raise TypeError(f"attempt_summary.json at {path} must contain a list")

    results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, value in enumerate(loaded):
        field = f"attempt_summary.json row {position}"
        if not isinstance(value, Mapping):
            raise TypeError(f"{field} must be an object")
        missing = _ATTEMPT_REQUIRED_FIELDS - set(value)
        if missing:
            raise ValueError(f"{field} is missing fields: {', '.join(sorted(missing))}")
        episode_index = _nonnegative_int(value["episode_index"], f"{field}.episode_index")
        if episode_index in seen:
            raise ValueError(f"{field}.episode_index is duplicated: {episode_index}")
        seen.add(episode_index)

        _optional_bool(value["metadata_episode_success"], f"{field}.metadata_episode_success")
        _optional_bool(value["episode_success"], f"{field}.episode_success")
        _optional_bool(value["needs_manual_review"], f"{field}.needs_manual_review")
        if not isinstance(value["vlm_valid"], bool):
            raise TypeError(f"{field}.vlm_valid must be a boolean")
        for name in (
            "pre_success_failed_attempt_count",
            "attempt_count",
            "success_count",
            "failed_count",
        ):
            _optional_count(value[name], f"{field}.{name}")
        attempts = value["failed_attempts_before_success"]
        if not isinstance(attempts, list):
            raise TypeError(f"{field}.failed_attempts_before_success must be a list")
        failed_before_success = value["pre_success_failed_attempt_count"]
        if failed_before_success is not None and failed_before_success != len(attempts):
            raise ValueError(f"{field}.pre_success_failed_attempt_count does not match attempts")

        confidence = value["confidence"]
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(f"{field}.confidence must be a finite number from 0 to 1 or null")
        warnings = value["auto_warning"]
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ValueError(f"{field}.auto_warning must be a list of strings")
        if value["review_mode"] not in {"manual_review", "auto_review"}:
            raise ValueError(f"{field}.review_mode is unsupported")
        for name in ("parse_error", "review_note", "reason"):
            if not isinstance(value[name], str):
                raise TypeError(f"{field}.{name} must be a string")
        results.append(dict(value))
    return results


def _validate_attempt_indices(results: list[dict[str, Any]], expected: frozenset[int]) -> None:
    actual = frozenset(row["episode_index"] for row in results)
    if actual != expected:
        raise ValueError(
            "attempt_summary.json episode indices do not match expected episodes: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _build_api_client_factory(api: VLMApiProfile) -> Callable[..., Any]:
    """Build a ``client_factory`` that constructs the API VLM client.

    The vendored runner calls ``factory(config.model_path, model_family=...,
    max_new_tokens=..., prompt_version=...)``. The API backend ignores that
    ``model_path`` (it takes its model name from the profile's ``api`` block) and
    reads its connection details from ``api``. Lazy-importing ``ApiVLMClient``
    keeps httpx out of module load, matching this module's existing
    "no GPU/network imports at module import time" contract. Returning a closure
    captures the api profile without modifying ``Genie02_report``.
    """
    from vla_eval.vlm_api import ApiVLMClient

    def factory(
        _model_path: Any,
        *,
        model_family: str,
        max_new_tokens: int,
        prompt_version: str,
    ) -> ApiVLMClient:
        return ApiVLMClient(
            base_url=api.base_url,
            model=api.model,
            api_key_env=api.api_key_env,
            max_new_tokens=max_new_tokens,
            prompt_version=prompt_version,
            timeout=api.timeout,
            max_retries=api.max_retries,
        )

    return factory


def run_profile_vlm(
    dataset_path: Path,
    output_dir: Path,
    profile: Profile,
    callbacks: EvaluationCallbacks,
    camera_keys: tuple[str, ...] | None = None,
) -> Path:
    """Run the optional Task 7 service without importing GPU packages at module import time."""
    from Genie02_report.attempt_eval.run_episode_attempt_eval import (
        AttemptEvalConfig,
        run_attempt_evaluation,
    )

    sampling = profile.vlm.sampling
    if profile.vlm.backend == "api":
        api = profile.vlm.api
        # AttemptEvalConfig requires model_path as a pathlib.Path (isinstance-checked
        # in the vendored library) but never reads it once client_factory is injected.
        # We pass the API model name as a placeholder so the vendored config passes
        # untouched; the API client takes its model from the api block below. This
        # avoids modifying Genie02_report to relax the required field.
        model_path = Path(api.model)
    else:
        model_path = Path(profile.vlm.model_path)
    config = AttemptEvalConfig(
        dataset_root=dataset_path,
        model_path=model_path,
        model_family=profile.vlm.model_family or "qwen2_5_vl",
        prompt_version=profile.vlm.prompt_version,
        image_key=profile.image_key,
        image_keys=tuple(camera_keys or (profile.image_key,)),
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

    if profile.vlm.backend == "api":
        run_attempt_evaluation(
            config,
            client_factory=_build_api_client_factory(profile.vlm.api),
            progress=progress,
            should_cancel=callbacks.should_cancel,
        )
    else:
        run_attempt_evaluation(
            config,
            progress=progress,
            should_cancel=callbacks.should_cancel,
        )
    summary_path = output_dir / "attempt_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"VLM evaluation did not create required artifact: {summary_path}")
    load_attempt_summary(summary_path)
    return summary_path


def run_evaluation(
    dataset_path: str | Path,
    output_dir: str | Path,
    profile: Profile,
    vlm_enabled: bool,
    callbacks: EvaluationCallbacks,
    resume_from: str = "METRICS",
    initial_progress: float = 0.0,
    camera_keys: tuple[str, ...] | None = None,
) -> EvaluationResult:
    """Run METRICS, optional VLM, then REPORT with resumable stage boundaries."""
    if resume_from not in {"METRICS", "VLM", "REPORT"}:
        raise ValueError("resume_from must be one of METRICS, VLM, or REPORT")
    if not isinstance(vlm_enabled, bool):
        raise TypeError("vlm_enabled must be a boolean")
    if resume_from == "VLM" and not vlm_enabled:
        raise ValueError("resume_from='VLM' requires vlm_enabled=True")
    if vlm_enabled and camera_keys is not None and not camera_keys:
        raise ValueError("camera_keys must contain at least one camera when VLM is enabled")
    resolved_camera_keys = tuple(camera_keys if camera_keys is not None else ())
    if vlm_enabled and camera_keys is None:
        resolved_camera_keys = (profile.image_key,)
    if len(resolved_camera_keys) > 3 or any(
        not isinstance(value, str) or not value for value in resolved_camera_keys
    ):
        raise ValueError("camera_keys must contain at most three non-empty strings")
    progress = _ProgressEmitter(callbacks.on_progress, _validate_initial_progress(initial_progress))
    stage_callbacks = EvaluationCallbacks(
        on_stage=callbacks.on_stage,
        on_progress=progress.emit,
        should_cancel=callbacks.should_cancel,
    )

    dataset = Path(dataset_path)
    output = _prepare_output_dir(Path(output_dir))
    _preflight_output_contract(output, profile)
    _check_cancelled(stage_callbacks)

    metrics_end = 30.0 if vlm_enabled else 80.0
    if resume_from == "METRICS":
        stage_callbacks.on_stage("METRICS")
        stage_callbacks.on_progress(0.0)
        episode_rows = generate_episode_metrics(dataset, output)
        expected_episode_indices = frozenset(row["episode_index"] for row in episode_rows)
        metrics = generate_metrics_core(dataset, output)
        stage_callbacks.on_progress(metrics_end)
    else:
        persisted = _load_persisted_metrics_state(dataset, output)
        metrics = persisted.metrics
        expected_episode_indices = persisted.episode_indices
        stage_callbacks.on_progress(metrics_end)

    _check_cancelled(stage_callbacks)
    vlm_path: Path | None = None
    if vlm_enabled and resume_from in {"METRICS", "VLM"}:
        stage_callbacks.on_stage("VLM")
        stage_callbacks.on_progress(30.0)
        if camera_keys is None:
            vlm_path = run_profile_vlm(
                dataset,
                output / "attempt_eval",
                profile,
                stage_callbacks,
            )
        else:
            vlm_path = run_profile_vlm(
                dataset,
                output / "attempt_eval",
                profile,
                stage_callbacks,
                camera_keys=resolved_camera_keys,
            )
        stage_callbacks.on_progress(90.0)
    elif vlm_enabled:
        vlm_path = output / "attempt_eval" / "attempt_summary.json"
        if not vlm_path.is_file():
            raise ValueError(
                f"cannot resume REPORT with VLM enabled; missing required artifact: {vlm_path}"
            )
        _validate_attempt_indices(load_attempt_summary(vlm_path), expected_episode_indices)

    _check_cancelled(stage_callbacks)
    if vlm_path is not None and resume_from in {"METRICS", "VLM"}:
        _validate_attempt_indices(load_attempt_summary(vlm_path), expected_episode_indices)
    stage_callbacks.on_stage("REPORT")
    stage_callbacks.on_progress(90.0 if vlm_enabled else 80.0)
    report_path = generate_markdown_report(dataset, output)
    _validate_output_contract(output, profile, report_path)
    stage_callbacks.on_progress(100.0)
    return EvaluationResult(metrics, report_path, vlm_path)
