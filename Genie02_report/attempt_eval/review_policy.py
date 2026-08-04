from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReviewConfig:
    mode: str = "manual_review"
    confidence_threshold: float = 0.7
    min_episode_duration: float = 3.0
    min_sampled_frames: int = 3


def _num(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) else None


def apply_review_policy(
    result: dict[str, Any],
    config: ReviewConfig,
    episode_duration: float,
    sampled_frame_count: int,
    vlm_json_valid: bool,
) -> dict[str, Any]:
    warnings = list(result.get("auto_warning") or [])

    confidence = _num(result.get("confidence"))
    reason = result.get("reason") if isinstance(result.get("reason"), str) else ""
    if confidence is not None and confidence < config.confidence_threshold:
        warnings.append("low_confidence")
    if confidence is not None and confidence <= 0.0 and not reason.strip():
        warnings.append("empty_reason_with_zero_confidence")
        warnings.append("vlm_invalid_response")
        result["vlm_valid"] = False
    if episode_duration < config.min_episode_duration:
        warnings.append("short_episode")
    if sampled_frame_count < config.min_sampled_frames:
        warnings.append("too_few_frames")
    if sampled_frame_count < 10 and episode_duration > 60:
        warnings.append("sparse_sampling")
    if result.get("dense_frame_count", 0) < 5 and episode_duration > 10:
        warnings.append("too_few_dense_frames")
    if not vlm_json_valid:
        warnings.append("vlm_invalid_response")

    if result.get("episode_success") is not False:
        attempt = result.get("attempt_count")
        success = result.get("success_count")
        failed = result.get("failed_count")
        if not all(isinstance(v, int) for v in [attempt, success, failed]):
            warnings.append("missing_counts")
        elif success + failed != attempt:
            warnings.append("inconsistent_counts")

    text = f"{result.get('reason', '')} {result.get('raw_response', '')}".lower()
    if "遮挡" in text or "occlusion" in text or "blocked" in text:
        warnings.append("severe_occlusion")

    result["auto_warning"] = sorted(set(warnings))
    result["review_note"] = result.get("review_note", "")
    result["review_mode"] = config.mode
    result["needs_manual_review"] = bool(result["auto_warning"]) if config.mode == "auto_review" else None
    return result
