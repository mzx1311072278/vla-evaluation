from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    if __package__:
        from .dataset_reader import EpisodeMeta
        from .vlm_client import LocalVLMClient
    else:
        from dataset_reader import EpisodeMeta
        from vlm_client import LocalVLMClient


PROMPT_VERSION = "genie02-attempt-v1"
SUPPORTED_PROMPT_VERSIONS = frozenset({PROMPT_VERSION})


@dataclass(frozen=True)
class AttemptEvalConfig:
    dataset_root: Path
    model_path: Path
    image_key: str = "observation.images.right_wrist"
    output_dir: Path = Path("outputs/attempt_eval")
    max_image_size: int = 336
    max_global_frames: int = 8
    global_sample_interval: float = 2.0
    max_dense_frames: int = 8
    dense_sample_interval: float = 0.5
    dense_region: str = "full"
    review_mode: str = "manual_review"
    confidence_threshold: float = 0.7
    min_episode_duration: float = 3.0
    min_sampled_frames: int = 3
    max_new_tokens: int = 256
    limit: int | None = None
    dry_run: bool = False
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_root", "model_path", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        if not isinstance(self.image_key, str) or not self.image_key.strip():
            raise ValueError("image_key must be a non-empty string")
        if self.prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_PROMPT_VERSIONS))
            raise ValueError(f"prompt_version must be one of: {supported}")
        if self.dense_region not in {"full", "last_half", "last_third"}:
            raise ValueError("dense_region must be one of: full, last_half, last_third")
        if self.review_mode not in {"manual_review", "auto_review"}:
            raise ValueError("review_mode must be one of: manual_review, auto_review")

        for name in (
            "max_image_size",
            "max_global_frames",
            "max_dense_frames",
            "min_sampled_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name == "max_image_size" else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int):
                raise TypeError("limit must be an integer or null")
            if self.limit < 0:
                raise ValueError("limit must be nonnegative")
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be a boolean")

        for name in (
            "global_sample_interval",
            "dense_sample_interval",
            "confidence_threshold",
            "min_episode_duration",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.global_sample_interval <= 0:
            raise ValueError("global_sample_interval must be greater than zero")
        if self.dense_sample_interval <= 0:
            raise ValueError("dense_sample_interval must be greater than zero")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.min_episode_duration < 0:
            raise ValueError("min_episode_duration must be nonnegative")


def run_attempt_evaluation(
    config: AttemptEvalConfig,
    *,
    episodes: list[EpisodeMeta] | None = None,
    client_factory: Callable[..., LocalVLMClient] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    if episodes is None:
        episodes = _read_episode_metadata(config.dataset_root, config.image_key)
    if config.limit is not None:
        episodes = episodes[: config.limit]

    if __package__:
        from .result_writer import (
            ensure_output_dirs,
            save_episode_result,
            write_summary,
        )
        from .review_policy import ReviewConfig, apply_review_policy
    else:
        from result_writer import ensure_output_dirs, save_episode_result, write_summary
        from review_policy import ReviewConfig, apply_review_policy

    output_dir = config.output_dir.expanduser()
    _, frame_root = ensure_output_dirs(output_dir)
    review_config = ReviewConfig(
        mode=config.review_mode,
        confidence_threshold=config.confidence_threshold,
        min_episode_duration=config.min_episode_duration,
        min_sampled_frames=config.min_sampled_frames,
    )
    total = len(episodes)
    if progress is not None:
        progress(0, total, "initial")

    vlm = None
    if episodes and not config.dry_run:
        _raise_if_cancelled(should_cancel)
        factory = client_factory or _create_local_vlm_client
        vlm = factory(
            config.model_path,
            max_new_tokens=config.max_new_tokens,
            prompt_version=config.prompt_version,
        )

    results: list[dict[str, Any]] = []
    for done, episode in enumerate(episodes, start=1):
        _raise_if_cancelled(should_cancel)
        duration = max(0.0, episode.to_timestamp - episode.from_timestamp)
        sampled_frames: list[Path] = []
        frame_timestamps: list[dict[str, Any]] = []
        vlm_json_valid = True
        try:
            sampled_frames, frame_timestamps = _sample_episode_frames(
                episode.video_file,
                frame_root / f"episode_{episode.episode_index:03d}",
                episode.from_timestamp,
                episode.to_timestamp,
                max_image_size=config.max_image_size,
                max_global_frames=config.max_global_frames,
                global_sample_interval=config.global_sample_interval,
                max_dense_frames=config.max_dense_frames,
                dense_sample_interval=config.dense_sample_interval,
                dense_region=config.dense_region,
            )
            if not sampled_frames:
                raise RuntimeError("No frames sampled from episode video segment")
            if episode.episode_success is False:
                vlm_result = {
                    "episode_success": False,
                    "pre_success_failed_attempt_count": None,
                    "failed_attempts_before_success": [],
                    "final_success_time": None,
                    "attempt_count": None,
                    "success_count": 0,
                    "failed_count": None,
                    "attempts": [],
                    "confidence": None,
                    "vlm_valid": True,
                    "reason": "metadata marks episode as failure; skipped attempt counting",
                    "parse_error": "",
                    "raw_response": "",
                    "auto_warning": ["skipped_failure_episode"],
                }
            elif config.dry_run:
                vlm_result = _fallback_result("", "dry_run")
                vlm_json_valid = False
            else:
                assert vlm is not None
                vlm_result, vlm_json_valid = vlm.analyze(sampled_frames, frame_timestamps, duration)
            result = merge_result(episode, vlm_result)
        except Exception as exc:
            if _is_evaluation_cancelled(exc):
                raise
            result = merge_result(episode, _fallback_result(str(exc), "episode_error"))
            vlm_json_valid = False

        result["sampled_frame_count"] = len(sampled_frames)
        result["frame_timestamps"] = frame_timestamps
        result["global_frame_count"] = sum(
            1 for item in frame_timestamps if item.get("frame_type") == "global"
        )
        result["dense_frame_count"] = sum(
            1 for item in frame_timestamps if item.get("frame_type") == "dense"
        )
        result = apply_review_policy(
            result,
            review_config,
            duration,
            len(sampled_frames),
            vlm_json_valid,
        )
        save_episode_result(output_dir, result)
        results.append(result)
        if progress is not None:
            progress(done, total, "episode_complete")

    write_summary(output_dir, results)
    if progress is not None:
        progress(total, total, "complete")
    return results


def _read_episode_metadata(dataset_root: Path, image_key: str) -> list[EpisodeMeta]:
    if __package__:
        from .dataset_reader import read_episode_metadata
    else:
        from dataset_reader import read_episode_metadata

    return read_episode_metadata(dataset_root, image_key)


def _sample_episode_frames(*args: Any, **kwargs: Any) -> tuple[list[Path], list[dict[str, Any]]]:
    if __package__:
        from .frame_sampler import sample_episode_frames
    else:
        from frame_sampler import sample_episode_frames

    return sample_episode_frames(*args, **kwargs)


def _create_local_vlm_client(*args: Any, **kwargs: Any) -> LocalVLMClient:
    if __package__:
        from .vlm_client import LocalVLMClient
    else:
        from vlm_client import LocalVLMClient

    return LocalVLMClient(*args, **kwargs)


def _fallback_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if __package__:
        from .vlm_client import fallback_result
    else:
        from vlm_client import fallback_result

    return fallback_result(*args, **kwargs)


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        from vla_eval.evaluation import EvaluationCancelled

        raise EvaluationCancelled("evaluation was cancelled")


def _is_evaluation_cancelled(exc: Exception) -> bool:
    from vla_eval.evaluation import EvaluationCancelled

    return isinstance(exc, EvaluationCancelled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp attempt counts with a local VLM.")
    parser.add_argument(
        "--dataset_root", required=True, type=Path, help="LeRobot dataset root directory."
    )
    parser.add_argument("--model_path", required=True, type=Path, help="Local VLM model path.")
    parser.add_argument(
        "--image_key", default="observation.images.right_wrist", help="Video image key."
    )
    parser.add_argument(
        "--output_dir", default=Path("outputs/attempt_eval"), type=Path, help="Output directory."
    )
    parser.add_argument(
        "--max_frames", default=None, type=int, help="Deprecated alias for --max_global_frames."
    )
    parser.add_argument(
        "--sample_interval",
        default=None,
        type=float,
        help="Deprecated alias for --global_sample_interval.",
    )
    parser.add_argument("--max_image_size", default=336, type=int, help="Max image side length.")
    parser.add_argument(
        "--max_global_frames", default=8, type=int, help="Max global frames per episode."
    )
    parser.add_argument(
        "--global_sample_interval", default=2.0, type=float, help="Global frame sampling interval."
    )
    parser.add_argument(
        "--max_dense_frames", default=8, type=int, help="Max dense frames per episode."
    )
    parser.add_argument(
        "--dense_sample_interval", default=0.5, type=float, help="Dense frame sampling interval."
    )
    parser.add_argument(
        "--dense_region",
        choices=["full", "last_half", "last_third"],
        default="full",
        help="Episode region used for dense sampling.",
    )
    parser.add_argument(
        "--review_mode",
        choices=["manual_review", "auto_review"],
        default="manual_review",
        help="Manual leaves needs_manual_review null; auto sets it from policy.",
    )
    parser.add_argument(
        "--confidence_threshold", default=0.7, type=float, help="Auto-review confidence threshold."
    )
    parser.add_argument(
        "--min_episode_duration",
        default=3.0,
        type=float,
        help="Auto-review short duration threshold.",
    )
    parser.add_argument(
        "--min_sampled_frames", default=3, type=int, help="Auto-review minimum sampled frames."
    )
    parser.add_argument("--max_new_tokens", default=256, type=int, help="Max VLM output tokens.")
    parser.add_argument("--limit", default=None, type=int, help="Process only first N episodes.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Read mapping and sample frames without VLM inference.",
    )
    parser.add_argument(
        "--prompt_version",
        choices=sorted(SUPPORTED_PROMPT_VERSIONS),
        default=PROMPT_VERSION,
        help="Versioned VLM prompt contract.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AttemptEvalConfig:
    return AttemptEvalConfig(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        image_key=args.image_key,
        output_dir=args.output_dir,
        max_image_size=args.max_image_size,
        max_global_frames=(
            args.max_frames if args.max_frames is not None else args.max_global_frames
        ),
        global_sample_interval=(
            args.sample_interval
            if args.sample_interval is not None
            else args.global_sample_interval
        ),
        max_dense_frames=args.max_dense_frames,
        dense_sample_interval=args.dense_sample_interval,
        dense_region=args.dense_region,
        review_mode=args.review_mode,
        confidence_threshold=args.confidence_threshold,
        min_episode_duration=args.min_episode_duration,
        min_sampled_frames=args.min_sampled_frames,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        dry_run=args.dry_run,
        prompt_version=args.prompt_version,
    )


def base_result(episode: EpisodeMeta) -> dict[str, Any]:
    return {
        "episode_index": episode.episode_index,
        "video_file": episode.video_file_rel,
        "from_timestamp": episode.from_timestamp,
        "to_timestamp": episode.to_timestamp,
        "length": episode.length,
        "metadata_episode_success": episode.episode_success,
        "episode_success": episode.episode_success,
        "pre_success_failed_attempt_count": None,
        "failed_attempts_before_success": [],
        "final_success_time": None,
        "attempt_count": None,
        "success_count": None,
        "failed_count": None,
        "attempts": [],
        "confidence": None,
        "vlm_valid": False,
        "parse_error": "",
        "raw_response": "",
        "frame_timestamps": [],
        "global_frame_count": 0,
        "dense_frame_count": 0,
        "needs_manual_review": None,
        "review_note": "",
        "auto_warning": [],
        "review_mode": "manual_review",
        "reason": "",
    }


def merge_result(episode: EpisodeMeta, vlm_result: dict[str, Any]) -> dict[str, Any]:
    result = base_result(episode)
    result.update(vlm_result)
    result["episode_index"] = episode.episode_index
    result["video_file"] = episode.video_file_rel
    result["from_timestamp"] = episode.from_timestamp
    result["to_timestamp"] = episode.to_timestamp
    result["length"] = episode.length
    result["metadata_episode_success"] = episode.episode_success
    if result.get("episode_success") is None:
        result["episode_success"] = episode.episode_success
    return result


def main() -> int:
    config = _config_from_args(parse_args())
    run_attempt_evaluation(config)
    print(f"wrote: {config.output_dir.expanduser() / 'attempt_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
