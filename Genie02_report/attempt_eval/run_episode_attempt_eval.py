from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

if __package__:
    from .dataset_reader import EpisodeMeta, read_episode_metadata
    from .frame_sampler import sample_episode_frames
    from .result_writer import ensure_output_dirs, save_episode_result, write_summary
    from .review_policy import ReviewConfig, apply_review_policy
    from .vlm_client import LocalVLMClient, fallback_result
else:
    from dataset_reader import EpisodeMeta, read_episode_metadata
    from frame_sampler import sample_episode_frames
    from result_writer import ensure_output_dirs, save_episode_result, write_summary
    from review_policy import ReviewConfig, apply_review_policy
    from vlm_client import LocalVLMClient, fallback_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp attempt counts with a local VLM.")
    parser.add_argument("--dataset_root", required=True, type=Path, help="LeRobot dataset root directory.")
    parser.add_argument("--model_path", required=True, type=Path, help="Local VLM model path.")
    parser.add_argument("--image_key", default="observation.images.right_wrist", help="Video image key.")
    parser.add_argument("--output_dir", default=Path("outputs/attempt_eval"), type=Path, help="Output directory.")
    parser.add_argument("--max_frames", default=None, type=int, help="Deprecated alias for --max_global_frames.")
    parser.add_argument("--sample_interval", default=None, type=float, help="Deprecated alias for --global_sample_interval.")
    parser.add_argument("--max_image_size", default=336, type=int, help="Max image side length.")
    parser.add_argument("--max_global_frames", default=8, type=int, help="Max global frames per episode.")
    parser.add_argument("--global_sample_interval", default=2.0, type=float, help="Global frame sampling interval.")
    parser.add_argument("--max_dense_frames", default=8, type=int, help="Max dense frames per episode.")
    parser.add_argument("--dense_sample_interval", default=0.5, type=float, help="Dense frame sampling interval.")
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
    parser.add_argument("--confidence_threshold", default=0.7, type=float, help="Auto-review confidence threshold.")
    parser.add_argument("--min_episode_duration", default=3.0, type=float, help="Auto-review short duration threshold.")
    parser.add_argument("--min_sampled_frames", default=3, type=int, help="Auto-review minimum sampled frames.")
    parser.add_argument("--max_new_tokens", default=256, type=int, help="Max VLM output tokens.")
    parser.add_argument("--limit", default=None, type=int, help="Process only first N episodes.")
    parser.add_argument("--dry_run", action="store_true", help="Read mapping and sample frames without VLM inference.")
    return parser.parse_args()


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
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    episode_dir, frame_root = ensure_output_dirs(output_dir)
    review_config = ReviewConfig(
        mode=args.review_mode,
        confidence_threshold=args.confidence_threshold,
        min_episode_duration=args.min_episode_duration,
        min_sampled_frames=args.min_sampled_frames,
    )

    episodes = read_episode_metadata(args.dataset_root, args.image_key)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    print(f"episodes to process: {len(episodes)}")

    vlm = None if args.dry_run else LocalVLMClient(args.model_path, max_new_tokens=args.max_new_tokens)
    results: list[dict[str, Any]] = []

    for n, episode in enumerate(episodes, start=1):
        print(f"[{n}/{len(episodes)}] episode {episode.episode_index}: sampling frames")
        duration = max(0.0, episode.to_timestamp - episode.from_timestamp)
        sampled_frames: list[Path] = []
        frame_timestamps: list[dict[str, Any]] = []
        vlm_json_valid = True
        try:
            sampled_frames, frame_timestamps = sample_episode_frames(
                episode.video_file,
                frame_root / f"episode_{episode.episode_index:03d}",
                episode.from_timestamp,
                episode.to_timestamp,
                sample_interval=args.sample_interval,
                max_frames=args.max_frames,
                max_image_size=args.max_image_size,
                max_global_frames=args.max_global_frames,
                global_sample_interval=args.global_sample_interval,
                max_dense_frames=args.max_dense_frames,
                dense_sample_interval=args.dense_sample_interval,
                dense_region=args.dense_region,
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
            elif args.dry_run:
                vlm_result = fallback_result("", "dry_run")
                vlm_json_valid = False
            else:
                print(f"[{n}/{len(episodes)}] episode {episode.episode_index}: VLM inference")
                vlm_result, vlm_json_valid = vlm.analyze(sampled_frames, frame_timestamps, duration)
            result = merge_result(episode, vlm_result)
        except Exception as exc:
            result = merge_result(episode, fallback_result(str(exc), "episode_error"))
            vlm_json_valid = False
            print(f"[{n}/{len(episodes)}] episode {episode.episode_index}: error: {exc}")

        result["sampled_frame_count"] = len(sampled_frames)
        result["frame_timestamps"] = frame_timestamps
        result["global_frame_count"] = sum(1 for item in frame_timestamps if item.get("frame_type") == "global")
        result["dense_frame_count"] = sum(1 for item in frame_timestamps if item.get("frame_type") == "dense")
        result = apply_review_policy(result, review_config, duration, len(sampled_frames), vlm_json_valid)
        save_episode_result(output_dir, result)
        results.append(result)

    write_summary(output_dir, results)
    print(f"wrote: {episode_dir}")
    print(f"wrote: {output_dir / 'attempt_summary.csv'}")
    print(f"wrote: {output_dir / 'attempt_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
