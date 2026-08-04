from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "episode_index",
    "video_file",
    "from_timestamp",
    "to_timestamp",
    "length",
    "metadata_episode_success",
    "episode_success",
    "pre_success_failed_attempt_count",
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
]


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    episode_dir = output_dir / "episode_results"
    frame_dir = output_dir / "sampled_frames"
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir, frame_dir


def save_episode_result(output_dir: Path, result: dict[str, Any]) -> Path:
    episode_dir = output_dir / "episode_results"
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / f"episode_{int(result['episode_index']):03d}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attempt_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "attempt_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in CSV_COLUMNS}
            row["auto_warning"] = json.dumps(row.get("auto_warning") or [], ensure_ascii=False)
            writer.writerow(row)
