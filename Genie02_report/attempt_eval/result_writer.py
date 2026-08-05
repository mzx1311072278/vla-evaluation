from __future__ import annotations

import csv
import io
import json
import os
import tempfile
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

_FIXED_ARTIFACTS = (
    "episode_results",
    "sampled_frames",
    "attempt_summary.json",
    "attempt_summary.csv",
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"output artifact path must not contain a symbolic link: {current}")


def _ensure_real_directory(path: Path) -> Path:
    intended = _absolute(path)
    _reject_symlink_components(intended)
    intended.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(intended)
    if not intended.is_dir():
        raise ValueError(f"output path is not a directory: {intended}")
    if intended.resolve(strict=True) != intended:
        raise ValueError(f"output directory did not resolve to the intended path: {intended}")
    return intended


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"output artifact must not be a symbolic link: {path}")


def _stage_text(path: Path, text: str, *, newline: str | None = None) -> Path:
    _reject_symlink(path)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _replace_staged(temp_path: Path, path: Path) -> None:
    _reject_symlink(path)
    os.replace(temp_path, path)


def _atomic_write_text(path: Path, text: str, *, newline: str | None = None) -> None:
    temp_path = _stage_text(path, text, newline=newline)
    try:
        _replace_staged(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _summary_csv(results: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for result in results:
        row = {key: result.get(key) for key in CSV_COLUMNS}
        row["auto_warning"] = json.dumps(row.get("auto_warning") or [], ensure_ascii=False)
        writer.writerow(row)
    return buffer.getvalue()


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir = _ensure_real_directory(output_dir)
    for child in output_dir.rglob("*"):
        _reject_symlink(child)
    for name in _FIXED_ARTIFACTS:
        _reject_symlink(output_dir / name)
    episode_dir = output_dir / "episode_results"
    frame_dir = output_dir / "sampled_frames"
    episode_dir = _ensure_real_directory(episode_dir)
    frame_dir = _ensure_real_directory(frame_dir)
    return episode_dir, frame_dir


def save_episode_result(output_dir: Path, result: dict[str, Any]) -> Path:
    output_dir = _ensure_real_directory(output_dir)
    episode_dir = _ensure_real_directory(output_dir / "episode_results")
    path = episode_dir / f"episode_{int(result['episode_index']):03d}.json"
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    _atomic_write_text(path, serialized)
    return path


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir = _ensure_real_directory(output_dir)
    json_path = output_dir / "attempt_summary.json"
    csv_path = output_dir / "attempt_summary.csv"
    _reject_symlink(json_path)
    _reject_symlink(csv_path)

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    csv_text = _summary_csv(results)
    json_temp = csv_temp = None
    try:
        json_temp = _stage_text(json_path, json_text)
        csv_temp = _stage_text(csv_path, csv_text, newline="")
        _replace_staged(json_temp, json_path)
        json_temp = None
        _replace_staged(csv_temp, csv_path)
        csv_temp = None
    finally:
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
        if csv_temp is not None:
            csv_temp.unlink(missing_ok=True)
