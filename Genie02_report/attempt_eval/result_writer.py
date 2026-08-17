from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SummaryPersistenceError(RuntimeError):
    """Raised when a summary commit fails and cannot be fully rolled back."""

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


def _stage_backup(path: Path) -> Path | None:
    _reject_symlink(path)
    if not path.exists():
        return None
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.backup.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with path.open("rb") as source, os.fdopen(fd, "wb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
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
    targets = (json_path, csv_path)
    staged: dict[Path, Path | None] = {json_path: None, csv_path: None}
    backups: dict[Path, Path | None] = {}
    try:
        staged[json_path] = _stage_text(json_path, json_text)
        staged[csv_path] = _stage_text(csv_path, csv_text, newline="")
        for path in targets:
            backups[path] = _stage_backup(path)

        try:
            for path in targets:
                temp_path = staged[path]
                if temp_path is None:
                    raise RuntimeError(f"summary artifact was not staged: {path}")
                _replace_staged(temp_path, path)
                staged[path] = None
        except BaseException as commit_error:
            rollback_errors: list[tuple[Path, BaseException]] = []
            for path in targets:
                backup_path = backups[path]
                try:
                    if backup_path is None:
                        path.unlink(missing_ok=True)
                    else:
                        _replace_staged(backup_path, path)
                        backups[path] = None
                except BaseException as rollback_error:
                    logger.exception("summary rollback failed for %s", path)
                    rollback_errors.append((path, rollback_error))
            if rollback_errors:
                details = "; ".join(
                    f"{path.name}: {type(error).__name__}: {error}"
                    for path, error in rollback_errors
                )
                raise SummaryPersistenceError(
                    f"summary commit failed and rollback was incomplete: {details}"
                ) from commit_error
            raise
    finally:
        for temp_path in (*staged.values(), *backups.values()):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
