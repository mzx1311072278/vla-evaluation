"""Dataset discovery, security preflight, and lightweight manifest fingerprints."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from Genie02_report.attempt_eval.dataset_reader import resolve_video_columns


class DatasetKind(StrEnum):
    LEROBOT = "lerobot"
    GENIE02_SESSION = "genie02_session"


@dataclass(frozen=True)
class DatasetInspection:
    kind: DatasetKind | None
    ready: bool
    fingerprint: str
    size_bytes: int
    episode_count: int | None
    errors: tuple[str, ...]


_SESSION_FIELDS = (
    "schema_version",
    "session_id",
    "created_at",
    "status",
    "rollout_config_path",
    "rollout_mode",
    "policy_path",
    "task",
    "num_episodes_target",
    "fps",
    "dataset_backend",
)
_EPISODE_FIELDS = (
    "session_id",
    "episode_index",
    "episode_path",
    "trajectory_path",
    "t_start",
    "t_end",
    "duration_s",
    "outcome",
    "operator_intervened",
    "notes",
)


@dataclass(frozen=True)
class _FileEntry:
    logical_path: Path
    resolved_path: Path


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int


class _DatasetFileError(RuntimeError):
    pass


class _Manifest:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.entries: dict[str, dict[str, Any]] = {}
        self.files: dict[Path, Path] = {}
        self.snapshots: dict[Path, _FileSnapshot] = {}
        self._sized_files: set[tuple[int, int]] = set()
        self.size_bytes = 0

    def add_file(self, logical_path: Path, resolved_path: Path) -> None:
        logical_path = Path(os.path.abspath(logical_path))
        relative = logical_path.relative_to(self.dataset_root).as_posix()
        if relative in self.entries:
            return
        file_stat = resolved_path.stat()
        record: dict[str, Any] = {
            "path": relative,
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        }
        self.entries[relative] = record
        self.files[logical_path] = resolved_path
        self.snapshots[logical_path] = _snapshot(file_stat)
        identity = (file_stat.st_dev, file_stat.st_ino)
        if identity not in self._sized_files:
            self._sized_files.add(identity)
            self.size_bytes += file_stat.st_size

    def mark_metadata(self, logical_path: Path) -> None:
        logical_path = Path(os.path.abspath(logical_path))
        relative = logical_path.relative_to(self.dataset_root).as_posix()
        if logical_path in self.files:
            self.entries[relative]["sha256"] = _hash_file(logical_path, self)

    def verify_unchanged(self) -> list[str]:
        errors: list[str] = []
        for logical_path, expected in sorted(self.snapshots.items()):
            try:
                path_stat = os.stat(logical_path, follow_symlinks=False)
                if stat.S_ISLNK(path_stat.st_mode):
                    resolved = logical_path.resolve(strict=True)
                    if not _is_relative_to(resolved, self.dataset_root):
                        raise _DatasetFileError(
                            f"symlink target moved outside dataset root: {logical_path} -> {resolved}"
                        )
                    path_stat = resolved.stat()
                if _snapshot(path_stat) != expected:
                    raise _DatasetFileError(
                        f"dataset file changed during inspection: {logical_path}"
                    )
            except (OSError, RuntimeError, _DatasetFileError) as exc:
                errors.append(str(exc))
        return errors

    def fingerprint(self) -> str:
        canonical = json.dumps(
            [self.entries[key] for key in sorted(self.entries)],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _snapshot(file_stat: os.stat_result) -> _FileSnapshot:
    return _FileSnapshot(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
    )


@contextmanager
def _stable_binary_file(logical_path: Path, manifest: _Manifest, label: str) -> Iterator[BinaryIO]:
    """Open one scanned regular file and verify it stays unchanged while parsed."""
    logical_path = Path(os.path.abspath(logical_path))
    expected = manifest.snapshots.get(logical_path)
    if expected is None:
        raise _DatasetFileError(f"{label} is not part of the dataset manifest: {logical_path}")
    before = os.stat(logical_path, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise _DatasetFileError(f"parsed symlink files are not supported: {logical_path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(logical_path, flags)
    except OSError as exc:
        raise _DatasetFileError(f"cannot safely open {label} {logical_path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise _DatasetFileError(f"{label} is not a regular file: {logical_path}")
        if _snapshot(opened) != expected:
            raise _DatasetFileError(f"dataset file changed during inspection: {logical_path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
            after_read = os.fstat(descriptor)
        after_path = os.stat(logical_path, follow_symlinks=False)
        if stat.S_ISLNK(after_path.st_mode) or _snapshot(after_read) != expected:
            raise _DatasetFileError(f"dataset file changed during inspection: {logical_path}")
        if _snapshot(after_path) != expected:
            raise _DatasetFileError(f"dataset file changed during inspection: {logical_path}")
    finally:
        os.close(descriptor)


def _hash_file(path: Path, manifest: _Manifest) -> str:
    digest = hashlib.sha256()
    with _stable_binary_file(path, manifest, "metadata") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_root(path: Path, label: str) -> tuple[Path | None, str | None]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError:
        return None, f"{label} does not exist: {path}"
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve {label} {path}: {exc}"
    if not resolved.is_dir():
        return None, f"{label} is not a directory: {resolved}"
    return resolved, None


def _resolve_dataset_root(path: Path, allowed_root: Path) -> tuple[Path | None, str | None]:
    logical = Path(os.path.abspath(path.expanduser()))
    if not _is_relative_to(logical, allowed_root):
        return None, f"dataset root is outside allowed root: {logical}"
    try:
        resolved = logical.resolve(strict=True)
    except FileNotFoundError:
        return None, f"dataset root does not exist: {logical}"
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve dataset root {logical}: {exc}"
    if not _is_relative_to(resolved, allowed_root):
        return None, f"dataset root is outside allowed root: {logical} -> {resolved}"
    if not resolved.is_dir():
        return None, f"dataset root is not a directory: {resolved}"
    return resolved, None


def _scan_dataset(root: Path, allowed_root: Path, manifest: _Manifest) -> list[str]:
    errors: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name, reverse=True)
        except OSError as exc:
            errors.append(f"cannot read directory {directory}: {exc}")
            continue
        for entry in entries:
            logical = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                errors.append(f"cannot inspect path {logical}: {exc}")
                continue
            if stat.S_ISLNK(mode):
                try:
                    resolved = logical.resolve(strict=True)
                except RuntimeError as exc:
                    errors.append(f"unsafe symlink loop at {logical}: {exc}")
                    continue
                except OSError as exc:
                    errors.append(f"cannot resolve symlink {logical}: {exc}")
                    continue
                if not _is_relative_to(resolved, allowed_root):
                    errors.append(f"path is outside allowed root: {logical} -> {resolved}")
                    continue
                if not resolved.is_file():
                    errors.append(f"symlink target is not a regular file: {logical} -> {resolved}")
                    continue
                try:
                    manifest.add_file(logical, resolved)
                except OSError as exc:
                    errors.append(f"cannot read file metadata {logical}: {exc}")
            elif stat.S_ISDIR(mode):
                pending.append(logical)
            elif stat.S_ISREG(mode):
                try:
                    resolved = logical.resolve(strict=True)
                    if not _is_relative_to(resolved, allowed_root):
                        errors.append(f"path is outside allowed root: {logical} -> {resolved}")
                        continue
                    manifest.add_file(logical, resolved)
                except OSError as exc:
                    errors.append(f"cannot read file metadata {logical}: {exc}")
            else:
                errors.append(f"special files are not supported: {logical}")
    return sorted(errors, key=lambda error: ("outside allowed root" not in error, error))


def _safe_reference(
    raw_path: str | Path,
    *,
    base: Path,
    allowed_root: Path,
    label: str,
) -> tuple[_FileEntry | None, str | None]:
    candidate = Path(raw_path).expanduser()
    logical = Path(os.path.abspath(candidate if candidate.is_absolute() else base / candidate))
    if not _is_relative_to(logical, allowed_root):
        return None, f"{label} is outside allowed root: {logical}"
    try:
        boundary_target = logical.resolve(strict=False)
        if not _is_relative_to(boundary_target, allowed_root):
            return None, f"{label} is outside allowed root: {logical} -> {boundary_target}"
        resolved = logical.resolve(strict=True)
    except FileNotFoundError:
        return None, f"{label} does not exist: {logical}"
    except RuntimeError as exc:
        return None, f"unsafe symlink loop in {label}: {logical}: {exc}"
    except OSError as exc:
        return None, f"cannot resolve {label} {logical}: {exc}"
    if not _is_relative_to(resolved, allowed_root):
        return None, f"{label} is outside allowed root: {logical} -> {resolved}"
    if not resolved.is_file():
        return None, f"{label} is not a regular file: {logical}"
    return _FileEntry(logical_path=logical, resolved_path=resolved), None


def _read_json(
    path: Path, label: str, errors: list[str], manifest: _Manifest
) -> dict[str, Any] | None:
    try:
        with _stable_binary_file(path, manifest, label) as handle:
            value = json.loads(handle.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {label}: {exc}")
        return None
    except (OSError, UnicodeError, _DatasetFileError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def _int_value(value: Any, label: str, errors: list[str], *, positive: bool = False) -> int | None:
    try:
        if isinstance(value, bool) or pd.isna(value):
            raise ValueError
        result = int(value)
        if float(value) != result or result < (1 if positive else 0):
            raise ValueError
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{label} must be {'a positive' if positive else 'a non-negative'} integer")
        return None
    return result


def _video_keys(info: dict[str, Any], columns: set[str]) -> set[str]:
    keys = (
        {
            str(key)
            for key, descriptor in info.get("features", {}).items()
            if isinstance(descriptor, dict) and descriptor.get("dtype") == "video"
        }
        if isinstance(info.get("features", {}), dict)
        else set()
    )
    suffix = "/file_index"
    for column in columns:
        if column.startswith("videos/") and column.endswith(suffix):
            keys.add(column[len("videos/") : -len(suffix)])
    return keys


def _find_video_file(
    root: Path, key: str, meta_chunk: str, file_index: int, manifest: _Manifest
) -> Path | None:
    video_root = root / "videos" / key
    candidates = (
        video_root / meta_chunk / f"file-{file_index:03d}.mp4",
        video_root / f"chunk-{file_index // 1000:03d}" / f"file-{file_index % 1000:03d}.mp4",
        video_root / f"chunk-{file_index // 1000:03d}" / f"file-{file_index:03d}.mp4",
    )
    for candidate in candidates:
        if candidate in manifest.files:
            return candidate
    matches = sorted(
        logical
        for logical in manifest.files
        if _is_relative_to(logical, video_root) and logical.name == f"file-{file_index:03d}.mp4"
    )
    return matches[0] if matches else None


def _validate_data_parquet(
    path: Path,
    references: dict[int, int],
    manifest: _Manifest,
    errors: list[str],
) -> None:
    frame_counts = {index: 0 for index in references}
    action_shape: tuple[int, ...] | None = None
    try:
        with _stable_binary_file(path, manifest, "referenced data parquet") as handle:
            parquet_file = pq.ParquetFile(handle)
            columns = set(parquet_file.schema_arrow.names)
            missing = sorted({"episode_index", "timestamp", "action"} - columns)
            if missing:
                errors.append(
                    f"referenced data parquet {path} is missing columns: {', '.join(missing)}"
                )
                return
            for batch in parquet_file.iter_batches(
                columns=["episode_index", "timestamp", "action"], batch_size=65_536
            ):
                values = batch.to_pydict()
                for index, timestamp, action in zip(
                    values["episode_index"],
                    values["timestamp"],
                    values["action"],
                    strict=True,
                ):
                    try:
                        episode_index = int(index)
                    except (TypeError, ValueError):
                        continue
                    if episode_index not in references:
                        continue
                    frame_counts[episode_index] += 1
                    if isinstance(timestamp, (bool, bytes, str)) or not isinstance(timestamp, Real):
                        errors.append(
                            f"episode {episode_index} must have a finite numeric timestamp in {path}"
                        )
                    elif not math.isfinite(float(timestamp)):
                        errors.append(
                            f"episode {episode_index} timestamp must be finite numeric timestamp in {path}"
                        )
                    raw_action = np.asarray(action)
                    if raw_action.dtype.kind not in "biufc":
                        errors.append(
                            f"episode {episode_index} action must be a numeric action vector in {path}"
                        )
                        continue
                    if raw_action.ndim != 1 or raw_action.size == 0:
                        errors.append(
                            f"episode {episode_index} must have a nonempty action vector in {path}"
                        )
                        continue
                    if action_shape is None:
                        action_shape = raw_action.shape
                    elif raw_action.shape != action_shape:
                        errors.append(f"inconsistent action vector shape in {path}")
    except Exception as exc:  # noqa: BLE001 - pyarrow exposes varied corrupt-file errors
        errors.append(f"cannot read referenced data parquet {path}: {exc}")
        return
    for episode_index, expected_length in references.items():
        frame_count = frame_counts[episode_index]
        if frame_count == 0:
            errors.append(f"episode {episode_index} is absent from referenced data parquet {path}")
        elif frame_count != expected_length:
            errors.append(
                f"episode {episode_index} length is {expected_length} in metadata, "
                f"but data parquet contains {frame_count} frames"
            )


def _inspect_lerobot(root: Path, manifest: _Manifest) -> tuple[int | None, list[str]]:
    errors: list[str] = []
    info_path = root / "meta/info.json"
    episodes_dir = root / "meta/episodes"
    data_dir = root / "data"
    if info_path not in manifest.files:
        errors.append("missing required LeRobot metadata: meta/info.json")
    if not episodes_dir.is_dir():
        errors.append("missing required LeRobot directory: meta/episodes")
    if not data_dir.is_dir():
        errors.append("missing required LeRobot directory: data")

    info = (
        _read_json(info_path, "meta/info.json", errors, manifest)
        if info_path in manifest.files
        else None
    )
    expected_count: int | None = None
    if info is not None:
        expected_count = _int_value(
            info.get("total_episodes"), "meta/info.json total_episodes", errors
        )
        try:
            fps = float(info.get("fps"))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("meta/info.json fps must be a positive finite number")

    episode_paths = sorted(
        logical
        for logical in manifest.files
        if _is_relative_to(logical, episodes_dir) and logical.suffix.lower() in {".parquet", ".pq"}
    )
    if not episode_paths:
        errors.append("no LeRobot episode parquet files found under meta/episodes")
        return expected_count, errors

    episode_indices: set[int] = set()
    data_references: dict[Path, dict[int, int]] = {}
    required_columns = {
        "episode_index",
        "length",
        "episode_success",
        "data/chunk_index",
        "data/file_index",
    }
    for logical_path in episode_paths:
        try:
            with _stable_binary_file(logical_path, manifest, "episode metadata") as handle:
                frame = pd.read_parquet(handle)
        except Exception as exc:  # noqa: BLE001 - pandas wraps backend-specific errors
            errors.append(f"cannot read episode metadata parquet {logical_path}: {exc}")
            continue
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            errors.append(
                f"episode metadata parquet {logical_path} is missing columns: {', '.join(missing)}"
            )
            continue
        if frame.empty:
            errors.append(f"episode metadata parquet is empty: {logical_path}")
            continue
        video_keys = _video_keys(info or {}, set(frame.columns))
        for row_number, row in frame.iterrows():
            prefix = f"{logical_path} row {row_number}"
            index = _int_value(row["episode_index"], f"{prefix} episode_index", errors)
            length = _int_value(row["length"], f"{prefix} length", errors, positive=True)
            chunk_index = _int_value(row["data/chunk_index"], f"{prefix} data/chunk_index", errors)
            file_index = _int_value(row["data/file_index"], f"{prefix} data/file_index", errors)
            if index is not None:
                if index in episode_indices:
                    errors.append(f"duplicate LeRobot episode_index: {index}")
                episode_indices.add(index)
            outcome = str(row["episode_success"]).strip().lower()
            if outcome not in {"success", "failure"}:
                errors.append(f"{prefix} episode_success must be success or failure")
            if index is None or length is None or chunk_index is None or file_index is None:
                continue
            data_path = (
                root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
            )
            if data_path not in manifest.files:
                errors.append(f"referenced data file does not exist: {data_path}")
            else:
                data_references.setdefault(data_path, {})[index] = length

            for key in sorted(video_keys):
                try:
                    video_columns = resolve_video_columns(list(frame.columns), key)
                except ValueError as exc:
                    errors.append(f"cannot resolve metadata for video stream {key}: {exc}")
                    continue
                video_index = _int_value(
                    row[video_columns.file_index], f"{prefix} {key} video file_index", errors
                )
                if video_index is None:
                    continue
                try:
                    start = float(row[video_columns.from_timestamp])
                    end = float(row[video_columns.to_timestamp])
                    if not math.isfinite(start) or not math.isfinite(end) or end < start:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"{prefix} video timestamps for {key} are invalid")
                video_path = _find_video_file(
                    root, key, logical_path.parent.name, video_index, manifest
                )
                if video_path is None:
                    errors.append(
                        f"referenced video file does not exist for stream {key}, file_index {video_index}"
                    )

    for data_path, references in sorted(data_references.items()):
        _validate_data_parquet(data_path, references, manifest, errors)

    actual_count = len(episode_indices)
    if expected_count is not None and actual_count != expected_count:
        errors.append(
            f"meta/info.json total_episodes is {expected_count}, but episode metadata contains {actual_count}"
        )
    return actual_count, errors


def _find_native_trajectory(directory: Path, index: int) -> Path | None:
    candidates = (
        directory / f"episode_{index:03d}.npz",
        directory / f"episode_{index:06d}.npz",
        directory / f"episode_{index:06d}" / "frames.npz",
        directory / "frames.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _numeric_matrix(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        try:
            array = np.stack([np.asarray(item, dtype=float) for item in value])
        except (TypeError, ValueError) as exc:
            raise _DatasetFileError(f"{label} is not a numeric matrix") from exc
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise _DatasetFileError(f"{label} must have shape [frames, dimensions]")
    return array


def _trajectory_action_names(
    logical_path: Path,
    value_key: str,
    manifest: _Manifest,
    errors: list[str],
) -> list[str]:
    if logical_path.suffix.lower() == ".npz":
        candidate = logical_path.parent / "meta.json"
        field = "action_names"
    elif logical_path.suffix.lower() in {".parquet", ".pq"} and len(logical_path.parents) >= 3:
        candidate = logical_path.parents[2] / "meta/info.json"
        field = value_key
    else:
        return []
    if candidate not in manifest.files:
        return []
    metadata = _read_json(candidate, "trajectory metadata", errors, manifest)
    if metadata is None:
        raise _DatasetFileError(f"cannot read trajectory metadata: {candidate}")
    names = (
        metadata.get(field, [])
        if field == "action_names"
        else metadata.get("features", {}).get(field, {}).get("names", [])
    )
    return [name for name in names if isinstance(name, str)] if isinstance(names, list) else []


def _select_ee_columns(values: np.ndarray, action_names: list[str]) -> np.ndarray:
    columns = [
        index
        for index, name in enumerate(action_names[: values.shape[1]])
        if "_ee." in name and name.rsplit(".", 1)[-1] in {"x", "y", "z"}
    ]
    if not columns:
        if values.shape[1] >= 14:
            columns = [0, 1, 2, 7, 8, 9]
        elif values.shape[1] >= 3:
            columns = [0, 1, 2]
        else:
            raise _DatasetFileError("EE trajectory has fewer than 3 xyz columns")
    return values[:, columns]


def _read_trajectory_arrays(
    logical_path: Path,
    episode_index: int,
    manifest: _Manifest,
) -> dict[str, Any]:
    value_keys = ("smooth_send_y", "sent_y", "action")
    time_keys = ("smooth_send_t", "sent_t", "timestamp")
    suffix = logical_path.suffix.lower()
    if suffix == ".npz":
        with (
            _stable_binary_file(logical_path, manifest, "trajectory") as handle,
            np.load(handle, allow_pickle=False) as archive,
        ):
            keys = set(archive.files)
            selected = [key for key in (*value_keys, *time_keys, "is_intervention") if key in keys]
            return {key: archive[key] for key in selected}
    if suffix in {".parquet", ".pq"}:
        with _stable_binary_file(logical_path, manifest, "trajectory") as handle:
            parquet_file = pq.ParquetFile(handle)
            columns = set(parquet_file.schema_arrow.names)
            selected = [
                key
                for key in (
                    "episode_index",
                    *value_keys,
                    *time_keys,
                    "is_intervention",
                    "complementary_info.is_intervention",
                )
                if key in columns
            ]
            frame = parquet_file.read(columns=selected).to_pandas()
        if "episode_index" in frame:
            frame = frame[frame["episode_index"] == episode_index]
        if frame.empty:
            raise _DatasetFileError(
                f"episode {episode_index} is absent from trajectory parquet {logical_path}"
            )
        return {key: frame[key].to_numpy() for key in selected if key != "episode_index"}
    raise _DatasetFileError(f"unsupported trajectory format: {logical_path}")


def _validate_trajectory_structure(
    logical_path: Path,
    episode_index: int,
    session: dict[str, Any],
    manifest: _Manifest,
    errors: list[str],
) -> None:
    data = _read_trajectory_arrays(logical_path, episode_index, manifest)
    value_time_pairs = (
        ("smooth_send_y", "smooth_send_t"),
        ("sent_y", "sent_t"),
        ("action", "timestamp"),
    )
    value_key, paired_time_key = next(
        ((value, time) for value, time in value_time_pairs if value in data),
        (None, None),
    )
    if value_key is None:
        raise _DatasetFileError(f"no supported trajectory array in {logical_path}")
    values = _numeric_matrix(data[value_key], f"{logical_path}:{value_key}")
    if session["rollout_mode"] == "ee":
        names = _trajectory_action_names(logical_path, value_key, manifest, errors)
        values = _select_ee_columns(values, names)
    if len(values) == 0:
        raise _DatasetFileError(f"trajectory is empty: {logical_path}")
    time_key = (
        paired_time_key
        if paired_time_key is not None and paired_time_key in data
        else next((key for key in ("smooth_send_t", "sent_t", "timestamp") if key in data), None)
    )
    times = np.asarray(data[time_key], dtype=float).reshape(-1) if time_key else None
    intervention_raw = data.get("is_intervention", data.get("complementary_info.is_intervention"))
    intervention = (
        np.asarray(intervention_raw, dtype=float).reshape(-1)
        if intervention_raw is not None
        else None
    )
    if times is not None and len(times) != len(values):
        raise _DatasetFileError("trajectory timestamp length mismatch")
    if intervention is not None and len(intervention) != len(values):
        raise _DatasetFileError("trajectory intervention-mask length mismatch")


def _validate_genie_trajectory(
    raw_path: str,
    *,
    session: dict[str, Any],
    root: Path,
    allowed_root: Path,
    manifest: _Manifest,
    episode_index: int,
    label: str,
    errors: list[str],
) -> bool:
    candidate = Path(raw_path).expanduser()
    logical = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    if not _is_relative_to(logical, manifest.dataset_root):
        errors.append(f"{label} is outside dataset root: {logical}")
        return False
    try:
        boundary_target = logical.resolve(strict=False)
        if not _is_relative_to(boundary_target, manifest.dataset_root):
            errors.append(f"{label} is outside dataset root: {logical} -> {boundary_target}")
            return False
        resolved_candidate = logical.resolve(strict=True)
    except FileNotFoundError:
        errors.append(f"{label} does not exist: {logical}")
        return False
    except (OSError, RuntimeError) as exc:
        errors.append(f"cannot resolve {label} {logical}: {exc}")
        return False
    if not _is_relative_to(resolved_candidate, manifest.dataset_root):
        errors.append(f"{label} is outside dataset root: {logical} -> {resolved_candidate}")
        return False
    entry, error = _safe_reference(
        logical, base=root, allowed_root=manifest.dataset_root, label=label
    )
    if error:
        errors.append(error)
        return False
    assert entry is not None
    if not _mark_trajectory_sidecar_metadata(entry, manifest.dataset_root, manifest, errors):
        return False
    try:
        _validate_trajectory_structure(entry.logical_path, episode_index, session, manifest, errors)
    except Exception as exc:  # noqa: BLE001 - numpy/pyarrow expose varied corrupt-file errors
        errors.append(f"cannot read {label} {logical}: {exc}")
        return False
    try:
        manifest.add_file(entry.logical_path, entry.resolved_path)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot add {label} to manifest: {exc}")
        return False
    return True


def _mark_trajectory_sidecar_metadata(
    entry: _FileEntry,
    allowed_root: Path,
    manifest: _Manifest,
    errors: list[str],
) -> bool:
    suffix = entry.logical_path.suffix.lower()
    if suffix == ".npz":
        candidates = (entry.logical_path.parent / "meta.json",)
    elif suffix in {".parquet", ".pq"} and len(entry.logical_path.parents) >= 3:
        candidates = (entry.logical_path.parents[2] / "meta/info.json",)
    else:
        candidates = ()
    for candidate in candidates:
        safe_candidate, error = _safe_logical_path(
            candidate,
            base=entry.logical_path.parent,
            allowed_root=allowed_root,
            label="trajectory metadata",
        )
        if error:
            errors.append(error)
            return False
        assert safe_candidate is not None
        if not os.path.lexists(safe_candidate):
            continue
        metadata_entry, error = _safe_reference(
            safe_candidate,
            base=entry.logical_path.parent,
            allowed_root=allowed_root,
            label="trajectory metadata",
        )
        if error:
            errors.append(error)
            return False
        assert metadata_entry is not None
        try:
            manifest.add_file(metadata_entry.logical_path, metadata_entry.resolved_path)
            manifest.mark_metadata(metadata_entry.logical_path)
        except (OSError, ValueError, _DatasetFileError) as exc:
            errors.append(f"cannot read trajectory metadata {safe_candidate}: {exc}")
            return False
    return True


def _mark_adapter_metadata(kind: DatasetKind | None, root: Path, manifest: _Manifest) -> list[str]:
    if kind is DatasetKind.LEROBOT:
        paths = []
        for path in manifest.files:
            relative = path.relative_to(root).as_posix()
            is_known_file = relative in {
                "meta/info.json",
                "meta/stats.json",
                "meta/tasks.parquet",
            }
            is_episode_metadata = relative.startswith("meta/episodes/") and path.suffix.lower() in {
                ".parquet",
                ".pq",
            }
            if is_known_file or is_episode_metadata:
                paths.append(path)
    elif kind is DatasetKind.GENIE02_SESSION:
        paths = [
            path
            for name in ("session.json", "episodes.csv", "raw_refs.json")
            if (path := root / name) in manifest.files
        ]
    else:
        paths = []
    errors: list[str] = []
    for path in sorted(paths):
        try:
            manifest.mark_metadata(path)
        except (OSError, _DatasetFileError) as exc:
            relative = path.relative_to(root).as_posix()
            errors.append(f"cannot hash adapter metadata {relative}: {exc}")
    return errors


def _safe_logical_path(
    raw_path: str | Path, *, base: Path, allowed_root: Path, label: str
) -> tuple[Path | None, str | None]:
    candidate = Path(raw_path).expanduser()
    logical = Path(os.path.abspath(candidate if candidate.is_absolute() else base / candidate))
    if not _is_relative_to(logical, allowed_root):
        return None, f"{label} is outside allowed root: {logical}"
    try:
        boundary_target = logical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve {label} {logical}: {exc}"
    if not _is_relative_to(boundary_target, allowed_root):
        return None, f"{label} is outside allowed root: {logical} -> {boundary_target}"
    return logical, None


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _DatasetFileError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise _DatasetFileError(f"{label} must be finite")
    return result


def _load_genie_session(
    root: Path, manifest: _Manifest, errors: list[str]
) -> dict[str, Any] | None:
    session = _read_json(root / "session.json", "session.json", errors, manifest)
    if session is None:
        return None
    missing = [field for field in _SESSION_FIELDS if field not in session]
    if (
        session.get("record_dataset", True) is not False
        and not str(session.get("trajectory_log_dir", "")).strip()
        and not str(session.get("dataset_root", "")).strip()
    ):
        missing.append("dataset_root")
    if missing:
        errors.append(f"session.json is missing fields: {', '.join(dict.fromkeys(missing))}")
        return None
    try:
        if session["schema_version"] != "1.0":
            raise _DatasetFileError("session.json schema_version must be '1.0'")
        if session["status"] not in {"recording", "completed", "aborted"}:
            raise _DatasetFileError(f"invalid session status: {session['status']!r}")
        if session["rollout_mode"] not in {"ee", "pi05", "default"}:
            raise _DatasetFileError(f"invalid rollout_mode: {session['rollout_mode']!r}")
        if session["dataset_backend"] not in {"lerobot", "native"}:
            raise _DatasetFileError(f"invalid dataset_backend: {session['dataset_backend']!r}")
        if _finite_float(session["fps"], "session.fps") <= 0:
            raise _DatasetFileError("session.fps must be greater than zero")
        target = session["num_episodes_target"]
        if not isinstance(target, int) or isinstance(target, bool) or target < 0:
            raise _DatasetFileError("session.num_episodes_target must be a non-negative integer")
    except _DatasetFileError as exc:
        errors.append(str(exc))
        return None
    return session


def _load_genie_episodes(
    root: Path,
    session: dict[str, Any],
    manifest: _Manifest,
    errors: list[str],
) -> list[dict[str, str]] | None:
    path = root / "episodes.csv"
    try:
        with _stable_binary_file(path, manifest, "episodes.csv") as handle:
            text = handle.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        missing = [field for field in _EPISODE_FIELDS if field not in (reader.fieldnames or ())]
        if missing:
            raise _DatasetFileError(f"episodes.csv is missing fields: {', '.join(missing)}")
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error, _DatasetFileError) as exc:
        errors.append(f"cannot read episodes.csv: {exc}")
        return None
    seen: set[int] = set()
    for line, row in enumerate(rows, 2):
        prefix = f"episodes.csv row {line}"
        try:
            index = int(row["episode_index"])
            if row["session_id"] != session["session_id"] or index < 0 or index in seen:
                raise _DatasetFileError(f"{prefix}: invalid session_id or episode_index")
            seen.add(index)
            has_path = bool(row["episode_path"].strip() or row["trajectory_path"].strip())
            has_session_path = bool(str(session.get("trajectory_log_dir", "")).strip())
            if not (has_path or has_session_path):
                raise _DatasetFileError(f"{prefix}: episode_path or trajectory_path is required")
            if row["outcome"].strip().lower() not in {"success", "failure"}:
                raise _DatasetFileError(f"{prefix}: invalid outcome")
            start = _finite_float(row["t_start"], f"{prefix} t_start")
            end = _finite_float(row["t_end"], f"{prefix} t_end")
            duration = _finite_float(row["duration_s"], f"{prefix} duration_s")
            if duration < 0 or end < start or abs(end - start - duration) > 0.0015:
                raise _DatasetFileError(f"{prefix}: invalid timestamps or duration_s")
            if row["operator_intervened"].strip().lower() not in {"true", "false"}:
                raise _DatasetFileError(f"{prefix}: invalid operator_intervened")
        except (TypeError, ValueError, _DatasetFileError) as exc:
            errors.append(str(exc))
    return rows


def _inspect_genie02(
    root: Path, allowed_root: Path, manifest: _Manifest
) -> tuple[int | None, list[str]]:
    allowed_root = root
    errors: list[str] = []
    session_path = root / "session.json"
    episodes_path = root / "episodes.csv"
    if session_path not in manifest.files:
        errors.append("missing required Genie02 metadata: session.json")
    if episodes_path not in manifest.files:
        errors.append("missing required Genie02 metadata: episodes.csv")
    if errors:
        if session_path in manifest.files:
            _read_json(session_path, "session.json", errors, manifest)
        return None, errors

    session = _load_genie_session(root, manifest, errors)
    if session is None:
        return None, errors
    rows = _load_genie_episodes(root, session, manifest, errors)
    if rows is None:
        return None, errors

    trajectory_directories: list[Path] = []
    if str(session.get("trajectory_log_dir", "")).strip():
        trajectory_directories.append(Path(str(session["trajectory_log_dir"])))
    raw_refs_path = root / "raw_refs.json"
    if raw_refs_path in manifest.files:
        refs = _read_json(raw_refs_path, "raw_refs.json", errors, manifest)
        if refs and str(refs.get("trajectory_log_dir", "")).strip():
            trajectory_directories.append(Path(str(refs["trajectory_log_dir"])))
    trajectory_directories.append(Path("trajectories"))

    resolved_directories: list[Path] = []
    for raw_directory in trajectory_directories:
        directory, error = _safe_logical_path(
            raw_directory,
            base=root,
            allowed_root=allowed_root,
            label="trajectory directory",
        )
        if error:
            errors.append(error)
            continue
        assert directory is not None
        if directory.is_dir() and directory not in resolved_directories:
            resolved_directories.append(directory)

    for row in rows:
        index = int(row["episode_index"])
        trajectory_path = row["trajectory_path"].strip()
        if trajectory_path:
            direct_path, error = _safe_logical_path(
                trajectory_path,
                base=root,
                allowed_root=allowed_root,
                label=f"episodes.csv trajectory_path for episode {index}",
            )
            if error:
                errors.append(error)
                continue
            assert direct_path is not None
            selected_path: Path | None = None
            if direct_path.is_dir():
                selected_path = _find_native_trajectory(direct_path, index)
            elif direct_path.is_file():
                selected_path = direct_path
            if selected_path is None:
                selected_path = next(
                    (
                        directory / direct_path.name
                        for directory in resolved_directories
                        if (directory / direct_path.name).is_file()
                    ),
                    None,
                )
            if selected_path is not None:
                _validate_genie_trajectory(
                    str(selected_path),
                    session=session,
                    root=root,
                    allowed_root=allowed_root,
                    manifest=manifest,
                    episode_index=index,
                    label=f"episodes.csv trajectory_path for episode {index}",
                    errors=errors,
                )
                continue

        selected_path = next(
            (
                found
                for directory in resolved_directories
                if (found := _find_native_trajectory(directory, index)) is not None
            ),
            None,
        )
        if selected_path is not None:
            _validate_genie_trajectory(
                str(selected_path),
                session=session,
                root=root,
                allowed_root=allowed_root,
                manifest=manifest,
                episode_index=index,
                label=f"trajectory for episode {index}",
                errors=errors,
            )
            continue

        episode_path = row["episode_path"].strip()
        if not episode_path:
            errors.append(f"no trajectory found for episode {index}")
            continue
        logical_episode_path, error = _safe_logical_path(
            episode_path,
            base=root,
            allowed_root=allowed_root,
            label=f"episodes.csv episode_path for episode {index}",
        )
        if error:
            errors.append(error)
            continue
        assert logical_episode_path is not None
        if session["dataset_backend"] == "native":
            native_path = _find_native_trajectory(logical_episode_path, index)
            if native_path is None:
                errors.append(f"no native frames.npz for episode {index}")
                continue
            _validate_genie_trajectory(
                str(native_path),
                session=session,
                root=root,
                allowed_root=allowed_root,
                manifest=manifest,
                episode_index=index,
                label=f"episodes.csv episode_path for episode {index}",
                errors=errors,
            )
            continue
        if not logical_episode_path.exists():
            errors.append(f"episode_path does not exist: {logical_episode_path}")
            continue
        scan_errors = _scan_dataset(logical_episode_path, allowed_root, manifest)
        errors.extend(scan_errors)
        parquet_paths = sorted(
            path
            for path in manifest.files
            if _is_relative_to(path, logical_episode_path / "data") and path.suffix == ".parquet"
        )
        last_errors: list[str] = []
        for parquet_path in parquet_paths:
            candidate_errors: list[str] = []
            if _validate_genie_trajectory(
                str(parquet_path),
                session=session,
                root=root,
                allowed_root=allowed_root,
                manifest=manifest,
                episode_index=index,
                label=f"episodes.csv episode_path for episode {index}",
                errors=candidate_errors,
            ):
                last_errors = []
                break
            last_errors = candidate_errors
        else:
            suffix = f": {last_errors[-1]}" if last_errors else ""
            errors.append(f"no LeRobot parquet for episode {index}{suffix}")
    return len(rows), errors


def _detect_kind(root: Path) -> tuple[DatasetKind | None, str | None]:
    lerobot_markers = (root / "meta/info.json", root / "meta/episodes", root / "data")
    genie_markers = (root / "session.json", root / "episodes.csv")
    complete_lerobot = (
        lerobot_markers[0].is_file() and lerobot_markers[1].is_dir() and lerobot_markers[2].is_dir()
    )
    complete_genie = all(marker.is_file() for marker in genie_markers)
    if complete_lerobot and complete_genie:
        return None, "ambiguous dataset format: complete LeRobot and Genie02 signatures"
    if complete_lerobot:
        return DatasetKind.LEROBOT, None
    if complete_genie:
        return DatasetKind.GENIE02_SESSION, None
    partial_lerobot = any(os.path.lexists(marker) for marker in lerobot_markers)
    partial_genie = any(os.path.lexists(marker) for marker in genie_markers)
    if partial_lerobot and partial_genie:
        return None, "ambiguous partial dataset markers for LeRobot and Genie02"
    if partial_lerobot:
        return DatasetKind.LEROBOT, None
    if partial_genie:
        return DatasetKind.GENIE02_SESSION, None
    return None, None


def inspect_dataset(path: Path, allowed_root: Path) -> DatasetInspection:
    """Inspect a quiescent dataset; portable filesystems cannot snapshot a whole tree atomically."""
    resolved_allowed, allowed_error = _resolve_root(allowed_root, "allowed root")
    if allowed_error:
        empty_manifest = _Manifest(Path.cwd())
        return DatasetInspection(
            None, False, empty_manifest.fingerprint(), 0, None, (allowed_error,)
        )
    assert resolved_allowed is not None

    resolved_root, root_error = _resolve_dataset_root(path, resolved_allowed)
    if root_error:
        empty_manifest = _Manifest(resolved_allowed)
        return DatasetInspection(None, False, empty_manifest.fingerprint(), 0, None, (root_error,))
    assert resolved_root is not None
    manifest = _Manifest(resolved_root)
    scan_errors = _scan_dataset(resolved_root, resolved_root, manifest)
    kind, detection_error = _detect_kind(resolved_root)
    episode_count: int | None = None
    validation_errors = _mark_adapter_metadata(kind, resolved_root, manifest)
    if detection_error:
        validation_errors.append(detection_error)
    elif kind is DatasetKind.LEROBOT:
        episode_count, adapter_errors = _inspect_lerobot(resolved_root, manifest)
        validation_errors.extend(adapter_errors)
    elif kind is DatasetKind.GENIE02_SESSION:
        episode_count, adapter_errors = _inspect_genie02(resolved_root, resolved_allowed, manifest)
        validation_errors.extend(adapter_errors)
    else:
        validation_errors.append("unknown dataset format")

    validation_errors.extend(manifest.verify_unchanged())

    errors = tuple(scan_errors + validation_errors)
    return DatasetInspection(
        kind=kind,
        ready=kind is not None and not errors,
        fingerprint=manifest.fingerprint(),
        size_bytes=manifest.size_bytes,
        episode_count=episode_count,
        errors=errors,
    )
