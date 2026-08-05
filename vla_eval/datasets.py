"""Dataset discovery, security preflight, and lightweight manifest fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from Genie02_report.genie02_episode_metrics import _load_file
from Genie02_report.genie02_eval_common import load_episodes, load_session


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


@dataclass(frozen=True)
class _FileEntry:
    logical_path: Path
    resolved_path: Path


class _Manifest:
    def __init__(self, allowed_root: Path) -> None:
        self.allowed_root = allowed_root
        self.entries: dict[str, dict[str, Any]] = {}
        self.files: dict[Path, Path] = {}
        self._sized_files: set[tuple[int, int]] = set()
        self.size_bytes = 0

    def add_file(self, logical_path: Path, resolved_path: Path) -> None:
        logical_path = Path(os.path.abspath(logical_path))
        relative = logical_path.relative_to(self.allowed_root).as_posix()
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
        identity = (file_stat.st_dev, file_stat.st_ino)
        if identity not in self._sized_files:
            self._sized_files.add(identity)
            self.size_bytes += file_stat.st_size

    def mark_metadata(self, logical_path: Path) -> None:
        logical_path = Path(os.path.abspath(logical_path))
        relative = logical_path.relative_to(self.allowed_root).as_posix()
        resolved_path = self.files.get(logical_path)
        if resolved_path is not None:
            self.entries[relative]["sha256"] = _hash_file(resolved_path)

    def fingerprint(self) -> str:
        canonical = json.dumps(
            [self.entries[key] for key in sorted(self.entries)],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {label}: {exc}")
        return None
    except (OSError, UnicodeError) as exc:
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
    path: Path, episode_index: int, expected_length: int, errors: list[str]
) -> None:
    try:
        parquet_file = pq.ParquetFile(path)
        columns = set(parquet_file.schema_arrow.names)
    except Exception as exc:  # noqa: BLE001 - pyarrow exposes varied corrupt-file errors
        errors.append(f"cannot read referenced data parquet {path}: {exc}")
        return
    missing = sorted({"episode_index", "timestamp", "action"} - columns)
    if missing:
        errors.append(f"referenced data parquet {path} is missing columns: {', '.join(missing)}")
        return
    frame_count = 0
    try:
        for batch in parquet_file.iter_batches(columns=["episode_index"], batch_size=65_536):
            frame_count += batch.column(0).to_pylist().count(episode_index)
    except Exception as exc:  # noqa: BLE001 - pyarrow exposes varied corrupt-file errors
        errors.append(f"cannot read episode_index from data parquet {path}: {exc}")
        return
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
        _read_json(manifest.files[info_path], "meta/info.json", errors)
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
    required_columns = {
        "episode_index",
        "length",
        "episode_success",
        "data/chunk_index",
        "data/file_index",
    }
    for logical_path in episode_paths:
        try:
            frame = pd.read_parquet(manifest.files[logical_path])
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
                _validate_data_parquet(manifest.files[data_path], index, length, errors)

            for key in sorted(video_keys):
                video_columns = {
                    "file_index": f"videos/{key}/file_index",
                    "from_timestamp": f"videos/{key}/from_timestamp",
                    "to_timestamp": f"videos/{key}/to_timestamp",
                }
                absent = [name for name in video_columns.values() if name not in frame]
                if absent:
                    errors.append(
                        f"episode metadata for video stream {key} is missing columns: {', '.join(absent)}"
                    )
                    continue
                video_index = _int_value(
                    row[video_columns["file_index"]], f"{prefix} {key} video file_index", errors
                )
                if video_index is None:
                    continue
                try:
                    start = float(row[video_columns["from_timestamp"]])
                    end = float(row[video_columns["to_timestamp"]])
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
    if not _is_relative_to(logical, allowed_root):
        errors.append(f"{label} is outside allowed root: {logical}")
        return False
    try:
        boundary_target = logical.resolve(strict=False)
        if not _is_relative_to(boundary_target, allowed_root):
            errors.append(f"{label} is outside allowed root: {logical} -> {boundary_target}")
            return False
        resolved_candidate = logical.resolve(strict=True)
    except FileNotFoundError:
        errors.append(f"{label} does not exist: {logical}")
        return False
    except (OSError, RuntimeError) as exc:
        errors.append(f"cannot resolve {label} {logical}: {exc}")
        return False
    if not _is_relative_to(resolved_candidate, allowed_root):
        errors.append(f"{label} is outside allowed root: {logical} -> {resolved_candidate}")
        return False
    entry, error = _safe_reference(logical, base=root, allowed_root=allowed_root, label=label)
    if error:
        errors.append(error)
        return False
    assert entry is not None
    if not _mark_trajectory_sidecar_metadata(entry, allowed_root, manifest, errors):
        return False
    try:
        trajectory = _load_file(entry.resolved_path, session, episode_index)
    except Exception as exc:  # noqa: BLE001 - the legacy reader wraps backend-specific errors
        errors.append(f"cannot read {label} {logical}: {exc}")
        return False
    if len(trajectory.values) == 0:
        errors.append(f"{label} is empty: {logical}")
        return False
    if trajectory.times is not None and len(trajectory.times) != len(trajectory.values):
        errors.append(f"{label} timestamp length does not match trajectory values: {logical}")
        return False
    if trajectory.intervention is not None and len(trajectory.intervention) != len(
        trajectory.values
    ):
        errors.append(f"{label} intervention length does not match trajectory values: {logical}")
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
    suffix = entry.resolved_path.suffix.lower()
    if suffix == ".npz":
        candidates = (entry.resolved_path.parent / "meta.json",)
    elif suffix in {".parquet", ".pq"} and len(entry.resolved_path.parents) >= 3:
        candidates = (entry.resolved_path.parents[2] / "meta/info.json",)
    else:
        candidates = ()
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        metadata_entry, error = _safe_reference(
            candidate,
            base=entry.resolved_path.parent,
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
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read trajectory metadata {candidate}: {exc}")
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
        except OSError as exc:
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


def _inspect_genie02(
    root: Path, allowed_root: Path, manifest: _Manifest
) -> tuple[int | None, list[str]]:
    errors: list[str] = []
    session_path = root / "session.json"
    episodes_path = root / "episodes.csv"
    if session_path not in manifest.files:
        errors.append("missing required Genie02 metadata: session.json")
    if episodes_path not in manifest.files:
        errors.append("missing required Genie02 metadata: episodes.csv")
    if errors:
        if session_path in manifest.files:
            _read_json(manifest.files[session_path], "session.json", errors)
        return None, errors

    try:
        session = load_session(root)
    except Exception as exc:  # noqa: BLE001 - legacy reader leaks decode/type errors
        errors.append(f"cannot read session.json: {exc}")
        session = None
    try:
        rows = load_episodes(root, session) if session is not None else []
    except Exception as exc:  # noqa: BLE001 - legacy reader leaks malformed CSV errors
        errors.append(f"cannot read episodes.csv: {exc}")
        return None, errors
    if session is None:
        return None, errors

    trajectory_directories: list[Path] = []
    if str(session.get("trajectory_log_dir", "")).strip():
        trajectory_directories.append(Path(str(session["trajectory_log_dir"])))
    raw_refs_path = root / "raw_refs.json"
    if raw_refs_path in manifest.files:
        refs = _read_json(manifest.files[raw_refs_path], "raw_refs.json", errors)
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


def _detect_kind(root: Path) -> DatasetKind | None:
    has_lerobot_marker = any(
        os.path.lexists(path)
        for path in (root / "meta/info.json", root / "meta/episodes", root / "data")
    )
    if has_lerobot_marker:
        return DatasetKind.LEROBOT
    if os.path.lexists(root / "session.json") or os.path.lexists(root / "episodes.csv"):
        return DatasetKind.GENIE02_SESSION
    return None


def inspect_dataset(path: Path, allowed_root: Path) -> DatasetInspection:
    """Inspect a registered dataset without reading bulk data file contents."""
    resolved_allowed, allowed_error = _resolve_root(allowed_root, "allowed root")
    if allowed_error:
        empty_manifest = _Manifest(Path.cwd())
        return DatasetInspection(
            None, False, empty_manifest.fingerprint(), 0, None, (allowed_error,)
        )
    assert resolved_allowed is not None

    resolved_root, root_error = _resolve_root(path, "dataset root")
    if root_error:
        empty_manifest = _Manifest(resolved_allowed)
        return DatasetInspection(None, False, empty_manifest.fingerprint(), 0, None, (root_error,))
    assert resolved_root is not None
    if not _is_relative_to(resolved_root, resolved_allowed):
        empty_manifest = _Manifest(resolved_allowed)
        error = f"dataset root is outside allowed root: {resolved_root}"
        return DatasetInspection(None, False, empty_manifest.fingerprint(), 0, None, (error,))

    manifest = _Manifest(resolved_allowed)
    scan_errors = _scan_dataset(resolved_root, resolved_allowed, manifest)
    kind = _detect_kind(resolved_root)
    episode_count: int | None = None
    validation_errors = _mark_adapter_metadata(kind, resolved_root, manifest)
    if kind is DatasetKind.LEROBOT:
        episode_count, adapter_errors = _inspect_lerobot(resolved_root, manifest)
        validation_errors.extend(adapter_errors)
    elif kind is DatasetKind.GENIE02_SESSION:
        episode_count, adapter_errors = _inspect_genie02(resolved_root, resolved_allowed, manifest)
        validation_errors.extend(adapter_errors)
    else:
        validation_errors.append("unknown dataset format")

    errors = tuple(scan_errors + validation_errors)
    return DatasetInspection(
        kind=kind,
        ready=kind is not None and not errors,
        fingerprint=manifest.fingerprint(),
        size_bytes=manifest.size_bytes,
        episode_count=episode_count,
        errors=errors,
    )
