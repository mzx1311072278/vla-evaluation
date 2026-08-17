"""Safe local dataset source resolution and rsync argument construction."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath

from vla_eval.config import LocalSource
from vla_eval.remote import normalize_remote_relative_path


def _directory_chain(path: Path) -> list[Path]:
    current = Path(path.anchor)
    directories = [current]
    for part in path.parts[1:]:
        current /= part
        directories.append(current)
    return directories


def _has_access(path: Path, mode: int) -> bool:
    try:
        return os.access(path, mode, effective_ids=True)
    except (NotImplementedError, TypeError):
        return os.access(path, mode)


def _validate_directory_chain(path: Path) -> None:
    for component in _directory_chain(path):
        try:
            component_stat = os.lstat(component)
        except OSError as error:
            raise ValueError("local source path must be an existing directory") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError("local source path must not contain symlink components")
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError("local source path must be an existing directory")


def resolve_local_source_directory(
    source: LocalSource,
    selected_root: str,
    relative_path: str,
) -> Path:
    """Resolve one configured local dataset directory without following symlinks."""
    if not isinstance(selected_root, str):
        raise TypeError("selected local root must be a string")
    configured_root = next(
        (root for root in source.roots if str(root) == selected_root),
        None,
    )
    if configured_root is None:
        raise ValueError("selected local root must be a configured local root")

    normalized_relative = normalize_remote_relative_path(relative_path)
    candidate = configured_root.joinpath(*PurePosixPath(normalized_relative).parts)
    _validate_directory_chain(candidate)
    if not _has_access(candidate, os.R_OK | os.X_OK):
        raise ValueError("local source path must be readable and searchable")

    resolved_root = configured_root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError("local source path escaped the configured local root")
    return resolved_candidate


def _absolute_directory(path: Path, field_name: str) -> Path:
    try:
        candidate = Path(path)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a filesystem path") from error
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field_name} must be an absolute normalized path")
    if not candidate.is_dir():
        raise ValueError(f"{field_name} must be an existing directory")
    return candidate


def build_local_rsync_argv(source: Path, staging: Path) -> list[str]:
    """Build argv for a local resumable directory copy without invoking a shell."""
    source_directory = _absolute_directory(source, "local source")
    staging_directory = _absolute_directory(staging, "staging destination")
    return [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "--",
        f"{source_directory}/",
        f"{staging_directory}/",
    ]
