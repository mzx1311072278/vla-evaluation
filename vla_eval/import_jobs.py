"""Resumable remote dataset transfer and atomic inbox publication."""

from __future__ import annotations

import codecs
import math
import os
import re
import selectors
import shlex
import stat
import subprocess
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from vla_eval.config import RemoteSource
from vla_eval.datasets import DatasetInspection, inspect_dataset
from vla_eval.remote import (
    build_rsync_argv,
    normalize_remote_relative_path,
    validate_remote_source_files,
    validate_staging_path,
)

CONNECTING: Final = "CONNECTING"
TRANSFERRING: Final = "TRANSFERRING"
VERIFYING: Final = "VERIFYING"
PREFLIGHT: Final = "PREFLIGHT"
READY: Final = "READY"
FAILED: Final = "FAILED"
IMPORT_STATE_ORDER: Final = (CONNECTING, TRANSFERRING, VERIFYING, PREFLIGHT, READY)

_PROGRESS_PATTERN = re.compile(r"\s*(\d{1,3})%")
_PROGRESS_PREFIX_PATTERN = re.compile(r"[\s\d,.]*\Z")
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(password|passwd|secret|token|credential|authorization)\b(\s*[:=]\s*)(\S+)"
)
_MAX_TAIL_LINES = 200
_MAX_TAIL_LINE_LENGTH = 500
_MAX_VALIDATION_ERRORS = 8
_MAX_VALIDATION_ERROR_LENGTH = 240
_PROCESS_WAIT_SECONDS = 2.0
_PROCESS_POLL_SECONDS = 0.1

StateCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
CancellationCallback = Callable[[], bool]
Transfer = Callable[[Sequence[str], ProgressCallback], None]
Inspector = Callable[[Path], DatasetInspection]


def _ignore_state(_state: str) -> None:
    return None


def _ignore_progress(_progress: float) -> None:
    return None


def _not_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class ImportCallbacks:
    on_state: StateCallback = _ignore_state
    on_progress: ProgressCallback = _ignore_progress
    is_cancelled: CancellationCallback = _not_cancelled


class _ProgressReporter:
    def __init__(self, callbacks: ImportCallbacks) -> None:
        self.callbacks = callbacks
        self.last_progress = 0.0

    def poll(self) -> None:
        if self.callbacks.is_cancelled():
            raise TransferError("import cancelled")

    def __call__(self, value: float) -> None:
        self.poll()
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = self.last_progress
        self.last_progress = max(
            self.last_progress,
            min(100.0, max(0.0, numeric)),
        )
        self.callbacks.on_progress(self.last_progress)


@dataclass(frozen=True)
class ImportSpec:
    job_id: str
    source_name: str
    remote_root: str
    remote_relative_path: str
    staging_path: Path
    target_path: Path
    mode: Literal["injected", "production"] = "injected"
    source: RemoteSource | None = None
    trusted_credentials_root: Path | None = None
    trusted_staging_root: Path | None = None
    trusted_inbox_root: Path | None = None


@dataclass(frozen=True)
class ImportResult:
    dataset_path: Path
    inspection: DatasetInspection


class TransferError(RuntimeError):
    """A retryable transfer failure with bounded, sanitized diagnostic output."""

    def __init__(self, message: str, safe_tail: Sequence[str] = ()) -> None:
        super().__init__(_sanitize_text(message))
        self.safe_tail = tuple(_sanitize_text(line) for line in safe_tail)[-_MAX_TAIL_LINES:]


class DatasetValidationError(RuntimeError):
    """A non-retryable dataset preflight failure."""

    def __init__(self, errors: Sequence[str]) -> None:
        bounded = tuple(
            _sanitize_text(str(error))[:_MAX_VALIDATION_ERROR_LENGTH]
            for error in errors[:_MAX_VALIDATION_ERRORS]
        )
        if not bounded:
            bounded = ("dataset did not pass preflight validation",)
        self.errors = bounded
        super().__init__("; ".join(bounded))


_DEFAULT_INSPECTOR = object()
_DEFAULT_CALLBACKS = ImportCallbacks()


def _sanitize_text(value: str, secrets: Sequence[str] = ()) -> str:
    sanitized = "".join(
        character
        for character in str(value)
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )
    for secret in sorted((secret for secret in secrets if secret), key=len, reverse=True):
        sanitized = sanitized.replace(secret, "[redacted]")
    sanitized = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2[redacted]", sanitized)
    return sanitized[:_MAX_TAIL_LINE_LENGTH]


def _argv_secrets(argv: Sequence[str]) -> tuple[str, ...]:
    secrets: set[str] = set()
    for argument in argv:
        if len(argument) > 4:
            secrets.add(argument)
        try:
            tokens = shlex.split(argument)
        except ValueError:
            tokens = []
        for index, token in enumerate(tokens):
            if token == "-i" and index + 1 < len(tokens):
                secrets.add(tokens[index + 1])
            if token.startswith("UserKnownHostsFile="):
                secrets.add(token.partition("=")[2])
            if token.startswith("/"):
                secrets.add(token)
    return tuple(secrets)


def _terminate_process(process: subprocess.Popen[str]) -> list[Exception]:
    errors: list[Exception] = []
    try:
        process.terminate()
    except Exception as error:  # noqa: BLE001 - aggregate arbitrary cleanup failures
        errors.append(error)
    try:
        process.wait(timeout=_PROCESS_WAIT_SECONDS)
        return errors
    except subprocess.TimeoutExpired:
        pass
    except Exception as error:  # noqa: BLE001 - cleanup must continue through kill
        errors.append(error)
    try:
        process.kill()
    except Exception as error:  # noqa: BLE001 - aggregate arbitrary cleanup failures
        errors.append(error)
    try:
        process.wait(timeout=_PROCESS_WAIT_SECONDS)
    except Exception as error:  # noqa: BLE001 - aggregate arbitrary cleanup failures
        errors.append(error)
    return errors


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)):
        raise TransferError("rsync argv must be a nonempty sequence of strings")
    try:
        normalized = list(argv)
    except (TypeError, ValueError) as error:
        raise TransferError("rsync argv must be a nonempty sequence of strings") from error
    if not normalized:
        raise TransferError("rsync argv must be a nonempty sequence of strings")
    for argument in normalized:
        if not isinstance(argument, str) or not argument:
            raise TransferError("rsync argv entries must be nonempty strings")
        if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in argument):
            raise TransferError("rsync argv entries must not contain control characters")
    return normalized


def _fallback_stream(output: object, feed_text: Callable[[str], None]) -> None:
    read = output.read  # type: ignore[attr-defined]
    while chunk := read(4096):
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        feed_text(chunk)


def _selector_stream(
    output: object,
    descriptor: int,
    feed_text: Callable[[str], None],
    on_poll: Callable[[], None],
) -> None:
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    encoding = getattr(output, "encoding", None) or "utf-8"
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    primary: BaseException | None = None
    try:
        while True:
            events = selector.select(_PROCESS_POLL_SECONDS)
            if not events:
                on_poll()
                continue
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            feed_text(decoder.decode(chunk))
        feed_text(decoder.decode(b"", final=True))
    except BaseException as error:  # noqa: BLE001 - preserve interrupts during cleanup
        primary = error
    try:
        selector.close()
    except Exception as close_error:
        if primary is not None:
            raise BaseExceptionGroup(
                "streaming and selector cleanup both failed",
                [primary, close_error],
            ) from primary
        raise
    if primary is not None:
        raise primary


def run_rsync(
    argv: Sequence[str],
    on_progress: ProgressCallback,
    *,
    on_poll: Callable[[], None] | None = None,
) -> None:
    """Run one argv-only rsync process and stream progress2 updates."""
    normalized_argv = _normalize_argv(argv)
    try:
        process = subprocess.Popen(
            normalized_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
    except FileNotFoundError as error:
        raise TransferError("rsync executable was not found; install rsync and retry") from error
    except PermissionError as error:
        raise TransferError("rsync could not be started due to a permission error") from error
    except (OSError, ValueError, TypeError, IndexError) as error:
        raise TransferError("rsync could not be started") from error

    output = process.stdout
    if output is None:
        cleanup_errors = _terminate_process(process)
        failure = TransferError("rsync output pipe was unavailable")
        if cleanup_errors:
            raise ExceptionGroup(
                "rsync output setup and process cleanup both failed",
                [failure, *cleanup_errors],
            ) from failure
        raise failure

    tail: deque[str] = deque(maxlen=_MAX_TAIL_LINES)
    secrets = _argv_secrets(normalized_argv)
    pending = ""
    last_progress = 0.0
    poll_callback = on_poll or getattr(on_progress, "poll", _not_cancelled)

    def consume_record(record: str) -> None:
        nonlocal last_progress
        if record:
            tail.append(_sanitize_text(record, secrets))
        if "|" in record:
            return
        for match in _PROGRESS_PATTERN.finditer(record):
            if _PROGRESS_PREFIX_PATTERN.fullmatch(record[: match.start()]) is None:
                continue
            value = min(100.0, max(0.0, float(match.group(1))))
            last_progress = max(last_progress, value)
            on_progress(last_progress)
            return

    def feed_text(chunk: str) -> None:
        nonlocal pending
        for character in chunk:
            if character in "\r\n":
                consume_record(pending)
                pending = ""
            else:
                pending += character

    try:
        try:
            descriptor = output.fileno()
        except (AttributeError, OSError, TypeError, ValueError):
            _fallback_stream(output, feed_text)
        else:
            _selector_stream(output, descriptor, feed_text, poll_callback)
        consume_record(pending)
        returncode = process.wait()
    except BaseException as primary:
        cleanup_errors = _terminate_process(process)
        try:
            output.close()
        except Exception as close_error:  # noqa: BLE001 - aggregate cleanup with primary
            cleanup_errors.append(close_error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "rsync execution and cleanup both failed",
                [primary, *cleanup_errors],
            ) from primary
        raise
    try:
        output.close()
    except Exception as close_error:
        raise TransferError("rsync output cleanup failed") from close_error

    if returncode != 0:
        raise TransferError(f"rsync exited with status {returncode}", tuple(tail))


_DEFAULT_RUN_RSYNC = run_rsync


def _absolute_path(value: Path, field_name: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a filesystem path") from error
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain '..' components")
    return Path(os.path.abspath(path))


def _is_contained(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _path_components(path: Path) -> list[Path]:
    current = Path(path.anchor)
    components = [current]
    for part in path.parts[1:]:
        current /= part
        components.append(current)
    return components


def _validate_no_symlink_directory(path: Path, field_name: str) -> None:
    for component in _path_components(path):
        try:
            component_stat = os.lstat(component)
        except OSError as error:
            raise ValueError(f"{field_name} must be an existing directory") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"{field_name} must not contain symlink components")
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory")


def _validate_protected_directory(path: Path, field_name: str) -> None:
    for component in _path_components(path):
        component_stat = os.lstat(component)
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"{field_name} must not contain symlink components")
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(f"{field_name} components must be directories")
        if component_stat.st_uid not in {0, os.geteuid()}:
            raise ValueError(f"{field_name} must be owned by root or the service user")
        if component_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"{field_name} must not be group or other writable")
    if not os.access(path, os.W_OK | os.X_OK):
        raise ValueError(f"{field_name} must be writable and searchable")


def _create_under_root(destination: Path, root: Path, field_name: str) -> None:
    if not _is_contained(destination, root):
        raise ValueError(f"{field_name} must be within its trusted root")
    _validate_protected_directory(root, f"trusted {field_name} root")
    current = root
    for part in destination.relative_to(root).parts:
        current /= part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        component_stat = os.lstat(current)
        if stat.S_ISLNK(component_stat.st_mode) or not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(f"{field_name} must contain only directories")
    _validate_protected_directory(destination, field_name)


def _ensure_safe_injected_directory(path: Path, field_name: str) -> None:
    path = _absolute_path(path, field_name)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child_descriptor = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{field_name} must not contain symlink or non-directory components"
                ) from error
            os.close(descriptor)
            descriptor = child_descriptor
    finally:
        os.close(descriptor)


def _validate_spec(spec: ImportSpec) -> tuple[Path, Path]:
    if not spec.job_id.strip() or not spec.source_name.strip():
        raise ValueError("job and source names must not be empty")
    normalize_remote_relative_path(spec.remote_relative_path)
    remote_root = PurePosixPath(spec.remote_root)
    if not remote_root.is_absolute() or str(remote_root) != spec.remote_root:
        raise ValueError("remote root must be an absolute canonical POSIX path")
    staging = _absolute_path(spec.staging_path, "staging path")
    target = _absolute_path(spec.target_path, "target path")
    if staging == target or _is_contained(target, staging) or _is_contained(staging, target):
        raise ValueError("staging and target paths must be separate")
    if target == Path(target.anchor):
        raise ValueError("filesystem root cannot be an import target")
    return staging, target


def _prepare_paths(spec: ImportSpec, staging: Path, target: Path, production: bool) -> None:
    if production:
        assert spec.trusted_staging_root is not None
        assert spec.trusted_inbox_root is not None
        staging_root = _absolute_path(spec.trusted_staging_root, "trusted staging root")
        inbox_root = _absolute_path(spec.trusted_inbox_root, "trusted inbox root")
        _create_under_root(staging, staging_root, "staging path")
        _create_under_root(target.parent, inbox_root, "inbox path")
    else:
        _ensure_safe_injected_directory(staging.parent, "staging parent")
        _ensure_safe_injected_directory(target.parent, "target parent")
        _validate_no_symlink_directory(staging.parent, "staging parent")
        _validate_no_symlink_directory(target.parent, "target parent")


def _require_production_context(spec: ImportSpec) -> None:
    values = (
        spec.source,
        spec.trusted_credentials_root,
        spec.trusted_staging_root,
        spec.trusted_inbox_root,
    )
    if any(value is None for value in values):
        raise ValueError("default transfer requires complete production trust context")
    assert spec.source is not None
    if spec.source.name != spec.source_name:
        raise ValueError("remote source does not match source name")


def _is_production_mode(spec: ImportSpec, transfer: Transfer) -> bool:
    context = (
        spec.source,
        spec.trusted_credentials_root,
        spec.trusted_staging_root,
        spec.trusted_inbox_root,
    )
    if spec.mode == "production":
        _require_production_context(spec)
        return True
    if spec.mode != "injected":
        raise ValueError("import mode must be 'injected' or 'production'")
    if any(value is not None for value in context):
        raise ValueError("injected mode must not include production context")
    if transfer is _DEFAULT_RUN_RSYNC or transfer is run_rsync:
        raise ValueError("injected mode requires an explicit custom transfer")
    return False


def _ensure_target_available(target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(f"import target already exists: {target.name}")


def _ensure_same_filesystem(staging: Path, target_parent: Path) -> None:
    if staging.stat().st_dev != target_parent.stat().st_dev:
        raise OSError("staging and target parent must be on the same filesystem")


def _verify_staging(spec: ImportSpec, staging: Path, production: bool) -> None:
    if production:
        assert spec.trusted_staging_root is not None
        validate_staging_path(staging, spec.trusted_staging_root)
    else:
        _validate_no_symlink_directory(staging, "staging path")
    if not any(staging.iterdir()):
        raise TransferError("transfer completed without receiving dataset files")


def _verify_target_parent(spec: ImportSpec, target: Path, production: bool) -> None:
    _ensure_target_available(target)
    if production:
        assert spec.trusted_inbox_root is not None
        root = _absolute_path(spec.trusted_inbox_root, "trusted inbox root")
        if not _is_contained(target, root):
            raise ValueError("import target must be within trusted inbox root")
        _validate_protected_directory(target.parent, "target parent")
        resolved_root = root.resolve(strict=True)
        resolved_parent = target.parent.resolve(strict=True)
        if not _is_contained(resolved_parent, resolved_root):
            raise ValueError("resolved import target must be within trusted inbox root")
    else:
        _validate_no_symlink_directory(target.parent, "target parent")


def _emit_state(callbacks: ImportCallbacks, state: str) -> None:
    if callbacks.is_cancelled():
        raise TransferError("import cancelled")
    callbacks.on_state(state)


def execute_import(
    spec: ImportSpec,
    *,
    transfer: Transfer = run_rsync,
    inspector: Inspector | object = _DEFAULT_INSPECTOR,
    callbacks: ImportCallbacks = _DEFAULT_CALLBACKS,
) -> ImportResult:
    """Transfer, preflight, and atomically publish one remote dataset."""
    if not callable(transfer):
        raise TypeError("transfer must be callable")
    if inspector is not _DEFAULT_INSPECTOR and not callable(inspector):
        raise TypeError("inspector must be callable")

    production = False
    failure_callback_needed = False
    injected_transfer_started = False
    published = False
    try:
        production = _is_production_mode(spec, transfer)
        staging, target = _validate_spec(spec)
        _emit_state(callbacks, CONNECTING)
        failure_callback_needed = True
        _prepare_paths(spec, staging, target, production)
        _verify_target_parent(spec, target, production)
        filesystem_probe = staging if staging.exists() else staging.parent
        _ensure_same_filesystem(filesystem_probe, target.parent)

        _emit_state(callbacks, TRANSFERRING)
        if production:
            assert spec.source is not None
            assert spec.trusted_credentials_root is not None
            assert spec.trusted_staging_root is not None
            validate_remote_source_files(
                spec.source,
                trusted_credentials_root=spec.trusted_credentials_root,
            )
            validate_staging_path(staging, spec.trusted_staging_root)
            argv = build_rsync_argv(
                spec.source,
                spec.remote_root,
                spec.remote_relative_path,
                staging,
                trusted_staging_root=spec.trusted_staging_root,
            )
        else:
            argv = []

        report_progress = _ProgressReporter(callbacks)
        injected_transfer_started = not production
        transfer(argv, report_progress)
        _emit_state(callbacks, VERIFYING)
        _verify_staging(spec, staging, production)
        _verify_target_parent(spec, target, production)

        _emit_state(callbacks, PREFLIGHT)
        if inspector is _DEFAULT_INSPECTOR:
            inspection = inspect_dataset(staging, allowed_root=staging)
        else:
            inspection = inspector(staging)
        if not inspection.ready:
            raise DatasetValidationError(inspection.errors)

        _verify_staging(spec, staging, production)
        _verify_target_parent(spec, target, production)
        _ensure_same_filesystem(staging, target.parent)
        report_progress.poll()
        staging.replace(target)
        published = True

        if not target.exists():
            raise OSError("published dataset target is missing")
        if production:
            assert spec.trusted_inbox_root is not None
            resolved_root = spec.trusted_inbox_root.resolve(strict=True)
            if not _is_contained(target.resolve(strict=True), resolved_root):
                raise OSError("published dataset escaped trusted inbox root")

        try:
            callbacks.on_state(READY)
        except Exception as ready_error:
            try:
                target.replace(staging)
            except Exception as rollback_error:  # noqa: BLE001 - report compound publish failure
                raise ExceptionGroup(
                    "READY callback failed and dataset rollback failed",
                    [ready_error, rollback_error],
                ) from ready_error
            published = False
            raise
        failure_callback_needed = False
        return ImportResult(dataset_path=target, inspection=inspection)
    except BaseException as original:
        if not production and injected_transfer_started and not published:
            staging_path = Path(spec.staging_path)
            if staging_path.is_absolute() and not os.path.lexists(staging_path):
                try:
                    _ensure_safe_injected_directory(staging_path, "staging path")
                except (OSError, ValueError) as placeholder_error:
                    original.add_note(f"staging placeholder was not created: {placeholder_error}")
        if failure_callback_needed:
            try:
                callbacks.on_state(FAILED)
            except Exception as failed_callback_error:  # noqa: BLE001 - preserve both callbacks
                raise BaseExceptionGroup(
                    "import and FAILED callback both failed",
                    [original, failed_callback_error],
                ) from original
        raise
