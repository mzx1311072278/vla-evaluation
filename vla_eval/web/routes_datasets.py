import errno
import fcntl
import os
import re
import secrets
import stat
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, or_, select
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User
from vla_eval.security import require_csrf, require_html_user
from vla_eval.web.list_management import (
    ListControls,
    literal_contains_pattern,
    parse_list_controls,
    validate_return_to,
)
from vla_eval.web.templating import templates

router = APIRouter()
_ATTACHMENT_FIELDS = frozenset({"csrf_token", "file"})
_ALLOWED_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".csv"})
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_MAX_DATASET_ATTACHMENTS_BYTES = 100 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_FILE_TOO_LARGE_MESSAGE = "attachment-file-too-large"
_ARCHIVE_KEY = "_archive"
_ARCHIVABLE_DATASET_STATES = frozenset({"READY", "PREFLIGHT_FAILED"})
_ARCHIVE_FORM_FIELDS = frozenset({"csrf_token", "return_to"})
_TERMINAL_EVALUATION_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
)


class _AttachmentValidationError(ValueError):
    pass


class _AttachmentTooLarge(ValueError):
    pass


class _AttachmentConflict(ValueError):
    pass


@dataclass(frozen=True)
class _DatasetLocation:
    inbox: Path
    path_parts: tuple[str, ...]


class _LimitedMultiPartParser(MultiPartParser):
    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        self._current_file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._current_file_bytes += end - start
            if self._current_file_bytes > _MAX_ATTACHMENT_BYTES:
                raise MultiPartException(_FILE_TOO_LARGE_MESSAGE)
        super().on_part_data(data, start, end)


async def _parse_attachment_form(request: Request):
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        return await request.form()
    parser = _LimitedMultiPartParser(request.headers, request.stream())
    try:
        return await parser.parse()
    except MultiPartException as error:
        if error.message == _FILE_TOO_LARGE_MESSAGE:
            raise _AttachmentTooLarge("attachment capacity exceeded") from error
        raise _AttachmentValidationError("invalid attachment form") from error


def _template_context(request: Request, current_user: User, **values):
    return {
        "current_user": current_user,
        "csrf_token": request.session["csrf_token"],
        **values,
    }


def _load_dataset(request: Request, dataset_id: str) -> Dataset:
    with session_scope(request.app.state.engine) as session:
        dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


async def _archive_form_target(request: Request, dataset_id: str) -> str:
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    if set(form.keys()) != _ARCHIVE_FORM_FIELDS:
        raise HTTPException(status_code=422, detail="Invalid dataset archive form")
    return_values = form.getlist("return_to")
    if len(return_values) != 1 or not isinstance(return_values[0], str):
        raise HTTPException(status_code=422, detail="Invalid dataset archive form")
    return validate_return_to(
        return_values[0],
        allowed_paths=frozenset({"/datasets", f"/datasets/{dataset_id}"}),
        fallback="/datasets",
    )


def _valid_archive_snapshot(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "previous_status",
        "archived_at",
        "archived_by",
    }:
        return False
    if value["previous_status"] not in _ARCHIVABLE_DATASET_STATES:
        return False
    archived_at = value["archived_at"]
    archived_by = value["archived_by"]
    if not isinstance(archived_at, str) or not isinstance(archived_by, str) or not archived_by:
        return False
    try:
        parsed_archived_at = datetime.fromisoformat(archived_at)
    except ValueError:
        return False
    return parsed_archived_at.tzinfo is not None


def _dataset_archived_at(dataset: Dataset) -> datetime | None:
    snapshot = dataset.inspection_json.get(_ARCHIVE_KEY)
    if dataset.status != "ARCHIVED" or not _valid_archive_snapshot(snapshot):
        return None
    return datetime.fromisoformat(snapshot["archived_at"])


def _dataset_order(controls: ListControls):
    if controls.sort == "oldest":
        return (Dataset.created_at.asc(), Dataset.id.asc())
    if controls.sort == "name_asc":
        return (
            func.lower(Dataset.name).asc(),
            Dataset.name.asc(),
            Dataset.created_at.desc(),
            Dataset.id.asc(),
        )
    if controls.sort == "name_desc":
        return (
            func.lower(Dataset.name).desc(),
            Dataset.name.desc(),
            Dataset.created_at.desc(),
            Dataset.id.desc(),
        )
    return (Dataset.created_at.desc(), Dataset.id.desc())


def _current_local_url(request: Request) -> str:
    query = request.url.query
    return request.url.path + (f"?{query}" if query else "")


def _validate_attachment_name(raw_name: str | None) -> str:
    if not isinstance(raw_name, str) or not raw_name:
        raise _AttachmentValidationError("invalid attachment name")
    basename = Path(raw_name).name
    if (
        basename != raw_name
        or basename in {"", ".", ".."}
        or "/" in raw_name
        or "\\" in raw_name
        or unicodedata.normalize("NFC", raw_name) != raw_name
        or any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in raw_name)
        or any(unicodedata.category(character) in {"Zl", "Zp"} for character in raw_name)
        or any(
            int(match.group(1), 16) < 32 or int(match.group(1), 16) == 127
            for match in _PERCENT_ESCAPE.finditer(raw_name)
        )
        or len(os.fsencode(raw_name)) > 255
        or Path(basename).suffix.lower() not in _ALLOWED_EXTENSIONS
    ):
        raise _AttachmentValidationError("invalid attachment name")
    return basename


def _resolve_dataset_root(configured_data_root: Path, stored_path: str) -> _DatasetLocation:
    try:
        inbox = (configured_data_root / "inbox").resolve(strict=True)
        raw_dataset_root = Path(stored_path)
        if not raw_dataset_root.is_absolute():
            raise _AttachmentValidationError("invalid dataset path")
        relative_path = raw_dataset_root.relative_to(inbox)
        path_parts = relative_path.parts
        if not path_parts or any(part in {"", ".", ".."} for part in path_parts):
            raise _AttachmentValidationError("invalid dataset path")
        dataset_root = raw_dataset_root.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _AttachmentValidationError("invalid dataset path") from error
    if dataset_root == inbox or inbox not in dataset_root.parents or not dataset_root.is_dir():
        raise _AttachmentValidationError("invalid dataset path")
    return _DatasetLocation(inbox=inbox, path_parts=path_parts)


def _write_all(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        remaining = remaining[written:]


def _open_attachments_directory(dataset_root_fd: int) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        os.mkdir("_attachments", mode=0o750, dir_fd=dataset_root_fd)
    except FileExistsError:
        pass
    try:
        return os.open("_attachments", flags, dir_fd=dataset_root_fd)
    except OSError as error:
        raise _AttachmentValidationError("invalid attachment directory") from error


def _existing_attachment_bytes(attachments_fd: int) -> int:
    total = 0
    try:
        entries = os.scandir(attachments_fd)
        with entries:
            for entry in entries:
                if entry.name == ".upload.lock":
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise _AttachmentValidationError("invalid attachment directory contents")
                total += info.st_size
    except OSError as error:
        raise _AttachmentValidationError("invalid attachment directory contents") from error
    return total


def _store_attachment(
    dataset_location: _DatasetLocation,
    filename: str,
    stream: BinaryIO,
) -> None:
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    dataset_root_fd = -1
    try:
        dataset_root_fd = os.open(dataset_location.inbox, root_flags)
        for path_part in dataset_location.path_parts:
            next_fd = os.open(path_part, root_flags, dir_fd=dataset_root_fd)
            os.close(dataset_root_fd)
            dataset_root_fd = next_fd
    except OSError as error:
        if dataset_root_fd >= 0:
            os.close(dataset_root_fd)
        raise _AttachmentValidationError("invalid dataset path") from error

    attachments_fd = -1
    lock_fd = -1
    temporary_name: str | None = None
    try:
        attachments_fd = _open_attachments_directory(dataset_root_fd)
        lock_flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            lock_fd = os.open(
                ".upload.lock",
                lock_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=attachments_fd,
            )
        except FileExistsError:
            try:
                lock_fd = os.open(".upload.lock", lock_flags, dir_fd=attachments_fd)
            except OSError as error:
                raise _AttachmentValidationError("invalid attachment lock") from error
        except OSError as error:
            raise _AttachmentValidationError("invalid attachment lock") from error
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        existing_bytes = _existing_attachment_bytes(attachments_fd)
        if existing_bytes > _MAX_DATASET_ATTACHMENTS_BYTES:
            raise _AttachmentTooLarge("attachment capacity exceeded")
        try:
            os.stat(filename, dir_fd=attachments_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise _AttachmentConflict("attachment already exists")

        temporary_name = f".upload-{secrets.token_hex(16)}"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=attachments_fd,
            )
        except OSError as error:
            raise _AttachmentValidationError("cannot create attachment") from error

        uploaded_bytes = 0
        try:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                uploaded_bytes += len(chunk)
                if (
                    uploaded_bytes > _MAX_ATTACHMENT_BYTES
                    or existing_bytes + uploaded_bytes > _MAX_DATASET_ATTACHMENTS_BYTES
                ):
                    raise _AttachmentTooLarge("attachment capacity exceeded")
                _write_all(temporary_fd, chunk)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)

        try:
            os.link(
                temporary_name,
                filename,
                src_dir_fd=attachments_fd,
                dst_dir_fd=attachments_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise _AttachmentConflict("attachment already exists") from error
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise _AttachmentConflict("attachment already exists") from error
            raise _AttachmentValidationError("cannot publish attachment") from error
        os.unlink(temporary_name, dir_fd=attachments_fd)
        temporary_name = None
        os.fsync(attachments_fd)
    finally:
        if temporary_name is not None and attachments_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=attachments_fd)
            except FileNotFoundError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        if attachments_fd >= 0:
            os.close(attachments_fd)
        os.close(dataset_root_fd)


@router.get("/datasets", name="datasets")
def dataset_list(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    controls = parse_list_controls(request)
    query = select(Dataset)
    if not controls.include_archived:
        query = query.where(Dataset.status != "ARCHIVED")
    if controls.q:
        pattern = literal_contains_pattern(controls.q)
        query = query.where(
            or_(
                Dataset.name.ilike(pattern, escape="\\"),
                Dataset.path.ilike(pattern, escape="\\"),
            )
        )
    query = query.order_by(*_dataset_order(controls))
    with session_scope(request.app.state.engine) as session:
        datasets = session.scalars(query).all()
    return templates.TemplateResponse(
        request=request,
        name="datasets/index.html",
        context=_template_context(
            request,
            current_user,
            datasets=datasets,
            controls=controls,
            current_url=_current_local_url(request),
        ),
    )


@router.get("/datasets/{dataset_id}")
def dataset_detail(
    request: Request,
    dataset_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    dataset = _load_dataset(request, dataset_id)
    with session_scope(request.app.state.engine) as session:
        recent_evaluations = session.scalars(
            select(EvaluationJob)
            .where(EvaluationJob.dataset_id == dataset_id)
            .order_by(EvaluationJob.created_at.desc())
            .limit(5)
        ).all()
    return templates.TemplateResponse(
        request=request,
        name="datasets/detail.html",
        context=_template_context(
            request,
            current_user,
            dataset=dataset,
            dataset_archived_at=_dataset_archived_at(dataset),
            recent_evaluations=recent_evaluations,
        ),
    )


@router.post("/datasets/{dataset_id}/archive")
async def archive_dataset(
    request: Request,
    dataset_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    return_to = await _archive_form_target(request, dataset_id)
    with session_scope(request.app.state.engine) as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.status not in _ARCHIVABLE_DATASET_STATES:
            raise HTTPException(status_code=409, detail="Dataset cannot be archived")
        active_job_id = session.scalar(
            select(EvaluationJob.id)
            .where(
                EvaluationJob.dataset_id == dataset_id,
                EvaluationJob.state.not_in(_TERMINAL_EVALUATION_STATES),
            )
            .limit(1)
        )
        if active_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail="Dataset has an active evaluation",
            )
        if not isinstance(dataset.inspection_json, dict):
            raise HTTPException(status_code=409, detail="Dataset metadata is invalid")
        metadata = dict(dataset.inspection_json)
        if _ARCHIVE_KEY in metadata:
            raise HTTPException(status_code=409, detail="Dataset metadata is invalid")
        metadata[_ARCHIVE_KEY] = {
            "previous_status": dataset.status,
            "archived_at": datetime.now(UTC).isoformat(),
            "archived_by": current_user.id,
        }
        dataset.inspection_json = metadata
        dataset.status = "ARCHIVED"
    return RedirectResponse(return_to, status_code=303)


@router.post("/datasets/{dataset_id}/restore")
async def restore_dataset(
    request: Request,
    dataset_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    return_to = await _archive_form_target(request, dataset_id)
    with session_scope(request.app.state.engine) as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.status != "ARCHIVED" or not isinstance(dataset.inspection_json, dict):
            raise HTTPException(status_code=409, detail="Dataset cannot be restored")
        metadata = dict(dataset.inspection_json)
        snapshot = metadata.get(_ARCHIVE_KEY)
        if not _valid_archive_snapshot(snapshot):
            raise HTTPException(status_code=409, detail="Dataset archive metadata is invalid")
        metadata.pop(_ARCHIVE_KEY)
        dataset.inspection_json = metadata
        dataset.status = snapshot["previous_status"]
    return RedirectResponse(return_to, status_code=303)


@router.post("/datasets/{dataset_id}/attachments")
async def upload_attachment(
    request: Request,
    dataset_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    try:
        form = await _parse_attachment_form(request)
    except _AttachmentTooLarge as error:
        raise HTTPException(status_code=413, detail="Attachment size limit exceeded") from error
    except _AttachmentValidationError as error:
        raise HTTPException(status_code=422, detail="Invalid attachment") from error
    require_csrf(request, form.getlist("csrf_token"))
    dataset = _load_dataset(request, dataset_id)
    if set(form.keys()) != _ATTACHMENT_FIELDS:
        raise HTTPException(status_code=422, detail="Invalid attachment form")
    uploads = form.getlist("file")
    if len(uploads) != 1 or not isinstance(uploads[0], UploadFile):
        raise HTTPException(status_code=422, detail="Exactly one attachment is required")
    upload = uploads[0]
    try:
        filename = _validate_attachment_name(upload.filename)
        dataset_root = _resolve_dataset_root(request.app.state.config.data_root, dataset.path)
        await run_in_threadpool(_store_attachment, dataset_root, filename, upload.file)
    except _AttachmentConflict as error:
        raise HTTPException(status_code=409, detail="Attachment already exists") from error
    except _AttachmentTooLarge as error:
        raise HTTPException(status_code=413, detail="Attachment size limit exceeded") from error
    except _AttachmentValidationError as error:
        raise HTTPException(status_code=422, detail="Invalid attachment") from error
    finally:
        await upload.close()
    return RedirectResponse(f"/datasets/{dataset.id}", status_code=303)
