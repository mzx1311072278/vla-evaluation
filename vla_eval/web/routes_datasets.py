import errno
import fcntl
import os
import re
import secrets
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from vla_eval.db import session_scope
from vla_eval.models import Dataset, User
from vla_eval.security import require_csrf, require_html_user

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
_ATTACHMENT_FIELDS = frozenset({"csrf_token", "file"})
_ALLOWED_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".csv"})
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_MAX_DATASET_ATTACHMENTS_BYTES = 100 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_FILE_TOO_LARGE_MESSAGE = "attachment-file-too-large"


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
    with session_scope(request.app.state.engine) as session:
        datasets = session.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return templates.TemplateResponse(
        request=request,
        name="datasets/index.html",
        context=_template_context(request, current_user, datasets=datasets),
    )


@router.get("/datasets/{dataset_id}")
def dataset_detail(
    request: Request,
    dataset_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    dataset = _load_dataset(request, dataset_id)
    return templates.TemplateResponse(
        request=request,
        name="datasets/detail.html",
        context=_template_context(request, current_user, dataset=dataset),
    )


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
