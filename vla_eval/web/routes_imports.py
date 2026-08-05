import os
import stat
import unicodedata
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Annotated

import paramiko
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, select

from vla_eval.db import session_scope
from vla_eval.models import ImportJob, User
from vla_eval.remote import normalize_remote_relative_path
from vla_eval.security import require_csrf, require_html_user
from vla_eval.tasks import run_import_task

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
_TERMINAL_STATES = frozenset({"READY", "FAILED", "CANCELLED"})
_IMPORT_FIELDS = frozenset({"csrf_token", "source_name", "root", "relative_path", "target_name"})
_TARGET_PUNCTUATION = frozenset(" ._-")
_IMPORT_PHASES = (
    ("CONNECTING", "\u8fde\u63a5"),
    ("TRANSFERRING", "\u4f20\u8f93"),
    ("VERIFYING", "\u9a8c\u8bc1"),
    ("PREFLIGHT", "\u9884检"),
    ("READY", "\u5b8c成"),
)


def _template_context(request: Request, current_user: User, **values):
    return {
        "current_user": current_user,
        "csrf_token": request.session["csrf_token"],
        **values,
    }


def _validate_target_name(value: str) -> str:
    if (
        not value
        or not value.strip()
        or value != value.strip()
        or value in {".", ".."}
        or len(value) > 255
        or len(os.fsencode(value)) > 255
        or unicodedata.normalize("NFC", value) != value
        or any(
            character not in _TARGET_PUNCTUATION
            and unicodedata.category(character)[0] not in {"L", "N"}
            for character in value
        )
    ):
        raise ValueError("invalid target name")
    return value


def _single_form_value(form, field: str) -> str:
    values = form.getlist(field)
    if len(values) != 1 or not isinstance(values[0], str):
        raise HTTPException(status_code=422, detail="Invalid import form")
    return values[0]


def _load_job(request: Request, job_id: str) -> ImportJob:
    with session_scope(request.app.state.engine) as session:
        job = session.get(ImportJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found")
    return job


@router.get("/imports")
def import_list(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    with session_scope(request.app.state.engine) as session:
        jobs = session.scalars(select(ImportJob).order_by(ImportJob.created_at.desc())).all()
    return templates.TemplateResponse(
        request=request,
        name="imports/index.html",
        context=_template_context(request, current_user, jobs=jobs),
    )


@router.get("/imports/new")
def import_new(
    request: Request,
    current_user: Annotated[User, Depends(require_html_user)],
):
    return templates.TemplateResponse(
        request=request,
        name="imports/new.html",
        context=_template_context(
            request,
            current_user,
            remote_sources=request.app.state.config.remote_sources,
        ),
    )


@router.post("/imports")
async def create_import(
    request: Request,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    form = await request.form()
    require_csrf(request, form.getlist("csrf_token"))
    if set(form.keys()) != _IMPORT_FIELDS:
        raise HTTPException(status_code=422, detail="Invalid import form")

    source_name = _single_form_value(form, "source_name")
    remote_root = _single_form_value(form, "root")
    remote_path = _single_form_value(form, "relative_path")
    target_name = _single_form_value(form, "target_name")
    source = request.app.state.config.remote_sources.get(source_name)
    if source is None or remote_root not in source.roots:
        raise HTTPException(status_code=422, detail="Invalid import source")
    try:
        remote_path = normalize_remote_relative_path(remote_path)
        target_name = _validate_target_name(target_name)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Invalid import path or target") from error

    with session_scope(request.app.state.engine) as session:
        job = ImportJob(
            source_name=source_name,
            remote_root=remote_root,
            remote_path=remote_path,
            target_name=target_name,
            state="QUEUED",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    try:
        request.app.state.queues.transfer.enqueue(run_import_task, job_id)
    except Exception as error:
        with session_scope(request.app.state.engine) as session:
            removed = session.execute(
                delete(ImportJob).where(
                    ImportJob.id == job_id,
                    ImportJob.state == "QUEUED",
                    ImportJob.execution_token.is_(None),
                )
            )
        if removed.rowcount == 1:
            raise HTTPException(status_code=503, detail="Import queue unavailable") from error
        return RedirectResponse(f"/imports/{job_id}", status_code=303)

    return RedirectResponse(f"/imports/{job_id}", status_code=303)


@router.get("/imports/{job_id}")
def import_detail(
    request: Request,
    job_id: str,
    current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request, job_id)
    phase_states = [state for state, _label in _IMPORT_PHASES]
    phase_index = phase_states.index(job.state) if job.state in phase_states else -1
    return templates.TemplateResponse(
        request=request,
        name="imports/detail.html",
        context=_template_context(
            request,
            current_user,
            job=job,
            import_phases=_IMPORT_PHASES,
            phase_index=phase_index,
        ),
    )


@router.get("/api/imports/{job_id}")
def import_status(
    request: Request,
    job_id: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    job = _load_job(request, job_id)
    finished = job.state in _TERMINAL_STATES
    response = JSONResponse(
        {
            "id": job.id,
            "state": job.state,
            "progress": job.progress,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "dataset_id": job.dataset_id,
            "finished": finished,
        }
    )
    if finished:
        response.headers["HX-Trigger"] = "job-finished"
    return response


@router.get("/api/remote-sources/{source_name}/directories")
def remote_directories(
    request: Request,
    source_name: str,
    _current_user: Annotated[User, Depends(require_html_user)],
):
    root_values = request.query_params.getlist("root")
    path_values = request.query_params.getlist("path")
    if len(root_values) != 1 or len(path_values) > 1:
        raise HTTPException(status_code=422, detail="Invalid remote source parameters")
    root = root_values[0]
    path = path_values[0] if path_values else ""
    source = request.app.state.config.remote_sources.get(source_name)
    if source is None or root not in source.roots:
        raise HTTPException(status_code=422, detail="Invalid remote source or root")
    if path:
        try:
            relative_path = normalize_remote_relative_path(path)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid remote path") from error
        directory_path = str(PurePosixPath(root) / relative_path)
    else:
        relative_path = ""
        directory_path = root

    client = None
    sftp = None
    try:
        client = request.app.state.ssh_client_factory()
        client.load_host_keys(str(source.known_hosts_path))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=source.host,
            port=source.port,
            username=source.username,
            key_filename=str(source.key_path),
            look_for_keys=False,
            allow_agent=False,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
        )
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(10)
        directories = sorted(
            entry.filename
            for entry in sftp.listdir_attr(directory_path)
            if isinstance(entry.filename, str)
            and entry.filename not in {".", ".."}
            and "/" not in entry.filename
            and "\\" not in entry.filename
            and isinstance(entry.st_mode, int)
            and stat.S_ISDIR(entry.st_mode)
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Remote directory listing is unavailable",
        ) from error
    finally:
        if sftp is not None:
            with suppress(Exception):
                sftp.close()
        if client is not None:
            with suppress(Exception):
                client.close()

    return {"directories": directories, "path": relative_path}
