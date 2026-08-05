"""Management CLI for the VLA evaluation service.

Invoke with ``python -m vla_eval.cli``. Commands share a ``--config`` option that
defaults to the ``VLA_EVAL_CONFIG`` environment variable. The CLI never prints
plaintext passwords or secret material.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rq import Worker
from sqlalchemy import Engine, select

from vla_eval.config import AppConfig, load_config
from vla_eval.datasets import DatasetInspection, inspect_dataset
from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.models import Dataset, User
from vla_eval.queueing import QueueBundle, create_queues
from vla_eval.readiness import collect_readiness_failures
from vla_eval.security import hash_password
from vla_eval.tasks import (
    TaskRuntime,
    clear_runtime,
    configure_runtime,
    recover_interrupted_jobs,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        envvar="VLA_EVAL_CONFIG",
        help="Path to the YAML config file (defaults to $VLA_EVAL_CONFIG).",
    ),
]

_INITIAL_PASSWORD_ENV = "VLA_EVAL_INITIAL_PASSWORD"
_WORKER_QUEUE_CHOICES = {"evaluations", "transfers"}


def _load_config(config_path: Path | None) -> AppConfig:
    if config_path is None:
        config_path = Path(os.environ["VLA_EVAL_CONFIG"])
    return load_config(config_path)


def _engine_for(config: AppConfig) -> Engine:
    engine = create_engine_for_url(config.database_url)
    init_db(engine)
    return engine


def _build_task_runtime(
    config: AppConfig, *, engine: Engine | None = None
) -> tuple[Engine, TaskRuntime]:
    resolved_engine = engine if engine is not None else _engine_for(config)
    profiles_root = Path(os.environ.get("VLA_EVAL_PROFILES_ROOT", "config/profiles"))
    credentials_root = Path(
        os.environ.get("VLA_EVAL_CREDENTIALS_ROOT", str(config.data_root / "credentials"))
    )
    runtime = TaskRuntime(
        engine=resolved_engine,
        config=config,
        profiles_root=profiles_root,
        credentials_root=credentials_root,
    )
    return resolved_engine, runtime


@app.command("init-db")
def init_db_cmd(config_path: ConfigOption = None) -> None:
    """Create database tables (idempotent)."""
    config = _load_config(config_path)
    engine = _engine_for(config)
    engine.dispose()
    typer.echo("database initialized")


@app.command("create-user")
def create_user_cmd(
    username: str,
    password: Annotated[
        str | None,
        typer.Option(
            "--password",
            help="Initial password. If omitted, read $VLA_EVAL_INITIAL_PASSWORD.",
        ),
    ] = None,
    admin: Annotated[
        bool, typer.Option("--admin", help="Grant administrator privileges.")
    ] = False,
    config_path: ConfigOption = None,
) -> None:
    """Create a login user with a hashed password."""
    config = _load_config(config_path)
    chosen = password if password is not None else os.environ.get(_INITIAL_PASSWORD_ENV)
    if not chosen:
        typer.echo(
            "error: provide --password or set VLA_EVAL_INITIAL_PASSWORD", err=True
        )
        raise typer.Exit(2)
    engine = _engine_for(config)
    try:
        with session_scope(engine) as session:
            existing = session.scalar(select(User).where(User.username == username))
            if existing is not None:
                typer.echo(f"error: user already exists: {username}", err=True)
                raise typer.Exit(1)
            session.add(
                User(
                    username=username,
                    password_hash=hash_password(chosen),
                    is_admin=admin,
                    active=True,
                )
            )
    finally:
        engine.dispose()
    typer.echo(f"created user '{username}'")


@app.command("disable-user")
def disable_user_cmd(username: str, config_path: ConfigOption = None) -> None:
    """Mark a user inactive."""
    config = _load_config(config_path)
    engine = _engine_for(config)
    try:
        with session_scope(engine) as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                typer.echo(f"error: user not found: {username}", err=True)
                raise typer.Exit(1)
            user.active = False
    finally:
        engine.dispose()
    typer.echo(f"disabled user '{username}'")


def _upsert_dataset(engine: Engine, entry: Path, inspection: DatasetInspection) -> None:
    fields = {
        "name": entry.name,
        "path": str(entry),
        "kind": inspection.kind.value if inspection.kind is not None else "unknown",
        "status": "READY",
        "fingerprint": inspection.fingerprint,
        "size_bytes": inspection.size_bytes,
        "episode_count": inspection.episode_count or 0,
        "inspection_json": {"errors": list(inspection.errors)},
    }
    with session_scope(engine) as session:
        existing = session.scalar(select(Dataset).where(Dataset.path == str(entry)))
        if existing is None:
            session.add(Dataset(**fields))
        else:
            for key, value in fields.items():
                setattr(existing, key, value)


@app.command("scan-datasets")
def scan_datasets_cmd(config_path: ConfigOption = None) -> None:
    """Inspect the inbox and register READY datasets."""
    config = _load_config(config_path)
    inbox = config.data_root / "inbox"
    entries = sorted(p for p in inbox.iterdir() if p.is_dir()) if inbox.is_dir() else []
    engine = _engine_for(config)
    ready = 0
    skipped = 0
    try:
        for entry in entries:
            inspection = inspect_dataset(entry, allowed_root=entry)
            if not inspection.ready or inspection.kind is None:
                skipped += 1
                continue
            _upsert_dataset(engine, entry, inspection)
            ready += 1
    finally:
        engine.dispose()
    typer.echo(f"scan complete: {ready} ready dataset(s), {skipped} skipped")


@app.command("recover-jobs")
def recover_jobs_cmd(config_path: ConfigOption = None) -> None:
    """Mark interrupted jobs so workers can reclaim them."""
    config = _load_config(config_path)
    engine, runtime = _build_task_runtime(config)
    try:
        configure_runtime(runtime)
        count = recover_interrupted_jobs()
    finally:
        clear_runtime()
        engine.dispose()
    typer.echo(f"recovered {count} interrupted job(s)")


@app.command("smoke")
def smoke_cmd(config_path: ConfigOption = None) -> None:
    """Verify the database, Redis, and data root are reachable/writable."""
    config = _load_config(config_path)
    engine = create_engine_for_url(config.database_url)
    queues = create_queues(config.redis_url)
    failures = collect_readiness_failures(engine, queues, config.data_root)
    engine.dispose()
    if failures:
        typer.echo(f"smoke failed: {', '.join(failures)}", err=True)
        raise typer.Exit(1)
    typer.echo("smoke ok")


def build_runtime(
    config_path: Path | None,
) -> tuple[Engine, QueueBundle, TaskRuntime]:
    """Build engine + queues + configured task runtime (used by workers)."""
    config = _load_config(config_path)
    engine = _engine_for(config)
    queues = create_queues(config.redis_url)
    _, runtime = _build_task_runtime(config, engine=engine)
    configure_runtime(runtime)
    recover_interrupted_jobs(runtime=runtime)
    return engine, queues, runtime


def run_worker(queue_name: str, config_path: Path | None) -> None:
    """Run an RQ worker on the chosen queue (one process, one queue)."""
    if queue_name not in _WORKER_QUEUE_CHOICES:
        typer.echo(
            f"error: unknown queue '{queue_name}'; choose evaluations or transfers",
            err=True,
        )
        raise typer.Exit(2)
    _engine, queues, _runtime = build_runtime(config_path)
    selected = queues.evaluation if queue_name == "evaluations" else queues.transfer
    Worker([selected], connection=selected.connection).work()


@app.command("worker")
def worker_cmd(
    queue: Annotated[
        str,
        typer.Option("--queue", help="Queue to consume: evaluations or transfers."),
    ],
    config_path: ConfigOption = None,
) -> None:
    """Run a background worker process."""
    run_worker(queue, config_path)


if __name__ == "__main__":
    app()
