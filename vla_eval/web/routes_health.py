"""Public `/health` readiness probe used by Docker healthchecks and the reverse proxy."""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter()


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Check SQLite reachability, Redis connectivity, and data-root writability.

    Returns 200 on success. Any failing component yields 503 with the component
    name only; no credentials, secrets, or absolute key paths are exposed.
    """
    engine = request.app.state.engine
    queues = request.app.state.queues
    data_root = request.app.state.config.data_root

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any DB failure is a degraded component
        return _degraded("sqlite")

    try:
        queues.evaluation.connection.ping()
    except Exception:  # noqa: BLE001 - any Redis failure is a degraded component
        return _degraded("redis")

    try:
        probe = data_root / f".health-{os.getpid()}-{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception:  # noqa: BLE001 - any filesystem failure is a degraded component
        return _degraded("data_root")

    return JSONResponse(status_code=200, content={"status": "ok"})


def _degraded(component: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "component": component},
    )
