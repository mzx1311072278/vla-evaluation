"""Public `/health` readiness probe used by Docker healthchecks and the reverse proxy."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vla_eval.readiness import collect_readiness_failures

router = APIRouter()


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Check SQLite reachability, Redis connectivity, and data-root writability.

    Returns 200 on success. Any failing component yields 503 with the first
    failing component name only; no credentials, secrets, or absolute key paths
    are ever exposed.
    """
    failures = collect_readiness_failures(
        request.app.state.engine,
        request.app.state.queues,
        request.app.state.config.data_root,
    )
    if not failures:
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "component": failures[0]},
    )
