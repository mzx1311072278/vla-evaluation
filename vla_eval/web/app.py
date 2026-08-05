from pathlib import Path

import paramiko
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Engine
from starlette.middleware.sessions import SessionMiddleware

from vla_eval.config import AppConfig, require_session_secret
from vla_eval.queueing import QueueBundle
from vla_eval.web.routes_auth import router as auth_router
from vla_eval.web.routes_datasets import router as datasets_router
from vla_eval.web.routes_imports import router as imports_router


def create_app(config: AppConfig, engine: Engine, queues: QueueBundle) -> FastAPI:
    require_session_secret(config)
    app = FastAPI(title="VLA Evaluation")
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.session_secret,
        https_only=True,
        same_site="lax",
        max_age=43200,
    )
    app.state.config = config
    app.state.engine = engine
    app.state.queues = queues
    app.state.ssh_client_factory = paramiko.SSHClient
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )
    app.include_router(auth_router)
    app.include_router(datasets_router)
    app.include_router(imports_router)
    return app
