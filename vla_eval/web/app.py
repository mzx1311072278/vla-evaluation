from fastapi import FastAPI
from sqlalchemy import Engine
from starlette.middleware.sessions import SessionMiddleware

from vla_eval.config import AppConfig, require_session_secret
from vla_eval.queueing import QueueBundle
from vla_eval.web.routes_auth import router as auth_router


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
    app.include_router(auth_router)
    return app
