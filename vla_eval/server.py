"""ASGI application factory for uvicorn, wired from environment variables.

Run with::

    uvicorn vla_eval.server:create_app_from_env --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from vla_eval.config import load_config, require_session_secret
from vla_eval.db import create_engine_for_url, init_db
from vla_eval.queueing import create_queues
from vla_eval.web.app import create_app


def create_app_from_env() -> FastAPI:
    """Build the FastAPI app from $VLA_EVAL_CONFIG and related environment."""
    config_path = Path(os.environ["VLA_EVAL_CONFIG"])
    config = load_config(config_path)
    require_session_secret(config)
    engine = create_engine_for_url(config.database_url)
    init_db(engine)
    queues = create_queues(config.redis_url)
    return create_app(config, engine, queues)
