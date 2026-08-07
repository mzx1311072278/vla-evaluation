"""Tests for the public `/health` endpoint."""

import os
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from vla_eval.config import AppConfig
from vla_eval.db import create_engine_for_url, init_db
from vla_eval.queueing import create_queues
from vla_eval.web.app import create_app


@pytest.fixture
def health_app(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    fake_redis = fakeredis.FakeRedis()
    queues = create_queues("redis://unused", connection=fake_redis)
    config = AppConfig(
        data_root=data_root,
        database_url="sqlite://",
        redis_url="redis://unused",
        session_secret="hush-secret-value",
        remote_sources={},
        local_sources={},
    )
    app = create_app(config, engine, queues)
    return {
        "app": app,
        "redis": fake_redis,
        "engine": engine,
        "config": config,
        "data_root": data_root,
    }


def _get(client_app):
    app = client_app["app"]
    with TestClient(app, base_url="https://testserver") as client:
        return client.get("/health")


def test_health_ok(health_app):
    response = _get(health_app)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "hush-secret-value" not in response.text


def test_health_redis_failure(health_app, monkeypatch):
    def _raise():
        raise OSError("redis down")

    monkeypatch.setattr(health_app["redis"], "ping", _raise)
    response = _get(health_app)
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "component": "redis"}
    assert "hush-secret-value" not in response.text


def test_health_sqlite_failure(health_app, monkeypatch):
    def _raise():
        raise RuntimeError("database down")

    monkeypatch.setattr(health_app["engine"], "connect", _raise)
    response = _get(health_app)
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "component": "sqlite"}
    assert "hush-secret-value" not in response.text


def test_health_data_root_failure(health_app):
    data_root: Path = health_app["data_root"]
    os.chmod(data_root, 0o555)
    try:
        response = _get(health_app)
    finally:
        os.chmod(data_root, 0o755)
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "component": "data_root"}
    assert "hush-secret-value" not in response.text
