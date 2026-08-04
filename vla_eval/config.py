import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SESSION_SECRET_ENV_VAR = "VLA_EVAL_SESSION_SECRET"
SESSION_SECRET_PLACEHOLDER = "${VLA_EVAL_SESSION_SECRET}"


@dataclass(frozen=True)
class RemoteSource:
    name: str
    host: str
    port: int
    username: str
    key_path: Path
    known_hosts_path: Path
    roots: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    database_url: str
    redis_url: str
    session_secret: str
    remote_sources: dict[str, RemoteSource]


def load_config(path: Path) -> AppConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data_root = Path(raw["data_root"]).expanduser().resolve()
    configured_secret = str(raw.get("session_secret") or "")
    environment_secret = os.environ.get(SESSION_SECRET_ENV_VAR, "")
    if environment_secret and configured_secret in ("", SESSION_SECRET_PLACEHOLDER):
        configured_secret = environment_secret
    sources = {
        name: RemoteSource(
            name=name,
            host=str(item["host"]),
            port=int(item.get("port", 22)),
            username=str(item["username"]),
            key_path=Path(item["key_path"]),
            known_hosts_path=Path(item["known_hosts_path"]),
            roots=tuple(str(value) for value in item["roots"]),
        )
        for name, item in (raw.get("remote_sources") or {}).items()
    }
    return AppConfig(
        data_root=data_root,
        database_url=str(raw.get("database_url", f"sqlite:///{data_root / 'db/app.sqlite3'}")),
        redis_url=str(raw.get("redis_url", "redis://redis:6379/0")),
        session_secret=configured_secret,
        remote_sources=sources,
    )


def require_session_secret(config: AppConfig) -> None:
    if not config.session_secret.strip() or config.session_secret == SESSION_SECRET_PLACEHOLDER:
        raise ValueError("session_secret must be set before server startup")


def resolve_local_dataset_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    allowed = root.resolve()
    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("path is outside allowed root")
    return candidate
