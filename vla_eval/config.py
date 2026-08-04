import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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
    session_secret: str = field(repr=False)
    remote_sources: Mapping[str, RemoteSource]

    def __post_init__(self) -> None:
        object.__setattr__(self, "remote_sources", MappingProxyType(dict(self.remote_sources)))


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_string(raw: Mapping[str, Any], field_name: str, default: str) -> str:
    value = raw.get(field_name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or null")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _remote_root(value: Any, field_name: str) -> str:
    root = _nonempty_string(value, field_name)
    if root != value:
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")
    if any(unicodedata.category(character) == "Cc" for character in root):
        raise ValueError(f"{field_name} must not contain control characters")

    path = PurePosixPath(root)
    if not path.is_absolute() or root.startswith("//"):
        raise ValueError(f"{field_name} must be an absolute POSIX path")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain '..' components")
    if str(path) != root:
        raise ValueError(f"{field_name} must be a normalized POSIX path")
    return root


def _load_remote_sources(value: Any) -> dict[str, RemoteSource]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("remote_sources must be a mapping")

    sources: dict[str, RemoteSource] = {}
    for name, item in value.items():
        source_name = _nonempty_string(name, "remote source name")
        if source_name in sources:
            raise ValueError(f"remote source name collision after normalization: {source_name!r}")
        if source_name != name:
            raise ValueError("remote source name must not have leading or trailing whitespace")
        field_prefix = f"remote_sources.{source_name}"
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_prefix} must be a mapping")

        port = item.get("port", 22)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{field_prefix}.port must be an integer from 1 to 65535")

        roots = item.get("roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError(f"{field_prefix}.roots must be a nonempty list of strings")
        normalized_roots = tuple(_remote_root(root, f"{field_prefix}.roots") for root in roots)

        sources[source_name] = RemoteSource(
            name=source_name,
            host=_nonempty_string(item.get("host"), f"{field_prefix}.host"),
            port=port,
            username=_nonempty_string(item.get("username"), f"{field_prefix}.username"),
            key_path=Path(_nonempty_string(item.get("key_path"), f"{field_prefix}.key_path")),
            known_hosts_path=Path(
                _nonempty_string(item.get("known_hosts_path"), f"{field_prefix}.known_hosts_path")
            ),
            roots=normalized_roots,
        )
    return sources


def load_config(path: Path) -> AppConfig:
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(loaded, Mapping):
        raw = loaded
    else:
        raise ValueError("configuration must be a top-level mapping")

    data_root = Path(_nonempty_string(raw.get("data_root"), "data_root")).expanduser().resolve()
    secret_value = raw.get("session_secret")
    if secret_value is None:
        configured_secret = ""
    elif isinstance(secret_value, str):
        configured_secret = secret_value.strip()
    else:
        raise ValueError("session_secret must be a string or null")
    environment_secret = os.environ.get(SESSION_SECRET_ENV_VAR, "")
    if environment_secret and configured_secret in ("", SESSION_SECRET_PLACEHOLDER):
        configured_secret = environment_secret
    default_database_url = f"sqlite:///{data_root / 'db/app.sqlite3'}"
    return AppConfig(
        data_root=data_root,
        database_url=_optional_string(raw, "database_url", default_database_url),
        redis_url=_optional_string(raw, "redis_url", "redis://redis:6379/0"),
        session_secret=configured_secret,
        remote_sources=_load_remote_sources(raw.get("remote_sources")),
    )


def require_session_secret(config: AppConfig) -> None:
    normalized_secret = config.session_secret.strip()
    if not normalized_secret or normalized_secret == SESSION_SECRET_PLACEHOLDER:
        raise ValueError("session_secret must be set before server startup")


def resolve_local_dataset_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("path must be relative to allowed root")
    candidate = (root / relative).resolve()
    allowed = root.resolve()
    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("path is outside allowed root")
    return candidate
