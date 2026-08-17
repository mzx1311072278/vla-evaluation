import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import yaml

SESSION_SECRET_ENV_VAR = "VLA_EVAL_SESSION_SECRET"
SESSION_SECRET_PLACEHOLDER = "${VLA_EVAL_SESSION_SECRET}"
StorageTrustMode = Literal["strict", "data_root_boundary"]
STORAGE_TRUST_MODES = frozenset({"strict", "data_root_boundary"})


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
class LocalSource:
    name: str
    roots: tuple[Path, ...]


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    database_url: str
    redis_url: str
    session_secret: str = field(repr=False)
    remote_sources: Mapping[str, RemoteSource]
    local_sources: Mapping[str, LocalSource]
    storage_trust_mode: StorageTrustMode = "strict"

    def __post_init__(self) -> None:
        object.__setattr__(self, "remote_sources", MappingProxyType(dict(self.remote_sources)))
        object.__setattr__(self, "local_sources", MappingProxyType(dict(self.local_sources)))
        object.__setattr__(
            self,
            "storage_trust_mode",
            _storage_trust_mode(self.storage_trust_mode),
        )


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


def _storage_trust_mode(value: Any) -> StorageTrustMode:
    if not isinstance(value, str):
        raise TypeError("storage_trust_mode must be a string")
    if not value.strip():
        raise ValueError("storage_trust_mode must not be blank")
    if value not in STORAGE_TRUST_MODES:
        raise ValueError(
            "storage_trust_mode must be 'strict' or 'data_root_boundary'"
        )
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


def _local_root(value: Any, field_name: str) -> Path:
    root = _nonempty_string(value, field_name)
    if root != value:
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")
    if any(unicodedata.category(character) == "Cc" for character in root):
        raise ValueError(f"{field_name} must not contain control characters")

    path = Path(root)
    if (
        not path.is_absolute()
        or root.startswith("//")
        or ".." in path.parts
        or path == Path(path.anchor)
        or str(path) != root
    ):
        raise ValueError(f"{field_name} must be a normalized absolute path below filesystem root")
    return path.resolve(strict=False)


def _load_local_sources(value: Any) -> dict[str, LocalSource]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("local_sources must be a mapping")

    sources: dict[str, LocalSource] = {}
    for name, item in value.items():
        source_name = _nonempty_string(name, "local source name")
        if source_name in sources:
            raise ValueError(f"local source name collision after normalization: {source_name!r}")
        if source_name != name:
            raise ValueError("local source name must not have leading or trailing whitespace")
        field_prefix = f"local_sources.{source_name}"
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_prefix} must be a mapping")
        roots = item.get("roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError(f"{field_prefix}.roots must be a nonempty list of strings")
        normalized_roots = tuple(_local_root(root, f"{field_prefix}.roots") for root in roots)
        if len(set(normalized_roots)) != len(normalized_roots):
            raise ValueError(f"{field_prefix}.roots contains duplicate normalized paths")
        sources[source_name] = LocalSource(name=source_name, roots=normalized_roots)
    return sources


def load_config(path: Path) -> AppConfig:
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(loaded, Mapping):
        raw = loaded
    else:
        raise ValueError("configuration must be a top-level mapping")

    configured_data_root = Path(
        _nonempty_string(raw.get("data_root"), "data_root")
    ).expanduser()
    storage_trust_mode = _storage_trust_mode(
        raw.get("storage_trust_mode", "strict")
    )
    if storage_trust_mode == "data_root_boundary":
        if not configured_data_root.is_absolute():
            raise ValueError("data_root must be absolute in data_root_boundary mode")
        try:
            os.lstat(configured_data_root)
        except OSError as error:
            raise ValueError(
                "data_root must be an existing directory in data_root_boundary mode"
            ) from error
        if not configured_data_root.is_dir() or os.path.islink(configured_data_root):
            raise ValueError(
                "data_root must be an existing non-symlink directory in "
                "data_root_boundary mode"
            )
    data_root = configured_data_root.resolve()
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
    remote_sources = _load_remote_sources(raw.get("remote_sources"))
    local_sources = _load_local_sources(raw.get("local_sources"))
    collisions = sorted(set(remote_sources) & set(local_sources))
    if collisions:
        raise ValueError(
            "source name cannot be configured as both local and remote: "
            + ", ".join(collisions)
        )
    return AppConfig(
        data_root=data_root,
        database_url=_optional_string(raw, "database_url", default_database_url),
        redis_url=_optional_string(raw, "redis_url", "redis://redis:6379/0"),
        session_secret=configured_secret,
        remote_sources=remote_sources,
        local_sources=local_sources,
        storage_trust_mode=storage_trust_mode,
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
