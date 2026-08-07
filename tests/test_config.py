from pathlib import Path
from typing import Any

import pytest
import yaml

from vla_eval.config import load_config, require_session_secret, resolve_local_dataset_path


def _valid_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "data_root": str(tmp_path / "data"),
        "remote_sources": {
            "lab-a": {
                "host": "10.0.0.8",
                "port": 22,
                "username": "eval-read",
                "key_path": "/run/secrets/lab_a_key",
                "known_hosts_path": "/run/secrets/known_hosts",
                "roots": ["/data/rollouts"],
            }
        },
    }


def _write_config(tmp_path: Path, raw: Any) -> Path:
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_resolve_local_dataset_path_rejects_escape(tmp_path: Path):
    root = tmp_path / "inbox"
    root.mkdir()
    with pytest.raises(ValueError, match="outside allowed root"):
        resolve_local_dataset_path(root, "../secret")


def test_resolve_local_dataset_path_rejects_absolute_path_inside_root(tmp_path: Path):
    root = tmp_path / "inbox"
    root.mkdir()

    with pytest.raises(ValueError, match="relative"):
        resolve_local_dataset_path(root, str(root / "run-1"))


def test_load_config_parses_remote_source(tmp_path: Path):
    path = tmp_path / "app.yaml"
    path.write_text(
        "data_root: /srv/vla-eval/data\n"
        "remote_sources:\n"
        "  lab-a:\n"
        "    host: 10.0.0.8\n"
        "    port: 22\n"
        "    username: eval-read\n"
        "    key_path: /run/secrets/lab_a_key\n"
        "    known_hosts_path: /run/secrets/known_hosts\n"
        "    roots: [/data/rollouts]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.remote_sources["lab-a"].roots == ("/data/rollouts",)


def test_load_config_parses_local_source(tmp_path: Path):
    local_root = tmp_path / "datasets"
    raw = _valid_config(tmp_path)
    raw["local_sources"] = {"this-host": {"roots": [str(local_root)]}}

    config = load_config(_write_config(tmp_path, raw))

    assert config.local_sources["this-host"].name == "this-host"
    assert config.local_sources["this-host"].roots == (local_root.resolve(),)


@pytest.mark.parametrize("local_sources", [[], "this-host", 1])
def test_load_config_rejects_non_mapping_local_sources(
    tmp_path: Path, local_sources: Any
):
    raw = _valid_config(tmp_path)
    raw["local_sources"] = local_sources

    with pytest.raises(TypeError, match="local_sources.*mapping"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("roots", ["/data/rollouts", [], [""], [None], [7]])
def test_load_config_rejects_invalid_local_source_roots(tmp_path: Path, roots: Any):
    raw = _valid_config(tmp_path)
    raw["local_sources"] = {"this-host": {"roots": roots}}

    with pytest.raises(ValueError, match="local_sources.this-host.roots"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "root",
    [
        "data/rollouts",
        "/data/../secret",
        "/data/\nrollouts",
        "/data/\x7frollouts",
        "/data/\x85rollouts",
        "/data//rollouts",
        "/data/./rollouts",
        "/data/rollouts/",
        "//data/rollouts",
        "/data/rollouts ",
        "/",
    ],
)
def test_load_config_rejects_unsafe_or_non_normalized_local_root(
    tmp_path: Path, root: str
):
    raw = _valid_config(tmp_path)
    raw["local_sources"] = {"this-host": {"roots": [root]}}

    with pytest.raises(ValueError, match="local_sources.this-host.roots"):
        load_config(_write_config(tmp_path, raw))


def test_load_config_rejects_source_name_collision_across_transports(tmp_path: Path):
    raw = _valid_config(tmp_path)
    raw["local_sources"] = {
        "lab-a": {"roots": [str((tmp_path / "datasets").resolve())]}
    }

    with pytest.raises(ValueError, match="source name.*both"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("configured", ["", "${VLA_EVAL_SESSION_SECRET}"])
def test_load_config_uses_environment_secret_for_empty_or_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: str
):
    path = tmp_path / "app.yaml"
    path.write_text(
        f"data_root: {tmp_path}\nsession_secret: '{configured}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLA_EVAL_SESSION_SECRET", "runtime-secret")

    assert load_config(path).session_secret == "runtime-secret"


def test_require_session_secret_rejects_empty_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VLA_EVAL_SESSION_SECRET", raising=False)
    path = tmp_path / "app.yaml"
    path.write_text(f"data_root: {tmp_path}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="session_secret"):
        require_session_secret(load_config(path))


def test_load_config_substitutes_whitespace_wrapped_secret_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_config(
        tmp_path,
        {"data_root": str(tmp_path), "session_secret": "  ${VLA_EVAL_SESSION_SECRET}\t"},
    )
    monkeypatch.setenv("VLA_EVAL_SESSION_SECRET", "runtime-secret")

    assert load_config(path).session_secret == "runtime-secret"


def test_require_session_secret_rejects_whitespace_wrapped_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_config(
        tmp_path,
        {"data_root": str(tmp_path), "session_secret": "  ${VLA_EVAL_SESSION_SECRET}\t"},
    )
    monkeypatch.delenv("VLA_EVAL_SESSION_SECRET", raising=False)

    with pytest.raises(ValueError, match="session_secret"):
        require_session_secret(load_config(path))


@pytest.mark.parametrize("raw", [["not", "a", "mapping"], "scalar"])
def test_load_config_rejects_non_mapping_document(tmp_path: Path, raw: Any):
    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("data_root", [None, "", "  ", 7])
def test_load_config_rejects_invalid_data_root(tmp_path: Path, data_root: Any):
    raw = _valid_config(tmp_path)
    raw["data_root"] = data_root

    with pytest.raises(ValueError, match="data_root"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("session_secret", [7, True, [], {}])
def test_load_config_rejects_non_string_session_secret(tmp_path: Path, session_secret: Any):
    raw = _valid_config(tmp_path)
    raw["session_secret"] = session_secret

    with pytest.raises(ValueError, match="session_secret"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("remote_sources", [[], "lab-a", 1])
def test_load_config_rejects_non_mapping_remote_sources(tmp_path: Path, remote_sources: Any):
    raw = _valid_config(tmp_path)
    raw["remote_sources"] = remote_sources

    with pytest.raises(TypeError, match="remote_sources.*mapping"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("source_name", ["", 7])
def test_load_config_rejects_invalid_remote_source_name(tmp_path: Path, source_name: Any):
    raw = _valid_config(tmp_path)
    source = raw["remote_sources"].pop("lab-a")
    raw["remote_sources"][source_name] = source

    with pytest.raises(ValueError, match="source name"):
        load_config(_write_config(tmp_path, raw))


def test_load_config_rejects_whitespace_around_remote_source_name(tmp_path: Path):
    raw = _valid_config(tmp_path)
    source = raw["remote_sources"].pop("lab-a")
    raw["remote_sources"][" lab "] = source

    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        load_config(_write_config(tmp_path, raw))


def test_load_config_rejects_normalized_remote_source_name_collision(tmp_path: Path):
    raw = _valid_config(tmp_path)
    source = raw["remote_sources"].pop("lab-a")
    raw["remote_sources"] = {"lab": source, " lab ": source.copy()}

    with pytest.raises(ValueError, match="collision"):
        load_config(_write_config(tmp_path, raw))


def test_load_config_rejects_non_mapping_remote_source(tmp_path: Path):
    raw = _valid_config(tmp_path)
    raw["remote_sources"]["lab-a"] = "not-a-mapping"

    with pytest.raises(TypeError, match="remote_sources.lab-a.*mapping"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("host", ""),
        ("host", None),
        ("username", 7),
        ("key_path", "  "),
        ("known_hosts_path", None),
    ],
)
def test_load_config_rejects_invalid_remote_source_strings(
    tmp_path: Path, field_name: str, value: Any
):
    raw = _valid_config(tmp_path)
    raw["remote_sources"]["lab-a"][field_name] = value

    with pytest.raises(ValueError, match=field_name):
        load_config(_write_config(tmp_path, raw))


def test_load_config_rejects_missing_remote_source_field(tmp_path: Path):
    raw = _valid_config(tmp_path)
    del raw["remote_sources"]["lab-a"]["host"]

    with pytest.raises(ValueError, match="host"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("port", [True, "22", 0, 65536, None])
def test_load_config_rejects_invalid_remote_source_port(tmp_path: Path, port: Any):
    raw = _valid_config(tmp_path)
    raw["remote_sources"]["lab-a"]["port"] = port

    with pytest.raises(ValueError, match="port"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("roots", ["/data/rollouts", [], [""], [None], [7]])
def test_load_config_rejects_invalid_remote_source_roots(tmp_path: Path, roots: Any):
    raw = _valid_config(tmp_path)
    raw["remote_sources"]["lab-a"]["roots"] = roots

    with pytest.raises(ValueError, match="roots"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "root",
    [
        "data/rollouts",
        "/data/../secret",
        "/data/\nrollouts",
        "/data/\x7frollouts",
        "/data/\x85rollouts",
        "/data//rollouts",
        "/data/./rollouts",
        "/data/rollouts/",
        "//data/rollouts",
        "/data/rollouts ",
    ],
)
def test_load_config_rejects_unsafe_or_non_normalized_remote_root(tmp_path: Path, root: str):
    raw = _valid_config(tmp_path)
    raw["remote_sources"]["lab-a"]["roots"] = [root]

    with pytest.raises(ValueError, match="roots"):
        load_config(_write_config(tmp_path, raw))


def test_load_config_uses_url_defaults_for_null_values(tmp_path: Path):
    raw = _valid_config(tmp_path)
    raw.update(database_url=None, redis_url=None)

    config = load_config(_write_config(tmp_path, raw))

    assert config.database_url == f"sqlite:///{(tmp_path / 'data').resolve() / 'db/app.sqlite3'}"
    assert config.redis_url == "redis://redis:6379/0"


@pytest.mark.parametrize("field_name", ["database_url", "redis_url"])
@pytest.mark.parametrize("value", [7, True, [], {}])
def test_load_config_rejects_non_string_url(tmp_path: Path, field_name: str, value: Any):
    raw = _valid_config(tmp_path)
    raw[field_name] = value

    with pytest.raises(TypeError, match=field_name):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("field_name", ["database_url", "redis_url"])
@pytest.mark.parametrize("value", ["", " \t "])
def test_load_config_rejects_blank_url(tmp_path: Path, field_name: str, value: str):
    raw = _valid_config(tmp_path)
    raw[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        load_config(_write_config(tmp_path, raw))


def test_remote_sources_mapping_is_immutable(tmp_path: Path):
    config = load_config(_write_config(tmp_path, _valid_config(tmp_path)))

    with pytest.raises(TypeError):
        config.remote_sources["lab-b"] = config.remote_sources["lab-a"]


def test_app_config_repr_hides_session_secret(tmp_path: Path):
    raw = _valid_config(tmp_path)
    raw["session_secret"] = "do-not-log-this-secret"

    assert "do-not-log-this-secret" not in repr(load_config(_write_config(tmp_path, raw)))
