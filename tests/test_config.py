from pathlib import Path

import pytest

from vla_eval.config import load_config, require_session_secret, resolve_local_dataset_path


def test_resolve_local_dataset_path_rejects_escape(tmp_path: Path):
    root = tmp_path / "inbox"
    root.mkdir()
    with pytest.raises(ValueError, match="outside allowed root"):
        resolve_local_dataset_path(root, "../secret")


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
