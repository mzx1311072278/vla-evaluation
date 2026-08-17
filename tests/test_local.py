from pathlib import Path

import pytest

from vla_eval.config import LocalSource
from vla_eval.local import build_local_rsync_argv, resolve_local_source_directory


def _source(root: Path) -> LocalSource:
    return LocalSource(name="this-host", roots=(root,))


def test_resolve_local_source_directory_accepts_configured_descendant(tmp_path: Path):
    root = tmp_path / "datasets"
    dataset = root / "team" / "run-01"
    dataset.mkdir(parents=True)

    resolved = resolve_local_source_directory(_source(root), str(root), "team/run-01")

    assert resolved == dataset


def test_resolve_local_source_directory_requires_exact_configured_root(tmp_path: Path):
    root = tmp_path / "datasets"
    (root / "run-01").mkdir(parents=True)

    with pytest.raises(ValueError, match="configured local root"):
        resolve_local_source_directory(_source(root), str(tmp_path), "datasets/run-01")


@pytest.mark.parametrize("relative_path", ["/run-01", "../run-01", "team//run-01"])
def test_resolve_local_source_directory_rejects_unsafe_relative_path(
    tmp_path: Path, relative_path: str
):
    root = tmp_path / "datasets"
    root.mkdir()

    with pytest.raises(ValueError, match="path"):
        resolve_local_source_directory(_source(root), str(root), relative_path)


def test_resolve_local_source_directory_rejects_missing_directory(tmp_path: Path):
    root = tmp_path / "datasets"
    root.mkdir()

    with pytest.raises(ValueError, match="existing directory"):
        resolve_local_source_directory(_source(root), str(root), "missing")


def test_resolve_local_source_directory_rejects_file(tmp_path: Path):
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "run-01").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="existing directory"):
        resolve_local_source_directory(_source(root), str(root), "run-01")


def test_resolve_local_source_directory_rejects_symlink_component(tmp_path: Path):
    root = tmp_path / "datasets"
    outside = tmp_path / "outside" / "run-01"
    outside.mkdir(parents=True)
    root.mkdir()
    (root / "linked").symlink_to(outside.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        resolve_local_source_directory(_source(root), str(root), "linked/run-01")


def test_build_local_rsync_argv_uses_separate_operands(tmp_path: Path):
    source = tmp_path / "source with spaces"
    staging = tmp_path / "staging with spaces"
    source.mkdir()
    staging.mkdir()

    argv = build_local_rsync_argv(source, staging)

    assert argv == [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "--",
        f"{source}/",
        f"{staging}/",
    ]
