from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for name in ("inbox", "staging", "runs", "models", "db"):
        (root / name).mkdir(parents=True)
    return root
