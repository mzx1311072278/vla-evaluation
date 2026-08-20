#!/usr/bin/env python3
"""Build a human-readable symlink index for evaluation run directories.

Reads evaluation_jobs + datasets from the app database and (re)creates
data/runs-index/<readable-name> -> ../runs/<uuid> symlinks. The canonical
runs/<uuid> layout is untouched; the index is a disposable view that can
be rebuilt at any time.

Usage:
    VLA_EVAL_CONFIG=/path/to/app.yaml python deploy/runs_index.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_STATE_ABBR = {
    "SUCCEEDED": "OK",
    "FAILED": "FAIL",
    "CANCELLED": "CANCEL",
    "RUNNING": "RUN",
    "QUEUED": "QUEUE",
    "PENDING": "PEND",
}


def _sanitize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = _SAFE.sub("-", value).strip("-")
    return value or "unnamed"


def _load_config(path: Path) -> tuple[Path, Path]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_root = Path(raw["data_root"])
    database_url = raw["database_url"]
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise SystemExit(f"only sqlite databases are supported, got: {database_url}")
    db_path = Path(database_url[len(prefix) :])
    return data_root, db_path


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/app.yaml")
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path} (pass path as argument)")
    data_root, db_path = _load_config(config_path)
    runs_root = data_root / "runs"
    index_root = data_root / "runs-index"

    if not db_path.is_file():
        raise SystemExit(f"database not found: {db_path}")
    if not runs_root.is_dir():
        raise SystemExit(f"runs root not found: {runs_root}")

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT j.id, j.state, j.created_at, d.name AS dataset, j.profile_name
        FROM evaluation_jobs j JOIN datasets d ON d.id = j.dataset_id
        ORDER BY j.created_at
        """
    ).fetchall()
    connection.close()

    index_root.mkdir(parents=True, exist_ok=True)
    for stale in index_root.iterdir():
        if stale.is_symlink():
            stale.unlink()

    used: set[str] = set()
    created = 0
    for row in rows:
        target = runs_root / row["id"]
        if not target.is_dir():
            continue
        stamp = (row["created_at"] or "").replace("-", "").replace(":", "").replace(" ", "-")
        stamp = stamp[:15]
        state = _STATE_ABBR.get(row["state"], _sanitize(row["state"])[:8])
        name = "_".join(
            part
            for part in (
                stamp,
                _sanitize(row["dataset"] or "dataset"),
                _sanitize(row["profile_name"] or "profile"),
                state,
                row["id"][:8],
            )
            if part
        )
        while name in used:
            name = f"{name}-{row['id'][:12]}"
        used.add(name)
        (index_root / name).symlink_to(f"../runs/{row['id']}")
        created += 1

    print(f"索引目录: {index_root}  （链接 {created} 个，指向 runs/<uuid>）")
    for entry in sorted(index_root.iterdir()):
        print(f"  {entry.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
