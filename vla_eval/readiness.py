"""Shared readiness probes used by the web `/health` endpoint and `cli smoke`.

Centralizing the three checks (SQLite ``SELECT 1``, Redis ``PING``, data-root
write/unlink) keeps the two call sites from drifting. Each probe is isolated so
one failing component cannot mask another, and only non-sensitive component
names are surfaced.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

from sqlalchemy import Engine, text

from vla_eval.queueing import QueueBundle


def collect_readiness_failures(
    engine: Engine, queues: QueueBundle, data_root: Path
) -> list[str]:
    """Run all readiness probes; return the names of failing components.

    An empty list means healthy. The temp file created for the write probe is
    always removed (its unlink is suppressed if it was never created).
    """
    failures: list[str] = []

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any DB failure marks the component
        failures.append("sqlite")

    try:
        queues.evaluation.connection.ping()
    except Exception:  # noqa: BLE001 - any Redis failure marks the component
        failures.append("redis")

    probe = data_root / f".readiness-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        probe.write_text("ok", encoding="utf-8")
    except Exception:  # noqa: BLE001 - any filesystem failure marks the component
        failures.append("data_root")
    finally:
        with contextlib.suppress(FileNotFoundError):
            probe.unlink()

    return failures
