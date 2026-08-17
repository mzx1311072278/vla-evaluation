"""Presentation-only summaries for persisted smoothness values."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median


def summarize_smoothness(values: Iterable[float]) -> dict[str, float]:
    """Return deterministic display statistics without changing metric values."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("smoothness values must not be empty")
    p90_index = max(0, math.ceil(len(ordered) * 0.9) - 1)
    return {
        "minimum": ordered[0],
        "median": float(median(ordered)),
        "p90": ordered[p90_index],
        "maximum": ordered[-1],
    }
