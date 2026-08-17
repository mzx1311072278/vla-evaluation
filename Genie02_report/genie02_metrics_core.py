#!/usr/bin/env python3
"""Aggregate Genie02 session-level core metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .genie02_eval_common import (
        SCHEMA_VERSION,
        EvaluationError,
        finite_float,
        load_episode_metrics,
        load_episodes,
        load_session,
        parse_session_args,
        prepare_output_dir,
        require_session_dir,
        write_json,
    )
else:
    from genie02_eval_common import (
        SCHEMA_VERSION,
        EvaluationError,
        finite_float,
        load_episode_metrics,
        load_episodes,
        load_session,
        parse_session_args,
        prepare_output_dir,
        require_session_dir,
        write_json,
    )


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    result: dict[str, Any] = {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "n_episodes": len(array),
    }
    if len(array):
        result.update(
            mean=float(np.mean(array)),
            std=float(np.std(array, ddof=0)),
            min=float(np.min(array)),
            max=float(np.max(array)),
        )
    return result


def _join_and_validate(
    episodes: Sequence[dict[str, str]],
    episode_metrics: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    metrics_by_index = {row["episode_index"]: row for row in episode_metrics}
    episode_indices = {int(row["episode_index"]) for row in episodes}
    metric_indices = set(metrics_by_index)
    if episode_indices != metric_indices:
        missing = sorted(episode_indices - metric_indices)
        extra = sorted(metric_indices - episode_indices)
        details = []
        if missing:
            details.append(f"missing episode indices {missing}")
        if extra:
            details.append(f"unexpected episode indices {extra}")
        raise EvaluationError("episode_metrics.csv does not match episodes.csv: " + "; ".join(details))

    joined: list[tuple[dict[str, str], dict[str, Any]]] = []
    for episode in episodes:
        index = int(episode["episode_index"])
        derived = metrics_by_index[index]
        outcome = episode["outcome"].strip().lower()
        if derived["outcome"] != outcome:
            raise EvaluationError(
                f"episode {index}: outcome differs between episodes.csv and episode_metrics.csv"
            )
        if derived["duration_s"] is not None and not math.isclose(
            derived["duration_s"],
            finite_float(episode["duration_s"], f"episode {index} duration_s"),
            abs_tol=0.0005,
        ):
            raise EvaluationError(
                f"episode {index}: duration differs between episodes.csv and episode_metrics.csv"
            )
        joined.append((episode, derived))
    return joined


def build_core_metrics(
    session: dict[str, Any],
    episodes: Sequence[dict[str, str]],
    episode_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the GSR, successful-TTS, and smoothness formulas."""
    joined = _join_and_validate(episodes, episode_metrics)
    successes = [pair for pair in joined if pair[0]["outcome"].strip().lower() == "success"]
    smooth_rows = [
        derived
        for _, derived in joined
        if derived["left_smoothness"] is not None
        or derived["right_smoothness"] is not None
    ]
    spaces = {row["smoothness_space"] for row in smooth_rows}
    if len(spaces) > 1:
        raise EvaluationError(
            "episode smoothness values use mixed coordinate spaces"
        )
    expected_space = "ee_xyz" if session["rollout_mode"] == "ee" else "joint"
    if spaces and next(iter(spaces)) != expected_space:
        raise EvaluationError(
            f"smoothness space does not match rollout_mode {session['rollout_mode']!r}"
        )

    smoothness: dict[str, Any] = {
        "space": next(iter(spaces)) if spaces else expected_space,
        "n_episodes": len(smooth_rows),
        "left": _summary(
            [
                row["left_smoothness"]
                for row in smooth_rows
                if row["left_smoothness"] is not None
            ]
        ),
        "right": _summary(
            [
                row["right_smoothness"]
                for row in smooth_rows
                if row["right_smoothness"] is not None
            ]
        ),
    }

    successful_durations = [
        finite_float(episode["duration_s"], "successful duration_s")
        for episode, _ in successes
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session["session_id"],
        "n_episodes": len(joined),
        "n_success": len(successes),
        "n_failure": len(joined) - len(successes),
        "gsr": len(successes) / len(joined) if joined else 0.0,
        "mean_tts_success_s": (
            float(np.mean(successful_durations)) if successful_durations else None
        ),
        "smoothness": smoothness,
    }


def generate_metrics_core(
    session_dir: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Read episodes plus derived rows and write metrics_core.json."""
    session_dir = require_session_dir(session_dir)
    output_root = prepare_output_dir(output_dir)
    session = load_session(session_dir)
    episodes = load_episodes(session_dir, session)
    episode_metrics = load_episode_metrics(output_root, session)
    metrics = build_core_metrics(session, episodes, episode_metrics)
    write_json(output_root / "metrics_core.json", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_session_args(
        argv, "Generate Genie02 metrics_core.json from episode CSV files."
    )
    try:
        metrics = generate_metrics_core(args.session_dir, args.output_dir)
    except EvaluationError as exc:
        print(f"error: {exc}")
        return 2
    output_root = prepare_output_dir(args.output_dir)
    print(
        f"Wrote {output_root / 'metrics_core.json'}: GSR={metrics['gsr']:.3f}, "
        f"successes={metrics['n_success']}/{metrics['n_episodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
