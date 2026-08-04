#!/usr/bin/env python3
"""Run the complete Genie02 B-side evaluation pipeline.

Each stage is implemented in its own module and communicates through the files
defined by ``Genie02 真机评测指标与行动计划``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

from genie02_episode_metrics import generate_episode_metrics
from genie02_eval_common import (
    EvaluationError,
    parse_session_args,
    prepare_output_dir,
    require_session_dir,
)
from genie02_markdown_report import generate_markdown_report
from genie02_metrics_core import generate_metrics_core


def generate_report(
    session_dir: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Run the three B-side tasks in their documented file-contract order."""
    session_dir = require_session_dir(session_dir)
    output_root = prepare_output_dir(output_dir)

    generate_episode_metrics(session_dir, output_root)
    metrics = generate_metrics_core(session_dir, output_root)
    generate_markdown_report(session_dir, output_root)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_session_args(argv, "Run all three Genie02 B-side evaluation tasks.")
    try:
        metrics = generate_report(args.session_dir, args.output_dir)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output_root = prepare_output_dir(args.output_dir)
    print(
        f"Generated B-side outputs for {metrics['session_id']}: "
        f"GSR={metrics['gsr']:.3f}, successes={metrics['n_success']}/"
        f"{metrics['n_episodes']}, "
        f"smoothness_n={metrics['smoothness']['n_episodes']}, "
        f"output_dir={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
