import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from Genie02_report.genie02_eval_report import generate_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(
    "Genie02_report/zqyh_2cm_mixed_ee_rot6_right_arm_only_eval_pi05_stage2_acp"
)


def test_metric_definitions_match_implemented_formulas():
    from Genie02_report.metric_definitions import (
        METRIC_DEFINITIONS,
        markdown_formula_lines,
    )

    assert [metric.key for metric in METRIC_DEFINITIONS] == [
        "gsr",
        "tts_success",
        "smoothness",
    ]
    formulas = "\n".join(markdown_formula_lines(METRIC_DEFINITIONS))
    assert "GSR = N_success / N_total" in formulas
    assert "TTS = mean(duration_s | outcome = success)" in formulas
    assert "S = log10(E + 1)" in formulas
    assert "E = sum(||j_k||^2) * delta_t" in formulas
    assert (
        "j_k = (x_k - 3 x_(k-1) + 3 x_(k-2) - x_(k-3)) / delta_t^3"
        in formulas
    )


@pytest.fixture
def minimal_native_session(tmp_path):
    session_dir = tmp_path / "native_session"
    trajectory_dir = session_dir / "trajectories"
    trajectory_dir.mkdir(parents=True)
    session = {
        "schema_version": "1.0",
        "session_id": "native-fixture",
        "created_at": "2026-01-02T03:04:05+08:00",
        "status": "completed",
        "rollout_config_path": "rollout.yaml",
        "rollout_mode": "default",
        "policy_path": "policy",
        "task": "deterministic fixture",
        "num_episodes_target": 2,
        "fps": 10,
        "dataset_backend": "native",
        "dataset_root": "unused",
        "date": "2026-01-02",
    }
    (session_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = (
        "session_id",
        "episode_index",
        "episode_path",
        "trajectory_path",
        "t_start",
        "t_end",
        "duration_s",
        "outcome",
        "operator_intervened",
        "notes",
    )
    rows = [
        {
            "session_id": "native-fixture",
            "episode_index": "0",
            "episode_path": "",
            "trajectory_path": "trajectories/episode_000.npz",
            "t_start": "0",
            "t_end": "2",
            "duration_s": "2",
            "outcome": "success",
            "operator_intervened": "false",
            "notes": "",
        },
        {
            "session_id": "native-fixture",
            "episode_index": "1",
            "episode_path": "",
            "trajectory_path": "trajectories/episode_001.npz",
            "t_start": "0",
            "t_end": "3",
            "duration_s": "3",
            "outcome": "failure",
            "operator_intervened": "false",
            "notes": "fixture failure",
        },
    ]
    with (session_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    values = np.arange(5, dtype=float)[:, None]
    np.savez(trajectory_dir / "episode_000.npz", action=values)
    np.savez(trajectory_dir / "episode_001.npz", action=values * 2)
    return session_dir


def test_minimal_native_session_generates_complete_report_in_output_dir(
    minimal_native_session, tmp_path, monkeypatch
):
    output_dir = tmp_path / "explicit-output"
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    source_files_before = {
        path.relative_to(minimal_native_session): path.read_bytes()
        for path in minimal_native_session.rglob("*")
        if path.is_file()
    }
    repository_artifacts_before = {
        path
        for pattern in (
            "episode_metrics.csv",
            "metrics_core.json",
            "smoothness_curve.svg",
            "report_*.md",
        )
        for path in PROJECT_ROOT.glob(pattern)
    }
    monkeypatch.chdir(unrelated_cwd)

    actual = generate_report(minimal_native_session, output_dir)

    assert set(actual) == {
        "schema_version",
        "session_id",
        "n_episodes",
        "n_success",
        "n_failure",
        "gsr",
        "mean_tts_success_s",
        "smoothness",
    }
    assert actual["schema_version"] == "1.0"
    assert actual["session_id"] == "native-fixture"
    assert actual["n_episodes"] == 2
    assert actual["n_success"] == 1
    assert actual["n_failure"] == 1
    assert actual["gsr"] == pytest.approx(0.5)
    assert actual["mean_tts_success_s"] == pytest.approx(2.0)
    assert actual["smoothness"] == {
        "space": "joint",
        "n_episodes": 2,
        "left": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_episodes": 2},
        "right": {"mean": None, "std": None, "min": None, "max": None, "n_episodes": 0},
    }

    report_files = list(output_dir.glob("report_*.md"))
    assert len(report_files) == 1
    assert {path.name for path in output_dir.iterdir()} == {
        "episode_metrics.csv",
        "metrics_core.json",
        "smoothness_curve.svg",
        report_files[0].name,
    }
    persisted_metrics = json.loads((output_dir / "metrics_core.json").read_text())
    assert persisted_metrics == actual
    with (output_dir / "episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
        episode_metrics = list(csv.DictReader(handle))
    assert [(row["outcome"], row["duration_s"]) for row in episode_metrics] == [
        ("success", "2.000"),
        ("failure", "3.000"),
    ]
    assert [row["smoothness"] for row in episode_metrics] == ["0", "0"]
    assert [row["smoothness_frames"] for row in episode_metrics] == ["5", "5"]
    assert all(not row["smoothness_skipped_reason"] for row in episode_metrics)
    assert "| GSR | 50.0% | 2 |" in report_files[0].read_text(encoding="utf-8")
    assert "Episode 平滑度概览" in (output_dir / "smoothness_curve.svg").read_text(
        encoding="utf-8"
    )

    source_files_after = {
        path.relative_to(minimal_native_session): path.read_bytes()
        for path in minimal_native_session.rglob("*")
        if path.is_file()
    }
    assert source_files_after == source_files_before
    assert not list(unrelated_cwd.iterdir())
    repository_artifacts_after = {
        path
        for pattern in (
            "episode_metrics.csv",
            "metrics_core.json",
            "smoothness_curve.svg",
            "report_*.md",
        )
        for path in PROJECT_ROOT.glob(pattern)
    }
    assert repository_artifacts_after == repository_artifacts_before


def test_markdown_report_uses_shared_metric_wording(
    minimal_native_session, tmp_path, monkeypatch
):
    from Genie02_report import genie02_markdown_report

    monkeypatch.setattr(
        genie02_markdown_report,
        "metric_definition_rows",
        lambda: [("sentinel", "shared-definition", "smaller")],
    )
    output_dir = tmp_path / "shared-definitions"
    generate_report(minimal_native_session, output_dir)
    report_path = next(output_dir.glob("report_*.md"))

    assert "shared-definition" in report_path.read_text(encoding="utf-8")


@pytest.mark.skipif(not SAMPLE.exists(), reason="large local sample is not installed")
def test_existing_lerobot_sample_matches_committed_metrics(tmp_path):
    expected = json.loads(
        Path("Genie02_report/report_20260708/metrics_core.json").read_text()
    )
    actual = generate_report(SAMPLE, tmp_path)
    assert actual["n_episodes"] == expected["n_episodes"]
    assert actual["n_success"] == expected["n_success"]
    assert actual["gsr"] == pytest.approx(expected["gsr"])


@pytest.mark.parametrize(
    ("command", "expected_options"),
    [
        pytest.param(
            (
                str(
                    PROJECT_ROOT
                    / "Genie02_report/attempt_eval/run_episode_attempt_eval.py"
                ),
                "--help",
            ),
            ("--dataset_root", "--model_path"),
            id="attempt-direct-script",
        ),
        pytest.param(
            ("-m", "Genie02_report.attempt_eval.run_episode_attempt_eval", "--help"),
            ("--dataset_root", "--model_path"),
            id="attempt-package-module",
        ),
        pytest.param(
            ("-m", "Genie02_report.genie02_eval_report", "--help"),
            ("session_dir", "--output-dir"),
            id="report-package-module",
        ),
    ],
)
def test_cli_help_does_not_require_runtime_inputs(command, expected_options, tmp_path):
    result = subprocess.run(
        [sys.executable, *command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in expected_options)
