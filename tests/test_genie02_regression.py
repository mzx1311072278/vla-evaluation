import json
import subprocess
import sys
from pathlib import Path

import pytest

from Genie02_report.genie02_eval_report import generate_report

SAMPLE = Path(
    "Genie02_report/zqyh_2cm_mixed_ee_rot6_right_arm_only_eval_pi05_stage2_acp"
)


@pytest.mark.skipif(not SAMPLE.exists(), reason="large local sample is not installed")
def test_existing_lerobot_sample_matches_committed_metrics(tmp_path):
    expected = json.loads(
        Path("Genie02_report/report_20260708/metrics_core.json").read_text()
    )
    actual = generate_report(SAMPLE, tmp_path)
    assert actual["n_episodes"] == expected["n_episodes"]
    assert actual["n_success"] == expected["n_success"]
    assert actual["gsr"] == pytest.approx(expected["gsr"])


def test_attempt_eval_help_does_not_require_optional_runtime_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "Genie02_report/attempt_eval/run_episode_attempt_eval.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset_root" in result.stdout
    assert "--model_path" in result.stdout
