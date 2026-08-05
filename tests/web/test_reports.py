import csv
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from Genie02_report.genie02_eval_common import EPISODE_METRIC_FIELDS
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, User

_METRICS_SUCCESS = {
    "schema_version": "1.0",
    "session_id": "ready-dataset",
    "n_episodes": 1,
    "n_success": 1,
    "n_failure": 0,
    "gsr": 1.0,
    "mean_tts_success_s": 1.0,
    "smoothness": {
        "space": "joint",
        "left": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_episodes": 1},
        "right": {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n_episodes": 0,
        },
        "n_episodes": 1,
    },
}


def _write_episode_csv(output_dir: Path, rows: list[dict]) -> None:
    with (output_dir / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EPISODE_METRIC_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_succeeded_job(
    db_engine: Engine,
    dataset: Dataset,
    user: User,
    data_root: Path,
    *,
    slug: str,
    metrics: dict | None = None,
    episode_rows: list[dict] | None = None,
    provenance: dict | None = None,
) -> tuple[EvaluationJob, Path]:
    output_dir = data_root / "runs" / slug
    output_dir.mkdir(parents=True)
    payload = metrics if metrics is not None else _METRICS_SUCCESS
    (output_dir / "metrics_core.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_episode_csv(
        output_dir,
        episode_rows
        if episode_rows is not None
        else [
            {
                "session_id": "ready-dataset",
                "episode_index": 0,
                "outcome": "success",
                "duration_s": "1.000",
                "smoothness": "0",
                "left_smoothness": "0",
                "right_smoothness": "",
                "smoothness_space": "joint",
                "smoothness_frames": 4,
                "smoothness_skipped_reason": "",
            }
        ],
    )
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="SUCCEEDED",
            stage="REPORT",
            progress=100.0,
            output_dir=str(output_dir),
            provenance_json=provenance or {},
            created_by=user.id,
        )
        session.add(job)
        session.flush()
        return job, output_dir


def _attempt_row(episode_index: int, **overrides) -> dict:
    row = {
        "episode_index": episode_index,
        "metadata_episode_success": None,
        "episode_success": None,
        "pre_success_failed_attempt_count": 0,
        "failed_attempts_before_success": [],
        "attempt_count": 1,
        "success_count": 1,
        "failed_count": 0,
        "confidence": 0.9,
        "vlm_valid": True,
        "parse_error": "",
        "needs_manual_review": False,
        "review_note": "",
        "auto_warning": [],
        "review_mode": "auto_review",
        "reason": "",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "path",
    [
        "/reports/missing",
        "/reports/missing/files/metrics_core.json",
    ],
)
def test_report_routes_require_login(client: TestClient, path: str):
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_report_page_shows_core_metrics(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}")
    assert response.status_code == 200
    assert "GSR" in response.text
    # successful_job fixture has gsr=1.0 -> rendered as 100.0% (one decimal).
    assert "100.0%" in response.text
    # Headline counts.
    assert "成功数" in response.text
    # Episode row rendered with original outcome preserved.
    assert "success" in response.text
    assert "1.000" in response.text
    # Provenance from the job columns.
    assert "genie02-full" in response.text
    assert "1.0.0" in response.text


def test_report_page_404_when_job_has_no_output_dir(auth_client, evaluation_job):
    response = auth_client.get(f"/reports/{evaluation_job.id}")
    assert response.status_code == 404


def test_report_page_404_when_metrics_core_missing(
    auth_client, db_engine, ready_dataset, user, data_root
):
    job, output_dir = _make_succeeded_job(
        db_engine, ready_dataset, user, data_root, slug="no-metrics"
    )
    (output_dir / "metrics_core.json").unlink()
    response = auth_client.get(f"/reports/{job.id}")
    assert response.status_code == 404


def test_report_page_404_for_missing_job(auth_client):
    response = auth_client.get(f"/reports/{uuid.uuid4()}")
    assert response.status_code == 404


def test_download_serves_metrics_core(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}/files/metrics_core.json")
    assert response.status_code == 200
    expected = json.loads(
        (Path(successful_job.output_dir) / "metrics_core.json").read_text(encoding="utf-8")
    )
    assert json.loads(response.text) == expected
    assert "metrics_core.json" in response.headers["content-disposition"]


def test_download_serves_episode_metrics_csv(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}/files/episode_metrics.csv")
    assert response.status_code == 200
    expected = (Path(successful_job.output_dir) / "episode_metrics.csv").read_text(
        encoding="utf-8"
    )
    assert response.text == expected
    assert "episode_metrics.csv" in response.headers["content-disposition"]


def test_download_serves_report_markdown_glob(
    auth_client, db_engine, ready_dataset, user, data_root
):
    job, output_dir = _make_succeeded_job(
        db_engine, ready_dataset, user, data_root, slug="report-md"
    )
    (output_dir / "report_20240101.md").write_text("# report body", encoding="utf-8")
    response = auth_client.get(f"/reports/{job.id}/files/report_20240101.md")
    assert response.status_code == 200
    assert response.text == "# report body"
    assert "report_20240101.md" in response.headers["content-disposition"]


def test_download_rejects_path_escape(auth_client, successful_job):
    # Plain `..` is normalized by the router, but must still be rejected (404/422)
    # and never serve a file outside the job output directory.
    response = auth_client.get(
        f"/reports/{successful_job.id}/files/../../app.sqlite3",
        follow_redirects=False,
    )
    assert response.status_code in {404, 422}


def test_download_rejects_traversal_outside_output_dir(
    auth_client, successful_job, data_root
):
    # Place a whitelisted-NAMED file OUTSIDE the job output_dir. If containment
    # were broken, the traversal would serve this outside file. URL-encode the
    # `..` so the route genuinely receives a traversal path (Starlette normalizes
    # the plain form away before routing).
    outside = data_root / "metrics_core.json"
    outside.write_text("SECRET-OUTSIDE", encoding="utf-8")
    response = auth_client.get(
        f"/reports/{successful_job.id}/files/%2e%2e/%2e%2e/metrics_core.json",
        follow_redirects=False,
    )
    assert response.status_code in {404, 422}
    assert "SECRET-OUTSIDE" not in response.text


def test_download_rejects_non_whitelisted_filename(auth_client, successful_job):
    secret = Path(successful_job.output_dir) / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    response = auth_client.get(f"/reports/{successful_job.id}/files/secret.txt")
    assert response.status_code in {404, 422}
    assert "private" not in response.text


def test_download_rejects_non_whitelisted_sqlite_name(auth_client, successful_job):
    db_file = Path(successful_job.output_dir) / "app.sqlite3"
    db_file.write_text("db-bytes", encoding="utf-8")
    response = auth_client.get(f"/reports/{successful_job.id}/files/app.sqlite3")
    assert response.status_code in {404, 422}
    assert "db-bytes" not in response.text


def test_download_rejects_absolute_path(auth_client, successful_job):
    response = auth_client.get(
        f"/reports/{successful_job.id}/files//etc/passwd",
        follow_redirects=False,
    )
    assert response.status_code in {404, 422}


def test_download_404_for_missing_job(auth_client):
    response = auth_client.get(f"/reports/{uuid.uuid4()}/files/metrics_core.json")
    assert response.status_code == 404


def test_download_404_when_job_has_no_output_dir(auth_client, evaluation_job):
    response = auth_client.get(f"/reports/{evaluation_job.id}/files/metrics_core.json")
    assert response.status_code == 404


def test_download_404_for_missing_whitelisted_file(auth_client, successful_job):
    # smoothness_curve.svg is whitelisted but not produced by the fixture.
    response = auth_client.get(f"/reports/{successful_job.id}/files/smoothness_curve.svg")
    assert response.status_code == 404


def test_report_renders_vlm_review_alongside_original_outcome(
    auth_client, db_engine, ready_dataset, user, data_root
):
    metrics = {
        "schema_version": "1.0",
        "session_id": "ready-dataset",
        "n_episodes": 2,
        "n_success": 1,
        "n_failure": 1,
        "gsr": 0.5,
        "mean_tts_success_s": 1.0,
        "smoothness": {
            "space": "joint",
            "left": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_episodes": 2},
            "right": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_episodes": 0,
            },
            "n_episodes": 2,
        },
    }
    episode_rows = [
        {
            "session_id": "ready-dataset",
            "episode_index": 0,
            "outcome": "success",
            "duration_s": "1.000",
            "smoothness": "0",
            "left_smoothness": "0",
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        },
        {
            "session_id": "ready-dataset",
            "episode_index": 1,
            "outcome": "failure",
            "duration_s": "2.000",
            "smoothness": "0",
            "left_smoothness": "0",
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        },
    ]
    job, output_dir = _make_succeeded_job(
        db_engine,
        ready_dataset,
        user,
        data_root,
        slug="vlm-job",
        metrics=metrics,
        episode_rows=episode_rows,
    )
    attempt_dir = output_dir / "attempt_eval"
    attempt_dir.mkdir()
    (attempt_dir / "attempt_summary.json").write_text(
        json.dumps(
            [
                _attempt_row(0, needs_manual_review=False, reason=""),
                _attempt_row(
                    1,
                    needs_manual_review=True,
                    review_mode="manual_review",
                    reason="gripper did not close",
                    confidence=0.2,
                ),
            ]
        ),
        encoding="utf-8",
    )

    response = auth_client.get(f"/reports/{job.id}")
    assert response.status_code == 200
    # GSR rendered for gsr=0.5.
    assert "50.0%" in response.text
    # Both original outcomes preserved (VLM must not override them).
    assert "success" in response.text
    assert "failure" in response.text
    # VLM-derived review info shown alongside, clearly labeled, with the reason.
    assert "gripper did not close" in response.text
    assert "VLM" in response.text
    # 待复核 counts the single needs_manual_review row.
    assert "待复核" in response.text


def test_download_rejects_attempt_summary_outside_dir(auth_client, successful_job):
    # Non-existent whitelisted name resolves to 404 (no traversal either).
    response = auth_client.get(
        f"/reports/{successful_job.id}/files/attempt_summary.json"
    )
    assert response.status_code == 404
