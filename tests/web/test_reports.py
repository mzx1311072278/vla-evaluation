import csv
import json
import re
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
    assert 'class="report-page"' in response.text
    assert 'class="report-headline"' in response.text
    assert 'class="metric-success"' in response.text
    assert 'class="metric-failure"' in response.text
    assert 'class="detail-band report-overview"' in response.text
    assert 'class="key-value-table provenance-table"' in response.text
    assert 'class="report-filter-form"' in response.text
    assert 'class="button secondary-button"' in response.text
    assert 'class="section-heading"' in response.text
    assert "2 个文件" in response.text
    assert 'data-lucide="download"' in response.text
    assert 'class="table-scroll report-files"' in response.text
    assert '<th><span class="sr-only">操作</span></th>' in response.text
    assert 'class="table-action"' in response.text
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


def test_report_styles_expose_report_layout_and_mobile_rules(client: TestClient):
    response = client.get("/static/app.css")

    assert response.status_code == 200
    for selector in (
        ".report-headline",
        ".report-overview",
        ".smoothness-preview img",
        ".report-files-table",
        ".report-filter-form",
        ".table-action",
    ):
        assert selector in response.text
    assert re.search(
        r"@media \(max-width: 720px\) \{[\s\S]*?\.report-headline",
        response.text,
    )
    assert ".page-heading .button { width: 100%; }" in response.text
    assert ".report-page .page-heading .commands { width: 100%; }" in response.text


def test_report_page_shows_artifact_metadata_and_smoothness_preview(
    auth_client, successful_job
):
    output_dir = Path(successful_job.output_dir)
    (output_dir / "smoothness_curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8"
    )
    (output_dir / "attempt_summary.json").write_text("[]", encoding="utf-8")
    (output_dir / "attempt_summary.csv").write_text("episode_index\n", encoding="utf-8")
    (output_dir / "report_summary.md").write_text("# Report", encoding="utf-8")

    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    svg_url = f"/reports/{successful_job.id}/files/smoothness_curve.svg"
    assert f'<img src="{svg_url}"' in response.text
    assert f'href="{svg_url}"' in response.text
    assert 'class="smoothness-preview"' in response.text
    assert re.search(
        r'<figure class="smoothness-preview">\s*'
        r'<figcaption class="section-heading">',
        response.text,
    )
    for filename in (
        "episode_metrics.csv",
        "metrics_core.json",
        "smoothness_curve.svg",
        "attempt_summary.json",
        "attempt_summary.csv",
        "report_summary.md",
    ):
        assert f'href="/reports/{successful_job.id}/files/{filename}"' in response.text
    for description, file_format in (
        ("Episode 逐项指标", "CSV"),
        ("评测汇总指标", "JSON"),
        ("平滑度矢量图", "SVG"),
        ("VLM 尝试汇总", "JSON"),
        ("VLM 尝试明细", "CSV"),
        ("完整文本报告", "MD"),
    ):
        assert description in response.text
        assert f">{file_format}<" in response.text


def test_report_page_omits_smoothness_preview_without_svg(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert 'class="smoothness-preview"' not in response.text
    assert 'class="report-files-table"' in response.text


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
                    video_file="rollouts/episode_001.mp4",
                    from_timestamp=1.5,
                    to_timestamp=3.25,
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
    # Failed episode surfaces the evidence-frame locator from the attempt row
    # (video_file + timestamp range), alongside the VLM reason.
    assert "rollouts/episode_001.mp4" in response.text
    assert "1.500" in response.text
    assert "3.250" in response.text


def test_report_episode_filters(auth_client, db_engine, ready_dataset, user, data_root):
    metrics = {
        "schema_version": "1.0",
        "session_id": "ready-dataset",
        "n_episodes": 3,
        "n_success": 2,
        "n_failure": 1,
        "gsr": 0.667,
        "mean_tts_success_s": 1.0,
        "smoothness": {
            "space": "joint",
            "left": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_episodes": 3},
            "right": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_episodes": 0,
            },
            "n_episodes": 3,
        },
    }
    episode_rows = [
        {
            "session_id": "ready-dataset",
            "episode_index": 0,
            "outcome": "success",
            "duration_s": "1.100",
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
            "outcome": "success",
            "duration_s": "2.200",
            "smoothness": "0",
            "left_smoothness": "0",
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        },
        {
            "session_id": "ready-dataset",
            "episode_index": 2,
            "outcome": "failure",
            "duration_s": "3.300",
            "smoothness": "0",
            "left_smoothness": "0",
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        },
    ]
    job, _output_dir = _make_succeeded_job(
        db_engine,
        ready_dataset,
        user,
        data_root,
        slug="filter-job",
        metrics=metrics,
        episode_rows=episode_rows,
    )
    attempt_dir = Path(job.output_dir) / "attempt_eval"
    attempt_dir.mkdir()
    # Realistic per-mode values (review_policy.apply_review_policy):
    #   auto_review   -> needs_manual_review is True (warnings) or False (clean)
    #   manual_review -> needs_manual_review is None (always, human must decide)
    (attempt_dir / "attempt_summary.json").write_text(
        json.dumps(
            [
                _attempt_row(0, needs_manual_review=False, review_mode="auto_review"),
                _attempt_row(1, needs_manual_review=True, review_mode="auto_review"),
                _attempt_row(2, needs_manual_review=None, review_mode="manual_review"),
            ]
        ),
        encoding="utf-8",
    )

    base = f"/reports/{job.id}"

    # No filter: all three episodes shown.
    response = auth_client.get(base)
    assert response.status_code == 200
    assert "显示 3 / 3 个 episode" in response.text
    assert "1.100" in response.text
    assert "2.200" in response.text
    assert "3.300" in response.text

    # Filter by outcome=failure -> only the failed episode (duration 3.300).
    response = auth_client.get(f"{base}?outcome=failure")
    assert response.status_code == 200
    assert "显示 1 / 3 个 episode" in response.text
    assert "3.300" in response.text
    assert "1.100" not in response.text
    assert "2.200" not in response.text

    # Filter by outcome=success -> the two success episodes.
    response = auth_client.get(f"{base}?outcome=success")
    assert response.status_code == 200
    assert "显示 2 / 3 个 episode" in response.text
    assert "1.100" in response.text
    assert "2.200" in response.text
    assert "3.300" not in response.text

    # review=needs_review (needs_manual_review is True) -> episode 1 only.
    response = auth_client.get(f"{base}?review=needs_review")
    assert response.status_code == 200
    assert "显示 1 / 3 个 episode" in response.text
    assert "2.200" in response.text
    assert "1.100" not in response.text
    assert "3.300" not in response.text

    # review=reviewed (needs_manual_review is None, manual_review mode) -> ep 2.
    response = auth_client.get(f"{base}?review=reviewed")
    assert response.status_code == 200
    assert "显示 1 / 3 个 episode" in response.text
    assert "3.300" in response.text
    assert "1.100" not in response.text
    assert "2.200" not in response.text

    # review=ok (needs_manual_review is False, auto-clean) -> episode 0 only.
    response = auth_client.get(f"{base}?review=ok")
    assert response.status_code == 200
    assert "显示 1 / 3 个 episode" in response.text
    assert "1.100" in response.text
    assert "2.200" not in response.text
    assert "3.300" not in response.text

    # Combined outcome=success + review=needs_review -> only episode 1.
    response = auth_client.get(f"{base}?outcome=success&review=needs_review")
    assert response.status_code == 200
    assert "显示 1 / 3 个 episode" in response.text
    assert "2.200" in response.text
    assert "1.100" not in response.text

    # Invalid filter values are rejected with 422.
    assert auth_client.get(f"{base}?outcome=bogus").status_code == 422
    assert auth_client.get(f"{base}?review=bogus").status_code == 422
    # Duplicate filter params are rejected.
    assert (
        auth_client.get(f"{base}?outcome=success&outcome=failure").status_code == 422
    )


def test_report_review_filter_noop_without_vlm(auth_client, successful_job):
    # successful_job has no attempt_summary.json; a review filter must be a no-op
    # (200, all episodes still shown).
    response = auth_client.get(f"/reports/{successful_job.id}?review=needs_review")
    assert response.status_code == 200
    assert "显示 1 / 1 个 episode" in response.text


def test_download_rejects_attempt_summary_outside_dir(auth_client, successful_job):
    # Non-existent whitelisted name resolves to 404 (no traversal either).
    response = auth_client.get(
        f"/reports/{successful_job.id}/files/attempt_summary.json"
    )
    assert response.status_code == 404


def test_report_review_filter_manual_review_mode(
    auth_client, db_engine, ready_dataset, user, data_root
):
    """In manual_review mode needs_manual_review is always None, so the only
    non-empty review bucket is ``reviewed``; ``needs_review``/``ok`` are empty.
    """
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
            "duration_s": "1.100",
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
            "duration_s": "2.200",
            "smoothness": "0",
            "left_smoothness": "0",
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        },
    ]
    job, _output_dir = _make_succeeded_job(
        db_engine,
        ready_dataset,
        user,
        data_root,
        slug="manual-mode-job",
        metrics=metrics,
        episode_rows=episode_rows,
    )
    attempt_dir = Path(job.output_dir) / "attempt_eval"
    attempt_dir.mkdir()
    (attempt_dir / "attempt_summary.json").write_text(
        json.dumps(
            [
                _attempt_row(0, needs_manual_review=None, review_mode="manual_review"),
                _attempt_row(1, needs_manual_review=None, review_mode="manual_review"),
            ]
        ),
        encoding="utf-8",
    )
    base = f"/reports/{job.id}"

    # review=reviewed captures every manual_review-mode episode.
    response = auth_client.get(f"{base}?review=reviewed")
    assert response.status_code == 200
    assert "显示 2 / 2 个 episode" in response.text
    assert "1.100" in response.text
    assert "2.200" in response.text

    # needs_review and ok are empty in manual_review mode (no True/False values).
    response = auth_client.get(f"{base}?review=needs_review")
    assert response.status_code == 200
    assert "暂无 Episode 指标" in response.text

    response = auth_client.get(f"{base}?review=ok")
    assert response.status_code == 200
    assert "暂无 Episode 指标" in response.text


def test_report_page_tolerates_pathological_episode_csv(
    auth_client, successful_job, monkeypatch
):
    """A pathological episode_metrics.csv that raises csv.Error must degrade
    gracefully (200, empty episode table) rather than 500 — headline metrics
    are still rendered from metrics_core.json.
    """
    import csv as _csv

    from vla_eval.web import routes_reports

    class _ExplodingDictReader:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            raise _csv.Error("simulated pathological CSV")

    monkeypatch.setattr(routes_reports.csv, "DictReader", _ExplodingDictReader)
    response = auth_client.get(f"/reports/{successful_job.id}")
    assert response.status_code == 200
    # Headline metrics still rendered from metrics_core.json.
    assert "100.0%" in response.text
    # Episode table degrades to the empty state instead of a 500.
    assert "暂无 Episode 指标" in response.text
