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
from vla_eval.models import Dataset, EvaluationJob, EvaluationJobArchive, User

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
    if episode_rows is not None:
        source_path = Path(dataset.path) / "episodes.csv"
        with source_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
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
                ),
            )
            writer.writeheader()
            for row in episode_rows:
                duration = row.get("duration_s", "")
                writer.writerow(
                    {
                        "session_id": row.get("session_id", "ready-dataset"),
                        "episode_index": row["episode_index"],
                        "episode_path": "",
                        "trajectory_path": (
                            f"trajectories/episode_{int(row['episode_index']):03d}.npz"
                        ),
                        "t_start": "0",
                        "t_end": duration,
                        "duration_s": duration,
                        "outcome": row["outcome"],
                        "operator_intervened": "false",
                        "notes": "",
                    }
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


def test_report_view_uses_current_persisted_evidence(successful_job, ready_dataset):
    from vla_eval.web.report_view import build_report_view

    view = build_report_view(
        job=successful_job,
        dataset=ready_dataset,
        output_dir=Path(successful_job.output_dir),
    )

    configuration = {row["label"]: row for row in view["configuration_rows"]}
    assert configuration["任务"]["value"] == "fixture"
    assert configuration["FPS"]["value"] == "10"
    assert configuration["数据后端"]["value"] == "native"
    assert configuration["数据指纹"]["value"] == ready_dataset.fingerprint
    assert view["headline"]["gsr"] == "100.0%"
    assert view["headline"]["n_success"] == 1
    assert view["headline"]["n_failure"] == 0
    assert view["summary_facts"]["episode_count"] == 1
    assert view["summary_facts"]["smoothness_coverage"] == 1
    assert view["summary_facts"]["vlm_executed"] is False
    assert view["episodes"] == [
        {
            "index": 0,
            "outcome": "success",
            "duration": "1.000",
            "smoothness": "0.000000",
            "left_smoothness": "0.000000",
            "right_smoothness": "—",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
            "operator_intervened": False,
            "notes": "",
            "vlm": None,
            "evidence_path": None,
            "evidence_range": None,
        }
    ]
    assert view["release_decision"] == "未配置自动发版判定"
    rendered = json.dumps(view, ensure_ascii=False, default=str)
    assert "30.8%" not in rendered
    assert "Ep 9" not in rendered
    assert "dc67326" not in rendered


def test_report_view_builds_smoothness_trend_and_worst_episode_rows(
    db_engine, ready_dataset, user, data_root
):
    values = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 10.0, 11.0, 0.5]
    rows = [
        {
            "session_id": "ready-dataset",
            "episode_index": index,
            "outcome": "failure" if index in {1, 10} else "success",
            "duration_s": "1.000",
            "smoothness": str(value),
            "left_smoothness": str(value),
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        }
        for index, value in enumerate(values)
    ]
    metrics = {
        **_METRICS_SUCCESS,
        "n_episodes": len(rows),
        "n_success": 10,
        "n_failure": 2,
        "gsr": 10 / 12,
        "smoothness": {
            **_METRICS_SUCCESS["smoothness"],
            "n_episodes": len(rows),
        },
    }
    job, output_dir = _make_succeeded_job(
        db_engine,
        ready_dataset,
        user,
        data_root,
        slug="smoothness-presentation",
        metrics=metrics,
        episode_rows=rows,
    )
    from vla_eval.web.report_view import build_report_view

    view = build_report_view(job=job, dataset=ready_dataset, output_dir=output_dir)

    chart = view["smoothness_chart"]
    assert len(chart["points"]) == 12
    assert chart["points"][1] == {
        "episode": 1,
        "value": 9.0,
        "outcome": "failure",
        "intervened": False,
    }
    assert chart["median"] == 5.5
    assert chart["p90"] == 10.0
    assert chart["minimum"] == 0.5
    assert chart["maximum"] == 11.0
    assert [row["episode"] for row in chart["worst"]] == [10, 9, 1, 3, 5, 7, 8, 6, 4, 2]


def test_report_page_renders_interactive_smoothness_diagnostics(
    auth_client, db_engine, ready_dataset, user, data_root
):
    row_count = 60
    rows = [
        {
            "session_id": "ready-dataset",
            "episode_index": index,
            "outcome": "failure" if index == 59 else "success",
            "duration_s": "1.000",
            "smoothness": str(index / 10),
            "left_smoothness": str(index / 10),
            "right_smoothness": "",
            "smoothness_space": "joint",
            "smoothness_frames": 4,
            "smoothness_skipped_reason": "",
        }
        for index in range(row_count)
    ]
    metrics = {
        **_METRICS_SUCCESS,
        "n_episodes": row_count,
        "n_success": row_count - 1,
        "n_failure": 1,
        "gsr": (row_count - 1) / row_count,
        "smoothness": {
            **_METRICS_SUCCESS["smoothness"],
            "n_episodes": row_count,
        },
    }
    job, output_dir = _make_succeeded_job(
        db_engine,
        ready_dataset,
        user,
        data_root,
        slug="interactive-smoothness",
        metrics=metrics,
        episode_rows=rows,
    )
    (output_dir / "smoothness_curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8"
    )

    response = auth_client.get(f"/reports/{job.id}")

    assert response.status_code == 200
    assert 'data-smoothness-chart' in response.text
    assert 'id="smoothness-chart-data"' in response.text
    assert 'data-chart-window' in response.text
    assert 'data-chart-start' in response.text
    assert 'class="smoothness-stats"' in response.text
    assert 'class="smoothness-outliers"' in response.text
    assert "最不平滑的 10 个 Episode" in response.text
    assert 'aria-label="下载平滑度矢量图"' in response.text
    assert 'data-chart-tooltip' in response.text
    assert 'class="smoothness-chart-shell table-scroll"' in response.text


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


def test_report_page_sections_follow_evidence_priority(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    section_ids = [
        "report-summary",
        "report-configuration",
        "report-sources",
        "report-quality",
        "report-metrics",
        "report-episodes",
        "report-components",
        "report-gaps",
        "report-downloads",
    ]
    positions = [response.text.index(f'id="{value}"') for value in section_ids]
    assert positions == sorted(positions)
    for table_class in (
        "configuration-table",
        "source-table",
        "quality-table",
        "metric-definition-table",
        "episode-table",
        "component-table",
        "evidence-gap-table",
        "report-files-table",
    ):
        assert f'class="{table_class}"' in response.text


def test_report_page_renders_shared_formulas_and_no_fake_release_decision(
    auth_client, successful_job
):
    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert 'class="formula-fraction"' in response.text
    assert "<sub>success</sub>" in response.text
    assert "<sub>total</sub>" in response.text
    assert "<sup>2</sup>" in response.text
    assert "<sup>3</sup>" in response.text
    assert "未配置自动发版判定" in response.text
    assert "建议暂缓生产发版" not in response.text
    assert "30.8%" not in response.text


def test_report_styles_expose_report_layout_and_mobile_rules(client: TestClient):
    response = client.get("/static/app.css")

    assert response.status_code == 200
    for selector in (
        ".report-headline",
        ".report-section",
        ".release-decision",
        ".report-overview",
        ".smoothness-preview img",
        ".metric-definition-table",
        ".formula-fraction",
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
    assert ".report-overview { grid-template-columns: minmax(0, 1fr); }" in response.text
    assert ".status-success { color: #176b53; }" in response.text
    assert ".status-failure { color: #a12b22; }" in response.text


def test_report_page_shows_artifact_metadata_and_smoothness_preview(
    auth_client, successful_job
):
    output_dir = Path(successful_job.output_dir)
    (output_dir / "smoothness_curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8"
    )
    attempt_dir = output_dir / "attempt_eval"
    attempt_dir.mkdir()
    (attempt_dir / "attempt_summary.json").write_text("[]", encoding="utf-8")
    (attempt_dir / "attempt_summary.csv").write_text(
        "episode_index\n", encoding="utf-8"
    )
    (output_dir / "report_summary.md").write_text("# Report", encoding="utf-8")

    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    svg_url = f"/reports/{successful_job.id}/files/smoothness_curve.svg"
    assert 'data-smoothness-chart' in response.text
    assert f'href="{svg_url}"' in response.text
    assert 'class="smoothness-preview"' in response.text
    assert 'aria-label="下载 metrics_core.json"' in response.text
    assert re.search(
        r'<figure class="smoothness-preview">\s*'
        r'<figcaption class="section-heading">',
        response.text,
    )
    for filename in (
        "episode_metrics.csv",
        "metrics_core.json",
        "smoothness_curve.svg",
        "attempt_eval/attempt_summary.json",
        "attempt_eval/attempt_summary.csv",
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


def test_archived_evaluation_preserves_report_and_every_available_download(
    auth_client, db_engine: Engine, successful_job
):
    output_dir = Path(successful_job.output_dir)
    (output_dir / "smoothness_curve.svg").write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
    )
    attempt_dir = output_dir / "attempt_eval"
    attempt_dir.mkdir()
    (attempt_dir / "attempt_summary.json").write_bytes(b"[]")
    (attempt_dir / "attempt_summary.csv").write_bytes(b"episode_index\n")
    (output_dir / "report_acceptance.md").write_bytes(b"# Full report\n")
    filenames = (
        "metrics_core.json",
        "episode_metrics.csv",
        "smoothness_curve.svg",
        "attempt_eval/attempt_summary.json",
        "attempt_eval/attempt_summary.csv",
        "report_acceptance.md",
    )
    report_url = f"/reports/{successful_job.id}"
    report_before = auth_client.get(report_url)
    downloads_before = {
        filename: auth_client.get(f"{report_url}/files/{filename}")
        for filename in filenames
    }

    archived = auth_client.post(
        f"/evaluations/{successful_job.id}/archive",
        data={
            "csrf_token": auth_client.csrf,
            "return_to": f"/evaluations/{successful_job.id}",
        },
        follow_redirects=False,
    )

    assert archived.status_code == 303
    report_after = auth_client.get(report_url)
    assert report_after.status_code == report_before.status_code == 200
    assert report_after.content == report_before.content
    for filename, before in downloads_before.items():
        after = auth_client.get(f"{report_url}/files/{filename}")
        assert before.status_code == after.status_code == 200
        assert after.content == before.content
        assert after.headers["content-disposition"] == before.headers["content-disposition"]
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, successful_job.id) is not None

    restored = auth_client.post(
        f"/evaluations/{successful_job.id}/restore",
        data={"csrf_token": auth_client.csrf, "return_to": report_url},
        follow_redirects=False,
    )

    assert restored.status_code == 303
    assert restored.headers["location"] == report_url
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, successful_job.id) is None


def test_report_page_keeps_interactive_smoothness_without_svg_download(
    auth_client, successful_job
):
    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert 'class="smoothness-preview"' in response.text
    assert 'data-smoothness-chart' in response.text
    assert 'aria-label="下载平滑度矢量图"' not in response.text
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


def test_report_page_404_when_dataset_and_metrics_sessions_do_not_match(
    auth_client, successful_job, ready_dataset
):
    session_path = Path(ready_dataset.path) / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["session_id"] = "different-session"
    session_path.write_text(json.dumps(session), encoding="utf-8")

    response = auth_client.get(f"/reports/{successful_job.id}")

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


def test_download_serves_smoothness_curve_svg(auth_client, successful_job):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
    (Path(successful_job.output_dir) / "smoothness_curve.svg").write_bytes(svg)

    response = auth_client.get(
        f"/reports/{successful_job.id}/files/smoothness_curve.svg"
    )

    assert response.status_code == 200
    assert response.content == svg
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert (
        'attachment; filename="smoothness_curve.svg"'
        in response.headers["content-disposition"]
    )


@pytest.mark.parametrize("filename", ["attempt_summary.json", "attempt_summary.csv"])
def test_download_serves_nested_vlm_artifacts(auth_client, successful_job, filename):
    attempt_dir = Path(successful_job.output_dir) / "attempt_eval"
    attempt_dir.mkdir()
    payload = "[]" if filename.endswith(".json") else "episode_index\n"
    (attempt_dir / filename).write_text(payload, encoding="utf-8")

    response = auth_client.get(
        f"/reports/{successful_job.id}/files/attempt_eval/{filename}"
    )

    assert response.status_code == 200
    assert response.text == payload
    assert filename in response.headers["content-disposition"]


def test_download_rejects_obsolete_root_vlm_artifact(auth_client, successful_job):
    path = Path(successful_job.output_dir) / "attempt_summary.json"
    path.write_text("SECRET-ROOT", encoding="utf-8")

    response = auth_client.get(
        f"/reports/{successful_job.id}/files/attempt_summary.json"
    )

    assert response.status_code == 404
    assert "SECRET-ROOT" not in response.text


def test_download_rejects_unknown_nested_artifact(auth_client, successful_job):
    attempt_dir = Path(successful_job.output_dir) / "attempt_eval"
    attempt_dir.mkdir()
    (attempt_dir / "unknown.json").write_text("SECRET-NESTED", encoding="utf-8")

    response = auth_client.get(
        f"/reports/{successful_job.id}/files/attempt_eval/unknown.json"
    )

    assert response.status_code == 404
    assert "SECRET-NESTED" not in response.text


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


def test_report_page_rejects_invalid_required_episode_csv(auth_client, successful_job):
    output_dir = Path(successful_job.output_dir)
    (output_dir / "episode_metrics.csv").write_text(
        "unexpected,column\ninvalid,row\n", encoding="utf-8"
    )

    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Report is not available"
