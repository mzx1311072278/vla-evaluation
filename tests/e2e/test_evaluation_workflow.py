"""End-to-end evaluation workflow acceptance test (Task 16, Step 1).

Exercises the full design path in-process, with every production component real
except the rsync network transfer (faked by copying a prepared fixture into the
job staging path):

  1. log in,
  2. POST /imports  -> run_import_task  (fake rsync)  -> Dataset READY,
  3. POST /evaluations -> run_evaluation_task (real Genie02 metrics pipeline),
  4. GET /reports/{id} and download metrics_core.json,
  5. reopen the job and observe the persisted terminal state.

State transitions, provenance traceability (data fingerprint, profile version,
code version), and persisted progress recovery are all asserted.
"""

from __future__ import annotations

import json

import pytest

import vla_eval
from tests.e2e.conftest import (
    REMOTE_ROOT,
    build_native_session,
    drain_queues,
    install_fake_rsync,
)
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, ImportJob

TARGET_NAME = "run-1"


def _redirect_id(response, prefix: str) -> str:
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert location.startswith(prefix), location
    return location.rsplit("/", 1)[-1]


def test_end_to_end_evaluation_workflow(
    auth_client,
    fake_queues,
    runtime,
    db_engine,
    data_root,
    tmp_path,
    monkeypatch,
):
    # --- Step 1: prepare a valid Genie02 native session on the (fake) remote and
    # replace only the rsync transfer with a local copy. ------------------------
    remote_fixture_dir = build_native_session(tmp_path / "remote")
    install_fake_rsync(monkeypatch, remote_fixture_dir)

    # --- Step 2: import via the web route, then run the real import task. ------
    import_response = auth_client.post(
        "/imports",
        data={
            "csrf_token": auth_client.csrf,
            "source_name": "lab-a",
            "root": REMOTE_ROOT,
            "relative_path": "run-1",
            "target_name": TARGET_NAME,
        },
        follow_redirects=False,
    )
    import_id = _redirect_id(import_response, "/imports/")

    # The transfer queue received exactly one run_import_task call.
    assert len(fake_queues.transfer.enqueued) == 1
    assert len(fake_queues.evaluation.enqueued) == 0

    drain_queues(fake_queues, runtime)

    # Import state machine reached READY (passed CONNECTING->...->PREFLIGHT->READY)
    # and published a READY dataset with a real fingerprint + episode count.
    with session_scope(db_engine) as session:
        import_job = session.get_one(ImportJob, import_id)
        assert import_job.state == "READY"
        assert import_job.progress == 100.0
        assert import_job.publish_fingerprint is not None
        assert import_job.dataset_id is not None
        dataset = session.get_one(Dataset, import_job.dataset_id)
        assert dataset.status == "READY"
        assert dataset.kind == "genie02_session"
        assert dataset.fingerprint == import_job.publish_fingerprint
        assert dataset.episode_count == 2
        dataset_id = dataset.id
        dataset_fingerprint = dataset.fingerprint
    assert (data_root / "inbox" / TARGET_NAME / "session.json").is_file()

    # The web detail + JSON status views reflect the persisted READY state.
    detail = auth_client.get(f"/imports/{import_id}")
    assert detail.status_code == 200
    assert "READY" in detail.text
    status = auth_client.get(f"/api/imports/{import_id}")
    assert status.status_code == 200
    assert status.json() == {
        "id": import_id,
        "state": "READY",
        "progress": 100.0,
        "error_code": None,
        "error_message": None,
        "dataset_id": dataset_id,
        "finished": True,
    }

    # --- Step 3: submit a no-VLM evaluation against the imported dataset. ------
    eval_response = auth_client.post(
        "/evaluations",
        data={
            "csrf_token": auth_client.csrf,
            "dataset_id": dataset_id,
            "profile": "genie02-full",
            "vlm_enabled": "false",
        },
        follow_redirects=False,
    )
    job_id = _redirect_id(eval_response, "/evaluations/")

    assert len(fake_queues.evaluation.enqueued) == 1
    drain_queues(fake_queues, runtime)

    # Evaluation state machine reached SUCCEEDED (PREFLIGHT->METRICS->REPORT)
    # at progress 100 with provenance traceability captured at submission time.
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        assert job.state == "SUCCEEDED"
        assert job.stage == "REPORT"
        assert job.progress == 100.0
        assert job.output_dir is not None
        assert job.error_code is None
        provenance = job.provenance_json
        assert provenance is not None
        assert provenance["dataset_fingerprint"] == dataset_fingerprint
        assert provenance["profile_name"] == "genie02-full"
        assert provenance["profile_version"] == "1.0.0"
        assert provenance["app_version"] == vla_eval.__version__
        assert "git_sha" in provenance

    # --- Step 4: report page renders and metrics_core.json downloads. ----------
    report = auth_client.get(f"/reports/{job_id}")
    assert report.status_code == 200
    assert "GSR" in report.text

    metrics_response = auth_client.get(f"/reports/{job_id}/files/metrics_core.json")
    assert metrics_response.status_code == 200
    metrics = json.loads(metrics_response.content)
    assert metrics["n_episodes"] == 2
    assert metrics["gsr"] == pytest.approx(0.5)

    # --- Step 5: reopening the job (fresh GET) shows the persisted progress. ---
    reopened_detail = auth_client.get(f"/evaluations/{job_id}")
    assert reopened_detail.status_code == 200
    assert "SUCCEEDED" in reopened_detail.text
    assert "查看报告" in reopened_detail.text
    reopened_status = auth_client.get(f"/api/evaluations/{job_id}")
    assert reopened_status.status_code == 200
    assert reopened_status.json()["state"] == "SUCCEEDED"
    assert reopened_status.json()["finished"] is True
