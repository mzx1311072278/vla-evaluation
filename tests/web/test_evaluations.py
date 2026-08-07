import json
import re
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select

from tests.conftest import reload_job
from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob, EvaluationJobArchive
from vla_eval.tasks import run_evaluation_task


def _listed_evaluation_ids(html: str) -> list[str]:
    matches = re.findall(r'href="/evaluations/([^"/?]+)"', html)
    return list(dict.fromkeys(matches))


def _evaluation_form(csrf: str, dataset_id: str, **overrides: str) -> dict[str, str]:
    values: dict[str, str] = {
        "csrf_token": csrf,
        "dataset_id": dataset_id,
        "profile": "genie02-full",
        "vlm_enabled": "false",
    }
    values.update(overrides)
    return values


def matching_form(job: EvaluationJob, csrf: str) -> dict[str, str]:
    """Build a form that matches an existing persisted evaluation's identity."""
    return {
        "csrf_token": csrf,
        "dataset_id": job.dataset_id,
        "profile": "genie02-full",
        "vlm_enabled": "false",
    }


def test_evaluation_list_shows_newest_jobs_and_report_links(
    auth_client, db_engine: Engine, ready_dataset
):
    created_at = datetime(2026, 8, 6, 9, 30, tzinfo=UTC)
    with session_scope(db_engine) as session:
        older = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            vlm_enabled=False,
            state="RUNNING",
            stage="METRICS",
            progress=35,
            created_at=created_at,
        )
        newer = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-api",
            profile_version="1.0.0",
            vlm_enabled=True,
            state="SUCCEEDED",
            stage="REPORT",
            progress=100,
            created_at=created_at + timedelta(hours=1),
        )
        session.add_all([older, newer])
        session.flush()
        older_id, newer_id = older.id, newer.id

    response = auth_client.get("/evaluations")

    assert response.status_code == 200
    assert response.text.index(newer_id) < response.text.index(older_id)
    for value in (ready_dataset.name, "genie02-api", "SUCCEEDED", "REPORT", "100%"):
        assert value in response.text
    assert "已启用" in response.text
    assert f'href="/evaluations/{newer_id}"' in response.text
    assert f'href="/reports/{newer_id}"' in response.text
    assert f'href="/reports/{older_id}"' not in response.text


def test_evaluation_list_filters_state_and_rejects_invalid_values(
    auth_client, db_engine: Engine, ready_dataset
):
    with session_scope(db_engine) as session:
        session.add_all(
            [
                EvaluationJob(
                    dataset_id=ready_dataset.id,
                    profile_name="visible-profile",
                    state="FAILED",
                ),
                EvaluationJob(
                    dataset_id=ready_dataset.id,
                    profile_name="hidden-profile",
                    state="SUCCEEDED",
                ),
            ]
        )

    filtered = auth_client.get("/evaluations?state=FAILED")

    assert filtered.status_code == 200
    assert "visible-profile" in filtered.text
    assert "hidden-profile" not in filtered.text
    assert auth_client.get("/evaluations?state=UNKNOWN").status_code == 422
    assert auth_client.get("/evaluations?state=FAILED&state=FAILED").status_code == 422


def test_evaluation_list_searches_dataset_profile_and_partial_id(
    auth_client, db_engine: Engine
):
    with session_scope(db_engine) as session:
        robot_dataset = Dataset(
            id="10000000-0000-0000-0000-000000000001",
            name="Robot Lab",
            path="/srv/robot",
            kind="fixture",
            status="READY",
        )
        vision_dataset = Dataset(
            id="10000000-0000-0000-0000-000000000002",
            name="Vision Lab",
            path="/srv/vision",
            kind="fixture",
            status="READY",
        )
        session.add_all([robot_dataset, vision_dataset])
        session.flush()
        jobs = [
            EvaluationJob(
                id="20000000-0000-0000-0000-000000000001",
                dataset_id=robot_dataset.id,
                profile_name="plain-profile",
                state="FAILED",
            ),
            EvaluationJob(
                id="20000000-0000-0000-0000-000000000002",
                dataset_id=vision_dataset.id,
                profile_name="ROBOT-policy",
                state="FAILED",
            ),
            EvaluationJob(
                id="c0ffee00-0000-0000-0000-000000000003",
                dataset_id=vision_dataset.id,
                profile_name="plain-profile",
                state="FAILED",
            ),
        ]
        session.add_all(jobs)

    robot = auth_client.get("/evaluations?q=robot")
    partial_id = auth_client.get("/evaluations?q=C0FFEE")

    assert set(_listed_evaluation_ids(robot.text)) == {jobs[0].id, jobs[1].id}
    assert _listed_evaluation_ids(partial_id.text) == [jobs[2].id]


def test_evaluation_list_combines_filters_and_archive_visibility(
    auth_client, db_engine: Engine
):
    base = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)
    with session_scope(db_engine) as session:
        target = Dataset(
            id="30000000-0000-0000-0000-000000000001",
            name="Target Dataset",
            path="/srv/target",
            kind="fixture",
            status="READY",
        )
        other = Dataset(
            id="30000000-0000-0000-0000-000000000002",
            name="Other Dataset",
            path="/srv/other",
            kind="fixture",
            status="READY",
        )
        session.add_all([target, other])
        session.flush()
        visible = EvaluationJob(
            id="40000000-0000-0000-0000-000000000001",
            dataset_id=target.id,
            profile_name="needle-visible",
            state="FAILED",
            created_at=base,
        )
        archived = EvaluationJob(
            id="40000000-0000-0000-0000-000000000002",
            dataset_id=target.id,
            profile_name="needle-archived",
            state="FAILED",
            created_at=base + timedelta(hours=1),
        )
        wrong_state = EvaluationJob(
            id="40000000-0000-0000-0000-000000000003",
            dataset_id=target.id,
            profile_name="needle-success",
            state="SUCCEEDED",
            created_at=base + timedelta(hours=2),
        )
        wrong_dataset = EvaluationJob(
            id="40000000-0000-0000-0000-000000000004",
            dataset_id=other.id,
            profile_name="needle-other",
            state="FAILED",
            created_at=base + timedelta(hours=3),
        )
        session.add_all([visible, archived, wrong_state, wrong_dataset])
        session.flush()
        session.add(EvaluationJobArchive(evaluation_job_id=archived.id))

    query = f"q=needle&sort=name_asc&state=FAILED&dataset_id={target.id}"
    default = auth_client.get(f"/evaluations?{query}")
    with_archived = auth_client.get(f"/evaluations?{query}&archived=1")

    assert _listed_evaluation_ids(default.text) == [visible.id]
    assert _listed_evaluation_ids(with_archived.text) == [archived.id, visible.id]


def test_evaluation_list_supports_four_stable_sorts(auth_client, db_engine: Engine):
    base = datetime(2026, 8, 7, 11, 0, tzinfo=UTC)
    with session_scope(db_engine) as session:
        alpha = Dataset(
            id="50000000-0000-0000-0000-000000000001",
            name="Alpha",
            path="/srv/alpha",
            kind="fixture",
            status="READY",
        )
        bravo = Dataset(
            id="50000000-0000-0000-0000-000000000002",
            name="bravo",
            path="/srv/bravo",
            kind="fixture",
            status="READY",
        )
        session.add_all([alpha, bravo])
        session.flush()
        jobs = [
            EvaluationJob(
                id=f"60000000-0000-0000-0000-00000000000{index}",
                dataset_id=dataset_id,
                profile_name="profile",
                state="FAILED",
                created_at=created_at,
            )
            for index, dataset_id, created_at in (
                (1, alpha.id, base),
                (2, bravo.id, base + timedelta(hours=1)),
                (3, alpha.id, base + timedelta(hours=2)),
                (4, alpha.id, base + timedelta(hours=2)),
            )
        ]
        session.add_all(jobs)

    expected = {
        "newest": [jobs[3].id, jobs[2].id, jobs[1].id, jobs[0].id],
        "oldest": [jobs[0].id, jobs[1].id, jobs[2].id, jobs[3].id],
        "name_asc": [jobs[2].id, jobs[3].id, jobs[0].id, jobs[1].id],
        "name_desc": [jobs[1].id, jobs[3].id, jobs[2].id, jobs[0].id],
    }
    for sort, ids in expected.items():
        response = auth_client.get(f"/evaluations?sort={sort}")
        assert response.status_code == 200
        assert _listed_evaluation_ids(response.text) == ids


@pytest.mark.parametrize("query", ["unknown=1", "q=one&q=two", "sort=invalid"])
def test_evaluation_list_rejects_invalid_duplicate_or_unknown_controls(
    auth_client, query: str
):
    assert auth_client.get(f"/evaluations?{query}").status_code == 422


@pytest.mark.parametrize(
    "path",
    ["/evaluations", "/evaluations/new", "/evaluations/missing", "/api/evaluations/missing"],
)
def test_evaluation_pages_require_login(client: TestClient, path: str):
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def _archive_evaluation(client, job_id: str, *, return_to: str = "/evaluations"):
    return client.post(
        f"/evaluations/{job_id}/archive",
        data={"csrf_token": client.csrf, "return_to": return_to},
        follow_redirects=False,
    )


def _restore_evaluation(client, job_id: str, *, return_to: str = "/evaluations"):
    return client.post(
        f"/evaluations/{job_id}/restore",
        data={"csrf_token": client.csrf, "return_to": return_to},
        follow_redirects=False,
    )


def _job_snapshot(job: EvaluationJob) -> dict[str, object]:
    return {
        column.name: deepcopy(getattr(job, column.name))
        for column in EvaluationJob.__table__.columns
    }


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"])
def test_archive_terminal_evaluation_preserves_every_job_field(
    auth_client, db_engine: Engine, evaluation_job, user, state: str
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = state
        job.stage = "REPORT"
        job.progress = 73.5
        job.error_code = "HISTORICAL_ERROR"
        job.error_message = "historical details"
        job.params_json = {"vlm_enabled": True}
        job.provenance_json = {"dataset_fingerprint": "abc123"}
        session.flush()
        before = _job_snapshot(job)

    response = _archive_evaluation(
        auth_client,
        evaluation_job.id,
        return_to=f"/evaluations?state={state}",
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations?state={state}"
    with session_scope(db_engine) as session:
        assert _job_snapshot(session.get_one(EvaluationJob, evaluation_job.id)) == before
        archive = session.get_one(EvaluationJobArchive, evaluation_job.id)
        assert archive.archived_by == user.id
        assert archive.archived_at.tzinfo is not None


@pytest.mark.parametrize("state", ["QUEUED", "RUNNING"])
def test_archive_evaluation_rejects_nonterminal_state(
    auth_client, db_engine: Engine, evaluation_job, state: str
):
    with session_scope(db_engine) as session:
        session.get_one(EvaluationJob, evaluation_job.id).state = state

    response = _archive_evaluation(auth_client, evaluation_job.id)

    assert response.status_code == 409
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, evaluation_job.id) is None


def test_archive_evaluation_requires_one_valid_csrf_token(
    auth_client, db_engine: Engine, evaluation_job
):
    with session_scope(db_engine) as session:
        session.get_one(EvaluationJob, evaluation_job.id).state = "FAILED"

    missing = auth_client.post(
        f"/evaluations/{evaluation_job.id}/archive",
        data={"return_to": "/evaluations"},
    )
    duplicate = auth_client.post(
        f"/evaluations/{evaluation_job.id}/archive",
        content=urlencode(
            [
                ("csrf_token", auth_client.csrf),
                ("csrf_token", auth_client.csrf),
                ("return_to", "/evaluations"),
            ]
        ),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert missing.status_code == duplicate.status_code == 403
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, evaluation_job.id) is None


@pytest.mark.parametrize(
    "fields",
    [
        [("csrf_token", "{csrf}")],
        [("csrf_token", "{csrf}"), ("return_to", "/evaluations"), ("extra", "1")],
        [
            ("csrf_token", "{csrf}"),
            ("return_to", "/evaluations"),
            ("return_to", "/evaluations?state=FAILED"),
        ],
    ],
)
def test_archive_evaluation_rejects_missing_unknown_or_duplicate_fields(
    auth_client, db_engine: Engine, evaluation_job, fields
):
    with session_scope(db_engine) as session:
        session.get_one(EvaluationJob, evaluation_job.id).state = "FAILED"
    submitted = [
        (key, auth_client.csrf if value == "{csrf}" else value) for key, value in fields
    ]

    response = auth_client.post(
        f"/evaluations/{evaluation_job.id}/archive",
        content=urlencode(submitted),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, evaluation_job.id) is None


def test_evaluation_archive_restore_lifecycle_and_safe_return_target(
    auth_client, db_engine: Engine, evaluation_job
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "INTERRUPTED"
        session.flush()
        before = _job_snapshot(job)

    archived = _archive_evaluation(
        auth_client,
        evaluation_job.id,
        return_to="https://evil.example/steal",
    )
    repeated_archive = _archive_evaluation(auth_client, evaluation_job.id)
    restored = _restore_evaluation(
        auth_client,
        evaluation_job.id,
        return_to=f"/evaluations/{evaluation_job.id}",
    )
    repeated_restore = _restore_evaluation(auth_client, evaluation_job.id)

    assert archived.status_code == 303
    assert archived.headers["location"] == "/evaluations"
    assert repeated_archive.status_code == 409
    assert restored.status_code == 303
    assert restored.headers["location"] == f"/evaluations/{evaluation_job.id}"
    assert repeated_restore.status_code == 409
    with session_scope(db_engine) as session:
        assert session.get(EvaluationJobArchive, evaluation_job.id) is None
        assert _job_snapshot(session.get_one(EvaluationJob, evaluation_job.id)) == before


@pytest.mark.parametrize("operation", ["archive", "restore"])
def test_evaluation_archive_operations_reject_missing_job(auth_client, operation: str):
    response = auth_client.post(
        f"/evaluations/not-a-job/{operation}",
        data={"csrf_token": auth_client.csrf, "return_to": "/evaluations"},
    )

    assert response.status_code == 404


def test_submit_evaluation_enqueues_business_id(auth_client, ready_dataset, fake_queues):
    response = auth_client.post(
        "/evaluations",
        data={
            "csrf_token": auth_client.csrf,
            "dataset_id": ready_dataset.id,
            "profile": "genie02-full",
            "vlm_enabled": "true",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert (
        fake_queues.evaluation.enqueued[0].args
        == (response.headers["location"].rsplit("/", 1)[-1],)
    )


def test_submit_evaluation_persists_provenance_and_enqueues_worker(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, user
):
    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id, vlm_enabled="true"),
        follow_redirects=False,
    )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/evaluations/")
    job = reload_job(db_engine, job_id)
    assert job.dataset_id == ready_dataset.id
    assert job.profile_name == "genie02-full"
    assert job.profile_version == "1.0.0"
    assert job.vlm_enabled is True
    assert len(job.run_key) == 64
    assert job.state == "QUEUED"
    assert job.stage == "PENDING"
    assert job.created_by == user.id
    assert job.params_json == {"vlm_enabled": True}
    assert job.provenance_json
    assert job.provenance_json["dataset_fingerprint"] == ready_dataset.fingerprint
    assert job.provenance_json["profile_name"] == "genie02-full"
    assert job.provenance_json["profile_version"] == "1.0.0"
    assert job.provenance_json["app_version"]
    assert job.provenance_json["git_sha"] == ""
    assert job.provenance_json["vlm_model_path"]
    assert job.provenance_json["vlm_backend"] == "local"
    assert "vlm_api_model" not in job.provenance_json
    assert job.provenance_json["prompt_version"]
    assert job.provenance_json["adapter"] == "genie02"
    assert job.provenance_json["plugin"] == "genie02-attempt-eval"
    assert job.provenance_json["image_key"] == "observation.images.right_wrist"
    assert job.provenance_json["max_image_size"] == 336
    assert job.provenance_json["max_new_tokens"] == 256
    assert job.provenance_json["sampling"] == {
        "max_global_frames": 8,
        "global_sample_interval": 2.0,
        "max_dense_frames": 8,
        "dense_sample_interval": 0.5,
        "dense_region": "full",
    }
    assert job.provenance_json["review"] == {
        "mode": "manual_review",
        "confidence_threshold": 0.7,
        "min_episode_duration": 3.0,
        "min_sampled_frames": 3,
    }
    assert job.provenance_json["outputs"] == {
        "required": ["episode_metrics.csv", "metrics_core.json", "report_*.md"],
        "optional": [
            "smoothness_curve.svg",
            "attempt_eval/attempt_summary.json",
            "attempt_eval/attempt_summary.csv",
        ],
    }
    assert job.provenance_json["params"] == {"vlm_enabled": True}

    assert fake_queues.evaluation.count == 1
    call = fake_queues.evaluation.enqueued[0]
    assert call.function is run_evaluation_task
    assert call.args == (job_id,)


def test_submit_evaluation_records_api_backend_provenance(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    """An api-profile submission records backend='api' and the api connection block.

    Provenance stores the env-var NAME only (vlm_api_key_env); the secret VALUE
    is never persisted -- vlm_model_path is null for the api backend so consumers
    see a stable key set across both backends.
    """
    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(
            auth_client.csrf, ready_dataset.id, profile="genie02-api", vlm_enabled="true"
        ),
        follow_redirects=False,
    )

    assert response.status_code == 303
    job_id = response.headers["location"].removeprefix("/evaluations/")
    job = reload_job(db_engine, job_id)
    assert job.profile_name == "genie02-api"
    provenance = job.provenance_json
    assert provenance["vlm_backend"] == "api"
    assert provenance["vlm_model_path"] is None
    assert provenance["vlm_api_base_url"] == "http://vlm-api.example.internal/v1"
    assert provenance["vlm_api_model"] == "qwen2.5-vl-7b-instruct"
    assert provenance["vlm_api_key_env"] == "VLA_EVAL_VLM_API_KEY"
    assert provenance["vlm_api_timeout"] == 60
    assert provenance["vlm_api_max_retries"] == 3
    assert fake_queues.evaluation.count == 1
    assert fake_queues.evaluation.enqueued[0].args == (job_id,)


def test_duplicate_successful_run_redirects_to_existing_evaluation(
    auth_client, db_engine: Engine, fake_queues, successful_job
):
    response = auth_client.post(
        "/evaluations",
        data=matching_form(successful_job, auth_client.csrf),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations/{successful_job.id}"
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert [job.id for job in session.scalars(select(EvaluationJob))] == [successful_job.id]


def test_force_submission_creates_new_evaluation_after_duplicate(
    auth_client, db_engine: Engine, fake_queues, successful_job
):
    response = auth_client.post(
        "/evaluations",
        data={**matching_form(successful_job, auth_client.csrf), "force": "true"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    new_job_id = response.headers["location"].removeprefix("/evaluations/")
    assert new_job_id != successful_job.id
    assert fake_queues.evaluation.count == 1
    assert fake_queues.evaluation.enqueued[0].args == (new_job_id,)


def test_submit_rejects_non_ready_dataset_without_side_effects(
    auth_client, db_engine: Engine, fake_queues, dataset
):
    response = auth_client.post(
        "/evaluations", data=_evaluation_form(auth_client.csrf, dataset.id)
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []


@pytest.mark.parametrize("profile", ["MissingProfile", "does-not-exist", "../etc"])
def test_submit_rejects_invalid_profile(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, profile: str
):
    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id, profile=profile),
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []


def test_submit_rejects_unknown_dataset(auth_client, fake_queues):
    response = auth_client.post(
        "/evaluations", data=_evaluation_form(auth_client.csrf, "not-a-dataset")
    )

    assert response.status_code == 404
    assert fake_queues.evaluation.count == 0


def test_submit_requires_valid_csrf_before_creating_job(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    response = auth_client.post(
        "/evaluations", data=_evaluation_form("wrong-token", ready_dataset.id)
    )

    assert response.status_code == 403
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []


def test_submit_rejects_unknown_form_field(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    response = auth_client.post(
        "/evaluations",
        data={**_evaluation_form(auth_client.csrf, ready_dataset.id), "evil": "1"},
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []


def test_submit_rejects_duplicate_form_values(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    fields = list(_evaluation_form(auth_client.csrf, ready_dataset.id).items())
    fields.append(("profile", "genie02-full"))

    response = auth_client.post(
        "/evaluations",
        content=urlencode(fields),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []


def test_submit_rejects_invalid_vlm_enabled_value(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(
            auth_client.csrf, ready_dataset.id, vlm_enabled="maybe"
        ),
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0


def test_enqueue_failure_removes_unclaimed_evaluation(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, monkeypatch
):
    def fail_enqueue(*_args):
        raise RuntimeError("redis password=secret")

    monkeypatch.setattr(fake_queues.evaluation, "enqueue", fail_enqueue)

    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
        follow_redirects=False,
    )

    assert response.status_code == 503
    with session_scope(db_engine) as session:
        assert session.scalars(select(EvaluationJob)).all() == []
    assert "secret" not in response.text


def test_retry_failed_evaluation_requeues(
    auth_client, db_engine: Engine, fake_queues, evaluation_job
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "FAILED"
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "boom"

    response = auth_client.post(
        f"/evaluations/{evaluation_job.id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations/{evaluation_job.id}"
    job = reload_job(db_engine, evaluation_job.id)
    assert job.state == "QUEUED"
    assert job.stage == "PENDING"
    assert job.error_code is None
    assert job.error_message is None
    assert job.run_key == evaluation_job.run_key
    assert fake_queues.evaluation.count == 1
    assert fake_queues.evaluation.enqueued[0].function is run_evaluation_task
    assert fake_queues.evaluation.enqueued[0].args == (evaluation_job.id,)


def test_retry_succeeds_when_dataset_fingerprint_matches(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    """A submitted job carries the submit-time fingerprint; an unchanged dataset may retry."""
    submit = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
        follow_redirects=False,
    )
    assert submit.status_code == 303
    job_id = submit.headers["location"].removeprefix("/evaluations/")

    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        assert job.provenance_json.get("dataset_fingerprint") == ready_dataset.fingerprint
        job.state = "FAILED"
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "boom"

    response = auth_client.post(
        f"/evaluations/{job_id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations/{job_id}"
    reloaded = reload_job(db_engine, job_id)
    assert reloaded.state == "QUEUED"
    assert reloaded.error_code is None
    assert reloaded.error_message is None
    # One enqueue from the original submit, plus one from the retry.
    assert fake_queues.evaluation.count == 2
    assert fake_queues.evaluation.enqueued[-1].function is run_evaluation_task
    assert fake_queues.evaluation.enqueued[-1].args == (job_id,)


def test_retry_rejects_when_dataset_fingerprint_changed(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    """If the on-disk dataset contents drifted since submission, retry is rejected (409)."""
    submit = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
        follow_redirects=False,
    )
    assert submit.status_code == 303
    job_id = submit.headers["location"].removeprefix("/evaluations/")

    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        original_fingerprint = job.provenance_json.get("dataset_fingerprint")
        assert original_fingerprint == ready_dataset.fingerprint
        job.state = "FAILED"
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "boom"

    # Mutate the dataset on disk so its fingerprint changes while staying READY.
    # session.json is a metadata file hashed into the fingerprint; the "task" field
    # value is not validated, so changing it keeps the dataset ready but changes the
    # sha256 and therefore the fingerprint.
    session_path = Path(ready_dataset.path) / "session.json"
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    session_data["task"] = "changed-before-retry"
    session_path.write_text(
        json.dumps(session_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    response = auth_client.post(
        f"/evaluations/{job_id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 409
    # The submit enqueued once; the rejected retry must not add another enqueue.
    assert fake_queues.evaluation.count == 1
    reloaded = reload_job(db_engine, job_id)
    assert reloaded.state == "FAILED"
    assert reloaded.error_code == "EVALUATION_FAILED"


def test_retry_rejected_for_successful_evaluation(auth_client, successful_job):
    response = auth_client.post(
        f"/evaluations/{successful_job.id}/retry",
        data={"csrf_token": auth_client.csrf},
    )
    assert response.status_code == 409


def test_retry_rejected_when_dataset_not_ready(
    auth_client, db_engine: Engine, fake_queues, dataset
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="FAILED",
            stage="METRICS",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.post(
        f"/evaluations/{job_id}/retry", data={"csrf_token": auth_client.csrf}
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0


def test_archived_dataset_cannot_open_new_evaluation(auth_client, db_engine, ready_dataset):
    with session_scope(db_engine) as session:
        session.get_one(Dataset, ready_dataset.id).status = "ARCHIVED"

    response = auth_client.get(f"/evaluations/new?dataset_id={ready_dataset.id}")

    assert response.status_code == 422


def test_archived_dataset_cannot_submit_evaluation(
    auth_client, db_engine, fake_queues, ready_dataset
):
    with session_scope(db_engine) as session:
        session.get_one(Dataset, ready_dataset.id).status = "ARCHIVED"

    response = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0


def test_failed_evaluation_cannot_retry_while_dataset_archived(
    auth_client, db_engine, fake_queues, evaluation_job
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "FAILED"
        session.get_one(Dataset, job.dataset_id).status = "ARCHIVED"

    response = auth_client.post(
        f"/evaluations/{evaluation_job.id}/retry",
        data={"csrf_token": auth_client.csrf},
    )

    assert response.status_code == 422
    assert fake_queues.evaluation.count == 0


@pytest.mark.parametrize("state", ["CANCELLED", "QUEUED", "RUNNING"])
def test_retry_rejected_for_non_retryable_state(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, state: str
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state=state,
            stage="METRICS",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.post(
        f"/evaluations/{job_id}/retry", data={"csrf_token": auth_client.csrf}
    )

    assert response.status_code == 409
    assert fake_queues.evaluation.count == 0


def test_retry_succeeds_for_interrupted_evaluation(
    auth_client, db_engine: Engine, fake_queues, evaluation_job
):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "INTERRUPTED"
        job.stage = "VLM"

    response = auth_client.post(
        f"/evaluations/{evaluation_job.id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert reload_job(db_engine, evaluation_job.id).state == "QUEUED"
    assert fake_queues.evaluation.count == 1


def test_retry_enqueue_failure_restores_previous_state(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, monkeypatch
):
    submit = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
        follow_redirects=False,
    )
    assert submit.status_code == 303
    job_id = submit.headers["location"].removeprefix("/evaluations/")

    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "FAILED"
        job.stage = "VLM"
        job.progress = 42.5
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "mid-run failure"

    def fail_enqueue(*_args):
        raise RuntimeError("redis password=top-secret")

    monkeypatch.setattr(fake_queues.evaluation, "enqueue", fail_enqueue)

    response = auth_client.post(
        f"/evaluations/{job_id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 503
    assert "top-secret" not in response.text
    job = reload_job(db_engine, job_id)
    assert job.state == "FAILED"
    assert job.stage == "VLM"
    assert job.progress == 42.5
    assert job.error_code == "EVALUATION_FAILED"
    assert job.error_message == "mid-run failure"


def test_retry_enqueue_failure_does_not_clobber_worker_claim(
    auth_client, db_engine: Engine, fake_queues, ready_dataset, monkeypatch
):
    submit = auth_client.post(
        "/evaluations",
        data=_evaluation_form(auth_client.csrf, ready_dataset.id),
        follow_redirects=False,
    )
    assert submit.status_code == 303
    job_id = submit.headers["location"].removeprefix("/evaluations/")

    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, job_id)
        job.state = "FAILED"
        job.stage = "VLM"
        job.progress = 42.5
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "mid-run failure"

    def claim_then_fail(_function, target_id):
        with session_scope(db_engine) as session:
            claimed = session.get_one(EvaluationJob, target_id)
            claimed.state = "RUNNING"
            claimed.execution_token = "worker-token"
        raise RuntimeError("connection dropped after enqueue")

    monkeypatch.setattr(fake_queues.evaluation, "enqueue", claim_then_fail)

    response = auth_client.post(
        f"/evaluations/{job_id}/retry",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations/{job_id}"
    job = reload_job(db_engine, job_id)
    assert job.state == "RUNNING"
    assert job.execution_token == "worker-token"


def test_cancel_marks_active_evaluation_requested(
    auth_client, db_engine: Engine, fake_queues, evaluation_job
):
    response = auth_client.post(
        f"/evaluations/{evaluation_job.id}/cancel",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/evaluations/{evaluation_job.id}"
    assert reload_job(db_engine, evaluation_job.id).cancel_requested is True
    assert fake_queues.evaluation.count == 0


def test_cancel_rejected_for_terminal_evaluation(auth_client, successful_job):
    response = auth_client.post(
        f"/evaluations/{successful_job.id}/cancel",
        data={"csrf_token": auth_client.csrf},
    )
    assert response.status_code == 409


def test_cancel_marks_running_evaluation_requested(
    auth_client, db_engine: Engine, fake_queues, ready_dataset
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="RUNNING",
            stage="VLM",
            execution_token="worker-token",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.post(
        f"/evaluations/{job_id}/cancel",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    job = reload_job(db_engine, job_id)
    assert job.cancel_requested is True
    assert job.execution_token == "worker-token"


def test_evaluation_status_api_returns_documented_fields(
    auth_client, db_engine: Engine, ready_dataset
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="RUNNING",
            stage="VLM",
            progress=55.0,
            vlm_enabled=True,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.get(f"/api/evaluations/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["state"] == "RUNNING"
    assert body["stage"] == "VLM"
    assert body["progress"] == 55.0
    assert body["error_code"] is None
    assert body["error_message"] is None
    assert isinstance(body["updated_at"], str)
    assert body["updated_at"].endswith("+00:00")
    assert body["finished"] is False
    assert "HX-Trigger" not in response.headers


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"])
def test_terminal_evaluation_status_triggers_polling_completion(
    auth_client, db_engine: Engine, ready_dataset, state: str
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state=state,
            stage="REPORT",
            progress=100.0,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.get(f"/api/evaluations/{job_id}")

    assert response.status_code == 200
    assert response.json()["finished"] is True
    assert response.headers["HX-Trigger"] == "job-finished"


@pytest.mark.parametrize("path", ["/evaluations/not-a-job", "/api/evaluations/not-a-job"])
def test_missing_evaluation_is_not_found(auth_client, path: str):
    assert auth_client.get(path).status_code == 404


def test_missing_evaluation_retry_is_not_found(auth_client):
    response = auth_client.post(
        "/evaluations/not-a-job/retry", data={"csrf_token": auth_client.csrf}
    )
    assert response.status_code == 404


def test_missing_evaluation_cancel_is_not_found(auth_client):
    response = auth_client.post(
        "/evaluations/not-a-job/cancel", data={"csrf_token": auth_client.csrf}
    )
    assert response.status_code == 404


def test_evaluation_new_page_preselects_dataset(auth_client, ready_dataset):
    response = auth_client.get(f"/evaluations/new?dataset_id={ready_dataset.id}")

    assert response.status_code == 200
    assert ready_dataset.name in response.text
    assert "genie02-full" in response.text


def test_evaluation_new_page_rejects_missing_dataset_param(auth_client):
    response = auth_client.get("/evaluations/new")
    assert response.status_code == 422


def test_evaluation_new_page_rejects_unknown_dataset(auth_client):
    response = auth_client.get("/evaluations/new?dataset_id=not-a-dataset")
    assert response.status_code == 404


def test_evaluation_new_page_rejects_non_ready_dataset(auth_client, dataset):
    response = auth_client.get(f"/evaluations/new?dataset_id={dataset.id}")
    assert response.status_code == 422


def test_evaluation_detail_shows_stage_progress_and_polls(
    auth_client, db_engine: Engine, ready_dataset
):
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="RUNNING",
            stage="METRICS",
            progress=30.0,
            vlm_enabled=True,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    response = auth_client.get(f"/evaluations/{job_id}")

    assert response.status_code == 200
    for label in ("预检", "指标", "VLM", "报告"):
        assert label in response.text
    assert 'hx-trigger="every 2s"' in response.text


def test_successful_evaluation_detail_links_to_report_and_stops_polling(
    auth_client, successful_job
):
    response = auth_client.get(f"/evaluations/{successful_job.id}")

    assert response.status_code == 200
    assert f"/reports/{successful_job.id}" in response.text
    assert 'hx-trigger="every 2s"' not in response.text


def test_failed_evaluation_detail_shows_retry_control(auth_client, db_engine, evaluation_job):
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "FAILED"
        job.error_code = "EVALUATION_FAILED"
        job.error_message = "something went wrong"

    response = auth_client.get(f"/evaluations/{evaluation_job.id}")

    assert response.status_code == 200
    assert f"/evaluations/{evaluation_job.id}/retry" in response.text
    assert "重试" in response.text
