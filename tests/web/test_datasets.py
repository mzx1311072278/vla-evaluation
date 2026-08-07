from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from vla_eval.db import session_scope
from vla_eval.models import Dataset, EvaluationJob


@pytest.mark.parametrize("path", ["/datasets", "/datasets/missing"])
def test_dataset_html_pages_require_login(client: TestClient, path: str):
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_dataset_list_and_ready_detail_show_evaluation_entry(auth_client, ready_dataset):
    listing = auth_client.get("/datasets")
    detail = auth_client.get(f"/datasets/{ready_dataset.id}")

    assert listing.status_code == detail.status_code == 200
    assert ready_dataset.name in listing.text
    assert "genie02_session" in detail.text
    assert "1" in detail.text
    assert str(ready_dataset.size_bytes) in detail.text
    assert f"/evaluations/new?dataset_id={ready_dataset.id}" in detail.text
    assert 'aria-disabled="true"' not in detail.text


def test_dataset_detail_shows_only_five_most_recent_evaluations(
    auth_client, db_engine: Engine, ready_dataset, dataset
):
    with session_scope(db_engine) as session:
        for index in range(6):
            session.add(
                EvaluationJob(
                    dataset_id=ready_dataset.id,
                    profile_name=f"recent-{index}",
                    state="SUCCEEDED",
                )
            )
            session.flush()
        session.add(
            EvaluationJob(
                dataset_id=dataset.id,
                profile_name="other-dataset-job",
                state="FAILED",
            )
        )

    detail = auth_client.get(f"/datasets/{ready_dataset.id}")

    assert detail.status_code == 200
    assert "最近评测" in detail.text
    assert "recent-5" in detail.text
    assert "recent-1" in detail.text
    assert "recent-0" not in detail.text
    assert "other-dataset-job" not in detail.text
    assert f'href="/evaluations?dataset_id={ready_dataset.id}"' in detail.text


def test_non_ready_dataset_detail_disables_evaluation_and_shows_preflight(
    auth_client, dataset, db_engine
):
    long_path = dataset.path + "/" + "nested/" * 20 + "dataset"
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, dataset.id)
        stored.path = long_path
        stored.inspection_json = {"errors": ["episodes.csv is missing"]}

    detail = auth_client.get(f"/datasets/{dataset.id}")

    assert detail.status_code == 200
    assert 'aria-disabled="true"' in detail.text
    assert f"/evaluations/new?dataset_id={dataset.id}" not in detail.text
    assert "episodes.csv is missing" in detail.text
    assert long_path in detail.text
    assert 'data-lucide="copy"' in detail.text
    assert 'title="\u590d\u5236\u8def\u5f84"' in detail.text


def test_missing_dataset_is_not_found(auth_client):
    assert auth_client.get("/datasets/not-a-dataset").status_code == 404


def _dataset_files(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def _archive_dataset(client, dataset_id: str, *, return_to: str = "/datasets"):
    return client.post(
        f"/datasets/{dataset_id}/archive",
        data={"csrf_token": client.csrf, "return_to": return_to},
        follow_redirects=False,
    )


def _restore_dataset(client, dataset_id: str, *, return_to: str = "/datasets"):
    return client.post(
        f"/datasets/{dataset_id}/restore",
        data={"csrf_token": client.csrf, "return_to": return_to},
        follow_redirects=False,
    )


def test_archive_ready_dataset_preserves_files_and_records_restore_snapshot(
    auth_client, db_engine: Engine, ready_dataset, user
):
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, ready_dataset.id)
        stored.inspection_json = {"errors": ["historical warning"]}
    files_before = _dataset_files(ready_dataset.path)

    response = _archive_dataset(
        auth_client,
        ready_dataset.id,
        return_to="/datasets?q=ready",
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/datasets?q=ready"
    assert _dataset_files(ready_dataset.path) == files_before
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, ready_dataset.id)
        assert stored.status == "ARCHIVED"
        assert stored.inspection_json["errors"] == ["historical warning"]
        archive = stored.inspection_json["_archive"]
        assert archive["previous_status"] == "READY"
        assert archive["archived_by"] == user.id
        assert datetime.fromisoformat(archive["archived_at"]).tzinfo is not None


def test_archive_dataset_requires_one_valid_csrf_token(
    auth_client, db_engine: Engine, ready_dataset
):
    response = auth_client.post(
        f"/datasets/{ready_dataset.id}/archive",
        data={"return_to": "/datasets"},
    )
    duplicate = auth_client.post(
        f"/datasets/{ready_dataset.id}/archive",
        content=urlencode(
            [
                ("csrf_token", auth_client.csrf),
                ("csrf_token", auth_client.csrf),
                ("return_to", "/datasets"),
            ]
        ),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == duplicate.status_code == 403
    with session_scope(db_engine) as session:
        assert session.get_one(Dataset, ready_dataset.id).status == "READY"


@pytest.mark.parametrize(
    "data",
    [
        [("csrf_token", "{csrf}")],
        [("csrf_token", "{csrf}"), ("return_to", "/datasets"), ("extra", "1")],
        [
            ("csrf_token", "{csrf}"),
            ("return_to", "/datasets"),
            ("return_to", "/datasets?q=ready"),
        ],
    ],
)
def test_archive_dataset_rejects_unknown_or_duplicate_fields(
    auth_client, db_engine: Engine, ready_dataset, data
):
    submitted = [(key, auth_client.csrf if value == "{csrf}" else value) for key, value in data]

    response = auth_client.post(
        f"/datasets/{ready_dataset.id}/archive",
        content=urlencode(submitted),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    with session_scope(db_engine) as session:
        assert session.get_one(Dataset, ready_dataset.id).status == "READY"


def test_archive_dataset_rejects_repeat_and_uses_safe_return_fallback(
    auth_client, ready_dataset
):
    archived = _archive_dataset(
        auth_client,
        ready_dataset.id,
        return_to="https://evil.example/steal",
    )
    repeated = _archive_dataset(auth_client, ready_dataset.id)

    assert archived.status_code == 303
    assert archived.headers["location"] == "/datasets"
    assert repeated.status_code == 409


@pytest.mark.parametrize("state", ["QUEUED", "RUNNING"])
def test_archive_dataset_rejects_active_evaluation(
    auth_client, db_engine: Engine, ready_dataset, state: str
):
    with session_scope(db_engine) as session:
        session.add(
            EvaluationJob(
                dataset_id=ready_dataset.id,
                profile_name="active-profile",
                state=state,
            )
        )

    response = _archive_dataset(auth_client, ready_dataset.id)

    assert response.status_code == 409
    with session_scope(db_engine) as session:
        assert session.get_one(Dataset, ready_dataset.id).status == "READY"


def test_restore_dataset_restores_previous_status_and_preserves_metadata(
    auth_client, db_engine: Engine, ready_dataset
):
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, ready_dataset.id)
        stored.status = "ARCHIVED"
        stored.inspection_json = {
            "errors": ["historical warning"],
            "_archive": {
                "previous_status": "READY",
                "archived_at": "2026-08-07T08:00:00+00:00",
                "archived_by": "user-id",
            },
        }

    response = _restore_dataset(
        auth_client,
        ready_dataset.id,
        return_to=f"/datasets/{ready_dataset.id}",
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/datasets/{ready_dataset.id}"
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, ready_dataset.id)
        assert stored.status == "READY"
        assert stored.inspection_json == {"errors": ["historical warning"]}


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"_archive": "invalid"},
        {"_archive": {"previous_status": "PENDING"}},
        {"_archive": {"previous_status": "READY"}},
    ],
)
def test_restore_dataset_rejects_missing_or_corrupt_archive_metadata(
    auth_client, db_engine: Engine, ready_dataset, metadata
):
    with session_scope(db_engine) as session:
        stored = session.get_one(Dataset, ready_dataset.id)
        stored.status = "ARCHIVED"
        stored.inspection_json = metadata

    response = _restore_dataset(auth_client, ready_dataset.id)

    assert response.status_code == 409
    with session_scope(db_engine) as session:
        assert session.get_one(Dataset, ready_dataset.id).status == "ARCHIVED"


def _upload(client, dataset_id: str, filename: str, content: bytes, csrf: str | None = None):
    return client.post(
        f"/datasets/{dataset_id}/attachments",
        data={"csrf_token": client.csrf if csrf is None else csrf},
        files={"file": (filename, content, "application/octet-stream")},
        follow_redirects=False,
    )


def test_attachment_upload_requires_csrf(auth_client, ready_dataset):
    response = _upload(auth_client, ready_dataset.id, "meta.json", b"{}", csrf="wrong")

    assert response.status_code == 403
    assert not (Path(ready_dataset.path) / "_attachments" / "meta.json").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "notes.txt",
        "archive.json.exe",
        "../evil.json",
        "nested/file.yaml",
        "nested\\file.csv",
        ".",
        "..",
        "bad\x00.json",
        "bad\u2028name.json",
    ],
)
def test_attachment_rejects_invalid_extension_or_filename(auth_client, ready_dataset, filename):
    response = _upload(auth_client, ready_dataset.id, filename, b"content")

    assert response.status_code == 422
    attachments = Path(ready_dataset.path) / "_attachments"
    assert not attachments.exists() or not any(
        child.name not in {".upload.lock"} for child in attachments.iterdir()
    )


@pytest.mark.parametrize("filename", ["meta.json", "config.yaml", "config.YML", "TABLE.CSV"])
def test_attachment_accepts_supported_extensions_case_insensitively(
    auth_client, ready_dataset, filename
):
    response = _upload(auth_client, ready_dataset.id, filename, b"attachment")

    assert response.status_code == 303
    assert response.headers["location"] == f"/datasets/{ready_dataset.id}"
    assert (Path(ready_dataset.path) / "_attachments" / filename).read_bytes() == b"attachment"


def test_attachment_does_not_overwrite_existing_attachment_or_original_data(
    auth_client, ready_dataset
):
    dataset_root = Path(ready_dataset.path)
    original = dataset_root / "metadata.json"
    original.write_bytes(b"original")
    attachments = dataset_root / "_attachments"
    attachments.mkdir()
    existing = attachments / "metadata.json"
    existing.write_bytes(b"first")

    response = _upload(auth_client, ready_dataset.id, "metadata.json", b"second")

    assert response.status_code == 409
    assert original.read_bytes() == b"original"
    assert existing.read_bytes() == b"first"
    assert not any(child.name.startswith(".upload-") for child in attachments.iterdir())


def test_attachment_rejects_file_larger_than_twenty_mib_without_partial_file(
    auth_client, ready_dataset, monkeypatch
):
    from vla_eval.web import routes_datasets

    store_calls = []
    monkeypatch.setattr(
        routes_datasets,
        "_store_attachment",
        lambda *_args: store_calls.append(_args),
    )
    response = _upload(auth_client, ready_dataset.id, "large.json", b"x" * (20 * 1024 * 1024 + 1))

    assert response.status_code == 413
    assert store_calls == []
    attachments = Path(ready_dataset.path) / "_attachments"
    assert not (attachments / "large.json").exists()
    assert not attachments.exists() or not any(
        child.name.startswith(".upload-") for child in attachments.iterdir()
    )


def test_attachment_rejects_dataset_total_larger_than_one_hundred_mib(auth_client, ready_dataset):
    attachments = Path(ready_dataset.path) / "_attachments"
    attachments.mkdir()
    with (attachments / "existing.csv").open("wb") as handle:
        handle.truncate(90 * 1024 * 1024)

    response = _upload(auth_client, ready_dataset.id, "more.json", b"x" * (11 * 1024 * 1024))

    assert response.status_code == 413
    assert not (attachments / "more.json").exists()
    assert not any(child.name.startswith(".upload-") for child in attachments.iterdir())


def test_attachment_rejects_dataset_path_outside_inbox(auth_client, dataset, db_engine, data_root):
    outside = data_root / "outside"
    outside.mkdir()
    with session_scope(db_engine) as session:
        session.get_one(Dataset, dataset.id).path = str(outside)

    response = _upload(auth_client, dataset.id, "meta.json", b"{}")

    assert response.status_code == 422
    assert not (outside / "_attachments").exists()


def test_attachment_maps_malformed_stored_dataset_path_to_validation_error(
    auth_client, dataset, db_engine
):
    with session_scope(db_engine) as session:
        session.get_one(Dataset, dataset.id).path = "/bad\x00path"

    response = _upload(auth_client, dataset.id, "meta.json", b"{}")

    assert response.status_code == 422


def test_attachment_rejects_attachment_directory_symlink_escape(
    auth_client, ready_dataset, data_root
):
    outside = data_root / "outside-attachments"
    outside.mkdir()
    (Path(ready_dataset.path) / "_attachments").symlink_to(outside, target_is_directory=True)

    response = _upload(auth_client, ready_dataset.id, "meta.json", b"{}")

    assert response.status_code == 422
    assert not (outside / "meta.json").exists()


def test_attachment_rejects_dataset_root_symlink_even_when_target_stays_in_inbox(
    auth_client, dataset, db_engine, data_root
):
    target = data_root / "inbox" / "other-dataset"
    target.mkdir()
    alias = data_root / "inbox" / "dataset-alias"
    alias.symlink_to(target, target_is_directory=True)
    with session_scope(db_engine) as session:
        session.get_one(Dataset, dataset.id).path = str(alias)

    response = _upload(auth_client, dataset.id, "meta.json", b"{}")

    assert response.status_code == 422
    assert not (target / "_attachments").exists()


def test_attachment_same_name_concurrency_has_one_winner(auth_client, ready_dataset):
    def upload(content: bytes):
        response = _upload(auth_client, ready_dataset.id, "race.json", content)
        return response.status_code, response.text

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(upload, [b"first", b"second"]))

    assert sorted(status for status, _text in responses) == [303, 409], responses
    stored = (Path(ready_dataset.path) / "_attachments" / "race.json").read_bytes()
    assert stored in {b"first", b"second"}


def test_attachment_rejects_duplicate_file_fields(auth_client, ready_dataset):
    response = auth_client.post(
        f"/datasets/{ready_dataset.id}/attachments",
        data={"csrf_token": auth_client.csrf},
        files=[("file", ("one.json", b"1")), ("file", ("two.json", b"2"))],
    )

    assert response.status_code == 422


def test_base_layout_loads_static_responsive_styles_and_keeps_login_usable(client):
    login = client.get("/login")
    stylesheet = client.get("/static/app.css")

    assert login.status_code == 200
    assert 'href="/static/app.css"' in login.text
    assert 'autocomplete="username"' in login.text
    assert stylesheet.status_code == 200
    assert "overflow-wrap: anywhere" in stylesheet.text
    assert "@media (max-width: 720px)" in stylesheet.text


def test_authenticated_layout_has_dataset_import_and_evaluation_navigation(auth_client):
    response = auth_client.get("/datasets")

    assert 'href="/datasets"' in response.text
    assert 'href="/imports"' in response.text
    assert 'href="/evaluations"' in response.text
    assert "htmx.org" in response.text
    assert "lucide" in response.text
