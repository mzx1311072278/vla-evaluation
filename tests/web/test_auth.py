import base64
import json
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner
from sqlalchemy import update

from vla_eval.config import AppConfig
from vla_eval.db import session_scope
from vla_eval.models import User

from .conftest import extract_csrf


def test_passwords_are_hashed_with_argon2_and_verified():
    from vla_eval.security import hash_password, verify_password

    password_hash = hash_password("secret")

    assert password_hash.startswith("$argon2")
    assert verify_password("secret", password_hash) is True
    assert verify_password("wrong", password_hash) is False


def test_verify_password_treats_malformed_hash_as_mismatch():
    from vla_eval.security import verify_password

    assert verify_password("secret", "not-a-password-hash") is False


@pytest.mark.parametrize("password", [None, 7, True, b"secret"])
def test_hash_password_rejects_non_string_values(password):
    from vla_eval.security import hash_password

    with pytest.raises(TypeError, match="password must be a string"):
        hash_password(password)


def test_hash_password_rejects_empty_value():
    from vla_eval.security import hash_password

    with pytest.raises(ValueError, match="password must not be empty"):
        hash_password("")


@pytest.mark.parametrize(
    ("password", "password_hash", "message"),
    [
        (None, "$argon2id$placeholder", "password must be a string"),
        ("secret", None, "password_hash must be a string"),
        ("", "$argon2id$placeholder", "password must not be empty"),
        ("secret", "", "password_hash must not be empty"),
    ],
)
def test_verify_password_rejects_invalid_values(password, password_hash, message):
    from vla_eval.security import verify_password

    expected_error = TypeError if "string" in message else ValueError
    with pytest.raises(expected_error, match=message):
        verify_password(password, password_hash)


def test_create_app_records_injected_dependencies(data_root, db_engine, fake_queues):
    from vla_eval.web.app import create_app

    config = AppConfig(
        data_root=data_root,
        database_url="sqlite://",
        redis_url="redis://unused.invalid/0",
        session_secret="test-session-secret",
        remote_sources={},
    )

    app = create_app(config, db_engine, fake_queues)

    assert app.title == "VLA Evaluation"
    assert app.state.config is config
    assert app.state.engine is db_engine
    assert app.state.queues is fake_queues


def test_create_app_rejects_missing_session_secret(data_root, db_engine, fake_queues):
    from vla_eval.web.app import create_app

    config = AppConfig(
        data_root=data_root,
        database_url="sqlite://",
        redis_url="redis://unused.invalid/0",
        session_secret=" ",
        remote_sources={},
    )

    with pytest.raises(ValueError, match="session_secret"):
        create_app(config, db_engine, fake_queues)


def test_login_page_sets_secure_session_cookie(client):
    response = client.get("/login")

    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "secure" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "max-age=43200" in cookie


def test_login_page_has_accessible_form_and_fresh_csrf(client):
    first = client.get("/login")
    first_token = extract_csrf(first.text)
    client.cookies.clear()
    second = client.get("/login")
    second_token = extract_csrf(second.text)

    assert first_token != second_token
    assert len(first_token) >= 32
    assert '<label for="username">' in first.text
    assert 'autocomplete="username"' in first.text
    assert '<label for="password">' in first.text
    assert 'autocomplete="current-password"' in first.text


def test_protected_page_redirects_to_login(client):
    response = client.get("/datasets", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_unknown_page_remains_not_found(client):
    assert client.get("/typo-that-does-not-exist").status_code == 404


def test_openapi_is_public(client):
    assert client.get("/openapi.json").status_code == 200


def test_login_rejects_missing_csrf(client, user):
    response = client.post("/login", data={"username": "alice", "password": "secret"})

    assert response.status_code == 403


@pytest.mark.parametrize("csrf_values", [["wrong"], ["错误"], ["first", "second"]])
def test_login_rejects_invalid_or_duplicate_csrf(client, user, csrf_values):
    response = client.post(
        "/login",
        content=urlencode(
            [
                ("username", "alice"),
                ("password", "secret"),
                *(("csrf_token", value) for value in csrf_values),
            ]
        ),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("username", "password"),
    [("alice", "wrong"), ("nobody", "wrong"), ("<b>alice</b>", "secret")],
)
def test_login_failure_is_generic_and_does_not_echo_credentials(client, user, username, password):
    page = client.get("/login")
    csrf_token = extract_csrf(page.text)

    response = client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": csrf_token},
    )

    assert response.status_code == 401
    assert "用户名或密码无效" in response.text
    assert username not in response.text
    assert password not in response.text
    assert 'role="alert"' in response.text


@pytest.mark.parametrize("stored_hash", ["", "not-a-password-hash"])
def test_login_with_invalid_stored_hash_is_generic_failure(app, user, db_engine, stored_hash):
    with session_scope(db_engine) as session:
        session.execute(update(User).where(User.id == user.id).values(password_hash=stored_hash))

    with TestClient(
        app,
        base_url="https://testserver",
        raise_server_exceptions=False,
    ) as safe_client:
        page = safe_client.get("/login")
        response = safe_client.post(
            "/login",
            data={
                "username": "alice",
                "password": "secret",
                "csrf_token": extract_csrf(page.text),
            },
        )

    assert response.status_code == 401
    assert "用户名或密码无效" in response.text
    assert "alice" not in response.text
    assert "secret" not in response.text


def test_unknown_user_still_runs_dummy_password_verification(db_engine, monkeypatch):
    from vla_eval import security

    verified_hashes = []

    def record_verify(password, password_hash):
        verified_hashes.append((password, password_hash))
        return False

    monkeypatch.setattr(security, "verify_password", record_verify)

    assert security.authenticate_user(db_engine, "nobody", "secret") is None
    assert len(verified_hashes) == 1
    assert verified_hashes[0][0] == "secret"
    assert verified_hashes[0][1].startswith("$argon2")


def test_inactive_user_cannot_log_in(client, user, db_engine):
    with session_scope(db_engine) as session:
        session.execute(update(User).where(User.id == user.id).values(active=False))
    page = client.get("/login")

    response = client.post(
        "/login",
        data={
            "username": "alice",
            "password": "secret",
            "csrf_token": extract_csrf(page.text),
        },
    )

    assert response.status_code == 401
    assert "用户名或密码无效" in response.text


def test_login_success_redirects_and_session_contains_only_safe_values(client, user):
    page = client.get("/login")
    old_csrf = extract_csrf(page.text)

    response = client.post(
        "/login",
        data={"username": "alice", "password": "secret", "csrf_token": old_csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/datasets"
    payload = client.cookies["session"].split(".", maxsplit=1)[0]
    session_data = json.loads(base64.b64decode(payload))
    assert set(session_data) == {"user_id", "csrf_token"}
    assert session_data["user_id"] == user.id
    serialized = json.dumps(session_data)
    assert "secret" not in serialized
    assert "argon2" not in serialized
    assert session_data["csrf_token"] != old_csrf


def test_logged_in_user_is_redirected_from_login(auth_client):
    response = auth_client.get("/login", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/datasets"


def test_login_ignores_external_next_target(client, user):
    page = client.get("/login")

    response = client.post(
        "/login",
        data={
            "username": "alice",
            "password": "secret",
            "csrf_token": extract_csrf(page.text),
            "next": "https://evil.example/steal",
        },
        follow_redirects=False,
    )

    assert response.headers["location"] == "/datasets"


def test_tampered_cookie_does_not_authenticate(auth_client):
    session_cookie = auth_client.cookies["session"]
    auth_client.cookies.set("session", session_cookie + "tampered")

    response = auth_client.get("/datasets", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_expired_cookie_does_not_authenticate(client, user):
    class ExpiredSigner(TimestampSigner):
        def get_timestamp(self):
            return 1

    payload = base64.b64encode(
        json.dumps({"user_id": user.id, "csrf_token": "expired-token"}).encode()
    )
    cookie = ExpiredSigner("test-session-secret").sign(payload).decode()
    client.cookies.set("session", cookie)

    response = client.get("/datasets", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@pytest.mark.parametrize("change", ["delete", "deactivate"])
def test_session_stops_working_when_user_is_removed_or_inactive(
    auth_client, user, db_engine, change
):
    with session_scope(db_engine) as session:
        persisted = session.get_one(User, user.id)
        if change == "delete":
            session.delete(persisted)
        else:
            persisted.active = False

    response = auth_client.get("/datasets", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_logout_requires_post_and_valid_current_csrf(auth_client):
    assert auth_client.get("/logout").status_code in {404, 405}
    assert auth_client.post("/logout", data={}).status_code == 403
    assert auth_client.post("/logout", data={"csrf_token": "wrong"}).status_code == 403

    response = auth_client.post(
        "/logout",
        data={"csrf_token": auth_client.csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert auth_client.get("/datasets", follow_redirects=False).status_code == 303


def test_logout_rejects_duplicate_csrf(auth_client):
    response = auth_client.post(
        "/logout",
        content=urlencode([("csrf_token", auth_client.csrf), ("csrf_token", auth_client.csrf)]),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 403


def test_login_rotates_csrf_and_old_token_cannot_log_out(client, user):
    page = client.get("/login")
    login_csrf = extract_csrf(page.text)
    client.post(
        "/login",
        data={"username": "alice", "password": "secret", "csrf_token": login_csrf},
        follow_redirects=False,
    )
    current_csrf = extract_csrf(client.get("/datasets").text)

    assert current_csrf != login_csrf
    assert client.post("/logout", data={"csrf_token": login_csrf}).status_code == 403
    assert (
        client.post(
            "/logout",
            data={"csrf_token": current_csrf},
            follow_redirects=False,
        ).status_code
        == 303
    )


def test_successful_job_fixture_has_deterministic_artifacts(successful_job):
    output_dir = successful_job.output_dir
    assert output_dir is not None
    metrics_path = Path(output_dir) / "metrics_core.json"
    csv_path = Path(output_dir) / "episode_metrics.csv"

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "episode_count": 1,
        "success_rate": 1.0,
    }
    assert csv_path.read_text(encoding="utf-8") == "episode_index,success\n0,True\n"
