import secrets
from collections.abc import Sequence

from fastapi import HTTPException, Request
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import Engine, select

from vla_eval.db import session_scope
from vla_eval.models import User

_PASSWORD_HASH = PasswordHash.recommended()
_DUMMY_PASSWORD_HASH = _PASSWORD_HASH.hash("vla-evaluation-dummy-password")


def _validated_secret(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def hash_password(password: str) -> str:
    return _PASSWORD_HASH.hash(_validated_secret(password, "password"))


def verify_password(password: str, password_hash: str) -> bool:
    validated_password = _validated_secret(password, "password")
    validated_hash = _validated_secret(password_hash, "password_hash")
    try:
        return _PASSWORD_HASH.verify(validated_password, validated_hash)
    except (UnknownHashError, ValueError):
        return False


def authenticate_user(engine: Engine, username: str, password: str) -> User | None:
    validated_username = _validated_secret(username, "username")
    validated_password = _validated_secret(password, "password")
    with session_scope(engine) as session:
        user = session.scalar(select(User).where(User.username == validated_username))
        password_hash = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
        try:
            password_matches = verify_password(validated_password, password_hash)
        except (TypeError, ValueError):
            password_matches = False
        if user is None or not user.active or not password_matches:
            return None
        return user


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def require_csrf(request: Request, submitted_tokens: Sequence[object]) -> None:
    stored_token = request.session.get("csrf_token")
    if (
        not isinstance(stored_token, str)
        or not stored_token
        or len(submitted_tokens) != 1
        or not isinstance(submitted_tokens[0], str)
        or not submitted_tokens[0]
        or not secrets.compare_digest(stored_token, submitted_tokens[0])
    ):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


def get_current_user(request: Request) -> User | None:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    with session_scope(request.app.state.engine) as session:
        user = session.scalar(select(User).where(User.id == user_id, User.active.is_(True)))
    if user is None:
        request.session.clear()
    return user


def require_html_user(request: Request) -> User:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user
