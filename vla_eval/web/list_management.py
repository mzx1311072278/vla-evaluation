"""Shared validation for searchable, sortable Web lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

ListSort = Literal["newest", "oldest", "name_asc", "name_desc"]
LIST_SORTS = frozenset({"newest", "oldest", "name_asc", "name_desc"})
_COMMON_FIELDS = frozenset({"q", "sort", "archived"})
_MAX_QUERY_LENGTH = 200


@dataclass(frozen=True)
class ListControls:
    q: str
    sort: ListSort
    include_archived: bool


def parse_list_controls(
    request: Request,
    *,
    allowed_extra: frozenset[str] = frozenset(),
) -> ListControls:
    """Parse common list controls and reject ambiguous query strings."""
    allowed = _COMMON_FIELDS | allowed_extra
    if set(request.query_params) - allowed:
        raise HTTPException(status_code=422, detail="Invalid list filters")
    if any(len(request.query_params.getlist(field)) > 1 for field in allowed):
        raise HTTPException(status_code=422, detail="Invalid list filters")

    query = request.query_params.get("q", "").strip()
    sort = request.query_params.get("sort", "newest")
    archived = request.query_params.get("archived")
    if (
        len(query) > _MAX_QUERY_LENGTH
        or sort not in LIST_SORTS
        or archived not in {None, "1"}
    ):
        raise HTTPException(status_code=422, detail="Invalid list filters")
    return ListControls(
        q=query,
        sort=sort,
        include_archived=archived == "1",
    )


def literal_contains_pattern(value: str) -> str:
    """Build a LIKE pattern that treats user metacharacters literally."""
    if not isinstance(value, str):
        raise TypeError("search value must be a string")
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def validate_return_to(
    value: object,
    *,
    allowed_paths: frozenset[str],
    fallback: str,
) -> str:
    """Return a local allowlisted target or a caller-provided safe fallback."""
    if not isinstance(value, str) or any(ord(character) < 32 for character in value):
        return fallback
    parsed = urlsplit(value)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or not parsed.path.startswith("/")
        or parsed.path not in allowed_paths
    ):
        return fallback
    return value
