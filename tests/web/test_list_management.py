from urllib.parse import urlsplit

import pytest
from fastapi import HTTPException, Request

from vla_eval.web.list_management import (
    ListControls,
    literal_contains_pattern,
    parse_list_controls,
    validate_return_to,
)


def _request(path: str) -> Request:
    parsed = urlsplit(path)
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": parsed.path,
            "query_string": parsed.query.encode("ascii"),
            "headers": [],
        }
    )


def test_parse_list_controls_normalizes_keyword_and_defaults():
    controls = parse_list_controls(_request("/datasets?q=%20Robot%20"))

    assert controls == ListControls(
        q="Robot",
        sort="newest",
        include_archived=False,
    )


def test_parse_list_controls_accepts_explicit_values_and_allowed_extras():
    request = _request(
        "/evaluations?q=robot&sort=name_desc&archived=1&state=FAILED&dataset_id=abc"
    )

    controls = parse_list_controls(
        request,
        allowed_extra=frozenset({"state", "dataset_id"}),
    )

    assert controls == ListControls(
        q="robot",
        sort="name_desc",
        include_archived=True,
    )


@pytest.mark.parametrize(
    "query",
    [
        "sort=unknown",
        "archived=true",
        "q=" + "x" * 201,
        "q=one&q=two",
        "sort=newest&sort=oldest",
        "unknown=value",
    ],
)
def test_parse_list_controls_rejects_invalid_or_duplicate_values(query: str):
    with pytest.raises(HTTPException) as captured:
        parse_list_controls(_request(f"/datasets?{query}"))

    assert captured.value.status_code == 422


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("robot", "%robot%"),
        ("100%", r"%100\%%"),
        ("a_b", r"%a\_b%"),
        (r"a\b", r"%a\\b%"),
    ],
)
def test_literal_contains_pattern_escapes_like_metacharacters(
    value: str,
    expected: str,
):
    assert literal_contains_pattern(value) == expected


@pytest.mark.parametrize("value", [None, 7, b"robot"])
def test_literal_contains_pattern_rejects_non_strings(value):
    with pytest.raises(TypeError, match="search value must be a string"):
        literal_contains_pattern(value)


def test_validate_return_to_accepts_local_allowed_path_with_query():
    assert (
        validate_return_to(
            "/datasets?q=robot&sort=oldest",
            allowed_paths=frozenset({"/datasets"}),
            fallback="/datasets",
        )
        == "/datasets?q=robot&sort=oldest"
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.example/",
        "//evil.example/",
        "/evaluations",
        "/datasets#fragment",
        "/datasets\nX-Header: bad",
        "datasets",
    ],
)
def test_validate_return_to_falls_back_for_external_or_disallowed_targets(value: str):
    assert (
        validate_return_to(
            value,
            allowed_paths=frozenset({"/datasets"}),
            fallback="/datasets",
        )
        == "/datasets"
    )


@pytest.mark.parametrize("value", [None, 7, b"/datasets"])
def test_validate_return_to_falls_back_for_non_strings(value):
    assert (
        validate_return_to(
            value,
            allowed_paths=frozenset({"/datasets"}),
            fallback="/datasets",
        )
        == "/datasets"
    )
