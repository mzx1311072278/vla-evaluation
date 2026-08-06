"""Unit tests for the OpenAI-compatible VLM API backend (ApiVLMClient).

All HTTP is mocked with ``httpx.MockTransport`` -- no live server, no GPU. These
tests pin the request shape, the prompt-identity invariant with the local
backend, the retry policy, and the secret-handling rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# Reused vendored helpers, to assert the API backend sends byte-identical text.
from Genie02_report.attempt_eval.prompt_registry import prompt_for_version
from Genie02_report.attempt_eval.vlm_client import (
    _prompt_with_frame_times,
)
from vla_eval.exceptions import ModelLoadError
from vla_eval.vlm_api import ApiVLMClient

VALID_VLM = {
    "episode_success": True,
    "pre_success_failed_attempt_count": 0,
    "failed_attempts_before_success": [],
    "final_success_time": 3.5,
    "confidence": 0.9,
    "reason": "final grasp visible",
}

DEFAULTS = {
    "base_url": "https://vlm.example.internal/v1",
    "model": "qwen2.5-vl-7b-instruct",
    "api_key_env": "VLA_EVAL_VLM_API_KEY",
    "max_new_tokens": 256,
    "prompt_version": "genie02-attempt-v1",
    "timeout": 60,
    "max_retries": 3,
}


def _success_body() -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps(VALID_VLM)}}]}).encode()


def _write_frames(tmp_path: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    frame = tmp_path / "frame_000.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nfakepixels")
    timestamps = [
        {"frame_type": "global", "frame": "f0", "episode_time": 0.0, "video_time": 0.0},
        {"frame_type": "dense", "frame": "f1", "episode_time": 1.0, "video_time": 1.0},
    ]
    return [frame], timestamps


def _make_client(monkeypatch: pytest.MonkeyPatch, transport: Any, **overrides: Any) -> ApiVLMClient:
    monkeypatch.setenv("VLA_EVAL_VLM_API_KEY", "test-secret-key")
    kwargs = {**DEFAULTS, "transport": transport, **overrides}
    return ApiVLMClient(**kwargs)


def test_analyze_success_validates_result_and_sends_expected_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (seen.append(req), httpx.Response(200, content=_success_body()))[1])
    client = _make_client(monkeypatch, transport)

    frames, timestamps = _write_frames(tmp_path)
    result, valid = client.analyze(frames, timestamps, 3.5)

    assert valid is True
    assert result["episode_success"] is True
    assert result["confidence"] == 0.9
    assert result["vlm_valid"] is True

    request = seen[0]
    assert str(request.url) == "https://vlm.example.internal/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-secret-key"
    body = json.loads(request.content)
    assert body["model"] == "qwen2.5-vl-7b-instruct"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0
    message = body["messages"][0]
    assert message["role"] == "user"
    image_parts = [c for c in message["content"] if c["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_analyze_sends_identical_prompt_text_to_local_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (seen.append(req), httpx.Response(200, content=_success_body()))[1])
    client = _make_client(monkeypatch, transport)

    frames, timestamps = _write_frames(tmp_path)
    client.analyze(frames, timestamps)

    body = json.loads(seen[0].content)
    text_parts = [c for c in body["messages"][0]["content"] if c["type"] == "text"]
    expected = _prompt_with_frame_times(timestamps, prompt_for_version("genie02-attempt-v1"))
    assert text_parts == [{"type": "text", "text": expected}]


def test_analyze_returns_fallback_on_non_json_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    body = json.dumps({"choices": [{"message": {"content": "not json {incomplete}"}}]}).encode()
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    client = _make_client(monkeypatch, transport)

    result, valid = client.analyze(*_write_frames(tmp_path))

    assert valid is False
    assert result["vlm_valid"] is False


def test_analyze_returns_fallback_on_malformed_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b'{"oops": true}'))
    client = _make_client(monkeypatch, transport)

    result, valid = client.analyze(*_write_frames(tmp_path))

    assert valid is False
    assert result["vlm_valid"] is False


def test_post_retries_5xx_then_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[httpx.Request] = []
    sleeps: list[float] = []
    monkeypatch.setattr("vla_eval.vlm_api.time.sleep", lambda s: sleeps.append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(500)

    client = _make_client(monkeypatch, httpx.MockTransport(handler), max_retries=3)

    with pytest.raises(httpx.HTTPStatusError):
        client.analyze(*_write_frames(tmp_path))
    assert len(attempts) == 4  # 1 initial + 3 retries
    assert len(sleeps) == 3


def test_post_retries_429_then_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[httpx.Request] = []
    monkeypatch.setattr("vla_eval.vlm_api.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429)

    client = _make_client(monkeypatch, httpx.MockTransport(handler), max_retries=2)

    with pytest.raises(httpx.HTTPStatusError):
        client.analyze(*_write_frames(tmp_path))
    assert len(attempts) == 3


def test_post_raises_immediately_on_non_retryable_401(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    attempts: list[httpx.Request] = []
    sleeps: list[float] = []
    monkeypatch.setattr("vla_eval.vlm_api.time.sleep", lambda s: sleeps.append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(401, content=b'{"error":"bad credentials"}')

    client = _make_client(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        client.analyze(*_write_frames(tmp_path))
    assert len(attempts) == 1
    assert sleeps == []


def test_post_retries_transport_error_then_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    attempts: list[httpx.Request] = []
    monkeypatch.setattr("vla_eval.vlm_api.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("boom", request=request)

    client = _make_client(monkeypatch, httpx.MockTransport(handler), max_retries=2)

    with pytest.raises(httpx.ConnectError):
        client.analyze(*_write_frames(tmp_path))
    assert len(attempts) == 3


def test_401_error_does_not_leak_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    body = b'{"error":"invalid test-secret-key"}'
    transport = httpx.MockTransport(lambda req: httpx.Response(401, content=body))
    client = _make_client(monkeypatch, transport)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.analyze(*_write_frames(tmp_path))
    assert "test-secret-key" not in str(exc_info.value)


def test_missing_api_key_raises_model_load_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLA_EVAL_VLM_API_KEY", raising=False)
    transport = httpx.MockTransport(lambda req: httpx.Response(200))

    with pytest.raises(ModelLoadError, match="VLA_EVAL_VLM_API_KEY"):
        ApiVLMClient(**{**DEFAULTS, "transport": transport})


def test_empty_api_key_raises_model_load_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLA_EVAL_VLM_API_KEY", "")
    transport = httpx.MockTransport(lambda req: httpx.Response(200))

    with pytest.raises(ModelLoadError, match="VLA_EVAL_VLM_API_KEY"):
        ApiVLMClient(**{**DEFAULTS, "transport": transport})


def test_missing_httpx_raises_model_load_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLA_EVAL_VLM_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "httpx", None)

    with pytest.raises(ModelLoadError, match="vlm-api"):
        ApiVLMClient(**DEFAULTS)
