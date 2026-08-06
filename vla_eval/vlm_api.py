"""OpenAI-compatible VLM backend -- the second VLM interface.

Mirrors the duck-typed surface of the vendored ``LocalVLMClient``
(``analyze(frame_paths, frame_timestamps, episode_duration) -> (dict, bool)``
plus ``close()``) so it can be injected through ``run_attempt_evaluation``'s
``client_factory`` seam without modifying ``Genie02_report``. Instead of loading
Qwen2.5-VL on a GPU it POSTs the sampled frames and the *same* prompt text to an
OpenAI-compatible ``/chat/completions`` endpoint (a cloud provider, or a
self-hosted vLLM server). Prompt construction, JSON extraction, and result
validation are reused from the vendored library so both backends produce
identical prompts and identical result semantics -- in particular the VLM never
overrides the original episode success label (design spec section 9).

httpx is imported lazily so a deployment that never uses the API backend needs
only the ``vlm-api`` extra when it does; absence surfaces as
``ModelLoadError`` (mapped to ``MODEL_LOAD_FAILED`` by the task layer).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

# Reused, unmodified vendored helpers -- this is the only way to guarantee
# byte-identical prompts and result validation between backends without touching
# Genie02_report. _prompt_with_frame_times is module-private in the vendored
# library; importing it explicitly is intentional and pinned by the
# prompt-identity unit test.
from Genie02_report.attempt_eval.prompt_registry import prompt_for_version
from Genie02_report.attempt_eval.vlm_client import (
    _prompt_with_frame_times,
    extract_json,
    fallback_result,
    validate_vlm_result,
)
from vla_eval.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ApiVLMClient:
    """VLM client that calls an OpenAI-compatible vision endpoint over HTTP.

    The API key is read from the environment variable named by ``api_key_env``
    (the NAME is what profiles/provenance store; the VALUE never is). On a
    retryable failure (network error, 5xx, 429) the request is retried with
    exponential backoff up to ``max_retries`` times; on exhaustion -- or on a
    non-retryable client error (4xx other than 429) -- the error is re-raised so
    the service layer can sanitize it into a per-episode fallback, matching
    ``LocalVLMClient`` semantics.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str,
        max_new_tokens: int,
        prompt_version: str,
        timeout: float,
        max_retries: int,
        *,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.prompt = prompt_for_version(prompt_version)

        try:
            key = os.environ[api_key_env]
        except KeyError as error:
            raise ModelLoadError(
                f"VLM API key is not set in environment variable {api_key_env}"
            ) from error
        if not key:
            raise ModelLoadError(
                f"VLM API key is not set in environment variable {api_key_env}"
            )

        try:
            import httpx
        except ImportError as error:  # pragma: no cover - exercised via monkeypatch
            raise ModelLoadError(
                "The VLM API backend requires the 'vlm-api' extra (httpx); "
                "install with: pip install \".[vlm-api]\""
            ) from error
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        )

    @staticmethod
    def _data_url(path: Path) -> str:
        """Encode an on-disk sampled frame as a base64 data URL."""
        data = path.read_bytes()
        if data.startswith(b"\x89PNG"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _post(self, payload: dict[str, Any]) -> Any:
        """POST a chat-completion request, retrying transient failures.

        Returns the ``httpx.Response`` on a 2xx. Retries 429 and 5xx responses
        and ``httpx.TransportError`` (timeouts/connection errors) up to
        ``max_retries`` times, then re-raises the last error. Any other 4xx is
        raised immediately (non-retryable).
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except self._httpx.TransportError as error:
                last_error = error
            else:
                if 200 <= response.status_code < 300:
                    return response
                status_error = self._httpx.HTTPStatusError(
                    f"VLM API returned status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if response.status_code != 429 and response.status_code < 500:
                    raise status_error
                last_error = status_error
            if attempt < self.max_retries:
                time.sleep(min(2.0**attempt, 30.0))
        if last_error is None:  # only if max_retries < 0, which profiles forbid
            raise RuntimeError("retry loop exited without capturing an error")
        raise last_error

    def analyze(
        self,
        frame_paths: list[Path],
        frame_timestamps: list[dict[str, Any]],
        episode_duration: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Run one episode through the API and return (result, vlm_valid).

        Mirrors ``LocalVLMClient.analyze``: build the multimodal content, obtain
        raw model text, then reuse ``extract_json`` + ``validate_vlm_result`` so
        the parsed result is identical to what the local backend would produce.
        """
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": self._data_url(path)}}
            for path in frame_paths
        ]
        content.append(
            {"type": "text", "text": _prompt_with_frame_times(frame_timestamps, self.prompt)}
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }

        response = self._post(payload)
        try:
            raw = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return (
                fallback_result("", "invalid_vlm_response", "VLM API response was malformed"),
                False,
            )
        if not isinstance(raw, str):
            raw = str(raw)
        parsed, ok, parse_error = extract_json(raw)
        if not ok or parsed is None:
            warning = "empty_vlm_response" if not raw else "invalid_vlm_json"
            return fallback_result(raw, warning, parse_error), False
        result = validate_vlm_result(parsed, raw, episode_duration)
        return result, bool(result.get("vlm_valid"))

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.exception("VLM API client cleanup failed")
