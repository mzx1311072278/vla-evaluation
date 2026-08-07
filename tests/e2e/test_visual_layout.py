"""Playwright desktop + mobile layout tests (Task 16, Step 2).

Drives the REAL FastAPI app (built via ``create_app``) under a real uvicorn
server over HTTPS (self-signed) -- the production ``SessionMiddleware`` marks
the session cookie ``Secure``, so a plaintext HTTP server would lose the
session in a real browser. Chromium accepts the self-signed cert via
``ignore_https_errors``.

These tests skip gracefully when the Chromium browser binary is unavailable
(``playwright install chromium`` not run), so the suite stays green on machines
without the browser installed.
"""

from __future__ import annotations

import json
import socket
import ssl
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from vla_eval.db import session_scope
from vla_eval.models import EvaluationJob
from vla_eval.web.app import create_app

_VIEWPORTS = [
    pytest.param({"width": 1440, "height": 1000}, id="desktop-1440"),
    pytest.param({"width": 390, "height": 844}, id="mobile-390"),
]


def _free_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]
    finally:
        listener.close()


def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def _wait_for_server(base_url: str, timeout: float = 15.0) -> None:
    """Poll until the server serves a page.

    ``/health`` is intentionally avoided: it pings Redis, which the in-process
    ``FakeQueueBundle`` cannot satisfy, so it reports 503 even though every page
    route works. ``/login`` renders without any backend dependency.
    """
    context = ssl._create_unverified_context()
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/login", context=context, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as error:  # noqa: BLE001 - retry until the server answers
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"uvicorn did not become ready at {base_url}") from last_error


@pytest.fixture
def live_server(app_config, db_engine, fake_queues, user, tmp_path):
    """Start the real app over HTTPS on an ephemeral port in a background thread."""
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    _generate_self_signed_cert(cert_path, key_path)

    app = create_app(app_config, db_engine, fake_queues)
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"https://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def page():
    """A Playwright Chromium page, skipping gracefully without a browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        pytest.skip(f"playwright not installed: {error}")
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as error:  # noqa: BLE001 - browser binary missing
            pytest.skip(f"chromium could not launch: {error}")
        try:
            context = browser.new_context(ignore_https_errors=True)
            yield context.new_page()
        finally:
            browser.close()


def login(page, live_server: str) -> None:
    """Authenticate through the real /login form, landing on /datasets."""
    page.goto(f"{live_server}/login")
    page.fill("#username", "alice")
    page.fill("#password", "secret")
    page.click("button[type='submit']")
    page.wait_for_url("**/datasets")


@pytest.fixture
def report_job(db_engine, ready_dataset, user, tmp_path):
    output_dir = tmp_path / "report-output"
    output_dir.mkdir()
    (output_dir / "metrics_core.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "session_id": "ready-dataset",
                "n_episodes": 1,
                "n_success": 1,
                "n_failure": 0,
                "gsr": 1.0,
                "mean_tts_success_s": 1.0,
                "smoothness": {
                    "space": "joint",
                    "left": {"mean": 0.0, "n_episodes": 1},
                    "right": {"mean": None, "n_episodes": 0},
                    "n_episodes": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="genie02-full",
            profile_version="1.0.0",
            state="SUCCEEDED",
            stage="REPORT",
            progress=100.0,
            output_dir=str(output_dir),
            provenance_json={
                "vlm_model_path": "vlm-model",
                "prompt_version": "prompt-v1",
                "app_version": "app-v1",
                "git_sha": "0123456789abcdef0123456789abcdef01234567",
            },
            created_by=user.id,
        )
        session.add(job)
        session.flush()
        return job


# A page "has no horizontal overflow" when its viewport-width chrome fits the
# viewport. The literal ``document.documentElement.scrollWidth`` check is the
# primary signal, but ``/imports`` and ``/datasets`` render their rows in a
# ``.table-scroll`` region that is *intentionally* horizontally scrollable on
# narrow viewports (``table { min-width: 46rem }`` inside
# ``.table-scroll { overflow-x: auto }``). Chromium's ``scrollWidth`` on the
# document element still reports that table's intrinsic 46rem width even though
# ``.table-scroll`` clips it and itself fits the viewport, so the raw check
# reports a false positive for that deliberate responsive-table pattern.
#
# The check below therefore:
#   1. passes immediately when ``scrollWidth <= clientWidth`` (no overflow), and
#   2. otherwise accepts the overflow only if it is fully contained inside
#      ``.table-scroll`` regions that each fit within the viewport -- i.e. the
#      page chrome never overflows. Any overflow originating outside a
#      ``.table-scroll`` (header, nav, headings, buttons) is treated as a real
#      layout defect and fails the test.
_NO_CHROME_OVERFLOW_JS = """
() => {
  const de = document.documentElement;
  const vw = de.clientWidth;
  if (de.scrollWidth <= vw) {
    return false;
  }
  const scrollers = Array.from(document.querySelectorAll('.table-scroll'));
  if (scrollers.length === 0) {
    return true;
  }
  for (const scroller of scrollers) {
    if (scroller.getBoundingClientRect().right > vw + 1) {
      return true;
    }
  }
  for (const el of document.querySelectorAll('body *')) {
    if (scrollers.some((scroller) => scroller.contains(el))) {
      continue;
    }
    if (el.getBoundingClientRect().right > vw + 1) {
      return true;
    }
  }
  return false;
}
"""


@pytest.mark.parametrize("viewport", _VIEWPORTS)
def test_core_pages_have_no_horizontal_overflow(
    page,
    live_server,
    ready_dataset,
    report_job,
    viewport,
):
    page.set_viewport_size(viewport)
    login(page, live_server)

    pages = [
        "/imports",
        "/datasets",
        f"/evaluations/new?dataset_id={ready_dataset.id}",
        f"/reports/{report_job.id}",
    ]
    for path in pages:
        page.goto(f"{live_server}{path}")
        page.wait_for_load_state("networkidle")
        if path.startswith("/reports/"):
            assert page.locator(".report-headline").is_visible()
        chrome_overflow = page.evaluate(_NO_CHROME_OVERFLOW_JS)
        assert chrome_overflow is False, (
            f"page-chrome horizontal overflow at viewport {page.viewport_size} on {path}"
        )
