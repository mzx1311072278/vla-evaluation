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

import csv
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
                    "left": {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "n_episodes": 1,
                    },
                    "right": {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "n_episodes": 0,
                    },
                    "n_episodes": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (output_dir / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "session_id",
                "episode_index",
                "outcome",
                "duration_s",
                "smoothness",
                "left_smoothness",
                "right_smoothness",
                "smoothness_space",
                "smoothness_frames",
                "smoothness_skipped_reason",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "ready-dataset",
                "episode_index": 0,
                "outcome": "success",
                "duration_s": "1.000",
                "smoothness": "0",
                "left_smoothness": "0",
                "right_smoothness": "",
                "smoothness_space": "joint",
                "smoothness_frames": 4,
                "smoothness_skipped_reason": "",
            }
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


@pytest.mark.parametrize("viewport", _VIEWPORTS)
def test_list_toolbars_and_row_actions_are_contained(
    page,
    live_server,
    ready_dataset,
    report_job,
    viewport,
):
    page.set_viewport_size(viewport)
    login(page, live_server)

    for path in ("/datasets", "/evaluations"):
        page.goto(f"{live_server}{path}")
        page.wait_for_load_state("networkidle")
        toolbar = page.locator(".list-toolbar")
        assert toolbar.count() == 1
        assert toolbar.is_visible()
        assert page.evaluate(_NO_CHROME_OVERFLOW_JS) is False
        measurements = toolbar.locator("input, select, button, a").evaluate_all(
            """
            (controls) => controls.map((control) => {
              const rect = control.getBoundingClientRect();
              return {left: rect.left, right: rect.right, width: rect.width};
            })
            """
        )
        assert all(
            item["left"] >= -1
            and item["right"] <= viewport["width"] + 1
            and item["width"] > 0
            for item in measurements
        ), measurements
        action_measurements = page.locator(".row-actions").evaluate_all(
            """
            (actions) => actions.map((action) => ({
              clientWidth: action.clientWidth,
              scrollWidth: action.scrollWidth,
              childrenContained: Array.from(action.children).every((child) => {
                const parentRect = action.getBoundingClientRect();
                const childRect = child.getBoundingClientRect();
                return childRect.left >= parentRect.left - 1 &&
                  childRect.right <= parentRect.right + 1;
              }),
            }))
            """
        )
        assert all(
            item["scrollWidth"] <= item["clientWidth"] + 1
            and item["childrenContained"]
            for item in action_measurements
        ), action_measurements


@pytest.mark.parametrize("viewport", _VIEWPORTS)
def test_report_sections_and_formulas_are_contained(
    page,
    live_server,
    report_job,
    viewport,
):
    page.set_viewport_size(viewport)
    login(page, live_server)
    page.goto(f"{live_server}/reports/{report_job.id}")
    page.wait_for_load_state("networkidle")

    section_ids = (
        "report-summary",
        "report-configuration",
        "report-sources",
        "report-quality",
        "report-metrics",
        "report-episodes",
        "report-components",
        "report-gaps",
        "report-downloads",
    )
    for section_id in section_ids:
        section = page.locator(f"#{section_id}")
        assert section.count() == 1
        section.scroll_into_view_if_needed()
        assert section.is_visible()

    assert page.evaluate(_NO_CHROME_OVERFLOW_JS) is False
    formula_measurements = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('.formula-row')).map((row) => {
          const container = row.closest('.table-scroll');
          const rowRect = row.getBoundingClientRect();
          const containerRect = container?.getBoundingClientRect();
          return {
            rowRight: rowRect.right,
            rowClientWidth: row.clientWidth,
            rowScrollWidth: row.scrollWidth,
            containerLeft: containerRect?.left,
            containerClientWidth: container?.clientWidth,
            containerScrollWidth: container?.scrollWidth,
          };
        })
        """
    )
    assert all(
        item["rowRight"]
        <= item["containerLeft"] + item["containerScrollWidth"] + 1
        and item["rowScrollWidth"] <= item["rowClientWidth"] + 1
        for item in formula_measurements
    ), formula_measurements
    assert page.evaluate(
        """
        () => {
          const controls = Array.from(
            document.querySelectorAll('.page-heading .button, .report-filter-form button')
          ).filter((item) => item.getClientRects().length > 0);
          return controls.every((item, index) => {
            const a = item.getBoundingClientRect();
            return controls.slice(index + 1).every((other) => {
              const b = other.getBoundingClientRect();
              return a.right <= b.left || b.right <= a.left ||
                a.bottom <= b.top || b.bottom <= a.top;
            });
          });
        }
        """
    )


def test_smoothness_ticks_do_not_overlap_and_follow_the_selected_window(
    page,
    live_server,
    report_job,
    ready_dataset,
):
    episode_count = 199
    session_path = Path(ready_dataset.path) / "session.json"
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    session_data["num_episodes_target"] = episode_count
    session_path.write_text(json.dumps(session_data), encoding="utf-8")

    def expand_csv(path, update_row):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            template = next(reader)
        assert fieldnames is not None
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(update_row(dict(template), index) for index in range(199))

    expand_csv(
        Path(ready_dataset.path) / "episodes.csv",
        lambda row, index: {
            **row,
            "episode_index": str(index),
            "trajectory_path": f"trajectories/episode_{index:03d}.npz",
        },
    )
    output_dir = Path(report_job.output_dir)
    expand_csv(
        output_dir / "episode_metrics.csv",
        lambda row, index: {
            **row,
            "episode_index": str(index),
            "smoothness": str(4.5 + index % 20 / 40),
            "left_smoothness": str(4.5 + index % 20 / 40),
        },
    )
    metrics_path = output_dir / "metrics_core.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        n_episodes=episode_count,
        n_success=episode_count,
        n_failure=0,
    )
    metrics["smoothness"]["n_episodes"] = episode_count
    metrics["smoothness"]["left"]["n_episodes"] = episode_count
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    report_url = f"{live_server}/reports/{report_job.id}"
    page.set_viewport_size({"width": 1440, "height": 1000})
    login(page, live_server)
    page.goto(report_url)
    page.wait_for_load_state("networkidle")

    page.locator("[data-chart-window]").select_option("50")
    tick_labels = page.locator("[data-smoothness-chart] .chart-axis-label").filter(
        has_text="Ep "
    )
    assert tick_labels.all_text_contents()[0] == "Ep 0"
    assert tick_labels.all_text_contents()[-1] == "Ep 49"

    page.locator("[data-chart-start]").evaluate(
        """
        (input) => {
          input.value = "149";
          input.dispatchEvent(new Event("input", {bubbles: true}));
        }
        """
    )
    assert page.locator("[data-chart-range-label]").text_content() == (
        "Episode 149–198 / 199"
    )
    assert tick_labels.all_text_contents()[0] == "Ep 149"
    assert tick_labels.all_text_contents()[-1] == "Ep 198"

    boxes = tick_labels.evaluate_all(
        """
        (labels) => labels.map((label) => {
          const box = label.getBoundingClientRect();
          return {left: box.left, right: box.right};
        })
        """
    )
    assert all(
        boxes[index]["right"] + 4 <= boxes[index + 1]["left"]
        for index in range(len(boxes) - 1)
    ), boxes
