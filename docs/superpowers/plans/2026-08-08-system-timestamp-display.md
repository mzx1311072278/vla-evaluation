# System Timestamp Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep persisted system timestamps in UTC while displaying all user-facing calendar times in 24-hour Beijing time with seconds, without modifying dataset contents.

**Architecture:** Add one standard-library time module for timezone conversion and formatting, and one shared Jinja environment that exposes it as a filter. Web routes continue to pass persisted models to templates; only validated archive timestamps need route-level parsing. Markdown generation uses the same time module and a single captured generation instant so report text and filenames cannot disagree around midnight.

**Tech Stack:** Python 3.11+, `datetime`, `zoneinfo`, FastAPI/Starlette `Jinja2Templates`, SQLAlchemy, Jinja2, pytest, Ruff.

---

## File Structure

- Create `vla_eval/time_utils.py`: define the Beijing timezone, timezone conversion, fixed display formatting, and current Beijing time.
- Create `vla_eval/web/templating.py`: own the shared `Jinja2Templates` instance and register the `beijing_time` filter.
- Modify `vla_eval/web/routes_auth.py`, `routes_datasets.py`, `routes_imports.py`, `routes_evaluations.py`, and `routes_reports.py`: import the shared template instance; parse only validated dataset archive timestamps for display.
- Modify `vla_eval/web/templates/datasets/index.html` and `detail.html`: show import-system, update, archive, and recent-evaluation times.
- Modify `vla_eval/web/templates/imports/index.html` and `detail.html`: show create and update times.
- Modify `vla_eval/web/templates/evaluations/index.html` and `detail.html`: format create/update/archive times.
- Modify `vla_eval/web/templates/reports/detail.html`: show task create and update times.
- Modify `Genie02_report/genie02_eval_common.py`: create synthetic LeRobot timestamps and default dated directories independently of the host timezone.
- Modify `Genie02_report/genie02_markdown_report.py`: show recorded/generated timestamps and date filenames in Beijing time.
- Create `tests/test_time_utils.py`: unit coverage for UTC conversion, existing `+08:00`, naive legacy values, and `None`.
- Modify `tests/web/test_datasets.py`, `test_imports.py`, `test_evaluations.py`, and `test_reports.py`: HTTP-level regression coverage for each affected page.
- Modify `tests/test_genie02_regression.py`: deterministic report and LeRobot timestamp coverage.
- Modify `progress.md`: record scope, implementation, commands, and results.

### Task 1: Shared Beijing Time Utilities

**Files:**
- Create: `vla_eval/time_utils.py`
- Create: `tests/test_time_utils.py`

- [ ] **Step 1: Write failing utility tests**

```python
from datetime import UTC, datetime, timedelta, timezone

from vla_eval.time_utils import as_beijing_time, format_beijing_time


def test_format_beijing_time_converts_utc_to_24_hour_display():
    value = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)

    assert format_beijing_time(value) == "2026-08-08 18:35:42（北京时间）"


def test_format_beijing_time_does_not_double_shift_existing_offset():
    value = datetime(
        2026, 8, 8, 18, 35, 42, tzinfo=timezone(timedelta(hours=8))
    )

    assert format_beijing_time(value) == "2026-08-08 18:35:42（北京时间）"


def test_legacy_naive_system_time_is_interpreted_as_utc():
    value = datetime(2026, 8, 8, 10, 35, 42)

    converted = as_beijing_time(value)

    assert converted.utcoffset() == timedelta(hours=8)
    assert converted.hour == 18


def test_missing_time_uses_placeholder():
    assert format_beijing_time(None) == "—"
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `.venv/bin/pytest -q tests/test_time_utils.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'vla_eval.time_utils'`.

- [ ] **Step 3: Implement the minimal shared utility**

```python
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def as_beijing_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(BEIJING_TIMEZONE)


def format_beijing_time(value: datetime | None) -> str:
    if value is None:
        return "—"
    return f"{as_beijing_time(value):%Y-%m-%d %H:%M:%S}（北京时间）"


def beijing_now() -> datetime:
    return datetime.now(UTC).astimezone(BEIJING_TIMEZONE)
```

- [ ] **Step 4: Run utility tests**

Run: `.venv/bin/pytest -q tests/test_time_utils.py`

Expected: `4 passed`.

- [ ] **Step 5: Commit the utility and tests**

```bash
git add vla_eval/time_utils.py tests/test_time_utils.py
git commit -m "feat: add Beijing timestamp formatting"
```

### Task 2: Web Page Timestamp Coverage and Rendering

**Files:**
- Create: `vla_eval/web/templating.py`
- Modify: `vla_eval/web/routes_auth.py`
- Modify: `vla_eval/web/routes_datasets.py`
- Modify: `vla_eval/web/routes_imports.py`
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `vla_eval/web/routes_reports.py`
- Modify: `vla_eval/web/templates/datasets/index.html`
- Modify: `vla_eval/web/templates/datasets/detail.html`
- Modify: `vla_eval/web/templates/imports/index.html`
- Modify: `vla_eval/web/templates/imports/detail.html`
- Modify: `vla_eval/web/templates/evaluations/index.html`
- Modify: `vla_eval/web/templates/evaluations/detail.html`
- Modify: `vla_eval/web/templates/reports/detail.html`
- Test: `tests/web/test_datasets.py`
- Test: `tests/web/test_imports.py`
- Test: `tests/web/test_evaluations.py`
- Test: `tests/web/test_reports.py`

- [ ] **Step 1: Add failing dataset page assertions**

Add a test that persists fixed UTC `created_at` and `updated_at`, plus a valid archived snapshot, then checks both list and detail output:

```python
def test_dataset_pages_show_beijing_system_timestamps(auth_client, db_engine):
    created = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)
    updated = datetime(2026, 8, 8, 11, 6, 7, tzinfo=UTC)
    dataset = Dataset(
        name="Timestamp Dataset",
        path="/srv/timestamp",
        kind="fixture",
        status="ARCHIVED",
        created_at=created,
        updated_at=updated,
        inspection_json={
            "_archive": {
                "previous_status": "READY",
                "archived_at": "2026-08-08T12:00:01+00:00",
                "archived_by": "user-id",
            }
        },
    )
    with session_scope(db_engine) as session:
        session.add(dataset)
        session.flush()
        dataset_id = dataset.id

    listing = auth_client.get("/datasets?archived=1")
    detail = auth_client.get(f"/datasets/{dataset_id}")

    assert "2026-08-08 18:35:42（北京时间）" in listing.text
    assert "2026-08-08 18:35:42（北京时间）" in detail.text
    assert "2026-08-08 19:06:07（北京时间）" in detail.text
    assert "2026-08-08 20:00:01（北京时间）" in detail.text
```

- [ ] **Step 2: Add failing import page assertions**

Add `from datetime import UTC, datetime` to `tests/web/test_imports.py`. Extend `test_import_pages_and_status_api_render_persisted_jobs` with fixed values and these assertions:

```python
created = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)
updated = datetime(2026, 8, 8, 11, 6, 7, tzinfo=UTC)
# Pass created_at=created and updated_at=updated to ImportJob.
assert "2026-08-08 18:35:42（北京时间）" in listing.text
assert "2026-08-08 18:35:42（北京时间）" in detail.text
assert "2026-08-08 19:06:07（北京时间）" in detail.text
```

- [ ] **Step 3: Add failing evaluation page assertions while preserving the API contract**

Extend the existing fixed-time evaluation list test and add a detail/archive test:

```python
assert "2026-08-06 17:30:00（北京时间）" in response.text


def test_evaluation_detail_shows_beijing_system_and_archive_times(
    auth_client, db_engine, ready_dataset
):
    created = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)
    updated = datetime(2026, 8, 8, 11, 6, 7, tzinfo=UTC)
    with session_scope(db_engine) as session:
        job = EvaluationJob(
            dataset_id=ready_dataset.id,
            profile_name="timestamp-profile",
            state="SUCCEEDED",
            created_at=created,
            updated_at=updated,
        )
        session.add(job)
        session.flush()
        job_id = job.id
        session.add(
            EvaluationJobArchive(
                evaluation_job_id=job_id,
                archived_at=datetime(2026, 8, 8, 12, 0, 1, tzinfo=UTC),
            )
        )

    detail = auth_client.get(f"/evaluations/{job_id}")
    status = auth_client.get(f"/api/evaluations/{job_id}")

    assert "2026-08-08 18:35:42（北京时间）" in detail.text
    assert "2026-08-08 19:06:07（北京时间）" in detail.text
    assert "2026-08-08 20:00:01（北京时间）" in detail.text
    assert status.json()["updated_at"] == "2026-08-08T11:06:07+00:00"
```

- [ ] **Step 4: Add a failing web report timestamp assertion**

Add `from datetime import UTC, datetime` to `tests/web/test_reports.py`, then add:

```python
def test_report_page_shows_beijing_task_timestamps(
    auth_client, db_engine, successful_job
):
    with session_scope(db_engine) as session:
        persisted = session.get_one(EvaluationJob, successful_job.id)
        persisted.created_at = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC)
        persisted.updated_at = datetime(2026, 8, 8, 11, 6, 7, tzinfo=UTC)

    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert "2026-08-08 18:35:42（北京时间）" in response.text
    assert "2026-08-08 19:06:07（北京时间）" in response.text
```

- [ ] **Step 5: Run the focused web tests and verify they fail on missing/UTC displays**

Run:

```bash
.venv/bin/pytest -q \
  tests/web/test_datasets.py::test_dataset_pages_show_beijing_system_timestamps \
  tests/web/test_imports.py::test_import_pages_and_status_api_render_persisted_jobs \
  tests/web/test_evaluations.py::test_evaluation_list_shows_newest_jobs_and_report_links \
  tests/web/test_evaluations.py::test_evaluation_detail_shows_beijing_system_and_archive_times \
  tests/web/test_reports.py::test_report_page_shows_beijing_task_timestamps
```

Expected: assertions fail because the pages omit the timestamps or render UTC to minutes.

- [ ] **Step 6: Create the shared Jinja environment**

```python
from pathlib import Path

from fastapi.templating import Jinja2Templates

from vla_eval.time_utils import format_beijing_time

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["beijing_time"] = format_beijing_time
```

Replace each route module's local `Jinja2Templates` construction with:

```python
from vla_eval.web.templating import templates
```

Remove now-unused `Jinja2Templates` imports and `Path` imports only where `Path` has no other use.

- [ ] **Step 7: Parse validated dataset archive time for the detail context**

Add a focused helper in `routes_datasets.py` and pass its result as `dataset_archived_at`:

```python
def _dataset_archived_at(dataset: Dataset) -> datetime | None:
    snapshot = dataset.inspection_json.get("_archive")
    if dataset.status != "ARCHIVED" or not _valid_archive_snapshot(snapshot):
        return None
    return datetime.fromisoformat(snapshot["archived_at"])
```

- [ ] **Step 8: Render all page timestamps with the shared filter**

Use these exact labels and expressions:

```jinja2
{{ dataset.created_at|beijing_time }}
{{ dataset.updated_at|beijing_time }}
{{ dataset_archived_at|beijing_time }}
{{ job.created_at|beijing_time }}
{{ job.updated_at|beijing_time }}
{{ job_archive.archived_at|beijing_time }}
```

Add “导入系统时间” to the dataset list and detail, “创建时间” to import/evaluation lists, and “创建时间”/“最后更新时间” to import/evaluation/report details. Only show “归档时间” when a validated archive exists. Update empty-table `colspan` values after adding list columns. Replace the two existing direct `strftime('%Y-%m-%d %H:%M')` calls with the filter.

- [ ] **Step 9: Run all web tests**

Run:

```bash
.venv/bin/pytest -q tests/web/test_datasets.py tests/web/test_imports.py \
  tests/web/test_evaluations.py tests/web/test_reports.py
```

Expected: all selected tests pass, including the existing API `+00:00` assertion.

- [ ] **Step 10: Commit web rendering changes**

```bash
git add vla_eval/web tests/web
git commit -m "fix(web): display system times in Beijing timezone"
```

### Task 3: Deterministic Markdown Report Times

**Files:**
- Modify: `Genie02_report/genie02_eval_common.py`
- Modify: `Genie02_report/genie02_markdown_report.py`
- Modify: `tests/test_genie02_regression.py`

- [ ] **Step 1: Add a failing synthetic LeRobot timestamp test**

Add `import os` and `from datetime import UTC, datetime` to `tests/test_genie02_regression.py`. Extend `test_lerobot_session_preserves_recorded_info_metadata` with a fixed directory modification time:

```python
from datetime import UTC, datetime
import os

fixed = datetime(2026, 8, 8, 10, 35, 42, tzinfo=UTC).timestamp()
os.utime(dataset, (fixed, fixed))

session = _synthesize_lerobot_session(dataset)

assert session["created_at"] == "2026-08-08T10:35:42+00:00"
```

- [ ] **Step 2: Add a failing Markdown report test with a fixed generation instant**

```python
def test_markdown_report_uses_beijing_record_and_generation_times(
    minimal_native_session, tmp_path, monkeypatch
):
    from Genie02_report import genie02_markdown_report

    fixed_now = datetime(2026, 8, 8, 16, 30, 45, tzinfo=UTC)
    monkeypatch.setattr(genie02_markdown_report, "beijing_now", lambda: fixed_now.astimezone(BEIJING_TIMEZONE))
    output_dir = tmp_path / "timestamp-report"

    generate_report(minimal_native_session, output_dir)
    report_path = output_dir / "report_20260809.md"
    report = report_path.read_text(encoding="utf-8")

    assert "数据记录时间：2026-01-02 03:04:05（北京时间）" in report
    assert "报告生成时间：2026-08-09 00:30:45（北京时间）" in report
```

Import `BEIJING_TIMEZONE` from `vla_eval.time_utils` in the test. The session fixture is already `+08:00`, so this also catches accidental double conversion.

- [ ] **Step 3: Run report tests and verify timezone-dependent failures**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_genie02_regression.py::test_lerobot_session_preserves_recorded_info_metadata \
  tests/test_genie02_regression.py::test_markdown_report_uses_beijing_record_and_generation_times
```

Expected: the synthetic session uses the host offset, and the Markdown report lacks the two explicit timestamps.

- [ ] **Step 4: Make synthetic session and default directory dates host-independent**

In `genie02_eval_common.py`, import `UTC` and `beijing_now`, then use:

```python
created_at = datetime.fromtimestamp(
    session_dir.stat().st_mtime, tz=UTC
).isoformat(timespec="seconds")

default_dir = f"{DEFAULT_OUTPUT_DIR}_{beijing_now():%Y%m%d}"
```

This only reads the dataset directory modification time and writes the derived session dictionary in memory; it does not modify dataset files.

- [ ] **Step 5: Parse recorded time strictly and capture generation time once**

Add this private helper to `genie02_markdown_report.py`:

```python
def _session_created_at(session: dict[str, Any]) -> datetime:
    value = session.get("created_at")
    if not isinstance(value, str):
        raise EvaluationError("session.json created_at must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise EvaluationError(
            "session.json created_at must be an ISO 8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise EvaluationError("session.json created_at must include a timezone")
    return parsed
```

Change `build_report` to accept a keyword-only `generated_at: datetime`, replace the ambiguous date line with:

```python
f"- 数据记录时间：{format_beijing_time(_session_created_at(session))}",
f"- 报告生成时间：{format_beijing_time(generated_at)}",
```

In `generate_markdown_report`, capture one instant and use it for both report contents and filename:

```python
generated_at = beijing_now()
report = build_report(
    session, episodes, episode_metrics, metrics, generated_at=generated_at
)
output = output_root / f"report_{generated_at:%Y%m%d}.md"
```

- [ ] **Step 6: Run focused and complete Genie02 regression tests**

Run:

```bash
.venv/bin/pytest -q tests/test_genie02_regression.py tests/test_cli.py \
  tests/test_evaluation.py
```

Expected: all selected tests pass; existing metric artifacts and relative Episode timestamps remain unchanged.

- [ ] **Step 7: Commit report changes**

```bash
git add Genie02_report/genie02_eval_common.py \
  Genie02_report/genie02_markdown_report.py tests/test_genie02_regression.py
git commit -m "fix(report): use explicit Beijing timestamps"
```

### Task 4: Change Record and Final Verification

**Files:**
- Modify: `progress.md`

- [ ] **Step 1: Record the completed behavior and evidence**

Append a dated section to `progress.md` containing:

```markdown
## 2026-08-08 系统时间戳统一

- 数据库和 API 继续使用 UTC；网页与 Markdown 报告统一显示 `Asia/Shanghai`，24 小时制并精确到秒。
- 数据集页面显示网站记录的“导入系统时间”，未修改或回写数据集文件及内部元数据。
- 数据集、导入任务、评测任务、归档记录和网页报告使用统一 Jinja 时间过滤器。
- Markdown 报告明确区分数据记录时间和报告生成时间；目录和文件名日期按北京时间生成。
- Episode 视频相对时间不进行时区转换。
```

After running the next steps, append bullets containing the exact commands, pass counts, and browser pages that produced successful results. Omit any check that was not executed.

- [ ] **Step 2: Run static checks**

Run:

```bash
.venv/bin/ruff check Genie02_report vla_eval tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Run the complete automated test suite once**

Run: `.venv/bin/pytest -q`

Expected: all tests pass. This is the single full-suite run; earlier runs remain scoped to affected modules.

- [ ] **Step 4: Start the local application and inspect real pages**

If port 8000 is not already serving this worktree, start the existing local environment with:

```bash
source data/dev/env.sh
.venv/bin/uvicorn vla_eval.server:create_app_from_env --factory --host 127.0.0.1 --port 8000
```

Verify `curl -fsS http://127.0.0.1:8000/health` returns `{"status":"ok"}`. Check `/datasets`, one dataset detail, `/imports`, one import detail, `/evaluations`, one evaluation detail, and one `/reports/<job-id>` page at desktop and narrow viewport widths. Confirm:

- every visible calendar timestamp includes seconds and “北京时间”;
- no 12-hour AM/PM display appears;
- added columns and summary values do not overlap;
- API `/api/evaluations/<job-id>` still returns UTC;
- no data file changes appear in `git status` or dataset directories.

- [ ] **Step 5: Update `progress.md` with actual verification results and commit**

```bash
git add progress.md
git commit -m "docs: record timestamp verification"
```

- [ ] **Step 6: Review final diff and repository state**

Run:

```bash
git status --short --branch
git log -5 --oneline
git diff origin/feature/vla-eval-web-vlm-api-backend...HEAD --stat
```

Expected: the branch contains only the design, plan, timestamp implementation, tests, and progress record; `uv.lock` remains untracked and unstaged.
