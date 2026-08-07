# Dataset and Evaluation List Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent archive/restore, keyword contains search, and stable time/name sorting to dataset and evaluation-task lists without deleting stored data or breaking historical reports.

**Architecture:** Dataset archival reuses `Dataset.status = "ARCHIVED"` with a strict `_archive` snapshot inside `inspection_json`. Evaluation tasks retain their execution state and use a new one-row-per-job `EvaluationJobArchive` table. Shared list-query parsing validates URL parameters and escapes literal LIKE metacharacters; route-specific SQL applies filtering and deterministic ordering.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2, SQLite, Jinja2, vanilla CSS/JavaScript, pytest, Ruff.

---

## File Map

- Create `vla_eval/web/list_management.py`: parse common list controls, escape literal contains-search patterns, and validate safe return targets.
- Modify `vla_eval/models.py`: define `EvaluationJobArchive` without changing existing table columns.
- Modify `vla_eval/web/routes_datasets.py`: query controls, dataset archive/restore transactions, active-job guard, and detail archive state.
- Modify `vla_eval/web/routes_evaluations.py`: query controls, task archive/restore transactions, archive joins, and detail archive state.
- Modify `vla_eval/web/templates/datasets/index.html` and `detail.html`: filter toolbar and archive/restore controls.
- Modify `vla_eval/web/templates/evaluations/index.html` and `detail.html`: combined filters and archive/restore controls.
- Modify `vla_eval/web/static/app.css`: compact responsive list toolbar, archive badge, and stable action layout.
- Modify `tests/test_models.py`: archive-table schema and foreign-key behavior.
- Create `tests/web/test_list_management.py`: pure query and redirect validation.
- Modify `tests/web/test_datasets.py`: dataset search/sort/archive/restore and file-preservation behavior.
- Modify `tests/web/test_evaluations.py`: task search/sort/archive/restore, report preservation, and active-state rejection.
- Modify `tests/e2e/test_visual_layout.py`: desktop/mobile list-toolbar overflow and visibility checks.
- Modify `task_plan.md` and `progress.md`: required change record, verification commands, and real acceptance IDs.

### Task 1: Shared List Query Contract

**Files:**
- Create: `vla_eval/web/list_management.py`
- Create: `tests/web/test_list_management.py`

- [ ] **Step 1: Write failing tests for common controls**

Specify a frozen value object and strict parser. Build requests without a new fixture dependency:

```python
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
    request = _request("/datasets?q=%20Robot%20")
    controls = parse_list_controls(request)
    assert controls == ListControls(q="Robot", sort="newest", include_archived=False)


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
def test_parse_list_controls_rejects_invalid_or_duplicate_values(query):
    with pytest.raises(HTTPException) as captured:
        parse_list_controls(_request(f"/datasets?{query}"))
    assert captured.value.status_code == 422
```

Also require `allowed_extra={"state", "dataset_id"}` to preserve the evaluation page's existing filters.

- [ ] **Step 2: Write failing tests for literal search and return targets**

```python
@pytest.mark.parametrize(
    ("value", "expected"),
    [("robot", "%robot%"), ("100%", "%100\\%%"), ("a_b", "%a\\_b%"), (r"a\\b", r"%a\\\\b%")],
)
def test_literal_contains_pattern_escapes_like_metacharacters(value, expected):
    assert literal_contains_pattern(value) == expected


def test_validate_return_to_accepts_only_local_allowed_paths():
    assert validate_return_to("/datasets?q=robot", {"/datasets"}, "/datasets") == "/datasets?q=robot"
    assert validate_return_to("https://evil.example/", {"/datasets"}, "/datasets") == "/datasets"
    assert validate_return_to("//evil.example/", {"/datasets"}, "/datasets") == "/datasets"
```

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_list_management.py -q`

Expected: collection fails because `vla_eval.web.list_management` does not exist.

- [ ] **Step 4: Implement the shared contract**

Create:

```python
@dataclass(frozen=True)
class ListControls:
    q: str
    sort: Literal["newest", "oldest", "name_asc", "name_desc"]
    include_archived: bool


def parse_list_controls(request: Request, *, allowed_extra: frozenset[str] = frozenset()) -> ListControls:
    allowed = {"q", "sort", "archived"} | set(allowed_extra)
    if set(request.query_params) - allowed:
        raise HTTPException(status_code=422, detail="Invalid list filters")
    for key in allowed:
        if len(request.query_params.getlist(key)) > 1:
            raise HTTPException(status_code=422, detail="Invalid list filters")
    q = request.query_params.get("q", "").strip()
    sort = request.query_params.get("sort", "newest")
    archived = request.query_params.get("archived")
    if len(q) > 200 or sort not in LIST_SORTS or archived not in {None, "1"}:
        raise HTTPException(status_code=422, detail="Invalid list filters")
    return ListControls(q=q, sort=sort, include_archived=archived == "1")


def literal_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
```

Implement `validate_return_to()` with `urllib.parse.urlsplit`: accept no scheme, netloc, fragment, username, or control characters, and require the parsed path to be in an endpoint-supplied exact allowlist; otherwise return the fallback.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_list_management.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vla_eval/web/list_management.py tests/web/test_list_management.py
git commit -m "feat(web): validate list controls"
```

### Task 2: Evaluation Archive Persistence

**Files:**
- Modify: `vla_eval/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

Add tests that create a user, dataset, evaluation job, and archive record:

```python
archive = EvaluationJobArchive(
    evaluation_job_id=job.id,
    archived_by=user.id,
)
session.add(archive)
session.flush()
assert archive.archived_at.tzinfo == UTC
```

Assert a second record for the same job raises `IntegrityError`, deleting an evaluation cascades its archive record, and deleting the user sets `archived_by` to `None` while preserving the archive.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/test_models.py -q`

Expected: import or attribute failure because `EvaluationJobArchive` is undefined.

- [ ] **Step 3: Implement the table**

Add:

```python
class EvaluationJobArchive(Base):
    __tablename__ = "evaluation_job_archives"

    evaluation_job_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    archived_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utc_now)
    archived_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
```

Do not add archive columns to `datasets` or `evaluation_jobs`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/test_models.py -q`

Expected: all model and foreign-key tests pass.

- [ ] **Step 5: Commit**

```bash
git add vla_eval/models.py tests/test_models.py
git commit -m "feat(model): persist evaluation archives"
```

### Task 3: Dataset Archive and Restore

**Files:**
- Modify: `vla_eval/web/routes_datasets.py`
- Modify: `tests/web/test_datasets.py`

- [ ] **Step 1: Write failing archive tests**

Add helpers that POST exactly `csrf_token` and `return_to`. Test that archiving a READY dataset:

```python
response = auth_client.post(
    f"/datasets/{ready_dataset.id}/archive",
    data={"csrf_token": auth_client.csrf, "return_to": "/datasets?q=robot"},
    follow_redirects=False,
)
assert response.status_code == 303
assert response.headers["location"] == "/datasets?q=robot"
with session_scope(db_engine) as session:
    stored = session.get_one(Dataset, ready_dataset.id)
    assert stored.status == "ARCHIVED"
    assert stored.inspection_json["_archive"]["previous_status"] == "READY"
```

Record all files and their bytes under the fixture dataset before the request and assert they are unchanged afterwards. Add RED tests for missing CSRF, unknown/duplicate form fields, repeat archive, invalid return target fallback, and active `QUEUED`/`RUNNING` evaluation conflicts.

- [ ] **Step 2: Write failing restore and evaluation-guard tests**

Test restoration removes only `_archive`, retains `errors`, and restores `READY`. Test missing/corrupt `_archive` returns 409. Test both GET `/evaluations/new?dataset_id=...` and POST `/evaluations` reject an archived dataset; retry of a FAILED job against it must also reject.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py -q`

Expected: archive/restore endpoints return 404 and archive guards are absent.

- [ ] **Step 4: Implement archive metadata and transactions**

Define constants:

```python
_ARCHIVE_KEY = "_archive"
_ARCHIVABLE_DATASET_STATES = frozenset({"READY", "PREFLIGHT_FAILED"})
_ARCHIVE_FORM_FIELDS = frozenset({"csrf_token", "return_to"})
_TERMINAL_EVALUATION_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
)
```

Archive using a server-generated snapshot:

```python
metadata = dict(dataset.inspection_json or {})
metadata[_ARCHIVE_KEY] = {
    "previous_status": dataset.status,
    "archived_at": datetime.now(UTC).isoformat(),
    "archived_by": current_user.id,
}
dataset.inspection_json = metadata
dataset.status = "ARCHIVED"
```

Before mutation, query for any `EvaluationJob.state.not_in(_TERMINAL_EVALUATION_STATES)` for this dataset and return 409 when found. Restore only from a complete dictionary whose `previous_status` is in `_ARCHIVABLE_DATASET_STATES`; assign a fresh metadata dictionary after removing `_archive` so `MutableDict` persists the change.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py -q`

Expected: all archive, restore, file-preservation, and existing attachment/evaluation tests pass.

- [ ] **Step 6: Commit**

```bash
git add vla_eval/web/routes_datasets.py tests/web/test_datasets.py tests/web/test_evaluations.py
git commit -m "feat(datasets): archive without deleting data"
```

### Task 4: Evaluation Task Archive and Restore

**Files:**
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `tests/web/test_evaluations.py`
- Modify: `tests/web/test_reports.py`

- [ ] **Step 1: Write failing task archive tests**

Parametrize `SUCCEEDED`, `FAILED`, `CANCELLED`, and `INTERRUPTED`: POST archive, assert 303, assert one `EvaluationJobArchive`, and assert every original job field remains unchanged. Parametrize nonterminal states such as `QUEUED` and `RUNNING` and assert 409 with no archive record.

Add CSRF, duplicate field, duplicate archive, duplicate restore, missing entity, and external `return_to` tests.

- [ ] **Step 2: Write failing report-preservation test**

Using an existing succeeded report fixture, capture report HTML and every whitelisted download before archival. Archive the job, then assert the report still returns 200 and each download has identical bytes and attachment headers. Restore and assert the archive record is gone.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_evaluations.py tests/web/test_reports.py -q`

Expected: archive/restore endpoints return 404.

- [ ] **Step 4: Implement task archive transactions**

Add `EvaluationJobArchive` imports and `_ARCHIVABLE_JOB_STATES = _TERMINAL_STATES`. POST archive must load the job, reject nonterminal states, and insert:

```python
session.add(
    EvaluationJobArchive(
        evaluation_job_id=job.id,
        archived_by=current_user.id,
    )
)
```

Catch `IntegrityError` outside the transaction and map it to 409. Restore deletes exactly one archive row and returns 409 when `rowcount != 1`. Do not modify `EvaluationJob.state`, `params_json`, `provenance_json`, or `output_dir`.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_evaluations.py tests/web/test_reports.py -q`

Expected: all task lifecycle and report-preservation tests pass.

- [ ] **Step 6: Commit**

```bash
git add vla_eval/web/routes_evaluations.py tests/web/test_evaluations.py tests/web/test_reports.py
git commit -m "feat(evaluations): archive completed tasks"
```

### Task 5: Server-Side Search, Sorting, and Combined Filters

**Files:**
- Modify: `vla_eval/web/routes_datasets.py`
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `tests/web/test_datasets.py`
- Modify: `tests/web/test_evaluations.py`

- [ ] **Step 1: Write failing dataset list tests**

Create datasets with mixed-case names, distinct paths, tied timestamps, and one `ARCHIVED` record. Assert:

- default excludes archived;
- `archived=1` includes active and archived;
- `q=robot` matches name and path case-insensitively;
- `q=100%` and `q=a_b` treat `%` and `_` literally;
- `newest`, `oldest`, `name_asc`, and `name_desc` produce exact stable ID sequences;
- invalid/duplicate/unknown query parameters return 422.

- [ ] **Step 2: Write failing evaluation list tests**

Create jobs that vary by dataset name, profile, partial UUID, state, created time, and archive record. Assert keyword, state, `dataset_id`, sort, and `archived=1` work alone and in one combined request.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py -q`

Expected: current routes ignore the new controls or return incorrect ordering.

- [ ] **Step 4: Implement dataset query composition**

Parse `ListControls`. Apply:

```python
if not controls.include_archived:
    query = query.where(Dataset.status != "ARCHIVED")
if controls.q:
    pattern = literal_contains_pattern(controls.q)
    query = query.where(
        or_(Dataset.name.ilike(pattern, escape="\\"), Dataset.path.ilike(pattern, escape="\\"))
    )
```

Map sort values to deterministic SQLAlchemy order tuples using `func.lower(Dataset.name)`, original name, created time, and ID. Pass controls and the current local URL to the template.

- [ ] **Step 5: Implement evaluation query composition**

Outer join `EvaluationJobArchive`, keep existing dataset join, and select `(EvaluationJob, Dataset, EvaluationJobArchive)`. Default-filter archive rows with `EvaluationJobArchive.evaluation_job_id.is_(None)`. Apply literal contains-search to dataset name, profile name, and job ID. Preserve strict `state` and `dataset_id` filters and deterministic sorting.

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py -q`

Expected: all list, archive, existing lifecycle, and combined-filter tests pass.

- [ ] **Step 7: Commit**

```bash
git add vla_eval/web/routes_datasets.py vla_eval/web/routes_evaluations.py tests/web/test_datasets.py tests/web/test_evaluations.py
git commit -m "feat(web): search and sort datasets and evaluations"
```

### Task 6: Responsive List Controls and Detail Actions

**Files:**
- Modify: `vla_eval/web/templates/datasets/index.html`
- Modify: `vla_eval/web/templates/datasets/detail.html`
- Modify: `vla_eval/web/templates/evaluations/index.html`
- Modify: `vla_eval/web/templates/evaluations/detail.html`
- Modify: `vla_eval/web/static/app.css`
- Modify: `tests/web/test_datasets.py`
- Modify: `tests/web/test_evaluations.py`
- Modify: `tests/e2e/test_visual_layout.py`

- [ ] **Step 1: Write failing template contract tests**

Assert both pages render labelled `q` and `sort` controls, a checkbox named `archived` with value `1`, selected values after filtering, a clear link only when filters are active, archive/restore POST forms with CSRF, Lucide archive icons, and confirmation text. Assert active tasks do not render an archive submit button.

- [ ] **Step 2: Write failing visual layout assertions**

Extend the real-app Playwright test to visit `/datasets` and `/evaluations` at 1440x1000 and 390x844. Assert the list toolbar is visible, every control stays within the viewport, no body-level horizontal overflow occurs, and action text/icons remain inside their cells.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py tests/e2e/test_visual_layout.py -q`

Expected: missing controls/actions and layout selectors fail.

- [ ] **Step 4: Implement semantic templates**

Use one GET form per list with labels and current values. Dataset row logic keys off `dataset.status == "ARCHIVED"`; evaluation row logic keys off whether its joined archive record is present. Use POST forms containing only CSRF and `return_to`, with `onsubmit="return confirm(...)"`. Add `archive`, `archive-restore`, and existing Lucide icons rather than custom SVG.

On detail pages, render archive/restore beside the primary command. Archived datasets show a disabled new-evaluation command but retain history. Evaluation detail retains report/retry/cancel behavior and adds archive/restore only when allowed.

- [ ] **Step 5: Implement compact responsive CSS**

Add unframed `.list-toolbar`, stable `.row-actions`, `.archive-badge`, and `.inline-action-form` rules. Use grid/flex constraints, 4px radii, no viewport-scaled type, and a 720px stacked layout. Do not create nested cards or decorative sections.

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_datasets.py tests/web/test_evaluations.py tests/e2e/test_visual_layout.py -q`

Expected: semantic and desktop/mobile layout tests pass.

- [ ] **Step 7: Commit**

```bash
git add vla_eval/web/templates/datasets vla_eval/web/templates/evaluations vla_eval/web/static/app.css tests/web/test_datasets.py tests/web/test_evaluations.py tests/e2e/test_visual_layout.py
git commit -m "feat(web): add archive and list controls"
```

### Task 7: Change Record, Real Acceptance, and Final Verification

**Files:**
- Modify: `task_plan.md`
- Modify: `progress.md`

- [ ] **Step 1: Restart the real local service on current code**

Restart Uvicorn and both workers using `/Users/xueyg/.config/vla-eval/app.yaml`. Confirm `GET https://127.0.0.1:8443/health` returns `{"status":"ok"}` and both RQ workers are registered.

- [ ] **Step 2: Verify real database initialization**

Start the app once so `create_all` creates `evaluation_job_archives`. Query SQLite metadata and assert the new table exists with the three designed columns and both foreign keys.

- [ ] **Step 3: Perform reversible real acceptance**

Record one current READY dataset ID and one terminal evaluation ID with report output. Through authenticated HTTPS forms:

1. search each by a partial keyword;
2. exercise all four sort values;
3. archive the dataset only after confirming it has no active tasks;
4. confirm it disappears by default and reappears with `archived=1`;
5. confirm its files and fingerprint are unchanged;
6. restore it and confirm READY is restored;
7. archive the terminal evaluation;
8. confirm report HTML and every download remain byte-identical;
9. restore it and confirm the archive record is removed.

- [ ] **Step 4: Run nonredundant final verification**

The focused suites have already run at each RED/GREEN checkpoint. Run the full suite once here:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check Genie02_report vla_eval tests
git diff --check
```

Expected: the full suite passes, Ruff reports no issues, and whitespace validation exits 0.

- [ ] **Step 5: Update the required change record**

Append a checked “数据集与评测任务列表管理” section to `task_plan.md`. Append implementation commits, new table contract, real dataset/evaluation IDs, archive/restore results, report/download preservation evidence, and exact verification output to `progress.md`. Record any encountered failures and resolutions. Do not stage `uv.lock`.

- [ ] **Step 6: Commit acceptance evidence**

```bash
git add task_plan.md progress.md
git commit -m "test(web): verify archive and list management"
```
