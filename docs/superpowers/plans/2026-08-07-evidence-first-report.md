# Evidence-First Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Web evaluation report whose metrics, configuration, data-quality facts, VLM state, formulas, evidence gaps, and downloads all trace to current persisted interfaces and artifacts.

**Architecture:** Add a shared, structured metric-definition module consumed by Markdown and Web. Move report data composition into a focused Web view-model module that joins the existing strict Genie02 loaders with job/dataset provenance, while the route retains authentication, filtering, and secure file delivery. Render the evidence-first hierarchy in Jinja with semantic tables and local formula markup.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Jinja2, pytest, Playwright, HTML/CSS, Lucide icons

---

## File Structure

- Create `Genie02_report/metric_definitions.py`: immutable metric metadata and structured formula fragments shared by report renderers.
- Modify `Genie02_report/genie02_markdown_report.py`: generate metric definitions and formulas from the shared module.
- Create `vla_eval/web/report_view.py`: strict cross-source report view construction and factual status derivation.
- Modify `vla_eval/web/routes_reports.py`: delegate content composition, retain filters, and support safe nested artifacts.
- Modify `vla_eval/web/routes_evaluations.py`: persist the submitted profile output contract in provenance.
- Modify `vla_eval/web/templates/reports/detail.html`: render the approved evidence-first hierarchy and semantic formulas/tables.
- Modify `vla_eval/web/static/app.css`: responsive report sections, formula layout, table containment, and print-safe styling.
- Modify `tests/test_genie02_regression.py`: shared-definition and Markdown formula regression coverage.
- Modify `tests/web/test_evaluations.py`: output-contract provenance coverage.
- Modify `tests/web/test_reports.py`: source mapping, quality facts, VLM states, formulas, evidence gaps, and nested download security.
- Modify `tests/e2e/test_visual_layout.py`: desktop/mobile report containment and critical-section visibility.

### Task 1: Shared Metric Definitions

**Files:**
- Create: `Genie02_report/metric_definitions.py`
- Modify: `Genie02_report/genie02_markdown_report.py`
- Test: `tests/test_genie02_regression.py`

- [ ] **Step 1: Write failing tests for one shared metric registry**

Add tests that import `METRIC_DEFINITIONS` and `markdown_formula_lines`, assert the registry order is `gsr`, `tts_success`, `smoothness`, and assert Markdown output contains all five implemented equations:

```python
def test_metric_definitions_match_implemented_formulas():
    from Genie02_report.metric_definitions import (
        METRIC_DEFINITIONS,
        markdown_formula_lines,
    )

    assert [metric.key for metric in METRIC_DEFINITIONS] == [
        "gsr",
        "tts_success",
        "smoothness",
    ]
    formulas = "\n".join(markdown_formula_lines(METRIC_DEFINITIONS))
    assert "GSR = N_success / N_total" in formulas
    assert "TTS = mean(duration_s | outcome = success)" in formulas
    assert "S = log10(E + 1)" in formulas
    assert "E = sum(||j_k||^2) * delta_t" in formulas
    assert "j_k = (x_k - 3 x_(k-1) + 3 x_(k-2) - x_(k-3)) / delta_t^3" in formulas


def test_markdown_report_uses_shared_metric_wording(
    minimal_native_session, tmp_path, monkeypatch
):
    from Genie02_report import genie02_markdown_report

    monkeypatch.setattr(
        genie02_markdown_report,
        "metric_definition_rows",
        lambda: [("sentinel", "shared-definition", "smaller")],
    )
    output_dir = tmp_path / "shared-definitions"
    generate_report(minimal_native_session, output_dir)
    report_path = next(output_dir.glob("report_*.md"))
    assert "shared-definition" in report_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_genie02_regression.py -k "metric_definitions or shared_metric" -q
```

Expected: FAIL because `Genie02_report.metric_definitions` and the shared helpers do not exist.

- [ ] **Step 3: Implement structured definitions**

Create these frozen dataclasses. A formula line contains escaped text fragments and either an inline right side or numerator/denominator fragments; no HTML is stored in the registry:

```python
@dataclass(frozen=True)
class FormulaFragment:
    text: str
    subscript: str = ""
    superscript: str = ""


@dataclass(frozen=True)
class FormulaLine:
    lhs: tuple[FormulaFragment, ...]
    rhs: tuple[FormulaFragment, ...] = ()
    numerator: tuple[FormulaFragment, ...] = ()
    denominator: tuple[FormulaFragment, ...] = ()


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    definition: str
    direction: str
    formulas: tuple[FormulaLine, ...]
    notes: tuple[str, ...] = ()
```

Export:

```python
METRIC_DEFINITIONS: tuple[MetricDefinition, ...]
def metric_definition_rows() -> list[tuple[str, str, str]]: ...
def markdown_formula_lines(metrics=METRIC_DEFINITIONS) -> list[str]: ...
```

Registry meanings must state:

- GSR is successful Episode count divided by total Episode count; larger is better.
- TTS is mean `duration_s` over successful Episodes only; smaller is better.
- Smoothness is the implemented log jerk-energy value; smaller is smoother.
- Smoothness preprocessing removes non-finite and intervention frames, sorts timestamps, removes duplicate timestamps, uses median positive `delta_t`, and requires at least four valid frames.

- [ ] **Step 4: Replace Markdown-local formula strings**

Import `metric_definition_rows` and `markdown_formula_lines` in `genie02_markdown_report.py`. Build the metric definition table and formula block from these helpers; remove the duplicated GSR/TTS/smoothness prose at the current `build_report()` formula section.

- [ ] **Step 5: Run focused and Genie02 regression tests**

Run:

```bash
.venv/bin/pytest tests/test_genie02_regression.py -q
```

Expected: PASS with the generated report still containing current metrics and the five shared equations.

- [ ] **Step 6: Commit the shared registry**

```bash
git add Genie02_report/metric_definitions.py Genie02_report/genie02_markdown_report.py tests/test_genie02_regression.py
git commit -m "refactor(report): share metric definitions"
```

### Task 2: Strict Evidence-First Report View

**Files:**
- Create: `vla_eval/web/report_view.py`
- Modify: `vla_eval/web/routes_reports.py`
- Modify: `Genie02_report/genie02_eval_common.py`
- Test: `tests/web/test_reports.py`
- Test: `tests/test_genie02_regression.py`

- [ ] **Step 1: Add failing report-source tests**

Extend the report fixture with a native `session.json` and `episodes.csv`, then assert the page contains current task, FPS, backend, intervention, notes, valid smoothness coverage, dataset fingerprint, and inspection errors. Add a sentinel assertion that old report values such as `30.8%`, `Ep 9`, and `dc67326` are absent.

Add a LeRobot loader regression asserting `_synthesize_lerobot_session()` exposes real optional metadata from `meta/info.json`:

```python
assert session["codebase_version"] == "v3.0"
assert session["robot_type"] == "genie02"
assert session["total_frames"] == 120
assert session["features"]["action"]["shape"] == [10]
```

- [ ] **Step 2: Run the source tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -k "source or quality or task_boundary" -q
.venv/bin/pytest tests/test_genie02_regression.py -k "lerobot_session_metadata" -q
```

Expected: FAIL because the current route does not load session/Episode facts or expose extended LeRobot metadata.

- [ ] **Step 3: Preserve optional LeRobot metadata in the existing loader**

In `_synthesize_lerobot_session()`, copy only validated JSON values already read from `meta/info.json`:

```python
session.update(
    codebase_version=str(info.get("codebase_version", "")),
    robot_type=str(info.get("robot_type", "")),
    total_frames=int(info.get("total_frames", 0)),
    total_tasks=int(info.get("total_tasks", 0)),
    features=info.get("features", {}) if isinstance(info.get("features"), dict) else {},
    splits=info.get("splits", {}) if isinstance(info.get("splits"), dict) else {},
)
```

- [ ] **Step 4: Implement `build_report_view`**

Create `vla_eval/web/report_view.py` with a public function:

```python
def build_report_view(
    *,
    job: EvaluationJob,
    dataset: Dataset,
    output_dir: Path,
) -> dict[str, Any]:
```

It must call `load_session(Path(dataset.path))`, `load_episodes(dataset_root, session)`, `load_episode_metrics(output_dir, session)`, and `load_metrics_core(output_dir, session)`. Join original and derived Episode rows by integer `episode_index`, rejecting duplicate/missing IDs through the existing loaders. Return:

- `headline` with current GSR, counts, TTS, smoothness, review count.
- `summary_facts` with Episode total, smoothness coverage, and VLM execution status.
- `configuration_rows` sourced from session/job/provenance.
- `source_rows` containing source label, interface/path, status, and purpose.
- `quality_rows` containing deterministic counts and inspection errors.
- `episodes` containing outcome, duration, combined/left/right smoothness, space, frames, intervention, notes, skip reason, and optional VLM row.
- `component_rows` from recorded adapter/plugin/image key and dataset schema.
- `evidence_gaps` for training, deployment, hardware, calibration, OOD/robustness/safety, and release policy.
- `release_decision = "未配置自动发版判定"` unless a future persisted release policy and decision are both present.
- `metric_definitions = METRIC_DEFINITIONS`.

Do not compute GSR, TTS, smoothness, action-state error, a risk grade, or a release decision.

- [ ] **Step 5: Make the route use the view builder**

In `routes_reports.py`, keep job/dataset loading, filter validation, download discovery, and response construction. Replace `_load_core_metrics`, `_load_episode_rows`, and route-local composition with `build_report_view`. Catch `EvaluationError`, `OSError`, and invalid persisted data as a 404 `Report is not available`. Apply outcome/review filters only to `view["episodes"]`.

- [ ] **Step 6: Run focused report and loader tests**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -q
.venv/bin/pytest tests/test_genie02_regression.py -q
```

Expected: PASS, including existing filters and the new source/quality assertions.

- [ ] **Step 7: Commit the strict view model**

```bash
git add Genie02_report/genie02_eval_common.py vla_eval/web/report_view.py vla_eval/web/routes_reports.py tests/web/test_reports.py tests/test_genie02_regression.py
git commit -m "feat(report): build views from persisted evidence"
```

### Task 3: Historical Output Contract and Nested VLM Downloads

**Files:**
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `vla_eval/web/routes_reports.py`
- Modify: `tests/web/test_evaluations.py`
- Modify: `tests/web/test_reports.py`

- [ ] **Step 1: Write failing provenance and nested-download tests**

Assert a newly submitted job persists:

```python
assert job.provenance_json["outputs"] == {
    "required": ["episode_metrics.csv", "metrics_core.json", "report_*.md"],
    "optional": [
        "smoothness_curve.svg",
        "attempt_eval/attempt_summary.json",
        "attempt_eval/attempt_summary.csv",
    ],
}
```

Create `attempt_eval/attempt_summary.json` and `.csv` in the report fixture. Assert both appear in the download table and return 200. Assert these requests return 404:

```text
/files/attempt_summary.json
/files/attempt_eval/unknown.json
/files/attempt_eval/../metrics_core.json
/files/%2e%2e/metrics_core.json
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/web/test_evaluations.py -k output_contract -q
.venv/bin/pytest tests/web/test_reports.py -k "nested or traversal" -q
```

Expected: FAIL because outputs are not persisted and `_safe_artifact_path` rejects all nested paths while the download list still checks obsolete root-level attempt files.

- [ ] **Step 3: Persist the immutable output contract**

When `routes_evaluations.py` builds provenance, add lists copied from the loaded profile:

```python
"outputs": {
    "required": list(profile.outputs.required),
    "optional": list(profile.outputs.optional),
},
```

No API key value or other secret is added.

- [ ] **Step 4: Replace filename whitelist with relative artifact paths**

Use exact supported paths:

```python
_EXACT_ARTIFACTS = frozenset({
    "metrics_core.json",
    "episode_metrics.csv",
    "smoothness_curve.svg",
    "attempt_eval/attempt_summary.json",
    "attempt_eval/attempt_summary.csv",
})
```

`_available_downloads()` must test `output_dir / PurePosixPath(relative_path)` and URL-quote each path segment. `_safe_artifact_path()` must parse a normalized `PurePosixPath`, require an exact artifact path or root-level `report_*.md`, resolve strictly, and require containment under the resolved output directory. It must reject absolute paths, `..`, backslashes, control characters, unknown nesting, directories, and symlink escapes.

- [ ] **Step 5: Run evaluation and report security tests**

Run:

```bash
.venv/bin/pytest tests/web/test_evaluations.py tests/web/test_reports.py -q
```

Expected: PASS; genuine nested VLM artifacts download and traversal remains blocked.

- [ ] **Step 6: Commit provenance and downloads**

```bash
git add vla_eval/web/routes_evaluations.py vla_eval/web/routes_reports.py tests/web/test_evaluations.py tests/web/test_reports.py
git commit -m "fix(report): align downloads with output contract"
```

### Task 4: Evidence-First Report Template and Formula Rendering

**Files:**
- Modify: `vla_eval/web/templates/reports/detail.html`
- Modify: `vla_eval/web/static/app.css`
- Modify: `tests/web/test_reports.py`

- [ ] **Step 1: Write failing semantic-structure tests**

Assert the report HTML contains, in this order, stable section IDs:

```python
section_ids = [
    "report-summary",
    "report-configuration",
    "report-sources",
    "report-quality",
    "report-metrics",
    "report-episodes",
    "report-components",
    "report-gaps",
    "report-downloads",
]
positions = [response.text.index(f'id="{value}"') for value in section_ids]
assert positions == sorted(positions)
```

Assert semantic tables exist for configuration, sources, quality, metric definitions, Episodes, components, gaps, and downloads. Assert formulas include `.formula-fraction`, `<sub>success</sub>`, `<sup>2</sup>`, and `<sup>3</sup>`. Assert the page says `未配置自动发版判定` and does not contain `建议暂缓生产发版`.

- [ ] **Step 2: Run the template tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -k "section_order or formula or evidence_gap" -q
```

Expected: FAIL because the current page has only the compact overview, files, and Episode sections.

- [ ] **Step 3: Render the approved hierarchy**

Update `reports/detail.html` to use the section IDs above. Preserve the current header, navigation, headline, SVG preview, filters, Episode states, and downloads. Render all view-model row collections with `<table>`, `<thead>`, `<tbody>`, `<th scope="row">`, and local horizontal `.table-scroll` wrappers.

For each `FormulaFragment`, render escaped `fragment.text` plus optional `fragment.subscript` in `<sub>` and `fragment.superscript` in `<sup>`. For fraction lines, render:

```html
<span class="formula-fraction" aria-label="除以">
  <span class="formula-numerator">...</span>
  <span class="formula-denominator">...</span>
</span>
```

Do not use `|safe`, dynamic HTML strings, MathJax, KaTeX, or a CDN.

- [ ] **Step 4: Add responsive and formula CSS**

Add stable report section spacing, restrained section navigation, a formula row with wrapping, fraction rules, tabular numeric alignment, and breakpoint behavior. Tables scroll only inside their wrapper. At `max-width: 720px`, summary metrics use two columns, headings and commands stack, and formula rows remain within the viewport. Add print rules that hide controls and preserve table headers.

- [ ] **Step 5: Run all report tests**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -q
```

Expected: PASS with all old and new report behaviors retained.

- [ ] **Step 6: Commit the complete report page**

```bash
git add vla_eval/web/templates/reports/detail.html vla_eval/web/static/app.css tests/web/test_reports.py
git commit -m "feat(report): render complete evidence-first report"
```

### Task 5: Focused Regression and Real HTTPS Acceptance

**Files:**
- Modify: `tests/e2e/test_visual_layout.py`
- Modify: `task_plan.md`
- Modify: `progress.md`

- [ ] **Step 1: Add the report to desktop/mobile layout checks**

Extend the authenticated E2E fixture to open a completed report at desktop `1440x1000` and mobile `390x844`. Assert every section heading is visible after scrolling, `document.documentElement.scrollWidth <= window.innerWidth`, table wrappers contain their own overflow, and no formula or button overlaps another visible element.

- [ ] **Step 2: Run the focused E2E test and verify behavior**

Run:

```bash
.venv/bin/pytest tests/e2e/test_visual_layout.py -k report -q
```

Expected: PASS after any necessary CSS-only corrections.

- [ ] **Step 3: Run proportional automated verification**

Run:

```bash
.venv/bin/pytest tests/test_genie02_regression.py tests/web/test_evaluations.py tests/web/test_reports.py -q
.venv/bin/ruff check Genie02_report vla_eval tests
.venv/bin/pytest -q
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and `git diff --check` prints nothing.

- [ ] **Step 4: Verify the real report and downloads over HTTPS**

Open `/reports/d9338238-e7b7-4559-870a-7b33153b9823` on the running local HTTPS service. Verify the page shows 4 Episodes, 2 success, 2 failure, GSR 50.0%, successful TTS 2.500 s, smoothness from the current artifact, VLM not executed, and no old 13-Episode values. Verify each displayed download returns 200 and the formula/long tables remain contained at desktop and mobile sizes.

- [ ] **Step 5: Update planning records and commit verification**

Mark Web report phases 5 and 6 complete in `task_plan.md`; record exact test commands and real-page observations in `progress.md`.

```bash
git add tests/e2e/test_visual_layout.py task_plan.md progress.md
git commit -m "test(report): verify evidence-first workflow"
```
