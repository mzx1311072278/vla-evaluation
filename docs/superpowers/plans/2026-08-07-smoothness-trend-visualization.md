# Smoothness Trend Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make smoothness results readable and diagnostically useful for reports containing hundreds of Episodes without changing metric formulas or downloadable evidence.

**Architecture:** Keep `smoothness_curve.svg` as a deterministic downloadable artifact, but replace the overflowing bar chart with a compact line/scatter overview. Build a numeric chart presentation model from validated `episode_metrics.csv` data so the report page can render an interactive SVG with tooltips and a range control, plus a server-rendered worst-Episode table that works without JavaScript.

**Tech Stack:** Python, FastAPI/Jinja2, deterministic SVG, vanilla JavaScript, CSS, pytest, Playwright.

---

### Task 1: Large-report chart contract

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `tests/web/test_reports.py`

- [ ] Add a generator regression test with 199 smoothness rows that asserts the SVG contains one point per Episode, sparse axis labels, median and P90 reference lines, and no per-point value labels.
- [ ] Add report-view tests that assert numeric chart points and the ten highest smoothness rows are derived only from validated persisted Episode metrics.
- [ ] Add page tests for the interactive chart container, range control, accessible summary, abnormal-Episode table, and retained SVG download link.
- [ ] Run the focused tests and confirm they fail because the new presentation contract is absent.

### Task 2: Deterministic static SVG overview

**Files:**
- Modify: `Genie02_report/genie02_markdown_report.py`
- Test: `tests/test_evaluation.py`

- [ ] Replace fixed-width bars with a line and one point per valid Episode inside the existing 820 by 330 view box.
- [ ] Calculate the median and nearest-rank P90 from the plotted smoothness values and render labelled reference lines.
- [ ] Limit x-axis labels to at most twelve evenly distributed Episode ticks while retaining every point.
- [ ] Encode failure and operator intervention through point colour and outline, preserving the existing evidence semantics.
- [ ] Run the focused generator tests and confirm they pass.

### Task 3: Interactive report trend and abnormal rows

**Files:**
- Modify: `vla_eval/web/report_view.py`
- Modify: `vla_eval/web/templates/reports/detail.html`
- Modify: `vla_eval/web/static/app.css`
- Modify: `tests/web/test_reports.py`

- [ ] Add a focused helper that returns numeric smoothness points, median, P90, maximum, and the ten highest Episode rows with deterministic tie ordering.
- [ ] Render summary statistics, an accessible interactive SVG region, a range slider shown for more than fifty points, and a compact abnormal-Episode table.
- [ ] Add a report-page script that redraws the selected Episode window, uses sparse x-axis ticks, and exposes exact values through an on-chart tooltip and keyboard-focusable points.
- [ ] Preserve the existing generated SVG and CSV download links as the authoritative portable evidence.
- [ ] Run report unit tests and static checks.

### Task 4: Verification and records

**Files:**
- Modify: `task_plan.md`
- Modify: `progress.md`

- [ ] Record the chosen visualization, formula non-change, tests, and browser acceptance results.
- [ ] Run focused report and evaluation tests, then the full test suite once.
- [ ] Run Ruff and `git diff --check`.
- [ ] Restart only the local web processes needed to load the modified Python/template/static assets.
- [ ] Verify a 199-Episode report at desktop and mobile widths: all points remain represented, labels do not overlap, the range control changes the visible window, tooltips identify exact Episodes, the worst-ten table matches the source data, and downloads still work.
