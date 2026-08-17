# Report Page Visual Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Turn the completed evaluation report into the approved layered operations layout with an authenticated smoothness SVG preview, semantic provenance and download tables, polished Episode details, and unchanged artifact downloads.

**Architecture:** Keep report generation and download security untouched. Extend the report route's presentation model with fixed artifact descriptions and formats, derive the optional SVG preview from the same whitelisted download list, and render the approved structure in the existing Jinja template with report-specific CSS.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Jinja2, pytest, HTML/CSS, Lucide icons

---

## File Structure

- Modify vla_eval/web/routes_reports.py: enrich whitelisted download view data and expose the optional SVG preview URL.
- Modify vla_eval/web/templates/reports/detail.html: render the approved report hierarchy, SVG preview, provenance table, result-file table, filters, and Episode table.
- Modify vla_eval/web/static/app.css: provide report-specific desktop/mobile layout without changing global report behavior.
- Modify tests/web/test_reports.py: verify the new view model, SVG preview behavior, semantic tables, and preserved downloads.

### Task 1: Download View Data and SVG Preview

**Files:**
- Modify: vla_eval/web/routes_reports.py
- Test: tests/web/test_reports.py

- [ ] **Step 1: Write failing tests for artifact metadata and SVG preview**

Add focused tests next to the existing report-page and download tests:

```python
def test_report_page_renders_download_table_metadata_and_svg_preview(
    auth_client, successful_job
):
    output_dir = Path(successful_job.output_dir)
    (output_dir / "smoothness_curve.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        encoding="utf-8",
    )

    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert 'class="report-files-table"' in response.text
    assert "Episode 逐项指标" in response.text
    assert "评测汇总指标" in response.text
    assert "平滑度矢量图" in response.text
    assert "CSV" in response.text
    assert "JSON" in response.text
    assert "SVG" in response.text
    svg_url = f"/reports/{successful_job.id}/files/smoothness_curve.svg"
    assert f'<img src="{svg_url}"' in response.text
    assert f'href="{svg_url}"' in response.text


def test_report_page_omits_svg_preview_when_artifact_is_missing(
    auth_client, successful_job
):
    response = auth_client.get(f"/reports/{successful_job.id}")

    assert response.status_code == 200
    assert 'class="smoothness-preview"' not in response.text
    assert "Episode 逐项指标" in response.text
```

- [ ] **Step 2: Run the new tests and verify the expected failures**

Run:

```bash
.venv/bin/pytest +  tests/web/test_reports.py::test_report_page_renders_download_table_metadata_and_svg_preview +  tests/web/test_reports.py::test_report_page_omits_svg_preview_when_artifact_is_missing -q
```

Expected: FAIL because the current page has no report-files-table, metadata labels, or inline SVG image.

- [ ] **Step 3: Add fixed artifact presentation metadata**

In vla_eval/web/routes_reports.py, add a fixed mapping near the whitelist:

```python
_ARTIFACT_PRESENTATION = {
    "episode_metrics.csv": ("Episode 逐项指标", "CSV"),
    "metrics_core.json": ("评测汇总指标", "JSON"),
    "smoothness_curve.svg": ("平滑度矢量图", "SVG"),
    "attempt_summary.json": ("VLM 尝试汇总", "JSON"),
    "attempt_summary.csv": ("VLM 尝试明细", "CSV"),
}
```

Add a helper that keeps markdown filenames safe and presentation-only:

```python
def _download_view(name: str, job_id: str) -> dict[str, str]:
    description, file_format = _ARTIFACT_PRESENTATION.get(
        name,
        ("完整文本报告", Path(name).suffix.removeprefix(".").upper() or "FILE"),
    )
    return {
        "name": name,
        "url": f"/reports/{job_id}/files/{name}",
        "description": description,
        "format": file_format,
    }
```

Replace both dictionaries appended by _available_downloads with:

```python
downloads.append(_download_view(name, job_id))
```

and:

```python
downloads.append(_download_view(path.name, job_id))
```

After downloads are built in report_detail, derive the optional preview only from the whitelisted list:

```python
smoothness_preview_url = next(
    (
        item["url"]
        for item in downloads
        if item["name"] == "smoothness_curve.svg"
    ),
    None,
)
```

Pass smoothness_preview_url into the template context.

- [ ] **Step 4: Run the helper and report tests**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -q
```

Expected: the new SVG preview test still fails at the template assertion; all existing route/download tests remain green.

- [ ] **Step 5: Commit the route/view-model change after Task 2 makes the feature green**

Do not commit a red intermediate state. Task 1 and Task 2 will be committed together after the template renders the new fields.

### Task 2: Approved Report Template

**Files:**
- Modify: vla_eval/web/templates/reports/detail.html
- Test: tests/web/test_reports.py

- [ ] **Step 1: Add failing structural assertions for the approved hierarchy**

Extend test_report_page_shows_core_metrics with:

```python
assert 'class="report-headline"' in response.text
assert 'class="report-overview"' in response.text
assert 'class="provenance-table"' in response.text
assert 'class="report-filter-form"' in response.text
assert 'data-lucide="download"' in response.text
```

Keep the existing assertions for GSR, success/failure counts, TTS, smoothness, provenance, and download URLs.

- [ ] **Step 2: Run the structural test and verify it fails**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py::test_report_page_shows_core_metrics -q
```

Expected: FAIL on report-headline because the old template does not contain the approved structure.

- [ ] **Step 3: Replace the report template structure**

Update vla_eval/web/templates/reports/detail.html so it contains these semantic sections:

```jinja2
{% extends "base.html" %}
{% block title %}{{ dataset.name }} | 评测报告{% endblock %}
{% block content %}
  <section class="report-page">
    <div class="page-heading">
      <div><p class="eyebrow">评测报告</p><h1>{{ dataset.name }}</h1></div>
      <div class="commands">
        <a class="button secondary-button" href="/evaluations/{{ job.id }}">查看任务</a>
        <a class="button secondary-button" href="/datasets/{{ dataset.id }}">查看数据集</a>
      </div>
    </div>

    <dl class="report-headline">
      <div><dt>GSR</dt><dd>{{ headline.gsr }}</dd></div>
      <div><dt>成功 / 失败</dt><dd><span class="metric-success">{{ headline.n_success }}</span> / <span class="metric-failure">{{ headline.n_failure }}</span></dd></div>
      <div><dt>成功 TTS</dt><dd>{{ headline.tts }}</dd></div>
      <div><dt>待复核</dt><dd>{{ headline.pending_review }}</dd></div>
    </dl>

    <section class="detail-band report-overview">
      {% if smoothness_preview_url %}
        <figure class="smoothness-preview">
          <div class="section-heading">
            <figcaption><h2>平滑度曲线</h2><p class="muted">{{ headline.smoothness }}</p></figcaption>
            <a class="icon-button" href="{{ smoothness_preview_url }}" title="下载平滑度曲线" aria-label="下载平滑度曲线"><i data-lucide="download"></i></a>
          </div>
          <img src="{{ smoothness_preview_url }}" alt="Episode 平滑度曲线">
        </figure>
      {% else %}
        <div class="smoothness-summary">
          <h2>平滑度</h2><p>{{ headline.smoothness }}</p>
        </div>
      {% endif %}

      <div>
        <h2>来源信息</h2>
        <div class="table-scroll">
          <table class="key-value-table provenance-table">
            <tbody>
              <tr><th scope="row">评测配置</th><td>{{ provenance.profile_name }}</td></tr>
              <tr><th scope="row">配置版本</th><td>{{ provenance.profile_version }}</td></tr>
              <tr><th scope="row">VLM 模型</th><td>{{ provenance.vlm_model }}</td></tr>
              <tr><th scope="row">Prompt 版本</th><td>{{ provenance.prompt_version }}</td></tr>
              <tr><th scope="row">应用版本</th><td>{{ provenance.app_version }}</td></tr>
              {% if provenance.git_sha %}<tr><th scope="row">Git SHA</th><td><code>{{ provenance.git_sha }}</code></td></tr>{% endif %}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="detail-band">
      <div class="section-heading"><h2>结果文件</h2><span class="muted">{{ downloads|length }} 个文件</span></div>
      {% if job.output_dir %}<div class="path-row"><code class="path">{{ job.output_dir }}</code><button class="icon-button" type="button" data-copy="{{ job.output_dir }}" title="复制路径" aria-label="复制路径"><i data-lucide="copy"></i></button></div>{% endif %}
      <div class="table-scroll report-files">
        <table class="report-files-table">
          <thead><tr><th>文件</th><th>内容</th><th>格式</th><th><span class="sr-only">操作</span></th></tr></thead>
          <tbody>
            {% for item in downloads %}
              <tr>
                <td><code>{{ item.name }}</code></td>
                <td>{{ item.description }}</td>
                <td>{{ item.format }}</td>
                <td><a class="table-action" href="{{ item.url }}"><i data-lucide="download"></i><span>下载</span></a></td>
              </tr>
            {% else %}<tr><td colspan="4" class="empty">暂无可下载文件</td></tr>{% endfor %}
          </tbody>
        </table>
      </div>
    </section>
```

Retain the existing Episode loop and VLM conditionals exactly, but:

- add class report-filter-form to the filter form and remove its inline style;
- keep the shown/total Episode count;
- keep table-scroll and add class episode-table to the table;
- retain the VLM disclaimer;
- remove the duplicate bottom navigation because it now appears in the page heading.

- [ ] **Step 4: Run report tests and make the template green**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -q
```

Expected: all report tests pass, including the SVG preview, download metadata, filters, malformed CSV fallback, and secure downloads.

- [ ] **Step 5: Commit route, template, and tests**

```bash
git add vla_eval/web/routes_reports.py vla_eval/web/templates/reports/detail.html tests/web/test_reports.py
git commit -m "feat(web): structure report results for scanning"
```

### Task 3: Responsive Styling and Final Verification

**Files:**
- Modify: vla_eval/web/static/app.css

- [ ] **Step 1: Add report-specific CSS**

Append focused styles in vla_eval/web/static/app.css:

```css
.secondary-button {
  border-color: #aeb7b3;
  background: #ffffff;
  color: #176b53;
}
.secondary-button:hover { background: #edf2f0; color: #0e4f3d; }
.report-headline {
  margin: 0;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border-block: 1px solid #d6dcda;
  background: #ffffff;
}
.report-headline > div { min-width: 0; padding: 1rem; border-right: 1px solid #d6dcda; }
.report-headline > div:last-child { border-right: 0; }
.report-headline dd { font-size: 1.2rem; }
.metric-success { color: #176b53; }
.metric-failure { color: #a12b22; }
.report-overview {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(18rem, .65fr);
  gap: 1.5rem;
}
.smoothness-preview { min-width: 0; margin: 0; }
.smoothness-preview figcaption p { margin: .25rem 0 0; }
.smoothness-preview img {
  width: 100%;
  max-height: 24rem;
  margin-top: .75rem;
  display: block;
  object-fit: contain;
  background: #ffffff;
  border: 1px solid #d6dcda;
  border-radius: 6px;
}
.smoothness-summary { align-self: start; }
.key-value-table { min-width: 0; }
.key-value-table th { width: 8rem; text-transform: none; }
.report-files { margin-top: .8rem; }
.report-files-table code { overflow-wrap: anywhere; }
.table-action { display: inline-flex; align-items: center; gap: .35rem; white-space: nowrap; }
.table-action svg { width: 1rem; height: 1rem; }
.report-filter-form {
  margin-bottom: 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: flex-end;
}
.report-filter-form label { min-width: 11rem; }
.episode-table td { vertical-align: middle; }
```

Add to the existing max-width: 720px media query:

```css
.report-page .page-heading .commands { width: 100%; }
.report-page .page-heading .commands .button { min-width: 0; flex: 1; }
.report-headline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.report-headline > div:nth-child(2) { border-right: 0; }
.report-overview { grid-template-columns: 1fr; }
.report-filter-form { align-items: stretch; flex-direction: column; }
.report-filter-form label, .report-filter-form button { width: 100%; }
```

- [ ] **Step 2: Run focused report tests**

Run:

```bash
.venv/bin/pytest tests/web/test_reports.py -q
```

Expected: all report route, filtering, rendering, and download tests pass.

- [ ] **Step 3: Run full automated verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
git diff --check
```

Expected: pytest exits 0, Ruff reports All checks passed, and git diff --check produces no output.

- [ ] **Step 4: Restart and verify the real local report**

Restart only the Uvicorn process on port 8443 with the existing runtime environment. Keep Redis and workers running.

Verify:

- GET /reports/d9338238-e7b7-4559-870a-7b33153b9823 returns 200 after login.
- Core metrics, provenance table, SVG preview, result-file table, and Episode table render.
- Every result-file download URL returns 200 and keeps Content-Disposition attachment.
- The report has no incoherent overlap at desktop width and at or below 720px.

- [ ] **Step 5: Commit styling and final verification changes**

```bash
git add vla_eval/web/static/app.css
git commit -m "style(web): polish evaluation report layout"
```
