# Controlled Local Dataset Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configured, whitelist-only local directory source that copies datasets through the existing staging, preflight, atomic publication, and READY registration workflow.

**Architecture:** Add `LocalSource` beside `RemoteSource`, with globally unique source names and normalized absolute roots. Reuse the existing `ImportJob` schema and import state machine; `run_import_task` selects a remote or local transport from configuration, while `execute_import` continues to own staging, validation, publication, rollback, progress, and cancellation.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, RQ/Redis, rsync argv execution, Jinja2, vanilla JavaScript, pytest, Ruff.

---

## File Map

- Modify `vla_eval/config.py`: define and load `LocalSource`; reject invalid roots and cross-kind name collisions.
- Create `vla_eval/local.py`: resolve a configured local source directory without symlink escape and build local rsync argv.
- Modify `vla_eval/import_jobs.py`: allow exactly one production transport (`RemoteSource` or `LocalSource`) while preserving shared staging and publication logic.
- Modify `vla_eval/tasks.py`: select the configured transport for persisted import jobs.
- Modify `vla_eval/web/routes_imports.py`: accept local sources and preserve pre-enqueue validation.
- Modify `vla_eval/web/templates/imports/new.html`: present local and SSH sources in one grouped form.
- Modify `vla_eval/web/static/app.js`: filter root choices to the selected source.
- Modify `config/app.example.yaml` and `docs/deployment/ubuntu-22.04.md`: document local source configuration and worker visibility.
- Modify focused tests under `tests/`: cover configuration, path security, transport dispatch, Web submission, regression, and layout.

### Task 1: Local Source Configuration

**Files:**
- Modify: `vla_eval/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/web/conftest.py`
- Modify: `tests/web/test_auth.py`
- Modify: `tests/web/test_health.py`
- Modify: `tests/test_tasks.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests that construct:

```python
raw["local_sources"] = {"this-host": {"roots": [str(tmp_path / "datasets")]}}
config = load_config(_write_config(tmp_path, raw))
assert config.local_sources["this-host"].roots == ((tmp_path / "datasets").resolve(),)
```

Also assert rejection of relative, non-normalized and control-character roots, empty roots, and a name present in both `local_sources` and `remote_sources`.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/test_config.py -q`

Expected: failures because `AppConfig` has no `local_sources` and no `LocalSource` loader.

- [ ] **Step 3: Implement the minimal configuration model**

Add:

```python
@dataclass(frozen=True)
class LocalSource:
    name: str
    roots: tuple[Path, ...]
```

Add `local_sources: Mapping[str, LocalSource]` to `AppConfig`, wrap it in `MappingProxyType`, parse `local_sources`, normalize absolute filesystem roots without requiring them to exist, and reject collisions with remote source names.

- [ ] **Step 4: Update direct AppConfig fixtures**

Every test constructing `AppConfig(...)` directly must pass `local_sources={}` or an explicit local source mapping.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest tests/test_config.py tests/web/test_auth.py tests/web/test_health.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vla_eval/config.py tests/test_config.py tests/web/conftest.py tests/web/test_auth.py tests/web/test_health.py tests/test_tasks.py
git commit -m "feat(import): configure local dataset sources"
```

### Task 2: Local Path Resolution and Transfer Arguments

**Files:**
- Create: `vla_eval/local.py`
- Create: `tests/test_local.py`

- [ ] **Step 1: Write failing path-resolution tests**

Specify this interface:

```python
source = LocalSource(name="this-host", roots=(root,))
resolved = resolve_local_source_directory(source, str(root), "team/run-01")
assert resolved == root / "team" / "run-01"
```

Tests must reject an unregistered root, absolute or traversing relative paths, missing paths, files, symlink components, and a resolved path outside the configured root.

- [ ] **Step 2: Write failing argv tests**

Specify:

```python
argv = build_local_rsync_argv(source_dir, staging_dir)
assert argv[0] == "rsync"
assert "--" in argv
assert argv[-2].endswith("/")
assert argv[-1].endswith("/")
```

Assert each path occupies one argv item and no shell command string is produced.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/test_local.py -q`

Expected: collection failure because `vla_eval.local` does not exist.

- [ ] **Step 4: Implement strict local path resolution**

Create `resolve_local_source_directory()` using the existing relative-path normalizer, exact configured-root matching, `os.lstat()` for each component, symlink rejection, directory checks, readability checks, and final resolved containment verification.

- [ ] **Step 5: Implement local rsync argv**

Create `build_local_rsync_argv()` with argv-only execution, archive/partial/progress flags, `--` option termination, and trailing separators so the dataset contents copy into the task staging directory.

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/pytest tests/test_local.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add vla_eval/local.py tests/test_local.py
git commit -m "feat(import): validate local dataset paths"
```

### Task 3: Shared Import Core Supports Local Transport

**Files:**
- Modify: `vla_eval/import_jobs.py`
- Modify: `tests/test_import_jobs.py`

- [ ] **Step 1: Write failing production-local tests**

Add a production `ImportSpec` containing `local_source=source` and `source=None`. Inject a recording transfer and assert it receives the local argv, reports progress, runs preflight, and atomically publishes the target.

Also test that production mode rejects both sources present, neither source present, mismatched source names, and source/staging/target overlap.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/test_import_jobs.py -q`

Expected: failures because `ImportSpec` has no local transport context.

- [ ] **Step 3: Implement transport selection**

Add `local_source: LocalSource | None` to `ImportSpec`. Require exactly one of `source` and `local_source` in production. In `execute_import`, build remote argv through the existing SSH path or local argv through `resolve_local_source_directory()` and `build_local_rsync_argv()`. Keep `run_rsync`, progress reporting, cancellation, verification, preflight, and publication common.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/test_import_jobs.py -q`

Expected: all pass, including existing remote security and recovery tests.

- [ ] **Step 5: Commit**

```bash
git add vla_eval/import_jobs.py tests/test_import_jobs.py
git commit -m "feat(import): reuse publication flow for local sources"
```

### Task 4: Web Form and Task Dispatch

**Files:**
- Modify: `vla_eval/web/routes_imports.py`
- Modify: `vla_eval/web/templates/imports/new.html`
- Modify: `vla_eval/web/static/app.js`
- Modify: `vla_eval/tasks.py`
- Modify: `tests/web/test_imports.py`
- Modify: `tests/test_tasks.py`

- [ ] **Step 1: Write failing Web tests**

Configure one local and one remote source. Assert the page renders grouped options, local roots are associated with the local source, and POSTing the local tuple persists one queued `ImportJob`. Assert an unknown source or mismatched local root returns 422 before database or queue side effects.

- [ ] **Step 2: Write failing Worker dispatch tests**

Persist a local-source job, inject the existing transfer seam, run `run_import_task`, and assert the resulting `ImportSpec` uses `local_source` without SSH credentials. Keep the existing remote assertions unchanged.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/web/test_imports.py tests/test_tasks.py -q`

Expected: local submissions fail because routes and Worker only consult `remote_sources`.

- [ ] **Step 4: Implement source lookup and form rendering**

Add one helper that resolves a unique source across `local_sources` and `remote_sources`. Render optgroups and root options with `data-source`; update the label to “数据来源”. Add progressive-enhancement JavaScript that disables roots not belonging to the selected source and selects the first valid root.

- [ ] **Step 5: Implement Worker dispatch**

In `run_import_task`, resolve the persisted source from exactly one configured mapping, revalidate its exact root, and construct the appropriate `ImportSpec`. Preserve current state ownership, idempotency, retry, and failure recording.

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/pytest tests/web/test_imports.py tests/test_tasks.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add vla_eval/web/routes_imports.py vla_eval/web/templates/imports/new.html vla_eval/web/static/app.js vla_eval/tasks.py tests/web/test_imports.py tests/test_tasks.py
git commit -m "feat(web): submit local dataset imports"
```

### Task 5: Configuration Documentation and Local Runtime

**Files:**
- Modify: `config/app.example.yaml`
- Modify: `docs/deployment/ubuntu-22.04.md`
- Modify outside Git: `/Users/xueyg/.config/vla-eval/app.yaml`

- [ ] **Step 1: Document the source**

Add an example:

```yaml
local_sources:
  this-mac:
    roots:
      - /Users/xueyg/Downloads/fangdianlang_data
```

Document that paths belong to the Worker host, SMB mounts may be roots, and imports copy into `inbox` before evaluation.

- [ ] **Step 2: Configure this machine**

Add `this-mac` and `/Users/xueyg/Downloads/fangdianlang_data` to the local runtime YAML without changing secrets or remote sources.

- [ ] **Step 3: Run configuration and focused regression checks**

Run: `.venv/bin/pytest tests/test_config.py tests/test_local.py tests/test_import_jobs.py tests/web/test_imports.py tests/test_tasks.py -q`

Expected: all pass.

- [ ] **Step 4: Commit tracked documentation**

```bash
git add config/app.example.yaml docs/deployment/ubuntu-22.04.md
git commit -m "docs(import): explain local dataset sources"
```

### Task 6: Real Dataset Acceptance and Final Verification

**Files:**
- Modify: `progress.md`
- Modify: `task_plan.md`

- [ ] **Step 1: Check capacity and runtime prerequisites**

Confirm at least 4 GB free in the data filesystem, `rsync` is available, Redis is healthy, and transfer/evaluation workers use the updated branch and runtime configuration.

- [ ] **Step 2: Restart local services on updated code**

Restart Uvicorn and both RQ workers without changing the HTTPS URL or Redis data.

- [ ] **Step 3: Submit the real local import**

Use source `this-mac`, root `/Users/xueyg/Downloads/fangdianlang_data`, relative path `fangdianlang_good_only_ee`, and target name `fangdianlang_good_only_ee`. Wait for the terminal state.

- [ ] **Step 4: Verify the published dataset**

Assert the target is under the configured `inbox`, the source remains untouched, status is `READY`, kind is `lerobot`, Episode count is 199, and the persisted fingerprint matches a fresh inspection.

- [ ] **Step 5: Run nonredundant final verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check Genie02_report vla_eval tests
git diff --check
```

Expected: test suite, Ruff, and whitespace checks all pass.

- [ ] **Step 6: Record acceptance and commit**

Update `task_plan.md` and `progress.md` with the real import ID, dataset ID, fingerprint, commands, and results, excluding the unrelated `uv.lock`.

```bash
git add task_plan.md progress.md
git commit -m "test(import): verify local dataset workflow"
```
