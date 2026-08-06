# VLM API Local Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VLM API branch pass every locally available quality and configuration gate without requiring a real provider or GPU.

**Architecture:** Keep production behavior stable and make narrowly scoped quality fixes in the existing Genie02 modules. Reuse the existing API client and orchestration tests, adding coverage only if inspection reveals an untested acceptance behavior.

**Tech Stack:** Python 3.11, Pytest, Ruff, Docker Compose

---

### Task 1: Keep coverage output out of Git status

**Files:**
- Modify: `.gitignore`

- [x] Add `.coverage` to the generated test artifacts in `.gitignore`.
- [x] Run `git status --short` and verify `.coverage` is absent.

### Task 2: Resolve the current Ruff findings

**Files:**
- Modify: `Genie02_report/genie02_episode_metrics.py`
- Modify: `Genie02_report/genie02_eval_common.py`
- Modify: `Genie02_report/genie02_markdown_report.py`

- [x] Run `.venv/bin/ruff check .` and retain its failing output as the red quality gate.
- [x] Apply only the import, cache, exception handling, timezone, and string-grouping changes required by Ruff.
- [x] Run focused Genie02 tests and confirm they pass.
- [x] Run `.venv/bin/ruff check .` and confirm it returns 0.

### Task 3: Verify VLM API acceptance coverage

**Files:**
- Inspect: `tests/test_vlm_api.py`
- Inspect: `tests/test_attempt_eval_service.py`
- Inspect: `tests/test_evaluation.py`
- Inspect: `tests/web/test_evaluations.py`

- [x] Map configuration, request encoding, retry, malformed response, secret handling, orchestration, and provenance requirements to existing tests.
- [x] Add a focused failing test only if one of those observable behaviors has no coverage.
- [x] Run the VLM API focused test set and confirm it passes.

### Task 4: Run local release gates

**Files:**
- Verify: `docker-compose.yml`
- Verify: `deploy/Dockerfile.evaluation`

- [x] Run `.venv/bin/pytest` and record pass, skip, and warning counts.
- [x] Run `.venv/bin/ruff check .` and confirm it returns 0.
- [x] Detect Docker Compose availability and run `docker compose config --quiet` when available.
- [x] Run a build/config check that does not require GPU or API credentials when supported locally.
- [x] Review `git diff --check`, `git diff --stat`, and `git status --short` before reporting results.
