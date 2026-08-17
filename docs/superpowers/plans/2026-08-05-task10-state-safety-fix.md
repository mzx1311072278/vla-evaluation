# Task 10 State Safety Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make queued import and evaluation executions generation-safe, confine all worker file access to trusted roots, revalidate dataset identity before work, and preserve actionable failure states.

**Architecture:** Each business job stores an execution token. A worker atomically claims an allowed source state and every callback or terminal write uses `job_id + execution_token + RUNNING state` as a compare-and-set boundary. Evaluation preflight resolves trusted dataset, output, and profile paths before recomputing the dataset fingerprint; recovery also uses conditional SQL updates so it cannot overwrite a terminal state.

**Tech Stack:** Python 3.11, SQLAlchemy 2, SQLite, Redis/RQ, pytest, Ruff.

---

### Task 1: Add Execution-Generation CAS

**Files:**
- Modify: `vla_eval/models.py`
- Modify: `vla_eval/tasks.py`
- Modify: `tests/test_models.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write failing stale-worker and cancellation race tests**

Add tests that claim one evaluation execution, replace its token with a newer execution, then invoke the old stage/progress/success/failure/cancel callbacks. Assert the old callbacks cannot change the new state. Add a barrier-based success/cancel interleaving test and the corresponding import READY/cancel and stale callback tests.

```python
def test_stale_evaluation_callback_cannot_revive_terminal_job(
    db_engine, evaluation_job, task_runtime
):
    token = "11111111-1111-1111-1111-111111111111"
    _claim_evaluation_execution(task_runtime, evaluation_job.id, token)
    with session_scope(db_engine) as session:
        job = session.get_one(EvaluationJob, evaluation_job.id)
        job.state = "SUCCEEDED"
        job.execution_token = None
    with pytest.raises(StaleTaskExecution):
        _update_evaluation_stage(task_runtime, evaluation_job.id, token, "REPORT")
    assert reload_job(db_engine, evaluation_job.id).state == "SUCCEEDED"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'stale or cancel_race' -v`

Expected: stale callbacks overwrite terminal state or success wins despite a committed cancellation.

- [ ] **Step 3: Implement atomic claims and token-scoped writes**

Add nullable indexed `execution_token` fields to `ImportJob` and `EvaluationJob`. Generate a UUID token per invocation. Claim with a conditional SQL update from allowed source states. Require that token and the allowed running state in every callback and terminal update; include `cancel_requested IS false` in successful terminal CAS operations. A row count of zero must raise `StaleTaskExecution` or `EvaluationCancelled` without changing a newer terminal state.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'stale or cancel or callback or success' -v`

Expected: all selected tests pass.

### Task 2: Constrain Worker Paths and Profiles

**Files:**
- Modify: `vla_eval/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write failing path-boundary tests**

Use real minimal Genie02 data. Persist a dataset path outside `data_root/inbox`, an output path outside `data_root/runs/<job_id>`, and profile names containing path separators. Assert the evaluation core is not called and no outside artifact is written. Add a profile file whose YAML `name` differs from the persisted name.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'trusted_path or profile_selector' -v`

Expected: the current worker reaches the evaluation core or writes outside trusted roots.

- [ ] **Step 3: Implement worker trust checks**

Validate the dataset with `validate_published_target(dataset_path, data_root / "inbox")`. Require output to equal the canonical absolute `data_root / "runs" / job_id`; reject symlink ancestors and unsafe ownership or modes before creating files. Accept profile names only as one identifier segment, resolve `<profiles_root>/<name>.yaml` under the trusted root, reject symlink escapes, then require both `profile.name == job.profile_name` and `profile.version == job.profile_version`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'trusted_path or profile_selector' -v`

Expected: all selected tests pass and outside directories remain unchanged.

### Task 3: Revalidate Dataset Identity Before Resume

**Files:**
- Modify: `vla_eval/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write a failing fingerprint-drift test**

Create a READY dataset and evaluation job, mutate the dataset after submission, and request a METRICS, VLM, and REPORT resume. Assert none reaches `run_evaluation`, the evaluation becomes FAILED with `DATASET_CHANGED`, and the Dataset becomes `PREFLIGHT_FAILED` without deleting files.

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'dataset_changed or fingerprint' -v`

Expected: the current worker runs with the stale database fingerprint.

- [ ] **Step 3: Implement preflight identity verification**

Before loading resume artifacts, call `inspect_dataset(dataset_path, allowed_root=dataset_path)`. Require `ready`, the persisted kind, and the exact persisted fingerprint. On mismatch, atomically fail the token-owned evaluation and mark the Dataset `PREFLIGHT_FAILED`; rethrow a stable internal exception so RQ keeps the traceback.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'dataset_changed or fingerprint or resume' -v`

Expected: all selected tests pass.

### Task 4: Make Recovery CAS-Safe and Classify Failures

**Files:**
- Modify: `vla_eval/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write failing recovery and error-code tests**

Use a barrier so recovery selects a RUNNING job, then a worker commits SUCCEEDED before recovery updates. Assert SUCCEEDED remains. Add evaluation failures for CUDA OOM, model loading, and disk-full errors; assert stable safe codes and messages while the original exception is rethrown.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/test_tasks.py -k 'recover or out_of_memory or disk_full or model_load' -v`

Expected: recovery overwrites SUCCEEDED or every failure receives `EVALUATION_FAILED`.

- [ ] **Step 3: Implement conditional recovery and safe classification**

Replace ORM read-then-write recovery with per-job `update(EvaluationJob).where(EvaluationJob.id == job_id, EvaluationJob.state.in_(running_states)).values(state="INTERRUPTED", execution_token=None)` and the equivalent `ImportJob` statement; count only `rowcount == 1`. Classify known exception types and messages into stable codes such as `CUDA_OUT_OF_MEMORY`, `MODEL_LOAD_FAILED`, and `DISK_FULL`, but persist only constant user-safe messages. Preserve the original exception and aggregate any persistence failure.

- [ ] **Step 4: Run focused and full verification**

Run: `.venv/bin/pytest tests/test_tasks.py tests/test_models.py tests/test_import_jobs.py tests/test_remote.py -q`

Expected: pass.

Run: `.venv/bin/pytest -q`

Expected: pass with only the existing hardware-related skip.

Run: `.venv/bin/ruff check vla_eval/models.py vla_eval/tasks.py vla_eval/import_jobs.py tests/test_models.py tests/test_tasks.py`

Expected: `All checks passed!`

Run: `.venv/bin/ruff format --check vla_eval/models.py vla_eval/tasks.py vla_eval/import_jobs.py tests/test_models.py tests/test_tasks.py`

Expected: all listed files already formatted.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short --branch`

Expected: no whitespace errors and only the intended Task 10 files before commit.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-05-task10-state-safety-fix.md \
  vla_eval/models.py vla_eval/tasks.py vla_eval/import_jobs.py \
  tests/test_models.py tests/test_tasks.py
git commit -m "fix: isolate persistent task executions"
```
