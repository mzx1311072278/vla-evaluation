# VLM Camera Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each evaluation persist up to three dataset camera streams and jointly analyze all selected views with exact context-length and per-Episode CUDA peak-memory safeguards.

**Architecture:** Dataset inspection owns camera discovery and persists a stable sorted list. The web layer validates and canonicalizes repeated camera form fields into the evaluation task snapshot, while the worker passes only that snapshot into evaluation. The attempt evaluator groups per-camera video references by Episode, samples every selected stream independently, performs one multimodal request, and records resource observations without changing the existing result schema contract.

**Tech Stack:** Python 3.11, FastAPI/Jinja, SQLAlchemy JSON fields, pandas/pyarrow, PyTorch/Transformers, pytest, Ruff.

---

### Task 1: Discover and Persist Dataset Cameras

**Files:**
- Modify: `vla_eval/datasets.py`
- Modify: `vla_eval/tasks.py`
- Modify: `vla_eval/cli.py`
- Test: `tests/test_datasets.py`
- Test: `tests/test_import_jobs.py`

- [x] **Step 1: Write failing discovery tests**

Add tests proving `DatasetInspection.camera_keys` merges `meta/info.json` video features and Episode parquet video prefixes, returns a sorted tuple, and defaults to empty for non-video formats. Extend import/CLI persistence expectations to require `inspection_json["camera_keys"]`.

- [x] **Step 2: Run discovery tests and verify RED**

Run: `uv run --isolated --extra dev pytest -q tests/test_datasets.py -k camera_keys tests/test_import_jobs.py -k camera_keys`

Expected: failures because `DatasetInspection` has no `camera_keys` and persistence stores only `errors`.

- [x] **Step 3: Implement camera discovery**

Add `camera_keys: tuple[str, ...] = ()` to `DatasetInspection` to preserve positional fixture compatibility. Make `_inspect_lerobot` return its validated camera union alongside episode count/errors, propagate it through `inspect_dataset`, and save `list(inspection.camera_keys)` in import and CLI `inspection_json`.

- [x] **Step 4: Run discovery tests and verify GREEN**

Run the Task 1 command and confirm all selected tests pass.

### Task 2: Canonicalize Web Camera Selection

**Files:**
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `vla_eval/web/templates/evaluations/new.html`
- Modify: `vla_eval/web/templates/evaluations/detail.html`
- Test: `tests/web/test_evaluations.py`

- [x] **Step 1: Write failing form and route tests**

Cover checkbox rendering, stored-camera use, safe inspection fallback for historical records, empty-selection-to-all normalization, stable dataset order, duplicate removal, unknown-key rejection, more-than-three rejection, VLM-disabled normalization to `[]`, params/provenance persistence, run-key separation, and task-detail display.

- [x] **Step 2: Run web tests and verify RED**

Run: `uv run --isolated --extra dev pytest -q tests/web/test_evaluations.py -k camera`

Expected: failures because `camera_keys` is currently an unknown form field and templates have no selector.

- [x] **Step 3: Implement authoritative normalization**

Allow repeated `camera_keys`, reject non-string values, resolve available keys from valid stored inspection metadata or a safe `inspect_dataset` fallback, and canonicalize the task list with a constant limit of three. Include the resolved list in `params_json`, provenance, embedded provenance params, and run-key hashing.

- [x] **Step 4: Implement checkbox UI**

Render all available cameras as native checkboxes with the “none means all” explanation and multi-camera cost warning. Add a small page-local client-side handler that disables unselected boxes after three selections while leaving backend validation authoritative. Display persisted task cameras on the detail page.

- [x] **Step 5: Run web tests and verify GREEN**

Run the Task 2 command and confirm all selected tests pass.

### Task 3: Pass Task Camera Snapshots Through the Worker

**Files:**
- Modify: `vla_eval/tasks.py`
- Modify: `vla_eval/evaluation.py`
- Test: `tests/test_tasks.py`
- Test: `tests/test_evaluation.py`

- [x] **Step 1: Write failing orchestration tests**

Assert the worker reads `params_json.camera_keys`, `run_evaluation` forwards them to `run_profile_vlm`, and an old task without the key falls back to `[profile.image_key]`. Assert VLM-disabled tasks use an empty list.

- [x] **Step 2: Run orchestration tests and verify RED**

Run: `uv run --isolated --extra dev pytest -q tests/test_tasks.py tests/test_evaluation.py -k camera`

Expected: failures because evaluation functions do not accept camera lists.

- [x] **Step 3: Implement snapshot propagation**

Read and validate the persisted JSON list in the worker before execution. Add optional `camera_keys` parameters to `run_evaluation` and `run_profile_vlm`, preserving existing callers, and map the normalized tuple into `AttemptEvalConfig.image_keys`. Do not rediscover cameras at worker runtime.

- [x] **Step 4: Run orchestration tests and verify GREEN**

Run the Task 3 command and confirm all selected tests pass.

### Task 4: Jointly Sample and Analyze Multiple Cameras

**Files:**
- Modify: `Genie02_report/attempt_eval/dataset_reader.py`
- Modify: `Genie02_report/attempt_eval/run_episode_attempt_eval.py`
- Modify: `Genie02_report/attempt_eval/vlm_client.py`
- Test: `tests/test_datasets.py`
- Test: `tests/test_attempt_eval_service.py`
- Test: `tests/test_vlm_api.py`

- [x] **Step 1: Write failing multi-camera service tests**

Define Episode video-stream fixtures and assert metadata groups selected keys by Episode, each stream receives the full per-camera sampling caps, output paths are camera-isolated, frame timestamps contain `camera_key`, the client is called once per Episode with every image, prompt text names each camera, and results contain camera lists plus per-camera counts.

- [x] **Step 2: Run service tests and verify RED**

Run: `uv run --isolated --extra dev pytest -q tests/test_datasets.py tests/test_attempt_eval_service.py tests/test_vlm_api.py -k 'camera or prompt_with_frame_times'`

Expected: failures because metadata and sampling currently carry one video only.

- [x] **Step 3: Implement grouped metadata and sampling**

Add an immutable per-camera video reference and make `read_episode_metadata` accept either the legacy string key or a sequence. Group rows by Episode, enforce consistent Episode fields, sample each video into a filesystem-safe camera subdirectory, append `camera_key` to timestamps, and aggregate paths in deterministic camera order.

- [x] **Step 4: Preserve result compatibility**

Keep legacy `video_file` as the first selected stream for existing consumers, add `video_files`, `camera_keys`, `sampled_frame_count_by_camera`, and preserve aggregate frame count fields. Update prompt rendering for camera labels without changing its output JSON request.

- [x] **Step 5: Run service tests and verify GREEN**

Run the Task 4 command and confirm all selected tests pass.

### Task 5: Enforce Context Budget and Record CUDA Peaks

**Files:**
- Modify: `Genie02_report/attempt_eval/vlm_client.py`
- Modify: `Genie02_report/attempt_eval/run_episode_attempt_eval.py`
- Modify: `vla_eval/vlm_api.py`
- Test: `tests/test_attempt_eval_service.py`
- Test: `tests/test_vlm_api.py`

- [x] **Step 1: Write failing local resource-protection tests**

Use fake Processor tensors/model configs/CUDA APIs to assert Qwen3 reads `text_config.max_position_embeddings`, Qwen2.5 reads top-level `max_position_embeddings`, malformed limits become model configuration failures, `input_tokens + max_new_tokens` over limit skips `generate`, and CUDA allocated/reserved peaks are captured after both successful and failed inference. Assert CPU and API results expose null resource metrics.

- [x] **Step 2: Run resource tests and verify RED**

Run: `uv run --isolated --extra dev pytest -q tests/test_attempt_eval_service.py tests/test_vlm_api.py -k 'context or peak_memory or resource_metrics'`

Expected: failures because the client does not inspect sequence length or report memory metrics.

- [x] **Step 3: Implement exact budget enforcement**

Resolve a positive context limit at model load, inspect the real processed `input_ids.shape[-1]`, and before generation return a sanitized `context_length_exceeded` fallback when the input plus configured output reserve is too large. Add `input_token_count` and `context_token_limit` to every local result.

- [x] **Step 4: Implement resource observations**

Reset CUDA peak stats before each local analysis, read `max_memory_allocated` and `max_memory_reserved` in `finally`, and attach byte counts to the result. Ensure API and CPU paths attach the same keys with null values.

- [x] **Step 5: Run resource tests and verify GREEN**

Run the Task 5 command and confirm all selected tests pass.

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/deployment/czj-shared-storage-guide.zh-CN.md`

- [x] **Step 1: Document operator behavior**

Document task-level camera selection, no-selection semantics, the three-camera limit, per-camera frame multiplication, exact context guard, local-only CUDA metrics, API observability limitations, and a 4090 three-camera smoke test.

- [x] **Step 2: Run targeted suites**

Run: `uv run --isolated --extra dev pytest -q tests/test_datasets.py tests/web/test_evaluations.py tests/test_tasks.py tests/test_evaluation.py tests/test_attempt_eval_service.py tests/test_vlm_api.py`

Expected: all targeted tests pass.

- [x] **Step 3: Run full verification**

Run:

```bash
uv run --isolated --extra dev pytest -q
uv run --isolated --extra dev ruff check vla_eval Genie02_report tests
git diff --check
```

Expected: the full suite and Ruff pass, and `git diff --check` emits no output.

- [x] **Step 4: Review and commit**

Review `git diff --stat`, `git diff`, and `git status --short`; then create one focused local feature commit without pushing.
