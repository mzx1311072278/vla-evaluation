# Qwen3-VL Local Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current Qwen2.5-VL local evaluator while adding a separately selectable `Qwen/Qwen3-VL-8B-Instruct` local backend with deterministic, traceable evaluation behavior.

**Architecture:** Extend the versioned Profile contract with an explicit local `model_family`, then pass it through the existing evaluation service seam into one `LocalVLMClient`. The client uses Transformers `AutoModelForImageTextToText` for both supported families, validates the checkpoint `model_type`, and applies Qwen3-compatible visual preprocessing without changing the prompt or result schema.

**Tech Stack:** Python 3.11, pytest, PyYAML, Transformers 4.57+, qwen-vl-utils 0.0.14, PyTorch, torchvision, FastAPI/RQ evaluation pipeline.

---

## File Map

- `vla_eval/profiles.py`: validate and expose the local `model_family` Profile field.
- `config/profiles/genie02-full.yaml`: explicitly identify the existing Qwen2.5-VL family.
- `config/profiles/genie02-qwen3-vl.yaml`: define the new Qwen3-VL evaluation identity and defaults.
- `Genie02_report/attempt_eval/run_episode_attempt_eval.py`: carry the model family through the reusable attempt-evaluation service and CLI.
- `Genie02_report/attempt_eval/vlm_client.py`: validate checkpoint family, load either model through AutoModel, and run family-compatible preprocessing.
- `vla_eval/evaluation.py`: map a Profile into the attempt service without changing API-backend behavior.
- `vla_eval/web/routes_evaluations.py`: persist the selected model family in job provenance.
- `pyproject.toml`: declare Qwen3-compatible local GPU dependencies.
- `README.md`, `docs/deployment/ubuntu-22.04.md`, `docs/deployment/czj-shared-storage-guide.zh-CN.md`: document model installation, selection, and GPU smoke checks.
- `tests/test_evaluation.py`, `tests/test_attempt_eval_service.py`, `tests/web/test_evaluations.py`: pin Profile, client, orchestration, and provenance behavior.

### Task 1: Extend the Profile contract

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `vla_eval/profiles.py`
- Modify: `config/profiles/genie02-full.yaml`
- Create: `config/profiles/genie02-qwen3-vl.yaml`

- [ ] **Step 1: Write failing Profile tests**

Add assertions that the shipped Qwen2.5 Profile exposes `qwen2_5_vl`, a new Qwen3 Profile exposes `qwen3_vl`, a legacy local Profile without the field defaults to `qwen2_5_vl`, invalid values fail, and API Profiles reject the local-only field.

```python
QWEN3_PROFILE_PATH = Path("config/profiles/genie02-qwen3-vl.yaml")

def test_qwen3_profile_contract():
    profile = load_profile(QWEN3_PROFILE_PATH)
    assert profile.name == "genie02-qwen3-vl"
    assert profile.version == "1.0.0"
    assert profile.vlm.backend == "local"
    assert profile.vlm.model_family == "qwen3_vl"
    assert profile.vlm.model_path.endswith("Qwen3-VL-8B-Instruct")

def test_legacy_local_profile_defaults_model_family(tmp_path):
    raw = _profile_data()
    raw["vlm"].pop("model_family", None)
    assert load_profile(_write_profile(tmp_path, raw)).vlm.model_family == "qwen2_5_vl"
```

- [ ] **Step 2: Run Profile tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -q
```

Expected: FAIL because `VLMProfile` has no `model_family` and the Qwen3 Profile does not exist.

- [ ] **Step 3: Implement the Profile field and files**

Add `model_family: str | None` to `VLMProfile`. For `backend: local`, parse an optional `model_family` defaulting to `qwen2_5_vl` and restrict it to `{qwen2_5_vl, qwen3_vl}`. For `backend: api`, reject `model_family` and store `None`.

Add this to the existing Profile:

```yaml
vlm:
  backend: local
  model_family: qwen2_5_vl
```

Create `genie02-qwen3-vl.yaml` by preserving the same adapter, prompt, sampling, review, and outputs while setting:

```yaml
name: genie02-qwen3-vl
version: 1.0.0
vlm:
  backend: local
  model_family: qwen3_vl
  model_path: /srv/vla-eval/data/models/Qwen3-VL-8B-Instruct
```

- [ ] **Step 4: Run Profile tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -q
```

Expected: PASS.

### Task 2: Pass the model family through evaluation

**Files:**
- Modify: `tests/test_attempt_eval_service.py`
- Modify: `tests/test_evaluation.py`
- Modify: `Genie02_report/attempt_eval/run_episode_attempt_eval.py`
- Modify: `vla_eval/evaluation.py`

- [ ] **Step 1: Write failing service and orchestration tests**

Extend the injected factory expectation so a local service constructs its client with the family:

```python
assert factory_calls == [
    (
        (config.model_path,),
        {
            "model_family": "qwen2_5_vl",
            "max_new_tokens": 256,
            "prompt_version": "genie02-attempt-v1",
        },
    )
]
```

Add a Qwen3 orchestration test that loads `genie02-qwen3-vl`, injects the attempt runner, and asserts `AttemptEvalConfig.model_family == "qwen3_vl"`. Add configuration validation tests for unsupported family values.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/pytest \
  tests/test_attempt_eval_service.py::test_run_attempt_evaluation_uses_injected_client_without_optional_imports \
  tests/test_evaluation.py -q
```

Expected: FAIL because `AttemptEvalConfig` does not carry or forward `model_family`.

- [ ] **Step 3: Implement service propagation**

Add to `AttemptEvalConfig`:

```python
model_family: str = "qwen2_5_vl"
```

Validate it against the two supported values, pass it into the local client factory, and add a CLI option:

```python
parser.add_argument(
    "--model_family",
    choices=["qwen2_5_vl", "qwen3_vl"],
    default="qwen2_5_vl",
)
```

In `run_profile_vlm`, set `AttemptEvalConfig.model_family` from the local Profile. API mode keeps the placeholder configuration value because its injected API factory ignores local model identity.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused pytest command. Expected: PASS.

### Task 3: Add unified Qwen2.5/Qwen3 local loading

**Files:**
- Modify: `tests/test_attempt_eval_service.py`
- Modify: `Genie02_report/attempt_eval/vlm_client.py`

- [ ] **Step 1: Write failing checkpoint and AutoModel tests**

Use temporary model directories with minimal `config.json` files and injected fake `torch`/`transformers` modules. Cover:

```python
def test_local_vlm_client_rejects_checkpoint_family_mismatch(...):
    model_dir.joinpath("config.json").write_text('{"model_type":"qwen2_5_vl"}')
    with pytest.raises(ModelLoadError):
        LocalVLMClient(model_dir, model_family="qwen3_vl")

def test_local_vlm_client_uses_auto_image_text_model_for_qwen3(...):
    model_dir.joinpath("config.json").write_text('{"model_type":"qwen3_vl"}')
    client = LocalVLMClient(model_dir, model_family="qwen3_vl")
    assert auto_model_calls[0][1] == {
        "dtype": "auto",
        "device_map": "auto",
        "local_files_only": True,
    }
```

Also pin the CPU fallback to FP32 with no device map and ensure public errors do not expose the model path.

- [ ] **Step 2: Run loader tests and verify RED**

Run the new loader tests with:

```bash
.venv/bin/pytest tests/test_attempt_eval_service.py -k 'local_vlm_client' -q
```

Expected: FAIL because the client still imports the concrete Qwen2.5 class and does not validate `config.json`.

- [ ] **Step 3: Implement validated AutoModel loading**

Read `config.json` before importing/loading weights. Require the expected `model_type`, then import:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
```

On CUDA, load with `dtype="auto"`, `device_map="auto"`, and `local_files_only=True`. On CPU, use `dtype=torch.float32`, no device map, and move the model to CPU. Preserve the existing safe `ModelLoadError` boundary.

- [ ] **Step 4: Run loader tests and verify GREEN**

Run the same `-k local_vlm_client` command. Expected: PASS.

### Task 4: Make preprocessing and generation Qwen3-compatible

**Files:**
- Modify: `tests/test_attempt_eval_service.py`
- Modify: `Genie02_report/attempt_eval/vlm_client.py`

- [ ] **Step 1: Write a failing inference-contract test**

Build a `LocalVLMClient` with fake processor, model, torch, and `qwen_vl_utils`. Assert:

```python
assert vision_call["image_patch_size"] == 16
assert processor_call["do_resize"] is False
assert inputs_target == model.device
assert generate_call["do_sample"] is False
assert decode_call["clean_up_tokenization_spaces"] is False
```

Use a fake valid JSON response and assert the existing validated result schema remains unchanged.

- [ ] **Step 2: Run the inference-contract test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_attempt_eval_service.py -k 'qwen3_preprocessing' -q
```

Expected: FAIL because the current client passes no patch size, resizes twice, assumes `cuda`, and inherits checkpoint sampling.

- [ ] **Step 3: Implement compatible inference behavior**

Pass `self.processor.image_processor.patch_size` to `process_vision_info`, set `do_resize=False`, move processor output to `self.model.device`, call `generate(..., do_sample=False)`, and decode with `clean_up_tokenization_spaces=False`.

- [ ] **Step 4: Run inference and service tests**

Run:

```bash
.venv/bin/pytest tests/test_attempt_eval_service.py tests/test_vlm_api.py -q
```

Expected: PASS, including unchanged API-backend behavior.

### Task 5: Persist provenance and update deployment dependencies

**Files:**
- Modify: `tests/web/test_evaluations.py`
- Modify: `vla_eval/web/routes_evaluations.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/deployment/ubuntu-22.04.md`
- Modify: `docs/deployment/czj-shared-storage-guide.zh-CN.md`

- [ ] **Step 1: Write a failing provenance test**

Create an evaluation with the Qwen3 Profile and assert:

```python
assert job.profile_name == "genie02-qwen3-vl"
assert job.provenance_json["vlm_model_family"] == "qwen3_vl"
assert job.provenance_json["vlm_model_path"].endswith("Qwen3-VL-8B-Instruct")
```

Also assert API provenance stores `vlm_model_family is None` or omits it consistently with the selected implementation.

- [ ] **Step 2: Run the provenance test and verify RED**

Run:

```bash
.venv/bin/pytest tests/web/test_evaluations.py -k 'provenance' -q
```

Expected: FAIL because provenance has no `vlm_model_family`.

- [ ] **Step 3: Implement provenance and dependency declarations**

Add `vlm_model_family` to local evaluation provenance. Update the GPU extra to:

```toml
"transformers>=4.57.0",
"qwen-vl-utils==0.0.14",
"torchvision",
```

while retaining existing GPU packages. Document separate Qwen2.5 and Qwen3 Profile selection, `/srv` and `/czj` model paths, `VLA_EVAL_PROFILES_ROOT`, dependency import checks, and a one-dataset GPU smoke run. Correct the `/czj` guide so local VLM installation uses `.[gpu,vlm-api]` rather than only `.[dev,vlm-api]`.

- [ ] **Step 4: Run provenance tests and static checks**

Run:

```bash
.venv/bin/pytest tests/web/test_evaluations.py -k 'provenance' -q
.venv/bin/ruff check vla_eval Genie02_report tests
git diff --check
```

Expected: all commands PASS.

### Task 6: Full regression and server handoff

**Files:**
- Verify all modified files above.

- [ ] **Step 1: Run the complete local suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: PASS with no failures.

- [ ] **Step 2: Review the final diff**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: only Qwen3-VL design, research, implementation, tests, Profiles, dependency, and deployment documentation changes are present; no generated model files or secrets are tracked.

- [ ] **Step 3: Provide server verification commands**

Handoff commands must verify:

```bash
python -c "import torch, torchvision, transformers, qwen_vl_utils; print(transformers.__version__, torch.cuda.is_available())"
grep -nE 'model_family|model_path' /czj/code/vla-evaluation/data/profiles/genie02-qwen3-vl.yaml
test -f /czj/model/Qwen3-VL-8B-Instruct/config.json
```

Then restart Web and Evaluation Worker, create a new evaluation using `genie02-qwen3-vl`, enable VLM, and confirm `attempt_eval/attempt_summary.json` plus Qwen3 provenance are present.

No commit is created unless the user explicitly requests one.
