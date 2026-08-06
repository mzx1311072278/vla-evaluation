import builtins
import copy
import csv
import json
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from vla_eval.evaluation import (
    EvaluationCallbacks,
    EvaluationCancelled,
    run_evaluation,
)
from vla_eval.profiles import load_profile

PROFILE_PATH = Path("config/profiles/genie02-full.yaml")


def _profile_data() -> dict[str, Any]:
    return yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))


def _write_profile(tmp_path: Path, raw: Any) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _callbacks(
    *,
    stages: list[str] | None = None,
    progress: list[float] | None = None,
    should_cancel=lambda: False,
) -> EvaluationCallbacks:
    return EvaluationCallbacks(
        on_stage=(stages if stages is not None else []).append,
        on_progress=(progress if progress is not None else []).append,
        should_cancel=should_cancel,
    )


def _mock_metrics_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def episode_metrics(_dataset, output):
        (output / "episode_metrics.csv").write_text("generated\n", encoding="utf-8")
        return [{"episode_index": 0}]

    def metrics_core(_dataset, output):
        (output / "metrics_core.json").write_text("{}", encoding="utf-8")
        return {"gsr": 1.0}

    def report(_dataset, output):
        path = output / "report_20260805.md"
        path.write_text("# report\n", encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", episode_metrics)
    monkeypatch.setattr("vla_eval.evaluation.generate_metrics_core", metrics_core)
    monkeypatch.setattr("vla_eval.evaluation.generate_markdown_report", report)


def _persist_resume_artifacts(output: Path) -> None:
    output.mkdir(parents=True)
    with (output / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "session_id",
                "episode_index",
                "outcome",
                "duration_s",
                "smoothness",
                "left_smoothness",
                "right_smoothness",
                "smoothness_space",
                "smoothness_frames",
                "smoothness_skipped_reason",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "test-session",
                "episode_index": 0,
                "outcome": "success",
                "duration_s": "1.000",
                "smoothness_skipped_reason": "trajectory unavailable",
            }
        )
    (output / "metrics_core.json").write_text("{}", encoding="utf-8")


def _attempt_result() -> dict[str, Any]:
    return {
        "episode_index": 0,
        "metadata_episode_success": True,
        "episode_success": True,
        "pre_success_failed_attempt_count": 0,
        "failed_attempts_before_success": [],
        "attempt_count": 1,
        "success_count": 1,
        "failed_count": 0,
        "confidence": 0.9,
        "vlm_valid": True,
        "parse_error": "",
        "needs_manual_review": None,
        "review_note": "",
        "auto_warning": [],
        "review_mode": "manual_review",
        "reason": "final grasp visible",
    }


def _source_episode(**overrides: str) -> dict[str, str]:
    return {
        "episode_index": "0",
        "outcome": "success",
        "duration_s": "1.000",
        **overrides,
    }


def _core_metrics() -> dict[str, Any]:
    empty_summary = {"mean": None, "std": None, "min": None, "max": None, "n_episodes": 0}
    return {
        "schema_version": "1.0",
        "session_id": "test-session",
        "n_episodes": 1,
        "n_success": 1,
        "n_failure": 0,
        "gsr": 1.0,
        "mean_tts_success_s": 1.0,
        "smoothness": {
            "space": "joint",
            "n_episodes": 0,
            "left": dict(empty_summary),
            "right": dict(empty_summary),
        },
    }


def _mock_persisted_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {
            "session_id": "test-session",
            "rollout_mode": "default",
        },
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [_source_episode()])
    monkeypatch.setattr(
        "vla_eval.evaluation.load_metrics_core",
        lambda *_args: _core_metrics(),
    )


def test_run_evaluation_calls_metrics_then_report(tmp_path, monkeypatch):
    stages = []
    _mock_metrics_and_report(monkeypatch)
    profile = load_profile("config/profiles/genie02-full.yaml")
    result = run_evaluation(
        dataset_path=tmp_path,
        output_dir=tmp_path / "run",
        profile=profile,
        vlm_enabled=False,
        callbacks=EvaluationCallbacks(
            on_stage=stages.append,
            on_progress=lambda _value: None,
            should_cancel=lambda: False,
        ),
    )
    assert stages == ["METRICS", "REPORT"]
    assert result.metrics["gsr"] == 1.0


def test_genie02_profile_matches_cli_contract_and_is_deeply_immutable():
    profile = load_profile(PROFILE_PATH)

    assert (profile.name, profile.version, profile.adapter, profile.plugin) == (
        "genie02-full",
        "1.0.0",
        "genie02",
        "genie02-attempt-eval",
    )
    assert profile.image_key == "observation.images.right_wrist"
    assert profile.vlm.sampling.max_global_frames == 8
    assert profile.vlm.sampling.global_sample_interval == 2.0
    assert profile.vlm.sampling.max_dense_frames == 8
    assert profile.vlm.sampling.dense_sample_interval == 0.5
    assert profile.vlm.sampling.dense_region == "full"
    assert profile.vlm.max_image_size == 336
    assert profile.vlm.max_new_tokens == 256
    assert profile.vlm.backend == "local"
    assert profile.vlm.api is None
    assert profile.vlm.model_path
    assert profile.review.confidence_threshold == 0.7
    assert profile.review.min_episode_duration == 3.0
    assert profile.review.min_sampled_frames == 3
    assert profile.outputs.required == (
        "episode_metrics.csv",
        "metrics_core.json",
        "report_*.md",
    )
    assert profile.outputs.optional == (
        "smoothness_curve.svg",
        "attempt_eval/attempt_summary.json",
        "attempt_eval/attempt_summary.csv",
    )
    with pytest.raises(FrozenInstanceError):
        profile.vlm.sampling.max_dense_frames = 9
    with pytest.raises(FrozenInstanceError):
        profile.review.mode = "auto_review"
    with pytest.raises(TypeError):
        profile.outputs.required[0] = "changed"


@pytest.mark.parametrize(
    "location",
    ["profile", "vlm", "sampling", "review", "outputs"],
)
def test_load_profile_rejects_unknown_fields_at_every_level(tmp_path: Path, location: str):
    raw = _profile_data()
    target = {
        "profile": raw,
        "vlm": raw["vlm"],
        "sampling": raw["vlm"]["sampling"],
        "review": raw["review"],
        "outputs": raw["outputs"],
    }[location]
    target["surprise"] = "value"

    with pytest.raises(ValueError, match="unknown fields.*surprise"):
        load_profile(_write_profile(tmp_path, raw))


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("profile", "adapter"),
        ("vlm", "prompt_version"),
        ("sampling", "dense_region"),
        ("review", "mode"),
        ("outputs", "optional"),
    ],
)
def test_load_profile_rejects_missing_fields_at_every_level(
    tmp_path: Path, location: str, field: str
):
    raw = _profile_data()
    target = {
        "profile": raw,
        "vlm": raw["vlm"],
        "sampling": raw["vlm"]["sampling"],
        "review": raw["review"],
        "outputs": raw["outputs"],
    }[location]
    del target[field]

    with pytest.raises(ValueError, match=f"missing required fields.*{field}"):
        load_profile(_write_profile(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("name",), ""),
        (("version",), "1.0"),
        (("adapter",), "unsupported"),
        (("plugin",), "unsupported"),
        (("vlm", "max_image_size"), True),
        (("vlm", "max_new_tokens"), 0),
        (("vlm", "sampling", "max_global_frames"), 1.5),
        (("vlm", "sampling", "max_dense_frames"), True),
        (("vlm", "sampling", "global_sample_interval"), 0),
        (("vlm", "sampling", "dense_sample_interval"), float("inf")),
        (("vlm", "sampling", "dense_region"), "middle"),
        (("review", "mode"), "automatic"),
        (("review", "confidence_threshold"), True),
        (("review", "confidence_threshold"), 1.1),
        (("review", "min_episode_duration"), -1),
        (("review", "min_sampled_frames"), 0),
        (("outputs", "required"), "metrics_core.json"),
    ],
)
def test_load_profile_rejects_wrong_types_ranges_and_enums(
    tmp_path: Path, path: tuple[str, ...], value: Any
):
    raw = _profile_data()
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError)):
        load_profile(_write_profile(tmp_path, raw))


def test_load_profile_rejects_unsupported_prompt_version(tmp_path: Path):
    raw = _profile_data()
    raw["vlm"]["prompt_version"] = "made-up-v99"

    with pytest.raises(ValueError, match="prompt_version must be one of.*genie02-attempt-v1"):
        load_profile(_write_profile(tmp_path, raw))


def test_load_profile_rejects_duplicate_yaml_mapping_keys(tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "name: first-name\n" + PROFILE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate mapping key.*name"):
        load_profile(path)


@pytest.mark.parametrize(
    "unsafe",
    ["/tmp/metrics.json", "../metrics.json", "sub/../metrics.json", "sub\\metrics.json", "*.json"],
)
def test_load_profile_rejects_unsafe_output_paths(tmp_path: Path, unsafe: str):
    raw = _profile_data()
    raw["outputs"]["optional"].append(unsafe)

    with pytest.raises(ValueError, match="output"):
        load_profile(_write_profile(tmp_path, raw))


def test_load_profile_rejects_duplicate_and_missing_required_outputs(tmp_path: Path):
    duplicate = _profile_data()
    duplicate["outputs"]["optional"].append("metrics_core.json")
    with pytest.raises(ValueError, match="duplicate"):
        load_profile(_write_profile(tmp_path, duplicate))

    incomplete = _profile_data()
    incomplete["outputs"]["required"].remove("metrics_core.json")
    with pytest.raises(ValueError, match="required output"):
        load_profile(_write_profile(tmp_path, incomplete))


def _api_profile_data() -> dict[str, Any]:
    """A valid API-backend profile derived from the shipped local profile."""
    raw = _profile_data()
    raw["name"] = "genie02-api"
    raw["vlm"] = {
        "backend": "api",
        "prompt_version": raw["vlm"]["prompt_version"],
        "api": {
            "base_url": "https://vlm.example.internal/v1",
            "model": "qwen2.5-vl-7b-instruct",
            "api_key_env": "VLA_EVAL_VLM_API_KEY",
            "timeout": 60,
            "max_retries": 3,
        },
        "sampling": raw["vlm"]["sampling"],
        "max_image_size": raw["vlm"]["max_image_size"],
        "max_new_tokens": raw["vlm"]["max_new_tokens"],
    }
    return raw


def test_api_profile_loads_with_backend_api(tmp_path: Path):
    profile = load_profile(_write_profile(tmp_path, _api_profile_data()))
    assert profile.vlm.backend == "api"
    assert profile.vlm.model_path is None
    assert profile.vlm.api is not None
    assert profile.vlm.api.base_url == "https://vlm.example.internal/v1"
    assert profile.vlm.api.model == "qwen2.5-vl-7b-instruct"
    assert profile.vlm.api.api_key_env == "VLA_EVAL_VLM_API_KEY"
    assert profile.vlm.api.timeout == 60.0
    assert profile.vlm.api.max_retries == 3


def test_api_profile_applies_optional_defaults(tmp_path: Path):
    raw = _api_profile_data()
    del raw["vlm"]["api"]["timeout"]
    del raw["vlm"]["api"]["max_retries"]
    profile = load_profile(_write_profile(tmp_path, raw))
    assert profile.vlm.api.timeout == 60.0
    assert profile.vlm.api.max_retries == 3


def test_load_profile_rejects_api_block_when_backend_local(tmp_path: Path):
    raw = _profile_data()
    raw["vlm"]["api"] = {
        "base_url": "https://vlm.example.internal/v1",
        "model": "qwen2.5-vl-7b-instruct",
        "api_key_env": "VLA_EVAL_VLM_API_KEY",
    }
    with pytest.raises(ValueError, match="backend=local"):
        load_profile(_write_profile(tmp_path, raw))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r["vlm"].pop("api"), "missing required fields.*api"),
        (lambda r: r["vlm"]["api"].pop("base_url"), "missing required fields.*base_url"),
        (lambda r: r["vlm"]["api"].pop("model"), "missing required fields.*model"),
        (lambda r: r["vlm"]["api"].pop("api_key_env"), "missing required fields.*api_key_env"),
        (lambda r: r["vlm"]["api"].__setitem__("api_key_env", "lowercase"), "api_key_env"),
        (lambda r: r["vlm"]["api"].__setitem__("api_key_env", "9LEADING"), "api_key_env"),
        (lambda r: r["vlm"]["api"].__setitem__("base_url", "ftp://x/v1"), "base_url"),
        (lambda r: r["vlm"]["api"].__setitem__("timeout", 0), "timeout"),
        (lambda r: r["vlm"]["api"].__setitem__("timeout", 601), "timeout"),
        (lambda r: r["vlm"]["api"].__setitem__("max_retries", -1), "max_retries"),
        (lambda r: r["vlm"]["api"].__setitem__("max_retries", 11), "max_retries"),
        (lambda r: r["vlm"].__setitem__("backend", "cloud"), "backend"),
        (lambda r: r["vlm"]["api"].__setitem__("surprise", "x"), "unknown fields.*surprise"),
    ],
)
def test_load_profile_rejects_invalid_api_backend_config(
    tmp_path: Path, mutate, match: str
):
    raw = _api_profile_data()
    mutate(raw)
    with pytest.raises((TypeError, ValueError), match=match):
        load_profile(_write_profile(tmp_path, raw))


def test_run_evaluation_calls_optional_vlm_between_metrics_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stages: list[str] = []
    _mock_metrics_and_report(monkeypatch)

    def vlm(_dataset, output, _profile, _callbacks):
        output.mkdir(parents=True)
        path = output / "attempt_summary.json"
        path.write_text(json.dumps([_attempt_result()]), encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.run_profile_vlm", vlm)

    result = run_evaluation(
        tmp_path,
        tmp_path / "run",
        load_profile(PROFILE_PATH),
        True,
        _callbacks(stages=stages),
    )

    assert stages == ["METRICS", "VLM", "REPORT"]
    assert result.vlm_summary_path == tmp_path / "run/attempt_eval/attempt_summary.json"


def test_run_evaluation_does_not_import_optional_vlm_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_metrics_and_report(monkeypatch)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith("attempt_eval.run_episode_attempt_eval"):
            raise AssertionError("optional VLM service imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    run_evaluation(
        tmp_path,
        tmp_path / "run",
        load_profile(PROFILE_PATH),
        False,
        _callbacks(),
    )


def test_run_profile_vlm_maps_profile_and_callbacks_to_task7_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from vla_eval import evaluation

    configs: list[Any] = []
    cancellation = lambda: False
    progress_values: list[float] = []
    fake_service = ModuleType("Genie02_report.attempt_eval.run_episode_attempt_eval")

    @dataclass
    class FakeAttemptEvalConfig:
        dataset_root: Path
        model_path: Path
        prompt_version: str
        image_key: str
        output_dir: Path
        max_image_size: int
        max_global_frames: int
        global_sample_interval: float
        max_dense_frames: int
        dense_sample_interval: float
        dense_region: str
        review_mode: str
        confidence_threshold: float
        min_episode_duration: float
        min_sampled_frames: int
        max_new_tokens: int

    def fake_run(config, *, progress, should_cancel):
        configs.append(config)
        assert should_cancel is cancellation
        progress(1, 4, "sampling")
        config.output_dir.mkdir(parents=True)
        (config.output_dir / "attempt_summary.json").write_text("[]", encoding="utf-8")

    fake_service.AttemptEvalConfig = FakeAttemptEvalConfig
    fake_service.run_attempt_evaluation = fake_run
    monkeypatch.setitem(sys.modules, fake_service.__name__, fake_service)
    profile = load_profile(PROFILE_PATH)
    output = tmp_path / "attempt_eval"

    result = evaluation.run_profile_vlm(
        tmp_path,
        output,
        profile,
        _callbacks(progress=progress_values, should_cancel=cancellation),
    )

    assert result == output / "attempt_summary.json"
    assert configs == [
        FakeAttemptEvalConfig(
            dataset_root=tmp_path,
            model_path=Path(profile.vlm.model_path),
            prompt_version="genie02-attempt-v1",
            image_key="observation.images.right_wrist",
            output_dir=output,
            max_image_size=336,
            max_global_frames=8,
            global_sample_interval=2.0,
            max_dense_frames=8,
            dense_sample_interval=0.5,
            dense_region="full",
            review_mode="manual_review",
            confidence_threshold=0.7,
            min_episode_duration=3.0,
            min_sampled_frames=3,
            max_new_tokens=256,
        )
    ]
    assert progress_values == [45.0]


@pytest.mark.parametrize("resume_from", ["", "metrics", "DONE", None])
def test_run_evaluation_rejects_invalid_resume_stage(tmp_path: Path, resume_from: Any):
    with pytest.raises(ValueError, match="resume_from"):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
            resume_from=resume_from,
        )


def test_run_evaluation_checks_cancellation_before_first_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stages: list[str] = []
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_episode_metrics",
        lambda *_args: pytest.fail("metrics started"),
    )
    with pytest.raises(EvaluationCancelled, match="cancelled"):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(stages=stages, should_cancel=lambda: True),
        )
    assert stages == []


def test_run_evaluation_cancellation_after_metrics_prevents_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_metrics_and_report(monkeypatch)
    checks = iter([False, True])
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda *_args: pytest.fail("report started"),
    )

    with pytest.raises(EvaluationCancelled):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(should_cancel=lambda: next(checks)),
        )


def test_run_evaluation_cancellation_after_vlm_prevents_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_metrics_and_report(monkeypatch)
    checks = iter([False, False, True])
    monkeypatch.setattr(
        "vla_eval.evaluation.run_profile_vlm", lambda *_args: Path("attempt_summary.json")
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda *_args: pytest.fail("report started"),
    )

    with pytest.raises(EvaluationCancelled):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            True,
            _callbacks(should_cancel=lambda: next(checks)),
        )


def test_resume_vlm_loads_metrics_and_runs_vlm_then_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    stages: list[str] = []
    _mock_persisted_loaders(monkeypatch)
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_episode_metrics",
        lambda *_args: pytest.fail("metrics regenerated"),
    )

    def fake_vlm(_dataset, vlm_output, _profile, _callbacks):
        vlm_output.mkdir(parents=True)
        path = vlm_output / "attempt_summary.json"
        path.write_text(json.dumps([_attempt_result()]), encoding="utf-8")
        return path

    def fake_report(_dataset, actual_output):
        path = actual_output / "report_20260805.md"
        path.write_text("# report\n", encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.run_profile_vlm", fake_vlm)
    monkeypatch.setattr("vla_eval.evaluation.generate_markdown_report", fake_report)

    result = run_evaluation(
        tmp_path,
        output,
        load_profile(PROFILE_PATH),
        True,
        _callbacks(stages=stages),
        resume_from="VLM",
    )

    assert result.metrics["gsr"] == 1.0
    assert stages == ["VLM", "REPORT"]


@pytest.mark.parametrize("missing", ["episode_metrics.csv", "metrics_core.json"])
def test_resume_rejects_missing_metrics_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    (output / missing).unlink()
    monkeypatch.setattr(
        "vla_eval.evaluation.load_metrics_core", lambda *_args: pytest.fail("loader called")
    )

    with pytest.raises(ValueError, match=f"missing required artifacts.*{missing}"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
            resume_from="REPORT",
        )


def test_resume_report_with_vlm_requires_existing_attempt_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    _mock_persisted_loaders(monkeypatch)

    with pytest.raises(ValueError, match="cannot resume REPORT.*attempt_summary.json"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            True,
            _callbacks(),
            resume_from="REPORT",
        )


@pytest.mark.parametrize(
    ("vlm_enabled", "resume_from"),
    [(False, "METRICS"), (True, "METRICS"), (True, "VLM"), (False, "REPORT")],
)
def test_progress_is_monotonic_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vlm_enabled: bool,
    resume_from: str,
):
    output = tmp_path / f"run-{vlm_enabled}-{resume_from}"
    progress: list[float] = []
    _mock_metrics_and_report(monkeypatch)
    if resume_from != "METRICS":
        _persist_resume_artifacts(output)
        _mock_persisted_loaders(monkeypatch)

    def fake_vlm(_dataset, vlm_output, _profile, callbacks):
        callbacks.on_progress(45.0)
        callbacks.on_progress(75.0)
        vlm_output.mkdir(parents=True)
        path = vlm_output / "attempt_summary.json"
        path.write_text(json.dumps([_attempt_result()]), encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.run_profile_vlm", fake_vlm)

    run_evaluation(
        tmp_path,
        output,
        load_profile(PROFILE_PATH),
        vlm_enabled,
        _callbacks(progress=progress),
        resume_from=resume_from,
    )

    assert progress == sorted(progress)
    assert all(0.0 <= value <= 100.0 for value in progress)
    assert progress[-1] == 100.0


def test_run_evaluation_creates_output_dir_and_rejects_unsafe_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "parent/run"
    _mock_metrics_and_report(monkeypatch)
    run_evaluation(
        tmp_path,
        output,
        load_profile(PROFILE_PATH),
        False,
        _callbacks(),
    )
    assert output.is_dir()

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="output"):
        run_evaluation(
            tmp_path,
            file_path,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
        )

    symlink_path = tmp_path / "output-link"
    symlink_path.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        run_evaluation(
            tmp_path,
            symlink_path,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
        )


@pytest.mark.parametrize(
    ("initial_progress", "error"),
    [
        (True, TypeError),
        ("50", TypeError),
        (-0.1, ValueError),
        (100.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_run_evaluation_rejects_invalid_initial_progress(
    tmp_path: Path,
    initial_progress: Any,
    error: type[Exception],
):
    with pytest.raises(error, match="initial_progress"):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
            initial_progress=initial_progress,
        )


def test_vlm_then_report_retries_keep_persisted_progress_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    progress: list[float] = []
    _mock_persisted_loaders(monkeypatch)

    def fake_vlm(_dataset, vlm_output, _profile, callbacks):
        callbacks.on_progress(45.0)
        vlm_output.mkdir(parents=True)
        summary = vlm_output / "attempt_summary.json"
        summary.write_text(json.dumps([_attempt_result()]), encoding="utf-8")
        return summary

    monkeypatch.setattr("vla_eval.evaluation.run_profile_vlm", fake_vlm)
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("report failed")),
    )
    callbacks = _callbacks(progress=progress)

    with pytest.raises(RuntimeError, match="report failed"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            True,
            callbacks,
            resume_from="VLM",
            initial_progress=30.0,
        )

    persisted = progress[-1]
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda _dataset, actual_output: actual_output / "report_20260805.md",
    )
    (output / "report_20260805.md").write_text("# report\n", encoding="utf-8")
    run_evaluation(
        tmp_path,
        output,
        load_profile(PROFILE_PATH),
        True,
        callbacks,
        resume_from="REPORT",
        initial_progress=persisted,
    )

    assert progress == sorted(progress)
    assert progress[-1] == 100.0


def test_run_evaluation_rejects_vlm_resume_when_vlm_is_disabled(tmp_path: Path):
    with pytest.raises(ValueError, match="resume_from.*VLM.*vlm_enabled"):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
            resume_from="VLM",
        )


@pytest.mark.parametrize("callback_name", ["on_stage", "on_progress", "should_cancel"])
def test_callback_exceptions_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_name: str,
):
    _mock_metrics_and_report(monkeypatch)

    def fail(*_args):
        raise RuntimeError(f"{callback_name} failed")

    callbacks = EvaluationCallbacks(
        on_stage=fail if callback_name == "on_stage" else lambda _stage: None,
        on_progress=fail if callback_name == "on_progress" else lambda _value: None,
        should_cancel=fail if callback_name == "should_cancel" else lambda: False,
    )

    with pytest.raises(RuntimeError, match=f"{callback_name} failed"):
        run_evaluation(
            tmp_path,
            tmp_path / f"run-{callback_name}",
            load_profile(PROFILE_PATH),
            False,
            callbacks,
        )


def test_load_persisted_metrics_parses_episode_csv_and_checks_core_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from vla_eval import evaluation

    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {"session_id": "test-session", "rollout_mode": "default"},
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [_source_episode()])
    monkeypatch.setattr(
        "vla_eval.evaluation.load_metrics_core",
        lambda *_args: _core_metrics(),
    )

    metrics = evaluation.load_persisted_metrics(tmp_path, output)

    assert metrics["n_episodes"] == 1


@pytest.mark.parametrize(
    "metrics",
    [
        {"n_episodes": 2, "n_success": 1, "n_failure": 1, "gsr": 0.5},
        {"n_episodes": 1, "n_success": 0, "n_failure": 1, "gsr": 0.0},
        {"n_episodes": 1, "n_success": 1, "n_failure": 0, "gsr": float("nan")},
        {"n_episodes": True, "n_success": 1, "n_failure": 0, "gsr": 1.0},
    ],
)
def test_load_persisted_metrics_rejects_inconsistent_numeric_essentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics: dict[str, Any],
):
    from vla_eval import evaluation

    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {"session_id": "test-session", "rollout_mode": "default"},
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [_source_episode()])
    monkeypatch.setattr("vla_eval.evaluation.load_metrics_core", lambda *_args: metrics)

    with pytest.raises(ValueError, match="persisted metrics"):
        evaluation.load_persisted_metrics(tmp_path, output)


def test_resume_validates_episode_metrics_before_starting_vlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    (output / "episode_metrics.csv").write_text("wrong,columns\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {"session_id": "test-session", "rollout_mode": "default"},
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [_source_episode()])
    monkeypatch.setattr(
        "vla_eval.evaluation.run_profile_vlm", lambda *_args: pytest.fail("VLM started")
    )

    with pytest.raises(ValueError, match="persisted metrics.*episode_metrics.csv"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            True,
            _callbacks(),
            resume_from="VLM",
        )


@pytest.mark.parametrize(
    "summary",
    [
        {"episode_index": 0},
        [{}],
        [{**_attempt_result(), "episode_index": True}],
        [{**_attempt_result(), "confidence": float("nan")}],
        [{**_attempt_result(), "auto_warning": "low_confidence"}],
        [_attempt_result(), _attempt_result()],
    ],
)
def test_load_attempt_summary_rejects_malformed_writer_schema(tmp_path: Path, summary: Any):
    from vla_eval import evaluation

    path = tmp_path / "attempt_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="attempt_summary.json"):
        evaluation.load_attempt_summary(path)


def test_resume_report_validates_attempt_summary_before_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    summary = output / "attempt_eval/attempt_summary.json"
    summary.parent.mkdir()
    summary.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {"session_id": "test-session", "rollout_mode": "default"},
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [_source_episode()])
    monkeypatch.setattr(
        "vla_eval.evaluation.load_metrics_core",
        lambda *_args: _core_metrics(),
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda *_args: pytest.fail("report started"),
    )

    with pytest.raises(ValueError, match="attempt_summary.json"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            True,
            _callbacks(),
            resume_from="REPORT",
        )


def test_required_outputs_and_report_path_are_validated_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    progress: list[float] = []

    def episode_metrics(_dataset, actual_output):
        (actual_output / "episode_metrics.csv").write_text("generated\n", encoding="utf-8")
        return []

    def report(_dataset, actual_output):
        path = actual_output / "report_20260805.md"
        path.write_text("# report\n", encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", episode_metrics)
    monkeypatch.setattr("vla_eval.evaluation.generate_metrics_core", lambda *_args: {"gsr": 1.0})
    monkeypatch.setattr("vla_eval.evaluation.generate_markdown_report", report)

    with pytest.raises(ValueError, match="required output.*metrics_core.json"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(progress=progress),
        )

    assert 100.0 not in progress


@pytest.mark.parametrize("case", ["outside", "symlink", "not-allowed"])
def test_returned_report_must_be_safe_and_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
):
    output = tmp_path / "run"
    _mock_metrics_and_report(monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")

    def unsafe_report(_dataset, actual_output):
        if case == "outside":
            return outside
        if case == "symlink":
            path = actual_output / "report_20260805.md"
            path.symlink_to(outside)
            return path
        path = actual_output / "unexpected.md"
        path.write_text("unexpected\n", encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.generate_markdown_report", unsafe_report)

    with pytest.raises(ValueError, match="report|symbolic link|allowlist"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
        )


@pytest.mark.parametrize(
    "relative",
    ["metrics_core.json", "attempt_eval", "smoothness_curve.svg"],
)
def test_output_preflight_rejects_child_symlinks_before_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
):
    output = tmp_path / "run"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=relative == "attempt_eval")
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_episode_metrics",
        lambda *_args: pytest.fail("writer started"),
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
        )


@pytest.mark.parametrize(
    ("corruption", "value"),
    [
        ("mean_tts_success_s", 99.0),
        ("smoothness", 99.0),
        ("source_index", "1"),
        ("source_outcome", "failure"),
        ("source_duration", "2.000"),
    ],
)
def test_persisted_metrics_rebuild_rejects_complete_core_or_source_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    value: Any,
):
    from vla_eval import evaluation

    output = tmp_path / "run"
    _persist_resume_artifacts(output)
    session = {"session_id": "test-session", "rollout_mode": "default"}
    source = _source_episode()
    persisted = copy.deepcopy(_core_metrics())
    if corruption == "smoothness":
        persisted["smoothness"]["left"]["mean"] = value
    elif corruption.startswith("source_"):
        source_field = {
            "source_index": "episode_index",
            "source_outcome": "outcome",
            "source_duration": "duration_s",
        }[corruption]
        source[source_field] = value
    else:
        persisted[corruption] = value
    monkeypatch.setattr("vla_eval.evaluation.load_session", lambda _dataset: session)
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [source])
    monkeypatch.setattr("vla_eval.evaluation.load_metrics_core", lambda *_args: persisted)

    with pytest.raises(ValueError, match="persisted metrics"):
        evaluation.load_persisted_metrics(tmp_path, output)


def test_zero_episode_metrics_support_initial_run_and_report_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "run"
    fieldnames = (
        "session_id",
        "episode_index",
        "outcome",
        "duration_s",
        "smoothness",
        "left_smoothness",
        "right_smoothness",
        "smoothness_space",
        "smoothness_frames",
        "smoothness_skipped_reason",
    )
    zero_core = _core_metrics()
    zero_core.update(
        n_episodes=0,
        n_success=0,
        n_failure=0,
        gsr=0.0,
        mean_tts_success_s=None,
    )

    def episode_metrics(_dataset, actual_output):
        with (actual_output / "episode_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return []

    def metrics_core(_dataset, actual_output):
        (actual_output / "metrics_core.json").write_text(json.dumps(zero_core), encoding="utf-8")
        return zero_core

    def report(_dataset, actual_output):
        path = actual_output / "report_20260805.md"
        path.write_text("# empty report\n", encoding="utf-8")
        return path

    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", episode_metrics)
    monkeypatch.setattr("vla_eval.evaluation.generate_metrics_core", metrics_core)
    monkeypatch.setattr("vla_eval.evaluation.generate_markdown_report", report)
    profile = load_profile(PROFILE_PATH)

    first = run_evaluation(tmp_path, output, profile, False, _callbacks())
    assert first.metrics["n_episodes"] == 0

    monkeypatch.setattr(
        "vla_eval.evaluation.load_session",
        lambda _dataset: {"session_id": "test-session", "rollout_mode": "default"},
    )
    monkeypatch.setattr("vla_eval.evaluation.load_episodes", lambda *_args: [])
    monkeypatch.setattr("vla_eval.evaluation.load_metrics_core", lambda *_args: zero_core)
    resumed = run_evaluation(
        tmp_path,
        output,
        profile,
        False,
        _callbacks(),
        resume_from="REPORT",
    )
    assert resumed.metrics["n_episodes"] == 0


@pytest.mark.parametrize("branch", ["new-vlm", "report-resume"])
@pytest.mark.parametrize("indices", [[], [0, 0], [1], [0, 1]])
def test_attempt_summary_indices_must_exactly_match_before_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    indices: list[int],
):
    output = tmp_path / "run"
    _mock_metrics_and_report(monkeypatch)
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda *_args: pytest.fail("report started"),
    )

    def write_summary(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([{**_attempt_result(), "episode_index": index} for index in indices]),
            encoding="utf-8",
        )
        return path

    if branch == "new-vlm":
        monkeypatch.setattr(
            "vla_eval.evaluation.run_profile_vlm",
            lambda _dataset, vlm_output, _profile, _callbacks: write_summary(
                vlm_output / "attempt_summary.json"
            ),
        )
        resume_from = "METRICS"
    else:
        _persist_resume_artifacts(output)
        _mock_persisted_loaders(monkeypatch)
        write_summary(output / "attempt_eval/attempt_summary.json")
        resume_from = "REPORT"

    with pytest.raises(ValueError, match="attempt_summary.json.*episode indices|duplicated"):
        run_evaluation(
            tmp_path,
            output,
            load_profile(PROFILE_PATH),
            True,
            _callbacks(),
            resume_from=resume_from,
        )


def test_returned_report_cannot_be_another_required_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_metrics_and_report(monkeypatch)
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda _dataset, output: output / "metrics_core.json",
    )

    with pytest.raises(ValueError, match="report output pattern"):
        run_evaluation(
            tmp_path,
            tmp_path / "run",
            load_profile(PROFILE_PATH),
            False,
            _callbacks(),
        )
