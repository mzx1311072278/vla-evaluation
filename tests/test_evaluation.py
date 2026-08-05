import builtins
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
    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", lambda *_args: [])
    monkeypatch.setattr("vla_eval.evaluation.generate_metrics_core", lambda *_args: {"gsr": 1.0})
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda _dataset, output: output / "report.md",
    )


def _persist_resume_artifacts(output: Path) -> None:
    output.mkdir(parents=True)
    (output / "episode_metrics.csv").write_text("header\n", encoding="utf-8")
    (output / "metrics_core.json").write_text("{}", encoding="utf-8")


def test_run_evaluation_calls_metrics_then_report(tmp_path, monkeypatch):
    stages = []
    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", lambda dataset, output: [])
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_metrics_core", lambda dataset, output: {"gsr": 1.0}
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda dataset, output: output / "report.md",
    )
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
    assert profile.review.confidence_threshold == 0.7
    assert profile.review.min_episode_duration == 3.0
    assert profile.review.min_sampled_frames == 3
    assert profile.outputs.required == (
        "episode_metrics.csv",
        "metrics_core.json",
        "smoothness_curve.svg",
        "report_*.md",
    )
    assert profile.outputs.optional == (
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


def test_run_evaluation_calls_optional_vlm_between_metrics_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stages: list[str] = []
    _mock_metrics_and_report(monkeypatch)
    monkeypatch.setattr(
        "vla_eval.evaluation.run_profile_vlm",
        lambda _dataset, output, _profile, _callbacks: output / "attempt_summary.json",
    )

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
    monkeypatch.setattr("vla_eval.evaluation.load_session", lambda dataset: {"path": dataset})
    monkeypatch.setattr(
        "vla_eval.evaluation.load_metrics_core",
        lambda actual_output, session: {"gsr": 0.5, "output": actual_output, "session": session},
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_episode_metrics",
        lambda *_args: pytest.fail("metrics regenerated"),
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.run_profile_vlm",
        lambda _dataset, vlm_output, _profile, _callbacks: vlm_output / "attempt_summary.json",
    )
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda _dataset, actual_output: actual_output / "report.md",
    )

    result = run_evaluation(
        tmp_path,
        output,
        load_profile(PROFILE_PATH),
        True,
        _callbacks(stages=stages),
        resume_from="VLM",
    )

    assert result.metrics["gsr"] == 0.5
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
    monkeypatch.setattr("vla_eval.evaluation.load_session", lambda _dataset: {})
    monkeypatch.setattr("vla_eval.evaluation.load_metrics_core", lambda *_args: {"gsr": 1.0})

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
        monkeypatch.setattr("vla_eval.evaluation.load_session", lambda _dataset: {})
        monkeypatch.setattr("vla_eval.evaluation.load_metrics_core", lambda *_args: {"gsr": 1.0})
    monkeypatch.setattr(
        "vla_eval.evaluation.run_profile_vlm",
        lambda _dataset, vlm_output, _profile, callbacks: (
            callbacks.on_progress(45.0),
            callbacks.on_progress(75.0),
            vlm_output / "attempt_summary.json",
        )[-1],
    )

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
