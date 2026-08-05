import builtins
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Genie02_report.attempt_eval import run_episode_attempt_eval as service
from Genie02_report.attempt_eval.run_episode_attempt_eval import (
    AttemptEvalConfig,
    run_attempt_evaluation,
)
from vla_eval.evaluation import (
    EvaluationCallbacks,
    EvaluationCancelled,
    load_attempt_summary,
    run_profile_vlm,
)
from vla_eval.profiles import load_profile


def _episode(tmp_path: Path, episode_index: int, *, success: bool | None = True):
    return SimpleNamespace(
        episode_index=episode_index,
        length=12,
        episode_success=success,
        video_file=tmp_path / f"episode-{episode_index}.mp4",
        video_file_rel=f"videos/episode-{episode_index}.mp4",
        from_timestamp=10.0,
        to_timestamp=14.0,
    )


def _valid_vlm_result() -> dict[str, Any]:
    return {
        "episode_success": True,
        "pre_success_failed_attempt_count": 0,
        "failed_attempts_before_success": [],
        "final_success_time": 3.5,
        "attempt_count": 1,
        "success_count": 1,
        "failed_count": 0,
        "attempts": [],
        "confidence": 0.9,
        "vlm_valid": True,
        "reason": "final grasp visible",
        "parse_error": "",
        "raw_response": "{}",
        "auto_warning": [],
    }


def test_run_attempt_evaluation_accepts_injected_dependencies(tmp_path: Path):
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
        dry_run=True,
    )
    results = run_attempt_evaluation(
        config,
        episodes=[],
        progress=lambda _done, _total, _stage: None,
    )
    assert results == []
    assert (config.output_dir / "attempt_summary.json").exists()


def test_empty_non_dry_run_writes_summary_without_constructing_client(tmp_path: Path):
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "missing-model",
        output_dir=tmp_path / "out",
    )

    progress: list[tuple[int, int, str]] = []
    results = run_attempt_evaluation(
        config,
        episodes=[],
        client_factory=lambda *_args, **_kwargs: pytest.fail("VLM client constructed"),
        progress=lambda done, total, stage: progress.append((done, total, stage)),
    )

    assert results == []
    assert load_attempt_summary(config.output_dir / "attempt_summary.json") == []
    assert progress == [(0, 0, "initial"), (0, 0, "complete")]


def test_attempt_eval_config_is_frozen(tmp_path: Path):
    config = AttemptEvalConfig(dataset_root=tmp_path, model_path=tmp_path / "model")

    with pytest.raises(FrozenInstanceError):
        config.limit = 1  # type: ignore[misc]


def test_prompt_registry_selects_exact_current_prompt():
    from Genie02_report.attempt_eval.vlm_client import PROMPT, PROMPTS, prompt_for_version

    assert PROMPTS == {"genie02-attempt-v1": PROMPT}
    assert prompt_for_version("genie02-attempt-v1") is PROMPT
    with pytest.raises(ValueError, match="genie02-attempt-v1"):
        prompt_for_version("made-up-v99")


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"prompt_version": "made-up-v99"}, "prompt_version"),
        ({"dense_region": "middle"}, "dense_region"),
        ({"review_mode": "sometimes"}, "review_mode"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
        ({"max_image_size": 0}, "max_image_size"),
        ({"global_sample_interval": 0}, "global_sample_interval"),
        ({"limit": -1}, "limit"),
        ({"dry_run": 1}, "dry_run"),
        ({"output_dir": "out"}, "output_dir"),
    ],
)
def test_attempt_eval_config_rejects_invalid_service_values(
    tmp_path: Path, overrides: dict[str, object], error: str
):
    with pytest.raises((TypeError, ValueError), match=error):
        AttemptEvalConfig(
            dataset_root=tmp_path,
            model_path=tmp_path / "model",
            **overrides,
        )


def test_run_attempt_evaluation_uses_injected_client_without_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sampling_calls: list[dict[str, Any]] = []
    factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    progress: list[tuple[int, int, str]] = []
    frame_path = tmp_path / "frame.jpg"

    def sample_frames(*_args, **kwargs):
        sampling_calls.append(kwargs)
        return [frame_path], [
            {
                "frame": "global/frame.jpg",
                "frame_type": "global",
                "episode_time": 1.0,
                "video_time": 11.0,
            }
        ]

    class FakeClient:
        def analyze(self, frame_paths, frame_timestamps, duration):
            assert frame_paths == [frame_path]
            assert frame_timestamps[0]["episode_time"] == 1.0
            assert duration == 4.0
            return _valid_vlm_result(), True

    def client_factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return FakeClient()

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "cv2",
            "PIL",
            "av",
            "qwen_vl_utils",
        }:
            raise AssertionError(f"optional dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    results = run_attempt_evaluation(
        config,
        episodes=[_episode(tmp_path, 7)],
        client_factory=client_factory,
        progress=lambda done, total, stage: progress.append((done, total, stage)),
    )

    assert [result["episode_index"] for result in results] == [7]
    assert load_attempt_summary(config.output_dir / "attempt_summary.json") == results
    assert factory_calls == [
        (
            (config.model_path,),
            {
                "max_new_tokens": 256,
                "prompt_version": "genie02-attempt-v1",
            },
        )
    ]
    assert sampling_calls == [
        {
            "max_image_size": 336,
            "max_global_frames": 8,
            "global_sample_interval": 2.0,
            "max_dense_frames": 8,
            "dense_sample_interval": 0.5,
            "dense_region": "full",
        }
    ]
    assert progress == [
        (0, 1, "initial"),
        (1, 1, "episode_complete"),
        (1, 1, "complete"),
    ]


def test_run_attempt_evaluation_reads_metadata_only_when_episodes_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reader_calls: list[tuple[Path, str]] = []

    def read_metadata(dataset_root: Path, image_key: str):
        reader_calls.append((dataset_root, image_key))
        return [_episode(tmp_path, 2, success=False), _episode(tmp_path, 3, success=False)]

    monkeypatch.setattr(service, "_read_episode_metadata", read_metadata)
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
        limit=1,
        dry_run=True,
    )

    results = run_attempt_evaluation(config)

    assert reader_calls == [(tmp_path, "observation.images.right_wrist")]
    assert [result["episode_index"] for result in results] == [2]


def test_cli_maps_deprecated_sampling_aliases_to_canonical_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attempt-eval",
            "--dataset_root",
            str(tmp_path),
            "--model_path",
            str(tmp_path / "model"),
            "--max_frames",
            "3",
            "--sample_interval",
            "1.25",
            "--prompt_version",
            "genie02-attempt-v1",
        ],
    )

    config = service._config_from_args(service.parse_args())

    assert config.max_global_frames == 3
    assert config.global_sample_interval == 1.25
    assert config.prompt_version == "genie02-attempt-v1"
    assert not hasattr(config, "max_frames")
    assert not hasattr(config, "sample_interval")


def test_dry_run_does_not_construct_vlm_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
        dry_run=True,
    )

    results = run_attempt_evaluation(
        config,
        episodes=[_episode(tmp_path, 0)],
        client_factory=lambda *_args, **_kwargs: pytest.fail("VLM client constructed"),
    )

    assert results[0]["vlm_valid"] is False
    assert "dry_run" in results[0]["auto_warning"]


def test_episode_error_uses_fallback_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def sample_frames(video_file: Path, *_args, **_kwargs):
        if video_file.name == "episode-0.mp4":
            raise RuntimeError("broken video")
        return [tmp_path / "frame.jpg"], []

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    progress: list[tuple[int, int, str]] = []
    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    results = run_attempt_evaluation(
        config,
        episodes=[_episode(tmp_path, 0), _episode(tmp_path, 1)],
        client_factory=lambda *_args, **_kwargs: FakeClient(),
        progress=lambda done, total, stage: progress.append((done, total, stage)),
    )

    assert [result["episode_index"] for result in results] == [0, 1]
    assert "episode_error" in results[0]["auto_warning"]
    assert results[0]["parse_error"] == ""
    assert results[1]["vlm_valid"] is True
    assert [done for done, _total, _stage in progress] == [0, 1, 2, 2]


def test_cancellation_preserves_episode_files_without_overwriting_final_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    summary_path = output_dir / "attempt_summary.json"
    summary_path.write_text("sentinel", encoding="utf-8")
    cancellation_checks = iter([False, False, True])
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
    )

    with pytest.raises(EvaluationCancelled, match="cancelled"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0), _episode(tmp_path, 1)],
            client_factory=lambda *_args, **_kwargs: FakeClient(),
            should_cancel=lambda: next(cancellation_checks),
        )

    assert (output_dir / "episode_results/episode_000.json").is_file()
    assert not (output_dir / "episode_results/episode_001.json").exists()
    assert summary_path.read_text(encoding="utf-8") == "sentinel"


def test_cancellation_after_final_episode_does_not_overwrite_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    summary_path = output_dir / "attempt_summary.json"
    summary_path.write_text("sentinel", encoding="utf-8")
    cancellation_checks = iter([False, False, True])
    progress: list[tuple[int, int, str]] = []
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
    )

    with pytest.raises(EvaluationCancelled, match="cancelled"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: FakeClient(),
            progress=lambda done, total, stage: progress.append((done, total, stage)),
            should_cancel=lambda: next(cancellation_checks),
        )

    assert (output_dir / "episode_results/episode_000.json").is_file()
    assert summary_path.read_text(encoding="utf-8") == "sentinel"
    assert progress == [(0, 1, "initial"), (1, 1, "episode_complete")]


@pytest.mark.parametrize("existing_summary", [None, "sentinel"])
def test_empty_cancelled_run_does_not_create_or_overwrite_summary(
    tmp_path: Path, existing_summary: str | None
):
    output_dir = tmp_path / "out"
    summary_path = output_dir / "attempt_summary.json"
    if existing_summary is not None:
        output_dir.mkdir()
        summary_path.write_text(existing_summary, encoding="utf-8")
    progress: list[tuple[int, int, str]] = []
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
    )

    with pytest.raises(EvaluationCancelled, match="cancelled"):
        run_attempt_evaluation(
            config,
            episodes=[],
            progress=lambda done, total, stage: progress.append((done, total, stage)),
            should_cancel=lambda: True,
        )

    if existing_summary is None:
        assert not summary_path.exists()
    else:
        assert summary_path.read_text(encoding="utf-8") == existing_summary
    assert progress == [(0, 0, "initial")]


def test_client_cancellation_is_not_converted_to_episode_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class CancellingClient:
        def analyze(self, *_args):
            raise EvaluationCancelled("cancelled in client")

    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with pytest.raises(EvaluationCancelled, match="cancelled in client"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: CancellingClient(),
        )

    assert not (config.output_dir / "attempt_summary.json").exists()
    assert not (config.output_dir / "episode_results/episode_000.json").exists()


def test_progress_callback_exceptions_propagate_without_success_summary(tmp_path: Path):
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
        dry_run=True,
    )

    def broken_progress(_done: int, _total: int, _stage: str) -> None:
        raise LookupError("progress failed")

    with pytest.raises(LookupError, match="progress failed"):
        run_attempt_evaluation(config, episodes=[], progress=broken_progress)

    assert not (config.output_dir / "attempt_summary.json").exists()


def test_task6_profile_mapping_runs_real_service_and_writes_compatible_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    episodes = [_episode(tmp_path, 4), _episode(tmp_path, 9)]
    monkeypatch.setattr(service, "_read_episode_metadata", lambda *_args: episodes)
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    client_calls: list[dict[str, Any]] = []

    def create_client(*_args, **kwargs):
        client_calls.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(service, "_create_local_vlm_client", create_client)
    progress: list[float] = []
    output_dir = tmp_path / "attempt_eval"

    summary_path = run_profile_vlm(
        tmp_path,
        output_dir,
        load_profile(Path("config/profiles/genie02-full.yaml")),
        EvaluationCallbacks(
            on_stage=lambda _stage: None,
            on_progress=progress.append,
            should_cancel=lambda: False,
        ),
    )

    summary = load_attempt_summary(summary_path)
    assert [result["episode_index"] for result in summary] == [4, 9]
    assert client_calls == [
        {
            "max_new_tokens": 256,
            "prompt_version": "genie02-attempt-v1",
        }
    ]
    assert progress == [30.0, 60.0, 90.0, 90.0]
