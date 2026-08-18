import builtins
import json
import logging
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from Genie02_report.attempt_eval import result_writer
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
from vla_eval.exceptions import EvaluationCancelled as LightweightEvaluationCancelled
from vla_eval.exceptions import ModelLoadError
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


def test_fallback_result_exposes_unavailable_resource_metrics():
    from Genie02_report.attempt_eval.vlm_client import fallback_result

    result = fallback_result()

    assert result["input_token_count"] is None
    assert result["context_token_limit"] is None
    assert result["cuda_peak_memory_allocated_bytes"] is None
    assert result["cuda_peak_memory_reserved_bytes"] is None


def test_base_result_exposes_unavailable_resource_metrics(tmp_path: Path):
    from Genie02_report.attempt_eval.run_episode_attempt_eval import base_result

    result = base_result(_episode(tmp_path, 0))

    assert result["input_token_count"] is None
    assert result["context_token_limit"] is None
    assert result["cuda_peak_memory_allocated_bytes"] is None
    assert result["cuda_peak_memory_reserved_bytes"] is None


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


def test_service_rejects_output_directory_symlink_before_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        service,
        "_read_episode_metadata",
        lambda *_args: pytest.fail("metadata reader called"),
    )
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_link,
        dry_run=True,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_attempt_evaluation(config)

    assert list(outside.iterdir()) == []


def test_service_rejects_output_path_through_symlinked_parent(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(outside, target_is_directory=True)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=parent_link / "attempt_eval",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="symbolic link|resolve"):
        run_attempt_evaluation(config, episodes=[])

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("artifact", "directory"),
    [
        ("episode_results", True),
        ("sampled_frames", True),
        ("sampled_frames/episode_000", True),
        ("attempt_summary.json", False),
        ("attempt_summary.csv", False),
    ],
)
def test_service_rejects_existing_child_artifact_symlinks(
    tmp_path: Path, artifact: str, directory: bool
):
    output_dir = tmp_path / "out"
    outside = tmp_path / "outside"
    output_dir.mkdir()
    if directory:
        outside.mkdir()
    else:
        outside.write_text("sentinel", encoding="utf-8")
    artifact_path = output_dir / artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.symlink_to(outside, target_is_directory=directory)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        dry_run=True,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_attempt_evaluation(config, episodes=[])

    if directory:
        assert list(outside.iterdir()) == []
    else:
        assert outside.read_text(encoding="utf-8") == "sentinel"


def test_episode_writer_rejects_symlink_target(tmp_path: Path):
    output_dir = tmp_path / "out"
    episode_dir = output_dir / "episode_results"
    episode_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    (episode_dir / "episode_000.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        result_writer.save_episode_result(output_dir, {"episode_index": 0})

    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_atomic_episode_write_preserves_previous_file_on_serialization_failure(tmp_path: Path):
    output_dir = tmp_path / "out"
    episode_dir = output_dir / "episode_results"
    episode_dir.mkdir(parents=True)
    final_path = episode_dir / "episode_000.json"
    final_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(TypeError):
        result_writer.save_episode_result(
            output_dir,
            {"episode_index": 0, "not_json": object()},
        )

    assert final_path.read_text(encoding="utf-8") == "sentinel"
    assert list(episode_dir.glob(".*.tmp")) == []


def test_atomic_summary_write_preserves_previous_files_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    json_path = output_dir / "attempt_summary.json"
    csv_path = output_dir / "attempt_summary.csv"
    json_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")
    monkeypatch.setattr(
        result_writer.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        result_writer.write_summary(output_dir, [])

    assert json_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"
    assert list(output_dir.glob(".*.tmp")) == []


@pytest.mark.parametrize("preexisting", [True, False])
def test_summary_commit_rolls_back_both_files_when_second_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    json_path = output_dir / "attempt_summary.json"
    csv_path = output_dir / "attempt_summary.csv"
    old_json = b"old-json\x00\xff"
    old_csv = b"old-csv\r\n\x80"
    if preexisting:
        json_path.write_bytes(old_json)
        csv_path.write_bytes(old_csv)

    real_replace = result_writer.os.replace
    final_replace_count = 0

    def fail_second_final_replace(source: Path, destination: Path) -> None:
        nonlocal final_replace_count
        if Path(destination) in {json_path, csv_path}:
            final_replace_count += 1
            if final_replace_count == 2:
                raise OSError("second final replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(result_writer.os, "replace", fail_second_final_replace)

    with pytest.raises(OSError, match="second final replace failed"):
        result_writer.write_summary(output_dir, [_valid_vlm_result()])

    if preexisting:
        assert json_path.read_bytes() == old_json
        assert csv_path.read_bytes() == old_csv
    else:
        assert not json_path.exists()
        assert not csv_path.exists()
    assert list(output_dir.glob(".*.tmp")) == []


def test_summary_commit_reports_compound_error_when_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    json_path = output_dir / "attempt_summary.json"
    csv_path = output_dir / "attempt_summary.csv"
    json_path.write_bytes(b"old-json")
    csv_path.write_bytes(b"old-csv")
    commit_error = OSError("second final replace failed")
    real_replace = result_writer.os.replace
    final_replace_count = 0

    def fail_commit_and_rollback(source: Path, destination: Path) -> None:
        nonlocal final_replace_count
        if Path(destination) in {json_path, csv_path}:
            final_replace_count += 1
            if final_replace_count == 2:
                raise commit_error
            if final_replace_count == 3:
                raise OSError("rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(result_writer.os, "replace", fail_commit_and_rollback)

    with (
        caplog.at_level(logging.ERROR, logger=result_writer.__name__),
        pytest.raises(result_writer.SummaryPersistenceError, match="rollback") as caught,
    ):
        result_writer.write_summary(output_dir, [_valid_vlm_result()])

    assert caught.value.__cause__ is commit_error
    assert "rollback failed" in caplog.text
    assert list(output_dir.glob(".*.tmp")) == []


def test_attempt_eval_config_is_frozen(tmp_path: Path):
    config = AttemptEvalConfig(dataset_root=tmp_path, model_path=tmp_path / "model")

    with pytest.raises(FrozenInstanceError):
        config.limit = 1  # type: ignore[misc]


def test_prompt_registry_selects_exact_current_prompt():
    from Genie02_report.attempt_eval import prompt_registry
    from Genie02_report.attempt_eval.vlm_client import PROMPT, PROMPTS, prompt_for_version
    from vla_eval import profiles

    assert PROMPTS == {"genie02-attempt-v1": PROMPT}
    assert prompt_for_version("genie02-attempt-v1") is PROMPT
    assert service.SUPPORTED_PROMPT_VERSIONS is prompt_registry.SUPPORTED_PROMPT_VERSIONS
    assert profiles._PROMPT_VERSIONS is prompt_registry.SUPPORTED_PROMPT_VERSIONS
    with pytest.raises(TypeError):
        PROMPTS["made-up-v99"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="genie02-attempt-v1"):
        prompt_for_version("made-up-v99")


def test_local_vlm_client_close_releases_loaded_resources():
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    empty_cache_calls: list[None] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: empty_cache_calls.append(None),
    )
    client = LocalVLMClient.__new__(LocalVLMClient)
    client.model = object()
    client.processor = object()
    client.torch = SimpleNamespace(cuda=fake_cuda)

    client.close()
    client.close()

    assert client.model is None
    assert client.processor is None
    assert empty_cache_calls


def test_local_vlm_client_missing_path_raises_safe_model_load_error(tmp_path):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    missing = tmp_path / "private-model"
    with pytest.raises(ModelLoadError, match="configured model could not be loaded") as raised:
        LocalVLMClient(missing)

    assert str(missing) not in str(raised.value)
    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_local_vlm_client_rejects_checkpoint_family_mismatch(tmp_path: Path):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    model_dir = tmp_path / "private-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl"}),
        encoding="utf-8",
    )

    with pytest.raises(ModelLoadError, match="configured model could not be loaded") as raised:
        LocalVLMClient(model_dir, model_family="qwen3_vl")

    assert str(model_dir) not in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "qwen3_vl" in str(raised.value.__cause__)


@pytest.mark.parametrize(
    ("model_family", "config", "expected"),
    [
        (
            "qwen3_vl",
            {
                "model_type": "qwen3_vl",
                "text_config": {"max_position_embeddings": 262_144},
            },
            262_144,
        ),
        (
            "qwen2_5_vl",
            {"model_type": "qwen2_5_vl", "max_position_embeddings": 128_000},
            128_000,
        ),
    ],
)
def test_local_vlm_client_reads_checkpoint_context_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_family, config, expected
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.float32 = object()
    transformers_module = ModuleType("transformers")
    transformers_module.AutoModelForImageTextToText = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(to=lambda *_args: None)
    )
    transformers_module.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: object()
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    client = LocalVLMClient(model_dir, model_family=model_family)

    assert client.context_token_limit == expected


def test_local_vlm_client_rejects_missing_context_limit(tmp_path: Path):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "text_config": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelLoadError) as raised:
        LocalVLMClient(model_dir, model_family="qwen3_vl")

    assert isinstance(raised.value.__cause__, ValueError)


def test_local_vlm_client_uses_auto_image_text_model_for_qwen3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {"max_position_embeddings": 262_144},
            }
        ),
        encoding="utf-8",
    )
    model_calls: list[tuple[Path, dict[str, Any]]] = []
    processor_calls: list[tuple[Path, dict[str, Any]]] = []
    fake_model = SimpleNamespace(device="cuda:0")
    fake_processor = SimpleNamespace()
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: True)
    torch_module.float32 = object()
    transformers_module = ModuleType("transformers")
    transformers_module.AutoModelForImageTextToText = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: (
            model_calls.append((Path(path), kwargs)) or fake_model
        )
    )
    transformers_module.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: (
            processor_calls.append((Path(path), kwargs)) or fake_processor
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    client = LocalVLMClient(model_dir, model_family="qwen3_vl")

    assert client.model is fake_model
    assert client.processor is fake_processor
    assert model_calls == [
        (
            model_dir,
            {
                "dtype": "auto",
                "device_map": "auto",
                "local_files_only": True,
            },
        )
    ]
    assert processor_calls == [(model_dir, {"local_files_only": True})]


def test_local_vlm_client_uses_qwen3_preprocessing_and_deterministic_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    vision_calls: list[dict[str, Any]] = []
    processor_calls: list[dict[str, Any]] = []
    generate_calls: list[dict[str, Any]] = []
    decode_calls: list[dict[str, Any]] = []
    input_targets: list[Any] = []

    class FakeInputs(dict):
        def __init__(self):
            super().__init__()
            self.input_ids = [[10, 11]]

        def to(self, target):
            input_targets.append(target)
            return self

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["type"] == "image"
            return "templated prompt"

        def __call__(self, **kwargs):
            processor_calls.append(kwargs)
            return FakeInputs()

        def batch_decode(self, generated, **kwargs):
            decode_calls.append({"generated": generated, **kwargs})
            return [
                json.dumps(
                    {
                        "episode_success": True,
                        "pre_success_failed_attempt_count": 0,
                        "failed_attempts_before_success": [],
                        "final_success_time": 3.5,
                        "confidence": 0.9,
                        "reason": "final grasp visible",
                    }
                )
            ]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            generate_calls.append(kwargs)
            return [[10, 11, 12]]

    qwen_utils = ModuleType("qwen_vl_utils")

    def process_vision_info(messages, **kwargs):
        vision_calls.append({"messages": messages, **kwargs})
        return ["image-input"], []

    qwen_utils.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    client = LocalVLMClient.__new__(LocalVLMClient)
    client.model = FakeModel()
    client.processor = FakeProcessor()
    client.max_new_tokens = 256
    client.context_token_limit = 262_144
    client.prompt = "prompt"
    client.torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
        inference_mode=FakeInferenceMode,
    )
    frame = tmp_path / "frame.jpg"

    result, valid = client.analyze(
        [frame],
        [{"frame_type": "global", "frame": "frame.jpg", "episode_time": 1.0, "video_time": 2.0}],
        4.0,
    )

    assert valid is True
    assert result["input_token_count"] == 2
    assert result["context_token_limit"] == 262_144
    assert result["pre_success_failed_attempt_count"] == 0
    assert vision_calls[0]["image_patch_size"] == 16
    assert processor_calls[0]["do_resize"] is False
    assert input_targets == ["cuda:0"]
    assert generate_calls[0]["do_sample"] is False
    assert generate_calls[0]["max_new_tokens"] == 256
    assert decode_calls == [
        {
            "generated": [[12]],
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
    ]


def test_local_vlm_client_skips_generation_when_context_budget_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    class FakeInputs(dict):
        input_ids = SimpleNamespace(shape=(1, 90))

        def to(self, _target):
            return self

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return FakeInputs()

    generated = []
    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = lambda *_args, **_kwargs: (["image"], [])
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    client = LocalVLMClient.__new__(LocalVLMClient)
    client.model = SimpleNamespace(device="cuda:0", generate=lambda **kwargs: generated.append(kwargs))
    client.processor = Processor()
    client.max_new_tokens = 16
    client.context_token_limit = 100
    client.prompt = "prompt"
    client.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
    )

    result, valid = client.analyze(
        [tmp_path / "frame.jpg"],
        [{"frame_type": "global", "frame": "frame.jpg", "episode_time": 0, "video_time": 0}],
    )

    assert valid is False
    assert generated == []
    assert "context_length_exceeded" in result["auto_warning"]
    assert result["input_token_count"] == 90
    assert result["context_token_limit"] == 100


def test_local_vlm_client_records_cuda_peak_memory_on_inference_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    events = []

    class FakeInputs(dict):
        input_ids = SimpleNamespace(shape=(1, 2))

        def to(self, _target):
            return self

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return FakeInputs()

    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = lambda *_args, **_kwargs: (["image"], [])
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: events.append("reset"),
        max_memory_allocated=lambda: 123,
        max_memory_reserved=lambda: 456,
        empty_cache=lambda: events.append("empty"),
    )
    client = LocalVLMClient.__new__(LocalVLMClient)
    client.model = SimpleNamespace(
        device="cuda:0",
        generate=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private boom")),
    )
    client.processor = Processor()
    client.max_new_tokens = 16
    client.context_token_limit = 100
    client.prompt = "prompt"
    client.torch = SimpleNamespace(cuda=cuda, inference_mode=lambda: nullcontext())

    result, valid = client.analyze(
        [tmp_path / "frame.jpg"],
        [{"frame_type": "global", "frame": "frame.jpg", "episode_time": 0, "video_time": 0}],
    )

    assert valid is False
    assert result["cuda_peak_memory_allocated_bytes"] == 123
    assert result["cuda_peak_memory_reserved_bytes"] == 456
    assert events == ["reset", "empty"]
    assert "private boom" not in json.dumps(result)


def test_local_vlm_client_records_cuda_peak_memory_on_processor_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.vlm_client import LocalVLMClient

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            raise RuntimeError("private processor failure")

    qwen_utils = ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = lambda *_args, **_kwargs: (["image"], [])
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 123,
        max_memory_reserved=lambda: 456,
        empty_cache=lambda: None,
    )
    client = LocalVLMClient.__new__(LocalVLMClient)
    client.model = SimpleNamespace(device="cuda:0")
    client.processor = Processor()
    client.max_new_tokens = 16
    client.context_token_limit = 100
    client.prompt = "prompt"
    client.torch = SimpleNamespace(cuda=cuda)

    result, valid = client.analyze(
        [tmp_path / "frame.jpg"],
        [{"frame_type": "global", "frame": "frame.jpg", "episode_time": 0, "video_time": 0}],
    )

    assert valid is False
    assert "inference_error" in result["auto_warning"]
    assert result["cuda_peak_memory_allocated_bytes"] == 123
    assert result["cuda_peak_memory_reserved_bytes"] == 456
    assert "private processor failure" not in json.dumps(result)


def test_evaluation_cancelled_is_reexported_from_lightweight_module():
    assert EvaluationCancelled is LightweightEvaluationCancelled


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"prompt_version": "made-up-v99"}, "prompt_version"),
        ({"model_family": "internvl"}, "model_family"),
        ({"prompt_version": []}, "prompt_version"),
        ({"dense_region": "middle"}, "dense_region"),
        ({"dense_region": []}, "dense_region"),
        ({"review_mode": "sometimes"}, "review_mode"),
        ({"review_mode": []}, "review_mode"),
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
                "model_family": "qwen2_5_vl",
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


def test_run_attempt_evaluation_reads_all_configured_camera_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reader_calls = []

    def read_metadata(dataset_root: Path, image_keys):
        reader_calls.append((dataset_root, image_keys))
        return []

    monkeypatch.setattr(service, "_read_episode_metadata", read_metadata)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        image_keys=("observation.images.front", "observation.images.right_wrist"),
        output_dir=tmp_path / "out",
    )

    run_attempt_evaluation(config)

    assert reader_calls == [(tmp_path, config.image_keys)]


def test_run_attempt_evaluation_jointly_analyzes_selected_cameras_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    episode = _episode(tmp_path, 4)
    episode.camera_videos = (
        SimpleNamespace(
            camera_key="observation.images.front",
            video_file=tmp_path / "front.mp4",
            video_file_rel="videos/observation.images.front/front.mp4",
            from_timestamp=10.0,
            to_timestamp=14.0,
        ),
        SimpleNamespace(
            camera_key="observation.images.right_wrist",
            video_file=tmp_path / "right.mp4",
            video_file_rel="videos/observation.images.right_wrist/right.mp4",
            from_timestamp=10.0,
            to_timestamp=15.0,
        ),
    )
    sampling_calls: list[tuple[str, Path, dict[str, Any]]] = []

    def sample_frames(video_file: Path, output_dir: Path, *_args, **kwargs):
        sampling_calls.append((video_file.stem, output_dir, kwargs))
        frame = output_dir / "global/frame_000.jpg"
        return [frame], [
            {
                "frame": "global/frame_000.jpg",
                "frame_type": "global",
                "episode_time": 0.0,
                "video_time": 10.0,
            }
        ]

    analyses: list[tuple[list[Path], list[dict[str, Any]], float]] = []

    class Client:
        def analyze(self, frame_paths, frame_timestamps, duration):
            analyses.append((frame_paths, frame_timestamps, duration))
            return _valid_vlm_result(), True

        def close(self):
            pass

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        image_keys=("observation.images.front", "observation.images.right_wrist"),
        output_dir=tmp_path / "out",
    )

    results = run_attempt_evaluation(
        config,
        episodes=[episode],
        client_factory=lambda *_args, **_kwargs: Client(),
    )

    assert [call[0] for call in sampling_calls] == ["front", "right"]
    assert all(call[2]["max_global_frames"] == 8 for call in sampling_calls)
    assert len(analyses) == 1
    assert analyses[0][2] == 5.0
    assert [item["camera_key"] for item in analyses[0][1]] == [
        "observation.images.front",
        "observation.images.right_wrist",
    ]
    assert results[0]["camera_keys"] == [
        "observation.images.front",
        "observation.images.right_wrist",
    ]
    assert results[0]["sampled_frame_count_by_camera"] == {
        "observation.images.front": 1,
        "observation.images.right_wrist": 1,
    }
    assert results[0]["sampled_frame_count"] == 2


def test_prompt_frame_list_includes_camera_identity():
    from Genie02_report.attempt_eval.vlm_client import _prompt_with_frame_times

    text = _prompt_with_frame_times(
        [
            {
                "frame": "front/global/frame_000.jpg",
                "frame_type": "global",
                "camera_key": "observation.images.front",
                "episode_time": 0.0,
                "video_time": 10.0,
            }
        ],
        "PROMPT",
    )

    assert "camera=observation.images.front" in text


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
            "--model_family",
            "qwen3_vl",
        ],
    )

    config = service._config_from_args(service.parse_args())

    assert config.max_global_frames == 3
    assert config.global_sample_interval == 1.25
    assert config.prompt_version == "genie02-attempt-v1"
    assert config.model_family == "qwen3_vl"
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


def test_metadata_failure_episode_does_not_construct_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    results = run_attempt_evaluation(
        config,
        episodes=[_episode(tmp_path, 0, success=False)],
        client_factory=lambda *_args, **_kwargs: pytest.fail("VLM client constructed"),
    )

    assert results[0]["episode_success"] is False


def test_client_is_created_after_first_successful_sampling_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []

    def sample_frames(video_file: Path, *_args, **_kwargs):
        events.append(f"sample-{video_file.stem}")
        if video_file.name == "episode-0.mp4":
            raise RuntimeError("broken video")
        return [tmp_path / "frame.jpg"], []

    class FakeClient:
        def analyze(self, *_args):
            events.append("analyze")
            return _valid_vlm_result(), True

        def close(self):
            events.append("close")

    def create_client(*_args, **_kwargs):
        events.append("factory")
        return FakeClient()

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    results = run_attempt_evaluation(
        config,
        episodes=[_episode(tmp_path, 0), _episode(tmp_path, 1)],
        client_factory=create_client,
    )

    assert [result["vlm_valid"] for result in results] == [False, True]
    assert events == ["sample-episode-0", "sample-episode-1", "factory", "analyze", "close"]


def test_client_factory_failure_is_fatal_after_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []

    def sample_frames(*_args, **_kwargs):
        events.append("sample")
        return [tmp_path / "frame.jpg"], []

    def create_client(*_args, **_kwargs):
        events.append("factory")
        raise RuntimeError("model setup failed")

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with pytest.raises(RuntimeError, match="model setup failed"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=create_client,
        )

    assert events == ["sample", "factory"]
    assert not (config.output_dir / "episode_results/episode_000.json").exists()
    assert not (config.output_dir / "attempt_summary.json").exists()


def test_cancellation_after_sampling_prevents_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sampled = False

    def sample_frames(*_args, **_kwargs):
        nonlocal sampled
        sampled = True
        return [tmp_path / "frame.jpg"], []

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with pytest.raises(EvaluationCancelled):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: pytest.fail("VLM client constructed"),
            should_cancel=lambda: sampled,
        )


def test_cancellation_immediately_before_analyze_closes_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client_created = False
    client_closed = False

    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class FakeClient:
        def analyze(self, *_args):
            pytest.fail("VLM analyze called")

        def close(self):
            nonlocal client_closed
            client_closed = True

    def create_client(*_args, **_kwargs):
        nonlocal client_created
        client_created = True
        return FakeClient()

    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with pytest.raises(EvaluationCancelled):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=create_client,
            should_cancel=lambda: client_created,
        )

    assert client_closed is True


def test_episode_error_uses_fallback_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def sample_frames(video_file: Path, *_args, **_kwargs):
        if video_file.name == "episode-0.mp4":
            raise RuntimeError("broken video")
        return [tmp_path / "frame.jpg"], []

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    progress: list[tuple[int, int, str]] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vla_eval.evaluation" or name.split(".", 1)[0] in {
            "yaml",
            "fastapi",
            "torch",
            "transformers",
        }:
            raise AssertionError(f"heavy dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(service, "_sample_episode_frames", sample_frames)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
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
    assert results[0]["raw_response"] == ""
    assert results[0]["parse_error"] == "sampling_error:RuntimeError"
    assert results[0]["reason"] == "Episode frame sampling failed"
    assert "broken video" not in str(results[0])
    assert results[1]["vlm_valid"] is True
    assert [done for done, _total, _stage in progress] == [0, 1, 2, 2]


def test_inference_error_is_sanitized_and_full_error_is_only_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    secret = "token=top-secret /private/customer/video.mp4"
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class BrokenClient:
        def analyze(self, *_args):
            raise RuntimeError(secret)

    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with caplog.at_level(logging.ERROR, logger=service.__name__):
        results = run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: BrokenClient(),
        )

    result = results[0]
    assert result["raw_response"] == ""
    assert result["parse_error"] == "inference_error:RuntimeError"
    assert result["reason"] == "Episode VLM inference failed"
    assert secret not in f"{result['raw_response']} {result['parse_error']} {result['reason']}"
    assert "episode 0" in caplog.text
    assert secret in caplog.text


def test_frame_sampler_import_is_optional_dependency_free():
    script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'av', 'cv2', 'PIL'}:
        raise ImportError(f'blocked {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from Genie02_report.attempt_eval import frame_sampler
assert issubclass(frame_sampler.SamplingDependencyError, RuntimeError)
print('dependency-free')
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "dependency-free"


def test_sampling_dependency_error_is_fatal_without_client_or_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from Genie02_report.attempt_eval.frame_sampler import SamplingDependencyError

    def missing_dependencies(*_args, **_kwargs):
        raise SamplingDependencyError("video decoder and image backend unavailable")

    monkeypatch.setattr(service, "_sample_episode_frames", missing_dependencies)
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
        dry_run=True,
    )

    with pytest.raises(SamplingDependencyError, match="backend unavailable"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: pytest.fail("VLM client constructed"),
        )

    assert not (config.output_dir / "attempt_summary.json").exists()


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
    first_episode_path = output_dir / "episode_results/episode_000.json"
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
            should_cancel=first_episode_path.exists,
        )

    assert first_episode_path.is_file()
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
    episode_path = output_dir / "episode_results/episode_000.json"
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
            should_cancel=episode_path.exists,
        )

    assert episode_path.is_file()
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


def test_client_close_failure_preserves_active_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    cancellation = EvaluationCancelled("cancelled in client")
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class CancellingClient:
        def analyze(self, *_args):
            raise cancellation

        def close(self):
            raise RuntimeError("close failed")

    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "out",
    )

    with (
        caplog.at_level(logging.ERROR, logger=service.__name__),
        pytest.raises(EvaluationCancelled) as caught,
    ):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: CancellingClient(),
        )

    assert caught.value is cancellation
    assert "client cleanup failed" in caplog.text
    assert "close failed" in caplog.text
    assert not (config.output_dir / "attempt_summary.json").exists()


def test_close_failure_after_inference_fallback_is_fatal_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class BrokenClient:
        def analyze(self, *_args):
            raise RuntimeError("analyze failed")

        def close(self):
            raise LookupError("close failed")

    output_dir = tmp_path / "out"
    progress: list[tuple[int, int, str]] = []
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
    )

    with pytest.raises(LookupError, match="close failed"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: BrokenClient(),
            progress=lambda done, total, stage: progress.append((done, total, stage)),
        )

    assert (output_dir / "episode_results/episode_000.json").is_file()
    assert not (output_dir / "attempt_summary.json").exists()
    assert progress == [(0, 1, "initial"), (1, 1, "episode_complete")]


def test_close_only_failure_is_fatal_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class CloseFailingClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

        def close(self):
            raise RuntimeError("close failed")

    output_dir = tmp_path / "out"
    progress: list[tuple[int, int, str]] = []
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
    )

    with pytest.raises(RuntimeError, match="close failed"):
        run_attempt_evaluation(
            config,
            episodes=[_episode(tmp_path, 0)],
            client_factory=lambda *_args, **_kwargs: CloseFailingClient(),
            progress=lambda done, total, stage: progress.append((done, total, stage)),
        )

    assert (output_dir / "episode_results/episode_000.json").is_file()
    assert not (output_dir / "attempt_summary.json").exists()
    assert progress == [(0, 1, "initial"), (1, 1, "episode_complete")]


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


def test_complete_progress_failure_preserves_previous_summary(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    json_path = output_dir / "attempt_summary.json"
    csv_path = output_dir / "attempt_summary.csv"
    json_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")
    progress: list[tuple[int, int, str]] = []
    config = AttemptEvalConfig(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        dry_run=True,
    )

    def fail_on_complete(done: int, total: int, stage: str) -> None:
        progress.append((done, total, stage))
        if stage == "complete":
            raise LookupError("final progress failed")

    with pytest.raises(LookupError, match="final progress failed"):
        run_attempt_evaluation(config, episodes=[], progress=fail_on_complete)

    assert progress == [(0, 0, "initial"), (0, 0, "complete")]
    assert json_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"


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
            "model_family": "qwen2_5_vl",
            "max_new_tokens": 256,
            "prompt_version": "genie02-attempt-v1",
        }
    ]
    assert progress == [30.0, 60.0, 90.0, 90.0]


def test_api_backend_runs_service_with_injected_api_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The api profile routes run_profile_vlm through the injected API factory."""
    episodes = [_episode(tmp_path, 1), _episode(tmp_path, 2)]
    monkeypatch.setattr(service, "_read_episode_metadata", lambda *_args: episodes)
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    class FakeClient:
        def analyze(self, *_args):
            return _valid_vlm_result(), True

    factory_calls: list[dict[str, Any]] = []

    def fake_builder(api):
        def factory(_model_path, *, model_family, max_new_tokens, prompt_version):
            factory_calls.append(
                {
                    "base_url": api.base_url,
                    "model": api.model,
                    "api_key_env": api.api_key_env,
                    "timeout": api.timeout,
                    "max_retries": api.max_retries,
                    "model_family": model_family,
                    "max_new_tokens": max_new_tokens,
                    "prompt_version": prompt_version,
                }
            )
            return FakeClient()

        return factory

    monkeypatch.setattr("vla_eval.evaluation._build_api_client_factory", fake_builder)
    progress: list[float] = []
    output_dir = tmp_path / "attempt_eval"

    summary_path = run_profile_vlm(
        tmp_path,
        output_dir,
        load_profile(Path("config/profiles/genie02-api.yaml")),
        EvaluationCallbacks(
            on_stage=lambda _stage: None,
            on_progress=progress.append,
            should_cancel=lambda: False,
        ),
    )

    summary = load_attempt_summary(summary_path)
    assert [result["episode_index"] for result in summary] == [1, 2]
    # The client is constructed once (lazy, on the first successful episode) and
    # reused; the factory must receive the api block plus the run-level kwargs,
    # and must ignore the vendored model_path placeholder.
    assert factory_calls == [
        {
            "base_url": "http://vlm-api.example.internal/v1",
            "model": "qwen2.5-vl-7b-instruct",
            "api_key_env": "VLA_EVAL_VLM_API_KEY",
            "timeout": 60,
            "max_retries": 3,
            "model_family": "qwen2_5_vl",
            "max_new_tokens": 256,
            "prompt_version": "genie02-attempt-v1",
        }
    ]
    assert progress == [30.0, 60.0, 90.0, 90.0]


def test_api_backend_sanitizes_inference_error_without_leaking_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An api-client inference failure becomes a sanitized per-episode fallback.

    Mirrors the local-backend sanitization contract: the vendored runner wraps
    .analyze() in a broad ``except Exception`` and converts it to a fallback, so
    a failure carrying the API key must never reach the persisted result fields.
    """
    monkeypatch.setattr(
        service, "_read_episode_metadata", lambda *_args: [_episode(tmp_path, 0)]
    )
    monkeypatch.setattr(
        service,
        "_sample_episode_frames",
        lambda *_args, **_kwargs: ([tmp_path / "frame.jpg"], []),
    )

    secret = "Authorization: Bearer top-secret-key"

    class FailingClient:
        def analyze(self, *_args):
            raise RuntimeError(secret)

    monkeypatch.setattr(
        "vla_eval.evaluation._build_api_client_factory",
        lambda _api: (lambda *_args, **_kwargs: FailingClient()),
    )

    summary_path = run_profile_vlm(
        tmp_path,
        tmp_path / "attempt_eval",
        load_profile(Path("config/profiles/genie02-api.yaml")),
        EvaluationCallbacks(
            on_stage=lambda _stage: None,
            on_progress=lambda _progress: None,
            should_cancel=lambda: False,
        ),
    )

    summary = load_attempt_summary(summary_path)
    assert summary[0]["vlm_valid"] is False
    assert summary[0]["parse_error"] == "inference_error:RuntimeError"
    assert summary[0]["reason"] == "Episode VLM inference failed"
    # The secret must not appear in ANY persisted field -- sweep all values so a
    # regression in any field (incl. future-added ones) is caught.
    rendered = " ".join(str(value) for value in summary[0].values())
    assert secret not in rendered


def test_build_api_client_factory_wires_profile_fields_into_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The real (un-monkeypatched) factory forwards the api block to ApiVLMClient.

    The two service tests above inject a fake factory, so without this test the
    7-kwarg pass-through from the profile's api block into ApiVLMClient.__init__
    is never exercised -- a transposition there would only surface in production.
    """
    from vla_eval.evaluation import _build_api_client_factory
    from vla_eval.profiles import load_profile

    monkeypatch.setenv("VLA_EVAL_VLM_API_KEY", "test-key")
    api = load_profile(Path("config/profiles/genie02-api.yaml")).vlm.api

    factory = _build_api_client_factory(api)
    # The vendored runner passes model identity plus generation settings; the
    # placeholder model_path and local model_family are accepted and ignored.
    client = factory(
        tmp_path / "ignored-model-path-placeholder",
        model_family="qwen2_5_vl",
        max_new_tokens=256,
        prompt_version="genie02-attempt-v1",
    )
    try:
        assert client.base_url == "http://vlm-api.example.internal/v1"
        assert client.model == "qwen2.5-vl-7b-instruct"
        assert client.max_new_tokens == 256
        assert client.max_retries == 3
    finally:
        client.close()
