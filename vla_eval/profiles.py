"""Strict, immutable evaluation profile loading."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError

_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DENSE_REGIONS = frozenset({"full", "last_half", "last_third"})
_REVIEW_MODES = frozenset({"manual_review", "auto_review"})
_REPORT_PATTERN = "report_*.md"
_ADAPTERS = frozenset({"genie02"})
_PLUGINS = frozenset({"genie02-attempt-eval"})
_PROMPT_VERSIONS = frozenset({"genie02-attempt-v1"})
_REQUIRED_OUTPUTS = frozenset({"episode_metrics.csv", "metrics_core.json", _REPORT_PATTERN})


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SamplingProfile:
    max_global_frames: int
    global_sample_interval: float
    max_dense_frames: int
    dense_sample_interval: float
    dense_region: str


@dataclass(frozen=True)
class VLMProfile:
    model_path: str
    prompt_version: str
    sampling: SamplingProfile
    max_image_size: int
    max_new_tokens: int


@dataclass(frozen=True)
class ReviewProfile:
    mode: str
    confidence_threshold: float
    min_episode_duration: float
    min_sampled_frames: int


@dataclass(frozen=True)
class OutputProfile:
    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    adapter: str
    plugin: str
    image_key: str
    vlm: VLMProfile
    review: ReviewProfile
    outputs: OutputProfile


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _fields(raw: Mapping[str, Any], expected: set[str], field: str) -> None:
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(sorted(map(str, unknown)))}")
    if missing:
        raise ValueError(f"{field} is missing required fields: {', '.join(sorted(missing))}")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    return value


def _identifier(value: Any, field: str) -> str:
    result = _string(value, field)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase identifier")
    return result


def _enum_identifier(value: Any, field: str, supported: frozenset[str]) -> str:
    result = _identifier(value, field)
    if result not in supported:
        raise ValueError(f"{field} must be one of {sorted(supported)}")
    return result


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return result


def _output_path(value: Any, field: str) -> str:
    result = _string(value, field)
    if "\\" in result or any(ord(character) < 32 for character in result):
        raise ValueError(f"{field} must be a safe POSIX relative path")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or str(path) != result or result in {".", ""}:
        raise ValueError(f"{field} must be a normalized relative path without '..'")
    if any(character in result for character in "?[]{}"):
        raise ValueError(f"{field} contains unsupported pattern characters")
    if "*" in result and result != _REPORT_PATTERN:
        raise ValueError(f"{field} contains an unsupported wildcard")
    return result


def _output_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    return tuple(_output_path(item, f"{field}[{index}]") for index, item in enumerate(value))


def load_profile(path: str | Path) -> Profile:
    """Load one fully specified profile, rejecting ambiguous or unsafe input."""
    profile_path = Path(path)
    try:
        loaded: Any = yaml.load(profile_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load evaluation profile {profile_path}: {exc}") from exc

    raw = _mapping(loaded, "profile")
    _fields(
        raw,
        {"name", "version", "adapter", "plugin", "image_key", "vlm", "review", "outputs"},
        "profile",
    )

    version = _string(raw["version"], "version")
    if not _SEMVER.fullmatch(version):
        raise ValueError("version must be a valid semantic version")

    vlm = _mapping(raw["vlm"], "vlm")
    _fields(
        vlm, {"model_path", "prompt_version", "sampling", "max_image_size", "max_new_tokens"}, "vlm"
    )
    sampling = _mapping(vlm["sampling"], "vlm.sampling")
    _fields(
        sampling,
        {
            "max_global_frames",
            "global_sample_interval",
            "max_dense_frames",
            "dense_sample_interval",
            "dense_region",
        },
        "vlm.sampling",
    )
    dense_region = _string(sampling["dense_region"], "vlm.sampling.dense_region")
    if dense_region not in _DENSE_REGIONS:
        raise ValueError(f"vlm.sampling.dense_region must be one of {sorted(_DENSE_REGIONS)}")

    review = _mapping(raw["review"], "review")
    _fields(
        review,
        {"mode", "confidence_threshold", "min_episode_duration", "min_sampled_frames"},
        "review",
    )
    review_mode = _string(review["mode"], "review.mode")
    if review_mode not in _REVIEW_MODES:
        raise ValueError(f"review.mode must be one of {sorted(_REVIEW_MODES)}")

    outputs = _mapping(raw["outputs"], "outputs")
    _fields(outputs, {"required", "optional"}, "outputs")
    required_outputs = _output_list(outputs["required"], "outputs.required")
    optional_outputs = _output_list(outputs["optional"], "outputs.optional")
    all_outputs = (*required_outputs, *optional_outputs)
    if len(set(all_outputs)) != len(all_outputs):
        raise ValueError("outputs must not contain duplicate paths")
    missing_outputs = _REQUIRED_OUTPUTS - set(required_outputs)
    if missing_outputs:
        raise ValueError(
            "outputs.required is missing required output paths: "
            f"{', '.join(sorted(missing_outputs))}"
        )

    return Profile(
        name=_identifier(raw["name"], "name"),
        version=version,
        adapter=_enum_identifier(raw["adapter"], "adapter", _ADAPTERS),
        plugin=_enum_identifier(raw["plugin"], "plugin", _PLUGINS),
        image_key=_string(raw["image_key"], "image_key"),
        vlm=VLMProfile(
            model_path=_string(vlm["model_path"], "vlm.model_path"),
            prompt_version=_enum_identifier(
                vlm["prompt_version"], "vlm.prompt_version", _PROMPT_VERSIONS
            ),
            sampling=SamplingProfile(
                max_global_frames=_integer(
                    sampling["max_global_frames"], "vlm.sampling.max_global_frames", 1, 10_000
                ),
                global_sample_interval=_number(
                    sampling["global_sample_interval"],
                    "vlm.sampling.global_sample_interval",
                    0.001,
                    86_400,
                ),
                max_dense_frames=_integer(
                    sampling["max_dense_frames"], "vlm.sampling.max_dense_frames", 1, 10_000
                ),
                dense_sample_interval=_number(
                    sampling["dense_sample_interval"],
                    "vlm.sampling.dense_sample_interval",
                    0.001,
                    86_400,
                ),
                dense_region=dense_region,
            ),
            max_image_size=_integer(vlm["max_image_size"], "vlm.max_image_size", 1, 16_384),
            max_new_tokens=_integer(vlm["max_new_tokens"], "vlm.max_new_tokens", 1, 1_000_000),
        ),
        review=ReviewProfile(
            mode=review_mode,
            confidence_threshold=_number(
                review["confidence_threshold"], "review.confidence_threshold", 0, 1
            ),
            min_episode_duration=_number(
                review["min_episode_duration"], "review.min_episode_duration", 0, 86_400
            ),
            min_sampled_frames=_integer(
                review["min_sampled_frames"], "review.min_sampled_frames", 1, 10_000
            ),
        ),
        outputs=OutputProfile(required=required_outputs, optional=optional_outputs),
    )
