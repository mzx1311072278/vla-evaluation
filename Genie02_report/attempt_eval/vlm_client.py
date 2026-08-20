from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_eval.exceptions import ModelLoadError

if __package__:
    from .prompt_registry import PROMPT, PROMPT_VERSION, prompt_for_version
    from .prompt_registry import PROMPTS as PROMPTS  # noqa: PLC0414 - compatibility re-export
else:
    from prompt_registry import PROMPT, PROMPT_VERSION, prompt_for_version
    from prompt_registry import PROMPTS as PROMPTS  # noqa: PLC0414 - compatibility re-export


def _prompt_with_frame_times(frame_timestamps: list[dict[str, Any]], prompt: str = PROMPT) -> str:
    lines = [
        "下面是该 episode 的抽帧序列：",
        "",
        "global frames：全局过程，用于理解 episode 起止和大致阶段。",
    ]
    for item in [item for item in frame_timestamps if item.get("frame_type") == "global"]:
        camera = f", camera={item['camera_key']}" if item.get("camera_key") else ""
        lines.append(
            f"{item['frame']}: episode_time={item['episode_time']}s, "
            f"video_time={item['video_time']}s{camera}"
        )
    lines += [
        "",
        "dense frames：局部细节，用于优先判断夹爪闭合、接触、夹住或滑落。",
    ]
    for item in [item for item in frame_timestamps if item.get("frame_type") == "dense"]:
        camera = f", camera={item['camera_key']}" if item.get("camera_key") else ""
        lines.append(
            f"{item['frame']}: episode_time={item['episode_time']}s, "
            f"video_time={item['video_time']}s{camera}"
        )
    return prompt + "\n\n" + "\n".join(lines)


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool, str]:
    if not text.strip():
        return None, False, "VLM output is empty"
    try:
        return json.loads(text), True, ""
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
    start = text.find("{")
    if start < 0:
        return None, False, parse_error
    depth = 0
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1]), True, ""
                except json.JSONDecodeError as exc:
                    return None, False, str(exc)
    return None, False, parse_error


def fallback_result(
    raw_response: str = "", warning: str = "invalid_vlm_json", parse_error: str = ""
) -> dict[str, Any]:
    warnings = ["vlm_invalid_response", warning]
    if not raw_response:
        warnings.append("empty_vlm_response")
        parse_error = parse_error or "VLM output is empty"
    return {
        "episode_success": None,
        "pre_success_failed_attempt_count": None,
        "failed_attempts_before_success": [],
        "final_success_time": None,
        "attempt_count": None,
        "success_count": None,
        "failed_count": None,
        "attempts": [],
        "confidence": 0.0,
        "vlm_valid": False,
        "reason": "VLM output is invalid or empty",
        "parse_error": parse_error,
        "raw_response": raw_response,
        "auto_warning": sorted(set(warnings)),
        "input_token_count": None,
        "context_token_limit": None,
        "cuda_peak_memory_allocated_bytes": None,
        "cuda_peak_memory_reserved_bytes": None,
    }


def _context_token_limit(config: dict[str, Any], model_family: str) -> int:
    if model_family in {"qwen3_vl", "qwen3_5"}:
        text_config = config.get("text_config")
        value = text_config.get("max_position_embeddings") if isinstance(text_config, dict) else None
    else:
        value = config.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("checkpoint config is missing a valid context token limit")
    return value


def _input_token_count(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    return len(input_ids[0])


def _safe_exception_name(error: Exception) -> str:
    value = "".join(
        character
        for character in type(error).__name__
        if character.isascii() and (character.isalnum() or character == "_")
    )[:64]
    return value or "Exception"


def _attach_resource_metrics(
    result: dict[str, Any],
    *,
    input_token_count: int | None,
    context_token_limit: int | None,
    cuda_allocated: int | None,
    cuda_reserved: int | None,
) -> dict[str, Any]:
    result["input_token_count"] = input_token_count
    result["context_token_limit"] = context_token_limit
    result["cuda_peak_memory_allocated_bytes"] = cuda_allocated
    result["cuda_peak_memory_reserved_bytes"] = cuda_reserved
    return result


def _is_non_grasp_attempt(attempt: dict[str, Any]) -> bool:
    evidence = str(attempt.get("evidence", ""))
    if "简短证据" in evidence or not evidence.strip():
        return True
    return ("未闭合" in evidence or "没有明确闭合" in evidence) and (
        "接近" in evidence or "靠近" in evidence
    )


def validate_vlm_result(
    parsed: dict[str, Any],
    raw_response: str,
    episode_duration: float | None = None,
) -> dict[str, Any]:
    required = [
        "episode_success",
        "pre_success_failed_attempt_count",
        "failed_attempts_before_success",
        "confidence",
        "reason",
    ]
    missing = [key for key in required if key not in parsed]
    if missing:
        return fallback_result(
            raw_response, "missing_required_fields", f"Missing required fields: {missing}"
        )
    if parsed["episode_success"] is not True:
        return fallback_result(
            raw_response, "missing_required_fields", "episode_success must be true"
        )
    attempts = parsed["failed_attempts_before_success"]
    if not isinstance(attempts, list):
        return fallback_result(
            raw_response, "missing_required_fields", "failed_attempts_before_success must be a list"
        )
    count = parsed["pre_success_failed_attempt_count"]
    if count is not None and (not isinstance(count, int) or count != len(attempts)):
        return fallback_result(
            raw_response,
            "missing_required_fields",
            "pre_success_failed_attempt_count must match failed_attempts_before_success length",
        )
    if count is None and attempts:
        return fallback_result(
            raw_response,
            "missing_required_fields",
            "failed_attempts_before_success must be empty when count is null",
        )
    for i, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            return fallback_result(
                raw_response,
                "missing_required_fields",
                f"failed_attempts_before_success[{i}] must be an object",
            )
    filtered_non_grasp_attempt = False
    filtered_attempts = [attempt for attempt in attempts if not _is_non_grasp_attempt(attempt)]
    if len(filtered_attempts) != len(attempts):
        filtered_non_grasp_attempt = True
        attempts = filtered_attempts
        count = len(attempts)
        parsed["failed_attempts_before_success"] = attempts
        parsed["pre_success_failed_attempt_count"] = count
    if not isinstance(parsed["confidence"], (int, float)):
        return fallback_result(
            raw_response, "missing_required_fields", "confidence must be a number"
        )
    if not isinstance(parsed["reason"], str):
        return fallback_result(raw_response, "missing_required_fields", "reason must be a string")
    if "简短原因" in parsed["reason"]:
        return fallback_result(
            raw_response, "placeholder_reason", "reason must describe observed frames"
        )
    final_success_time = parsed.get("final_success_time")
    if final_success_time is not None and not isinstance(final_success_time, (int, float)):
        return fallback_result(
            raw_response, "missing_required_fields", "final_success_time must be a number or null"
        )
    result = dict(parsed)
    out_of_range_final_success_time = False
    if (
        episode_duration is not None
        and isinstance(final_success_time, (int, float))
        and not (0 <= final_success_time <= episode_duration + 0.5)
    ):
        result["final_success_time"] = None
        out_of_range_final_success_time = True
    result["confidence"] = float(result["confidence"])
    result["vlm_valid"] = True
    result["raw_response"] = raw_response
    result["parse_error"] = ""
    result.setdefault("auto_warning", [])
    if filtered_non_grasp_attempt:
        result["auto_warning"].append("filtered_non_grasp_attempt")
    if out_of_range_final_success_time:
        result["auto_warning"].append("final_success_time_out_of_range")
    result["attempts"] = attempts
    result["failed_count"] = count
    result["success_count"] = 1
    result["attempt_count"] = None if count is None else count + 1
    return result


class LocalVLMClient:
    def __init__(
        self,
        model_path: Path,
        model_family: str = "qwen2_5_vl",
        max_new_tokens: int = 256,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.model_path = model_path.expanduser()
        if not self.model_path.exists():
            error = FileNotFoundError("local VLM model path does not exist")
            error.add_note(str(self.model_path))
            raise ModelLoadError("The configured model could not be loaded.") from error
        try:
            config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
            model_type = config.get("model_type")
            if model_type != model_family:
                raise ValueError(
                    f"configured model family {model_family} does not match checkpoint "
                    f"model_type {model_type}"
                )
            self.context_token_limit = _context_token_limit(config, model_family)
        except Exception as error:
            raise ModelLoadError("The configured model could not be loaded.") from error
        self.model_family = model_family
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt_for_version(prompt_version)

        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.torch = torch
            if torch.cuda.is_available():
                print("CUDA available: using GPU for VLM inference.")
                model_kwargs = {"dtype": "auto", "device_map": "auto"}
            else:
                print("CUDA not available: using CPU. VLM inference will be very slow.")
                model_kwargs = {"dtype": torch.float32}

            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                local_files_only=True,
                **model_kwargs,
            )
            if not torch.cuda.is_available():
                self.model.to("cpu")
        except Exception as error:
            raise ModelLoadError("The configured model could not be loaded.") from error

    def close(self) -> None:
        self.model = None
        self.processor = None
        torch = getattr(self, "torch", None)
        cuda = getattr(torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must be best-effort.
            return

    def analyze(
        self,
        frame_paths: list[Path],
        frame_timestamps: list[dict[str, Any]],
        episode_duration: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        from qwen_vl_utils import process_vision_info

        cuda = self.torch.cuda
        cuda_available = bool(cuda.is_available())
        reset_peak = getattr(cuda, "reset_peak_memory_stats", None)
        if cuda_available and callable(reset_peak):
            reset_peak()
        input_token_count: int | None = None
        cuda_allocated: int | None = None
        cuda_reserved: int | None = None
        result: dict[str, Any] | None = None
        valid = False

        image_inputs = video_inputs = inputs = generated = None
        try:
            content = [{"type": "image", "image": str(path.resolve())} for path in frame_paths]
            content.append(
                {"type": "text", "text": _prompt_with_frame_times(frame_timestamps, self.prompt)}
            )
            messages = [{"role": "user", "content": content}]
            # Qwen3.8 (qwen3_5) defaults to thinking mode on; its reasoning tokens
            # would exhaust max_new_tokens before the JSON answer is produced, so
            # disable it per request. Older templates do not know this kwarg.
            template_kwargs = (
                {"enable_thinking": False} if self.model_family == "qwen3_5" else {}
            )
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            image_inputs, video_inputs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            input_token_count = _input_token_count(inputs.input_ids)
            if input_token_count + self.max_new_tokens > self.context_token_limit:
                result = fallback_result(
                    "",
                    "context_length_exceeded",
                    "input and output token budget exceeds the model context limit",
                )
            else:
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                generated = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
                raw = self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                parsed, ok, parse_error = extract_json(raw)
                if not ok or parsed is None:
                    warning = "empty_vlm_response" if not raw else "invalid_vlm_json"
                    result = fallback_result(raw, warning, parse_error)
                else:
                    result = validate_vlm_result(parsed, raw, episode_duration)
                    valid = bool(result.get("vlm_valid"))
        except Exception as error:  # noqa: BLE001 - sanitize per-Episode inference errors.
            result = fallback_result(
                "",
                "inference_error",
                f"inference_error:{_safe_exception_name(error)}",
            )
        finally:
            if cuda_available:
                maximum_allocated = getattr(cuda, "max_memory_allocated", None)
                maximum_reserved = getattr(cuda, "max_memory_reserved", None)
                if callable(maximum_allocated):
                    cuda_allocated = int(maximum_allocated())
                if callable(maximum_reserved):
                    cuda_reserved = int(maximum_reserved())
            image_inputs = video_inputs = inputs = generated = None
            if cuda_available:
                cuda.empty_cache()
        if result is None:
            raise RuntimeError("VLM analysis completed without a result")
        return (
            _attach_resource_metrics(
                result,
                input_token_count=input_token_count,
                context_token_limit=self.context_token_limit,
                cuda_allocated=cuda_allocated,
                cuda_reserved=cuda_reserved,
            ),
            valid,
        )
