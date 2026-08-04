from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT = """你正在分析机器人抓取任务的成功 episode 抽帧序列。

该 episode 的元数据已经标注为 success。你的任务不是重新评估整条是否成功，而是统计“最终成功抓取之前，发生了几次失败抓取”。

失败抓取定义：
夹爪已经接近目标并发生闭合/夹取动作，但目标物体（药盒）没有被夹爪抓住、夹空、滑落，或闭合后目标没有随夹爪稳定移动。

不计入失败抓取：
1. 机械臂只是移动或调整姿态；
2. 夹爪靠近/接近目标但没有明确闭合/夹取动作；如果 evidence 是“接近但未闭合”，必须删除该 attempt，不能计数；
3. 最终成功抓取本身；
4. 成功之后的移动、放置、调整动作。

最终成功抓取定义：
夹爪闭合后，目标物体被夹住，并随夹爪移动、抬起或保持在夹爪中。

连续抓取事件合并规则：
如果若干连续帧表现为同一次靠近、对准、接触、闭合过程，并且该过程最终成功夹住目标，则整个过程只算 1 次成功抓取；不要把前面的靠近、对准、尚未闭合、刚接触但未完成闭合的帧拆成失败尝试。
失败尝试必须是一个已经结束的独立事件：夹爪闭合/夹取后目标未被带起或滑落，随后夹爪离开目标、重新打开，或开始下一轮对准/抓取。

判断顺序：
1. 先定位最终成功抓取发生的大致时间；
2. 只统计这个时间之前的失败抓取次数；只有看到“明确闭合/夹取动作 + 未抓住/夹空/滑落”才算失败抓取；
3. 如果能定位最终成功抓取，但没有看到明确失败抓取，则 pre_success_failed_attempt_count=0，failed_attempts_before_success=[]；
4. 只有最终成功抓取也看不清，或严重遮挡导致完全无法判断失败次数时，才输出 pre_success_failed_attempt_count=null；
5. 由于视频来自腕部相机，画面中目标移动可能由相机运动导致。不要仅凭目标在画面中移动判断成功或失败；优先看夹爪闭合、接触、夹住、滑落关系。

rough_start_time 和 rough_end_time 必须使用 episode_time，不要使用原 mp4 的 video_time。
所有时间字段必须是 JSON 数字，例如 19.2；不要写 "19.2s" 或 19.2s。

请只输出合法 JSON，不要输出额外解释。

JSON schema:
{
  "episode_success": true,
  "pre_success_failed_attempt_count": 1,
  "failed_attempts_before_success": [
    {
      "attempt_id": 1,
      "rough_start_time": 0.0,
      "rough_end_time": 0.0,
      "evidence": "简短证据"
    }
  ],
  "final_success_time": 0.0,
  "confidence": 0.0,
  "reason": "简短原因"
}

一致性要求：
1. episode_success 必须为 true；
2. pre_success_failed_attempt_count 必须等于 failed_attempts_before_success 数组长度；
3. 看到最终成功但没有明确失败抓取时，pre_success_failed_attempt_count=0，failed_attempts_before_success=[]；
4. 只有最终成功也无法定位或严重遮挡时，pre_success_failed_attempt_count=null，failed_attempts_before_success=[]；
5. 不要把最终成功抓取算进失败次数。
6. “接近目标但未闭合/没有明确闭合”不是失败抓取，不能出现在 failed_attempts_before_success。
7. 不要逐帧计数；时间相邻且动作连续的帧属于同一个抓取事件。
"""


def _prompt_with_frame_times(frame_timestamps: list[dict[str, Any]]) -> str:
    lines = [
        "下面是该 episode 的抽帧序列：",
        "",
        "global frames：全局过程，用于理解 episode 起止和大致阶段。",
    ]
    for item in [item for item in frame_timestamps if item.get("frame_type") == "global"]:
        lines.append(
            f"{item['frame']}: episode_time={item['episode_time']}s, "
            f"video_time={item['video_time']}s"
        )
    lines += [
        "",
        "dense frames：局部细节，用于优先判断夹爪闭合、接触、夹住或滑落。",
    ]
    for item in [item for item in frame_timestamps if item.get("frame_type") == "dense"]:
        lines.append(
            f"{item['frame']}: episode_time={item['episode_time']}s, "
            f"video_time={item['video_time']}s"
        )
    return PROMPT + "\n\n" + "\n".join(lines)


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool, str]:
    if not text.strip():
        return None, False, "VLM output is empty"
    try:
        return json.loads(text), True, ""
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        pass
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


def fallback_result(raw_response: str = "", warning: str = "invalid_vlm_json", parse_error: str = "") -> dict[str, Any]:
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
    }


def _is_non_grasp_attempt(attempt: dict[str, Any]) -> bool:
    evidence = str(attempt.get("evidence", ""))
    if "简短证据" in evidence or not evidence.strip():
        return True
    return ("未闭合" in evidence or "没有明确闭合" in evidence) and ("接近" in evidence or "靠近" in evidence)


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
        return fallback_result(raw_response, "missing_required_fields", f"Missing required fields: {missing}")
    if parsed["episode_success"] is not True:
        return fallback_result(raw_response, "missing_required_fields", "episode_success must be true")
    attempts = parsed["failed_attempts_before_success"]
    if not isinstance(attempts, list):
        return fallback_result(raw_response, "missing_required_fields", "failed_attempts_before_success must be a list")
    count = parsed["pre_success_failed_attempt_count"]
    if count is not None and (not isinstance(count, int) or count != len(attempts)):
        return fallback_result(raw_response, "missing_required_fields", "pre_success_failed_attempt_count must match failed_attempts_before_success length")
    if count is None and attempts:
        return fallback_result(raw_response, "missing_required_fields", "failed_attempts_before_success must be empty when count is null")
    for i, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            return fallback_result(raw_response, "missing_required_fields", f"failed_attempts_before_success[{i}] must be an object")
    filtered_non_grasp_attempt = False
    filtered_attempts = [attempt for attempt in attempts if not _is_non_grasp_attempt(attempt)]
    if len(filtered_attempts) != len(attempts):
        filtered_non_grasp_attempt = True
        attempts = filtered_attempts
        count = len(attempts)
        parsed["failed_attempts_before_success"] = attempts
        parsed["pre_success_failed_attempt_count"] = count
    if not isinstance(parsed["confidence"], (int, float)):
        return fallback_result(raw_response, "missing_required_fields", "confidence must be a number")
    if not isinstance(parsed["reason"], str):
        return fallback_result(raw_response, "missing_required_fields", "reason must be a string")
    if "简短原因" in parsed["reason"]:
        return fallback_result(raw_response, "placeholder_reason", "reason must describe observed frames")
    final_success_time = parsed.get("final_success_time")
    if final_success_time is not None and not isinstance(final_success_time, (int, float)):
        return fallback_result(raw_response, "missing_required_fields", "final_success_time must be a number or null")
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
    def __init__(self, model_path: Path, max_new_tokens: int = 256):
        self.model_path = model_path.expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local VLM model path does not exist: {self.model_path}")
        self.max_new_tokens = max_new_tokens

        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        if torch.cuda.is_available():
            print("CUDA available: using GPU for VLM inference.")
            device_map = "auto"
            torch_dtype = "auto"
        else:
            print("CUDA not available: using CPU. VLM inference will be very slow.")
            device_map = None
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        if device_map is None:
            self.model.to("cpu")

    def analyze(
        self,
        frame_paths: list[Path],
        frame_timestamps: list[dict[str, Any]],
        episode_duration: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": str(path.resolve())} for path in frame_paths]
        content.append({"type": "text", "text": _prompt_with_frame_times(frame_timestamps)})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = video_inputs = inputs = generated = None
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if self.torch.cuda.is_available():
            inputs = inputs.to("cuda")

        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
            raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        finally:
            image_inputs = video_inputs = inputs = generated = None
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        parsed, ok, parse_error = extract_json(raw)
        if not ok or parsed is None:
            warning = "empty_vlm_response" if not raw else "invalid_vlm_json"
            return fallback_result(raw, warning, parse_error), False
        result = validate_vlm_result(parsed, raw, episode_duration)
        return result, bool(result.get("vlm_valid"))
