"""Immutable, dependency-free prompt registry for Genie02 attempt evaluation."""

from collections.abc import Mapping
from types import MappingProxyType

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

PROMPT_VERSION = "genie02-attempt-v1"
PROMPTS: Mapping[str, str] = MappingProxyType({PROMPT_VERSION: PROMPT})
SUPPORTED_PROMPT_VERSIONS = frozenset(PROMPTS)


def prompt_for_version(prompt_version: str) -> str:
    if not isinstance(prompt_version, str):
        raise TypeError("prompt_version must be a string")
    try:
        return PROMPTS[prompt_version]
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPTS))
        raise ValueError(f"prompt_version must be one of: {supported}") from exc
