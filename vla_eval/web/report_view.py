"""Build the report presentation model from persisted evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Genie02_report.genie02_eval_common import (
    EvaluationError,
    load_episode_metrics,
    load_episodes,
    load_metrics_core,
    load_session,
)
from Genie02_report.metric_definitions import METRIC_DEFINITIONS
from vla_eval.models import Dataset, EvaluationJob


def _format_percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) * 100:.1f}%"


def _format_float(value: Any, digits: int = 3, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _smoothness_summary(smoothness: Any) -> str:
    if not isinstance(smoothness, dict):
        return "—"
    space = str(smoothness.get("space") or "—")
    for side_name in ("left", "right"):
        side = smoothness.get(side_name)
        mean = side.get("mean") if isinstance(side, dict) else None
        if isinstance(mean, (int, float)) and not isinstance(mean, bool):
            return f"{space} · 平均 {float(mean):.3f}"
    return space


def _format_timestamp(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _episode_evidence(attempt: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not attempt:
        return None, None
    video_file = attempt.get("video_file")
    evidence_path = video_file if isinstance(video_file, str) and video_file else None
    from_ts = _format_timestamp(attempt.get("from_timestamp"))
    to_ts = _format_timestamp(attempt.get("to_timestamp"))
    evidence_range = None
    if from_ts is not None or to_ts is not None:
        evidence_range = f"{from_ts or '—'} → {to_ts or '—'}"
    return evidence_path, evidence_range


def _load_attempts(output_dir: Path) -> tuple[dict[int, dict[str, Any]], str]:
    path = output_dir / "attempt_eval" / "attempt_summary.json"
    if not path.is_file():
        return {}, "missing"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}, "invalid"
    if not isinstance(loaded, list):
        return {}, "invalid"
    attempts: dict[int, dict[str, Any]] = {}
    for row in loaded:
        if not isinstance(row, dict):
            return {}, "invalid"
        index = row.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in attempts:
            return {}, "invalid"
        attempts[index] = row
    return attempts, "available"


def _configuration_rows(
    job: EvaluationJob, dataset: Dataset, session: dict[str, Any]
) -> list[dict[str, str]]:
    provenance = job.provenance_json or {}
    values = [
        ("任务", session.get("task"), "session.task"),
        ("数据集", dataset.name, "Dataset.name"),
        ("评测配置", job.profile_name, "EvaluationJob.profile_name"),
        ("配置版本", job.profile_version, "EvaluationJob.profile_version"),
        ("运行模式", session.get("rollout_mode"), "session.rollout_mode"),
        ("FPS", session.get("fps"), "session.fps"),
        ("数据后端", session.get("dataset_backend"), "session.dataset_backend"),
        ("数据指纹", dataset.fingerprint, "Dataset.fingerprint"),
        ("数据版本", session.get("codebase_version"), "meta/info.json"),
        ("机器人类型", session.get("robot_type"), "meta/info.json"),
        ("应用版本", provenance.get("app_version"), "job.provenance_json"),
        ("Git SHA", provenance.get("git_sha"), "job.provenance_json"),
    ]
    return [
        {
            "label": label,
            "value": "—" if value in (None, "") else str(value),
            "source": source,
        }
        for label, value, source in values
    ]


def _source_rows(
    dataset: Dataset,
    output_dir: Path,
    attempt_status: str,
) -> list[dict[str, str]]:
    sources = [
        ("数据集元信息", dataset.path, "可用", "任务、FPS、后端和 Episode 原始记录"),
        ("核心指标", "metrics_core.json", "可用", "GSR、TTS 和平滑度汇总"),
        ("Episode 指标", "episode_metrics.csv", "可用", "逐 Episode 派生指标"),
        (
            "VLM 尝试结果",
            "attempt_eval/attempt_summary.json",
            {"available": "可用", "missing": "未产生", "invalid": "格式无效"}[
                attempt_status
            ],
            "VLM 尝试、置信度和复核状态",
        ),
    ]
    reports = sorted(path.name for path in output_dir.glob("report_*.md") if path.is_file())
    sources.append(
        (
            "文本报告",
            ", ".join(reports) if reports else "report_*.md",
            "可用" if reports else "未产生",
            "可下载的运行时 Markdown 报告",
        )
    )
    return [
        {"label": label, "source": source, "status": status, "purpose": purpose}
        for label, source, status, purpose in sources
    ]


def _evidence_gaps() -> list[dict[str, str]]:
    return [
        {
            "item": "模型架构与权重",
            "impact": "无法确认本次被测模型的结构、权重和 checkpoint。",
            "required_source": "版本化模型清单或模型注册表",
        },
        {
            "item": "训练记录",
            "impact": "无法确认实际训练超参数、步数、日志和时间。",
            "required_source": "训练任务与制品元数据接口",
        },
        {
            "item": "部署环境与硬件",
            "impact": "无法确认 PyTorch/CUDA/cuDNN、GPU 和工控机版本。",
            "required_source": "部署环境快照",
        },
        {
            "item": "标定与传感器",
            "impact": "无法确认标定版本、力传感器和控制安全阈值。",
            "required_source": "机器人配置与标定记录",
        },
        {
            "item": "泛化、鲁棒性与安全覆盖",
            "impact": "无法形成 OOD、干扰恢复或碰撞安全结论。",
            "required_source": "版本化测试矩阵与结果接口",
        },
        {
            "item": "发版门禁与审批",
            "impact": "系统不能自动给出准许或暂缓发版结论。",
            "required_source": "release-gate 策略和审批记录",
        },
    ]


def build_report_view(
    *, job: EvaluationJob, dataset: Dataset, output_dir: Path
) -> dict[str, Any]:
    dataset_root = Path(dataset.path)
    session = load_session(dataset_root)
    source_episodes = load_episodes(dataset_root, session)
    episode_metrics = load_episode_metrics(output_dir, session)
    metrics = load_metrics_core(output_dir, session)

    original_by_index = {int(row["episode_index"]): row for row in source_episodes}
    derived_by_index = {row["episode_index"]: row for row in episode_metrics}
    if original_by_index.keys() != derived_by_index.keys():
        raise EvaluationError("episode metrics do not match dataset episodes")

    attempts, attempt_status = _load_attempts(output_dir)
    unknown_attempts = set(attempts) - set(original_by_index)
    if unknown_attempts:
        attempt_status = "invalid"
        attempts = {}

    episodes: list[dict[str, Any]] = []
    for index in sorted(original_by_index):
        source = original_by_index[index]
        derived = derived_by_index[index]
        attempt = attempts.get(index)
        evidence_path, evidence_range = _episode_evidence(attempt)
        episodes.append(
            {
                "index": index,
                "outcome": derived["outcome"],
                "duration": _format_float(derived["duration_s"]),
                "smoothness": _format_float(derived["smoothness"], 6),
                "left_smoothness": _format_float(derived["left_smoothness"], 6),
                "right_smoothness": _format_float(derived["right_smoothness"], 6),
                "smoothness_space": derived["smoothness_space"] or "—",
                "smoothness_frames": derived["smoothness_frames"],
                "smoothness_skipped_reason": derived["smoothness_skipped_reason"],
                "operator_intervened": source["operator_intervened"].lower() == "true",
                "notes": source.get("notes", ""),
                "vlm": attempt,
                "evidence_path": evidence_path,
                "evidence_range": evidence_range,
            }
        )

    intervention_count = sum(
        1 for episode in episodes if episode["operator_intervened"]
    )
    short_success_count = sum(
        1
        for episode in episode_metrics
        if episode["outcome"] == "success"
        and episode["duration_s"] is not None
        and episode["duration_s"] < 1.0
    )
    smoothness_count = int(metrics["smoothness"]["n_episodes"])
    inspection_errors = (dataset.inspection_json or {}).get("errors", [])
    if not isinstance(inspection_errors, list):
        inspection_errors = []
    quality_rows = [
        {"label": "Episode 总数", "value": str(metrics["n_episodes"]), "status": "已记录"},
        {
            "label": "成功 / 失败",
            "value": f"{metrics['n_success']} / {metrics['n_failure']}",
            "status": "已记录",
        },
        {"label": "存在操作员介入", "value": str(intervention_count), "status": "已记录"},
        {"label": "成功且时长 < 1s", "value": str(short_success_count), "status": "诊断项"},
        {
            "label": "平滑度有效 / 缺失",
            "value": f"{smoothness_count} / {metrics['n_episodes'] - smoothness_count}",
            "status": "已记录",
        },
        {
            "label": "数据集预检错误",
            "value": "; ".join(str(error) for error in inspection_errors) or "无",
            "status": "通过" if not inspection_errors else "需处理",
        },
    ]

    pending_review = sum(
        1 for attempt in attempts.values() if attempt.get("needs_manual_review") is True
    )
    provenance = job.provenance_json or {}
    release_decision = "未配置自动发版判定"
    if provenance.get("release_policy_version") and provenance.get("release_decision"):
        release_decision = str(provenance["release_decision"])

    return {
        "headline": {
            "gsr": _format_percent(metrics.get("gsr")),
            "n_success": metrics["n_success"],
            "n_failure": metrics["n_failure"],
            "tts": _format_float(metrics.get("mean_tts_success_s"), suffix=" s"),
            "smoothness": _smoothness_summary(metrics.get("smoothness")),
            "pending_review": pending_review,
        },
        "summary_facts": {
            "episode_count": metrics["n_episodes"],
            "smoothness_coverage": smoothness_count,
            "vlm_configured": bool(provenance.get("vlm_backend")),
            "vlm_enabled": bool(job.vlm_enabled),
            "vlm_executed": attempt_status == "available",
            "vlm_artifact_status": attempt_status,
        },
        "configuration_rows": _configuration_rows(job, dataset, session),
        "source_rows": _source_rows(dataset, output_dir, attempt_status),
        "quality_rows": quality_rows,
        "episodes": episodes,
        "component_rows": [
            {
                "component": "评测适配器",
                "value": str(provenance.get("adapter") or "—"),
                "source": "job.provenance_json",
            },
            {
                "component": "VLM 插件",
                "value": str(provenance.get("plugin") or "—"),
                "source": "job.provenance_json",
            },
            {
                "component": "图像字段",
                "value": str(provenance.get("image_key") or "—"),
                "source": "job.provenance_json",
            },
            {
                "component": "动作 / 状态 schema",
                "value": json.dumps(session.get("features", {}), ensure_ascii=False)
                if session.get("features")
                else "—",
                "source": "meta/info.json",
            },
        ],
        "evidence_gaps": _evidence_gaps(),
        "release_decision": release_decision,
        "metric_definitions": METRIC_DEFINITIONS,
        "attempt_status": attempt_status,
        "has_vlm": bool(attempts),
        "pending_review": pending_review,
    }
