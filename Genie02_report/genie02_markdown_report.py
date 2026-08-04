#!/usr/bin/env python3
"""Render report.md from the four inputs required by document section 6.7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from datetime import date

from genie02_eval_common import (
    EvaluationError,
    load_episode_metrics,
    load_episodes,
    load_metrics_core,
    load_session,
    parse_session_args,
    prepare_output_dir,
    require_session_dir,
)


SMOOTHNESS_CHART = "smoothness_curve.svg"


def _format_number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _md_cell(value: Any) -> str:
    return (
        str(value if value not in (None, "") else "—")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def _mean_number(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None and value != ""]
    return sum(numbers) / len(numbers) if numbers else None


def _write_smoothness_chart(
    output_root: Path,
    episodes: Sequence[dict[str, str]],
    episode_metrics: Sequence[dict[str, Any]],
) -> bool:
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    rows = [
        row
        for row in sorted(episode_metrics, key=lambda item: item["episode_index"])
        if row["smoothness"] is not None
    ]
    if not rows:
        (output_root / SMOOTHNESS_CHART).unlink(missing_ok=True)
        return False

    values = [float(row["smoothness"]) for row in rows]
    width, height = 820, 330
    left_pad, right_pad, top_pad, bottom_pad = 68, 28, 44, 62
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad
    y_max = max(1.0, max(values) * 1.15)
    bar_gap = 8
    bar_w = max(10, (plot_w - bar_gap * (len(rows) - 1)) / len(rows))

    def y_pos(value: float) -> float:
        return top_pad + plot_h - (value / y_max * plot_h)

    y_ticks = [y_max * index / 4 for index in range(5)]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif}</style>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="16">Episode 平滑度概览</text>',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + plot_h}" stroke="#333"/>',
        f'<line x1="{left_pad}" y1="{top_pad + plot_h}" x2="{left_pad + plot_w}" y2="{top_pad + plot_h}" stroke="#333"/>',
    ]
    for tick in y_ticks:
        y = y_pos(tick)
        svg.extend(
            [
                f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left_pad - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{tick:.1f}</text>',
            ]
        )
    for index, row in enumerate(rows):
        episode = episodes_by_index[row["episode_index"]]
        outcome = episode["outcome"].strip().lower()
        intervened = episode["operator_intervened"].strip().lower() == "true"
        value = float(row["smoothness"])
        x = left_pad + index * (bar_w + bar_gap)
        y = y_pos(value)
        fill = "#2f7d59" if outcome == "success" else "#c44e3b"
        stroke = "#111827" if intervened else "#ffffff"
        stroke_width = 2 if intervened else 1
        svg.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top_pad + plot_h - y:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>',
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="10">{value:.2f}</text>',
                f'<text x="{x + bar_w / 2:.1f}" y="{top_pad + plot_h + 18}" text-anchor="middle" font-size="11">Ep {row["episode_index"]}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{left_pad + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="13">Episode</text>',
            f'<text x="18" y="{top_pad + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top_pad + plot_h / 2:.1f})" font-size="13">平滑度 S</text>',
            f'<rect x="{width - 220}" y="38" width="12" height="12" fill="#2f7d59"/><text x="{width - 202}" y="49" font-size="12">success</text>',
            f'<rect x="{width - 140}" y="38" width="12" height="12" fill="#c44e3b"/><text x="{width - 122}" y="49" font-size="12">failure</text>',
            f'<rect x="{width - 220}" y="58" width="12" height="12" fill="#ffffff" stroke="#111827" stroke-width="2"/><text x="{width - 202}" y="69" font-size="12">有遥操介入</text>',
        ]
    )
    svg.append("</svg>")
    (output_root / SMOOTHNESS_CHART).write_text("\n".join(svg) + "\n", encoding="utf-8")
    return True


def _validate_report_inputs(
    episodes: Sequence[dict[str, str]],
    episode_metrics: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    episode_indices = {int(row["episode_index"]) for row in episodes}
    metric_indices = {row["episode_index"] for row in episode_metrics}
    if episode_indices != metric_indices:
        raise EvaluationError(
            "report inputs do not join one-to-one on episode_index"
        )
    n_success = sum(row["outcome"].strip().lower() == "success" for row in episodes)
    n_failure = sum(row["outcome"].strip().lower() == "failure" for row in episodes)
    if (
        metrics["n_episodes"],
        metrics["n_success"],
        metrics["n_failure"],
    ) != (len(episodes), n_success, n_failure):
        raise EvaluationError(
            "metrics_core.json counts do not match episodes.csv"
        )


def build_report(
    session: dict[str, Any],
    episodes: Sequence[dict[str, str]],
    episode_metrics: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    """Build the five report sections prescribed by document section 6.7."""
    _validate_report_inputs(episodes, episode_metrics, metrics)
    smoothness = metrics["smoothness"]
    left_smoothness = smoothness["left"]
    right_smoothness = smoothness["right"]
    mean_smoothness = _mean_number(row["smoothness"] for row in episode_metrics)
    lines = [
        f"# 真机评测报告",
        "",
        "## 1. 评测配置",
        "",
        f"- 日期：{_md_cell(session.get('date', date.today().isoformat()))}",
        f"- 任务：{_md_cell(session['task'])}",
        f"- 数据：{session['session_id']}",
        f"- 配置：{_md_cell(session['rollout_config_path'])}",
        f"- 模式 / 后端：{_md_cell(session['rollout_mode'])} / {_md_cell(session['dataset_backend'])}",
        f"- Episode：{metrics['n_episodes']} / {session['num_episodes_target']}（已完成 / 计划）",
        f"- FPS：{_md_cell(session['fps'])}",
        f"- 状态：{_md_cell(session['status'])}",
        "- 评测公式：<br>"
        "&emsp;&emsp;1. GSR = 成功 Episode 数 / Episode 总数<br>"
        "&emsp;&emsp;2. TTS = 成功 Episode 的 duration_s 均值<br>"
        "&emsp;&emsp;3. 报告平滑度 $S=log10(E+1)$，原始平滑度 $E=Σ||j_k||² \cdot Δt$，其中 $j_k≈(x_k-3x_{k-1}+3x_{k-2}-x_{k-3})/(Δt)^3$ <br>"
        "&emsp;&emsp;&emsp;参数：$S$ 为报告展示的平滑度，$E$ 为平滑度原始量，$k$ 为帧索引，$j_k$ 为第 k 帧 jerk，$x_k$ 为第 k 帧末端 xyz 位置向量，$Δt$ 为相邻轨迹帧时间间隔；综合与左右臂分别计算，越小越平滑",
        "",
        "## 2. 核心指标",
        "",
        "| 指标 | 数值 | 样本量 |",
        "|:---:|:---:|:---:|",
        f"| GSR | {_format_number(metrics['gsr'] * 100, 1, '%')} | {metrics['n_episodes']} |",
        f"| TTS（成功） | {_format_number(metrics['mean_tts_success_s'], 3, ' s')} | {metrics['n_success']} |",
        f"| 平滑度（越小越好） | {_format_number(left_smoothness['mean'], 6)}（左臂）<br>{_format_number(right_smoothness['mean'], 6)}（右臂） | {smoothness['n_episodes']} |",
        "",
        *([f"![平滑度概览]({SMOOTHNESS_CHART})", ""] if smoothness["n_episodes"] else []),
        "## 3. Episode 明细",
        "",
        "| Episode | 结果 | 时长 (s) | 平滑度 | 左臂平滑度 | 右臂平滑度 | 空间 | 帧数 | 跳过原因 | 备注 |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    for row in sorted(episode_metrics, key=lambda item: item["episode_index"]):
        source = episodes_by_index[row["episode_index"]]
        lines.append(
            "| {episode_index} | {outcome} | {duration} | {smoothness} | "
            "{left_smoothness} | {right_smoothness} | {space} | {frames} | {reason} | {notes} |".format(
                episode_index=row["episode_index"],
                outcome=_md_cell(row["outcome"]),
                duration=_format_number(source["duration_s"]),
                smoothness=_format_number(row["smoothness"], 6),
                left_smoothness=_format_number(row["left_smoothness"], 6),
                right_smoothness=_format_number(row["right_smoothness"], 6),
                space=_md_cell(row["smoothness_space"]),
                frames=_md_cell(row["smoothness_frames"]),
                reason=_md_cell(row["smoothness_skipped_reason"]),
                notes=_md_cell(source.get("notes", "")),
            )
        )

    failures = [
        row for row in episodes if row["outcome"].strip().lower() == "failure"
    ]
    lines.extend(["", "## 4. 失败案例", ""])
    if failures:
        lines.extend(
            ["| Episode | 时长 (s) | 遥操介入 | 备注 |", "|:---:|:---:|:---:|:---:|"]
        )
        for row in sorted(failures, key=lambda item: int(item["episode_index"])):
            lines.append(
                f"| {int(row['episode_index'])} | {_format_number(row['duration_s'])} | "
                f"{_md_cell(row['operator_intervened'])} | {_md_cell(row.get('notes', ''))} |"
            )
    else:
        lines.append("无失败 episode。")

    missing_smoothness = metrics["n_episodes"] - smoothness["n_episodes"]
    conclusion = (
        f"本次共完成 {metrics['n_episodes']} 条 episode，成功 {metrics['n_success']} 条、"
        f"失败 {metrics['n_failure']} 条，GSR 为 "
        f"{_format_number(metrics['gsr'] * 100, 1, '%')}。"
    )
    conclusion += (
        f"成功 episode 的平均 TTS 为 "
        f"{_format_number(metrics['mean_tts_success_s'], 3, ' s')}。"
        if metrics["n_success"]
        else "本次无成功 episode，TTS 不可用。"
    )
    lines.extend(["", "## 5. 结论", "", conclusion])
    if smoothness["n_episodes"]:
        lines.append(
            f"平滑度在 {smoothness['n_episodes']} 条 episode 上有效；"
            f"综合均值 {_format_number(mean_smoothness, 6)}，"
            f"左臂均值 {_format_number(left_smoothness['mean'], 6)}，"
            f"右臂均值 {_format_number(right_smoothness['mean'], 6)}。"
        )
    else:
        lines.append("没有 episode 具备可用轨迹，无法汇总平滑度。")
    if missing_smoothness > 0:
        lines.append(
            f"另有 {missing_smoothness} 条 episode 因轨迹不可用而未计入平滑度。"
        )
    return "\n".join(lines) + "\n"


def generate_markdown_report(
    session_dir: Path, output_dir: Path | None = None
) -> Path:
    """Read all four documented inputs from disk and write only dated report markdown."""
    session_dir = require_session_dir(session_dir)
    output_root = prepare_output_dir(output_dir)
    session = load_session(session_dir)
    episodes = load_episodes(session_dir, session)
    episode_metrics = load_episode_metrics(output_root, session)
    metrics = load_metrics_core(output_root, session)
    _write_smoothness_chart(output_root, episodes, episode_metrics)
    report = build_report(session, episodes, episode_metrics, metrics)
    output = output_root / f"report_{date.today().strftime('%Y%m%d')}.md"
    output.write_text(report, encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_session_args(
        argv, "Generate Genie02 report.md from the four documented inputs."
    )
    try:
        output = generate_markdown_report(args.session_dir, args.output_dir)
    except EvaluationError as exc:
        print(f"error: {exc}")
        return 2
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
