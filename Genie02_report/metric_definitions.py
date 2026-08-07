"""Versioned metric definitions shared by Markdown and Web reports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaFragment:
    text: str
    subscript: str = ""
    superscript: str = ""


@dataclass(frozen=True)
class FormulaLine:
    lhs: tuple[FormulaFragment, ...]
    rhs: tuple[FormulaFragment, ...] = ()
    numerator: tuple[FormulaFragment, ...] = ()
    denominator: tuple[FormulaFragment, ...] = ()


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    definition: str
    direction: str
    formulas: tuple[FormulaLine, ...]
    notes: tuple[str, ...] = ()


def _fragment(
    text: str, *, subscript: str = "", superscript: str = ""
) -> FormulaFragment:
    return FormulaFragment(text, subscript, superscript)


METRIC_DEFINITIONS = (
    MetricDefinition(
        key="gsr",
        label="GSR",
        definition="成功 Episode 数除以 Episode 总数。",
        direction="越大越好",
        formulas=(
            FormulaLine(
                lhs=(_fragment("GSR"),),
                numerator=(_fragment("N", subscript="success"),),
                denominator=(_fragment("N", subscript="total"),),
            ),
        ),
    ),
    MetricDefinition(
        key="tts_success",
        label="TTS（成功）",
        definition="仅对成功 Episode 的 duration_s 取算术平均值。",
        direction="越小越好",
        formulas=(
            FormulaLine(
                lhs=(_fragment("TTS"),),
                rhs=(
                    _fragment("mean("),
                    _fragment("duration", subscript="s"),
                    _fragment(" | outcome = success)"),
                ),
            ),
        ),
    ),
    MetricDefinition(
        key="smoothness",
        label="平滑度",
        definition="末端位置或关节轨迹的离散 jerk 能量经 log10 压缩后的数值。",
        direction="越小越平滑",
        formulas=(
            FormulaLine(
                lhs=(_fragment("S"),),
                rhs=(_fragment("log10(E + 1)"),),
            ),
            FormulaLine(
                lhs=(_fragment("E"),),
                rhs=(
                    _fragment("sum(||j", subscript="k"),
                    _fragment("||", superscript="2"),
                    _fragment(") * delta", subscript="t"),
                ),
            ),
            FormulaLine(
                lhs=(_fragment("j", subscript="k"),),
                numerator=(
                    _fragment("x", subscript="k"),
                    _fragment(" - 3 x", subscript="(k-1)"),
                    _fragment(" + 3 x", subscript="(k-2)"),
                    _fragment(" - x", subscript="(k-3)"),
                ),
                denominator=(_fragment("delta", subscript="t", superscript="3"),),
            ),
        ),
        notes=(
            "过滤非有限帧和操作员介入帧。",
            "按时间戳稳定排序并去除重复时间戳。",
            "使用相邻有效时间差的中位数作为 delta_t；无时间戳时使用 1/FPS。",
            "少于 4 个有效帧时不计算平滑度。",
        ),
    ),
)


def metric_definition_rows() -> list[tuple[str, str, str]]:
    return [
        (metric.label, metric.definition, metric.direction)
        for metric in METRIC_DEFINITIONS
    ]


def _plain_fragments(fragments: Iterable[FormulaFragment]) -> str:
    rendered: list[str] = []
    for fragment in fragments:
        value = fragment.text
        if fragment.subscript:
            value += f"_{fragment.subscript}"
        if fragment.superscript:
            value += f"^{fragment.superscript}"
        rendered.append(value)
    return "".join(rendered)


def markdown_formula_lines(
    metrics: Iterable[MetricDefinition] = METRIC_DEFINITIONS,
) -> list[str]:
    lines: list[str] = []
    for metric in metrics:
        for formula in metric.formulas:
            lhs = _plain_fragments(formula.lhs)
            if formula.rhs:
                rhs = _plain_fragments(formula.rhs)
            else:
                numerator = _plain_fragments(formula.numerator)
                if len(formula.numerator) > 1:
                    numerator = f"({numerator})"
                rhs = f"{numerator} / {_plain_fragments(formula.denominator)}"
            lines.append(f"{lhs} = {rhs}")
    return lines
