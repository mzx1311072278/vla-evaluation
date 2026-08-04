#!/usr/bin/env python3
"""根据 Genie02 A 侧数据生成 episode_metrics.csv。"""
from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from genie02_eval_common import EPISODE_METRIC_FIELDS, EvaluationError, finite_float
from genie02_eval_common import load_episodes, load_session, parse_session_args
from genie02_eval_common import read_json, resolve_path
from genie02_eval_common import prepare_output_dir, require_session_dir
TIME_KEYS = ("smooth_send_t", "sent_t", "timestamp")
VALUE_TIME_KEYS = (
    ("smooth_send_y", "smooth_send_t"),
    ("sent_y", "sent_t"),
    ("action", "timestamp"),
)
VALUE_KEYS = tuple(value for value, _ in VALUE_TIME_KEYS)

@dataclass
class Trajectory:
    values: np.ndarray
    times: np.ndarray | None
    intervention: np.ndarray | None
    space: str
    arm: str = ""

def _trajectory_directories(session: dict[str, Any], session_dir: Path) -> list[Path]:
    """返回按优先级排列的轨迹目录，支持采集机绝对路径迁移后的本地兜底。"""
    directories: list[Path] = []
    if session.get("trajectory_log_dir"):
        directories.append(resolve_path(session["trajectory_log_dir"], session_dir))
    refs_path = session_dir / "raw_refs.json"
    if refs_path.is_file():
        refs = read_json(refs_path)
        if refs.get("trajectory_log_dir"):
            directories.append(resolve_path(refs["trajectory_log_dir"], session_dir))
    directories.append(session_dir / "trajectories")
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        path = directory.expanduser()
        key = path if path.is_absolute() else (session_dir / path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique

@lru_cache(maxsize=None)
def _action_names_for_path(path: Path) -> list[str]:
    """读取轨迹目录 meta.json 中的 action_names；不存在时返回空列表。"""
    meta_path = path.parent / "meta.json"
    if not meta_path.is_file():
        return []
    meta = read_json(meta_path)
    names = meta.get("action_names", [])
    return [name for name in names if isinstance(name, str)]

@lru_cache(maxsize=None)
def _lerobot_action_names_for_path(path: Path, key: str) -> list[str]:
    """读取 LeRobot meta/info.json 中的向量列名。"""
    root = path.parents[2] if len(path.parents) >= 3 else path.parent
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return []
    info = read_json(info_path)
    names = (
        info.get("features", {})
        .get(key, {})
        .get("names", [])
    )
    return [name for name in names if isinstance(name, str)]

def _ee_xyz_columns(values: np.ndarray, action_names: Sequence[str]) -> list[int]:
    """末端模式只取位置 xyz，优先依据 action_names，兼容 Genie02 16 维 EE 向量。"""
    columns = [
        index
        for index, name in enumerate(action_names[: values.shape[1]])
        if "_ee." in name and name.rsplit(".", 1)[-1] in {"x", "y", "z"}
    ]
    if columns:
        return columns
    if values.shape[1] >= 14:
        return [0, 1, 2, 7, 8, 9]
    if values.shape[1] >= 3:
        return [0, 1, 2]
    raise EvaluationError("EE trajectory has fewer than 3 xyz columns")

def _matrix(value: Any, label: str) -> np.ndarray:
    """将轨迹值转换为二维浮点矩阵 [帧数, 维度数]。"""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        try:
            array = np.stack([np.asarray(item, dtype=float) for item in value])
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"{label} is not a numeric matrix") from exc
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise EvaluationError(f"{label} must have shape [frames, dimensions]")
    return array

def _to_trajectory(
    data: Any,
    session: dict[str, Any],
    source: str,
    action_names: Sequence[str] = (),
) -> Trajectory:
    """按文档优先级从 NPZ/Parquet 字段中提取轨迹、时间和介入标记。"""
    value_key, paired_time_key = next(
        ((value, time) for value, time in VALUE_TIME_KEYS if value in data),
        (None, None),
    )
    if value_key is None:
        raise EvaluationError(f"no supported trajectory array in {source}")
    values = _matrix(data[value_key], f"{source}:{value_key}")
    space = "ee_xyz" if session["rollout_mode"] == "ee" else "joint"
    arm = ""
    if space == "ee_xyz":
        columns = _ee_xyz_columns(values, action_names)
        selected_names = [action_names[index] for index in columns if index < len(action_names)]
        if selected_names and all(name.startswith("right_ee.") for name in selected_names):
            arm = "right"
        elif selected_names and all(name.startswith("left_ee.") for name in selected_names):
            arm = "left"
        elif values.shape[1] < 6:
            arm = str(session.get("single_arm", ""))
        values = values[:, columns]
    time_key = (
        paired_time_key
        if paired_time_key is not None and paired_time_key in data
        else next((key for key in TIME_KEYS if key in data), None)
    )
    times = np.asarray(data[time_key], dtype=float).reshape(-1) if time_key else None
    intervention = np.asarray(data["is_intervention"]).reshape(-1) if "is_intervention" in data else None
    return Trajectory(values, times, intervention, space, arm)

def _load_npz(path: Path, session: dict[str, Any]) -> Trajectory:
    """读取 NPZ 并返回统一轨迹对象。"""
    try:
        with np.load(path, allow_pickle=False) as data:
            return _to_trajectory(data, session, str(path), _action_names_for_path(path))
    except FileNotFoundError as exc:
        raise EvaluationError(f"trajectory does not exist: {path}") from exc
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot read trajectory {path}: {exc}") from exc

def _load_parquet(path: Path, session: dict[str, Any], index: int) -> Trajectory:
    """从 LeRobot Parquet 中读取指定 episode 的轨迹。"""
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise EvaluationError(f"cannot read parquet {path}: {exc}") from exc
    if "episode_index" in frame:
        frame = frame[frame["episode_index"] == index]
    if frame.empty:
        raise EvaluationError(f"episode {index} is absent from {path}")
    keys = (*VALUE_KEYS, *TIME_KEYS, "is_intervention", "complementary_info.is_intervention")
    data = {key: frame[key].to_numpy() for key in keys if key in frame}
    if "is_intervention" not in data and "complementary_info.is_intervention" in data:
        data["is_intervention"] = data["complementary_info.is_intervention"]
    names_key = next((value for value in VALUE_KEYS if value in data), "action")
    return _to_trajectory(
        data,
        session,
        str(path),
        _lerobot_action_names_for_path(path, names_key),
    )

def _find_npz(directory: Path, index: int) -> Path | None:
    """在轨迹目录或 native episode 目录中查找约定命名的 NPZ。"""
    candidates = (
        directory / f"episode_{index:03d}.npz",
        directory / f"episode_{index:06d}.npz",
        directory / f"episode_{index:06d}" / "frames.npz",
        directory / "frames.npz",
    )
    return next((path for path in candidates if path.is_file()), None)

def _load_file(path: Path, session: dict[str, Any], index: int) -> Trajectory:
    """根据扩展名分派 NPZ 或 Parquet 读取器。"""
    if path.suffix.lower() == ".npz":
        return _load_npz(path, session)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return _load_parquet(path, session, index)
    raise EvaluationError(f"unsupported trajectory format: {path}")

def _load_trajectory(
    row: dict[str, str], session: dict[str, Any], session_dir: Path
) -> Trajectory:
    """依次从 trajectory_path、轨迹日志目录和 episode_path 定位轨迹。"""
    index = int(row["episode_index"])
    directories = _trajectory_directories(session, session_dir)
    if row["trajectory_path"].strip():
        path = resolve_path(row["trajectory_path"], session_dir)
        if path.is_dir():
            found = _find_npz(path, index)
            if found:
                return _load_npz(found, session)
        elif path.is_file():
            return _load_file(path, session, index)
        for directory in directories:
            found = directory / path.name
            if found.is_file():
                return _load_file(found, session, index)
    for directory in directories:
        found = _find_npz(directory, index)
        if found:
            return _load_npz(found, session)
    if not row["episode_path"].strip():
        raise EvaluationError("no_trajectory")
    episode_path = resolve_path(row["episode_path"], session_dir)
    if session["dataset_backend"] == "native":
        found = _find_npz(episode_path, index)
        if found:
            return _load_npz(found, session)
        raise EvaluationError(f"no native frames.npz for episode {index}")
    if not episode_path.exists():
        raise EvaluationError(f"episode_path does not exist: {episode_path}")
    last_error = ""
    for path in sorted((episode_path / "data").glob("**/*.parquet")):
        try:
            return _load_parquet(path, session, index)
        except EvaluationError as exc:
            last_error = str(exc)
    suffix = f" ({last_error})" if last_error else ""
    raise EvaluationError(f"no LeRobot parquet for episode {index}{suffix}")

def _jerk(values: np.ndarray, dt: float) -> np.ndarray:
    return (values[3:] - 3 * values[2:-1] + 3 * values[1:-2] - values[:-3]) / (
        dt**3
    )


def _smoothness_value(values: np.ndarray, dt: float) -> float:
    jerk = _jerk(values, dt)
    raw = float(np.sum(jerk * jerk) * dt)
    if not math.isfinite(raw):
        raise EvaluationError("non_finite_smoothness")
    return math.log10(raw + 1.0)


def _smoothness_inputs(
    trajectory: Trajectory, fps: float
) -> tuple[np.ndarray, np.ndarray, float]:
    values, times, intervention = trajectory.values, trajectory.times, trajectory.intervention
    if times is not None and len(times) != len(values):
        raise EvaluationError("trajectory timestamp length mismatch")
    if intervention is not None and len(intervention) != len(values):
        raise EvaluationError("trajectory intervention-mask length mismatch")
    valid = np.all(np.isfinite(values), axis=1)
    if times is not None:
        valid &= np.isfinite(times)
    if intervention is not None:
        valid &= np.asarray(intervention, dtype=float) == 0
    values = values[valid]
    if times is not None:
        times = times[valid]
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        keep = np.r_[True, np.diff(times) > 0]
        times, values = times[keep], values[keep]
    if len(values) < 4:
        raise EvaluationError("insufficient_frames")
    dt = 1.0 / fps
    if times is not None and len(times) >= 2:
        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            dt = float(np.median(diffs))
    if times is None:
        times = np.arange(len(values), dtype=float) * dt
    else:
        times = times - times[0]
    return values, times, dt


def _smoothness_curve(trajectory: Trajectory, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """返回综合平滑度曲线。"""
    values, times, dt = _smoothness_inputs(trajectory, fps)
    jerk = _jerk(values, dt)
    curve = np.log10(np.sum(jerk * jerk, axis=1) * dt + 1.0)
    return times[3:], curve


def _smoothness(trajectory: Trajectory, fps: float) -> tuple[float, float | None, float | None, int]:
    """过滤无效/介入帧并分别计算综合、左右臂平滑度与有效帧数。"""
    values, _, dt = _smoothness_inputs(trajectory, fps)
    if trajectory.space == "ee_xyz":
        combined = _smoothness_value(values, dt)
        if values.shape[1] >= 6:
            left = _smoothness_value(values[:, :3], dt)
            right = _smoothness_value(values[:, 3:6], dt)
        elif trajectory.arm == "right":
            left, right = None, combined
        else:
            left, right = combined, None
        return combined, left, right, len(values)
    return _smoothness_value(values, dt), _smoothness_value(values, dt), None, len(values)

def _metric_row(row: dict[str, str], session: dict[str, Any], root: Path) -> dict[str, Any]:
    """生成一行派生指标；轨迹错误写入 skipped_reason 而不中断 session。"""
    result = {
        "session_id": row["session_id"],
        "episode_index": int(row["episode_index"]),
        "outcome": row["outcome"].strip().lower(),
        "duration_s": finite_float(row["duration_s"], "duration_s"),
        "smoothness": None,
        "left_smoothness": None,
        "right_smoothness": None,
        "smoothness_space": "",
        "smoothness_frames": None,
        "smoothness_skipped_reason": "",
    }
    try:
        trajectory = _load_trajectory(row, session, root)
        combined, left, right, frames = _smoothness(trajectory, float(session["fps"]))
        result.update(
            smoothness=combined,
            left_smoothness=left,
            right_smoothness=right,
            smoothness_space=trajectory.space,
            smoothness_frames=frames,
        )
    except EvaluationError as exc:
        result["smoothness_skipped_reason"] = str(exc)
    return result

def generate_episode_metrics(
    session_dir: Path, output_dir: Path | None = None
) -> list[dict[str, Any]]:
    """读取 A 侧文件、计算每条 episode 指标并写 episode_metrics.csv。"""
    root = require_session_dir(session_dir)
    output_root = prepare_output_dir(output_dir)
    session = load_session(root)
    rows = [_metric_row(row, session, root) for row in load_episodes(root, session)]
    with (output_root / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["duration_s"] = f"{row['duration_s']:.3f}"
            for field in ("smoothness", "left_smoothness", "right_smoothness"):
                output[field] = f"{row[field]:.12g}" if row[field] is not None else ""
            output["smoothness_frames"] = row["smoothness_frames"] or ""
            writer.writerow(output)
    return rows

def main(argv: Sequence[str] | None = None) -> int:
    """解析 session 目录并执行 episode 指标生成命令。"""
    args = parse_session_args(argv, "Generate Genie02 episode metrics.")
    try:
        rows = generate_episode_metrics(args.session_dir, args.output_dir)
    except EvaluationError as exc:
        print(f"error: {exc}")
        return 2
    computed = sum(
        row["smoothness"] is not None
        for row in rows
    )
    output_root = prepare_output_dir(args.output_dir)
    print(
        f"Wrote {output_root / 'episode_metrics.csv'}: "
        f"rows={len(rows)}, smoothness={computed}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
