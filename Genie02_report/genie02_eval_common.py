"""Genie02 B 侧各阶段共用的数据契约与文件读写函数。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "report"
logger = logging.getLogger(__name__)
EPISODE_FIELDS = (
    "session_id",
    "episode_index",
    "episode_path",
    "trajectory_path",
    "t_start",
    "t_end",
    "duration_s",
    "outcome",
    "operator_intervened",
    "notes",
)
EPISODE_METRIC_FIELDS = (
    "session_id",
    "episode_index",
    "outcome",
    "duration_s",
    "smoothness",
    "left_smoothness",
    "right_smoothness",
    "smoothness_space",
    "smoothness_frames",
    "smoothness_skipped_reason",
)
SESSION_FIELDS = (
    "schema_version",
    "session_id",
    "created_at",
    "status",
    "rollout_config_path",
    "rollout_mode",
    "policy_path",
    "task",
    "num_episodes_target",
    "fps",
    "dataset_backend",
)
SESSION_DATASET_FIELDS = (
    "dataset_root",
)
CORE_FIELDS = (
    "schema_version",
    "session_id",
    "n_episodes",
    "n_success",
    "n_failure",
    "gsr",
    "mean_tts_success_s",
    "smoothness",
)
SMOOTHNESS_FIELDS = ("space", "left", "right", "n_episodes")
SMOOTHNESS_SUMMARY_FIELDS = ("mean", "std", "min", "max", "n_episodes")
class EvaluationError(RuntimeError):
    """输入文件或指标计算不符合评测契约。"""
def finite_float(value: Any, name: str) -> float:
    """将值转换为有限浮点数，并给出包含字段名的错误信息。"""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be finite, got {value!r}")
    return result

def _require_fields(data: Any, fields: Iterable[str], source: str) -> None:
    """校验字典或 CSV 表头是否包含契约字段。"""
    missing = [field for field in fields if field not in data]
    if missing:
        raise EvaluationError(f"{source} is missing fields: {', '.join(missing)}")

def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} must contain a JSON object")
    return value

def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()

def _lerobot_task(session_dir: Path) -> str:
    path = session_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        return ""
    try:
        import pandas as pd

        tasks = pd.read_parquet(path)
    except Exception as exc:
        raise EvaluationError(f"cannot read LeRobot tasks {path}: {exc}") from exc
    if len(tasks.index) and not isinstance(tasks.index, pd.RangeIndex):
        return str(tasks.index[0])
    for field in ("task", "tasks"):
        if field in tasks and len(tasks[field]):
            return str(tasks[field].iloc[0])
    return ""

def _lerobot_single_arm(info: dict[str, Any]) -> str:
    names = (
        info.get("features", {})
        .get("action", {})
        .get("names", [])
    )
    has_left = any(str(name).startswith("left_ee.") for name in names)
    has_right = any(str(name).startswith("right_ee.") for name in names)
    if has_right and not has_left:
        return "right"
    if has_left and not has_right:
        return "left"
    return ""

def _synthesize_lerobot_session(session_dir: Path) -> dict[str, Any]:
    info = read_json(session_dir / "meta" / "info.json")
    created_at = datetime.fromtimestamp(
        session_dir.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")
    session = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_dir.name,
        "created_at": created_at,
        "status": "completed",
        "rollout_config_path": "meta/info.json",
        "rollout_mode": "ee",
        "policy_path": "lerobot_dataset",
        "task": _lerobot_task(session_dir),
        "num_episodes_target": int(info.get("total_episodes", 0)),
        "fps": float(info.get("fps", 30)),
        "dataset_backend": "lerobot",
        "dataset_root": str(session_dir),
        "single_arm": _lerobot_single_arm(info),
    }
    return session

def _synthesize_lerobot_episodes(
    session_dir: Path, session: dict[str, Any]
) -> list[dict[str, str]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise EvaluationError(f"pandas is required for LeRobot episodes: {exc}") from exc

    rows: list[dict[str, str]] = []
    episode_files = sorted((session_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise EvaluationError("missing episodes.csv and no LeRobot episode metadata found")
    fps = finite_float(session["fps"], "session.fps")
    for meta_path in episode_files:
        try:
            meta = pd.read_parquet(meta_path)
        except Exception as exc:
            raise EvaluationError(f"cannot read LeRobot episode metadata {meta_path}: {exc}") from exc
        for _, item in meta.iterrows():
            index = int(item["episode_index"])
            data_file = (
                session_dir
                / "data"
                / f"chunk-{int(item['data/chunk_index']):03d}"
                / f"file-{int(item['data/file_index']):03d}.parquet"
            )
            duration = (int(item["length"]) - 1) / fps
            intervened = False
            try:
                columns = [
                    "timestamp",
                    "episode_index",
                    "complementary_info.is_intervention",
                ]
                frame = pd.read_parquet(data_file, columns=columns)
                frame = frame[frame["episode_index"] == index]
                if not frame.empty:
                    duration = float(frame["timestamp"].max() - frame["timestamp"].min())
                    intervened = bool(
                        (frame["complementary_info.is_intervention"].astype(float) != 0).any()
                    )
            except (ImportError, KeyError, OSError, ValueError) as exc:
                # Metadata duration remains valid when optional frame columns are unavailable.
                logger.debug(
                    "cannot read optional episode frame metadata from %s: %s",
                    data_file,
                    exc,
                )
            outcome = str(item["episode_success"]).strip().lower()
            notes = "时长低于 1s" if outcome == "success" and duration < 1 else ""
            rows.append(
                {
                    "session_id": session["session_id"],
                    "episode_index": str(index),
                    "episode_path": "",
                    "trajectory_path": str(data_file),
                    "t_start": "0.000",
                    "t_end": f"{duration:.3f}",
                    "duration_s": f"{duration:.3f}",
                    "outcome": outcome,
                    "operator_intervened": str(intervened).lower(),
                    "notes": notes,
                }
            )
    return sorted(rows, key=lambda row: int(row["episode_index"]))

def write_json(path: Path, value: dict[str, Any]) -> None:
    """按文档要求以 UTF-8、两空格缩进写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def prepare_output_dir(output_dir: Path | None = None) -> Path:
    """解析并创建输出目录。"""
    default_dir = f"{DEFAULT_OUTPUT_DIR}_{datetime.now().astimezone().strftime('%Y%m%d')}"
    root = (output_dir or Path.cwd() / default_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

def parse_session_args(
    argv: Sequence[str] | None, description: str
) -> argparse.Namespace:
    """解析这组 CLI 共同的 session_dir / --output-dir 参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)

def require_session_dir(session_dir: Path) -> Path:
    """解析 session 目录，并确认它存在。"""
    root = session_dir.expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError(f"session directory does not exist: {root}")
    if not (root / "session.json").is_file() and not _is_lerobot_dataset(root):
        children = [path for path in root.iterdir() if path.is_dir()]
        lerobot_children = [path for path in children if _is_lerobot_dataset(path)]
        if len(lerobot_children) == 1:
            return lerobot_children[0]
    return root

def resolve_path(raw: str | Path, base_dir: Path) -> Path:
    """同时支持相对 session 目录和绝对路径。"""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else base_dir / path

def _read_csv(path: Path, fields: Iterable[str]) -> list[dict[str, str]]:
    """读取 CSV 并校验表头。"""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_fields(reader.fieldnames or (), fields, path.name)
            return list(reader)
    except FileNotFoundError as exc:
        raise EvaluationError(f"missing required file: {path}") from exc

def load_session(session_dir: Path) -> dict[str, Any]:
    """读取并校验 session.json 的必填字段和枚举值。"""
    session_path = session_dir / "session.json"
    session = (
        read_json(session_path)
        if session_path.is_file()
        else _synthesize_lerobot_session(session_dir)
        if _is_lerobot_dataset(session_dir)
        else read_json(session_path)
    )
    _require_fields(session, SESSION_FIELDS, "session.json")
    has_dataset_root = bool(str(session.get("dataset_root", "")).strip())
    records_dataset = session.get("record_dataset", True) is not False
    has_trajectory_source = bool(str(session.get("trajectory_log_dir", "")).strip())
    if records_dataset and not has_dataset_root and not has_trajectory_source:
        _require_fields(session, SESSION_DATASET_FIELDS, "session.json")
    if session["schema_version"] != SCHEMA_VERSION:
        raise EvaluationError("session.json schema_version must be '1.0'")
    if session["status"] not in {"recording", "completed", "aborted"}:
        raise EvaluationError(f"invalid session status: {session['status']!r}")
    if session["rollout_mode"] not in {"ee", "pi05", "default"}:
        raise EvaluationError(f"invalid rollout_mode: {session['rollout_mode']!r}")
    if session["dataset_backend"] not in {"lerobot", "native"}:
        raise EvaluationError(f"invalid dataset_backend: {session['dataset_backend']!r}")
    if finite_float(session["fps"], "session.fps") <= 0:
        raise EvaluationError("session.fps must be greater than zero")
    target = session["num_episodes_target"]
    if not isinstance(target, int) or isinstance(target, bool) or target < 0:
        raise EvaluationError("num_episodes_target must be a non-negative integer")
    return session

def load_episodes(
    session_dir: Path, session: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """读取 episodes.csv，并校验文档规定的逐行约束。"""
    session = session or load_session(session_dir)
    episodes_path = session_dir / "episodes.csv"
    rows = (
        _read_csv(episodes_path, EPISODE_FIELDS)
        if episodes_path.is_file()
        else _synthesize_lerobot_episodes(session_dir, session)
        if session["dataset_backend"] == "lerobot" and _is_lerobot_dataset(session_dir)
        else _read_csv(episodes_path, EPISODE_FIELDS)
    )
    seen: set[int] = set()
    for line, row in enumerate(rows, 2):
        prefix = f"episodes.csv row {line}"
        try:
            index = int(row["episode_index"])
        except ValueError as exc:
            raise EvaluationError(f"{prefix}: invalid episode_index") from exc
        if row["session_id"] != session["session_id"] or index < 0 or index in seen:
            raise EvaluationError(f"{prefix}: invalid session_id or episode_index")
        seen.add(index)
        has_episode_path = bool(row["episode_path"].strip())
        has_trajectory_path = bool(row["trajectory_path"].strip())
        has_session_trajectory = bool(str(session.get("trajectory_log_dir", "")).strip())
        if not (has_episode_path or has_trajectory_path or has_session_trajectory):
            raise EvaluationError(
                f"{prefix}: episode_path or trajectory_path is required"
            )
        if row["outcome"].strip().lower() not in {"success", "failure"}:
            raise EvaluationError(f"{prefix}: invalid outcome")
        start = finite_float(row["t_start"], f"{prefix} t_start")
        end = finite_float(row["t_end"], f"{prefix} t_end")
        duration = finite_float(row["duration_s"], f"{prefix} duration_s")
        if duration < 0 or end < start or abs(end - start - duration) > 0.0015:
            raise EvaluationError(f"{prefix}: invalid timestamps or duration_s")
        if row["operator_intervened"].strip().lower() not in {"true", "false"}:
            raise EvaluationError(f"{prefix}: invalid operator_intervened")
    return rows

def load_episode_metrics(
    session_dir: Path, session: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """读取 episode_metrics.csv，并把可选数值字段转换为 Python 类型。"""
    session = session or load_session(session_dir)
    raw_rows = _read_csv(session_dir / "episode_metrics.csv", EPISODE_METRIC_FIELDS)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line, raw in enumerate(raw_rows, 2):
        prefix = f"episode_metrics.csv row {line}"
        try:
            index = int(raw["episode_index"])
            frames = int(raw["smoothness_frames"]) if raw["smoothness_frames"] else None
        except ValueError as exc:
            raise EvaluationError(f"{prefix}: invalid integer field") from exc
        if raw["session_id"] != session["session_id"] or index < 0 or index in seen:
            raise EvaluationError(f"{prefix}: invalid session_id or episode_index")
        seen.add(index)
        outcome = raw["outcome"].strip().lower()
        space = raw["smoothness_space"].strip()
        if outcome not in {"success", "failure"} or space not in {"", "ee_xyz", "joint"}:
            raise EvaluationError(f"{prefix}: invalid outcome or smoothness_space")
        rows.append(
            {
                "session_id": raw["session_id"],
                "episode_index": index,
                "outcome": outcome,
                "duration_s": finite_float(raw["duration_s"], f"{prefix} duration_s")
                if raw["duration_s"]
                else None,
                "smoothness": finite_float(
                    raw["smoothness"], f"{prefix} smoothness"
                )
                if raw["smoothness"]
                else None,
                "left_smoothness": finite_float(
                    raw["left_smoothness"], f"{prefix} left_smoothness"
                )
                if raw["left_smoothness"]
                else None,
                "right_smoothness": finite_float(
                    raw["right_smoothness"], f"{prefix} right_smoothness"
                )
                if raw["right_smoothness"]
                else None,
                "smoothness_space": space,
                "smoothness_frames": frames,
                "smoothness_skipped_reason": raw["smoothness_skipped_reason"].strip(),
            }
        )
    return rows

def load_metrics_core(
    session_dir: Path, session: dict[str, Any] | None = None
) -> dict[str, Any]:
    """读取 report.md 所需的 metrics_core.json。"""
    session = session or load_session(session_dir)
    metrics = read_json(session_dir / "metrics_core.json")
    _require_fields(metrics, CORE_FIELDS, "metrics_core.json")
    if metrics["schema_version"] != SCHEMA_VERSION:
        raise EvaluationError("metrics_core.json schema_version must be '1.0'")
    if metrics["session_id"] != session["session_id"]:
        raise EvaluationError("metrics_core.json session_id does not match session.json")
    if not isinstance(metrics["smoothness"], dict):
        raise EvaluationError("metrics_core.smoothness must be an object")
    _require_fields(metrics["smoothness"], SMOOTHNESS_FIELDS, "metrics_core.smoothness")
    for side in ("left", "right"):
        _require_fields(
            metrics["smoothness"][side],
            SMOOTHNESS_SUMMARY_FIELDS,
            f"metrics_core.smoothness.{side}",
        )
    return metrics
