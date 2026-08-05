import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vla_eval.datasets import DatasetKind, inspect_dataset

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


def _write_native_session(
    root: Path, *, trajectory_path: str = "trajectories/episode_000.npz"
) -> Path:
    (root / "trajectories").mkdir(parents=True)
    session = {
        "schema_version": "1.0",
        "session_id": "native-fixture",
        "created_at": "2026-01-02T03:04:05+08:00",
        "status": "completed",
        "rollout_config_path": "rollout.yaml",
        "rollout_mode": "default",
        "policy_path": "policy",
        "task": "fixture",
        "num_episodes_target": 1,
        "fps": 10,
        "dataset_backend": "native",
        "dataset_root": "unused",
    }
    (root / "session.json").write_text(json.dumps(session), encoding="utf-8")
    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "native-fixture",
                "episode_index": "0",
                "episode_path": "",
                "trajectory_path": trajectory_path,
                "t_start": "0",
                "t_end": "1",
                "duration_s": "1",
                "outcome": "success",
                "operator_intervened": "false",
                "notes": "",
            }
        )
    trajectory = root / trajectory_path if not Path(trajectory_path).is_absolute() else None
    if trajectory is not None:
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez(trajectory, action=np.arange(12, dtype=float).reshape(4, 3))
    return root


def _write_lerobot(root: Path, *, with_video_metadata: bool = False) -> Path:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info: dict[str, object] = {"total_episodes": 1, "fps": 30}
    episode_data: dict[str, list[object]] = {
        "episode_index": [0],
        "length": [1],
        "episode_success": ["success"],
        "data/chunk_index": [0],
        "data/file_index": [0],
    }
    if with_video_metadata:
        key = "observation.images.right_wrist"
        info["features"] = {key: {"dtype": "video", "shape": [3, 8, 8]}}
        episode_data.update(
            {
                f"videos/{key}/file_index": [0],
                f"videos/{key}/from_timestamp": [0.0],
                f"videos/{key}/to_timestamp": [0.1],
            }
        )
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame(episode_data).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame(
        {"episode_index": [0], "timestamp": [0.0], "action": [[0.0, 0.0, 0.0]]}
    ).to_parquet(root / "data/chunk-000/file-000.parquet")
    return root


def test_inspect_lerobot_dataset(tmp_path: Path):
    root = tmp_path / "run"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 1, "fps": 30}')
    pd.DataFrame(
        {
            "episode_index": [0],
            "length": [1],
            "episode_success": ["success"],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame(
        {"episode_index": [0], "timestamp": [0.0], "action": [[0.0, 0.0, 0.0]]}
    ).to_parquet(root / "data/chunk-000/file-000.parquet")
    result = inspect_dataset(root, allowed_root=tmp_path)
    assert result.kind is DatasetKind.LEROBOT
    assert result.ready is True
    assert len(result.fingerprint) == 64


def test_inspect_rejects_symlink_outside_allowed_root(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "session.json").write_text("{}")
    (root / "leak").symlink_to(Path("/etc/passwd"))
    result = inspect_dataset(root, allowed_root=tmp_path)
    assert result.ready is False
    assert "outside allowed root" in result.errors[0]


def test_inspect_genie02_session(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.kind is DatasetKind.GENIE02_SESSION
    assert result.ready is True
    assert result.episode_count == 1
    assert result.size_bytes > 0


def test_inspect_unknown_and_incomplete_formats(tmp_path: Path):
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "notes.txt").write_text("not a dataset")
    unknown_result = inspect_dataset(unknown, allowed_root=tmp_path)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "session.json").write_text("{}")
    incomplete_result = inspect_dataset(incomplete, allowed_root=tmp_path)

    assert unknown_result.kind is None
    assert unknown_result.ready is False
    assert "unknown dataset format" in unknown_result.errors
    assert incomplete_result.kind is DatasetKind.GENIE02_SESSION
    assert incomplete_result.ready is False
    assert any("episodes.csv" in error for error in incomplete_result.errors)
    assert len(incomplete_result.fingerprint) == 64


def test_inspect_reports_invalid_json_and_parquet_metadata(tmp_path: Path):
    root = tmp_path / "run"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text("not-json")
    (root / "meta/episodes/chunk-000/file-000.parquet").write_bytes(b"not-parquet")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.kind is DatasetKind.LEROBOT
    assert result.ready is False
    assert any("invalid JSON" in error for error in result.errors)
    assert any("cannot read" in error and "parquet" in error for error in result.errors)


def test_inspect_accepts_internal_file_symlink_without_double_counting(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    target = root / "trajectories/episode_000.npz"
    alias = root / "trajectory-link.npz"
    alias.symlink_to(target.relative_to(root))
    episodes = (root / "episodes.csv").read_text(encoding="utf-8")
    (root / "episodes.csv").write_text(
        episodes.replace("trajectories/episode_000.npz", "trajectory-link.npz"),
        encoding="utf-8",
    )

    result = inspect_dataset(root, allowed_root=tmp_path)
    expected_size = sum(
        path.stat().st_size for path in (root / "session.json", root / "episodes.csv", target)
    )

    assert result.ready is True
    assert result.size_bytes == expected_size


def test_inspect_rejects_symlink_loop(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    (root / "loop-a").symlink_to("loop-b")
    (root / "loop-b").symlink_to("loop-a")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("symlink" in error and "loop" in error for error in result.errors)


def test_metadata_hash_changes_fingerprint_when_size_and_mtime_do_not(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    info_path = root / "meta/info.json"
    original_stat = info_path.stat()
    first = inspect_dataset(root, allowed_root=tmp_path)
    info_path.write_text(info_path.read_text(encoding="utf-8").replace("30", "31"))
    os.utime(info_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = inspect_dataset(root, allowed_root=tmp_path)
    repeated = inspect_dataset(root, allowed_root=tmp_path)

    assert first.fingerprint != second.fingerprint
    assert second.fingerprint == repeated.fingerprint


def test_large_data_content_is_not_hashed_when_size_and_mtime_do_not_change(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run", with_video_metadata=True)
    video = root / "videos/observation.images.right_wrist/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"AAAA")
    original_stat = video.stat()
    first = inspect_dataset(root, allowed_root=tmp_path)
    video.write_bytes(b"BBBB")
    os.utime(video, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = inspect_dataset(root, allowed_root=tmp_path)

    assert first.ready is True
    assert second.ready is True
    assert first.fingerprint == second.fingerprint


def test_inspect_validates_referenced_lerobot_data_and_video(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run", with_video_metadata=True)
    missing_video = inspect_dataset(root, allowed_root=tmp_path)
    (root / "data/chunk-000/file-000.parquet").unlink()
    missing_both = inspect_dataset(root, allowed_root=tmp_path)

    assert missing_video.ready is False
    assert any("video" in error and "does not exist" in error for error in missing_video.errors)
    assert any("data" in error and "does not exist" in error for error in missing_both.errors)


def test_inspect_reconciles_lerobot_episode_length_with_data(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "timestamp": [0.0, 0.1],
            "action": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("contains 2 frames" in error for error in result.errors)


def test_inspect_rejects_genie02_trajectory_reference_outside_allowed_root(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    episodes = (root / "episodes.csv").read_text(encoding="utf-8")
    (root / "episodes.csv").write_text(
        episodes.replace("trajectories/episode_000.npz", "/etc/passwd"),
        encoding="utf-8",
    )

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("outside allowed root" in error for error in result.errors)


def test_inspect_reports_invalid_metadata_encoding_without_crashing(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    (root / "session.json").write_bytes(b"\xff")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("session.json" in error and "read" in error for error in result.errors)


def test_inspect_reports_missing_outside_reference_as_outside_allowed_root(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    episodes = (root / "episodes.csv").read_text(encoding="utf-8")
    (root / "episodes.csv").write_text(
        episodes.replace("trajectories/episode_000.npz", "/etc/vla-eval-definitely-missing.npz"),
        encoding="utf-8",
    )

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("outside allowed root" in error for error in result.errors)
