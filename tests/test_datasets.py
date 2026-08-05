import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vla_eval import datasets
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
    root: Path,
    *,
    trajectory_path: str = "trajectories/episode_000.npz",
    episode_path: str = "",
    dataset_backend: str = "native",
    create_trajectory: bool = True,
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
        "dataset_backend": dataset_backend,
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
                "episode_path": episode_path,
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
    if create_trajectory and trajectory is not None:
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez(trajectory, action=np.arange(12, dtype=float).reshape(4, 3))
    return root


def test_inspect_genie02_accepts_direct_parquet_trajectory(tmp_path: Path):
    trajectory_path = "recording/data/chunk-000/episode_000.parquet"
    root = _write_native_session(
        tmp_path / "run",
        trajectory_path=trajectory_path,
        create_trajectory=False,
    )
    (root / trajectory_path).parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0, 0.0, 0.0]] * 4,
        }
    ).to_parquet(root / trajectory_path)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is True


def test_inspect_genie02_falls_back_by_missing_trajectory_basename(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run",
        trajectory_path="stale/custom.npz",
        create_trajectory=False,
    )
    np.savez(root / "trajectories/custom.npz", action=np.ones((4, 3)))
    np.savez(root / "trajectories/episode_000.npz", action=np.array([object()]))

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is True


def test_inspect_genie02_rejects_object_npz_without_falling_back(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="bad.npz", create_trajectory=False
    )
    np.savez(root / "bad.npz", action=np.array([object()]))
    np.savez(root / "trajectories/episode_000.npz", action=np.ones((4, 3)))

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("cannot read" in error for error in result.errors)


def test_inspect_genie02_rejects_plain_csv_trajectory(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="trajectory.csv", create_trajectory=False
    )
    (root / "trajectory.csv").write_text("action\n0.0\n", encoding="utf-8")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("unsupported trajectory format" in error for error in result.errors)


def test_inspect_genie02_rejects_csv_named_symlink_to_npz(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="trajectory.csv", create_trajectory=False
    )
    target = root / "trajectories/source.npz"
    np.savez(target, action=np.ones((4, 3)))
    (root / "trajectory.csv").symlink_to(target)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("unsupported trajectory format" in error for error in result.errors)


def test_inspect_genie02_rejects_parquet_symlink_before_target_sidecar_read(tmp_path: Path):
    logical_path = "alias/data/chunk-000/file-000.parquet"
    root = _write_native_session(
        tmp_path / "run", trajectory_path=logical_path, create_trajectory=False
    )
    target = root / "target/data/chunk-000/file-000.parquet"
    target.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0, 0.0, 0.0]] * 4,
        }
    ).to_parquet(target)
    logical = root / logical_path
    logical.parent.mkdir(parents=True)
    logical.symlink_to(target)
    logical_info = root / "alias/meta/info.json"
    logical_info.parent.mkdir(parents=True)
    logical_info.write_text('{"features": {"action": {"names": []}}}', encoding="utf-8")
    target_info = root / "target/meta/info.json"
    target_info.parent.mkdir(parents=True)
    target_info.write_text("invalid-json", encoding="utf-8")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("parsed symlink files are not supported" in error for error in result.errors)


def test_inspect_rejects_shallow_parquet_symlink_with_reader_sidecar_escape(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="shallow.parquet", create_trajectory=False
    )
    target = root / "target/data/chunk-000/file-000.parquet"
    target.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0, 0.0, 0.0]] * 4,
        }
    ).to_parquet(target)
    target_info = root / "target/meta/info.json"
    target_info.parent.mkdir(parents=True)
    target_info.write_text('{"features": {}}', encoding="utf-8")
    (root / "shallow.parquet").symlink_to(target)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("trajectory metadata is outside allowed root" in error for error in result.errors)


def test_inspect_genie02_rejects_corrupt_npz(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="corrupt.npz", create_trajectory=False
    )
    (root / "corrupt.npz").write_bytes(b"not-a-zip-archive")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("cannot read" in error for error in result.errors)


def test_inspect_genie02_rejects_mismatched_npz_timestamps(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run", trajectory_path="mismatch.npz", create_trajectory=False
    )
    np.savez(root / "mismatch.npz", action=np.ones((4, 3)), timestamp=np.arange(3))

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("timestamp length" in error for error in result.errors)


def test_inspect_genie02_rejects_empty_parquet_trajectory(tmp_path: Path):
    trajectory_path = "recording/data/chunk-000/empty.parquet"
    root = _write_native_session(
        tmp_path / "run", trajectory_path=trajectory_path, create_trajectory=False
    )
    (root / trajectory_path).parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": pd.Series(dtype="int64"),
            "timestamp": pd.Series(dtype="float64"),
            "action": pd.Series(dtype="object"),
        }
    ).to_parquet(root / trajectory_path)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("absent" in error for error in result.errors)


def test_inspect_genie02_rejects_wrong_episode_parquet_trajectory(tmp_path: Path):
    trajectory_path = "recording/data/chunk-000/wrong.parquet"
    root = _write_native_session(
        tmp_path / "run", trajectory_path=trajectory_path, create_trajectory=False
    )
    (root / trajectory_path).parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [1, 1, 1, 1],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0, 0.0, 0.0]] * 4,
        }
    ).to_parquet(root / trajectory_path)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("episode 0 is absent" in error for error in result.errors)


def test_inspect_genie02_lerobot_backend_uses_episode_path_data(tmp_path: Path):
    root = _write_native_session(
        tmp_path / "run",
        trajectory_path="",
        episode_path="recording",
        dataset_backend="lerobot",
        create_trajectory=False,
    )
    data_path = root / "recording/data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0, 0.0, 0.0]] * 4,
        }
    ).to_parquet(data_path)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is True


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


def test_inspect_rejects_internal_symlink_for_parsed_trajectory(tmp_path: Path):
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

    assert result.ready is False
    assert any("symlink" in error for error in result.errors)
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


def test_manifest_hashes_only_adapter_metadata_independent_of_ancestors(
    tmp_path: Path, monkeypatch
):
    allowed_root = tmp_path / "meta/allowed"
    root = _write_lerobot(allowed_root / "run")
    unrelated = root / "unrelated.json"
    unrelated.write_text('{"same_size": true}', encoding="utf-8")
    unrelated_meta = root / "meta/unrelated.json"
    unrelated_meta.write_text('{"not_adapter_metadata": true}', encoding="utf-8")
    unrelated_csv = root / "unrelated.csv"
    unrelated_csv.write_text("not,adapter,metadata\n", encoding="utf-8")
    hashed: list[Path] = []
    original_hash_file = datasets._hash_file

    def record_hash(path: Path, *args) -> str:
        hashed.append(path.resolve())
        return original_hash_file(path, *args)

    monkeypatch.setattr(datasets, "_hash_file", record_hash)

    result = inspect_dataset(root, allowed_root=allowed_root)

    assert result.ready is True
    assert (root / "meta/info.json").resolve() in hashed
    assert (root / "meta/episodes/chunk-000/file-000.parquet").resolve() in hashed
    assert (root / "data/chunk-000/file-000.parquet").resolve() not in hashed
    assert unrelated.resolve() not in hashed
    assert unrelated_meta.resolve() not in hashed
    assert unrelated_csv.resolve() not in hashed


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


def test_fingerprint_is_stable_when_dataset_is_relocated(tmp_path: Path):
    source = _write_lerobot(tmp_path / "staging/run")
    destination = tmp_path / "inbox/run"
    shutil.copytree(source, destination, copy_function=shutil.copy2)

    source_result = inspect_dataset(source, allowed_root=tmp_path)
    destination_result = inspect_dataset(destination, allowed_root=tmp_path)

    assert source_result.ready is True
    assert destination_result.ready is True
    assert source_result.fingerprint == destination_result.fingerprint


def test_complete_genie_signature_wins_over_ordinary_data_directory(tmp_path: Path):
    root = _write_native_session(tmp_path / "run")
    (root / "data").mkdir()

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.kind is DatasetKind.GENIE02_SESSION
    assert result.ready is True


def test_dataset_with_both_complete_signatures_is_ambiguous(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    _write_native_session(root)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.kind is None
    assert result.ready is False
    assert any("ambiguous" in error for error in result.errors)


def test_dataset_root_outside_allowed_is_rejected_before_type_error(tmp_path: Path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("not a directory")

    result = inspect_dataset(outside_file, allowed_root=allowed_root)

    assert result.ready is False
    assert "outside allowed root" in result.errors[0]


def test_inspect_rejects_string_action_vectors(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    pd.DataFrame(
        {"episode_index": [0], "timestamp": [0.0], "action": [["not-numeric"]]}
    ).to_parquet(root / "data/chunk-000/file-000.parquet")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("numeric action" in error for error in result.errors)


def test_inspect_rejects_non_finite_lerobot_timestamps(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    pd.DataFrame(
        {"episode_index": [0], "timestamp": [float("nan")], "action": [[0.0, 0.0]]}
    ).to_parquet(root / "data/chunk-000/file-000.parquet")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("finite numeric timestamp" in error for error in result.errors)


def test_inspect_rejects_empty_lerobot_action_vectors(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run")
    pd.DataFrame({"episode_index": [0], "timestamp": [0.0], "action": [[]]}).to_parquet(
        root / "data/chunk-000/file-000.parquet"
    )

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is False
    assert any("nonempty action vector" in error for error in result.errors)


def test_shared_lerobot_data_file_is_validated_once(tmp_path: Path, monkeypatch):
    root = tmp_path / "run"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 100, "fps": 30}')
    indices = list(range(100))
    pd.DataFrame(
        {
            "episode_index": indices,
            "length": [1] * 100,
            "episode_success": ["success"] * 100,
            "data/chunk_index": [0] * 100,
            "data/file_index": [0] * 100,
        }
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame(
        {
            "episode_index": indices,
            "timestamp": [float(index) for index in indices],
            "action": [[0.0, 0.0, 0.0]] * 100,
            "unrelated": ["do-not-read"] * 100,
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet")
    calls = 0
    original_validate = datasets._validate_data_parquet

    def count_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(datasets, "_validate_data_parquet", count_validation)

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is True
    assert calls == 1


def test_video_column_resolver_supports_all_reader_schemas():
    from Genie02_report.attempt_eval.dataset_reader import resolve_video_columns

    key = "observation.images.right_wrist"
    schemas = (
        {
            "episode_index": "episode_index",
            "length": "length",
            "file_index": f"videos/{key}/file_index",
            "from_timestamp": f"videos/{key}/from_timestamp",
            "to_timestamp": f"videos/{key}/to_timestamp",
        },
        {
            "episode_index": "episode_idx",
            "length": "episode_length",
            "file_index": f"{key}/file_index",
            "from_timestamp": f"{key}/from_timestamp",
            "to_timestamp": f"{key}/to_timestamp",
        },
        {
            "episode_index": "my_episode_number",
            "length": "recording_length",
            "file_index": f"metadata/{key}/video_file_index",
            "from_timestamp": f"metadata/{key}/video_from_timestamp",
            "to_timestamp": f"metadata/{key}/video_to_timestamp",
        },
    )

    for expected in schemas:
        resolved = resolve_video_columns(list(expected.values()), key)
        assert resolved.episode_index == expected["episode_index"]
        assert resolved.length == expected["length"]
        assert resolved.file_index == expected["file_index"]
        assert resolved.from_timestamp == expected["from_timestamp"]
        assert resolved.to_timestamp == expected["to_timestamp"]


def test_inspect_lerobot_uses_shared_video_column_resolution(tmp_path: Path):
    root = _write_lerobot(tmp_path / "run", with_video_metadata=True)
    key = "observation.images.right_wrist"
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path).rename(
        columns={
            f"videos/{key}/file_index": f"{key}/file_index",
            f"videos/{key}/from_timestamp": f"{key}/from_timestamp",
            f"videos/{key}/to_timestamp": f"{key}/to_timestamp",
        }
    )
    metadata.to_parquet(metadata_path)
    video = root / f"videos/{key}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video-placeholder")

    result = inspect_dataset(root, allowed_root=tmp_path)

    assert result.ready is True


def test_symlink_swap_between_check_and_parse_is_rejected(tmp_path: Path, monkeypatch):
    root = _write_native_session(tmp_path / "run")
    trajectory = root / "trajectories/episode_000.npz"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.npz"
    np.savez(outside, action=np.ones((4, 3)))
    original_mark = datasets._mark_trajectory_sidecar_metadata
    swapped = False

    def swap_after_check(*args, **kwargs):
        nonlocal swapped
        result = original_mark(*args, **kwargs)
        if not swapped:
            trajectory.unlink()
            trajectory.symlink_to(outside)
            swapped = True
        return result

    monkeypatch.setattr(datasets, "_mark_trajectory_sidecar_metadata", swap_after_check)
    try:
        result = inspect_dataset(root, allowed_root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert result.ready is False
    assert any(
        "changed during inspection" in error or "symlink" in error for error in result.errors
    )
