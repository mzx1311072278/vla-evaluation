from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int | None
    episode_success: bool | None
    video_file: Path
    video_file_rel: str
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class VideoColumns:
    episode_index: str
    length: str
    file_index: str
    from_timestamp: str
    to_timestamp: str


def _pick_column(columns: list[str], candidates: list[str], contains: list[str]) -> str:
    for name in candidates:
        if name in columns:
            return name

    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        hit = lowered.get(name.lower())
        if hit:
            return hit

    hits = [c for c in columns if all(part.lower() in c.lower() for part in contains)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return min(hits, key=len)
    raise ValueError(
        f"Cannot find column containing {contains}. Available columns:\n- " + "\n- ".join(columns)
    )


def resolve_video_columns(columns: list[str], image_key: str) -> VideoColumns:
    """Resolve every episode/video column accepted by the attempt evaluator."""
    video_prefix = f"videos/{image_key}"
    return VideoColumns(
        episode_index=_pick_column(columns, ["episode_index", "episode_idx"], ["episode"]),
        length=_pick_column(columns, ["length", "episode_length"], ["length"]),
        file_index=_pick_column(
            columns,
            [f"{video_prefix}/file_index", f"{image_key}/file_index"],
            [image_key, "file_index"],
        ),
        from_timestamp=_pick_column(
            columns,
            [f"{video_prefix}/from_timestamp", f"{image_key}/from_timestamp"],
            [image_key, "from_timestamp"],
        ),
        to_timestamp=_pick_column(
            columns,
            [f"{video_prefix}/to_timestamp", f"{image_key}/to_timestamp"],
            [image_key, "to_timestamp"],
        ),
    )


def _file_index(value: object) -> int:
    if pd.isna(value):
        raise ValueError("file_index is empty")
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if not match:
            raise ValueError(f"Cannot parse file_index from {value!r}")
        return int(match.group(1))
    return int(value)


def _video_path(dataset_root: Path, image_key: str, meta_chunk: str, file_index: int) -> Path:
    video_root = dataset_root / "videos" / image_key
    candidates = [
        video_root / meta_chunk / f"file-{file_index:03d}.mp4",
        video_root / f"chunk-{file_index // 1000:03d}" / f"file-{file_index % 1000:03d}.mp4",
        video_root / f"chunk-{file_index // 1000:03d}" / f"file-{file_index:03d}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(video_root.glob(f"**/file-{file_index:03d}.mp4"))
    return matches[0] if matches else candidates[0]


def read_episode_metadata(dataset_root: Path, image_key: str) -> list[EpisodeMeta]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"meta/episodes not found: {episodes_dir}")

    parquet_files = sorted(episodes_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files under: {episodes_dir}")

    frames = []
    for path in parquet_files:
        frame = pd.read_parquet(path)
        frame["__meta_chunk"] = path.parent.name
        frames.append(frame)
    meta = pd.concat(frames, ignore_index=True)
    columns = list(meta.columns)

    print("episode metadata columns:")
    for column in columns:
        print(f"  - {column}")

    video_columns = resolve_video_columns(columns, image_key)
    success_col = "episode_success" if "episode_success" in columns else None

    episodes = []
    for _, row in meta.iterrows():
        file_index = _file_index(row[video_columns.file_index])
        video_file = _video_path(dataset_root, image_key, str(row["__meta_chunk"]), file_index)
        success_value = (
            None if success_col is None else str(row[success_col]).strip().lower() == "success"
        )
        episodes.append(
            EpisodeMeta(
                episode_index=int(row[video_columns.episode_index]),
                length=(
                    None if pd.isna(row[video_columns.length]) else int(row[video_columns.length])
                ),
                episode_success=success_value,
                video_file=video_file,
                video_file_rel=(
                    video_file.relative_to(dataset_root).as_posix()
                    if video_file.is_relative_to(dataset_root)
                    else video_file.as_posix()
                ),
                from_timestamp=float(row[video_columns.from_timestamp]),
                to_timestamp=float(row[video_columns.to_timestamp]),
            )
        )
    return sorted(episodes, key=lambda item: item.episode_index)
