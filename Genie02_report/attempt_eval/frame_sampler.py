from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def _sample_times(start: float, end: float, interval: float, max_frames: int) -> list[float]:
    if max_frames <= 0:
        return []
    duration = max(0.0, end - start)
    if duration == 0:
        return [start]
    if max_frames == 1:
        return [start + duration / 2]
    interval = max(interval, 0.001)
    times = []
    current = start
    while current <= end + 1e-6:
        times.append(current)
        current += interval
    if not times or end - times[-1] >= interval * 0.5:
        times.append(end)
    if len(times) > max_frames:
        last = len(times) - 1
        indexes = sorted({round(i * last / (max_frames - 1)) for i in range(max_frames)})
        times = [times[i] for i in indexes]
    if len(times) < min(3, max_frames):
        extras = [start, start + duration / 2, end]
        times = sorted({round(t, 6) for t in times + extras if start <= t <= end})[:max_frames]
    return times


def _resize(image: Image.Image, max_image_size: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_image_size / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def _dense_start(from_timestamp: float, to_timestamp: float, dense_region: str) -> float:
    duration = max(0.0, to_timestamp - from_timestamp)
    if duration < 10 or dense_region == "full":
        return from_timestamp
    if dense_region == "last_half":
        return from_timestamp + duration / 2
    return from_timestamp + duration * 2 / 3


def _sample_episode_frames_with_pyav(
    video_file: Path,
    output_dir: Path,
    from_timestamp: float,
    to_timestamp: float,
    sample_interval: float,
    max_frames: int,
    max_image_size: int,
) -> list[tuple[Path, float]]:
    import av

    target_times = _sample_times(from_timestamp, to_timestamp, sample_interval, max_frames)
    saved: list[tuple[Path, float]] = []
    target_i = 0
    with av.open(str(video_file)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            while target_i < len(target_times) and frame.time >= target_times[target_i]:
                image = _resize(frame.to_image().convert("RGB"), max_image_size)
                out = output_dir / f"frame_{len(saved):03d}.jpg"
                image.save(out, quality=92)
                saved.append((out, target_times[target_i]))
                target_i += 1
            if target_i >= len(target_times):
                break
    return saved


def _sample_episode_frames_with_cv2(
    video_file: Path,
    output_dir: Path,
    from_timestamp: float,
    to_timestamp: float,
    sample_interval: float,
    max_frames: int,
    max_image_size: int,
) -> list[tuple[Path, float]]:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")

    saved: list[tuple[Path, float]] = []
    try:
        for timestamp in _sample_times(from_timestamp, to_timestamp, sample_interval, max_frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = _resize(Image.fromarray(frame), max_image_size)
            out = output_dir / f"frame_{len(saved):03d}.jpg"
            image.save(out, quality=92)
            saved.append((out, timestamp))
    finally:
        cap.release()
    return saved


def _sample_one_region(
    video_file: Path,
    output_dir: Path,
    from_timestamp: float,
    to_timestamp: float,
    sample_interval: float,
    max_frames: int,
    max_image_size: int,
) -> list[tuple[Path, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        saved = _sample_episode_frames_with_pyav(
            video_file,
            output_dir,
            from_timestamp,
            to_timestamp,
            sample_interval,
            max_frames,
            max_image_size,
        )
        if saved:
            return saved
    except ImportError:
        pass
    except Exception:
        pass
    saved = _sample_episode_frames_with_cv2(
        video_file,
        output_dir,
        from_timestamp,
        to_timestamp,
        sample_interval,
        max_frames,
        max_image_size,
    )
    if saved:
        return saved
    return _sample_episode_frames_with_pyav(
        video_file,
        output_dir,
        from_timestamp,
        to_timestamp,
        sample_interval,
        max_frames,
        max_image_size,
    )


def sample_episode_frames(
    video_file: Path,
    output_dir: Path,
    from_timestamp: float,
    to_timestamp: float,
    sample_interval: float | None = None,
    max_frames: int | None = None,
    max_image_size: int = 384,
    max_global_frames: int = 16,
    global_sample_interval: float = 2.0,
    max_dense_frames: int = 16,
    dense_sample_interval: float = 0.5,
    dense_region: str = "last_third",
) -> tuple[list[Path], list[dict[str, float | str]]]:
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    global_interval = sample_interval if sample_interval is not None else global_sample_interval
    global_limit = max_frames if max_frames is not None else max_global_frames
    global_samples = _sample_one_region(
        video_file,
        output_dir / "global",
        from_timestamp,
        to_timestamp,
        global_interval,
        global_limit,
        max_image_size,
    )
    dense_samples = _sample_one_region(
        video_file,
        output_dir / "dense",
        _dense_start(from_timestamp, to_timestamp, dense_region),
        to_timestamp,
        dense_sample_interval,
        max_dense_frames,
        max_image_size,
    )
    return _split_samples(
        [("global", *item) for item in global_samples] + [("dense", *item) for item in dense_samples],
        from_timestamp,
        output_dir,
    )


def _split_samples(
    samples: list[tuple[str, Path, float]],
    from_timestamp: float,
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, float | str]]]:
    paths = [path for _, path, _ in samples]
    timestamps = [
        {
            "frame": path.relative_to(output_dir).as_posix(),
            "frame_type": frame_type,
            "episode_time": round(video_time - from_timestamp, 4),
            "video_time": round(video_time, 4),
        }
        for frame_type, path, video_time in samples
    ]
    return paths, timestamps
