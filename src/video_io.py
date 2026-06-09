"""Video frame extraction: scan a folder of videos and dump frames as image files
at a fixed time interval. OpenCV (cv2) does the decoding and is an *optional*
dependency — imported lazily so the image-only GUI still runs without it.

Frames are read sequentially and every Nth one is kept (N derived from fps and the
requested interval). We deliberately avoid CAP_PROP_POS_FRAMES seeking: on most
codecs it snaps to the nearest keyframe and returns the wrong frame."""

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from .image_io import next_available_filename

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg",
}

# Used when a file reports no/garbage fps (some webm/streaming files report 0 or NaN).
# Without this we'd fall back to "keep every frame" and silently dump thousands of
# images from a single clip.
DEFAULT_FPS = 30.0


def scan_video_folder(folder) -> List[Path]:
    """Return a sorted list of video files in `folder`. Missing folder returns []."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def frame_step(fps: float, interval_sec: float) -> int:
    """How many frames to advance between saved frames so consecutive saves are
    ~`interval_sec` apart. Falls back to DEFAULT_FPS when `fps` is missing or
    non-finite (NaN/0/negative). Always returns at least 1. Pure — no cv2."""
    if not (math.isfinite(fps) and fps > 0):
        fps = DEFAULT_FPS
    if interval_sec <= 0:
        return 1
    return max(1, round(fps * interval_sec))


def frame_filename(stem: str, ordinal: int, ext: str = ".jpg", width: int = 4) -> str:
    """`<stem>_frame<NNNN><ext>`, e.g. ('clip', 1) -> 'clip_frame0001.jpg'."""
    return f"{stem}_frame{ordinal:0{width}d}{ext}"


def _import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as e:  # pragma: no cover - exercised only without cv2 installed
        raise RuntimeError(
            "OpenCV is required for video frame extraction. Install it with:\n"
            "    pip install opencv-python"
        ) from e


def extract_frames(
    video_path,
    output_folder,
    interval_sec: float = 1.0,
    image_ext: str = ".jpg",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> int:
    """Extract frames from one video into `output_folder` as image files, keeping
    one frame roughly every `interval_sec` seconds. Returns the number of frames
    written.

    `progress_cb(saved, est_total)` is called after each saved frame; `est_total`
    is a best-effort estimate (the file's frame count may be unreliable).
    `should_cancel()` is polled per frame — return True to stop early and return
    what's been written so far."""
    cv2 = _import_cv2()
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path.name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = frame_step(fps, interval_sec)
        est_total = (total + step - 1) // step if total > 0 else 0

        existing = {p.name for p in output_folder.iterdir() if p.is_file()}
        saved = 0
        frame_idx = 0
        while True:
            if should_cancel is not None and should_cancel():
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                # cv2 gives BGR; PIL expects RGB. Save through PIL for format/alpha
                # handling consistent with the rest of the tool.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                desired = frame_filename(video_path.stem, saved + 1, image_ext)
                final = next_available_filename(desired, existing)
                existing.add(final)
                img.save(output_folder / final)
                saved += 1
                if progress_cb is not None:
                    progress_cb(saved, max(est_total, saved))
            frame_idx += 1
        return saved
    finally:
        cap.release()


def extract_folder(
    input_folder,
    output_folder,
    interval_sec: float = 1.0,
    image_ext: str = ".jpg",
    progress_cb: Optional[Callable[[int, int, str, int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """Extract frames from every video in `input_folder`. Returns a summary dict:
    {videos, frames, per_video: {name: count}, failed: [(name, error)], cancelled}.

    `progress_cb(video_index, video_total, video_name, saved, est_total)` is
    forwarded for each saved frame so a UI can show overall + per-video progress.
    A failure on one video is recorded and the batch continues."""
    videos = scan_video_folder(input_folder)
    results: Dict[str, object] = {
        "videos": 0,
        "frames": 0,
        "per_video": {},
        "failed": [],
        "cancelled": False,
    }
    total_videos = len(videos)
    for vi, video in enumerate(videos, start=1):
        if should_cancel is not None and should_cancel():
            results["cancelled"] = True
            break

        def _cb(saved: int, est_total: int, _name=video.name, _vi=vi):
            if progress_cb is not None:
                progress_cb(_vi, total_videos, _name, saved, est_total)

        try:
            n = extract_frames(
                video, output_folder, interval_sec, image_ext,
                progress_cb=_cb, should_cancel=should_cancel,
            )
            results["per_video"][video.name] = n  # type: ignore[index]
            results["frames"] = int(results["frames"]) + n
            results["videos"] = int(results["videos"]) + 1
        except Exception as e:  # keep the batch going if one file is broken
            results["failed"].append((video.name, str(e)))  # type: ignore[union-attr]
    return results
