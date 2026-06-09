"""Unit tests for the cv2-free parts of video frame extraction: folder scanning,
frame-step math (including the NaN/0 fps fallback), and frame filenames. The cv2
decode path is not unit-tested — it needs OpenCV and real video files."""

import math

from src.video_io import (
    DEFAULT_FPS,
    frame_filename,
    frame_step,
    scan_video_folder,
)


# --- scan_video_folder --------------------------------------------------------

def test_scan_video_folder_missing_returns_empty(tmp_path):
    assert scan_video_folder(tmp_path / "nope") == []


def test_scan_video_folder_filters_and_sorts(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.mov").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "pic.jpg").write_bytes(b"")  # image, not video
    got = [p.name for p in scan_video_folder(tmp_path)]
    assert got == ["a.mov", "b.mp4"]


def test_scan_video_folder_extension_case_insensitive(tmp_path):
    (tmp_path / "CLIP.MP4").write_bytes(b"")
    assert [p.name for p in scan_video_folder(tmp_path)] == ["CLIP.MP4"]


# --- frame_step ---------------------------------------------------------------

def test_frame_step_one_second_at_30fps():
    assert frame_step(30.0, 1.0) == 30


def test_frame_step_half_second_rounds():
    assert frame_step(25.0, 0.5) == 12  # 12.5 -> 12 (banker's rounding)


def test_frame_step_sub_frame_interval_floors_to_one():
    # 30fps, 0.01s -> 0.3 frames -> clamped up to 1 (never skip everything)
    assert frame_step(30.0, 0.01) == 1


def test_frame_step_zero_fps_uses_default():
    assert frame_step(0.0, 1.0) == round(DEFAULT_FPS * 1.0)


def test_frame_step_nan_fps_uses_default():
    # NaN <= 0 is False, so a naive guard would crash on round(nan). Verify the
    # isfinite guard kicks in instead.
    assert frame_step(math.nan, 2.0) == round(DEFAULT_FPS * 2.0)


def test_frame_step_negative_fps_uses_default():
    assert frame_step(-5.0, 1.0) == round(DEFAULT_FPS * 1.0)


def test_frame_step_zero_interval_returns_one():
    assert frame_step(30.0, 0.0) == 1


# --- frame_filename -----------------------------------------------------------

def test_frame_filename_zero_pads():
    assert frame_filename("clip", 1) == "clip_frame0001.jpg"


def test_frame_filename_large_ordinal():
    assert frame_filename("clip", 12345) == "clip_frame12345.jpg"


def test_frame_filename_custom_extension():
    assert frame_filename("beach", 7, ext=".png") == "beach_frame0007.png"
