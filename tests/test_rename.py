"""Tests for the batch-rename utility — template formatting, plan validation,
and end-to-end behavior including cyclic swaps."""

import pytest
from PIL import Image

from src.rename import (
    apply_renames,
    format_name,
    plan_renames,
    run,
    select_files,
)


def _make_image(path):
    Image.new("RGB", (10, 10), "red").save(path)


# --- format_name -------------------------------------------------------------

def test_format_name_simple_counter():
    assert format_name("img_{n}{ext}", n=1, stem="x", ext=".jpg") == "img_1.jpg"


def test_format_name_zero_padded_counter():
    assert format_name("img_{n:03d}{ext}", n=5, stem="x", ext=".jpg") == "img_005.jpg"


def test_format_name_stem_token():
    assert format_name("{stem}_renamed{ext}", n=1, stem="photo", ext=".png") == "photo_renamed.png"


def test_format_name_all_tokens():
    assert format_name("{stem}_{n:02d}{ext}", n=7, stem="img", ext=".jpg") == "img_07.jpg"


def test_format_name_unknown_token_raises():
    with pytest.raises(ValueError, match="Unknown template token"):
        format_name("foo_{bogus}", n=1, stem="x", ext=".jpg")


# --- select_files ------------------------------------------------------------

def test_select_files_filters_to_image_extensions(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.png")
    (tmp_path / "c.txt").write_text("not an image")
    files = select_files(tmp_path)
    assert sorted(f.name for f in files) == ["a.jpg", "b.png"]


def test_select_files_glob_match(tmp_path):
    _make_image(tmp_path / "IMG_001.jpg")
    _make_image(tmp_path / "IMG_002.jpg")
    _make_image(tmp_path / "DSC_001.jpg")
    files = select_files(tmp_path, match="IMG_*.jpg")
    assert sorted(f.name for f in files) == ["IMG_001.jpg", "IMG_002.jpg"]


def test_select_files_sort_by_name(tmp_path):
    for n in ("c.jpg", "a.jpg", "b.jpg"):
        _make_image(tmp_path / n)
    files = select_files(tmp_path, sort="name")
    assert [f.name for f in files] == ["a.jpg", "b.jpg", "c.jpg"]


def test_select_files_missing_folder_returns_empty(tmp_path):
    assert select_files(tmp_path / "does_not_exist") == []


# --- plan_renames ------------------------------------------------------------

def test_plan_renames_builds_sequential_targets(tmp_path):
    paths = []
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        p = tmp_path / n
        _make_image(p)
        paths.append(p)
    plan = plan_renames(paths, "vacation_{n:02d}{ext}")
    assert [d.name for _, d in plan] == [
        "vacation_01.jpg", "vacation_02.jpg", "vacation_03.jpg",
    ]


def test_plan_renames_respects_start_value(tmp_path):
    a = tmp_path / "a.jpg"; _make_image(a)
    b = tmp_path / "b.jpg"; _make_image(b)
    plan = plan_renames([a, b], "img_{n}{ext}", start=100)
    assert [d.name for _, d in plan] == ["img_100.jpg", "img_101.jpg"]


def test_plan_renames_detects_duplicate_destinations(tmp_path):
    a = tmp_path / "a.jpg"; _make_image(a)
    b = tmp_path / "b.jpg"; _make_image(b)
    # Constant template — both sources collide on the same destination.
    with pytest.raises(ValueError, match="duplicate name"):
        plan_renames([a, b], "constant.jpg")


def test_plan_renames_refuses_to_overwrite_unrelated_file(tmp_path):
    a = tmp_path / "a.jpg"; _make_image(a)
    # An unrelated file already occupies the target name.
    _make_image(tmp_path / "vacation_01.jpg")
    with pytest.raises(ValueError, match="already exists"):
        plan_renames([a], "vacation_{n:02d}{ext}")


def test_plan_renames_empty_input_returns_empty():
    assert plan_renames([], "img_{n}{ext}") == []


# --- apply_renames / end-to-end ---------------------------------------------

def test_apply_renames_executes_plan(tmp_path):
    a = tmp_path / "a.jpg"; _make_image(a)
    b = tmp_path / "b.jpg"; _make_image(b)
    plan = plan_renames([a, b], "out_{n}{ext}")
    n = apply_renames(plan)
    assert n == 2
    assert not a.exists() and not b.exists()
    assert (tmp_path / "out_1.jpg").exists()
    assert (tmp_path / "out_2.jpg").exists()


def test_apply_renames_skips_noop_when_src_equals_dst(tmp_path):
    a = tmp_path / "img_1.jpg"; _make_image(a)
    # Plan that maps the file to its own current name.
    n = apply_renames([(a, a)])
    assert n == 0
    assert a.exists()


def test_apply_renames_handles_cyclic_swap(tmp_path):
    # a.jpg and b.jpg swap names — the two-pass approach must handle this.
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"A-content")
    b.write_bytes(b"B-content")
    apply_renames([(a, b), (b, a)])
    # After the swap, the file *named* a.jpg should hold what was originally
    # in b.jpg, and vice versa.
    assert (tmp_path / "a.jpg").read_bytes() == b"B-content"
    assert (tmp_path / "b.jpg").read_bytes() == b"A-content"


def test_run_end_to_end(tmp_path):
    for n in ("a.jpg", "b.jpg"):
        _make_image(tmp_path / n)
    rc = run(folder=tmp_path, template="img_{n:02d}{ext}")
    assert rc == 0
    assert (tmp_path / "img_01.jpg").exists()
    assert (tmp_path / "img_02.jpg").exists()
    assert not (tmp_path / "a.jpg").exists()
    assert not (tmp_path / "b.jpg").exists()


def test_run_dry_run_makes_no_changes(tmp_path):
    a = tmp_path / "a.jpg"; _make_image(a)
    b = tmp_path / "b.jpg"; _make_image(b)
    rc = run(folder=tmp_path, template="x_{n}{ext}", dry_run=True)
    assert rc == 0
    assert a.exists() and b.exists()
    assert not (tmp_path / "x_1.jpg").exists()


def test_run_missing_template_errors():
    assert run(folder=".", template="") == 2


def test_run_collision_with_unrelated_file_aborts(tmp_path):
    # Only "a.jpg" is in the rename set thanks to --match. "img_1.jpg" exists
    # but is *not* selected, so it counts as an unrelated file that the plan
    # must refuse to overwrite.
    a = tmp_path / "a.jpg"; _make_image(a)
    _make_image(tmp_path / "img_1.jpg")
    rc = run(folder=tmp_path, match="a.jpg", template="img_{n}{ext}")
    assert rc == 2
    assert a.exists()  # unchanged because the plan failed validation
    assert (tmp_path / "img_1.jpg").exists()
