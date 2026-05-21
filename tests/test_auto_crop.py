"""Unit tests for v2 auto-crop helpers: padding math, class-name parsing, and
the labeled-filename code path in image_io.save_crop."""

import pytest
from PIL import Image

from src.auto_crop import COCO_NAMES, pad_box, parse_classes
from src.image_io import save_crop


# --- pad_box: fractional padding ---------------------------------------------

def test_pad_box_fraction_grows_box_by_longer_side():
    # 100x100 bbox, 10% of longer side = 10px each side.
    assert pad_box((100, 100, 200, 200), 0.10, (500, 500)) == (90, 90, 210, 210)


def test_pad_box_zero_padding_is_identity():
    assert pad_box((100, 100, 200, 200), 0.0, (500, 500)) == (100, 100, 200, 200)


def test_pad_box_fraction_uses_longer_side():
    # 100x50 bbox — longer side is 100; 20% → 20px on both axes.
    assert pad_box((100, 100, 200, 150), 0.20, (500, 500)) == (80, 80, 220, 170)


# --- pad_box: pixel-suffix padding -------------------------------------------

def test_pad_box_pixels():
    assert pad_box((100, 100, 200, 200), "20px", (500, 500)) == (80, 80, 220, 220)


def test_pad_box_fractional_string_is_treated_as_fraction():
    # CLI passes the raw string; "0.10" without 'px' must behave like float 0.10.
    assert pad_box((100, 100, 200, 200), "0.10", (500, 500)) == (90, 90, 210, 210)


# --- pad_box: clamping -------------------------------------------------------

def test_pad_box_clamps_at_topleft_corner():
    assert pad_box((5, 5, 50, 50), "20px", (500, 500)) == (0, 0, 70, 70)


def test_pad_box_clamps_at_bottomright_corner():
    assert pad_box((400, 400, 495, 495), "20px", (500, 500)) == (380, 380, 500, 500)


# --- parse_classes -----------------------------------------------------------

def test_parse_classes_single_name():
    assert parse_classes("person") == [0]


def test_parse_classes_multiple_names():
    assert parse_classes("person,dog") == [0, 16]


def test_parse_classes_strips_whitespace():
    assert parse_classes(" person , dog ") == [0, 16]


def test_parse_classes_empty_means_all():
    assert parse_classes("") is None
    assert parse_classes(None) is None


def test_parse_classes_invalid_raises_with_helpful_message():
    with pytest.raises(ValueError, match="Unknown class"):
        parse_classes("person,wizard")


def test_coco_names_has_80_entries():
    assert len(COCO_NAMES) == 80


# --- save_crop with label (v2 filename scheme) -------------------------------

def test_save_crop_with_label_inserts_class_into_filename(tmp_path):
    img = Image.new("RGB", (100, 100), color="red")
    out = save_crop(img, tmp_path, "photo.jpg", 1, (10, 10, 50, 50), label="person")
    assert out.name == "photo_person_crop1.jpg"
    assert out.exists()


def test_save_crop_without_label_matches_v1_scheme(tmp_path):
    img = Image.new("RGB", (100, 100), color="red")
    out = save_crop(img, tmp_path, "photo.jpg", 1, (10, 10, 50, 50))
    assert out.name == "photo_crop1.jpg"
    assert out.exists()


def test_save_crop_with_label_collision_gets_numeric_suffix(tmp_path):
    img = Image.new("RGB", (100, 100), color="red")
    first = save_crop(img, tmp_path, "photo.jpg", 1, (10, 10, 50, 50), label="dog")
    second = save_crop(img, tmp_path, "photo.jpg", 1, (10, 10, 50, 50), label="dog")
    assert first.name == "photo_dog_crop1.jpg"
    assert second.name == "photo_dog_crop1_2.jpg"
