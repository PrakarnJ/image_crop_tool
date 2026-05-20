"""Unit tests for coordinate conversion, rect normalization, and collision-safe naming."""

import pytest

from src.geometry import canvas_to_image_coords, fit_scale, normalize_rect
from src.image_io import next_available_filename


# --- normalize_rect: all four drag directions ---------------------------------

def test_normalize_rect_topleft_to_bottomright():
    assert normalize_rect(10, 20, 100, 200) == (10, 20, 100, 200)


def test_normalize_rect_bottomright_to_topleft():
    assert normalize_rect(100, 200, 10, 20) == (10, 20, 100, 200)


def test_normalize_rect_topright_to_bottomleft():
    assert normalize_rect(100, 20, 10, 200) == (10, 20, 100, 200)


def test_normalize_rect_bottomleft_to_topright():
    assert normalize_rect(10, 200, 100, 20) == (10, 20, 100, 200)


# --- canvas_to_image_coords ---------------------------------------------------

def test_canvas_to_image_coords_half_scale():
    # Canvas shows the image at 50% — original pixel = canvas / 0.5
    box = canvas_to_image_coords((50, 40, 250, 240), scale=0.5, image_size=(1000, 800))
    assert box == (100, 80, 500, 480)


def test_canvas_to_image_coords_quarter_scale():
    box = canvas_to_image_coords((0, 0, 100, 75), scale=0.25, image_size=(1000, 800))
    assert box == (0, 0, 400, 300)


def test_canvas_to_image_coords_unity_scale():
    box = canvas_to_image_coords((12, 34, 56, 78), scale=1.0, image_size=(200, 200))
    assert box == (12, 34, 56, 78)


def test_canvas_to_image_coords_clamps_to_image_bounds():
    box = canvas_to_image_coords((-50, -10, 2000, 2000), scale=1.0, image_size=(500, 400))
    assert box == (0, 0, 500, 400)


def test_canvas_to_image_coords_normalizes_reverse_drag():
    # Same rect, drawn bottom-right → top-left, must round-trip the same as left→right.
    box = canvas_to_image_coords((250, 240, 50, 40), scale=0.5, image_size=(1000, 800))
    assert box == (100, 80, 500, 480)


def test_canvas_to_image_coords_rounds_subpixel_drag():
    # scale=0.333... → 100/0.333 ≈ 300.3 rounds to 300
    box = canvas_to_image_coords((0, 0, 100, 50), scale=1 / 3, image_size=(1000, 1000))
    assert box == (0, 0, 300, 150)


# --- fit_scale ---------------------------------------------------------------

def test_fit_scale_landscape_image_width_limited():
    # 1000x500 in 800x600: 0.8 by width vs 1.2 by height → 0.8
    assert fit_scale((1000, 500), (800, 600)) == pytest.approx(0.8)


def test_fit_scale_portrait_image_height_limited():
    # 500x1000 in 800x600: 1.6 vs 0.6 → 0.6
    assert fit_scale((500, 1000), (800, 600)) == pytest.approx(0.6)


def test_fit_scale_handles_zero_dimensions():
    assert fit_scale((0, 0), (800, 600)) == 1.0
    assert fit_scale((100, 100), (0, 0)) == 1.0


# --- next_available_filename -------------------------------------------------

def test_next_available_filename_no_collision():
    assert next_available_filename("photo_crop1.jpg", []) == "photo_crop1.jpg"


def test_next_available_filename_one_collision():
    assert next_available_filename("photo_crop1.jpg", ["photo_crop1.jpg"]) == "photo_crop1_2.jpg"


def test_next_available_filename_chain_of_collisions():
    existing = ["photo_crop1.jpg", "photo_crop1_2.jpg", "photo_crop1_3.jpg"]
    assert next_available_filename("photo_crop1.jpg", existing) == "photo_crop1_4.jpg"


def test_next_available_filename_unrelated_existing_files():
    existing = ["other.png", "photo_crop2.jpg", "different_crop1.jpg"]
    assert next_available_filename("photo_crop1.jpg", existing) == "photo_crop1.jpg"


def test_next_available_filename_preserves_extension():
    assert next_available_filename("img.png", ["img.png"]) == "img_2.png"


def test_next_available_filename_no_extension():
    assert next_available_filename("file", ["file"]) == "file_2"
