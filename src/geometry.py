"""Pure geometry helpers: canvas <-> image coordinate conversion and rect normalization.

Kept dependency-free so it can be unit-tested without a display.
"""

from typing import Tuple

Rect = Tuple[int, int, int, int]
FloatRect = Tuple[float, float, float, float]
Size = Tuple[int, int]


def normalize_rect(x1: float, y1: float, x2: float, y2: float) -> FloatRect:
    """Return (left, top, right, bottom) with left <= right and top <= bottom,
    regardless of which corner the drag started from."""
    left, right = (x1, x2) if x1 <= x2 else (x2, x1)
    top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
    return left, top, right, bottom


def fit_scale(image_size: Size, canvas_size: Size) -> float:
    """Scale factor to fit `image_size` into `canvas_size` while preserving aspect ratio."""
    img_w, img_h = image_size
    canvas_w, canvas_h = canvas_size
    if img_w <= 0 or img_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        return 1.0
    return min(canvas_w / img_w, canvas_h / img_h)


def canvas_to_image_coords(rect: FloatRect, scale: float, image_size: Size) -> Rect:
    """Convert a canvas-space rectangle (in image-local canvas coords — i.e. the image's
    top-left is (0, 0)) into integer pixel coords on the original image.

    - Normalizes the rect first, so any drag direction works.
    - Divides by `scale` (displayed / original) and rounds to integer pixels.
    - Clamps to image bounds so a drag past the edge yields a valid crop box.
    """
    left, top, right, bottom = normalize_rect(*rect)
    img_w, img_h = image_size
    s = scale if scale > 0 else 1.0
    left_i = int(round(left / s))
    top_i = int(round(top / s))
    right_i = int(round(right / s))
    bottom_i = int(round(bottom / s))
    left_i = max(0, min(left_i, img_w))
    right_i = max(0, min(right_i, img_w))
    top_i = max(0, min(top_i, img_h))
    bottom_i = max(0, min(bottom_i, img_h))
    return left_i, top_i, right_i, bottom_i


def image_to_canvas_coords(rect: Rect, scale: float, offset: Tuple[float, float]) -> FloatRect:
    """Inverse of canvas_to_image_coords (without clamping). `offset` is the (x, y) of
    the image's top-left on the canvas."""
    ox, oy = offset
    return (rect[0] * scale + ox, rect[1] * scale + oy,
            rect[2] * scale + ox, rect[3] * scale + oy)
