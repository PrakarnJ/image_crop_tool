"""Tkinter Canvas widget that displays an image fitted to its size and lets the user
draw multiple crop rectangles by click-dragging. Rectangles are stored in the original
image's pixel coordinates, so a window resize doesn't drift them."""

import tkinter as tk
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageTk

from .geometry import (
    Rect,
    canvas_to_image_coords,
    fit_scale,
    image_to_canvas_coords,
)


class CanvasView(tk.Canvas):
    OUTLINE_COLOR = "#00FF88"
    OUTLINE_WIDTH = 2
    MIN_DRAG_PIXELS = 3  # ignore tiny accidental drags

    def __init__(self, master, on_change: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, bg="#222", highlightthickness=0, cursor="cross", **kwargs)
        self._on_change = on_change or (lambda: None)

        self.image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None  # held to dodge GC
        self.scale_factor: float = 1.0
        self.image_offset: Tuple[int, int] = (0, 0)
        self.image_rects: List[Rect] = []  # in ORIGINAL image pixel coords

        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_id: Optional[int] = None

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", lambda _e: self._redraw())

    # --- Public API -----------------------------------------------------------

    def set_image(self, image: Optional[Image.Image]) -> None:
        self.image = image
        self.image_rects = []
        self.cancel_drag()
        self._redraw()
        self._on_change()

    def clear(self) -> None:
        self.set_image(None)

    def undo_last(self) -> None:
        if self.image_rects:
            self.image_rects.pop()
            self._redraw()
            self._on_change()

    def clear_rects(self) -> None:
        if self.image_rects:
            self.image_rects = []
            self._redraw()
            self._on_change()

    def get_image_rects(self) -> List[Rect]:
        """Rectangles in original-image pixel coords, ready for PIL.Image.crop()."""
        return list(self.image_rects)

    def num_rects(self) -> int:
        return len(self.image_rects)

    def cancel_drag(self) -> None:
        if self._drag_id is not None:
            self.delete(self._drag_id)
        self._drag_id = None
        self._drag_start = None

    # --- Mouse handlers -------------------------------------------------------

    def _on_press(self, event):
        if self.image is None:
            return
        self._drag_start = (event.x, event.y)
        self._drag_id = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=self.OUTLINE_COLOR, width=self.OUTLINE_WIDTH,
        )

    def _on_drag(self, event):
        if self._drag_start is None or self._drag_id is None:
            return
        x0, y0 = self._drag_start
        self.coords(self._drag_id, x0, y0, event.x, event.y)

    def _on_release(self, event):
        if self._drag_start is None or self.image is None:
            self.cancel_drag()
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self.cancel_drag()

        if abs(x1 - x0) < self.MIN_DRAG_PIXELS or abs(y1 - y0) < self.MIN_DRAG_PIXELS:
            return

        # Convert canvas coords (relative to canvas) into image-local canvas coords
        # by subtracting the image's top-left offset, then map to original pixels.
        ox, oy = self.image_offset
        canvas_rect = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
        img_rect = canvas_to_image_coords(canvas_rect, self.scale_factor, self.image.size)

        if img_rect[2] - img_rect[0] <= 0 or img_rect[3] - img_rect[1] <= 0:
            return  # entirely outside the image, or zero-area after clamping

        self.image_rects.append(img_rect)
        self._redraw()
        self._on_change()

    # --- Rendering ------------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("all")
        if self.image is None:
            return

        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            return  # not yet laid out

        scale = fit_scale(self.image.size, (cw, ch))
        self.scale_factor = scale
        disp_w = max(1, int(round(self.image.width * scale)))
        disp_h = max(1, int(round(self.image.height * scale)))
        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2
        self.image_offset = (ox, oy)

        resized = self.image.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.create_image(ox, oy, anchor="nw", image=self._tk_image)

        for i, rect in enumerate(self.image_rects, start=1):
            cx1, cy1, cx2, cy2 = image_to_canvas_coords(rect, scale, (ox, oy))
            self.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=self.OUTLINE_COLOR, width=self.OUTLINE_WIDTH,
            )
            self.create_text(
                cx1 + 4, cy1 + 2, anchor="nw",
                text=str(i), fill=self.OUTLINE_COLOR,
                font=("Helvetica", 12, "bold"),
            )
