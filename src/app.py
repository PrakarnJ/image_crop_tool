"""Tkinter application shell: toolbar, status bar, navigation, folder pickers, and
keyboard shortcuts. Delegates drawing/mouse work to CanvasView and file I/O to image_io."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional

from PIL import Image

from .canvas_view import CanvasView
from .image_io import load_image, save_crop, scan_input_folder


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multi-Crop Image Tool")
        self.root.geometry("1200x800")

        self.input_folder: Path = Path("./input").resolve()
        self.output_folder: Path = Path("./output").resolve()
        self.image_paths: List[Path] = []
        self.current_index: int = 0
        self.current_image: Optional[Image.Image] = None
        self.saved_for_current: bool = False

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_input()

    # --- UI construction ------------------------------------------------------

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        buttons = [
            ("Choose Input Folder", self.choose_input),
            ("Choose Output Folder", self.choose_output),
            ("Save Crops (s)", self.save_crops),
            ("Previous (p)", self.previous_image),
            ("Next (n)", self.next_image),
            ("Skip", self.skip_image),
            ("Undo (u)", self.undo),
            ("Clear (c)", self.clear_all),
        ]
        for label, cmd in buttons:
            tk.Button(toolbar, text=label, command=cmd).pack(side=tk.LEFT, padx=2, pady=2)

        self.canvas = CanvasView(self.root, on_change=self._update_status)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="No images loaded.")
        tk.Label(
            self.root, textvariable=self.status_var,
            anchor="w", bd=1, relief=tk.SUNKEN,
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_shortcuts(self) -> None:
        self.root.bind("<n>", lambda _e: self.next_image())
        self.root.bind("<p>", lambda _e: self.previous_image())
        self.root.bind("<s>", lambda _e: self.save_crops())
        self.root.bind("<u>", lambda _e: self.undo())
        self.root.bind("<c>", lambda _e: self.clear_all())
        self.root.bind("<Escape>", lambda _e: self.canvas.cancel_drag())

    # --- Folder management ----------------------------------------------------

    def choose_input(self) -> None:
        folder = filedialog.askdirectory(
            initialdir=str(self.input_folder) if self.input_folder.exists() else str(Path.cwd()),
            title="Choose input folder",
        )
        if folder:
            self.input_folder = Path(folder)
            self._refresh_input()

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(
            initialdir=str(self.output_folder) if self.output_folder.exists() else str(Path.cwd()),
            title="Choose output folder",
        )
        if folder:
            self.output_folder = Path(folder)
            self._update_status()

    def _refresh_input(self) -> None:
        self.image_paths = scan_input_folder(self.input_folder)
        self.current_index = 0
        self.current_image = None
        if self.image_paths:
            self._load_current()
        else:
            self.canvas.clear()
            self.status_var.set(f"No images in {self.input_folder}")

    # --- Image loading --------------------------------------------------------

    def _load_current(self) -> None:
        if not self.image_paths:
            return
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        path = self.image_paths[self.current_index]
        try:
            self.current_image = load_image(path)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load {path.name}:\n{e}")
            self.current_image = None
            return
        self.canvas.set_image(self.current_image)
        self.saved_for_current = False
        self._update_status()

    # --- Status bar -----------------------------------------------------------

    def _update_status(self) -> None:
        if not self.image_paths:
            self.status_var.set(f"No images in {self.input_folder}")
            return
        n = self.canvas.num_rects()
        path = self.image_paths[self.current_index]
        plural = "s" if n != 1 else ""
        self.status_var.set(
            f"image {self.current_index + 1} / {len(self.image_paths)} "
            f"— {path.name} — {n} crop{plural} pending"
        )

    # --- Actions --------------------------------------------------------------

    def save_crops(self, show_message: bool = True) -> int:
        if not self.image_paths or self.current_image is None:
            return 0
        rects = self.canvas.get_image_rects()
        if not rects:
            if show_message:
                messagebox.showinfo("No crops", "No rectangles drawn.")
            return 0
        src_path = self.image_paths[self.current_index]
        try:
            for i, box in enumerate(rects, start=1):
                save_crop(self.current_image, self.output_folder, src_path.name, i, box)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return 0
        self.saved_for_current = True
        count = len(rects)
        if show_message:
            plural = "s" if count != 1 else ""
            messagebox.showinfo(
                "Saved",
                f"Saved {count} crop{plural} to {self.output_folder}",
            )
        return count

    def next_image(self) -> None:
        if not self.image_paths:
            return
        if self.canvas.num_rects() > 0 and not self.saved_for_current:
            self.save_crops(show_message=False)
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_current()
        else:
            messagebox.showinfo("End", "This is the last image.")

    def previous_image(self) -> None:
        if not self.image_paths:
            return
        if self.canvas.num_rects() > 0 and not self.saved_for_current:
            if not messagebox.askyesno(
                "Unsaved crops",
                "You have unsaved crops on this image. Go back anyway?",
            ):
                return
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()
        else:
            messagebox.showinfo("Start", "This is the first image.")

    def skip_image(self) -> None:
        if not self.image_paths:
            return
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_current()
        else:
            messagebox.showinfo("End", "This is the last image.")

    def undo(self) -> None:
        self.canvas.undo_last()

    def clear_all(self) -> None:
        self.canvas.clear_rects()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
