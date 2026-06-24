"""ColorAugmentTab — Tkinter tab for label-based color augmentation.

Reads existing YOLO .txt label files to locate object regions, then for each
source→target color mapping produces a full-image copy where only pixels inside
labeled boxes whose hue matches the source color are remapped to the target hue.
"""

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk


class _GalleryWindow(tk.Toplevel):
    """Scrollable thumbnail grid that populates in real-time as images are saved."""

    THUMB = 160
    COLS = 4
    PAD = 6

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("Color Augment — Results")
        self.geometry("740x520")
        self._photos: list = []  # hold refs so GC doesn't collect them
        self._count = 0
        self._cur_row = 0
        self._cur_col = 0
        self._build()

    def _build(self) -> None:
        self._status = ttk.Label(self, text="Waiting for results…", foreground="gray")
        self._status.pack(anchor="w", padx=8, pady=(6, 2))

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(container, bg="#2b2b2b", highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._grid = tk.Frame(self._canvas, bg="#2b2b2b")
        self._grid_id = self._canvas.create_window((0, 0), window=self._grid, anchor="nw")

        self._grid.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._grid_id, width=e.width),
        )
        # Mouse-wheel scrolling
        self._canvas.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._canvas.bind("<Button-4>", lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind("<Button-5>", lambda e: self._canvas.yview_scroll(1, "units"))

    def add_image(self, path: Path) -> None:
        """Add a thumbnail for `path`. Must be called from the main thread."""
        try:
            img = Image.open(path)
            img.thumbnail((self.THUMB, self.THUMB), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
        except Exception:
            return

        cell = tk.Frame(self._grid, bg="#2b2b2b", padx=self.PAD, pady=self.PAD)
        cell.grid(row=self._cur_row, column=self._cur_col, padx=2, pady=2)

        lbl = tk.Label(cell, image=photo, bg="#3c3f41", cursor="hand2", relief="flat", bd=1)
        lbl.image = photo
        lbl.pack()

        name = path.name
        short = name if len(name) <= 22 else name[:20] + "…"
        tk.Label(cell, text=short, bg="#2b2b2b", fg="#cccccc", font=("Courier", 8)).pack()

        lbl.bind("<Button-1>", lambda _e, p=path: self._open_full(p))

        self._photos.append(photo)
        self._count += 1
        self._status.config(text=f"{self._count} image(s) generated", foreground="#4caf50")

        self._cur_col += 1
        if self._cur_col >= self.COLS:
            self._cur_col = 0
            self._cur_row += 1

        # Auto-scroll to show latest thumbnail
        self._canvas.yview_moveto(1.0)

    def _open_full(self, path: Path) -> None:
        try:
            img = Image.open(path)
        except Exception:
            return

        popup = tk.Toplevel(self)
        popup.title(path.name)

        max_w, max_h = self.winfo_screenwidth() - 100, self.winfo_screenheight() - 100
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        frame = ttk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, width=img.width, height=img.height)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # keep ref


class _QueueHandler(logging.Handler):
    """Logging handler that enqueues formatted records for the UI to poll."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put_nowait(self.format(record))


class ColorAugmentTab(ttk.Frame):
    """A ttk.Frame presenting the colour-augmentation form inside a Notebook tab."""

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        notebook.add(self, text="Color Augment")

        self._log_q: queue.Queue = queue.Queue()
        self._img_q: queue.Queue = queue.Queue()
        self._gallery: _GalleryWindow | None = None

        self._build_ui()
        self._poll_log()
        self._poll_img()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True)

        form_frame = ttk.Frame(pane)
        log_frame = ttk.Frame(pane)
        pane.add(form_frame, weight=1)
        pane.add(log_frame, weight=2)

        self._build_form(form_frame)
        self._build_log(log_frame)

    def _build_form(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent, padding=8)
        form.pack(fill=tk.BOTH, expand=True)

        self._dataset_var = tk.StringVar(value="./dataset")
        self._output_var = tk.StringVar(value="./output")
        self._dry_run_var = tk.BooleanVar(value=False)
        self._color_mappings: list = []  # list of (src_rgb, dst_rgb) tuples
        self._tolerance_var = tk.IntVar(value=30)
        self._class_id_list: list = []  # parallel to listbox rows, holds int class IDs

        row = 0

        # --- Dataset folder -----------------------------------------------------
        ttk.Label(form, text="Dataset folder:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._dataset_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        folder_btn_frame = ttk.Frame(form)
        folder_btn_frame.grid(row=row, column=2, padx=(4, 0), pady=3, sticky="w")
        ttk.Button(folder_btn_frame, text="Browse…", command=self._browse_dataset).pack(side=tk.LEFT, padx=(0, 4))
        self._scan_btn = ttk.Button(folder_btn_frame, text="Scan Classes", command=self._scan_dataset)
        self._scan_btn.pack(side=tk.LEFT)
        ttk.Label(form, text="must contain train/  val/  and *.yaml", foreground="gray").grid(
            row=row + 1, column=1, sticky="w", pady=(0, 4)
        )
        row += 2

        # --- Classes ------------------------------------------------------------
        ttk.Label(form, text="Classes:").grid(
            row=row, column=0, sticky="ne", padx=(0, 4), pady=3
        )
        class_list_frame = ttk.Frame(form)
        class_list_frame.grid(row=row, column=1, sticky="nsew", pady=3)
        self._class_lb = tk.Listbox(
            class_list_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False
        )
        class_vsb = ttk.Scrollbar(class_list_frame, orient="vertical", command=self._class_lb.yview)
        self._class_lb.config(yscrollcommand=class_vsb.set)
        self._class_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        class_ctrl = ttk.Frame(form)
        class_ctrl.grid(row=row, column=2, sticky="nw", padx=(4, 0), pady=3)
        ttk.Button(class_ctrl, text="All", width=6, command=self._select_all_classes).pack(anchor="w")
        ttk.Button(class_ctrl, text="Clear", width=6, command=self._clear_classes).pack(anchor="w", pady=(2, 6))
        self._class_status = ttk.Label(
            class_ctrl, text="Click\n'Scan Classes'", foreground="gray", wraplength=110
        )
        self._class_status.pack(anchor="w")
        classes_row = row
        row += 1

        # --- Output folder ------------------------------------------------------
        ttk.Label(form, text="Output folder:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._output_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(form, text="Browse…", command=self._browse_output).grid(
            row=row, column=2, padx=(4, 0), pady=3
        )
        ttk.Label(form, text="train/  val/  and *.yaml written here", foreground="gray").grid(
            row=row + 1, column=1, sticky="w", pady=(0, 4)
        )
        row += 2

        # --- Color mappings -------------------------------------------------------
        ttk.Label(form, text="Color mappings:").grid(
            row=row, column=0, sticky="ne", padx=(0, 4), pady=6
        )
        mapping_outer = ttk.Frame(form)
        mapping_outer.grid(row=row, column=1, sticky="nsew", pady=3)
        ttk.Button(mapping_outer, text="+ Add Mapping", command=self._add_mapping).pack(anchor="w", pady=(0, 4))
        self._mappings_frame = tk.Frame(mapping_outer, bg="#2b2b2b")
        self._mappings_frame.pack(fill=tk.X)

        ttk.Label(form, text="click swatch\nto change color", foreground="gray").grid(
            row=row, column=2, sticky="nw", padx=(4, 0), pady=3
        )
        mappings_row = row
        row += 1

        # --- Tolerance ----------------------------------------------------------
        ttk.Label(form, text="Tolerance:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        tol_frame = ttk.Frame(form)
        tol_frame.grid(row=row, column=1, sticky="ew", pady=3)
        self._tol_label = ttk.Label(tol_frame, text="30°", width=5)
        self._tol_label.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Scale(
            tol_frame,
            from_=5,
            to=90,
            variable=self._tolerance_var,
            orient="horizontal",
            command=lambda v: self._tol_label.config(text=f"{int(float(v))}°"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(form, text="hue range to replace", foreground="gray").grid(
            row=row, column=2, sticky="w", padx=(4, 0), pady=3
        )
        row += 1

        # --- Dry run ------------------------------------------------------------
        ttk.Checkbutton(
            form,
            text="Dry run (preview only, no files written)",
            variable=self._dry_run_var,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        # --- Run button ---------------------------------------------------------
        self._run_btn = ttk.Button(
            form, text="Run Color Augment", command=self._on_run
        )
        self._run_btn.grid(row=row, column=1, sticky="w", pady=(8, 4))

        form.columnconfigure(1, weight=1)
        form.rowconfigure(classes_row, weight=1)
        form.rowconfigure(mappings_row, weight=1)

    def _build_log(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Log:").pack(anchor="w", padx=4)
        self._log_text = ScrolledText(
            parent, state="disabled", height=12, font=("Courier", 10)
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------ Browse

    def _browse_dataset(self) -> None:
        current = self._dataset_var.get().strip()
        initial = str(Path(current).resolve()) if current else str(Path.cwd())
        folder = filedialog.askdirectory(initialdir=initial, title="Choose dataset folder")
        if folder:
            self._dataset_var.set(folder)

    # ------------------------------------------------------------------ Class scan

    def _scan_dataset(self) -> None:
        dataset_folder = self._dataset_var.get().strip() or "./dataset"
        self._scan_btn.config(state="disabled")
        self._class_status.config(text="Scanning…", foreground="gray")
        self._class_lb.delete(0, "end")
        self._class_id_list = []
        threading.Thread(target=self._scan_worker, args=(dataset_folder,), daemon=True).start()

    def _scan_worker(self, dataset_folder: str) -> None:
        try:
            from .color_augment import scan_dataset_classes
            class_map = scan_dataset_classes(dataset_folder)
            self.after(0, lambda: self._populate_classes(class_map))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.after(0, lambda: self._class_status.config(text=f"Scan failed:\n{msg}", foreground="red"))
            self.after(0, lambda: self._scan_btn.config(state="normal"))

    def _populate_classes(self, class_map: dict) -> None:
        self._class_id_list = sorted(class_map.keys())
        self._class_lb.delete(0, "end")
        for cls_id in self._class_id_list:
            self._class_lb.insert("end", f"{cls_id}: {class_map[cls_id]}")
        if self._class_id_list:
            self._class_lb.select_set(0, "end")  # default: all selected
            self._class_status.config(
                text=f"{len(self._class_id_list)} class(es)\n(empty sel. = all)",
                foreground="#4caf50",
            )
        else:
            self._class_status.config(text="No labels found", foreground="orange")
        self._scan_btn.config(state="normal")

    def _select_all_classes(self) -> None:
        self._class_lb.select_set(0, "end")

    def _clear_classes(self) -> None:
        self._class_lb.selection_clear(0, "end")

    def _browse_output(self) -> None:
        current = self._output_var.get().strip()
        initial = str(Path(current).resolve()) if current else str(Path.cwd())
        folder = filedialog.askdirectory(initialdir=initial, title="Choose output folder")
        if folder:
            self._output_var.set(folder)

    # ------------------------------------------------------------------ Color picker

    @staticmethod
    def _rgb_to_hue(r: int, g: int, b: int) -> int:
        import colorsys
        h, _s, _v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return round(h * 360) % 360

    def _rebuild_mappings(self) -> None:
        for w in self._mappings_frame.winfo_children():
            w.destroy()
        for idx, (src_rgb, dst_rgb) in enumerate(self._color_mappings):
            self._make_mapping_row(idx, src_rgb, dst_rgb)

    def _make_mapping_row(self, idx: int, src_rgb: tuple, dst_rgb: tuple) -> None:
        def swatch(parent, rgb, click_cmd):
            r, g, b = rgb
            hue = self._rgb_to_hue(r, g, b)
            cell = tk.Frame(parent, bg="#2b2b2b")
            cell.pack(side=tk.LEFT, padx=(0, 4))
            btn = tk.Label(cell, bg=f"#{r:02x}{g:02x}{b:02x}", width=4, height=2,
                           relief="raised", bd=2, cursor="hand2")
            btn.pack()
            btn.bind("<Button-1>", click_cmd)
            tk.Label(cell, text=f"{hue}°", bg="#2b2b2b", fg="#aaaaaa",
                     font=("Courier", 8)).pack()

        row = tk.Frame(self._mappings_frame, bg="#2b2b2b", pady=3)
        row.pack(fill=tk.X)

        swatch(row, src_rgb, lambda _e, i=idx: self._change_src_color(i))
        tk.Label(row, text="→", bg="#2b2b2b", fg="white", font=("Arial", 14)).pack(side=tk.LEFT, padx=6)
        swatch(row, dst_rgb, lambda _e, i=idx: self._change_dst_color(i))

        rem = tk.Label(row, text="×", bg="#2b2b2b", fg="#ff6b6b",
                       cursor="hand2", font=("Arial", 13, "bold"))
        rem.pack(side=tk.LEFT, padx=(4, 0))
        rem.bind("<Button-1>", lambda _e, i=idx: self._remove_mapping(i))

    def _add_mapping(self) -> None:
        from tkinter import colorchooser
        src = colorchooser.askcolor(title="Step 1 — Pick the source color (color to replace)")
        if src[0] is None:
            return
        dst = colorchooser.askcolor(title="Step 2 — Pick the target color (replace with)")
        if dst[0] is None:
            return
        self._color_mappings.append((
            tuple(int(c) for c in src[0]),
            tuple(int(c) for c in dst[0]),
        ))
        self._rebuild_mappings()

    def _change_src_color(self, idx: int) -> None:
        from tkinter import colorchooser
        src_rgb, dst_rgb = self._color_mappings[idx]
        r, g, b = src_rgb
        result = colorchooser.askcolor(color=f"#{r:02x}{g:02x}{b:02x}",
                                       title="Change source color")
        if result[0] is not None:
            self._color_mappings[idx] = (tuple(int(c) for c in result[0]), dst_rgb)
            self._rebuild_mappings()

    def _change_dst_color(self, idx: int) -> None:
        from tkinter import colorchooser
        src_rgb, dst_rgb = self._color_mappings[idx]
        r, g, b = dst_rgb
        result = colorchooser.askcolor(color=f"#{r:02x}{g:02x}{b:02x}",
                                       title="Change target color")
        if result[0] is not None:
            self._color_mappings[idx] = (src_rgb, tuple(int(c) for c in result[0]))
            self._rebuild_mappings()

    def _remove_mapping(self, idx: int) -> None:
        if 0 <= idx < len(self._color_mappings):
            self._color_mappings.pop(idx)
            self._rebuild_mappings()

    # ------------------------------------------------------------------ Log helpers

    def _clear_log(self) -> None:
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _poll_log(self) -> None:
        while not self._log_q.empty():
            msg = self._log_q.get_nowait()
            self._log_text.config(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        self.after(100, self._poll_log)

    def _poll_img(self) -> None:
        while not self._img_q.empty():
            path = self._img_q.get_nowait()
            if self._gallery and self._gallery.winfo_exists():
                self._gallery.add_image(path)
        self.after(100, self._poll_img)

    # ------------------------------------------------------------------ Run

    def _on_run(self) -> None:
        dataset_folder = self._dataset_var.get().strip() or "./dataset"
        output_folder = self._output_var.get().strip() or "./output"
        if not self._color_mappings:
            self._log_q.put_nowait("ERROR: Add at least one color mapping using '+ Add Mapping'.")
            return
        color_mappings = [
            (self._rgb_to_hue(*src), self._rgb_to_hue(*dst))
            for src, dst in self._color_mappings
        ]
        tolerance = self._tolerance_var.get()
        dry_run = self._dry_run_var.get()
        selected = list(self._class_lb.curselection())
        class_ids = [self._class_id_list[i] for i in selected] if selected else None

        self._run_btn.config(state="disabled")
        self._clear_log()

        if self._gallery and self._gallery.winfo_exists():
            self._gallery.destroy()
        if not dry_run:
            self._gallery = _GalleryWindow(self)

        threading.Thread(
            target=self._worker,
            args=(dataset_folder, output_folder, color_mappings, tolerance, class_ids, dry_run),
            daemon=True,
        ).start()

    def _worker(
        self,
        dataset_folder: str,
        output_folder: str,
        color_mappings: list,
        tolerance: int,
        class_ids,
        dry_run: bool,
    ) -> None:
        from .color_augment import run as color_augment_run

        logger = logging.getLogger("color_augment")
        handler = _QueueHandler(self._log_q)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            color_augment_run(
                dataset_folder=dataset_folder,
                output_folder=output_folder,
                color_mappings=color_mappings,
                tolerance=tolerance,
                class_ids=class_ids,
                dry_run=dry_run,
                on_save=lambda p: self._img_q.put_nowait(p),
            )
        except ValueError as e:
            self._log_q.put_nowait(f"ERROR (invalid class name): {e}")
        except Exception as e:
            self._log_q.put_nowait(f"ERROR: {e}")
        finally:
            logger.removeHandler(handler)
            self.after(0, lambda: self._run_btn.config(state="normal"))
