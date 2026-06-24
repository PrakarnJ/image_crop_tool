"""AutoCropTab — Tkinter tab that wraps the YOLO auto-crop backend.

Provides a form UI, a background-thread runner, and a scrollable log panel.
Heavy imports (ultralytics, torch, auto_crop) are deferred to the worker thread.
"""

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText


class _QueueHandler(logging.Handler):
    """Logging handler that enqueues formatted records for the UI to poll."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put_nowait(self.format(record))


class AutoCropTab(ttk.Frame):
    """A ttk.Frame that presents the YOLO auto-crop form inside a Notebook tab."""

    def __init__(self, notebook: ttk.Notebook) -> None:
        super().__init__(notebook)
        notebook.add(self, text="Auto Crop")

        self._log_q: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_log()  # start the recurring log-drain loop

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        # Top: form area; bottom: log panel.
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

        # StringVars / control variables
        self._input_var = tk.StringVar(value="./input")
        self._output_var = tk.StringVar(value="./output")
        self._model_var = tk.StringVar(value="yolov8n.pt")
        self._conf_var = tk.DoubleVar(value=0.25)
        self._pad_var = tk.StringVar(value="0.10")
        self._device_var = tk.StringVar(value="")
        self._dry_run_var = tk.BooleanVar(value=False)
        self._class_id_list: list = []  # parallel to listbox rows; holds int class IDs

        row = 0

        # --- Input folder -------------------------------------------------------
        ttk.Label(form, text="Input folder:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._input_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(
            form, text="Browse…", command=self._browse_input
        ).grid(row=row, column=2, padx=(4, 0), pady=3)
        row += 1

        # --- Output folder ------------------------------------------------------
        ttk.Label(form, text="Output folder:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._output_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(
            form, text="Browse…", command=self._browse_output
        ).grid(row=row, column=2, padx=(4, 0), pady=3)
        row += 1

        # --- Model --------------------------------------------------------------
        ttk.Label(form, text="Model:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._model_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        model_btn_frame = ttk.Frame(form)
        model_btn_frame.grid(row=row, column=2, padx=(4, 0), pady=3, sticky="w")
        ttk.Button(model_btn_frame, text="Browse…", command=self._browse_model).pack(side=tk.LEFT, padx=(0, 4))
        self._load_btn = ttk.Button(model_btn_frame, text="Load Model", command=self._load_model)
        self._load_btn.pack(side=tk.LEFT)
        row += 1

        # --- Classes ------------------------------------------------------------
        ttk.Label(form, text="Classes:").grid(
            row=row, column=0, sticky="ne", padx=(0, 4), pady=3
        )
        class_list_frame = ttk.Frame(form)
        class_list_frame.grid(row=row, column=1, sticky="nsew", pady=3)
        self._class_lb = tk.Listbox(
            class_list_frame, selectmode=tk.MULTIPLE, height=5, exportselection=False
        )
        class_vsb = ttk.Scrollbar(class_list_frame, orient="vertical", command=self._class_lb.yview)
        self._class_lb.config(yscrollcommand=class_vsb.set)
        self._class_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        class_ctrl = ttk.Frame(form)
        class_ctrl.grid(row=row, column=2, sticky="nw", padx=(4, 0), pady=3)
        ttk.Button(class_ctrl, text="All", width=6, command=self._select_all_classes).pack(anchor="w")
        ttk.Button(class_ctrl, text="Clear", width=6, command=self._clear_classes).pack(anchor="w", pady=(2, 6))
        self._class_status = ttk.Label(class_ctrl, text="Load a model\nfirst", foreground="gray", wraplength=110)
        self._class_status.pack(anchor="w")
        classes_row = row
        row += 1

        # --- Confidence ---------------------------------------------------------
        ttk.Label(form, text="Confidence:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        conf_frame = ttk.Frame(form)
        conf_frame.grid(row=row, column=1, sticky="ew", pady=3)

        self._conf_label = ttk.Label(conf_frame, text="0.25", width=4)
        self._conf_label.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            variable=self._conf_var,
            orient="horizontal",
            command=lambda v: self._conf_label.config(text=f"{float(v):.2f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        row += 1

        # --- Padding ------------------------------------------------------------
        ttk.Label(form, text="Padding:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._pad_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Label(form, text="0.10 or 20px", foreground="gray").grid(
            row=row, column=2, sticky="w", padx=(4, 0), pady=3
        )
        row += 1

        # --- Device -------------------------------------------------------------
        ttk.Label(form, text="Device:").grid(
            row=row, column=0, sticky="e", padx=(0, 4), pady=3
        )
        ttk.Entry(form, textvariable=self._device_var, width=40).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Label(form, text="cpu / cuda / mps, empty = auto", foreground="gray").grid(
            row=row, column=2, sticky="w", padx=(4, 0), pady=3
        )
        row += 1

        # --- Dry run ------------------------------------------------------------
        ttk.Checkbutton(
            form, text="Dry run (preview only, no files written)",
            variable=self._dry_run_var
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        # --- Run button ---------------------------------------------------------
        self._run_btn = ttk.Button(form, text="Run Auto Crop", command=self._on_run)
        self._run_btn.grid(row=row, column=1, sticky="w", pady=(8, 4))

        form.columnconfigure(1, weight=1)
        form.rowconfigure(classes_row, weight=1)

    def _build_log(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Log:").pack(anchor="w", padx=4)
        self._log_text = ScrolledText(
            parent, state="disabled", height=12, font=("Courier", 10)
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------ Browse

    def _browse_input(self) -> None:
        current = self._input_var.get().strip()
        initial = str(Path(current).resolve()) if current else str(Path.cwd())
        folder = filedialog.askdirectory(initialdir=initial, title="Choose input folder")
        if folder:
            self._input_var.set(folder)

    def _browse_output(self) -> None:
        current = self._output_var.get().strip()
        initial = str(Path(current).resolve()) if current else str(Path.cwd())
        folder = filedialog.askdirectory(initialdir=initial, title="Choose output folder")
        if folder:
            self._output_var.set(folder)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose model weights",
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._model_var.set(path)

    def _load_model(self) -> None:
        model_name = self._model_var.get().strip() or "yolov8n.pt"
        self._load_btn.config(state="disabled")
        self._class_status.config(text="Loading…", foreground="gray")
        self._class_lb.delete(0, "end")
        threading.Thread(target=self._load_model_worker, args=(model_name,), daemon=True).start()

    def _load_model_worker(self, model_name: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(model_name)
            names: dict = dict(model.names)
            self.after(0, lambda: self._populate_classes(names, model_name))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.after(0, lambda: self._class_status.config(text=f"Load failed:\n{msg}", foreground="red"))
            self.after(0, lambda: self._load_btn.config(state="normal"))

    def _populate_classes(self, names: dict, model_name: str) -> None:
        self._class_id_list = sorted(names.keys())
        self._class_lb.delete(0, "end")
        for cls_id in self._class_id_list:
            self._class_lb.insert("end", f"{cls_id}: {names[cls_id]}")
        stem = Path(model_name).name
        self._class_status.config(
            text=f"{stem}\n{len(names)} class(es)\n(empty sel. = all)",
            foreground="#4caf50",
        )
        self._load_btn.config(state="normal")

    def _select_all_classes(self) -> None:
        self._class_lb.select_set(0, "end")

    def _clear_classes(self) -> None:
        self._class_lb.selection_clear(0, "end")

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

    # ------------------------------------------------------------------ Run

    def _on_run(self) -> None:
        # Read all form values on the main thread before handing off to the worker
        input_folder = self._input_var.get().strip() or "./input"
        output_folder = self._output_var.get().strip() or "./output"
        model_name = self._model_var.get().strip() or "yolov8n.pt"
        selected = list(self._class_lb.curselection())
        class_ids = [self._class_id_list[i] for i in selected] if selected else None
        conf = float(self._conf_var.get())
        pad = self._pad_var.get().strip() or "0.10"
        device = self._device_var.get().strip() or None
        dry_run = self._dry_run_var.get()

        self._run_btn.config(state="disabled")
        self._clear_log()

        threading.Thread(
            target=self._worker,
            args=(input_folder, output_folder, model_name, class_ids, conf, pad, device, dry_run),
            daemon=True,
        ).start()

    def _worker(
        self,
        input_folder: str,
        output_folder: str,
        model_name: str,
        class_ids,
        conf: float,
        pad: str,
        device,
        dry_run: bool,
    ) -> None:
        from .auto_crop import run as auto_crop_run  # lazy — keeps torch out of startup

        logger = logging.getLogger("auto_crop")
        handler = _QueueHandler(self._log_q)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            auto_crop_run(
                input_folder=input_folder,
                output_folder=output_folder,
                model_name=model_name,
                class_ids=class_ids,
                conf=conf,
                pad=pad,
                device=device,
                dry_run=dry_run,
            )
        except ValueError as e:
            self._log_q.put_nowait(f"ERROR (invalid class name): {e}")
        except Exception as e:
            self._log_q.put_nowait(f"ERROR: {e}")
        finally:
            logger.removeHandler(handler)
            self.after(0, lambda: self._run_btn.config(state="normal"))
