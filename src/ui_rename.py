"""Rename tab UI — wraps the batch-rename backend (src/rename.py) with a
Tkinter form, a Preview button, a Run button, and a scrollable log panel.
"""

import logging
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText


class _QueueHandler(logging.Handler):
    """Logging handler that puts formatted records into a queue."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put_nowait(self.format(record))


class RenameTab(ttk.Frame):
    def __init__(self, notebook: ttk.Notebook):
        super().__init__(notebook)
        notebook.add(self, text="Rename")

        self._log_q: queue.Queue = queue.Queue()

        self._build_form()
        self._build_log()

        # Start polling the log queue
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_form(self) -> None:
        """Build the top form grid."""
        form = ttk.LabelFrame(self, text="Options", padding=(8, 4))
        form.pack(fill=tk.X, padx=6, pady=6)

        # StringVars / IntVars
        self._folder_var = tk.StringVar(value="./input")
        self._match_var = tk.StringVar(value="*")
        self._template_var = tk.StringVar(value="")
        self._start_var = tk.StringVar(value="1")
        self._sort_var = tk.StringVar(value="name")
        self._dry_run_var = tk.BooleanVar(value=False)

        row = 0

        # Folder row
        ttk.Label(form, text="Folder:").grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
        folder_entry = ttk.Entry(form, textvariable=self._folder_var, width=40)
        folder_entry.grid(row=row, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(form, text="Browse…", command=self._browse_folder).grid(
            row=row, column=2, padx=4, pady=2
        )
        row += 1

        # Match glob row
        ttk.Label(form, text="Match glob:").grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(form, textvariable=self._match_var, width=40).grid(
            row=row, column=1, sticky=tk.EW, padx=4, pady=2
        )
        row += 1

        # Template row
        ttk.Label(form, text="Template\n(e.g. vacation_{n:03d}{ext}):").grid(
            row=row, column=0, sticky=tk.W, padx=4, pady=2
        )
        ttk.Entry(form, textvariable=self._template_var, width=40).grid(
            row=row, column=1, sticky=tk.EW, padx=4, pady=2
        )
        row += 1

        # Start counter row
        ttk.Label(form, text="Start counter:").grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Spinbox(
            form,
            textvariable=self._start_var,
            from_=1,
            to=9999,
            width=8,
        ).grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        row += 1

        # Sort row
        ttk.Label(form, text="Sort:").grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Combobox(
            form,
            textvariable=self._sort_var,
            values=["name", "mtime"],
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        row += 1

        # Dry run row
        ttk.Checkbutton(form, text="Dry run (preview only, no files changed)", variable=self._dry_run_var).grid(
            row=row, column=1, sticky=tk.W, padx=4, pady=2
        )
        row += 1

        # Buttons row
        btn_frame = ttk.Frame(form)
        btn_frame.grid(row=row, column=1, sticky=tk.W, padx=4, pady=6)
        ttk.Button(btn_frame, text="Preview", command=self._on_preview).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Run Rename", command=self._on_run).pack(side=tk.LEFT)

        form.columnconfigure(1, weight=1)

    def _build_log(self) -> None:
        """Build the bottom scrollable log panel."""
        self._log_text = ScrolledText(
            self,
            state="disabled",
            height=12,
            font=("Courier", 10),
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Browse helper
    # ------------------------------------------------------------------

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder")
        if folder:
            self._folder_var.set(folder)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _clear_log(self) -> None:
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _write_log(self, msg: str) -> None:
        self._log_text.config(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _poll_log(self) -> None:
        while not self._log_q.empty():
            msg = self._log_q.get_nowait()
            self._log_text.config(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_preview(self) -> None:
        from .rename import select_files, plan_renames

        self._clear_log()
        folder = Path(self._folder_var.get())
        match = self._match_var.get() or "*"
        template = self._template_var.get().strip()
        if not template:
            self._write_log("Template is required.")
            return
        try:
            start = int(self._start_var.get())
        except ValueError:
            self._write_log("Start must be an integer.")
            return
        sort = self._sort_var.get()
        files = select_files(folder, match=match, sort=sort)
        if not files:
            self._write_log(f"No image files matched '{match}' in {folder}")
            return
        try:
            plan = plan_renames(files, template, start=start)
        except ValueError as e:
            self._write_log(f"ERROR: {e}")
            return
        for src, dst in plan:
            if src.name == dst.name:
                self._write_log(f"  {src.name}  (unchanged)")
            else:
                self._write_log(f"  {src.name}  →  {dst.name}")
        self._write_log(f"\n{len(plan)} file(s) would be renamed.")

    def _on_run(self) -> None:
        from .rename import run as rename_run

        self._clear_log()
        logger = logging.getLogger("rename")
        handler = _QueueHandler(self._log_q)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            rename_run(
                folder=self._folder_var.get(),
                match=self._match_var.get() or "*",
                template=self._template_var.get().strip(),
                start=int(self._start_var.get()),
                sort=self._sort_var.get(),
                dry_run=bool(self._dry_run_var.get()),
            )
        except Exception as e:
            self._log_q.put_nowait(f"ERROR: {e}")
        finally:
            logger.removeHandler(handler)
        # drain immediately (synchronous run)
        self._poll_log()
