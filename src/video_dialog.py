"""Toplevel dialog for batch-extracting frames from a folder of videos into image
files. The cv2 decode loop runs on a worker thread; progress is streamed back to
the Tk main loop through a thread-safe queue polled with `after()` (Tk widgets must
only be touched from the main thread)."""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .video_io import extract_folder, scan_video_folder


class VideoExtractDialog(tk.Toplevel):
    POLL_MS = 100

    def __init__(self, master, input_folder: Path, output_folder: Path):
        super().__init__(master)
        self.title("Extract Frames from Video")
        self.geometry("640x520")
        self.transient(master)

        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)

        self._queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._cancel = threading.Event()

        self._build_ui()
        self._refresh_video_list()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI construction ------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # Input folder row
        in_row = tk.Frame(self)
        in_row.pack(fill=tk.X, **pad)
        tk.Label(in_row, text="Video folder:", width=12, anchor="w").pack(side=tk.LEFT)
        self.input_var = tk.StringVar(value=str(self.input_folder))
        tk.Entry(in_row, textvariable=self.input_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(in_row, text="Choose…", command=self.choose_input).pack(side=tk.LEFT)

        # Output folder row
        out_row = tk.Frame(self)
        out_row.pack(fill=tk.X, **pad)
        tk.Label(out_row, text="Output folder:", width=12, anchor="w").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=str(self.output_folder))
        tk.Entry(out_row, textvariable=self.output_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(out_row, text="Choose…", command=self.choose_output).pack(side=tk.LEFT)

        # Options row
        opt_row = tk.Frame(self)
        opt_row.pack(fill=tk.X, **pad)
        tk.Label(opt_row, text="Every", width=12, anchor="w").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="1.0")
        tk.Entry(opt_row, textvariable=self.interval_var, width=8).pack(side=tk.LEFT)
        tk.Label(opt_row, text="seconds").pack(side=tk.LEFT, padx=(2, 16))
        tk.Label(opt_row, text="Format:").pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value=".jpg")
        ttk.Combobox(
            opt_row, textvariable=self.format_var, values=[".jpg", ".png"],
            width=6, state="readonly",
        ).pack(side=tk.LEFT, padx=4)

        # Found-videos list
        list_frame = tk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, **pad)
        tk.Label(list_frame, text="Videos found:", anchor="w").pack(fill=tk.X)
        self.video_list = tk.Listbox(list_frame)
        self.video_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(list_frame, command=self.video_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_list.config(yscrollcommand=sb.set)

        # Progress + status
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill=tk.X, **pad)
        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, **pad)

        # Action buttons
        btn_row = tk.Frame(self)
        btn_row.pack(fill=tk.X, **pad)
        self.extract_btn = tk.Button(
            btn_row, text="Extract All Frames", command=self.start_extract
        )
        self.extract_btn.pack(side=tk.LEFT)
        self.cancel_btn = tk.Button(
            btn_row, text="Cancel", command=self.cancel_extract, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Close", command=self._on_close).pack(side=tk.RIGHT)

    # --- Folder selection -----------------------------------------------------

    def choose_input(self) -> None:
        folder = filedialog.askdirectory(
            parent=self, title="Choose video folder",
            initialdir=self._safe_initialdir(self.input_var.get()),
        )
        if folder:
            self.input_var.set(folder)
            self._refresh_video_list()

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(
            parent=self, title="Choose output folder",
            initialdir=self._safe_initialdir(self.output_var.get()),
        )
        if folder:
            self.output_var.set(folder)

    @staticmethod
    def _safe_initialdir(value: str) -> str:
        p = Path(value)
        return str(p) if p.is_dir() else str(Path.cwd())

    def _refresh_video_list(self) -> None:
        self.video_list.delete(0, tk.END)
        videos = scan_video_folder(self.input_var.get())
        for v in videos:
            self.video_list.insert(tk.END, v.name)
        n = len(videos)
        self.status_var.set(
            f"{n} video{'s' if n != 1 else ''} found." if n
            else "No video files in this folder."
        )

    # --- Extraction lifecycle -------------------------------------------------

    def start_extract(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            interval = float(self.interval_var.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid interval", "Interval must be a positive number of seconds.",
                parent=self,
            )
            return

        videos = scan_video_folder(self.input_var.get())
        if not videos:
            messagebox.showinfo("Nothing to do", "No videos in the chosen folder.", parent=self)
            return

        input_folder = self.input_var.get()
        output_folder = self.output_var.get()
        image_ext = self.format_var.get()

        self._cancel.clear()
        self.extract_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress.config(value=0, maximum=len(videos))
        self.status_var.set("Starting…")

        self._worker = threading.Thread(
            target=self._run_extract,
            args=(input_folder, output_folder, interval, image_ext),
            daemon=True,
        )
        self._worker.start()
        self.after(self.POLL_MS, self._poll_queue)

    def _run_extract(self, input_folder, output_folder, interval, image_ext) -> None:
        def progress_cb(vi, vtotal, name, saved, est_total):
            self._queue.put(("progress", vi, vtotal, name, saved, est_total))

        try:
            results = extract_folder(
                input_folder, output_folder, interval, image_ext,
                progress_cb=progress_cb,
                should_cancel=self._cancel.is_set,
            )
            self._queue.put(("done", results))
        except Exception as e:  # surfaces e.g. cv2-not-installed
            self._queue.put(("error", str(e)))

    def cancel_extract(self) -> None:
        self._cancel.set()
        self.status_var.set("Cancelling…")
        self.cancel_btn.config(state=tk.DISABLED)

    def _poll_queue(self) -> None:
        terminal = False
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, vi, vtotal, name, saved, est_total = msg
                    self.progress.config(maximum=max(vtotal, 1))
                    self.progress.config(value=min(vi, vtotal))
                    self.status_var.set(
                        f"video {vi}/{vtotal}: {name} — {saved} frame"
                        f"{'s' if saved != 1 else ''}"
                    )
                elif kind == "done":
                    self._on_done(msg[1])
                    terminal = True
                elif kind == "error":
                    messagebox.showerror("Extraction failed", msg[1], parent=self)
                    self._finish()
                    terminal = True
        except queue.Empty:
            pass

        if terminal:
            return
        if (self._worker is not None and self._worker.is_alive()) or not self._queue.empty():
            self.after(self.POLL_MS, self._poll_queue)
        else:
            self._finish()

    def _on_done(self, results) -> None:
        self._finish()
        frames = results.get("frames", 0)
        videos = results.get("videos", 0)
        failed = results.get("failed", [])
        cancelled = results.get("cancelled", False)
        self.progress.config(value=self.progress.cget("maximum"))

        lines = [f"Extracted {frames} frame(s) from {videos} video(s)."]
        if cancelled:
            lines.append("Cancelled before finishing all videos.")
        if failed:
            lines.append("")
            lines.append(f"{len(failed)} video(s) failed:")
            lines.extend(f"  • {name}: {err}" for name, err in failed)
        self.status_var.set(lines[0])
        messagebox.showinfo("Done", "\n".join(lines), parent=self)

    def _finish(self) -> None:
        self.extract_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno(
                "Extraction running",
                "Frame extraction is still running. Cancel and close?",
                parent=self,
            ):
                return
            self._cancel.set()
        self.destroy()
