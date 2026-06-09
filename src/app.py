"""Root window: creates the ttk.Notebook and mounts the three tab modules."""

import tkinter as tk
from tkinter import ttk

from .ui_manual_crop import ManualCropTab
from .ui_auto_crop import AutoCropTab
from .ui_rename import RenameTab


class App:
    def __init__(self, root: tk.Tk):
        root.title("Multi-Crop Image Tool")
        root.geometry("1200x800")
        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)
        ManualCropTab(nb, root)
        AutoCropTab(nb)
        RenameTab(nb)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
