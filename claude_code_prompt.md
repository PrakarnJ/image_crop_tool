# Claude Code Prompt — Multi-Crop Image Tool (Desktop GUI)

Copy the prompt below into Claude Code at the root of an empty project folder.

---

## PROMPT

Build a Python desktop GUI tool that lets a user load images from an input folder, draw **multiple crop rectangles** on each image with the mouse, save all crops to an output folder, then advance to the next image. Use **Tkinter + Pillow** (standard library + one dependency) so it runs anywhere with no heavy setup.

### Requirements

**Folders**
- `input_folder/` — source images (jpg, jpeg, png, bmp, webp). Path configurable via a "Choose Input Folder" button in the GUI, default to `./input`.
- `output_folder/` — cropped results. Path configurable via "Choose Output Folder" button, default to `./output`. Create it if missing.

**Workflow (one image at a time)**
1. On launch, scan the input folder and build a sorted list of image paths.
2. Display the first image fitted to the canvas (preserve aspect ratio, remember scale factor so crops map back to original-resolution pixels).
3. User clicks-and-drags to draw a rectangle → rectangle stays visible on the canvas with a numbered label (Crop 1, Crop 2, ...).
4. User can draw **multiple rectangles** on the same image before moving on.
5. Buttons:
   - **Save Crops** — write every rectangle as a separate file to the output folder, then keep the image loaded (in case user wants more).
   - **Next Image** — auto-save any pending crops, clear rectangles, load the next image.
   - **Previous Image** — go back (without auto-saving; warn if unsaved crops exist).
   - **Undo Last Box** — remove the most recently drawn rectangle.
   - **Clear All Boxes** — remove all rectangles on the current image.
   - **Skip** — move to next without saving.
6. Status bar shows: `image 3 / 47 — filename.jpg — 2 crops pending`.

**Output naming**
- For source `photo.jpg` with two crops → `photo_crop1.jpg`, `photo_crop2.jpg` in the output folder.
- If `photo_crop1.jpg` already exists, append a suffix: `photo_crop1_2.jpg` to avoid overwrite.
- Preserve the source file extension (and image mode — RGB/RGBA).

**Cropping correctness (this is the part that's easy to get wrong)**
- The canvas displays a *scaled* version of the image. Store the scale factor `s = displayed_size / original_size`.
- When the user drags from `(x1, y1)` to `(x2, y2)` in canvas coordinates, convert to original-image coordinates by dividing by `s` and rounding. Clamp to image bounds.
- Use `PIL.Image.crop((left, top, right, bottom))` on the **original** full-resolution image, not the scaled preview.
- Normalize the rectangle so left < right and top < bottom regardless of drag direction.

**UX details**
- Window resizable; canvas should redraw image and existing rectangles on resize.
- Rectangles drawn with a 2px outline in a bright color (e.g. `#00FF88`), with the crop number drawn in the top-left corner of the box.
- Keyboard shortcuts: `n` = Next, `p` = Previous, `s` = Save Crops, `u` = Undo, `c` = Clear, `Esc` = cancel in-progress drag.
- Show a small toast/messagebox after a successful save: "Saved 3 crops to ./output".

### Project structure

```
.
├── README.md
├── requirements.txt        # just: Pillow
├── src/
│   ├── __init__.py
│   ├── app.py              # Tkinter App class, main loop
│   ├── canvas_view.py      # Canvas widget, mouse handlers, rectangle store
│   ├── image_io.py         # folder scanning, save_crop(), filename collision logic
│   └── geometry.py         # canvas↔image coordinate conversion, rect normalization
└── tests/
    └── test_geometry.py    # unit tests for coord conversion + collision-safe naming
```

### Implementation notes

- Use `PIL.ImageTk.PhotoImage` for displaying images on the Tkinter canvas. Keep a reference to avoid garbage collection.
- Recompute the displayed image whenever the window is resized (bind to `<Configure>`).
- Use `tkinter.filedialog.askdirectory()` for the folder pickers.
- Wrap file I/O in try/except and surface errors via `messagebox.showerror`.
- Don't load all images into memory — load lazily, one at a time.

### Tests (pytest)

Cover at minimum:
- `canvas_to_image_coords()` returns correct integer pixel coords for several scale factors and drag directions.
- `normalize_rect()` handles drags in all four directions.
- `next_available_filename("photo_crop1.jpg", existing=[...])` returns the expected non-colliding name.

### Deliverables

1. Working code in the structure above.
2. `requirements.txt` with `Pillow>=10.0`.
3. `README.md` with: install (`pip install -r requirements.txt`), run (`python -m src.app`), keyboard shortcuts, and a screenshot placeholder.
4. All tests passing (`pytest tests/`).

Start by scaffolding the project, then implement `geometry.py` + tests first (pure logic, easy to verify), then `image_io.py`, then the GUI in `canvas_view.py` and `app.py`. Run the tests at the end and show the output.

---

## Notes on choices

- **Tkinter over PyQt**: ships with Python, zero install friction. PyQt is nicer-looking but adds a 50MB dep and licensing thoughts. Switch to PyQt later if the UI needs to look polished.
- **Pillow** for image I/O — handles all common formats including EXIF orientation (call `ImageOps.exif_transpose()` after open).
- The coordinate-conversion bug is the #1 thing that goes wrong in tools like this — that's why `geometry.py` is isolated and tested first.
