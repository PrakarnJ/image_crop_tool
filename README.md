# Multi-Crop Image Tool

A small Pillow-based image batch-cropper with two modes:

- **v1 — manual GUI** (`src/app.py`): a Tkinter window where you draw rectangles
  on each image and save them as separate files. Also includes a **Video → Frames**
  tool that extracts frames from a folder of videos into image files.
- **v2 — automated batch** (`src/auto_crop.py`): a headless CLI that runs an
  Ultralytics YOLO detector over a folder and saves one crop per detected object.

## Install

```sh
pip install -r requirements.txt
```

This installs Pillow (used by both modes), `ultralytics` (only needed for v2;
pulls torch and OpenCV), and `opencv-python` (needed for the **Video → Frames**
feature; `pip install opencv-python`). For v1 image cropping alone, Pillow is
sufficient — OpenCV is imported lazily and only required once you extract video
frames.

Tkinter ships with Python on macOS and Windows. On some Linux distros you need
`sudo apt install python3-tk`. On macOS with Homebrew Python it's a separate formula:
`brew install python-tk@3.13` (adjust the version to match your Python).

---

## v1 — Manual GUI

### Run

```sh
python -m src.app
```

Defaults: reads from `./input`, writes to `./output` (created if missing). Use the
**Choose Input/Output Folder** buttons to point elsewhere.

### Workflow

1. Click-and-drag on the image to draw a rectangle. Drawing keeps the existing
   rectangles — draw as many as you need.
2. Press `s` (or **Save Crops**) to write each rectangle as a separate file.
3. Press `n` to advance. Pending crops are auto-saved on **Next**.

### Output naming

For source `photo.jpg` with three crops you get `photo_crop1.jpg`, `photo_crop2.jpg`,
`photo_crop3.jpg` in the output folder. If `photo_crop1.jpg` already exists, the new
file becomes `photo_crop1_2.jpg` (then `_3`, `_4`, ...). The source extension and
image mode are preserved (JPEG-incompatible alpha channels are flattened to RGB).

### Keyboard shortcuts

| Key   | Action                                       |
|-------|----------------------------------------------|
| `n`   | Next image (auto-saves pending crops)        |
| `p`   | Previous image (warns if unsaved crops)      |
| `s`   | Save crops                                   |
| `u`   | Undo last rectangle                          |
| `c`   | Clear all rectangles on the current image    |
| `Esc` | Cancel a drag in progress                    |

### Video → Frames

The **Video → Frames** toolbar button opens a dialog that turns a folder of
videos into image files — useful for harvesting still frames to crop afterwards.

1. Pick a **video folder** (the dialog lists every video it finds) and an
   **output folder**.
2. Set the sampling interval (**every _N_ seconds**) and the image format
   (`.jpg` or `.png`).
3. Click **Extract All Frames**. Decoding runs on a background thread with a live
   progress bar; **Cancel** stops cleanly and keeps whatever was already written.

Frames are saved as `<video-stem>_frame0001.jpg`, `<video-stem>_frame0002.jpg`,
… (zero-padded, per video), with the same `_2`/`_3` collision suffixing as the
crop modes. Supported containers: `.mp4 .mov .avi .mkv .webm .m4v .wmv .flv
.mpg .mpeg`.

Frames are read sequentially and every Nth one is kept (N = `fps × interval`);
we avoid frame-seeking because it snaps to keyframes on most codecs. When a file
reports no/garbage fps (some webm/streaming files report `0`/`NaN`), a default of
30 fps is assumed so a single clip can't silently dump thousands of images.

This feature needs **OpenCV**: `pip install opencv-python`. It's imported lazily,
so the rest of the GUI works without it; you'll get a clear install message if you
try to extract without it.

---

## v2 — YOLO auto-crop

Point it at a folder; it detects objects with YOLO and writes one cropped file
per detection. No GUI, no per-image clicks.

### Run

```sh
# Crop everything YOLO finds (default: yolov8n.pt, ~6MB, auto-downloaded on first run)
python -m src.auto_crop --input ./input --output ./output

# Only crop persons and dogs, with 20% padding around each detection
python -m src.auto_crop --classes person,dog --pad 0.20

# Absolute pixel padding, higher confidence threshold, preview only
python -m src.auto_crop --pad 30px --conf 0.5 --dry-run
```

### CLI flags

| Flag         | Default        | Description                                                                 |
|--------------|----------------|-----------------------------------------------------------------------------|
| `--input`    | `./input`      | Input folder                                                                |
| `--output`   | `./output`     | Output folder (created if missing)                                          |
| `--model`    | `yolov8n.pt`   | Ultralytics weights file; auto-downloaded on first run                      |
| `--classes`  | _all_          | Comma-separated COCO names (e.g. `person,dog`). Invalid names error out.    |
| `--conf`     | `0.25`         | Minimum detection confidence                                                |
| `--pad`      | `0.10`         | Padding around each detection: fraction of bbox's longer side, or `Npx`     |
| `--device`   | _auto_         | `cpu`, `cuda`, or `mps`                                                     |
| `--dry-run`  | off            | Print what would be written, don't create files                             |
| `--quiet`    | off            | Suppress INFO logging                                                       |

### Output naming

The detected class is embedded in the filename. Source `beach.jpg` with two persons
and one dog produces:

```
beach_person_crop1.jpg
beach_person_crop2.jpg
beach_dog_crop1.jpg
```

Collision suffixing (`_2`, `_3`, …) is the same as v1.

### Python API

```python
from src.auto_crop import run

run(
    input_folder="./input",
    output_folder="./output",
    classes=["person", "dog"],
    pad="20px",
    conf=0.5,
)
```

---

## Utility — Batch rename

A standalone CLI for renaming image files in a folder with a template. Useful
for normalizing input filenames before cropping, or tidying up output stems.

### Run

```sh
# Renumber every image in ./input as vacation_001.jpg, vacation_002.jpg, ...
python -m src.rename --folder ./input --to "vacation_{n:03d}{ext}"

# Only files matching a glob, starting the counter at 100
python -m src.rename --match "IMG_*.jpg" --to "trip_{n}{ext}" --start 100

# Preview first — nothing is written
python -m src.rename --to "img_{n:04d}{ext}" --dry-run

# Sort by modification time instead of name (numbering follows file age)
python -m src.rename --sort mtime --to "shot_{n:03d}{ext}"
```

### Template tokens

| Token   | Meaning                                              |
|---------|------------------------------------------------------|
| `{n}`   | Running counter; supports format specs (`{n:03d}`)   |
| `{stem}`| Original filename stem (no extension)                |
| `{ext}` | Original extension including the leading dot         |

### Safety

- Refuses to overwrite any destination that exists outside the rename set —
  the whole plan aborts before any file is touched.
- Two-pass execution via temp names handles cycles (`a.jpg ↔ b.jpg` swap).
- `--dry-run` shows the full plan without changing the filesystem.

### CLI flags

| Flag        | Default    | Description                                            |
|-------------|------------|--------------------------------------------------------|
| `--folder`  | `./input`  | Folder to operate on                                   |
| `--match`   | `*`        | Glob filter applied before numbering                   |
| `--to`      | _required_ | Filename template (see tokens above)                   |
| `--start`   | `1`        | Counter start value                                    |
| `--sort`    | `name`     | `name` (alphabetical) or `mtime` (oldest first)        |
| `--dry-run` | off        | Preview the plan; don't change files                   |
| `--quiet`   | off        | Suppress INFO logging                                  |

---

## Tests

```sh
pytest tests/
```

Covers coordinate conversion (all drag directions, scale factors, clamping),
collision-safe filename generation, padding math (fraction + pixel forms,
clamping at image edges), COCO class-name parsing, the rename utility
(template formatting, plan validation, cyclic swaps, dry-run), and the
video-frame math (folder scanning, frame-step including the NaN/0-fps fallback,
frame filenames). The YOLO inference path and the cv2 video-decode path aren't
unit-tested — they require model weights / OpenCV and real media; the cv2 path
was validated end-to-end with a generated test clip.

## Screenshot

_TODO: add screenshot._
