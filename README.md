# Multi-Crop Image Tool

A small Pillow-based image batch-cropper with two modes:

- **v1 — manual GUI** (`src/app.py`): a Tkinter window where you draw rectangles
  on each image and save them as separate files.
- **v2 — automated batch** (`src/auto_crop.py`): a headless CLI that runs an
  Ultralytics YOLO detector over a folder and saves one crop per detected object.

## Install

```sh
pip install -r requirements.txt
```

This installs Pillow (used by both modes) and `ultralytics` (only needed for v2;
pulls torch and OpenCV). For v1 alone, Pillow is sufficient.

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

## Tests

```sh
pytest tests/
```

Covers coordinate conversion (all drag directions, scale factors, clamping),
collision-safe filename generation, padding math (fraction + pixel forms,
clamping at image edges), and COCO class-name parsing. The YOLO inference path
itself isn't unit-tested — it requires the model weights and a CV stack.

## Screenshot

_TODO: add screenshot._
