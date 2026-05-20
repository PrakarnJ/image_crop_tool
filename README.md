# Multi-Crop Image Tool

A small Tkinter + Pillow desktop GUI for batch-cropping images. Open a folder, draw
as many rectangles as you like on each photo, save them all as separate files, advance
to the next image.

## Install

```sh
pip install -r requirements.txt
```

Tkinter ships with Python on macOS and Windows. On some Linux distros you need
`sudo apt install python3-tk`. On macOS with Homebrew Python it's a separate formula:
`brew install python-tk@3.13` (adjust the version to match your Python).

## Run

```sh
python -m src.app
```

Defaults: reads from `./input`, writes to `./output` (created if missing). Use the
**Choose Input/Output Folder** buttons to point elsewhere.

## Workflow

1. Click-and-drag on the image to draw a rectangle. Drawing keeps the existing
   rectangles — draw as many as you need.
2. Press `s` (or **Save Crops**) to write each rectangle as a separate file.
3. Press `n` to advance. Pending crops are auto-saved on **Next**.

## Output naming

For source `photo.jpg` with three crops you get `photo_crop1.jpg`, `photo_crop2.jpg`,
`photo_crop3.jpg` in the output folder. If `photo_crop1.jpg` already exists, the new
file becomes `photo_crop1_2.jpg` (then `_3`, `_4`, ...). The source extension and
image mode are preserved (JPEG-incompatible alpha channels are flattened to RGB).

## Keyboard shortcuts

| Key   | Action                                       |
|-------|----------------------------------------------|
| `n`   | Next image (auto-saves pending crops)        |
| `p`   | Previous image (warns if unsaved crops)      |
| `s`   | Save crops                                   |
| `u`   | Undo last rectangle                          |
| `c`   | Clear all rectangles on the current image    |
| `Esc` | Cancel a drag in progress                    |

## Tests

```sh
pytest tests/
```

Covers coordinate conversion (all drag directions, scale factors, clamping) and
collision-safe filename generation.

## Screenshot

_TODO: add screenshot._
