# Multi-Crop Image Tool — v2 Requirements (YOLO Auto-Crop)

## 1. Overview

v2 replaces the manual draw-rectangle workflow with a **fully automated batch
pipeline**: point the tool at an input folder, and it uses a pretrained YOLO
(Ultralytics) model to detect objects and write one cropped file per detection
to the output folder. No interactive canvas, no per-image clicks.

v1 (`src/app.py` Tk GUI) remains in the repo and continues to work; v2 is a new
entry point that shares the existing `image_io` helpers where it makes sense.

## 2. Goals

- Crop every relevant detection across a folder of images in one command.
- Let the user choose **which COCO classes** to crop (e.g. only `person`, only
  `dog`+`cat`, or all 80).
- Let the user **pad each detection** before cropping so the output isn't
  uncomfortably tight (configurable percentage or absolute pixels).
- Drop low-confidence detections via a configurable threshold.
- Preserve v1's collision-safe filename behavior (`<stem>_crop<N><ext>`).

## 3. Non-goals

- No interactive GUI in v2 (manual mode stays in v1).
- No custom-model training; v2 only consumes pretrained Ultralytics weights.
- No video, no streaming input, no multi-folder recursion (flat folder only,
  matching v1).
- No re-identification, no tracking, no dedupe across images.

## 4. Functional requirements

### 4.1 Entry point

- New module `src/auto_crop.py` exposing a `main()` CLI entry, runnable as
  `python -m src.auto_crop`.
- Arguments (argparse):
  - `--input <folder>` (default `./input`)
  - `--output <folder>` (default `./output`)
  - `--model <weights>` (default `yolov8n.pt`; Ultralytics auto-downloads on
    first run)
  - `--classes <name[,name...]>` (default: all COCO classes). Accepts class
    names (`person,dog`) — invalid names error out with the list of valid
    names.
  - `--conf <float>` (default `0.25`) — minimum detection confidence.
  - `--pad <value>` (default `0.10`) — padding around each detection. Accepts
    either a fraction of the bbox's longer side (`0.10` = 10%) or an absolute
    pixel count when suffixed with `px` (`20px`). Padding is clamped to the
    image bounds.
  - `--device <cpu|cuda|mps>` (default: auto-detect via Ultralytics).
  - `--dry-run` — print what would be written, don't create files.

### 4.2 Pipeline

For each image file in the input folder (using `image_io.scan_input_folder`
for sort order and extension filter):

1. Load the image via `image_io.load_image` (EXIF orientation applied).
2. Run inference: `model.predict(image, conf=<conf>, classes=<class-ids>)`.
3. For each detection above threshold and in the requested class set:
   - Convert the bbox to integer pixel coords.
   - Apply padding (per `--pad` rule) and clamp to `[0, W] × [0, H]`.
   - Skip detections that collapse to zero area after clamping.
4. Save each surviving box via a slight extension of `image_io.save_crop` so
   the filename includes the class:
   `<stem>_<class>_crop<N><ext>` (e.g. `vacation_person_crop1.jpg`).
   Collision suffixing (`_2`, `_3`, …) is unchanged.
5. Continue to the next image. Per-image failures (corrupt file, inference
   error) log a warning and do not abort the batch.

### 4.3 Reporting

- Print a per-image line: `vacation.jpg: 3 crops (person×2, dog×1)`.
- End-of-run summary: total images, total crops, skipped images, elapsed time.
- Non-zero exit code if **all** images failed; zero otherwise.

### 4.4 Reuse from v1

- `image_io.scan_input_folder` — unchanged.
- `image_io.load_image` — unchanged.
- `image_io.next_available_filename` — unchanged.
- `image_io.save_crop` — extend to accept an optional `label` segment in the
  filename. v1 calls it without the label (no behavior change).

## 5. Non-functional requirements

### 5.1 Dependencies

Add to `requirements.txt`:

- `ultralytics>=8.0` (pulls `torch`, `torchvision`, `opencv-python`).

Pillow stays. Tkinter still only used by v1. The new code must not import
`tkinter` so headless servers can run v2.

### 5.2 Performance

- First run downloads model weights (~6 MB for `yolov8n.pt`) — acceptable.
- Subsequent runs reuse the cached weights.
- Throughput target on CPU: ≥ 1 image/sec on a 1080p input with `yolov8n`.
  Not a hard SLA; just a sanity check.

### 5.3 Logging

- Use `logging`, not `print`, for warnings/errors. INFO-level for the
  per-image and summary lines. `--quiet` flag suppresses INFO.

## 6. Output naming examples

Source: `beach.jpg`, two persons and one dog detected.

```
beach_person_crop1.jpg
beach_person_crop2.jpg
beach_dog_crop1.jpg
```

If `beach_person_crop1.jpg` already exists in `output/`, the next write
becomes `beach_person_crop1_2.jpg` — same collision rule as v1.

## 7. Acceptance criteria

1. `python -m src.auto_crop --input fixtures/ --output out/` runs end-to-end
   on a folder of mixed-class images and produces correctly named crops.
2. `--classes person` filters out all non-person detections.
3. `--pad 0.20` produces visibly larger crops than `--pad 0` on the same
   input; clamping at image edges is verified (no out-of-bounds crashes when
   a detection touches the border).
4. `--conf 0.9` returns strictly fewer (or equal) crops than `--conf 0.25`.
5. `--dry-run` writes no files but prints the same per-image report.
6. v1 (`python -m src.app`) still launches and behaves identically.
7. All existing tests in `tests/test_geometry.py` continue to pass.

## 8. Tests to add

`tests/test_auto_crop.py` (pytest, no GPU, no real model required):

- **Padding math** — a pure helper `pad_box(box, pad_spec, image_size)` that
  applies fractional and pixel-suffixed padding and clamps. Cover: positive
  fraction, zero padding, `"20px"` form, clamp at top-left corner, clamp at
  bottom-right corner.
- **Class filter parsing** — `parse_classes("person,dog")` → `[0, 16]`
  (COCO IDs); invalid name raises with a helpful message.
- **Filename labeling** — extend the existing `save_crop` tests to cover the
  new `label` parameter and confirm v1's no-label path is unchanged.

YOLO inference itself is not unit-tested (would require bundling weights and
a CV stack into CI). One optional integration test, gated behind an env var,
can exercise the full pipeline on a small fixture image.

## 9. Open questions / decisions deferred to implementation

- Whether to expose a Python API (`auto_crop.run(...)`) alongside the CLI, or
  keep it CLI-only. Lean: expose both — the CLI is a thin wrapper.
- Whether to write a per-run JSON manifest (`output/manifest.json`) recording
  which source image and bbox each crop came from. Useful for downstream
  pipelines but not required by the acceptance criteria above.
