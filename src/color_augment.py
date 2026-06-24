"""Color augmentation: replace target colors in YOLO-labeled regions.

Reads existing YOLO .txt label files to find object bounding boxes, then for
each (source color → target color) mapping produces one augmented full image
where only the pixels inside labeled boxes whose hue matches the source color
are remapped to the target hue. The label file is copied unchanged (bounding
boxes are identical; only pixel values change).

Expected dataset layout::

    dataset/
        images/   ← scanned for images
        labels/   ← YOLO .txt files (same stem as each image)

Output layout::

    output/
        images/   ← augmented images
        labels/   ← copied label files

Importable: ``from src.color_augment import run``.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .image_io import load_image, next_available_filename, scan_input_folder

Box = Tuple[int, int, int, int]

log = logging.getLogger("color_augment")


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------

def hue_replace_region(
    img: Image.Image,
    box: Box,
    src_hue: int,
    dst_hue: int,
    tolerance: int = 30,
) -> Image.Image:
    """Return a copy of `img` where pixels inside `box` whose hue is within
    `tolerance` degrees of `src_hue` are remapped to `dst_hue`. Pixels outside
    the tolerance band (and outside `box`) are left unchanged.

    All hue values in degrees [0, 360)."""
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    x1, y1, x2, y2 = box
    region = arr[y1:y2, x1:x2].copy()

    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]  # OpenCV H ∈ [0, 179]

    src_h_cv = (src_hue % 360) / 2.0
    dst_h_cv = (dst_hue % 360) / 2.0
    tol_cv = tolerance / 2.0

    # Angular distance on [0, 180) circle — handles red wrap-around correctly
    diff = np.abs(h - src_h_cv)
    diff = np.minimum(diff, 180.0 - diff)

    hsv[:, :, 0] = np.where(diff <= tol_cv, dst_h_cv, h)

    replaced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    result = arr.copy()
    result[y1:y2, x1:x2] = replaced
    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# YOLO label I/O
# ---------------------------------------------------------------------------

def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, Box]]:
    """Parse a YOLO .txt label file into pixel-coordinate boxes.

    Returns a list of ``(class_id, (x1, y1, x2, y2))`` tuples.
    Lines that are malformed or produce zero-area boxes are silently skipped."""
    results: List[Tuple[int, Box]] = []
    if not label_path.exists():
        return results
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        x1 = max(0, int((cx - w / 2) * img_w))
        y1 = max(0, int((cy - h / 2) * img_h))
        x2 = min(img_w, int((cx + w / 2) * img_w))
        y2 = min(img_h, int((cy + h / 2) * img_h))
        if x2 > x1 and y2 > y1:
            results.append((cls_id, (x1, y1, x2, y2)))
    return results


# ---------------------------------------------------------------------------
# Dataset introspection
# ---------------------------------------------------------------------------

def scan_dataset_classes(dataset_folder: Union[str, Path]) -> Dict[int, str]:
    """Return a ``{class_id: name}`` dict for every class referenced in the
    train split's label files.

    Class names are read from the first ``*.yaml`` found at the dataset root
    (standard Ultralytics dataset config).  Falls back to ``"class <id>"`` when
    no name is available.  Returns an empty dict when no labels are found."""
    dataset_folder = Path(dataset_folder)
    labels_in = dataset_folder / "train" / "labels"

    found_ids: set = set()
    for txt in labels_in.glob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    found_ids.add(int(parts[0]))
                except ValueError:
                    pass

    # Try to load class names from any *.yaml at the dataset root
    names: Dict[int, str] = {}
    yaml_files = sorted(dataset_folder.glob("*.yaml"))
    if yaml_files:
        try:
            import yaml  # pyyaml — installed as ultralytics dependency
            data = yaml.safe_load(yaml_files[0].read_text())
            raw = data.get("names", {})
            if isinstance(raw, list):
                names = {i: n for i, n in enumerate(raw)}
            elif isinstance(raw, dict):
                names = {int(k): v for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            pass

    return {cid: names.get(cid, f"class {cid}") for cid in sorted(found_ids)}


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_augmented(
    image: Image.Image,
    output_folder: Path,
    source_name: str,
    tag: str,
) -> Path:
    """Save the full augmented image. Naming: ``{stem}_{tag}{ext}``."""
    output_folder.mkdir(parents=True, exist_ok=True)
    src = Path(source_name)
    desired = f"{src.stem}_{tag}{src.suffix}"
    existing = {p.name for p in output_folder.iterdir() if p.is_file()}
    final_name = next_available_filename(desired, existing)
    out_path = output_folder / final_name

    save_img = image
    if src.suffix.lower() in (".jpg", ".jpeg") and image.mode in ("RGBA", "LA", "P"):
        save_img = image.convert("RGB")
    save_img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dataset_folder: Union[str, Path],
    output_folder: Union[str, Path],
    color_mappings: Sequence[Tuple[int, int]] = ((0, 180),),
    tolerance: int = 30,
    class_ids: Optional[List[int]] = None,
    dry_run: bool = False,
    on_save: Optional[Callable[[Path], None]] = None,
) -> int:
    """Programmatic entry point. Returns exit code (0 = ok, 1 = all failed).

    Expected input layout::

        dataset_folder/
            train/images/   ← augmented; one output image per color mapping
            train/labels/   ← YOLO .txt files; copied unchanged alongside each output
            val/            ← copied to output unchanged
            *.yaml          ← copied to output unchanged

    Output layout mirrors the input::

        output_folder/
            train/images/
            train/labels/
            val/
            *.yaml
    """
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)

    images_in = dataset_folder / "train" / "images"
    labels_in = dataset_folder / "train" / "labels"
    images_out = output_folder / "train" / "images"
    labels_out = output_folder / "train" / "labels"

    images = scan_input_folder(images_in)
    if not images:
        log.error("No images found in %s", images_in)
        return 1

    class_filter = set(class_ids) if class_ids is not None else None
    log.info(
        "Dataset: %s  |  %d train image(s)  |  mappings: %s  |  tolerance ±%d°  |  classes: %s",
        dataset_folder, len(images), list(color_mappings), tolerance,
        sorted(class_filter) if class_filter is not None else "all",
    )

    total_saved = 0
    skipped = 0
    no_label = 0
    start = time.time()

    for img_path in images:
        label_path = labels_in / f"{img_path.stem}.txt"
        if not label_path.exists():
            log.warning("No label for %s — skipped", img_path.name)
            no_label += 1
            continue

        try:
            img = load_image(img_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping %s: load failed (%s)", img_path.name, e)
            skipped += 1
            continue

        all_boxes = read_yolo_labels(label_path, *img.size)
        boxes = (
            [(cid, box) for cid, box in all_boxes if cid in class_filter]
            if class_filter is not None
            else all_boxes
        )
        if not boxes:
            if not all_boxes:
                log.info("%s: label file empty, skipped", img_path.name)
            else:
                log.info(
                    "%s: no boxes match selected class(es) %s, skipped",
                    img_path.name, sorted(class_filter),
                )
            no_label += 1
            continue

        img_saved = 0
        for src_hue, dst_hue in color_mappings:
            aug = img
            for _cls_id, box in boxes:
                aug = hue_replace_region(aug, box, src_hue, dst_hue, tolerance)

            tag = f"h{src_hue}to{dst_hue}"
            if not dry_run:
                out_img = _save_augmented(aug, images_out, img_path.name, tag)
                labels_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, labels_out / f"{out_img.stem}.txt")
                if on_save is not None:
                    on_save(out_img)
            img_saved += 1
            total_saved += 1

        log.info(
            "%s: %d box(es), %d mapping(s) → %d augmented image(s)",
            img_path.name, len(boxes), len(color_mappings), img_saved,
        )

    elapsed = time.time() - start
    log.info(
        "Done (train): %d image(s), %d augmented, %d skipped (load), "
        "%d skipped (no/empty label), %.2fs",
        len(images), total_saved, skipped, no_label, elapsed,
    )

    # --- Copy val split unchanged --------------------------------------------
    val_in = dataset_folder / "val"
    if val_in.exists():
        val_out = output_folder / "val"
        if not dry_run:
            shutil.copytree(val_in, val_out, dirs_exist_ok=True)
        log.info("val split → %s (%s)", val_out, "skipped (dry run)" if dry_run else "copied")

    # --- Copy dataset yaml(s) -------------------------------------------------
    for yaml_path in sorted(dataset_folder.glob("*.yaml")):
        if not dry_run:
            output_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yaml_path, output_folder / yaml_path.name)
        log.info("%s → output root (%s)", yaml_path.name, "skipped (dry run)" if dry_run else "copied")

    return 1 if (skipped + no_label) == len(images) else 0
