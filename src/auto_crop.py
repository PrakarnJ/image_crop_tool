"""v2 entry point: YOLO-powered batch auto-crop.

Headless pipeline — no Tkinter. Scans an input folder, runs an Ultralytics
YOLO model over each image, and writes one cropped file per detection. The
detected COCO class becomes part of the output filename.

CLI: `python -m src.auto_crop --input <in> --output <out> [...]`.
Also importable: `from src.auto_crop import run`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .image_io import load_image, save_crop, scan_input_folder

# COCO 80 class names in the canonical Ultralytics order. Hardcoded so
# parse_classes() can work without importing torch / loading weights — keeps
# unit tests light and CI green on machines without the CV stack installed.
COCO_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

Box = Tuple[int, int, int, int]
PadSpec = Union[float, str]

log = logging.getLogger("auto_crop")


# --- Pure helpers (unit-tested without YOLO) ---------------------------------

def parse_classes(spec: Optional[str]) -> Optional[List[int]]:
    """Convert a comma-separated COCO class-name string into a sorted list of
    class IDs. Returns None to mean "all classes" (matches Ultralytics' API,
    where `classes=None` disables the filter)."""
    if spec is None or spec.strip() == "":
        return None
    name_to_id = {n: i for i, n in enumerate(COCO_NAMES)}
    requested = [n.strip() for n in spec.split(",") if n.strip()]
    ids: List[int] = []
    invalid: List[str] = []
    for n in requested:
        if n in name_to_id:
            ids.append(name_to_id[n])
        else:
            invalid.append(n)
    if invalid:
        raise ValueError(
            f"Unknown class name(s): {', '.join(invalid)}. "
            f"Valid names: {', '.join(COCO_NAMES)}"
        )
    return ids


def pad_box(box: Box, pad_spec: PadSpec, image_size: Tuple[int, int]) -> Box:
    """Grow `box` by `pad_spec` and clamp to image bounds.

    `pad_spec` is either:
      - a numeric fraction of the bbox's longer side (`0.10` → 10%), or
      - a string like `"20px"` meaning an absolute pixel count.
    """
    x1, y1, x2, y2 = box
    img_w, img_h = image_size

    if isinstance(pad_spec, str):
        s = pad_spec.strip()
        if s.endswith("px"):
            pad_x = pad_y = int(s[:-2])
        else:
            frac = float(s)
            longer = max(x2 - x1, y2 - y1)
            pad_x = pad_y = int(round(frac * longer))
    else:
        frac = float(pad_spec)
        longer = max(x2 - x1, y2 - y1)
        pad_x = pad_y = int(round(frac * longer))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)
    return nx1, ny1, nx2, ny2


# --- CLI ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.auto_crop",
        description="Batch auto-crop images using a YOLO detector.",
    )
    p.add_argument("--input", default="./input", help="Input folder (default: ./input)")
    p.add_argument("--output", default="./output", help="Output folder (default: ./output)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Ultralytics weights (default: yolov8n.pt; auto-downloaded)")
    p.add_argument("--classes", default=None,
                   help="Comma-separated COCO class names to keep (default: all)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Minimum detection confidence (default: 0.25)")
    p.add_argument("--pad", default="0.10",
                   help='Padding around each detection: fraction (e.g. "0.10") '
                        'or absolute pixels (e.g. "20px"). Default: 0.10')
    p.add_argument("--device", default=None,
                   help="cpu | cuda | mps (default: Ultralytics auto-detect)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without creating files")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logging")
    return p


def _setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
    )


def run(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    model_name: str = "yolov8n.pt",
    classes: Optional[Sequence[str]] = None,
    conf: float = 0.25,
    pad: PadSpec = 0.10,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Programmatic entry point. Returns process exit code (0 = ok, 1 = all failed)."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    class_ids = parse_classes(",".join(classes)) if classes else None

    images = scan_input_folder(input_folder)
    if not images:
        log.error("No images found in %s", input_folder)
        return 1

    # Lazy import: keeps `import src.auto_crop` cheap (and possible) when
    # ultralytics/torch aren't installed — useful for unit tests.
    from ultralytics import YOLO  # type: ignore
    model = YOLO(model_name)

    total_crops = 0
    skipped = 0
    start = time.time()

    for img_path in images:
        try:
            img = load_image(img_path)
        except Exception as e:  # noqa: BLE001 — log and continue on per-image error
            log.warning("Skipping %s: load failed (%s)", img_path.name, e)
            skipped += 1
            continue

        try:
            predict_kwargs = dict(conf=conf, classes=class_ids, verbose=False)
            if device is not None:
                predict_kwargs["device"] = device
            results = model.predict(img, **predict_kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping %s: inference failed (%s)", img_path.name, e)
            skipped += 1
            continue

        per_class_count: dict = {}
        per_class_index: dict = {}

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else f"class{cls_id}"
                xyxy = boxes.xyxy[i].tolist()
                raw_box = tuple(int(round(v)) for v in xyxy)
                padded = pad_box(raw_box, pad, img.size)
                if padded[2] - padded[0] <= 0 or padded[3] - padded[1] <= 0:
                    continue

                per_class_index[cls_name] = per_class_index.get(cls_name, 0) + 1
                idx = per_class_index[cls_name]
                if not dry_run:
                    save_crop(img, output_folder, img_path.name, idx, padded, label=cls_name)

                per_class_count[cls_name] = per_class_count.get(cls_name, 0) + 1
                total_crops += 1

        n = sum(per_class_count.values())
        if per_class_count:
            breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(per_class_count.items()))
            log.info("%s: %d crops (%s)", img_path.name, n, breakdown)
        else:
            log.info("%s: 0 crops", img_path.name)

    elapsed = time.time() - start
    log.info(
        "Done: %d image(s), %d crop(s), %d skipped, %.2fs",
        len(images), total_crops, skipped, elapsed,
    )

    return 1 if skipped == len(images) else 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    _setup_logging(args.quiet)
    try:
        class_ids = parse_classes(args.classes)  # validate early, fail fast
    except ValueError as e:
        log.error("%s", e)
        return 2
    # Reconstruct the class-name list for the run() API (simpler signature there).
    class_names = (
        [COCO_NAMES[i] for i in class_ids] if class_ids is not None else None
    )
    return run(
        input_folder=args.input,
        output_folder=args.output,
        model_name=args.model,
        classes=class_names,
        conf=args.conf,
        pad=args.pad,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
