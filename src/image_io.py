"""Folder scanning, image loading, and crop-save with collision-safe naming."""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_input_folder(folder) -> List[Path]:
    """Return a sorted list of image files in `folder`. Missing folder returns []."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path) -> Image.Image:
    """Open an image and apply EXIF orientation so it displays the right way up."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.load()
    return img


def next_available_filename(desired_name: str, existing: Iterable[str]) -> str:
    """Return `desired_name`, or a `_2`/`_3`/... suffixed variant that does not collide
    with anything in `existing`. Suffix is inserted before the extension."""
    existing_set = set(existing)
    if desired_name not in existing_set:
        return desired_name
    p = Path(desired_name)
    stem, suffix = p.stem, p.suffix
    n = 2
    while True:
        candidate = f"{stem}_{n}{suffix}"
        if candidate not in existing_set:
            return candidate
        n += 1


def save_crop(
    image: Image.Image,
    output_folder,
    source_name: str,
    crop_index: int,
    box: Tuple[int, int, int, int],
    label: Optional[str] = None,
) -> Path:
    """Crop `image` to `box` and save it under `output_folder`. Naming scheme:
    `<stem>_crop<N><ext>` by default, or `<stem>_<label>_crop<N><ext>` when `label`
    is given (used by v2 to embed the detected class). Returns the actual path
    written. Creates the output folder if missing."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    src = Path(source_name)
    if label:
        desired = f"{src.stem}_{label}_crop{crop_index}{src.suffix}"
    else:
        desired = f"{src.stem}_crop{crop_index}{src.suffix}"
    existing = {p.name for p in output_folder.iterdir() if p.is_file()}
    final_name = next_available_filename(desired, existing)
    out_path = output_folder / final_name

    cropped = image.crop(box)
    # JPEG can't hold an alpha channel — flatten to RGB to avoid a save error.
    if src.suffix.lower() in (".jpg", ".jpeg") and cropped.mode in ("RGBA", "LA", "P"):
        cropped = cropped.convert("RGB")
    cropped.save(out_path)
    return out_path
