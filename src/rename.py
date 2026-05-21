"""Batch image-rename utility.

Glob-match files in a folder, apply a template using {n}, {stem}, {ext} tokens,
and rename in place. Safe by design:
  - Two-pass execution via temp names, so cycles like a.jpg <-> b.jpg work.
  - Refuses to overwrite any destination that exists outside the rename set.
  - --dry-run prints the plan without touching the filesystem.

CLI:
  python -m src.rename --folder ./input --match "IMG_*.jpg" --to "vacation_{n:03d}{ext}"
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .image_io import IMAGE_EXTENSIONS

log = logging.getLogger("rename")


# --- Pure helpers (unit-tested without touching the filesystem) --------------

def format_name(template: str, n: int, stem: str, ext: str) -> str:
    """Render a new filename from `template`. Supported tokens:
      - {n}    running counter; format specs work (e.g. {n:03d} -> 001)
      - {stem} the original file's stem
      - {ext}  the original file's extension, including the leading dot

    Raises ValueError on unknown tokens or malformed format specs.
    """
    try:
        return template.format(n=n, stem=stem, ext=ext)
    except KeyError as e:
        raise ValueError(
            f"Unknown template token: {e}. Valid tokens: {{n}}, {{stem}}, {{ext}}"
        ) from None
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid template '{template}': {e}") from None


# --- Filesystem-aware operations --------------------------------------------

def select_files(folder, match: str = "*", sort: str = "name") -> List[Path]:
    """List image files in `folder` matching the glob `match`, sorted as requested."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and fnmatch.fnmatch(p.name, match)
    ]
    if sort == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name)
    return files


def plan_renames(
    files: List[Path],
    template: str,
    start: int = 1,
) -> List[Tuple[Path, Path]]:
    """Build a validated (src -> dst) rename plan.

    Validation rules:
      - No two sources may map to the same destination filename.
      - No destination may collide with an existing file that is NOT itself in
        the rename set (so we never silently overwrite an unrelated file).
    """
    if not files:
        return []
    folder = files[0].parent
    source_resolved = {f.resolve() for f in files}
    targets: dict = {}
    plan: List[Tuple[Path, Path]] = []

    for i, src in enumerate(files):
        n = start + i
        new_name = format_name(template, n=n, stem=src.stem, ext=src.suffix)
        new_path = folder / new_name

        if new_name in targets:
            raise ValueError(
                f"Template produces duplicate name '{new_name}' for both "
                f"'{targets[new_name].name}' and '{src.name}'"
            )
        targets[new_name] = src

        if new_path.exists() and new_path.resolve() not in source_resolved:
            raise ValueError(
                f"Destination '{new_name}' already exists and is not in the "
                f"rename set; refusing to overwrite. Move it aside or change --to."
            )
        plan.append((src, new_path))
    return plan


def apply_renames(plan: List[Tuple[Path, Path]]) -> int:
    """Execute the rename plan. Two-pass so cycles work:
      pass 1: src -> .rename_<token>_<i><ext>
      pass 2: temp -> final dst
    Returns the number of files actually renamed (no-ops excluded).
    """
    token = uuid.uuid4().hex[:8]
    temps: List[Tuple[Path, Path]] = []
    for i, (src, dst) in enumerate(plan):
        if src == dst:
            continue
        tmp = src.with_name(f".rename_{token}_{i}{src.suffix}")
        src.rename(tmp)
        temps.append((tmp, dst))
    for tmp, dst in temps:
        tmp.rename(dst)
    return len(temps)


# --- CLI / programmatic entry -----------------------------------------------

def run(
    folder,
    match: str = "*",
    template: str = "",
    start: int = 1,
    sort: str = "name",
    dry_run: bool = False,
) -> int:
    """Programmatic entry. Returns process exit code (0 ok, 2 on bad input)."""
    if not template:
        log.error("--to template is required")
        return 2
    folder = Path(folder)
    files = select_files(folder, match=match, sort=sort)
    if not files:
        log.warning("No image files matched '%s' in %s", match, folder)
        return 0

    try:
        plan = plan_renames(files, template, start=start)
    except ValueError as e:
        log.error("%s", e)
        return 2

    for src, dst in plan:
        if src.name == dst.name:
            log.info("%s (unchanged)", src.name)
        else:
            log.info("%s -> %s", src.name, dst.name)

    if dry_run:
        log.info("(dry-run: no files changed)")
        return 0

    n_renamed = apply_renames(plan)
    log.info("Renamed %d file(s)", n_renamed)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.rename",
        description="Batch-rename image files in a folder using a template.",
    )
    p.add_argument("--folder", default="./input", help="Folder to operate on (default: ./input)")
    p.add_argument("--match", default="*", help='Glob to filter files (default: "*")')
    p.add_argument("--to", required=True, dest="template",
                   help='Filename template. Tokens: {n}, {stem}, {ext}. '
                        'Example: "vacation_{n:03d}{ext}"')
    p.add_argument("--start", type=int, default=1, help="Counter start value (default: 1)")
    p.add_argument("--sort", choices=("name", "mtime"), default="name",
                   help="Sort order before numbering (default: name)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without renaming anything")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logging")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    return run(
        folder=args.folder,
        match=args.match,
        template=args.template,
        start=args.start,
        sort=args.sort,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
