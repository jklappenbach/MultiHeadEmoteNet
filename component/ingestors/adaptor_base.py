from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


class BaseAdaptor:
    """
    Base interface for dataset ingestors.

    Subclasses should implement iterate(root) yielding tuples of (image_path, raw_label),
    where image_path is a str or Path to an image file and raw_label is a str label
    deduced from folders, filenames, annotations, etc.
    """

    name: str = "BaseLoader"

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        raise NotImplementedError

    # Utility for subclasses
    @staticmethod
    def walk_images(root: Path) -> Generator[Path, None, None]:
        for p in root.rglob("*"):
            if is_image_file(p):
                yield p
