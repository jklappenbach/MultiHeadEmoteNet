from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import logging

from component.ingestors.adaptor_base import BaseAdaptor, is_image_file

logger = logging.getLogger(__name__)

SPLIT_DIRS = {"train", "val", "valid", "validation", "test", "public_test"}


class FERAdaptor(BaseAdaptor):
    """
    Handles datasets with split directories (train/val/test) containing class subfolders.
    Flattens by ignoring the split directory and using the immediate class folder as label.
    """

    name = "FERAdaptor"

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        root = Path(root)
        for split in SPLIT_DIRS:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                label = cls_dir.name
                for img in cls_dir.rglob("*"):
                    if is_image_file(img):
                        logger.debug("%s: %s", label, str(img))
                        yield img, label
        # Fallback: if structure differs, just walk images and take parent folder as label
        for img in root.rglob("*"):
            if is_image_file(img):
                label = img.parent.name
                logger.debug("label: %s, path: %s", label, str(img))
                yield img, label
