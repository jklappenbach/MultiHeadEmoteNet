from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import logging

from component.ingestors.adaptor_base import BaseAdaptor, is_image_file

logger = logging.getLogger(__name__)

SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test", "final", "final_test", "final test", "public_test"}


class GenericImageFolderAdaptor(BaseAdaptor):
    name = "GenericImageFolderAdaptor"

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        root = Path(root)
        for img in root.rglob("*"):
            if not is_image_file(img):
                continue
            # Determine label from nearest non-split parent directory
            label = self._infer_label_from_parents(img.parent)
            if not label:
                # fallback to immediate parent name anyway
                label = img.parent.name
            logger.debug("label: %s, path: %s", label, str(img))
            yield img, label

    def _infer_label_from_parents(self, d: Path) -> str:
        cur = d
        while cur is not None and cur != cur.parent:
            name = cur.name.lower().replace("-", " ")
            if name not in SPLIT_DIR_NAMES:
                return cur.name
            cur = cur.parent
        return d.name
