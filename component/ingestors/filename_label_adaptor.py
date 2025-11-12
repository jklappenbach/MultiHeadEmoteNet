from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple
import logging

from component.ingestors.adaptor_base import BaseAdaptor, is_image_file

TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
logger = logging.getLogger(__name__)


class FilenameLabelAdaptor(BaseAdaptor):
    """
    Derive label tokens from filenames like 'happy_001.png' or 'img-angry-42.jpg'.
    Falls back to parent folder if no meaningful token is found.
    """

    name = "FilenameLabelAdaptor"

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        root = Path(root)
        for img in root.rglob("*"):
            if not is_image_file(img):
                continue
            fname = img.stem.lower()
            tokens = [t for t in TOKEN_RE.split(fname) if t]
            label = None
            # Heuristic: first token is often the class name
            if tokens:
                label = tokens[0]
            if not label:
                label = img.parent.name
            logger.debug("label: %s, path: %s", label, str(img))
            yield img, label
