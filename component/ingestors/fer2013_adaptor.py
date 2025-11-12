from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import logging

from component.ingestors.adaptor_base import BaseAdaptor, is_image_file

logger = logging.getLogger(__name__)


class FER2013Adaptor(BaseAdaptor):
    """
    Stub loader: recognizes fer2013.csv if present. If only CSV is present, no images are yielded.
    If images exist (some variants ship pre-extracted images), falls back to scanning and inferring labels from parent folders.
    """

    name = "FER2013Adaptor"

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        root = Path(root)
        csv_path = root / "fer2013.csv"
        if csv_path.exists():
            # CSV-based format; image reconstruction not implemented here.
            # Users can pre-extract images using external tools; we will fall back to any images found.
            pass
        for img in root.rglob("*"):
            if is_image_file(img):
                label = img.parent.name
                logger.debug("label: %s, path: %s", label, str(img))
                yield img, label
