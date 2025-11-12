from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import logging

from component.ingestors.adaptor_base import BaseAdaptor, is_image_file

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - allow running without PyYAML, fallback behavior
    yaml = None

logger = logging.getLogger(__name__)


class YOLOEmotionAdaptor(BaseAdaptor):
    """
    Attempts to read a YOLO-style data.yaml to map class ids to names.
    Falls back to folder names (parent directory of images) if yaml is not present.
    """

    name = "YOLOEmotionAdaptor"

    def _load_yaml_classes(self, root: Path) -> Dict[int, str] | None:
        if yaml is None:
            return None
        # Common locations: root/data.yaml or nested under root
        candidates = list(root.glob("data.yaml")) + list(root.rglob("*/data.yaml"))
        for y in candidates:
            try:
                data = yaml.safe_load(y.read_text(encoding="utf-8"))
                names = data.get("names") if isinstance(data, dict) else None
                if isinstance(names, dict):
                    # {id: name}
                    return {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list):
                    return {i: str(n) for i, n in enumerate(names)}
            except Exception:
                continue
        return None

    def iterate(self, root: Path | str) -> Iterable[Tuple[Path, str]]:
        root = Path(root)

        # Fixed mapping as specified
        id_to_name: Dict[int, str] = {
            0: "Angry",
            1: "Contempt",
            2: "Disgust",
            3: "Fear",
            4: "Happy",
            5: "Neutral",
            6: "Sad",
            7: "Sleepy",
            8: "Surprise",
        }

        # Expected YOLO split structure: <root>/{train,valid,test}/{images,labels}
        found_any = False
        for split in ("train", "valid", "test"):
            labels_dir = root / split / "labels"
            images_dir = root / split / "images"
            if not (labels_dir.exists() and images_dir.exists()):
                continue
            found_any = True
            for lbl in labels_dir.rglob("*.txt"):
                try:
                    lines = lbl.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                # Extract the first valid class id (0-8) from the first non-empty line
                cls_id = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except Exception:
                        continue
                    if 0 <= cid <= 8:
                        cls_id = cid
                        break
                if cls_id is None:
                    continue
                cls_name = id_to_name.get(cls_id)
                if not cls_name:
                    continue
                # Match image by stem in the corresponding split/images directory
                stem = lbl.stem
                for img in images_dir.rglob(stem + ".*"):
                    if is_image_file(img):
                        logger.debug("%s: %s", cls_name, str(img))
                        yield img, cls_name
        if found_any:
            return

        # Fallbacks for unexpected layouts
        # 1) Try non-split root-level images/labels with fixed mapping
        labels_dir = root / "labels"
        images_dir = root / "images"
        if labels_dir.exists() and images_dir.exists():
            for lbl in labels_dir.rglob("*.txt"):
                try:
                    lines = lbl.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                cls_id = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except Exception:
                        continue
                    if 0 <= cid <= 8:
                        cls_id = cid
                        break
                if cls_id is None:
                    continue
                cls_name = id_to_name.get(cls_id)
                if not cls_name:
                    continue
                stem = lbl.stem
                for img in images_dir.rglob(stem + ".*"):
                    if is_image_file(img):
                        logger.debug("%s: %s", cls_name, str(img))
                        yield img, cls_name
            return
