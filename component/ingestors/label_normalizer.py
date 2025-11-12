from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

# Canonical IDs as specified by the user (global, stable IDs)
# Note: canonical keys are lowercase, spaces replaced with underscores
_CANONICAL_ID_MAP_DEFAULT: Dict[str, int] = {
    "angry": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "natural": 5,   # user chose "Natural" instead of "Neutral"
    "sad": 6,
    "surprise": 8,
    "sleepy": 7,
    "ahegao": 9,
}

# Simple synonym/alias map; keys are lowercase tokens, values are canonical keys above.
# This also implements the "map to nearest canonical class via alias" behavior.
_SYNONYMS: Dict[str, str] = {
    # Basic variants
    "anger": "angry",
    "angry_face": "angry",
    "happiness": "happy",
    "joy": "happy",
    "smile": "happy",
    "smiling": "happy",
    "sadness": "sad",
    "surprised": "surprise",
    "surprised_face": "surprise",
    "disgusted": "disgust",
    "fearful": "fear",
    # Neutral/Natural variations
    "neutral": "natural",
    "neutrality": "natural",
    "calm": "natural",
    "normal": "natural",
    # Sleepy variations
    "sleep": "sleepy",
    "sleeping": "sleepy",
    "drowsy": "sleepy",
}


def normalize_label(raw: str) -> str:
    """Normalize a raw label string to a canonical key.

    Returns a canonical key if recognized; otherwise returns a cleaned token
    (lowercased, spaces/hyphens to underscores) so callers can decide to skip.
    """
    if raw is None:
        return ""
    t = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    # singularize simple plurals
    if t.endswith("es") and t[:-2] in _CANONICAL_ID_MAP_DEFAULT:
        t = t[:-2]
    if t.endswith("s") and t[:-1] in _CANONICAL_ID_MAP_DEFAULT:
        t = t[:-1]
    # alias mapping
    if t in _SYNONYMS:
        t = _SYNONYMS[t]
    return t


def load_or_init_registry(path: Path) -> Dict[str, int]:
    """Load the SSOT label registry from JSON, or initialize with defaults if missing.
    The registry is a mapping of canonical_name -> global_id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # If existing file is empty or invalid, fall back to defaults
            if isinstance(data, dict) and data:
                return {str(k).lower(): int(v) for k, v in data.items()}
    # Initialize with defaults
    with path.open("w", encoding="utf-8") as f:
        json.dump(_CANONICAL_ID_MAP_DEFAULT, f, indent=2, ensure_ascii=False)
    return dict(_CANONICAL_ID_MAP_DEFAULT)


def save_registry(path: Path, mapping: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def build_reverse_map(id_map: Dict[str, int]) -> Dict[int, str]:
    rev: Dict[int, str] = {}
    for name, _id in id_map.items():
        rev[int(_id)] = str(name)
    return rev
