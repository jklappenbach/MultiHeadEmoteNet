from __future__ import annotations

from typing import Type, Dict

from .generic_image_adaptor import GenericImageFolderAdaptor
from .fer_adaptor import FERAdaptor
from .filename_label_adaptor import FilenameLabelAdaptor
from .yolo_emotion_adaptor import YOLOEmotionAdaptor
from .fer2013_adaptor import FER2013Adaptor

LOADER_BY_NAME: Dict[str, Type] = {
    "GenericImageFolderAdaptor": GenericImageFolderAdaptor,
    "FERAdaptor": FERAdaptor,
    "FilenameLabelAdaptor": FilenameLabelAdaptor,
    "YOLOEmotionAdaptor": YOLOEmotionAdaptor,
    "FER2013Adaptor": FER2013Adaptor,
}


def resolve_adaptor(name: str):
    if name not in LOADER_BY_NAME:
        raise KeyError(f"Unknown loader class name: {name}. Available: {sorted(LOADER_BY_NAME.keys())}")
    return LOADER_BY_NAME[name]


def resolve_loader():
    return None