"""Pretrained COCO detectors used for annotation suggestions.

torchvision detection models are BSD-3-Clause and ship no AGPL obligations.
They are used only to propose boxes; the user always reviews the result, and
any accepted box is stored as an ordinary annotation.
"""

from __future__ import annotations

import io
import threading
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

_MODELS: Dict[str, "object"] = {}
_LOCK = threading.Lock()

CATALOG = {
    "fasterrcnn-mobilenet": {
        "label": "Faster R-CNN MobileNet v3 (COCO)",
        "note": "Fast on CPU, 80 everyday object classes. ~75 MB on first download.",
        "license": "BSD-3-Clause",
    },
    "fcos-resnet50": {
        "label": "FCOS ResNet-50 FPN (COCO)",
        "note": "Slower but better on small objects. ~124 MB on first download.",
        "license": "BSD-3-Clause",
    },
}


def _build(name: str):
    import torchvision

    if name == "fcos-resnet50":
        weights = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fcos_resnet50_fpn(weights=weights)
    else:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    return model, list(weights.meta["categories"])


def get_detector(name: str):
    if name not in CATALOG:
        raise ValueError(f"unknown pretrained detector {name!r}")
    with _LOCK:
        if name not in _MODELS:
            _MODELS[name] = _build(name)
        return _MODELS[name]


def is_loaded(name: str) -> bool:
    return name in _MODELS


def _normalise(label: str) -> str:
    return "".join(ch for ch in label.lower().strip() if ch.isalnum())


def suggest(
    image_bytes: bytes,
    classes: Sequence[str],
    model_name: str = "fasterrcnn-mobilenet",
    conf: float = 0.25,
) -> Dict:
    """Run a real COCO detector and map its classes onto the project's classes."""
    import torch

    model, categories = get_detector(model_name)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)).float() / 255.0
    with torch.no_grad():
        output = model([tensor])[0]
    wanted = {_normalise(c): i for i, c in enumerate(classes)}
    boxes: List[Dict] = []
    seen_categories: Dict[str, int] = {}
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        sc = float(score)
        if sc < conf:
            continue
        name = categories[int(label)]
        seen_categories[name] = seen_categories.get(name, 0) + 1
        cls = wanted.get(_normalise(name))
        if cls is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = sorted([max(0.0, x1), min(float(width), x2)])
        y1, y2 = sorted([max(0.0, y1), min(float(height), y2)])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append(
            {
                "cls": cls,
                "score": sc,
                "x": x1 / width,
                "y": y1 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
                "source": name,
            }
        )
    boxes.sort(key=lambda b: b["score"], reverse=True)
    return {
        "width": width,
        "height": height,
        "boxes": boxes,
        "matched": len(boxes),
        "detected_categories": seen_categories,
        "model": model_name,
        "license": CATALOG[model_name]["license"],
    }
