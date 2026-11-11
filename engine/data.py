"""Dataset loading from an on-disk VisionLLM project.

The original files are written to disk byte-for-byte at upload time and are
never rewritten. Training reads them, letterboxes them in memory, and applies
only reversible photometric augmentation. Nothing here crops or squashes an
image, and nothing overwrites a source file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset

from .letterbox import letterbox_image, letterbox_mask


def load_manifest(root: Path) -> Dict:
    path = root / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"no dataset uploaded for this project ({path})")
    return json.loads(path.read_text())


def class_names(manifest: Dict) -> List[str]:
    return list(manifest.get("classes") or [])


def _image_path(root: Path, item: Dict) -> Path:
    return root / "images" / item["file"]


def _boxes_pixels(item: Dict) -> Tuple[np.ndarray, np.ndarray]:
    boxes = []
    labels = []
    for box in item.get("boxes") or []:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        boxes.append([x * item["width"], y * item["height"], (x + w) * item["width"], (y + h) * item["height"]])
        labels.append(int(box["cls"]))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class DetectionDataset(Dataset):
    def __init__(self, root: Path, manifest: Dict, split: str, size: int, augment: bool = False):
        self.root = root
        self.size = size
        self.augment = augment
        self.items = [
            item
            for item in manifest["images"]
            if item.get("split") == split and (augment or item.get("boxes"))
        ]
        self.includes_empty = any(not item.get("boxes") for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(_image_path(self.root, item)).convert("RGB")
        width, height = image.size
        boxes, labels = _boxes_pixels(item)
        boxes = boxes.copy()

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        if self.augment and random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))

        canvas, ratio, pad_x, pad_y = letterbox_image(image, self.size)
        boxes = boxes * ratio
        boxes[:, [0, 2]] += pad_x
        boxes[:, [1, 3]] += pad_y
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.size)

        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.from_numpy(labels.astype(np.int64)),
        }
        meta = {
            "image_id": item.get("id", str(index)),
            "width": width,
            "height": height,
            "ratio": ratio,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }
        return tensor, target, meta


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch], 0)
    return images, [b[1] for b in batch], [b[2] for b in batch]


class ClassificationDataset(Dataset):
    def __init__(self, root: Path, manifest: Dict, split: str, size: int, augment: bool = False):
        self.root = root
        self.size = size
        self.augment = augment
        self.items = [item for item in manifest["images"] if item.get("split") == split and item.get("label") is not None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(_image_path(self.root, item)).convert("RGB")
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment and random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        canvas, _, _, _ = letterbox_image(image, self.size)
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)).float() / 255.0
        return tensor, int(item["label"]), {"image_id": item.get("id", str(index))}


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, manifest: Dict, split: str, size: int, augment: bool = False):
        self.root = root
        self.size = size
        self.augment = augment
        self.num_classes = len(manifest.get("classes") or []) + 1
        self.items = [item for item in manifest["images"] if item.get("split") == split and item.get("shapes")]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        path = _image_path(self.root, item)
        image = Image.open(path).convert("RGB")
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for shape in item.get("shapes") or []:
            cls = int(shape["cls"]) + 1
            points = [(float(p[0]) * width, float(p[1]) * height) for p in shape.get("points") or []]
            if len(points) >= 3:
                draw.polygon(points, fill=cls)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        canvas, _, _, _ = letterbox_image(image, self.size)
        mask_canvas = letterbox_mask(mask, self.size)
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)).float() / 255.0
        target = torch.from_numpy(np.asarray(mask_canvas, dtype=np.int64))
        return tensor, target, {"image_id": item.get("id", str(index)), "width": width, "height": height}


def split_counts(manifest: Dict) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "unassigned": 0}
    for item in manifest.get("images", []):
        split = item.get("split") or "unassigned"
        counts[split] = counts.get(split, 0) + 1
    return counts
