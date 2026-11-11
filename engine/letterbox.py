"""Aspect-preserving preprocessing shared by training and inference.

Images are never cropped or squashed. Every model input is produced by
letterboxing into a square canvas, with the geometry recorded so predictions
can be mapped back onto the untouched original.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

PAD_VALUE = 114
PAD_RGB = (PAD_VALUE, PAD_VALUE, PAD_VALUE)


def letterbox_size(width: int, height: int, size: int) -> Tuple[int, int, int, int]:
    """Return (new_w, new_h, pad_x, pad_y) for an aspect-preserving fit."""
    ratio = min(size / width, size / height)
    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    new_w = max(1, min(size, new_w))
    new_h = max(1, min(size, new_h))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    return new_w, new_h, pad_x, pad_y


def letterbox_image(image: Image.Image, size: int, resample=Image.BILINEAR) -> Tuple[Image.Image, float, int, int]:
    """Letterbox a PIL image. Returns (canvas, ratio, pad_x, pad_y)."""
    width, height = image.size
    new_w, new_h, pad_x, pad_y = letterbox_size(width, height, size)
    resized = image if (new_w, new_h) == (width, height) else image.resize((new_w, new_h), resample)
    canvas = Image.new("RGB", (size, size), PAD_RGB)
    canvas.paste(resized, (pad_x, pad_y))
    ratio = new_w / width
    return canvas, ratio, pad_x, pad_y


def letterbox_mask(mask: Image.Image, size: int) -> Image.Image:
    """Letterbox a single-channel label mask with nearest-neighbour sampling."""
    width, height = mask.size
    new_w, new_h, pad_x, pad_y = letterbox_size(width, height, size)
    resized = mask if (new_w, new_h) == (width, height) else mask.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)
