"""Pillow-based reference implementations for the first experimental stage."""

from __future__ import annotations

import numpy as np
from PIL import Image

PILLOW_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
}


def resize_with_pillow(image: np.ndarray, size: tuple[int, int], method: str) -> np.ndarray:
    """Resize with Pillow; accepts uint8 grayscale, RGB or RGBA arrays."""
    if method not in PILLOW_FILTERS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(PILLOW_FILTERS)}")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    return np.asarray(pil_image.resize(size, resample=PILLOW_FILTERS[method])).astype(np.float64)
