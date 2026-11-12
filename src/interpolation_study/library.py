"""Optional Pillow reference implementations for the library-baseline stage.

This module is intentionally isolated from ``resize.py``.  The from-scratch
resampler does not import or call Pillow.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PILLOW_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}
PILLOW_METHODS = tuple(PILLOW_FILTERS)


def resize_with_pillow(image: np.ndarray, size: tuple[int, int], method: str) -> np.ndarray:
    """Resize with Pillow; only used as an optional external baseline."""
    if method not in PILLOW_FILTERS:
        raise ValueError(f"Pillow baseline does not provide {method!r}; choose from {', '.join(PILLOW_METHODS)}")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be grayscale or have channels in the last axis")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    squeeze_channel = array.ndim == 3 and array.shape[2] == 1
    if squeeze_channel:
        array = array[..., 0]
    pil_image = Image.fromarray(array)
    result = np.asarray(pil_image.resize(size, resample=PILLOW_FILTERS[method])).astype(np.float64)
    return result[..., None] if squeeze_channel else result
