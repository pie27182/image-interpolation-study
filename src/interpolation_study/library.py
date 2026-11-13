"""Необязательные эталонные реализации на Pillow для этапа библиотечного сравнения.

Этот модуль намеренно отделён от ``resize.py``. Самостоятельный ресэмплер
не импортирует Pillow и не вызывает его функции.
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
    """Изменить размер средствами Pillow; используется только как необязательная внешняя базовая реализация."""
    if method not in PILLOW_FILTERS:
        raise ValueError(f"Базовая реализация Pillow не предоставляет {method!r}; выберите один из методов: {', '.join(PILLOW_METHODS)}")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("изображение должно быть в градациях серого или иметь каналы по последней оси")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    squeeze_channel = array.ndim == 3 and array.shape[2] == 1
    if squeeze_channel:
        array = array[..., 0]
    pil_image = Image.fromarray(array)
    result = np.asarray(pil_image.resize(size, resample=PILLOW_FILTERS[method])).astype(np.float64)
    return result[..., None] if squeeze_channel else result
