"""Image resizing implemented from interpolation kernels using NumPy only.

Pillow/OpenCV resize functions are deliberately not used in this module.
Images are sampled using the common half-pixel-centre mapping:

    source_position = (destination_index + 0.5) * source_size / destination_size - 0.5

The resize is separable: horizontal filtering followed by vertical filtering.
For bilinear, bicubic and Lanczos-3, the kernel support is widened while
shrinking to provide a simple scale-aware low-pass filter and reduce aliasing.
Nearest-neighbour remains point sampling.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import kernels

Kernel = Callable[[np.ndarray], np.ndarray]
METHODS = ("nearest", "bilinear", "bicubic", "lanczos3")
_KERNELS: dict[str, tuple[Kernel, float]] = {
    "bilinear": (kernels.linear, 1.0),
    "bicubic": (kernels.keys_cubic, 2.0),
    "lanczos3": (kernels.lanczos3, 3.0),
}


def _as_image_array(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D channel array")
    if array.ndim == 3 and array.shape[2] < 1:
        raise ValueError("image must contain at least one channel")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("image dimensions must be non-zero")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must contain numeric values")
    return array.astype(np.float64, copy=False)


def _axis_contributions(
    source_length: int,
    destination_length: int,
    kernel: Kernel,
    radius: float,
    antialias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute clipped source indices and normalized weights per output pixel."""
    scale = source_length / destination_length
    filter_scale = max(1.0, scale) if antialias else 1.0
    support = radius * filter_scale
    positions = (np.arange(destination_length, dtype=np.float64) + 0.5) * scale - 0.5

    # The union width is fixed for this axis; zero-weight padding makes it vectorizable.
    max_taps = int(np.ceil(2.0 * support)) + 2
    indices = np.zeros((destination_length, max_taps), dtype=np.intp)
    weights = np.zeros((destination_length, max_taps), dtype=np.float64)

    for out_index, position in enumerate(positions):
        first = int(np.ceil(position - support))
        last = int(np.floor(position + support))
        sample_positions = np.arange(first, last + 1, dtype=np.intp)
        distances = (position - sample_positions.astype(np.float64)) / filter_scale
        sample_weights = kernel(distances)
        weight_sum = float(sample_weights.sum())
        if abs(weight_sum) < 1e-12:
            # Defensive fallback for pathological numerical cases.
            nearest_index = int(np.floor(position + 0.5))
            indices[out_index, 0] = np.clip(nearest_index, 0, source_length - 1)
            weights[out_index, 0] = 1.0
            continue

        count = min(sample_positions.size, max_taps)
        indices[out_index, :count] = np.clip(sample_positions[:count], 0, source_length - 1)
        weights[out_index, :count] = sample_weights[:count] / weight_sum

    return indices, weights


def _resize_axis(
    image: np.ndarray,
    destination_length: int,
    axis: int,
    kernel: Kernel,
    radius: float,
    antialias: bool,
) -> np.ndarray:
    source_length = image.shape[axis]
    indices, weights = _axis_contributions(
        source_length, destination_length, kernel, radius, antialias
    )
    gathered = np.take(image, indices, axis=axis)
    weight_shape = [1] * (image.ndim + 1)
    weight_shape[axis] = destination_length
    weight_shape[axis + 1] = weights.shape[1]
    weighted = gathered * weights.reshape(weight_shape)
    return weighted.sum(axis=axis + 1)


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    x_positions = (np.arange(width, dtype=np.float64) + 0.5) * (src_w / width) - 0.5
    y_positions = (np.arange(height, dtype=np.float64) + 0.5) * (src_h / height) - 0.5
    x_indices = np.clip(np.floor(x_positions + 0.5).astype(np.intp), 0, src_w - 1)
    y_indices = np.clip(np.floor(y_positions + 0.5).astype(np.intp), 0, src_h - 1)
    return image[y_indices[:, None], x_indices[None, :]]


def resize(
    image: np.ndarray,
    size: tuple[int, int],
    method: str = "bilinear",
    *,
    antialias: bool = True,
) -> np.ndarray:
    """Resize a grayscale or channel image using a hand-coded interpolator.

    Args:
        image: HxW or HxWxC array. Integer inputs are converted to float64.
        size: Target ``(width, height)``.
        method: One of ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos3``.
        antialias: Widen reconstruction kernels when downscaling. This option
            does not alter nearest-neighbour sampling.

    Returns:
        A float64 array with shape ``(height, width)`` or ``(height, width, C)``.
    """
    array = _as_image_array(image)
    if len(size) != 2:
        raise ValueError("size must be a (width, height) pair")
    width, height = size
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise TypeError("target width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")

    if method == "nearest":
        return _resize_nearest(array, int(width), int(height))

    kernel, radius = _KERNELS[method]
    # Separable filtering: x first, then y. Each pass works on float64 values.
    horizontal = _resize_axis(array, int(width), axis=1, kernel=kernel,
                              radius=radius, antialias=antialias)
    return _resize_axis(horizontal, int(height), axis=0, kernel=kernel,
                        radius=radius, antialias=antialias)
