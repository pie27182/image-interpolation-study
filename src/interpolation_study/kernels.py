"""Interpolation reconstruction kernels.

The cubic kernel is Keys' cubic convolution kernel with a=-0.5 (Catmull-Rom).
Lanczos-3 is the windowed-sinc kernel with support radius 3.
"""

from __future__ import annotations

import numpy as np


def linear(distance: np.ndarray) -> np.ndarray:
    """Triangle/hat kernel, with support radius 1."""
    x = np.abs(distance)
    return np.maximum(1.0 - x, 0.0)


def keys_cubic(distance: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys cubic convolution kernel.

    Args:
        distance: Distances from an input sample, in source-pixel units.
        a: Cubic parameter. -0.5 gives the Catmull-Rom variant.
    """
    x = np.abs(distance)
    x2 = x * x
    x3 = x2 * x
    first = ((a + 2.0) * x3) - ((a + 3.0) * x2) + 1.0
    second = (a * x3) - (5.0 * a * x2) + (8.0 * a * x) - (4.0 * a)
    return np.where(x < 1.0, first, np.where(x < 2.0, second, 0.0))


def sinc(distance: np.ndarray) -> np.ndarray:
    """Normalized sinc function, sin(pi*x)/(pi*x), with sinc(0)=1."""
    return np.sinc(distance)


def lanczos3(distance: np.ndarray) -> np.ndarray:
    """Lanczos windowed-sinc kernel with support radius 3."""
    x = np.abs(distance)
    return np.where(x < 3.0, sinc(distance) * sinc(distance / 3.0), 0.0)
