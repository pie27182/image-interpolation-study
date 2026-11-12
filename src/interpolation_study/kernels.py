"""Scalar reconstruction kernels used by the from-scratch resampler.

Only NumPy's elementary array operations are used.  No image-resizing API is
called anywhere in this module.
"""

from __future__ import annotations

import numpy as np


def nearest(distance: np.ndarray) -> np.ndarray:
    """Unit-width box reconstruction kernel (mainly for analysis/plots)."""
    return (np.abs(distance) < 0.5).astype(np.float64)


def linear(distance: np.ndarray) -> np.ndarray:
    """Linear/hat kernel, support radius 1."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    return np.maximum(1.0 - x, 0.0)


def keys_cubic(distance: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys' cubic-convolution kernel (a=-0.5 is Catmull-Rom-like)."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    x2 = x * x
    x3 = x2 * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def mitchell_netravali(
    distance: np.ndarray, b: float = 1.0 / 3.0, c: float = 1.0 / 3.0
) -> np.ndarray:
    """Mitchell-Netravali family of cubic reconstruction filters.

    The common compromise is B=C=1/3. Unlike Keys cubic convolution, this
    default is not cardinal: its value at the origin is 8/9, not 1.
    """
    x = np.abs(np.asarray(distance, dtype=np.float64))
    x2 = x * x
    x3 = x2 * x
    inner = (
        (12.0 - 9.0 * b - 6.0 * c) * x3
        + (-18.0 + 12.0 * b + 6.0 * c) * x2
        + (6.0 - 2.0 * b)
    ) / 6.0
    outer = (
        (-b - 6.0 * c) * x3
        + (6.0 * b + 30.0 * c) * x2
        + (-12.0 * b - 48.0 * c) * x
        + (8.0 * b + 24.0 * c)
    ) / 6.0
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def cubic_bspline(distance: np.ndarray) -> np.ndarray:
    """Centered cubic B-spline basis function, support radius 2.

    A B-spline basis is not cardinal.  The resampler therefore prefilters the
    sample array to obtain spline coefficients before evaluating this basis.
    """
    x = np.abs(np.asarray(distance, dtype=np.float64))
    x2 = x * x
    inner = (4.0 - 6.0 * x2 + 3.0 * x2 * x) / 6.0
    outer = np.maximum(2.0 - x, 0.0) ** 3 / 6.0
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def sinc(distance: np.ndarray) -> np.ndarray:
    """Normalized sinc, sin(pi*x)/(pi*x), with the removable limit at zero."""
    return np.sinc(np.asarray(distance, dtype=np.float64))


def lanczos(distance: np.ndarray, radius: int = 3) -> np.ndarray:
    """Windowed-sinc (Lanczos) kernel with a configurable integer radius."""
    if radius < 1:
        raise ValueError("Lanczos radius must be a positive integer")
    x = np.asarray(distance, dtype=np.float64)
    ax = np.abs(x)
    values = sinc(x) * sinc(x / float(radius))
    return np.where(ax < radius, values, 0.0)


def lanczos3(distance: np.ndarray) -> np.ndarray:
    """Convenience wrapper for the radius-three Lanczos kernel."""
    return lanczos(distance, radius=3)
