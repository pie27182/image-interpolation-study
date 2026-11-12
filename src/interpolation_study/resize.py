"""Image resampling implemented directly from mathematical kernels.

The only computational dependency is NumPy.  Pillow, OpenCV, SciPy and
scikit-image are not imported or called here.

Default coordinates use the half-pixel-centre convention:

    x_source = (x_destination + 1/2) * source_length / destination_length - 1/2

Filtering is separable (horizontal then vertical).  During reduction, compact
kernels are widened in source coordinates to provide a scale-aware low-pass
filter.  ``area`` instead integrates exact one-dimensional pixel-footprint
overlaps and is especially useful for shrinking images.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import kernels

Kernel = Callable[[np.ndarray], np.ndarray]
METHODS = (
    "nearest",
    "bilinear",
    "bicubic",
    "mitchell",
    "bspline3",
    "lanczos3",
    "area",
)
BOUNDARIES = ("clamp", "reflect", "wrap", "constant")
COORDINATE_MODES = ("half_pixel", "align_corners")
# Limit temporary gather arrays for large images.  The output is evaluated in
# slices along the resized axis so memory use does not scale with the full
# destination length times the kernel support.
_MAX_GATHER_BYTES = 64 * 1024 * 1024

_KERNELS: dict[str, tuple[Kernel, float]] = {
    "bilinear": (kernels.linear, 1.0),
    "bicubic": (kernels.keys_cubic, 2.0),
    "mitchell": (kernels.mitchell_netravali, 2.0),
    "bspline3": (kernels.cubic_bspline, 2.0),
    "lanczos3": (kernels.lanczos3, 3.0),
}


def _as_image_array(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D channel array")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("image dimensions must be non-zero")
    if array.ndim == 3 and array.shape[2] < 1:
        raise ValueError("image must contain at least one channel")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must contain numeric values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("complex-valued images are not supported")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    return array.astype(np.float64, copy=False)


def _source_positions(
    source_length: int, destination_length: int, coordinate_mode: str
) -> tuple[np.ndarray, float]:
    scale = source_length / destination_length
    if coordinate_mode == "half_pixel":
        positions = (np.arange(destination_length, dtype=np.float64) + 0.5) * scale - 0.5
    elif coordinate_mode == "align_corners":
        if destination_length == 1:
            positions = np.array([(source_length - 1.0) / 2.0])
            scale = float(max(source_length, 1))
        else:
            positions = np.linspace(0.0, source_length - 1.0, destination_length)
            scale = (source_length - 1.0) / (destination_length - 1.0)
    else:
        raise ValueError(
            f"unknown coordinate mode {coordinate_mode!r}; choose from {', '.join(COORDINATE_MODES)}"
        )
    return positions, scale


def _map_indices(indices: np.ndarray, length: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Map integer sample indices to the source extent.

    ``reflect`` is whole-sample symmetric reflection (index -1 maps to 1),
    often called reflect-101.  ``constant`` retains a validity mask so that
    out-of-range taps contribute the caller's constant value.
    """
    valid = (indices >= 0) & (indices < length)
    if boundary == "constant":
        return np.clip(indices, 0, length - 1), valid
    if boundary == "clamp":
        return np.clip(indices, 0, length - 1), np.ones(indices.shape, dtype=bool)
    if boundary == "wrap":
        return np.mod(indices, length), np.ones(indices.shape, dtype=bool)
    if boundary == "reflect":
        if length == 1:
            mapped = np.zeros(indices.shape, dtype=np.intp)
        else:
            period = 2 * (length - 1)
            folded = np.mod(indices, period)
            mapped = np.where(folded < length, folded, period - folded)
        return mapped.astype(np.intp, copy=False), np.ones(indices.shape, dtype=bool)
    raise ValueError(f"unknown boundary mode {boundary!r}; choose from {', '.join(BOUNDARIES)}")


def _axis_kernel_contributions(
    source_length: int,
    destination_length: int,
    kernel: Kernel,
    radius: float,
    *,
    antialias: bool,
    boundary: str,
    coordinate_mode: str,
    kernel_kwargs: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute source indices, validity flags and normalized tap weights."""
    positions, scale = _source_positions(source_length, destination_length, coordinate_mode)
    filter_scale = max(1.0, scale) if antialias else 1.0
    support = radius * filter_scale
    max_taps = int(np.ceil(2.0 * support)) + 3
    indices = np.zeros((destination_length, max_taps), dtype=np.intp)
    valid = np.zeros((destination_length, max_taps), dtype=bool)
    weights = np.zeros((destination_length, max_taps), dtype=np.float64)
    kwargs = kernel_kwargs or {}

    for out_index, position in enumerate(positions):
        first = int(np.ceil(position - support))
        last = int(np.floor(position + support))
        source_indices = np.arange(first, last + 1, dtype=np.intp)
        distances = (position - source_indices.astype(np.float64)) / filter_scale
        tap_weights = kernel(distances, **kwargs) if kwargs else kernel(distances)
        total = float(np.sum(tap_weights))
        if not np.isfinite(total) or abs(total) < 1e-12:
            raise ArithmeticError("interpolation kernel produced a near-zero or invalid weight sum")
        tap_weights = tap_weights / total
        mapped, is_valid = _map_indices(source_indices, source_length, boundary)
        count = source_indices.size
        if count > max_taps:
            raise RuntimeError("internal tap allocation was too small")
        indices[out_index, :count] = mapped
        valid[out_index, :count] = is_valid
        weights[out_index, :count] = tap_weights

    return indices, valid, weights


def _axis_area_contributions(
    source_length: int, destination_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact overlap between destination footprints and source pixel cells.

    Source pixel i occupies [i-1/2, i+1/2].  Destination cell j covers the
    corresponding interval [j*s-1/2, (j+1)*s-1/2] in source coordinates.
    Its weights are overlap lengths divided by the destination footprint size.
    """
    scale = source_length / destination_length
    max_taps = int(np.ceil(scale)) + 2
    indices = np.zeros((destination_length, max_taps), dtype=np.intp)
    valid = np.ones((destination_length, max_taps), dtype=bool)
    weights = np.zeros((destination_length, max_taps), dtype=np.float64)
    for out_index in range(destination_length):
        left = out_index * scale - 0.5
        right = (out_index + 1) * scale - 0.5
        first = max(0, int(np.floor(left + 0.5)))
        last = min(source_length - 1, int(np.ceil(right + 0.5) - 1))
        source_indices = np.arange(first, last + 1, dtype=np.intp)
        cell_left = source_indices.astype(np.float64) - 0.5
        cell_right = source_indices.astype(np.float64) + 0.5
        overlap = np.maximum(0.0, np.minimum(right, cell_right) - np.maximum(left, cell_left))
        total = float(overlap.sum())
        if total <= 0.0:
            raise ArithmeticError("area resampling generated an empty pixel footprint")
        count = source_indices.size
        indices[out_index, :count] = source_indices
        weights[out_index, :count] = overlap / total
    return indices, valid, weights


def _resize_axis(
    image: np.ndarray,
    destination_length: int,
    axis: int,
    *,
    kernel: Kernel | None,
    radius: float = 0.0,
    antialias: bool = True,
    boundary: str = "clamp",
    coordinate_mode: str = "half_pixel",
    cval: float = 0.0,
    area: bool = False,
    kernel_kwargs: dict[str, float] | None = None,
) -> np.ndarray:
    source_length = image.shape[axis]
    if area:
        indices, valid, weights = _axis_area_contributions(source_length, destination_length)
    else:
        if kernel is None:
            raise ValueError("a reconstruction kernel is required")
        indices, valid, weights = _axis_kernel_contributions(
            source_length,
            destination_length,
            kernel,
            radius,
            antialias=antialias,
            boundary=boundary,
            coordinate_mode=coordinate_mode,
            kernel_kwargs=kernel_kwargs,
        )

    output_shape = list(image.shape)
    output_shape[axis] = destination_length
    result = np.empty(output_shape, dtype=np.float64)
    other_elements = image.size // source_length
    bytes_per_output_position = max(1, other_elements * indices.shape[1] * 8)
    chunk_size = max(1, min(destination_length, _MAX_GATHER_BYTES // bytes_per_output_position))

    for start in range(0, destination_length, chunk_size):
        stop = min(destination_length, start + chunk_size)
        chunk_indices = indices[start:stop]
        chunk_weights = weights[start:stop]
        gathered = np.take(image, chunk_indices, axis=axis)
        expanded_shape = [1] * (image.ndim + 1)
        expanded_shape[axis] = stop - start
        expanded_shape[axis + 1] = weights.shape[1]
        weights_view = chunk_weights.reshape(expanded_shape)
        if boundary == "constant" and not area:
            valid_view = valid[start:stop].reshape(expanded_shape)
            gathered = np.where(valid_view, gathered, cval)
        filtered = np.sum(gathered * weights_view, axis=axis + 1)
        output_slice = [slice(None)] * image.ndim
        output_slice[axis] = slice(start, stop)
        result[tuple(output_slice)] = filtered

    return result


def _spline_prefilter_axis(image: np.ndarray, axis: int, boundary: str) -> np.ndarray:
    """Solve the cubic B-spline sample-to-coefficient system along one axis.

    At integer knots, beta_3(0)=2/3 and beta_3(+-1)=1/6.  Mirror extension
    gives endpoint off-diagonal 1/3; edge-clamp extension gives endpoint
    diagonal 5/6 and off-diagonal 1/6.  The resulting system is tridiagonal,
    so a vectorized Thomas algorithm solves all rows/columns in O(n) per line.
    """
    if boundary not in ("reflect", "clamp"):
        raise ValueError("bspline3 supports only 'reflect' and 'clamp' boundaries")
    moved = np.moveaxis(image, axis, 0)
    length = moved.shape[0]
    if length == 1:
        return image.copy()
    rhs = moved.reshape(length, -1).copy()

    diagonal = np.full(length, 2.0 / 3.0, dtype=np.float64)
    lower = np.full(length - 1, 1.0 / 6.0, dtype=np.float64)
    upper = np.full(length - 1, 1.0 / 6.0, dtype=np.float64)
    if boundary == "reflect":
        lower[-1] = 1.0 / 3.0
        upper[0] = 1.0 / 3.0
    else:  # constant extension of the spline coefficients at the endpoints
        diagonal[0] = 5.0 / 6.0
        diagonal[-1] = 5.0 / 6.0

    c_prime = np.empty(length - 1, dtype=np.float64)
    d_prime = np.empty_like(rhs)
    denominator = diagonal[0]
    c_prime[0] = upper[0] / denominator
    d_prime[0] = rhs[0] / denominator
    for row in range(1, length):
        denominator = diagonal[row] - lower[row - 1] * c_prime[row - 1]
        if row < length - 1:
            c_prime[row] = upper[row] / denominator
        d_prime[row] = (rhs[row] - lower[row - 1] * d_prime[row - 1]) / denominator

    coefficients = np.empty_like(rhs)
    coefficients[-1] = d_prime[-1]
    for row in range(length - 2, -1, -1):
        coefficients[row] = d_prime[row] - c_prime[row] * coefficients[row + 1]

    restored = coefficients.reshape(moved.shape)
    return np.moveaxis(restored, 0, axis)


def _resize_nearest(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    boundary: str,
    coordinate_mode: str,
) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    x_positions, _ = _source_positions(source_w, width, coordinate_mode)
    y_positions, _ = _source_positions(source_h, height, coordinate_mode)
    x_raw = np.floor(x_positions + 0.5).astype(np.intp)
    y_raw = np.floor(y_positions + 0.5).astype(np.intp)
    x_indices, x_valid = _map_indices(x_raw, source_w, boundary)
    y_indices, y_valid = _map_indices(y_raw, source_h, boundary)
    result = image[y_indices[:, None], x_indices[None, :]]
    if boundary == "constant":
        valid = y_valid[:, None] & x_valid[None, :]
        if image.ndim == 3:
            valid = valid[..., None]
        result = np.where(valid, result, 0.0)
    return result


def resize(
    image: np.ndarray,
    size: tuple[int, int],
    method: str = "bilinear",
    *,
    antialias: bool = True,
    boundary: str = "clamp",
    coordinate_mode: str = "half_pixel",
    cval: float = 0.0,
    keys_a: float = -0.5,
    mitchell_b: float = 1.0 / 3.0,
    mitchell_c: float = 1.0 / 3.0,
) -> np.ndarray:
    """Resize a grayscale or multichannel image using NumPy-only algorithms.

    Args:
        image: HxW grayscale or HxWxC array. Values are converted to float64.
        size: Destination ``(width, height)``.
        method: nearest, bilinear, bicubic (Keys), mitchell, bspline3, lanczos3 or area.
        antialias: Widen reconstruction kernels for reduction. ``area`` is exact
            footprint integration and does not use this flag or point-kernel options.
        boundary: ``clamp``, ``reflect``, ``wrap`` or ``constant`` extension.
            Here ``reflect`` is whole-sample symmetric (index -1 maps to 1),
            matching SciPy ``mirror`` / OpenCV ``BORDER_REFLECT_101``. Cubic
            B-spline coefficient prefiltering supports clamp and reflect. The
            ``area`` method uses source pixel footprints and ignores this option.
        coordinate_mode: ``half_pixel`` or ``align_corners``. The ``area``
            method uses exact source pixel-cell footprints and ignores this option.
        cval: Constant boundary value when ``boundary='constant'``.
        keys_a: Parameter of the Keys cubic-convolution family (default -0.5).
        mitchell_b, mitchell_c: Parameters of the Mitchell-Netravali cubic family.

    Returns:
        A float64 array with dimensions (height, width[, channels]).

    Notes:
        This function does not call Pillow, OpenCV, SciPy, scikit-image or any
        other image interpolation/resizing implementation. NumPy is used only
        for array storage and elementary numerical operations.
    """
    array = _as_image_array(image)
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size must be a (width, height) pair")
    width, height = size
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise TypeError("target width and height must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary mode {boundary!r}; choose from {', '.join(BOUNDARIES)}")
    if coordinate_mode not in COORDINATE_MODES:
        raise ValueError(
            f"unknown coordinate mode {coordinate_mode!r}; choose from {', '.join(COORDINATE_MODES)}"
        )
    if not np.isfinite(cval):
        raise ValueError("cval must be finite")
    if method == "bspline3" and boundary not in ("reflect", "clamp"):
        raise ValueError("bspline3 supports only 'reflect' and 'clamp' boundaries")

    # A resampling operation whose grid is unchanged is exactly the identity,
    # including for non-cardinal reconstruction filters such as Mitchell.
    if array.shape[1] == width and array.shape[0] == height:
        return array.copy()

    if method == "nearest":
        result = _resize_nearest(
            array, width, height, boundary=boundary, coordinate_mode=coordinate_mode
        )
        if boundary == "constant" and cval != 0.0:
            source_h, source_w = array.shape[:2]
            x_positions, _ = _source_positions(source_w, width, coordinate_mode)
            y_positions, _ = _source_positions(source_h, height, coordinate_mode)
            x_valid = (np.floor(x_positions + 0.5) >= 0) & (np.floor(x_positions + 0.5) < source_w)
            y_valid = (np.floor(y_positions + 0.5) >= 0) & (np.floor(y_positions + 0.5) < source_h)
            valid = y_valid[:, None] & x_valid[None, :]
            if array.ndim == 3:
                valid = valid[..., None]
            result = np.where(valid, result, cval)
        return result

    if method == "area":
        horizontal = _resize_axis(array, width, axis=1, kernel=None, area=True)
        return _resize_axis(horizontal, height, axis=0, kernel=None, area=True)

    if method == "bspline3":
        coefficients = _spline_prefilter_axis(array, axis=1, boundary=boundary)
        coefficients = _spline_prefilter_axis(coefficients, axis=0, boundary=boundary)
        kernel = kernels.cubic_bspline
        radius = 2.0
        kernel_kwargs: dict[str, float] = {}
    elif method == "bicubic":
        kernel, radius = kernels.keys_cubic, 2.0
        kernel_kwargs = {"a": float(keys_a)}
    elif method == "mitchell":
        kernel, radius = kernels.mitchell_netravali, 2.0
        kernel_kwargs = {"b": float(mitchell_b), "c": float(mitchell_c)}
    else:
        kernel, radius = _KERNELS[method]
        kernel_kwargs = {}

    source = coefficients if method == "bspline3" else array
    horizontal = _resize_axis(
        source,
        width,
        axis=1,
        kernel=kernel,
        radius=radius,
        antialias=antialias,
        boundary=boundary,
        coordinate_mode=coordinate_mode,
        cval=cval,
        kernel_kwargs=kernel_kwargs,
    )
    return _resize_axis(
        horizontal,
        height,
        axis=0,
        kernel=kernel,
        radius=radius,
        antialias=antialias,
        boundary=boundary,
        coordinate_mode=coordinate_mode,
        cval=cval,
        kernel_kwargs=kernel_kwargs,
    )
