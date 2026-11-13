"""Передискретизация изображений, реализованная непосредственно на основе математических ядер.

Единственная вычислительная зависимость — NumPy. Pillow, OpenCV, SciPy и
scikit-image здесь не импортируются и не вызываются.

По умолчанию используется соглашение о координатах с полуцелым центром:

    x_source = (x_destination + 1/2) * source_length / destination_length - 1/2

Фильтрация сепарабельна (сначала по горизонтали, затем по вертикали).
При уменьшении компактные ядра расширяются в координатах исходного
изображения, обеспечивая низкочастотную фильтрацию с учётом масштаба.
Метод ``area`` вместо этого точно интегрирует одномерные перекрытия
пиксельных отпечатков и особенно полезен для уменьшения изображений.
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
# Ограничить размер временных массивов выборки для больших изображений.
# Результат вычисляется фрагментами вдоль изменяемой оси, поэтому расход
# памяти не растёт пропорционально полной длине целевой оси и поддержке ядра.
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
        raise ValueError("изображение должно быть двумерным массивом в градациях серого или трёхмерным массивом каналов")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("размеры изображения должны быть ненулевыми")
    if array.ndim == 3 and array.shape[2] < 1:
        raise ValueError("изображение должно содержать хотя бы один канал")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("изображение должно содержать числовые значения")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("изображения с комплексными значениями не поддерживаются")
    if not np.all(np.isfinite(array)):
        raise ValueError("изображение должно содержать только конечные значения")
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
            f"неизвестный режим координат {coordinate_mode!r}; выберите один из вариантов: {', '.join(COORDINATE_MODES)}"
        )
    return positions, scale


def _map_indices(indices: np.ndarray, length: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Отобразить целочисленные индексы отсчётов в пределах исходного изображения.

    Режим ``reflect`` означает симметричное отражение относительно целого
    отсчёта (индекс -1 отображается в 1); его также называют reflect-101.
    Для режима ``constant`` сохраняется маска допустимости, чтобы отсчёты
    за пределами изображения получили заданное вызывающим кодом постоянное значение.
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
    raise ValueError(f"неизвестный режим границ {boundary!r}; выберите один из вариантов: {', '.join(BOUNDARIES)}")


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
    """Заранее вычислить исходные индексы, флаги допустимости и нормированные веса отсчётов."""
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
            raise ArithmeticError("ядро интерполяции вернуло близкую к нулю или недопустимую сумму весов")
        tap_weights = tap_weights / total
        mapped, is_valid = _map_indices(source_indices, source_length, boundary)
        count = source_indices.size
        if count > max_taps:
            raise RuntimeError("внутреннего буфера отсчётов оказалось недостаточно")
        indices[out_index, :count] = mapped
        valid[out_index, :count] = is_valid
        weights[out_index, :count] = tap_weights

    return indices, valid, weights


def _axis_area_contributions(
    source_length: int, destination_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точное перекрытие целевых отпечатков с ячейками исходных пикселей.

    Исходный пиксель i занимает [i-1/2, i+1/2]. Целевая ячейка j покрывает
    соответствующий интервал [j*s-1/2, (j+1)*s-1/2] в исходных координатах.
    Её веса равны длинам перекрытия, делённым на размер целевого отпечатка.
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
            raise ArithmeticError("при передискретизации по площади получен пустой пиксельный отпечаток")
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
            raise ValueError("требуется реконструкционное ядро")
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
    """Решить вдоль одной оси систему перехода от отсчётов к коэффициентам кубического B-сплайна.

    В целочисленных узлах beta_3(0)=2/3 и beta_3(+-1)=1/6.
    Зеркальное продолжение даёт конечный внедиагональный элемент 1/3;
    продолжение с прижатием к краю — конечный диагональный элемент 5/6
    и внедиагональный 1/6. Полученная система трёхдиагональна, поэтому
    векторизованный алгоритм Томаса решает все строки/столбцы за O(n) на строку.
    """
    if boundary not in ("reflect", "clamp"):
        raise ValueError("bspline3 поддерживает только граничные режимы 'reflect' и 'clamp'")
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
    else:  # постоянное продолжение коэффициентов сплайна на концах
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
    """Изменить размер одноканального или многоканального изображения алгоритмами только на NumPy.

    Args:
        image: массив HxW в градациях серого или HxWxC. Значения преобразуются в float64.
        size: целевой размер ``(width, height)``.
        method: nearest, bilinear, bicubic (Keys), mitchell, bspline3, lanczos3 или area.
        antialias: расширять реконструкционные ядра при уменьшении. ``area``
            выполняет точное интегрирование отпечатка и не использует этот
            флаг или параметры точечных ядер.
        boundary: продолжение за границами: ``clamp``, ``reflect``, ``wrap``
            или ``constant``. Здесь ``reflect`` означает симметрию
            относительно целого отсчёта (индекс -1 отображается в 1),
            как в SciPy ``mirror`` / OpenCV ``BORDER_REFLECT_101``.
            Предварительная фильтрация коэффициентов кубического B-сплайна
            поддерживает clamp и reflect. Метод ``area`` использует
            отпечатки исходных пикселей и игнорирует этот параметр.
        coordinate_mode: ``half_pixel`` или ``align_corners``. Метод
            ``area`` использует точные отпечатки исходных ячеек пикселей
            и игнорирует этот параметр.
        cval: постоянное значение за границами при ``boundary='constant'``.
        keys_a: параметр семейства кубической свёртки Keys (по умолчанию -0.5).
        mitchell_b, mitchell_c: параметры кубического семейства Mitchell-Netravali.

    Returns:
        Массив float64 с размерностями (height, width[, channels]).

    Notes:
        Эта функция не вызывает Pillow, OpenCV, SciPy, scikit-image или любую
        другую реализацию интерполяции/изменения размера изображений. NumPy
        используется только для хранения массивов и элементарных численных операций.
    """
    array = _as_image_array(image)
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size должен быть парой значений (width, height)")
    width, height = size
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise TypeError("целевая ширина и высота должны быть целыми числами")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("целевая ширина и высота должны быть положительными")
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"неизвестный метод {method!r}; выберите один из вариантов: {', '.join(METHODS)}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"неизвестный режим границ {boundary!r}; выберите один из вариантов: {', '.join(BOUNDARIES)}")
    if coordinate_mode not in COORDINATE_MODES:
        raise ValueError(
            f"неизвестный режим координат {coordinate_mode!r}; выберите один из вариантов: {', '.join(COORDINATE_MODES)}"
        )
    if not np.isfinite(cval):
        raise ValueError("cval должен быть конечным числом")
    if method == "bspline3" and boundary not in ("reflect", "clamp"):
        raise ValueError("bspline3 поддерживает только граничные режимы 'reflect' и 'clamp'")

    # Операция передискретизации без изменения сетки в точности тождественна,
    # в том числе для некардинальных реконструкционных фильтров, таких как Mitchell.
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
