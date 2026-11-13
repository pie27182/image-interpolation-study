"""Метрики качества изображения, реализованные непосредственно с помощью NumPy.

Здесь SSIM использует невзвешенное квадратное локальное окно (равномерное
окно) и поправку на выборочную ковариацию. Это явно описанный вариант SSIM;
не утверждается, что он побитово совпадает с реализацией любой библиотеки.
"""

from __future__ import annotations

import numpy as np


def mean_squared_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = _validated_pair(reference, candidate)
    return float(np.mean((reference - candidate) ** 2))


def peak_signal_to_noise_ratio(
    reference: np.ndarray, candidate: np.ndarray, data_range: float = 255.0
) -> float:
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range должен быть положительным конечным числом")
    mse = mean_squared_error(reference, candidate)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range * data_range) / mse))


def _validated_pair(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw_ref = np.asarray(reference)
    raw_test = np.asarray(candidate)
    if np.iscomplexobj(raw_ref) or np.iscomplexobj(raw_test):
        raise TypeError("изображения с комплексными значениями не поддерживаются")
    ref = np.asarray(raw_ref, dtype=np.float64)
    test = np.asarray(raw_test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(f"формы массивов изображений различаются: {ref.shape} != {test.shape}")
    if ref.ndim not in (2, 3):
        raise ValueError("метрики ожидают изображение в градациях серого или массив с каналами")
    if ref.size == 0:
        raise ValueError("изображения не должны быть пустыми")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(test)):
        raise ValueError("изображения должны содержать только конечные значения")
    return ref, test


def _box_sum_axis(array: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Скользящая сумма по допустимым окнам вдоль одной оси с помощью накопительных сумм."""
    moved = np.moveaxis(array, axis, 0)
    cumulative = np.cumsum(moved, axis=0, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[0] = 1
    cumulative = np.concatenate((np.zeros(zero_shape, dtype=np.float64), cumulative), axis=0)
    result = cumulative[width:] - cumulative[:-width]
    return np.moveaxis(result, 0, axis)


def structural_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    data_range: float = 255.0,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Вычислить среднюю локальную SSIM с равномерным окном нечётного размера.

    Для каждого допустимого окна локальные средние, дисперсии и ковариация
    вычисляются по суммам в прямоугольном окне. Применяется поправка на
    выборочную ковариацию N/(N-1), как в распространённой эталонной формуле.
    Для массивов RGB/RGBA каналы вычисляются независимо и усредняются;
    альфа-канал рассматривается как обычный канал.
    """
    ref, test = _validated_pair(reference, candidate)
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range должен быть положительным конечным числом")
    if not isinstance(win_size, int) or win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size должен быть нечётным целым числом не меньше 3")
    if ref.shape[0] < win_size or ref.shape[1] < win_size:
        raise ValueError("размеры изображения должны быть не меньше win_size")

    if ref.ndim == 2:
        ref = ref[..., None]
        test = test[..., None]
    n = win_size * win_size
    correction = n / (n - 1.0)
    mean_ref = _box_sum_axis(_box_sum_axis(ref, win_size, 0), win_size, 1) / n
    mean_test = _box_sum_axis(_box_sum_axis(test, win_size, 0), win_size, 1) / n
    mean_ref_sq = mean_ref * mean_ref
    mean_test_sq = mean_test * mean_test
    mean_cross = mean_ref * mean_test

    e_ref_sq = _box_sum_axis(_box_sum_axis(ref * ref, win_size, 0), win_size, 1) / n
    e_test_sq = _box_sum_axis(_box_sum_axis(test * test, win_size, 0), win_size, 1) / n
    e_cross = _box_sum_axis(_box_sum_axis(ref * test, win_size, 0), win_size, 1) / n

    var_ref = np.maximum(0.0, (e_ref_sq - mean_ref_sq) * correction)
    var_test = np.maximum(0.0, (e_test_sq - mean_test_sq) * correction)
    covariance = (e_cross - mean_cross) * correction
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mean_cross + c1) * (2.0 * covariance + c2)
    denominator = (mean_ref_sq + mean_test_sq + c1) * (var_ref + var_test + c2)
    return float(np.mean(numerator / denominator))


def compare_images(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    data_range: float = 255.0,
    win_size: int = 7,
) -> dict[str, float]:
    """Вернуть MSE, PSNR и документированный в этом пакете вариант SSIM с равномерным окном."""
    ref, test = _validated_pair(reference, candidate)
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range должен быть положительным конечным числом")
    if not isinstance(win_size, int) or win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size должен быть нечётным целым числом не меньше 3")
    mse = float(np.mean((ref - test) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range**2 / mse))
    if min(ref.shape[:2]) < win_size:
        ssim = float("nan")
    else:
        ssim = structural_similarity(ref, test, data_range=data_range, win_size=win_size)
    return {"mse": mse, "psnr": psnr, "ssim": ssim}
