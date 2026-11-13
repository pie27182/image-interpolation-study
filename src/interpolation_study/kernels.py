"""Скалярные реконструкционные ядра для самостоятельного ресэмплера.

Используются только базовые операции с массивами NumPy. В этом модуле
не вызывается ни один API изменения размера изображений.
"""

from __future__ import annotations

import numpy as np


def nearest(distance: np.ndarray) -> np.ndarray:
    """Реконструкционное ядро-прямоугольник единичной ширины (в основном для анализа и графиков)."""
    return (np.abs(distance) < 0.5).astype(np.float64)


def linear(distance: np.ndarray) -> np.ndarray:
    """Линейное/треугольное ядро с радиусом поддержки 1."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    return np.maximum(1.0 - x, 0.0)


def keys_cubic(distance: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Ядро кубической свёртки Keys (значение a=-0.5 близко к варианту Catmull--Rom)."""
    x = np.abs(np.asarray(distance, dtype=np.float64))
    x2 = x * x
    x3 = x2 * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def mitchell_netravali(
    distance: np.ndarray, b: float = 1.0 / 3.0, c: float = 1.0 / 3.0
) -> np.ndarray:
    """Семейство кубических реконструкционных фильтров Mitchell--Netravali.

    Распространённый компромиссный вариант использует B=C=1/3. В отличие
    от кубической свёртки Keys, этот вариант не является кардинальным:
    значение в начале координат равно 8/9, а не 1.
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
    """Базисная функция центрированного кубического B-сплайна с радиусом поддержки 2.

    Базис B-сплайна не является кардинальным. Поэтому перед вычислением
    этой функции ресэмплер предварительно фильтрует массив отсчётов и
    получает коэффициенты сплайна.
    """
    x = np.abs(np.asarray(distance, dtype=np.float64))
    x2 = x * x
    inner = (4.0 - 6.0 * x2 + 3.0 * x2 * x) / 6.0
    outer = np.maximum(2.0 - x, 0.0) ** 3 / 6.0
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def sinc(distance: np.ndarray) -> np.ndarray:
    """Нормированная функция sinc, sin(pi*x)/(pi*x), с устранимым пределом в нуле."""
    return np.sinc(np.asarray(distance, dtype=np.float64))


def lanczos(distance: np.ndarray, radius: int = 3) -> np.ndarray:
    """Оконное ядро sinc (Lanczos) с настраиваемым целочисленным радиусом."""
    if radius < 1:
        raise ValueError("радиус Lanczos должен быть положительным целым числом")
    x = np.asarray(distance, dtype=np.float64)
    ax = np.abs(x)
    values = sinc(x) * sinc(x / float(radius))
    return np.where(ax < radius, values, 0.0)


def lanczos3(distance: np.ndarray) -> np.ndarray:
    """Удобная обёртка для ядра Lanczos с радиусом три."""
    return lanczos(distance, radius=3)
