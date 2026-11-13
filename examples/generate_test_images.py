"""Генерация воспроизводимых синтетических тестов для интерполяции изображений."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def generate_test_images(output_dir: str | Path, size: int = 512) -> tuple[Path, ...]:
    """Создать восемь детерминированных тестовых изображений без случайных данных.

    Синтетические изображения подходят для изолированной проверки границ,
    импульсного отклика, муара и поведения алгоритмов на возрастающих частотах.
    """
    if size < 16:
        raise ValueError("размер тестовых изображений должен быть не меньше 16 пикселей")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:size, :size]
    paths: list[Path] = []

    def save(name: str, pixels: np.ndarray) -> None:
        path = output / name
        Image.fromarray(pixels).save(path)
        paths.append(path)

    # Цветовой градиент: проверка плавности и точности интерполяции.
    gradient = np.empty((size, size, 3), dtype=np.uint8)
    gradient[..., 0] = np.rint(x * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 1] = np.rint(y * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 2] = 128
    save("synthetic_gradient.png", gradient)

    # Шахматная решётка с вложенным квадратом: контрастные границы и алиасинг.
    tile = max(1, size // 32)
    checker = (((x // tile + y // tile) % 2) * 255).astype(np.uint8)
    edges = np.repeat(checker[..., None], 3, axis=2)
    center = size // 2
    outer = max(2, size // 14)
    inner = max(1, size // 28)
    edges[center - outer:center + outer, center - outer:center + outer] = 255
    edges[center - inner:center + inner, center - inner:center + inner] = 0
    save("synthetic_edges.png", edges)

    # Единичный импульс: пространственный отклик реконструкционного фильтра.
    impulse = np.zeros((size, size), dtype=np.uint8)
    impulse[center, center] = 255
    save("synthetic_impulse.png", impulse)

    # Наклонная ступень: размытие, звон и оценка отклика на границу.
    slanted = np.where(x - 0.37 * y > size * 0.28, 255, 0).astype(np.uint8)
    save("synthetic_slanted_edge.png", slanted)

    # Периодическая решётка фиксированной частоты: интерференция и муар.
    grating = (
        127.5 + 127.5 * np.sin(2.0 * np.pi * x / 7.0)
    ).clip(0, 255).astype(np.uint8)
    save("synthetic_grating.png", np.repeat(grating[..., None], 3, axis=2))

    # Zone plate: радиальная частота растёт от центра к краям и пересекает Найквист.
    dx = x.astype(np.float64) - (size - 1) / 2
    dy = y.astype(np.float64) - (size - 1) / 2
    zone_phase = np.pi * (dx * dx + dy * dy) / size
    zone_plate = np.rint(127.5 + 127.5 * np.cos(zone_phase)).astype(np.uint8)
    save("synthetic_zone_plate.png", zone_plate)

    # Siemens star: радиальная звезда с 48 чередующимися секторами.
    theta = np.arctan2(dy, dx)
    siemens = np.where(np.sin(24.0 * theta) >= 0.0, 255, 0).astype(np.uint8)
    siemens[dx * dx + dy * dy < (size * 0.012) ** 2] = 127
    save("synthetic_siemens_star.png", siemens)

    # Линейный частотный sweep: частота вертикальных полос растёт слева направо.
    x_float = x.astype(np.float64)
    start_frequency = 0.01
    end_frequency = 0.75  # часть диапазона намеренно выше предела Найквиста (0.5).
    phase = 2.0 * np.pi * (
        start_frequency * x_float
        + (end_frequency - start_frequency) * x_float * x_float / (2.0 * (size - 1))
    )
    sweep = np.rint(127.5 + 127.5 * np.cos(phase)).astype(np.uint8)
    save("synthetic_frequency_sweep.png", sweep)

    return tuple(paths)


def build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    parser = argparse.ArgumentParser(
        description="Создать восемь детерминированных синтетических тестов интерполяции."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"каталог результата (по умолчанию: {default_output})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="ширина и высота каждого квадратного изображения (по умолчанию: 512)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = generate_test_images(args.output_dir, args.size)
    print(f"Создано {len(paths)} тестовых изображений в {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
