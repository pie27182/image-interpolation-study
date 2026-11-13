"""Графики для анализа ядер и результатов сравнительных экспериментов.

Matplotlib загружается только при вызове функций визуализации. Численные
значения алгоритмов рассчитываются модулями проекта, а не Matplotlib.
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from . import kernels


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Для построения графиков установите дополнительную зависимость: "
            "python -m pip install -e '.[plots]'"
        ) from exc
    return plt


def plot_kernel_diagnostics(output: str | Path) -> Path:
    """Сохранить графики ядер, частотных характеристик и переходных откликов."""
    plt = _pyplot()
    x = np.linspace(-4.5, 4.5, 9001)
    dx = x[1] - x[0]
    definitions = {
        "Ближайший сосед": (kernels.nearest, 0.5),
        "Билинейная": (kernels.linear, 1.0),
        "Keys (a=-0.5)": (kernels.keys_cubic, 2.0),
        "Mitchell (B=C=1/3)": (kernels.mitchell_netravali, 2.0),
        "Кубический B-сплайн": (kernels.cubic_bspline, 2.0),
        "Lanczos-3": (kernels.lanczos3, 3.0),
    }
    frequencies = np.linspace(0.0, 0.5, 401)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for label, (kernel, support) in definitions.items():
        values = kernel(x)
        axes[0].plot(x, values, label=label)
        inside = np.abs(x) <= support
        response = np.array([
            np.sum(values[inside] * np.cos(2.0 * np.pi * f * x[inside])) * dx
            for f in frequencies
        ])
        dc = response[0]
        if abs(dc) > 1e-12:
            response = response / dc
        axes[1].plot(frequencies, np.abs(response), label=label)
        axes[2].plot(x, np.cumsum(values) * dx, label=label)

    axes[0].set(
        title="Пространственные ядра",
        xlabel="Расстояние (исходные пиксели)",
        ylabel="K(x)",
    )
    axes[1].set(
        title="Амплитудная частотная характеристика",
        xlabel="Циклы на исходный пиксель",
        ylabel="|H(f)|",
    )
    axes[2].set(
        title="Переходная характеристика",
        xlabel="Положение (исходные пиксели)",
        ylabel="Накопленный отклик",
    )
    axes[1].set_xlim(0.0, 0.5)
    axes[2].set_xlim(-3.5, 3.5)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def _finite_values(values: Iterable[str | float]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def plot_benchmark_results(
    results_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Построить агрегированные графики PSNR, SSIM и времени по файлу benchmark CSV.

    При наличии нескольких изображений агрегирование выполняется по значениям
    отдельных изображений. Время берётся как медиана повторов из каждой строки,
    после чего усредняется по изображениям. Это не смешивает отдельные прогоны
    одного изображения с независимыми изображениями набора.
    """
    source = Path(results_csv)
    if not source.is_file():
        raise FileNotFoundError(f"CSV с результатами не найден: {source}")

    required = {"implementation", "method", "psnr_db", "ssim_uniform", "time_median_ms"}
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "В CSV отсутствуют обязательные столбцы: " + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"В CSV нет строк с результатами: {source}")

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"psnr_db": [], "ssim_uniform": [], "time_median_ms": []}
    )
    for row in rows:
        key = (row["implementation"], row["method"])
        for metric in ("psnr_db", "ssim_uniform", "time_median_ms"):
            grouped[key][metric].extend(_finite_values([row.get(metric, "")]))

    summary: list[dict[str, str | int | float]] = []
    for (implementation, method), metrics in sorted(grouped.items()):
        item: dict[str, str | int | float] = {
            "implementation": implementation,
            "method": method,
        }
        for metric, values in metrics.items():
            item[f"{metric}_n"] = len(values)
            item[f"{metric}_mean"] = statistics.fmean(values) if values else float("nan")
            item[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    plt = _pyplot()
    labels = [f"{row['implementation']}\n{row['method']}" for row in summary]
    x = np.arange(len(summary))

    def draw(metric: str, title: str, ylabel: str, filename: str) -> Path:
        means = np.array([float(row[f"{metric}_mean"]) for row in summary])
        deviations = np.array([float(row[f"{metric}_stdev"]) for row in summary])
        valid = np.isfinite(means)
        if not np.any(valid):
            raise ValueError(f"В файле нет конечных значений метрики {metric}")
        fig, axis = plt.subplots(figsize=(max(9, len(summary) * 0.8), 5.5))
        axis.bar(x[valid], means[valid], yerr=deviations[valid], capsize=3)
        axis.set_xticks(x[valid], [labels[index] for index in x[valid]], rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        fig.tight_layout()
        path = target_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    psnr_path = draw(
        "psnr_db", "Качество восстановления: PSNR", "PSNR, дБ", "psnr.png"
    )
    ssim_path = draw(
        "ssim_uniform", "Структурное сходство: SSIM", "SSIM", "ssim.png"
    )
    time_path = draw(
        "time_median_ms", "Время выполнения", "Медианное время, мс", "runtime.png"
    )
    return summary_path, psnr_path, ssim_path, time_path
