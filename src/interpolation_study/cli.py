"""Командный интерфейс для изменения размера и исследования интерполяции."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from . import __version__
from .benchmark import run_benchmark
from .library import PILLOW_METHODS, resize_with_pillow
from .resize import BOUNDARIES, COORDINATE_MODES, METHODS, resize
from .visualization import plot_benchmark_results, plot_kernel_diagnostics

_METHOD_DESCRIPTIONS = {
    "nearest": "ближайший сосед; сохраняет дискретные значения",
    "bilinear": "билинейная интерполяция; быстрый сглаженный результат",
    "bicubic": "кубическая свёртка Keys; параметр задаётся через --keys-a",
    "mitchell": "кубический фильтр Mitchell–Netravali; параметры --mitchell-b/c",
    "bspline3": "кубический B-сплайн с предварительной фильтрацией коэффициентов",
    "lanczos3": "оконный sinc-фильтр с радиусом 3",
    "area": "усреднение перекрывающихся площадей пикселей; прежде всего для уменьшения",
}


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            image = image.convert("RGB")
        return np.asarray(image)


def _print_methods() -> None:
    print("Методы самостоятельной реализации (NumPy):")
    for method in METHODS:
        print(f"  {method:<10} {_METHOD_DESCRIPTIONS[method]}")
    print("\nМетоды библиотечной базовой реализации Pillow:")
    print("  " + ", ".join(PILLOW_METHODS))
    print("\nMitchell и bspline3 доступны только в реализации NumPy.")


def build_parser() -> argparse.ArgumentParser:
    examples = """Примеры:
  image-interpolation --help
  image-interpolation methods
  image-interpolation resize input.jpg output.png --width 1200 --height 800 --method lanczos3
  image-interpolation benchmark reference.png --scale 2 --repeats 7 --output-dir results/run
  image-interpolation plot results/run/results.csv --output-dir results/run/plots
  image-interpolation plot-kernels --output results/kernel_diagnostics.png
"""
    parser = argparse.ArgumentParser(
        prog="image-interpolation",
        description=(
            "Сравнительное исследование алгоритмов интерполяции цифровых изображений. "
            "Самостоятельные алгоритмы реализованы на NumPy."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="показать версию программы",
    )
    commands = parser.add_subparsers(
        dest="command",
        title="команды",
        metavar="COMMAND",
    )

    resize_parser = commands.add_parser(
        "resize",
        help="изменить размер одного изображения",
        description="Изменить размер изображения указанным алгоритмом.",
        epilog=(
            "Пример: image-interpolation resize input.jpg output.png "
            "--width 1200 --height 800 --method bicubic"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    resize_parser.add_argument("input", help="путь к исходному изображению")
    resize_parser.add_argument("output", help="путь для сохранения результата")
    resize_parser.add_argument("--width", "-w", type=int, required=True, help="целевая ширина в пикселях")
    resize_parser.add_argument("--height", "-H", type=int, required=True, help="целевая высота в пикселях")
    resize_parser.add_argument(
        "--method", choices=METHODS, default="bilinear",
        help="алгоритм интерполяции (по умолчанию: bilinear)",
    )
    resize_parser.add_argument(
        "--implementation", choices=("scratch", "pillow"), default="scratch",
        help="scratch — собственный код на NumPy; pillow — базовая реализация Pillow",
    )
    resize_parser.add_argument(
        "--no-antialias", action="store_true",
        help="отключить расширение фильтра при уменьшении (может усилить алиасинг)",
    )
    resize_parser.add_argument(
        "--boundary", choices=BOUNDARIES, default="clamp",
        help="обработка отсчётов за границей изображения (по умолчанию: clamp)",
    )
    resize_parser.add_argument(
        "--coordinate-mode", choices=COORDINATE_MODES, default="half_pixel",
        help="соглашение о координатах пикселей (по умолчанию: half_pixel)",
    )
    resize_parser.add_argument(
        "--fill-value", type=float, default=0.0,
        help="постоянное значение за границами при --boundary constant (по умолчанию: 0)",
    )
    resize_parser.add_argument(
        "--keys-a", type=float, default=-0.5,
        help="параметр кубического ядра Keys для bicubic (по умолчанию: -0.5)",
    )
    resize_parser.add_argument(
        "--mitchell-b", type=float, default=1.0 / 3.0,
        help="параметр B ядра Mitchell (по умолчанию: 1/3)",
    )
    resize_parser.add_argument(
        "--mitchell-c", type=float, default=1.0 / 3.0,
        help="параметр C ядра Mitchell (по умолчанию: 1/3)",
    )

    benchmark_parser = commands.add_parser(
        "benchmark",
        help="сравнить качество и скорость всех доступных реализаций",
        description=(
            "Уменьшить эталонное изображение, восстановить исходный размер разными методами "
            "и записать метрики качества и время выполнения."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark_parser.add_argument("input", help="эталонное изображение высокого разрешения")
    benchmark_parser.add_argument(
        "--output-dir", default="results",
        help="каталог для изображений и results.csv (по умолчанию: results)",
    )
    benchmark_parser.add_argument(
        "--scale", type=float, default=2.0,
        help="во сколько раз уменьшить исходное изображение (должно быть больше 1; по умолчанию: 2)",
    )
    benchmark_parser.add_argument(
        "--degradation", choices=METHODS, default="lanczos3",
        help="фильтр, используемый для создания входа низкого разрешения (по умолчанию: lanczos3)",
    )
    benchmark_parser.add_argument(
        "--degradation-implementation", choices=("scratch", "pillow"), default="scratch",
        help="реализация фильтра уменьшения (по умолчанию: scratch)",
    )
    benchmark_parser.add_argument(
        "--warmup", type=int, default=1,
        help="число прогревочных запусков, не включаемых в замер (по умолчанию: 1)",
    )
    benchmark_parser.add_argument(
        "--repeats", type=int, default=5,
        help="число измеряемых повторов (по умолчанию: 5)",
    )

    plot_parser = commands.add_parser(
        "plot",
        help="построить графики по results.csv после benchmark",
        description=(
            "Построить столбчатые диаграммы PSNR, SSIM и времени выполнения. "
            "При пакетном эксперименте значения агрегируются по изображениям."
        ),
    )
    plot_parser.add_argument("results_csv", help="путь к results.csv")
    plot_parser.add_argument(
        "--output-dir", default="results/plots",
        help="каталог для PNG-графиков и сводного summary.csv (по умолчанию: results/plots)",
    )

    kernels_parser = commands.add_parser(
        "plot-kernels",
        help="построить графики интерполяционных ядер и их характеристик",
        description=(
            "Показать пространственные ядра, численную частотную характеристику "
            "и переходный отклик. Расчёты выполняются проектом; Matplotlib только рисует графики."
        ),
    )
    kernels_parser.add_argument(
        "--output", default="results/kernel_diagnostics.png",
        help="путь к PNG-файлу (по умолчанию: results/kernel_diagnostics.png)",
    )

    commands.add_parser(
        "methods",
        help="показать описание алгоритмов и доступность в Pillow",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "methods":
        _print_methods()
        return

    if args.command == "plot":
        summary, psnr, ssim, runtime = plot_benchmark_results(
            args.results_csv, args.output_dir
        )
        print("Визуализация завершена:")
        for path in (summary, psnr, ssim, runtime):
            print(f"  {path}")
        return

    if args.command == "plot-kernels":
        target = plot_kernel_diagnostics(args.output)
        print(f"Графики ядер сохранены: {target}")
        return

    if args.command == "resize":
        image = _load_image(args.input)
        size = (args.width, args.height)
        if args.implementation == "pillow":
            if args.method not in PILLOW_METHODS:
                parser.error(
                    f"Pillow не поддерживает метод {args.method!r}; "
                    f"доступные методы: {', '.join(PILLOW_METHODS)}"
                )
            result = resize_with_pillow(image, size, args.method)
        else:
            result = resize(
                image,
                size,
                args.method,
                antialias=not args.no_antialias,
                boundary=args.boundary,
                coordinate_mode=args.coordinate_mode,
                cval=args.fill_value,
                keys_a=args.keys_a,
                mitchell_b=args.mitchell_b,
                mitchell_c=args.mitchell_c,
            )
        output_array = np.clip(np.rint(result), 0, 255).astype(np.uint8)
        Image.fromarray(output_array).save(args.output)
        print(
            f"Сохранено: {args.output} ({args.width}x{args.height}, "
            f"{args.method}, {args.implementation})."
        )
        return

    rows = run_benchmark(
        args.input,
        args.output_dir,
        args.scale,
        args.degradation,
        degradation_implementation=args.degradation_implementation,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(
        f"Сохранено изображений с результатами: {len(rows)}; "
        f"файл results.csv: {Path(args.output_dir).resolve()}"
    )
    print(f"{'Реализация':<14} {'Метод':<10} {'медиана, мс':>11} {'PSNR, дБ':>10} {'SSIM':>9}")
    for row in rows:
        print(
            f"{row['implementation']:<14} {row['method']:<10} "
            f"{row['time_median_ms']:>11.2f} {row['psnr_db']:>10.3f} "
            f"{row['ssim_uniform']:>9.5f}"
        )


if __name__ == "__main__":
    main()
