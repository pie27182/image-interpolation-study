"""Пакетный запуск сравнительного эксперимента на наборе изображений.

Пример:
    python examples/run_dataset_experiment.py data/images results/dataset --scale 2 --repeats 7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from interpolation_study.benchmark import run_benchmark

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить все доступные методы на каждом изображении каталога."
    )
    parser.add_argument("input_dir", type=Path, help="каталог с исходными изображениями")
    parser.add_argument("output_dir", type=Path, help="каталог для результатов эксперимента")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="во сколько раз уменьшать изображение перед восстановлением")
    parser.add_argument("--degradation", default="lanczos3",
                        choices=("nearest", "bilinear", "bicubic", "mitchell", "bspline3", "lanczos3", "area"))
    parser.add_argument("--degradation-implementation", choices=("scratch", "pillow"), default="scratch")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"каталог не найден: {args.input_dir}")
    files = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        parser.error(f"в каталоге нет поддерживаемых изображений: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for index, image_path in enumerate(files, start=1):
        run_dir = args.output_dir / f"{index:03d}_{image_path.stem}"
        rows = run_benchmark(
            image_path,
            run_dir,
            args.scale,
            args.degradation,
            degradation_implementation=args.degradation_implementation,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for row in rows:
            all_rows.append({"image": str(image_path), **row})
        print(f"[{index}/{len(files)}] {image_path.name}: {len(rows)} вариантов")

    csv_path = args.output_dir / "results.csv"
    fieldnames = list(all_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Готово: {len(files)} изображений, {len(all_rows)} измерений.")
    print(f"Сводная таблица: {csv_path.resolve()}")
    print("Для итоговых выводов группируйте результаты по методу и отдельно анализируйте изображения.")


if __name__ == "__main__":
    main()
