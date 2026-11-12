"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .benchmark import run_benchmark
from .library import resize_with_pillow
from .resize import METHODS, resize


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            image = image.convert("RGB")
        return np.asarray(image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-interpolation",
        description="Compare Pillow filters with NumPy interpolation algorithms implemented from scratch.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    resize_parser = commands.add_parser("resize", help="resize one image")
    resize_parser.add_argument("input", help="input image path")
    resize_parser.add_argument("output", help="output image path")
    resize_parser.add_argument("--width", type=int, required=True)
    resize_parser.add_argument("--height", type=int, required=True)
    resize_parser.add_argument("--method", choices=METHODS, default="bilinear")
    resize_parser.add_argument("--implementation", choices=("scratch", "pillow"), default="scratch")
    resize_parser.add_argument("--no-antialias", action="store_true",
                               help="do not widen kernels while downscaling (scratch implementation only)")

    benchmark_parser = commands.add_parser("benchmark", help="run all algorithms and save metrics")
    benchmark_parser.add_argument("input", help="high-resolution reference image")
    benchmark_parser.add_argument("--output-dir", default="results")
    benchmark_parser.add_argument("--scale", type=float, default=2.0,
                                  help="downscale factor used to generate the low-resolution input")
    benchmark_parser.add_argument("--degradation", choices=METHODS, default="lanczos3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "resize":
        image = _load_image(args.input)
        size = (args.width, args.height)
        if args.implementation == "pillow":
            output = resize_with_pillow(image, size, args.method)
        else:
            output = resize(image, size, args.method, antialias=not args.no_antialias)
        output_array = np.clip(np.rint(output), 0, 255).astype(np.uint8)
        Image.fromarray(output_array).save(args.output)
        print(f"Saved {args.output} ({args.width}x{args.height}, {args.method}, {args.implementation}).")
        return

    rows = run_benchmark(args.input, args.output_dir, args.scale, args.degradation)
    print(f"Saved {len(rows)} result images and results.csv in {Path(args.output_dir).resolve()}")
    print(f"{'Implementation':<14} {'Method':<10} {'ms':>10} {'PSNR dB':>10} {'SSIM':>9}")
    for row in rows:
        print(f"{row['implementation']:<14} {row['method']:<10} {row['time_ms']:>10.2f} "
              f"{row['psnr_db']:>10.3f} {row['ssim']:>9.5f}")


if __name__ == "__main__":
    main()
