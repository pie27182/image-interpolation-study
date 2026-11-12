"""Command-line interface for image resampling and experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .benchmark import run_benchmark
from .library import PILLOW_METHODS, resize_with_pillow
from .resize import BOUNDARIES, COORDINATE_MODES, METHODS, resize


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            image = image.convert("RGB")
        return np.asarray(image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-interpolation",
        description="Study classical image resampling with a NumPy-only implementation.",
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
                               help="disable scale-aware low-pass filtering during reduction")
    resize_parser.add_argument("--boundary", choices=BOUNDARIES, default="clamp")
    resize_parser.add_argument("--coordinate-mode", choices=COORDINATE_MODES, default="half_pixel")
    resize_parser.add_argument("--fill-value", type=float, default=0.0)
    resize_parser.add_argument("--keys-a", type=float, default=-0.5,
                               help="Keys cubic parameter (used by bicubic)")
    resize_parser.add_argument("--mitchell-b", type=float, default=1.0 / 3.0)
    resize_parser.add_argument("--mitchell-c", type=float, default=1.0 / 3.0)

    benchmark_parser = commands.add_parser("benchmark", help="run a repeatable image-quality benchmark")
    benchmark_parser.add_argument("input", help="high-resolution reference image")
    benchmark_parser.add_argument("--output-dir", default="results")
    benchmark_parser.add_argument("--scale", type=float, default=2.0,
                                  help="factor by which to reduce the source to create a low-resolution input")
    benchmark_parser.add_argument("--degradation", choices=METHODS, default="lanczos3",
                                  help="method used to create the low-resolution input")
    benchmark_parser.add_argument("--degradation-implementation", choices=("scratch", "pillow"),
                                  default="scratch",
                                  help="use the independent NumPy resizer (default) or Pillow")
    benchmark_parser.add_argument("--warmup", type=int, default=1)
    benchmark_parser.add_argument("--repeats", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "resize":
        image = _load_image(args.input)
        size = (args.width, args.height)
        if args.implementation == "pillow":
            if args.method not in PILLOW_METHODS:
                raise SystemExit(
                    f"Pillow baseline does not implement {args.method!r}; choose from {', '.join(PILLOW_METHODS)}"
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
            f"Saved {args.output} ({args.width}x{args.height}, {args.method}, "
            f"{args.implementation})."
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
    print(f"Saved {len(rows)} result images and results.csv in {Path(args.output_dir).resolve()}")
    print(f"{'Implementation':<14} {'Method':<10} {'median ms':>11} {'PSNR dB':>10} {'SSIM':>9}")
    for row in rows:
        print(
            f"{row['implementation']:<14} {row['method']:<10} "
            f"{row['time_median_ms']:>11.2f} {row['psnr_db']:>10.3f} "
            f"{row['ssim_uniform']:>9.5f}"
        )


if __name__ == "__main__":
    main()
