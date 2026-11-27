"""Run the DIV2K interpolation benchmark from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interpolation_benchmark.benchmark import run_benchmark
from src.interpolation_benchmark.dataset import discover_div2k_pairs
from src.interpolation_benchmark.methods import DEFAULT_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/DIV2K"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/DIV2K_x4"))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visual-samples", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    args = parser.parse_args()

    pairs = discover_div2k_pairs(args.data_root, scale=args.scale)
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        pairs = pairs[: args.max_images]

    print(f"Found {len(pairs)} paired DIV2K images.")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Scale: x{args.scale}; repeats: {args.repeats}; CPU only.")

    _, summary, significance = run_benchmark(
        pairs=pairs,
        output_dir=args.output_dir,
        scale=args.scale,
        repeats=args.repeats,
        methods=args.methods,
        visual_sample_count=args.visual_samples,
    )

    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults written to: {args.output_dir.resolve()}")
    if not significance.empty:
        print(
            "\nPaired Wilcoxon tests are in "
            f"{(args.output_dir / 'paired_tests_vs_bicubic.csv').resolve()}"
        )


if __name__ == "__main__":
    main()
