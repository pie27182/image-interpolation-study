"""Benchmark runner: identical inputs, target sizes, and CPU timing protocol."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from .dataset import ImagePair
from .methods import BASELINE_METHOD, get_method_filters
from .metrics import compute_metrics
from .visualization import save_comparison_grid


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    if not p_values:
        return []
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted


def _paired_tests(
    frame: pd.DataFrame,
    methods: Sequence[str],
    baseline: str = BASELINE_METHOD,
) -> pd.DataFrame:
    """Paired Wilcoxon tests per image, with Holm correction across all tests."""
    rows: list[dict[str, object]] = []
    metrics = ("psnr_y_db", "ssim_y", "psnr_rgb_db", "ssim_rgb")

    base = frame[frame["method"] == baseline].set_index("image_id")
    for metric in metrics:
        for method in methods:
            if method == baseline:
                continue
            candidate = frame[frame["method"] == method].set_index("image_id")
            common = base.index.intersection(candidate.index)
            base_values = base.loc[common, metric].to_numpy(dtype=float)
            method_values = candidate.loc[common, metric].to_numpy(dtype=float)
            differences = method_values - base_values

            if len(common) == 0 or np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    method_values,
                    base_values,
                    alternative="two-sided",
                    zero_method="pratt",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)

            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "n_images": len(common),
                    "median_difference_vs_baseline": (
                        float(np.median(differences)) if len(common) else float("nan")
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )

    adjusted = _holm_adjust([float(row["p_value"]) for row in rows])
    for row, corrected in zip(rows, adjusted):
        row["p_value_holm"] = corrected

    return pd.DataFrame(rows)


def _summarize(frame: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    summary = (
        frame.groupby("method", sort=False)
        .agg(
            n_images=("image_id", "nunique"),
            psnr_y_mean_db=("psnr_y_db", "mean"),
            psnr_y_std_db=("psnr_y_db", "std"),
            psnr_y_median_db=("psnr_y_db", "median"),
            ssim_y_mean=("ssim_y", "mean"),
            ssim_y_std=("ssim_y", "std"),
            ssim_y_median=("ssim_y", "median"),
            psnr_rgb_mean_db=("psnr_rgb_db", "mean"),
            ssim_rgb_mean=("ssim_rgb", "mean"),
            time_ms_median=("time_ms", "median"),
            time_ms_mean=("time_ms", "mean"),
            ms_per_megapixel_median=("ms_per_megapixel", "median"),
        )
        .reindex(methods)
        .reset_index()
    )
    return summary


def run_benchmark(
    pairs: Sequence[ImagePair],
    output_dir: str | Path,
    scale: int = 4,
    repeats: int = 5,
    border: int | None = None,
    methods: Sequence[str] | None = None,
    visual_sample_count: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the fixed-filter comparison and write CSV/JSON/PNG results.

    Timing starts after image decoding and measures only Image.resize().
    Each method receives one warm-up call and then a configurable number of
    timed calls. The median timed call is reported for each image/method pair.
    """
    if not pairs:
        raise ValueError("No image pairs supplied.")
    if scale < 2:
        raise ValueError("scale must be >= 2.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filter_map = get_method_filters(methods)
    method_names = list(filter_map)
    metric_border = scale if border is None else border
    rows: list[dict[str, object]] = []
    visual_samples: list[dict[str, object]] = []

    for pair_index, pair in enumerate(tqdm(pairs, desc="Images")):
        with Image.open(pair.lr_path) as image:
            low_resolution = image.convert("RGB")
        with Image.open(pair.hr_path) as image:
            reference = image.convert("RGB")

        target_size = reference.size
        reference_array = np.asarray(reference, dtype=np.uint8)
        output_arrays: dict[str, np.ndarray] = {}

        for method_name, resampler in filter_map.items():
            # Warm-up is deliberately excluded from the timing sample.
            low_resolution.resize(target_size, resample=resampler)

            timings: list[float] = []
            resized = None
            for _ in range(repeats):
                start = time.perf_counter()
                resized = low_resolution.resize(target_size, resample=resampler)
                timings.append((time.perf_counter() - start) * 1000.0)

            assert resized is not None
            result_array = np.asarray(resized, dtype=np.uint8)
            elapsed_ms = float(statistics.median(timings))
            metrics = compute_metrics(
                result_array,
                reference_array,
                border=metric_border,
            )

            output_megapixels = (
                (target_size[0] * target_size[1]) / 1_000_000.0
            )
            rows.append(
                {
                    "image_id": pair.image_id,
                    "method": method_name,
                    "scale": scale,
                    "width": target_size[0],
                    "height": target_size[1],
                    "psnr_y_db": metrics["psnr_y_db"],
                    "ssim_y": metrics["ssim_y"],
                    "psnr_rgb_db": metrics["psnr_rgb_db"],
                    "ssim_rgb": metrics["ssim_rgb"],
                    "time_ms": elapsed_ms,
                    "time_ms_std_repeats": float(np.std(timings, ddof=0)),
                    "ms_per_megapixel": elapsed_ms / output_megapixels,
                }
            )
            if pair_index < visual_sample_count:
                output_arrays[method_name] = result_array

        if pair_index < visual_sample_count:
            visual_samples.append(
                {
                    "image_id": pair.image_id,
                    "reference": reference_array,
                    "outputs": output_arrays,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "metrics_per_image.csv", index=False)

    summary = _summarize(frame, method_names)
    summary.to_csv(output_path / "summary.csv", index=False)

    significance = _paired_tests(frame, method_names)
    significance.to_csv(output_path / "paired_tests_vs_bicubic.csv", index=False)

    save_comparison_grid(
        visual_samples,
        method_names,
        output_path / "comparison_grid.png",
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DIV2K validation (official HR and bicubic LR pairs)",
        "scale": scale,
        "n_images": len(pairs),
        "methods": method_names,
        "baseline": BASELINE_METHOD,
        "pillow_version": _package_version("Pillow"),
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timing": {
            "device": "CPU",
            "scope": "Image.resize() only; file I/O and metric calculation excluded",
            "warmup_calls_per_image_method": 1,
            "timed_repeats": repeats,
            "reported_statistic": "median",
        },
        "metric_protocol": {
            "border_shave_pixels": metric_border,
            "psnr_ssim_channels": "RGB and BT.601-style luma",
            "uint8_data_range": 255,
        },
        "packages": {
            package: _package_version(package)
            for package in ("Pillow", "numpy", "pandas", "scikit-image", "scipy", "matplotlib")
        },
    }
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return frame, summary, significance
