"""Run a repeatable upscale benchmark against a known high-resolution image."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .library import PILLOW_FILTERS, resize_with_pillow
from .metrics import compare_images
from .resize import METHODS, resize


def run_benchmark(
    image_path: str | Path,
    output_dir: str | Path,
    scale_factor: float = 2.0,
    degradation: str = "lanczos3",
) -> list[dict[str, float | str]]:
    """Downsample a source image, then upscale it with each implementation.

    The original image is treated as the reference. The low-resolution image is
    generated with Pillow using ``degradation``; this choice is written to the
    CSV and should be kept fixed when comparing experiments.
    """
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1")
    if degradation not in PILLOW_FILTERS:
        raise ValueError(f"degradation must be one of {', '.join(METHODS)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        original = np.asarray(opened.convert("RGB"), dtype=np.uint8)

    height, width = original.shape[:2]
    low_size = (max(1, round(width / scale_factor)), max(1, round(height / scale_factor)))
    low_res = resize_with_pillow(original, low_size, degradation).astype(np.uint8)
    Image.fromarray(low_res).save(output / "low_resolution_input.png")

    rows: list[dict[str, float | str]] = []
    implementations = (
        ("pillow", lambda arr, size, method: resize_with_pillow(arr, size, method)),
        ("scratch", lambda arr, size, method: resize(arr, size, method)),
    )
    for implementation, resize_fn in implementations:
        for method in METHODS:
            start = time.perf_counter()
            result = resize_fn(low_res, (width, height), method)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result_u8 = np.clip(np.rint(result), 0, 255).astype(np.uint8)
            Image.fromarray(result_u8).save(output / f"{implementation}_{method}.png")
            metrics = compare_images(original, result_u8)
            rows.append({
                "implementation": implementation,
                "method": method,
                "time_ms": elapsed_ms,
                "mse": metrics["mse"],
                "psnr_db": metrics["psnr"],
                "ssim": metrics["ssim"],
                "source_width": width,
                "source_height": height,
                "low_width": low_size[0],
                "low_height": low_size[1],
                "scale_factor": scale_factor,
                "degradation_filter": degradation,
            })

    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
