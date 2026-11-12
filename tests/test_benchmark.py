import csv

import numpy as np
from PIL import Image

from interpolation_study.benchmark import run_benchmark
from interpolation_study.library import PILLOW_METHODS
from interpolation_study.resize import METHODS


def test_benchmark_runs_library_and_scratch_methods(tmp_path):
    y, x = np.mgrid[:24, :28]
    image = np.stack((x * 9, y * 10, (x + y) * 4), axis=-1).clip(0, 255).astype(np.uint8)
    input_path = tmp_path / "reference.png"
    Image.fromarray(image).save(input_path)

    output_dir = tmp_path / "results"
    rows = run_benchmark(
        input_path, output_dir, scale_factor=2, degradation="area", warmup=0, repeats=1
    )

    assert len(rows) == len(PILLOW_METHODS) + len(METHODS)
    assert all(row["degradation_implementation"] == "scratch" for row in rows)
    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == len(rows)
    assert (output_dir / "low_resolution_input.png").exists()
    assert (output_dir / "scratch_bspline3.png").exists()
    assert (output_dir / "pillow_area.png").exists()
    assert all("time_median_ms" in row for row in saved_rows)



def test_benchmark_can_use_pillow_for_degradation_when_explicit(tmp_path):
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    input_path = tmp_path / "reference.png"
    Image.fromarray(image).save(input_path)
    rows = run_benchmark(
        input_path,
        tmp_path / "pillow-degradation",
        scale_factor=2,
        degradation="lanczos3",
        degradation_implementation="pillow",
        warmup=0,
        repeats=1,
    )
    assert rows
    assert all(row["degradation_implementation"] == "pillow" for row in rows)
