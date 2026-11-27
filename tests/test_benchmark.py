import numpy as np
from PIL import Image

from src.interpolation_benchmark.benchmark import run_benchmark
from src.interpolation_benchmark.dataset import ImagePair
from src.interpolation_benchmark.methods import DEFAULT_METHODS


def test_benchmark_writes_expected_artifacts(tmp_path):
    pairs = []
    rng = np.random.default_rng(2026)

    for index in range(3):
        image_id = f"{index + 1:04d}"
        hr_array = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        hr = Image.fromarray(hr_array, mode="RGB")
        lr = hr.resize((20, 20), resample=Image.Resampling.BICUBIC)

        hr_path = tmp_path / f"{image_id}.png"
        lr_path = tmp_path / f"{image_id}x4.png"
        hr.save(hr_path)
        lr.save(lr_path)
        pairs.append(ImagePair(image_id, lr_path, hr_path))

    output_dir = tmp_path / "results"
    per_image, summary, tests = run_benchmark(
        pairs=pairs,
        output_dir=output_dir,
        scale=4,
        repeats=1,
        visual_sample_count=1,
    )

    assert len(per_image) == len(pairs) * len(DEFAULT_METHODS)
    assert set(summary["method"]) == set(DEFAULT_METHODS)
    assert not tests.empty
    for filename in (
        "metrics_per_image.csv",
        "summary.csv",
        "paired_tests_vs_bicubic.csv",
        "comparison_grid.png",
        "run_metadata.json",
    ):
        assert (output_dir / filename).is_file()
