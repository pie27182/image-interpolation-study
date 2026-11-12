import numpy as np
import pytest

from interpolation_study.metrics import (
    compare_images,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    structural_similarity,
)


def test_mse_known_value():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 10, dtype=np.uint8)
    assert mean_squared_error(a, b) == pytest.approx(100.0)


def test_psnr_identical_images_is_infinite():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert np.isinf(peak_signal_to_noise_ratio(image, image))


def test_ssim_identical_images_is_one():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(24, 19, 3), dtype=np.uint8)
    assert structural_similarity(image, image) == pytest.approx(1.0, abs=1e-12)


def test_ssim_constant_offset_is_finite_and_below_one():
    a = np.full((24, 24), 100.0)
    b = np.full((24, 24), 110.0)
    score = structural_similarity(a, b)
    assert np.isfinite(score)
    assert score < 1.0


def test_compare_images_returns_expected_keys():
    image = np.full((10, 10), 120, dtype=np.uint8)
    result = compare_images(image, image)
    assert set(result) == {"mse", "psnr", "ssim"}
    assert result["mse"] == 0.0
    assert np.isinf(result["psnr"])
    assert result["ssim"] == pytest.approx(1.0)


def test_metric_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mean_squared_error(np.zeros((2, 2)), np.zeros((2, 3)))


def test_metric_rejects_invalid_data_range():
    image = np.ones((10, 10))
    with pytest.raises(ValueError):
        compare_images(image, image, data_range=0.0)


def test_metric_rejects_even_ssim_window():
    image = np.ones((10, 10))
    with pytest.raises(ValueError):
        compare_images(image, image, win_size=4)
