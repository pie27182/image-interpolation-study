import numpy as np

from src.interpolation_benchmark.metrics import compute_metrics


def test_identical_images_have_infinite_psnr_and_unit_ssim():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = compute_metrics(image, image, border=4)
    assert np.isinf(result["psnr_y_db"])
    assert np.isclose(result["ssim_y"], 1.0)
    assert np.isinf(result["psnr_rgb_db"])
    assert np.isclose(result["ssim_rgb"], 1.0)
