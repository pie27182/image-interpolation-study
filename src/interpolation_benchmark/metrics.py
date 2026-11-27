"""Image-fidelity metrics computed on paired 8-bit RGB images."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601-style luma on the 0..255 scale."""
    values = rgb.astype(np.float64, copy=False)
    return (
        0.299 * values[..., 0]
        + 0.587 * values[..., 1]
        + 0.114 * values[..., 2]
    )


def _shave(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    if min(image.shape[:2]) <= 2 * border + 7:
        raise ValueError(
            f"Image is too small ({image.shape[:2]}) for border={border}."
        )
    return image[border:-border, border:-border, ...]


def compute_metrics(
    reconstructed_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    border: int = 4,
) -> dict[str, float]:
    """Calculate PSNR/SSIM on RGB and luma, after the stated border shave."""
    reconstructed = np.asarray(reconstructed_rgb)
    reference = np.asarray(reference_rgb)

    if reconstructed.shape != reference.shape:
        raise ValueError(
            f"Image shapes differ: {reconstructed.shape} vs {reference.shape}"
        )
    if reconstructed.ndim != 3 or reconstructed.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB arrays.")
    if reconstructed.dtype != np.uint8 or reference.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB arrays.")

    sr = _shave(reconstructed, border)
    hr = _shave(reference, border)

    sr_y = _luma(sr)
    hr_y = _luma(hr)

    return {
        "psnr_y_db": float(
            peak_signal_noise_ratio(hr_y, sr_y, data_range=255.0)
        ),
        "ssim_y": float(
            structural_similarity(hr_y, sr_y, data_range=255.0)
        ),
        "psnr_rgb_db": float(
            peak_signal_noise_ratio(hr, sr, data_range=255.0)
        ),
        "ssim_rgb": float(
            structural_similarity(hr, sr, channel_axis=-1, data_range=255.0)
        ),
    }
