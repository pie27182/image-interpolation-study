"""Objective image-comparison metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def compare_images(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute MSE, PSNR and SSIM for equally shaped uint8-range images."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"image shapes differ: {reference.shape} != {candidate.shape}")
    if reference.ndim not in (2, 3):
        raise ValueError("metrics expect a grayscale or channel image")
    channel_axis = -1 if reference.ndim == 3 else None
    min_side = min(reference.shape[:2])
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    if win_size < 3:
        ssim = float("nan")
    else:
        ssim = float(structural_similarity(
            reference, candidate, data_range=255.0, channel_axis=channel_axis,
            win_size=win_size,
        ))
    return {
        "mse": float(mean_squared_error(reference, candidate)),
        "psnr": float(peak_signal_noise_ratio(reference, candidate, data_range=255.0)),
        "ssim": ssim,
    }
