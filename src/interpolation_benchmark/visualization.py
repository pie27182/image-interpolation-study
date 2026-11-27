"""Publication-friendly visual comparison grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _center_crop(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = min(height, max_side)
    crop_w = min(width, max_side)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[top : top + crop_h, left : left + crop_w]


def save_comparison_grid(
    samples: list[dict[str, Any]],
    method_names: list[str],
    output_path: str | Path,
) -> None:
    """Save HR reference and method outputs for selected real DIV2K images."""
    if not samples:
        return

    n_rows = len(samples)
    n_cols = len(method_names) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for row, sample in enumerate(samples):
        reference = _center_crop(sample["reference"])
        axes[row, 0].imshow(reference)
        axes[row, 0].set_title(f'{sample["image_id"]} — HR reference', fontsize=9)
        axes[row, 0].axis("off")

        for col, method in enumerate(method_names, start=1):
            image = _center_crop(sample["outputs"][method])
            axes[row, col].imshow(image)
            axes[row, col].set_title(method, fontsize=9)
            axes[row, col].axis("off")

    fig.suptitle(
        "DIV2K validation: ×4 interpolation results (center crops)",
        fontsize=12,
    )
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
