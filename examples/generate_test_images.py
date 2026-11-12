"""Generate deterministic synthetic images for a first benchmark run."""

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    output = Path(__file__).resolve().parent
    size = 512
    y, x = np.mgrid[0:size, 0:size]

    gradient = np.zeros((size, size, 3), dtype=np.uint8)
    gradient[..., 0] = np.rint(x * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 1] = np.rint(y * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 2] = 128
    Image.fromarray(gradient).save(output / "synthetic_gradient.png")

    tile = 16
    checker = (((x // tile + y // tile) % 2) * 255).astype(np.uint8)
    patterns = np.repeat(checker[..., None], 3, axis=2)
    patterns[220:292, 220:292] = 255
    patterns[236:276, 236:276] = 0
    Image.fromarray(patterns).save(output / "synthetic_edges.png")
    print(f"Generated synthetic_gradient.png and synthetic_edges.png in {output}")


if __name__ == "__main__":
    main()
