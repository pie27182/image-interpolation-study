"""Generate deterministic synthetic images for interpolation experiments."""

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    output = Path(__file__).resolve().parent
    size = 512
    y, x = np.mgrid[:size, :size]

    gradient = np.empty((size, size, 3), dtype=np.uint8)
    gradient[..., 0] = np.rint(x * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 1] = np.rint(y * 255 / (size - 1)).astype(np.uint8)
    gradient[..., 2] = 128
    Image.fromarray(gradient).save(output / "synthetic_gradient.png")

    tile = 16
    checker = (((x // tile + y // tile) % 2) * 255).astype(np.uint8)
    edges = np.repeat(checker[..., None], 3, axis=2)
    edges[220:292, 220:292] = 255
    edges[236:276, 236:276] = 0
    Image.fromarray(edges).save(output / "synthetic_edges.png")

    impulse = np.zeros((size, size), dtype=np.uint8)
    impulse[size // 2, size // 2] = 255
    Image.fromarray(impulse).save(output / "synthetic_impulse.png")

    slanted = np.where(x - 0.37 * y > size * 0.28, 255, 0).astype(np.uint8)
    Image.fromarray(slanted).save(output / "synthetic_slanted_edge.png")

    grating = (127.5 + 127.5 * np.sin(2.0 * np.pi * x / 7.0)).clip(0, 255).astype(np.uint8)
    grating_rgb = np.repeat(grating[..., None], 3, axis=2)
    Image.fromarray(grating_rgb).save(output / "synthetic_grating.png")

    print(f"Generated five deterministic test images in {output}")


if __name__ == "__main__":
    main()
