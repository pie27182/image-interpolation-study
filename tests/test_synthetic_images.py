import numpy as np
import pytest
from PIL import Image

from examples.generate_test_images import generate_test_images


EXPECTED_FILES = {
    "synthetic_gradient.png",
    "synthetic_edges.png",
    "synthetic_impulse.png",
    "synthetic_slanted_edge.png",
    "synthetic_grating.png",
    "synthetic_zone_plate.png",
    "synthetic_siemens_star.png",
    "synthetic_frequency_sweep.png",
}


def test_generator_creates_eight_square_images(tmp_path):
    paths = generate_test_images(tmp_path, size=128)

    assert {path.name for path in paths} == EXPECTED_FILES
    assert all(path.is_file() for path in paths)

    for path in paths:
        with Image.open(path) as image:
            assert image.size == (128, 128)
            assert image.mode in {"L", "RGB"}


def test_generator_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = generate_test_images(first, size=96)
    second_paths = generate_test_images(second, size=96)

    assert {path.name for path in first_paths} == {path.name for path in second_paths}
    for name in EXPECTED_FILES:
        with Image.open(first / name) as first_image, Image.open(second / name) as second_image:
            assert np.array_equal(np.asarray(first_image), np.asarray(second_image))


def test_generator_rejects_too_small_images(tmp_path):
    with pytest.raises(ValueError, match="не меньше 16"):
        generate_test_images(tmp_path, size=15)
