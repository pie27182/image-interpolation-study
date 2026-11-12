import numpy as np
import pytest

from interpolation_study.library import resize_with_pillow
from interpolation_study.resize import METHODS, resize


@pytest.mark.parametrize("method", METHODS)
def test_output_shape_for_rgb(method):
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    result = resize(image, (17, 9), method)
    assert result.shape == (9, 17, 3)
    assert result.dtype == np.float64


@pytest.mark.parametrize("method", METHODS)
def test_constant_image_is_preserved(method):
    image = np.full((8, 9, 3), 73, dtype=np.uint8)
    result = resize(image, (15, 13), method)
    np.testing.assert_allclose(result, 73.0, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_identity_resize_is_identity(method):
    image = np.arange(5 * 6, dtype=np.float64).reshape(5, 6)
    np.testing.assert_allclose(resize(image, (6, 5), method), image, atol=1e-9)


def test_nearest_neighbor_uses_discrete_source_values():
    image = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    result = resize(image, (4, 4), "nearest")
    assert set(np.unique(result)).issubset({0.0, 50.0, 100.0, 150.0})


def test_grayscale_is_supported():
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert resize(image, (3, 2), "lanczos3").shape == (2, 3)


def test_pillow_baseline_shape():
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    assert resize_with_pillow(image, (17, 9), "bicubic").shape == (9, 17, 3)


@pytest.mark.parametrize("size", [(0, 3), (3, 0), (-1, 3)])
def test_invalid_dimensions_are_rejected(size):
    with pytest.raises(ValueError):
        resize(np.zeros((3, 3)), size)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        resize(np.zeros((3, 3)), (4, 4), "magic")


def test_invalid_image_dimensions_are_rejected():
    with pytest.raises(ValueError):
        resize(np.zeros((2, 2, 2, 2)), (4, 4))
