import numpy as np
import pytest

from interpolation_study.library import PILLOW_METHODS, resize_with_pillow
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
    np.testing.assert_allclose(result, 73.0, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_identity_resize_is_identity(method):
    image = np.arange(5 * 6, dtype=np.float64).reshape(5, 6)
    np.testing.assert_allclose(resize(image, (6, 5), method), image, atol=1e-8)


def test_nearest_neighbor_uses_discrete_source_values():
    image = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    result = resize(image, (4, 4), "nearest")
    assert set(np.unique(result)).issubset({0.0, 50.0, 100.0, 150.0})


def test_grayscale_is_supported():
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert resize(image, (3, 2), "lanczos3").shape == (2, 3)


def test_pillow_baseline_shape():
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    for method in PILLOW_METHODS:
        assert resize_with_pillow(image, (17, 9), method).shape == (9, 17, 3)


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


def test_bilinear_reproduces_linear_ramp_at_interior_coordinates():
    y, x = np.mgrid[:8, :9]
    image = 3.0 * x + 2.0 * y + 7.0
    result = resize(image, (18, 16), "bilinear")
    # Исключаем граничные позиции, где выбранное продолжение clamp изменяет линейную рампу.
    for out_y in range(2, 14):
        src_y = (out_y + 0.5) * 8 / 16 - 0.5
        for out_x in range(2, 16):
            src_x = (out_x + 0.5) * 9 / 18 - 0.5
            assert result[out_y, out_x] == pytest.approx(3 * src_x + 2 * src_y + 7)


def test_keys_kernel_reproduces_samples_at_identity_grid():
    rng = np.random.default_rng(12)
    image = rng.normal(size=(9, 11))
    np.testing.assert_allclose(resize(image, (11, 9), "bicubic"), image, atol=1e-10)


def test_cubic_bspline_prefilter_reproduces_original_samples():
    from interpolation_study.kernels import cubic_bspline
    from interpolation_study.resize import _resize_axis, _spline_prefilter_axis

    rng = np.random.default_rng(13)
    image = rng.normal(size=(7, 9))
    coefficients = _spline_prefilter_axis(image, axis=1, boundary="reflect")
    coefficients = _spline_prefilter_axis(coefficients, axis=0, boundary="reflect")
    horizontal = _resize_axis(
        coefficients, 9, axis=1, kernel=cubic_bspline, radius=2.0,
        antialias=False, boundary="reflect", coordinate_mode="half_pixel",
    )
    reconstructed = _resize_axis(
        horizontal, 7, axis=0, kernel=cubic_bspline, radius=2.0,
        antialias=False, boundary="reflect", coordinate_mode="half_pixel",
    )
    np.testing.assert_allclose(reconstructed, image, atol=1e-9)


def test_cubic_bspline_prefilter_clamp_boundary_reproduces_samples():
    from interpolation_study.kernels import cubic_bspline
    from interpolation_study.resize import _resize_axis, _spline_prefilter_axis

    rng = np.random.default_rng(14)
    image = rng.normal(size=(5, 8))
    coefficients = _spline_prefilter_axis(image, axis=1, boundary="clamp")
    coefficients = _spline_prefilter_axis(coefficients, axis=0, boundary="clamp")
    horizontal = _resize_axis(
        coefficients, 8, axis=1, kernel=cubic_bspline, radius=2.0,
        antialias=False, boundary="clamp", coordinate_mode="half_pixel",
    )
    reconstructed = _resize_axis(
        horizontal, 5, axis=0, kernel=cubic_bspline, radius=2.0,
        antialias=False, boundary="clamp", coordinate_mode="half_pixel",
    )
    np.testing.assert_allclose(reconstructed, image, atol=1e-9)


def test_mitchell_default_is_not_cardinal_but_catmull_rom_variant_is():
    from interpolation_study.kernels import mitchell_netravali

    assert float(mitchell_netravali(np.array([0.0]))[0]) == pytest.approx(8.0 / 9.0)
    values = mitchell_netravali(np.array([0.0, 1.0, 2.0]), b=0.0, c=0.5)
    np.testing.assert_allclose(values, [1.0, 0.0, 0.0], atol=1e-12)


def test_mitchell_kernel_is_not_cardinal_but_preserves_constants():
    image = np.full((5, 6), 127.0)
    result = resize(image, (12, 10), "mitchell")
    np.testing.assert_allclose(result, 127.0, atol=1e-8)


def test_area_downsampling_averages_pixel_footprints():
    image = np.array(
        [[0.0, 2.0, 4.0, 6.0], [8.0, 10.0, 12.0, 14.0],
         [16.0, 18.0, 20.0, 22.0], [24.0, 26.0, 28.0, 30.0]]
    )
    expected = np.array([[5.0, 9.0], [21.0, 25.0]])
    np.testing.assert_allclose(resize(image, (2, 2), "area"), expected)


@pytest.mark.parametrize("boundary", ["clamp", "reflect", "wrap", "constant"])
def test_boundary_modes_produce_finite_values(boundary):
    image = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = resize(image, (13, 11), "lanczos3", boundary=boundary, cval=17.0)
    assert np.isfinite(result).all()


def test_constant_boundary_fill_value_is_used_outside_source_extent():
    image = np.array([[5.0]])
    result_zero = resize(image, (3, 3), "bilinear", boundary="constant", cval=0.0)
    result_nonzero = resize(image, (3, 3), "bilinear", boundary="constant", cval=17.0)
    assert result_zero[0, 0] < 5.0
    assert result_nonzero[0, 0] > result_zero[0, 0]


def test_reflect_spline_boundary_handles_single_pixel_axis():
    image = np.array([[9.0, 3.0, 7.0]])
    result = resize(image, (9, 1), "bspline3", boundary="reflect")
    assert result.shape == (1, 9)
    assert np.isfinite(result).all()


def test_align_corners_preserves_corner_samples():
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = resize(image, (5, 5), "bilinear", coordinate_mode="align_corners")
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, -1] == pytest.approx(2.0)
    assert result[-1, 0] == pytest.approx(3.0)
    assert result[-1, -1] == pytest.approx(4.0)


def test_from_scratch_resizer_has_no_image_library_import_dependency():
    import inspect
    import interpolation_study.resize as resize_module

    source = inspect.getsource(resize_module)
    for forbidden in ("PIL", "cv2", "scipy", "skimage"):
        assert forbidden not in source


def test_one_pixel_image_can_be_enlarged_by_every_method():
    image = np.array([[42.0]])
    for method in METHODS:
        result = resize(image, (9, 7), method)
        np.testing.assert_allclose(result, 42.0, atol=1e-8)


def test_bicubic_parameter_is_exposed_and_changes_output():
    image = np.array([[0.0, 0.0, 0.0, 255.0], [0.0, 0.0, 255.0, 255.0]])
    first = resize(image, (19, 7), "bicubic", keys_a=-0.5)
    second = resize(image, (19, 7), "bicubic", keys_a=-0.75)
    assert not np.allclose(first, second)


def test_invalid_spline_boundary_is_rejected_even_for_identity_size():
    with pytest.raises(ValueError):
        resize(np.ones((3, 4)), (4, 3), "bspline3", boundary="wrap")


def test_empty_channel_dimension_is_rejected():
    with pytest.raises(ValueError):
        resize(np.empty((3, 4, 0)), (5, 6))


def test_area_resampling_preserves_spatial_integral_for_non_integer_scale():
    rng = np.random.default_rng(18)
    image = rng.random((5, 7))
    destination = resize(image, (4, 3), "area")
    scale_x = image.shape[1] / destination.shape[1]
    scale_y = image.shape[0] / destination.shape[0]
    np.testing.assert_allclose(destination.sum() * scale_x * scale_y, image.sum(), atol=1e-12)



def test_reflect_boundary_is_whole_sample_symmetric():
    from interpolation_study.resize import _map_indices

    mapped, valid = _map_indices(np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5]), 3, "reflect")
    np.testing.assert_array_equal(mapped, [1, 2, 1, 0, 1, 2, 1, 0, 1])
    assert valid.all()


def test_area_resampling_uses_pixel_footprints_not_point_mapping_options():
    image = np.arange(35, dtype=np.float64).reshape(5, 7)
    expected = resize(image, (3, 2), "area")
    actual = resize(
        image,
        (3, 2),
        "area",
        antialias=False,
        boundary="constant",
        coordinate_mode="align_corners",
        cval=999.0,
    )
    np.testing.assert_allclose(actual, expected)


def test_antialiasing_changes_high_frequency_downsampling():
    x = np.arange(64)
    pattern = ((np.sin(2.0 * np.pi * x / 4.0) > 0) * 255.0).astype(np.float64)
    image = np.tile(pattern[None, :], (16, 1))
    filtered = resize(image, (8, 16), "lanczos3", antialias=True)
    unfiltered = resize(image, (8, 16), "lanczos3", antialias=False)
    assert np.mean(np.abs(filtered - unfiltered)) > 10.0



def test_chunked_resampling_matches_unchunked_result(monkeypatch):
    import importlib

    resize_module = importlib.import_module("interpolation_study.resize")
    rng = np.random.default_rng(19)
    image = rng.normal(size=(37, 41, 3))
    expected = resize(image, (53, 29), "lanczos3")
    monkeypatch.setattr(resize_module, "_MAX_GATHER_BYTES", 1024)
    actual = resize(image, (53, 29), "lanczos3")
    np.testing.assert_allclose(actual, expected, atol=1e-12)
