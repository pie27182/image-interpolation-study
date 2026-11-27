from PIL import Image

from src.interpolation_benchmark.methods import (
    BASELINE_METHOD,
    DEFAULT_METHODS,
    get_method_filters,
)


def test_all_primary_methods_are_available():
    filters = get_method_filters()
    assert list(filters) == list(DEFAULT_METHODS)
    assert BASELINE_METHOD in filters
    assert all(isinstance(value, Image.Resampling) for value in filters.values())


def test_resize_returns_requested_size():
    image = Image.new("RGB", (12, 10), color=(80, 120, 160))
    target = (48, 40)
    for resampler in get_method_filters().values():
        assert image.resize(target, resample=resampler).size == target
