"""Use maintained Pillow resampling implementations; do not reimplement kernels."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, __version__ as PILLOW_VERSION

BASELINE_METHOD = "Bicubic"

# Keep this list fixed for the primary experiment. All methods are applied by
# the same Pillow Image.resize() API and to the same target dimensions.
DEFAULT_METHODS = (
    "Bilinear",
    "Bicubic",
    "Lanczos",
    "MKS2013",
    "MKS2021",
)


def get_method_filters(
    method_names: Sequence[str] | None = None,
) -> dict[str, Image.Resampling]:
    """Return Pillow resampling enums by the human-readable method name.

    Pillow 13.0.0 is pinned because MKS2013 and MKS2021 were added in that
    release. The code intentionally calls Pillow's maintained implementation
    instead of copying third-party C code into this repository.
    """
    available = {
        "Bilinear": Image.Resampling.BILINEAR,
        "Bicubic": Image.Resampling.BICUBIC,
        "Lanczos": Image.Resampling.LANCZOS,
    }

    for name in ("MKS2013", "MKS2021"):
        value = getattr(Image.Resampling, name, None)
        if value is not None:
            available[name] = value

    requested = tuple(method_names) if method_names is not None else DEFAULT_METHODS
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise RuntimeError(
            "Unavailable resampling method(s): "
            + ", ".join(unknown)
            + f". Installed Pillow version: {PILLOW_VERSION}. "
            "Install the pinned requirements with: pip install -r requirements.txt"
        )

    if len(set(requested)) != len(requested):
        raise ValueError("The methods list contains duplicates.")
    if BASELINE_METHOD not in requested:
        raise ValueError(
            f"The primary comparison must include the baseline {BASELINE_METHOD!r}."
        )

    return {name: available[name] for name in requested}
