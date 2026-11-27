"""Discovery of the official DIV2K validation HR/LR image pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImagePair:
    image_id: str
    lr_path: Path
    hr_path: Path


def _find_directory(root: Path, exact_name: str) -> Path | None:
    direct = root / exact_name
    if direct.is_dir():
        return direct
    matches = sorted(
        (path for path in root.rglob(exact_name) if path.is_dir()),
        key=lambda path: (len(path.parts), str(path)),
    )
    return matches[0] if matches else None


def discover_div2k_pairs(data_root: str | Path, scale: int = 4) -> list[ImagePair]:
    """Find DIV2K validation images using their official filenames.

    Expected archive layout after extraction:
      DIV2K_valid_HR/0801.png ... 0900.png
      DIV2K_valid_LR_bicubic/X4/0801x4.png ... 0900x4.png

    The archives can be extracted under an additional parent directory;
    directory discovery is recursive to support both layouts.
    """
    if scale < 2:
        raise ValueError("scale must be >= 2")

    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. "
            "Run scripts/download_div2k.py first."
        )

    hr_dir = _find_directory(root, "DIV2K_valid_HR")
    lr_dir = _find_directory(root, f"X{scale}")

    if hr_dir is None:
        raise FileNotFoundError(
            f"Could not find DIV2K_valid_HR under {root}. "
            "Check that DIV2K_valid_HR.zip has been extracted."
        )
    if lr_dir is None or "LR_bicubic" not in str(lr_dir):
        candidates = sorted(
            (
                path for path in root.rglob("*")
                if path.is_dir()
                and "LR_bicubic" in str(path)
                and (path.name == f"X{scale}" or f"X{scale}" in str(path))
            ),
            key=lambda path: (len(path.parts), str(path)),
        )
        lr_dir = candidates[0] if candidates else None

    if lr_dir is None:
        raise FileNotFoundError(
            f"Could not find the DIV2K bicubic LR X{scale} directory under {root}. "
            "Check that DIV2K_valid_LR_bicubic_X4.zip has been extracted."
        )

    hr_by_id: dict[str, Path] = {}
    for path in hr_dir.glob("*.png"):
        if re.fullmatch(r"\d{4}", path.stem):
            hr_by_id[path.stem] = path

    lr_by_id: dict[str, Path] = {}
    pattern = re.compile(rf"(\d{{4}})x{scale}")
    for path in lr_dir.glob("*.png"):
        match = pattern.fullmatch(path.stem)
        if match:
            lr_by_id[match.group(1)] = path

    common_ids = sorted(set(hr_by_id) & set(lr_by_id))
    if not common_ids:
        raise RuntimeError(
            "No matching DIV2K HR/LR files were found. "
            f"HR directory: {hr_dir}; LR directory: {lr_dir}"
        )

    missing_lr = sorted(set(hr_by_id) - set(lr_by_id))
    missing_hr = sorted(set(lr_by_id) - set(hr_by_id))
    if missing_lr or missing_hr:
        print(
            "Warning: unmatched files skipped. "
            f"HR without LR: {len(missing_lr)}; LR without HR: {len(missing_hr)}"
        )

    return [
        ImagePair(image_id=image_id, lr_path=lr_by_id[image_id], hr_path=hr_by_id[image_id])
        for image_id in common_ids
    ]
