"""Download the official DIV2K validation HR and bicubic x4 LR archives."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "DIV2K_valid_HR.zip": (
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    ),
    "DIV2K_valid_LR_bicubic_X4.zip": (
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
    ),
}


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "image-interpolation-study/1.0"},
    )
    print(f"Downloading {url}")
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    print(f"Saved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(
                    f"Unsafe path in archive {archive_path.name}: {member.filename}"
                )
        archive.extractall(destination)

    print(f"Extracted {archive_path.name} into {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/DIV2K"))
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP files after extraction.",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        archive_path = args.root / filename
        if not archive_path.exists():
            download(url, archive_path)
        else:
            print(f"Using existing archive: {archive_path}")

        safe_extract(archive_path, args.root)
        if not args.keep_archives:
            archive_path.unlink(missing_ok=True)

    print(
        "\nDataset download complete. DIV2K is provided for academic research "
        "only; see the official dataset page before reuse or redistribution."
    )


if __name__ == "__main__":
    main()
