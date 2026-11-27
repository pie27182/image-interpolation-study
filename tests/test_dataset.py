from PIL import Image

from src.interpolation_benchmark.dataset import discover_div2k_pairs


def _write_rgb(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(90, 120, 150)).save(path)


def test_discover_pairs_by_official_div2k_filenames(tmp_path):
    hr_dir = tmp_path / "DIV2K_valid_HR"
    lr_dir = tmp_path / "DIV2K_valid_LR_bicubic" / "X4"

    _write_rgb(hr_dir / "0801.png")
    _write_rgb(hr_dir / "0802.png")
    _write_rgb(lr_dir / "0801x4.png")
    _write_rgb(lr_dir / "0802x4.png")

    pairs = discover_div2k_pairs(tmp_path, scale=4)

    assert [pair.image_id for pair in pairs] == ["0801", "0802"]
    assert pairs[0].hr_path.name == "0801.png"
    assert pairs[0].lr_path.name == "0801x4.png"
