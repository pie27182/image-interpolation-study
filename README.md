# Comparative Analysis of Digital Image Interpolation Algorithms Using Python

A reproducible research project for comparing classical image-resizing methods.
It has two deliberately separate stages:

1. **Library baseline:** Pillow's built-in nearest, bilinear, bicubic and Lanczos
   resampling filters.
2. **From-scratch implementation:** the same families reimplemented from their
   mathematical kernels using NumPy. The scratch implementation does not call
   Pillow, OpenCV, SciPy or scikit-image resize functions.

The goal is not to claim that one method is universally best. It is to measure
quality and runtime under clearly stated experimental conditions.

## Algorithms

| Method | Mathematical idea | Typical properties |
|---|---|---|
| Nearest neighbour | Select the closest source pixel | Very fast; preserves hard labels and pixel-art blocks; visibly jagged for photographs |
| Bilinear | Weighted average of a 2x2 neighbourhood | Smooth and inexpensive; can soften detail |
| Bicubic | Keys cubic convolution, 4x4 neighbourhood | Often sharper than bilinear; may overshoot near strong edges |
| Lanczos-3 | Windowed sinc, radius 3 | Retains detail well; can create ringing around sharp transitions |

Implementation conventions and equations are described in
[`docs/algorithm-notes.md`](docs/algorithm-notes.md). A broader, thesis-oriented
mathematical treatment is available in
[`docs/theoretical-foundations.md`](docs/theoretical-foundations.md). Research
references are in [`docs/research-sources.md`](docs/research-sources.md).

## Project layout

```text
image-interpolation-study/
├── docs/
│   ├── algorithm-notes.md
│   └── research-sources.md
├── examples/                 # Put source photographs/test images here
├── src/interpolation_study/
│   ├── benchmark.py           # Reproducible quality/runtime experiment
│   ├── cli.py                 # Command-line interface
│   ├── kernels.py             # Linear, Keys cubic and Lanczos-3 kernels
│   ├── library.py             # Pillow reference implementation
│   ├── metrics.py             # MSE, PSNR, SSIM
│   └── resize.py              # NumPy from-scratch resizer
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/<your-account>/image-interpolation-study.git
cd image-interpolation-study
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Resize one image

The default implementation is the hand-coded NumPy version:

```bash
image-interpolation resize examples/input.jpg results/bilinear.png \
  --width 1200 --height 800 --method bilinear --implementation scratch
```

Run the Pillow reference for comparison:

```bash
image-interpolation resize examples/input.jpg results/pillow-bilinear.png \
  --width 1200 --height 800 --method bilinear --implementation pillow
```

Available methods: `nearest`, `bilinear`, `bicubic`, `lanczos3`.

## Run the full benchmark

Place a high-resolution reference image in `examples/`, then run:

```bash
image-interpolation benchmark examples/input.jpg --scale 2 \
  --degradation lanczos3 --output-dir results/experiment-01
```

The experiment performs the following steps:

1. Reads the supplied image as the high-resolution reference.
2. Creates one low-resolution input at the chosen scale using the selected
   Pillow degradation filter (Lanczos-3 by default).
3. Upscales that same low-resolution image with four Pillow filters and four
   from-scratch algorithms.
4. Saves every reconstructed image, the low-resolution input, and `results.csv`.
5. Reports elapsed milliseconds, MSE, PSNR and SSIM for each reconstruction.

The original image is the reference for quality metrics. The selected
low-resolution degradation is part of the experiment and is recorded in the
CSV. Repeat the benchmark with several images, scale factors and degradation
filters; the default Lanczos degradation gives the Pillow Lanczos baseline an
advantage, so it must not be the only experiment in a thesis.

## Run tests

```bash
pytest
```

## Methodological limitations

- Pillow and the scratch implementation may differ because libraries can use
  different coordinate mappings, edge policies, cubic parameters, internal
  precision and downsampling strategies. Such differences are documented, not
  silently treated as bugs.
- The scratch implementation returns floating-point arrays. The command-line
  tool rounds and clips to 8-bit values when saving images.
- Timing depends on image dimensions, CPU, NumPy build, warm-up and machine load.
  Run repeated trials and report medians for thesis-grade measurements.
- MSE/PSNR/SSIM are useful but do not capture every perceptual property. Include
  side-by-side crops and discuss ringing, sharpness, aliasing and edge quality.
- The benchmark measures reconstruction from an artificially degraded source;
  it is not a claim about every real-world image pipeline.

## Suggested thesis experiments

1. Upscaling: 2x, 3x and 4x.
2. Downscaling: 0.75x, 0.5x and 0.25x (use the resize command and inspect aliasing).
3. Image classes: portraits, landscapes, fine textures, text/line art and pixel art.
4. Degradation filters: repeat the benchmark using nearest, bilinear, bicubic and
   Lanczos-3 to test whether the result depends on the low-resolution generation.
5. Runtime: run each method several times and report median and standard deviation.
6. Quality: report MSE, PSNR and SSIM together with visual crops and a discussion
   of the limits of each metric.

## License

MIT. See `LICENSE`.

## Generate deterministic test images

```bash
python examples/generate_test_images.py
image-interpolation benchmark examples/synthetic_edges.png --scale 2 \
  --degradation lanczos3 --output-dir results/synthetic-edges
```

For thesis experiments, see [`docs/experiment-protocol.md`](docs/experiment-protocol.md).
