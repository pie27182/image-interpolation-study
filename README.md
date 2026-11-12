# Comparative Analysis of Digital Image Interpolation Algorithms Using Python

A reproducible research project for a thesis on classical digital image
interpolation. Its central contribution is a transparent, independently
written resampling implementation whose mathematical operations can be
studied and compared with an established library.

## Two deliberately separate stages

1. **Library baseline:** Pillow filters are used as a reference for the methods
   that it provides.
2. **From-scratch implementation:** interpolation kernels, coordinate
   transforms, tap selection, edge handling, anti-aliasing, separable filtering,
   B-spline coefficient prefiltering, area integration, and quality metrics
   are implemented explicitly using NumPy.

The scratch interpolation code does **not** call Pillow, OpenCV, SciPy,
scikit-image, or another image interpolation/resizing API. Pillow is retained
for image file I/O and the optional baseline only. The project does not copy
code from those libraries; its equations and algorithms are based on the
original papers and references in `docs/research-sources.md`.

## Implemented methods

| Method | Mathematical model | Main properties |
|---|---|---|
| Nearest neighbour | Closest sample | Very fast; discrete labels; blocky enlargement |
| Bilinear | Triangle/hat kernel | Simple, nonnegative, smooth but soft |
| Bicubic | Keys cubic convolution, `a=-0.5` | Cardinal cubic; sharp; may ring |
| Mitchell--Netravali | Two-parameter cubic, default `B=C=1/3` | Non-cardinal reconstruction filter; blur/ringing compromise |
| Cubic B-spline | Cubic B-spline plus coefficient prefilter | Smooth interpolating spline; prefilter is essential |
| Lanczos-3 | Windowed sinc, radius 3 | Good detail retention; possible ringing |
| Area | Exact pixel-footprint overlap | Explicit area averaging; especially useful for downsampling |

The repository also includes a generated comparison of the spatial kernels,
approximate frequency responses and step responses:

![Kernel and response diagnostics](examples/kernel_diagnostics.png)

The full derivations, mathematical connections, implementation conventions,
limitations and research literature are in:

- [`docs/theoretical-foundations.md`](docs/theoretical-foundations.md) — the
  broad mathematical theory: sampling, Fourier analysis, reconstruction,
  approximation theory, splines, anti-aliasing, colour, geometry and metrics.
- [`docs/implementation-from-scratch.md`](docs/implementation-from-scratch.md)
  — equations and step-by-step design of the actual NumPy implementation.
- [`docs/algorithm-notes.md`](docs/algorithm-notes.md) — concise method and
  reproducibility notes.
- [`docs/research-sources.md`](docs/research-sources.md) — original papers,
  books, official docs and implementation-source audits.
- [`docs/experiment-protocol.md`](docs/experiment-protocol.md) — experimental
  methodology for thesis work.

## Project layout

```text
image-interpolation-study/
├── .github/workflows/tests.yml
├── docs/
│   ├── algorithm-notes.md
│   ├── experiment-protocol.md
│   ├── implementation-from-scratch.md
│   ├── research-sources.md
│   └── theoretical-foundations.md
├── examples/
│   └── generate_test_images.py
├── src/interpolation_study/
│   ├── benchmark.py       # repeated quality/runtime experiments
│   ├── cli.py             # command line interface
│   ├── kernels.py         # explicit mathematical kernel formulas
│   ├── library.py         # optional Pillow reference baseline
│   ├── metrics.py         # NumPy MSE, PSNR and uniform-window SSIM
│   └── resize.py          # NumPy-only interpolation and resampling
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

Dependencies are NumPy and Pillow. Pillow is for image I/O and the optional
baseline; the `scratch` resizer and metrics use NumPy only. Matplotlib is an
optional extra for any user-created plots: `pip install -e '.[plots]'`.

## Resize an image

From-scratch Keys bicubic enlargement:

```bash
image-interpolation resize examples/input.jpg results/bicubic.png \
  --width 1200 --height 800 --method bicubic --implementation scratch
```

From-scratch cubic B-spline interpolation:

```bash
image-interpolation resize examples/input.jpg results/bspline.png \
  --width 1200 --height 800 --method bspline3 --boundary reflect
```

Area downsampling:

```bash
image-interpolation resize examples/input.jpg results/half.png \
  --width 600 --height 400 --method area
```

Optional Pillow baseline:

```bash
image-interpolation resize examples/input.jpg results/pillow-bicubic.png \
  --width 1200 --height 800 --method bicubic --implementation pillow
```

Available scratch methods: `nearest`, `bilinear`, `bicubic`, `mitchell`,
`bspline3`, `lanczos3`, `area`.

Additional controls:

- `--boundary clamp|reflect|wrap|constant` (B-spline supports clamp/reflect).
  Here `reflect` means whole-sample symmetry (index -1 maps to 1), matching
  SciPy `mirror` / OpenCV `BORDER_REFLECT_101`, not SciPy `reflect`.
- `--coordinate-mode half_pixel|align_corners`.
- `--no-antialias` disables scale-aware kernel widening during reduction.
- `--keys-a` selects the Keys cubic parameter.
- `--mitchell-b` and `--mitchell-c` select Mitchell--Netravali parameters.
- `--fill-value` controls the constant boundary value.

## Run a benchmark

Generate deterministic synthetic images:

```bash
python examples/generate_test_images.py
```

Run a benchmark against the synthetic edge image:

```bash
image-interpolation benchmark examples/synthetic_edges.png \
  --scale 2 --degradation lanczos3 --warmup 1 --repeats 5 \
  --output-dir results/synthetic-edges
```

The benchmark creates one common low-resolution input with the scratch
implementation by default, applies every available Pillow method and every
scratch method, saves reconstructed images, and writes `results.csv`. To
explicitly use Pillow for the degradation step, add
`--degradation-implementation pillow`. The CSV reports median/mean/standard deviation/minimum
runtime, MSE, PSNR, the uniform-window SSIM variant, dimensions and degradation
settings. Pillow offers no directly equivalent named Mitchell or B-spline
baseline, so those methods are scratch-only in the comparison table.

**Research caution:** repeat the experiment with several degradation filters.
A Lanczos-generated input can favour a Lanczos reconstruction; this is not
evidence that Lanczos is universally best. Use multiple images, content classes
and scale factors. The original high-resolution image is the known reference
for synthetic degradation experiments, not a perfect ground truth for every
real-world resizing task.

## Tests

```bash
pytest -q
```

Tests cover mathematical invariants, dimensions, boundaries, coordinate
conventions, spline coefficient reconstruction, area averaging, and metrics.
The CI workflow runs tests on supported Python versions.

## Methodological limitations

- The implementation currently operates on scalar channels independently.
  It does not perform linear-light colour conversion or alpha premultiplication
  automatically. For physically meaningful colour averaging, decode sRGB to
  linear light before interpolation; for RGBA, consider premultiplying colour
  by alpha before filtering and unpremultiplying afterwards.
- Mitchell--Netravali with `B=C=1/3` is a non-cardinal reconstruction filter.
  It preserves constants but does not reproduce each input sample. A same-size
  resize is special-cased as the identity.
- The B-spline coefficient system has explicit finite-image whole-sample
  symmetric `reflect` and `clamp` boundary models. Its prefilter is required;
  applying the basis
  directly to raw pixel samples is not cubic B-spline interpolation.
- Area resampling integrates pixel footprints and is designed primarily for
  reduction; it is not intended as a sharp photographic upscaler.
- Coordinate maps, boundary models, anti-aliasing filters, colour treatment,
  rounding and precision all affect results. Record them in thesis experiments.
- MSE, PSNR and SSIM are not complete perceptual models. Include visual
  inspection and discuss aliasing, blur, ringing, edge sharpness and texture.

## License

MIT. See `LICENSE`.
