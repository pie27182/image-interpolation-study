# From-scratch implementation guide

This document describes the actual implementation in `src/interpolation_study/`.
The routines are written for this project from the equations in the cited
literature. NumPy supplies arrays, elementary arithmetic, indexing, cumulative
sums and linear-algebra-friendly storage; no call to Pillow, OpenCV, SciPy or
scikit-image performs interpolation in the scratch path.

Pillow remains in a separate module for reading/writing common image formats
and for the optional library-baseline experiment. This separation matters:
`resize.py` and `kernels.py` do not import it.

## 1. Common coordinate model

For source length $S$, destination length $D$, and destination index
$j$, the default half-pixel mapping is

$$
 x_j=(j+\tfrac12)\frac{S}{D}-\tfrac12.
$$

It maps destination pixel centres to source pixel-centre coordinates. The
inverse mapping is used because each output pixel asks where to sample the
input. When source and destination dimensions are equal, the operation is
explicitly the identity (including for reconstruction filters which are not
cardinal).

The alternative `align_corners` mapping is

$$
 x_j=j\frac{S-1}{D-1},\qquad D>1,
$$

with the centre of the source chosen when $D=1$. These coordinate choices
are not interchangeable. A comparison must record which one it uses.

## 2. General separable resampling

For a one-dimensional kernel $K$, output coordinate $x_j$, support radius
$R$, and source-to-destination scale $s=S/D$, a source sample $i$
contributes the weight

$$
 w_{j,i}=K\left(\frac{x_j-i}{q}\right),\qquad
 q=\max(1,s)
$$

when anti-aliasing is enabled. Its normalized weight is

$$
 \widetilde w_{j,i}=\frac{w_{j,i}}{\sum_k w_{j,k}}.
$$

Only integer samples in the effective support $|x_j-i|<Rq$ are included.
The width grows during downscaling, so the resampler integrates a wider source
neighbourhood before producing a more sparsely sampled output. The implementation
precomputes the indices, validity flags and normalized weights for each axis.
It then performs horizontal and vertical passes. For a separable kernel,

$$
 K_{2D}(x,y)=K(x)K(y),
$$

so the two one-dimensional passes are mathematically equivalent to a direct
2D weighted sum, while using substantially fewer multiply-accumulate operations.

The filter weights are normalized for each output position. This preserves a
constant image even when part of a kernel footprint falls outside the finite
image. The selected boundary mode determines what those outside samples mean.

## 3. Implemented kernels

### Nearest neighbour

The source index is the closest integer centre:

$$
 i_j=\lfloor x_j+\tfrac12\rfloor.
$$

No averaging occurs. This keeps label values discrete, but is not an
anti-aliased reduction filter.

### Bilinear

The one-dimensional triangle kernel is

$$
 K_{lin}(x)=\max(1-|x|,0).
$$

In two dimensions its product form uses the four surrounding source samples.
The weights are nonnegative and sum to one. The method reproduces constants
and linear ramps away from boundary-extension effects, but its frequency
response attenuates fine detail.

### Keys bicubic / cubic convolution

For $r=|x|$, Keys' kernel is

$$
 K_a(x)=\begin{cases}
 (a+2)r^3-(a+3)r^2+1,&0\le r<1,\\
 ar^3-5ar^2+8ar-4a,&1\le r<2,\\
 0,&r\ge2.
 \end{cases}
$$

The default is $a=-0.5$; `keys_a` makes the parameter explicit. This is a
cardinal kernel: it is one at zero and zero at other integer offsets. Its
support radius is two, giving four taps per axis for ordinary enlargement.
Negative lobes can increase apparent sharpness and also produce overshoot or
undershoot next to sharp transitions.

### Mitchell--Netravali cubic family

The two-parameter cubic family is

$$
 K_{B,C}(x)=\begin{cases}
 \frac{(12-9B-6C)r^3+(-18+12B+6C)r^2+(6-2B)}{6},&0\le r<1,\\[3pt]
 \frac{(-B-6C)r^3+(6B+30C)r^2+(-12B-48C)r+(8B+24C)}{6},&1\le r<2,\\[3pt]
 0,&r\ge2,
 \end{cases}
 \qquad r=|x|.
$$

The default $B=C=1/3$ is a non-cardinal reconstruction filter: its central
value is $8/9$, not one. The kernel is normalized at each output position,
which preserves constants, but individual input samples are not necessarily
reproduced. The `mitchell_b` and `mitchell_c` CLI/API parameters are recorded
in experiment metadata when a custom run is prepared. The family includes
other useful cubic behaviours; for example $B=0,C=1/2$ is Catmull--Rom-like.

### Cubic B-spline with coefficient prefilter

The centred cubic B-spline basis is

$$
 \beta^3(x)=\begin{cases}
 \frac{4-6|x|^2+3|x|^3}{6},&|x|<1,\\
 \frac{(2-|x|)^3}{6},&1\le |x|<2,\\
 0,&|x|\ge2.
 \end{cases}
$$

It is smooth and compactly supported, but it is not cardinal. Substituting raw
pixel values as spline coefficients would blur even at the original sample
locations. The code therefore computes coefficients first. At integer knots,

$$
 f[n]=\frac16c[n-1]+\frac23c[n]+\frac16c[n+1].
$$

For an interior line, this is a tridiagonal linear system. Under the selected
finite-image boundary condition, the endpoint equations are modified:

- Whole-sample reflection (the implementation's `reflect` mode) uses
  $f[0]=\frac23c[0]+\frac13c[1]$, with the symmetric equation at the
  opposite end.
- Edge clamping uses
  $f[0]=\frac56c[0]+\frac16c[1]$, again symmetrically at the other end.

The implementation solves each line with the Thomas algorithm, a specialized
$O(N)$ solver for tridiagonal systems. It solves many image rows/columns at
once by keeping the non-filtered dimensions in a matrix of right-hand sides.
The prefilter is applied horizontally and vertically, after which the cubic
B-spline basis is evaluated at the requested coordinates. The spline mode
therefore supports `reflect` and `clamp` boundaries only.

This is an interpolating cubic B-spline implementation, not a direct
convolution of the original samples with a cubic B-spline basis. The
coefficient prefilter is mathematically essential.

### Lanczos-3

The normalized sinc and radius-three windowed-sinc kernel are

$$
 \mathrm{sinc}(x)=\frac{\sin(\pi x)}{\pi x},\quad
 \mathrm{sinc}(0)=1,
$$

$$
 K_{L,3}(x)=\begin{cases}
 \mathrm{sinc}(x)\mathrm{sinc}(x/3),&|x|<3,\\
 0,&|x|\ge3.
 \end{cases}
$$

The support radius is three source pixels. Windowing the ideal sinc makes the
filter finite, but it introduces a trade-off: the filter is no longer ideal
brick-wall reconstruction and can exhibit ringing because of its oscillatory
side lobes.

### Area / pixel-footprint averaging

Area resampling is implemented separately from point-kernel reconstruction.
Source pixel $i$ occupies $[i-1/2,i+1/2]$. For scale $s=S/D$,
destination pixel $j$ covers

$$
 A_j=[js-\tfrac12,(j+1)s-\tfrac12]
$$

in source coordinates. The weight of source pixel $i$ is its exact overlap
with that footprint:

$$
 w_{j,i}=\max\left(0,
 \min((j+1)s-\tfrac12,i+\tfrac12)
 -\max(js-\tfrac12,i-\tfrac12)\right).
$$

Weights are divided by their sum. The 2D operation is separable rectangle
integration. It is particularly meaningful for reduction when pixels represent
area averages. For enlargement, a shrinking footprint can behave like a
piecewise-constant reconstruction, so area is not intended as a photographic
sharp upscaler.

## 4. Boundary extension

A finite image has no samples outside its stored extent. A filter footprint may
nevertheless extend beyond the edge. This project makes that policy explicit:

- `clamp`: all out-of-range indices map to the nearest edge sample.
- `reflect`: whole-sample symmetric reflection without repeating the edge;
  for example, index -1 maps to 1. This corresponds to SciPy `mirror` and
  OpenCV `BORDER_REFLECT_101`, not SciPy `reflect` (half-sample symmetry).
- `wrap`: indices are periodic modulo the image length.
- `constant`: out-of-range taps use `cval`.

Boundary policies can materially change the first and last few output pixels.
The B-spline prefilter uses endpoint equations matched to the `reflect` or
`clamp` coefficient-extension model.

## 5. Image-quality metrics, implemented in this project

MSE and PSNR are direct NumPy calculations:

$$
 \mathrm{MSE}=\frac1N\sum_n(I_n-\widehat I_n)^2,
 \qquad
 \mathrm{PSNR}=10\log_{10}\frac{L^2}{\mathrm{MSE}}.
$$

An exact match has zero MSE and infinite PSNR. `data_range` is 255 by default
for 8-bit images and must be set to 1 for normalized [0,1] data.

SSIM is implemented from its local statistics equation:

$$
 \mathrm{SSIM}(x,y)=
 \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
 {(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
$$

This repository uses a uniform square window (default 7x7) and the sample
covariance correction $N/(N-1)$. The local means, second moments and cross
moments are computed by separable running sums (cumulative sums), not by
scikit-image. This setting agrees with the standard uniform-window formulation
when the same window, channel treatment and covariance convention are used.
Other SSIM implementations may use Gaussian windows or different border and
channel rules, so record these settings when comparing results.

## 6. What is and is not an external dependency

| Component | Implementation | External image-interpolation API? |
|---|---|---:|
| Kernels | `kernels.py`, explicit equations | No |
| Coordinate transform | `resize.py`, explicit equations | No |
| Tap generation and edge mapping | `resize.py`, explicit loops and NumPy arrays | No |
| Separable resampling | `resize.py`, explicit gather / multiply / sum | No |
| B-spline coefficient prefilter | `resize.py`, Thomas tridiagonal solver | No |
| Area overlap | `resize.py`, explicit interval intersections | No |
| MSE / PSNR / SSIM | `metrics.py`, NumPy | No |
| Image file I/O | Pillow | Not an interpolation operation |
| Optional comparison baseline | Pillow resampling functions | Yes, baseline only |

For academic transparency, the formulas and algorithmic design are attributed
to the literature in `research-sources.md`. The project code is a new,
independently written implementation and does not copy the source code of
Pillow, OpenCV, SciPy or scikit-image.

## 7. Reproduction and validation checklist

The automated tests cover:

- Output dimensions, grayscale and multichannel inputs, and invalid arguments.
- Identity resizing and constant-image preservation for every method.
- Nearest-neighbour discrete-value preservation.
- Bilinear reproduction of an interior affine ramp.
- Cardinal Keys kernel sample reproduction.
- B-spline coefficient prefilter reconstruction under both supported boundary
  conditions.
- Mitchell's non-cardinal centre value and the Catmull--Rom-like parameter set.
- Exact area averages for a 2x2 pixel-footprint reduction.
- All boundary modes and both coordinate modes.
- Metric values for known images, including identical-image SSIM and infinite
  PSNR.

These tests verify implementation properties, not universal perceptual quality.
For thesis conclusions, use diverse reference images, multiple degradation
filters, multiple scale factors, repeated timings and visual inspection.
