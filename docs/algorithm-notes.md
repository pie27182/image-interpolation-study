# Algorithm notes

This project deliberately defines its own conventions. Results from two image
libraries need not be pixel-identical even when their filters share a name:
coordinate transforms, edge extension, coefficient choices, and downsampling
anti-aliasing can differ.

## Pixel-centre mapping

For one axis, let `S` be the source length and `D` the destination length. The
source coordinate corresponding to destination pixel `j` is

\[
 x_j = (j + 1/2)\frac{S}{D} - 1/2.
\]

This maps pixel centres rather than treating the first pixel as a corner.
Coordinates outside the image are handled by clamping sample indices to the
nearest border pixel (edge extension).

## Nearest neighbour

Choose the source pixel whose centre is closest to the mapped coordinate:

\[
 i = \lfloor x_j + 1/2 \rfloor.
\]

The selected source value is copied without averaging. This preserves hard
pixel boundaries and is useful for pixel art and label masks, but can produce
jagged edges and aliasing.

## Bilinear interpolation

Let `i=floor(x)`, `t=x-i`, and use the two adjacent samples. The one-dimensional
weights are `1-t` and `t`; in two dimensions the four products of horizontal
and vertical weights are applied to the 2x2 neighbourhood. The triangle kernel
is

\[
 h(x)=\max(1-|x|,0).
\]

## Bicubic interpolation (Keys cubic convolution)

The implementation uses Keys' cubic convolution kernel with `a=-0.5`:

\[
 h(x)=\begin{cases}
 (a+2)|x|^3-(a+3)|x|^2+1,&0\le |x|<1,\\
 a|x|^3-5a|x|^2+8a|x|-4a,&1\le |x|<2,\\
 0,&|x|\ge2.
 \end{cases}
\]

A 4x4 neighbourhood contributes in two dimensions. Other applications and
libraries may choose a different cubic parameter, so “bicubic” is not one
unique numerical method.

## Lanczos-3

Lanczos-3 is a windowed sinc reconstruction filter:

\[
 h(x)=\begin{cases}
 \operatorname{sinc}(x)\operatorname{sinc}(x/3),&|x|<3,\\
 0,&|x|\ge3,
 \end{cases}
 \qquad \operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x},
\]

with `sinc(0)=1`. Its support radius is 3 input pixels on each axis, so up to
6 samples per axis contribute (more around fractional endpoints because of
inclusive integer support bounds).

## Separable filtering and downsampling

Bilinear, bicubic and Lanczos-3 are separable: the 2-D filter is the product of
one 1-D horizontal and one 1-D vertical filter. The code therefore performs
horizontal and vertical passes.

When shrinking, the implementation widens each reconstruction kernel by the
source-to-destination scale. This integrates over a wider source region and
acts as a low-pass filter before sampling, reducing aliasing. Nearest-neighbour
is intentionally left as point sampling. All weights are normalized per
output coordinate, including at image borders.

## Research caveat

The benchmark creates one low-resolution image using a stated Pillow filter,
then reconstructs it to the original dimensions. The original is the reference
for MSE, PSNR and SSIM. This is a controlled experiment, not a universal ranking:
results depend on image content, scale factor, degradation filter, colour space,
edge policy, and implementation details. Run more than one source image and
more than one scale factor for a defensible thesis conclusion.
