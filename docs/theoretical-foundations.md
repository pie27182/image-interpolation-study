# Digital Image Interpolation and Resampling: Theoretical Foundations

This chapter is a research-oriented introduction to the mathematics behind classical digital image interpolation. It is intended to support a thesis comparing reconstruction methods and their implementations. It is not a claim that one filter is universally best: the appropriate method depends on the sampling model, scale change, image content, quality criterion, and implementation conventions.

## 1. What problem does interpolation solve?

A physical scene is not a matrix. A useful idealization is a continuous image or spatial signal

$$
f: \mathbb R^2\rightarrow\mathbb R^C, 
$$

where $(x,y)$ denotes spatial position and $C$ is the number of channels (one for grayscale, three for RGB, and so on). A digital image is a finite collection of values on a discrete lattice:

$$
 I[i,j,c],\qquad i=0,\ldots,W-1,\quad j=0,\ldots,H-1.
$$

The digital array alone does not tell us what happens between its samples. To rotate, scale, warp, or otherwise transform the image, an algorithm needs a model of the underlying continuous image. **Interpolation** constructs or evaluates such a model at non-integer coordinates.

Three operations should be distinguished:

- **Sampling** converts a continuous signal into discrete measurements.
- **Interpolation/reconstruction** estimates a continuous signal from discrete measurements, or evaluates an estimate at locations where no sample was recorded.
- **Resampling** evaluates an image on a new grid. Upscaling is usually dominated by interpolation; downscaling also requires low-pass filtering to prevent aliasing.

Interpolation does not recover scene detail that was never recorded. It makes a mathematically or visually motivated estimate from the available samples. Super-resolution is a different, more ambitious problem: it uses additional assumptions, learned priors, multiple frames, or other information to infer plausible high-frequency detail.

There are also two useful pixel models. In a point-sample model, $I[i,j]$ approximates the continuous image at the sample centre. In an area-sample model, a pixel represents an integral or average over a finite footprint. A real camera sensor has a finite photosensitive area and an optical point-spread function, so its measurement process is more elaborate than ideal point sampling. A resizing algorithm must choose which model it is approximating; area/box reduction is especially natural when input values are interpreted as pixel-area averages.

## 2. Sampling theory and the Nyquist–Shannon theorem

### 2.1 Sampling a continuous signal

For a one-dimensional signal $f(x)$ sampled at spacing $T$, ideal point sampling produces

$$
 f_s[n]=f(nT).
$$

An impulse-train representation is

$$
 f_{\delta}(x)=f(x)\sum_{n\in\mathbb Z}\delta(x-nT).
$$

Multiplication by a periodic impulse train in the spatial domain creates periodic replicas of the signal spectrum in the Fourier domain. If those replicas overlap, different continuous frequencies become indistinguishable after sampling. This is **aliasing**. In images it can appear as jagged diagonal edges, false patterns, moiré, or texture that changes its apparent orientation or frequency.

### 2.2 Nyquist condition

Let the continuous signal be band-limited, with no spectral energy above angular frequency $\omega_{\max}$. Perfect reconstruction is possible under the ideal assumptions when the sampling angular frequency satisfies

$$
 \omega_s=\frac{2\pi}{T}>2\omega_{\max}.
$$

Equivalently, the sampling period must be shorter than half the period of the highest frequency component. The corresponding Nyquist frequency is one half cycle per sample in normalized spatial units.

The theorem is powerful but its assumptions are important: exact band limitation, an infinite signal or ideal boundary treatment, exact samples, and exact arithmetic. A finite photograph is not strictly band-limited. Sharp edges and fine detail have broad spectra, and a finite image has boundaries. Thus practical image resizing approximates the ideal theory rather than satisfying it exactly.

### 2.3 Ideal sinc reconstruction

Under the band-limited assumptions, the one-dimensional Whittaker–Shannon reconstruction formula is

$$
 f_r(x)=\sum_{n=-\infty}^{\infty} f_s[n]\,
 \mathrm{sinc}\!\left(\frac{x}{T}-n\right),
$$

where the normalized sinc is

$$
 \mathrm{sinc}(u)=
 \begin{cases}
 \dfrac{\sin(\pi u)}{\pi u},&u\ne0,
 1,&u=0.
 \end{cases}
$$

For unit-spaced, two-dimensional samples, the separable ideal reconstruction is

$$
 \widehat I(x,y)=\sum_{i\in\mathbb Z}\sum_{j\in\mathbb Z}
 I[i,j]\,\mathrm{sinc}(x-i)\mathrm{sinc}(y-j).
$$

The sinc is an ideal interpolating kernel: it equals one at zero and zero at every other integer. It preserves every frequency in the representable baseband and rejects frequencies outside it. Its limitations are practical: it has infinite support, decays slowly, requires infinitely many samples in theory, and its oscillatory side lobes can cause ringing when a finite approximation is used. MIT's sampling chapter derives this ideal reconstruction and explains why local kernels are used in practice.

### 2.4 Frequency-domain view of resampling

A reconstruction kernel has a frequency response $H(\omega)$, the Fourier transform of its spatial impulse response. Its shape predicts much of its visual behaviour:

- A response that rolls off too early attenuates detail and produces blur.
- Weak stop-band rejection allows high frequencies to fold into the output band and produce aliasing.
- Sharp transitions and truncated oscillatory kernels can produce side lobes and ringing.
- Passband ripple can produce variations in contrast or fine repeating patterns.

No finite-support kernel perfectly achieves the ideal brick-wall low-pass response. Kernel selection is therefore a trade-off among spatial support, passband fidelity, stop-band attenuation, smoothness, and computational cost.

## 3. A general mathematical model of interpolation

### 3.1 Basis-function reconstruction

A broad family of linear reconstruction methods can be expressed as

$$
 f_r(x)=\sum_{k\in\mathbb Z}c[k]\,\varphi(x-k),
$$

where $\varphi$ is a basis or reconstruction function and $c[k]$ are expansion coefficients. For an interpolating basis,

$$
 \varphi(n)=\delta[n]
 \quad\Longrightarrow\quad c[k]=f_s[k].
$$

The reconstructed function then passes through the input samples. This is the familiar cardinal-interpolation case.

For a non-interpolating basis, the coefficients are not generally the sample values. At the integer sample locations,

$$
 f_s[n]=\sum_k c[k]\,\varphi(n-k).
$$

This is a linear system. In matrix notation, $\mathbf f=\mathbf P\mathbf c$, so the coefficients must be computed as $\mathbf c=\mathbf P^{-1}\mathbf f$, subject to suitable boundary conditions. That prefiltering step is central to correct B-spline interpolation.

For a separable two-dimensional basis,

$$
 I_r(x,y)=\sum_i\sum_j c[i,j]\,
 \varphi_x(x-i)\varphi_y(y-j).
$$

If the same one-dimensional kernel is used on both axes, then $\varphi_x=\varphi_y=\varphi$. The separable form allows a two-dimensional operation to be performed as one horizontal pass followed by one vertical pass.

### 3.2 Kernel conditions and useful properties

Common design conditions include:

1. **Interpolation:** $\varphi(0)=1$ and $\varphi(n)=0$ for every nonzero integer $n$.
2. **Symmetry:** $\varphi(x)=\varphi(-x)$, avoiding directional phase bias for a centred kernel.
3. **Partition of unity:** $\sum_k\varphi(x-k)=1$, preserving constant signals.
4. **Linear reproduction:** $\sum_k k\varphi(x-k)=x$, reproducing a linear ramp exactly.
5. **Compact support:** the kernel is zero outside a finite interval, limiting the number of contributing samples.
6. **Smoothness:** continuity of the kernel and its derivatives affects the smoothness of the reconstructed image.

More generally, polynomial reproduction conditions can be written as

$$
 \sum_{k\in\mathbb Z} k^r\varphi(x-k)=x^r,
 \qquad r=0,\ldots,L-1.
$$

These moment/reproduction conditions are closely connected to approximation order: how quickly an approximation error decreases as the sampling grid is refined. The exact relationship depends on the function space and assumptions, but Taylor expansion and Fourier-domain analysis are standard ways to derive error orders.

A kernel need not have only positive weights. Negative lobes can retain high-frequency contrast and make an image look sharper, but can overshoot near a sharp transition. Positive weights are more likely to stay within the local input range, but often produce more blur. This is one reason sharpness and ringing must be evaluated together.

### 3.3 A standard resampling equation

For an output pixel at destination coordinate $(p,q)$, first map it back to the corresponding source coordinate $(x_s,y_s)$. A separable resampler can then be written as

$$
 I_{out}[p,q]=\sum_i\sum_j I_{in}[i,j]
 K_x(x_s-i)K_y(y_s-j),
$$

with appropriate normalization and boundary extension. This equation is the core abstraction behind many resize routines. The algorithm is determined not only by the word used for the kernel, but also by the coordinate map, kernel parameters, filter dilation, edge policy, precision, and rounding.

## 4. Classical interpolation methods

### 4.1 Nearest-neighbour interpolation

Choose the source sample whose centre is closest to the query coordinate:

$$
 \widehat f(x)=f_s[\mathrm{round}(x)].
$$

A centred one-dimensional box basis can be described (up to the tie convention at exactly half-integer positions) by

$$
 \varphi_{NN}(x)=
 \begin{cases}
 1,&|x|<\tfrac12,\\
 0,&\text{otherwise}.
 \end{cases}
$$

The reconstructed function is piecewise constant. It is exceptionally cheap and does not invent intermediate class values, which is why nearest-neighbour is appropriate for segmentation masks, label maps, categorical rasters, and pixel art. It gives blocky enlargements and jagged transformed edges for photographs, and point-sampling during reduction can alias badly.

### 4.2 Linear and bilinear interpolation

For a one-dimensional query $x$, let $i=\lfloor x\rfloor$ and $t=x-i$. Linear interpolation is

$$
 \widehat f(x)=(1-t)f_s[i]+t f_s[i+1].
$$

Its triangular reconstruction kernel is

$$
 \varphi_{lin}(x)=\max(1-|x|,0).
$$

For a two-dimensional point, set $i=\lfloor x\rfloor$, $j=\lfloor y\rfloor$, $t_x=x-i$, and $t_y=y-j$. The bilinear value is

$$
\begin{aligned}
\widehat I(x,y)={}&(1-t_x)(1-t_y)I[i,j]\\
&+t_x(1-t_y)I[i+1,j]\\
&+(1-t_x)t_yI[i,j+1]\\
&+t_xt_yI[i+1,j+1].
\end{aligned}
$$

This is a weighted average of the four corners of the containing cell. It is a linear operator with respect to the pixel values, although its polynomial expression contains the coordinate product $xy$; “bilinear” does not mean a single plane in the two coordinates. The reconstructed surface is continuous, but its first derivatives can jump at grid lines. Bilinear is inexpensive and stable, but its triangular kernel attenuates high spatial frequencies, so enlarged images can look soft.

### 4.3 Bicubic interpolation is a family, not a single formula

“Bicubic” usually means a separable two-dimensional interpolation using a cubic kernel along each axis. A compactly supported cubic kernel with support radius two uses up to four samples per axis, or a 4×4 neighbourhood in two dimensions. But many different cubic kernels satisfy reasonable interpolation and smoothness constraints. The label *bicubic* does not uniquely identify a coefficient set or frequency response.

#### Keys cubic convolution

Keys derived a piecewise cubic convolution kernel. In normalized coordinates $r=|x|$, its one-dimensional form is

$$
 K_a(x)=
 \begin{cases}
 (a+2)r^3-(a+3)r^2+1,&0\le r<1,\\
 ar^3-5ar^2+8ar-4a,&1\le r<2,\\
 0,&r\ge2.
 \end{cases}
$$

Keys obtains this family by imposing symmetry, compact support, interpolation at integer nodes, continuity constraints, and accuracy conditions. A Taylor-series argument selects $a=-\tfrac12$ for third-order convergence in the paper's construction; this parameter is often associated with Catmull–Rom cubic convolution. Other values are used by real libraries, producing visibly and numerically different results.

The two-dimensional separable form is

$$
 K_a^{(2D)}(x,y)=K_a(x)K_a(y).
$$

It can preserve more detail than bilinear interpolation, but negative portions of the kernel can create overshoot or undershoot near high-contrast edges (ringing/halos).

#### Mitchell–Netravali cubic filters

Mitchell and Netravali studied a two-parameter family of piecewise cubic reconstruction filters, commonly parameterized by $B$ and $C$. One common expression is

$$
K_{B,C}(x)=\frac{1}{6}
\begin{cases}
(12-9B-6C)r^3+(-18+12B+6C)r^2+(6-2B),&0\le r<1,\\
(-B-6C)r^3+(6B+30C)r^2\\
\quad+(-12B-48C)r+(8B+24C),&1\le r<2,\\
0,&r\ge2.
\end{cases}
$$

Different parameter choices trade blur, ringing, and alias rejection. The paper's well-known compromise is $(B,C)=(\tfrac13,\tfrac13)$; other choices include cubic B-spline-like smoothing and sharper Catmull–Rom-like interpolation. The paper emphasizes that filters should be judged not only by their fit at sample points but also by their frequency response and visible artifacts.

### 4.4 Spline interpolation and the B-spline distinction

A spline is a piecewise polynomial whose segments join with prescribed smoothness. Cubic splines are popular partly because their first and second derivatives can be made continuous, and because natural cubic splines have a variational characterization related to minimizing bending energy, commonly expressed in one dimension through an integral of the squared second derivative.

A centred B-spline of degree $n$ is the $(n+1)$-fold convolution of a unit-width box function:

$$
 \beta^n=\underbrace{\beta^0*\beta^0*\cdots*\beta^0}_{n+1\text{ factors}}.
$$

The cubic B-spline basis function is

$$
\beta^3(x)=
\begin{cases}
\frac23-|x|^2+\frac{|x|^3}{2},&0\le|x|<1,\\[3pt]
\frac{(2-|x|)^3}{6},&1\le|x|<2,\\[3pt]
0,&|x|\ge2.
\end{cases}
$$

A crucial distinction: the cubic B-spline basis is smooth and compactly supported, but it is **not interpolating**. If raw pixel samples are used directly as B-spline coefficients, the reconstructed spline generally does not pass exactly through the pixels; it is a smoothed approximation. To obtain an exact interpolating spline, one must first calculate coefficients by inverting the discrete sampling relation. In practice this is implemented by a digital prefilter, often using efficient recursive causal and anti-causal filters rather than explicitly inverting a large matrix.

This explains why “cubic B-spline interpolation” is not the same method as Keys bicubic convolution. The former uses a non-interpolating basis plus coefficient prefilter; the latter uses an interpolating cubic convolution kernel whose coefficients can be the samples themselves. SciPy's `ndimage.zoom` uses spline interpolation of selectable order and exposes a `prefilter` option; its default order is three. It should not be treated as an implementation of Keys' `a=-0.5` bicubic kernel.

### 4.5 Lanczos interpolation

The Lanczos kernel is a windowed sinc. With radius $a>0$,

$$
 K_{L,a}(x)=
 \begin{cases}
 \mathrm{sinc}(x)\mathrm{sinc}(x/a),&|x|<a,\\
 0,&|x|\ge a.
 \end{cases}
$$

Lanczos-3 uses $a=3$; in one dimension the kernel has finite radius three, and a separable two-dimensional version applies it along both axes. The window truncates the infinite sinc while retaining much of its frequency-selective behaviour. A wider support follows the ideal sinc more closely but requires more work. Oscillatory side lobes can cause ringing around high-contrast edges, especially when a sharp dark-to-light transition is present.

Pillow describes Lanczos as a truncated sinc. ImageMagick's resampling guide explains the role of the sinc window and the trade-off between support size, frequency response, and ringing. Lanczos is often a strong general-purpose choice, not a universal optimum.

### 4.6 Area/box resampling

When shrinking an image, a destination pixel covers a region of the source image. An area method estimates the average value over that source footprint:

$$
 I_{out}[p,q]\approx\frac{1}{|A_{pq}|}
 \int_{A_{pq}} I(x,y)\,dx\,dy.
$$

Under a piecewise-constant source-pixel model, the integral can be evaluated by summing each source pixel value multiplied by the area of overlap between its square and the destination footprint. In a separable axis-aligned resize, overlap weights can be computed independently in x and y. This makes area resampling especially natural for reduction. It is not simply another name for bilinear or bicubic interpolation; it integrates a footprint rather than evaluating a local point estimate.

## 5. Downsampling and anti-aliasing

### 5.1 Why reduction is a different problem

Suppose an image is reduced by a linear scale factor $s<1$. The output grid has a lower Nyquist limit. Frequencies that were representable in the source may lie above the output Nyquist limit. If they are sampled without filtering, they fold into lower frequencies and become aliasing. Therefore the correct conceptual sequence is

$$
 \text{low-pass filter}\;\longrightarrow\;\text{sample on the coarser grid}.
$$

A useful mathematical model is

$$
 I_{out}[m,n]=(I_{in}*h_s)(m/s,n/s),
$$

where $h_s$ removes frequencies that cannot be represented on the output grid. In one dimension, scaling a reconstruction kernel to the reduced sampling rate gives a broadened filter of the form

$$
 K_s(x)=sK(sx),\qquad 0<s<1,
$$

with corresponding normalization. Its support in source-pixel coordinates grows by $1/s$. For example, reducing by a factor of four should combine information from a substantially wider source neighbourhood than enlarging by four. In two dimensions with possibly different axis scales, use the corresponding axis-specific widened kernels.

A common implementation error is to use the same narrow 4×4 or 6×6 footprint for a severe reduction as for enlargement. That does not adequately suppress high-frequency content and produces aliasing. MIT's downsampling chapter explicitly derives low-pass filtering followed by decimation and describes windowed-sinc and binomial approximations.

### 5.2 Windowed low-pass filters

The ideal low-pass filter is a rectangular response in the Fourier domain and a sinc-shaped impulse response in the spatial domain. Since that impulse response is long/infinite, practical filters window and truncate it. Hamming, Hann, Blackman, Kaiser, and Lanczos-like windows change the compromise between transition width, stop-band attenuation, ripple, and ringing.

A finite binomial filter such as

$$
 [1,2,1]/4
$$

is inexpensive and gives useful smoothing, but it is not an ideal brick-wall filter. A stronger binomial kernel such as $[1,4,6,4,1]/16$ attenuates more high frequency detail and may over-smooth. The right anti-alias strength depends on the reduction factor and the content.

### 5.3 Upsampling is not the inverse of downsampling

Downsampling discards information. Upsampling increases the number of sample locations but cannot uniquely reconstruct the discarded high-frequency content. In multirate signal processing, integer upsampling can be represented as **zero insertion** (expansion) followed by an interpolation low-pass filter. Downsampling can be represented as an anti-aliasing low-pass filter followed by **decimation** (keeping every kth sample). These are different operators; one is not a lossless inverse of the other.

## 6. Geometric transformations and spatially varying filters

Resizing is only a special case of image resampling. Rotation, perspective projection, affine transforms, lens correction, texture mapping, and image warping map each output coordinate to a location in the source image.

A robust implementation generally uses **inverse mapping**: loop over destination pixels, map each destination coordinate back into the source, then interpolate there. Forward mapping can leave holes because source pixels may not land exactly on destination pixel centres. OpenCV's geometric-transform documentation describes this destination-to-source mapping approach.

For a general transform $\mathbf x_s=\Phi(\mathbf x_d)$, the local Jacobian

$$
 J=\frac{\partial\Phi}{\partial\mathbf x_d}
$$

describes how a small destination pixel footprint maps into the source. Under non-uniform scale or shear, that footprint is generally an ellipse, not a circle or axis-aligned square. A single fixed isotropic kernel can therefore be inadequate. Elliptical weighted average (EWA) resampling uses a spatially varying elliptical footprint; Heckbert's Berkeley technical report is a foundational treatment of filtering for arbitrary texture mappings and image warps. MIP mapping and image pyramids are practical multiscale approaches for selecting appropriately prefiltered source levels when a texture is minified.

## 7. Practical implementation choices that change the answer

Two functions both called “bicubic” or “Lanczos” may return different pixels. A fair comparison must record the whole resampling definition.

### 7.1 Coordinate conventions and pixel centres

Let the source length be $S$ and destination length be $D$. One common half-pixel mapping is

$$
 x_s(j)=\left(j+\frac12\right)\frac{S}{D}-\frac12.
$$

Another convention aligns the first and last pixel centres:

$$
 x_s(j)=j\frac{S-1}{D-1},
$$

when the dimensions make this expression meaningful. These conventions differ, particularly for small dimensions and non-integer scale factors. The resampling result can shift or stretch by fractions of a pixel. SciPy exposes a related distinction through `grid_mode`: it can scale centre-to-centre distances or the full pixel extent. Record the convention explicitly.

### 7.2 Boundary extension

A kernel near an image border requests samples outside the finite array. The algorithm must decide how the image is extended. Common policies are:

- **Clamp/nearest:** repeat the first or last sample.
- **Half-sample symmetric reflection:** repeat the edge sample at the reflection boundary.
- **Whole-sample symmetric reflection:** reflect about the centre of the edge sample without repeating it.
- **Wrap:** use periodic extension.
- **Constant:** extend with a specified fill value.

The words `reflect` and `mirror` are not used consistently across libraries. SciPy calls half-sample symmetry `reflect` and whole-sample symmetry `mirror`; OpenCV's `BORDER_REFLECT_101` follows whole-sample symmetry. This project's public option `boundary="reflect"` means whole-sample symmetric reflection, where index `-1` maps to source index `1`. It therefore corresponds to SciPy `mirror` and OpenCV `BORDER_REFLECT_101`, not SciPy `reflect`. Always compare the explicit index mapping, not just the mode name.

### 7.3 Kernel support, dilation, and normalization

A kernel's support determines which source pixels contribute. For a downscale, that support should grow in source coordinates. Weights are commonly normalized so that

$$
 \sum_i w_i=1,
$$

which preserves a constant image (partition of unity). At borders, some implementations truncate the kernel and renormalize; others include padded samples. This choice changes edge pixels.

### 7.4 Numeric precision, clipping, and quantization

Computing every weighted sum in 8-bit integer arithmetic is unsafe because intermediate rounding can accumulate error and negative-lobe filters may temporarily produce values below zero or above the nominal maximum. A sound implementation normally computes in floating point, applies the full filter, then clips and rounds once when converting back to an integer image. Record dtype, intermediate precision, rounding rule, and clipping policy.

### 7.5 Colour space and gamma

sRGB values are gamma-encoded, not proportional to physical light. The ICC registry gives the piecewise sRGB transfer function. For physically meaningful averaging, filtering should ideally be done after decoding sRGB to linear-light RGB, followed by re-encoding to sRGB. Filtering encoded values directly can produce different brightness around edges and gradients. To isolate interpolation mathematics, a thesis should either use linear-light data or clearly declare that it compares the libraries' standard encoded-channel behaviour.

For images with transparency, filtering non-premultiplied RGB can create coloured fringes around transparent pixels. A robust pipeline generally premultiplies colour by alpha, filters premultiplied channels and alpha consistently, then un-premultiplies where alpha is nonzero.

### 7.6 Library names are not complete mathematical specifications

The source code makes the ambiguity concrete. Pillow's current `Resample.c` defines its bicubic cubic-convolution parameter as `a = -0.5` and its Lanczos filter with support radius 3. OpenCV's current 4.x `resize.cpp` defines its cubic parameter `A = -0.75`; its `INTER_LANCZOS4` implementation builds eight coefficients per axis, corresponding to an 8×8 source neighbourhood in two dimensions. OpenCV's API documentation likewise describes `INTER_CUBIC` as a 4×4 neighbourhood and `INTER_LANCZOS4` as an 8×8 neighbourhood.

These are related filter families, but they are not numerically identical. The cubic parameter changes the shape and frequency response of the kernel. The Lanczos radius changes the window/support and the number of contributing samples. Coordinate mapping, boundary policy, coefficient quantization, and output rounding can introduce further differences. Therefore a result from `PIL.Image.Resampling.BICUBIC` should not be expected to equal `cv2.INTER_CUBIC`, even when the input and output dimensions match.

For a controlled mathematical comparison, implement one explicitly stated formula in the independent code and compare it against a reference implementation only after verifying that the reference uses the same parameter. For a practical library comparison, do not force different presets into a false claim of algorithmic equivalence: compare the end-to-end results and document the exact library version.

## 8. Adaptive and edge-directed interpolation

Classical fixed kernels use the same spatially invariant weights everywhere. Natural images, however, contain edges and textures whose local structure is directional. A fixed isotropic kernel may average across an edge and blur it.

Edge-directed methods estimate local orientation or local statistics and adapt the weights. One influential example is **New Edge-Directed Interpolation (NEDI)** by Li and Orchard (2001). It estimates local covariance from the low-resolution image and uses a geometric relationship between low- and high-resolution covariance to adapt interpolation coefficients. The intuition is that image intensity tends to vary more slowly along an edge than across it. The method can better follow oblique edges than an isotropic linear interpolator, at greater computational and modelling complexity.

A generic local linear predictor can be written as

$$
 \widehat y=\mathbf w^T\mathbf x,
$$

where $\mathbf x$ contains known neighbourhood values. In a minimum-mean-square-error/Wiener setting, coefficients are related to covariance matrices, for example

$$
 \mathbf w=\mathbf R_{xx}^{-1}\mathbf r_{xy},
$$

when the inverse exists (or a regularized/pseudoinverse version is used). This connects adaptive interpolation to statistical estimation and linear prediction. NEDI's specific covariance estimation and geometric-duality construction should be read in the original paper rather than reduced to this generic formula.

Other adaptive families include directional filters, edge-guided polynomial fitting, gradient-based interpolation, iterative reconstruction, and demosaicing-specific algorithms. Demosaicing is related but distinct: the missing values are colour channels sampled on a colour-filter-array pattern, and cross-channel correlations are used in addition to spatial information.

## 9. Mathematical disciplines connected to image interpolation

| Field | Connection to image interpolation |
|---|---|
| **Signal processing** | Sampling, reconstruction, convolution, multirate systems, low-pass filtering, polyphase implementations, and linear shift-invariant systems. |
| **Fourier/harmonic analysis** | Frequency responses, Nyquist limits, spectral replicas, aliasing, sinc reconstruction, stop-band attenuation, and ringing. |
| **Approximation theory** | Basis functions, cardinal interpolation, polynomial reproduction, approximation order, convergence, and approximation error. |
| **Spline theory** | Piecewise polynomials, B-spline bases, smoothness constraints, cardinal splines, and spline prefilters. |
| **Numerical analysis** | Taylor expansions, truncation error, stability, conditioning, floating-point precision, and boundary-value problems. |
| **Linear algebra** | Solving for spline coefficients; covariance estimation; matrix formulations of filtering, expansion, and decimation. |
| **Optimization and variational methods** | Minimum-curvature splines, least-squares reconstruction, regularization, and energy minimization. |
| **Probability and statistics** | MMSE estimation, covariance-based adaptive kernels, noise models, and uncertainty in missing samples. |
| **Computer graphics and geometry** | Inverse coordinate maps, affine/projective transforms, texture mapping, local Jacobians, EWA filtering, and MIP maps. |
| **Colour science** | Linear-light colour, transfer functions, RGB encoding, premultiplied alpha, and perceptual colour differences. |
| **Human vision and psychophysics** | Perceived sharpness, contrast sensitivity, edge visibility, ringing tolerance, and perceptual image-quality metrics. |
| **Machine learning** | Learned super-resolution and image priors. These methods can hallucinate plausible detail and should be distinguished from classical interpolation. |

## 10. How to evaluate interpolation methods scientifically

### 10.1 Controlled degradation and ground truth

A defensible upscaling experiment begins with a high-resolution reference. Generate a lower-resolution input through a documented degradation operator (ideally a low-pass blur followed by sampling), then reconstruct it to the reference dimensions. Compare the reconstruction with the original high-resolution image. The degradation kernel matters: if every low-resolution test image is created with Lanczos, the Lanczos reconstruction has a built-in model-match advantage. Repeat with more than one degradation model and with a variety of image classes.

Keep the following variables fixed within each comparison: source image, source and destination dimensions, scale factor, coordinate convention, colour handling, output dtype, boundary mode, and degradation process. If the purpose is to compare implementations of the same named mathematical filter, also fix its exact kernel formula and parameter values.

Use two separate comparisons:

1. **Implementation fidelity:** compare a hand-coded method with a reference implementation that uses the same mathematical kernel, coordinate map, boundary mode, and precision. This tests correctness and implementation differences.
2. **Library preset comparison:** compare Pillow, OpenCV, SciPy, or other out-of-the-box presets. This compares complete software pipelines, not necessarily identical mathematical filters.

### 10.2 Objective metrics

Let the reference image be $I$, reconstruction be $\widehat I$, and there be $N$ scalar samples across all compared channels.

**Mean squared error (MSE)**

$$
 \mathrm{MSE}=\frac1N\sum_{n=1}^N(I_n-\widehat I_n)^2.
$$

Lower is better. MSE is mathematically simple but treats all sample errors independently and equally.

**Peak signal-to-noise ratio (PSNR)**

$$
 \mathrm{PSNR}=10\log_{10}\left(\frac{L^2}{\mathrm{MSE}}\right)\;\mathrm{dB},
$$

where $L$ is the maximum possible sample value (255 for ordinary 8-bit samples, or 1 for normalized floating-point data). Higher is better; an exact match yields infinite PSNR. PSNR is a logarithmic restatement of MSE, not an independent perceptual model.

**Structural Similarity (SSIM)**

SSIM compares local luminance, contrast, and structure. A commonly used combined form is

$$
 \mathrm{SSIM}(x,y)=
 \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
 {(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},
$$

where $\mu_x,\mu_y$ are local means, $\sigma_x^2,\sigma_y^2$ local variances, and $\sigma_{xy}$ local covariance. Constants stabilize the expression near zero denominators. The original SSIM paper develops this structural approach as a complement to error-visibility models. Values closer to one indicate greater structural similarity under the chosen implementation and settings.

**LPIPS and other learned perceptual metrics** may be added as exploratory metrics, but they depend on a pretrained network and model version. They should not replace transparent pixel/structural metrics or human visual inspection in a classical interpolation thesis.

No one metric completely describes image quality. Report metrics alongside enlarged crops and discuss blur, edge sharpness, ringing/halos, jaggedness, moiré, and texture preservation.

### 10.3 Runtime and statistical reporting

Measure the resampling computation separately from image loading, saving, display, and disk I/O. Use the same hardware and software environment, warm up each method, repeat measurements, and report a robust statistic such as median and interquartile range. Record image size, source/output dimensions, scaling factor, CPU, OS, Python, NumPy, Pillow/OpenCV/SciPy versions, and threading settings. Avoid treating one runtime measurement as conclusive.

For a separable kernel with $K$ taps per axis, a naive two-dimensional convolution needs roughly $K^2$ products per output pixel; two one-dimensional passes need roughly $2K$. This makes separability an important efficiency property. Actual runtime will also depend on vectorization, memory access, cache behaviour, language-level loops, and native library implementations.

## 11. Implications for the thesis project

A strong comparative thesis could be structured around the following questions:

1. How do nearest, bilinear, Keys bicubic, B-spline, and Lanczos differ in spatial and frequency-domain behaviour?
2. How does performance change for enlargement versus reduction, especially when anti-aliasing is enabled?
3. How much of the output difference between libraries is caused by kernel choice versus coordinate mapping, edge policy, and numeric precision?
4. How do image class and degradation process change the ranking under MSE, PSNR, SSIM, and visual inspection?
5. What quality/runtime trade-offs are observed for Python/NumPy implementations compared with optimized native-library routines?

The current from-scratch implementation goes beyond the initial four-method
baseline. It includes nearest, bilinear, Keys bicubic, Mitchell--Netravali,
coefficient-prefiltered cubic B-spline, Lanczos-3, and exact area resampling.
It also makes coordinate mode and boundary extension configurable, and
implements MSE, PSNR and a documented uniform-window SSIM variant directly
with NumPy. The detailed equations and implementation choices are in
[`implementation-from-scratch.md`](implementation-from-scratch.md).

Further research extensions could include:

- **Edge-directed interpolation** such as NEDI, treated separately from fixed
  linear kernels.
- **EWA resampling** for arbitrary affine/projective warps with an elliptical
  source footprint.
- **Linear-light colour and premultiplied-alpha filtering** as explicit
  experimental modes.
- **Polyphase or recursive implementations** for performance comparisons.
- **Learned super-resolution** as a distinct comparison category, not as a
  classical interpolation kernel.
- Diagnostics for kernel plots, frequency response, impulse response,
  step-edge response, constant preservation, linear-ramp reproduction and
  boundary behaviour.

These extensions should be added as separately defined experiments so that a
fixed-kernel comparison does not conflate fundamentally different problem
classes.

## 12. Selected bibliography and online resources

The following sources range from mathematical foundations to original algorithm papers and current implementation documentation. Page and equation numbers should be checked in the cited original document when preparing formal thesis citations.

1. **MIT Foundations of Computer Vision**, “Image Sampling and Aliasing” and “Downsampling and Upsampling Images.” Clear derivations of sampling, Nyquist, sinc reconstruction, anti-alias filtering, decimation, expansion, and interpolation kernels. <https://visionbook.mit.edu/sampling_and_aliasing.html> · <https://visionbook.mit.edu/upsamplig_downsampling_2.html>
2. **R. G. Keys**, “Cubic Convolution Interpolation for Digital Image Processing,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 29(6), 1153–1160, 1981. Original derivation of the cubic-convolution kernel and its accuracy conditions. DOI: <https://doi.org/10.1109/TASSP.1981.1163711> · accessible paper: <https://lapi.fi-p.unam.mx/wp-content/uploads/Articulo_CubConvInterp.pdf>
3. **D. P. Mitchell and A. N. Netravali**, “Reconstruction Filters in Computer Graphics,” *Computer Graphics* (SIGGRAPH), 22(4), 221–228, 1988. Studies reconstruction artefacts and the two-parameter cubic filter family. DOI: <https://doi.org/10.1145/378456.378514>
4. **M. Unser**, “Splines: A Perfect Fit for Signal and Image Processing,” *IEEE Signal Processing Magazine*, 16(6), 22–38, 1999. Tutorial on B-spline representation, interpolation prefilters, approximation, and links to wavelets. DOI: <https://doi.org/10.1109/79.799930> · <https://bigwww.epfl.ch/publications/unser9902.html> · PDF: <https://bigwww.epfl.ch/publications/unser9902.pdf>
5. **P. Thévenaz, T. Blu, and M. Unser**, “Interpolation Revisited,” *IEEE Transactions on Medical Imaging*, 19(7), 739–758, 2000. Unified approximation-theoretic treatment of interpolation, basis functions, approximation order, and generalized interpolation. DOI: <https://doi.org/10.1109/42.875199> · <https://bigwww.epfl.ch/publications/thevenaz0002.html>
6. **P. S. Heckbert**, “Fundamentals of Texture Mapping and Image Warping,” UC Berkeley Technical Report UCB/CSD-89-516, 1989. Important background for arbitrary geometric mapping and space-varying anti-alias filters. <https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html>
7. **X. Li and M. T. Orchard**, “New Edge-Directed Interpolation,” *IEEE Transactions on Image Processing*, 10(10), 1521–1527, 2001. Covariance-based, orientation-adaptive interpolation for natural images. DOI: <https://doi.org/10.1109/83.951537> · paper: <https://www.ece.lsu.edu/gunturk/EE7700/NEDI.pdf>
8. **Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli**, “Image Quality Assessment: From Error Visibility to Structural Similarity,” *IEEE Transactions on Image Processing*, 13(4), 600–612, 2004. Original SSIM paper. DOI: <https://doi.org/10.1109/TIP.2003.819861> · <https://utw10503.utweb.utexas.edu/publications/2004/zwang_ssim_ieeeip2004.pdf>
9. **Pillow**, Image and Concepts documentation. Available filters include nearest, box, bilinear, bicubic, and Lanczos; the documentation describes their intended behaviour and supported resize operations. <https://pillow.readthedocs.io/en/stable/reference/Image.html> · <https://pillow.readthedocs.io/en/stable/handbook/concepts.html>
10. **SciPy**, `scipy.ndimage.zoom`. Documents spline interpolation order, the prefilter option, boundary extension modes, and the `grid_mode` coordinate convention. <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html>
11. **OpenCV**, “Geometric Image Transformations.” Documents inverse mapping and common resize interpolation flags, including nearest, linear, cubic, area, and Lanczos. <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html>
12. **scikit-image**, `skimage.metrics`. Reference documentation for MSE, PSNR, SSIM, and related metrics. <https://scikit-image.org/docs/stable/api/skimage.metrics.html>
13. **International Color Consortium**, sRGB colour encoding registry. Specifies the piecewise sRGB component transfer function. <https://registry.color.org/rgb-registry/srgb>
14. **ImageMagick**, “Resampling Filters.” Practical discussion of filter support, windowed sinc filters, ringing, blurring, aliasing, and filter parameter trade-offs. <https://usage.imagemagick.org/filter/>
15. **R. C. Gonzalez and R. E. Woods**, *Digital Image Processing*, 4th ed., Pearson, 2018. Broad textbook treatment of image fundamentals and interpolation for geometric transformations. Publisher information: <https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003190/9780133356724>.

## 13. Search and citation notes

This chapter synthesizes the linked papers and documentation for study. It is not a replacement for reading and citing the original publication. For a formal thesis, cite the original paper or standard where the claim originates, verify its exact publication metadata, and use a consistent citation style required by the university. Library documentation is appropriate when describing what a particular software implementation does; it is not a substitute for the mathematical source of an algorithm.

### Additional implementation-source references

- **Pillow**, [C resampling implementation](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c). Source for the bicubic parameter `a = -0.5` and Lanczos support radius 3.
- **OpenCV**, [4.x resize implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp). Source for the cubic parameter `A = -0.75` and the Lanczos4 coefficient construction.

## 14. Companion implementation chapter

The complete equations used by the current Python implementation, including
the Keys and Mitchell cubic kernels, cubic B-spline coefficient prefilter,
Lanczos window, area-overlap weights, separable tap calculation, boundary
models, and NumPy-only MSE/PSNR/SSIM calculations, are documented in
[`implementation-from-scratch.md`](implementation-from-scratch.md). The code is
an independent implementation based on the cited mathematics, not copied
source from an existing interpolation library.
