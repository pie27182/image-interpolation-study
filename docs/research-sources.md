# Research and implementation sources

The project uses original scientific publications for mathematical claims and
official software documentation/source for library-specific behaviour. The
scratch code is independently written from the equations and algorithmic
ideas below; source-code links are used to audit conventions, not as code to
copy into this project.

## 1. Sampling, reconstruction and signal processing

1. **MIT, Foundations of Computer Vision**, Chapter 20, [Image Sampling and
   Aliasing](https://visionbook.mit.edu/sampling_and_aliasing.html). Derives
   sampling, aliasing, Nyquist, ideal sinc reconstruction and practical local
   interpolation kernels.
2. **MIT, Foundations of Computer Vision**, Chapter 21, [Downsampling and
   Upsampling Images](https://visionbook.mit.edu/upsamplig_downsampling_2.html).
   Explains low-pass filtering, decimation, zero insertion, reconstruction and
   practical resizing. The book's [sampling and multiscale overview](https://visionbook.mit.edu/part_sampling_and_multiscale.html)
   situates resizing in image pyramids and multiscale vision.
3. **R. C. Gonzalez and R. E. Woods**, *Digital Image Processing*, 4th ed.,
   Pearson, 2018. General textbook reference for sampling, geometric
   transformations, spatial filtering and image resampling.
4. **A. V. Oppenheim and R. W. Schafer**, *Discrete-Time Signal Processing*,
   3rd ed., Pearson, 2010. General foundation for discrete-time systems,
   convolution, frequency response, sampling and multirate signal processing.

## 2. Classical interpolation filters and spline theory

5. **R. G. Keys**, [“Cubic Convolution Interpolation for Digital Image
   Processing”](https://doi.org/10.1109/TASSP.1981.1163711), *IEEE Transactions
   on Acoustics, Speech, and Signal Processing*, 29(6), 1153–1160, 1981.
   Original cubic-convolution interpolation derivation. An accessible scanned
   copy is available [here](https://www.ncorr.com/download/publications/keysbicubic.pdf).
6. **D. P. Mitchell and A. N. Netravali**, [“Reconstruction Filters in
   Computer Graphics”](https://doi.org/10.1145/378456.378514), *Computer
   Graphics* (SIGGRAPH), 22(4), 221–228, 1988. Two-parameter cubic filters and
   their blur, aliasing, sharpness and ringing trade-offs. [Scanned paper](https://citeseerx.ist.psu.edu/document?doi=6731642027d82aa37289331a77f06c0871689c39&repid=rep1&type=pdf).
7. **M. Unser**, [“Splines: A Perfect Fit for Signal and Image Processing”](https://doi.org/10.1109/79.799930),
   *IEEE Signal Processing Magazine*, 16(6), 22–38, 1999. B-spline bases,
   coefficient prefiltering, approximation order and signal processing.
   [EPFL publication page](https://bigwww.epfl.ch/publications/unser9902.html).
8. **P. Thévenaz, T. Blu, and M. Unser**, [“Interpolation Revisited”](https://doi.org/10.1109/42.875199),
   *IEEE Transactions on Medical Imaging*, 19(7), 739–758, 2000. Unified
   approximation-theoretic treatment of interpolation, generalized
   interpolation, basis functions, prefiltering and approximation order.
   [EPFL publication page](https://bigwww.epfl.ch/publications/thevenaz0002.html).
9. **P. S. Heckbert**, [*Fundamentals of Texture Mapping and Image Warping*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html),
   UC Berkeley Technical Report UCB/CSD-89-516, 1989. Geometric mappings and
   spatially varying anti-alias filters.
10. **X. Li and M. T. Orchard**, [“New Edge-Directed Interpolation”](https://doi.org/10.1109/83.951537),
    *IEEE Transactions on Image Processing*, 10(10), 1521–1527, 2001.
    Covariance-based adaptation to local image structure. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/18255495/).

## 3. Quality assessment

11. **Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli**, [“Image
    Quality Assessment: From Error Visibility to Structural Similarity”](https://doi.org/10.1109/TIP.2003.819861),
    *IEEE Transactions on Image Processing*, 13(4), 600–612, 2004. Original
    SSIM paper. [Authors' publication page](https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html).
12. **Wang et al.**, [SSIM research page](https://www.cns.nyu.edu/~lcv/research/ssim/).
    Background, equations and reference implementation information.

## 4. Official implementation documentation and source audits

13. **Pillow**, [Image resampling documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html)
    and [filter concepts](https://pillow.readthedocs.io/en/stable/handbook/concepts.html).
    Documentation for Pillow's nearest, box, bilinear, bicubic and Lanczos
    resampling options.
14. **Pillow**, [C resampling implementation](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c).
    Source audit for Pillow's Keys-style bicubic parameter and its radius-three
    Lanczos filter. This project does not copy the C implementation.
15. **OpenCV**, [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
    Documents `resize`, reverse mapping and the supported interpolation flags.
16. **OpenCV**, [4.x resize implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp).
    Source audit for OpenCV's cubic-convolution parameter and Lanczos4
    coefficient construction. This project does not copy the C++ code.
17. **SciPy**, [`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html).
    Documents spline interpolation order, prefiltering, boundary extension and
    grid coordinate conventions. It is not a runtime dependency of the scratch
    implementation.
18. **scikit-image**, [image metrics documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html).
    Reference for MSE, PSNR and SSIM options. This project's metric routines
    are written directly in NumPy and scikit-image is not a dependency.
19. **International Color Consortium**, [sRGB registry entry](https://registry.color.org/rgb-registry/srgb).
    sRGB transfer function and colour encoding.
20. **ImageMagick**, [Resampling Filters](https://usage.imagemagick.org/filter/).
    Practical explanations of support, windowed-sinc filters, ringing, blur,
    aliasing and resize-filter trade-offs.
21. **Cambridge in Colour**, [Understanding Digital Image Interpolation](https://www.cambridgeincolour.com/tutorials/image-interpolation.htm).
    Accessible practical overview of common interpolation methods and visual
    effects.

## 5. How to cite this project's implementation

The algorithm equations should be cited to the original publications above:
Keys for cubic convolution, Mitchell--Netravali for their cubic family, Unser
and Thévenaz/Blu/Unser for spline interpolation and prefiltering, and the
sampling-theory sources for sinc reconstruction and anti-aliasing. Cite Pillow
or OpenCV documentation only when describing those libraries' specific
behaviour.

The Python implementation in this repository is an independent implementation
written for the study. It uses NumPy as the numeric array substrate; it does
not incorporate copied implementation source from Pillow, OpenCV, SciPy or
scikit-image. This does not remove the need to cite the mathematical works on
which the methods are based.

## 6. Reproducibility notes

For each experiment, record:

- Python, NumPy, Pillow and operating-system versions.
- Source and destination dimensions and scale factors.
- Exact method and parameters (for example Keys `a=-0.5`, Lanczos radius 3,
  Mitchell `B=C=1/3`).
- Coordinate mapping, boundary extension and anti-alias setting.
- Degradation filter used to create low-resolution input.
- Channel, alpha and colour-space treatment.
- Number of warmups and repeated timings, CPU and machine load.
- Metric settings, especially data range and SSIM window/covariance convention.

The name “bicubic” alone is not a sufficient algorithm specification. Nor is
“Lanczos” without the support radius and window convention.
