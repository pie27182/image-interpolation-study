# Research and implementation sources

The project combines original mathematical papers, teaching material, and software documentation. Use original publications for claims about the mathematics and library documentation for claims about a particular implementation.

## Mathematical foundations and original papers

1. MIT, *Foundations of Computer Vision*, [Image Sampling and Aliasing](https://visionbook.mit.edu/sampling_and_aliasing.html) and [Downsampling and Upsampling Images](https://visionbook.mit.edu/upsamplig_downsampling_2.html). Sampling, Nyquist, sinc reconstruction, aliasing, anti-alias filters, decimation, expansion, and practical interpolation kernels.
2. R. G. Keys, [“Cubic Convolution Interpolation for Digital Image Processing”](https://doi.org/10.1109/TASSP.1981.1163711), *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 29(6), 1153–1160, 1981. Original derivation of cubic convolution interpolation.
3. D. P. Mitchell and A. N. Netravali, [“Reconstruction Filters in Computer Graphics”](https://doi.org/10.1145/378456.378514), *Computer Graphics*, 22(4), 221–228, 1988. Cubic reconstruction filter family and visual/spectral trade-offs.
4. M. Unser, [“Splines: A Perfect Fit for Signal and Image Processing”](https://doi.org/10.1109/79.799930), *IEEE Signal Processing Magazine*, 16(6), 22–38, 1999. B-spline bases, coefficient prefiltering, approximation order, and the relationship between splines and signal processing. [EPFL publication page](https://bigwww.epfl.ch/publications/unser9902.html).
5. P. Thévenaz, T. Blu, and M. Unser, [“Interpolation Revisited”](https://doi.org/10.1109/42.875199), *IEEE Transactions on Medical Imaging*, 19(7), 739–758, 2000. A unified approximation-theoretic treatment of interpolating and generalized interpolation.
6. P. S. Heckbert, [*Fundamentals of Texture Mapping and Image Warping*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html), UC Berkeley Technical Report UCB/CSD-89-516, 1989. Arbitrary geometric mappings and space-varying anti-alias filters.
7. X. Li and M. T. Orchard, [“New Edge-Directed Interpolation”](https://doi.org/10.1109/83.951537), *IEEE Transactions on Image Processing*, 10(10), 1521–1527, 2001. Covariance-based, orientation-adaptive interpolation.
8. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, [“Image Quality Assessment: From Error Visibility to Structural Similarity”](https://doi.org/10.1109/TIP.2003.819861), *IEEE Transactions on Image Processing*, 13(4), 600–612, 2004. Original SSIM paper.
9. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018. General textbook reference for sampling, image transforms, and interpolation.

## Software and standards documentation

10. Pillow, [Image module](https://pillow.readthedocs.io/en/stable/reference/Image.html) and [Concepts: Filters](https://pillow.readthedocs.io/en/stable/handbook/concepts.html). The library baseline for nearest, bilinear, bicubic, and Lanczos filters.
11. OpenCV, [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html). Reverse mapping and common resize/warp interpolation modes.
12. SciPy, [`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html). Spline interpolation order, prefilter option, boundary modes, and coordinate-grid convention.
13. scikit-image, [`skimage.metrics`](https://scikit-image.org/docs/stable/api/skimage.metrics.html). MSE, PSNR, SSIM and other image comparison metrics.
14. International Color Consortium, [sRGB registry entry](https://registry.color.org/rgb-registry/srgb). sRGB component transfer function and colour encoding characteristics.
15. ImageMagick, [Resampling Filters](https://usage.imagemagick.org/filter/). Practical filter support, windowed sinc, aliasing, ringing, blur, and resampling trade-offs.
16. Cambridge in Colour, [Understanding Digital Image Interpolation](https://www.cambridgeincolour.com/tutorials/image-interpolation.htm). Accessible overview of common interpolation families and visual effects.

## Reproducibility notes

- Pillow is the library baseline; its filter definitions and coordinate conventions are not assumed to match the hand-coded versions.
- NumPy is used for arrays, indexing and arithmetic. No high-level resize API is called by the from-scratch implementation.
- scikit-image is used only for evaluation metrics, not interpolation.
- Record package versions, image dimensions, scale factor, degradation filter, colour-space handling, coordinate convention, edge mode, and machine details when collecting thesis results.
- The label “bicubic” is not a complete mathematical specification. Record the exact kernel and parameters (for example, Keys with `a=-0.5`).
- Distinguish cubic B-spline interpolation with its coefficient prefilter from Keys cubic convolution; they are different methods.
