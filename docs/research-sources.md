# Research and implementation sources

These references informed the algorithms and experimental design. The code in
`src/interpolation_study/resize.py` implements the interpolation operations
itself; it does not call Pillow or OpenCV resize routines.

1. Cambridge in Colour, [Understanding Digital Image Interpolation](https://www.cambridgeincolour.com/tutorials/image-interpolation.htm). Overview of nearest-neighbour, bilinear, bicubic, spline, sinc and Lanczos interpolation, including trade-offs between neighbourhood size, quality and processing cost.
2. Pillow documentation, [Image module — resampling filters](https://pillow.readthedocs.io/en/stable/reference/Image.html). Describes the available reference filters and Pillow's resize interface; used for the library-baseline stage.
3. OpenCV, [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html). Documents common interpolation modes and the role of neighbourhoods in image resizing.
4. scikit-image, [Image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html). Defines MSE, PSNR and structural similarity (SSIM) metrics used to evaluate reconstructed images.
5. R. G. Keys, “Cubic Convolution Interpolation for Digital Image Processing,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153–1160, 1981. DOI: [10.1109/TASSP.1981.1163711](https://doi.org/10.1109/TASSP.1981.1163711).

## Reproducibility notes

- Pillow is the baseline implementation; its filter definitions and internal
  coordinate conventions are not assumed to match the hand-coded versions.
- NumPy is used for arrays, indexing and arithmetic. No high-level resize API is
  called by the from-scratch implementation.
- scikit-image is used only for evaluation metrics, not interpolation.
- Record package versions, image dimensions, scale factor, degradation filter,
  and machine details when collecting thesis results.
