# Suggested experiment protocol for the thesis

This protocol is a starting point, not a substitute for the supervisor's
requirements or a formal statistical plan.

## 1. Image set

Use at least 15–30 images across several content groups if practical:

- natural scenes and landscapes;
- portraits;
- fine textures (foliage, fabric, hair, brick);
- text, diagrams and thin lines;
- pixel art or segmentation masks (nearest-neighbour is particularly relevant).

Keep the original images unchanged and record their dimensions, format, colour
mode and source/licence in a dataset manifest.

## 2. Scale factors

For enlargement test 2x, 3x and 4x. For reduction test 0.75x, 0.5x and 0.25x.
Use exact integer output dimensions in all runs and record them in the results.

## 3. Library baseline

Use Pillow's `Image.Resampling` filters as the first baseline. Record Python,
Pillow and NumPy versions. Do not assume Pillow's cubic coefficients or pixel
mapping are identical to those of the independent implementation.

## 4. Independent implementation

Use the NumPy implementation in this repository. It defines its own half-pixel
coordinate mapping, clamped edge extension, Keys cubic coefficient `a=-0.5`,
Lanczos support radius 3, separable filtering and scale-aware kernel widening
for reductions. These are fixed experimental parameters; change one at a time
when testing alternative conventions.

## 5. Quality evaluation

When a high-resolution original is available, use it as a reference, generate a
low-resolution input with an explicitly stated degradation filter, then
reconstruct to the original dimensions. Report:

- MSE: lower is better;
- PSNR: higher is better;
- SSIM: closer to 1 is more structurally similar.

Also include visual inspection of representative crops. Metrics alone do not
fully describe perceived sharpness, ringing, edge halos or aliasing.

The benchmark's default Lanczos degradation is useful as a fixed baseline but
favours methods similar to Lanczos. Repeat with all four degradation filters
or use an independently defined degradation process. Do not compare scores
across experiments without reporting the degradation process.

## 6. Runtime evaluation

For every image, method, implementation and scale factor:

1. Run once as an unreported warm-up.
2. Run at least 5–10 measured repetitions.
3. Report median and interquartile range (or mean and standard deviation if
   required by the study design).
4. Record CPU, operating system, Python, NumPy and Pillow versions.
5. Avoid other heavy workloads during measurement.

The included CLI currently reports one measured runtime per output. For final
thesis tables, wrap repeated benchmark runs and aggregate the CSVs rather than
relying on one measurement.

## 7. Interpretation

Separate conclusions about (a) visual quality, (b) objective metrics, and
(c) processing time. State where methods behave differently for upscaling and
downscaling. Explain any deviation between Pillow and the scratch version by
examining coordinate conventions, edge handling, kernel definitions and
anti-aliasing—not just by assigning one implementation a higher score.
