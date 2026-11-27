# Third-party software and dataset notices

## Pillow

This project calls the public resampling API of Pillow 13.0.0. No Pillow source code is copied into this repository. Pillow is distributed under the MIT-CMU License:
https://github.com/python-pillow/Pillow/blob/main/LICENSE

MKS2013 and MKS2021 are resampling filters added to Pillow 13.0.0:
https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html

## Other Python dependencies

NumPy, pandas, scikit-image, SciPy, Matplotlib and tqdm are installed as dependencies and used through their public APIs. No source code from these packages is vendored here. Their own licenses are included in their distributions.

## DIV2K dataset

DIV2K images are not included in this Git repository. The download script retrieves the official validation HR and bicubic LR x4 archives from:
https://data.vision.ee.ethz.ch/cvl/DIV2K/

The official DIV2K page states academic research use and that image copyrights remain with their original owners. Users must follow the official terms; this repository does not grant image redistribution or commercial-use rights.

## LeRF (optional future extension)

The authors' LeRF PyTorch repository is not copied or vendored:
https://github.com/ddlee-cn/LeRF-PyTorch

Its repository declares the MIT License. It is noted as a candidate optional extension only.
