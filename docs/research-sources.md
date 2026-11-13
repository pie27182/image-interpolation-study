# Источники

Для математических утверждений следует ссылаться на оригинальные научные публикации; документацию библиотек используйте при описании поведения конкретных программных реализаций.

## Теория дискретизации и реконструкции

1. **MIT, Foundations of Computer Vision**, главы [Image Sampling and Aliasing](https://visionbook.mit.edu/sampling_and_aliasing.html) и [Downsampling and Upsampling Images](https://visionbook.mit.edu/upsamplig_downsampling_2.html). Дискретизация, частота Найквиста, реконструкция sinc, подавление наложения спектров и передискретизация.
2. **R. C. Gonzalez и R. E. Woods**, *Digital Image Processing*, 4-е изд., Pearson, 2018. Учебный источник по обработке цифровых изображений и геометрическим преобразованиям.
3. **P. S. Heckbert**, [*Fundamentals of Texture Mapping and Image Warping*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/5504.html), UC Berkeley Technical Report UCB/CSD-89-516, 1989. Геометрические отображения и фильтрация при деформации изображений.

## Интерполяционные фильтры и сплайны

4. **R. G. Keys**, [“Cubic Convolution Interpolation for Digital Image Processing”](https://doi.org/10.1109/TASSP.1981.1163711), *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 29(6), 1153–1160, 1981. Кубическая свёртка.
5. **D. P. Mitchell и A. N. Netravali**, [“Reconstruction Filters in Computer Graphics”](https://doi.org/10.1145/378456.378514), *Computer Graphics* (SIGGRAPH), 22(4), 221–228, 1988. Двухпараметрические кубические фильтры.
6. **M. Unser**, [“Splines: A Perfect Fit for Signal and Image Processing”](https://doi.org/10.1109/79.799930), *IEEE Signal Processing Magazine*, 16(6), 22–38, 1999. B-сплайны и предварительная фильтрация коэффициентов.
7. **P. Thévenaz, T. Blu и M. Unser**, [“Interpolation Revisited”](https://doi.org/10.1109/42.875199), *IEEE Transactions on Medical Imaging*, 19(7), 739–758, 2000. Теория интерполяции и аппроксимации.
8. **X. Li и M. T. Orchard**, [“New Edge-Directed Interpolation”](https://doi.org/10.1109/83.951537), *IEEE Transactions on Image Processing*, 10(10), 1521–1527, 2001. Адаптивная интерполяция с учётом структуры границ.

## Метрики и реализации библиотек

9. **Z. Wang, A. C. Bovik, H. R. Sheikh и E. P. Simoncelli**, [“Image Quality Assessment: From Error Visibility to Structural Similarity”](https://doi.org/10.1109/TIP.2003.819861), *IEEE Transactions on Image Processing*, 13(4), 600–612, 2004. Оригинальная статья о SSIM.
10. **Pillow**, [документация по ресэмплингу](https://pillow.readthedocs.io/en/stable/reference/Image.html) и [исходный код фильтров](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c). Поведение доступных фильтров и параметры реализации.
11. **OpenCV**, [геометрические преобразования](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) и [исходный код изменения размера](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp). Режимы интерполяции и детали реализации.
12. **SciPy**, [`scipy.ndimage.zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html). Сплайн-интерполяция, предварительная фильтрация и режимы границ.
13. **International Color Consortium**, [реестр sRGB](https://registry.color.org/rgb-registry/srgb). Передаточная функция и кодирование цвета.
14. **scikit-image**, [документация метрик](https://scikit-image.org/docs/stable/api/skimage.metrics.html). Справочник по MSE, PSNR и SSIM.

## Примечание о цитировании

Уравнения Keys, Mitchell--Netravali и B-сплайнов следует связывать с соответствующими оригинальными статьями (пункты 4–7). Для описания конкретных отличий Pillow, OpenCV или SciPy цитируйте документацию этих библиотек. Реализация этого проекта написана независимо на NumPy; ссылки на исходный код библиотек приведены для проверки их поведения, а не как источники скопированного кода.
