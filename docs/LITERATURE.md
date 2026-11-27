# Литература и обоснование экспериментального дизайна

Ссылки ведут на публикации издателей, открытые версии и документацию.

## Основы интерполяции

1. Parker, J. A., Kenyon, R. V., Troxel, D. E. “Comparison of Interpolating Methods for Image Resampling.” IEEE Transactions on Medical Imaging, 2(1), 1983, pp. 31–39. DOI: https://doi.org/10.1109/TMI.1983.4307610

   Классическая журнальная работа непосредственно о ресэмплинге изображений. Сравнивает интерполяционные подходы, качество и сглаживание. Подтверждает, что сравнение фильтров при одинаковых входных данных — полноценная исследовательская постановка.

2. Keys, R. G. “Cubic Convolution Interpolation for Digital Image Processing.” IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(6), 1981, pp. 1153–1160. DOI: https://doi.org/10.1109/TASSP.1981.1163711

   Фундаментальная работа о cubic-convolution interpolation — математическая основа бикубического метода.

3. Mitchell, D. P., Netravali, A. N. “Reconstruction Filters in Computer Graphics.” Computer Graphics (SIGGRAPH '88), 22(4), 1988, pp. 221–228. DOI: https://doi.org/10.1145/378456.378514

   Рассматривает свойства реконструкционных фильтров и компромисс между резкостью, сглаживанием и артефактами.

4. Meijering, E. “A Chronology of Interpolation: From Ancient Astronomy to Modern Signal and Image Processing.” Proceedings of the IEEE, 90(3), 2002, pp. 319–342. DOI: https://doi.org/10.1109/5.993400

   Исторический и технический обзор развития интерполяции.

5. Getreuer, P. “Linear Methods for Image Interpolation.” Image Processing On Line, 1, 2011, pp. 238–259. DOI: https://doi.org/10.5201/ipol.2011.g_lmii

   Открытая рецензируемая статья с описанием nearest, bilinear, bicubic, splines и sinc interpolation. На странице доступны статья, демо и исходный код. Полезна для терминологии и математического описания: https://www.ipol.im/pub/art/2011/g_lmii/

## Benchmark и данные

6. Agustsson, E., Timofte, R. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.” CVPR Workshops, 2017, pp. 126–135. https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.html

   Публикация представляет DIV2K и его использование в benchmark single-image super-resolution.

7. Timofte, R., Agustsson, E., Van Gool, L., Yang, M.-H., Zhang, L. “NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results.” CVPR Workshops, 2017, pp. 114–125. https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Timofte_NTIRE_2017_Challenge_CVPR_2017_paper.html

   Описывает стандартный bicubic downscaling track и отдельный track с неизвестными операторами деградации. Это важно для объяснения того, почему benchmark bicubic — контролируемый, но не универсальный сценарий.

8. “A Comparative Study Of Super-Resolution Interpolation Techniques: Insights For Selecting The Most Appropriate Method.” Procedia Computer Science, 233, 2024, pp. 504–517. DOI: https://doi.org/10.1016/j.procs.2024.03.240

   Прямо близкая по постановке сравнительная работа: nearest-neighbor, bilinear, bicubic, Lanczos, spline; изображения с разными характеристиками; PSNR, SSIM и MOS. Это хороший пример методики, но Procedia Computer Science — proceedings, не следует позиционировать как IEEE/TPAMI-журнальную публикацию.

## Magic Kernel Sharp

9. Costella, J. P. “Solving the Mystery of Magic Kernel Sharp”, 2021. Авторский технический материал: https://johncostella.com/magic/

   Автор описывает происхождение MKS и математико-частотные свойства. Это первичный материал автора фильтра, но не независимая peer-reviewed журнальная оценка.

10. Pillow 13.0.0 release notes (2026-10-16): https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html

   Официальная документация фиксирует добавление Image.Resampling.MKS2013 и Image.Resampling.MKS2021 для resize()/thumbnail(). Это свидетельствует о доступности в реальном программном инструменте, а не доказывает качество.

11. Pillow Image concepts, resampling filters: https://pillow.readthedocs.io/en/stable/handbook/concepts.html

   Документирует фильтры, их назначение и качественный компромисс по upscaling/downscaling/performance. Это документация реализации, а не независимая статья.

## Адаптивные методы для возможного расширения

12. Li, J., Chen, C., Huang, W., Lang, Z., Song, F., Yan, Y., Xiong, Z. “Learning Steerable Function for Efficient Image Resampling.” CVPR, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Li_Learning_Steerable_Function_for_Efficient_Image_Resampling_CVPR_2023_paper.html

13. Li, J., Chen, C., Song, F., Yan, Y., Xiong, Z. “LeRF: Learning Resampling Function for Adaptive and Efficient Image Interpolation.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. DOI: https://doi.org/10.1109/TPAMI.2025.3577227

   LeRF учит адаптивную функцию ресэмплинга. Код авторов: https://github.com/ddlee-cn/LeRF-PyTorch (репозиторий объявляет MIT License). Это достойный кандидат для отдельного расширения ВКР, но он требует обученных весов и PyTorch, поэтому не смешан с пятью фиксированными фильтрами основного эксперимента.

## OCR и реальные текстовые изображения

14. Wang, W., Xie, E., Liu, X., Wang, W., Liang, D., Shen, C., Bai, X. “Scene Text Image Super-resolution in the Wild.” ECCV, 2020. DOI: https://doi.org/10.1007/978-3-030-58607-2_38

   Работа вводит TextZoom — paired real low/high-resolution text images, снятые камерами с различными focal lengths. Авторы подчёркивают, что synthetic bicubic degradation не достаточно отражает real-world scene text, а конечная задача — улучшение распознавания. Публикация: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1186_ECCV_2020_paper.php

## Как литература определила эксперимент

| Решение | Основание |
|---|---|
| Bicubic baseline | Pillow API default для обычного режима + распространённый baseline в SR benchmark |
| Bilinear, Bicubic, Lanczos | Parker et al.; Getreuer; сравнительная работа 2024 года |
| MKS2013/MKS2021 | Pillow 13.0 API и технический материал автора фильтра |
| DIV2K validation | Agustsson & Timofte / NTIRE 2017 |
| PSNR, SSIM | Сравнительная работа 2024 года и практика SR benchmark |
| Отдельная OCR проверка | Wang et al., TextZoom, ECCV 2020 |
| LeRF как опциональное расширение | CVPR 2023 и IEEE TPAMI 2025 |

## Критическая оценка источников

- Parker, Keys, Meijering, Mitchell & Netravali, Getreuer — основные журнальные/конференционные труды по интерполяции.
- CVPR Workshops/NTIRE и ECCV — конференционные публикации по benchmark и реальным данным.
- TPAMI 2025 — журнальная публикация об адаптивном ресэмплинге.
- Procedia Computer Science 2024 — полезная прямая сравнительная работа, но её престиж и тип публикации нужно указывать корректно.
- Pillow release notes/API — документация программной реализации, не научная статья.
- Costella — авторский технический материал по MKS, не независимое peer-reviewed исследование.

Для окончательного текста ВКР нужно оформить библиографию согласно требованиям кафедры, указать DOI и процитировать официальный DIV2K dataset paper. Проверить права на изображения по официальной странице DIV2K.
