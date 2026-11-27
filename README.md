# Сравнение алгоритмов интерполяции цифровых изображений

Воспроизводимый эксперимент для ВКР: пять фильтров масштабирования в Python, одна библиотека, один датасет и фиксированный масштаб ×4.

[Открыть ноутбук в Google Colab](https://colab.research.google.com/github/pie27182/image-interpolation-study/blob/study/interpolation-benchmark/notebooks/interpolation_benchmark_colab.ipynb)

## Методы

| Алгоритм | Роль |
|---|---|
| Bilinear | Классический ориентир по скорости и сглаживанию |
| **Bicubic** | **Baseline** |
| Lanczos | Оконная sinc-интерполяция |
| MKS2013 | Magic Kernel Sharp, версия 2013 |
| MKS2021 | Magic Kernel Sharp, версия 2021 |

Baseline обоснован конкретно для Pillow: документация указывает BICUBIC как default для обычных режимов изображения. Это не утверждение о едином default для всех библиотек. Pillow 13.0.0 закреплён, поскольку в нём появились MKS2013 и MKS2021. Все фильтры применяются через один API Pillow Image.resize(); собственные ядра не переписываются.

## Данные и измерения

Официальный DIV2K validation: до 100 реальных HR фотографий и соответствующие bicubic LR изображения ×4. Для каждого LR алгоритм строит результат в размере соответствующего HR эталона.

Метрики:
- PSNR и SSIM по яркостной компоненте (BT.601-style luma);
- PSNR и SSIM по RGB;
- медианное время вызова resize() и миллисекунды на мегапиксель результата.

Тайминг: CPU, один прогревочный вызов, затем пять повторов; чтение данных и расчёт метрик исключены из таймера. Парные сравнения с Bicubic: критерий Уилкоксона и поправка Holm.

## Google Colab

Откройте ноутбук выше и выполните все ячейки сверху вниз. Он устанавливает зависимости, скачивает официальный DIV2K validation и запускает полный эксперимент.

## Локальный запуск

    python -m pip install -r requirements.txt
    python scripts/download_div2k.py --root data/DIV2K
    python scripts/run_benchmark.py --data-root data/DIV2K --output-dir results/smoke --scale 4 --repeats 3 --max-images 5

Полный прогон:

    python scripts/run_benchmark.py --data-root data/DIV2K --output-dir results/DIV2K_x4 --scale 4 --repeats 5

## Файлы результатов

- metrics_per_image.csv — метрики каждой пары «изображение—метод»;
- summary.csv — сводные показатели;
- paired_tests_vs_bicubic.csv — парная статистика относительно baseline;
- comparison_grid.png — визуальные примеры;
- run_metadata.json — параметры эксперимента и среда.

Данные и результаты не коммитятся: data/ и results/ добавлены в .gitignore.

## Ограничения

DIV2K — контролируемый benchmark с bicubic degradation; он не воспроизводит все реальные процессы камеры (шум, смаз, фокус, JPEG и неизвестная оптика). Так как LR создавались bicubic downsampling, baseline Bicubic может иметь преимущество. См. docs/METHODOLOGY.md.

Основной эксперимент не измеряет OCR и не делает выводов о распознавании автомобильных номеров. Для такого вывода нужен отдельный эксперимент на реальных парных текстовых изображениях (например, TextZoom), фиксированный OCR-движок и метрики CER / exact recognition accuracy. PSNR/SSIM не гарантируют улучшения распознавания.

Энергия и пиковая память не заявляются: стандартная Colab VM не даёт стабильного переносимого измерителя энергии CPU, а tracemalloc не учитывает все нативные аллокации Pillow.

## Научная документация и лицензии

- docs/METHODOLOGY.md — протокол и ограничения.
- docs/LITERATURE.md — публикации, DOI и обоснование решений.
- THIRD_PARTY_NOTICES.md — лицензии программных компонентов и условия DIV2K.

Код проекта распространяется по MIT License. Это не меняет условия DIV2K: официальная страница указывает academic research use, а авторские права на изображения остаются у правообладателей.
