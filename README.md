# Сравнение алгоритмов интерполяции цифровых изображений на Python

Исследовательский проект для сравнения классических методов изменения размера изображений. Включает самостоятельную реализацию на NumPy и отдельную базовую реализацию на Pillow.

## Методы

| Метод | Особенность |
|---|---|
| `nearest` | Сохраняет дискретные значения; подходит для масок и пиксельной графики |
| `bilinear` | Быстрый и плавный, но может размывать детали |
| `bicubic` | Кубическая свёртка Keys (`a=-0.5`); резкий результат, возможен звон |
| `mitchell` | Компромисс между резкостью и звоном (`B=C=1/3`) |
| `bspline3` | Кубический B-сплайн с предварительной фильтрацией коэффициентов |
| `lanczos3` | Хорошо сохраняет детали; возможен звон у контрастных границ |
| `area` | Усредняет пиксельные области; особенно полезен при уменьшении |

Самостоятельная реализация (`scratch`) использует NumPy и не вызывает API изменения размера Pillow, OpenCV, SciPy или scikit-image. Pillow применяется для чтения и записи изображений и как отдельный эталон для сравнения. Методы `mitchell` и `bspline3` доступны только в самостоятельной реализации.

![Диагностика интерполяционных ядер](examples/kernel_diagnostics.png)

## Документация

- [Теоретические основы](docs/theoretical-foundations.md) — дискретизация, частотный анализ, интерполяционные ядра и оценка качества.
- [Реализация с нуля](docs/implementation-from-scratch.md) — формулы, граничные условия и устройство алгоритмов на NumPy.
- [Протокол экспериментов](docs/experiment-protocol.md) — подготовка данных и проведение сравнения.
- [Источники](docs/research-sources.md) — научные публикации и документация библиотек.
- [Краткие заметки](docs/algorithm-notes.md) — соглашения координат, фильтрация и режимы границ.\n- [Наборы тестовых изображений](docs/test-image-sources.md) — синтетические паттерны, внешние датасеты и условия использования.\n- [Аудит зависимостей](docs/dependency-audit.md) — какие библиотеки используются в scratch, CLI и визуализации.

## Установка

Требуется Python 3.10 или новее.

```bash
git clone https://github.com/pie27182/image-interpolation-study.git
cd image-interpolation-study
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Вычислительное ядро `scratch` использует NumPy и стандартную библиотеку Python. Pillow нужна для чтения/записи файлов изображений и отдельного библиотечного сравнения. Для построения графиков установите дополнительную зависимость: `python -m pip install -e '.[plots]'`. Подробности — в [аудите зависимостей](docs/dependency-audit.md).

## Справка командной строки

Справка встроена в CLI на базе стандартного модуля Python `argparse`. Флаги `-h` и `--help` доступны для самой программы и каждой команды:

```bash
image-interpolation --help
image-interpolation resize --help
image-interpolation benchmark --help
image-interpolation plot --help
image-interpolation plot-kernels --help
image-interpolation methods
image-interpolation --version
```

Запуск `image-interpolation` без команды также показывает общую справку. Команда `methods` выводит описание алгоритмов и отмечает, какие из них доступны в Pillow.

## Изменение размера изображения

Пример увеличения методом Keys:

```bash
image-interpolation resize input.jpg output.png \
  --width 1200 --height 800 --method bicubic
```

Для Pillow добавьте `--implementation pillow`. Доступные методы самостоятельной реализации: `nearest`, `bilinear`, `bicubic`, `mitchell`, `bspline3`, `lanczos3`, `area`.

Основные дополнительные параметры:

- `--boundary clamp|reflect|wrap|constant` — обработка границ;
- `--coordinate-mode half_pixel|align_corners` — отображение координат;
- `--no-antialias` — отключение расширения фильтра при уменьшении;
- `--keys-a`, `--mitchell-b`, `--mitchell-c` — параметры кубических ядер;
- `--fill-value` — значение за границами при `constant`.

## Сравнительный эксперимент

Сначала создайте восемь синтетических тестовых изображений: градиент, шахматную решётку, импульс, наклонную границу, периодическую решётку, Zone Plate, звезду Siemens и частотный sweep. Файлы сохраняются в `data/synthetic/`, который исключён из Git. Описание внешних источников и лицензий — в [руководстве по тестовым наборам](docs/test-image-sources.md).

```bash
python examples/generate_test_images.py --output-dir data/synthetic --size 512
```

Затем запустите тест качества и времени выполнения:

```bash
image-interpolation benchmark data/synthetic/synthetic_edges.png \
  --scale 2 --degradation lanczos3 --warmup 1 --repeats 5 \
  --output-dir results/synthetic-edges
```

Тест сохраняет восстановленные изображения и `results.csv` с метриками и временем выполнения. По умолчанию изображение низкого разрешения создаётся самостоятельной реализацией; для Pillow используйте `--degradation-implementation pillow`. Повторите сравнение с несколькими фильтрами ухудшения: фильтр реконструкции не следует оценивать только на данных, созданных им самим.

### Эксперимент на наборе изображений

Для исследования по нескольким изображениям подготовьте каталог с 15–30 файлами (например, фотографиями, текстурами, текстом и тонкими линиями). Сохраните сведения об источнике и лицензии каждого изображения. Запустите пакетное сравнение:

```bash
python examples/run_dataset_experiment.py data/images results/dataset \
  --scale 2 --degradation lanczos3 --warmup 1 --repeats 7
```

Скрипт проверит все изображения в каталоге и его подкаталогах, запустит все доступные методы NumPy и Pillow, сохранит отдельные результаты в подпапках и общую таблицу `results/dataset/results.csv`. Повторите эксперимент с другими коэффициентами масштабирования и фильтрами ухудшения. Не делайте вывод о лучшем методе по одному изображению или единственному сценарию ухудшения.

### Визуализация результатов

После команды `benchmark` или пакетного эксперимента постройте графики по CSV:

```bash
image-interpolation plot results/dataset/results.csv --output-dir results/dataset/plots
```

Будут созданы:

- `psnr.png` — средний PSNR по изображениям;
- `ssim.png` — средний SSIM по изображениям;
- `runtime.png` — среднее медианное время выполнения по изображениям;
- `summary.csv` — сводная таблица со средними значениями и стандартными отклонениями.

Для анализа формы ядер, частотной характеристики и переходного отклика:

```bash
image-interpolation plot-kernels --output results/kernel_diagnostics.png
```

Обе команды визуализации требуют необязательный пакет Matplotlib (`python -m pip install -e '.[plots]'`). Он используется только для отрисовки: расчёт интерполяции и метрик выполняется кодом проекта.

## Тесты

```bash
pytest -q
```

Реализация обрабатывает каналы независимо и автоматически не переводит sRGB в линейное цветовое пространство и не выполняет premultiplied-alpha. Подробности численных соглашений и ограничений приведены в [руководстве реализации](docs/implementation-from-scratch.md) и [теоретическом обзоре](docs/theoretical-foundations.md).

## Лицензия

MIT. См. `LICENSE`.
