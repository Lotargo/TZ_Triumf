# 3D Face Reconstruction

Проект исследует и прототипирует пайплайн 3D-реконструкции лица по 1-3 фото со
смартфона. Цель — получить web-ready 3D-ассет головы: стабильный mesh,
согласованную UV-текстуру, маски/карты материала и удобный браузерный preview
для проверки результата.

Основной отчет: [docs/report.md](docs/report.md)

Публичный лендинг: [https://lotargo.github.io/TZ_Triumf/](https://lotargo.github.io/TZ_Triumf/)

Локальный viewer: [site/index.html](site/index.html)

## Что уже есть

- Python CLI для реконструкции и генерации 3D-моделей.
- DECA integration для image-to-mesh reconstruction.
- Standalone `Flame2023Decoder` на PyTorch без зависимости от DECA/chumpy.
- Экспорт OBJ/GLB/PLY.
- Mesh diagnostics: bounds, connected components, degenerate faces, UV checks.
- Three.js viewer для просмотра DECA/FLAME ассетов в браузере.
- Документация по исследованиям, архитектуре, multi-view выводам и texture/mask
  pipeline.
- Тесты на декодер, CLI, экспорт и diagnostics.

## Текущий фокус

Практические эксперименты показали, что качество результата нельзя улучшить
только увеличением числа полигонов. Плотная сетка лучше показывает микрорельеф,
но одновременно раскрывает все ошибки UV, швов, текстуры и масок.

Текущий production-oriented вектор:

```text
coarse mesh
  + clean UV baseColor/albedo
  + semantic masks
  + normal/detail normal
  + optional height/bump
  + material maps
```

То есть проект движется от разовых vertex-color экспериментов к полноценному
2D/UV texture baking workflow.

## Быстрый старт

Установка:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Проверка тестов:

```bash
pytest -q
```

Проверка окружения для DECA:

```bash
python -m src.reconstruction.main --check-env
```

## Лицензированные модели и ассеты

После клонирования репозитория полноценная генерация DECA/FLAME результатов не
заработает автоматически. В репозитории хранится только код, документация,
viewer и демонстрационные web-ассеты. Предобученные модели, FLAME templates,
texture space и checkpoint-файлы нужно скачать самостоятельно с официальных
источников после регистрации, принятия лицензий и соблюдения условий
распространения.

Минимальный набор для практического запуска:

- `DECA/data/deca_model.tar` — checkpoint DECA с официального источника DECA.
- `DECA/data/generic_model.pkl` — FLAME2020 model для DECA.
- `DECA/data/flame2023_Open.pkl` или `DECA/data/flame2023.pkl` — FLAME2023 для
  standalone `Flame2023Decoder`.
- `DECA/data/FLAME_texture.npz` — FLAME texture space / mean texture.
- сопутствующие DECA/FLAME UV, mask, landmark и template assets, если они нужны
  выбранному режиму реконструкции.

Без этих файлов можно запускать тесты, mock-реконструкцию, читать документацию и
открывать подготовленный web-viewer, но нельзя воспроизвести полноценный
image-to-mesh результат с нуля.

Официальные источники:

- DECA: <https://github.com/yfeng95/DECA>
- FLAME: <https://flame.is.tue.mpg.de/>
- DECA project page: <https://deca.is.tue.mpg.de/>

Mock-реконструкция без внешних весов:

```bash
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/mock_face.glb \
  --device cpu \
  --mock \
  --no-texture
```

Генерация нейтрального FLAME2023 mesh:

```bash
python -m src.reconstruction.main \
  --flame-model flame2023_Open \
  --output outputs/flame2023_neutral.glb \
  --format glb
```

Для FLAME2023 нужен файл `DECA/data/flame2023_Open.pkl`, скачанный с
официального сайта FLAME после принятия лицензии. Для полноценной
DECA-реконструкции нужны `generic_model.pkl`, `deca_model.tar` и совместимые
texture/UV assets.

## Веб-просмотрщик

Локальный запуск:

```bash
python -m http.server 3000 -d site
```

Затем открыть:

```text
http://localhost:3000/#demo
```

В viewer доступны:

- `DECA result` — coarse DECA/FLAME-topology mesh из трех проекций;
- `DECA baked detail` — легкий mesh с baked normal/bump maps;
- `DECA detail` — плотная DECA detail-геометрия;
- `FLAME textured` — FLAME2023 с UV/mean texture;
- `FLAME mask` — FLAME texture-space baseline в vertex colors;
- `FLAME detail textured` / `FLAME detail` — subdivided FLAME варианты;
- `FLAME neutral` — нейтральная параметрическая голова без текстуры.

## Документация

Главные документы:

- [docs/report.md](docs/report.md) — текущий проектный отчет и направление.
- [docs/practical/architecture.md](docs/practical/architecture.md) — архитектура
  пайплайна.
- [docs/practical/deca-integration.md](docs/practical/deca-integration.md) —
  детали интеграции DECA.
- [docs/practical/multiview-reconstruction.md](docs/practical/multiview-reconstruction.md)
  — выводы по three-view reconstruction, detail mesh, UV fusion и mask pipeline.
- [docs/research/08-comfyui-masks-meshes.md](docs/research/08-comfyui-masks-meshes.md)
  — ComfyUI, segmentation, masks, meshes и semi-manual workflow.
- [docs/research/09-game-face-textures-masks.md](docs/research/09-game-face-textures-masks.md)
  — игровые texture/mask практики для лица персонажа.

## Структура проекта

```text
TZ_Triumf/
├── README.md
├── pyproject.toml
├── docs/
│   ├── report.md
│   ├── research/
│   └── practical/
├── src/
│   ├── reconstruction/
│   │   ├── flame_decoder_2023.py
│   │   ├── face_reconstructor.py
│   │   ├── main.py
│   │   └── ...
│   └── visualization/
├── tests/
├── site/
│   ├── index.html
│   ├── css/style.css
│   ├── js/
│   └── models/
└── DECA/                  # external assets, ignored by git
```

## Важные выводы

### Multi-view

Три проекции не нужно склеивать как три разных меша. Правильнее решать shared
identity fitting:

```text
images[3]
  -> DECA/MICA/EMOCA initialization per view
  -> shared shape
  -> per-view pose/camera/expression/light
  -> visibility-aware UV texture fusion
  -> one canonical mesh
```

### Маски

Неудачный эксперимент `DECA + FLAME mask` удален из viewer. Он показал, что
FLAME texture-space mask нельзя напрямую переносить в DECA vertex colors без
явного UV remap. Правильный путь:

```text
semantic 2D masks -> projection -> target UV texture -> material maps
```

### Детализация

Для real-time/WebGL чаще полезнее держать легкий mesh и переносить микродетали
в normal/detail normal/height maps, чем делать всю детализацию геометрией.

## GitHub Pages

Публикация настроена через GitHub Actions. Workflow копирует `docs/` внутрь
`site/docs` перед публикацией, затем выкладывает `site/` на Pages.

Корневой [index.html](index.html) перенаправляет на `site/`.

## Статус

Проект находится в активной R&D/prototype стадии. Ближайшие направления:

- явный UV texture/mask bake в целевую развертку;
- semantic face parsing для кожи, губ, глаз, ушей, волос и шеи;
- visibility-aware multi-view texture fusion;
- отдельные режимы viewer для geometry, texture, normal, mask и wireframe;
- API-слой для загрузки фото и запуска серверного инференса.
