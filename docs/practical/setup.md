# Инструкция по запуску

## Предварительные требования

### Системные требования

- **ОС:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **Python:** 3.10 или выше
- **GPU:** Рекомендуется NVIDIA с CUDA (не обязательно)

### Установка зависимостей

#### 1. Клонирование репозитория

```bash
git clone https://github.com/Lotargo/TZ_Triumf.git
cd TZ_Triumf
```

#### 2. Python окружение

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (macOS/Linux)
source .venv/bin/activate

# Установка зависимостей
pip install -e .

# Для запуска тестов
pip install -e ".[dev]"
```

#### 3. FLAME модели

Модели FLAME размещаются в `DECA/data/`:

- `generic_model.pkl` (FLAME2020) — обязателен для DECA
- `flame2023_Open.pkl` (FLAME2023 Open, CC-BY-4.0)
- `flame2023.pkl` (FLAME2023)
- `FLAME_masks.pkl`

Для FLAME2020 добавлена воспроизводимая загрузка из Hugging Face-зеркала с
проверкой SHA256:

```bash
python -m src.reconstruction.download_assets flame2020

# После установки пакета:
download-reconstruction-assets flame2020
```

Скрипт сохраняет файл в `DECA/data/generic_model.pkl` и проверяет checksum
`efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b`.
Перед использованием учитывайте лицензионные условия FLAME.

#### 4. Дополнительные зависимости для DECA

```bash
# Клонирование DECA
git clone https://github.com/yfeng95/DECA.git
python -m gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar

# Современное окружение Python 3.12:
pip install scikit-image yacs kornia ninja fvcore face-alignment
pip install chumpy --no-build-isolation
```

Подробности и статус интеграции: [`deca-integration.md`](deca-integration.md).

Проверить, готово ли окружение к полноценному DECA renderer path:

```bash
python -m src.reconstruction.main --check-env
```

Для персональной текстуры с фото и detail normals нужны CUDA-сборка PyTorch,
PyTorch3D или рабочий DECA rasterizer, `deca_model.tar`, `generic_model.pkl` и
`head_template.obj`.

#### 4. Веб-интерфейс (опционально)

```bash
python -m http.server 3000 -d site
```

## Запуск

### Вариант 1: Командная строка

```bash
# Реконструкция одного изображения
python -m src.reconstruction.main --input photo.jpg --output result.glb

# Быстрая проверка без установленной DECA
python -m src.reconstruction.main --input photo.jpg --output result.glb --mock --device cpu

# Проверка реального DECA backend после установки FLAME asset
python -m src.reconstruction.main \
    --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
    --output outputs/deca_result.glb \
    --device cpu \
    --no-texture

# Проверка готовности full DECA renderer path
python -m src.reconstruction.main --check-env

# Реконструкция с настройками
python -m src.reconstruction.main \
    --input photo.jpg \
    --output result.glb \
    --device cuda \
    --detail-level high

# Генерация нейтральной FLAME2023 Open модели без входного фото
python -m src.reconstruction.main \
    --output result.glb \
    --flame-model flame2023_Open

# После установки пакета доступна короткая команда
face-reconstruct --input photo.jpg --output result.glb --mock --device cpu
```

### Вариант 2: Python скрипт

```python
from src.reconstruction import FaceReconstructor

# Инициализация
reconstructor = FaceReconstructor(device='cuda')

# Реконструкция
result = reconstructor.reconstruct('photo.jpg')

# Сохранение
result.to_glb('output.glb')
result.to_obj('output.obj')

print(f"Реконструкция завершена: {result.vertices.shape[0]} вершин")
```

### Вариант 3: Веб-интерфейс

```bash
# Запуск статического сервера
python -m http.server 3000 -d site

# Открыть в браузере
# http://localhost:3000
```

Лендинг также доступен на GitHub Pages:
[https://lotargo.github.io/TZ_Triumf/](https://lotargo.github.io/TZ_Triumf/)

## Использование

### Просмотр 3D-демо

На лендинге доступны три предсгенерированных GLB-модели для интерактивного просмотра:

1. Откройте страницу в браузере
2. Перейдите в секцию "3D-демо"
3. Переключайтесь между моделями: DECA result, FLAME textured, FLAME neutral
4. Управляйте вьювером: вращение, масштаб, wireframe, авто-вращение

### Управление 3D-моделью

- **Вращение:** Левая кнопка мыши + перетаскивание
- **Масштаб:** Колёсико мыши
- **Панорамирование:** Правая кнопка мыши + перетаскивание

### Экспорт

- **Скачивание GLB:** Кнопка "Download GLB"
- **Скачивание OBJ:** Кнопка "Download OBJ"
- **Скачивание PDF:** Кнопка "Download Report"

## Решение проблем

### Проблема: CUDA out of memory

```bash
# Решение: Использовать CPU
python -m src.reconstruction.main --input photo.jpg --output result.glb --device cpu
```

### Проблема: Модель не загружается

```bash
# Проверка пути к модели
ls -la DECA/data/

# Скачивание модели заново
cd DECA && bash models/download_models.sh
```

### Проблема: Медленная обработка

```bash
# Проверка доступности GPU
python -c "import torch; print(torch.cuda.is_available())"

# Установка CUDA версии PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Ошибка при экспорте в GLB

```bash
# Установка trimesh
pip install trimesh

# Проверка версии
python -c "import trimesh; print(trimesh.__version__)"
```

## Структура проекта после установки

```
TZ_Triumf/
├── .venv/                            # Виртуальное окружение
├── DECA/                             # Клонированный репозиторий DECA
├── src/
│   ├── reconstruction/
│   │   ├── __init__.py               # Lazy imports
│   │   ├── flame_decoder_2023.py     # Standalone LBS-декодер FLAME 2023
│   │   ├── face_reconstructor.py     # Фасад: DECA + FLAME2023 режимы
│   │   ├── preprocessor.py           # Детекция и выравнивание лица
│   │   ├── postprocessor.py          # Очистка меша, экспорт OBJ/GLB/PLY
│   │   ├── main.py                   # CLI точка входа
│   │   ├── runtime_check.py          # Проверка окружения
│   │   └── download_assets.py        # Загрузчик FLAME2020 с SHA256
│   └── visualization/
│       └── __init__.py               # Stub (Python-визуализация отложена)
├── site/                             # Лендинг (аналитический веб-отчёт)
│   ├── index.html
│   ├── css/style.css
│   ├── js/viewer.js                  # Three.js 3D-вьювер
│   ├── js/main.js                    # Документ-вьювер
│   ├── js/export.js                  # PDF и Markdown экспорт
│   ├── models/                       # GLB-модели для демо
│   └── assets/                       # Декоративные SVG
├── tests/
├── docs/
├── pyproject.toml
├── .github/workflows/
│   └── deploy-pages.yml              # CI/CD на GitHub Pages
└── README.md
```

## Тестирование

### Запуск тестов

```bash
# Unit тесты
pytest
```

### Проверка качества

```bash
# Diagnostics меша встроены в постпроцессинг:
python -m src.reconstruction.main \
    --input photo.jpg --output result.glb --mock --device cpu

# Проверка готовности окружения к полноценной DECA-реконструкции:
python -m src.reconstruction.main --check-env
```

## Дополнительные ресурсы

- [Документация DECA](https://github.com/yfeng95/DECA)
- [Three.js документация](https://threejs.org/docs/)
- [PyTorch документация](https://pytorch.org/docs/stable/)

---

*При проблемах с установкой создайте issue в репозитории.*
