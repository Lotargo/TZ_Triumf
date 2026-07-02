# Инструкция по запуску

## Предварительные требования

### Системные требования

- **ОС:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **Python:** 3.10 или выше
- **Node.js:** 18 или выше (для веб-интерфейса)
- **GPU:** Рекомендуется NVIDIA с CUDA (не обязательно)

### Установка зависимостей

#### 1. Клонирование репозитория

```bash
git clone https://github.com/BoikoOleg/TZ_Triumf.git
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

Модели FLAME 2020/2023 нужно скачать вручную после регистрации на
[flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) и разместить в `DECA/data/`:

- `generic_model.pkl` (FLAME2020) — обязателен для DECA
- `flame2023_Open.pkl` (FLAME2023 Open, CC-BY-4.0)
- `flame2023.pkl` (FLAME2023)
- `FLAME_masks.pkl`

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

# Реконструкция с настройками
python -m src.reconstruction.main \
    --input photo.jpg \
    --output result.glb \
    --device cuda \
    --detail-level high

# Использование FLAME2023 Open (для альтернативных бэкендов)
python -m src.reconstruction.main \
    --input photo.jpg \
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

## Использование

### Загрузка изображения

1. Откройте веб-интерфейс
2. Нажмите "Upload Photo"
3. Выберите фотографию лица (JPEG/PNG)
4. Дождитесь обработки (1–3 секунды)
5. Просмотрите 3D-модель

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
├── .venv/                   # Виртуальное окружение
├── DECA/                    # Клонированный репозиторий DECA
├── src/
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   ├── face_reconstructor.py
│   │   ├── preprocessor.py
│   │   └── postprocessor.py
│   └── visualization/
│       └── face_viewer.js
├── site/
│   ├── index.html
│   ├── css/
│   └── js/
├── tests/
├── docs/
├── pyproject.toml
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
# Запуск оценки качества
python -m src.evaluation.run \
    --input test_images/ \
    --output results/ \
    --metrics chamfer,normal_consistency
```

## Дополнительные ресурсы

- [Документация DECA](https://github.com/yfeng95/DECA)
- [Three.js документация](https://threejs.org/docs/)
- [PyTorch документация](https://pytorch.org/docs/stable/)

---

*При проблемах с установкой создайте issue в репозитории.*
