# Интеграция DECA

## Текущий статус

DECA подключается как внешний каталог `DECA/` в корне проекта. Каталог добавлен в
`.gitignore`, потому что это сторонний репозиторий и набор model assets, а не
исходный код тестового задания.

Проверенный локальный статус:

- исходный код DECA скачан из `https://github.com/yfeng95/DECA`;
- публичный checkpoint `data/deca_model.tar` скачан через Google Drive;
- Python-зависимости для запуска DECA установлены;
- `data/generic_model.pkl` можно скачать через `src.reconstruction.download_assets`
  из Hugging Face-зеркала с проверкой SHA256.
- CPU image-to-mesh запуск проверен в mesh-only режиме без PyTorch3D:
  `outputs/deca_result.glb` создаётся из `DECA/TestSamples/examples/IMG_0392_inputs.jpg`.

`generic_model.pkl` относится к FLAME2020 и не должен распространяться в
репозитории. Перед использованием нужно учитывать лицензионные условия FLAME.

## Модели FLAME

### FLAME2020 (требуется DECA)

DECA обучена с FLAME2020 (`generic_model.pkl`). Модельный чекпоинт `deca_model.tar`
использует shape/expression базисы именно FLAME2020 — замена на FLAME2023 приведёт к
некорректной геометрии. Поэтому для DECA backend всегда используется FLAME2020.

### FLAME 2023 / FLAME 2023 Open (для альтернативных бэкендов)

С мая 2023 доступна обновлённая версия FLAME с исправленной областью глаз (`flame2023.pkl`).
С ноября 2025 — Open-версия под лицензией CC-BY-4.0 (`flame2023_Open.pkl`).

Эти модели имеют другой формат (без chumpy) и несовместимы с DECA, но могут
использоваться с другими бэкендами (EMOCA, MICA, или собственный FLAME decoder).

## Установка

### Загрузка моделей

```bash
# DECA checkpoint
git clone --depth 1 https://github.com/yfeng95/DECA.git DECA
python -m gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar
```

FLAME2020 для DECA можно скачать в `DECA/data/generic_model.pkl` командой:

```bash
git clone --depth 1 https://github.com/yfeng95/DECA.git DECA
python -m gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar
python -m src.reconstruction.download_assets flame2020
```

Скрипт использует Hugging Face-зеркало `camenduru/show` и проверяет SHA256:

```text
efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b
```

FLAME2023 / FLAME2023 Open размещаются в том же каталоге как
`flame2023.pkl` и `flame2023_Open.pkl`.

## Проверка

```bash
# DECA backend с FLAME2020
python -m src.reconstruction.download_assets flame2020
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/deca_result.glb \
  --device cpu \
  --no-texture

# Генерация FLAME2023 без входного фото
python -m src.reconstruction.main \
  --output result.glb \
  --flame-model flame2023_Open
```

Если FLAME asset отсутствует, CLI завершится понятной ошибкой и перечислит
недостающие файлы. Если `pytorch3d` не установлен, пайплайн автоматически
переключается в mesh-only режим: DECA восстанавливает вершины и faces без
рендера, detail normals и texture extraction. Для проверки остального пайплайна
без FLAME можно использовать:

```bash
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/mock_result.glb \
  --device cpu \
  --mock \
  --no-texture
```
