# Интеграция DECA

## Текущий статус

DECA подключается как внешний каталог `DECA/` в корне проекта. Каталог добавлен в
`.gitignore`, потому что это сторонний репозиторий и набор model assets, а не
исходный код тестового задания.

Проверенный локальный статус:

- исходный код DECA скачан из `https://github.com/yfeng95/DECA`;
- публичный checkpoint `data/deca_model.tar` скачан через Google Drive;
- Python-зависимости для запуска DECA установлены;
- запуск доходит до инициализации FLAME decoder;
- дальнейший запуск требует `data/generic_model.pkl`.

`generic_model.pkl` относится к FLAME2020 и не должен распространяться в
репозитории. Его нужно скачать после регистрации и принятия лицензии на сайте FLAME.

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

FLAME модели (generic_model.pkl, flame2023.pkl, flame2023_Open.pkl) нужно
скачать вручную после регистрации на https://flame.is.tue.mpg.de/ и
разместить в `DECA/data/`:

```bash
git clone --depth 1 https://github.com/yfeng95/DECA.git DECA
python -m gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar
```

Затем:

1. Зарегистрироваться на https://flame.is.tue.mpg.de/.
2. Скачать `FLAME2020.zip`.
3. Распаковать `generic_model.pkl`.
4. Положить файл в `DECA/data/generic_model.pkl`.

## Проверка

```bash
# DECA backend с FLAME2020
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/deca_result.glb \
  --device cpu \
  --no-texture

# С указанием FLAME модели (DECA всё равно использует FLAME2020)
python -m src.reconstruction.main \
  --input photo.jpg \
  --output result.glb \
  --flame-model flame2023_Open
```

Если FLAME asset отсутствует, CLI завершится понятной ошибкой и перечислит
недостающие файлы. Для проверки остального пайплайна без FLAME можно использовать:

```bash
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/mock_result.glb \
  --device cpu \
  --mock \
  --no-texture
```
