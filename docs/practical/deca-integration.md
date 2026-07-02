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
репозитории. Его нужно скачать вручную после регистрации и принятия лицензии на
сайте FLAME.

## Установка

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
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output outputs/deca_result.glb \
  --device cpu \
  --no-texture
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
