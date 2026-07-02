# 3D Face Reconstruction

3D реконструкция лица по фотографии с использованием DECA и визуализацией через Three.js.

## Быстрый старт

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 2. Установить зависимости
pip install -e .

# 3. Запустить реконструкцию
python -m src.reconstruction.main --input photo.jpg --output result.glb
```

## Структура

```
src/
├── reconstruction/     # Python: реконструкция лица
│   ├── face_reconstructor.py
│   ├── preprocessor.py
│   └── postprocessor.py
└── visualization/      # Three.js: визуализация
    └── face_viewer.js
```

## Документация

- [Архитектура решения](docs/practical/architecture.md)
- [Инструкция по запуску](docs/practical/setup.md)
- [Исследование методов](docs/research/)

## Лицензия

MIT License - Boiko Oleg, 2026
