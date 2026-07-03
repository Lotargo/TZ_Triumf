# Проверка DECA/FLAME пайплайна

## Выводы по первичным источникам

Проверка выполнена по локальной копии официального DECA в `DECA/` и публичной
документации проекта:

- официальный demo вызывает `deca.encode(images)` и затем `deca.decode(codedict)`
  с renderer path по умолчанию;
- персональная UV-текстура появляется как `uv_texture_gt` внутри
  `DECA.decode(..., return_vis=True)`;
- `DECA.save_obj()` берёт `verts` из `opdict`, а faces/UV layout из
  `self.render.faces`, `self.render.raw_uvcoords`, `self.render.uvfaces`;
- detail normals, displacement map и detailed OBJ требуют renderer path и не
  появляются в CPU mesh-only fallback;
- mean texture FLAME не является персональной текстурой с фото.

Ссылки:

- https://github.com/yfeng95/DECA
- https://pytorch3d.org/tutorials/render_textured_meshes

## Что исправлено в локальном пайплайне

1. Gaussian smoothing по порядку массива вершин отключён по умолчанию
   (`smooth_sigma=0.0`). Это важно для FLAME/DECA, потому что соседство вершин
   определяется faces, а не индексом в массиве.
2. Добавлен mesh sanity-check:
   - количество вершин и faces;
   - конечность координат;
   - валидность индексов faces;
   - количество degenerate faces;
   - количество connected components;
   - совместимость `uv`/`uv_faces`.
3. DECA image-to-mesh теперь явно разделяет два режима:
   - mesh-only fallback: `rendering=False`, `use_detail=False`, без UV-текстуры;
   - renderer path: `rendering=True`, `return_vis=True`, `use_detail=True`, с
     извлечением `uv_texture_gt`, `raw_uvcoords` и `uvfaces`.
4. Добавлена проверка окружения:

```bash
python -m src.reconstruction.main --check-env
```

На текущей машине результат:

```text
torch: ok (2.12.1+cu130)
CUDA: ok (NVIDIA GeForce RTX 2070 Super)
PyTorch3D: missing
DECA standard rasterizer: ok
DECA assets: ok
full renderer path: ready
```

## Что сделано для GPU

1. Установлена CUDA-сборка PyTorch `2.12.1+cu130`, совместимая с локальным CUDA
   Toolkit 13.0.
2. PyTorch3D wheel для текущей связки Windows/Python 3.12 не найден, поэтому
   собран DECA `standard_rasterize_cuda` extension.
3. `python -m src.reconstruction.main --check-env` показывает
   `full renderer path: ready`.
4. Локальный pipeline генерирует `site/models/deca_result.glb` через
   `DECA renderer enabled: standard`.

## Практическая проверка

```bash
python -m src.reconstruction.main \
  --input DECA/TestSamples/examples/IMG_0392_inputs.jpg \
  --output site/models/deca_result.glb \
  --device cuda
```

## Что остаётся сделать

1. Доскачать/починить face-alignment SFD detector cache для официального DECA demo
   с `--iscrop True`, если нужен именно FAN crop path.
2. Сравнить официальный OBJ с GLB, который экспортирует локальный pipeline.
3. Проверить diagnostics для обоих мешей: connected components, degenerate faces,
   bounds, face index range, UV face compatibility.
4. При наличии `FLAME_albedo_from_BFM.npz` включить DECA albedo model и сравнить
   качество UV texture с текущим texture-from-photo fallback.
