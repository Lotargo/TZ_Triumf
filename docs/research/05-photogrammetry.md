# Фотограмметрия

## Введение

Фотограмметрия — метод восстановления трёхмерной структуры сцены по серии двухмерных изображений, снятых с разных ракурсов. Основан на принципах проективной геометрии и триангуляции ключевых точек. Фотограмметрия обеспечивает высокую точность реконструкции, но требует множества изображений и контролируемых условий.

## Основы фотограмметрии

### Принцип работы

```
Изображение 1 ──┐
                ├──▶ Триангуляция ──▶ 3D точки ──▶ Меш
Изображение 2 ──┘
```

**Этапы:**
1. **Детекция ключевых точек** (SIFT, SURF, ORB)
2. **Матчинг точек** между изображениями
3. **Восстановление позы камеры** (Structure from Motion)
4. **Триангуляция 3D точек**
5. **Построение mesh** (Poisson reconstruction)
6. **Текстурирование**

### Математический аппарат

**Проективная камера:**
```
[u]   [f  0  cx] [R | t] [X]
[v] = [0  f  cy]         [Y]
[1]   [0  0  1 ]         [Z]
                          [1]
```

Где:
- (u, v) — координаты на изображении
- f — фокусное расстояние
- (cx, cy) — principal point
- [R | t] — extrinsic parameters (поза камеры)
- (X, Y, Z) — 3D координаты

**Триангуляция:**
```python
def triangulate(point1, point2, P1, P2):
    """
    Триангуляция 3D точки по двум 2D проекциям
    
    point1: координаты на изображении 1
    point2: координаты на изображении 2
    P1, P2: projection matrices
    """
    # Дженерированная матрица
    A = np.array([
        point1[0] * P1[2] - P1[0],
        point1[1] * P1[2] - P1[1],
        point2[0] * P2[2] - P2[0],
        point2[1] * P2[2] - P2[1]
    ])
    
    # SVD решение
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    return X[:3] / X[3]
```

## Pipeline фотограмметрии

### 1. Structure from Motion (SfM)

Восстановление 3D структуры и поз камер:

```python
class SfMPipeline:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def reconstruct(self, images):
        # 1. Детекция ключевых точек
        keypoints = [self.detect(img) for img in images]
        
        # 2. Матчинг
        matches = self.match_keypoints(keypoints)
        
        # 3. Восстановление поз
        poses = self.estimate_poses(matches)
        
        # 4. Триангуляция
        points_3d = self.triangulate_points(keypoints, poses)
        
        return points_3d, poses
```

### 2. Multi-View Stereo (MVS)

Плотная реконструкция на основе Sparse points:

```python
class MVSPipeline:
    def __init__(self, sparse_points, images):
        self.sparse = sparse_points
        self.images = images
        
    def dense_reconstruction(self):
        # 1. Depth map estimation
        depth_maps = self.estimate_depth_maps()
        
        # 2. Fusion
        fused_cloud = self.fuse_depth_maps(depth_maps)
        
        # 3. Meshing
        mesh = self.poisson_reconstruction(fused_cloud)
        
        return mesh
```

### 3. Текстурирование

Проецирование изображений на mesh:

```python
def texture_mesh(mesh, images, poses):
    """
    Текстурирование mesh'а с помощью проекции
    """
    # Создать UV coordinates
    uv_coords = compute_uv_mapping(mesh)
    
    # Для каждого треугольника
    for face in mesh.faces:
        # Найти лучшее изображение
        best_image = select_best_view(face, poses)
        
        # Спроецировать текстуру
        texture = project_face_to_image(
            face, uv_coords, best_image
        )
        
        mesh.visual.face_colors[face] = texture
    
    return mesh
```

## Инструменты фотограмметрии

### COLMAP

Open-source решение для SfM и MVS:

```bash
# Feature extraction
colmap feature_extractor \
    --database_path db.db \
    --image_path ./images

# Feature matching
colmap exhaustive_matcher \
    --database_path db.db

# Sparse reconstruction
colmap mapper \
    --database_path db.db \
    --image_path ./images \
    --output_path ./sparse

# Dense reconstruction
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/0 \
    --output_path ./dense

colmap patch_match_stereo \
    --workspace_path ./dense

colmap stereo_fusion \
    --workspace_path ./dense \
    --output_path ./dense/fused.ply
```

### OpenMVS

Коммерческое качество, открытый код:

```cpp
// Пример использования OpenMVS
#include <OpenMVS/OpenMVS.h>

int main() {
    // Загрузка sparse model
    Scene scene;
    scene.Load("scene.mvs");
    
    // Dense reconstruction
    scene.DenseReconstruction();
    
    // Texturing
    scene.TextureMesh();
    
    // Экспорт
    scene.Save("output.ply");
}
```

### RealityCapture

Коммерческое ПО с высоким качеством:
- Облачная и локальная версия
- Поддержка数千 изображений
- Автоматическая обработка
- Высокая скорость

## Фотограмметрия для лица

### Особенности

**Преимущества:**
- Высокая точность (sub-mm)
- Детализация (поры, морщины)
- Фотореалистичная текстура

**Ограничения:**
- Требуется 10–50 изображений
- Стабильное освещение
- Фиксированный expression
- Длительная обработка

### Требования к данным

| Параметр | Минимум | Рекомендуется |
|----------|---------|---------------|
| Количество фото | 10 | 30–50 |
| Разрешение | 2 Мп | 10+ Мп |
| Углы | ±45° | 360° |
| Освещение | Равномерное | Студийное |

### Пример датасета

```
face_scan/
├── images/
│   ├── img_001.jpg  # Frontal
│   ├── img_002.jpg  # 30° left
│   ├── img_003.jpg  # 30° right
│   ├── img_004.jpg  # 45° left
│   ├── img_005.jpg  # 45° right
│   ├── img_006.jpg  # Up
│   ├── img_007.jpg  # Down
│   └── ... (30-50 images total)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Сравнение с другими методами

| Критерий | Фотограмметрия | 3DMM | DECA | NeRF |
|----------|----------------|------|------|------|
| Кол-во фото | 10–50 | 1 | 1 | 50–100 |
| Точность | Высокая | Средняя | Высокая | Очень высокая |
| Скорость | Низкая | Высокая | Высокая | Низкая |
| Детализация | Очень высокая | Средняя | Высокая | Очень высокая |
| Текстура | Встроенная | Отдельная | Albedo | Встроенная |

## Применение

### Сфера

- **Археология** — сканирование артефактов
- **Архитектура** — 3D модели зданий
- **Кинематограф** — спецэффекты
- **Медицина** — планирование операций

### Face-specific

- **Протезирование** — создание протезов
- **Пластическая хирургия** — планирование операций
- **Идентификация** — 3D人脸识别
- **VR/AR** — создание аватаров

## Выводы

Фотограмметрия обеспечивает highest quality 3D-реконструкции, но требует:
- Множества изображений (10–50)
- Контролируемых условий
- Значительных вычислительных ресурсов

Для practical применения 3D-реконструкции лица по фотографиям со смартфона фотограмметрия не подходит из-за высоких требований к входным данным. Однако она остаётся эталоном качества для сравнения с другими методами.

---

*См. также: [01-overview.md](01-overview.md), [03-nerf.md](03-nerf.md)*
