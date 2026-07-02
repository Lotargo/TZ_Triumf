# Архитектура решения

## Обзор

Данное решение реализует пайплайн 3D-реконструкции лица по фотографии с использованием предобученной нейросетевой модели DECA и визуализацией результата в браузере через Three.js.

## Системная архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                          │
│                      (HTML/CSS/JS + Three.js)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                         │
│                   POST /api/reconstruct                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Reconstruction Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Preprocessing│→ │    DECA      │→ │Postprocessing│         │
│  │  (OpenCV)    │  │  (PyTorch)   │  │  (NumPy)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Formats                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │   OBJ   │  │   GLB   │  │   PLY   │  │   3D    │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Модули

### 1. Reconstruction Module (`src/reconstruction/`)

Отвечает за 3D-реконструкцию лица из изображения.

```python
# src/reconstruction/__init__.py
from .face_reconstructor import FaceReconstructor
from .preprocessor import FacePreprocessor
from .postprocessor import MeshPostprocessor

__all__ = [
    'FaceReconstructor',
    'FacePreprocessor', 
    'MeshPostprocessor'
]
```

**Основной класс:**
```python
class FaceReconstructor:
    """
    Основной класс для 3D-реконструкции лица
    
    Использует DECA модель для предсказания
    3D-геометрии из одного изображения
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model()
        self.preprocessor = FacePreprocessor()
        self.postprocessor = MeshPostprocessor()
    
    def reconstruct(self, image_path: str) -> ReconstructionResult:
        """
        Реконструкция 3D-модели лица
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            ReconstructionResult с 3D-мешем и метаданными
        """
        # 1. Загрузка и предобработка
        image = self.preprocessor.process(image_path)
        
        # 2. Инференс модели
        with torch.no_grad():
            output = self.model(image)
        
        # 3. Постобработка
        result = self.postprocessor.process(output)
        
        return result
```

### 2. Preprocessing Module (`src/reconstruction/preprocessor`)

Предобработка изображения перед инференсом.

```python
class FacePreprocessor:
    """
    Предобработка изображений для 3D-реконструкции
    
    Включает:
    - Детекцию лица
    - Выравнивание
    - Нормализацию освещения
    - Изменение размера
    """
    
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.target_size = (224, 224)
    
    def process(self, image_path: str) -> torch.Tensor:
        """
        Предобработка изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Тензор [1, 3, 224, 224]
        """
        # Загрузка
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция лица
        face = self._detect_face(image)
        
        # Выравнивание
        aligned = self._align_face(face)
        
        # Нормализация
        normalized = self._normalize(aligned)
        
        # Конвертация в тензор
        tensor = self._to_tensor(normalized)
        
        return tensor
```

### 3. Postprocessing Module (`src/reconstruction/postprocessor`)

Постобработка результатов модели.

```python
class MeshPostprocessor:
    """
    Постобработка 3D-меша
    
    Включает:
    - Сглаживание
    - Оптимизацию топологии
    - Экспорт в различные форматы
    """
    
    def process(self, model_output: dict) -> ReconstructionResult:
        """
        Постобработка результатов
        
        Args:
            model_output: Выход модели DECA
            
        Returns:
            ReconstructionResult
        """
        vertices = model_output['vertices']
        faces = model_output['faces']
        
        # Сглаживание
        vertices = self._smooth(vertices)
        
        # Оптимизация
        vertices, faces = self._optimize(vertices, faces)
        
        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            texture=model_output.get('texture')
        )
```

### 4. Visualization Module (`src/visualization/`)

Three.js визуализация в браузере.

```javascript
// src/visualization/face_viewer.js

class FaceViewer {
    /**
     * 3D визуализация лица
     */
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        this.init();
    }
    
    init() {
        // Настройка рендерера
        this.renderer.setSize(500, 500);
        this.container.appendChild(this.renderer.domElement);
        
        // Освещение
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Камера
        this.camera.position.z = 2;
    }
    
    loadModel(glbUrl) {
        /**
         * Загрузка 3D-модели
         */
        const loader = new THREE.GLTFLoader();
        
        loader.load(glbUrl, (gltf) => {
            this.model = gltf.scene;
            this.scene.add(this.model);
            
            this.animate();
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.model) {
            this.model.rotation.y += 0.01;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
}
```

## Данные

### ReconstructionResult

```python
@dataclass
class ReconstructionResult:
    """
    Результат 3D-реконструкции
    """
    vertices: np.ndarray      # (N, 3) - 3D вершины
    faces: np.ndarray         # (M, 3) - треугольники
    texture: Optional[np.ndarray] = None  # Текстура
    landmarks: Optional[np.ndarray] = None  # 2D landmarks
    
    def to_obj(self, path: str):
        """Экспорт в OBJ формат"""
        with open(path, 'w') as f:
            for v in self.vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in self.faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    
    def to_glb(self, path: str):
        """Экспорт в GLB формат (для Three.js)"""
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces
        )
        mesh.export(path, file_type='glb')
```

## API

### FastAPI Server

```python
# src/api/server.py

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from reconstruction import FaceReconstructor

app = FastAPI(title="3D Face Reconstruction API")

# CORS для Three.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reconstructor = FaceReconstructor()

@app.post("/api/reconstruct")
async def reconstruct_face(image: UploadFile):
    """
    Реконструкция 3D-модели лица
    
    Принимает: изображение (JPEG/PNG)
    Возвращает: GLB файл с 3D-моделью
    """
    # Сохранение временного файла
    temp_path = f"temp/{image.filename}"
    with open(temp_path, "wb") as f:
        f.write(await image.read())
    
    # Реконструкция
    result = reconstructor.reconstruct(temp_path)
    
    # Экспорт в GLB
    output_path = f"output/{image.filename}.glb"
    result.to_glb(output_path)
    
    return {"model_url": f"/output/{image.filename}.glb"}
```

## Потоки данных

### Основной пайплайн

```
┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌─────────┐
│  Image  │ →  │ Preprocess  │ →  │  DECA   │ →  │ Postprocess │
│ (JPEG)  │    │ (OpenCV)    │    │(PyTorch)│    │ (NumPy) │
└─────────┘    └─────────────┘    └─────────┘    └─────────┘
                                                      │
                                                      ▼
┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌─────────┐
│ Browser │ ←  │  Three.js   │ ←  │  GLB    │ ←  │ Export  │
│         │    │ Visualization│    │  File   │    │         │
└─────────┘    └─────────────┘    └─────────┘    └─────────┘
```

## Производительность

### Ожидаемые метрики

| Этап | Время | Память |
|------|-------|--------|
| Предобработка | 10–20 мс | 50 МБ |
| Инференс DECA | 15–50 мс | 500 МБ |
| Постобработка | 5–10 мс | 100 МБ |
| Экспорт | 5–15 мс | 50 МБ |
| **Итого** | **35–95 мс** | **~700 МБ** |

### Оптимизации

1. **Кеширование модели** — загрузка при старте сервера
2. **Батчевая обработка** — несколько изображений за раз
3. **GPU inference** — использование CUDA для ускорения
4. **Ленивая загрузка** — загрузка модулей по требованию

## Требования

### Программное обеспечение

- Python 3.10+
- PyTorch 2.0+
- OpenCV 4.8+
- Node.js 18+ (для сборки Three.js)

### Аппаратное обеспечение

**Минимальные:**
- CPU: 4 ядра
- RAM: 8 ГБ
- GPU: Не требуется (будет медленнее)

**Рекомендуемые:**
- CPU: 8 ядер
- RAM: 16 ГБ
- GPU: NVIDIA с поддержкой CUDA 11.8+

---

*См. также: [setup.md](setup.md), [04-deep-learning.md](../research/04-deep-learning.md)*
