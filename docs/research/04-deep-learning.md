# Deep Learning based Reconstruction: DECA, EMOCA, MICA

## Введение

Deep learning подходы к 3D-реконструкции лица позволяют восстанавливать трёхмерную геометрию из одного или нескольких изображений с помощью нейросетевых моделей. Ключевые модели этого направления — DECA, EMOCA и MICA — обеспечивают сильный baseline при минимальных входных данных.

## DECA (Detailed Expression Capture and Animation)

### Архитектура

DECA использует encoder-decoder архитектуру с несколькими ветвями:

```
Input Image → [ResNet Encoder] → latent code
                                    ↓
                        [Shape Decoder] → 3DMM params + detail displacement
                        [Texture Decoder] → Albedo map
```

### Детали реализации

```python
class DECAModel(nn.Module):
    def __init__(self):
        # Encoder
        self.encoder = ResNet50(pretrained=True)
        
        # Decoders
        self.shape_decoder = FLAMEShapedecoder()
        self.expression_decoder = ExpressionDecoder()
        self.texture_decoder = TextureDecoder()
        self.detail_decoder = DetailDecoder()
        
    def forward(self, image):
        # Encode
        latent = self.encoder(image)
        
        # Decode shape
        shape_params = self.shape_decoder(latent)
        expr_params = self.expression_decoder(latent)
        
        # Decode texture
        texture = self.texture_decoder(latent)
        
        # Detail displacement
        detail = self.detail_decoder(latent)
        
        return {
            'shape': shape_params,
            'expression': expr_params,
            'texture': texture,
            'detail': detail
        }
```

### Выходные параметры

| Параметр | Размер | Описание |
|----------|--------|----------|
| Shape | 100 | Форма лица (3DMM) |
| Expression | 50 | Выражение лица |
| Pose | 6 | Поза (yaw, pitch, roll) |
| Texture | 192×160×3 | Карта альбедо |
| Detail | 1 | Displacement map |

### Качество реконструкции

**Оценка по публикациям и открытым демо:**
- хорошо восстанавливает общую форму лица, позу и выражение;
- добавляет детальную карту смещений для морщин и складок;
- качество сильно зависит от входного изображения и корректного face alignment;
- точные метрики нужно брать из конкретной статьи/benchmark, а не переносить между датасетами без оговорок.

**Оценка визуального качества:**
- Хорошо восстанавливает общую форму
- Детали (морщины, складки) частично восстанавливаются
- Текстура может содержать артефакты при extreme lighting

## EMOCA (Emotional Model Capture and Animation)

### Улучшения по сравнению с DECA

1. **Улучшенное восстановление мимики**
   - emotion-consistency loss
   - лучшая устойчивость на сильных выражениях лица

2. **Стабильная геометрия**
   - Регуляризация формы
   - Уменьшение артефактов

3. **Улучшенная текстура**
   - Более детальная карта альбедо
   - Уменьшение шума

### Архитектура

```python
class EMOCAModel(nn.Module):
    def __init__(self):
        self.encoder = EfficientNetEncoder()
        
        # Расширенные декодеры
        self.shape_decoder = ShapeDecoder(
            input_dim=512,
            output_dim=100,
            num_layers=8
        )
        self.expression_decoder = ExpressionDecoder(
            input_dim=512,
            output_dim=75,  # Увеличено с 50
            num_layers=6
        )
        self.texture_decoder = TextureDecoder(
            input_dim=512,
            output_dim=192*160*3
        )
        
        # Дополнительные головы
        self.detail_head = DetailHead()
        self.lmk_head = LandmarkHead()  # Для 2D landmarks
```

### Особенности

**Emotion-aware loss:**
```python
def emotion_loss(predicted, target):
    # Специальный loss для выражений лица
    # Учитывает семантику эмоций
    
    # AU (Action Units) based loss
    au_loss = F.mse_loss(predicted_aus, target_aus)
    
    # Geometric loss
    geom_loss = chamfer_distance(predicted_mesh, target_mesh)
    
    return alpha * au_loss + beta * geom_loss
```

**Multi-task learning:**
- Основная задача: 3D реконструкция
- Дополнительные: детекция landmarks, классификация выражения лица

## MICA (MetrIC fAce)

### Фокус на метрическую точность

MICA специализируется на:
- **Точной форме** — метрически корректные пропорции
- **Уменьшении искажений** — стабильная геометрия
- **Identity preservation** — сохранение индивидуальных особенностей

### Архитектура

```python
class MICAModel(nn.Module):
    def __init__(self):
        # Encoder с метрической калибровкой
        self.encoder = MetricEncoder()
        
        # Shape decoder с regularization
        self.shape_decoder = MetricShapeDecoder(
            num_coeffs=200,  # Увеличено для детализации
            regularization='procrustes'
        )
        
        # Identity-specific parameters
        self.identity_head = IdentityHead()
    
    def forward(self, image):
        # Encode с метрическими ограничениями
        features = self.encoder(image)
        
        # Shape с Procrustes alignment
        shape = self.shape_decoder(features)
        
        # Identity metrics
        identity = self.identity_head(features)
        
        return {
            'shape': shape,
            'identity': identity,
            'metrics': self.compute_metrics(shape)
        }
```

### Метрики качества

**Procrustes distance:**
```python
def procrustes_distance(source, target):
    # Выравнивание по Procrustes
    # Минимизирует расстояние после alignment
    
    aligned_source = procrustes_align(source, target)
    return torch.norm(aligned_source - target)
```

**Landmark consistency:**
- Расстояние до 68 facial landmarks
- Стабильность при изменении позы

## Сравнение моделей

| Модель | Вход | Shape | Expression | Texture | Скорость |
|--------|------|-------|------------|---------|----------|
| DECA | 1 фото | 100 params | 50 params | Albedo map | 15 мс |
| EMOCA | 1 фото | 100 params | 75 params | Albedo map | 18 мс |
| MICA | 1 фото | 200 params | — | — | 12 мс |

### Качество на benchmark

Публикации DECA, EMOCA и MICA используют разные акценты и наборы данных: DECA — детальную геометрию и анимацию, EMOCA — воспринимаемое качество эмоций, MICA — метрическую точность формы. Поэтому корректнее сравнивать их по роли в пайплайне, а не одной универсальной таблицей чисел.

## Использование в проекте

### Установка

```bash
# Clone DECA
git clone https://github.com/yfeng95/DECA.git
cd DECA

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt

# Download pre-trained model
bash models/download_models.sh
```

### Инференс

```python
from deca import DECA
from utils import load_image

# Load model
deca = DECA(device='cuda')

# Load image
image = load_image('photo.jpg')

# Reconstruct
output = deca.reconstruct(image)

# Get 3D mesh
vertices = output['vertices']  # (1, 5023, 3)
faces = output['faces']        # (1, 10004, 3)

# Export to OBJ
deca.save_obj('output.obj', vertices, faces)
```

### Экспорт для Three.js

```python
import trimesh
import numpy as np

def export_for_threejs(vertices, faces, texture, output_path):
    # Создать mesh
    mesh = trimesh.Trimesh(
        vertices=vertices[0].cpu().numpy(),
        faces=faces[0].cpu().numpy()
    )
    
    # Добавить texture
    if texture is not None:
        # Создать UV coordinates
        uv = create_uv_mapping(mesh)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            image=texture
        )
    
    # Экспорт в GLB (для Three.js)
    mesh.export(output_path, file_type='glb')
```

## Ограничения и проблемы

### Domain Gap

Модели обучены на контролируемых датасетах:
- **BIWI:** контролируемое освещение, ограниченные позы и выражения
- **MEAD:** Ограниченные углы, controlled conditions

На реальных фотографиях:
- Изменчивое освещение
- Экстремальные angles
- Аксессуары (очки, шляпы)

### Детализация

- Мелкие детали (поры, тонкие морщины) восстанавливаются нестабильно
- Текстура может быть размытой
- Глаза и рот требуют дополнительной обработки

### Стабильность

- Inter-frame consistency в видео
- Робастность к occlusion
- Устойчивость к изменению мимики

## Выводы

DECA, EMOCA и MICA представляют собой state-of-the-art в monocular 3D face reconstruction. Каждая модель имеет свои преимущества:

- **DECA** — сбалансированный выбор для general purpose
- **EMOCA** — лучший для emotion capture
- **MICA** — лучший для metric accuracy

Для данного проекта рекомендуется использовать **DECA** или **EMOCA** как основу, поскольку они обеспечивают оптимальный баланс между качеством, скоростью и простотой использования.

---

*См. также: [02-3dmm.md](02-3dmm.md), [06-domain-gap.md](06-domain-gap.md), [07-comparison.md](07-comparison.md)*
