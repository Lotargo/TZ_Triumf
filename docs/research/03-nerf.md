# NeRF и нейросетевые подходы

## Введение

Neural Radiance Fields (NeRF) — метод представления сцены в виде непрерывной функции, отображающей пространственные координаты и направление наблюдения в цвет и плотность. NeRF произвёл революцию в 3D-реконструкции, демонстрируя unprecedented качество восстановления сложных сцен по набору 2D-изображений.

## Основы NeRF

### Математическая модель

NeRF представляет собой функцию:

```
F: (x, y, z, θ, φ) → (r, g, b, σ)
```

Где:
- (x, y, z) — пространственные координаты
- (θ, φ) — направление наблюдения (viewing direction)
- (r, g, b) — цвет точки
- σ — плотность (opacity)

### Архитектура

```python
class NeRF(nn.Module):
    def __init__(self):
        # Position encoding
        self.pos_encoder = PositionalEncoder(input_dim=3, num_freqs=10)
        self.dir_encoder = PositionalEncoder(input_dim=3, num_freqs=4)
        
        # Main network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # ... 8 layers
            nn.Linear(256, 256)
        )
        
        # Output heads
        self.sigma_head = nn.Linear(256, 1)  # Density
        self.color_head = nn.Linear(256, 3)  # RGB
    
    def forward(self, pos, direction):
        pos_enc = self.pos_encoder(pos)
        dir_enc = self.dir_encoder(direction)
        
        h = self.layers(pos_enc)
        sigma = self.sigma_head(h)
        color = self.color_head(torch.cat([h, dir_enc], dim=-1))
        
        return color, sigma
```

### Volume Rendering

Рендеринг луча через сцену:

```python
def volume_rendering(rgb, sigma, t_values):
    # Дифференцируемый рендеринг
    delta = t_values[1:] - t_values[:-1]
    
    # Alpha compositing
    alpha = 1 - torch.exp(-sigma * delta)
    transmittance = torch.cumprod(1 - alpha + 1e-10, dim=0)
    
    weights = alpha * transmittance
    rgb_rendered = (weights[..., None] * rgb).sum(dim=0)
    
    return rgb_rendered
```

## NeRF для лица

### HoloFilies

Адаптация NeRF для портретов с контролируемым expression.

**Характеристики:**
- Вход: 200–500 изображений одного человека
- Выход:olumetric portrait с возможностью анимации
- Качество: photo-realistic rendering

**Ограничения:**
- Требует множество изображений
- Длительное обучение (часы)
- Ограниченная гибкость expression

### Variational Radiance Fields

Вариационный подход для faces:

```python
class VariationalNeRF(nn.Module):
    def __init__(self):
        self.encoder = FaceEncoder()
        self.decoder = NeRFDecoder()
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, images, directions):
        # Encode face
        features = self.encoder(images)
        
        # Latent space
        mu = self.mu(features)
        logvar = self.logvar(features)
        z = self.reparameterize(mu, logvar)
        
        # Decode NeRF
        rgb, sigma = self.decoder(z, directions)
        return rgb, sigma, mu, logvar
```

### Neural Head Avatars

Создание аватаров головы для видеозвонков:

**Подходы:**
1. **Neural Talking Heads** — генерация лица по аудио
2. **First Order Motion Model** — анимация по ключевым точкам
3. **GAN-based Avatars** — генеративные модели для лица

## Ускорение NeRF

### Методы ускорения

| Метод | Принцип | Ускорение |
|-------|---------|-----------|
| Instant-NGP | Multi-resolution hash encoding | 1000x |
| Plenoxels | Sparse voxel grids | 100x |
| TensoRF | Tensor decomposition | 50x |
| Mip-NeRF 360 | Антиалиасинг | 2x (качество) |

### Instant-NGP (NVIDIA)

```python
class InstantNGP(nn.Module):
    def __init__(self):
        # Multi-resolution hash encoding
        self.hash_encoding = HashEncoding(
            num_levels=16,
            features_per_level=2,
            log2_hashmap_size=19
        )
        
        # Small MLP
        self.mlp = nn.Sequential(
            nn.Linear(32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # RGB + sigma
        )
    
    def forward(self, pos, direction):
        # Hash encoding (очень быстрый)
        features = self.hash_encoding(pos)
        
        # MLP
        h = self.mlp(torch.cat([features, direction], dim=-1))
        rgb, sigma = h[..., :3], h[..., 3:]
        return rgb, sigma
```

## Ограничения NeRF для лица

### Требования к данным

- **Количество изображений:** 50–500 (vs 1 для DECA)
- **Качество:** Высокое разрешение, стабильное освещение
- **Разнообразие:** Множество углов и expressions

### Вычислительная стоимость

- **Обучение:** Часы на GPU (vs миллисекунды для DECA)
- **Инференс:** Секунды на изображение (vs миллисекунды)
- **Память:** Гигабайты для хранения модели

### Ограничения по expression

- Сложность в анимации выражений лица
- Требуется специальный датасет для expression control
- Interpolation между expressions не всегда корректна

## Сравнение с 3DMM подходами

| Аспект | NeRF | 3DMM + DECA |
|--------|------|-------------|
| Входные данные | 50–500 фото | 1 фото |
| Качество | Photo-realistic | Высокое |
| Скорость обучения | Часы | Не требуется |
| Скорость инференса | Секунды | Миллисекунды |
| Анимация | Сложная | Простая |
| Текстура | Встроенная | Отдельная |

## Современные developments

### Gaussian Splatting

Новый подход, заменяющий volumetric rendering:

```python
class GaussianSplatting(nn.Module):
    def __init__(self):
        # Gaussians: position, covariance, color, opacity
        self.means = nn.Parameter(torch.randn(num_gaussians, 3))
        self.covariances = nn.Parameter(torch.randn(num_gaussians, 6))
        self.colors = nn.Parameter(torch.randn(num_gaussians, 3))
        self.opacities = nn.Parameter(torch.randn(num_gaussians, 1))
    
    def forward(self, camera):
        # Splatting Gaussians to image
        # Much faster than volume rendering
        pass
```

**Преимущества:**
- Real-time rendering
- Легче обучение
- Better for dynamic scenes

### Diffusion Models for 3D

Генеративные модели для 3D-реконструкции:

- **Zero-1-to-3** — генерация нового вида из одного изображения
- **Magic3D** — генерация 3D из текста
- **DreamFusion** — SDS loss для 3D generation

## Выводы

NeRF обеспечивает unprecedented качество 3D-реконструкции, но требует значительных вычислительных ресурсов и множество изображений. Для practical применения 3D-реконструкции лица по фотографиям со смартфона NeRF не подходит из-за высоких требований к входным данным.

Однако NeRF остаётся перспективным направлением для:
- Высококачественного сканирования
- Создания photorealistic аватаров
- Видео-конференций нового поколения

---

*См. также: [01-overview.md](01-overview.md), [04-deep-learning.md](04-deep-learning.md)*
