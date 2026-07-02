# Проблема Domain Gap

## Введение

Domain gap (разрыв доменов) — проблема снижения качества модели при работе на данных, отличающихся от тренировочных. В контексте 3D-реконструкции лица это проявляется в виде артефактов, искажений и потери деталей при обработке реальных фотографий, не похожих на данные обучения.

## Типы Domain Gap

### 1. Освещение (Lighting Gap)

| Обучающий домен | Реальные входные фото | Что контролировать |
|-----------------|-----------------------|--------------------|
| Студийное освещение | Естественное освещение | нормализация яркости и white balance |
| Равномерный свет | Тени и блики | детекция пересветов, теней и specular highlights |
| Нейтральная температура | Теплые или холодные оттенки | цветовая коррекция перед реконструкцией |

**Проявления:**
- Искажение формы из-за теней
- Потеря деталей в пересветах
- Некорректная текстура

**Примеры:**
```python
# Проблемное освещение
problematic_lighting = [
    "side_lighting",      # Боковой свет
    "backlighting",       # Контражур
    "mixed_lighting",     # Смешанное освещение
    "harsh_shadows",      # Жёсткие тени
    "overexposure",       # Пересвет
    "underexposure",      # Недодержка
]
```

### 2. Поза (Pose Gap)

| Обучающий домен | Реальные входные фото | Что контролировать |
|-----------------|-----------------------|--------------------|
| Frontal, примерно ±15° | Profile и extreme angles | скоринг позы перед запуском реконструкции |
| Стабильная поза | Динамическая поза | отбор резких кадров и fallback для плохого ракурса |

**Проявления:**
- Искажение geometry при profile view
- Потеря деталей на occluded areas
- Некорректная реконструкция jaw/neck

### 3. Expression (Expression Gap)

| Обучающий домен | Реальные входные фото | Что контролировать |
|-----------------|-----------------------|--------------------|
| Нейтральное выражение | Эмоциональные выражения | оценка выражения и запрос нейтрального кадра |
| Ограниченный набор мимики | Спонтанная мимика | устойчивость identity shape к выражению |
| Синхронизированная съемка | Motion blur | проверка резкости в области глаз, рта и контура лица |

**Проявления:**
- Искажение параметров мимики
- Артефакты в mouth/eye areas
- Потеря деталей мимики

### 4. Качество изображения (Quality Gap)

| Обучающий домен | Реальные входные фото | Что контролировать |
|-----------------|-----------------------|--------------------|
| Высокое разрешение | Низкое разрешение | минимальный размер лица в кадре |
| Низкий шум | Высокий шум | denoise или отказ от плохого входа |
| Без артефактов | JPEG compression | оценка сжатия и предупреждение о качестве |

**Проявления:**
- Размытие текстуры
- Шум в деталях
- Артефакты сжатия

### 5. Аксессуары (Accessory Gap)

| Обучающий домен | Реальные входные фото | Что контролировать |
|-----------------|-----------------------|--------------------|
| Без аксессуаров | Очки, шляпы, серьги | сегментация аксессуаров и маскирование |
| Открытое лицо | Частичная occlusion | fallback-сценарий и явное предупреждение о качестве |

**Проявления:**
- Реконструкция geometry под очками
- Искажение формы с шляпой
- Потерра деталей в occluded areas

## Количественная оценка

### Метрики Domain Gap

```python
def measure_domain_gap(source_features, target_features):
    """
    Оценка разрыва между доменами
    """
    # Maximum Mean Discrepancy (MMD)
    mmd = compute_mmd(source_features, target_features)
    
    # Correlation Distance
    corr_dist = 1 - np.corrcoef(
        source_features.mean(axis=0),
        target_features.mean(axis=0)
    )[0, 1]
    
    # Feature distribution comparison
    kl_div = compute_kl_divergence(
        source_features, target_features
    )
    
    return {
        'mmd': mmd,
        'correlation_distance': corr_dist,
        'kl_divergence': kl_div
    }
```

### Влияние на качество

| Тип Gap | Снижение качества | Пример |
|---------|-------------------|--------|
| Освещение | 15–30% | Chamfer Distance +0.2 мм |
| Поза | 20–40% | NME +1.5% |
| Expression | 10–25% | SSIM -0.05 |
| Качество | 10–20% | PSNR -3 дБ |
| Аксессуары | 25–50% | Chamfer Distance +0.5 мм |

## Методы борьбы

### 1. Аугментация данных

```python
class FaceAugmentation:
    def __init__(self):
        self.transforms = A.Compose([
            # Освещение
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            
            # Шум и размытие
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            
            # Геометрические
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            
            # Специфичные для лица
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        ])
    
    def __call__(self, image):
        return self.transforms(image=image)['image']
```

### 2. Domain Adaptation

**Adversarial Domain Adaptation:**
```python
class DomainAdaptation(nn.Module):
    def __init__(self, feature_extractor, domain_classifier):
        self.feature_extractor = feature_extractor
        self.domain_classifier = domain_classifier
        
    def forward(self, source, target):
        # Features
        src_features = self.feature_extractor(source)
        tgt_features = self.feature_extractor(target)
        
        # Domain classification
        src_domain = self.domain_classifier(src_features)
        tgt_domain = self.domain_classifier(tgt_features)
        
        # Adversarial loss
        domain_loss = F.binary_cross_entropy_with_logits(
            src_domain, torch.zeros_like(src_domain)
        ) + F.binary_cross_entropy_with_logits(
            tgt_domain, torch.ones_like(tgt_domain)
        )
        
        return domain_loss
```

**Gradient Reversal Layer:**
```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

### 3. Test-Time Adaptation

```python
class TestTimeAdaptation:
    def __init__(self, model):
        self.model = model
        
    def adapt(self, image):
        # Self-supervised adaptation
        with torch.enable_grad():
            # Pseudo-labels
            with torch.no_grad():
                pseudo_label = self.model(image)
            
            # Adaptation step
            output = self.model(image)
            loss = self.compute_self_loss(output, pseudo_label)
            loss.backward()
            
            # Update only batch norm stats
            self.update_batch_norm()
        
        return output
```

### 4. Ensembling

```python
class DomainRobustEnsemble:
    def __init__(self, models):
        self.models = models  # Разные модели, обученные на разных доменах
        
    def predict(self, image):
        predictions = []
        for model in self.models:
            pred = model(image)
            predictions.append(pred)
        
        # Weighted average based on confidence
        weights = self.compute_confidence(predictions)
        return self.weighted_average(predictions, weights)
```

## Domain Gap в DECA/EMOCA

### Известные проблемы

**На контрольных фото (ideal conditions):**
- Chamfer Distance: 0.7–0.8 мм
- SSIM: 0.90+

**На реальных фото:**
- Chamfer Distance: 1.0–1.5 мм (+30–50%)
- SSIM: 0.80–0.85 (-10–15%)

### Примеры проблем

```python
# Проблемные кейсы
problematic_cases = [
    {
        'type': 'extreme_lighting',
        'example': 'side_lighting_with_shadows',
        'artifact': 'face_distortion_on_shadowed_side',
        'severity': 'high'
    },
    {
        'type': 'extreme_pose',
        'example': 'profile_view_90_degrees',
        'artifact': 'severe_geometry_distortion',
        'severity': 'very_high'
    },
    {
        'type': 'accessories',
        'example': 'thick_frame_glasses',
        'artifact': 'face_reconstructed_under_glasses',
        'severity': 'medium'
    },
    {
        'type': 'low_quality',
        'example': 'noisy_low_light_photo',
        'artifact': 'texture_noise_and_blur',
        'severity': 'medium'
    }
]
```

## Стратегии mitigation

### 1. Pre-processing

```python
def preprocess_for_reconstruction(image):
    """
    Предобработка для уменьшения domain gap
    """
    # 1. Face detection and alignment
    face = detect_and_align(image)
    
    # 2. Normalization
    face = normalize_lighting(face)
    
    # 3. Super-resolution (если нужно)
    if face.shape[0] < 224:
        face = super_resolve(face)
    
    # 4. Denoising
    face = denoise(face)
    
    return face
```

### 2. Post-processing

```python
def postprocess_reconstruction(output, original_image):
    """
    Постобработка для улучшения качества
    """
    # 1. Temporal consistency (для видео)
    output = smooth_temporal(output)
    
    # 2. Detail enhancement
    output = enhance_details(output, original_image)
    
    # 3. Texture refinement
    output = refine_texture(output, original_image)
    
    return output
```

### 3. Multi-model approach

```python
class MultiModelPipeline:
    def __init__(self):
        self.models = {
            'deca': DECAModel(),
            'emoca': EMOCAModel(),
            'mica': MICAModel()
        }
        
    def reconstruct(self, image):
        # Detect characteristics
        characteristics = self.analyze_image(image)
        
        # Select best model
        model_name = self.select_model(characteristics)
        
        # Run inference
        output = self.models[model_name](image)
        
        return output, model_name
```

## Выводы

Domain gap остаётся существенной проблемой для deep learning моделей 3D-реконструкции лица. Основные стратегии борьбы:

1. **Аугментация данных** — simplest approach, частично решает проблему
2. **Domain Adaptation** — требует additional training, эффективна
3. **Test-Time Adaptation** — адаптация в runtime, promising direction
4. **Multi-model approach** — выбор лучшей модели для каждого случая

Для практической реализации рекомендуется комбинировать:
- Аугментация при training
- Pre/post-processing при inference
- Multi-model selection для различных conditions

---

*См. также: [04-deep-learning.md](04-deep-learning.md), [07-comparison.md](07-comparison.md)*
