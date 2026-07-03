# Игровой пайплайн текстур и масок лица персонажа на open-source / mixed-source решениях

## 1. Главная идея

В игровой индустрии лицо персонажа обычно не строится вокруг одной «текстуры лица» или одной модели, которая делает всё автоматически. Рабочий подход другой: создаётся набор карт материала и масок, которые вместе формируют реалистичную кожу, мимику, блеск, микрорельеф, поры, морщины, щетину, губы, веки и зоны subsurface scattering.

Движки вроде Unreal Engine, Unity и Godot в основном не являются инструментами генерации таких карт. Они выступают как runtime/preview/target-платформа, где проверяется качество материала и шейдера. Основная подготовка делается в DCC и texture-authoring инструментах: Blender, ArmorPaint, Substance Painter, Krita, GIMP, Photoshop/Photopea, Material Maker, ComfyUI и специализированных 3D/AI-пайплайнах.

Ключевой вывод:

```text
Движок показывает, какие карты и маски нужны.
DCC/painting/AI-инструменты создают эти карты.
Шейдер в движке превращает набор карт в реалистичную кожу.
```

---

## 2. Какие карты обычно нужны для лица персонажа

Для игрового персонажа лицо собирается из нескольких карт:

```text
BaseColor / Albedo        - цвет кожи без запечённого света
Normal Map                - крупный рельеф лица
Detail Normal / Micro     - поры, мелкий рельеф, микроморщины
Roughness                 - жирность/матовость кожи
Specular                  - сила и характер блика
AO / Cavity               - углубления, складки, поры, зоны затемнения
SSS / Thickness / Scatter - зоны подповерхностного рассеивания света
Wrinkle Maps              - морщины, активируемые мимикой
Mask / ID Maps            - зоны лица и материалов
Opacity                   - ресницы, волосы, борода, hair cards
```

Важно: в играх детальность лица редко держат только геометрией. Обычно меш остаётся относительно чистым, а реализм вытаскивают через material system:

```text
Base mesh        - форма лица
Normal map       - крупные формы
Detail normal    - поры и микрорельеф
Wrinkle normal   - мимические морщины
Cavity map       - складки, поры, углубления
Roughness map    - жирность кожи
SSS mask         - уши, нос, губы, щеки
ID masks         - кожа, губы, веки, щетина, брови, макияж
```

---

## 3. Маски, которые особенно полезны для лица

Для production-подхода к лицу желательно иметь не одну общую маску лица, а набор зон:

```text
skin
lips
eyelids
eyes
eyebrows
nose area
ears
cheeks
forehead
neck
hairline
beard / stubble
freckles / moles
scars
makeup
wrinkle zones
oily zones
SSS zones
```

Эти маски можно использовать для:

- раздельного inpaint/refine через ComfyUI;
- projection painting в Blender;
- PBR-покраски в ArmorPaint/Substance;
- настройки roughness/specular/SSS;
- blending detail normals;
- wrinkle maps;
- channel packing для UE/Unity/Godot;
- финальной оптимизации материала в движке.

---

## 4. Unreal Engine как ориентир

Unreal Engine полезен не как open-source генератор лицевых текстур, а как сильный ориентир для архитектуры материала.

Что можно подсмотреть у UE/MetaHuman:

```text
- skin material instances
- Subsurface Profile
- wrinkle texture sets
- roughness variation
- freckles / moles / stubble masks
- skin tone controls
- separate skin zones
- channel-packed masks
- groom / hair отдельно от кожи
```

Unreal активно использует texture masks и channel packing. Маска может быть grayscale-текстурой или отдельным каналом R/G/B/A внутри одной packed-текстуры. Это экономит texture samples и позволяет одной текстурой управлять несколькими эффектами.

Пример UE-подхода:

```text
RGBA Mask Texture:
R - cavity / redness / region mask
G - roughness modifier
B - stubble / freckles / detail mask
A - SSS / opacity / custom mask
```

Для кожи важен Subsurface Profile. Он управляет тем, как свет рассеивается внутри кожи. Без SSS реалистичное лицо часто выглядит пластиковым.

Вывод по Unreal:

```text
Использовать как:
- эталон skin shader
- проверку материала в real-time
- Subsurface Profile
- material instances
- channel-packed masks
- MetaHuman-like структуру карт

Не использовать как:
- open-source генератор текстур
- замену Blender/ArmorPaint/ComfyUI
```

---

## 5. Unity HDRP как target-формат

Unity HDRP тоже работает через набор PBR-карт и packed masks. В HDRP есть Mask Map, где разные данные упакованы по каналам:

```text
Unity HDRP Mask Map:
R - Metallic
G - Ambient Occlusion
B - Detail Mask
A - Smoothness
```

Для органических материалов Unity HDRP поддерживает Subsurface Scattering через Diffusion Profiles. Это помогает коже выглядеть менее пластиковой и лучше передавать мягкое рассеивание света.

Вывод по Unity:

```text
Использовать как:
- target для HDRP MaskMap / DetailMap
- проверку PBR-текстур
- SSS через Diffusion Profile
- real-time preview

Не использовать как:
- основной authoring tool для лицевых текстур
```

---

## 6. Godot как open-source runtime

Godot интересен тем, что сам движок open-source. Но готового аналога MetaHuman или специализированного production digital human pipeline для генерации лицевых текстур в Godot нет.

Godot можно использовать как:

```text
- open-source runtime target
- площадку для кастомного skin shader
- тестовую среду для PBR-материалов
- связку с Material Maker
- lightweight preview для персонажей
```

Godot поддерживает PBR-материалы, ORM/packed maps и кастомные шейдеры. Для серьёзного лица придётся самостоятельно собирать skin shader и material pipeline.

Вывод по Godot:

```text
Использовать как:
- open-source runtime
- кастомный shader target
- тестовую площадку
- связку с Material Maker

Не использовать как:
- готовый digital human authoring stack
```

---

## 7. Open-source / near-open-source инструменты для создания карт

### Blender

Blender остаётся главным open-source центром для 3D-подготовки.

Что закрывает Blender:

```text
- UV unwrap
- projection painting
- texture paint
- baking high-poly → low-poly
- normal / AO / cavity / displacement bake
- vertex groups
- material slots
- mesh cleanup
- retopo
- shape keys
- facial rig
```

Роль Blender в пайплайне:

```text
DECA/FLAME или mesh из ComfyUI
→ clean UV
→ projection paint front/profile
→ bake normal/AO/cavity
→ retopo/cleanup
→ export в UE/Unity/Godot
```

### ArmorPaint

ArmorPaint — близкая open-source-ish альтернатива Substance Painter для PBR texture painting.

Полезен для:

```text
- skin roughness map
- pores / cavity painting
- lips / eyelids masks
- stubble / beard mask
- scars / freckles / moles layers
- material zones
- PBR preview
- painting прямо по 3D-модели
```

Важный нюанс: исходники доступны, но готовые сборки могут быть платными для поддержки проекта.

### Material Maker

Material Maker — open-source procedural material authoring tool, построенный на Godot.

Полезен для генерации:

```text
- pore/noise masks
- procedural roughness variation
- skin micro detail
- freckles/moles distribution
- subtle color variation
- generic material masks
```

Он не заменяет ручную покраску лица, но может быть полезен как генератор процедурных слоёв.

### Krita / GIMP / Photoshop / Photopea

Эти инструменты нужны для ручной и полуручной подготовки масок:

```text
- clean alpha
- face region masks
- hairline cleanup
- skin/lips/eyes masks
- freckles/scars/makeup masks
- ручная коррекция AI-сегментации
- подготовка grayscale maps
- редактирование UV texture sheets
```

Photoshop исторически часто использовался в 3D-пайплайнах не как mesh tool, а как инструмент подготовки texture maps и masks. Сегодня его open-source/бесплатные альтернативы — Krita, GIMP и Photopea.

---

## 8. Связка с ComfyUI

ComfyUI полезен как генератор черновиков и вспомогательных карт:

```text
ComfyUI:
- face cleanup
- segmentation
- SAM / BiSeNet masks
- background removal
- depth maps
- normal maps
- texture inpaint
- missing UV areas
- front/profile generation
- image-to-3D / multi-view-to-3D
```

Практическая схема:

```text
ComfyUI генерирует черновую маску / depth / normal / texture
→ Krita/GIMP/Photoshop/Photopea чистят маски вручную
→ Blender/ArmorPaint используют чистые карты на меше
→ UE/Unity/Godot проверяют skin shader в real-time
```

Важно: ручная чистка масок не является устаревшим костылём. Это production-контроль. Нейросети дают скорость, но чистота силуэта, волос, ушей, губ и век часто требует ручной коррекции.

---

## 9. Рекомендуемый пайплайн для лица

```text
1. Base mesh:
   DECA / FLAME / Pixel3DMM / Hunyuan3D / InstantMesh / SAM 3D Body

2. UV:
   Blender unwrap или canonical UV, если используется стабильная топология

3. Base texture:
   projection painting из front/profile images
   ComfyUI для inpaint missing zones
   ручная чистка в Krita/GIMP/Photoshop/Photopea

4. Bake maps:
   normal
   AO
   cavity
   curvature
   displacement / height

5. Paint material masks:
   skin
   lips
   eyelids
   nose
   ears
   eyebrows
   beard/stubble
   freckles/moles
   oily zones
   wrinkle zones
   SSS zones

6. Pack masks:
   UE-style RGBA mask texture
   Unity HDRP MaskMap
   Godot ORM/material masks

7. Shader:
   SSS / thickness
   detail normal
   wrinkle normal blending
   roughness variation
   cavity darkening
   optional redness/blood/skin tone masks

8. Export:
   GLB / FBX / OBJ + packed textures
```

---

## 10. Рекомендуемая связка инструментов

```text
ComfyUI:
- segmentation
- depth / normal
- texture inpaint
- front/profile generation
- image-to-3D helpers

DECA / FLAME:
- canonical face mesh
- stable topology
- pose / expression layer

Blender:
- UV
- projection painting
- baking
- cleanup
- retopo
- export

ArmorPaint:
- PBR painting
- roughness / specular / SSS masks
- skin details
- material layers

Material Maker:
- procedural pores
- noise
- freckles
- roughness variation

Krita / GIMP / Photoshop / Photopea:
- manual mask cleanup
- texture sheet edits
- alpha masks
- grayscale maps

UE / Unity / Godot:
- real-time shader preview
- packed masks
- SSS/material tuning
- final runtime validation
```

---

## 11. Итоговый вывод

Для игрового качества лица нужно мыслить не в терминах «получить одну текстуру», а в терминах **skin material system**.

Главная формула пайплайна:

```text
Base mesh + UV
→ BaseColor
→ Normal + DetailNormal
→ Roughness / Specular
→ AO / Cavity
→ SSS / Thickness
→ Wrinkle maps
→ ID / Region masks
→ Real-time skin shader
```

Движки вроде Unreal, Unity и Godot не заменяют создание масок и текстур, но отлично показывают, какие именно карты должны быть подготовлены. Для open-source/mixed-source пайплайна наиболее практичная связка выглядит так:

```text
ComfyUI = генерация и вспомогательные карты
DECA/FLAME = стабильная лицевая геометрия
Blender = UV, bake, retopo, projection painting
ArmorPaint = PBR painting
Material Maker = процедурные skin details
Krita/GIMP/Photopea = ручная чистка масок
UE/Unity/Godot = проверка и настройка финального skin shader
```

Главный практический вывод: **реализм лица в играх создаётся не только мешем, а системой карт и масок.** Поэтому для твоего пайплайна стоит не пытаться заменить всё одной 3D-моделью, а собрать управляемый набор карт, который потом можно применять в UE/Unity/Godot или в собственном renderer/export pipeline.
