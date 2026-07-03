# Multi-view reconstruction notes

## Current finding

DECA is a single-image regressor. Running it on front/left/right images gives
three independent FLAME parameter estimates, not one jointly optimized mesh.
A simple average of predicted shape codes can stabilize identity a little, but
it is not true multi-view fitting because cameras, visibility, masks, lighting,
and photo consistency are not solved jointly.

The current browser `DECA result` is a coarse FLAME-topology mesh:

- 5,023 vertices
- 9,975 faces after removing one degenerate face
- front-view UV texture
- global pose reset to zero for stable preview
- UV texture extracted in the original front-view pose, then applied to the
  neutral/global-zero preview mesh
- `DECA baked detail` keeps the same 5,023-vertex mesh but adds normal and bump
  maps baked from the DECA detail pass.

The official DECA detail path upsamples the mesh from the displacement map:

- 59,315 vertices
- 117,380 faces
- exported in the landing demo without vertex colors or the failed mask layer
- exported locally as `outputs/gpt_three_view_detail.glb`
- copied to `site/models/deca_detail_neutral.glb` for browser comparison

This means better source photos alone cannot add much visible geometry to the
coarse GLB. Higher visual smoothness needs either the official detail mesh, a
subdivision/displacement export path, or a denser target model.

FLAME2023 is also shown in coarse/detail variants:

- coarse FLAME: 5,023 vertices, 9,976 faces;
- FLAME mask baseline: 5,023 vertices, 9,976 faces, with the FLAME mean texture
  baked into vertex colors for stable browser display;
- detail FLAME: 59,856 vertices, 39,904 faces;
- the FLAME detail meshes use midpoint subdivision with preserved UVs for the
  textured variant;
- unlike DECA detail, FLAME detail has no learned displacement map, so it is a
  smoother topology baseline rather than recovered facial microgeometry.
- `FLAME_texture.npz` stores its mean texture in BGR channel order; the export
  pipeline converts it to RGB before writing GLB.
- The FLAME texture space is useful as an out-of-the-box head/face UV mask
  baseline, even though the mean texture is not personalized.

The visual tests also showed an important presentation trade-off: more
microdetail does not automatically mean better perceived quality. Around 5k-10k
vertices, texture and geometry errors are visually smoothed. Dense detail meshes
make real displacement visible, but they also expose every texture seam,
projection mismatch, and generative artifact. The landing demo should therefore
keep both coarse and detail variants.

The most useful production-oriented compromise is:

```text
coarse mesh + albedo/baseColor + baked normal map + optional height/bump map
```

This keeps the model light enough for browser/real-time use while allowing the
detail branch to contribute shading-level microgeometry. It also avoids the
main failure mode of dense vertex-color exports: the dense mesh reveals every
projection and texture artifact. The first spike exports:

- `site/models/deca_baked_detail.glb` — same coarse DECA geometry as
  `DECA result`;
- `site/models/deca_baked_normal.png` — UV normal map from DECA detail normals;
- `site/models/deca_baked_height.png` — normalized DECA displacement map used as
  a subtle bump map in Three.js.

It is not meaningful to substantially increase triangle count while keeping the
FLAME mesh at 5k-6k vertices. A clean triangular head mesh already has roughly
two faces per vertex, so the coarse FLAME topology is close to that practical
limit. To improve the 5k baseline visually, bake the FLAME texture-space mask
into vertex colors. To increase faces materially, add vertices through
subdivision or a displacement/detail layer.

## Direction decision

The experimental UV-fusion branch should not be developed further as the main
solution. It was useful as a diagnostic step: it proved that denser DECA detail
geometry improves surface smoothness, but also showed that texture/mask quality
is bottlenecked by missing semantic segmentation and true multi-view fitting.

The browser demo should therefore keep:

- `DECA result` for the coarse FLAME topology baseline;
- `DECA detail` for dense DECA displacement geometry;
- `FLAME textured`, `FLAME mask`, and `FLAME neutral` for the original
  FLAME2023 baseline;
- `FLAME detail textured` and `FLAME detail` for subdivided FLAME2023 topology.

The failed UV-fusion assets are intentionally removed from the landing demo so
the page does not imply that visibility-weighted UV blending is the chosen
production path.

The later `DECA + FLAME mask` spike is also intentionally removed. It baked a
FLAME texture-space mask directly into DECA vertex colors and exposed the core
problem: FLAME and DECA preview geometry can share a broad head prior, but their
texture/UV assumptions are not interchangeable without an explicit UV remap.
The result looked like a spatially desynchronized mask rather than a clean face
texture. Increasing or decreasing polygon count does not fix that mismatch; it
only changes how visible the interpolation artifacts are.

## Texture and mask caveat

DECA extracts `uv_texture_gt` by projecting the currently decoded mesh back into
the source image. If global pose is reset before this extraction, the UV texture
is sampled from the wrong image locations. The preview pipeline must therefore
use two decode passes:

1. Texture pass: original front-view pose/camera for `uv_texture_gt`.
2. Geometry pass: global-zero pose for the browser-facing neutral mesh.

Even after this fix, the dense vertex-color texture is not a production-quality
mask. DECA's built-in `uv_face_eye_mask` is a UV face/eye prior, not semantic
segmentation of skin, hair, ears, neck, background, and occluders. A production
texture pass needs a separate segmentation/visibility pipeline before fusing
colors into UV space.

The project direction is therefore 2D/UV-first for masks:

```text
semantic image masks -> visibility-aware projection -> target UV texture
                    -> material maps on the chosen mesh
```

If a mask is authored in FLAME UV space and the target mesh uses a DECA/export
UV layout, the required operation is not vertex-color transfer. It is a
texture-space bake/remap into the target UV layout, ideally with visibility,
occlusion and seam handling.

The first runtime UV-fusion experiment followed DECA's own training hint:

```text
uv_visibility = self_occlusion_mask * uv_face_eye_mask
fused_uv = sum(view_uv_texture * uv_visibility * angle_weight)
           / sum(uv_visibility * angle_weight)
```

On the GPT three-view sample this produced about 42% UV coverage under the DECA
face/eye prior. That is expected: the current mask only covers DECA's face UV
area, not a full semantic head/skin/hair/neck mask. It improves side coverage
compared with front-only projection, but it cannot solve full 3D masking by
itself and is not the direction to keep extending.

The next useful spike should be:

1. Run face/head segmentation on each input image.
2. Use BiSeNet-style face parsing for semantic face parts.
3. Use SAM2-style segmentation for head/hair/neck/silhouette prompts.
4. Project these semantic masks into UV or onto the mesh.
5. Only then revisit texture baking and multi-view mask fusion.

For multi-view geometry, use learned geometry/matching tools such as VGGT or
MASt3R as input to fitting, not as a direct replacement for the FLAME topology.

## Correct programmatic multi-view path

For three smartphone projections, the right formulation is joint optimization:

1. Detect/crop face and landmarks for each image.
2. Initialize each view with DECA/MICA/EMOCA.
3. Share one identity shape vector across all views.
4. Keep per-view pose, camera, expression, lighting, and visibility masks.
5. Optimize a combined loss:
   - 2D landmark reprojection per view
   - photometric loss on visible skin/face pixels
   - silhouette or face-mask consistency
   - regularization for shape, expression, jaw pose, and lighting
   - optional identity consistency loss
6. Fuse texture in UV space using per-view visibility and view angle weights.
7. Export both coarse FLAME mesh and dense/detail mesh.

Minimal useful implementation for this project:

```text
images[3]
  -> DECA encode per image
  -> initialize shared_shape = robust mean(shape_i)
  -> optimize shared_shape + per_view(cam, pose, exp, light)
  -> decode shared mesh in neutral/global-zero pose
  -> build UV texture from all views with visibility masks
  -> export coarse GLB and detail GLB
```

## Why manual mesh stitching is the wrong target

All views already decode into the same FLAME topology. We do not need to stitch
three meshes geometrically. We need to solve the parameter disagreement between
views and then decode one shared parameter set. Stitching would create seams and
double surfaces; parameter fitting preserves clean topology.

## Source anchors

- DECA official README describes reconstruction from a single input image and
  notes that the demo can save coarse/detailed geometry and extracted texture.
- DECA `save_obj()` exports coarse OBJ plus `_detail.obj` using displacement
  upsampling.
- FLAME fitting examples show the expected fitting pattern: optimize FLAME
  parameters to landmarks/scans rather than merging unrelated meshes.
- MICA is useful for metric identity shape initialization because it outputs
  FLAME parameters under a common topology.
- EMOCA is also monocular, but can be a better initializer when expression
  capture matters.
