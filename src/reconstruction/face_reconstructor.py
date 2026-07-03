"""
Face Reconstruction Module.

Main class for 3D face reconstruction from a single image.
Uses DECA (Detailed Expression Capture and Animation) model with FLAME2020.

The module also supports FLAME 2023 models (flame2023.pkl, flame2023_Open.pkl)
for parametric mesh generation via Flame2023Decoder.
"""

from pathlib import Path
from importlib.util import find_spec
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None


def _install_legacy_chumpy_shims() -> None:
    """Provide Python/NumPy aliases expected by legacy chumpy code."""
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


class ReconstructionResult:
    """
    Result of 3D face reconstruction.
    
    Attributes:
        vertices: 3D vertex coordinates (N, 3)
        faces: Triangle face indices (M, 3)
        texture: Optional texture map
        landmarks: Optional 2D facial landmarks
        params: Model parameters (shape, expression, pose)
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        texture: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        uv_faces: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        normals: Optional[np.ndarray] = None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.texture = texture
        self.uv = uv
        self.uv_faces = uv_faces
        self.landmarks = landmarks
        self.params = params or {}
        self.normals = normals
    
    def to_obj(self, path: Union[str, Path]) -> None:
        """
        Export mesh to OBJ format.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            has_uv = self.texture is not None and self.uv is not None

            if has_uv:
                for vt in self.uv:
                    f.write(f"vt {vt[0]:.6f} {1.0 - vt[1]:.6f}\n")
            
            # Write faces (1-indexed)
            for idx, face in enumerate(self.faces):
                if has_uv:
                    uv_face = (
                        self.uv_faces[idx]
                        if self.uv_faces is not None
                        else face
                    )
                    f.write(
                        f"f {face[0] + 1}/{uv_face[0] + 1} "
                        f"{face[1] + 1}/{uv_face[1] + 1} "
                        f"{face[2] + 1}/{uv_face[2] + 1}\n"
                    )
                else:
                    f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    
    def to_glb(self, path: Union[str, Path]) -> None:
        """
        Export mesh to GLB format (for Three.js).
        
        Args:
            path: Output file path
        """
        import trimesh
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        vertices = self.vertices
        faces = self.faces
        uv = self.uv
        
        if self.texture is not None:
            if uv is None:
                uv = self._create_uv_mapping(vertices)
            elif self.uv_faces is not None:
                vertices, faces, uv = self._expand_mesh_for_uv()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if self.texture is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                image=Image.fromarray(self.texture),
            )
        
        mesh.export(str(path), file_type="glb")
    
    def to_ply(self, path: Union[str, Path]) -> None:
        """
        Export mesh to PLY format.
        
        Args:
            path: Output file path
        """
        import trimesh
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
        )
        
        mesh.export(str(path), file_type="ply")
    
    def _create_uv_mapping(self, vertices: np.ndarray) -> np.ndarray:
        """Create simple UV mapping for texture."""
        uv = np.zeros((len(vertices), 2))
        x_range = vertices[:, 0].max() - vertices[:, 0].min()
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        uv[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / max(x_range, 1e-8)
        uv[:, 1] = (vertices[:, 1] - vertices[:, 1].min()) / max(y_range, 1e-8)
        return uv

    def _expand_mesh_for_uv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Duplicate seam vertices so exporters can use FLAME's per-face UVs."""
        expanded_vertices = self.vertices[self.faces.reshape(-1)]
        expanded_uv = self.uv[self.uv_faces.reshape(-1)]
        expanded_faces = np.arange(len(expanded_vertices), dtype=np.int64).reshape(-1, 3)
        return expanded_vertices, expanded_faces, expanded_uv
    
    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)
    
    @property
    def face_count(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces)
    
    def __repr__(self) -> str:
        return (
            f"ReconstructionResult(vertices={self.vertex_count}, "
            f"faces={self.face_count})"
        )


class FaceReconstructor:
    """
    3D Face Reconstruction from a single image.
    
    Uses DECA (Detailed Expression Capture and Animation) model
    for high-quality 3D face reconstruction.
    
    Example:
        >>> reconstructor = FaceReconstructor(device='cuda')
        >>> result = reconstructor.reconstruct('photo.jpg')
        >>> result.to_glb('output.glb')
    """
    
    # FLAME model identifiers
    FLAME2020 = "FLAME2020"
    FLAME2023 = "flame2023"
    FLAME2023_OPEN = "flame2023_Open"

    FLAME_FILENAMES: Dict[str, str] = {
        FLAME2020: "generic_model.pkl",
        FLAME2023: "flame2023.pkl",
        FLAME2023_OPEN: "flame2023_Open.pkl",
    }

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        use_mock: bool = False,
        flame_model: str = FLAME2020,
    ):
        """
        Initialize face reconstructor.
        
        Args:
            device: Device for inference ('cuda' or 'cpu')
            model_path: Path to pre-trained DECA model
            flame_model: FLAME model variant.
                FLAME2020 — default, required by DECA's trained checkpoint.
                flame2023_Open — CC-BY-4.0 model for alternative backends.
        """
        if torch is None:
            raise ImportError(
                "PyTorch is required for reconstruction. "
                "Install the project dependencies with `pip install -e .`."
            )

        from .postprocessor import MeshPostprocessor
        from .preprocessor import FacePreprocessor

        if flame_model not in self.FLAME_FILENAMES:
            raise ValueError(
                f"Unknown FLAME model: {flame_model!r}. "
                f"Choose from: {', '.join(self.FLAME_FILENAMES)}"
            )
        self.flame_model = flame_model
        self.flame_filename = self.FLAME_FILENAMES[flame_model]

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model_path = model_path
        self.use_mock = use_mock
        self.deca_mesh_only = False
        
        # Initialize components
        self.preprocessor = FacePreprocessor(device=str(self.device))
        self.postprocessor = MeshPostprocessor()
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load DECA model.
        
        Returns:
            Loaded DECA model
        """
        if self.use_mock:
            print("Using mock model for reproducible local demo.")
            self.deca_mesh_only = True
            return MockDECA()

        # --- FLAME 2023 path: standalone parametric decoder ---
        if self.flame_model != self.FLAME2020:
            return self._load_flame2023_model()

        # --- DECA (FLAME2020) path: full image-to-mesh pipeline ---
        try:
            import sys
            deca_root = Path(__file__).parent.parent.parent / "DECA"
            sys.path.append(str(deca_root))

            self._validate_deca_assets(deca_root)
            _install_legacy_chumpy_shims()

            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            albedo_path = deca_root / "data" / "FLAME_albedo_from_BFM.npz"
            standard_rasterizer = next(
                (deca_root / "decalib" / "utils" / "rasterizer").glob(
                    "standard_rasterize_cuda*.pyd"
                ),
                None,
            )
            if find_spec("pytorch3d") is not None:
                rasterizer_type = "pytorch3d"
            elif standard_rasterizer is not None:
                rasterizer_type = "standard"
            else:
                rasterizer_type = None
            mesh_only = rasterizer_type is None
            self.deca_mesh_only = mesh_only
            deca_cfg.model.use_tex = albedo_path.exists()
            deca_cfg.model.extract_tex = True
            if rasterizer_type is not None:
                deca_cfg.rasterizer_type = rasterizer_type
            deca_cfg.model.topology_path = str(deca_root / "data" / "head_template.obj")
            deca_cfg.model.mano_path = "data"
            deca_cfg.model.flame_path = "data"
            if not deca_cfg.model.use_tex:
                print(
                    "DECA texture model disabled: "
                    f"{albedo_path} was not found.",
                    flush=True,
                )
            if mesh_only:
                print(
                    "DECA renderer disabled: neither pytorch3d nor the "
                    "standard rasterizer extension is available; mesh-only "
                    "inference will be used.",
                    flush=True,
                )
            else:
                print(f"DECA renderer enabled: {rasterizer_type}", flush=True)

            original_torch_load = torch.load
            original_setup_renderer = DECA._setup_renderer

            def _torch_load_with_map_location(*args, **kwargs):
                if self.device.type == "cpu" and "map_location" not in kwargs:
                    kwargs["map_location"] = self.device
                return original_torch_load(*args, **kwargs)

            def _setup_mesh_only_renderer(deca_self, model_cfg):
                from decalib.utils import util

                _, uvcoords, faces, uv_faces = util.load_obj(model_cfg.topology_path)
                deca_self.faces = faces.to(deca_self.device)
                deca_self.raw_uvcoords = uvcoords.to(deca_self.device)
                deca_self.uvfaces = uv_faces.to(deca_self.device)

            try:
                torch.load = _torch_load_with_map_location
                if mesh_only:
                    DECA._setup_renderer = _setup_mesh_only_renderer
                return DECA(config=deca_cfg, device=self.device)
            finally:
                torch.load = original_torch_load
                DECA._setup_renderer = original_setup_renderer

        except Exception as exc:
            raise RuntimeError(
                "DECA backend is unavailable. Install DECA dependencies and required model assets, "
                f"or run with `--mock` for the deterministic local demo.\nReason: {exc}"
            ) from exc

    def _load_flame2023_model(self):
        """Load standalone Flame2023Decoder for parametric mesh generation.

        Unlike DECA, this decoder does NOT perform image-based reconstruction;
        it only generates meshes from shape/expression/pose parameters.
        """
        from .flame_decoder_2023 import Flame2023Decoder

        deca_root = Path(__file__).parent.parent.parent / "DECA"
        model_file = deca_root / "data" / self.flame_filename

        if not model_file.exists():
            raise FileNotFoundError(
                f"{self.flame_model} model not found at {model_file}. "
                "Download from https://flame.is.tue.mpg.de/ after registration."
            )

        print(
            f"Loading {self.flame_model} decoder from {model_file}",
            flush=True,
        )
        return Flame2023Decoder(model_file)

    def _validate_deca_assets(self, deca_root: Path) -> None:
        """Fail early with actionable guidance when DECA assets are missing."""
        if not deca_root.exists():
            raise FileNotFoundError(
                f"DECA repository was not found at {deca_root}. "
                "Clone it with `git clone https://github.com/yfeng95/DECA.git DECA`."
            )

        # Report alternative FLAME models available (not required by DECA)
        alt_models = []
        for model_key, fname in self.FLAME_FILENAMES.items():
            if model_key == self.FLAME2020:
                continue
            model_file = deca_root / "data" / fname
            if model_file.exists():
                alt_models.append(f"  + {model_key} ({fname}) — available at {model_file}")
        if alt_models:
            print("Alternative FLAME models found:")
            print("\n".join(alt_models))

        # DECA requires FLAME2020 (generic_model.pkl) — the model checkpoint
        # was trained specifically with FLAME2020's shape/expression bases.
        # FLAME2023 models have different basis vectors and are NOT compatible
        # with DECA without retraining.
        required_assets = {
            deca_root / "data" / "deca_model.tar": (
                "Download the released DECA checkpoint with "
                "`python -m gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O DECA/data/deca_model.tar`."
            ),
            deca_root / "data" / "generic_model.pkl": (
                "Download FLAME2020 from https://flame.is.tue.mpg.de/ after registration, "
                "accept the license, and copy `generic_model.pkl` to `DECA/data/generic_model.pkl`."
            ),
        }

        missing = [
            f"{path}: {hint}" for path, hint in required_assets.items() if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing DECA assets:\n" + "\n".join(missing))
    
    def reconstruct(
        self,
        image_path: Union[str, Path],
        with_texture: bool = True,
        detail_level: str = "high",
    ) -> ReconstructionResult:
        """
        Reconstruct 3D face from a single image.

        When the FLAME2023 decoder is active, image-based reconstruction
        is not supported. Use :meth:`generate` instead.

        Args:
            image_path: Path to input image
            with_texture: Whether to extract texture
            detail_level: Detail level ('low', 'medium', 'high')

        Returns:
            ReconstructionResult with 3D mesh
        """
        image_path = Path(image_path)

        # Flame2023Decoder does not support image-based reconstruction
        if not hasattr(self.model, "encode"):
            raise RuntimeError(
                f"The {self.flame_model} backend does not support image-based "
                f"reconstruction. Use `--flame-model FLAME2020` for image-to-mesh, "
                f"or call `.generate(shape=..., expression=..., pose=...)` directly."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        input_tensor = self._preprocess_for_deca(image_path)

        renderer_available = (
            not self.deca_mesh_only
            and hasattr(self.model, "render")
            and hasattr(self.model.render, "faces")
        )
        use_detail = renderer_available and detail_level != "low"
        return_vis = renderer_available and with_texture

        with torch.no_grad():
            codedict = self.model.encode(input_tensor, use_detail=use_detail)
            decoded = self.model.decode(
                codedict,
                rendering=renderer_available,
                vis_lmk=renderer_available,
                return_vis=return_vis,
                use_detail=use_detail,
            )
            if return_vis:
                opdict, _visdict = decoded
            else:
                opdict = decoded

            vertices = opdict["verts"][0].cpu().numpy()
            if renderer_available:
                faces = self.model.render.faces[0].cpu().numpy()
            elif hasattr(self.model, "faces"):
                faces = self.model.faces.cpu().numpy()
            else:
                faces = self.model.render.faces[0].cpu().numpy()

            texture_tensor = opdict.get("uv_texture_gt")
            if texture_tensor is None:
                texture_tensor = opdict.get("uv_texture")
            texture = texture_tensor
            if texture is not None:
                texture = texture[0].permute(1, 2, 0).cpu().numpy()
                texture = (texture * 255).astype(np.uint8)
            if not with_texture:
                texture = None

            uv = None
            uv_faces = None
            if texture is not None and renderer_available:
                uv = self.model.render.raw_uvcoords[0].cpu().numpy()
                uv_faces = self.model.render.uvfaces[0].cpu().numpy()

            params = {
                "shape": codedict["shape"].cpu().numpy(),
                "expression": codedict["exp"].cpu().numpy(),
                "pose": codedict["pose"].cpu().numpy(),
                "backend": {
                    "flame_model": self.flame_model,
                    "renderer_available": renderer_available,
                    "mesh_only": self.deca_mesh_only,
                    "use_detail": use_detail,
                    "texture_source": (
                        "deca_uv_texture_gt"
                        if texture is not None and "uv_texture_gt" in opdict
                        else "deca_uv_texture"
                        if texture is not None
                        else None
                    ),
                },
            }

        # Postprocess
        result = ReconstructionResult(
            vertices=vertices,
            faces=faces,
            texture=texture,
            uv=uv,
            uv_faces=uv_faces,
            params=params,
        )

        result = self.postprocessor.process(result)

        return result

    def _preprocess_for_deca(self, image_path: Path):
        """Use DECA's official test transform when the DECA package is present."""
        try:
            from decalib.datasets import datasets

            testdata = datasets.TestData(str(image_path), iscrop=False)
            return testdata[0]["image"].to(self.device)[None, ...]
        except Exception:
            return self.preprocessor.process(str(image_path))

    def generate(
        self,
        shape: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None,
        with_texture: bool = True,
    ) -> ReconstructionResult:
        """Generate FLAME mesh from parametric coefficients.

        This is the primary interface for the standalone FLAME2023 decoder.
        When using DECA (FLAME2020), use :meth:`reconstruct` instead.

        Args:
            shape: (B, N_shape) or (N_shape,) shape coefficients, or None.
            expression: (B, N_exp) or (N_exp,) expression coefficients, or None.
            pose: (B, J*3) or (J*3,) or (J, 3) axis-angle joint rotations, or None.
            with_texture: Whether to attach the FLAME mean texture and UV layout.

        Returns:
            ReconstructionResult with 3D mesh.
        """
        if hasattr(self.model, "encode"):
            raise RuntimeError(
                "generate() requires a standalone FLAME decoder backend. "
                f"Current backend ({self.flame_model}) does not support it."
            )

        def _ensure_batch(arr, name):
            if arr is None:
                return None
            if arr.ndim == 1:
                return arr[np.newaxis, :]
            return arr

        shape = _ensure_batch(shape, "shape")
        expression = _ensure_batch(expression, "expression")
        pose = _ensure_batch(pose, "pose")

        with torch.no_grad():
            vertices, faces, _ = self.model(
                shape=shape,
                expression=expression,
                pose=pose,
                return_joints=False,
            )

        texture = None
        uv = None
        uv_faces = None
        if with_texture:
            texture, uv, uv_faces = self._load_flame_texture()
            if uv_faces is not None and len(uv_faces) != len(faces):
                texture = None
                uv = None
                uv_faces = None

        result = ReconstructionResult(
            vertices=vertices[0].cpu().numpy(),
            faces=faces.cpu().numpy(),
            texture=texture,
            uv=uv,
            uv_faces=uv_faces,
            params={
                "shape": shape,
                "expression": expression,
                "pose": pose,
            },
        )

        result = self.postprocessor.process(result)
        return result

    def _load_flame_texture(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load FLAME mean texture plus UV topology when available."""
        texture_file = Path(__file__).parent.parent.parent / "DECA" / "data" / "FLAME_texture.npz"
        if not texture_file.exists():
            return None, None, None

        data = np.load(texture_file)
        # FLAME_texture.npz stores the mean texture in BGR order. Pillow,
        # trimesh, and browser GLB viewers expect RGB.
        texture = np.clip(data["mean"], 0, 255).astype(np.uint8)[:, :, ::-1]
        uv = data["vt"].astype(np.float32)
        uv_faces = data["ft"].astype(np.int64)
        return texture, uv, uv_faces
    
    def reconstruct_from_array(
        self,
        image_array: np.ndarray,
        with_texture: bool = True,
    ) -> ReconstructionResult:
        """
        Reconstruct from numpy array.

        Only supported when using DECA (FLAME2020) backend.

        Args:
            image_array: Image as numpy array (H, W, 3)
            with_texture: Whether to extract texture

        Returns:
            ReconstructionResult
        """
        if not hasattr(self.model, "encode"):
            raise RuntimeError(
                f"The {self.flame_model} backend does not support image-based "
                f"reconstruction. Use `generate()` instead."
            )

        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            codedict = self.model.encode(tensor)
            opdict = self.model.decode(codedict)
            vertices = opdict["verts"][0].cpu().numpy()
            faces = self.model.faces.cpu().numpy()

        return ReconstructionResult(vertices=vertices, faces=faces)


class MockDECA:
    """
    Mock DECA model for demonstration.
    
    Generates a simple face-like mesh for testing.  This class satisfies the
    interface that :meth:`FaceReconstructor.reconstruct` expects from DECA:
    ``encode(input_tensor)`` → ``codedict``, ``decode(codedict)`` → ``opdict``,
    and a ``.faces`` attribute (torch.Tensor, shape (F, 3)).
    """

    def __init__(self):
        self.faces = _make_mock_faces()

    def encode(self, tensor, use_detail=True):
        codedict = {
            "shape": torch.zeros((1, 300), dtype=torch.float32),
            "exp": torch.zeros((1, 100), dtype=torch.float32),
            "pose": torch.zeros((1, 15), dtype=torch.float32),
            "images": tensor,
        }
        if use_detail:
            codedict["detail"] = torch.zeros((1, 128), dtype=torch.float32)
        return codedict

    def decode(self, codedict, **kwargs):
        verts, _ = _make_mock_mesh()
        return {"verts": verts}


def _make_mock_mesh():
    """Simple sphere-like mesh as (verts_tensor, faces_tensor)."""
    n = 30
    verts = torch.zeros((n * n, 3), dtype=torch.float32)
    for i, p in enumerate(torch.linspace(0, torch.pi, n)):
        for j, t in enumerate(torch.linspace(0, 2 * torch.pi, n)):
            idx = i * n + j
            verts[idx] = torch.tensor(
                [torch.sin(p) * torch.cos(t), torch.cos(p), torch.sin(p) * torch.sin(t)]
            )
    verts = verts * 0.5

    faces = _make_mock_faces()

    return verts[None, ...], faces  # (1, V, 3)


def _make_mock_faces():
    n = 30
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            faces.append([idx, idx + n, idx + 1])
            faces.append([idx + 1, idx + n, idx + n + 1])
    return torch.tensor(faces, dtype=torch.long)
