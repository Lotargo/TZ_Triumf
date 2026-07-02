"""
Face Reconstruction Module.

Main class for 3D face reconstruction from a single image.
Uses DECA (Detailed Expression Capture and Animation) model.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .preprocessor import FacePreprocessor
from .postprocessor import MeshPostprocessor


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
        landmarks: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.texture = texture
        self.landmarks = landmarks
        self.params = params or {}
    
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
            
            # Write texture coordinates if available
            if self.texture is not None:
                # Placeholder UV coordinates
                for i in range(len(self.vertices)):
                    f.write(f"vt {i % 2} {i % 2}\n")
            
            # Write faces (1-indexed)
            for face in self.faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    def to_glb(self, path: Union[str, Path]) -> None:
        """
        Export mesh to GLB format (for Three.js).
        
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
        
        # Add texture if available
        if self.texture is not None:
            # Create UV mapping
            uv = self._create_uv_mapping(mesh)
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
    
    def _create_uv_mapping(self, mesh) -> np.ndarray:
        """Create simple UV mapping for texture."""
        # Simple planar projection
        uv = np.zeros((len(mesh.vertices), 2))
        uv[:, 0] = (mesh.vertices[:, 0] - mesh.vertices[:, 0].min()) / (
            mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        )
        uv[:, 1] = (mesh.vertices[:, 1] - mesh.vertices[:, 1].min()) / (
            mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()
        )
        return uv
    
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
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize face reconstructor.
        
        Args:
            device: Device for inference ('cuda' or 'cpu')
            model_path: Path to pre-trained DECA model
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Initialize components
        self.preprocessor = FacePreprocessor()
        self.postprocessor = MeshPostprocessor()
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load DECA model.
        
        Returns:
            Loaded DECA model
        """
        try:
            # Try to import DECA
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent / "DECA"))
            
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            
            # Configure DECA
            deca_cfg.model.use_tex = True
            deca_cfg.model.topology_path = "data/head_template.obj"
            deca_cfg.model.mano_path = "data"
            deca_cfg.model.flame_path = "data"
            
            # Initialize DECA
            deca = DECA(config=deca_cfg, device=self.device)
            
            return deca
            
        except ImportError:
            print("DECA not found. Using mock model for demonstration.")
            return MockDECA()
    
    def reconstruct(
        self,
        image_path: Union[str, Path],
        with_texture: bool = True,
        detail_level: str = "high",
    ) -> ReconstructionResult:
        """
        Reconstruct 3D face from a single image.
        
        Args:
            image_path: Path to input image
            with_texture: Whether to extract texture
            detail_level: Detail level ('low', 'medium', 'high')
        
        Returns:
            ReconstructionResult with 3D mesh
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        input_tensor = self.preprocessor.process(str(image_path))
        
        # Run inference
        with torch.no_grad():
            if hasattr(self.model, "encode"):
                # Real DECA model
                with torch.no_grad():
                    codedict = self.model.encode(input_tensor)
                    opdict = self.model.decode(codedict)
                
                vertices = opdict["verts"][0].cpu().numpy()
                faces = self.model.faces.cpu().numpy()
                texture = opdict.get("tex_image", None)
                if texture is not None:
                    texture = texture[0].permute(1, 2, 0).cpu().numpy()
                    texture = (texture * 255).astype(np.uint8)
                
                params = {
                    "shape": codedict["shape"].cpu().numpy(),
                    "expression": codedict["exp"].cpu().numpy(),
                    "pose": codedict["pose"].cpu().numpy(),
                }
            else:
                # Mock model for demonstration
                vertices, faces, texture = self.model.reconstruct(input_tensor)
                params = {}
        
        # Postprocess
        result = ReconstructionResult(
            vertices=vertices,
            faces=faces,
            texture=texture,
            params=params,
        )
        
        result = self.postprocessor.process(result)
        
        return result
    
    def reconstruct_from_array(
        self,
        image_array: np.ndarray,
        with_texture: bool = True,
    ) -> ReconstructionResult:
        """
        Reconstruct from numpy array.
        
        Args:
            image_array: Image as numpy array (H, W, 3)
            with_texture: Whether to extract texture
        
        Returns:
            ReconstructionResult
        """
        # Convert to tensor
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            if hasattr(self.model, "encode"):
                codedict = self.model.encode(tensor)
                opdict = self.model.decode(codedict)
                
                vertices = opdict["verts"][0].cpu().numpy()
                faces = self.model.faces.cpu().numpy()
            else:
                vertices, faces, _ = self.model.reconstruct(tensor)
        
        return ReconstructionResult(vertices=vertices, faces=faces)


class MockDECA:
    """
    Mock DECA model for demonstration.
    
    Generates a simple face-like mesh for testing.
    """
    
    def __init__(self):
        # Generate simple face mesh
        self.vertices, self.faces = self._generate_simple_face()
    
    def _generate_simple_face(self):
        """Generate a simple face-like mesh."""
        # Create sphere-like face
        phi = np.linspace(0, np.pi, 30)
        theta = np.linspace(0, 2 * np.pi, 30)
        
        vertices = []
        for p in phi:
            for t in theta:
                x = np.sin(p) * np.cos(t)
                y = np.cos(p)
                z = np.sin(p) * np.sin(t)
                vertices.append([x, y, z])
        
        vertices = np.array(vertices) * 0.5  # Scale
        
        # Create faces (quads converted to triangles)
        faces = []
        n = 30
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.append([idx, idx + n, idx + 1])
                faces.append([idx + 1, idx + n, idx + n + 1])
        
        return vertices, np.array(faces)
    
    def reconstruct(self, tensor):
        """Mock reconstruction."""
        return self.vertices, self.faces, None
