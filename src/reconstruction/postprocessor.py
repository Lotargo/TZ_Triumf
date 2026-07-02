"""
Post-processing of 3D reconstruction results.

Handles mesh smoothing, optimization, and format conversion.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


class MeshPostprocessor:
    """
    Post-process 3D mesh from reconstruction.
    
    Pipeline:
        1. Smooth vertices
        2. Optimize topology
        3. Compute normals
        4. Prepare for export
    """
    
    def __init__(
        self,
        smooth_sigma: float = 0.5,
        remove_degenerate: bool = True,
    ):
        """
        Initialize postprocessor.
        
        Args:
            smooth_sigma: Gaussian smoothing sigma
            remove_degenerate: Remove degenerate faces
        """
        self.smooth_sigma = smooth_sigma
        self.remove_degenerate = remove_degenerate
    
    def process(self, result) -> "ReconstructionResult":
        """
        Post-process reconstruction result.
        
        Args:
            result: ReconstructionResult
        
        Returns:
            Processed ReconstructionResult
        """
        from .face_reconstructor import ReconstructionResult
        
        vertices = result.vertices.copy()
        faces = result.faces.copy()
        
        # Smooth vertices
        vertices = self._smooth_vertices(vertices)
        
        # Remove degenerate faces
        uv_faces = result.uv_faces
        if self.remove_degenerate:
            vertices, faces, valid_mask = self._remove_degenerate(vertices, faces)
            if uv_faces is not None:
                uv_faces = uv_faces[valid_mask]
        
        # Compute normals
        normals = self._compute_normals(vertices, faces)
        
        # Create new result
        processed = ReconstructionResult(
            vertices=vertices,
            faces=faces,
            texture=result.texture,
            uv=result.uv,
            uv_faces=uv_faces,
            landmarks=result.landmarks,
            params=result.params,
            normals=normals,
        )
        
        return processed
    
    def _smooth_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Smooth vertex positions.
        
        Args:
            vertices: Vertex positions (N, 3)
        
        Returns:
            Smoothed vertices
        """
        if self.smooth_sigma <= 0:
            return vertices
        
        smoothed = vertices.copy()
        
        # Apply Gaussian smoothing to each coordinate
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(
                vertices[:, i], sigma=self.smooth_sigma
            )
        
        return smoothed
    
    def _remove_degenerate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple:
        """
        Remove degenerate faces (zero area).
        
        Args:
            vertices: Vertex positions
            faces: Face indices
        
        Returns:
            Tuple of (vertices, faces)
        """
        # Compute face areas
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Cross product for area
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2
        
        # Remove degenerate faces
        valid_mask = areas > 1e-8
        faces = faces[valid_mask]
        
        return vertices, faces, valid_mask
    
    def _compute_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Compute vertex normals.
        
        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (M, 3)
        
        Returns:
            Vertex normals (N, 3)
        """
        normals = np.zeros_like(vertices)
        
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        # Accumulate normals to vertices
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
        
        # Normalize
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        return normals
    
    def optimize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: Optional[int] = None,
    ) -> tuple:
        """
        Optimize mesh by decimation.
        
        Args:
            vertices: Vertex positions
            faces: Face indices
            target_faces: Target number of faces
        
        Returns:
            Tuple of (vertices, faces)
        """
        if target_faces is None or len(faces) <= target_faces:
            return vertices, faces
        
        try:
            import trimesh
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            simplified = mesh.simplify_quadric_decimation(target_faces)
            
            return simplified.vertices, simplified.faces
        
        except ImportError:
            return vertices, faces
    
    def center_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """
        Center mesh at origin.
        
        Args:
            vertices: Vertex positions
        
        Returns:
            Centered vertices
        """
        centroid = vertices.mean(axis=0)
        return vertices - centroid
    
    def scale_mesh(
        self,
        vertices: np.ndarray,
        target_height: float = 1.0,
    ) -> np.ndarray:
        """
        Scale mesh to target height.
        
        Args:
            vertices: Vertex positions
            target_height: Target height
        
        Returns:
            Scaled vertices
        """
        current_height = vertices[:, 1].max() - vertices[:, 1].min()
        
        if current_height > 0:
            scale = target_height / current_height
            vertices = vertices * scale
        
        return vertices
