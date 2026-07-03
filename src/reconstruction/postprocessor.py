"""Post-processing and validation of 3D reconstruction results."""

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:  # pragma: no cover - scipy is a project dependency
    gaussian_filter1d = None


class MeshPostprocessor:
    """
    Post-process 3D mesh from reconstruction.

    Pipeline:
        1. Validate vertices/faces/UV compatibility
        2. Optionally smooth vertices when explicitly enabled
        3. Remove degenerate faces
        4. Compute normals
        5. Attach diagnostics for export/debugging
    """
    
    def __init__(
        self,
        smooth_sigma: float = 0.0,
        remove_degenerate: bool = True,
    ):
        """
        Initialize postprocessor.
        
        Args:
            smooth_sigma: Optional Gaussian smoothing sigma. Defaults to 0 because
                smoothing by vertex array index can corrupt FLAME/DECA topology.
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
        pre_diagnostics = self.validate_mesh(
            vertices,
            faces,
            uv=result.uv,
            uv_faces=result.uv_faces,
        )
        
        vertices = self._smooth_vertices(vertices)
        
        uv_faces = result.uv_faces
        if self.remove_degenerate:
            vertices, faces, valid_mask = self._remove_degenerate(vertices, faces)
            if uv_faces is not None:
                uv_faces = uv_faces[valid_mask]
        
        post_diagnostics = self.validate_mesh(
            vertices,
            faces,
            uv=result.uv,
            uv_faces=uv_faces,
        )
        normals = self._compute_normals(vertices, faces)
        params = dict(result.params)
        params["mesh_diagnostics"] = {
            "before_postprocess": pre_diagnostics,
            "after_postprocess": post_diagnostics,
            "smooth_sigma": self.smooth_sigma,
            "removed_degenerate_faces": (
                pre_diagnostics["degenerate_face_count"]
                - post_diagnostics["degenerate_face_count"]
            ),
        }
        
        processed = ReconstructionResult(
            vertices=vertices,
            faces=faces,
            texture=result.texture,
            uv=result.uv,
            uv_faces=uv_faces,
            landmarks=result.landmarks,
            params=params,
            normals=normals,
        )
        
        return processed

    def validate_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: Optional[np.ndarray] = None,
        uv_faces: Optional[np.ndarray] = None,
    ) -> dict:
        """Return mesh sanity diagnostics without mutating the mesh."""
        diagnostics = {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "finite_vertices": bool(np.isfinite(vertices).all()),
            "faces_shape_valid": bool(faces.ndim == 2 and faces.shape[1] == 3),
            "face_indices_valid": False,
            "degenerate_face_count": 0,
            "connected_components": 0,
            "bounds_min": None,
            "bounds_max": None,
            "uv_count": int(len(uv)) if uv is not None else 0,
            "uv_face_count": int(len(uv_faces)) if uv_faces is not None else 0,
            "uv_faces_compatible": uv is None and uv_faces is None,
        }

        if len(vertices) > 0:
            diagnostics["bounds_min"] = vertices.min(axis=0).tolist()
            diagnostics["bounds_max"] = vertices.max(axis=0).tolist()

        if diagnostics["faces_shape_valid"] and len(vertices) > 0 and len(faces) > 0:
            diagnostics["face_indices_valid"] = bool(
                faces.min() >= 0 and faces.max() < len(vertices)
            )
            if diagnostics["face_indices_valid"]:
                diagnostics["degenerate_face_count"] = self._count_degenerate_faces(
                    vertices, faces
                )
                diagnostics["connected_components"] = self._count_connected_components(
                    len(vertices), faces
                )

        if uv is not None:
            uv_faces_match = uv_faces is None or len(uv_faces) == len(faces)
            uv_indices_valid = True
            if uv_faces is not None and len(uv_faces) > 0:
                uv_indices_valid = bool(uv_faces.min() >= 0 and uv_faces.max() < len(uv))
            diagnostics["uv_faces_compatible"] = bool(uv_faces_match and uv_indices_valid)

        return diagnostics
    
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
        if gaussian_filter1d is None:
            raise ImportError("scipy is required when smooth_sigma > 0")
        
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

    def _count_degenerate_faces(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> int:
        """Count zero-area or near-zero-area triangles."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2
        return int(np.count_nonzero(areas <= 1e-8))

    def _count_connected_components(
        self,
        vertex_count: int,
        faces: np.ndarray,
    ) -> int:
        """Count connected components among vertices referenced by faces."""
        referenced = np.unique(faces.reshape(-1))
        if len(referenced) == 0:
            return 0

        adjacency = {int(idx): set() for idx in referenced}
        for a, b, c in faces:
            a = int(a)
            b = int(b)
            c = int(c)
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))

        seen = set()
        components = 0
        for start in referenced:
            start = int(start)
            if start in seen:
                continue
            components += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        return components
    
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
