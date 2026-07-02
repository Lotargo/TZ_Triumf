"""
3D Face Reconstruction from single image using DECA.

This module provides a complete pipeline for 3D face reconstruction
from a single photograph, including preprocessing, inference, and
postprocessing.
"""

from .face_reconstructor import FaceReconstructor
from .preprocessor import FacePreprocessor
from .postprocessor import MeshPostprocessor

__version__ = "0.1.0"
__author__ = "Boiko Oleg"

__all__ = [
    "FaceReconstructor",
    "FacePreprocessor",
    "MeshPostprocessor",
]
