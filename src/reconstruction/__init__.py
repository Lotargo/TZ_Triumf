"""3D face reconstruction pipeline package."""

__version__ = "0.1.0"
__author__ = "Boiko Oleg"

__all__ = [
    "FaceReconstructor",
    "FacePreprocessor",
    "MeshPostprocessor",
]


def __getattr__(name: str):
    if name == "FaceReconstructor":
        from .face_reconstructor import FaceReconstructor

        return FaceReconstructor
    if name == "FacePreprocessor":
        from .preprocessor import FacePreprocessor

        return FacePreprocessor
    if name == "MeshPostprocessor":
        from .postprocessor import MeshPostprocessor

        return MeshPostprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
