from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reconstruction.face_reconstructor import FaceReconstructor, ReconstructionResult
from src.reconstruction.main import build_parser
from src.reconstruction.postprocessor import MeshPostprocessor
from src.reconstruction.runtime_check import format_runtime_report


def test_cli_accepts_documented_detail_level_flag():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--input",
            "photo.jpg",
            "--output",
            "result.glb",
            "--detail-level",
            "high",
            "--mock",
        ]
    )

    assert args.input == "photo.jpg"
    assert args.output == "result.glb"
    assert args.detail_level == "high"
    assert args.mock is True


def test_cli_accepts_flame2023_generation_without_input():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--output",
            "neutral.glb",
            "--flame-model",
            "flame2023_Open",
        ]
    )

    assert args.input is None
    assert args.output == "neutral.glb"
    assert args.flame_model == "flame2023_Open"


def test_cli_accepts_check_env_without_input():
    parser = build_parser()

    args = parser.parse_args(["--check-env"])

    assert args.check_env is True
    assert args.input is None


def test_runtime_report_marks_full_renderer_not_ready_when_cuda_missing():
    report = {
        "torch_installed": True,
        "torch_version": "2.0.0",
        "cuda_available": False,
        "cuda_device": None,
        "pytorch3d_installed": True,
        "standard_rasterizer": False,
        "deca_repo": True,
        "deca_checkpoint": True,
        "flame2020": True,
        "flame_texture": True,
        "head_template": True,
        "full_deca_renderer_ready": False,
    }

    text = format_runtime_report(report)

    assert "CUDA: missing" in text
    assert "full renderer path: not ready" in text


def test_obj_export_without_texture_has_valid_face_indices(tmp_path: Path):
    result = ReconstructionResult(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    output = tmp_path / "triangle.obj"

    result.to_obj(output)

    content = output.read_text(encoding="utf-8")
    assert "v 0.000000 0.000000 0.000000" in content
    assert "f 1 2 3" in content
    assert "/1" not in content


def test_obj_export_with_flame_uv_uses_texture_indices(tmp_path: Path):
    result = ReconstructionResult(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texture=np.zeros((4, 4, 3), dtype=np.uint8),
        uv=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=np.float32,
        ),
        uv_faces=np.array([[2, 1, 0]], dtype=np.int64),
    )
    output = tmp_path / "triangle_textured.obj"

    result.to_obj(output)

    content = output.read_text(encoding="utf-8")
    assert "vt 0.100000 0.800000" in content
    assert "f 1/3 2/2 3/1" in content


def test_mock_reconstruction_exports_glb(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    pytest.importorskip("trimesh")

    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (128, 128), color=(210, 180, 165)).save(image_path)

    reconstructor = FaceReconstructor(device="cpu", use_mock=True)
    result = reconstructor.reconstruct(image_path, with_texture=False)

    output = tmp_path / "face.glb"
    result.to_glb(output)

    assert result.vertex_count > 0
    assert result.face_count > 0
    assert result.normals is not None
    assert output.exists()
    assert output.stat().st_size > 0


def test_postprocessor_does_not_smooth_vertices_by_default():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = ReconstructionResult(
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )

    processed = MeshPostprocessor().process(result)

    assert np.allclose(processed.vertices, vertices)
    assert processed.params["mesh_diagnostics"]["smooth_sigma"] == 0.0


def test_mesh_diagnostics_report_invalid_face_indices():
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 5]], dtype=np.int64)

    diagnostics = MeshPostprocessor().validate_mesh(vertices, faces)

    assert diagnostics["faces_shape_valid"] is True
    assert diagnostics["face_indices_valid"] is False


def test_mesh_diagnostics_report_uv_incompatibility():
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.zeros((2, 2), dtype=np.float32)
    uv_faces = np.array([[0, 1, 2]], dtype=np.int64)

    diagnostics = MeshPostprocessor().validate_mesh(
        vertices,
        faces,
        uv=uv,
        uv_faces=uv_faces,
    )

    assert diagnostics["uv_faces_compatible"] is False


def test_mesh_diagnostics_count_connected_components():
    vertices = np.zeros((6, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    diagnostics = MeshPostprocessor().validate_mesh(vertices, faces)

    assert diagnostics["connected_components"] == 2


def test_deca_renderer_path_attaches_uv_texture(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")

    class _FakeRender:
        faces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
        raw_uvcoords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        uvfaces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)

    class _FakeDECA:
        render = _FakeRender()

        def encode(self, tensor, use_detail=True):
            assert use_detail is True
            return {
                "shape": torch.zeros((1, 300)),
                "exp": torch.zeros((1, 100)),
                "pose": torch.zeros((1, 15)),
                "images": tensor,
                "detail": torch.zeros((1, 128)),
            }

        def decode(self, codedict, **kwargs):
            assert kwargs["rendering"] is True
            assert kwargs["return_vis"] is True
            assert kwargs["use_detail"] is True
            opdict = {
                "verts": torch.tensor(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
                ),
                "uv_texture_gt": torch.ones((1, 3, 2, 2)),
            }
            return opdict, {}

    monkeypatch.setattr(
        "src.reconstruction.face_reconstructor.FaceReconstructor._load_model",
        lambda self: _FakeDECA(),
    )

    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (128, 128), color=(210, 180, 165)).save(image_path)

    rec = FaceReconstructor(device="cpu", use_mock=False)
    result = rec.reconstruct(image_path, with_texture=True)

    assert result.texture.shape == (2, 2, 3)
    assert result.uv.shape == (3, 2)
    assert result.uv_faces.shape == (1, 3)
    assert result.params["backend"]["texture_source"] == "deca_uv_texture_gt"
    assert result.params["backend"]["renderer_available"] is True
