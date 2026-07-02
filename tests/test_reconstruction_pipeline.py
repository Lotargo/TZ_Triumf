from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reconstruction.face_reconstructor import FaceReconstructor, ReconstructionResult
from src.reconstruction.main import build_parser


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
