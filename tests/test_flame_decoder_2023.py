"""Tests for Flame2023Decoder."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Parameters used by the mock model — kept as module-level constants so tests
# can reference them when constructing the decoder with matching n_shape/n_exp.
MOCK_N_SHAPE = 10
MOCK_N_EXP = 4


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a minimal FLAME-like .pkl that passes basic shape checks.

    The pickle stores shapedirs with 300 shape + 100 expression dims (matching
    the FLAME 2023 format).  The decoder's ``n_shape`` / ``n_exp`` arguments
    select which slices to keep; tests should pass the *MOCK_N_SHAPE /
    *MOCK_N_EXP values.
    """
    n_vert = 100
    n_joint = 5
    total_shape_dims = 300
    total_exp_dims = 100

    v_template = np.random.randn(n_vert, 3).astype(np.float32)

    model = {
        "v_template": v_template,
        "shapedirs": np.zeros(
            (n_vert, 3, total_shape_dims + total_exp_dims), dtype=np.float32
        ),
        # Store posedirs in the pkl's native format: (V*3, P)
        "posedirs": np.random.randn(n_vert * 3, (n_joint - 1) * 9).astype(
            np.float32
        ),
        "J_regressor": np.zeros((n_joint, n_vert), dtype=np.float32),
        "kintree_table": np.zeros((2, n_joint), dtype=np.int32),
        "weights": np.zeros((n_vert, n_joint), dtype=np.float32),
        "f": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
    }
    # Give the first shape / first expression dims non-zero entries for the
    # "blend shapes change vertices" smoke check.
    model["shapedirs"][:, :, 0] = 0.01
    model["shapedirs"][:, :, 300] = 0.01
    # Skinning: every vertex fully bound to the first joint.
    model["weights"][:, 0] = 1.0
    model["kintree_table"][0] = [-1, 0, 1, 1, 1]
    model["kintree_table"][1] = [0, 1, 2, 3, 4]

    pkl_path = tmp_path / "flame2023_mock.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    return pkl_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlame2023DecoderConstruction:
    def test_loads_mock_model(self, mock_model_path):
        from src.reconstruction.flame_decoder_2023 import Flame2023Decoder

        dec = Flame2023Decoder(
            mock_model_path, n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP
        )
        assert dec.v_template.shape == (100, 3)
        assert dec.shapedirs.shape == (100, 3, MOCK_N_SHAPE + MOCK_N_EXP)
        assert dec.posedirs.shape == ((5 - 1) * 9, 100 * 3)
        assert dec.J_regressor.shape == (5, 100)
        assert dec.parents.shape == (5,)
        assert dec.lbs_weights.shape == (100, 5)
        assert dec.faces.shape == (2, 3)
        assert dec.n_vert == 100
        assert dec.n_joint == 5

    def test_raises_on_missing_file(self):
        from src.reconstruction.flame_decoder_2023 import Flame2023Decoder

        with pytest.raises(FileNotFoundError):
            Flame2023Decoder("nonexistent.pkl")


class TestFlame2023DecoderForward:
    def _make(self, path):
        from src.reconstruction.flame_decoder_2023 import Flame2023Decoder

        return Flame2023Decoder(path, n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP)

    def test_forward_all_zeros(self, mock_model_path):
        dec = self._make(mock_model_path)

        shape = torch.zeros(1, MOCK_N_SHAPE)
        exp = torch.zeros(1, MOCK_N_EXP)
        pose = torch.zeros(1, 5 * 3)

        verts, faces, joints = dec(shape=shape, expression=exp, pose=pose)
        assert verts.shape == (1, 100, 3)
        assert faces.shape == (2, 3)
        assert joints is None

    def test_forward_none_params(self, mock_model_path):
        dec = self._make(mock_model_path)

        verts, faces, _ = dec(shape=None, expression=None, pose=None)
        assert verts.shape == (1, 100, 3)

    def test_forward_numpy_inputs(self, mock_model_path):
        dec = self._make(mock_model_path)

        shape = np.zeros(MOCK_N_SHAPE, dtype=np.float32)
        exp = np.zeros(MOCK_N_EXP, dtype=np.float32)

        verts, faces, _ = dec(shape=shape, expression=exp, pose=None)
        assert verts.shape == (1, 100, 3)

    def test_forward_with_shape_expression(self, mock_model_path):
        dec = self._make(mock_model_path)

        shape = torch.ones(1, MOCK_N_SHAPE) * 0.5
        exp = torch.ones(1, MOCK_N_EXP) * 0.3
        pose = torch.zeros(1, 5 * 3)

        verts, faces, _ = dec(shape=shape, expression=exp, pose=pose)
        assert not torch.allclose(verts, dec.v_template.unsqueeze(0))

    def test_forward_with_pose_rotation(self, mock_model_path):
        dec = self._make(mock_model_path)

        shape = torch.zeros(1, MOCK_N_SHAPE)
        exp = torch.zeros(1, MOCK_N_EXP)
        angle = np.deg2rad(30)
        pose = torch.zeros(1, 5 * 3)
        pose[0, 1] = angle

        verts, faces, _ = dec(shape=shape, expression=exp, pose=pose)
        zero_pose = torch.zeros(1, 5 * 3)
        verts_zero, _, _ = dec(shape=shape, expression=exp, pose=zero_pose)
        assert not torch.allclose(verts, verts_zero)

    def test_forward_return_joints(self, mock_model_path):
        dec = self._make(mock_model_path)

        shape = torch.zeros(1, MOCK_N_SHAPE)
        exp = torch.zeros(1, MOCK_N_EXP)
        pose = torch.zeros(1, 5 * 3)

        verts, faces, joints = dec(
            shape=shape, expression=exp, pose=pose, return_joints=True
        )
        assert verts.shape == (1, 100, 3)
        assert joints.shape == (1, 5, 3)

    def test_forward_batch(self, mock_model_path):
        dec = self._make(mock_model_path)
        B = 4

        shape = torch.zeros(B, MOCK_N_SHAPE)
        exp = torch.zeros(B, MOCK_N_EXP)
        pose = torch.zeros(B, 5 * 3)

        verts, faces, _ = dec(shape=shape, expression=exp, pose=pose)
        assert verts.shape == (B, 100, 3)

    def test_pose_3d_shape_auto_flatten(self, mock_model_path):
        dec = self._make(mock_model_path)

        pose_2d = torch.zeros(1, 5 * 3)
        pose_3d = pose_2d.reshape(1, 5, 3)

        v1, _, _ = dec(pose=pose_2d)
        v2, _, _ = dec(pose=pose_3d)
        assert torch.allclose(v1, v2)


class TestFlame2023DecoderAPI:
    def test_repr(self, mock_model_path):
        from src.reconstruction.flame_decoder_2023 import Flame2023Decoder

        dec = Flame2023Decoder(
            mock_model_path, n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP
        )
        assert "Flame2023Decoder" in repr(dec)

    def test_has_expected_attributes(self, mock_model_path):
        from src.reconstruction.flame_decoder_2023 import Flame2023Decoder

        dec = Flame2023Decoder(
            mock_model_path, n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP
        )
        assert dec.n_shape == MOCK_N_SHAPE
        assert dec.n_exp == MOCK_N_EXP


# ---------------------------------------------------------------------------
# Integration test with FaceReconstructor.generate()
# ---------------------------------------------------------------------------


class TestFaceReconstructorGenerate:
    def test_generate_with_flame2023(self, mock_model_path, monkeypatch):
        """FaceReconstructor with flame2023 generates a mesh via generate()."""
        monkeypatch.setattr(
            "src.reconstruction.face_reconstructor.FaceReconstructor._load_flame2023_model",
            lambda self: __import__(
                "src.reconstruction.flame_decoder_2023", fromlist=["Flame2023Decoder"]
            ).Flame2023Decoder(
                str(mock_model_path), n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP
            ),
        )
        from src.reconstruction.face_reconstructor import FaceReconstructor

        rec = FaceReconstructor(use_mock=False, flame_model="flame2023_Open")
        result = rec.generate(
            shape=np.zeros(MOCK_N_SHAPE, dtype=np.float32),
            expression=np.zeros(MOCK_N_EXP, dtype=np.float32),
        )
        assert result.vertices.shape == (100, 3)
        assert result.faces.shape == (2, 3)

    def test_generate_raises_on_deca(self, monkeypatch):
        """generate() raises RuntimeError when a DECA-like encoder is loaded."""
        class _MockEncoderModel:
            encode = lambda self, x: None

        monkeypatch.setattr(
            "src.reconstruction.face_reconstructor.FaceReconstructor._load_model",
            lambda self: _MockEncoderModel(),
        )
        from src.reconstruction.face_reconstructor import FaceReconstructor

        rec = FaceReconstructor(use_mock=False, flame_model="FLAME2020")
        with pytest.raises(RuntimeError, match="standalone FLAME decoder"):
            rec.generate()

    def test_reconstruct_raises_on_flame2023(self, mock_model_path, monkeypatch):
        """reconstruct() raises RuntimeError with FLAME2023 decoder."""
        monkeypatch.setattr(
            "src.reconstruction.face_reconstructor.FaceReconstructor._load_flame2023_model",
            lambda self: __import__(
                "src.reconstruction.flame_decoder_2023", fromlist=["Flame2023Decoder"]
            ).Flame2023Decoder(
                str(mock_model_path), n_shape=MOCK_N_SHAPE, n_exp=MOCK_N_EXP
            ),
        )
        from src.reconstruction.face_reconstructor import FaceReconstructor

        rec = FaceReconstructor(use_mock=False, flame_model="flame2023_Open")
        with pytest.raises(RuntimeError, match="does not support image-based"):
            rec.reconstruct("nonexistent.jpg")
