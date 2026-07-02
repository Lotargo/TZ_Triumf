"""
FLAME 2023 Decoder.

Self-contained PyTorch module that loads flame2023_Open.pkl (or flame2023.pkl)
and runs the full FLAME forward pass: shape + expression + pose -> vertices + faces.

Based on:

- DECA (https://github.com/yfeng95/DECA) — FLAME.py, lbs.py
- GaussianAvatars (https://github.com/ShenhanQian/GaussianAvatars) — flame.py
- SMPL-X / FLAME official releases

The model file (flame2023_Open.pkl) is NOT chumpy-based; it can be loaded
with plain pickle and converted to torch tensors directly.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LBS (Linear Blend Skinning) helpers — adapted from DECA/decalib/models/lbs.py
# ---------------------------------------------------------------------------

def blend_shapes(betas: torch.Tensor, shapedirs: torch.Tensor) -> torch.Tensor:
    """Per-vertex displacement from blend-shape coefficients.

    shapedirs: (V, 3, N_betas)   beta = [shape | exp]
    betas:     (B, N_betas)
    returns:   (B, V, 3)
    """
    return torch.einsum("bl,mkl->bmk", [betas, shapedirs])


def batch_rodrigues(
    rot_vecs: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Axis-angle (N, 3) -> rotation matrix (N, 3, 3)."""
    batch = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle.clamp_min(eps)

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros(batch, 1, dtype=dtype, device=device)

    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    ).view(batch, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return ident + sin * K + (1.0 - cos) * torch.bmm(K, K)


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R: (B, 3, 3), t: (B, 3, 1) -> T: (B, 4, 4)"""
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics.

    rot_mats: (B, J, 3, 3)
    joints:   (B, J, 3)
    parents:  (J,)  — kinematic parent indices, parent[0] = -1

    returns:
        posed_joints:    (B, J, 3)
        rel_transforms:  (B, J, 4, 4)  — T_j
    """
    joints = joints.unsqueeze(-1)  # (B, J, 3, 1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[parents[i]], transforms_mat[:, i]))
    transforms = torch.stack(chain, dim=1)

    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )
    return posed_joints, rel_transforms


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """J_regressor: (J, V), vertices: (B, V, 3) -> joints: (B, J, 3)"""
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear Blend Skinning.

    Follows DECA's lbs() exactly but does not accept pose2rot=False (always
    converts axis-angle to rotation matrices internally).

    betas:       (B, N_betas)
    pose:        (B, (J+1)*3)  — axis-angle per joint (global + J body)
    v_template:  (V, 3)
    shapedirs:   (V, 3, N_betas)
    posedirs:    (P, V*3)      — P = (J-1) * 9
    J_regressor: (J, V)
    parents:     (J,)          — kinematic tree
    lbs_weights: (V, J+1)

    returns:
        verts:  (B, V, 3)
        joints: (B, J, 3)
    """
    batch = max(betas.shape[0], pose.shape[0])
    device = betas.device
    dtype = betas.dtype

    # 1. Shape blend shapes
    v_shaped = v_template.unsqueeze(0) + blend_shapes(betas, shapedirs)

    # 2. Joint positions
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Pose blend shapes
    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch, -1, 3)
    v_posed = v_shaped + pose_offsets

    # 4. Forward kinematics
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)

    # 5. Skinning
    num_joints = J_regressor.shape[0]
    W = lbs_weights.unsqueeze(0).expand(batch, -1, -1)
    T = (
        torch.matmul(W, A.view(batch, num_joints, 16))
        .view(batch, -1, 4, 4)
    )

    ones = torch.ones(batch, v_posed.shape[1], 1, dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, ones], dim=2)
    verts = torch.matmul(T, v_posed_homo.unsqueeze(-1))[:, :, :3, 0]

    return verts, J_transformed


# ---------------------------------------------------------------------------
# FLAME 2023 Decoder
# ---------------------------------------------------------------------------

ShapeLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.tensor(arr, dtype=dtype)


def _install_legacy_pickle_shims() -> None:
    """Provide aliases needed by legacy chumpy FLAME pickles on Python 3.12."""
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


class Flame2023Decoder(nn.Module):
    """PyTorch FLAME 2023 decoder — no chumpy dependency.

    Loads ``flame2023_Open.pkl`` (or ``flame2023.pkl``) and runs the
    differentiable FLAME head model forward pass.

    Usage
    -----
    >>> decoder = Flame2023Decoder("models/flame2023_Open.pkl")
    >>> verts, faces, joints = decoder(
    ...     shape=torch.zeros(1, 300),
    ...     expression=torch.zeros(1, 100),
    ...     pose=torch.zeros(1, 15, 3),          # axis-angle per joint
    ... )
    >>> verts.shape
    (1, N_vertices, 3)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        n_shape: int = 300,
        n_exp: int = 100,
        use_face_contour: bool = True,
    ):
        super().__init__()

        self.n_shape = n_shape
        self.n_exp = n_exp
        self.dtype = torch.float32

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at {model_path}. "
                "Download from https://flame.is.tue.mpg.de/ after registration."
            )

        # FLAME2023 Open is plain pickle; the full registered release may
        # contain legacy chumpy objects that expect removed Python/NumPy aliases.
        _install_legacy_pickle_shims()
        import pickle

        with open(model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            # ss is a dict with keys matching SMPL-family format

        # -- template vertices -------------------------------------------------
        self.register_buffer(
            "v_template", _to_tensor(np.array(ss["v_template"]), self.dtype)
        )
        n_vert = self.v_template.shape[0]

        # -- shape / expression blend shapes -----------------------------------
        shapedirs = _to_tensor(np.array(ss["shapedirs"]), self.dtype)
        # shapedirs shape: (V, 3, 400) — first 300 = shape, next 100 = exp
        shapedirs = torch.cat(
            [shapedirs[:, :, :n_shape], shapedirs[:, :, 300 : 300 + n_exp]], dim=2
        )
        self.register_buffer("shapedirs", shapedirs)
        # Track the actual number of shape/exp dims after slicing
        self.n_shape = shapedirs.shape[-1] - n_exp
        self.n_exp = n_exp

        # -- pose blend shapes -------------------------------------------------
        posedirs = np.array(ss["posedirs"])  # (V*3, P)  or  (V, 3, P)
        if posedirs.ndim == 3:
            # (V, 3, P) -> (P, V*3) with Fortran ordering (DECA convention)
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1], order="F").T
        else:
            # (V*3, P) -> (P, V*3)
            posedirs = posedirs.T
        self.register_buffer("posedirs", _to_tensor(posedirs, self.dtype))

        # -- joint regressor ---------------------------------------------------
        J_regressor = ss["J_regressor"]
        if hasattr(J_regressor, "toarray"):
            J_regressor = J_regressor.toarray()
        self.register_buffer(
            "J_regressor", _to_tensor(np.array(J_regressor), self.dtype)
        )

        # -- kinematic tree ----------------------------------------------------
        kintree = np.array(ss["kintree_table"])  # (2, J)
        parents = torch.from_numpy(kintree[0].astype(np.int64))
        parents[0] = -1
        self.register_buffer("parents", parents)

        # -- skinning weights --------------------------------------------------
        self.register_buffer(
            "lbs_weights", _to_tensor(np.array(ss["weights"]), self.dtype)
        )

        # -- faces -------------------------------------------------------------
        faces = np.array(ss.get("f", ss.get("faces")))
        self.register_buffer(
            "faces", torch.from_numpy(faces.astype(np.int64)), persistent=False
        )

        # -- convenience count -------------------------------------------------
        self.n_vert = n_vert
        self.n_joint = self.J_regressor.shape[0]

    def forward(
        self,
        shape: Optional[ShapeLike] = None,
        expression: Optional[ShapeLike] = None,
        pose: Optional[ShapeLike] = None,
        return_joints: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run FLAME 2023 forward pass.

        Parameters
        ----------
        shape : (B, N_shape) or None
            Shape coefficients (zero-filled if None).
        expression : (B, N_exp) or None
            Expression coefficients (zero-filled if None).
        pose : (B, J*3) or (B, J, 3) or None
            Axis-angle per joint. The first 3 values are the global rotation;
            remaining joints are neck, jaw, eyes (standard FLAME layout).
            Zero-filled if None.
        return_joints : bool
            If True, also return the posed joint locations.

        Returns
        -------
        vertices : (B, V, 3)
        faces : (F, 3) — shared across the batch
        joints : (B, J, 3) or None
        """
        device = self.v_template.device
        batched = False

        if shape is not None:
            if isinstance(shape, np.ndarray):
                shape = torch.from_numpy(shape).float()
            shape = shape.to(device)
            if shape.ndim == 1:
                shape = shape.unsqueeze(0)
            batched = True
        if expression is not None:
            if isinstance(expression, np.ndarray):
                expression = torch.from_numpy(expression).float()
            expression = expression.to(device)
            if expression.ndim == 1:
                expression = expression.unsqueeze(0)
            batched = True
        if pose is not None:
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float()
            pose = pose.to(device)
            if pose.ndim == 1:
                pose = pose.unsqueeze(0)
            batched = True

        B = 1
        if shape is not None:
            B = max(B, shape.shape[0])
        if expression is not None:
            B = max(B, expression.shape[0])
        if pose is not None:
            B = max(B, pose.shape[0])

        if shape is None:
            shape = torch.zeros(B, self.n_shape, device=device, dtype=self.dtype)
        if expression is None:
            expression = torch.zeros(B, self.n_exp, device=device, dtype=self.dtype)
        if pose is None:
            pose = torch.zeros(B, self.n_joint * 3, device=device, dtype=self.dtype)

        # Flatten pose if given as (B, J, 3)
        if pose.ndim == 3:
            pose = pose.reshape(B, -1)

        betas = torch.cat([shape, expression], dim=1)

        verts, joints = lbs(
            betas=betas,
            pose=pose,
            v_template=self.v_template,
            shapedirs=self.shapedirs,
            posedirs=self.posedirs,
            J_regressor=self.J_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
        )

        faces = self.faces
        if return_joints:
            return verts, faces, joints
        return verts, faces, None
