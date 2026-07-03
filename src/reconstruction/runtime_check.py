"""Runtime checks for DECA/FLAME reconstruction environments."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def collect_runtime_report(project_root: Path | None = None) -> dict:
    """Collect local readiness checks for the full DECA renderer path."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    standard_rasterizer = next(
        (project_root / "DECA" / "decalib" / "utils" / "rasterizer").glob(
            "standard_rasterize_cuda*.pyd"
        ),
        None,
    )
    report = {
        "python": True,
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device": None,
        "pytorch3d_installed": find_spec("pytorch3d") is not None,
        "standard_rasterizer": standard_rasterizer is not None,
        "deca_repo": (project_root / "DECA").exists(),
        "deca_checkpoint": (project_root / "DECA" / "data" / "deca_model.tar").exists(),
        "flame2020": (project_root / "DECA" / "data" / "generic_model.pkl").exists(),
        "flame_texture": (project_root / "DECA" / "data" / "FLAME_texture.npz").exists(),
        "head_template": (project_root / "DECA" / "data" / "head_template.obj").exists(),
    }

    try:
        import torch

        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    report["full_deca_renderer_ready"] = all(
        [
            report["torch_installed"],
            report["cuda_available"],
            report["pytorch3d_installed"] or report["standard_rasterizer"],
            report["deca_repo"],
            report["deca_checkpoint"],
            report["flame2020"],
            report["head_template"],
        ]
    )
    return report


def format_runtime_report(report: dict) -> str:
    """Format runtime report for CLI output."""
    lines = [
        "DECA/FLAME runtime check",
        f"  torch: {'ok' if report['torch_installed'] else 'missing'}"
        + (f" ({report['torch_version']})" if report["torch_version"] else ""),
        f"  CUDA: {'ok' if report['cuda_available'] else 'missing'}"
        + (f" ({report['cuda_device']})" if report["cuda_device"] else ""),
        f"  PyTorch3D: {'ok' if report['pytorch3d_installed'] else 'missing'}",
        "  DECA standard rasterizer: "
        + ("ok" if report["standard_rasterizer"] else "missing"),
        f"  DECA repo: {'ok' if report['deca_repo'] else 'missing'}",
        f"  deca_model.tar: {'ok' if report['deca_checkpoint'] else 'missing'}",
        f"  generic_model.pkl: {'ok' if report['flame2020'] else 'missing'}",
        f"  head_template.obj: {'ok' if report['head_template'] else 'missing'}",
        f"  FLAME_texture.npz: {'ok' if report['flame_texture'] else 'missing'}",
        "  full renderer path: "
        + ("ready" if report["full_deca_renderer_ready"] else "not ready"),
    ]
    return "\n".join(lines)
