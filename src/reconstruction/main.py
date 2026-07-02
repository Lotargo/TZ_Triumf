"""
Main entry point for face reconstruction.
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for reconstruction runs."""
    parser = argparse.ArgumentParser(
        description="3D Face Reconstruction from single image"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.glb",
        help="Path to output file (default: output.glb)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device for inference"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["glb", "obj", "ply"],
        default="glb",
        help="Output format"
    )

    parser.add_argument(
        "--detail-level",
        type=str,
        choices=["low", "medium", "high"],
        default="high",
        help="Reconstruction detail level (reserved for real DECA backend)"
    )

    parser.add_argument(
        "--flame-model",
        type=str,
        choices=["FLAME2020", "flame2023", "flame2023_Open"],
        default="FLAME2020",
        help="FLAME model variant. DECA requires FLAME2020 (default). "
             "FLAME2023 models are for alternative backends."
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force deterministic mock model instead of trying to load DECA"
    )

    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip texture extraction/export"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from .face_reconstructor import FaceReconstructor
    
    # Initialize reconstructor
    print("Initializing Face Reconstructor...")
    try:
        reconstructor = FaceReconstructor(
            device=args.device,
            use_mock=args.mock,
            flame_model=args.flame_model,
        )

        # Reconstruct
        print(f"Reconstructing face from: {args.input}")
        result = reconstructor.reconstruct(
            args.input,
            with_texture=not args.no_texture,
            detail_level=args.detail_level,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "glb":
        result.to_glb(output_path)
    elif args.format == "obj":
        result.to_obj(output_path)
    elif args.format == "ply":
        result.to_ply(output_path)
    
    print(f"Done! Output saved to: {output_path}")
    print(f"  Vertices: {result.vertex_count}")
    print(f"  Faces: {result.face_count}")


if __name__ == "__main__":
    main()
