"""
Main entry point for face reconstruction.
"""

import argparse
from pathlib import Path

from .face_reconstructor import FaceReconstructor


def main():
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
    
    args = parser.parse_args()
    
    # Initialize reconstructor
    print(f"Initializing Face Reconstructor...")
    reconstructor = FaceReconstructor(device=args.device)
    
    # Reconstruct
    print(f"Reconstructing face from: {args.input}")
    result = reconstructor.reconstruct(args.input)
    
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
