"""Download optional external model assets used by the reconstruction pipeline."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    output: Path
    sha256: str
    description: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSETS = {
    "flame2020": Asset(
        name="flame2020",
        url=(
            "https://huggingface.co/camenduru/show/resolve/"
            "064a379f415f674051145ec4862f54bd6a65073f/"
            "models/models_MICA/FLAME2020/generic_model.pkl?download=true"
        ),
        output=PROJECT_ROOT / "DECA" / "data" / "generic_model.pkl",
        sha256="efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b",
        description="FLAME2020 generic model required by DECA",
    ),
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, force: bool = False) -> Path:
    """Download an asset and verify its SHA256 checksum."""
    asset.output.parent.mkdir(parents=True, exist_ok=True)

    if asset.output.exists() and not force:
        current_hash = sha256_file(asset.output)
        if current_hash == asset.sha256:
            print(f"{asset.output} already exists and checksum is valid.")
            return asset.output
        raise RuntimeError(
            f"{asset.output} exists but checksum does not match.\n"
            f"Expected: {asset.sha256}\n"
            f"Actual:   {current_hash}\n"
            "Run again with --force to replace it."
        )

    temp_path = asset.output.with_suffix(asset.output.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    print(f"Downloading {asset.description}...")
    print(f"Source: {asset.url}")
    print(f"Target: {asset.output}")

    with urllib.request.urlopen(asset.url) as response, open(temp_path, "wb") as f:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            received += len(chunk)
            if total:
                percent = received / total * 100
                print(f"\r{received / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="")
            else:
                print(f"\r{received / 1024 / 1024:.1f} MB", end="")
    print()

    actual_hash = sha256_file(temp_path)
    if actual_hash != asset.sha256:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum failed for {asset.name}.\n"
            f"Expected: {asset.sha256}\n"
            f"Actual:   {actual_hash}"
        )

    temp_path.replace(asset.output)
    print(f"Saved: {asset.output}")
    return asset.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download optional model assets for TZ_Triumf."
    )
    parser.add_argument(
        "asset",
        choices=sorted(ASSETS),
        help="Asset to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file even if it is already present.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        download(ASSETS[args.asset], force=args.force)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
