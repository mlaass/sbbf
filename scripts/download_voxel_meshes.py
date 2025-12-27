#!/usr/bin/env python3
"""
Download and Voxelize 3D Meshes

Downloads classic 3D meshes from public repositories and voxelizes them
at configurable resolutions. Saves voxel coordinates to HDF5 files.

Meshes:
- Stanford Bunny (35K vertices)
- Utah Teapot (classic CG model)
- Stanford Dragon (high detail)
- Stanford Armadillo (high detail)

Usage:
    pip install sbbf[voxel]
    python scripts/download_voxel_meshes.py

    # Download specific mesh
    python scripts/download_voxel_meshes.py --mesh bunny

    # Custom resolution
    python scripts/download_voxel_meshes.py --resolution 256
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
import gzip
import shutil

try:
    import numpy as np
    import h5py
    import trimesh
except ImportError as e:
    print("ERROR: Missing required dependencies.")
    print("Install with: pip install trimesh h5py numpy")
    print(f"Details: {e}")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MESH_DIR = PROJECT_ROOT / "datasets" / "meshes"
HDF5_DIR = PROJECT_ROOT / "datasets" / "hdf5"

# Mesh URLs (public domain / permissive license)
MESH_SOURCES = {
    "bunny": {
        "url": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj",
        "file": "bunny.obj",
        "description": "Stanford Bunny",
    },
    "teapot": {
        "url": "https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj",
        "file": "teapot.obj",
        "description": "Utah Teapot",
    },
    "dragon": {
        "url": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/dragon.obj",
        "file": "dragon.obj",
        "description": "Stanford Dragon",
    },
    "armadillo": {
        "url": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/armadillo.obj",
        "file": "armadillo.obj",
        "description": "Stanford Armadillo",
    },
}

# Default resolutions to generate
DEFAULT_RESOLUTIONS = [64, 128]


def download_mesh(mesh_name: str, force: bool = False) -> Path:
    """Download a mesh file if not already present."""
    if mesh_name not in MESH_SOURCES:
        raise ValueError(f"Unknown mesh: {mesh_name}. Available: {list(MESH_SOURCES.keys())}")

    info = MESH_SOURCES[mesh_name]
    mesh_path = MESH_DIR / info["file"]

    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if mesh_path.exists() and not force:
        print(f"  Using cached: {mesh_path.name}")
        return mesh_path

    print(f"  Downloading {info['description']}...")
    try:
        urlretrieve(info["url"], mesh_path)
        print(f"  Saved to: {mesh_path}")
    except Exception as e:
        print(f"  ERROR downloading {mesh_name}: {e}")
        # Try alternative sources
        alt_urls = [
            f"https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/{mesh_name}.obj",
        ]
        for alt_url in alt_urls:
            try:
                print(f"  Trying alternative: {alt_url}")
                urlretrieve(alt_url, mesh_path)
                print(f"  Saved to: {mesh_path}")
                return mesh_path
            except Exception:
                continue
        raise RuntimeError(f"Failed to download {mesh_name}")

    return mesh_path


def voxelize_mesh(mesh_path: Path, resolution: int) -> np.ndarray:
    """Voxelize a mesh at the given resolution."""
    print(f"  Loading mesh: {mesh_path.name}")
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries in scene
        mesh = mesh.dump(concatenate=True)

    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Voxelize
    print(f"  Voxelizing at resolution {resolution}...")

    # Calculate pitch to fit mesh in resolution^3 grid
    bounds = mesh.bounds
    max_dim = (bounds[1] - bounds[0]).max()
    pitch = max_dim / (resolution - 1)

    voxelized = mesh.voxelized(pitch=pitch)

    # Get filled voxel coordinates
    coords = voxelized.points

    # Normalize to [0, resolution-1] range
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1  # Avoid division by zero

    normalized = (coords - coords_min) / coords_range * (resolution - 1)
    voxel_coords = normalized.astype(np.int32)

    # Remove duplicates
    voxel_coords = np.unique(voxel_coords, axis=0)

    print(f"  Generated {len(voxel_coords):,} voxels")

    return voxel_coords


def save_voxels(coords: np.ndarray, mesh_name: str, resolution: int) -> Path:
    """Save voxel coordinates to HDF5."""
    HDF5_DIR.mkdir(parents=True, exist_ok=True)

    output_path = HDF5_DIR / f"{mesh_name}_{resolution}.h5"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        f.attrs["mesh_name"] = mesh_name
        f.attrs["resolution"] = resolution
        f.attrs["num_voxels"] = len(coords)

    print(f"  Saved: {output_path.name} ({len(coords):,} voxels)")

    return output_path


def process_mesh(mesh_name: str, resolutions: list[int], force_download: bool = False):
    """Download and voxelize a mesh at multiple resolutions."""
    print(f"\n{'=' * 50}")
    print(f"Processing: {mesh_name}")
    print("=" * 50)

    try:
        mesh_path = download_mesh(mesh_name, force=force_download)

        for resolution in resolutions:
            print(f"\n  Resolution: {resolution}x{resolution}x{resolution}")

            output_path = HDF5_DIR / f"{mesh_name}_{resolution}.h5"
            if output_path.exists() and not force_download:
                print(f"  Skipping (already exists): {output_path.name}")
                continue

            coords = voxelize_mesh(mesh_path, resolution)
            save_voxels(coords, mesh_name, resolution)

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and voxelize 3D meshes for SBBF experiments"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        choices=list(MESH_SOURCES.keys()) + ["all"],
        default="all",
        help="Mesh to process (default: all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help=f"Voxel resolutions (default: {DEFAULT_RESOLUTIONS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-voxelize",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available meshes and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available meshes:")
        for name, info in MESH_SOURCES.items():
            print(f"  {name}: {info['description']}")
        return

    print("=" * 60)
    print("SBBF Voxel Dataset Generator")
    print("=" * 60)
    print(f"Resolutions: {args.resolution}")
    print(f"Output: {HDF5_DIR}")

    meshes = list(MESH_SOURCES.keys()) if args.mesh == "all" else [args.mesh]

    success_count = 0
    for mesh_name in meshes:
        if process_mesh(mesh_name, args.resolution, force_download=args.force):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Complete! Processed {success_count}/{len(meshes)} meshes")
    print(f"Datasets saved to: {HDF5_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
