#!/usr/bin/env python3
"""
SBBF Voxel Experiment

Compares SBBF seed strategies on voxelized 3D meshes.
Measures false positive rates, applies neighbor-based denoising,
and renders comparison images.

Usage:
    pip install sbbf[voxel]
    python scripts/sbbf_voxel_experiment.py

Prerequisites:
    - Run download_voxel_meshes.py first to generate HDF5 datasets
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from tqdm import tqdm

try:
    import sbbf
except ImportError as e:
    print("ERROR: Could not import sbbf module.")
    print("Install with: pip install sbbf")
    print(f"Details: {e}")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
HDF5_DIR = PROJECT_ROOT / "datasets" / "hdf5"
RESULTS_DIR = PROJECT_ROOT / "sbbf_results"
FIGURES_DIR = PROJECT_ROOT / "sbbf_results" / "figures"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENTS = [
    {
        "name": "bunny_128_medium",
        "mesh": "bunny",
        "resolution": 128,
        "log_num_blocks": 15,
        "hash_k": 4,
        "sfc_type": "HILBERT_3D",
        "min_neighbors": 2,
    },
    {
        "name": "teapot_128_medium",
        "mesh": "teapot",
        "resolution": 128,
        "log_num_blocks": 15,
        "hash_k": 4,
        "sfc_type": "HILBERT_3D",
        "min_neighbors": 2,
    },
    {
        "name": "armadillo_128_medium",
        "mesh": "armadillo",
        "resolution": 128,
        "log_num_blocks": 15,
        "hash_k": 4,
        "sfc_type": "HILBERT_3D",
        "min_neighbors": 2,
    },
    {
        "name": "dragon_128_medium",
        "mesh": "dragon",
        "resolution": 128,
        "log_num_blocks": 15,
        "hash_k": 4,
        "sfc_type": "HILBERT_3D",
        "min_neighbors": 2,
    },
]


def fix_orientation(coords: np.ndarray) -> np.ndarray:
    """Swap Y and Z axes to correct vertical orientation."""
    if len(coords) == 0:
        return coords
    return coords[:, [0, 2, 1]]


def rotate_for_mesh(coords: np.ndarray, mesh_name: str) -> np.ndarray:
    """Apply mesh-specific rotations to face front."""
    if len(coords) == 0:
        return coords
    if mesh_name == "armadillo":
        rot_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        rot_matrix = rot_x_180 @ rot_y
        return coords @ rot_matrix.T
    return coords


def transform_coords(coords: np.ndarray, mesh_name: str) -> np.ndarray:
    """Apply all coordinate transformations."""
    coords = fix_orientation(coords)
    coords = rotate_for_mesh(coords, mesh_name)
    return coords


def load_voxels(mesh_name: str, resolution: int) -> np.ndarray:
    """Load voxel coordinates from HDF5 file."""
    path = HDF5_DIR / f"{mesh_name}_{resolution}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Voxel file not found: {path}")

    with h5py.File(path, "r") as f:
        coords = f["coords"][:]
        print(f"  Loaded {len(coords):,} voxels from {path.name}")
    return coords


def get_sfc_type(name: str) -> sbbf.SFCType:
    """Convert string SFC type to enum."""
    sfc_map = {
        "HILBERT_3D": sbbf.SFCType.HILBERT_3D,
        "MORTON_3D": sbbf.SFCType.MORTON_3D,
        "HILBERT_2D": sbbf.SFCType.HILBERT_2D,
        "MORTON_2D": sbbf.SFCType.MORTON_2D,
    }
    return sfc_map.get(name, sbbf.SFCType.HILBERT_3D)


def run_sbbf(
    voxel_coords: np.ndarray,
    true_set: set,
    resolution: int,
    min_neighbors: int,
    sfc_type: sbbf.SFCType,
    log_num_blocks: int,
    hash_k: int,
    seed_strategy: sbbf.SeedStrategy,
    strategy_name: str,
) -> dict:
    """Run SBBF with a specific seed strategy."""
    num_negatives = resolution**3 - len(voxel_coords)

    # Configure SBBF
    config = sbbf.SBBFConfig()
    config.sfc_type = sfc_type
    config.log_num_blocks = log_num_blocks
    config.hash_k = hash_k
    config.bits_per_block = 64
    config.sfc_bits = max(8, int(math.ceil(math.log2(resolution + 1))))
    config.seed_strategy = seed_strategy

    bf = sbbf.SpatialBlockedBloomFilter(config)

    # Insert all voxels
    for x, y, z in tqdm(voxel_coords, desc=f"  Insert ({strategy_name})", leave=False):
        bf.put3d(int(x), int(y), int(z))

    # Query all grid points
    queried_points = []
    false_positives = []

    for x in tqdm(range(resolution), desc=f"  Query ({strategy_name})", leave=False):
        for y in range(resolution):
            for z in range(resolution):
                if bf.query3d(x, y, z):
                    queried_points.append((x, y, z))
                    if (x, y, z) not in true_set:
                        false_positives.append((x, y, z))

    queried = np.array(queried_points) if queried_points else np.empty((0, 3))
    fps = np.array(false_positives) if false_positives else np.empty((0, 3))

    raw_fpr = len(fps) / num_negatives if num_negatives > 0 else 0.0

    # Apply denoising
    denoised = []
    for x, y, z in tqdm(queried, desc=f"  Denoise ({strategy_name})", leave=False):
        neighbor_count = bf.neighbors3d(int(x), int(y), int(z), full_26=True)
        if neighbor_count >= min_neighbors:
            denoised.append((x, y, z))

    denoised = np.array(denoised) if denoised else np.empty((0, 3))

    denoised_fps = sum(1 for p in denoised if tuple(p) not in true_set)
    denoised_fpr = denoised_fps / num_negatives if num_negatives > 0 else 0.0

    fps_removed = len(fps) - denoised_fps
    correction_rate = fps_removed / len(fps) if len(fps) > 0 else 0.0

    return {
        "queried": queried,
        "fps": fps,
        "denoised": denoised,
        "raw_fpr": raw_fpr,
        "raw_fps": len(fps),
        "denoised_fpr": denoised_fpr,
        "denoised_fps": denoised_fps,
        "fps_removed": fps_removed,
        "correction_rate": correction_rate,
        "memory_kb": bf.memory_bytes() / 1024,
        "fill_ratio": bf.fill_ratio(),
    }


def run_single_experiment(exp: dict) -> dict:
    """Run a single experiment comparing SBBF XOR vs MULTIPLY_SHIFT."""
    mesh_name = exp["mesh"]
    resolution = exp["resolution"]
    log_num_blocks = exp["log_num_blocks"]
    hash_k = exp["hash_k"]
    sfc_type = get_sfc_type(exp["sfc_type"])
    min_neighbors = exp.get("min_neighbors", 2)

    # Load voxels
    voxel_coords = load_voxels(mesh_name, resolution)
    true_set = set(map(tuple, voxel_coords))

    # Run SBBF with XOR strategy
    print("  Running SBBF (XOR)...")
    sbbf_xor = run_sbbf(
        voxel_coords, true_set, resolution, min_neighbors, sfc_type, log_num_blocks, hash_k, sbbf.SeedStrategy.XOR, "XOR"
    )
    print(f"  SBBF XOR FPR: {sbbf_xor['raw_fpr']:.4%} -> {sbbf_xor['denoised_fpr']:.4%}")

    # Run SBBF with MULTIPLY_SHIFT strategy
    print("  Running SBBF (MULTIPLY_SHIFT)...")
    sbbf_ms = run_sbbf(
        voxel_coords,
        true_set,
        resolution,
        min_neighbors,
        sfc_type,
        log_num_blocks,
        hash_k,
        sbbf.SeedStrategy.MULTIPLY_SHIFT,
        "MS",
    )
    print(f"  SBBF MultShift FPR: {sbbf_ms['raw_fpr']:.4%} -> {sbbf_ms['denoised_fpr']:.4%}")

    return {
        "name": exp["name"],
        "config": {
            "mesh": mesh_name,
            "resolution": resolution,
            "log_num_blocks": log_num_blocks,
            "hash_k": hash_k,
            "sfc_type": exp["sfc_type"],
            "min_neighbors": min_neighbors,
        },
        "stats": {
            "num_voxels": len(voxel_coords),
            "grid_points": resolution**3,
            "sbbf_xor_memory_kb": sbbf_xor["memory_kb"],
            "sbbf_xor_fill_ratio": sbbf_xor["fill_ratio"],
            "sbbf_xor_raw_fpr": sbbf_xor["raw_fpr"],
            "sbbf_xor_denoised_fpr": sbbf_xor["denoised_fpr"],
            "sbbf_xor_correction_rate": sbbf_xor["correction_rate"],
            "sbbf_ms_memory_kb": sbbf_ms["memory_kb"],
            "sbbf_ms_fill_ratio": sbbf_ms["fill_ratio"],
            "sbbf_ms_raw_fpr": sbbf_ms["raw_fpr"],
            "sbbf_ms_denoised_fpr": sbbf_ms["denoised_fpr"],
            "sbbf_ms_correction_rate": sbbf_ms["correction_rate"],
        },
        "_ground_truth": voxel_coords,
        "_sbbf_xor_queried": sbbf_xor["queried"],
        "_sbbf_xor_fps": sbbf_xor["fps"],
        "_sbbf_xor_denoised": sbbf_xor["denoised"],
        "_sbbf_ms_queried": sbbf_ms["queried"],
        "_sbbf_ms_fps": sbbf_ms["fps"],
        "_sbbf_ms_denoised": sbbf_ms["denoised"],
    }


def render_comparison(result: dict, output_path: Path):
    """Render 5-panel comparison with PyVista."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth = result["_ground_truth"]
    mesh_name = result["config"]["mesh"]
    resolution = result["config"]["resolution"]
    stats = result["stats"]
    font_size = 20

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(shape=(1, 5), window_size=(3000, 600), off_screen=True)
    point_size = max(2, 8 - resolution // 64)

    # Panel 0: Ground truth
    plotter.subplot(0, 0)
    actor = plotter.add_text(
        f"Ground Truth\n({len(ground_truth):,} voxels)",
        position="upper_edge",
        font_size=font_size,
        color="black",
    )
    actor.prop.background_color = "white"
    actor.prop.background_opacity = 0.7
    if len(ground_truth) > 0:
        cloud = pv.PolyData(transform_coords(ground_truth, mesh_name).astype(float))
        plotter.add_mesh(cloud, color="green", point_size=point_size, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.camera_position = "iso"

    # Panels 1-2: SBBF XOR
    for i, (data_key, title, fpr_key) in enumerate([
        ("_sbbf_xor_queried", "SBBF XOR Raw", "sbbf_xor_raw_fpr"),
        ("_sbbf_xor_denoised", "SBBF XOR Denoised", "sbbf_xor_denoised_fpr"),
    ], start=1):
        plotter.subplot(0, i)
        fpr = stats[fpr_key]
        actor = plotter.add_text(f"{title}\n(FPR={fpr:.2%})", position="upper_edge", font_size=font_size, color="black")
        actor.prop.background_color = "white"
        actor.prop.background_opacity = 0.7
        data = result[data_key]
        if len(data) > 0:
            color = "cyan" if "Denoised" in title else "blue"
            cloud = pv.PolyData(transform_coords(data, mesh_name).astype(float))
            plotter.add_mesh(cloud, color=color, point_size=point_size, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.camera_position = "iso"

    # Panels 3-4: SBBF MultShift
    for i, (data_key, title, fpr_key) in enumerate([
        ("_sbbf_ms_queried", "SBBF MS Raw", "sbbf_ms_raw_fpr"),
        ("_sbbf_ms_denoised", "SBBF MS Denoised", "sbbf_ms_denoised_fpr"),
    ], start=3):
        plotter.subplot(0, i)
        fpr = stats[fpr_key]
        actor = plotter.add_text(f"{title}\n(FPR={fpr:.2%})", position="upper_edge", font_size=font_size, color="black")
        actor.prop.background_color = "white"
        actor.prop.background_opacity = 0.7
        data = result[data_key]
        if len(data) > 0:
            color = "cyan" if "Denoised" in title else "blue"
            cloud = pv.PolyData(transform_coords(data, mesh_name).astype(float))
            plotter.add_mesh(cloud, color=color, point_size=point_size, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.camera_position = "iso"

    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"  Saved: {output_path.name}")


def save_results(results: list[dict], output_path: Path):
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_results = []
    for r in results:
        clean = {
            "name": r["name"],
            "config": r["config"],
            "stats": r["stats"],
            "image": f"{r['name']}.png",
        }
        clean_results.append(clean)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(clean_results),
        "experiments": clean_results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SBBF Voxel Experiment")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering images")
    args = parser.parse_args()

    print("=" * 60)
    print("SBBF Voxel Experiment: XOR vs MULTIPLY_SHIFT")
    print(f"Running {len(EXPERIMENTS)} experiments")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(EXPERIMENTS)}] {exp['name']}")
        print("=" * 60)

        try:
            result = run_single_experiment(exp)
            results.append(result)

            if not args.no_render:
                img_path = FIGURES_DIR / f"{exp['name']}.png"
                render_comparison(result, img_path)

        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if results:
        save_results(results, RESULTS_DIR / "voxel_experiments.json")

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print(f"  Results: {RESULTS_DIR / 'voxel_experiments.json'}")
    print(f"  Images:  {FIGURES_DIR}/*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
