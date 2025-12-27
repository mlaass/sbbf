#!/usr/bin/env python3
"""
SBBF K-Sensitivity Benchmark for Voxelized Meshes

Measures FPR as a function of k (hash functions) for SBBF on voxelized 3D meshes.
Tests k = 2..12 with constant fill rate by adjusting memory per k.

Configurations:
- Meshes: bunny_128, teapot_128, armadillo_128, dragon_128
- SFC type: HILBERT_3D (best for structured spatial data)
- Intra-block strategies: double_hash, pattern_lookup
- Seed strategies: XOR, MULTIPLY_SHIFT

Usage:
    pip install sbbf[benchmarks]
    python scripts/sbbf_k_sensitivity.py
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

try:
    import sbbf
except ImportError as e:
    print("ERROR: Could not import sbbf module.")
    print("Install with: pip install sbbf")
    print(f"Details: {e}")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
HDF5_DIR = PROJECT_ROOT / "datasets" / "hdf5"
OUTPUT_DIR = PROJECT_ROOT / "sbbf_results" / "k_sensitivity"
TARGET_FILL_RATE = 0.5

# K values to test
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Meshes to test
MESHES = [
    {"name": "bunny", "resolution": 128},
    {"name": "teapot", "resolution": 128},
    {"name": "armadillo", "resolution": 128},
    {"name": "dragon", "resolution": 128},
]

# Configurations
INTRA_STRATEGIES = [
    ("double_hash", sbbf.IntraBlockStrategy.DOUBLE_HASH),
    ("pattern_lookup", sbbf.IntraBlockStrategy.PATTERN_LOOKUP),
]

SEED_STRATEGIES = [
    ("XOR", sbbf.SeedStrategy.XOR),
    ("MULTIPLY_SHIFT", sbbf.SeedStrategy.MULTIPLY_SHIFT),
]


# ============================================================================
# Helper Functions
# ============================================================================


def compute_log_blocks(n: int, k: int, target_fill: float = 0.5, bits_per_block: int = 64) -> int:
    """Compute log_num_blocks to achieve target fill rate."""
    required_blocks = (n * k) / (target_fill * bits_per_block)
    log_blocks = int(math.ceil(math.log2(required_blocks)))
    return max(log_blocks, 10)


def load_voxels(mesh_name: str, resolution: int) -> np.ndarray:
    """Load voxel coordinates from HDF5 file."""
    path = HDF5_DIR / f"{mesh_name}_{resolution}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Voxel file not found: {path}")

    with h5py.File(path, "r") as f:
        coords = f["coords"][:]
    return coords


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_benchmark(
    voxel_coords: np.ndarray,
    resolution: int,
    k: int,
    intra_strategy: sbbf.IntraBlockStrategy,
    seed_strategy: sbbf.SeedStrategy,
) -> dict:
    """Run SBBF benchmark for a single configuration."""
    num_voxels = len(voxel_coords)
    num_negatives = resolution**3 - num_voxels
    true_set = set(map(tuple, voxel_coords))

    log_num_blocks = compute_log_blocks(num_voxels, k, TARGET_FILL_RATE)

    config = sbbf.SBBFConfig()
    config.sfc_type = sbbf.SFCType.HILBERT_3D
    config.log_num_blocks = log_num_blocks
    config.hash_k = k
    config.bits_per_block = 64
    config.sfc_bits = max(8, int(math.ceil(math.log2(resolution + 1))))
    config.intra_strategy = intra_strategy
    config.seed_strategy = seed_strategy
    if intra_strategy == sbbf.IntraBlockStrategy.PATTERN_LOOKUP:
        config.pattern_table_size = 1024

    bf = sbbf.SpatialBlockedBloomFilter(config)

    # Insert all voxels
    for x, y, z in voxel_coords:
        bf.put3d(int(x), int(y), int(z))

    # Query all grid points to measure FPR
    false_positives = 0
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                if bf.query3d(x, y, z):
                    if (x, y, z) not in true_set:
                        false_positives += 1

    fpr = false_positives / num_negatives if num_negatives > 0 else 0.0

    return {
        "fpr": fpr,
        "false_positives": false_positives,
        "actual_fill_rate": bf.fill_ratio(),
        "memory_bytes": bf.memory_bytes(),
        "log_num_blocks": log_num_blocks,
    }


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("SBBF K-Sensitivity Benchmark (Voxelized Meshes)")
    print("=" * 60)
    print(f"Target fill rate: {TARGET_FILL_RATE}")
    print(f"K values: {K_VALUES}")
    print(f"Meshes: {[m['name'] for m in MESHES]}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    available_meshes = []

    for mesh in MESHES:
        mesh_name = mesh["name"]
        resolution = mesh["resolution"]
        path = HDF5_DIR / f"{mesh_name}_{resolution}.h5"
        if path.exists():
            available_meshes.append(mesh)
        else:
            print(f"Warning: {mesh_name}_{resolution}.h5 not found, skipping.")

    if not available_meshes:
        print("ERROR: No voxel datasets found!")
        print("Run: python scripts/download_voxel_meshes.py")
        sys.exit(1)

    total_configs = len(available_meshes) * len(K_VALUES) * len(INTRA_STRATEGIES) * len(SEED_STRATEGIES)
    current = 0

    for mesh in available_meshes:
        mesh_name = mesh["name"]
        resolution = mesh["resolution"]

        print(f"\n--- {mesh_name}_{resolution} ---")

        try:
            voxel_coords = load_voxels(mesh_name, resolution)
            print(f"Loaded {len(voxel_coords):,} voxels")
        except FileNotFoundError as e:
            print(f"SKIPPED: {e}")
            continue

        for k in K_VALUES:
            for intra_name, intra_strategy in INTRA_STRATEGIES:
                for seed_name, seed_strategy in SEED_STRATEGIES:
                    current += 1
                    print(
                        f"[{current}/{total_configs}] k={k} {intra_name} {seed_name}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_benchmark(
                        voxel_coords,
                        resolution,
                        k,
                        intra_strategy,
                        seed_strategy,
                    )

                    print(f"FPR={result['fpr']:.4%}")

                    results.append(
                        {
                            "config": {
                                "mesh": mesh_name,
                                "resolution": resolution,
                                "k": k,
                                "sfc_type": "HILBERT_3D",
                                "intra_strategy": intra_name,
                                "seed_strategy": seed_name,
                                "log_num_blocks": result["log_num_blocks"],
                            },
                            "metrics": {
                                "fpr": result["fpr"],
                                "false_positives": result["false_positives"],
                                "actual_fill_rate": result["actual_fill_rate"],
                                "memory_bytes": result["memory_bytes"],
                            },
                        }
                    )

    output = {
        "experiment": "k_sensitivity_voxel",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target_fill_rate": TARGET_FILL_RATE,
            "k_values": K_VALUES,
            "meshes": available_meshes,
        },
        "results": results,
    }

    output_path = OUTPUT_DIR / "voxel.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Total configurations: {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
