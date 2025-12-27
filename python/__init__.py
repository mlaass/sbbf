"""
SBBF: Spatial-Blocked Bloom Filter

A bloom filter variant that uses Space-Filling Curves (SFC) for block indexing
instead of hash functions. This preserves spatial locality, making neighborhood
queries significantly faster due to cache coherence.

Example:
    import sbbf

    # Create configuration
    config = sbbf.SBBFConfig()
    config.sfc_type = sbbf.SFCType.HILBERT_2D
    config.log_num_blocks = 14  # 2^14 = 16K blocks
    config.hash_k = 4

    # Create filter
    bf = sbbf.SpatialBlockedBloomFilter(config)

    # Insert points
    bf.put2d(100, 200)
    bf.put2d(101, 200)

    # Query membership
    assert bf.query2d(100, 200)
    assert not bf.query2d(500, 500)

    # Query neighborhood (key SBBF advantage)
    neighbors = bf.neighbors2d(100, 200, radius=1)  # 3x3 neighborhood
"""

from ._sbbf import (
    # Enums
    SFCType,
    IntraBlockStrategy,
    SeedStrategy,
    # Config
    SBBFConfig,
    # Main class
    SpatialBlockedBloomFilter,
    # Helper
    make_config,
    # Space-Filling Curve functions
    morton2d_encode,
    morton2d_decode,
    morton3d_encode,
    morton3d_decode,
    hilbert2d_encode,
    hilbert2d_decode,
    hilbert3d_encode,
    hilbert3d_decode,
    # Version
    __version__,
)

__all__ = [
    "SFCType",
    "IntraBlockStrategy",
    "SeedStrategy",
    "SBBFConfig",
    "SpatialBlockedBloomFilter",
    "make_config",
    # Space-Filling Curve functions
    "morton2d_encode",
    "morton2d_decode",
    "morton3d_encode",
    "morton3d_decode",
    "hilbert2d_encode",
    "hilbert2d_decode",
    "hilbert3d_encode",
    "hilbert3d_decode",
    "__version__",
]
