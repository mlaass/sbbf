# SBBF - Spatial-Blocked Bloom Filter

A high-performance bloom filter designed for spatial data with space-filling curve locality. SBBF uses Morton (Z-order) and Hilbert curves to group spatially adjacent points into the same cache-line-sized blocks, enabling efficient neighbor queries.

## Features

- **Space-Filling Curve Indexing**: Morton and Hilbert curves for 2D and 3D data
- **Cache-Friendly Design**: 64-bit blocks aligned for CPU cache efficiency
- **Native Neighbor Queries**: Count adjacent cells with a single operation
- **Multiple Hash Strategies**: Double-hashing and pattern lookup
- **Seed Strategies**: XOR and MULTIPLY_SHIFT for hash diversification
- **Header-Only C++17**: Zero dependencies for core library
- **Python Bindings**: Full API access via pybind11

## Installation

### C++ (Header-Only)

```cpp
#include <sbbf/spatial_blocked_bloom_filter.hpp>

using namespace sbbf;

// Configure the filter
SBBFConfig config;
config.sfc_type = SFCType::HILBERT_2D;
config.log_num_blocks = 16;  // 2^16 = 64K blocks
config.hash_k = 8;
config.bits_per_block = 64;
config.sfc_bits = 16;

// Create filter (16-bit coordinates)
SpatialBlockedBloomFilter64<16> filter(config);

// Insert points
filter.put2D(100, 200);
filter.put2D(101, 200);

// Query membership
bool present = filter.get_bool_2D(100, 200);  // true

// Count neighbors (6-connected or 26-connected)
int neighbors = filter.count_neighbors_2D(100, 200, /* full_8 */ true);
```

### Python

```bash
pip install sbbf

# With voxel experiment dependencies
pip install sbbf[voxel]

# With benchmark dependencies
pip install sbbf[benchmarks]
```

```python
import sbbf

# Create configuration
config = sbbf.SBBFConfig()
config.sfc_type = sbbf.SFCType.HILBERT_3D
config.log_num_blocks = 15
config.hash_k = 4
config.bits_per_block = 64
config.sfc_bits = 8

# Create filter
bf = sbbf.SpatialBlockedBloomFilter(config)

# Insert 3D voxels
bf.put3d(10, 20, 30)
bf.put3d(10, 20, 31)

# Query
present = bf.query3d(10, 20, 30)  # True

# Count neighbors (26-connected)
neighbors = bf.neighbors3d(10, 20, 30, full_26=True)
```

### Space-Filling Curves (Python)

The SFC functions are also available directly for use without the bloom filter:

```python
import sbbf

# Morton (Z-order) curve - 2D
code = sbbf.morton2d_encode(100, 200)
x, y = sbbf.morton2d_decode(code)
assert (x, y) == (100, 200)

# Morton curve - 3D
code = sbbf.morton3d_encode(10, 20, 30)
x, y, z = sbbf.morton3d_decode(code)
assert (x, y, z) == (10, 20, 30)

# Hilbert curve - 2D (better locality than Morton)
code = sbbf.hilbert2d_encode(100, 200)
x, y = sbbf.hilbert2d_decode(code)
assert (x, y) == (100, 200)

# Hilbert curve - 3D
code = sbbf.hilbert3d_encode(10, 20, 30)
x, y, z = sbbf.hilbert3d_decode(code)
assert (x, y, z) == (10, 20, 30)
```

## Building from Source

### C++ Library and Tests

```bash
# Clone with submodules
git clone --recursive https://github.com/mlaass/sbbf.git
cd sbbf

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
./test_sbbf
```

### Python Bindings

```bash
# Install pybind11 submodule
git submodule update --init lib/pybind11

# Build with Python bindings
cmake -DSBBF_BUILD_PYTHON=ON ..
make -j$(nproc)
```

### Benchmarks with HDF5 Support

```bash
# Install HDF5 and HighFive
git submodule update --init lib/HighFive

cmake -DSBBF_BUILD_BENCHMARKS=ON -DSBBF_WITH_HDF5=ON ..
make -j$(nproc)

# Run benchmarks
./sbbf_k_sensitivity
./sbbf_seed_strategy
```

## Configuration Options

### SFC Types

| Type | Description |
|------|-------------|
| `MORTON_2D` | Z-order curve for 2D coordinates |
| `MORTON_3D` | Z-order curve for 3D coordinates |
| `HILBERT_2D` | Hilbert curve for 2D coordinates (better locality) |
| `HILBERT_3D` | Hilbert curve for 3D coordinates (better locality) |

### Intra-Block Strategies

| Strategy | Description |
|----------|-------------|
| `DOUBLE_HASH` | Classic double hashing: `h1 + i*h2` |
| `PATTERN_LOOKUP` | Pre-computed bit patterns for faster queries |
| `MULTIPLEXED` | Hybrid approach |

### Seed Strategies

| Strategy | Description |
|----------|-------------|
| `XOR` | XOR block index with seed |
| `MULTIPLY_SHIFT` | Multiply-shift hash combination |

## Voxel Experiments

Run experiments on voxelized 3D meshes:

```bash
# Download and voxelize meshes
python scripts/download_voxel_meshes.py

# Run voxel experiment (XOR vs MULTIPLY_SHIFT)
python scripts/sbbf_voxel_experiment.py

# Run k-sensitivity benchmark
python scripts/sbbf_k_sensitivity.py
```

## API Reference

### C++ API

```cpp
// Configuration
struct SBBFConfig {
    SFCType sfc_type = SFCType::HILBERT_2D;
    unsigned sfc_bits = 16;           // Coordinate bits
    unsigned log_num_blocks = 16;     // 2^16 blocks
    unsigned hash_k = 8;              // Hash functions
    unsigned bits_per_block = 64;     // Block size
    IntraBlockStrategy intra_strategy = IntraBlockStrategy::DOUBLE_HASH;
    SeedStrategy seed_strategy = SeedStrategy::XOR;
    unsigned pattern_table_size = 0;  // For PATTERN_LOOKUP
};

// Filter class
template<unsigned SFCBits>
class SpatialBlockedBloomFilter64 {
    // 2D operations
    void put2D(uint32_t x, uint32_t y);
    bool get_bool_2D(uint32_t x, uint32_t y) const;
    int count_neighbors_2D(uint32_t x, uint32_t y, bool full_8) const;

    // 3D operations
    void put3D(uint32_t x, uint32_t y, uint32_t z);
    bool get_bool_3D(uint32_t x, uint32_t y, uint32_t z) const;
    int count_neighbors_3D(uint32_t x, uint32_t y, uint32_t z, bool full_26) const;

    // Utility
    size_t memory_usage() const;
    double block_fill_ratio() const;
    void clear();
};
```

### Python API

```python
class SpatialBlockedBloomFilter:
    def put2d(self, x: int, y: int) -> None
    def put3d(self, x: int, y: int, z: int) -> None
    def query2d(self, x: int, y: int) -> bool
    def query3d(self, x: int, y: int, z: int) -> bool
    def neighbors2d(self, x: int, y: int, full_8: bool = True) -> int
    def neighbors3d(self, x: int, y: int, z: int, full_26: bool = True) -> int
    def has_neighbor2d(self, x: int, y: int, min_neighbors: int = 1) -> bool
    def memory_bytes(self) -> int
    def fill_ratio(self) -> float
    def clear(self) -> None
    def summary(self) -> str
```

## Performance

SBBF is optimized for:

- **Cache Efficiency**: Blocks are 64-bit aligned for single cache line access
- **Neighbor Queries**: Native support without separate queries per neighbor
- **Spatial Locality**: Space-filling curves keep nearby points together
- **SIMD-Friendly**: Pattern lookup uses pre-computed bit patterns

Typical performance (on modern x86-64):
- Insert: 30-50 ns/op
- Query: 20-40 ns/op
- Neighbor count: 50-100 ns/op

## License

MIT License - see LICENSE file for details.


