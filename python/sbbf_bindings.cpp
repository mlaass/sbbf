/**
 * SBBF Python Bindings via pybind11
 *
 * Exposes the Spatial-Blocked Bloom Filter to Python with full configuration
 * support for SFC types, intra-block strategies, and seed strategies.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sbbf/spatial_blocked_bloom_filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_sbbf, m) {
    m.doc() = "SBBF: Spatial-Blocked Bloom Filter with Space-Filling Curve locality";

    // ========================================================================
    // Enumerations
    // ========================================================================

    py::enum_<sbbf::SFCType>(m, "SFCType", "Space-filling curve type for block indexing")
        .value("MORTON_2D", sbbf::SFCType::MORTON_2D, "Morton (Z-order) curve for 2D coordinates")
        .value("MORTON_3D", sbbf::SFCType::MORTON_3D, "Morton (Z-order) curve for 3D coordinates")
        .value("HILBERT_2D", sbbf::SFCType::HILBERT_2D, "Hilbert curve for 2D coordinates (better locality)")
        .value("HILBERT_3D", sbbf::SFCType::HILBERT_3D, "Hilbert curve for 3D coordinates (better locality)");

    py::enum_<sbbf::IntraBlockStrategy>(m, "IntraBlockStrategy", "Strategy for setting bits within a block")
        .value("DOUBLE_HASH", sbbf::IntraBlockStrategy::DOUBLE_HASH, "h_i = (h1 + i*h2) % B - standard double hashing")
        .value("PATTERN_LOOKUP", sbbf::IntraBlockStrategy::PATTERN_LOOKUP, "Pre-computed k-bit patterns from lookup table")
        .value("MULTIPLEXED", sbbf::IntraBlockStrategy::MULTIPLEXED, "OR multiple patterns together");

    py::enum_<sbbf::SeedStrategy>(m, "SeedStrategy", "Strategy for deriving intra-block seed from SFC code")
        .value("XOR", sbbf::SeedStrategy::XOR, "XOR high and low bits of SFC code")
        .value("MULTIPLY_SHIFT", sbbf::SeedStrategy::MULTIPLY_SHIFT, "Multiply-shift hash for better mixing");

    // ========================================================================
    // Configuration
    // ========================================================================

    py::class_<sbbf::SBBFConfig>(m, "SBBFConfig", "Configuration for Spatial-Blocked Bloom Filter")
        .def(py::init<>(), "Create default configuration")
        .def_readwrite("sfc_type", &sbbf::SBBFConfig::sfc_type,
            "Space-filling curve type (MORTON_2D, HILBERT_2D, etc.)")
        .def_readwrite("sfc_bits", &sbbf::SBBFConfig::sfc_bits,
            "Bits per coordinate in SFC encoding (max resolution, default 16)")
        .def_readwrite("log_num_blocks", &sbbf::SBBFConfig::log_num_blocks,
            "Log2 of number of blocks (num_blocks = 2^log_num_blocks)")
        .def_readwrite("bits_per_block", &sbbf::SBBFConfig::bits_per_block,
            "Bits per block (64, 256, or 512)")
        .def_readwrite("hash_k", &sbbf::SBBFConfig::hash_k,
            "Number of bits set per element within a block")
        .def_readwrite("intra_strategy", &sbbf::SBBFConfig::intra_strategy,
            "Intra-block hashing strategy")
        .def_readwrite("seed_strategy", &sbbf::SBBFConfig::seed_strategy,
            "Seed derivation strategy (XOR or MULTIPLY_SHIFT)")
        .def_readwrite("pattern_table_size", &sbbf::SBBFConfig::pattern_table_size,
            "Pattern table size for PATTERN_LOOKUP strategy (default 1024)")
        .def("num_blocks", &sbbf::SBBFConfig::computed_num_blocks,
            "Get computed number of blocks")
        .def("memory_bytes", &sbbf::SBBFConfig::computed_memory_bytes,
            "Get computed memory usage in bytes")
        .def("validate", &sbbf::SBBFConfig::validate,
            "Validate configuration (raises exception on invalid config)")
        .def("__repr__", &sbbf::SBBFConfig::to_string);

    // ========================================================================
    // SBBF with 16-bit coordinates (default)
    // ========================================================================

    using SBBF16 = sbbf::SpatialBlockedBloomFilter64<16>;
    py::class_<SBBF16>(m, "SpatialBlockedBloomFilter",
        "Spatial-Blocked Bloom Filter with 64-bit blocks and 16-bit coordinates.\n\n"
        "Uses Space-Filling Curves (SFC) for block indexing instead of hash functions.\n"
        "This preserves spatial locality, making neighborhood queries significantly\n"
        "faster due to cache coherence.\n\n"
        "Example:\n"
        "    config = sbbf.SBBFConfig()\n"
        "    config.sfc_type = sbbf.SFCType.HILBERT_2D\n"
        "    config.log_num_blocks = 14\n"
        "    config.hash_k = 4\n"
        "    bf = sbbf.SpatialBlockedBloomFilter(config)\n"
        "    bf.put2d(100, 200)\n"
        "    assert bf.query2d(100, 200)")
        .def(py::init<const sbbf::SBBFConfig&>(),
             py::arg("config"),
             "Create a Spatial-Blocked Bloom Filter with the given configuration")

        // Insert operations
        .def("put2d", &SBBF16::put2D,
             py::arg("x"), py::arg("y"),
             "Insert a 2D point using SFC encoding")
        .def("put3d", &SBBF16::put3D,
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Insert a 3D point using SFC encoding")
        .def("put", &SBBF16::put,
             py::arg("point"),
             "Insert a point using vector interface [x, y] or [x, y, z]")

        // Query operations
        .def("query2d", &SBBF16::get_bool_2D,
             py::arg("x"), py::arg("y"),
             "Query membership for 2D point. Returns True if point may exist.")
        .def("query3d", &SBBF16::get_bool_3D,
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Query membership for 3D point. Returns True if point may exist.")
        .def("query", &SBBF16::get_bool,
             py::arg("point"),
             "Query membership using vector interface")

        // Raw SFC code access
        .def("put_by_sfc", &SBBF16::put_by_sfc,
             py::arg("sfc_code"),
             "Insert using raw SFC code (skips coordinate encoding)")
        .def("query_by_sfc", &SBBF16::get_bool_by_sfc,
             py::arg("sfc_code"),
             "Query using raw SFC code (skips coordinate encoding)")

        // Neighborhood queries (key SBBF advantage)
        .def("neighbors2d", &SBBF16::query_neighborhood_2D,
             py::arg("x"), py::arg("y"), py::arg("radius") = 1,
             "Query 2D neighborhood around point.\n\n"
             "Args:\n"
             "    x, y: Center coordinates\n"
             "    radius: Chebyshev radius (1 = 3x3, 2 = 5x5)\n"
             "Returns:\n"
             "    Bitmask of found neighbors (row-major, (2r+1)^2 bits)")
        .def("neighbors3d", &SBBF16::query_neighborhood_3D,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("full_26") = false,
             "Query 3D neighborhood around point.\n\n"
             "Args:\n"
             "    x, y, z: Center coordinates\n"
             "    full_26: If True, check all 26 neighbors; otherwise only 6 face neighbors\n"
             "Returns:\n"
             "    Count of present neighbors")
        .def("has_neighbor2d", &SBBF16::has_any_neighbor_2D,
             py::arg("x"), py::arg("y"), py::arg("radius") = 1,
             "Check if any neighbor exists (faster than full neighborhood query)")

        // Utility functions
        .def("clear", &SBBF16::clear,
             "Clear all blocks")
        .def("memory_bytes", &SBBF16::memory_usage,
             "Get memory usage in bytes")
        .def("fill_ratio", &SBBF16::block_fill_ratio,
             "Get fraction of blocks with any bits set")
        .def("avg_bits_per_block", &SBBF16::avg_bits_per_filled_block,
             "Get average bits set per filled block")
        .def("summary", &SBBF16::summary,
             "Get JSON summary of filter state")

        // Debug/analysis
        .def("get_block_index_2d", &SBBF16::get_block_index_2D,
             py::arg("x"), py::arg("y"),
             "Get the block index for a 2D point (for debugging)")
        .def("get_sfc_value_2d", &SBBF16::get_sfc_value_2D,
             py::arg("x"), py::arg("y"),
             "Get the SFC value for a 2D point (for debugging)")

        // Config access
        .def_readonly("config", &SBBF16::config,
             "Filter configuration")

        .def("__repr__", [](const SBBF16 &f) {
            return "<SpatialBlockedBloomFilter blocks=" +
                   std::to_string(f.config.computed_num_blocks()) +
                   " k=" + std::to_string(f.config.hash_k) +
                   " memory=" + std::to_string(f.memory_usage()) + "B>";
        });

    // ========================================================================
    // Helper functions
    // ========================================================================

    m.def("make_config", [](
        const std::string& sfc_type_str,
        unsigned log_num_blocks,
        unsigned hash_k,
        unsigned sfc_bits,
        const std::string& intra_strategy_str,
        const std::string& seed_strategy_str
    ) {
        sbbf::SBBFConfig config;

        // Parse SFC type
        if (sfc_type_str == "MORTON_2D") config.sfc_type = sbbf::SFCType::MORTON_2D;
        else if (sfc_type_str == "MORTON_3D") config.sfc_type = sbbf::SFCType::MORTON_3D;
        else if (sfc_type_str == "HILBERT_2D") config.sfc_type = sbbf::SFCType::HILBERT_2D;
        else if (sfc_type_str == "HILBERT_3D") config.sfc_type = sbbf::SFCType::HILBERT_3D;
        else throw std::invalid_argument("Unknown SFC type: " + sfc_type_str);

        // Parse intra-block strategy
        if (intra_strategy_str == "DOUBLE_HASH") config.intra_strategy = sbbf::IntraBlockStrategy::DOUBLE_HASH;
        else if (intra_strategy_str == "PATTERN_LOOKUP") config.intra_strategy = sbbf::IntraBlockStrategy::PATTERN_LOOKUP;
        else if (intra_strategy_str == "MULTIPLEXED") config.intra_strategy = sbbf::IntraBlockStrategy::MULTIPLEXED;
        else throw std::invalid_argument("Unknown intra-block strategy: " + intra_strategy_str);

        // Parse seed strategy
        if (seed_strategy_str == "XOR") config.seed_strategy = sbbf::SeedStrategy::XOR;
        else if (seed_strategy_str == "MULTIPLY_SHIFT") config.seed_strategy = sbbf::SeedStrategy::MULTIPLY_SHIFT;
        else throw std::invalid_argument("Unknown seed strategy: " + seed_strategy_str);

        config.log_num_blocks = log_num_blocks;
        config.hash_k = hash_k;
        config.sfc_bits = sfc_bits;

        return config;
    },
    py::arg("sfc_type") = "HILBERT_2D",
    py::arg("log_num_blocks") = 14,
    py::arg("hash_k") = 4,
    py::arg("sfc_bits") = 16,
    py::arg("intra_strategy") = "DOUBLE_HASH",
    py::arg("seed_strategy") = "XOR",
    "Create an SBBFConfig with string parameters for convenience.\n\n"
    "Args:\n"
    "    sfc_type: 'MORTON_2D', 'MORTON_3D', 'HILBERT_2D', or 'HILBERT_3D'\n"
    "    log_num_blocks: Log2 of number of blocks (default 14 = 16K blocks)\n"
    "    hash_k: Number of bits per element (default 4)\n"
    "    sfc_bits: Bits per coordinate (default 16)\n"
    "    intra_strategy: 'DOUBLE_HASH', 'PATTERN_LOOKUP', or 'MULTIPLEXED'\n"
    "    seed_strategy: 'XOR' or 'MULTIPLY_SHIFT'\n"
    "Returns:\n"
    "    SBBFConfig object\n\n"
    "Example:\n"
    "    config = sbbf.make_config('HILBERT_3D', log_num_blocks=16, hash_k=8)");

    // Version info
    m.attr("__version__") = "1.0.0";
}
