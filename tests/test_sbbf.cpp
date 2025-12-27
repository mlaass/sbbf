/*
 * Unit tests for Spatial-Blocked Bloom Filter (SBBF)
 *
 * Tests:
 * - Configuration validation
 * - Basic put/get operations (2D and 3D)
 * - No false negatives guarantee
 * - FPR measurement
 * - Neighborhood queries
 * - Strategy comparison
 */

#include <sbbf/spatial_blocked_bloom_filter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace sbbf;

// Simple test framework
int test_count = 0;
int test_passed = 0;

#define TEST(name)                                        \
    void test_##name();                                   \
    void run_test_##name() {                              \
        test_count++;                                     \
        std::cout << "Running test: " << #name << "... "; \
        std::cout.flush();                                \
        try {                                             \
            test_##name();                                \
            test_passed++;                                \
            std::cout << "PASSED\n";                      \
        } catch (const std::exception &e) {               \
            std::cout << "FAILED: " << e.what() << "\n";  \
        } catch (...) {                                   \
            std::cout << "FAILED: Unknown exception\n";   \
        }                                                 \
    }                                                     \
    void test_##name()

#define ASSERT_TRUE(cond)                                 \
    if (!(cond))                                          \
        throw std::runtime_error("Assertion failed: " #cond)

#define ASSERT_FALSE(cond)                                \
    if (cond)                                             \
        throw std::runtime_error("Assertion failed: !" #cond)

#define ASSERT_EQ(a, b)                                   \
    if ((a) != (b))                                       \
        throw std::runtime_error("Assertion failed: " #a " == " #b)

#define ASSERT_NE(a, b)                                   \
    if ((a) == (b))                                       \
        throw std::runtime_error("Assertion failed: " #a " != " #b)

#define ASSERT_LT(a, b)                                   \
    if ((a) >= (b))                                       \
        throw std::runtime_error("Assertion failed: " #a " < " #b)

#define ASSERT_LE(a, b)                                   \
    if ((a) > (b))                                        \
        throw std::runtime_error("Assertion failed: " #a " <= " #b)

#define ASSERT_GT(a, b)                                   \
    if ((a) <= (b))                                       \
        throw std::runtime_error("Assertion failed: " #a " > " #b)

#define ASSERT_GE(a, b)                                   \
    if ((a) < (b))                                        \
        throw std::runtime_error("Assertion failed: " #a " >= " #b)

// ============================================================================
// Configuration Tests
// ============================================================================

TEST(config_validation_valid) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.sfc_bits = 16;
    conf.log_num_blocks = 10;
    conf.bits_per_block = 64;
    conf.hash_k = 4;
    conf.split_point_q = 12;
    conf.validate();  // Should not throw
}

TEST(config_validation_invalid_sfc_bits) {
    SBBFConfig conf;
    conf.sfc_bits = 0;  // Invalid
    try {
        conf.validate();
        ASSERT_TRUE(false);  // Should have thrown
    } catch (const std::invalid_argument &) {
        // Expected
    }
}

TEST(config_validation_invalid_block_bits) {
    SBBFConfig conf;
    conf.bits_per_block = 128;  // Invalid (must be 64, 256, or 512)
    try {
        conf.validate();
        ASSERT_TRUE(false);
    } catch (const std::invalid_argument &) {
        // Expected
    }
}

TEST(config_computed_memory) {
    SBBFConfig conf;
    conf.log_num_blocks = 10;  // 1024 blocks
    conf.bits_per_block = 64;   // 8 bytes per block

    ASSERT_EQ(conf.computed_num_blocks(), 1024ULL);
    ASSERT_EQ(conf.computed_memory_bytes(), 8192ULL);  // 1024 * 8
}

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST(put_get_2d_morton) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    // Insert a point
    bf.put2D(100, 200);

    // Should be found
    ASSERT_TRUE(bf.get_bool_2D(100, 200));

    // Non-inserted point should likely not be found (with some FP probability)
    // Just check it doesn't crash
    bf.get_bool_2D(999, 999);
}

TEST(put_get_2d_hilbert) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::HILBERT_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    bf.put2D(100, 200);
    ASSERT_TRUE(bf.get_bool_2D(100, 200));
}

TEST(put_get_3d_morton) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_3D;
    conf.sfc_bits = 10;  // Smaller for 3D to fit in split
    conf.log_num_blocks = 10;
    conf.hash_k = 4;
    conf.split_point_q = 10;

    SpatialBlockedBloomFilter64<10> bf(conf);

    bf.put3D(50, 60, 70);
    ASSERT_TRUE(bf.get_bool_3D(50, 60, 70));
}

TEST(put_get_3d_hilbert) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::HILBERT_3D;
    conf.sfc_bits = 8;
    conf.log_num_blocks = 8;
    conf.hash_k = 4;
    conf.split_point_q = 10;

    SpatialBlockedBloomFilter64<8> bf(conf);

    bf.put3D(50, 60, 70);
    ASSERT_TRUE(bf.get_bool_3D(50, 60, 70));
}

TEST(put_get_vector_interface) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    std::vector<uint64_t> point = {100, 200};
    bf.put(point);
    ASSERT_TRUE(bf.get_bool(point));
    ASSERT_EQ(bf.get_min(point), 1ULL);
}

// ============================================================================
// No False Negatives Tests
// ============================================================================

TEST(no_false_negatives_2d) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 14;  // 16K blocks
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    std::mt19937 rng(42);
    std::vector<std::pair<uint32_t, uint32_t>> inserted;

    // Insert 1000 random points
    for (int i = 0; i < 1000; ++i) {
        uint32_t x = rng() % 10000;
        uint32_t y = rng() % 10000;
        bf.put2D(x, y);
        inserted.emplace_back(x, y);
    }

    // All inserted points must be found
    for (const auto &p : inserted) {
        ASSERT_TRUE(bf.get_bool_2D(p.first, p.second));
    }
}

TEST(no_false_negatives_3d) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_3D;
    conf.sfc_bits = 10;
    conf.log_num_blocks = 12;
    conf.hash_k = 4;
    conf.split_point_q = 10;

    SpatialBlockedBloomFilter64<10> bf(conf);

    std::mt19937 rng(42);
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> inserted;

    for (int i = 0; i < 500; ++i) {
        uint32_t x = rng() % 500;
        uint32_t y = rng() % 500;
        uint32_t z = rng() % 500;
        bf.put3D(x, y, z);
        inserted.emplace_back(x, y, z);
    }

    for (const auto &p : inserted) {
        ASSERT_TRUE(bf.get_bool_3D(std::get<0>(p), std::get<1>(p), std::get<2>(p)));
    }
}

// ============================================================================
// FPR Tests
// ============================================================================

TEST(fpr_2d_reasonable) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 14;  // 16K blocks
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    std::mt19937 rng(42);

    // Insert 1000 points in region [0, 5000)
    for (int i = 0; i < 1000; ++i) {
        uint32_t x = rng() % 5000;
        uint32_t y = rng() % 5000;
        bf.put2D(x, y);
    }

    // Query 10000 points in different region [10000, 15000)
    int false_positives = 0;
    int queries = 10000;
    for (int i = 0; i < queries; ++i) {
        uint32_t x = 10000 + (rng() % 5000);
        uint32_t y = 10000 + (rng() % 5000);
        if (bf.get_bool_2D(x, y)) {
            ++false_positives;
        }
    }

    double fpr = static_cast<double>(false_positives) / queries;
    std::cout << "(FPR=" << fpr << ") ";

    // FPR should be reasonable (< 20% for this configuration)
    ASSERT_LT(fpr, 0.20);
}

// ============================================================================
// Intra-Block Strategy Tests
// ============================================================================

TEST(double_hash_strategy) {
    SBBFConfig conf;
    conf.intra_strategy = IntraBlockStrategy::DOUBLE_HASH;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    for (int i = 0; i < 100; ++i) {
        bf.put2D(i * 10, i * 20);
    }

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(bf.get_bool_2D(i * 10, i * 20));
    }
}

TEST(pattern_lookup_strategy) {
    SBBFConfig conf;
    conf.intra_strategy = IntraBlockStrategy::PATTERN_LOOKUP;
    conf.pattern_table_size = 256;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    for (int i = 0; i < 100; ++i) {
        bf.put2D(i * 10, i * 20);
    }

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(bf.get_bool_2D(i * 10, i * 20));
    }
}

TEST(multiplexed_strategy) {
    SBBFConfig conf;
    conf.intra_strategy = IntraBlockStrategy::MULTIPLEXED;
    conf.multiplex_count = 2;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    for (int i = 0; i < 100; ++i) {
        bf.put2D(i * 10, i * 20);
    }

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(bf.get_bool_2D(i * 10, i * 20));
    }
}

// ============================================================================
// Neighborhood Query Tests
// ============================================================================

TEST(neighborhood_query_2d_simple) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    // Insert center and some neighbors
    bf.put2D(100, 100);  // center
    bf.put2D(99, 100);   // left
    bf.put2D(101, 100);  // right
    bf.put2D(100, 99);   // up

    // Query 3x3 neighborhood
    uint64_t result = bf.query_neighborhood_2D(100, 100, 1);

    // Center should be present (bit 4 in row-major 3x3)
    ASSERT_TRUE(result & (1ULL << 4));

    // At least some neighbors should be found
    int neighbors_found = __builtin_popcountll(result);
    ASSERT_GE(neighbors_found, 4);  // At least center + 3 neighbors
}

TEST(neighborhood_query_2d_radius) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    // Insert center only
    bf.put2D(100, 100);

    // Query with radius 1 (3x3)
    uint64_t result1 = bf.query_neighborhood_2D(100, 100, 1);
    ASSERT_TRUE(result1 & (1ULL << 4));  // Center bit

    // Query with radius 2 (5x5)
    uint64_t result2 = bf.query_neighborhood_2D(100, 100, 2);
    ASSERT_TRUE(result2 & (1ULL << 12));  // Center bit in 5x5
}

TEST(neighborhood_query_3d) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_3D;
    conf.sfc_bits = 10;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;
    conf.split_point_q = 10;

    SpatialBlockedBloomFilter64<10> bf(conf);

    // Insert center and all 6 face neighbors
    bf.put3D(100, 100, 100);
    bf.put3D(99, 100, 100);
    bf.put3D(101, 100, 100);
    bf.put3D(100, 99, 100);
    bf.put3D(100, 101, 100);
    bf.put3D(100, 100, 99);
    bf.put3D(100, 100, 101);

    // Query 6-connected neighborhood
    unsigned count6 = bf.query_neighborhood_3D(100, 100, 100, false);
    ASSERT_EQ(count6, 6u);  // All 6 neighbors

    // Query 26-connected neighborhood
    unsigned count26 = bf.query_neighborhood_3D(100, 100, 100, true);
    ASSERT_GE(count26, 6u);  // At least the 6 face neighbors
}

TEST(has_any_neighbor_2d) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    // Insert only center
    bf.put2D(100, 100);

    // Center has no neighbors
    ASSERT_FALSE(bf.has_any_neighbor_2D(100, 100, 1));

    // Insert a neighbor
    bf.put2D(101, 100);

    // Now center has a neighbor
    ASSERT_TRUE(bf.has_any_neighbor_2D(100, 100, 1));
}

// ============================================================================
// SFC Type Comparison Tests
// ============================================================================

TEST(morton_vs_hilbert_2d_locality) {
    // Both should work, we just compare block indices
    SBBFConfig conf_m, conf_h;
    conf_m.sfc_type = SFCType::MORTON_2D;
    conf_h.sfc_type = SFCType::HILBERT_2D;
    conf_m.log_num_blocks = 10;
    conf_h.log_num_blocks = 10;

    SpatialBlockedBloomFilter64<16> bf_m(conf_m);
    SpatialBlockedBloomFilter64<16> bf_h(conf_h);

    // Insert same points
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            bf_m.put2D(x, y);
            bf_h.put2D(x, y);
        }
    }

    // Both should find all points
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            ASSERT_TRUE(bf_m.get_bool_2D(x, y));
            ASSERT_TRUE(bf_h.get_bool_2D(x, y));
        }
    }
}

// ============================================================================
// Utility Tests
// ============================================================================

TEST(memory_usage_accurate) {
    SBBFConfig conf;
    conf.log_num_blocks = 10;  // 1024 blocks
    conf.bits_per_block = 64;   // 8 bytes each

    SpatialBlockedBloomFilter64<16> bf(conf);

    uint64_t mem = bf.memory_usage();
    ASSERT_GE(mem, 1024 * 8);  // At least the blocks
}

TEST(summary_format) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 10;

    SpatialBlockedBloomFilter64<16> bf(conf);

    std::string summary = bf.summary();
    ASSERT_TRUE(summary.find("SpatialBlockedBloomFilter64") != std::string::npos);
    ASSERT_TRUE(summary.find("MORTON_2D") != std::string::npos);
}

TEST(clear_resets_all) {
    SBBFConfig conf;
    conf.log_num_blocks = 10;
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    bf.put2D(100, 200);
    ASSERT_TRUE(bf.get_bool_2D(100, 200));

    bf.clear();
    ASSERT_FALSE(bf.get_bool_2D(100, 200));
}

TEST(block_fill_ratio) {
    SBBFConfig conf;
    conf.log_num_blocks = 10;  // 1024 blocks
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    // Initially empty
    ASSERT_EQ(bf.block_fill_ratio(), 0.0);

    // Insert some points
    for (int i = 0; i < 100; ++i) {
        bf.put2D(i, i);
    }

    // Some blocks should be filled
    double ratio = bf.block_fill_ratio();
    ASSERT_GT(ratio, 0.0);
    ASSERT_LE(ratio, 1.0);
}

// ============================================================================
// Performance Test (informational)
// ============================================================================

TEST(performance_insert_query) {
    SBBFConfig conf;
    conf.sfc_type = SFCType::MORTON_2D;
    conf.log_num_blocks = 16;  // 64K blocks
    conf.hash_k = 4;

    SpatialBlockedBloomFilter64<16> bf(conf);

    constexpr int N = 100000;
    std::mt19937 rng(42);
    std::vector<std::pair<uint32_t, uint32_t>> points;
    points.reserve(N);
    for (int i = 0; i < N; ++i) {
        points.emplace_back(rng() % 50000, rng() % 50000);
    }

    // Measure insert time
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &p : points) {
        bf.put2D(p.first, p.second);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double insert_ns = std::chrono::duration<double, std::nano>(end - start).count() / N;

    // Measure query time
    start = std::chrono::high_resolution_clock::now();
    volatile int found = 0;
    for (const auto &p : points) {
        if (bf.get_bool_2D(p.first, p.second)) ++found;
    }
    end = std::chrono::high_resolution_clock::now();
    double query_ns = std::chrono::duration<double, std::nano>(end - start).count() / N;

    std::cout << "(insert=" << insert_ns << "ns, query=" << query_ns << "ns) ";

    // All should be found
    ASSERT_EQ(found, N);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n===== Spatial-Blocked Bloom Filter Unit Tests =====\n\n";

    // Configuration
    run_test_config_validation_valid();
    run_test_config_validation_invalid_sfc_bits();
    run_test_config_validation_invalid_block_bits();
    run_test_config_computed_memory();

    // Basic operations
    run_test_put_get_2d_morton();
    run_test_put_get_2d_hilbert();
    run_test_put_get_3d_morton();
    run_test_put_get_3d_hilbert();
    run_test_put_get_vector_interface();

    // No false negatives
    run_test_no_false_negatives_2d();
    run_test_no_false_negatives_3d();

    // FPR
    run_test_fpr_2d_reasonable();

    // Strategies
    run_test_double_hash_strategy();
    run_test_pattern_lookup_strategy();
    run_test_multiplexed_strategy();

    // Neighborhood queries
    run_test_neighborhood_query_2d_simple();
    run_test_neighborhood_query_2d_radius();
    run_test_neighborhood_query_3d();
    run_test_has_any_neighbor_2d();

    // SFC comparison
    run_test_morton_vs_hilbert_2d_locality();

    // Utility
    run_test_memory_usage_accurate();
    run_test_summary_format();
    run_test_clear_resets_all();
    run_test_block_fill_ratio();

    // Performance
    run_test_performance_insert_query();

    std::cout << "\n===== Test Summary =====\n";
    std::cout << "Tests passed: " << test_passed << "/" << test_count << "\n";

    if (test_passed == test_count) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED!\n";
        return 1;
    }
}
