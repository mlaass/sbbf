/*
 * SBBF K-Sensitivity Benchmark
 *
 * Measures FPR as a function of k (hash functions) for SBBF variants.
 * Tests k = 2..12 with constant fill rate by adjusting memory per k.
 *
 * Configurations:
 * - SFC types: Morton2D, Morton3D, Hilbert2D, Hilbert3D
 * - Intra-block strategies: double_hash, pattern_lookup
 * - Seed strategies: XOR, MULTIPLY_SHIFT
 *
 * Output: JSON file with FPR for each configuration
 */

#include <sbbf/spatial_blocked_bloom_filter.hpp>
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <vector>

using json = nlohmann::json;
using namespace sbbf;

// ============================================================================
// Configuration
// ============================================================================

const size_t NUM_ELEMENTS = 100000;
const size_t NUM_QUERIES = 100000;
const double TARGET_FILL_RATE = 0.5;
const unsigned COORD_BITS = 16;  // Must be multiple of 4 for Hilbert
const uint32_t MAX_COORD = (1 << COORD_BITS) - 1;
const std::string OUTPUT_DIR = "./sbbf_results/k_sensitivity/";

// K values to test
const std::vector<unsigned> K_VALUES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// ============================================================================
// Data Structures
// ============================================================================

struct Point2D {
    uint32_t x, y;
};

struct Point3D {
    uint32_t x, y, z;
};

// ============================================================================
// Helper Functions
// ============================================================================

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Compute log_num_blocks to achieve target fill rate
// Fill rate = (n * k) / (num_blocks * bits_per_block)
// For target fill f: num_blocks = (n * k) / (f * bits_per_block)
unsigned compute_log_blocks(size_t n, unsigned k, double target_fill = 0.5, unsigned bits_per_block = 64) {
    double required_blocks = static_cast<double>(n * k) / (target_fill * bits_per_block);
    unsigned log_blocks = static_cast<unsigned>(std::ceil(std::log2(required_blocks)));
    return std::max(log_blocks, 10u);  // Minimum 2^10 = 1024 blocks
}

// ============================================================================
// Data Generation
// ============================================================================

std::vector<Point2D> generate_uniform_2d(size_t n, uint32_t max_coord, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, max_coord);
    std::vector<Point2D> points;
    points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        points.push_back({dist(rng), dist(rng)});
    }
    return points;
}

std::vector<Point3D> generate_uniform_3d(size_t n, uint32_t max_coord, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, max_coord);
    std::vector<Point3D> points;
    points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        points.push_back({dist(rng), dist(rng), dist(rng)});
    }
    return points;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

struct BenchResult {
    unsigned k;
    std::string sfc_type;
    std::string intra_strategy;
    std::string seed_strategy;
    unsigned dimensions;
    unsigned log_num_blocks;
    size_t memory_bytes;
    double actual_fill_rate;
    double fpr;
};

std::string sfc_type_to_string(SFCType sfc) {
    switch (sfc) {
        case SFCType::MORTON_2D: return "MORTON_2D";
        case SFCType::MORTON_3D: return "MORTON_3D";
        case SFCType::HILBERT_2D: return "HILBERT_2D";
        case SFCType::HILBERT_3D: return "HILBERT_3D";
    }
    return "UNKNOWN";
}

std::string intra_strategy_to_string(IntraBlockStrategy strategy) {
    switch (strategy) {
        case IntraBlockStrategy::DOUBLE_HASH: return "double_hash";
        case IntraBlockStrategy::PATTERN_LOOKUP: return "pattern_lookup";
        case IntraBlockStrategy::MULTIPLEXED: return "multiplexed";
    }
    return "unknown";
}

std::string seed_strategy_to_string(SeedStrategy strategy) {
    switch (strategy) {
        case SeedStrategy::XOR: return "XOR";
        case SeedStrategy::MULTIPLY_SHIFT: return "MULTIPLY_SHIFT";
    }
    return "unknown";
}

template<unsigned SFCBits>
BenchResult run_benchmark_2d(unsigned k, SFCType sfc_type,
                              IntraBlockStrategy intra_strategy,
                              SeedStrategy seed_strategy,
                              const std::vector<Point2D>& insert_data,
                              const std::vector<Point2D>& query_data) {
    unsigned log_blocks = compute_log_blocks(insert_data.size(), k, TARGET_FILL_RATE);

    SBBFConfig conf;
    conf.sfc_type = sfc_type;
    conf.sfc_bits = SFCBits;
    conf.log_num_blocks = log_blocks;
    conf.hash_k = k;
    conf.bits_per_block = 64;
    conf.intra_strategy = intra_strategy;
    conf.seed_strategy = seed_strategy;
    if (intra_strategy == IntraBlockStrategy::PATTERN_LOOKUP) {
        conf.pattern_table_size = 1024;
    }

    SpatialBlockedBloomFilter64<SFCBits> filter(conf);

    // Insert all elements
    for (const auto& p : insert_data) {
        filter.put2D(p.x, p.y);
    }

    // Measure FPR on query data (points not in filter)
    size_t false_positives = 0;
    for (const auto& p : query_data) {
        if (filter.get_bool_2D(p.x, p.y)) ++false_positives;
    }

    BenchResult result;
    result.k = k;
    result.sfc_type = sfc_type_to_string(sfc_type);
    result.intra_strategy = intra_strategy_to_string(intra_strategy);
    result.seed_strategy = seed_strategy_to_string(seed_strategy);
    result.dimensions = 2;
    result.log_num_blocks = log_blocks;
    result.memory_bytes = filter.memory_usage();
    result.actual_fill_rate = filter.block_fill_ratio();
    result.fpr = static_cast<double>(false_positives) / query_data.size();

    return result;
}

template<unsigned SFCBits>
BenchResult run_benchmark_3d(unsigned k, SFCType sfc_type,
                              IntraBlockStrategy intra_strategy,
                              SeedStrategy seed_strategy,
                              const std::vector<Point3D>& insert_data,
                              const std::vector<Point3D>& query_data) {
    unsigned log_blocks = compute_log_blocks(insert_data.size(), k, TARGET_FILL_RATE);

    SBBFConfig conf;
    conf.sfc_type = sfc_type;
    conf.sfc_bits = SFCBits;
    conf.log_num_blocks = log_blocks;
    conf.hash_k = k;
    conf.bits_per_block = 64;
    conf.intra_strategy = intra_strategy;
    conf.seed_strategy = seed_strategy;
    if (intra_strategy == IntraBlockStrategy::PATTERN_LOOKUP) {
        conf.pattern_table_size = 1024;
    }

    SpatialBlockedBloomFilter64<SFCBits> filter(conf);

    // Insert all elements
    for (const auto& p : insert_data) {
        filter.put3D(p.x, p.y, p.z);
    }

    // Measure FPR on query data
    size_t false_positives = 0;
    for (const auto& p : query_data) {
        if (filter.get_bool_3D(p.x, p.y, p.z)) ++false_positives;
    }

    BenchResult result;
    result.k = k;
    result.sfc_type = sfc_type_to_string(sfc_type);
    result.intra_strategy = intra_strategy_to_string(intra_strategy);
    result.seed_strategy = seed_strategy_to_string(seed_strategy);
    result.dimensions = 3;
    result.log_num_blocks = log_blocks;
    result.memory_bytes = filter.memory_usage();
    result.actual_fill_rate = filter.block_fill_ratio();
    result.fpr = static_cast<double>(false_positives) / query_data.size();

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SBBF K-Sensitivity Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Elements: " << NUM_ELEMENTS << "\n";
    std::cout << "Queries: " << NUM_QUERIES << "\n";
    std::cout << "Target fill rate: " << TARGET_FILL_RATE << "\n";
    std::cout << "Coord bits: " << COORD_BITS << "\n";
    std::cout << "K values: ";
    for (auto k : K_VALUES) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "========================================\n\n";

    ensure_dir("./sbbf_results");
    ensure_dir(OUTPUT_DIR);

    // Generate data
    std::cout << "Generating synthetic data...\n";
    auto insert_2d = generate_uniform_2d(NUM_ELEMENTS, MAX_COORD, 42);
    auto query_2d = generate_uniform_2d(NUM_QUERIES, MAX_COORD, 123);  // Different seed!
    auto insert_3d = generate_uniform_3d(NUM_ELEMENTS, MAX_COORD, 42);
    auto query_3d = generate_uniform_3d(NUM_QUERIES, MAX_COORD, 123);
    std::cout << "Done.\n\n";

    std::vector<BenchResult> results;

    // Configuration matrix
    std::vector<std::pair<SFCType, unsigned>> sfc_configs = {
        {SFCType::MORTON_2D, 2},
        {SFCType::HILBERT_2D, 2},
        {SFCType::MORTON_3D, 3},
        {SFCType::HILBERT_3D, 3}
    };
    std::vector<IntraBlockStrategy> intra_strategies = {
        IntraBlockStrategy::DOUBLE_HASH,
        IntraBlockStrategy::PATTERN_LOOKUP
    };
    std::vector<SeedStrategy> seed_strategies = {
        SeedStrategy::XOR,
        SeedStrategy::MULTIPLY_SHIFT
    };

    size_t total_configs = K_VALUES.size() * sfc_configs.size() *
                           intra_strategies.size() * seed_strategies.size();
    size_t current = 0;

    // Run all configurations
    for (auto k : K_VALUES) {
        for (const auto& [sfc_type, dim] : sfc_configs) {
            for (auto intra : intra_strategies) {
                for (auto seed : seed_strategies) {
                    ++current;
                    std::cout << "[" << current << "/" << total_configs << "] "
                              << "k=" << k
                              << " " << sfc_type_to_string(sfc_type)
                              << " " << intra_strategy_to_string(intra)
                              << " " << seed_strategy_to_string(seed)
                              << "... ";
                    std::cout.flush();

                    BenchResult result;
                    if (dim == 2) {
                        result = run_benchmark_2d<COORD_BITS>(
                            k, sfc_type, intra, seed, insert_2d, query_2d);
                    } else {
                        result = run_benchmark_3d<COORD_BITS>(
                            k, sfc_type, intra, seed, insert_3d, query_3d);
                    }

                    std::cout << "FPR=" << std::fixed << std::setprecision(4)
                              << (result.fpr * 100) << "%\n";

                    results.push_back(result);
                }
            }
        }
    }

    // Save results to JSON
    json output;
    output["experiment"] = "k_sensitivity";
    output["timestamp"] = get_iso_timestamp();
    output["config"] = {
        {"num_elements", NUM_ELEMENTS},
        {"num_queries", NUM_QUERIES},
        {"target_fill_rate", TARGET_FILL_RATE},
        {"coord_bits", COORD_BITS},
        {"k_values", K_VALUES}
    };

    json results_json = json::array();
    for (const auto& r : results) {
        results_json.push_back({
            {"config", {
                {"k", r.k},
                {"sfc_type", r.sfc_type},
                {"intra_strategy", r.intra_strategy},
                {"seed_strategy", r.seed_strategy},
                {"dimensions", r.dimensions},
                {"log_num_blocks", r.log_num_blocks}
            }},
            {"metrics", {
                {"fpr", r.fpr},
                {"actual_fill_rate", r.actual_fill_rate},
                {"memory_bytes", r.memory_bytes}
            }}
        });
    }
    output["results"] = results_json;

    std::string output_path = OUTPUT_DIR + "synthetic.json";
    std::ofstream out(output_path);
    out << output.dump(2);
    out.close();

    std::cout << "\n========================================\n";
    std::cout << "Results saved to: " << output_path << "\n";
    std::cout << "Total configurations: " << results.size() << "\n";
    std::cout << "========================================\n";

    return 0;
}
