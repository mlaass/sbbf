/*
 * SBBF Seed Strategy Benchmark
 *
 * Compares XOR vs MULTIPLY_SHIFT seed strategies across multiple datasets:
 * - Synthetic 2D: 100K uniform random points
 * - Synthetic 3D: 100K uniform random points
 * - Clustered 2D/3D: 100K clustered points
 * - GDELT 2D: Real-world geographic events (requires HDF5)
 *
 * Metrics: FPR, insert latency (ns/op), query latency (ns/op)
 *
 * Output: Separate JSON files for each dataset type
 */

#include <sbbf/spatial_blocked_bloom_filter.hpp>
#include "benchmark_config.hpp"
#include "json.hpp"

#ifdef SBBF_HAS_HDF5
#include <highfive/H5File.hpp>
#endif

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

const size_t SYNTHETIC_ELEMENTS = 100000;
const size_t SYNTHETIC_QUERIES = 100000;
const double BITS_PER_ELEMENT = 10.0;  // Standard BF sizing
const unsigned COORD_BITS = 16;  // Must be multiple of 4 for Hilbert
const uint32_t MAX_COORD = (1 << COORD_BITS) - 1;
const unsigned HASH_K = SBBF_BENCHMARK_HASH_K;  // From benchmark_config.hpp
const std::string OUTPUT_DIR = "./sbbf_results/seed_strategy/";
const std::string GDELT_PATH = "./datasets/hdf5/gdelt_events.h5";

// ============================================================================
// Data Structures
// ============================================================================

struct Point2D {
    uint32_t x, y;
};

struct Point3D {
    uint32_t x, y, z;
};

struct BenchResult {
    std::string sfc_type;
    std::string intra_strategy;
    std::string seed_strategy;
    unsigned dimensions;
    unsigned log_num_blocks;
    size_t memory_bytes;
    double actual_fill_rate;
    double fpr;
    double insert_ns;  // nanoseconds per insert
    double query_ns;   // nanoseconds per query
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

// Compute log_num_blocks for target bits per element
unsigned compute_log_blocks(size_t n, double bits_per_element, unsigned bits_per_block = 64) {
    double required_blocks = static_cast<double>(n) * bits_per_element / bits_per_block;
    unsigned log_blocks = static_cast<unsigned>(std::ceil(std::log2(required_blocks)));
    return std::max(log_blocks, 10u);  // Minimum 2^10 = 1024 blocks
}

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

// Generate clustered 2D data (Gaussian clusters)
std::vector<Point2D> generate_clustered_2d(size_t n, uint32_t max_coord, uint64_t seed,
                                            size_t n_clusters = 50, double sigma = 500.0) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> center_dist(0, max_coord);
    std::uniform_int_distribution<size_t> cluster_dist(0, n_clusters - 1);
    std::normal_distribution<double> gaussian(0.0, sigma);

    // Generate cluster centers
    std::vector<std::pair<double, double>> centers(n_clusters);
    for (size_t i = 0; i < n_clusters; ++i) {
        centers[i] = {static_cast<double>(center_dist(rng)), static_cast<double>(center_dist(rng))};
    }

    // Generate points around clusters
    std::vector<Point2D> points;
    points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        size_t cluster_idx = cluster_dist(rng);
        double x = centers[cluster_idx].first + gaussian(rng);
        double y = centers[cluster_idx].second + gaussian(rng);
        // Clamp to valid range
        x = std::max(0.0, std::min(static_cast<double>(max_coord), x));
        y = std::max(0.0, std::min(static_cast<double>(max_coord), y));
        points.push_back({static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
    }
    return points;
}

// Generate clustered 3D data (Gaussian clusters)
std::vector<Point3D> generate_clustered_3d(size_t n, uint32_t max_coord, uint64_t seed,
                                            size_t n_clusters = 50, double sigma = 500.0) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> center_dist(0, max_coord);
    std::uniform_int_distribution<size_t> cluster_dist(0, n_clusters - 1);
    std::normal_distribution<double> gaussian(0.0, sigma);

    // Generate cluster centers
    std::vector<std::tuple<double, double, double>> centers(n_clusters);
    for (size_t i = 0; i < n_clusters; ++i) {
        centers[i] = {static_cast<double>(center_dist(rng)),
                      static_cast<double>(center_dist(rng)),
                      static_cast<double>(center_dist(rng))};
    }

    // Generate points around clusters
    std::vector<Point3D> points;
    points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        size_t cluster_idx = cluster_dist(rng);
        double x = std::get<0>(centers[cluster_idx]) + gaussian(rng);
        double y = std::get<1>(centers[cluster_idx]) + gaussian(rng);
        double z = std::get<2>(centers[cluster_idx]) + gaussian(rng);
        // Clamp to valid range
        x = std::max(0.0, std::min(static_cast<double>(max_coord), x));
        y = std::max(0.0, std::min(static_cast<double>(max_coord), y));
        z = std::max(0.0, std::min(static_cast<double>(max_coord), z));
        points.push_back({static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)});
    }
    return points;
}

#ifdef SBBF_HAS_HDF5
// Load GDELT data from HDF5
std::vector<Point2D> load_gdelt(const std::string& path, uint32_t width, uint32_t height) {
    using namespace HighFive;

    std::vector<Point2D> points;

    try {
        File file(path, File::ReadOnly);
        DataSet dataset = file.getDataSet("coords");
        auto shape = dataset.getDimensions();
        size_t total_points = shape[0];

        std::cout << "  Loading " << total_points << " GDELT points...\n";

        // Load in batches
        const size_t batch_size = 8192;
        std::vector<std::vector<double>> batch_data;
        points.reserve(total_points);

        for (size_t start = 0; start < total_points; start += batch_size) {
            size_t count = std::min(batch_size, total_points - start);
            dataset.select({start, 0}, {count, 2}).read(batch_data);

            for (const auto& p : batch_data) {
                // Convert lat/lon to grid coordinates
                // lon: [-180, 180] -> [0, width]
                // lat: [-90, 90] -> [0, height]
                double x = static_cast<double>(width) * ((p[0] + 180.0) / 360.0);
                double y = static_cast<double>(height) * ((p[1] + 90.0) / 180.0);
                points.push_back({static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
            }
        }

        std::cout << "  Loaded " << points.size() << " points.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading GDELT: " << e.what() << "\n";
    }

    return points;
}
#endif

// ============================================================================
// Benchmark Functions
// ============================================================================

template<unsigned SFCBits>
BenchResult run_benchmark_2d(SFCType sfc_type,
                              IntraBlockStrategy intra_strategy,
                              SeedStrategy seed_strategy,
                              const std::vector<Point2D>& insert_data,
                              const std::vector<Point2D>& query_data) {
    using namespace std::chrono;

    unsigned log_blocks = compute_log_blocks(insert_data.size(), BITS_PER_ELEMENT);

    SBBFConfig conf;
    conf.sfc_type = sfc_type;
    conf.sfc_bits = SFCBits;
    conf.log_num_blocks = log_blocks;
    conf.hash_k = HASH_K;
    conf.bits_per_block = 64;
    conf.intra_strategy = intra_strategy;
    conf.seed_strategy = seed_strategy;
    if (intra_strategy == IntraBlockStrategy::PATTERN_LOOKUP) {
        conf.pattern_table_size = 1024;
    }

    SpatialBlockedBloomFilter64<SFCBits> filter(conf);

    // Measure insert time
    auto t1 = high_resolution_clock::now();
    for (const auto& p : insert_data) {
        filter.put2D(p.x, p.y);
    }
    auto t2 = high_resolution_clock::now();
    double insert_ns = duration<double, std::nano>(t2 - t1).count() / insert_data.size();

    // Measure query time and FPR
    size_t false_positives = 0;
    auto t3 = high_resolution_clock::now();
    for (const auto& p : query_data) {
        if (filter.get_bool_2D(p.x, p.y)) ++false_positives;
    }
    auto t4 = high_resolution_clock::now();
    double query_ns = duration<double, std::nano>(t4 - t3).count() / query_data.size();

    BenchResult result;
    result.sfc_type = sfc_type_to_string(sfc_type);
    result.intra_strategy = intra_strategy_to_string(intra_strategy);
    result.seed_strategy = seed_strategy_to_string(seed_strategy);
    result.dimensions = 2;
    result.log_num_blocks = log_blocks;
    result.memory_bytes = filter.memory_usage();
    result.actual_fill_rate = filter.block_fill_ratio();
    result.fpr = static_cast<double>(false_positives) / query_data.size();
    result.insert_ns = insert_ns;
    result.query_ns = query_ns;

    return result;
}

template<unsigned SFCBits>
BenchResult run_benchmark_3d(SFCType sfc_type,
                              IntraBlockStrategy intra_strategy,
                              SeedStrategy seed_strategy,
                              const std::vector<Point3D>& insert_data,
                              const std::vector<Point3D>& query_data) {
    using namespace std::chrono;

    unsigned log_blocks = compute_log_blocks(insert_data.size(), BITS_PER_ELEMENT);

    SBBFConfig conf;
    conf.sfc_type = sfc_type;
    conf.sfc_bits = SFCBits;
    conf.log_num_blocks = log_blocks;
    conf.hash_k = HASH_K;
    conf.bits_per_block = 64;
    conf.intra_strategy = intra_strategy;
    conf.seed_strategy = seed_strategy;
    if (intra_strategy == IntraBlockStrategy::PATTERN_LOOKUP) {
        conf.pattern_table_size = 1024;
    }

    SpatialBlockedBloomFilter64<SFCBits> filter(conf);

    // Measure insert time
    auto t1 = high_resolution_clock::now();
    for (const auto& p : insert_data) {
        filter.put3D(p.x, p.y, p.z);
    }
    auto t2 = high_resolution_clock::now();
    double insert_ns = duration<double, std::nano>(t2 - t1).count() / insert_data.size();

    // Measure query time and FPR
    size_t false_positives = 0;
    auto t3 = high_resolution_clock::now();
    for (const auto& p : query_data) {
        if (filter.get_bool_3D(p.x, p.y, p.z)) ++false_positives;
    }
    auto t4 = high_resolution_clock::now();
    double query_ns = duration<double, std::nano>(t4 - t3).count() / query_data.size();

    BenchResult result;
    result.sfc_type = sfc_type_to_string(sfc_type);
    result.intra_strategy = intra_strategy_to_string(intra_strategy);
    result.seed_strategy = seed_strategy_to_string(seed_strategy);
    result.dimensions = 3;
    result.log_num_blocks = log_blocks;
    result.memory_bytes = filter.memory_usage();
    result.actual_fill_rate = filter.block_fill_ratio();
    result.fpr = static_cast<double>(false_positives) / query_data.size();
    result.insert_ns = insert_ns;
    result.query_ns = query_ns;

    return result;
}

// ============================================================================
// JSON Output
// ============================================================================

void save_results(const std::vector<BenchResult>& results,
                  const std::string& dataset_name,
                  size_t num_elements,
                  size_t num_queries,
                  const std::string& output_path) {
    json output;
    output["experiment"] = "seed_strategy";
    output["dataset"] = dataset_name;
    output["timestamp"] = get_iso_timestamp();
    output["config"] = {
        {"num_elements", num_elements},
        {"num_queries", num_queries},
        {"bits_per_element", BITS_PER_ELEMENT},
        {"coord_bits", COORD_BITS},
        {"hash_k", HASH_K}
    };

    json results_json = json::array();
    for (const auto& r : results) {
        results_json.push_back({
            {"config", {
                {"sfc_type", r.sfc_type},
                {"intra_strategy", r.intra_strategy},
                {"seed_strategy", r.seed_strategy},
                {"dimensions", r.dimensions},
                {"log_num_blocks", r.log_num_blocks}
            }},
            {"metrics", {
                {"fpr", r.fpr},
                {"actual_fill_rate", r.actual_fill_rate},
                {"memory_bytes", r.memory_bytes},
                {"insert_ns", r.insert_ns},
                {"query_ns", r.query_ns}
            }}
        });
    }
    output["results"] = results_json;

    std::ofstream out(output_path);
    out << output.dump(2);
    out.close();

    std::cout << "  Saved to: " << output_path << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SBBF Seed Strategy Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Comparing XOR vs MULTIPLY_SHIFT\n";
    std::cout << "Hash k: " << HASH_K << "\n";
    std::cout << "Bits per element: " << BITS_PER_ELEMENT << "\n";
    std::cout << "========================================\n\n";

    ensure_dir("./sbbf_results");
    ensure_dir(OUTPUT_DIR);

    // Configuration matrix
    std::vector<IntraBlockStrategy> intra_strategies = {
        IntraBlockStrategy::DOUBLE_HASH,
        IntraBlockStrategy::PATTERN_LOOKUP
    };
    std::vector<SeedStrategy> seed_strategies = {
        SeedStrategy::XOR,
        SeedStrategy::MULTIPLY_SHIFT
    };

    // ========================================================================
    // Synthetic 2D
    // ========================================================================
    std::cout << "----------------------------------------\n";
    std::cout << "Synthetic 2D Benchmark\n";
    std::cout << "----------------------------------------\n";

    auto insert_2d = generate_uniform_2d(SYNTHETIC_ELEMENTS, MAX_COORD, 42);
    auto query_2d = generate_uniform_2d(SYNTHETIC_QUERIES, MAX_COORD, 123);

    std::vector<BenchResult> results_2d;

    // Morton 2D and Hilbert 2D
    std::vector<SFCType> sfc_types_2d = {SFCType::MORTON_2D, SFCType::HILBERT_2D};

    for (auto sfc_type : sfc_types_2d) {
        for (auto intra : intra_strategies) {
            for (auto seed : seed_strategies) {
                std::cout << "  " << sfc_type_to_string(sfc_type)
                          << " " << intra_strategy_to_string(intra)
                          << " " << seed_strategy_to_string(seed) << "... ";
                std::cout.flush();

                auto result = run_benchmark_2d<COORD_BITS>(
                    sfc_type, intra, seed, insert_2d, query_2d);

                std::cout << "FPR=" << std::fixed << std::setprecision(4)
                          << (result.fpr * 100) << "%, "
                          << "insert=" << std::fixed << std::setprecision(1)
                          << result.insert_ns << "ns, "
                          << "query=" << result.query_ns << "ns\n";

                results_2d.push_back(result);
            }
        }
    }

    save_results(results_2d, "synthetic_2d", SYNTHETIC_ELEMENTS, SYNTHETIC_QUERIES,
                 OUTPUT_DIR + "synthetic_2d.json");

    // ========================================================================
    // Synthetic 3D
    // ========================================================================
    std::cout << "\n----------------------------------------\n";
    std::cout << "Synthetic 3D Benchmark\n";
    std::cout << "----------------------------------------\n";

    auto insert_3d = generate_uniform_3d(SYNTHETIC_ELEMENTS, MAX_COORD, 42);
    auto query_3d = generate_uniform_3d(SYNTHETIC_QUERIES, MAX_COORD, 123);

    std::vector<BenchResult> results_3d;

    // Morton 3D and Hilbert 3D
    std::vector<SFCType> sfc_types_3d = {SFCType::MORTON_3D, SFCType::HILBERT_3D};

    for (auto sfc_type : sfc_types_3d) {
        for (auto intra : intra_strategies) {
            for (auto seed : seed_strategies) {
                std::cout << "  " << sfc_type_to_string(sfc_type)
                          << " " << intra_strategy_to_string(intra)
                          << " " << seed_strategy_to_string(seed) << "... ";
                std::cout.flush();

                auto result = run_benchmark_3d<COORD_BITS>(
                    sfc_type, intra, seed, insert_3d, query_3d);

                std::cout << "FPR=" << std::fixed << std::setprecision(4)
                          << (result.fpr * 100) << "%, "
                          << "insert=" << std::fixed << std::setprecision(1)
                          << result.insert_ns << "ns, "
                          << "query=" << result.query_ns << "ns\n";

                results_3d.push_back(result);
            }
        }
    }

    save_results(results_3d, "synthetic_3d", SYNTHETIC_ELEMENTS, SYNTHETIC_QUERIES,
                 OUTPUT_DIR + "synthetic_3d.json");

    // ========================================================================
    // Clustered 2D
    // ========================================================================
    std::cout << "\n----------------------------------------\n";
    std::cout << "Clustered 2D Benchmark (50 Gaussian clusters)\n";
    std::cout << "----------------------------------------\n";

    auto insert_clustered_2d = generate_clustered_2d(SYNTHETIC_ELEMENTS, MAX_COORD, 42);
    auto query_clustered_2d = generate_clustered_2d(SYNTHETIC_QUERIES, MAX_COORD, 123);

    std::vector<BenchResult> results_clustered_2d;

    for (auto sfc_type : sfc_types_2d) {
        for (auto intra : intra_strategies) {
            for (auto seed : seed_strategies) {
                std::cout << "  " << sfc_type_to_string(sfc_type)
                          << " " << intra_strategy_to_string(intra)
                          << " " << seed_strategy_to_string(seed) << "... ";
                std::cout.flush();

                auto result = run_benchmark_2d<COORD_BITS>(
                    sfc_type, intra, seed, insert_clustered_2d, query_clustered_2d);

                std::cout << "FPR=" << std::fixed << std::setprecision(4)
                          << (result.fpr * 100) << "%, "
                          << "insert=" << std::fixed << std::setprecision(1)
                          << result.insert_ns << "ns, "
                          << "query=" << result.query_ns << "ns\n";

                results_clustered_2d.push_back(result);
            }
        }
    }

    save_results(results_clustered_2d, "clustered_2d", SYNTHETIC_ELEMENTS, SYNTHETIC_QUERIES,
                 OUTPUT_DIR + "clustered_2d.json");

    // ========================================================================
    // Clustered 3D
    // ========================================================================
    std::cout << "\n----------------------------------------\n";
    std::cout << "Clustered 3D Benchmark (50 Gaussian clusters)\n";
    std::cout << "----------------------------------------\n";

    auto insert_clustered_3d = generate_clustered_3d(SYNTHETIC_ELEMENTS, MAX_COORD, 42);
    auto query_clustered_3d = generate_clustered_3d(SYNTHETIC_QUERIES, MAX_COORD, 123);

    std::vector<BenchResult> results_clustered_3d;

    for (auto sfc_type : sfc_types_3d) {
        for (auto intra : intra_strategies) {
            for (auto seed : seed_strategies) {
                std::cout << "  " << sfc_type_to_string(sfc_type)
                          << " " << intra_strategy_to_string(intra)
                          << " " << seed_strategy_to_string(seed) << "... ";
                std::cout.flush();

                auto result = run_benchmark_3d<COORD_BITS>(
                    sfc_type, intra, seed, insert_clustered_3d, query_clustered_3d);

                std::cout << "FPR=" << std::fixed << std::setprecision(4)
                          << (result.fpr * 100) << "%, "
                          << "insert=" << std::fixed << std::setprecision(1)
                          << result.insert_ns << "ns, "
                          << "query=" << result.query_ns << "ns\n";

                results_clustered_3d.push_back(result);
            }
        }
    }

    save_results(results_clustered_3d, "clustered_3d", SYNTHETIC_ELEMENTS, SYNTHETIC_QUERIES,
                 OUTPUT_DIR + "clustered_3d.json");

#ifdef SBBF_HAS_HDF5
    // ========================================================================
    // GDELT 2D
    // ========================================================================
    std::cout << "\n----------------------------------------\n";
    std::cout << "GDELT 2D Benchmark\n";
    std::cout << "----------------------------------------\n";

    // Check if GDELT dataset exists
    struct stat buffer;
    if (stat(GDELT_PATH.c_str(), &buffer) != 0) {
        std::cout << "  Warning: GDELT dataset not found at " << GDELT_PATH << "\n";
        std::cout << "  Skipping GDELT benchmark.\n";
    } else {
        // Load GDELT data (use full coordinate space)
        auto gdelt_data = load_gdelt(GDELT_PATH, MAX_COORD, MAX_COORD);

        if (!gdelt_data.empty()) {
            // Generate query points (random coordinates not in dataset)
            auto gdelt_query = generate_uniform_2d(SYNTHETIC_QUERIES, MAX_COORD, 456);

            std::vector<BenchResult> results_gdelt;

            for (auto sfc_type : sfc_types_2d) {
                for (auto intra : intra_strategies) {
                    for (auto seed : seed_strategies) {
                        std::cout << "  " << sfc_type_to_string(sfc_type)
                                  << " " << intra_strategy_to_string(intra)
                                  << " " << seed_strategy_to_string(seed) << "... ";
                        std::cout.flush();

                        auto result = run_benchmark_2d<COORD_BITS>(
                            sfc_type, intra, seed, gdelt_data, gdelt_query);

                        std::cout << "FPR=" << std::fixed << std::setprecision(4)
                                  << (result.fpr * 100) << "%, "
                                  << "insert=" << std::fixed << std::setprecision(1)
                                  << result.insert_ns << "ns, "
                                  << "query=" << result.query_ns << "ns\n";

                        results_gdelt.push_back(result);
                    }
                }
            }

            save_results(results_gdelt, "gdelt_2d", gdelt_data.size(), gdelt_query.size(),
                         OUTPUT_DIR + "gdelt_2d.json");
        }
    }
#else
    std::cout << "\n----------------------------------------\n";
    std::cout << "GDELT 2D Benchmark (HDF5 support disabled)\n";
    std::cout << "----------------------------------------\n";
    std::cout << "  Build with -DSBBF_WITH_HDF5=ON to enable.\n";
#endif

    std::cout << "\n========================================\n";
    std::cout << "Seed Strategy Benchmark Complete!\n";
    std::cout << "Results saved to: " << OUTPUT_DIR << "\n";
    std::cout << "========================================\n";

    return 0;
}
