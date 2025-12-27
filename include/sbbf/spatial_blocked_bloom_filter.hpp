/*
 * Spatial-Blocked Bloom Filter (SBBF)
 *
 * A bloom filter variant that uses Space-Filling Curves (SFC) for block indexing
 * instead of hash functions. This preserves spatial locality, making neighborhood
 * queries significantly faster due to cache coherence.
 *
 * Key innovation: block_idx = SFC(x, y, z) & (N-1)  (low bits for sequential access)
 *
 * References:
 * - Putze et al. (2009): Cache-, Hash-, and Space-Efficient Bloom Filters
 * - Lang et al. (2019): Performance-Optimal Filtering
 */

#ifndef SBBF_SPATIAL_BLOCKED_BLOOM_FILTER_HPP
#define SBBF_SPATIAL_BLOCKED_BLOOM_FILTER_HPP

#include "sbbf/space_filling_curves.hpp"
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbbf {

// ============================================================================
// Enumerations
// ============================================================================

/// Space-filling curve type for block indexing
enum class SFCType {
    MORTON_2D,
    MORTON_3D,
    HILBERT_2D,
    HILBERT_3D
};

/// Intra-block hashing strategy
enum class IntraBlockStrategy {
    DOUBLE_HASH,     // h_i = (h1 + i*h2) % B
    PATTERN_LOOKUP,  // Pre-computed k-bit patterns
    MULTIPLEXED      // OR x patterns of k/x bits each
};

/// Seed derivation strategy for intra-block hashing
enum class SeedStrategy {
    XOR,            // (sfc >> log_blocks) ^ (sfc & mask) - combines high and low bits
    MULTIPLY_SHIFT  // (sfc * prime) >> shift - mixes all bits via multiplication
};

// ============================================================================
// Configuration
// ============================================================================

struct SBBFConfig {
    /// Space-filling curve type
    SFCType sfc_type = SFCType::MORTON_2D;

    /// Bits per coordinate in SFC encoding (max resolution)
    unsigned sfc_bits = 16;

    /// Log2 of number of blocks (num_blocks = 2^log_num_blocks)
    unsigned log_num_blocks = 10;

    /// Bits per block (64, 256, 512, etc.)
    unsigned bits_per_block = 64;

    /// Number of bits set per element within a block
    unsigned hash_k = 4;

    /// Intra-block hashing strategy
    IntraBlockStrategy intra_strategy = IntraBlockStrategy::DOUBLE_HASH;

    /// Unused legacy parameter (kept for config compatibility)
    /// The split now happens at log_num_blocks: low bits -> block index, high bits -> seed
    unsigned split_point_q = 12;

    /// Pattern table size for PATTERN_LOOKUP strategy
    unsigned pattern_table_size = 1024;

    /// Number of patterns to OR for MULTIPLEXED strategy
    unsigned multiplex_count = 2;

    /// Seed derivation strategy
    SeedStrategy seed_strategy = SeedStrategy::XOR;

    void validate() const {
        if (sfc_bits == 0 || sfc_bits > 32) {
            throw std::invalid_argument("sfc_bits must be in range [1, 32]");
        }
        if (log_num_blocks == 0 || log_num_blocks > 30) {
            throw std::invalid_argument("log_num_blocks must be in range [1, 30]");
        }
        if (bits_per_block != 64 && bits_per_block != 256 && bits_per_block != 512) {
            throw std::invalid_argument("bits_per_block must be 64, 256, or 512");
        }
        if (hash_k == 0 || hash_k > bits_per_block) {
            throw std::invalid_argument("hash_k must be in range [1, bits_per_block]");
        }
        if (split_point_q == 0) {
            throw std::invalid_argument("split_point_q must be > 0");
        }

        // Validate SFC type matches dimensionality expectation
        unsigned sfc_output_bits = 0;
        switch (sfc_type) {
            case SFCType::MORTON_2D:
            case SFCType::HILBERT_2D:
                sfc_output_bits = 2 * sfc_bits;
                break;
            case SFCType::MORTON_3D:
            case SFCType::HILBERT_3D:
                sfc_output_bits = 3 * sfc_bits;
                break;
        }
        if (log_num_blocks > sfc_output_bits) {
            throw std::invalid_argument(
                "log_num_blocks exceeds SFC output bits");
        }
    }

    uint64_t computed_num_blocks() const {
        return 1ULL << log_num_blocks;
    }

    uint64_t computed_memory_bytes() const {
        return computed_num_blocks() * (bits_per_block / 8);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SBBFConfig(sfc=";
        switch (sfc_type) {
            case SFCType::MORTON_2D: oss << "MORTON_2D"; break;
            case SFCType::MORTON_3D: oss << "MORTON_3D"; break;
            case SFCType::HILBERT_2D: oss << "HILBERT_2D"; break;
            case SFCType::HILBERT_3D: oss << "HILBERT_3D"; break;
        }
        oss << ", sfc_bits=" << sfc_bits
            << ", blocks=2^" << log_num_blocks
            << ", block_bits=" << bits_per_block
            << ", k=" << hash_k
            << ", strategy=";
        switch (intra_strategy) {
            case IntraBlockStrategy::DOUBLE_HASH: oss << "DOUBLE_HASH"; break;
            case IntraBlockStrategy::PATTERN_LOOKUP: oss << "PATTERN_LOOKUP"; break;
            case IntraBlockStrategy::MULTIPLEXED: oss << "MULTIPLEXED"; break;
        }
        oss << ", seed=";
        switch (seed_strategy) {
            case SeedStrategy::XOR: oss << "XOR"; break;
            case SeedStrategy::MULTIPLY_SHIFT: oss << "MULTIPLY_SHIFT"; break;
        }
        oss << ", memory=" << computed_memory_bytes() << "B)";
        return oss.str();
    }
};

// ============================================================================
// 64-bit Block Spatial-Blocked Bloom Filter
// ============================================================================

/**
 * Spatial-Blocked Bloom Filter with 64-bit blocks.
 *
 * Template Parameters:
 *   SFCBits - Bits per coordinate in SFC encoding (default 16)
 */
template <unsigned SFCBits = 16>
class SpatialBlockedBloomFilter64 {
public:
    SBBFConfig config;

private:
    std::vector<uint64_t> blocks_;
    uint64_t block_mask_;  // Mask for block index (low bits)
    std::vector<uint64_t> pattern_table_;  // For PATTERN_LOOKUP

public:
    explicit SpatialBlockedBloomFilter64(const SBBFConfig &conf) : config(conf) {
        config.validate();

        if (config.bits_per_block != 64) {
            throw std::invalid_argument(
                "SpatialBlockedBloomFilter64 requires bits_per_block=64");
        }

        // Allocate blocks
        blocks_.resize(config.computed_num_blocks(), 0);

        // Compute mask for block index (low bits of SFC code)
        block_mask_ = (1ULL << config.log_num_blocks) - 1;

        // Initialize pattern table if needed
        if (config.intra_strategy == IntraBlockStrategy::PATTERN_LOOKUP) {
            init_pattern_table();
        }
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Insert a 2D point using SFC encoding
    void put2D(uint32_t x, uint32_t y) {
        uint64_t sfc_code = encode_2d(x, y);
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        blocks_[block_idx] |= mask;
    }

    /// Insert a 3D point using SFC encoding
    void put3D(uint32_t x, uint32_t y, uint32_t z) {
        uint64_t sfc_code = encode_3d(x, y, z);
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        blocks_[block_idx] |= mask;
    }

    /// Insert using vector interface (for compatibility)
    void put(const std::vector<uint64_t> &point) {
        if (point.size() >= 3 &&
            (config.sfc_type == SFCType::MORTON_3D ||
             config.sfc_type == SFCType::HILBERT_3D)) {
            put3D(static_cast<uint32_t>(point[0]),
                  static_cast<uint32_t>(point[1]),
                  static_cast<uint32_t>(point[2]));
        } else if (point.size() >= 2) {
            put2D(static_cast<uint32_t>(point[0]),
                  static_cast<uint32_t>(point[1]));
        }
    }

    /// Query membership for 2D point
    bool get_bool_2D(uint32_t x, uint32_t y) const {
        uint64_t sfc_code = encode_2d(x, y);
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        return (blocks_[block_idx] & mask) == mask;
    }

    /// Query membership for 3D point
    bool get_bool_3D(uint32_t x, uint32_t y, uint32_t z) const {
        uint64_t sfc_code = encode_3d(x, y, z);
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        return (blocks_[block_idx] & mask) == mask;
    }

    /// Query using vector interface
    bool get_bool(const std::vector<uint64_t> &point) const {
        if (point.size() >= 3 &&
            (config.sfc_type == SFCType::MORTON_3D ||
             config.sfc_type == SFCType::HILBERT_3D)) {
            return get_bool_3D(static_cast<uint32_t>(point[0]),
                               static_cast<uint32_t>(point[1]),
                               static_cast<uint32_t>(point[2]));
        } else if (point.size() >= 2) {
            return get_bool_2D(static_cast<uint32_t>(point[0]),
                               static_cast<uint32_t>(point[1]));
        }
        return false;
    }

    /// For counting interface compatibility (returns 1 if present, 0 otherwise)
    uint64_t get_min(const std::vector<uint64_t> &point) const {
        return get_bool(point) ? 1 : 0;
    }

    // ========================================================================
    // Direct SFC Code Access (for sequential iteration benchmarks)
    // ========================================================================

    /**
     * Query membership by raw SFC code (skips coordinate encoding).
     *
     * This enables optimal cache performance during sequential SFC traversal:
     *   for (uint64_t sfc = 0; sfc < max_sfc; ++sfc)
     *       filter.get_bool_by_sfc(sfc);
     *
     * Since block_idx = sfc & mask, sequential SFC codes yield sequential
     * block accesses, enabling hardware prefetching.
     *
     * @param sfc_code  Raw space-filling curve code
     * @return true if all k bits are set in the corresponding block
     */
    bool get_bool_by_sfc(uint64_t sfc_code) const {
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        return (blocks_[block_idx] & mask) == mask;
    }

    /**
     * Insert using raw SFC code (skips coordinate encoding).
     *
     * @param sfc_code  Raw space-filling curve code
     */
    void put_by_sfc(uint64_t sfc_code) {
        uint64_t block_idx, seed;
        compute_location(sfc_code, block_idx, seed);
        uint64_t mask = construct_mask(seed);
        blocks_[block_idx] |= mask;
    }

    // ========================================================================
    // Neighborhood Queries (key SBBF advantage)
    // ========================================================================

    /**
     * Query 2D neighborhood around point.
     *
     * @param x,y  Center coordinates
     * @param radius  Chebyshev radius (1 = 3x3, 2 = 5x5)
     * @return Bitmask of found neighbors (row-major, (2r+1)^2 bits)
     */
    uint64_t query_neighborhood_2D(uint32_t x, uint32_t y, unsigned radius = 1) const {
        // Fast path: Use batch encoding for Hilbert 3x3 neighborhoods
        if (radius == 1 && config.sfc_type == SFCType::HILBERT_2D) {
            return query_neighborhood_2D_hilbert_batch(x, y);
        }

        uint64_t result = 0;
        unsigned bit_idx = 0;
        int r = static_cast<int>(radius);

        // Prefetch center block (SFC locality means neighbors likely nearby)
        uint64_t center_sfc = encode_2d(x, y);
        uint64_t center_block = center_sfc & block_mask_;  // LOW bits -> block index
        __builtin_prefetch(&blocks_[center_block], 0, 3);

        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int nx = static_cast<int>(x) + dx;
                int ny = static_cast<int>(y) + dy;

                // Check bounds
                if (nx >= 0 && ny >= 0 &&
                    static_cast<uint32_t>(nx) <= sfc::Morton2D<SFCBits>::max_coord &&
                    static_cast<uint32_t>(ny) <= sfc::Morton2D<SFCBits>::max_coord) {
                    if (get_bool_2D(static_cast<uint32_t>(nx),
                                    static_cast<uint32_t>(ny))) {
                        result |= (1ULL << bit_idx);
                    }
                }
                ++bit_idx;
            }
        }
        return result;
    }

    /**
     * Optimized 3x3 neighborhood query for Hilbert2D using batch encoding.
     * Achieves ~3x speedup by encoding upper chunks once and varying only
     * the final chunk for all 9 neighbors (when within same 16x16 block).
     */
    uint64_t query_neighborhood_2D_hilbert_batch(uint32_t x, uint32_t y) const {
        // Get all 9 SFC codes in a single batch operation
        uint64_t codes[9];
        sfc::Hilbert2D<SFCBits>::encode_neighborhood_2d(x, y, codes);

        // Prefetch the center block
        uint64_t center_block = codes[4] & block_mask_;
        __builtin_prefetch(&blocks_[center_block], 0, 3);

        uint64_t result = 0;

        // Check all 9 codes
        for (unsigned i = 0; i < 9; ++i) {
            uint64_t block_idx, seed;
            compute_location(codes[i], block_idx, seed);
            uint64_t mask = construct_mask(seed);

            if ((blocks_[block_idx] & mask) == mask) {
                result |= (1ULL << i);
            }
        }

        return result;
    }

    /**
     * Query 3D neighborhood (6-connected or 26-connected).
     *
     * @param x,y,z  Center coordinates
     * @param full_26_connected  If true, check all 26 neighbors; otherwise 6
     * @return Count of present neighbors
     */
    unsigned query_neighborhood_3D(uint32_t x, uint32_t y, uint32_t z,
                                   bool full_26_connected = false) const {
        // Use optimized batch encoding for 26-connected Hilbert3D
        if (full_26_connected && config.sfc_type == SFCType::HILBERT_3D &&
            x > 0 && y > 0 && z > 0 &&
            x < sfc::Hilbert3D<SFCBits>::max_coord &&
            y < sfc::Hilbert3D<SFCBits>::max_coord &&
            z < sfc::Hilbert3D<SFCBits>::max_coord) {
            return query_neighborhood_3D_hilbert_batch(x, y, z);
        }

        unsigned count = 0;

        // 6-connected: face neighbors
        static const int offsets6[6][3] = {
            {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
        };

        // 26-connected: all neighbors (including edges and corners)
        static const int offsets26[26][3] = {
            {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
            {-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
            {-1,  1, -1}, {0,  1, -1}, {1,  1, -1},
            {-1, -1,  0}, {0, -1,  0}, {1, -1,  0},
            {-1,  0,  0},              {1,  0,  0},
            {-1,  1,  0}, {0,  1,  0}, {1,  1,  0},
            {-1, -1,  1}, {0, -1,  1}, {1, -1,  1},
            {-1,  0,  1}, {0,  0,  1}, {1,  0,  1},
            {-1,  1,  1}, {0,  1,  1}, {1,  1,  1}
        };

        const int (*offsets)[3] = full_26_connected ? offsets26 : offsets6;
        int num_offsets = full_26_connected ? 26 : 6;

        for (int i = 0; i < num_offsets; ++i) {
            int nx = static_cast<int>(x) + offsets[i][0];
            int ny = static_cast<int>(y) + offsets[i][1];
            int nz = static_cast<int>(z) + offsets[i][2];

            if (nx >= 0 && ny >= 0 && nz >= 0 &&
                static_cast<uint32_t>(nx) <= sfc::Morton3D<SFCBits>::max_coord &&
                static_cast<uint32_t>(ny) <= sfc::Morton3D<SFCBits>::max_coord &&
                static_cast<uint32_t>(nz) <= sfc::Morton3D<SFCBits>::max_coord) {
                if (get_bool_3D(static_cast<uint32_t>(nx),
                                static_cast<uint32_t>(ny),
                                static_cast<uint32_t>(nz))) {
                    ++count;
                }
            }
        }
        return count;
    }

private:
    /// Optimized 26-connected 3D neighborhood query using batch Hilbert encoding
    unsigned query_neighborhood_3D_hilbert_batch(uint32_t x, uint32_t y, uint32_t z) const {
        uint64_t codes[27];
        sfc::Hilbert3D<SFCBits>::encode_neighborhood_3d(x, y, z, codes);

        unsigned count = 0;
        // Check all 27 codes, skip center (index 13)
        for (unsigned i = 0; i < 27; ++i) {
            if (i == 13) continue;  // Skip center

            uint64_t block_idx, seed;
            compute_location(codes[i], block_idx, seed);
            uint64_t mask = construct_mask(seed);

            if ((blocks_[block_idx] & mask) == mask) {
                ++count;
            }
        }
        return count;
    }

public:

    /// Check if any neighbor exists (faster than full neighborhood query)
    bool has_any_neighbor_2D(uint32_t x, uint32_t y, unsigned radius = 1) const {
        int r = static_cast<int>(radius);
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = static_cast<int>(x) + dx;
                int ny = static_cast<int>(y) + dy;

                if (nx >= 0 && ny >= 0 &&
                    static_cast<uint32_t>(nx) <= sfc::Morton2D<SFCBits>::max_coord &&
                    static_cast<uint32_t>(ny) <= sfc::Morton2D<SFCBits>::max_coord) {
                    if (get_bool_2D(static_cast<uint32_t>(nx),
                                    static_cast<uint32_t>(ny))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    void clear() {
        std::fill(blocks_.begin(), blocks_.end(), 0ULL);
    }

    uint64_t memory_usage() const {
        return blocks_.size() * sizeof(uint64_t) +
               pattern_table_.size() * sizeof(uint64_t);
    }

    std::string summary() const {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"type\": \"SpatialBlockedBloomFilter64\",\n"
            << "  \"sfc_type\": \"";
        switch (config.sfc_type) {
            case SFCType::MORTON_2D: oss << "MORTON_2D"; break;
            case SFCType::MORTON_3D: oss << "MORTON_3D"; break;
            case SFCType::HILBERT_2D: oss << "HILBERT_2D"; break;
            case SFCType::HILBERT_3D: oss << "HILBERT_3D"; break;
        }
        oss << "\",\n"
            << "  \"sfc_bits\": " << config.sfc_bits << ",\n"
            << "  \"num_blocks\": " << blocks_.size() << ",\n"
            << "  \"bits_per_block\": " << config.bits_per_block << ",\n"
            << "  \"hash_k\": " << config.hash_k << ",\n"
            << "  \"memory_bytes\": " << memory_usage() << "\n"
            << "}";
        return oss.str();
    }

    /// Get the block index for a 2D point (for debugging/analysis)
    uint64_t get_block_index_2D(uint32_t x, uint32_t y) const {
        uint64_t sfc_code = encode_2d(x, y);
        return sfc_code & block_mask_;  // LOW bits -> block index
    }

    /// Get the SFC value for a 2D point (for debugging/analysis)
    uint64_t get_sfc_value_2D(uint32_t x, uint32_t y) const {
        return encode_2d(x, y);
    }

    /// Calculate block fill ratio (fraction of blocks with any bits set)
    double block_fill_ratio() const {
        size_t filled = 0;
        for (const auto &block : blocks_) {
            if (block != 0) ++filled;
        }
        return static_cast<double>(filled) / blocks_.size();
    }

    /// Calculate average bits set per filled block
    double avg_bits_per_filled_block() const {
        size_t total_bits = 0;
        size_t filled = 0;
        for (const auto &block : blocks_) {
            if (block != 0) {
                total_bits += __builtin_popcountll(block);
                ++filled;
            }
        }
        return filled > 0 ? static_cast<double>(total_bits) / filled : 0.0;
    }

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// Encode 2D coordinates using configured SFC type
    uint64_t encode_2d(uint32_t x, uint32_t y) const {
        switch (config.sfc_type) {
            case SFCType::MORTON_2D:
                return sfc::Morton2D<SFCBits>::encode(x, y);
            case SFCType::HILBERT_2D:
                return sfc::Hilbert2D<SFCBits>::encode(x, y);
            default:
                return sfc::Morton2D<SFCBits>::encode(x, y);
        }
    }

    /// Encode 3D coordinates using configured SFC type
    uint64_t encode_3d(uint32_t x, uint32_t y, uint32_t z) const {
        switch (config.sfc_type) {
            case SFCType::MORTON_3D:
                return sfc::Morton3D<SFCBits>::encode(x, y, z);
            case SFCType::HILBERT_3D:
                return sfc::Hilbert3D<SFCBits>::encode(x, y, z);
            default:
                return sfc::Morton3D<SFCBits>::encode(x, y, z);
        }
    }

    /// Split SFC code into block index and intra-block seed
    void compute_location(uint64_t sfc_code, uint64_t &block_idx,
                          uint64_t &seed) const {
        // LOW bits -> block index (enables hardware prefetching during SFC traversal)
        block_idx = sfc_code & block_mask_;

        // Derive seed using configured strategy
        switch (config.seed_strategy) {
            case SeedStrategy::XOR:
                // Combine high and low bits via XOR
                seed = (sfc_code >> config.log_num_blocks) ^ block_idx;
                break;
            case SeedStrategy::MULTIPLY_SHIFT:
                // Mix all bits via multiply-shift hash
                seed = (sfc_code * 0x9E3779B97F4A7C15ULL) >> 16;
                break;
            default:
                seed = (sfc_code >> config.log_num_blocks) ^ block_idx;
                break;
        }
    }

    /// Construct k-bit mask using configured strategy
    uint64_t construct_mask(uint64_t seed) const {
        switch (config.intra_strategy) {
            case IntraBlockStrategy::DOUBLE_HASH:
                return construct_mask_double_hash(seed);
            case IntraBlockStrategy::PATTERN_LOOKUP:
                return construct_mask_pattern(seed);
            case IntraBlockStrategy::MULTIPLEXED:
                return construct_mask_multiplexed(seed);
            default:
                return construct_mask_double_hash(seed);
        }
    }

    /// Double-hashing mask construction
    uint64_t construct_mask_double_hash(uint64_t seed) const {
        constexpr uint64_t B = 64;  // Block size in bits
        uint32_t h1 = seed % B;
        uint32_t h2 = (seed / B) % B;
        if (h2 == 0) h2 = 1;  // Ensure h2 != 0 for good distribution

        uint64_t mask = 0;
        for (unsigned i = 1; i <= config.hash_k; ++i) {
            uint32_t bit_pos = (h1 + i * h2) % B;
            mask |= (1ULL << bit_pos);
        }
        return mask;
    }

    /// Pattern lookup mask construction
    uint64_t construct_mask_pattern(uint64_t seed) const {
        if (pattern_table_.empty()) {
            return construct_mask_double_hash(seed);
        }
        size_t idx = seed % pattern_table_.size();
        return pattern_table_[idx];
    }

    /// Multiplexed pattern mask construction
    uint64_t construct_mask_multiplexed(uint64_t seed) const {
        uint64_t mask = 0;
        unsigned bits_per_pattern = config.hash_k / config.multiplex_count;
        if (bits_per_pattern == 0) bits_per_pattern = 1;

        for (unsigned p = 0; p < config.multiplex_count; ++p) {
            uint64_t pattern_seed = (seed >> (p * 8)) & 0xFF;
            uint32_t h1 = pattern_seed % 64;
            uint32_t h2 = ((pattern_seed >> 4) % 63) + 1;

            for (unsigned i = 1; i <= bits_per_pattern; ++i) {
                uint32_t bit_pos = (h1 + i * h2) % 64;
                mask |= (1ULL << bit_pos);
            }
        }
        return mask;
    }

    /// Initialize pattern table for PATTERN_LOOKUP strategy
    void init_pattern_table() {
        pattern_table_.resize(config.pattern_table_size);

        // Generate patterns with exactly k bits set
        for (size_t i = 0; i < pattern_table_.size(); ++i) {
            uint64_t mask = 0;
            // Use a simple PRNG seeded by index
            uint64_t state = i * 0x9e3779b97f4a7c15ULL + 0x6c62272e07bb0142ULL;

            for (unsigned j = 0; j < config.hash_k; ++j) {
                // Generate next bit position
                state ^= state >> 33;
                state *= 0xff51afd7ed558ccdULL;
                state ^= state >> 33;

                uint32_t bit_pos = state % 64;

                // Ensure we don't set the same bit twice
                while (mask & (1ULL << bit_pos)) {
                    state *= 0xc4ceb9fe1a85ec53ULL;
                    state ^= state >> 33;
                    bit_pos = state % 64;
                }
                mask |= (1ULL << bit_pos);
            }
            pattern_table_[i] = mask;
        }
    }
};

// ============================================================================
// Type Aliases
// ============================================================================

/// Default SBBF with 64-bit blocks and 16-bit coordinates
using SpatialBlockedBloomFilter = SpatialBlockedBloomFilter64<16>;

} // namespace sbbf

#endif // SBBF_SPATIAL_BLOCKED_BLOOM_FILTER_HPP
