/*
 * Space-Filling Curves for Spatial Indexing
 *
 * Provides Morton (Z-order) and Hilbert curve encoding/decoding for 2D and 3D
 * coordinates. These curves map multidimensional coordinates to 1D values while
 * preserving spatial locality.
 *
 * Morton curves use bit-interleaving (fast, uses PDEP/PEXT when available).
 * Hilbert curves use state-machine lookup tables (better locality, more complex).
 *
 * References:
 * - Morton: G.M. Morton, "A Computer Oriented Geodetic Data Base", IBM 1966
 * - Hilbert 2D: Standard state-machine approach
 * - Hilbert 3D: Jia et al., "Efficient 3D Hilbert Curve Encoding and Decoding
 *   Algorithms", Chinese J. of Electronics, 2022 (JFK-3HE/JFK-3HD)
 */

#ifndef SBBF_SPACE_FILLING_CURVES_HPP
#define SBBF_SPACE_FILLING_CURVES_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

// Detect BMI2 support for PDEP/PEXT instructions
#if defined(__BMI2__) && defined(__x86_64__)
#define SFC_HAS_BMI2 1
#include <immintrin.h>
#else
#define SFC_HAS_BMI2 0
#endif

// Detect AVX2 support for SIMD operations
// Define SFC_FORCE_SCALAR=1 to disable SIMD even when available
#if defined(__AVX2__) && defined(__x86_64__) && !defined(SFC_FORCE_SCALAR)
#define SFC_HAS_AVX2 1
#ifndef SFC_HAS_BMI2
#include <immintrin.h>
#endif
#else
#define SFC_HAS_AVX2 0
#endif

namespace sfc {

// ============================================================================
// Morton 2D - Z-order curve for 2D coordinates
// ============================================================================

/**
 * Morton (Z-order) encoding for 2D coordinates.
 * Interleaves bits: (x, y) -> ...y2x2y1x1y0x0
 *
 * Template parameter Bits: max bits per coordinate (up to 32).
 * Output is 2*Bits wide (max 64 bits).
 */
template <unsigned Bits = 16>
struct Morton2D {
    static_assert(Bits <= 32, "Morton2D supports up to 32 bits per coordinate");
    static_assert(Bits > 0, "Bits must be positive");

    /// Maximum valid coordinate value
    static constexpr uint64_t max_coord = (1ULL << Bits) - 1;

    /// Maximum Morton code value
    static constexpr uint64_t max_code = (1ULL << (2 * Bits)) - 1;

#if SFC_HAS_BMI2
    /**
     * Encode (x, y) to Morton code using BMI2 PDEP instruction.
     * PDEP spreads bits according to a mask pattern.
     */
    static inline uint64_t encode(uint32_t x, uint32_t y) {
        // 0x5555... = 0101010101... (every other bit for x)
        // 0xAAAA... = 1010101010... (every other bit for y, shifted left 1)
        return _pdep_u64(x, 0x5555555555555555ULL) |
               _pdep_u64(y, 0xAAAAAAAAAAAAAAAAULL);
    }

    /**
     * Decode Morton code to (x, y) using BMI2 PEXT instruction.
     * PEXT extracts bits according to a mask pattern.
     */
    static inline void decode(uint64_t code, uint32_t &x, uint32_t &y) {
        x = static_cast<uint32_t>(_pext_u64(code, 0x5555555555555555ULL));
        y = static_cast<uint32_t>(_pext_u64(code, 0xAAAAAAAAAAAAAAAAULL));
    }
#else
    /**
     * Portable Morton encode using magic-number bit spreading.
     * Based on: https://graphics.stanford.edu/~seander/bithacks.html
     */
    static inline uint64_t encode(uint32_t x, uint32_t y) {
        return spread_bits(x) | (spread_bits(y) << 1);
    }

    /**
     * Portable Morton decode using magic-number bit compaction.
     */
    static inline void decode(uint64_t code, uint32_t &x, uint32_t &y) {
        x = compact_bits(code);
        y = compact_bits(code >> 1);
    }

private:
    /// Spread bits of a 32-bit value to occupy every other bit position
    static inline uint64_t spread_bits(uint32_t v) {
        uint64_t w = v;
        w = (w | (w << 16)) & 0x0000FFFF0000FFFFULL;
        w = (w | (w << 8)) & 0x00FF00FF00FF00FFULL;
        w = (w | (w << 4)) & 0x0F0F0F0F0F0F0F0FULL;
        w = (w | (w << 2)) & 0x3333333333333333ULL;
        w = (w | (w << 1)) & 0x5555555555555555ULL;
        return w;
    }

    /// Compact bits from every other position back to contiguous
    static inline uint32_t compact_bits(uint64_t w) {
        w &= 0x5555555555555555ULL;
        w = (w | (w >> 1)) & 0x3333333333333333ULL;
        w = (w | (w >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
        w = (w | (w >> 4)) & 0x00FF00FF00FF00FFULL;
        w = (w | (w >> 8)) & 0x0000FFFF0000FFFFULL;
        w = (w | (w >> 16)) & 0x00000000FFFFFFFFULL;
        return static_cast<uint32_t>(w);
    }
#endif
};

// ============================================================================
// Morton 3D - Z-order curve for 3D coordinates
// ============================================================================

/**
 * Morton (Z-order) encoding for 3D coordinates.
 * Interleaves bits: (x, y, z) -> ...z2y2x2z1y1x1z0y0x0
 *
 * Template parameter Bits: max bits per coordinate (up to 21 for 63-bit output).
 * Output is 3*Bits wide.
 */
template <unsigned Bits = 16>
struct Morton3D {
    static_assert(Bits <= 21, "Morton3D supports up to 21 bits per coordinate (63-bit output)");
    static_assert(Bits > 0, "Bits must be positive");

    /// Maximum valid coordinate value
    static constexpr uint64_t max_coord = (1ULL << Bits) - 1;

    /// Maximum Morton code value
    static constexpr uint64_t max_code = (1ULL << (3 * Bits)) - 1;

#if SFC_HAS_BMI2
    /**
     * Encode (x, y, z) to Morton code using BMI2 PDEP instruction.
     */
    static inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        // Pattern for x: bits at positions 0, 3, 6, 9, ... (every 3rd bit starting at 0)
        // Pattern for y: bits at positions 1, 4, 7, 10, ... (every 3rd bit starting at 1)
        // Pattern for z: bits at positions 2, 5, 8, 11, ... (every 3rd bit starting at 2)
        constexpr uint64_t mask_x = 0x1249249249249249ULL; // 001001001... in binary
        constexpr uint64_t mask_y = 0x2492492492492492ULL; // 010010010... in binary
        constexpr uint64_t mask_z = 0x4924924924924924ULL; // 100100100... in binary

        return _pdep_u64(x, mask_x) | _pdep_u64(y, mask_y) | _pdep_u64(z, mask_z);
    }

    /**
     * Decode Morton code to (x, y, z) using BMI2 PEXT instruction.
     */
    static inline void decode(uint64_t code, uint32_t &x, uint32_t &y, uint32_t &z) {
        constexpr uint64_t mask_x = 0x1249249249249249ULL;
        constexpr uint64_t mask_y = 0x2492492492492492ULL;
        constexpr uint64_t mask_z = 0x4924924924924924ULL;

        x = static_cast<uint32_t>(_pext_u64(code, mask_x));
        y = static_cast<uint32_t>(_pext_u64(code, mask_y));
        z = static_cast<uint32_t>(_pext_u64(code, mask_z));
    }
#else
    /**
     * Portable Morton 3D encode using magic-number bit spreading.
     */
    static inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        return spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2);
    }

    /**
     * Portable Morton 3D decode.
     */
    static inline void decode(uint64_t code, uint32_t &x, uint32_t &y, uint32_t &z) {
        x = compact_bits_3d(code);
        y = compact_bits_3d(code >> 1);
        z = compact_bits_3d(code >> 2);
    }

private:
    /// Spread bits of a 21-bit value to occupy every 3rd bit position
    static inline uint64_t spread_bits_3d(uint32_t v) {
        uint64_t w = v & 0x1FFFFF; // Mask to 21 bits
        w = (w | (w << 32)) & 0x1F00000000FFFFULL;
        w = (w | (w << 16)) & 0x1F0000FF0000FFULL;
        w = (w | (w << 8)) & 0x100F00F00F00F00FULL;
        w = (w | (w << 4)) & 0x10C30C30C30C30C3ULL;
        w = (w | (w << 2)) & 0x1249249249249249ULL;
        return w;
    }

    /// Compact bits from every 3rd position back to contiguous
    static inline uint32_t compact_bits_3d(uint64_t w) {
        w &= 0x1249249249249249ULL;
        w = (w | (w >> 2)) & 0x10C30C30C30C30C3ULL;
        w = (w | (w >> 4)) & 0x100F00F00F00F00FULL;
        w = (w | (w >> 8)) & 0x1F0000FF0000FFULL;
        w = (w | (w >> 16)) & 0x1F00000000FFFFULL;
        w = (w | (w >> 32)) & 0x1FFFFFULL;
        return static_cast<uint32_t>(w);
    }
#endif
};

// ============================================================================
// Hilbert 2D - Hilbert curve for 2D coordinates
// ============================================================================

/**
 * Hilbert curve encoding for 2D coordinates.
 *
 * This implementation uses a LUT-based approach that processes 4 bits at a time,
 * reducing iterations from 16 to 4 for 16-bit coordinates. The LUT captures the
 * state transitions and Hilbert code contributions for all 4-bit coordinate chunks.
 *
 * Performance: ~5-10ns per encode (vs ~70ns for bit-by-bit approach)
 *
 * References:
 * - fast-hilbert (Rust): https://github.com/becheran/fast-hilbert
 * - hilbert_gen: https://github.com/wzli/hilbert_gen
 *
 * Template parameter Bits: max bits per coordinate (must be multiple of 4, up to 32).
 * Output is 2*Bits wide.
 */
template <unsigned Bits = 16>
struct Hilbert2D {
    static_assert(Bits <= 32, "Hilbert2D supports up to 32 bits per coordinate");
    static_assert(Bits > 0, "Bits must be positive");
    static_assert(Bits % 4 == 0 || Bits < 4, "Bits should be multiple of 4 for optimal performance");

    /// Maximum valid coordinate value
    static constexpr uint64_t max_coord = (1ULL << Bits) - 1;

    /// Maximum Hilbert code value
    static constexpr uint64_t max_code = (1ULL << (2 * Bits)) - 1;

private:
    // ========================================================================
    // LUT-based fast implementation (processes 4 bits at a time)
    // ========================================================================

    // The 2D Hilbert curve has 4 orientations/states:
    // State 0: Original H curve
    // State 1: Rotated 90 deg clockwise
    // State 2: Rotated 180 deg
    // State 3: Rotated 270 deg clockwise
    //
    // LUT index: (state << 8) | (x_chunk << 4) | y_chunk = 1024 entries
    // Each entry encodes: hilbert code chunk (8 bits) and next state (2 bits)

    // Combined LUT: low 8 bits = Hilbert code chunk, bits 8-9 = next state
    // Size: 1024 * 2 bytes = 2KB
    // Generated at program startup via static initialization
    struct LUT {
        alignas(64) uint16_t data[1024];

        LUT() {
            // Quadrant order for each state (index 0-3 gives position in Hilbert order)
            // For each state, defines {x, y, child_state} for positions 0-3
            static constexpr uint8_t state_info[4][4][3] = {
                // State 0: standard H curve
                {{0,0, 1}, {0,1, 0}, {1,1, 0}, {1,0, 3}},
                // State 1: A curve (rotated)
                {{0,0, 0}, {1,0, 1}, {1,1, 1}, {0,1, 2}},
                // State 2: H' curve (180 deg rotated)
                {{1,1, 3}, {1,0, 2}, {0,0, 2}, {0,1, 1}},
                // State 3: A' curve
                {{1,1, 2}, {0,1, 3}, {0,0, 3}, {1,0, 0}},
            };

            for (uint8_t initial_state = 0; initial_state < 4; ++initial_state) {
                for (uint8_t x4 = 0; x4 < 16; ++x4) {
                    for (uint8_t y4 = 0; y4 < 16; ++y4) {
                        // Process 4 bits (4 levels of recursion)
                        uint64_t code = 0;
                        uint8_t state = initial_state;

                        for (int level = 3; level >= 0; --level) {
                            uint8_t rx = (x4 >> level) & 1;
                            uint8_t ry = (y4 >> level) & 1;

                            // Find position in current state's order
                            uint8_t pos = 0;
                            uint8_t child_state = 0;
                            for (int i = 0; i < 4; ++i) {
                                if (state_info[state][i][0] == rx && state_info[state][i][1] == ry) {
                                    pos = i;
                                    child_state = state_info[state][i][2];
                                    break;
                                }
                            }

                            uint32_t s = 1U << level;
                            code += static_cast<uint64_t>(pos) * s * s;
                            state = child_state;
                        }

                        size_t idx = (static_cast<size_t>(initial_state) << 8) | (x4 << 4) | y4;
                        data[idx] = static_cast<uint8_t>(code) | (static_cast<uint16_t>(state) << 8);
                    }
                }
            }
        }
    };

    static const LUT& get_lut() {
        static const LUT lut;
        return lut;
    }

public:
    /**
     * Encode (x, y) to Hilbert code using LUT-based approach.
     * Processes 4 bits at a time for ~10x speedup over bit-by-bit.
     */
    static inline uint64_t encode(uint32_t x, uint32_t y) {
        // Use the reference implementation for non-standard bit widths
        if constexpr (Bits < 4) {
            return encode_reference(x, y);
        }

        const auto& lut = get_lut().data;
        uint64_t code = 0;
        uint8_t state = 0;

        // Process 4 bits at a time, MSB first
        constexpr unsigned chunks = Bits / 4;

        for (unsigned i = 0; i < chunks; ++i) {
            unsigned shift = Bits - 4 - i * 4;
            uint8_t x4 = (x >> shift) & 0xF;
            uint8_t y4 = (y >> shift) & 0xF;

            // LUT index: state (2 bits) | x (4 bits) | y (4 bits)
            size_t idx = (static_cast<size_t>(state) << 8) | (x4 << 4) | y4;
            uint16_t entry = lut[idx];

            // Low 8 bits: Hilbert code chunk, High 8 bits: contains next state
            code = (code << 8) | (entry & 0xFF);
            state = (entry >> 8) & 0x3;  // Next state is in bits 8-9
        }

        return code;
    }

    /**
     * Reference bit-by-bit encode (for verification and non-4-aligned Bits).
     */
    static inline uint64_t encode_reference(uint32_t x, uint32_t y) {
        uint64_t code = 0;
        uint32_t rx, ry, s;

        for (s = (1U << (Bits - 1)); s > 0; s >>= 1) {
            rx = (x & s) > 0 ? 1 : 0;
            ry = (y & s) > 0 ? 1 : 0;
            code += s * s * ((3 * rx) ^ ry);
            rotate(s, x, y, rx, ry);
        }
        return code;
    }

    /**
     * Decode Hilbert code to (x, y).
     */
    static inline void decode(uint64_t code, uint32_t &x, uint32_t &y) {
        uint32_t rx, ry, s, t = code;
        x = y = 0;

        for (s = 1; s < (1U << Bits); s <<= 1) {
            rx = 1 & (t / 2);
            ry = 1 & (t ^ rx);
            rotate(s, x, y, rx, ry);
            x += s * rx;
            y += s * ry;
            t /= 4;
        }
    }

    /**
     * Encode all chunks except the last one.
     * Returns the code prefix and the state after processing upper chunks.
     * Used for batch neighborhood encoding optimization.
     */
    static inline void encode_upper_chunks(uint32_t x, uint32_t y,
                                            uint64_t& prefix, uint8_t& state_out) {
        if constexpr (Bits < 4) {
            prefix = 0;
            state_out = 0;
            return;
        }

        const auto& lut = get_lut().data;
        uint64_t code = 0;
        uint8_t state = 0;

        // Process all chunks except the last one
        constexpr unsigned chunks = Bits / 4;
        constexpr unsigned upper_chunks = chunks > 1 ? chunks - 1 : 0;

        for (unsigned i = 0; i < upper_chunks; ++i) {
            unsigned shift = Bits - 4 - i * 4;
            uint8_t x4 = (x >> shift) & 0xF;
            uint8_t y4 = (y >> shift) & 0xF;

            size_t idx = (static_cast<size_t>(state) << 8) | (x4 << 4) | y4;
            uint16_t entry = lut[idx];

            code = (code << 8) | (entry & 0xFF);
            state = (entry >> 8) & 0x3;
        }

        prefix = code;
        state_out = state;
    }

    /**
     * Batch encode a 3x3 neighborhood around (x, y).
     * Optimized for the common case where all 9 points share the same upper chunks.
     *
     * @param x Center x coordinate
     * @param y Center y coordinate
     * @param codes Output array of 9 Hilbert codes in row-major order:
     *              [0,1,2] = row y-1, [3,4,5] = row y, [6,7,8] = row y+1
     *              Center point is at index 4.
     */
    static inline void encode_neighborhood_2d(uint32_t x, uint32_t y, uint64_t codes[9]) {
        if constexpr (Bits < 4) {
            // Fallback for small bit widths
            int idx = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    codes[idx++] = encode(x + dx, y + dy);
                }
            }
            return;
        }

        const auto& lut = get_lut().data;

        // Check if all 9 points are in the same 16x16 block (77% of cases)
        // This is true when x and y are not at the edges of their 4-bit chunk
        bool same_block = ((x & 0xF) >= 1 && (x & 0xF) <= 14 &&
                           (y & 0xF) >= 1 && (y & 0xF) <= 14);

        if (same_block) {
#if SFC_HAS_AVX2
            // SIMD fast path: gather 8 LUT entries in parallel
            encode_neighborhood_2d_simd(x, y, codes, lut);
#else
            // Scalar fast path: compute upper chunks once, vary only the last chunk
            encode_neighborhood_2d_scalar(x, y, codes, lut);
#endif
        } else {
            // Block boundary path: neighbors span up to 4 different 16x16 blocks
            // Group neighbors by their upper chunks and batch within each group
            encode_neighborhood_2d_boundary(x, y, codes, lut);
        }
    }

    /// Encode neighborhood when center is at a 16x16 block boundary
    /// Optimized: only compute 1-4 upper chunk encodings as needed
    static inline void encode_neighborhood_2d_boundary(
        uint32_t x, uint32_t y, uint64_t codes[9], const uint16_t* lut) {

        const uint8_t x4 = x & 0xF;
        const uint8_t y4 = y & 0xF;

        // Determine boundary type
        const bool x_low = (x4 == 0);
        const bool x_high = (x4 == 15);
        const bool y_low = (y4 == 0);
        const bool y_high = (y4 == 15);

        // Compute prefix/state for center (always needed, covers 4-6 neighbors)
        uint64_t prefix_cc;
        uint8_t state_cc;
        encode_upper_chunks(x, y, prefix_cc, state_cc);

        // Compute alternative prefixes only when crossing boundaries
        uint64_t prefix_xc = prefix_cc, prefix_cy = prefix_cc, prefix_xy = prefix_cc;
        uint8_t state_xc = state_cc, state_cy = state_cc, state_xy = state_cc;

        const uint32_t x_alt = x_low ? (x - 1) : (x_high ? (x + 1) : x);
        const uint32_t y_alt = y_low ? (y - 1) : (y_high ? (y + 1) : y);

        if (x_low || x_high) {
            encode_upper_chunks(x_alt, y, prefix_xc, state_xc);
        }
        if (y_low || y_high) {
            encode_upper_chunks(x, y_alt, prefix_cy, state_cy);
        }
        if ((x_low || x_high) && (y_low || y_high)) {
            encode_upper_chunks(x_alt, y_alt, prefix_xy, state_xy);
        }

        // Helper to select correct prefix/state based on neighbor position
        auto get_prefix_state = [&](int dx, int dy) -> std::pair<uint64_t, uint8_t> {
            bool use_alt_x = (dx == -1 && x_low) || (dx == 1 && x_high);
            bool use_alt_y = (dy == -1 && y_low) || (dy == 1 && y_high);
            if (use_alt_x && use_alt_y) return {prefix_xy, state_xy};
            if (use_alt_x) return {prefix_xc, state_xc};
            if (use_alt_y) return {prefix_cy, state_cy};
            return {prefix_cc, state_cc};
        };

        // Encode all 9 neighbors
        int idx = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                auto [pref, st] = get_prefix_state(dx, dy);
                uint8_t nx4 = (x + dx) & 0xF;
                uint8_t ny4 = (y + dy) & 0xF;
                size_t lut_idx = (static_cast<size_t>(st) << 8) | (nx4 << 4) | ny4;
                codes[idx++] = (pref << 8) | (lut[lut_idx] & 0xFF);
            }
        }
    }

private:
    /// Scalar implementation of same-block fast path
    static inline void encode_neighborhood_2d_scalar(
        uint32_t x, uint32_t y, uint64_t codes[9], const uint16_t* lut) {

        uint64_t prefix;
        uint8_t state;
        encode_upper_chunks(x, y, prefix, state);

        // Process all 9 points using only the final chunk lookup
        int idx = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint8_t x4 = (x + dx) & 0xF;
                uint8_t y4 = (y + dy) & 0xF;

                size_t lut_idx = (static_cast<size_t>(state) << 8) | (x4 << 4) | y4;
                uint16_t entry = lut[lut_idx];

                // Combine prefix with final chunk (ignore final state, we don't need it)
                codes[idx++] = (prefix << 8) | (entry & 0xFF);
            }
        }
    }

#if SFC_HAS_AVX2
    /// SIMD implementation using AVX2 gather for 8 parallel LUT lookups
    static inline void encode_neighborhood_2d_simd(
        uint32_t x, uint32_t y, uint64_t codes[9], const uint16_t* lut) {

        uint64_t prefix;
        uint8_t state;
        encode_upper_chunks(x, y, prefix, state);

        // Compute base offset for state
        const int32_t state_offset = static_cast<int32_t>(state) << 8;

        // Neighbor offsets in row-major order: (dx, dy) for indices 0-8
        // Index: 0=(-1,-1), 1=(0,-1), 2=(1,-1), 3=(-1,0), 4=(0,0),
        //        5=(1,0), 6=(-1,1), 7=(0,1), 8=(1,1)

        // Compute x4 and y4 for center
        const int32_t x4_center = static_cast<int32_t>(x & 0xF);
        const int32_t y4_center = static_cast<int32_t>(y & 0xF);

        // Create SIMD vectors for the 8 neighbors (excluding center at index 4)
        // We'll handle index 4 (center) separately
        // Indices in codes array: 0,1,2,3, (4=center), 5,6,7,8
        // We gather indices 0,1,2,3,5,6,7,8 with SIMD

        // dx offsets for indices 0,1,2,3,5,6,7,8: -1,0,1,-1,1,-1,0,1
        // dy offsets for indices 0,1,2,3,5,6,7,8: -1,-1,-1,0,0,1,1,1

        __m256i dx_vec = _mm256_setr_epi32(-1, 0, 1, -1, 1, -1, 0, 1);
        __m256i dy_vec = _mm256_setr_epi32(-1, -1, -1, 0, 0, 1, 1, 1);

        // Compute x4 and y4 for all 8 neighbors
        __m256i x4_vec = _mm256_add_epi32(_mm256_set1_epi32(x4_center), dx_vec);
        __m256i y4_vec = _mm256_add_epi32(_mm256_set1_epi32(y4_center), dy_vec);

        // Mask to 4 bits (not needed if same_block is true, but safe)
        x4_vec = _mm256_and_si256(x4_vec, _mm256_set1_epi32(0xF));
        y4_vec = _mm256_and_si256(y4_vec, _mm256_set1_epi32(0xF));

        // Compute LUT indices: state_offset | (x4 << 4) | y4
        __m256i lut_indices = _mm256_or_si256(
            _mm256_set1_epi32(state_offset),
            _mm256_or_si256(
                _mm256_slli_epi32(x4_vec, 4),
                y4_vec
            )
        );

        // The LUT is uint16_t, but gather works on int32_t
        // We need to gather from the 32-bit aligned view
        // LUT index i corresponds to 16-bit offset i, or 32-bit offset i/2
        // We use scale=2 to read uint16_t values

        // Gather 8 entries (as 32-bit values, but we only use low 16 bits)
        // Note: _mm256_i32gather_epi32 reads 32-bit values at base + index*scale
        // For uint16_t array, we use scale=2 and cast base pointer
        __m256i entries = _mm256_i32gather_epi32(
            reinterpret_cast<const int32_t*>(lut),
            lut_indices,
            2  // scale=2 for uint16_t
        );

        // Extract low 8 bits of each entry (Hilbert code chunk)
        __m256i chunks = _mm256_and_si256(entries, _mm256_set1_epi32(0xFF));

        // Store results - we need to combine prefix with each chunk
        alignas(32) int32_t chunk_array[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(chunk_array), chunks);

        // Map back to codes array (indices 0,1,2,3,5,6,7,8)
        codes[0] = (prefix << 8) | static_cast<uint64_t>(chunk_array[0]);
        codes[1] = (prefix << 8) | static_cast<uint64_t>(chunk_array[1]);
        codes[2] = (prefix << 8) | static_cast<uint64_t>(chunk_array[2]);
        codes[3] = (prefix << 8) | static_cast<uint64_t>(chunk_array[3]);
        codes[5] = (prefix << 8) | static_cast<uint64_t>(chunk_array[4]);
        codes[6] = (prefix << 8) | static_cast<uint64_t>(chunk_array[5]);
        codes[7] = (prefix << 8) | static_cast<uint64_t>(chunk_array[6]);
        codes[8] = (prefix << 8) | static_cast<uint64_t>(chunk_array[7]);

        // Handle center (index 4) with scalar lookup
        size_t center_idx = (static_cast<size_t>(state) << 8) | (x4_center << 4) | y4_center;
        codes[4] = (prefix << 8) | (lut[center_idx] & 0xFF);
    }
#endif

    /// Rotate/flip quadrant appropriately
    static inline void rotate(uint32_t n, uint32_t &x, uint32_t &y, uint32_t rx, uint32_t ry) {
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            // Swap x and y
            uint32_t t = x;
            x = y;
            y = t;
        }
    }
};

// ============================================================================
// Hilbert 3D - Hilbert curve for 3D coordinates
// ============================================================================

/**
 * 3D Hilbert curve using rotation-based algorithm.
 * Based on the approach from "Programming the Hilbert Curve" by John Skilling.
 *
 * This implementation uses coordinate transformations (rotations and reflections)
 * to recursively construct the 3D Hilbert curve.
 *
 * Template parameter Bits: max bits per coordinate (up to 21 for 63-bit output).
 * Output is 3*Bits wide.
 */
template <unsigned Bits = 10>
struct Hilbert3D {
    static_assert(Bits <= 21, "Hilbert3D supports up to 21 bits per coordinate");
    static_assert(Bits > 0, "Bits must be positive");

    /// Maximum valid coordinate value
    static constexpr uint64_t max_coord = (1ULL << Bits) - 1;

    /// Maximum Hilbert code value
    static constexpr uint64_t max_code = (1ULL << (3 * Bits)) - 1;

private:
    /**
     * Morton->Hilbert transform LUT (96 bytes).
     * Based on threadlocalmutex.com algorithm using A4 group with 12 states.
     *
     * Index: (state * 8) + octant, where state in [0,11] and octant in [0,7]
     * Entry: bits [2:0] = Hilbert octant, bits [6:3] = (next_state << 3)
     *
     * Source: https://threadlocalmutex.com/?p=149
     */
    static constexpr uint8_t MORTON_TO_HILBERT_LUT[96] = {
        48, 33, 27, 34, 47, 78, 28, 77,  // state 0
        66, 29, 51, 52, 65, 30, 72, 63,  // state 1
        76, 95, 75, 24, 53, 54, 82, 81,  // state 2
        18,  3, 17, 80, 61,  4, 62, 15,  // state 3
         0, 59, 71, 60, 49, 50, 86, 85,  // state 4
        84, 83,  5, 90, 79, 56,  6, 89,  // state 5
        32, 23,  1, 94, 11, 12,  2, 93,  // state 6
        42, 41, 13, 14, 35, 88, 36, 31,  // state 7
        92, 37, 87, 38, 91, 74,  8, 73,  // state 8
        46, 45,  9, 10,  7, 20, 64, 19,  // state 9
        70, 25, 39, 16, 69, 26, 44, 43,  // state 10
        22, 55, 21, 68, 57, 40, 58, 67   // state 11
    };

    /**
     * Hilbert->Morton transform LUT (96 bytes) for decoding.
     */
    static constexpr uint8_t HILBERT_TO_MORTON_LUT[96] = {
        48, 33, 35, 26, 30, 79, 77, 44,  // state 0
        78, 68, 64, 50, 51, 25, 29, 63,  // state 1
        27, 87, 86, 74, 72, 52, 53, 89,  // state 2
        83, 18, 16,  1,  5, 60, 62, 15,  // state 3
         0, 52, 53, 57, 59, 87, 86, 66,  // state 4
        61, 95, 91, 81, 80,  2,  6, 76,  // state 5
        32,  2,  6, 12, 13, 95, 91, 17,  // state 6
        93, 41, 40, 36, 38, 10, 11, 31,  // state 7
        14, 79, 77, 92, 88, 33, 35, 82,  // state 8
        70, 10, 11, 23, 21, 41, 40,  4,  // state 9
        19, 25, 29, 47, 46, 68, 64, 34,  // state 10
        45, 60, 62, 71, 67, 18, 16, 49   // state 11
    };

public:
    /**
     * Encode (x, y, z) to 3D Hilbert code using LUT-based Morton->Hilbert transform.
     * ~10-20x faster than reference bit-by-bit implementation.
     */
    static inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        // Step 1: Morton encode (fast with PDEP or bit magic)
        uint64_t morton = Morton3D<Bits>::encode(x, y, z);

        // Step 2: Transform Morton -> Hilbert using 96-byte LUT
        uint64_t hilbert = 0;
        uint32_t transform = 0;

        for (int i = 3 * (static_cast<int>(Bits) - 1); i >= 0; i -= 3) {
            transform = MORTON_TO_HILBERT_LUT[transform | ((morton >> i) & 7)];
            hilbert = (hilbert << 3) | (transform & 7);
            transform &= ~7U;  // Clear octant bits, keep state
        }

        return hilbert;
    }

    /**
     * Reference bit-by-bit encode (for verification).
     */
    static inline uint64_t encode_reference(uint32_t x, uint32_t y, uint32_t z) {
        uint32_t coords[3] = {x, y, z};
        uint64_t code = 0;

        for (uint32_t s = (1U << (Bits - 1)); s > 0; s >>= 1) {
            uint32_t rx = (coords[0] & s) > 0 ? 1 : 0;
            uint32_t ry = (coords[1] & s) > 0 ? 1 : 0;
            uint32_t rz = (coords[2] & s) > 0 ? 1 : 0;

            // Calculate the 3-bit index for this iteration
            uint32_t idx = rx | (ry << 1) | (rz << 2);

            // Gray code transformation for Hilbert ordering
            // The index determines which subcube we're in
            code = (code << 3) | gray_encode_3d(idx);

            // Rotate coordinates for next level
            rotate_3d(s, coords, rx, ry, rz);
        }
        return code;
    }

    /**
     * Batch encode a 3x3x3 neighborhood around (x, y, z).
     * Optimized for the common case where all 27 points share upper transform state.
     *
     * @param x Center x coordinate
     * @param y Center y coordinate
     * @param z Center z coordinate
     * @param codes Output array of 27 Hilbert codes in z-major order:
     *              [0-8] = z-1 plane, [9-17] = z plane, [18-26] = z+1 plane
     *              Within each plane: row-major (y varies, then x)
     *              Center point is at index 13.
     */
    static inline void encode_neighborhood_3d(uint32_t x, uint32_t y, uint32_t z,
                                               uint64_t codes[27]) {
        // Check if all 27 points share the same upper Morton bits (upper 24 bits).
        // This happens when all of x-1,x,x+1 and y-1,y,y+1 and z-1,z,z+1 have same upper bits.
        // For coordinates: (coord & 3) must be 1 or 2 (middle of 4-block, no carry/borrow to bit 2)
        // Covers (2/4)^3 = 12.5% of positions
        bool same_block = ((x & 3) >= 1 && (x & 3) <= 2) &&
                          ((y & 3) >= 1 && (y & 3) <= 2) &&
                          ((z & 3) >= 1 && (z & 3) <= 2);

        if (same_block) {
            encode_neighborhood_3d_same_block(x, y, z, codes);
        } else {
            encode_neighborhood_3d_boundary(x, y, z, codes);
        }
    }

private:
    /**
     * Helper: Transform Morton code to Hilbert, processing upper iterations only.
     * Processes iterations from bit position 3*(Bits-1) down to stop_bit (exclusive).
     * Returns the intermediate Hilbert prefix and transform state.
     */
    static inline void transform_upper_chunks(uint64_t morton, int stop_bit,
                                               uint64_t& prefix, uint32_t& state) {
        prefix = 0;
        state = 0;

        for (int i = 3 * (static_cast<int>(Bits) - 1); i >= stop_bit; i -= 3) {
            state = MORTON_TO_HILBERT_LUT[state | ((morton >> i) & 7)];
            prefix = (prefix << 3) | (state & 7);
            state &= ~7U;
        }
    }

    /**
     * Helper: Complete the transform for remaining iterations.
     */
    static inline uint64_t transform_final_chunks(uint64_t morton, int stop_bit,
                                                   uint64_t prefix, uint32_t state) {
        for (int i = stop_bit - 3; i >= 0; i -= 3) {
            state = MORTON_TO_HILBERT_LUT[state | ((morton >> i) & 7)];
            prefix = (prefix << 3) | (state & 7);
            state &= ~7U;
        }
        return prefix;
    }

    /**
     * Fast path: all 27 neighbors share the same upper transform state.
     * Shares first (Bits-2) iterations, does 2 final iterations per neighbor.
     */
    static inline void encode_neighborhood_3d_same_block(uint32_t x, uint32_t y, uint32_t z,
                                                          uint64_t codes[27]) {
        // Compute center's Morton code and upper transform (first Bits-2 iterations)
        uint64_t morton_center = Morton3D<Bits>::encode(x, y, z);
        uint64_t prefix;
        uint32_t state;
        // Stop at bit 6 (2 remaining iterations for bits 5:3 and 2:0)
        transform_upper_chunks(morton_center, 6, prefix, state);

        // Process all 27 neighbors - only final 2 chunks (6 bits) differ
        int idx = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint64_t morton = Morton3D<Bits>::encode(x + dx, y + dy, z + dz);

                    // Final 2 transform iterations
                    codes[idx++] = transform_final_chunks(morton, 6, prefix, state);
                }
            }
        }
    }

    /**
     * Boundary path: neighbors may span different transform prefixes.
     * Group by upper Morton bits to minimize redundant prefix computations.
     */
    static inline void encode_neighborhood_3d_boundary(uint32_t x, uint32_t y, uint32_t z,
                                                        uint64_t codes[27]) {
        // Compute prefixes for up to 8 different upper Morton groups
        // Groups are indexed by (dz_cross << 2) | (dy_cross << 1) | dx_cross
        // where d*_cross indicates whether neighbor has different upper coord bits
        uint64_t prefixes[8];
        uint32_t states[8];
        bool computed[8] = {false};

        int idx = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            uint32_t nz = z + dz;

            for (int dy = -1; dy <= 1; ++dy) {
                uint32_t ny = y + dy;

                for (int dx = -1; dx <= 1; ++dx) {
                    uint32_t nx = x + dx;

                    // Compute Morton code
                    uint64_t morton = Morton3D<Bits>::encode(nx, ny, nz);

                    // Find or compute the prefix for this group
                    // Group by upper coordinate bits (bit 1 and above)
                    int group = 0;
                    if ((nx >> 2) != (x >> 2)) group |= 1;
                    if ((ny >> 2) != (y >> 2)) group |= 2;
                    if ((nz >> 2) != (z >> 2)) group |= 4;

                    if (!computed[group]) {
                        // Share first (Bits-2) iterations
                        transform_upper_chunks(morton, 6, prefixes[group], states[group]);
                        computed[group] = true;
                    }

                    // Final 2 transform iterations
                    codes[idx++] = transform_final_chunks(morton, 6, prefixes[group], states[group]);
                }
            }
        }
    }

public:
    /**
     * Decode 3D Hilbert code to (x, y, z) using LUT-based Hilbert->Morton transform.
     */
    static inline void decode(uint64_t hilbert, uint32_t &x, uint32_t &y, uint32_t &z) {
        // Step 1: Transform Hilbert -> Morton using inverse LUT
        uint64_t morton = 0;
        uint32_t transform = 0;

        for (int i = 3 * (static_cast<int>(Bits) - 1); i >= 0; i -= 3) {
            transform = HILBERT_TO_MORTON_LUT[transform | ((hilbert >> i) & 7)];
            morton = (morton << 3) | (transform & 7);
            transform &= ~7U;
        }

        // Step 2: Morton decode (fast with PEXT or bit magic)
        Morton3D<Bits>::decode(morton, x, y, z);
    }

    /**
     * Reference decode (for the reference encode).
     */
    static inline void decode_reference(uint64_t code, uint32_t &x, uint32_t &y, uint32_t &z) {
        uint32_t coords[3] = {0, 0, 0};

        for (uint32_t s = 1; s < (1U << Bits); s <<= 1) {
            // Extract 3 bits from code
            uint32_t gray = code & 7;
            code >>= 3;

            // Inverse Gray code
            uint32_t idx = gray_decode_3d(gray);

            uint32_t rx = idx & 1;
            uint32_t ry = (idx >> 1) & 1;
            uint32_t rz = (idx >> 2) & 1;

            // Inverse rotation
            rotate_3d_inv(s, coords, rx, ry, rz);

            coords[0] += s * rx;
            coords[1] += s * ry;
            coords[2] += s * rz;
        }

        x = coords[0];
        y = coords[1];
        z = coords[2];
    }

private:
    /// 3D Gray code encode (for Hilbert ordering)
    static inline uint32_t gray_encode_3d(uint32_t i) {
        return i ^ (i >> 1);
    }

    /// 3D Gray code decode (inverse)
    static inline uint32_t gray_decode_3d(uint32_t g) {
        uint32_t i = g;
        i ^= (i >> 1);
        i ^= (i >> 2);
        return i;
    }

    /// Rotate/flip coordinates for encoding
    static inline void rotate_3d(uint32_t n, uint32_t *coords, uint32_t rx, uint32_t ry, uint32_t rz) {
        if (rz == 0) {
            if (ry == 0) {
                if (rx == 0) {
                    // Rotation A: swap x and y
                    uint32_t t = coords[0];
                    coords[0] = coords[1];
                    coords[1] = t;
                } else {
                    // Rotation B: swap y and z, flip x
                    coords[0] = n - 1 - coords[0];
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                }
            } else {
                if (rx == 0) {
                    // Rotation C: swap x and z
                    uint32_t t = coords[0];
                    coords[0] = coords[2];
                    coords[2] = t;
                } else {
                    // Rotation D: swap y and z
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                }
            }
        } else {
            if (ry == 0) {
                if (rx == 0) {
                    // Rotation E: swap x and y, flip z
                    coords[2] = n - 1 - coords[2];
                    uint32_t t = coords[0];
                    coords[0] = coords[1];
                    coords[1] = t;
                } else {
                    // Rotation F: flip y, swap y and z
                    coords[1] = n - 1 - coords[1];
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                }
            } else {
                if (rx == 0) {
                    // Rotation G: flip x, swap x and z
                    coords[0] = n - 1 - coords[0];
                    uint32_t t = coords[0];
                    coords[0] = coords[2];
                    coords[2] = t;
                } else {
                    // Rotation H: flip all, swap y and z
                    coords[0] = n - 1 - coords[0];
                    coords[1] = n - 1 - coords[1];
                    coords[2] = n - 1 - coords[2];
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                }
            }
        }
    }

    /// Inverse rotation for decoding
    static inline void rotate_3d_inv(uint32_t n, uint32_t *coords, uint32_t rx, uint32_t ry, uint32_t rz) {
        if (rz == 0) {
            if (ry == 0) {
                if (rx == 0) {
                    // Inverse of A: swap x and y
                    uint32_t t = coords[0];
                    coords[0] = coords[1];
                    coords[1] = t;
                } else {
                    // Inverse of B
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                    coords[0] = n - 1 - coords[0];
                }
            } else {
                if (rx == 0) {
                    // Inverse of C: swap x and z
                    uint32_t t = coords[0];
                    coords[0] = coords[2];
                    coords[2] = t;
                } else {
                    // Inverse of D: swap y and z
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                }
            }
        } else {
            if (ry == 0) {
                if (rx == 0) {
                    // Inverse of E
                    uint32_t t = coords[0];
                    coords[0] = coords[1];
                    coords[1] = t;
                    coords[2] = n - 1 - coords[2];
                } else {
                    // Inverse of F
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                    coords[1] = n - 1 - coords[1];
                }
            } else {
                if (rx == 0) {
                    // Inverse of G
                    uint32_t t = coords[0];
                    coords[0] = coords[2];
                    coords[2] = t;
                    coords[0] = n - 1 - coords[0];
                } else {
                    // Inverse of H
                    uint32_t t = coords[1];
                    coords[1] = coords[2];
                    coords[2] = t;
                    coords[0] = n - 1 - coords[0];
                    coords[1] = n - 1 - coords[1];
                    coords[2] = n - 1 - coords[2];
                }
            }
        }
    }
};

// ============================================================================
// SFC Type Enumeration and Utilities
// ============================================================================

/// Space-filling curve type selector
enum class SFCType { MORTON_2D, MORTON_3D, HILBERT_2D, HILBERT_3D };

/**
 * Compute the Manhattan distance between two 2D points.
 * Useful for measuring SFC locality.
 */
inline uint64_t manhattan_distance_2d(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2) {
    int64_t dx = static_cast<int64_t>(x1) - static_cast<int64_t>(x2);
    int64_t dy = static_cast<int64_t>(y1) - static_cast<int64_t>(y2);
    return static_cast<uint64_t>((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
}

/**
 * Compute the Manhattan distance between two 3D points.
 */
inline uint64_t manhattan_distance_3d(uint32_t x1, uint32_t y1, uint32_t z1,
                                       uint32_t x2, uint32_t y2, uint32_t z2) {
    int64_t dx = static_cast<int64_t>(x1) - static_cast<int64_t>(x2);
    int64_t dy = static_cast<int64_t>(y1) - static_cast<int64_t>(y2);
    int64_t dz = static_cast<int64_t>(z1) - static_cast<int64_t>(z2);
    return static_cast<uint64_t>((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy) + (dz < 0 ? -dz : dz));
}

} // namespace sfc

#endif // SBBF_SPACE_FILLING_CURVES_HPP
