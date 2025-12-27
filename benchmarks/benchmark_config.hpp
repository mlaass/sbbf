#pragma once

// ============================================================================
// SBBF Benchmark Configuration
// ============================================================================
//
// Centralized k value for SBBF benchmarks used in FPR tables and figures.
// Change this single value to update all FPR experiments.
//
// Note: Topological Verification (sbbf_voxel_experiment.py) uses k=4 separately
// and is NOT affected by this constant.
//
// Python scripts should use HASH_K = 8 (matching this value) with a comment
// referencing this header.
// ============================================================================

constexpr unsigned SBBF_BENCHMARK_HASH_K = 8;
