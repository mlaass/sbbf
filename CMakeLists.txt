cmake_minimum_required(VERSION 3.14)
project(sbbf VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(SBBF_BUILD_TESTS "Build unit tests" ON)
option(SBBF_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SBBF_BUILD_PYTHON "Build Python bindings" OFF)
option(SBBF_WITH_HDF5 "Enable HDF5 support for benchmarks" OFF)
option(SBBF_WITH_BASELINES "Include baseline bloom filter implementations" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    endif()
endif()

# ============================================================================
# Header-only library target
# ============================================================================

add_library(sbbf INTERFACE)
target_include_directories(sbbf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(sbbf INTERFACE cxx_std_17)

# ============================================================================
# Tests
# ============================================================================

if(SBBF_BUILD_TESTS)
    enable_testing()

    add_executable(test_sbbf tests/test_sbbf.cpp)
    target_link_libraries(test_sbbf PRIVATE sbbf)
    target_compile_options(test_sbbf PRIVATE -O2)

    add_test(NAME sbbf_unit_tests COMMAND test_sbbf)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(SBBF_BUILD_BENCHMARKS)
    # Check for nanobench
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/nanobench.h")
        message(STATUS "Found nanobench.h")
    else()
        message(WARNING "nanobench.h not found in external/. Benchmarks may not build.")
    endif()

    # Check for nlohmann/json
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/json.hpp")
        message(STATUS "Found json.hpp")
    else()
        message(WARNING "json.hpp not found in external/. Benchmarks may not build.")
    endif()

    # Main benchmark
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/sbbf_benchmark.cpp")
        add_executable(sbbf_benchmark benchmarks/sbbf_benchmark.cpp)
        target_link_libraries(sbbf_benchmark PRIVATE sbbf)
        target_include_directories(sbbf_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
        target_compile_options(sbbf_benchmark PRIVATE -O3)

        if(SBBF_WITH_HDF5)
            find_package(HDF5 REQUIRED)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/HighFive/CMakeLists.txt")
                add_subdirectory(lib/HighFive EXCLUDE_FROM_ALL)
                target_link_libraries(sbbf_benchmark PRIVATE HighFive)
                target_compile_definitions(sbbf_benchmark PRIVATE SBBF_HAS_HDF5)
            endif()
        endif()

        if(SBBF_WITH_BASELINES)
            target_include_directories(sbbf_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baselines)
            target_compile_definitions(sbbf_benchmark PRIVATE SBBF_WITH_BASELINES)
        endif()
    endif()

    # K-sensitivity benchmark
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/sbbf_k_sensitivity.cpp")
        add_executable(sbbf_k_sensitivity benchmarks/sbbf_k_sensitivity.cpp)
        target_link_libraries(sbbf_k_sensitivity PRIVATE sbbf)
        target_include_directories(sbbf_k_sensitivity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
        target_compile_options(sbbf_k_sensitivity PRIVATE -O3)
    endif()

    # Seed strategy benchmark
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/sbbf_seed_strategy.cpp")
        add_executable(sbbf_seed_strategy benchmarks/sbbf_seed_strategy.cpp)
        target_link_libraries(sbbf_seed_strategy PRIVATE sbbf)
        target_include_directories(sbbf_seed_strategy PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/external
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
        target_compile_options(sbbf_seed_strategy PRIVATE -O3)

        if(SBBF_WITH_HDF5)
            if(TARGET HighFive)
                target_link_libraries(sbbf_seed_strategy PRIVATE HighFive)
                target_compile_definitions(sbbf_seed_strategy PRIVATE SBBF_HAS_HDF5)
            endif()
        endif()
    endif()
endif()

# ============================================================================
# Python bindings
# ============================================================================

if(SBBF_BUILD_PYTHON)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
        add_subdirectory(lib/pybind11)

        pybind11_add_module(_sbbf python/sbbf_bindings.cpp)
        target_link_libraries(_sbbf PRIVATE sbbf)
        target_compile_options(_sbbf PRIVATE -O3)

        # Install Python module to python directory
        install(TARGETS _sbbf DESTINATION python)
    else()
        message(WARNING "pybind11 not found in lib/pybind11. Python bindings will not be built.")
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/sbbf
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install library target
install(TARGETS sbbf
    EXPORT sbbf-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT sbbf-targets
    FILE sbbf-targets.cmake
    NAMESPACE sbbf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sbbf
)

# Package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sbbf-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sbbf-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sbbf
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sbbf-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sbbf-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sbbf-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sbbf
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "SBBF Configuration Summary:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build tests:    ${SBBF_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${SBBF_BUILD_BENCHMARKS}")
message(STATUS "  Build Python:   ${SBBF_BUILD_PYTHON}")
message(STATUS "  With HDF5:      ${SBBF_WITH_HDF5}")
message(STATUS "  With baselines: ${SBBF_WITH_BASELINES}")
message(STATUS "")
