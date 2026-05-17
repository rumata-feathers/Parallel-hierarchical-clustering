cmake_minimum_required(VERSION 3.18)
project(ParallelHAC CXX)

option(ENABLE_CUDA "Build with CUDA HAC support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(APPLE)
    set(CMAKE_CXX_COMPILER clang++)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found. Please install libomp using Homebrew: brew install libomp")
endif()

set(HAC_SOURCES
    src/main.cpp
    src/hac/hac_serial.cpp
    src/hac/hac_parallel.cpp
    src/io/data_loader.cpp
    src/io/output_writer.cpp
)

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
    list(APPEND HAC_SOURCES src/hac/hac_cuda.cu)
endif()

add_executable(hac ${HAC_SOURCES})

target_include_directories(hac PRIVATE src)
target_link_libraries(hac PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hac PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall>)

if(ENABLE_CUDA)
    target_compile_definitions(hac PRIVATE ENABLE_CUDA)
    target_compile_options(hac PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-relaxed-constexpr>)
endif()

# ---------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------
set(TEST_SOURCES
    tests/test_correctness.cpp
    src/hac/hac_serial.cpp
    src/hac/hac_parallel.cpp
)
if(ENABLE_CUDA)
    list(APPEND TEST_SOURCES src/hac/hac_cuda.cu)
endif()

add_executable(test_hac ${TEST_SOURCES})
target_include_directories(test_hac PRIVATE src)
target_link_libraries(test_hac PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(test_hac PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall>)
if(ENABLE_CUDA)
    target_compile_definitions(test_hac PRIVATE ENABLE_CUDA)
    target_compile_options(test_hac PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-relaxed-constexpr>)
endif()

enable_testing()
add_test(NAME correctness COMMAND test_hac)
