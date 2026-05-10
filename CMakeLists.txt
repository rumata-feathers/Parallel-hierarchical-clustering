cmake_minimum_required(VERSION 3.16)
project(ParallelHAC CXX)

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

add_executable(hac
    src/main.cpp
    src/hac/hac_serial.cpp
    src/hac/hac_parallel.cpp
    src/io/data_loader.cpp
    src/io/output_writer.cpp
)

target_include_directories(hac PRIVATE src)
target_link_libraries(hac PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hac PRIVATE -O2 -Wall)
