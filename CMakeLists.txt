cmake_minimum_required(VERSION 3.19)

project(HySet
    VERSION 0.1
    DESCRIPTION "A Set Similarity Join Framework Using GPUs"
    LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD  14)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

# Add external dependencies
add_subdirectory(external/concurrentqueue)
add_subdirectory(external/cuda-api-wrappers)
add_subdirectory(external/cxxopts)
add_subdirectory(external/fmt)
add_subdirectory(external/json)

# The executable code is here
add_subdirectory(src)
