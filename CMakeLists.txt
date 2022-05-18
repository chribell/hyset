cmake_minimum_required(VERSION 3.19)

project(HySet
    VERSION 0.1
    DESCRIPTION "A Set Similarity Join Framework Using GPUs"
    LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD  14)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

# Add external dependencies
add_subdirectory(external/concurrentqueue)
add_subdirectory(external/cuda-api-wrappers)
add_subdirectory(external/cxxopts)
add_subdirectory(external/fmt)
add_subdirectory(external/json)

add_executable(cpu_single src/cpu_single.cu)
target_link_libraries(cpu_single PRIVATE cxxopts fmt::fmt)
target_include_directories(cpu_single PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        cpu_single
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cpu_single PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(cpu_multi src/cpu_multi.cu)
target_link_libraries(cpu_multi PRIVATE cxxopts fmt::fmt)
target_include_directories(cpu_multi PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        cpu_multi
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cpu_multi PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(dichotomy src/dichotomy.cu)
target_link_libraries(dichotomy PRIVATE cxxopts fmt::fmt)
target_include_directories(dichotomy PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        dichotomy
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(dichotomy PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(hybrid src/hybrid.cu)
target_link_libraries(hybrid PRIVATE cxxopts fmt::fmt)
target_include_directories(hybrid PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        hybrid
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(hybrid PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(fgss src/fgss.cu)
target_link_libraries(fgss PRIVATE cxxopts fmt::fmt)
target_include_directories(fgss PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        fgss
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(fgss PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(bitmap src/bitmap.cu)
target_link_libraries(bitmap PRIVATE cxxopts fmt::fmt)
target_include_directories(bitmap PRIVATE include external/cuda-api-wrappers/src)
set_target_properties(
        bitmap
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(bitmap PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(queue src/queue.cu)
target_link_libraries(queue PRIVATE cxxopts fmt::fmt)
target_include_directories(queue PRIVATE include external/concurrentqueue external/cuda-api-wrappers/src)
set_target_properties(
        queue
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(queue PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)



