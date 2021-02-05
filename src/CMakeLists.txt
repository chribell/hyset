add_executable(cpu_single cpu_single.cu)
target_link_libraries(cpu_single PRIVATE cxxopts fmt::fmt)
target_include_directories(cpu_single PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        cpu_single
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cpu_single PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(cpu_multi cpu_multi.cu)
target_link_libraries(cpu_multi PRIVATE cxxopts fmt::fmt)
target_include_directories(cpu_multi PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        cpu_multi
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cpu_multi PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(dichotomy dichotomy.cu)
target_link_libraries(dichotomy PRIVATE cxxopts fmt::fmt)
target_include_directories(dichotomy PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        dichotomy
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(dichotomy PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(hybrid hybrid.cu)
target_link_libraries(hybrid PRIVATE cxxopts fmt::fmt)
target_include_directories(hybrid PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        hybrid
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(hybrid PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(fgss fgss.cu)
target_link_libraries(fgss PRIVATE cxxopts fmt::fmt)
target_include_directories(fgss PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        fgss
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(fgss PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(bitmap bitmap.cu)
target_link_libraries(bitmap PRIVATE cxxopts fmt::fmt)
target_include_directories(bitmap PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        bitmap
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(bitmap PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(queue queue.cu)
target_link_libraries(queue PRIVATE cxxopts fmt::fmt)
target_include_directories(queue PRIVATE ../include ${concurrentqueue_SOURCE_DIR} ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        queue
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(queue PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)
