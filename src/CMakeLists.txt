add_executable(cpu cpu.cu)
target_link_libraries(cpu PRIVATE cxxopts fmt::fmt)
target_include_directories(cpu PRIVATE ../include ${cuda-api-wrappers_SOURCE_DIR}/src)
set_target_properties(
        cpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(cpu PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

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
