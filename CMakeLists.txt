cmake_minimum_required(VERSION 3.19)

project(HySet
    VERSION 0.1
    DESCRIPTION "A Set Similarity Join Framework Using GPUs"
    LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD  14)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

include(FetchContent)

# CXX options library
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)
# Adds cxxopts

# JSON library
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
    GIT_TAG        v3.8.0
)
FetchContent_MakeAvailable(json)
# Adds cxxopts

# cuda-api-wrappers library
FetchContent_Declare(
    cuda-api-wrappers
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG        v0.4
)
FetchContent_MakeAvailable(cuda-api-wrappers)
# Adds cuda-api-wrappers

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
)
FetchContent_MakeAvailable(fmt)
# Adds fmt::fmt

FetchContent_Declare(
        concurrentqueue
        GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
        GIT_TAG        9cfda6cc61065d016ae3f51f486ce0fae563ea87
)
FetchContent_MakeAvailable(concurrentqueue)
# Adds moodycamel::ConcurrentQueue

# The executable code is here
add_subdirectory(src)
