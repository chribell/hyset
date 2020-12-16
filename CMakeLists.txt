# Works with 3.11 and tested through 3.16
cmake_minimum_required(VERSION 3.16)

# Project name and a few useful settings. Other commands can pick up the results
project(HySet
    VERSION 0.1
    DESCRIPTION "A Set Similarity Join Framework Using GPUs"
    LANGUAGES CUDA CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here
    
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

include(FetchContent)

# CXX options library
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v2.2.0
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
    GIT_TAG        v0.3.3
)
FetchContent_MakeAvailable(cuda-api-wrappers)
# Adds cuda-api-wrappers

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.0.2
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

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)
