cmake_minimum_required(VERSION 3.22.1)
project(openjev C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Same llama.cpp source as the Termux package and convert/build.py (v0.4.1),
# so the app runs exactly the graph that convert/verify.py measured.
set(LLAMA_SRC_DIR "" CACHE PATH "Local llama.cpp checkout at v0.4.1 (empty: download it)")
if(LLAMA_SRC_DIR)
    add_subdirectory(${LLAMA_SRC_DIR} llama-build)
else()
    include(FetchContent)
    FetchContent_Declare(llama
        URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/v0.4.1.tar.gz
        URL_HASH SHA256=ef3d5b1907a391500ae11b5e61a8e2022e0deaac9790899cad9c4e02f03bfb9a)
    FetchContent_MakeAvailable(llama)
endif()

add_library(openjev SHARED openjev.cpp)
target_link_libraries(openjev llama ggml android log)
