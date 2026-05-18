cmake_minimum_required(VERSION 3.15...3.26)
project(odepack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set root path and include external dependencies
set(ODEPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include(${ODEPACK_ROOT}/cmake/external.cmake)

# Main odepack interface target
add_library(odepack INTERFACE)
add_library(odepack::odepack ALIAS odepack)
target_link_libraries(odepack INTERFACE
    odepack::headers
    odepack::autodiff
    odepack::qhull
)

# Optional MPREAL support
option(ODEPACK_ENABLE_MPREAL "Enable MPFR multi-precision support" OFF)
if(ODEPACK_ENABLE_MPREAL)
    find_library(MPFR_LIBRARY mpfr REQUIRED)
    find_library(GMP_LIBRARY gmp REQUIRED)
    target_link_libraries(odepack INTERFACE odepack::mpreal ${MPFR_LIBRARY} ${GMP_LIBRARY})
    target_compile_definitions(odepack INTERFACE MPREAL)
endif()
