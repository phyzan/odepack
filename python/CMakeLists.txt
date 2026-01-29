cmake_minimum_required(VERSION 3.15...3.26)

# optional flags: -DMPREAL=ON -DDEBUG=ON
option(MPREAL "Enable MPFR multi-precision support" OFF)
option(DEBUG "Enable debug build" OFF)

set(CMAKE_CXX_COMPILER "g++")

project(odepack)

# Set C++ standard (using C++20 as per .clangd)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find pybind11 via find_package or fetch it
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Common compile options
set(COMMON_COMPILE_OPTIONS
    -Wall
    -Wextra
    -ftemplate-backtrace-limit=0
    -Wno-unused-parameter
    -fopenmp
    -march=x86-64
    -DPYBIND11_DETAILED_ERROR_MESSAGES
)

if(MPREAL)
    list(APPEND COMMON_COMPILE_OPTIONS -DMPREAL)
endif()

# Add optimization and debug flags based on debug mode
if(DEBUG)
    list(APPEND COMMON_COMPILE_OPTIONS -O0 -g -ggdb3 -fno-omit-frame-pointer)
else()
    list(APPEND COMMON_COMPILE_OPTIONS -O3 -DNDEBUG -fno-math-errno -fno-trapping-math)
endif()


if(MPREAL)
    find_library(MPFR_LIBRARY mpfr REQUIRED)
    find_library(GMP_LIBRARY gmp REQUIRED)
endif()

# Find OpenMP
find_package(OpenMP)

# Create the main odesolvers module
pybind11_add_module(odesolvers src/pyode.cpp)
target_include_directories(odesolvers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(odesolvers PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_libraries(odesolvers PRIVATE OpenMP::OpenMP_CXX)
if(MPREAL)
    target_link_libraries(odesolvers PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})
endif()

# Installation rules
# Install the compiled module into the odepack package
install(TARGETS odesolvers
    LIBRARY DESTINATION odepack
)