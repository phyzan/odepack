cmake_minimum_required(VERSION 3.15...3.26)


set(CMAKE_CXX_COMPILER "clang++")

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

# Create the Python extension module
pybind11_add_module(odesolvers src/pyode.cpp)

# Include directories - point to the parent directory to make includes work
target_include_directories(odesolvers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add compile options from .clangd
target_compile_options(odesolvers PRIVATE
    -O3
    -DNDEBUG
    -Wall
    -Wextra
    -ftemplate-backtrace-limit=0
    -Wno-unused-parameter
    -fno-math-errno
    -fno-trapping-math
    -fopenmp
    -march=x86-64
)

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(odesolvers PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add MPFR library support
find_library(MPFR_LIBRARY mpfr REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(odesolvers PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})

# Installation rules
# Install the compiled module into the odepack package
install(TARGETS odesolvers
    LIBRARY DESTINATION odepack
)

# Install Python files from the odepack source directory
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/odepack/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/odepack/odesolvers.pyi
    DESTINATION odepack
)
