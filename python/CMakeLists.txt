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

# Common compile options
set(COMMON_COMPILE_OPTIONS
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

# Find MPFR and GMP libraries
find_library(MPFR_LIBRARY mpfr REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)

# Find OpenMP
find_package(OpenMP)

# Helper macro to create backend modules
macro(add_ode_backend backend_name source_file)
    pybind11_add_module(${backend_name} ${source_file})

    target_include_directories(${backend_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_options(${backend_name} PRIVATE ${COMMON_COMPILE_OPTIONS})

    target_link_libraries(${backend_name} PRIVATE OpenMP::OpenMP_CXX)

    # All backends need MPFR support (for eventual compatibility)
    target_link_libraries(${backend_name} PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})
endmacro()

# Create the main module that defines common tools (compiled once, shared by all backends)
pybind11_add_module(common src/common.cpp)
target_include_directories(common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(common PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_libraries(common PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(common PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})

# Create backend modules for different precision types
add_ode_backend(odesolvers_float src/pyode_float.cpp)
add_ode_backend(odesolvers_double src/pyode_double.cpp)
add_ode_backend(odesolvers_longdouble src/pyode_longdouble.cpp)
add_ode_backend(odesolvers_mpreal src/pyode_mpreal.cpp)

# Installation rules
# Install the compiled modules into the odepack package
install(TARGETS common odesolvers_float odesolvers_double odesolvers_longdouble odesolvers_mpreal
    LIBRARY DESTINATION odepack
)