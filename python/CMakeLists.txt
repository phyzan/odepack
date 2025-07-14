cmake_minimum_required(VERSION 3.14)
project(odepack LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
add_compile_options(-O3 -fno-math-errno -Wall -march=x86-64 -fopenmp -fPIC)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Create the shared library as a Python module
pybind11_add_module(odepack MODULE pyode.cpp)

# Link extra libraries to the module
target_link_libraries(odepack PRIVATE
    OpenMP::OpenMP_CXX
    mpfr
    gmp
)

# Include directories if not found automatically
target_include_directories(odepack PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    /usr/include/pybind11
    /usr/include/python3.12
)
