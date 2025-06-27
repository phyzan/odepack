cmake_minimum_required(VERSION 3.14)
project(odepack LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
add_compile_options(-O3 -fno-math-errno -Wall -march=x86-64 -fopenmp -fPIC)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
find_package(pybind11 REQUIRED)

# Create the shared library as a Python module
pybind11_add_module(odepack MODULE pyode.cpp)

# Link extra libraries
target_link_libraries(odepack PRIVATE mpfr gmp)

# Include directories explicitly, if not found via pybind11
target_include_directories(odepack PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    /usr/include/pybind11
    /usr/include/python3.12
)