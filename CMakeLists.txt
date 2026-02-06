cmake_minimum_required(VERSION 3.15...3.26)

# optional flags: -DMPREAL=ON -DDEBUG=ON, -DRK4_DENSE=ON, -DBUILD_PYTHON=ON
option(MPREAL "Enable MPFR multi-precision support" OFF)
option(DEBUG "Enable debug build" OFF)
option(RK4_DENSE "Enable accurate RK4 dense output" OFF) #by default Hermite interpolation is used
option(BUILD_PYTHON "Build Python extension module" OFF)

set(CMAKE_CXX_COMPILER "g++")

project(odepack)

# Check for g++ compiler and minimum version 13.3.0
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "This project requires g++ (GNU C++ compiler). Found: ${CMAKE_CXX_COMPILER_ID}")
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.3.0")
    message(FATAL_ERROR "g++ version 13.3.0 or higher is required. Found: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

message(STATUS "Using g++ version: ${CMAKE_CXX_COMPILER_VERSION}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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
if(RK4_DENSE)
    list(APPEND COMMON_COMPILE_OPTIONS -DRK4_DENSE)
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

#===========================================================================================
#                           INTERFACE target: pure C++ header-only library
#===========================================================================================

add_library(odepack_headers INTERFACE)
target_include_directories(odepack_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/odepack>
)
target_compile_features(odepack_headers INTERFACE cxx_std_20)

#===========================================================================================
#                           STATIC library: compiled Py* wrapper classes
#===========================================================================================

add_library(odepack_lib STATIC python/src/pyode.cpp)
target_link_libraries(odepack_lib PUBLIC odepack_headers pybind11::pybind11 Python::Module)
if(OpenMP_FOUND)
    target_link_libraries(odepack_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(odepack_lib PRIVATE ${COMMON_COMPILE_OPTIONS})
set_target_properties(odepack_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME odepack
)
if(MPREAL)
    target_link_libraries(odepack_lib PUBLIC ${MPFR_LIBRARY} ${GMP_LIBRARY})
endif()

#===========================================================================================
#                           Install rules (C++ library)
#===========================================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/odepack
    COMPONENT cpp
    FILES_MATCHING PATTERN "*.hpp"
)

# Install static library
install(TARGETS odepack_lib odepack_headers
    EXPORT odepackTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cpp
)

# Export targets
install(EXPORT odepackTargets
    FILE odepackTargets.cmake
    NAMESPACE odepack::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odepack
    COMPONENT cpp
)

# Generate config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/odepackConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/odepackConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odepack
)

# Generate version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/odepackConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/odepackConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/odepackConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/odepack
    COMPONENT cpp
)

#===========================================================================================
#                           Python extension (optional)
#===========================================================================================

if(SKBUILD OR BUILD_PYTHON)
    add_subdirectory(python)
endif()
