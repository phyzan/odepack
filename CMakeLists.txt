cmake_minimum_required(VERSION 3.15...3.26)

# optional flags: -DMPREAL=ON -DDEBUG=ON, -DRK4_DENSE=ON, -DBUILD_PYTHON=ON, -DNO_ODE_WARN=ON
option(MPREAL "Enable MPFR multi-precision support" OFF)
option(DEBUG "Enable debug build" OFF)
option(RK4_DENSE "Enable accurate RK4 dense output" OFF) #by default Hermite interpolation is used
option(BUILD_PYTHON "Build Python extension module" OFF)
option(NO_ODE_WARN "Disable ODE warnings" OFF)
option(NO_NAN_CHECK "Disable NaN/inf checks for performance (not recommended)" OFF)
option(ODEPACK_NATIVE_ARCH "Enable -march=native for local builds" OFF)
option(ODEPACK_STRIP_INSTALL "Strip binaries at install time" OFF)

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

# Prefer standard CMake build type semantics for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# Reduce exported symbol surface and inline symbol leakage in release-oriented builds.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

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
    -Wno-attributes
    -DPYBIND11_DETAILED_ERROR_MESSAGES
)

if(ODEPACK_NATIVE_ARCH)
    list(APPEND COMMON_COMPILE_OPTIONS -march=native)
else()
    list(APPEND COMMON_COMPILE_OPTIONS -march=x86-64)
endif()

if(MPREAL)
    list(APPEND COMMON_COMPILE_OPTIONS -DMPREAL)
endif()
if(NO_NAN_CHECK)
    list(APPEND COMMON_COMPILE_OPTIONS -DNO_NAN_CHECK)
endif()
if(RK4_DENSE)
    list(APPEND COMMON_COMPILE_OPTIONS -DRK4_DENSE)
endif()

# Add optimization and debug flags based on debug mode
set(ODEPACK_DEBUG_BUILD OFF)
if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ODEPACK_DEBUG_BUILD ON)
endif()

set(COMMON_LINK_OPTIONS)

if(ODEPACK_DEBUG_BUILD)
    list(APPEND COMMON_COMPILE_OPTIONS
        -O0
        -g
        -ggdb3
        -fno-omit-frame-pointer
        -UNDEBUG
    )
else()
    list(APPEND COMMON_COMPILE_OPTIONS
        -O3
        -DNDEBUG
        -g0
        -fno-math-errno
        -fno-trapping-math
        -ffunction-sections
        -fdata-sections
    )
    list(APPEND COMMON_LINK_OPTIONS
        -Wl,--gc-sections
        -Wl,--as-needed
    )

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ODEPACK_IPO_SUPPORTED OUTPUT ODEPACK_IPO_ERROR)
    if(ODEPACK_IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "IPO/LTO not enabled: ${ODEPACK_IPO_ERROR}")
    endif()
endif()

if(ODEPACK_DEBUG_BUILD)
    message(STATUS "ODEPACK_DEBUG_BUILD=ON (asserts enabled)")
else()
    message(STATUS "ODEPACK_DEBUG_BUILD=OFF (asserts disabled via NDEBUG)")
endif()

if(NO_ODE_WARN)
    list(APPEND COMMON_COMPILE_OPTIONS -DNO_ODE_WARN)
endif()

if(MPREAL)
    find_library(MPFR_LIBRARY mpfr REQUIRED)
    find_library(GMP_LIBRARY gmp REQUIRED)
endif()

# Find OpenMP
find_package(OpenMP)

# Find Qhull (for Delaunay triangulation in LinearNdInterpolator)
# libqhull-dev on Ubuntu installs to /usr/include/libqhull_r and /usr/lib
find_library(QHULL_R_LIBRARY NAMES qhull_r)
find_path(QHULL_INCLUDE_DIR libqhull_r/qhull_ra.h)

# Find Qhull (for Delaunay triangulation in LinearNdInterpolator)
find_package(Qhull CONFIG QUIET)
if(NOT Qhull_FOUND)
    # Try pkg-config as fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(QHULL QUIET qhull_r)
    endif()
endif()

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
#                           STATIC libraries: compiled Python wrapper classes
#===========================================================================================

# Check for Qhull first (needed by odepack_fields)
if(NOT QHULL_R_LIBRARY OR NOT QHULL_INCLUDE_DIR)
    message(FATAL_ERROR "Qhull not found. LinearNdInterpolator requires Qhull. Install with: sudo apt install libqhull-dev")
else()
    message(STATUS "Found Qhull: ${QHULL_R_LIBRARY}")
endif()

# Helper function to configure static library targets
function(configure_odepack_lib target_name)
    target_link_libraries(${target_name} PUBLIC odepack_headers pybind11::pybind11 Python::Module)
    if(OpenMP_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(${target_name} PRIVATE ${COMMON_COMPILE_OPTIONS})
    if(COMMON_LINK_OPTIONS)
        target_link_options(${target_name} PRIVATE ${COMMON_LINK_OPTIONS})
    endif()
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MPREAL)
        target_link_libraries(${target_name} PUBLIC ${MPFR_LIBRARY} ${GMP_LIBRARY})
    endif()
endfunction()

if(ODEPACK_STRIP_INSTALL)
    set(CMAKE_INSTALL_DO_STRIP TRUE)
endif()

# 0. odepack_pytools: Shared Python utilities (DtypeDispatcher, etc.) - no dependencies
add_library(odepack_pytools STATIC
    python/src/staticlib/PyTools.cpp
)
configure_odepack_lib(odepack_pytools)
set_target_properties(odepack_pytools PROPERTIES OUTPUT_NAME odepack_pytools)

# 1. odepack_events: Event handling implementations (depends on pytools)
add_library(odepack_events STATIC
    python/src/staticlib/PyEvents.cpp
)
configure_odepack_lib(odepack_events)
target_link_libraries(odepack_events PUBLIC odepack_pytools)
set_target_properties(odepack_events PROPERTIES OUTPUT_NAME odepack_events)

# 2. odepack_solvers: Core solver implementations (depends on events, pytools)
add_library(odepack_solvers STATIC
    python/src/staticlib/PySolver.cpp
    python/src/staticlib/PySubSolver.cpp
    python/src/staticlib/PyTools.cpp
)
configure_odepack_lib(odepack_solvers)
target_link_libraries(odepack_solvers PUBLIC odepack_events odepack_pytools)
set_target_properties(odepack_solvers PROPERTIES OUTPUT_NAME odepack_solvers)

# 3. odepack_oderesult: ODE result implementations
add_library(odepack_oderesult STATIC
    python/src/staticlib/PyResult.cpp
)
configure_odepack_lib(odepack_oderesult)
target_link_libraries(odepack_oderesult PUBLIC odepack_solvers)
set_target_properties(odepack_oderesult PROPERTIES OUTPUT_NAME odepack_oderesult)

# 4. odepack_ode: Low-level ODE interface implementations
add_library(odepack_ode STATIC
    python/src/staticlib/PyOde.cpp
)
configure_odepack_lib(odepack_ode)
target_link_libraries(odepack_ode PUBLIC odepack_solvers odepack_events odepack_oderesult odepack_pytools)
set_target_properties(odepack_ode PROPERTIES OUTPUT_NAME odepack_ode)

# 5. odepack_chaos: Variational/Lyapunov solver implementations
add_library(odepack_chaos STATIC
    python/src/staticlib/PyChaos.cpp
)
configure_odepack_lib(odepack_chaos)
target_link_libraries(odepack_chaos PUBLIC odepack_solvers odepack_events odepack_oderesult odepack_ode odepack_pytools)
set_target_properties(odepack_chaos PROPERTIES OUTPUT_NAME odepack_chaos)

# 6. odepack_fields: Vector field implementations (needs Qhull)
add_library(odepack_fields STATIC
    python/src/staticlib/PyField.cpp
)
configure_odepack_lib(odepack_fields)
target_include_directories(odepack_fields PUBLIC ${QHULL_INCLUDE_DIR})
target_link_libraries(odepack_fields PUBLIC odepack_solvers odepack_oderesult odepack_ode odepack_pytools ${QHULL_R_LIBRARY})
set_target_properties(odepack_fields PROPERTIES OUTPUT_NAME odepack_fields)

# Convenience target that groups all static libraries
add_library(odepack_all INTERFACE)
target_link_libraries(odepack_all INTERFACE
    odepack_pytools
    odepack_solvers
    odepack_events
    odepack_oderesult
    odepack_ode
    odepack_chaos
    odepack_fields
)

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

# Install static libraries
install(TARGETS
    odepack_pytools
    odepack_headers
    odepack_solvers
    odepack_events
    odepack_oderesult
    odepack_ode
    odepack_chaos
    odepack_fields
    odepack_all
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