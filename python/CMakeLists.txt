cmake_minimum_required(VERSION 3.15...3.26)
project(odepack_python LANGUAGES CXX)

# Build options
option(MPREAL "Enable MPFR multi-precision support" OFF)
option(DEBUG "Enable debug build" OFF)
option(RK4_DENSE "Enable accurate RK4 dense output" OFF)
option(NO_ODE_WARN "Disable ODE warnings" OFF)
option(NO_NAN_CHECK "Disable NaN/inf checks for performance" OFF)
option(ODEPACK_NATIVE_ARCH "Enable -march=native for local builds" OFF)
option(ODEPACK_STRIP_INSTALL "Strip binaries at install time" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#===========================================================================================
#                           Compiler checks
#===========================================================================================

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "This project requires g++ (GNU C++ compiler). Found: ${CMAKE_CXX_COMPILER_ID}")
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.3.0")
    message(FATAL_ERROR "g++ version 13.3.0 or higher is required. Found: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

message(STATUS "Using g++ version: ${CMAKE_CXX_COMPILER_VERSION}")

#===========================================================================================
#                           Build type configuration
#===========================================================================================

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

#===========================================================================================
#                           Dependencies
#===========================================================================================

# Python and pybind11
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

# OpenMP
find_package(OpenMP)

# Bundled external dependencies
set(ODEPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include(${ODEPACK_ROOT}/cmake/external.cmake)

# MPFR/GMP (optional)
if(MPREAL)
    find_library(MPFR_LIBRARY mpfr REQUIRED)
    find_library(GMP_LIBRARY gmp REQUIRED)
endif()

#===========================================================================================
#                           Compile options
#===========================================================================================

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
if(NO_ODE_WARN)
    list(APPEND COMMON_COMPILE_OPTIONS -DNO_ODE_WARN)
endif()

set(ODEPACK_DEBUG_BUILD OFF)
if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ODEPACK_DEBUG_BUILD ON)
endif()

set(COMMON_LINK_OPTIONS)

if(ODEPACK_DEBUG_BUILD)
    list(APPEND COMMON_COMPILE_OPTIONS -O0 -g3 -ggdb3 -fno-omit-frame-pointer -UNDEBUG)
    message(STATUS "ODEPACK_DEBUG_BUILD=ON (asserts enabled)")
else()
    list(APPEND COMMON_COMPILE_OPTIONS -O3 -DNDEBUG -g0 -fno-math-errno -fno-trapping-math -ffunction-sections -fdata-sections)
    list(APPEND COMMON_LINK_OPTIONS -Wl,--gc-sections -Wl,--as-needed)

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ODEPACK_IPO_SUPPORTED OUTPUT ODEPACK_IPO_ERROR)
    if(ODEPACK_IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "IPO/LTO not enabled: ${ODEPACK_IPO_ERROR}")
    endif()
    message(STATUS "ODEPACK_DEBUG_BUILD=OFF (asserts disabled via NDEBUG)")
endif()

if(ODEPACK_STRIP_INSTALL)
    set(CMAKE_INSTALL_DO_STRIP TRUE)
endif()

#===========================================================================================
#                           Static libraries
#===========================================================================================

function(configure_odepack_lib target_name)
    target_link_libraries(${target_name} PUBLIC
        odepack_headers
        odepack_autodiff
        pybind11::pybind11
        Python::Module
    )
    if(OpenMP_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(${target_name} PRIVATE ${COMMON_COMPILE_OPTIONS})
    if(COMMON_LINK_OPTIONS)
        target_link_options(${target_name} PRIVATE ${COMMON_LINK_OPTIONS})
    endif()
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MPREAL)
        target_link_libraries(${target_name} PUBLIC odepack_mpreal ${MPFR_LIBRARY} ${GMP_LIBRARY})
    endif()
endfunction()

# 0. odepack_pytools: Shared Python utilities
add_library(odepack_pytools STATIC src/staticlib/PyTools.cpp)
configure_odepack_lib(odepack_pytools)

# 1. odepack_events: Event handling
add_library(odepack_events STATIC src/staticlib/PyEvents.cpp)
configure_odepack_lib(odepack_events)
target_link_libraries(odepack_events PUBLIC odepack_pytools)

# 2. odepack_solvers: Core solver implementations
add_library(odepack_solvers STATIC
    src/staticlib/PySolver.cpp
    src/staticlib/PySubSolver.cpp
    src/staticlib/PyTools.cpp
)
configure_odepack_lib(odepack_solvers)
target_link_libraries(odepack_solvers PUBLIC odepack_events odepack_pytools)

# 3. odepack_oderesult: ODE result implementations
add_library(odepack_oderesult STATIC src/staticlib/PyResult.cpp)
configure_odepack_lib(odepack_oderesult)
target_link_libraries(odepack_oderesult PUBLIC odepack_solvers)

# 4. odepack_ode: Low-level ODE interface
add_library(odepack_ode STATIC src/staticlib/PyOde.cpp)
configure_odepack_lib(odepack_ode)
target_link_libraries(odepack_ode PUBLIC odepack_solvers odepack_events odepack_oderesult odepack_pytools)

# 5. odepack_chaos: Variational/Lyapunov solvers
add_library(odepack_chaos STATIC src/staticlib/PyChaos.cpp)
configure_odepack_lib(odepack_chaos)
target_link_libraries(odepack_chaos PUBLIC odepack_solvers odepack_events odepack_oderesult odepack_ode odepack_pytools)

# 6. odepack_interp: N-D interpolation (needs Qhull)
add_library(odepack_interp STATIC src/staticlib/PyInterp.cpp)
configure_odepack_lib(odepack_interp)
target_link_libraries(odepack_interp PUBLIC odepack_solvers odepack_oderesult odepack_ode odepack_pytools odepack_qhull)

# 7. odepack_fields: Vector field implementations
add_library(odepack_fields STATIC src/staticlib/PyField.cpp)
configure_odepack_lib(odepack_fields)
target_link_libraries(odepack_fields PUBLIC odepack_interp)

#===========================================================================================
#                           Python extension modules
#===========================================================================================

function(configure_py_module target_name)
    target_compile_options(${target_name} PRIVATE ${COMMON_COMPILE_OPTIONS})
    if(MPREAL)
        target_compile_definitions(${target_name} PRIVATE MPREAL)
    endif()
endfunction()

pybind11_add_module(solvers src/bindings/solvers.cpp)
target_link_libraries(solvers PRIVATE odepack_solvers)
configure_py_module(solvers)

pybind11_add_module(events src/bindings/events.cpp)
target_link_libraries(events PRIVATE odepack_events)
configure_py_module(events)

pybind11_add_module(oderesult src/bindings/oderesult.cpp)
target_link_libraries(oderesult PRIVATE odepack_oderesult)
configure_py_module(oderesult)

pybind11_add_module(ode src/bindings/ode.cpp)
target_link_libraries(ode PRIVATE odepack_ode)
configure_py_module(ode)

pybind11_add_module(chaos src/bindings/chaos.cpp)
target_link_libraries(chaos PRIVATE odepack_chaos)
configure_py_module(chaos)

pybind11_add_module(interp src/bindings/interp.cpp)
target_link_libraries(interp PRIVATE odepack_interp)
configure_py_module(interp)

pybind11_add_module(fields src/bindings/fields.cpp)
target_link_libraries(fields PRIVATE odepack_fields)
configure_py_module(fields)

#===========================================================================================
#                           Install
#===========================================================================================

install(TARGETS solvers events oderesult ode chaos interp fields
    LIBRARY DESTINATION odepack
    COMPONENT python
)

install(TARGETS
    odepack_pytools
    odepack_solvers
    odepack_events
    odepack_oderesult
    odepack_ode
    odepack_chaos
    odepack_interp
    odepack_fields
    ARCHIVE DESTINATION odepack/lib
    COMPONENT python
)

install(DIRECTORY ${ODEPACK_ROOT}/include/odepack/
    DESTINATION odepack/include
    COMPONENT python
    FILES_MATCHING PATTERN "*.hpp"
)
