# External dependencies setup
# Include this from any CMakeLists.txt that needs the bundled dependencies

# Ensure ODEPACK_ROOT is set
if(NOT DEFINED ODEPACK_ROOT)
    message(FATAL_ERROR "ODEPACK_ROOT must be set before including external.cmake")
endif()

# Guard against multiple inclusion
if(TARGET odepack_xdiff)
    return()
endif()

#===========================================================================================
#                           xdiff (header-only)
#===========================================================================================
add_library(odepack_xdiff INTERFACE)
add_library(odepack::xdiff ALIAS odepack_xdiff)
target_include_directories(odepack_xdiff INTERFACE
    ${ODEPACK_ROOT}/external/xdiff/include
    ${ODEPACK_ROOT}/external/xdiff/external/lazy/include
)

#===========================================================================================
#                           ndspan (header-only)
#===========================================================================================
add_library(odepack_ndspan INTERFACE)
add_library(odepack::ndspan ALIAS odepack_ndspan)
target_include_directories(odepack_ndspan INTERFACE
    ${ODEPACK_ROOT}/external/ndspan/include
)

#===========================================================================================
#                           polybox (header-only)
#===========================================================================================
add_library(odepack_polybox INTERFACE)
add_library(odepack::polybox ALIAS odepack_polybox)
target_include_directories(odepack_polybox INTERFACE
    ${ODEPACK_ROOT}/external/polybox/include
)


#===========================================================================================
#                           mpreal (header-only, requires system MPFR/GMP)
#===========================================================================================
add_library(odepack_mpreal INTERFACE)
add_library(odepack::mpreal ALIAS odepack_mpreal)
target_include_directories(odepack_mpreal INTERFACE
    ${ODEPACK_ROOT}/external/mpreal
)

#===========================================================================================
#                           qhull (static library)
#===========================================================================================
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${ODEPACK_ROOT}/external/qhull ${CMAKE_CURRENT_BINARY_DIR}/qhull EXCLUDE_FROM_ALL)

add_library(odepack_qhull INTERFACE)
add_library(odepack::qhull ALIAS odepack_qhull)
target_include_directories(odepack_qhull INTERFACE
    ${ODEPACK_ROOT}/external/qhull/src
)
target_link_libraries(odepack_qhull INTERFACE qhullstatic_r)

#===========================================================================================
#                           odepack headers (interface target)
#===========================================================================================
add_library(odepack_headers INTERFACE)
add_library(odepack::headers ALIAS odepack_headers)
target_include_directories(odepack_headers INTERFACE
    ${ODEPACK_ROOT}/include
)
target_compile_features(odepack_headers INTERFACE cxx_std_20)
