cmake_minimum_required(VERSION 3.15...3.26)

project(odepack LANGUAGES CXX)

# ----------- Main odepack interface target ---------------

add_library(odepack INTERFACE)
add_library(odepack::odepack ALIAS odepack)

target_compile_features(odepack INTERFACE cxx_std_20)

target_include_directories(odepack INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------- Add interface for macros ---------------

option(DPK_DENSE_RK4 "Enable accurate RK4 dense output for the RK4 solver (extra memory, slightly slower performance)" OFF)
option(DPK_NO_WARN "Disable ODE solver console warnings" OFF)
option(DPK_NO_NAN_CHECK "Disable NaN/inf checks on solver output for performance" OFF)
option(DEBUG "Enable debug build (-O0 -g3, asserts enabled) instead of the optimized release build" OFF)

target_compile_definitions(odepack INTERFACE
    $<$<BOOL:${DPK_DENSE_RK4}>:RK4_DENSE>
    $<$<BOOL:${DPK_NO_WARN}>:NO_ODE_WARN>
    $<$<BOOL:${DPK_NO_NAN_CHECK}>:DPK_NO_NAN_CHECK>
)

if((DEFINED PROJECT_IS_TOP_LEVEL AND PROJECT_IS_TOP_LEVEL) OR
   (NOT DEFINED PROJECT_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    set(ODEPACK_BUILD_TESTS_DEFAULT ON)
else()
    set(ODEPACK_BUILD_TESTS_DEFAULT OFF)
endif()
option(ODEPACK_BUILD_TESTS "Build the odepack_tests executable from tests/src/*.cpp" ${ODEPACK_BUILD_TESTS_DEFAULT})

if(ODEPACK_BUILD_TESTS)
    set(LAZY_ENABLE_MPREAL ON CACHE BOOL "" FORCE)
endif()

#===========================================================================================
#                                           xdiff
#===========================================================================================
add_subdirectory(external/xdiff)

#===========================================================================================
#                           ndspan (header-only, no CMakeLists of its own)
#===========================================================================================
add_library(odepack_ndspan INTERFACE)
add_library(odepack::ndspan ALIAS odepack_ndspan)
target_include_directories(odepack_ndspan INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ndspan/include
)

#===========================================================================================
#                           polybox (header-only, no CMakeLists of its own)
#===========================================================================================
add_library(odepack_polybox INTERFACE)
add_library(odepack::polybox ALIAS odepack_polybox)
target_include_directories(odepack_polybox INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/polybox/include
)

#===========================================================================================
#                           qhull (static library)
#===========================================================================================
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(external/qhull EXCLUDE_FROM_ALL)

add_library(odepack_qhull INTERFACE)
add_library(odepack::qhull ALIAS odepack_qhull)
target_include_directories(odepack_qhull INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/qhull/src
)
target_link_libraries(odepack_qhull INTERFACE qhullstatic_r)

# ----------- Link everything into the umbrella target ---------------

target_link_libraries(odepack INTERFACE
    xdiff
    odepack::ndspan
    odepack::polybox
    odepack::qhull
)

#===========================================================================================
#                           Tests (combined into a single executable)
#===========================================================================================
# Each tests/src/*.cpp (besides main.cpp) implements a void test_<name>() entry point declared
# in the matching tests/include/<name>.hpp, and tests/src/main.cpp includes those headers and
# calls each in turn. All of tests/src/*.cpp are compiled and linked into one odepack_tests
# executable.
#
# Only built when odepack is configured directly (not when pulled in via add_subdirectory by
# another project), so consumers who only want the interface never build or even see this
# target.

if(ODEPACK_BUILD_TESTS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    file(GLOB ODEPACK_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/*.cpp)

    add_executable(odepack_tests ${ODEPACK_TEST_SOURCES})
    target_include_directories(odepack_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)
    target_link_libraries(odepack_tests PRIVATE odepack::odepack lazy::mpfr)

    set(ODEPACK_TEST_COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wno-sign-conversion
        -fopenmp
    )
    set(ODEPACK_TEST_LINK_OPTIONS -fopenmp)

    set(ODEPACK_DEBUG_BUILD OFF)
    if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ODEPACK_DEBUG_BUILD ON)
    endif()

    if(ODEPACK_DEBUG_BUILD)
        list(APPEND ODEPACK_TEST_COMPILE_OPTIONS -O0 -g3 -ggdb3 -fno-omit-frame-pointer -UNDEBUG -Wconversion)
        message(STATUS "ODEPACK_DEBUG_BUILD=ON (asserts enabled)")
    else()
        list(APPEND ODEPACK_TEST_COMPILE_OPTIONS -Wno-unused-parameter -O3 -DNDEBUG -g0 -fno-math-errno -fno-trapping-math -ffunction-sections -fdata-sections)
        list(APPEND ODEPACK_TEST_LINK_OPTIONS -Wl,--gc-sections -Wl,--as-needed)

        include(CheckIPOSupported)
        check_ipo_supported(RESULT ODEPACK_IPO_SUPPORTED OUTPUT ODEPACK_IPO_ERROR LANGUAGES CXX)
        if(ODEPACK_IPO_SUPPORTED)
            set_target_properties(odepack_tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        else()
            message(STATUS "IPO/LTO not enabled: ${ODEPACK_IPO_ERROR}")
        endif()
        message(STATUS "ODEPACK_DEBUG_BUILD=OFF (asserts disabled via NDEBUG)")
    endif()

    target_compile_options(odepack_tests PRIVATE ${ODEPACK_TEST_COMPILE_OPTIONS})
    target_link_options(odepack_tests PRIVATE ${ODEPACK_TEST_LINK_OPTIONS})
endif()
