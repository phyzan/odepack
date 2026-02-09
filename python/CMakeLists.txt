# Python extension module — thin wrapper linked against libodepack

pybind11_add_module(odesolvers src/bindings.cpp)
target_link_libraries(odesolvers PRIVATE odepack_lib)

if(MPREAL)
    target_compile_definitions(odesolvers PRIVATE MPREAL)
endif()

# Install the compiled module into the odepack Python package
install(TARGETS odesolvers
    LIBRARY DESTINATION odepack
    COMPONENT python
)

# Install the static library and headers into the Python package
# so C++ projects can link against them after pip install
install(TARGETS odepack_lib
    ARCHIVE DESTINATION odepack/lib
    COMPONENT python
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    DESTINATION odepack/include
    COMPONENT python
    FILES_MATCHING PATTERN "*.hpp"
)
