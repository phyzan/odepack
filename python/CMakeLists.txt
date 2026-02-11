# Python extension modules — thin wrappers linked against static libraries

# Helper function to configure Python extension modules
function(configure_py_module target_name)
    if(MPREAL)
        target_compile_definitions(${target_name} PRIVATE MPREAL)
    endif()
endfunction()

# 1. solvers module: ODE solvers (RK23, RK45, DOP853, BDF, RK4)
pybind11_add_module(solvers src/bindings/solvers.cpp)
target_link_libraries(solvers PRIVATE odepack_solvers)
configure_py_module(solvers)

# 2. events module: Event handling
pybind11_add_module(events src/bindings/events.cpp)
target_link_libraries(events PRIVATE odepack_events)
configure_py_module(events)

# 3. oderesult module: ODE result containers
pybind11_add_module(oderesult src/bindings/oderesult.cpp)
target_link_libraries(oderesult PRIVATE odepack_oderesult)
configure_py_module(oderesult)

# 4. ode module: Low-level ODE interface
pybind11_add_module(ode src/bindings/ode.cpp)
target_link_libraries(ode PRIVATE odepack_ode)
configure_py_module(ode)

# 5. chaos module: Variational/Lyapunov solvers
pybind11_add_module(chaos src/bindings/chaos.cpp)
target_link_libraries(chaos PRIVATE odepack_chaos)
configure_py_module(chaos)

# 6. fields module: Vector field interpolation
pybind11_add_module(fields src/bindings/fields.cpp)
target_link_libraries(fields PRIVATE odepack_fields)
configure_py_module(fields)

# Install all Python extension modules into the odepack Python package
install(TARGETS solvers events oderesult ode chaos fields
    LIBRARY DESTINATION odepack
    COMPONENT python
)

# Install the static libraries and headers into the Python package
# so C++ projects can link against them after pip install
install(TARGETS
    odepack_pytools
    odepack_solvers
    odepack_events
    odepack_oderesult
    odepack_ode
    odepack_chaos
    odepack_fields
    ARCHIVE DESTINATION odepack/lib
    COMPONENT python
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    DESTINATION odepack/include
    COMPONENT python
    FILES_MATCHING PATTERN "*.hpp"
)
