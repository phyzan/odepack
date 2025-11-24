#include "include/pyode/pyode.hpp"
#include "include/pyode/mpreal_caster.hpp"

PYBIND11_MODULE(odesolvers_float, m) {
    m.doc() = "ODE solvers for float precision";
    define_ode_module<float>(m, "_Float");
}
