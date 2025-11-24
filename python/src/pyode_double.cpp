#include "include/pyode/pyode.hpp"
#include "include/pyode/mpreal_caster.hpp"

PYBIND11_MODULE(odesolvers_double, m) {
    m.doc() = "ODE solvers for double precision";
    define_ode_module<double>(m, "_Double");
}
