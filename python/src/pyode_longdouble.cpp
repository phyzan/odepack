#include "include/pyode/pyode.hpp"
#include "include/pyode/mpreal_caster.hpp"

PYBIND11_MODULE(odesolvers_longdouble, m) {
    m.doc() = "ODE solvers for long double precision";
    define_ode_module<long double>(m, "_LongDouble");
}
