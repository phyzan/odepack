#include "include/pyode/pyode.hpp"
#include "include/pyode/mpreal_caster.hpp"

PYBIND11_MODULE(odesolvers_mpreal, m) {
    m.doc() = "ODE solvers for MPFR arbitrary precision";
    define_ode_module<mpfr::mpreal>(m, "_MpReal");
}
