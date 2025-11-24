#include "include/pyode/pyode.hpp"
#include "include/pyode/mpreal_caster.hpp"

PYBIND11_MODULE(odesolvers_mpreal, m) {
    m.doc() = "ODE solvers for MPFR arbitrary precision";
    define_ode_module<mpfr::mpreal>(m, "_MpReal");

    m.def("set_mpreal_prec",
      [](int bits) {
          mpfr::mpreal::set_default_prec(bits);
      },
      py::arg("bits"),
      "Set the default MPFR precision (in bits) for mpfr::mpreal.")
    .def("mpreal_prec", []() {
          return mpfr::mpreal::get_default_prec();
      });
}
