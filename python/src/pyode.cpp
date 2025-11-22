#include "include/pyode/mpreal_caster.hpp"
#include "include/pyode/pyode.hpp"


PYBIND11_MODULE(odesolvers, m) {
    // Set a default precision for MPFR at module initialization
    // This MUST be set before any mpreal objects are created
    // mpfr::mpreal::set_default_prec(10);

    define_ode_module<double>(m);

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