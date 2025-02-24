#include "pyode.hpp"

PYBIND11_MODULE(pytest, m) {
    define_ode_module<double, py::array_t<double>>(m);
}
