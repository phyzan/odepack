#include "pyode.hpp"

PYBIND11_MODULE(pytest, m) {
    define_ode_module<double, 0>(m);
}
