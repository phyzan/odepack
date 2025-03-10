#include "src/pyode.hpp"

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, vec<double>>(m);
}