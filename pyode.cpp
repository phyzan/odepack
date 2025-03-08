#include "src/pyode.hpp"

#include <iostream>

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, vec<double>>(m);
}