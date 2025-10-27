#include "include/pyode/pyode.hpp"

PYBIND11_MODULE(odesolvers, m) {
    define_ode_module<double, 0>(m);
}