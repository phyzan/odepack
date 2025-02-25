#include "src/pyode.hpp"

PYBIND11_MODULE(_lowlevelode, m) {
    define_ode_module<double, Eigen::Array<double, Eigen::Dynamic, 1>>(m);
}
