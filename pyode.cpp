#include "src/pyode.hpp"

PYBIND11_MODULE(_integrate, m) {
    define_ode_module<double, Eigen::Array<double, 1, -1>>(m);
}


//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o _integrate$(python3-config --extension-suffix)