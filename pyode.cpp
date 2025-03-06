#include "src/pyode.hpp"

#include <iostream>

#ifndef N
#define N -1  // Default value if not provided
#endif

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, Eigen::Array<double, 1, N>>(m);
}


//g++ -DN=-1 -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)