#include "src/pyode.hpp"

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, vec<double>>(m);
}

//g++ -O3 -Wall -march=native -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix) -lmpfr -lgmp