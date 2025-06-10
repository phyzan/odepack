#include "../include/odepack/pyode.hpp"

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, -1>(m);
}

//g++ -O3 -fno-math-errno -Wall -march=x86-64 -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) python/pyode.cpp -o python/odepack$(python3-config --extension-suffix) -lmpfr -lgmp