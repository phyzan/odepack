#include "src/pyode.hpp"

#include <iostream>

PYBIND11_MODULE(odepack, m) {
    define_ode_module<double, Eigen::Array<double, 1, 4>>(m);
}


//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) pyode.cpp -o odepack$(python3-config --extension-suffix)

//g++ -O3 -march=native -flto -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) -fno-math-errno bohm.cpp -o bohm$(python3-config --extension-suffix)