#include "../include/odepack/variational.hpp"
#include <iostream>
#include <fstream>
#include <string>

const int N = -1;
using T = double;
using Ty = vec<T, N>;

double memory() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::string value = line.substr(6);
            size_t pos = value.find_first_of("0123456789");
            if (pos != std::string::npos) {
                size_t end = value.find(" kB", pos);
                std::string number = value.substr(pos, end - pos);
                size_t bytes = std::stoul(number) * 1024; // kB -> bytes
                return static_cast<double>(bytes) / (1024 * 1024 * 1024); // bytes -> GB
            }
        }
    }
    return 0.0;
}

void f(Ty& df, const T& t, const Ty& q, const std::vector<T>& args){
    df[0] = q[1];
    df[1] = -q[0] + (1. - pow(q[0], 2.))*args[0]*q[1];
}

void jac(JacMat<T, N>& result, const T& t, const vec<T, N>& q, const std::vector<T>& args){
    result(0, 0) = -1. - 2.*args[0]*q[0]*q[1];
    result(0, 1) = (1. - pow(q[0], 2.))*args[0];
    result(1, 0) = -1. - 2.*args[0]*q[0]*q[1];
    result(1, 1) = (1. - pow(q[0], 2.))*args[0];
}


int main(){
    
    std::cin.get();
    T t0 = 0;
    Ty q0(2);
    q0 << 2, 0;
    T first_step = 0;
    T rtol = 1e-14;
    T atol = 1e-12;
    T min_step = 0.;
    T max_step = inf<T>();
    T k = 1000;

    T tmax = 20000;

    ODE<T, N> ode({f, jac}, t0, q0, rtol, atol, min_step, max_step, first_step, {k}, {}, "RK23");
    //g++ -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}