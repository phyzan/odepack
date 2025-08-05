#include "../include/odepack/variational.hpp"

const int N = -1;
using T = double;
using Ty = vec<T, N>;

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
    T t0 = 0;
    Ty q0(2);
    q0 << 2, 0;
    T first_step = 0;
    T rtol = 1e-9;
    T atol = 1e-10;
    T min_step = 0.;
    T max_step = inf<T>();
    T k = 1000;

    T tmax = 1000;

    ODE<T, N> ode({f, jac}, t0, q0, rtol, atol, min_step, max_step, first_step, {k}, {}, "BDF");
    
    ode.integrate(tmax).examine();

    //g++ -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}