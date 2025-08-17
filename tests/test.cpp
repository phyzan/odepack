#include "../include/odepack/variational.hpp"
#include <iostream>
#include <fstream>
#include <odepack/events.hpp>
#include <string>

const int N = -1;
using T = double;
using Ty = vec<T, N>;

T event(const T& t, const T* q, const T* args){
    return q[1];
}

void f(T* df, const T& t, const T* q, const T* args){
    df[0] = q[1];
    df[1] = -q[0] + (1. - pow(q[0], 2.))*args[0]*q[1];
}

void jac(JacMat<T, N>& result, const T& t, const vec<T, N>& q, const std::vector<T>& args){
    result(0, 0) = T(0);
    result(0, 1) = T(1);
    result(1, 0) = T(-1) + T(-2)*args[0]*q[0]*q[1];
    result(1, 1) = (T(1) + T(-1)*pow(q[0], T(2)))*args[0];
}

void jac2(T* jac, const T& t, const T* q, const T* args){
    jac[0] = 0;
    jac[1] = 1;
    jac[2] = T(-1) + T(-2)*args[0]*q[0]*q[1];
    jac[3] = (T(1) + T(-1)*pow(q[0], T(2)))*args[0];
}

int main(){
    
    T t0 = 0;
    Ty q0(2);
    q0 << 2, 0;
    T first_step = 0;
    T rtol = 1e-12;
    T atol = 1e-12;
    T min_step = 0;
    T max_step = inf<T>();
    T k = 1000;

    T tmax = 20000;

    PreciseEvent<T, N> ev("Event", event);

    std::vector<Event<T, N>*> evs = {&ev};

    ODE<T, N> ode({f, jac2}, t0, q0, rtol, atol, min_step, max_step, first_step, {k}, {&ev}, "BDF");

    ode.integrate(tmax, -1, {}).examine();

    //g++ -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}