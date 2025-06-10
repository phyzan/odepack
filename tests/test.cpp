#include "../include/odepack/variational.hpp"

const int N = -1;
using T = double;
using Ty = vec<T, N>;

void f(Ty& df, const T& t, const Ty& q, const std::vector<T>& args){
    df[0] = q[2];
    df[1] = q[3];
    df[2] = -q[0];
    df[3] = -q[1];
}

void jacm (JacMat<T, N>& j, const T& t, const Ty& q, const std::vector<T>& args){
    JacMat<T, N> J(4, 4);
    J <<  0.0,  0.0,  1.0,  0.0,
          0.0,  0.0,  0.0,  1.0,
         -1.0,  0.0,  0.0,  0.0,
          0.0, -1.0,  0.0,  0.0;
    j=J;
}

T ps_func(const T& t, const Ty& q, const std::vector<T>& args){
    return q[1];
}

inline T stopfunc(const T& t, const Ty& q, const std::vector<T>& args){
    return q[1]-15;
}

bool check(const T& t, const Ty& q, const std::vector<T>& args){
    return q[3]>0;
}

int main(){
    T pi = 3.141592653589793238462;

    T t0 = 0;
    Ty q0(4);
    q0 << 1., 1., 2.3, 4.5;
    T first_step = 1e-3;
    T rtol = 1e-16;
    T atol = 1e-16;
    T min_step = 0.;
    T max_step = 100;

    // T tmax = 10001*pi/2;
    T tmax = 1;

    PreciseEvent<T, N> ps("Poincare Section", ps_func);
    PeriodicEvent<T, N> ev2("periodic", 1, 0.998);

    VariationalODE<T, N> ode({f, jacm}, t0, q0, 1, rtol, atol, min_step, max_step, first_step, {}, {&ps}, "RK45");
    ode.var_integrate(10000, 100).examine();
    ode.state().show();

    //g++ -g -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}