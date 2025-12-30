#include "include/odepack/ode.hpp"

using T = double;

void df_dt(T* res, const T& t, const T* q, const T* args, const void* obj){
    res[0] = q[1];
    res[1] = -q[0];
}

void mask(T* res, const T& t, const T* q, const T* args, const void* obj){
    res[0] = q[1];
    res[1] = q[0];
}

int main(){

    double t0 = 0;
    std::vector<T> ics {-1, 3};
    Array1D<double> q0(ics.data(), 2);
    double rtol = 1e-10;
    double atol = 1e-10;

    PeriodicEvent<double, 0> periodic_event("PERIOD", 1, mask);

    RK45<double, 0> solver({.rhs=df_dt, .jacobian=nullptr, .obj=nullptr}, t0, q0, rtol, atol, 0, inf<double>(), 0, 1, {}, {&periodic_event});

    solver.advance_to_event();

    // solver.state().show();

    while (true) {
        solver.state().show();
        std::cin.get();
        solver.advance_to_event();
    }
}