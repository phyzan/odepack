#include "../include/odepack/rk_adaptive.hpp"
#include <string>

const int N = -1;
using T = double;
using Ty = vec<T, N>;
const T pi = 3.14159265359;

T event(const T&, const T* q, const T*, const void*){
    return q[1];
}

T obj_fun(const T& t, const T*, const T*, const void*){
    return t-9.95;
}

void f(T* df, const T&, const T* q, const T*, const void*){
    df[0] = q[2];
    df[1] = q[3];
    df[2] = -4*pi*pi*q[0];
    df[3] = -4*pi*pi*q[1];
}

void jac(T* jac, const T&, const T* q, const T* args, const void*){
    jac[0] = 0;
    jac[1] = 1;
    jac[2] = T(-1) + T(-2)*args[0]*q[0]*q[1];
    jac[3] = (T(1) + T(-1)*pow(q[0], T(2)))*args[0];
}

int main(){
    
    Ty q0(4);
    q0 << 1, 1, 2.3, 4.5;
    T first_step = 0;
    T rtol = 1e-10;
    T atol = 1e-10;
    T min_step = 0;
    T max_step = inf<T>();
    T k = 1000;

    // T tmax = 20000;

    PeriodicEvent<T, N> ev_per("Periodic", 0.2);
    PeriodicEvent<T, N> ev_per2("sd", 0.2);
    PreciseEvent<T, N> ev_prec("precise", obj_fun, 0);

    RK45<T, N> solver({f, nullptr, nullptr}, 0, q0, rtol, atol, min_step, max_step, first_step, {k}, {&ev_per, &ev_per2});
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 100000; i++) {
        solver.advance();
        solver.state().show(5);
        std::cin.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    solver.state().show();

    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    // ode.integrate(tmax, -1, {}).examine();

    //g++ -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}