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

// void jacm (JacMat<T, N>& j, const T& t, const Ty& q, const std::vector<T>& args){
//     JacMat<T, N> J(4, 4);
//     J <<  0.0,  0.0,  1.0,  0.0,
//           0.0,  0.0,  0.0,  1.0,
//          -1.0,  0.0,  0.0,  0.0,
//           0.0, -1.0,  0.0,  0.0;
//     j=J;
// }

T ps_func(const T& t, const Ty& q, const std::vector<T>& args){
    return q[1]-1;
}

bool check(const T& t, const Ty& q, const std::vector<T>& args){
    return q[3]>0;
}

int main(){
    T t0 = 0;
    Ty q0(4);
    q0 << 1, -2, 3, 5;
    T first_step = 0;
    T rtol = 0;
    T atol = 1e-10;
    T min_step = 0.;
    T max_step = 100;
    T pi = 3.14159265359;

    T tmax = 1000;


    PreciseEvent<T, N> ps("Poincare Section", ps_func, check);
    PeriodicEvent<T, N> ev2("periodic", 5);

    ODE<T, N> ode(f, t0, q0, rtol, atol, min_step, max_step, first_step, {}, {&ps, &ev2}, "RK45");

    auto res = ode.rich_integrate(tmax);

    // std::cout << res(9999).transpose() << std::endl;
    std::cout << res(1).transpose() << std::endl;
    std::cout << res(1.0000001).transpose() << std::endl;
    // RK45<T, N> s(f, t0, q0, rtol, atol, min_step, max_step, first_step, {}, {&ps});
    // s.start_interpolation();
    // s.set_goal(10000);
    // while (s.is_running()){
    //     s.advance();
    // }


    // s.set_goal(-1);
    // while (s.is_running()){
    //     s.advance();
    // }
    // s.state().show();
    
    // std::cout << s.interpolators().front()->call(2*pi*105).transpose() << std::endl;
    // std::cout << s.interpolators().back()->call(2*pi*105).transpose() << std::endl;

    // ode.integrate(tmax, 100).examine();
    // ode.state().show();

    //g++ -g -O3 -fopenmp -Wall -std=c++20 tests/test.cpp -o tests/test -lmpfr -lgmp

}