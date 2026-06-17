#include "../include/odepack/odepack.hpp"


struct MyODE{

    template<typename T>
    static void Rhs(T* dy_dt, const T& t, const T* y, const T* args) {
        //3D lorenz system, args = {sigma, rho, beta}
        dy_dt[0] = args[0]*(y[1] - y[0]);
        dy_dt[1] = y[0]*(args[1] - y[2]) - y[1];
        dy_dt[2] = y[0]*y[1] - args[2]*y[2];
    }

};


using namespace ode;


int main(){

    using T = double;

    static constexpr size_t NSYS = 3;

    std::array<T, 3> y0 = {1.0, 1.0, 1.0};
    std::array<T, 3> y0_var = {1.0, 1.0, 1.0};
    std::vector<T> args = {10.0, 28.0, 8.0/3.0}; // sigma, rho, beta

    chaos::VariationalSolver<RK45, T, NSYS, ode::SolverPolicy::Static, MyODE> solver(
        MyODE{},
        0.0,
        y0.data(),
        y0_var.data(),
        NSYS,
        0.1,
        1e-9,
        1e-12,
        0.0,
        inf<T>(),
        0.0,
        1,
        args
    );


    solver.advance_until(10000);
    print("Expected Lyapunov exponent: ~0.905");
    print("Computed Lyapunov exponent: ", solver.lyapunov_exponent());

    // g++ -std=c++20 -O3 -DMPREAL -Iexternal/autodiff/include tests/variational_test.cpp -o test -lmpfr -lgmp
}