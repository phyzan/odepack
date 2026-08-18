#include "odepack/ode/IntegratorEnum.hpp"
#include <odepack/odepack.hpp>


struct MyODE{

    static void Rhs(auto* dy_dt, const auto& /*t*/, const auto* y, const auto* args) {
        //3D lorenz system, args = {sigma, rho, beta}
        dy_dt[0] = args[0]*(y[1] - y[0]);
        dy_dt[1] = y[0]*(args[1] - y[2]) - y[1];
        dy_dt[2] = y[0]*y[1] - args[2]*y[2];
    }

};


using namespace ode;


void test_variational_solver(){

    std::cout << "\n---------- Testing VariationalSolver ------------------\n" << std::endl;

    std::cout << " (This might take too long if compiled in debug mode)" << std::endl;

    using T = double;

    static constexpr size_t NSYS = 3;

    std::array<T, 3> y0 = {1.0, 1.0, 1.0};
    std::array<T, 3> y0_var = {1.0, 1.0, 1.0};
    std::vector<T> args = {10.0, 28.0, 8.0/3.0}; // sigma, rho, beta

    chaos::VariationalSolver<Integrator::RK45, T, NSYS, ode::SolverPolicy::Static, MyODE> solver(
        MyODE{},
        0.0,
        View1D<T, NSYS>{y0.data()},
        View1D<T, NSYS>{y0_var.data()},
        0.1,
        1e-9,
        1e-12,
        0.0,
        inf<T>(),
        0.0,
        1,
        std::move(args)
    );

    
    solver.advance_until(10000);
    print("Expected Lyapunov exponent: ~0.905");
    print("Computed Lyapunov exponent: ", solver.lyapunov_exponent());

}