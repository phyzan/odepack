#include "odepack/ode/IntegratorEnum.hpp"
#include <odepack/odepack.hpp>


struct MyODE{

    static void Rhs(auto* dy_dt, const auto& /*t*/, const auto* q) {
        //3D lorenz system, args = {sigma, rho, beta}
        const double sigma = 10.0;
        const double rho = 28.0;
        const double beta = 8.0/3.0;
        dy_dt[0] = sigma*(q[1] - q[0]);
        dy_dt[1] = q[0]*(rho - q[2]) - q[1];
        dy_dt[2] = q[0]*q[1] - beta*q[2];
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
        1
    );

    
    solver.advance_until(10000);
    print("Expected Lyapunov exponent: ~0.905");
    print("Computed Lyapunov exponent: ", solver.lyapunov_exponent());

}