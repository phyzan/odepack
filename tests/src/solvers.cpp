#include <odepack/odepack.hpp>

using namespace ode;

/*
Testing a simple non-virtual solver, using all available methods
*/

static void rhs(double* out, const double& /*t*/, const double* q, const double* args){
    const double& mu = args[0];
    const double& x = q[0];
    const double& xdot = q[1];
    out[0] = xdot;
    out[1] = mu * (1 - x*x) * xdot - x;
}

template<SolverTemplate typename Solver>
void run(const char* label){
    // declare ics
    const double t0 = 0;
    std::array<double, 2> q0 = {1, 0};
    std::vector<double> args = {300.};
    const double rtol = 1e-10;
    const double atol = 1e-10;
    const double min_step = 0;
    const auto max_step = ode::inf<double>();
    const double stepsize = 0.0001;

    const size_t nsys = 2;
    const double tmax = 1.0;
    std::cout << "\n---------- Testing " << label << " method------------------\n" << std::endl;

    // Stack allocated solver (nsys is passed as a compile-time parameter)
    auto solver_1 = getSolver<Solver, SolverPolicy::Static>(
        OdeData{.Rhs=rhs}, t0, View1D<double, nsys>{q0.data()}, rtol, atol, min_step, max_step, stepsize, 1, args);

    // Heap allocated solver (nsys is passed as a runtime parameter)
    auto solver_2 = getSolver<Solver, SolverPolicy::Static>(
        OdeData{.Rhs=rhs}, t0, View1D{q0.data(), nsys}, rtol, atol, min_step, max_step, stepsize, 1, args);

    
    
    solver_1.advance_until(tmax);
    solver_2.advance_until(tmax);



    if (solver_1.vector()[0] != solver_2.vector()[0] || solver_1.vector()[1] != solver_2.vector()[1]){
        std::cerr << "Error: Stack and Heap solvers produced different results!" << std::endl;
    } else {
        std::cout << "Result ( t = " << tmax << " ): x = " << solver_1.vector()[0] << ", xdot = " << solver_1.vector()[1] << std::endl;

        std::array<double, 2> arr;
        solver_1.Rhs(arr.data(), tmax, solver_1.vector().data());
        
        std::cout << "xdotdot = " << arr[0] << " " << arr[1] << std::endl;
    }
}

void test_solvers(){

    run<ode::Euler>("Euler");
    run<ode::RK23>("RK23");
    run<ode::RK45>("RK45");
    run<ode::DOP853>("DOP853");
    run<ode::BDF>("BDF");
    run<ode::RK4>("RK4");

}