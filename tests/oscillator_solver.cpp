#include "../include/odepack/solvers.hpp"


using namespace ode;

using mpfr::mpreal;

template<typename T>
void df_dt(T* dy_dt, const T& t, const T* y, const T* args, const void* ptr) {
    //2D oscillator: y'' + y = 0
    dy_dt[0] = y[1];
    dy_dt[1] = -y[0];
}

template<typename T>
T crossing(const T& t, const T* y, const T* args, const void* ptr) {
    return y[1] - 1;
}

template<typename Scalar, SolverTemplate typename Solver>
void oscillator_test(){
    // Initial conditions
    std::array<Scalar, 2> y0 = {3, 0};

    // Define the y' = 1 crossing
    PreciseEvent<Scalar> event("event", crossing<Scalar>);

    constexpr SolverPolicy SP = SolverPolicy::RichStatic;

    // Create solver
    Scalar t = 0;
    Scalar rtol = 1e-6;
    Scalar atol = 1e-9;
    Scalar min_step = 0;
    Scalar max_step = 1;
    Scalar first_step = 0;
    constexpr size_t nsys = 2;
    int dir = 1;
    auto solver = getSolver<Solver, Scalar, nsys, SP>(
        OdeData{.rhs=df_dt<Scalar>},
        t,
        y0.data(),
        nsys,
        rtol,
        atol,
        min_step,
        max_step,
        first_step,
        dir,
        {},
        {&event}
    );

    // Advance until event is detected
    while (!solver.at_event()) {
        solver.advance();
    }

    std::cout << "Event detected at t = " << solver.t() << "\n";
    std::cout << "State at event: ";
    auto v = solver.vector();
    for (size_t i = 0; i < 2; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n" << std::endl;
}

template<typename Scalar, SolverTemplate typename Solver>
void lambda_test(){
    // Initial conditions
    std::array<Scalar, 2> y0 = {3, 0};


    // Define the y' = 1 crossing
    PreciseEvent<Scalar> event("event", crossing<Scalar>);

    constexpr SolverPolicy SP = SolverPolicy::RichStatic;

    // Create solver
    Scalar t = 0;
    Scalar rtol = 1e-6;
    Scalar atol = 1e-9;
    Scalar min_step = 0;
    Scalar max_step = 1;
    Scalar first_step = 0;
    constexpr size_t nsys = 2;
    int dir = 1;

    Scalar omega = 1;

    auto ode_rhs = [&](Scalar* dy_dt, const Scalar& t, const Scalar* y, const Scalar* args, const void* ptr) {
        dy_dt[0] = y[1];
        dy_dt[1] = - omega*omega*y[0];
    };

    auto solver = getSolver<Solver, Scalar, nsys, SP>(
        OdeData{.rhs=ode_rhs},
        t,
        y0.data(),
        nsys,
        rtol,
        atol,
        min_step,
        max_step,
        first_step,
        dir,
        {},
        {&event}
    );

    // Advance until event is detected
    while (!solver.at_event()) {
        solver.advance();
    }

    std::cout << "Event detected at t = " << solver.t() << "\n";
    std::cout << "State at event: ";
    auto v = solver.vector();
    for (size_t i = 0; i < 2; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n" << std::endl;
}



int main() {

    /*
    Compile with:
    g++ -std=c++20 -O3 -DMPREAL tests/oscillator_solver.cpp -o test -lmpfr -lgmp
    */

    // Test with double precision
    std::cout << "Testing with double precision:\n";
    oscillator_test<double, RK45>();
    lambda_test<double, RK45>();

    oscillator_test<double, RK23>();
    lambda_test<double, RK23>();

    oscillator_test<double, BDF>();
    lambda_test<double, BDF>();

    oscillator_test<double, DOP853>();
    lambda_test<double, DOP853>();

    // Test with arbitrary precision
    std::cout << "\nTesting with arbitrary precision (mpreal):\n";
    mpreal::set_default_prec(256); // Set precision to 256 bits

    oscillator_test<mpreal, RK45>();
    lambda_test<mpreal, RK45>();

    oscillator_test<mpreal, RK23>();
    lambda_test<mpreal, RK23>();

    oscillator_test<mpreal, BDF>();
    lambda_test<mpreal, BDF>();

    oscillator_test<mpreal, DOP853>();
    lambda_test<mpreal, DOP853>();

    std::cout << "Expected event at t = 3.48143\n";
    std::cout << "Even state expected: {-2.82843, 1}\n";
}