#include "odepack/ode/Core/Events.hpp"
#include "odepack/ode/Core/VirtualBase.hpp"
#include <odepack/odepack.hpp>
#include <mpreal.h>

using namespace ode;

template<typename T>
T crossing(const T& /*t*/, const T* y, const T* /*args*/) {
    return y[1] - 1;
}


template<typename Scalar>
void crossing_test(Integrator method) {
    // Initial conditions
    std::array<Scalar, 2> y0 = {3, 0};


    // Define the y' = 1 crossing
    pbox::Box<Event<Scalar>> event = make_event<Scalar, PreciseEvent>("event", crossing<Scalar>, Scalar(0));

    // Create solver
    Scalar t = 0;
    Scalar rtol = 1e-6;
    Scalar atol = 1e-9;
    Scalar min_step = 0;
    Scalar max_step = 1;
    Scalar stepsize = 0;
    constexpr size_t nsys = 2;
    int dir = 1;

    Scalar omega = 1;

    pbox::Box<OdeRichSolver<Scalar, nsys>> solver = make_solver<UtilPolicy::RichVirtual>(
        method,
        OdeData{.Rhs=
            ODE_LAMBDA(out, /*t*/, q, /*args*/) {
                out[0] = q[1];
                out[1] = - omega*omega*q[0];
            }
        },
        t,
        View1D<Scalar, 2>{y0.data()}, // q0
        rtol,
        atol,
        min_step,
        max_step,
        stepsize,
        dir,
        std::vector<Scalar>{},
        make_event_list<Scalar>(std::move(event))
    );

    // Advance until event is detected
    while (!solver->at_event()) {
        solver->advance();
    }

    std::cout << "Event detected at t = " << solver->t() << "\n";
    std::cout << "State at event: ";
    auto v = solver->vector();
    for (size_t i = 0; i < 2; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n" << std::endl;
}



void test_oscillator_solver() {

    // Test with double precision
    std::cout << "Testing with double precision:\n";
    crossing_test<double>(Integrator::RK45);

    // Test with arbitrary precision
    std::cout << "\nTesting with arbitrary precision (mpreal):\n";
    mpfr::mpreal::set_default_prec(256); // Set precision to 256 bits

    crossing_test<mpfr::mpreal>(Integrator::RK45);
    std::cout << "Expected event at t = 3.48143\n";
}